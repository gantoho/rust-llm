//! Checkpoint 保存 / 恢复
//!
//! 文件格式（自描述二进制，`serde_json` 只序列化头信息，张量数据一律按原生 f32 小端写入）：
//!
//! ```text
//! 魔数 "LLMCP2\n"（7 字节）
//! u32 小端：JSON 头长度
//! JSON 头：step、best_val_loss、模型配置、优化器步数 opt_t、参数元信息（名字+形状）
//! 参数数据块：按参数顺序拼接每个参数的 f32 小端数据
//! 一阶动量 m 数据块：与参数同样的顺序与形状（resume 用）
//! 二阶动量 v 数据块：与参数同样的顺序与形状（resume 用）
//! ```
//!
//! 三个数据块等长（都是「参数总元素数 × 4」字节），所以**文件大小只由模型结构决定、与数值无关**。
//!
//! 为什么优化器状态放在数据块里、而不是塞进 JSON 头（v1 的 `LLMCP1` 就是这么干的）：
//! JSON 用十进制文本表示浮点数，一个 f32 平均要十几个字节，二进制只要 4 字节，
//! 优化器状态与参数量级相同，文本编码会把存档体积放大数倍。
//! 文本还有个隐患：JSON 没有 NaN / Infinity，`serde_json` 会把它们写成 `null`，
//! 读回时反序列化失败——数值一旦发散，存档就无法恢复。

use std::fs::File;
use std::io::{Read, Write};

use crate::config::LoRAConfig;
use crate::model::{GPT, GPTConfig};
use crate::optim::{AdamW, Optimizer};
use crate::quant::{QuantMeta, QuantMethod, QuantOpts};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 7] = b"LLMCP2\n";

/// checkpoint 头（JSON 部分）：只放标量和元信息，张量数据一律走后面的二进制块
#[derive(Serialize, Deserialize)]
struct CkptHeader {
    step: usize,
    /// `None` 表示"还没有可用的 best"，等价于 `f32::INFINITY`。
    /// 用 `Option` 而不是直接存 `f32::INFINITY`：JSON 没有 Infinity / NaN 字面量，
    /// `serde_json` 会把它写成 `null`，读回时直接反序列化失败。
    best_val_loss: Option<f32>,
    model: GPTConfig,
    opt_t: usize,
    params: Vec<ParamMeta>,
    /// 存档时的 LoRA 形态（`None` = 普通全参模型）。
    ///
    /// 加载端必须据此**先注入同样的适配器再恢复参数**：参数是按名字逐个对齐的，
    /// 少了 `*.lora_a` / `*.lora_b` 这些名字，参数数量与形状就对不上，直接报错。
    /// `serde(default)` 让 LoRA 之前存的旧档（没有这个字段）照常读入。
    #[serde(default)]
    lora: Option<LoRAConfig>,
    /// 存档时的量化记录（`None` = 没量化过）。
    ///
    /// 只记**参数**（位宽 / 算法 / 字节口径），不记整数码：数据块里写的始终是 f32 权重，
    /// 于是任何现有加载路径（训练续跑、推理、合并）都不用知道量化这件事。
    /// 真要在部署时把显存压下来，加载端按这份记录原地重放一次 RTN 量化即可
    /// （见 [`requantize_after_load`]）。`serde(default)` 让量化之前的旧档照常读入。
    #[serde(default)]
    quant: Option<QuantMeta>,
}

/// 单个参数的元信息
#[derive(Serialize, Deserialize)]
struct ParamMeta {
    name: String,
    shape: Vec<usize>,
}

/// 加载 checkpoint 后得到的元信息（训练 / 评估 / 生成共用）
#[derive(Clone)]
pub struct Checkpoint {
    pub step: usize,
    pub best_val_loss: f32,
    pub model: GPTConfig,
    /// 存档时的 LoRA 形态；加载端据此在恢复参数**之前**重放注入（见 [`CkptHeader::lora`]）
    pub lora: Option<LoRAConfig>,
    /// 存档时的量化记录；`Some` 表示这份权重当初被量化过（权重本身仍是 f32，
    /// 见 [`requantize_after_load`]）
    pub quant: Option<QuantMeta>,
}

/// 保存 checkpoint：模型参数 + 优化器状态 + 元信息
///
/// 参数块按 [`GPT::named_parameters`] 的顺序写（LoRA 形态下会把 `*.lora_a` / `*.lora_b`
/// 一并带上，三段数据块仍然等长）——优化器状态必须覆盖同一组参数，格式才成立。
/// 冻结的主干参数照样写进去：它们的数值自始至终没变，存档因此是**自包含**的，
/// 单独一个文件就能加载推理，不必再去找基座 checkpoint。
///
/// 写出去的权重永远是 f32：量化过的层必须先 [`GPT::dequantize_weights`] 烘焙回来
/// （否则这里写的是量化**之前**的原始 f32，而头部却记着"已量化"——两份数据对不上，
/// 加载端重放量化得到的模型与内存里那个不是同一个）。真写错了会在下面直接断言拦下。
pub fn save(path: &str, model: &GPT, opt: &AdamW, step: usize, best_val_loss: f32) {
    assert!(
        !model.has_quant(),
        "保存前必须先 GPT::dequantize_weights()：存档里写的是 f32 权重，\
         而量化层的 weight 字段仍是最初的未量化数值（量化结果只在 quant 里）"
    );
    // 权重目录（默认 checkpoints/）不存在时自动创建，保证产物不会落到根目录
    crate::config::ensure_parent_dir(path);
    let named = model.named_parameters();
    let (opt_t, opt_m, opt_v) = opt.state();
    let params = named
        .iter()
        .map(|(name, t)| ParamMeta {
            name: name.clone(),
            shape: t.shape().to_vec(),
        })
        .collect();
    let header = CkptHeader {
        step,
        best_val_loss: best_val_loss.is_finite().then_some(best_val_loss),
        model: model.cfg.clone(),
        opt_t,
        params,
        lora: model.lora.clone(),
        quant: model.quant_meta(),
    };
    let json = serde_json::to_vec(&header).expect("序列化 checkpoint 头失败");

    let mut f = File::create(path).unwrap_or_else(|e| panic!("无法创建 checkpoint {path}: {e}"));
    let write = |r: std::io::Result<()>| {
        r.unwrap_or_else(|e| panic!("写入 checkpoint {path} 失败: {e}"))
    };
    write(f.write_all(MAGIC));
    write(f.write_all(&(json.len() as u32).to_le_bytes()));
    write(f.write_all(&json));
    // 三段数据块，顺序固定：参数 → 一阶动量 m → 二阶动量 v。
    // 每段内部按参数顺序拼接，长度都是「参数总元素数 × 4」。
    for (_, t) in &named {
        write_f32s(&mut f, path, &t.data_ref());
    }
    for m in opt_m {
        write_f32s(&mut f, path, m);
    }
    for v in opt_v {
        write_f32s(&mut f, path, v);
    }
}

/// 零拷贝地把 f32 切片按字节写出（参数 / 动量共用，避免逐元素 write_all 的系统调用开销）
fn write_f32s(f: &mut File, path: &str, data: &[f32]) {
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    f.write_all(bytes)
        .unwrap_or_else(|e| panic!("写入 checkpoint {path} 失败: {e}"));
}

/// 只读取 checkpoint 头（模型配置、步数、best loss），不加载参数。
/// 用于 eval / generate 先按 checkpoint 里的配置构造模型。
/// 只读头部 + JSON，不读参数数据，大模型下也能秒开。
pub fn load_header(path: &str) -> Checkpoint {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("无法打开 checkpoint {path}: {e}"));
    let (h, _) = read_head(&mut f, path);
    Checkpoint {
        step: h.step,
        best_val_loss: h.best_val_loss.unwrap_or(f32::INFINITY),
        model: h.model,
        lora: h.lora,
        quant: h.quant,
    }
}

/// 只加载参数，不涉及优化器（eval / generate 用）。
/// `model` 必须先按 checkpoint 里的配置构造好，参数按名字逐个恢复。
/// 动量数据块会被跳过（不做解码），所以推理端不会为用不到的优化器状态花内存。
pub fn load_params(path: &str, model: &GPT) -> Checkpoint {
    let (ckpt, metas, data, _) = read_file(path);
    restore_params(model, &metas, &data[..numel_total(&metas) * 4]);
    ckpt
}

/// 加载参数并恢复优化器状态（resume 用）
pub fn load_with_opt(path: &str, model: &GPT, opt: &mut AdamW) -> Checkpoint {
    let (ckpt, metas, data, opt_t) = read_file(path);
    let block = numel_total(&metas) * 4;
    restore_params(model, &metas, &data[..block]);
    let mut m = decode_f32s(&metas, &data[block..block * 2]);
    let mut v = decode_f32s(&metas, &data[block * 2..]);
    // 词表被扩大时（[`GPT::resize_vocab`]），存档里的动量只覆盖旧词表的那些行。
    // 多出来的行按"从未更新过"补零：AdamW 的 m=v=0 加上偏置校正，第一步就等价于
    // 用该行自己的梯度从头开始累积，不会污染旧行的历史。
    for (i, p) in opt.params().iter().enumerate() {
        if m[i].len() < p.numel() {
            m[i].resize(p.numel(), 0.0);
            v[i].resize(p.numel(), 0.0);
        }
    }
    opt.restore_state(opt_t, m, v);
    ckpt
}

/// 加载后按存档头里的记录**原地重建权重量化**。
///
/// 为什么只做 RTN：GPTQ 要 `H = XᵀX`、AWQ 要 `mean|x|`，两者都依赖**当次校准集**的激活
/// 统计，而 checkpoint 里没有（也不该有）——把统计量塞进存档会让文件带上"当时那个
/// 校准集"的痕迹，换个领域就等于用错的统计去补偿。所以这里只做"不需要任何额外信息"的
/// RTN：误差比 GPTQ 略大，但完全确定、与校准集无关。要拿到 GPTQ 的精度，正确做法是
/// 加载后在**目标领域**的数据上重跑一次 [`GPT::calibrate`] + [`GPT::quantize_weights`]。
///
/// `meta` 只用其中最"硬"的一项——位宽 `bits`。重建之后的模型自己的量化记录以本次
/// 实测结果为准（`method` 记为 [`QuantMethod::Rtn`]，因为确实是用 RTN 建的），
/// 存档里那份原始记录属于**那份存档**，不属于这个模型。
///
/// 调用前模型必须是 f32（刚 [`load_params`] 完就是），否则量化会作用在已经量化过的
/// 权重上，误差叠加一层。
pub fn requantize_after_load(model: &mut GPT, meta: &QuantMeta) {
    assert!(
        !model.has_quant(),
        "模型已经带量化权重了，请先 GPT::dequantize_weights() 再重建"
    );
    // 算法参数走默认（RTN 本就不看它们；换个 method 也不会改变这份默认的行为）
    model.quantize_weights(meta.bits, QuantMethod::Rtn, None, QuantOpts::default());
}

/// 参数总元素数：三段数据块的长度基准（每段 `total * 4` 字节）
fn numel_total(metas: &[ParamMeta]) -> usize {
    metas
        .iter()
        .map(|m| m.shape.iter().product::<usize>())
        .sum()
}

/// 按参数顺序把一段数据块解码成 `Vec<Vec<f32>>`（形状与元信息一致）
fn decode_f32s(metas: &[ParamMeta], bytes: &[u8]) -> Vec<Vec<f32>> {
    let mut out = Vec::with_capacity(metas.len());
    let mut pos = 0usize;
    for meta in metas {
        let numel = meta.shape.iter().product::<usize>();
        let mut data = vec![0.0f32; numel];
        for (j, item) in data.iter_mut().enumerate() {
            let start = pos + j * 4;
            *item = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        }
        pos += numel * 4;
        out.push(data);
    }
    assert_eq!(pos, bytes.len(), "优化器状态数据块长度不匹配（文件可能损坏）");
    out
}

/// 从文件流读取并解析 checkpoint 头部（魔数 + JSON 头），返回 (header, json 长度)。
/// 用 `read_exact` 而非切片索引，文件过短/损坏时给出可读错误而非越界 panic。
fn read_head(f: &mut File, path: &str) -> (CkptHeader, usize) {
    let mut magic = [0u8; 7];
    f.read_exact(&mut magic)
        .unwrap_or_else(|e| panic!("读取 {path} 头部失败（文件过短或损坏）: {e}"));
    if &magic != MAGIC {
        panic!(
            "checkpoint 魔数不匹配：{path} 不是当前格式的 checkpoint\n\
             （旧格式与当前实现不兼容，需要重新训练）"
        );
    }
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)
        .unwrap_or_else(|e| panic!("读取 {path} 头长度失败（文件过短或损坏）: {e}"));
    let json_len = u32::from_le_bytes(len_buf) as usize;
    let mut json = vec![0u8; json_len];
    f.read_exact(&mut json)
        .unwrap_or_else(|e| panic!("读取 {path} JSON 头失败（文件过短或损坏）: {e}"));
    let header: CkptHeader = serde_json::from_slice(&json)
        .unwrap_or_else(|e| panic!("解析 checkpoint 头失败: {e}"));
    (header, json_len)
}

/// 读取并解析整个 checkpoint 文件：头部 + 三段数据块（参数 / m / v），
/// 数据块原样返回字节，由调用方按需解码（推理端只解参数段，不碰动量）。
fn read_file(path: &str) -> (Checkpoint, Vec<ParamMeta>, Vec<u8>, usize) {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("无法打开 checkpoint {path}: {e}"));
    let (header, _json_len) = read_head(&mut f, path);

    let ckpt = Checkpoint {
        step: header.step,
        best_val_loss: header.best_val_loss.unwrap_or(f32::INFINITY),
        model: header.model,
        lora: header.lora,
        quant: header.quant,
    };

    // 读取并校验数据段总长度：参数 + m + v 三段等长
    let mut data = Vec::new();
    f.read_to_end(&mut data)
        .unwrap_or_else(|e| panic!("读取 {path} 参数数据失败: {e}"));
    let block = numel_total(&header.params) * 4;
    assert_eq!(
        data.len(),
        block * 3,
        "checkpoint 数据段长度不匹配：期望 {} 字节（参数+m+v 各 {}），实得 {}（文件可能损坏）",
        block * 3,
        block,
        data.len()
    );
    (ckpt, header.params, data, header.opt_t)
}

/// 词嵌入的名称。只有它允许"模型行数 > 存档行数"——
/// 词表扩展只能是"表尾追加行"，其它参数扩大都意味着配置不一致，必须报错。
const EMBEDDING_PARAM: &str = "tok_emb.table";

/// 存档里的 `tok_emb.table` 是否按**前缀行**恢复到模型上（模型词表更大）。
///
/// 判据：非首维形状完全一致，且模型首维不小于存档首维。这样"给已训模型加
/// BOS/EOS/PAD 三个特殊 token"不必重训——旧行的数值逐位保留，只有新增的三行
/// 取 [`GPT::resize_vocab`] 的初始化值，继续训练即可收敛。
fn is_vocab_extension(name: &str, meta: &ParamMeta, t: &Tensor) -> bool {
    name == EMBEDDING_PARAM
        && t.shape().len() == meta.shape.len()
        && t.shape()[1..] == meta.shape[1..]
        && t.shape()[0] >= meta.shape[0]
}

/// 按名字、形状把参数数据写回模型
fn restore_params(model: &GPT, metas: &[ParamMeta], bytes: &[u8]) {
    let named = model.named_parameters();
    assert_eq!(
        named.len(),
        metas.len(),
        "checkpoint 参数数量（{}）与模型（{}）不匹配，请检查配置是否一致",
        metas.len(),
        named.len()
    );
    // 先逐个校验名字与形状，避免形状不符时把错位的数据写进模型
    for ((name, t), meta) in named.iter().zip(metas) {
        assert_eq!(
            name, &meta.name,
            "参数名不匹配：checkpoint='{}' vs 模型='{}'",
            meta.name, name
        );
        if is_vocab_extension(name, meta, t) {
            continue; // 词表扩展：行数允许更多，下面只写前缀行
        }
        assert_eq!(
            t.shape(),
            &meta.shape[..],
            "参数 {} 形状不匹配：checkpoint={:?} vs 模型={:?}",
            name,
            meta.shape,
            t.shape()
        );
    }
    for ((name, t), data) in named.iter().zip(decode_f32s(metas, bytes)) {
        if t.numel() == data.len() {
            t.set_data(data);
        } else {
            // 词表扩展：前缀行照写，新增行保持模型里的初始化值
            let mut full = t.data_ref().clone();
            assert!(
                name == EMBEDDING_PARAM,
                "参数 {name} 的数据长度（{}）与模型（{}）不符",
                data.len(),
                full.len()
            );
            full[..data.len()].copy_from_slice(&data);
            t.set_data(full);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GPTConfig;
    use crate::module::Module;
    use crate::optim::Optimizer;
    use crate::quant::QBits;
    use crate::rng::Rng;
    use crate::tensor::Tensor;

    fn tmp_path(tag: &str) -> String {
        let mut p = std::env::temp_dir();
        p.push(format!("llm_ckpt_test_{tag}_{}.ckpt", std::process::id()));
        p.to_string_lossy().into_owned()
    }

    /// 构造一个"训练过几步"的模型 + 优化器：手动灌梯度并真实调 `opt.step()`，
    /// 这样 m / v / t 都是非平凡值，才能验证优化器状态是否被完整保存。
    fn trained_tiny(seed: u64) -> (GPT, AdamW) {
        let mut rng = Rng::new(seed);
        let model = GPT::new(GPTConfig::tiny(64), &mut rng);
        let mut opt = AdamW::new(1e-3, model.parameters(), 0.1);
        for k in 0..3 {
            for p in opt.params() {
                let mut g = p.grad.borrow_mut();
                for (i, x) in g.iter_mut().enumerate() {
                    *x = ((i + k) as f32 * 0.01).sin();
                }
            }
            opt.step();
        }
        (model, opt)
    }

    fn assert_bits_eq(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what} 长度不一致");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}] 不是逐位还原：{x} vs {y}");
        }
    }

    /// 存档 → 恢复应**逐位**一致：参数、一阶动量、二阶动量、优化器步数。
    /// 这是"精确续训"的前提，任何一处有损都会让续训起点与预期不符。
    #[test]
    fn test_save_load_roundtrip_is_bit_exact() {
        let (model, opt) = trained_tiny(11);
        let (opt_t, opt_m, opt_v) = opt.state();
        let (opt_t, opt_m, opt_v) = (opt_t, opt_m.to_vec(), opt_v.to_vec());

        let path = tmp_path("roundtrip");
        save(&path, &model, &opt, 42, 1.2345);

        // 换一个全新初始化的模型 + 优化器，从存档恢复
        let mut rng = Rng::new(999);
        let model2 = GPT::new(GPTConfig::tiny(64), &mut rng);
        let mut opt2 = AdamW::new(1e-3, model2.parameters(), 0.1);
        let ckpt = load_with_opt(&path, &model2, &mut opt2);
        let _ = std::fs::remove_file(&path);

        assert_eq!(ckpt.step, 42);
        assert_eq!(ckpt.best_val_loss, 1.2345);
        assert_eq!(ckpt.model.vocab_size, 64);

        for ((name, a), (_, b)) in model.named_parameters().iter().zip(model2.named_parameters()) {
            assert_bits_eq(&a.data_ref(), &b.data_ref(), &format!("参数 {name}"));
        }
        let (t2, m2, v2) = opt2.state();
        assert_eq!(t2, opt_t, "优化器步数未还原");
        for (i, (a, b)) in opt_m.iter().zip(m2).enumerate() {
            assert_bits_eq(a, b, &format!("一阶动量 m[{i}]"));
        }
        for (i, (a, b)) in opt_v.iter().zip(v2).enumerate() {
            assert_bits_eq(a, b, &format!("二阶动量 v[{i}]"));
        }
    }

    /// 推理路径（`load_params`）只解参数块、跳过动量块，参数同样逐位还原。
    #[test]
    fn test_load_params_skips_optimizer_state() {
        let (model, opt) = trained_tiny(23);
        let path = tmp_path("params_only");
        save(&path, &model, &opt, 7, 0.5);

        let mut rng = Rng::new(5);
        let model2 = GPT::new(GPTConfig::tiny(64), &mut rng);
        let ckpt = load_params(&path, &model2);
        let _ = std::fs::remove_file(&path);

        assert_eq!(ckpt.step, 7);
        for ((name, a), (_, b)) in model.named_parameters().iter().zip(model2.named_parameters()) {
            assert_bits_eq(&a.data_ref(), &b.data_ref(), &format!("参数 {name}"));
        }
    }

    /// NaN / Inf 应原样存活。JSON 没有 NaN / Infinity 字面量，`serde_json` 会把它们写成
    /// `null`，读回时反序列化失败；二进制数据块不受这个限制。
    #[test]
    fn test_non_finite_values_survive_roundtrip() {
        let mut rng = Rng::new(31);
        let model = GPT::new(GPTConfig::tiny(64), &mut rng);
        let opt = AdamW::new(1e-3, model.parameters(), 0.1);
        let targets: Vec<Tensor> = model.parameters();
        let mut d = targets[0].data_ref().to_vec();
        d[0] = f32::NAN;
        d[1] = f32::INFINITY;
        d[2] = f32::NEG_INFINITY;
        targets[0].set_data(d);

        let path = tmp_path("non_finite");
        save(&path, &model, &opt, 1, f32::INFINITY);

        let mut rng = Rng::new(77);
        let model2 = GPT::new(GPTConfig::tiny(64), &mut rng);
        let ckpt = load_params(&path, &model2);
        let _ = std::fs::remove_file(&path);

        assert_eq!(ckpt.best_val_loss, f32::INFINITY);
        let got = targets[0].data_ref();
        let restored = model2.parameters();
        let want = restored[0].data_ref();
        assert!(want[0].is_nan(), "NaN 未存活");
        assert_eq!(got[1..3], want[1..3], "Inf 未存活");
    }

    /// LoRA 存档：头部记下 rank/alpha，加载端据此重建适配层后参数才能按名字对齐。
    /// 冻结的主干照样写入，所以单个文件就是自包含的——不需要再额外指定基座 checkpoint。
    #[test]
    fn test_lora_checkpoint_roundtrip() {
        let mut rng = Rng::new(13);
        let mut model = GPT::new(GPTConfig::tiny(64), &mut rng);
        let lora = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        model.apply_lora(&lora, &mut rng);
        // 让适配层带上非零数值：B 初始为 0，只灌零梯度的话"存没存"看不出来
        let mut opt = AdamW::new(1e-2, model.parameters(), 0.1);
        for p in opt.params() {
            if p.requires_grad() {
                p.grad
                    .borrow_mut()
                    .iter_mut()
                    .enumerate()
                    .for_each(|(i, g)| *g = (i as f32 * 0.05).sin());
            }
        }
        opt.step();

        let path = tmp_path("lora");
        save(&path, &model, &opt, 5, 2.5);

        // 头部必须带上 LoRA 形态，否则加载端不知道要重建适配层
        let head = load_header(&path);
        assert_eq!(head.lora.as_ref().map(|l| l.rank), Some(4));
        assert_eq!(head.lora.as_ref().map(|l| l.alpha), Some(8.0));

        // 按头部记录重建适配层后再加载：参数（含 lora_a / lora_b）逐位还原
        let mut rng2 = Rng::new(77);
        let mut model2 = GPT::new(GPTConfig::tiny(64), &mut rng2);
        model2.apply_lora(&lora, &mut rng2);
        let ckpt = load_params(&path, &model2);

        assert_eq!(ckpt.lora.as_ref().map(|l| l.rank), Some(4));
        let np = model.named_parameters();
        let np2 = model2.named_parameters();
        assert_eq!(np.len(), np2.len());
        assert!(np.iter().any(|(n, _)| n.ends_with(".lora_b")));
        for ((name, a), (_, b)) in np.iter().zip(np2.iter()) {
            assert_bits_eq(&a.data_ref(), &b.data_ref(), &format!("参数 {name}"));
        }

        // 反过来：不重建适配层、拿普通模型直接加载，参数名对不上必须报错，
        // 而不是悄悄装进去一半。这就是头部要记 LoRA 形态的原因。
        let plain = GPT::new(GPTConfig::tiny(64), &mut Rng::new(78));
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {})); // 这条断言是**故意**触发 panic 的，别让它刷屏
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            load_params(&path, &plain);
        }));
        std::panic::set_hook(hook);
        let _ = std::fs::remove_file(&path);
        assert!(err.is_err(), "没注入适配层就加载 LoRA 存档，应该直接报错");
    }

    /// LoRA 之前存的旧档没有 `lora` 字段，必须照常读入为 None（serde default）
    #[test]
    fn test_header_without_lora_field_is_accepted() {
        let json = r#"{"step":3,"best_val_loss":1.0,"model":{"vocab_size":64},"opt_t":3,"params":[]}"#;
        let h: CkptHeader = serde_json::from_str(json).expect("旧档应能读入");
        assert!(h.lora.is_none());
        assert_eq!(h.step, 3);
    }

    /// 量化之前存的旧档没有 `quant` 字段，同样必须照常读入（serde default）。
    /// 这是"加字段不能把已有存档读废"的同一条约束，两个字段各测一次。
    #[test]
    fn test_header_without_quant_field_is_accepted() {
        let json = r#"{"step":4,"best_val_loss":1.0,"model":{"vocab_size":64},"opt_t":4,"params":[]}"#;
        let h: CkptHeader = serde_json::from_str(json).expect("旧档应能读入");
        assert!(h.quant.is_none());
        assert_eq!(h.step, 4);
    }

    /// 量化元信息必须穿过存档往返：存档里写的是 f32 权重（逐位还原），
    /// 头部单独记下"当初怎么量化的"，加载端据此原地重放量化。
    #[test]
    fn checkpoint_roundtrip_preserves_quant_meta() {
        let mut rng = Rng::new(53);
        let mut model = GPT::new(GPTConfig::tiny(64), &mut rng);
        let reports =
            model.quantize_weights(QBits::Int8, QuantMethod::Rtn, None, QuantOpts::default());
        assert!(model.has_quant());
        // 逐层报告必须覆盖全部投影：RTN 不需要校准集，没有任何层该被跳过
        let summary = model.quant_summary(reports);
        assert!(!summary.layers.is_empty(), "tiny 模型也该有可量化的投影");
        assert_eq!(summary.skipped, 0, "RTN 不需要校准集，不该有层被跳过");
        let (orig, quantized) = model.weight_bytes();
        assert_eq!((summary.f32_bytes, summary.quant_bytes), (orig, quantized));
        assert!(quantized < orig, "int8 量化后字节数应少于 f32");

        // 存档前烘焙回 f32：存档里永远是 f32，量化只在内存里活过一段
        model.dequantize_weights();
        assert!(!model.has_quant(), "烘焙后不应再留着量化表示");

        let opt = AdamW::new(1e-3, model.parameters(), 0.1);
        let path = tmp_path("quant_meta");
        save(&path, &model, &opt, 12, 0.8);

        // 只读头就能看到量化记录（eval / 部署据此决定要不要重建量化）
        let head = load_header(&path);
        let meta = head.quant.expect("头部应记下量化元信息");
        assert_eq!(meta.bits, QBits::Int8);
        assert_eq!(meta.method, QuantMethod::Rtn);
        assert_eq!(meta.orig_bytes, orig);
        assert_eq!(meta.quant_bytes, quantized);

        // 参数逐位还原（量化结果已经烘焙进 f32，所以这里比的是量化后的数值）
        let mut rng2 = Rng::new(54);
        let mut model2 = GPT::new(GPTConfig::tiny(64), &mut rng2);
        let ckpt = load_params(&path, &model2);
        assert_eq!(ckpt.quant.map(|m| m.bits), Some(QBits::Int8));
        for ((name, a), (_, b)) in model.named_parameters().iter().zip(model2.named_parameters()) {
            assert_bits_eq(&a.data_ref(), &b.data_ref(), &format!("参数 {name}"));
        }

        // 按头部记录原地重建：模型重新带上量化表示，省下的字节与最初一致
        requantize_after_load(&mut model2, &meta);
        assert!(model2.has_quant(), "重建后应重新带上量化表示");
        let (o2, q2) = model2.weight_bytes();
        assert_eq!((o2, q2), (orig, quantized), "同一位宽重建，字节口径应一致");
        let _ = std::fs::remove_file(&path);
    }

    /// 词表扩展：存档词表 64、模型词表 67（多了 BOS/EOS/PAD）时——
    /// 旧行逐位恢复、新行保持初始化值，并且可以接着续训（动量对新增行补零）。
    #[test]
    fn test_vocab_extension_restores_prefix_rows_and_pads_optimizer() {
        let (model, opt) = trained_tiny(41);
        let path = tmp_path("vocab_ext");
        save(&path, &model, &opt, 9, 0.7);
        let (opt_t, _, _) = opt.state();

        // 模型按更大的词表构造，再把 embedding 扩到 67 行
        let mut rng = Rng::new(123);
        let mut model2 = GPT::new(GPTConfig::tiny(64), &mut rng);
        model2.resize_vocab(67, &mut rng);
        assert_eq!(model2.cfg.vocab_size, 67);

        let mut opt2 = AdamW::new(1e-3, model2.parameters(), 0.1);
        let ckpt = load_with_opt(&path, &model2, &mut opt2);
        let _ = std::fs::remove_file(&path);
        assert_eq!(ckpt.step, 9);

        // 旧行逐位一致；新增的三行是 resize_vocab 的初始化值（非零、有限）
        let d = model2.cfg.n_embd;
        let old = model.named_parameters();
        let new = model2.named_parameters();
        for ((name, a), (_, b)) in old.iter().zip(&new) {
            let (ad, bd) = (a.data_ref(), b.data_ref());
            if name == EMBEDDING_PARAM {
                continue; // 行数变了（64 -> 67），前缀行在下面单独比对
            }
            assert_bits_eq(&ad[..], &bd[..], &format!("参数 {name}"));
        }
        let emb_old = &old.iter().find(|(n, _)| n == EMBEDDING_PARAM).unwrap().1;
        let emb_new = &new.iter().find(|(n, _)| n == EMBEDDING_PARAM).unwrap().1;
        assert_bits_eq(&emb_old.data_ref()[..], &emb_new.data_ref()[..64 * d], "词嵌入旧行");
        let emb_d = emb_new.data_ref();
        assert_eq!(emb_d.len(), 67 * d);
        assert!(
            emb_d[64 * d..].iter().all(|v| v.is_finite() && *v != 0.0),
            "新增行应保留初始化值"
        );

        // 续训：优化器步数还原，新增行的动量是从零补齐的（长度已对齐）
        let (t2, m2, v2) = opt2.state();
        assert_eq!(t2, opt_t);
        let emb_idx = new.iter().position(|(n, _)| n == EMBEDDING_PARAM).unwrap();
        assert_eq!(m2[emb_idx].len(), 67 * d);
        assert!(v2[emb_idx][64 * d..].iter().all(|x| *x == 0.0));
    }

    /// 非词表参数形状不符时必须照旧报错——扩展只开放给 `tok_emb.table`。
    #[test]
    fn test_shape_mismatch_outside_embedding_still_panics() {
        let (model, opt) = trained_tiny(43);
        let path = tmp_path("shape_guard");
        save(&path, &model, &opt, 3, 1.0);

        // 改 n_embd 会让 c_q.weight 等一堆参数形状不符
        let mut cfg = GPTConfig::tiny(64);
        cfg.n_embd = 32; // n_head=4，head_dim 仍整除
        let mut rng = Rng::new(9);
        let bad = GPT::new(cfg, &mut rng);
        let hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            load_params(&path, &bad);
        }));
        std::panic::set_hook(hook);
        let _ = std::fs::remove_file(&path);
        assert!(err.is_err(), "配置不一致时必须报错，而不是悄悄装进去一半");
    }
}
