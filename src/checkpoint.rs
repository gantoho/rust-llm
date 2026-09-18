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

use crate::model::{GPT, GPTConfig};
use crate::optim::AdamW;
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
}

/// 保存 checkpoint：模型参数 + 优化器状态 + 元信息
pub fn save(path: &str, model: &GPT, opt: &AdamW, step: usize, best_val_loss: f32) {
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
    let m = decode_f32s(&metas, &data[block..block * 2]);
    let v = decode_f32s(&metas, &data[block * 2..]);
    opt.restore_state(opt_t, m, v);
    ckpt
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
        assert_eq!(
            t.shape(),
            &meta.shape[..],
            "参数 {} 形状不匹配：checkpoint={:?} vs 模型={:?}",
            name,
            meta.shape,
            t.shape()
        );
    }
    for ((_, t), data) in named.iter().zip(decode_f32s(metas, bytes)) {
        t.set_data(data);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GPTConfig;
    use crate::module::Module;
    use crate::optim::Optimizer;
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
}
