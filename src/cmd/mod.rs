//! cmd 模块：各子命令实现从 crate 根（main.rs）机械搬迁而来，逻辑保持不变。
//!
//! - `core`：核心命令（train / eval / quant / generate / sft / finetune / preset / bench）
//! - `deploy`：服务相关命令（serve / chat）
//! - `experiments`：实验类命令（scaling / moe / distributed / align / rag / speculative / distill / fp8 / fp4 / csa / mla / demo）
//!
//! 本文件另存放跨文件共享的辅助函数。

pub(crate) mod core;
pub(crate) mod deploy;
pub(crate) mod experiments;

use crate::cli::RopeArgs;
use crate::data::load_text;
use crate::model::Transformer;
use crate::quant::QBits;
use crate::rope::RopeScaling;
use crate::rng::Rng;
use crate::sample::{KvOpts, StopReason};
use crate::tokenizer::Tokenizer;

use crate::checkpoint;
use crate::config;

/// 命令行上的 KV cache 量化位宽名字 -> 枚举（`none` = 不量化）
pub(crate) fn kv_bits_from_name(name: &str) -> Option<QBits> {
    match name {
        "none" => None,
        "int8" => Some(QBits::Int8),
        "int4" => Some(QBits::Int4),
        other => panic!("未知的 KV cache 量化位宽 '{other}'（可选：none / int8 / int4）"),
    }
}

/// 命令行上的 RoPE 外推方式名字 -> 枚举（未给出时返回 `None` = 沿用 checkpoint）
pub(crate) fn rope_scaling_from_cli(args: &RopeArgs) -> Option<RopeScaling> {
    match args.rope_scaling.as_deref()? {
        "linear" => Some(RopeScaling::Linear { factor: args.rope_factor }),
        "ntk" => Some(RopeScaling::Ntk { factor: args.rope_factor }),
        // beta_fast / beta_slow / mscale 用论文与社区实践的默认值（32 / 1 / 1.0）：
        // 它们决定"哪些维度算高频"的分段边界，除非有明确理由，调到它们只会让结果更难复现。
        "yarn" => Some(RopeScaling::Yarn {
            factor: args.rope_factor,
            beta_fast: 32.0,
            beta_slow: 1.0,
            mscale: 1.0,
        }),
        other => panic!("未知的 RoPE 外推方式 '{other}'（可选：linear / ntk / yarn）"),
    }
}

/// 把命令行上的 RoPE 覆盖项套到已加载的模型上，返回一行说明（没有覆盖项时返回 `None`）。
///
/// 注意拿 `block_size`（= checkpoint 里记录的**训练**窗口）当 YaRN 的分段基准，
/// 而不是覆盖后的 `--max-ctx`：模型"见过"的位置差上限没有因为推理放宽而变大。
pub(crate) fn apply_rope_override(model: &mut Transformer, args: &RopeArgs) -> Option<String> {
    let scaling = rope_scaling_from_cli(args);
    let base = args.rope_base;
    let max_ctx = args.max_ctx;
    if scaling.is_none() && base.is_none() && max_ctx.is_none() {
        return None;
    }
    let train_ctx = model.cfg.rope_train_ctx();
    let new_ctx = max_ctx.unwrap_or(train_ctx);
    let new_base = base.unwrap_or(model.cfg.rope_base);
    let old_ctx = model.set_rope(new_base, scaling.unwrap_or(model.cfg.rope_scaling), train_ctx, new_ctx);
    Some(format!(
        "上下文窗口 {old_ctx} → {new_ctx}，base = {new_base}，外推 = {}",
        model.cfg.rope_scaling.describe()
    ))
}

/// 读取文本文件
pub(crate) fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("无法读取数据文件 {path}: {e}"))
}

/// 按**字符**（不是字节）截断到至多 `n` 个字符。
///
/// 必须按字符切：语料里有中文，按字节切会把一个三字节的汉字劈成两半，
/// 之后 `String` 就不是合法 UTF-8 了（`&text[..i]` 会直接 panic）。
pub(crate) fn truncate_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Windows 控制台默认 GBK，把输出代码页切到 UTF-8，避免中文乱码
#[cfg(windows)]
pub(crate) fn init_console_utf8() {
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleOutputCP(65001);
        windows_sys::Win32::System::Console::SetConsoleCP(65001);
    }
}

/// 解析 checkpoint 路径：None 时取 out_dir/latest.ckpt
pub(crate) fn resolve_ckpt<'a>(ckpt: Option<&'a str>, out_dir: &str) -> std::borrow::Cow<'a, str> {
    match ckpt {
        Some(p) => std::borrow::Cow::Borrowed(p),
        None => std::borrow::Cow::Owned(format!("{}/latest.ckpt", out_dir)),
    }
}

/// 从 checkpoint 加载模型和分词器的通用辅助函数（推理 / 微调共用）
///
/// 分词器加载策略（按优先级）：
/// 1. `tokenizer_path` 参数（用户通过 --tokenizer 显式指定）
/// 2. `tcfg.tokenizer_file`（config/config.json 中配置的路径）
/// 3. 自动查找 `{out_dir}/tokenizer.json`（训练时自动保存的）
/// 4. 最后才从语料训练（需要 train_file 存在）
pub(crate) fn load_model_and_tokenizer(
    ckpt_path: &str,
    tcfg: &config::TrainConfig,
    seed: u64,
    tokenizer_path: Option<&str>,
) -> (Transformer, Tokenizer, checkpoint::Checkpoint) {
    let ckpt = checkpoint::load_header(ckpt_path);
    let tokenizer = if let Some(path) = tokenizer_path {
        let loaded = Tokenizer::load(path);
        if ckpt.model.vocab_size != 0 {
            // 只要求"分词器的 id 都落在模型词表内"，**不要求两者相等**：
            // HF 模型常把 `vocab_size` 补到 128 的倍数（Qwen2.5：真实词表 151665、
            // config 写 151936），多出来的那几行是永不会被采到的占位 id。
            // 反过来若分词器更大，采样就可能产出越界 id —— 那才是必须拦下的情况。
            assert!(
                loaded.vocab_size() <= ckpt.model.vocab_size,
                "分词器词表（{}）大于 checkpoint 词表（{}）：会采出越界 token",
                loaded.vocab_size(),
                ckpt.model.vocab_size
            );
        }
        logln!("分词器：从 {} 加载（{}，词表 {}）", path, loaded.kind(), loaded.vocab_size());
        loaded
    } else {
        load_tokenizer_for_inference(tcfg, ckpt.model.vocab_size)
    };
    let mut rng = Rng::new(seed);
    let mut model = Transformer::new(ckpt.model.clone(), &mut rng);
    // LoRA 存档：先按头部记录重放注入（冻结主干 + 每层 Q/K/V 挂适配器），再恢复参数。
    // 顺序不能反——参数是**按名字**逐个对齐的，模型里少一套 `*.lora_a` / `*.lora_b`
    // 就会在 restore_params 处断言失败（见 [`checkpoint::CkptHeader`] 的 lora 字段）。
    if let Some(lora) = ckpt.lora.as_ref() {
        model.apply_lora(lora, &mut rng);
    }
    checkpoint::load_params(ckpt_path, &model);
    // 存档头里的量化记录只是"这份权重当初被量化过"的注记：数据块里始终是 f32，
    // 加载出来的模型也是普通 f32 模型（见 [`checkpoint::save`]）。打印出来是为了让
    // "这次推理为什么没省显存"一目了然——想真省，得按这条记录重建量化。
    if let Some(meta) = ckpt.quant {
        logln!(
            "[quant] 该存档记录为 {} {}（{} → {} 字节）；权重是 f32，\
             需要省显存时按此记录重建（checkpoint::requantize_after_load）",
            meta.bits.name(),
            meta.method.name(),
            meta.orig_bytes,
            meta.quant_bytes
        );
    }
    (model, tokenizer, ckpt)
}

/// 推理命令共用：`--merge-lora` 时把适配层增量**并入主干权重**再推理。
///
/// 只影响速度、不影响数值（`W + ΔW` 与"两条支路相加"在数学上同一个结果，只有浮点
/// 累加顺序的差异）。但它是**不可逆**的——合并后模型里再没有 A/B，也就无法链式续训，
/// 所以只在推理命令上显式开启，且不写回任何存档。
pub(crate) fn maybe_merge_lora(model: &mut Transformer, merge_lora: bool) {
    if !merge_lora {
        return;
    }
    if !model.has_lora() {
        logln!("[warn] --merge-lora 无效：该存档是普通模型，本来就没有适配层");
        return;
    }
    let adapters = model.lora_parameters().len() / 2; // 每个适配层一对 A/B
    model.merge_lora();
    logln!("LoRA 合并：{adapters} 个适配层的增量已并入主干权重（前向不再有额外矩阵乘）");
}

/// 推理 / 评估族命令共用的模型装配序列（generate / serve / chat 四步全用，
/// eval 不传 RoPE）：解析 checkpoint 路径 → 加载模型与分词器 → 可选合并 LoRA
/// → 可选 RoPE 覆盖。
///
/// 各步顺序不能换：先建结构再按名恢复参数（LoRA 注入在 [`load_model_and_tokenizer`]
/// 内部完成）；RoPE 覆盖必须在调用方读 `model.cfg.block_size` 分配上下文预算
/// **之前**生效（否则 `block_size` 还是训练窗口，`--max-ctx` 扩出来的空间会被
/// 白白裁掉），也要在 runlog 打「模型结构」之前生效（`ckpt.model` 只记了 head 数
/// 与维度，上下文窗口与频率表不在里面，不打印出来就没法从日志复现这次运行）。
///
/// 返回 `(模型, 分词器, checkpoint 头, 实际 ckpt 路径, RoPE 覆盖说明)`；
/// `rope_opt = None` 表示该命令不支持长度外推（eval），末项即为 `None`。
#[allow(clippy::type_complexity)] // 五元组一步解构到位，调用方才不必重写后续数百行
pub(crate) fn resolve_model_spec<'a>(
    ckpt: Option<&'a str>,
    tcfg: &config::TrainConfig,
    seed: u64,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    rope_opt: Option<&RopeArgs>,
) -> (
    Transformer,
    Tokenizer,
    checkpoint::Checkpoint,
    std::borrow::Cow<'a, str>,
    Option<String>,
) {
    let ckpt_path = resolve_ckpt(ckpt, &tcfg.out_dir);
    let (mut model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    maybe_merge_lora(&mut model, merge_lora);
    let rope_note = rope_opt.and_then(|r| apply_rope_override(&mut model, r));
    (model, tokenizer, ckpt, ckpt_path, rope_note)
}

/// 推理时加载分词器：优先从文件加载，避免依赖语料
pub(crate) fn load_tokenizer_for_inference(tcfg: &config::TrainConfig, expect_vocab: usize) -> Tokenizer {
    // 1. 用户显式配置了 tokenizer_file
    if let Some(ref path) = tcfg.tokenizer_file {
        let loaded = Tokenizer::load(path);
        if expect_vocab != 0 {
            assert_eq!(loaded.vocab_size(), expect_vocab,
                "分词器词表（{}）与 checkpoint（{}）不一致", loaded.vocab_size(), expect_vocab);
        }
        logln!("分词器：从 {} 加载（{}，词表 {}）", path, loaded.kind(), loaded.vocab_size());
        return loaded;
    }
    // 2. 自动查找训练时保存的 tokenizer.json
    let auto_path = format!("{}/tokenizer.json", tcfg.out_dir);
    if std::path::Path::new(&auto_path).exists() {
        let loaded = Tokenizer::load(&auto_path);
        if expect_vocab != 0 {
            assert_eq!(loaded.vocab_size(), expect_vocab,
                "分词器词表（{}）与 checkpoint（{}）不一致", loaded.vocab_size(), expect_vocab);
        }
        logln!("分词器：从 {} 加载（{}，词表 {}）", auto_path, loaded.kind(), loaded.vocab_size());
        return loaded;
    }
    // 3. 都没有，从语料训练（兜底）
    let train_text = load_text(&tcfg.train_file);
    build_tokenizer(tcfg, &train_text, expect_vocab)
}

#[cfg(not(windows))]
pub(crate) fn init_console_utf8() {}

/// 按配置重建分词器（char / bpe）。
/// 优先从文件加载（如果 tokenizer_file 配置了），否则从语料训练。
/// `expect_vocab = 0` 表示不校验（训练时词表由分词器决定）。
pub(crate) fn build_tokenizer(tcfg: &config::TrainConfig, train_text: &str, expect_vocab: usize) -> Tokenizer {
    let tok = if let Some(ref path) = tcfg.tokenizer_file {
        if std::path::Path::new(path).exists() {
            let loaded = Tokenizer::load(path);
            logln!("已从 {} 加载分词器（{}，词表 {}）", path, loaded.kind(), loaded.vocab_size());
            loaded
        } else {
            logln!("分词器文件 {} 不存在，从语料训练新分词器", path);
            Tokenizer::from_name(&tcfg.tokenizer, train_text, tcfg.bpe_vocab)
        }
    } else {
        Tokenizer::from_name(&tcfg.tokenizer, train_text, tcfg.bpe_vocab)
    };
    if expect_vocab != 0 {
        assert_eq!(
            tok.vocab_size(),
            expect_vocab,
            "分词器词表（{}）与模型/checkpoint（{}）不一致：请确认 config/config.json 与训练时保持一致",
            tok.vocab_size(),
            expect_vocab
        );
    }
    tok
}

/// 把 KV cache 设置说成一句人话（日志与命令行回显共用）
pub(crate) fn describe_kv(kv: &KvOpts) -> String {
    if !kv.enable {
        return "关（每次全量前向）".to_string();
    }
    let bits = match kv.bits {
        None => "f32".to_string(),
        Some(b) => format!("{b:?} 量化"),
    };
    format!("开（{bits}，Attention Sink {})", kv.sink)
}

/// 把收尾原因说成一句人话。生成段为空时终端上只有一行空白，
/// 不写清楚"模型没说话"还是"程序没跑"，用户无从判断（见 [`StopReason`]）。
pub(crate) fn describe_stop(reason: StopReason) -> String {
    match reason {
        StopReason::Eos => "模型采到 EOS，自己收了尾".to_string(),
        StopReason::StopMark(mark) => format!("模型开口就是停止标记「{mark}」，没答就转下一轮"),
        StopReason::MaxNew => "生成到 --max-new 上限，内容全为空白".to_string(),
    }
}

/// 收尾原因的短标签，**每一轮**都写进运行日志。
///
/// 与 [`describe_stop`] 分开写：那句长话是给"空输出"解释用的，
/// "开口就是停止标记「用户：」，没答就转下一轮"套在一段有内容的回答上是错的。
/// 只记空输出的那几轮时，非空回答究竟是采到 EOS 还是命中停止标记停下的就查不到——
/// 而这两种收尾写进历史的方式并不相同（见 `cmd_chat` 的历史更新）。
pub(crate) fn stop_label(reason: StopReason) -> String {
    match reason {
        StopReason::Eos => "采到 EOS".to_string(),
        StopReason::StopMark(mark) => format!("命中停止标记「{mark}」"),
        StopReason::MaxNew => "跑满 --max-new 被截断".to_string(),
    }
}
