//! 从零实现大语言模型（纯 Rust，不依赖深度学习框架）
//!
//! 用法（cli 子命令）：
//! - `cargo run --release -- train    --config config/config.json [--resume checkpoints/latest.ckpt]`
//! - `cargo run --release -- eval     --config config/config.json [--ckpt checkpoints/latest.ckpt]`
//! - `cargo run --release -- generate --config config/config.json [--ckpt ...] --prompt "Once" --max-new 100`
//! - `cargo run --release -- chat     --config config/config.json [--ckpt ...] [--system "..."]`
//! - `cargo run --release -- sft      --config config/config.json --pretrained ckpt [--sft-file "..."]`
//! - `cargo run --release -- finetune --config config/config.json --pretrained ckpt [--lora-rank 16]`
//! - `cargo run --release -- preset   [--name small] [--output config/config.json]`
//! - `cargo run --release -- demo`    # 端到端演示（XOR / BPE / 内置语料小 Transformer）
//!
//! 配套教程文档见 `docs/` 目录。
//!
//! 每次训练 / 推理都会自动在 `logs/` 下生成一份运行日志（文件名含操作名与毫秒级时间），
//! 内容包含完整命令行、完整配置与全部过程输出，见 [`runlog`]。

// `runlog` 里的 `logln!` 宏要覆盖后面所有模块，必须最先声明并带 #[macro_use]
#[macro_use]
mod runlog;

mod align;
mod attention;
mod autograd;
mod checkpoint;
mod cli;
mod config;
mod data;
mod distributed;
#[cfg(feature = "gpu")]
mod gpu;
mod layers;
mod loss;
mod model;
mod moe;
mod module;
mod optim;
mod quant;
mod rag;
mod rng;
mod rope;
mod sample;
mod scaling;
mod speculative;
mod tensor;
mod tokenizer;
mod train;

use cli::{AlignArgs, Cli, Cmd, DistArgs, RagArgs, RopeArgs, SpecArgs};
use autograd::clear_tape;
use config::Config;
use data::{
    BatchSource, CORPUS, DataLoader, SFT_ASSISTANT, SFT_END, SFT_USER, SftLoader, load_documents,
    load_text, load_texts,
};
use layers::{Linear, tanh};
use loss::cross_entropy_loss;
use model::{Transformer, TransformerConfig};
use module::Module;
use optim::{AdamW, Optimizer, SGD};
use quant::{CalibOpts, HessOpts, QBits, QuantMethod, QuantOpts};
use rng::Rng;
use rope::RopeScaling;
use sample::{KvOpts, SampleOpts, generate, probs_from_logits, sample_from_probs};
use tensor::Tensor;
use tokenizer::{BPETokenizer, CharTokenizer, Tokenizer};

/// 命令行上的 KV cache 量化位宽名字 -> 枚举（`none` = 不量化）
fn kv_bits_from_name(name: &str) -> Option<QBits> {
    match name {
        "none" => None,
        "int8" => Some(QBits::Int8),
        "int4" => Some(QBits::Int4),
        other => panic!("未知的 KV cache 量化位宽 '{other}'（可选：none / int8 / int4）"),
    }
}

/// 命令行上的权重位宽名字 -> 枚举（`quant` 子命令用；这里不存在"不量化"的选项，
/// 因为不量化就不该走这个子命令）
fn q_bits_from_name(name: &str) -> QBits {
    match name {
        "int8" => QBits::Int8,
        "int4" => QBits::Int4,
        other => panic!("未知的量化位宽 '{other}'（可选：int8 / int4）"),
    }
}

/// 命令行上的量化算法名字 -> 枚举
fn q_method_from_name(name: &str) -> QuantMethod {
    match name {
        "rtn" => QuantMethod::Rtn,
        "hess" => QuantMethod::Hess,
        "awq" => QuantMethod::Awq,
        other => panic!("未知的量化算法 '{other}'（可选：rtn / hess / awq）"),
    }
}

/// 命令行上的布尔字面量 -> `bool`（只认 `true` / `false`）。
///
/// 不接受 `1` / `yes` / 空串之类的变体：量化参数直接决定产出的权重，
/// 一个拼错的写法应当当场报错，而不是被静默当成"没给"从而悄悄用上默认值。
fn bool_from_name(flag: &str, name: &str) -> bool {
    match name {
        "true" => true,
        "false" => false,
        other => panic!("{flag} 只接受 true / false，收到 '{other}'"),
    }
}

/// 命令行上的 RoPE 外推方式名字 -> 枚举（未给出时返回 `None` = 沿用 checkpoint）
fn rope_scaling_from_cli(args: &RopeArgs) -> Option<RopeScaling> {
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
fn apply_rope_override(model: &mut Transformer, args: &RopeArgs) -> Option<String> {
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

fn main() {
    runlog::mark_start(); // 记下命令开始执行的时刻，作为运行日志总耗时的基准
    init_console_utf8();
    #[cfg(feature = "gpu")]
    gpu::init();
    let cli = Cli::parse_args();
    match cli.cmd {
        Cmd::Train { config, resume } => cmd_train(&config, resume.as_deref()),
        Cmd::Eval { config, ckpt, tokenizer, merge_lora } => {
            cmd_eval(&config, ckpt.as_deref(), tokenizer.as_deref(), merge_lora)
        }
        Cmd::Generate {
            config,
            ckpt,
            tokenizer,
            merge_lora,
            prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_window,
            seed,
            no_kv_cache,
            kv_bits,
            kv_sink,
            beam,
            length_penalty,
            rope,
        } => cmd_generate(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
            merge_lora,
            &prompt,
            max_new,
            SampleOpts {
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                repetition_window,
                stop: &[],
            },
            seed,
            KvOpts {
                enable: !no_kv_cache,
                sink: kv_sink,
                bits: kv_bits_from_name(&kv_bits),
            },
            beam,
            length_penalty,
            rope,
        ),
        Cmd::Chat {
            config,
            ckpt,
            tokenizer,
            merge_lora,
            system,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_window,
            max_new,
            kv_bits,
            kv_sink,
            seed,
            prompt_format,
            rope,
        } => cmd_chat(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
            merge_lora,
            &system,
            SampleOpts {
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                repetition_window,
                stop: &[],
            },
            max_new,
            KvOpts::on(kv_sink, kv_bits_from_name(&kv_bits)),
            seed,
            &prompt_format,
            rope,
        ),
        Cmd::Sft {
            config,
            pretrained,
            sft_file,
            steps,
            lr,
            out_dir,
        } => cmd_sft(
            &config,
            &pretrained,
            sft_file.as_deref(),
            steps,
            lr,
            out_dir.as_deref(),
        ),
        Cmd::Finetune {
            config,
            pretrained,
            lora_rank,
            lora_alpha,
            lora_targets,
            resume_lora,
            sft_file,
            steps,
            lr,
            out_dir,
        } => cmd_finetune(
            &config,
            &pretrained,
            lora_rank,
            lora_alpha,
            lora_targets.as_deref(),
            resume_lora,
            steps,
            lr,
            sft_file.as_deref(),
            out_dir.as_deref(),
        ),
        Cmd::Preset { name, output } => cmd_preset(&name, &output),
        Cmd::Demo => run_demo(),
        Cmd::Bench { steps, gen_tokens } => cmd_bench(steps, gen_tokens),
        Cmd::Scaling {
            config,
            sizes,
            steps,
            data_multiples,
            batch_size,
            block_size,
            lr,
            seed,
            budget,
            gpu_tflops,
            n_gpu,
            mfu,
            out,
        } => cmd_scaling(
            &config,
            &sizes,
            steps,
            &data_multiples,
            batch_size,
            block_size,
            lr,
            seed,
            budget,
            gpu_tflops,
            n_gpu,
            mfu,
            out.as_deref(),
        ),
        Cmd::Moe {
            experts,
            top_k,
            steps,
            batch_size,
            block_size,
            lr,
            n_embd,
            n_layer,
            aux_coef,
            z_loss,
            capacity_factors,
            seed,
        } => cmd_moe(
            &experts,
            top_k,
            steps,
            batch_size,
            block_size,
            lr,
            n_embd,
            n_layer,
            aux_coef,
            z_loss,
            &capacity_factors,
            seed,
        ),
        Cmd::Quant {
            config,
            ckpt,
            tokenizer,
            bits,
            method,
            calib_file,
            calib_samples,
            calib_tokens,
            act_order,
            damp,
            block,
            alpha,
            out,
            eval,
        } => cmd_quant(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
            &bits,
            &method,
            calib_file.as_deref(),
            calib_samples,
            calib_tokens,
            &act_order,
            damp,
            block,
            alpha,
            out.as_deref(),
            eval,
        ),
        Cmd::Distributed { dist } => cmd_distributed(&dist),
        Cmd::Align { align } => cmd_align(&align),
        Cmd::Rag { rag } => cmd_rag(&rag),
        Cmd::Speculative { spec } => cmd_speculative(&spec),
    }
}

/// 读取文本文件
fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("无法读取数据文件 {path}: {e}"))
}

/// 按**字符**（不是字节）截断到至多 `n` 个字符。
///
/// 必须按字符切：语料里有中文，按字节切会把一个三字节的汉字劈成两半，
/// 之后 `String` 就不是合法 UTF-8 了（`&text[..i]` 会直接 panic）。
fn truncate_chars(s: &str, n: usize) -> &str {
    match s.char_indices().nth(n) {
        Some((i, _)) => &s[..i],
        None => s,
    }
}

/// Windows 控制台默认 GBK，把输出代码页切到 UTF-8，避免中文乱码
#[cfg(windows)]
fn init_console_utf8() {
    unsafe {
        windows_sys::Win32::System::Console::SetConsoleOutputCP(65001);
        windows_sys::Win32::System::Console::SetConsoleCP(65001);
    }
}

/// 解析 checkpoint 路径：None 时取 out_dir/latest.ckpt
fn resolve_ckpt<'a>(ckpt: Option<&'a str>, out_dir: &str) -> std::borrow::Cow<'a, str> {
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
fn load_model_and_tokenizer(
    ckpt_path: &str,
    tcfg: &config::TrainConfig,
    seed: u64,
    tokenizer_path: Option<&str>,
) -> (Transformer, Tokenizer, checkpoint::Checkpoint) {
    let ckpt = checkpoint::load_header(ckpt_path);
    let tokenizer = if let Some(path) = tokenizer_path {
        let loaded = Tokenizer::load(path);
        if ckpt.model.vocab_size != 0 {
            assert_eq!(loaded.vocab_size(), ckpt.model.vocab_size,
                "分词器词表（{}）与 checkpoint（{}）不一致", loaded.vocab_size(), ckpt.model.vocab_size);
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
fn maybe_merge_lora(model: &mut Transformer, merge_lora: bool) {
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

/// 推理时加载分词器：优先从文件加载，避免依赖语料
fn load_tokenizer_for_inference(tcfg: &config::TrainConfig, expect_vocab: usize) -> Tokenizer {
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
fn init_console_utf8() {}

/// 按配置重建分词器（char / bpe）。
/// 优先从文件加载（如果 tokenizer_file 配置了），否则从语料训练。
/// `expect_vocab = 0` 表示不校验（训练时词表由分词器决定）。
fn build_tokenizer(tcfg: &config::TrainConfig, train_text: &str, expect_vocab: usize) -> Tokenizer {
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

/// 训练：`train --config config/config.json [--resume ckpt]`
fn cmd_train(config_path: &str, resume: Option<&str>) {
    let log_path = runlog::start("train");
    println!("运行日志：{log_path}");
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("resume", resume.unwrap_or("无（从头训练）").to_string()),
        ],
    );
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    #[cfg(feature = "gpu")]
    if gpu::is_available() {
        logln!("GPU: {}（{}）", gpu::name(), gpu::backend());
    } else {
        logln!("未检测到可用 GPU，本次训练走 CPU");
    }

    let train_docs = load_documents(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    // 分词器语料 = 所有文档以 "\n" 相连，与 load_text 的拼接口径逐字一致
    let tokenizer = build_tokenizer(tcfg, &train_docs.join("\n"), 0); // 训练时词表由分词器决定

    // 训练完成后保存分词器（out_dir 不存在时 save 会自动创建）
    let tok_path = tcfg.tokenizer_file.clone()
        .unwrap_or_else(|| format!("{}/tokenizer.json", tcfg.out_dir));
    tokenizer.save(&tok_path);
    logln!("分词器已保存到 {tok_path}");

    // 词表大小 0 表示"由分词器决定"
    let mut model_cfg = cfg.model.clone();
    if model_cfg.vocab_size == 0 {
        model_cfg.vocab_size = tokenizer.vocab_size();
    }

    let mut rng = Rng::new(tcfg.seed);
    let mut model = Transformer::new(model_cfg.clone(), &mut rng);
    // 续训一个 LoRA 存档时同样要先重建适配层：checkpoint 头里记着 rank/alpha，
    // 少了这套参数名，load_with_opt 恢复参数时会直接断言失败。
    if let Some(path) = resume {
        let ckpt = checkpoint::load_header(path);
        if let Some(lora) = ckpt.lora.as_ref() {
            logln!(
                "续训：{path} 是 LoRA 存档（rank={} alpha={}），重建适配层后再恢复参数",
                lora.rank,
                lora.alpha
            );
            model.apply_lora(lora, &mut rng);
        }
    }
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    runlog::fields(
        "数据与模型",
        &[
            ("训练语料", tcfg.train_file.clone()),
            (
                "验证语料",
                tcfg.val_file
                    .clone()
                    .unwrap_or_else(|| "（未配置，自动从训练文本末尾切 10%）".to_string()),
            ),
            (
                "分词器",
                format!("{}（词表 {}）→ {tok_path}", tokenizer.kind(), tokenizer.vocab_size()),
            ),
            ("模型结构", format!("{model_cfg:?}")),
            ("模型参数", format!("{param_count}")),
            ("输出目录", tcfg.out_dir.clone()),
            (
                "指标 CSV",
                tcfg.log_file.clone().unwrap_or_else(|| "不记录".to_string()),
            ),
        ],
    );
    let loader = DataLoader::from_documents(
        &train_docs,
        val_text.as_deref(),
        &tokenizer,
        model_cfg.block_size,
        tcfg.batch_size,
    );
    let best = train::train_transformer(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&tcfg.out_dir),
        resume,
        &mut rng,
    );
    runlog::append(&format!("训练结束：best val loss = {best:.4}"));
    runlog::finish();
}

/// 评估：在验证集上计算 loss 与困惑度
fn cmd_eval(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
) {
    let log_path = runlog::start("eval");
    println!("运行日志：{log_path}");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", ckpt_path.to_string()),
            (
                "分词器文件",
                tokenizer_path.unwrap_or("自动（config.tokenizer_file → out_dir/tokenizer.json）").to_string(),
            ),
            ("评估批数", tcfg.eval_iters.to_string()),
        ],
    );
    let (mut model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, tcfg.seed, tokenizer_path);
    maybe_merge_lora(&mut model, merge_lora);

    let train_docs = load_documents(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    let loader = DataLoader::from_documents(
        &train_docs,
        val_text.as_deref(),
        &tokenizer,
        ckpt.model.block_size,
        tcfg.batch_size,
    );

    let mut eval_rng = Rng::new(tcfg.seed); // 固定种子，结果可复现
    let loss = train::eval_loss(&model, &loader, tcfg.eval_iters, &mut eval_rng);
    logln!(
        "评估 step {}：val_loss {:.4} | perplexity {:.2}（{} 个 token 的验证集上采 {} 批）",
        ckpt.step,
        loss,
        loss.exp(),
        loader.num_val_tokens(),
        tcfg.eval_iters
    );
    runlog::finish();
}

/// 缺省校准语料目录：仓库自带的混合语料（中英文章 / 对话 / 代码 / 诗歌）。
///
/// 为什么不用配置里的 `train_file`：那份语料是拿去训模型的，量化校准要的是
/// "**推理时会遇到的输入**"。两者在领域上可能差很远（例如用中文小说训的模型
/// 被拿去跑英文代码），此时拿训练语料当校准集，`H` 与激活均值估的就是错的分布。
/// 所以缺省给一份覆盖面最广的语料，真要贴合部署领域就用 `--calib-file` 显式指定。
const DEFAULT_CALIB_DIR: &str = "data/corpus/";

/// 权重量化：`quant --config config/config.json --bits int8 --method hess --eval`
///
/// 完整流程：加载 f32 模型 → （非 RTN 时）在校准集上采激活统计 → 逐层量化并出报告 →
/// 可选地对比量化前后的验证 loss / 困惑度 → **把量化结果烘焙回 f32** 再落盘。
///
/// 存档里为什么仍写 f32：checkpoint 格式的语义是"按名字对齐的 f32 参数表"，
/// 让它长出第二种参数编码，会把加载、续训、合并、分词器对齐这一整条链路都变成
/// 两种可能的分支。量化真正的收益在**推理时的显存与带宽**，而推理本来就该在
/// 进程里重建量化（见 [`checkpoint::requantize_after_load`]），没必要为此污染格式。
/// 存档头里的 `quant` 字段负责记住"当初是怎么量化的"。
#[allow(clippy::too_many_arguments)]
fn cmd_quant(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    bits_name: &str,
    method_name: &str,
    calib_file: Option<&str>,
    calib_samples: usize,
    calib_tokens: usize,
    act_order_name: &str,
    damp: f32,
    block: usize,
    alpha: Option<f32>,
    out: Option<&str>,
    do_eval: bool,
) {
    let log_path = runlog::start("quant");
    println!("运行日志：{log_path}");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let bits = q_bits_from_name(bits_name);
    let method = q_method_from_name(method_name);
    let act_order = bool_from_name("--act-order", act_order_name);
    // 分组方向固定为 `Col`（每个输出通道一个 scale）：这是权重量化的通行口径，
    // 见 [`QuantOpts::axis`]。算法参数一次性装进 `QuantOpts` 往下传——
    // 这样"用户选的算法"与"该算法的参数"永远是同一个对象里的两份，不会出现
    // "命令行给了 Hessian 量化的 damp，但实际跑的是 AWQ"这种没人会发现的错配。
    let opts = QuantOpts {
        hess: HessOpts {
            act_order,
            damp,
            block,
        },
        awq_alpha: alpha,
        ..QuantOpts::default()
    };
    let out_path = out
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/quant.ckpt", tcfg.out_dir));
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", ckpt_path.to_string()),
            ("位宽", bits.name().to_string()),
            ("算法", format!("{}{}", method.name(), method.describe())),
            ("Hessian 量化选项", opts.hess.describe()),
            (
                "校准集",
                match method {
                    QuantMethod::Rtn => "不适用（RTN 只用权重本身）".to_string(),
                    _ => calib_file.unwrap_or(DEFAULT_CALIB_DIR).to_string(),
                },
            ),
            (
                "校准预算",
                format!("{calib_samples} 个窗口 × 上下文长度（硬上限 {calib_tokens} token，0 = 不设限）"),
            ),
            (
                "AWQ alpha",
                match alpha {
                    Some(a) => a.to_string(),
                    None => format!(
                        "网格搜索 0~1，步长 {}（论文默认 {}）",
                        crate::quant::AWQ_ALPHA_GRID_STEP,
                        crate::quant::AWQ_ALPHA
                    ),
                },
            ),
            ("输出", out_path.clone()),
            ("量化前后对比", if do_eval { "是" } else { "否（加 --eval 开启）" }.to_string()),
        ],
    );
    let (mut model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, tcfg.seed, tokenizer_path);
    // LoRA 存档必须先合并：量化会跳过挂着适配器的层（见 [`Linear::quantize_weight`]），
    // 不合并的话报告只会打印"N 层被跳过"，等于什么都没做。
    if model.has_lora() {
        maybe_merge_lora(&mut model, true);
    }

    // 1) 量化前基线（固定种子的 Rng，保证前后两次取的是同一批验证数据）。
    //    **loader 只在这里（`--eval` 时）才建**：它要把整份训练语料编码成 token 流，
    //    语料上百万字符时这一步比量化本身还慢，而"只想看看压缩比和逐层误差"的
    //    调用根本不需要它。所以把 `Option` 一路留着，不用就不付这份代价。
    let mut eval_rng = Rng::new(tcfg.seed);
    let (loss_before, eval_loader) = if do_eval {
        let train_docs = load_documents(&tcfg.train_file);
        let val_text = tcfg.val_file.as_deref().map(read_text);
        let loader = DataLoader::from_documents(
            &train_docs,
            val_text.as_deref(),
            &tokenizer,
            ckpt.model.block_size,
            tcfg.batch_size,
        );
        let l = train::eval_loss(&model, &loader, tcfg.eval_iters, &mut eval_rng);
        logln!("量化前：val_loss {l:.4} | perplexity {:.2}", l.exp());
        (Some(l), Some(loader))
    } else {
        (None, None)
    };

    // 2) 校准（RTN 不需要，跳过能省一整遍前向）。
    //    token 预算 = 窗口数 × 上下文长度：校准是按**窗口**喂进去的，所以这是最贴近
    //    调用方直觉的口径；`--calib-tokens` 只作为硬上限（0 = 不设限）。
    let seq = ckpt.model.block_size.max(1);
    let window_budget = calib_samples.max(1).saturating_mul(seq);
    let max_tokens = if calib_tokens == 0 {
        window_budget
    } else {
        window_budget.min(calib_tokens)
    };
    let stats = if method.needs_calib() {
        let path = calib_file.unwrap_or(DEFAULT_CALIB_DIR);
        // 缺省语料被精简掉时要**当场说清怎么补救**：`load_text` 对不存在的路径
        // 只是在读文件时报"无法读取数据文件"，不会告诉你"这是缺省值、可以用
        // --calib-file 换一个"。
        assert!(
            std::path::Path::new(path).exists(),
            "校准语料 {path} 不存在。请用 --calib-file 指定一个文本文件/目录/通配符路径\
             （缺省语料在 {DEFAULT_CALIB_DIR}）"
        );
        // 只把需要的**前缀**切出来喂给分词器：缺省语料 `data/corpus/` 有近 4 MB，
        // 整个编码一遍再 truncate 掉 99% 的 token，等于把时间全花在丢弃上。
        // 字符数按 token 预算的 8 倍留余量——任何子词/字符级分词器一个 token 都不会
        // 吃掉 8 个字符，所以这段前缀一定够填满预算（宁可多编一点，也不要因为截短了
        // 而少喂校准数据：`H` 估得不准会被 `H⁻¹` 放大）。
        let text = truncate_chars(&load_text(path), max_tokens.saturating_mul(8)).to_string();
        let s = model.calibrate(
            &tokenizer,
            &[text],
            CalibOpts {
                max_tokens,
                max_seq: seq,
                ..CalibOpts::default()
            },
        );
        let n = s.per_layer.first().map(|(_, c)| c.n_tokens).unwrap_or(0);
        let full = s
            .per_layer
            .iter()
            .filter(|(_, c)| c.hessian.as_ref().is_some_and(|h| h.is_full()))
            .count();
        // 把 Hessian 的**内存代价**打出来：这是整条流水线里唯一会随 `in_features²`
        // 增长的开销，性能问题里最容易被忽略的一项。超预算的层应自动退化成对角。
        let h_bytes: usize = s
            .per_layer
            .iter()
            .filter_map(|(_, c)| c.hessian.as_ref())
            .map(|h| h.byte_len())
            .sum();
        logln!(
            "校准：{path} 上跑了 {n} 个 token（预算 {max_tokens}，窗口 {seq}），\
             命中 {} 层（{full} 层存全矩阵 Hessian，共 {:.1} MB，其余退化为对角）",
            s.per_layer.len(),
            h_bytes as f64 / (1024.0 * 1024.0)
        );
        assert!(
            s.per_layer.iter().all(|(_, c)| c.n_tokens > 0),
            "有层没采到激活统计——继续下去它会静默退回 RTN，报告也看不出来"
        );
        Some(s)
    } else {
        None
    };

    // 3) 量化 + 报告
    let reports = model.quantize_weights(bits, method, stats.as_ref(), opts);
    // 量化表示与逐层报告必须同时成立：有报告却没带上量化表示 = 权重白量化了，
    // 这种"静默无效"比直接报错更危险（下游还在按报告的压缩率做显存预算）。
    assert_eq!(
        model.is_quantized(),
        !reports.is_empty(),
        "量化表示与逐层报告不一致：报告 {} 层，is_quantized = {}",
        reports.len(),
        model.is_quantized()
    );
    let summary = model.quant_summary(reports);
    logln!("{}", summary.describe());
    // 逐层报告按**相对误差降序**打印：量化的问题几乎总是集中在少数几层上，
    // 按名字顺序打出来会让人在几十行里自己找最大值（而且很容易看漏）。
    let mut layers = summary.layers.clone();
    layers.sort_by(|a, b| {
        b.rel_err
            .partial_cmp(&a.rel_err)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    logln!("逐层误差（按相对误差降序，最多 16 行）：");
    for r in layers.iter().take(16) {
        logln!("  {}", r.describe());
    }
    if layers.len() > 16 {
        logln!("  …（其余 {} 层误差更小）", layers.len() - 16);
    }

    // 4) 量化后的验证集对比
    if let (Some(before), Some(loader)) = (loss_before, eval_loader.as_ref()) {
        let after = train::eval_loss(&model, loader, tcfg.eval_iters, &mut eval_rng);
        logln!(
            "量化后：val_loss {after:.4} | perplexity {:.2}（Δ loss {:+.4}，Δ ppl {:+.2}）",
            after.exp(),
            after - before,
            after.exp() - before.exp()
        );
    }

    // 5) 烘焙回 f32 再落盘（见本函数文档）
    model.dequantize_weights();
    // 权重 only 档：量化产物没有优化器状态可言，只写参数段（体积 1× 参数量而非 3×）。
    // 头部 `weights_only: true` 让读取端按一段校验；续训方读到它会把动量补零，
    // 拿到一个全新 AdamW 的合法起点（见 checkpoint::save_weights）。
    checkpoint::save_weights(&out_path, &model, ckpt.step, ckpt.best_val_loss);
    logln!("已写出：{out_path}（权重为反量化后的 f32，头部记录量化参数，权重 only 格式）");
    runlog::finish();
}

/// 生成：`generate --config config/config.json --ckpt ckpt --prompt "..."`
#[allow(clippy::too_many_arguments)]
fn cmd_generate(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    prompt: &str,
    max_new: usize,
    opts: SampleOpts,
    seed: u64,
    kv: KvOpts,
    beam: Option<usize>,
    length_penalty: f32,
    rope_opt: RopeArgs,
) {
    let log_path = runlog::start("generate");
    println!("运行日志：{log_path}");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let (mut model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    maybe_merge_lora(&mut model, merge_lora);
    let mut rng = Rng::new(seed);
    // RoPE 覆盖要在报告"模型结构"之前生效：`ckpt.model` 只记了 head 数与维度，
    // 上下文窗口与频率表不在里面，不打印出来就没法从日志复现这次运行
    let rope_note = apply_rope_override(&mut model, &rope_opt);
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", format!("{ckpt_path}（step {}）", ckpt.step)),
            (
                "分词器文件",
                tokenizer_path.unwrap_or("自动（config.tokenizer_file → out_dir/tokenizer.json）").to_string(),
            ),
            ("模型结构", format!("{:?}", ckpt.model)),
            ("prompt", format!("{prompt:?}")),
            ("max_new", max_new.to_string()),
            ("seed", seed.to_string()),
            (
                "生成方式",
                match beam {
                    Some(b) => format!("Beam Search（beam_size={b}，length_penalty={length_penalty}）"),
                    None => "采样".to_string(),
                },
            ),
            ("temperature", opts.temperature.to_string()),
            ("top_k", opts.top_k.to_string()),
            ("top_p", opts.top_p.to_string()),
            (
                "重复惩罚",
                format!("{}（回看窗口 {}）", opts.repetition_penalty, opts.repetition_window),
            ),
            ("KV cache", describe_kv(&kv)),
            (
                "RoPE / 上下文",
                rope_note.clone().unwrap_or_else(|| {
                    format!("沿用 checkpoint：窗口 {}，base = {}", model.cfg.block_size, model.cfg.rope_base)
                }),
            ),
        ],
    );
    if let Some(note) = &rope_note {
        logln!("[rope] {note}");
    }

    let out = if let Some(beam_size) = beam {
        // Beam Search 生成（每条 beam 持有独立缓存，见 `sample::beam_search`）
        logln!(
            "Beam Search 生成（beam_size={} length_penalty={}，{}）：",
            beam_size, length_penalty, describe_kv(&kv)
        );
        sample::beam_search(
            &model,
            &tokenizer,
            prompt,
            max_new,
            beam_size,
            length_penalty,
            kv,
        )
    } else {
        // 采样生成
        logln!(
            "生成（temperature={} top-k={} top-p={} 重复惩罚={}（窗口 {}），{}）：",
            opts.temperature,
            opts.top_k,
            opts.top_p,
            opts.repetition_penalty,
            opts.repetition_window,
            describe_kv(&kv)
        );
        generate(&model, &tokenizer, prompt, max_new, &opts, kv, &mut rng)
    };
    logln!("{out}");
    runlog::append(&format!("[done] 输出总长 {} 字符（含 prompt）", out.chars().count()));
    runlog::finish();
}

/// 把 KV cache 设置说成一句人话（日志与命令行回显共用）
fn describe_kv(kv: &KvOpts) -> String {
    if !kv.enable {
        return "关（每次全量前向）".to_string();
    }
    let bits = match kv.bits {
        None => "f32".to_string(),
        Some(b) => format!("{b:?} 量化"),
    };
    format!("开（{bits}，Attention Sink {})", kv.sink)
}

/// 把对话历史裁剪到不超过 `budget` 个 token：从最老的一行开始丢，保留最近的内容。
///
/// 必须真的调 `encode` 来数 token —— 按"字节数 / 字符数"估算的偏差很大：
/// 中英文、BPE 合并数都不同，同样长度的文本 token 数能差一倍以上。
fn trim_history(tokenizer: &Tokenizer, history: &str, budget: usize) -> String {
    if history.is_empty() || tokenizer.encode(history).len() <= budget {
        return history.to_string();
    }
    // 行粒度足够细：每轮对话都会写入多行
    let lines: Vec<&str> = history.split('\n').collect();
    for start in 1..lines.len() {
        let candidate = lines[start..].join("\n");
        if tokenizer.encode(&candidate).len() <= budget {
            return candidate;
        }
    }
    String::new()
}

/// SFT 模板下的历史裁剪：以 `SFT_USER` 为切点按**轮**丢，而不是按行。
///
/// 按行丢会把某轮拦腰截断（历史里一行只是回答中的一句），留下半截上下文。
/// 每轮都以 `SFT_USER` 开头，从它的位置切就天然对齐到轮边界。
fn trim_sft_history(tokenizer: &Tokenizer, history: &str, budget: usize) -> String {
    if history.is_empty() || tokenizer.encode(history).len() <= budget {
        return history.to_string();
    }
    for (pos, _) in history.match_indices(SFT_USER) {
        let candidate = &history[pos..];
        if tokenizer.encode(candidate).len() <= budget {
            return candidate.to_string();
        }
    }
    String::new()
}

/// SFT 模板下的停止标记：模型答完会自己吐 `。。`；
/// 万一没学会收尾，它就会顺着模板编下一轮提问，所以 `用户：` 也是停止标记。
const SFT_STOP: &[&str] = &[SFT_END, SFT_USER];

/// 把 `i` 向前收敛到最近的字符边界（`String` 按字节切片时用，避免切出非法 UTF-8）
fn floor_char_boundary(s: &str, mut i: usize) -> usize {
    if i >= s.len() {
        return s.len();
    }
    while i > 0 && !s.is_char_boundary(i) {
        i -= 1;
    }
    i
}

/// 交互式对话模式
#[allow(clippy::too_many_arguments)]
fn cmd_chat(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    system: &str,
    opts: SampleOpts,
    max_new: usize,
    kv: KvOpts,
    seed: u64,
    prompt_format: &str,
    rope_opt: RopeArgs,
) {
    let log_path = runlog::start("chat");
    println!("运行日志：{log_path}");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    // SFT 模板：prompt 按训练时的对话模板拼，模型才会"回答"而不是"续写"。
    // 没做过 SFT 的预训练权重用这个模板只会一脸茫然（它没见过这些标记），要 `--prompt-format raw`。
    let use_sft = prompt_format == "sft";
    let mut opts = opts;
    if use_sft {
        opts.stop = SFT_STOP;
    }
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let (mut model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    maybe_merge_lora(&mut model, merge_lora);
    let mut rng = Rng::new(seed);
    // 长度外推必须在下面读 `model.cfg.block_size` 分配上下文预算**之前**生效：
    // 否则 `block_size` 还是训练窗口，`--max-ctx` 扩出来的空间会被白白裁掉。
    let rope_note = apply_rope_override(&mut model, &rope_opt);
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", format!("{ckpt_path}（step {}）", ckpt.step)),
            (
                "分词器文件",
                tokenizer_path.unwrap_or("自动（config.tokenizer_file → out_dir/tokenizer.json）").to_string(),
            ),
            ("模型结构", format!("{:?}", ckpt.model)),
            ("系统提示", if system.is_empty() { "无".to_string() } else { system.to_string() }),
            ("prompt 模板", prompt_format.to_string()),
            ("每轮最大新 token", max_new.to_string()),
            ("seed", seed.to_string()),
            ("temperature", opts.temperature.to_string()),
            ("top_k", opts.top_k.to_string()),
            ("top_p", opts.top_p.to_string()),
            (
                "重复惩罚",
                format!("{}（回看窗口 {}）", opts.repetition_penalty, opts.repetition_window),
            ),
            ("KV cache", describe_kv(&kv)),
            (
                "RoPE / 上下文",
                rope_note.clone().unwrap_or_else(|| {
                    format!("沿用 checkpoint：窗口 {}，base = {}", model.cfg.block_size, model.cfg.rope_base)
                }),
            ),
        ],
    );
    if let Some(note) = &rope_note {
        logln!("[rope] {note}");
    }

    logln!("交互式对话模式（输入文本后按回车生成，输入 :quit 退出）");
    logln!(
        "参数：temperature={} top-k={} top-p={} 重复惩罚={}（窗口 {}）",
        opts.temperature, opts.top_k, opts.top_p, opts.repetition_penalty, opts.repetition_window
    );
    if !system.is_empty() {
        logln!("系统提示：{}", system);
    }
    logln!("---");

    // 上下文窗口要在「system prompt」「对话历史」「本轮生成」三者之间分配。
    //
    // 分配顺序（前面的优先保）：
    //   1. system prompt —— 永远保留。它是序列最开头的"注意力锚点"（attention sink），
    //      丢掉它不只是失忆，还会让整条序列的注意力分布失稳
    //   2. 本轮生成 —— 预留 max_new 个 token，让它整段留在窗口内。缓存是滑动窗口，
    //      生成本身不受 block_size 限制（不会提前停），但窗口只保留最近 block_size 个 token：
    //      prompt + 生成超过它时，最前面的内容会滑出模型视野（system prompt 与早期历史被"遗忘"）
    //   3. 对话历史 —— 剩下的额度都给它，不够就按 token 数从最老的开始丢
    let block_size = model.cfg.block_size;
    if max_new >= block_size {
        logln!(
            "[warn] --max-new={max_new} 不小于上下文窗口 {block_size}：本轮生成到后半程时，\
             最早的输入已滑出视野（滑动窗口只保留最近 {block_size} 个 token）"
        );
    }
    let prompt_budget = block_size.saturating_sub(max_new);

    let system = system.trim();
    let n_system = tokenizer.encode(system).len();
    let n_system_prefix = if system.is_empty() { 0 } else { n_system + 1 }; // +1 是它后面的 '\n'
    if n_system_prefix >= prompt_budget {
        logln!(
            "[warn] system prompt 已有 {n_system} 个 token，占满了 {prompt_budget} 的输入预算，本轮没有空间留给对话历史"
        );
    }

    // 裁剪不能静默：首次立即提示，之后每 HISTORY_REPORT_EVERY 次汇总一次，
    // 否则跑几十轮下来用户不知道"已经无记忆多久了"
    const HISTORY_REPORT_EVERY: usize = 10;
    let mut first_trim_turn: Option<usize> = None;
    let mut trim_count = 0usize;
    let mut dropped_tokens_total = 0usize;
    let mut warned_prompt_overflow = false;

    // 只存「对话历史」，不含 system：system 在拼接 prompt 时单独放在最前面，
    // 这样它就不会被 `trim_history` 当成最老的内容丢掉。
    let mut context_history = String::new();
    let mut turn = 0usize;

    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush().ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input == ":quit" || input == ":exit" || input.is_empty() {
            break;
        }
        turn += 1;

        // 构造 prompt：system + 历史 + 本轮。
        // system 固定在序列最前面并独立于裁剪，历史按 **token** 裁剪（不是字节/字符），
        // 并给本轮生成留够 max_new 个 token。
        //
        // 本轮追加到 prompt 末尾的内容：raw 模式就是输入本身；
        // SFT 模式下要补上角色标记，并且**停在「助手：」这一行之后**——
        // 这正是训练时"轮到模型说话"的位置，模型才会接着写回答，而不是继续续写前文。
        let tail = if use_sft {
            format!("{SFT_USER}\n{input}\n{SFT_ASSISTANT}\n")
        } else {
            input.to_string()
        };
        let n_tail = tokenizer.encode(&tail).len();
        // 历史预算 = 输入预算 - system（含其后 '\n'）- 本轮追加内容 - 历史与它之间的 '\n'
        let history_budget = prompt_budget.saturating_sub(n_system_prefix + n_tail + 1);
        let history = if use_sft {
            trim_sft_history(&tokenizer, &context_history, history_budget)
        } else {
            trim_history(&tokenizer, &context_history, history_budget)
        };
        if history.len() < context_history.len() {
            let n_before = tokenizer.encode(&context_history).len();
            let n_after = tokenizer.encode(&history).len();
            let dropped = n_before.saturating_sub(n_after);
            trim_count += 1;
            dropped_tokens_total += dropped;
            match first_trim_turn {
                None => {
                    first_trim_turn = Some(turn);
                    logln!(
                        "[info] 对话历史超出预算（{n_before} > {history_budget} token），\
                         从本轮起丢弃最早的轮次（本次丢弃 {dropped} token，保留 {n_after} token）；\
                         想保留更多历史请调小 --max-new"
                    );
                }
                Some(t0) if trim_count % HISTORY_REPORT_EVERY == 0 => {
                    logln!(
                        "[info] 已丢弃历史 {trim_count} 次（起始于第 {t0} 轮），\
                         累计丢弃 {dropped_tokens_total} token，当前保留 {n_after} token"
                    );
                }
                Some(_) => {}
            }
        }

        let mut prompt = String::new();
        if !system.is_empty() {
            prompt.push_str(system);
            prompt.push('\n');
        }
        if !history.is_empty() {
            prompt.push_str(&history);
            prompt.push('\n');
        }
        prompt.push_str(&tail);
        runlog::append(&format!("\n[第 {turn} 轮] 输入：{input}"));

        let n_prompt = tokenizer.encode(&prompt).len();
        if n_prompt > block_size && !warned_prompt_overflow {
            warned_prompt_overflow = true;
            logln!(
                "[warn] 本轮输入本身已有 {n_prompt} 个 token，超过上下文窗口 {block_size}，只有末尾内容参与生成"
            );
        }

        // 每轮都重建缓存：上一轮的缓存对不上本轮重新裁剪过的 prompt，
        // 沿用会让模型看到"已被丢弃的历史"（见 `generate` 里的 KV 复用说明）。
        let out = generate(&model, &tokenizer, &prompt, max_new, &opts, kv, &mut rng);

        // 只打印新生成的部分（去掉 prompt 前缀）
        let start = floor_char_boundary(&out, prompt.len());
        let response = out[start..].trim();
        logln!("{response}");
        runlog::append(&format!("[第 {turn} 轮] 输出：{response}"));

        // 更新历史：历史 = 旧历史 + 本轮问答（不含 system，它单独拼接）。
        // 不要重复拼 `input`：prompt 里已含本轮输入，那会让每轮输入在窗口里占两份。
        // 长度控制交给下一轮开头的裁剪（按真实 token 数、SFT 模式下按轮）。
        context_history = history;
        if !context_history.is_empty() {
            context_history.push('\n');
        }
        if use_sft {
            // 写成模板形态，下一轮就能被 `trim_sft_history` 按轮切开
            context_history.push_str(&format!("{SFT_USER}\n{input}\n{SFT_ASSISTANT}\n{response}"));
        } else {
            context_history.push_str(input);
            if !response.is_empty() {
                context_history.push('\n');
                context_history.push_str(response);
            }
        }
    }
    runlog::append(&format!("\n对话结束：共 {} 轮", turn));
    runlog::finish();
}

/// 监督微调（SFT）：用「提问→回答」语料教只会续写的预训练模型"回答"。
///
/// 与 `train` 的区别只有数据与 loss：样本是对话，且**只有回答段参与 loss**
///（提问与角色标记被掩码屏蔽，见 [`data::SftLoader`]）。训练循环本身完全共用。
fn cmd_sft(
    config_path: &str,
    pretrained_path: &str,
    sft_file: Option<&str>,
    steps: Option<usize>,
    lr: Option<f32>,
    out_dir: Option<&str>,
) {
    let log_path = runlog::start("sft");
    println!("运行日志：{log_path}");
    let mut cfg = Config::load(config_path);

    // 步数与学习率：CLI 优先，其次配置文件。
    // 学习率**不**默认沿用预训练量级：SFT 是在已收敛的权重上继续训，
    // 用预训练的步长会把预训练攒下的语言能力一起冲掉（实测几百步就退化成乱码）。
    // 未显式指定时取配置里 max_lr 的 1/10，并把实际取值记进日志。
    if let Some(s) = steps {
        cfg.train.steps = s;
    }
    let (max_lr, lr_source) = match lr {
        Some(l) => (l, "--lr 指定".to_string()),
        None => (
            cfg.train.max_lr / 10.0,
            "--lr 未指定，取 config max_lr 的 1/10".to_string(),
        ),
    };
    cfg.train.max_lr = max_lr;
    cfg.train.min_lr = max_lr * 0.1;
    // 预热太长会让 SFT 大部分步数都耗在爬坡上
    cfg.train.warmup_steps = cfg.train.warmup_steps.min(cfg.train.steps / 10).max(1);
    // 评估间隔按步数收窄：配置里那是给预训练（上万步）的值，直接用在几百步的 SFT 上，
    // 会把整段训练压成"只在最后评估一次"——best.ckpt 退化成 final.ckpt，
    // 连续不改善才触发的早停也永远等不到第二次评估。收到约 1/10 就有十条曲线可看。
    cfg.train.eval_every = cfg.train.eval_every.min(cfg.train.steps / 10).max(1);

    let sft_paths = sft_file
        .map(str::to_string)
        .or_else(|| cfg.train.sft_file.clone())
        .unwrap_or_else(|| {
            panic!(
                "未指定 SFT 语料：用 --sft-file，或在 {config_path} 里设置 train.sft_file"
            )
        });

    // 输出目录默认加 `-sft` 后缀：绝不能直接写回 tcfg.out_dir，
    // 那会覆盖预训练攒下的 latest.ckpt / best.ckpt —— SFT 效果不好就再也回不去了。
    // 指标 CSV 同理，否则会把预训练那条 loss 曲线整条抹掉。
    let out_dir = match out_dir {
        Some(d) => d.to_string(),
        None => format!("{}-sft", cfg.train.out_dir),
    };
    cfg.train.log_file = Some(format!("{out_dir}/sft.csv"));

    runlog::json(&format!("完整配置（{config_path} + CLI 覆盖后）"), &cfg);
    let tcfg = &cfg.train;

    // SFT 是"带 loss mask 的"训练，GPU 的常驻输出头路径不吃逐位置权重，整段会回落到
    // 逐算子路径。实测这条回落路径比纯 CPU 还慢（MX150：392 tok/s vs CPU 780 tok/s），
    // 长跑还会崩。用户多半是照着预训练的命令加 `--features gpu` 过来的，所以要说清楚。
    #[cfg(feature = "gpu")]
    logln!(
        "提示：SFT 的 loss 带逐位置掩码，GPU 常驻输出头路径不可用，会回落到逐算子路径——\
         实测比纯 CPU 更慢。建议改用不带 `--features gpu` 的构建跑 SFT。"
    );

    let (model, tokenizer, ckpt) = load_model_and_tokenizer(pretrained_path, tcfg, tcfg.seed, None);
    // 语料可能散在多处，用逗号分隔（支持目录与 `*` 通配）。
    // 按文件加载而不是拼成一份：说话人角色是文件级属性，拼接会让后面文件的角色弄反。
    let sft_texts = load_texts(&sft_paths);

    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            (
                "预训练权重",
                format!("{pretrained_path}（step {}）", ckpt.step),
            ),
            ("预训练模型结构", format!("{:?}", ckpt.model)),
            ("SFT 语料", sft_paths.clone()),
            (
                "步数 / 学习率",
                format!("{} / {}（{lr_source}）", tcfg.steps, tcfg.max_lr),
            ),
            ("输出目录", out_dir.clone()),
            ("prompt 模板", format!("{SFT_USER} … {SFT_ASSISTANT} … {SFT_END}")),
        ],
    );

    let mut rng = Rng::new(tcfg.seed);
    let loader = SftLoader::from_texts(
        &sft_texts,
        &tokenizer,
        ckpt.model.block_size,
        tcfg.batch_size,
    );
    logln!(
        "SFT 语料：{} 段对话，打包 {} token | 监督位置（回答段）占 {:.1}%",
        loader.num_conversations(),
        loader.num_tokens(),
        100.0 * loader.supervised_ratio(),
    );
    if loader.supervised_ratio() < 0.05 {
        logln!(
            "[warn] 监督位置只有 {:.1}%，训练信号很稀——语料里提问与标记远多于回答",
            100.0 * loader.supervised_ratio()
        );
    }

    let best = train::train_transformer(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&out_dir),
        None,
        &mut rng,
    );

    // 把分词器一并放进输出目录，这个目录就是自包含的，推理时不必再指回预训练目录
    tokenizer.save(&format!("{out_dir}/tokenizer.json"));

    runlog::append(&format!("SFT 结束：best val loss = {best:.4}"));
    logln!("SFT 完成，best val loss = {best:.4}");
    // 两个 checkpoint 都给出，因为 SFT 语料往往很小：验证区与训练区同分布，val 曲线
    // 常在头几十步就见底，之后 train 还在降而 val 回升——`best.ckpt` 取的是那个
    // 「刚开始像样」的快照，`final.ckpt` 反而把回答的句式学得更足。
    // 哪个更好只有自己聊两句才知道，所以两个都列出来。
    logln!(
        "用下面的命令对话（{SFT_ASSISTANT} 之后的模板会自动拼好）；\
         建议 best.ckpt 与 final.ckpt 各聊几句再定——语料小时 final 常常更「会答」：\n  \
         cargo run --release -- chat --ckpt {out_dir}/best.ckpt --tokenizer {out_dir}/tokenizer.json\n  \
         cargo run --release -- chat --ckpt {out_dir}/final.ckpt --tokenizer {out_dir}/tokenizer.json"
    );
    runlog::finish();
}

/// LoRA 微调：加载预训练模型，冻结全部主干，只训练挂上去的低秩适配层。
///
/// 与 `sft` 共用同一套训练循环、同一份语料（`train.sft_file`，可用 `--sft-file` 覆盖），
/// 差别只在**参数集合**：
/// - `sft` 全参更新，可训练参数 = 全部
/// - `finetune` 冻结主干，可训练参数 = 各层 `lora_a` / `lora_b`（本项目实测 1.32%）
///
/// 冻结是三处一起做实的，缺一不可：前向用 `matmul_frozen` 省掉 `dW`、优化器跳过冻结参数
/// （连带不吃权重衰减）、GPU 常驻显存快路整体让路。原理见 `docs/29-LoRA低秩适配.md`。
///
/// 挂载位置由 `--lora-targets` 决定（缺省 Q/K/V）。基座本身是 LoRA 存档时有两种走法：
/// 缺省**重挂一套**（旧的增量丢弃，只继承主干），加 `--resume-lora` 则**链式续训**
/// 存档里那套（保留已学的增量），后者不能再传 rank / alpha / targets。
#[allow(clippy::too_many_arguments)]
fn cmd_finetune(
    config_path: &str,
    pretrained_path: &str,
    lora_rank: Option<usize>,
    lora_alpha: Option<f32>,
    lora_targets: Option<&str>,
    resume_lora: bool,
    steps: usize,
    lr: f32,
    sft_file: Option<&str>,
    out_dir: Option<&str>,
) {
    let log_path = runlog::start("finetune");
    println!("运行日志：{log_path}");
    let mut cfg = Config::load(config_path);

    // 先加载基座：该怎么挂适配层，取决于基座本身是不是 LoRA 存档
    let (mut model, tokenizer, ckpt) =
        load_model_and_tokenizer(pretrained_path, &cfg.train, cfg.train.seed, None);
    let mut rng = Rng::new(cfg.train.seed);

    // 这两个值只用于日志（运行时结构一律读模型上的实际状态，不做第二本账）
    let (rank_used, alpha_used, targets_used) = if resume_lora {
        assert!(
            model.has_lora(),
            "--resume-lora 要求基座是 LoRA 存档，但 {pretrained_path} 里没有适配层"
        );
        assert!(
            lora_rank.is_none() && lora_alpha.is_none() && lora_targets.is_none(),
            "--resume-lora 的结构由存档决定，不能再传 --lora-rank / --lora-alpha / --lora-targets"
        );
        model.resume_lora();
        let l = model.lora.clone().expect("resume_lora 之后 model.lora 必有值");
        (l.rank, l.alpha, l.targets.to_string())
    } else {
        // 基线形态来自配置文件（老代码里这一块是死字段：CLI 的默认值无条件盖过它，写进
        // config 也不生效）。现在 CLI 的 rank / alpha / targets 都是 Option，缺谁才用谁：
        // rank   = --lora-rank  → config.lora.rank  → 16
        // alpha  = --lora-alpha → config.lora.alpha（仅当配置文件显式写了 lora 块）→ rank
        // targets= --lora-targets → config.lora.targets → q,k,v
        let from_config = cfg.train.lora.clone();
        let base = from_config.clone().unwrap_or_default();
        let rank = lora_rank.unwrap_or(base.rank);
        let lora_cfg = config::LoRAConfig {
            rank,
            // 配置文件没写 lora 时，缺省 α = rank（增量不额外缩放）；写了就以它的 alpha 为准
            alpha: lora_alpha
                .or_else(|| from_config.map(|l| l.alpha))
                .unwrap_or(rank as f32),
            targets: match lora_targets {
                Some(spec) => config::LoRATargets::parse(spec).unwrap_or_else(|e| panic!("{e}")),
                None => base.targets,
            },
        };
        if model.has_lora() {
            // 拿一个 LoRA 存档当基座：默认语义是**重挂一套**，原适配层权重随之丢弃
            // ——只有主干被继承。说清楚，并把链式续训的开关指出来。
            logln!(
                "[warn] 基座存档本身已是 LoRA 形态，这里会重新挂一套适配层（原适配层权重丢弃）；\
                 要接着训存档里那套，加 --resume-lora"
            );
        }
        model.apply_lora(&lora_cfg, &mut rng);
        (lora_cfg.rank, lora_cfg.alpha, lora_cfg.targets.to_string())
    };
    // LoRA 形态写回配置，交给训练循环与 runlog（train_transformer 的实际行为仍只读模型上的形态）
    cfg.train.lora = model.lora.clone();
    cfg.train.steps = steps;
    cfg.train.max_lr = lr;
    cfg.train.min_lr = lr * 0.1;
    // 预热与评估间隔按步数收窄（与 sft 同理：配置里那是给上万步预训练的值）
    cfg.train.warmup_steps = cfg.train.warmup_steps.min(cfg.train.steps / 10).max(1);
    cfg.train.eval_every = cfg.train.eval_every.min(cfg.train.steps / 10).max(1);

    // 输出目录默认加 `-lora` 后缀：绝不写回 tcfg.out_dir，否则会覆盖预训练攒下的
    // latest.ckpt / best.ckpt。指标 CSV 同理，避免把预训练那条 loss 曲线抹掉。
    let out_dir = match out_dir {
        Some(d) => d.to_string(),
        None => format!("{}-lora", cfg.train.out_dir),
    };
    cfg.train.log_file = Some(format!("{out_dir}/lora.csv"));

    // 日志记的是**覆盖 CLI 参数之后**的最终配置（这才是本次实际使用的配置）
    runlog::json(&format!("完整配置（{config_path} + CLI 覆盖后）"), &cfg);
    let tcfg = &cfg.train;
    if resume_lora {
        logln!(
            "链式续训：沿用存档里的适配层（rank={rank_used} alpha={alpha_used} 挂载={targets_used}）\
             steps={steps} lr={lr}｜已学的增量保留，只训 A/B"
        );
    } else {
        logln!(
            "LoRA 微调：rank={rank_used} alpha={alpha_used} 挂载={targets_used} steps={steps} lr={lr}\
             ｜冻结主干，只训适配层"
        );
    }

    let total_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    let trainable_params: usize = model
        .trainable_parameters()
        .iter()
        .map(|p| p.numel())
        .sum();
    // 打印**实测**参数（不是按公式预估）：可训练集合就是 trainable_parameters，
    // 和"主干冻结是否真的生效"是同一件事，估错了这里会立刻露馅。
    logln!(
        "模型参数：{total_params}（含冻结主干）｜可训练：{trainable_params}（{:.3}%）\
         ｜{} 对低秩矩阵（挂载 {targets_used}），共 {} 个适配参数张量",
        100.0 * trainable_params as f32 / total_params.max(1) as f32,
        model.lora_parameters().len() / 2,
        model.trainable_parameters().len(),
    );
    // 语料与 `sft` 用的是同一份、同一套加载方式：LoRA 微调就是 SFT 的省算力版，
    // 只有语料对齐，"全参 SFT vs LoRA" 的对比才成立（都能直接吃 `chat --prompt-format sft`）。
    // 语料可能散在多处（逗号分隔）；按文件加载而不是拼成一份，因为说话人角色是文件级属性。
    let sft_paths = sft_file
        .map(str::to_string)
        .or_else(|| tcfg.sft_file.clone())
        .unwrap_or_else(|| {
            panic!("未指定 SFT 语料：用 --sft-file，或在 {config_path} 里设置 train.sft_file")
        });

    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            (
                "预训练权重",
                format!("{pretrained_path}（step {}）", ckpt.step),
            ),
            ("预训练模型结构", format!("{:?}", ckpt.model)),
            (
                "LoRA 形态",
                format!(
                    "rank {rank_used} / alpha {alpha_used}（alpha/rank = {:.4}）｜挂载 {targets_used}｜{}",
                    alpha_used / rank_used as f32,
                    if resume_lora { "链式续训（沿用存档的适配层）" } else { "新挂一套适配层" }
                ),
            ),
            ("总参数", total_params.to_string()),
            (
                "可训练参数（适配层）",
                format!("{trainable_params}（{:.3}%）",
                    100.0 * trainable_params as f32 / total_params.max(1) as f32),
            ),
            ("SFT 语料", sft_paths.clone()),
            ("输出目录", out_dir.clone()),
        ],
    );

    let sft_texts = load_texts(&sft_paths);

    let loader = SftLoader::from_texts(&sft_texts, &tokenizer, ckpt.model.block_size, tcfg.batch_size);
    logln!(
        "SFT 语料：{} 段对话，打包 {} token | 监督位置（回答段）占 {:.1}%",
        loader.num_conversations(),
        loader.num_tokens(),
        100.0 * loader.supervised_ratio(),
    );
    let best = train::train_transformer(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&out_dir),
        None,
        &mut rng,
    );

    // 分词器一并放进输出目录，这个目录就是自包含的，推理时不必再指回预训练目录
    tokenizer.save(&format!("{out_dir}/tokenizer.json"));

    let what = if resume_lora { "链式续训" } else { "LoRA 微调" };
    runlog::append(&format!("{what}结束：best val loss = {best:.4}"));
    logln!("{what}完成，best val loss = {best:.4}");
    logln!(
        "用下面的命令对话（存档里已含冻结主干，不必再指回预训练权重）：\n  \
         cargo run --release -- chat --ckpt {out_dir}/best.ckpt --tokenizer {out_dir}/tokenizer.json\n  \
         cargo run --release -- chat --ckpt {out_dir}/final.ckpt --tokenizer {out_dir}/tokenizer.json\n  \
         cargo run --release -- chat --ckpt {out_dir}/best.ckpt --tokenizer {out_dir}/tokenizer.json --merge-lora  # 增量并入主干后再推理\n\
         接着训这套适配层（链式续训，别加 rank/alpha/targets）：\n  \
         cargo run --release -- finetune --pretrained {out_dir}/best.ckpt --resume-lora"
    );
    runlog::finish();
}

/// 生成预设配置文件
fn cmd_preset(name: &str, output: &str) {
    let cfg = Config::from_preset(name);
    cfg.save(output);
    println!("已生成 '{}' 预设配置到 {}", name, output);
    println!("  模型：n_embd={} n_head={} n_layer={} block_size={}",
        cfg.model.n_embd, cfg.model.n_head, cfg.model.n_layer, cfg.model.block_size);
    println!("  训练：steps={} batch_size={} max_lr={}", 
        cfg.train.steps, cfg.train.batch_size, cfg.train.max_lr);
    if cfg.model.use_rmsnorm {
        println!("  架构：LLaMA 风格（RMSNorm + SwiGLU + GQA）");
    } else {
        println!("  架构：经典风格（LayerNorm + GELU + MHA）");
    }
}

// ==================== 性能基准（bench 子命令） ====================

/// 性能基准：用**固定、可复现、短时**的任务测训练与推理吞吐（tok/s）。
///
/// 目的：优化改动前后在同一台机器、同一套参数下对比，不必跑完整训练。
/// 模型/数据/步数全部写死，只受 `--steps`、`--gen-tokens` 影响，
/// 因此两次运行的差异只来自代码本身。
fn cmd_bench(steps: usize, gen_tokens: usize) {
    use std::time::Instant;

    println!("=== 性能基准（bench）===");
    println!("rayon 线程数：{}", rayon::current_num_threads());

    let mut rng = Rng::new(1234);
    let tokenizer = Tokenizer::char(data::CORPUS);
    let vocab_size = tokenizer.vocab_size();
    let gcfg = TransformerConfig {
        vocab_size,
        n_embd: 128,
        n_head: 4,
        n_layer: 2,
        block_size: 64,
        ..TransformerConfig::default()
    };
    let model = Transformer::new(gcfg.clone(), &mut rng);
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!(
        "模型：n_layer={} n_embd={} n_head={} block={} vocab={} | 参数 {}",
        gcfg.n_layer, gcfg.n_embd, gcfg.n_head, gcfg.block_size, vocab_size, param_count
    );

    // ---- 训练吞吐 ----
    let batch = 4;
    let loader = DataLoader::new(data::CORPUS, &tokenizer, gcfg.block_size, batch);
    let tcfg = config::TrainConfig {
        seed: 42,
        batch_size: batch,
        steps,
        max_lr: 6e-4,
        min_lr: 6e-5,
        warmup_steps: (steps / 10).max(1),
        eval_every: steps + 1, // 基准不评估，避免干扰计时
        log_file: None,        // 基准不落盘
        ..config::TrainConfig::default()
    };
    let t0 = Instant::now();
    train::train_transformer(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);
    let train_secs = t0.elapsed().as_secs_f64();
    let train_tokens = steps * batch * gcfg.block_size;
    println!(
        "[bench] train     : {} steps | {:.3}s | {:.4}s/step | {:.0} tok/s",
        steps,
        train_secs,
        train_secs / steps.max(1) as f64,
        train_tokens as f64 / train_secs
    );

    // ---- 推理吞吐 ----
    // 单次生成只几十毫秒，计时抖动大；先预热一次（线程池/首次分配），
    // 再重复若干次取**最短**耗时作为吞吐上限，前后对比才稳定。
    fn bench_generate(
        model: &Transformer,
        tokenizer: &Tokenizer,
        prompt: &str,
        n: usize,
        kv: KvOpts,
        reps: usize,
    ) -> f64 {
        let mut rng = Rng::new(42);
        let opts = SampleOpts::default();
        let _ = generate(model, tokenizer, prompt, n, &opts, kv, &mut rng);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = generate(model, tokenizer, prompt, n, &opts, kv, &mut rng);
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    }

    let prompt = "Once upon a time";
    let prompt_len = tokenizer.encode(prompt).len();
    // KV cache 模式下上下文总长达到 block_size 就会停，这里取不超过该上限
    let kv_new = gen_tokens.min(gcfg.block_size.saturating_sub(prompt_len + 1));
    let kv_secs = bench_generate(&model, &tokenizer, prompt, kv_new, KvOpts::on(0, None), 5);
    println!(
        "[bench] infer/kv  : {} tok | {:.4}s | {:.1} tok/s",
        kv_new,
        kv_secs,
        kv_new as f64 / kv_secs
    );

    let full_new = gen_tokens.min(24);
    let full_secs = bench_generate(&model, &tokenizer, prompt, full_new, KvOpts::off(), 3);
    println!(
        "[bench] infer/full: {} tok | {:.4}s | {:.1} tok/s",
        full_new,
        full_secs,
        full_new as f64 / full_secs
    );
    println!("（以上 tok/s 越高越好；优化前后同机对比即可看出收益）");
}

// ==================== Scaling Laws 实验（scaling 子命令） ====================

/// 解析 `2x64,4x128` 形式的规模网格（层数 x 宽度）
fn parse_sizes(spec: &str) -> Vec<(usize, usize)> {
    let sizes: Vec<(usize, usize)> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let (l, d) = s
                .split_once(|c: char| c == 'x' || c == 'X' || c == '*')
                .unwrap_or_else(|| panic!("规模 `{s}` 格式不对，应为 `层数x宽度`（如 4x128）"));
            let l: usize = l
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("规模 `{s}` 的层数不是整数"));
            let d: usize = d
                .trim()
                .parse()
                .unwrap_or_else(|_| panic!("规模 `{s}` 的宽度不是整数"));
            assert!(l >= 1 && d >= 1, "规模 `{s}` 的层数与宽度都必须 >= 1");
            (l, d)
        })
        .collect();
    assert!(!sizes.is_empty(), "规模网格不能为空（形如 2x64,4x128）");
    sizes
}

/// 解析 `1,2,4,8` 形式的数据量倍数列表
fn parse_multiples(spec: &str) -> Vec<usize> {
    let v: Vec<usize> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse()
                .unwrap_or_else(|_| panic!("数据量倍数 `{s}` 不是正整数"))
        })
        .collect();
    assert!(!v.is_empty(), "数据量倍数列表不能为空（形如 1,2,4,8）");
    assert!(v.iter().all(|&k| k >= 1), "数据量倍数必须 >= 1");
    v
}

/// 打印一次幂律拟合的解读。
///
/// 两点讲究：
/// - `α` 与论文值**不该**接近（规模区间差好几个数量级），所以这里只做中性对照，
///   不写"应该更大 / 更小"这种会随数据翻转的话；
/// - `b` 贴住搜索下界（0）时明说"这个参数没信息量"——否则"b = 0 ⇒ 压不到 0 以下"
///   是句废话，还会让读者以为真测出了不可约损失。
fn print_fit_readout(name: &str, paper_alpha: f64, law: &scaling::PowerLaw, min_loss: f64) {
    logln!(
        "  解读：α_{name} = {:.4}（论文在它自己的规模区间测得 {paper_alpha}；本项目尺度差几个数量级，\n  \
         数值不必接近——要对齐的是方法与口径，不是这个数）。",
        law.alpha
    );
    if law.b <= 1e-6 * min_loss {
        logln!(
            "  不可约损失 b ≈ 0（{:.2e}，不到最小 loss 的 {:.3}%）：在给定的跨度内还看不出不可约损失，\n  \
             纯幂律就够解释——继续放大规模 / 加数据仍然有效，还没到「压不下去」的那一档。",
            law.b,
            100.0 * law.b / min_loss
        );
    } else {
        logln!(
            "  不可约损失 b = {:.4}：这套语料 + 这个架构下，规模再大也压不到它以下。",
            law.b
        );
    }
}

/// 打印一个扫描点的实测结果与幂律拟合的预测值
fn print_scan_row(p: &scaling::ScanPoint, x: f64, law: &scaling::PowerLaw) {
    let pred = law.predict(x);
    logln!(
        "  {:<8} | 非嵌参 {:>9} | token {:>10.3e} | D/N {:>6.1} | C {:>9.2e} | 实测 loss {:.4} | 拟合 {:.4} | 偏差 {:+.4}",
        p.label,
        p.params_non_embedding,
        p.tokens,
        p.tokens_per_param(),
        p.compute(),
        p.loss,
        pred,
        p.loss - pred
    );
}

/// Scaling Laws 实验：先做**预算规划**（最优配比 + 算力/时长估算），再**实测扫描**
/// （放大模型测 loss vs N、放大数据测 loss vs D），最后用实测点拟合幂律。
///
/// 与只把论文数字抄一遍的区别：这里的每条幂律指数都是从真实训练里拟合出来的，
/// 数字可能与论文差很远（tiny 模型 + 小语料，根本不在论文的规模区间），
/// 但**方法**是同一套：口径统一（非嵌入参数、6ND）、同一 token 预算、固定随机种子。
#[allow(clippy::too_many_arguments)]
fn cmd_scaling(
    config_path: &str,
    sizes_spec: &str,
    steps: usize,
    multiples_spec: &str,
    batch_size: usize,
    block_size: usize,
    lr: f32,
    seed: u64,
    budget: f64,
    gpu_tflops: f64,
    n_gpu: usize,
    mfu: f64,
    out: Option<&str>,
) {
    let log_path = runlog::start("scaling");
    println!("运行日志：{log_path}");
    let cfg = Config::load(config_path);
    let sizes = parse_sizes(sizes_spec);
    let multiples = parse_multiples(multiples_spec);
    assert!(steps >= 2, "每个规模至少要训 2 步（实际 {steps}）");
    assert!(budget > 0.0, "算力预算必须为正（实际 {budget}）");
    // 幂律 `L = a·x^(-α) + b` 有三个待定参数，少于 3 个点连方程都列不出。
    // 在入口处挡住（而不是等拟合函数深处 panic）才能给出"该加哪个参数"的提示。
    assert!(
        sizes.len() >= 3,
        "规模网格至少 3 个（拟合 L(N) = a·N^(-α) + b 需要 3 个点，实际 {} 个：{sizes_spec}）",
        sizes.len()
    );
    assert!(
        multiples.len() >= 3,
        "数据量倍数至少 3 个（拟合 L(D) = a·D^(-α) + b 需要 3 个点，实际 {} 个：{multiples_spec}）",
        multiples.len()
    );

    // 语料与分词器：整场扫描共用同一份，否则不同规模的 loss 之间没有可比性
    let train_docs = load_documents(&cfg.train.train_file);
    let val_text = cfg.train.val_file.as_deref().map(read_text);
    let tokenizer =
        Tokenizer::from_name(&cfg.train.tokenizer, &train_docs.join("\n"), cfg.train.bpe_vocab);

    // 模型基底：沿用配置里的结构开关（RMSNorm / SwiGLU / GQA / dropout），只改层数与宽度
    let mut base = cfg.model.clone();
    base.vocab_size = tokenizer.vocab_size();
    base.block_size = block_size;

    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("规模网格", sizes_spec.to_string()),
            ("每规模步数", steps.to_string()),
            ("数据量倍数", multiples_spec.to_string()),
            (
                "批大小 / 上下文 / 学习率",
                format!("{batch_size} / {block_size} / {lr}"),
            ),
            ("分词器", format!("{}（词表 {}）", tokenizer.kind(), tokenizer.vocab_size())),
            (
                "模型基底",
                format!(
                    "n_head={} n_kv_head={} rmsnorm={} swiglu={}",
                    base.n_head, base.n_kv_head, base.use_rmsnorm, base.use_swiglu
                ),
            ),
            ("算力预算", format!("{budget:.3e} FLOPs")),
            (
                "硬件假设",
                format!("{n_gpu} × {gpu_tflops} TFLOPS，MFU {mfu}"),
            ),
        ],
    );

    // ---------- 1. 预算规划：最优配比 + 时长/电费 ----------
    logln!("=== 一、算力预算 {budget:.3e} FLOPs 该怎么分配 ===");
    let ratio20 = scaling::ratio20_optimal(budget);
    let parametric = scaling::parametric_optimal(budget);
    let hw = scaling::Hardware {
        gpu_tflops,
        n_gpu,
        mfu,
        ..scaling::Hardware::default()
    };
    logln!(
        "  20:1 法则（论文头条结论，IsoFLOP 实测）：N {:.3e} 参数 | D {:.3e} token | {:.1} token/参数 | 预测 loss {:.3}",
        ratio20.params,
        ratio20.tokens,
        ratio20.tokens_per_param(),
        ratio20.loss
    );
    logln!(
        "  参数化损失闭式解（Approach 3）：          N {:.3e} 参数 | D {:.3e} token | {:.1} token/参数 | 预测 loss {:.3}",
        parametric.params,
        parametric.tokens,
        parametric.tokens_per_param(),
        parametric.loss
    );
    logln!(
        "  两条路线的模型规模相差 {:.2}×——论文的三个方法本身就不完全一致（正文常数经四舍五入后\n  \
         会让 Approach 3 偏离前两个方法），所以实践里只把它当**量级**指导，不要当精确解。",
        (parametric.params / ratio20.params).max(ratio20.params / parametric.params)
    );
    logln!("  Chinchilla 论文配比表（算力每行 ×10，N 与 D 各 ×√10；逐行满足 C = 6ND、D/N ≈ 20）：");
    for (c, n, d) in scaling::CHINCHILLA_TABLE {
        let w = hw.estimate(c);
        logln!(
            "    C {:.2e} FLOPs → N {:.2e} 参数 | D {:.2e} token | 按本机估算 {:.2} 天 | 预测 loss {:.3}",
            c,
            n,
            d,
            w.days,
            scaling::chinchilla_loss(n, d)
        );
    }
    let wc = hw.estimate(ratio20.compute);
    logln!(
        "  按上面的 20:1 配比：C ≈ 6ND = {:.3e} FLOPs | 有效算力 {:.2e} FLOPS/s | 约 {:.1} 天（{:.3e} 秒）| {:.0} kWh | 电费 ${:.0}",
        ratio20.compute,
        wc.effective_flops_per_sec,
        wc.days,
        wc.seconds,
        wc.energy_kwh,
        wc.cost_usd
    );
    logln!("  常见规模的 Chinchilla 最优配比（20:1，给定 N 反解 D 与 C）：");
    for p in [1.0e9f64, 7.0e9, 13.0e9, 70.0e9] {
        let a = scaling::optimal_for_params(p);
        let d = hw.estimate(a.compute);
        logln!(
            "    {:.0}B 参数 → {:.3e} token | C {:.2e} FLOPs | 约 {:.1} 天 / {} 卡 | 电费 ${:.0}",
            p / 1e9,
            a.tokens,
            a.compute,
            d.days,
            n_gpu,
            d.cost_usd
        );
    }
    logln!("  过训练/欠训练（固定算力 C，把数据量放大 k 倍、模型相应变小）：");
    for k in [0.25f64, 0.5, 1.0, 2.0, 4.0, 8.0] {
        let a = scaling::overtrain(budget, k);
        logln!(
            "    k = {:>4}× → N {:.3e} | D {:.3e} | 预测 loss {:.4}{}",
            k,
            a.params,
            a.tokens,
            a.loss,
            if (k - 1.0).abs() < 1e-9 { "  ← 20:1 配比点（D/N = 20）" } else { "" }
        );
    }
    // 同一条 IsoFLOP 曲线上，参数化损失的最小值不在 k = 1：
    // 曲线上 D/N = 20k²，令它等于参数的闭式解 D*/N* 即得 k* = √((D*/N*)/20)。
    let k_star = (parametric.tokens_per_param() / scaling::TOKENS_PER_PARAM_OPTIMAL).sqrt();
    logln!(
        "  注意：这条曲线上预测 loss 的最小值在 k = {k_star:.2}（由参数化闭式解的 D*/N* 反解），\n  \
         而不是 k = 1——k = 1 是 20:1 经验法则的落点。两个口径的分歧就摆在同一张表里。"
    );

    // ---------- 2. 参数量扫描 ----------
    let sc = scaling::ScanConfig {
        steps,
        batch_size,
        block_size,
        max_lr: lr,
        seed,
    };
    logln!("=== 二、实测：loss vs 参数量 N（固定 {} token 预算） ===", sc.tokens_per_point(steps));
    let pts = scaling::params_scan(
        &base,
        &sizes,
        &train_docs,
        val_text.as_deref(),
        &tokenizer,
        &sc,
    );
    let fit_n = scaling::fit_over_params(&pts);
    logln!("  —— 实测点 ——");
    for p in &pts {
        print_scan_row(p, p.params_non_embedding as f64, &fit_n);
    }
    logln!(
        "  幂律拟合：L(N) = {:.4}·N^(-{:.4}) + {:.4}｜r² = {:.6}（{} 个点）",
        fit_n.a,
        fit_n.alpha,
        fit_n.b,
        fit_n.r2,
        fit_n.n_points
    );
    print_fit_readout(
        "N",
        0.076,
        &fit_n,
        pts.iter().map(|p| p.loss).fold(f64::INFINITY, f64::min),
    );
    // 固定 token 预算时，模型一大就掉进"数据受限"区间：参数量涨、val loss 可能不降反升。
    // 这不是 bug，是 Chinchilla 讲的那件事本身——不提示的话，读表的人会把过拟合当成噪声。
    let max_ratio = pts
        .iter()
        .map(|p| p.tokens_per_param())
        .fold(0.0f64, f64::max);
    if max_ratio < 1.0 {
        logln!(
            "  ⚠ 读表提示：本表所有点的 D/N < 1（最大 {max_ratio:.2}），模型已进入**数据受限**区间——\n  \
             固定 token 预算放大模型时，val loss 可能不降反升（过拟合压过容量收益），曲线因此可能非单调。\n  \
             要测纯 N 的幂律，应把 --sizes 收到更小的规模，或把 --steps 加大到 token 预算跟得上参数。"
        );
    }

    // ---------- 3. 数据量扫描 ----------
    // 用**最小**的模型：过训练分析的目的是看数据量的边际收益，模型越小越便宜，
    // 而且论文的练习也是"在小模型上做"
    let (small_layer, small_embd) = sizes[0];
    let small_cfg = scaling::scan_model_config(&base, small_layer, small_embd, block_size);
    let max_mult = multiples.iter().copied().max().unwrap_or(1);
    let base_steps = (steps / max_mult).max(2);
    let tsc = scaling::ScanConfig {
        steps: base_steps,
        ..sc
    };
    logln!(
        "=== 三、实测：loss vs 数据量 D（固定模型 {}x{}，基准 {} 步，倍数 {:?}） ===",
        small_layer,
        small_embd,
        base_steps,
        multiples
    );
    let tpts = scaling::tokens_scan(
        &small_cfg,
        &multiples,
        &train_docs,
        val_text.as_deref(),
        &tokenizer,
        &tsc,
    );
    let fit_d = scaling::fit_over_tokens(&tpts);
    logln!("  —— 实测点 ——");
    for p in &tpts {
        print_scan_row(p, p.tokens, &fit_d);
    }
    logln!(
        "  幂律拟合：L(D) = {:.4}·D^(-{:.4}) + {:.4}｜r² = {:.6}（{} 个点）",
        fit_d.a,
        fit_d.alpha,
        fit_d.b,
        fit_d.r2,
        fit_d.n_points
    );
    print_fit_readout(
        "D",
        0.095,
        &fit_d,
        tpts.iter().map(|p| p.loss).fold(f64::INFINITY, f64::min),
    );
    logln!(
        "  数据量翻倍带来的 loss 下降 = {:.4} nats（在第一个点处），幂律的「回报递减」就体现在\n  \
         这个数是个常数：翻倍一次拿一次，不会越拿越多。",
        fit_d.loss_drop_per_doubling(tpts[0].tokens)
    );

    // ---------- 4. CSV ----------
    if let Some(path) = out {
        let mut s = String::from("axis,label,n_layer,n_embd,params,params_non_embedding,steps,tokens,compute_flops,loss\n");
        for (axis, list) in [("params", &pts), ("tokens", &tpts)] {
            for p in list.iter() {
                s.push_str(&format!(
                    "{},{},{},{},{},{},{},{:.0},{:.6e},{:.6}\n",
                    axis,
                    p.label,
                    p.n_layer,
                    p.n_embd,
                    p.params,
                    p.params_non_embedding,
                    p.steps,
                    p.tokens,
                    p.compute(),
                    p.loss
                ));
            }
        }
        config::ensure_parent_dir(path);
        std::fs::write(path, s).unwrap_or_else(|e| panic!("无法写入 {path}: {e}"));
        logln!("扫描结果已写入 {path}（含 params / tokens 两组点，可直接画图）");
    }

    runlog::finish();
}

// ==================== MoE 稀疏专家实验（moe 子命令） ====================

/// 解析 `0,1.0,1.25` 形式的容量因子列表
fn parse_factors(spec: &str) -> Vec<f32> {
    let v: Vec<f32> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| {
            let f: f32 = s
                .parse()
                .unwrap_or_else(|_| panic!("容量因子 `{s}` 不是数字"));
            assert!(f >= 0.0, "容量因子不能为负（0 = 不限容量）：{s}");
            f
        })
        .collect();
    assert!(!v.is_empty(), "容量因子列表不能为空（形如 0,1.0,1.25）");
    v
}

/// 按 MoE 配置建一个模型（同 seed ⇒ 逐位相同的初始权重，可直接做对照）
///
/// `top_k == 1` 时自动开 `moe_switch_gate`（Switch Transformer 式原概率门控）：
/// 重归一化口径在 K = 1 时 `w ≡ 1`，对 logits 的雅可比恒为 0，主损失给不了路由器
/// 任何梯度——那样"α = 0 的那一份"连路由器都不更新，对照实验就没意义了。
fn moe_model(
    vocab: usize,
    n_embd: usize,
    n_layer: usize,
    block_size: usize,
    n_expert: usize,
    top_k: usize,
    aux_coef: f32,
    z_loss: f32,
    seed: u64,
) -> Transformer {
    assert!(
        n_embd % 4 == 0,
        "n_embd 必须能被 n_head=4 整除（实际 {n_embd}）"
    );
    let cfg = TransformerConfig {
        vocab_size: vocab,
        n_embd,
        n_head: 4,
        n_layer,
        block_size,
        dropout: 0.0,
        n_expert,
        moe_top_k: top_k,
        moe_capacity_factor: 0.0,
        moe_aux_coef: aux_coef,
        moe_z_loss_coef: z_loss,
        moe_switch_gate: n_expert > 1 && top_k == 1,
        ..TransformerConfig::default()
    };
    let mut rng = Rng::new(seed);
    Transformer::new(cfg, &mut rng)
}

/// 跑若干批前向，累计各层 MoE 的专家负载（纯诊断：`no_grad`，不建计算图）。
///
/// 单批的负载有随机性（哪些 token 恰好落在哪个专家上），要看"训练后路由是否真的被均衡了"
/// 必须跨多批累计。这里用**训练区**采样：`moe` 子命令的加载器不切验证集
/// （[`data::CORPUS`] 只有几百个 token），而且路由分布本就该看训练分布。
fn moe_probe(model: &Transformer, loader: &dyn BatchSource, batches: usize, rng: &mut Rng) -> moe::RouteStats {
    let mut counts: Vec<usize> = Vec::new();
    let (mut dropped, mut n_tokens) = (0usize, 0usize);
    let mut aux = 0.0f32;
    for _ in 0..batches {
        let (x, _, _) = loader.sample_batch(rng);
        // 诊断只看路由，前向不需要建图；`let _ =` 让张量在 no_grad 作用域内就析构
        let _ = tensor::no_grad(|| {
            model.forward(&x, loader.batch_size(), loader.block_size(), None, false)
        });
        let st = model.route_stats().expect("MoE 模型必有路由统计");
        if counts.is_empty() {
            counts = vec![0; st.counts.len()];
        }
        for (i, c) in st.counts.iter().enumerate() {
            counts[i] += c;
        }
        dropped += st.dropped;
        n_tokens += st.n_tokens;
        aux += st.aux;
    }
    moe::RouteStats {
        counts,
        dropped,
        n_tokens,
        aux: aux / batches as f32,
    }
}

/// 把模型里所有 MoE 路由器改造成"**已经塌缩**"的样子：权重清零，偏置只给专家 0
/// 一个巨大的正值。
///
/// 为什么要人为造这个初始状态：路由塌缩是"富者愈富"的**长期**训练动力学，小模型、
/// 几百步是跑不出来的（见 `cmd_moe` 第三节的实测）。但"辅助损失能不能把塌缩的路由
/// 拉回均衡"是个可直接验证的命题——把初始状态直接摆到塌缩点上，隔离地只优化 L_aux
/// 就够了，不必等主训练跑到那里。
fn skew_router(model: &Transformer, n_expert: usize) {
    let mut hit = 0usize;
    for (name, p) in model.named_parameters() {
        if name.ends_with(".moe.router.weight") {
            p.set_data(vec![0.0f32; p.numel()]);
            hit += 1;
        } else if name.ends_with(".moe.router.bias") {
            let mut b = vec![0.0f32; p.numel()];
            // 偏置差 3 ⇒ p_0 ≈ 0.94：所有 token 的首选都是专家 0，但 softmax **没有饱和**
            // （差 8 会让 p_0 ≈ 1.0，别的专家概率只剩 3e-4，辅助损失要走很远才能把它们抬起来）
            b[0] = 3.0;
            p.set_data(b);
            assert_eq!(p.numel(), n_expert);
        }
    }
    assert!(hit > 0, "模型里没有 MoE 路由器（n_expert 是否 ≤ 1？）");
}

/// 隔离实验：**不跑主损失**，只优化 L_aux 若干步，看负载能不能从塌缩被推平。
///
/// 返回 `(优化前的路由统计, 优化后的路由统计)`。调用前模型的 `moe_aux_coef` 必须 > 0
/// （否则 [`Transformer::aux_loss`] 返回 `None`）。
fn aux_only_balance(
    model: &Transformer,
    loader: &dyn BatchSource,
    steps: usize,
    lr: f32,
    seed: u64,
) -> (moe::RouteStats, moe::RouteStats) {
    let before = {
        let mut probe_rng = Rng::new(seed + 1);
        moe_probe(model, loader, 16, &mut probe_rng)
    };
    let mut opt = AdamW::new(lr, model.parameters(), 0.0);
    let mut rng = Rng::new(seed);
    for _ in 0..steps {
        let (x, _, _) = loader.sample_batch(&mut rng);
        // 只看 L_aux：主损失对路由分布完全不敏感，加不加进这条实验都不影响结论
        model.forward(&x, loader.batch_size(), loader.block_size(), None, true);
        let aux = model.aux_loss().expect("α > 0 时必有辅助损失");
        opt.zero_grad();
        aux.backward();
        // 主损失那半张图被丢弃、永远不会被反向，其 tape 条目不可达也就不会被
        // backward 回收——这里整图反完 aux 后显式清空，防止 demo 循环无限增长
        clear_tape();
        opt.step();
    }
    let after = {
        let mut probe_rng = Rng::new(seed + 1);
        moe_probe(model, loader, 16, &mut probe_rng)
    };
    (before, after)
}

/// 打印一次路由负载的直方图 + 诊断指标（`top_k` 用于算丢弃率的分母）
fn print_route_report(tag: &str, st: &moe::RouteStats, top_k: usize) {
    let max = st.counts.iter().copied().max().unwrap_or(0).max(1);
    let used = st.counts.iter().filter(|&&c| c > 0).count();
    let e = st.counts.len();
    logln!(
        "  {tag}：不均衡度 {:.2}（1.0 = 完美均衡，E = 全压一个）｜用到的专家 {}/{}｜丢弃 {}/{} = {:.2}%｜L_aux {:.3}",
        st.imbalance(),
        used,
        e,
        st.dropped,
        st.n_tokens * top_k,
        100.0 * st.dropped_ratio(top_k),
        st.aux
    );
    // 40 列宽的条形图：一眼看出是"几个专家吃满"还是"大体摊平"
    for (i, &c) in st.counts.iter().enumerate() {
        let w = (40.0 * c as f64 / max as f64).round() as usize;
        logln!(
            "    专家 {i:>2} |{}{}| {:>7}",
            "█".repeat(w),
            " ".repeat(40 - w),
            c
        );
    }
}

/// MoE 稀疏专家实验（第 32 课，`moe` 子命令）
///
/// 三件事：
/// 1. **口径**：MoE 省的是 FLOPs 不是显存——全部专家都要驻留显存，每 token 只算 Top-K 个。
///    用真实建出的稠密模型与 MoE 模型逐位核对参数量，而不是只背公式。
/// 2. **负载均衡对照实验**：同样的初始权重、同样的语料与步数，一份加辅助损失、一份不加。
///    不加的那份会赢者通吃，大部分专家拿不到任何梯度——这就是辅助损失存在的理由。
/// 3. **容量因子与 Token Dropping**：拿上面那个偏斜的模型，扫不同的容量因子看丢弃比例。
#[allow(clippy::too_many_arguments)]
fn cmd_moe(
    experts_spec: &str,
    top_k: usize,
    steps: usize,
    batch_size: usize,
    block_size: usize,
    lr: f32,
    n_embd: usize,
    n_layer: usize,
    aux_coef: f32,
    z_loss: f32,
    factors_spec: &str,
    seed: u64,
) {
    let log_path = runlog::start("moe");
    println!("运行日志：{log_path}");

    let experts_list = parse_multiples(experts_spec);
    let factors = parse_factors(factors_spec);
    assert!(steps >= 2, "至少要训 2 步（实际 {steps}）");

    // 内置语料 + 字符分词器：自包含、可复现，不需要外部数据文件
    let tokenizer = Tokenizer::char(data::CORPUS);
    let vocab = tokenizer.vocab_size();
    let loader = DataLoader::new(data::CORPUS, &tokenizer, block_size, batch_size);

    runlog::fields(
        "本次运行参数",
        &[
            ("专家数网格", experts_spec.to_string()),
            ("Top-K", top_k.to_string()),
            ("步数", steps.to_string()),
            (
                "批大小 / 上下文 / 学习率",
                format!("{batch_size} / {block_size} / {lr}"),
            ),
            (
                "模型",
                format!("n_layer={n_layer} n_embd={n_embd} n_head=4（GELU FFN）"),
            ),
            ("辅助损失系数 α", aux_coef.to_string()),
            ("router z-loss 系数 β", z_loss.to_string()),
            ("容量因子扫描", factors_spec.to_string()),
            (
                "分词器",
                format!("{}（词表 {vocab}）", tokenizer.kind()),
            ),
        ],
    );

    // ---------- 一、稀疏口径 ----------
    logln!("=== 一、参数 / 计算量口径：MoE 省的是 FLOPs，不是显存 ===");
    logln!(
        "  每个专家就是一个普通 FFN（GELU 版 8d²+5d = {} 参数，d = {n_embd}）；\n  \
         全部专家都要驻留显存，每个 token 只走 Top-K 个。",
        moe::expert_param_count(n_embd, false)
    );
    // 稠密基线：与 MoE 只差前馈子层，逐位可比
    let dense = moe_model(vocab, n_embd, n_layer, block_size, 1, 1, 0.0, 0.0, seed);
    let dense_params: usize = dense.parameters().iter().map(|p| p.numel()).sum();
    logln!("  稠密基线（n_expert=1）实测参数：{dense_params}");
    for &e in &experts_list {
        if e < 2 {
            logln!("  跳过 n_expert={e}（< 2 就是稠密，没有稀疏可言）");
            continue;
        }
        let k = top_k.min(e);
        let model = moe_model(vocab, n_embd, n_layer, block_size, e, k, aux_coef, z_loss, seed);
        let measured: usize = model.parameters().iter().map(|p| p.numel()).sum();
        let ss = moe::sparse_stats(e, k, n_embd, false);
        let per_layer_total = ss.total_params();
        let per_layer_active = ss.active_params();
        // 每 token 激活参数 = 模型总参数 - 每层没被激活的 (E-K) 个专家
        let activated = measured - n_layer * (e - k) * ss.expert_params;
        // 口径核对：实测模型参数必须与"单层公式 × 层数 + 非 FFN 部分"吻合
        let expect = dense_params + n_layer * (per_layer_total - ss.expert_params);
        assert_eq!(
            measured, expect,
            "E={e}：实测参数 {measured} 与公式 {expect} 不符（公式与建层漂移了）"
        );
        logln!(
            "  E={e:<3} K={k} | 单层 总 {per_layer_total} / 激活 {per_layer_active}\
             （{:.2}× / 省 {:.1}% FLOPs） | 模型 总 {measured} / 激活 {activated}\
             （{:.2}× / 省 {:.1}% FLOPs）",
            ss.param_ratio(),
            100.0 * ss.flops_saving(),
            measured as f64 / activated as f64,
            100.0 * (1.0 - activated as f64 / measured as f64)
        );
    }
    logln!(
        "  读法：MoE 的显存开销按**总参数**算（全部专家都在显存里），省下的是每 token 的乘加次数。\n  \
         所以 MoE 的卖点不是「显存换性能」，而是「同样的 FLOPs 预算下能塞进更多参数」。"
    );

    // ---------- 二、负载均衡辅助损失的隔离实验 ----------
    let e = *experts_list
        .iter()
        .find(|&&x| x >= 2)
        .unwrap_or_else(|| panic!("专家数网格里至少要有一个 >= 2 的值（实际 {experts_spec}）"));
    let k = top_k.min(e);
    logln!("=== 二、负载均衡辅助损失：隔离实验（不跑主损失，只优化 L_aux） ===");
    logln!(
        "  门控口径：{}",
        if k == 1 {
            "K = 1 ⇒ Switch Transformer 式（全部专家上的 softmax 原概率，Σw < 1）。\n  \
             若改用 Mixtral 式重归一化，Top-K 只有一个元素会让权重恒等于 1，对 logits 的雅可比\n  \
             恒为 0——主损失给不了路由器任何梯度，α = 0 的那一份连路由器都不会动。"
        } else {
            "K ≥ 2 ⇒ Mixtral / DeepSeek 式（Top-K 内部重归一化，Σw = 1）"
        }
    );
    // 路由塌缩是「富者愈富」的**长期**训练动力学，小模型几百步跑不出来。所以这里直接把
    // 路由器的状态摆到塌缩点上（权重清零、偏置只给专家 0 一个正数），再隔离地只优化
    // L_aux——"辅助损失能把塌缩拉回来吗"这个问题本身不需要等主训练跑到塌缩。
    //
    // 这条实验要跑几百步，但对模型规模毫无要求（关心的是路由器动力学），所以用单层
    // 32 维的小模型 + 小批次，跑得动就能多做几组对照。
    let balance_steps = 300usize;
    let balance_lr = 0.02f32;
    let iso_loader = DataLoader::new(CORPUS, &tokenizer, 32, 4);
    let mut iso_k_list = vec![1usize];
    if k > 1 {
        iso_k_list.push(k);
    }
    let mut iso_summary: Vec<String> = Vec::new();
    for &iso_k in &iso_k_list {
        let model = moe_model(vocab, 32, 1, 32, e, iso_k, 1.0, 0.0, seed);
        skew_router(&model, e);
        let (b, a) = aux_only_balance(&model, &iso_loader, balance_steps, balance_lr, seed);
        logln!("  —— K = {iso_k} ——");
        print_route_report("塌缩的初始路由", &b, iso_k);
        print_route_report(
            &format!("只优化 L_aux {balance_steps} 步之后"),
            &a,
            iso_k,
        );
        // 唯一稳定的不变量：只优化 L_aux 必然把它自己压下去（它就是被优化的目标）。
        // 负载会不会跟着走，是另一回事——这正是下面要说的。
        assert!(
            a.aux <= b.aux + 1e-6,
            "只优化 L_aux 竟然没把它压下去：{:.3} -> {:.3}",
            b.aux,
            a.aux
        );
        iso_summary.push(format!(
            "K={iso_k}：L_aux {:.3} → {:.3}（p 摊平 ⇒ 1.0，不是下界），不均衡度 {:.2} → {:.2}，用到的专家 {}/{} → {}/{}",
            b.aux,
            a.aux,
            b.imbalance(),
            a.imbalance(),
            b.counts.iter().filter(|&&c| c > 0).count(),
            e,
            a.counts.iter().filter(|&&c| c > 0).count(),
            e
        ));
    }
    logln!("  汇总：{}", iso_summary.join("\n        ｜"));
    logln!(
        "  读法：L_aux = E·Σ f_i p_i 对 p 是**软**的、对 f 是**硬**的（f 由 argmax 给出，\n  \
         不可导、是常数）。若 p 均匀（**不管 f 长什么样**）：Σ f_i p_i = (1/E)·Σ f_i = 1/E，\n  \
         于是 L_aux = 1——但这不是下界（f 与 p 支撑集不交时 Σ f_i p_i = 0，L_aux 可以是 0）。\n  \
         所以「L_aux 掉到 1」根本不是负载均衡的证书：\n  \
         上面两组的 L_aux 都被压到了 1.0 附近，而硬路由的不均衡度几乎没动。\n  \
         梯度方向确实指向「把被过度使用的专家按下去」（∂L/∂logit_j ∝ p_j·(f_j − Σ f_i p_i)），\n  \
         但它的**大小正比于 p_j**：p 越平，梯度越小。于是最快的下降路径是先把 p 摊平\n  \
         （L_aux 一步到位到 1），而不是把硬路由摊平——负载因此可能原地不动。\n  \
         这正是后续工作（DeepSeek-V3 的 loss-free 均衡偏置、expert-choice 路由）要绕开的东西，\n  \
         也是「辅助损失系数要调小」的真实原因：它压的是概率分布，不是分配结果。"
    );

    // ---------- 三、端到端对照实验 ----------
    logln!("=== 三、端到端对照（同种子 / 同语料 / 同 {steps} 步） ===");
    let tcfg = config::TrainConfig {
        seed,
        batch_size,
        steps,
        max_lr: lr,
        min_lr: lr * 0.1,
        warmup_steps: (steps / 10).max(1),
        eval_every: steps + 1, // 只在收尾评估一次，别让评估插进对照实验的计时里
        log_file: None,        // 实验不落训练 CSV（别覆盖 logs/train.csv）
        ..config::TrainConfig::default()
    };

    let mut skewed: Option<Transformer> = None;
    let mut end_to_end = Vec::new();
    for &alpha in &[0.0f32, aux_coef] {
        let tag = if alpha == 0.0 {
            "α = 0（不加辅助损失）".to_string()
        } else {
            format!("α = {alpha}（加辅助损失）")
        };
        let model = moe_model(vocab, n_embd, n_layer, block_size, e, k, alpha, z_loss, seed);
        let mut rng = Rng::new(seed);
        logln!("  —— {tag} ——");
        let loss = train::train_transformer(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);
        let mut probe_rng = Rng::new(seed + 1);
        let st = moe_probe(&model, &loader, 16, &mut probe_rng);
        logln!("  训练收尾 loss = {loss:.4}");
        print_route_report("路由负载（16 批累计）", &st, k);
        end_to_end.push(format!("α={alpha}: loss {loss:.4} / 不均衡度 {:.2}", st.imbalance()));
        if alpha == 0.0 {
            skewed = Some(model);
        }
    }
    logln!("  汇总：{}", end_to_end.join("｜"));
    logln!(
        "  诚实的读法：\n  \
         1) 这个规模（{n_layer} 层 × {n_embd} 维、{steps} 步）下 α = 0 那一份**不会**塌缩——\n  \
            随机初始化的路由器在几百步内大体保持对称。路由塌缩是「富者愈富」的长期动力学\n  \
            （路由器在自己已经擅长的方向上越走越专），要靠长时间训练才显形。\n  \
         2) 加了 α 之后不均衡度确实更低、loss 也没被拖累：这个小改善是真的，但别把它当万能药——\n  \
            第二节的隔离实验已经说明它压的是**概率分布**而不是**分配结果**（L_aux 掉到 1.0，\n  \
            硬路由可以原地不动）。所以真实系统里它只是个温和的正则项，系数要小（α ≲ 0.01）。\n  \
         3) 但塌缩的**代价**在任何规模下都一样：主损失只关心算得准不准，完全不关心是谁在算。\n  \
            未被选中的专家参数拿不到任何梯度（等于白占显存），而 loss 曲线看不出异常——\n  \
            所以必须有别的东西盯着路由分布。"
    );

    // ---------- 四、容量因子与 Token Dropping ----------
    logln!("=== 四、容量因子与 Token Dropping（用上面 α = 0 那份模型，负载最不均） ===");
    let mut model = skewed.expect("α = 0 那组必然建过模型");
    let n_tokens = batch_size * block_size;
    logln!("  每批 {n_tokens} 个 token，K={k} ⇒ 路由分配共 {} 条，平均每个专家 {}", n_tokens * k, n_tokens * k / e);
    for &cf in &factors {
        model.set_moe_capacity_factor(cf);
        let cap = moe::expert_capacity(n_tokens, e, k, cf);
        let mut probe_rng = Rng::new(seed + 2);
        let st = moe_probe(&model, &loader, 16, &mut probe_rng);
        logln!(
            "  cf = {cf:<5} → 每专家容量 {:<10} 丢弃 {}/{} = {:.2}% ｜读到 token 的专家 {}/{}",
            if cap == usize::MAX {
                "不限".to_string()
            } else {
                cap.to_string()
            },
            st.dropped,
            st.n_tokens * k,
            100.0 * st.dropped_ratio(k),
            st.counts.iter().filter(|&&c| c > 0).count(),
            e
        );
    }
    logln!(
        "  读法：容量是**训练时**给显存与 All-to-All 通信量封顶用的，代价是被挤掉的分配直接丢弃\n  \
         （Switch Transformer 的做法：不把这条分配退给次优专家，那样会让路由变得不可预测）。\n  \
         注意 cf = 1.0 在偏斜的路由上照样丢很多——容量按**平均负载**算，而负载根本不均。\n  \
         推理时务必设 cf = 0：否则同一个 token 的输出会取决于同批次里别的 token 挤没挤占容量，\n  \
         同一句话换个批大小就换个答案。"
    );

    runlog::finish();
}

// ==================== 第 38 课：分布式训练实验 ====================

/// 分布式训练实验（第 38 课）：集合通信 / 数据并行 / ZeRO / 张量并行 / 流水线并行 / 3D 并行。
///
/// 全部跑在**单进程**里：一个 rank 就是一个普通数据结构，相邻 rank 之间的「通信」是一次
/// 内存拷贝。集合通信按真实算法的**逐轮依赖**推进（环形 allreduce = reduce-scatter +
/// all-gather，每轮每 rank 只和左右邻居换一块），所以每节的数值都能与单卡逐位对拍；
/// 换成 NCCL / MPI 时，变的只是「这块内存由谁来搬」。
///
/// 为什么值得这么绕：分布式的坑几乎全在**切分边界**上——错开一格、少算一段、
/// 拼回时块序没转回来，这些错都不会崩、不会 NaN，只是让模型悄悄收敛到别的地方。
/// 单进程模拟能让这些边界被逐位对拍钉死，真机调试则要拿 N 张卡去猜。
fn cmd_distributed(args: &DistArgs) {
    let log_path = runlog::start("distributed");
    println!("运行日志：{log_path}");

    let (dp, tp, pp) = (args.dp, args.tp, args.pp);
    let (micros, steps) = (args.micro_batches, args.steps);
    let (per_rank_b, t) = (args.batch_size, args.block_size);
    let (lr, wd) = (args.lr, args.weight_decay);
    let (n_embd, n_layer, seed) = (args.n_embd, args.n_layer, args.seed);

    assert!(dp >= 1 && tp >= 1 && pp >= 1, "三个轴的并行度都必须 >= 1");
    assert!(micros >= 1, "micro-batch 数至少为 1（实际 {micros}）");
    assert!(steps >= 2, "至少要训 2 步（实际 {steps}）");
    assert!(
        n_embd % 4 == 0,
        "n_embd 必须能被 4 整除（n_head = 4），实际 {n_embd}"
    );
    assert!(
        wd > 0.0,
        "权重衰减必须非 0：ZeRO 分片若忘了把真实 θ 灌进分片优化器，衰减项会静默失效，\
         只有带 wd 的轨迹对拍才抓得到"
    );

    let tokenizer = Tokenizer::char(CORPUS);
    let vocab = tokenizer.vocab_size();
    let corpus_ids = tokenizer.encode(CORPUS);
    // 全局 batch = 每 rank × 数据并行度；样本按 rank 切片，切法就是 `chunk_range`
    let total_b = per_rank_b * dp;
    let need = total_b * t;
    assert!(
        corpus_ids.len() > need + 8,
        "语料只有 {} 个 token，装不下一个全局 batch（{need}）",
        corpus_ids.len()
    );
    let start = (seed as usize * 13) % (corpus_ids.len() - need);
    let ids: Vec<usize> = (0..need).map(|i| corpus_ids[start + i]).collect();
    // 目标 = 错位一位的下一个 token。刻意写成**整体可切分**的纯函数（`targets[i]` 只由 `ids`
    // 决定，不看「自己在第几片」）：若改成「每片最后一位复用自身」，各分片边界上算的就不是
    // 同一个损失，DP 与单卡的 loss 会差出一个固定偏差——那不是浮点误差。
    let targets: Vec<usize> = (0..need).map(|i| ids[(i + 1) % need]).collect();

    let make_model = |s: u64| {
        let cfg = TransformerConfig {
            n_embd,
            n_head: 4,
            n_layer,
            block_size: t,
            ..TransformerConfig::tiny(vocab)
        };
        Transformer::new(cfg, &mut Rng::new(s))
    };
    let batch_loss = |model: &Transformer, ids: &[usize], targets: &[usize], b: usize| {
        let logits = model.forward(ids, b, t, None, false);
        loss::cross_entropy_loss_masked(&logits, targets, None)
    };
    let total_params: usize = make_model(seed).parameters().iter().map(|p| p.numel()).sum();
    let shard = per_rank_b * t; // 每个 rank 一个 batch 的 token 数

    runlog::fields(
        "本次运行参数",
        &[
            (
                "切分",
                format!("dp={dp} tp={tp} pp={pp}（卡数 {}）", dp * tp * pp),
            ),
            ("流水线 micro-batch", micros.to_string()),
            ("DP / ZeRO 步数", steps.to_string()),
            (
                "全局 batch",
                format!("{total_b}（每 rank {per_rank_b}）× 上下文 {t}"),
            ),
            (
                "模型",
                format!("n_layer={n_layer} n_embd={n_embd} n_head=4（GELU FFN）"),
            ),
            ("优化器", format!("AdamW lr={lr} wd={wd}")),
            ("参数量", total_params.to_string()),
            ("分词器", format!("{}（词表 {vocab}）", tokenizer.kind())),
        ],
    );

    // ---------- 一、集合通信 ----------
    logln!("=== 一、集合通信：环形 allreduce = reduce-scatter + all-gather ===");
    logln!(
        "  通信域 N = {dp}，缓冲长度取模型参数量 {total_params}（一次梯度同步要搬的量）。\n  \
         各 rank 的本地数据与自己的编号有关，这样「某个 rank 少搬了一块」会在结果里露出来。"
    );
    let locals: Vec<Vec<f32>> = (0..dp)
        .map(|r| (0..total_params).map(|i| ((i * 37 + r * 11) % 101) as f32 * 0.01).collect())
        .collect();
    let mut world = distributed::World::new(dp);
    let ring = world.all_reduce_sum(&locals);
    let ring_log = world.log().clone();
    let naive = world.all_reduce_sum_naive(&locals);
    let naive_log = world.log().clone();
    let disagree = max_abs_diff(&ring[0], &naive[0]);
    assert!(
        disagree < 1e-4,
        "环形与主从广播的 allreduce 结果不一致（最大差 {disagree}）"
    );

    logln!("  rank ｜ 环形每 rank 发送 ｜ 主从每 rank 发送");
    for r in 0..dp {
        logln!(
            "  {:>4} ｜ {:>15} B ｜ {:>15} B",
            r,
            ring_log.per_rank_bytes()[r],
            naive_log.per_rank_bytes()[r]
        );
    }
    logln!(
        "  合计 ｜ {:>15} B ｜ {:>15} B",
        ring_log.total_sent() * 4,
        naive_log.total_sent() * 4
    );
    if dp > 1 {
        logln!(
            "  瓶颈 ｜ {:>15} B ｜ {:>15} B ← 主从是环形的 {:.1} 倍（= N/2）",
            ring_log.max_rank_bytes(),
            naive_log.max_rank_bytes(),
            naive_log.max_rank_bytes() as f64 / ring_log.max_rank_bytes() as f64
        );
    } else {
        logln!("  N = 1：没有通信，两条路径都是恒等操作");
    }

    // 两阶段拆开看：reduce-scatter 之后，每个 rank 恰好持有「某一块」的全和
    let mut scat = distributed::World::new(dp);
    let owned = scat.reduce_scatter_sum(&locals);
    logln!(
        "  reduce-scatter（{} 轮）后各 rank 持有的块下标：{:?}",
        scat.log().rounds,
        owned.iter().map(|(c, _)| *c).collect::<Vec<usize>>()
    );
    logln!(
        "  —— 块序整体左移一格：rank r 拿到的是第 (r+1) mod N 块。拼回完整参数时要按块下标\n  \
         转回来（`rotate_chunks`），否则每张卡拿着错位的参数继续训练：不崩、不 NaN，只是模型悄悄跑偏。"
    );
    logln!(
        "  读法：环形两阶段每 rank 只发 2(N-1)/N 份，而主从广播里 rank 0 要下发 (N-1) 份——\n  \
         瓶颈 rank 的通信量是环形的 N/2 倍。N 一大，主从广播的那张卡就是整个训练的地板。"
    );

    // ---------- 二、数据并行与 ZeRO ----------
    logln!("=== 二、数据并行与 ZeRO：同一条轨迹，不同的显存 ===");
    logln!(
        "  全局 batch {total_b}（每 rank {per_rank_b}）· 上下文 {t} · {n_layer} 层 × {n_embd} 维 · \
         {steps} 步 AdamW（lr {lr}、wd {wd}）"
    );
    // 不切分时，**每个** rank 都要常驻的优化器状态：m、v 各一份完整长度
    let full_state = 2 * total_params * std::mem::size_of::<f32>();

    // 参照组：vanilla DP —— 每个 rank 存一份完整的 m / v
    let ref_losses: Vec<f32> = {
        let replicas: Vec<Transformer> = (0..dp).map(|_| make_model(seed)).collect();
        let per_rank: Vec<Vec<Tensor>> = replicas.iter().map(|m| m.parameters()).collect();
        let mut opts: Vec<AdamW> = per_rank.iter().map(|p| AdamW::new(lr, p.clone(), wd)).collect();
        let mut sync = distributed::DataParallel::new(dp);
        let mut losses = Vec::new();
        for _ in 0..steps {
            for r in 0..dp {
                for p in &per_rank[r] {
                    p.zero_grad();
                }
                let lo = r * shard;
                batch_loss(
                    &replicas[r],
                    &ids[lo..lo + shard],
                    &targets[lo..lo + shard],
                    per_rank_b,
                )
                .backward();
            }
            sync.sync_gradients(&per_rank, 1.0);
            for o in opts.iter_mut() {
                o.step();
            }
            losses.push(batch_loss(&replicas[0], &ids, &targets, total_b).item());
        }
        logln!(
            "  vanilla DP：一次梯度同步 = 一个完整 allreduce（{} 轮），各 rank 合计搬 {} B / 步；\n  \
             代价是每个 rank 都常驻完整的优化器状态 m/v = {} B",
            sync.log().rounds,
            sync.log().total_sent() * 4,
            full_state
        );
        losses
    };

    // ZeRO：优化器状态（stage 1）与梯度（stage 2）按 rank 切开，单卡只养 1/N
    let mut zero_losses: Vec<(String, Vec<f32>)> = Vec::new();
    for (stage, name) in [
        (distributed::ZeroStage::One, "ZeRO-1"),
        (distributed::ZeroStage::Two, "ZeRO-2"),
    ] {
        let replicas: Vec<Transformer> = (0..dp).map(|_| make_model(seed)).collect();
        let per_rank: Vec<Vec<Tensor>> = replicas.iter().map(|m| m.parameters()).collect();
        let mut zero = distributed::ZeroOptimizer::new(stage, dp, total_params, lr, wd);
        let mut flat: Vec<Vec<f32>> =
            (0..dp).map(|_| distributed::flatten_params(&per_rank[0])).collect();
        let mut losses = Vec::new();
        for _ in 0..steps {
            let mut grads = Vec::with_capacity(dp);
            for r in 0..dp {
                for p in &per_rank[r] {
                    p.zero_grad();
                }
                let lo = r * shard;
                batch_loss(
                    &replicas[r],
                    &ids[lo..lo + shard],
                    &targets[lo..lo + shard],
                    per_rank_b,
                )
                .backward();
                grads.push(distributed::flatten_grads(&per_rank[r]));
            }
            flat = zero.step(&flat, &grads, 1.0);
            // 更新后的完整参数要写回模型，下一步才算得出正确的 loss（分片只在优化器里）
            for r in 0..dp {
                distributed::write_params(&per_rank[r], &flat[r]);
            }
            losses.push(batch_loss(&replicas[0], &ids, &targets, total_b).item());
        }

        // 轨迹必须与 vanilla DP **逐步一致**：Adam 的更新逐元素独立，第 j 个元素由谁算都不影响结果。
        // 分片最常出的错是边界错位（第 j 个元素的 m/v 配到了第 j+1 个元素的梯度），
        // 那种错不会崩也不会 NaN，只会让曲线「抖一点」——所以必须逐步对拍，不能只看趋势。
        for (i, (a, b)) in losses.iter().zip(&ref_losses).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "{name} 第 {i} 步的 loss 与 vanilla DP 不一致：{a} vs {b}"
            );
        }
        let states = zero.state_bytes_per_rank();
        // 一次 ZeRO 更新的集合通信总量：reduce-scatter (N-1) 份 + all-gather (N-1) 份
        // —— 与 vanilla DP 的完整 allreduce 完全相同（都是 2(N-1)·参数量）。
        // 这里用公式而不是 `zero.log()`：CommLog 每次集合调用都会重新计数，一次 step 里调了好几次，
        // 读到的只会是最后那一次（all-gather）的量。
        let per_step = 2 * (dp - 1) * total_params * std::mem::size_of::<f32>();
        logln!(
            "  {}：每步通信 {} B（= 2(N-1) × 参数量，与 vanilla DP 的 allreduce 同量；\
             它内部拆成 reduce-scatter + all-gather 两段）\n  \u{3000}\u{3000}\u{3000}\
             每 rank 状态 {} ~ {} B（合计 {} B），vanilla DP 每 rank {} B → 单卡状态降到 1/{}",
            name,
            per_step,
            states.iter().min().copied().unwrap_or(0),
            states.iter().max().copied().unwrap_or(0),
            states.iter().sum::<usize>(),
            full_state,
            dp
        );
        zero_losses.push((name.to_string(), losses));
    }

    logln!("  loss 曲线（均匀取样，共 {steps} 步）：");
    logln!("    vanilla DP ｜ {}", loss_curve(&ref_losses, 6));
    for (name, l) in &zero_losses {
        logln!("    {:>10} ｜ {}", name, loss_curve(l, 6));
    }
    logln!(
        "  读法：ZeRO 不是近似——它把「谁算第 j 个元素」换了个人，数学上与原方案一模一样。\n  \
         stage 1 只切优化器状态（最大的一块：每个参数要 m、v 各 4 字节，比参数本身还大）；\n  \
         stage 2 再切梯度，代价是每步通信从「完整 allreduce」换成 reduce-scatter + all-gather，\n  \
         总量不变，但完整梯度缓冲从头到尾不出现。"
    );

    // ---------- 三、张量并行 ----------
    logln!("=== 三、张量并行：层内按列 / 行切开，拼回来必须与单卡逐位一致 ===");
    let d = n_embd;
    let hidden = 2 * n_embd;
    assert!(
        hidden % tp == 0 && d % tp == 0,
        "隐藏维度 {hidden} / {d} 必须能被张量并行度 {tp} 整除"
    );
    let mut rng = Rng::new(seed + 7);
    let (w1, w2) = (
        xavier_data(d, hidden, &mut rng),
        xavier_data(hidden, d, &mut rng),
    );
    // 偏置刻意做成**位置相关**的：全用一个常数的话，「列对错位」这类错误在所有元素上都一样，
    // 梯度对拍完全看不出来
    let b1: Vec<f32> = (0..hidden).map(|i| 0.03 + 0.01 * i as f32).collect();
    let b2: Vec<f32> = (0..d).map(|i| 0.05 - 0.02 * i as f32).collect();
    let x_data: Vec<f32> = (0..per_rank_b * d).map(|_| rng.randn()).collect();

    let w1_ref = Tensor::param(w1.clone(), vec![d, hidden]);
    let b1_ref = Tensor::param(b1.clone(), vec![hidden]);
    let w2_ref = Tensor::param(w2.clone(), vec![hidden, d]);
    let b2_ref = Tensor::param(b2.clone(), vec![d]);
    let x_ref = Tensor::param(x_data.clone(), vec![per_rank_b, d]);
    let y_ref = x_ref
        .matmul(&w1_ref)
        .add(&b1_ref)
        .gelu()
        .matmul(&w2_ref)
        .add(&b2_ref);
    y_ref.mul(&y_ref).sum().backward();

    let mut tpm = distributed::TensorParallelMlp::from_full(&w1, &b1, &w2, &b2, d, hidden, tp);
    let x_tp = Tensor::param(x_data.clone(), vec![per_rank_b, d]);
    let y_tp = tpm.forward(&x_tp);
    y_tp.mul(&y_tp).sum().backward();

    // 各 rank 的权重梯度按列 / 按行拼回完整矩阵，再与单卡比
    let mut dw1_full = vec![0.0f32; d * hidden];
    let mut db1_full = vec![0.0f32; hidden];
    for r in 0..tp {
        let cols = tpm.fc1().columns(r).to_vec();
        let local = cols.len();
        let gw = tpm.fc1().weight_shard(r).grad();
        let gb = tpm.fc1().bias_shard(r).grad();
        for row in 0..d {
            for (j, &c) in cols.iter().enumerate() {
                dw1_full[row * hidden + c] = gw[row * local + j];
            }
        }
        for (j, &c) in cols.iter().enumerate() {
            db1_full[c] = gb[j];
        }
    }
    let mut dw2_full = Vec::with_capacity(hidden * d);
    for r in 0..tp {
        let (s, e) = tpm.fc2().input_range(r);
        assert_eq!(
            e - s,
            tpm.fc1().columns(r).len(),
            "列并行的输出分片必须与行并行的输入分片切在同一处"
        );
        dw2_full.extend_from_slice(&tpm.fc2().weight_shard(r).grad());
    }

    let errs = [
        ("前向输出", max_abs_diff(&y_tp.data(), &y_ref.data())),
        ("dx", max_abs_diff(&x_tp.grad(), &x_ref.grad())),
        ("dW1（按列拼回）", max_abs_diff(&dw1_full, &w1_ref.grad())),
        ("db1", max_abs_diff(&db1_full, &b1_ref.grad())),
        ("dW2（按行拼回）", max_abs_diff(&dw2_full, &w2_ref.grad())),
        ("db2", max_abs_diff(&tpm.fc2().bias().grad(), &b2_ref.grad())),
    ];
    logln!(
        "  N = {} 的切分 vs 单卡（最大逐元素差）：{}",
        tp,
        errs.iter()
            .map(|(n, e)| format!("{n} {e:.1e}"))
            .collect::<Vec<_>>()
            .join("｜")
    );
    for (n, e) in errs {
        assert!(e < 1e-5, "{n} 与单卡不一致（最大差 {e}）");
    }
    logln!(
        "  通信：整个 MLP 只在行并行的输出上做一次 allreduce（{} 轮，每 rank {} B）。\n  \
         列并行的接缝上是**零通信**（输出按列切开，各 rank 各算各的列）；\n  \
         代价是列并行后面只有「半成品」输出，谁要完整输出谁就得补一次 all_gather——\n  \
         注意力的 QKV 正是这么切的：按 head 切、每 rank 拿整 head，否则 attention 每一步都要通信。",
        tpm.log().rounds,
        tpm.log().per_rank_bytes()[0]
    );

    // ---------- 四、流水线并行 ----------
    logln!("=== 四、流水线并行：GPipe 与 1F1B 算得一样、驻留不一样 ===");
    let (p, mb) = (pp, micros);
    let total_rows = mb * per_rank_b;
    let chain = |s: u64| {
        let mut r = Rng::new(s);
        (0..p)
            .map(|_| Linear::new(d, d, &mut r))
            .collect::<Vec<Linear>>()
    };
    let ref_layers = chain(seed + 3);
    let mut rng_pp = Rng::new(seed + 11);
    let x_pp: Vec<f32> = (0..total_rows * d).map(|_| rng_pp.randn()).collect();

    // 参照组：单卡把 p 层串起来，整个 batch 一次算完
    let x_ref_pp = Tensor::param(x_pp.clone(), vec![total_rows, d]);
    let mut y = x_ref_pp.clone();
    for l in &ref_layers {
        y = l.forward(&y);
    }
    y.mul(&y).sum().mul_scalar(1.0 / total_rows as f32).backward();

    // 每个 micro-batch 的损失之和 = 全 batch 的损失，缩放因子按同一个分母给
    let loss_of = |y: &Tensor| y.mul(y).sum().mul_scalar(1.0 / total_rows as f32);
    let inputs = |i: usize| {
        Tensor::param(
            x_pp[i * per_rank_b * d..(i + 1) * per_rank_b * d].to_vec(),
            vec![per_rank_b, d],
        )
    };

    let gpipe = distributed::Pipeline::new(chain(seed + 3), distributed::Schedule::gpipe(mb, p));
    let rep_g = gpipe.run(&(0..mb).map(|i| inputs(i)).collect::<Vec<Tensor>>(), loss_of);
    let onef1b = distributed::Pipeline::new(
        chain(seed + 3),
        distributed::Schedule::one_forward_one_backward(mb, p),
    );
    let rep_o = onef1b.run(&(0..mb).map(|i| inputs(i)).collect::<Vec<Tensor>>(), loss_of);

    let outs =
        |rep: &distributed::RunReport| -> Vec<f32> { rep.outputs.iter().flat_map(|t| t.data()).collect() };
    let (g_out, o_out) = (outs(&rep_g), outs(&rep_o));
    assert!(
        max_abs_diff(&g_out, &y.data()) < 1e-5,
        "GPipe 输出（micro-batch 拼回）与单卡整批不一致"
    );
    assert!(
        max_abs_diff(&o_out, &g_out) < 1e-5,
        "1F1B 与 GPipe 的输出不一致——调度只该影响「什么时候算」，不该影响「算成什么」"
    );
    for (s, l) in gpipe.stages().iter().enumerate() {
        let (pa, pb) = (l.parameters(), ref_layers[s].parameters());
        assert!(
            max_abs_diff(&pa[0].grad(), &pb[0].grad()) < 1e-5
                && max_abs_diff(&pa[1].grad(), &pb[1].grad()) < 1e-5,
            "第 {s} 段的梯度与单卡不一致"
        );
    }

    logln!(
        "  切 {p} 段 × {mb} 个 micro-batch（每个 {per_rank_b} 行）：GPipe 输出 = 单卡整批，\
         1F1B 与 GPipe 的输出、每段梯度完全一致"
    );
    logln!(
        "  驻留峰值：GPipe 同时压着 {} 个 micro-batch，1F1B 只压 {} 个（= 段数 p）",
        rep_g.peak_in_flight,
        rep_o.peak_in_flight
    );
    logln!(
        "  跨段激活缓冲峰值：{} 块 → {} 块（= 在飞 micro-batch 数 × (p-1)）",
        rep_g.peak_boundary_buffers,
        rep_o.peak_boundary_buffers
    );
    logln!(
        "  事件表：两者都是 {} 步（= 2 × micro-batch × 段数，每个 (micro-batch, 段) 前向/反向各一次）；\n  \
         调度画像与实测峰值一致（GPipe {} / 1F1B {}），说明「省」是计划里就算出来的，不是碰巧。",
        gpipe.schedule().steps.len(),
        gpipe.schedule().peak_in_flight,
        onef1b.schedule().peak_in_flight
    );
    logln!(
        "  读法：1F1B 的驻留由**段数**决定，与 micro-batch 数无关——这正是 micro-batch 可以开到\n  \
         几十个的原因：更大的 batch 不会让显存跟着涨，气泡反而更小。"
    );

    // ---------- 五、3D 并行 ----------
    logln!(
        "=== 五、3D 并行：dp × tp × pp = {} 张卡，三轴正交 ===",
        dp * tp * pp
    );
    let cfg = distributed::DistConfig::new(dp, tp, pp);
    // 用 3 倍层数当「要切的层数」，好让 pp 除不尽时看到余数怎么摊
    let layers_demo = n_layer * 3;
    logln!(
        "  三个虚拟维度：层 {layers_demo}（pp 轴）、输出列 {hidden}（tp 轴）、样本 {total_b}（dp 轴）"
    );
    for rank in 0..cfg.world_size() {
        let plan = cfg.plan(rank, layers_demo, hidden, total_b);
        logln!("  {}", plan.summary(&cfg));
    }

    // 自检：参数在 (tp, pp) 网格上不重不漏、数据在 dp 轴上不重不漏。
    //
    // 「在哪个轴上」必须说清楚：columns 只由 tp 决定、layers 只由 pp 决定，所以查列时固定一个 pp
    // （否则每个 pp 都会把列重新覆盖一遍，正常也会数出 pp 次），查层时固定一个 tp。
    let mut col_cover = vec![0usize; hidden];
    for tp_i in 0..tp {
        let plan = cfg.plan(
            cfg.rank_of(distributed::RankCoord {
                dp: 0,
                tp: tp_i,
                pp: 0,
            }),
            layers_demo,
            hidden,
            total_b,
        );
        for &c in &plan.columns {
            col_cover[c] += 1;
        }
    }
    assert!(col_cover.iter().all(|&c| c == 1), "输出列在 tp 轴上不重不漏被破坏");

    let mut layer_cover = vec![0usize; layers_demo];
    for pp_i in 0..pp {
        let plan = cfg.plan(
            cfg.rank_of(distributed::RankCoord {
                dp: 0,
                tp: 0,
                pp: pp_i,
            }),
            layers_demo,
            hidden,
            total_b,
        );
        for k in plan.layers.0..plan.layers.1 {
            layer_cover[k] += 1;
        }
    }
    assert!(layer_cover.iter().all(|&c| c == 1), "层在 pp 轴上不重不漏被破坏");

    // dp 轴上是**复制**：同一个 (tp, pp) 上的 dp 个 rank 持有完全一样的参数分片，
    // 只有样本区间不同——这正是"参数不按 dp 切"的意思
    for tp_i in 0..tp {
        for pp_i in 0..pp {
            let plan = cfg.plan(
                cfg.rank_of(distributed::RankCoord {
                    dp: 0,
                    tp: tp_i,
                    pp: pp_i,
                }),
                layers_demo,
                hidden,
                total_b,
            );
            for dp_i in 1..dp {
                let other = cfg.plan(
                    cfg.rank_of(distributed::RankCoord {
                        dp: dp_i,
                        tp: tp_i,
                        pp: pp_i,
                    }),
                    layers_demo,
                    hidden,
                    total_b,
                );
                assert_eq!(other.columns, plan.columns, "dp 副本的参数分片必须完全相同");
                assert_eq!(other.layers, plan.layers);
                assert_ne!(other.batch, plan.batch, "dp 轴切的是数据：各副本的样本区间不能相同");
            }
        }
    }

    // 数据在 dp 轴上不重不漏（固定 tp / pp，把 dp 轴的样本区间拼起来应铺满 [0, total_b)）
    let mut batch_cover = vec![0usize; total_b];
    for dp_i in 0..dp {
        let plan = cfg.plan(
            cfg.rank_of(distributed::RankCoord {
                dp: dp_i,
                tp: 0,
                pp: 0,
            }),
            layers_demo,
            hidden,
            total_b,
        );
        for k in plan.batch.0..plan.batch.1 {
            batch_cover[k] += 1;
        }
    }
    assert!(batch_cover.iter().all(|&c| c == 1), "样本在 dp 轴上不重不漏被破坏");
    logln!(
        "  自检通过：参数在 (tp, pp) 网格上不重不漏，dp 轴上是**复制**（同一分片、不同数据）；\n  \
         数据在 dp 轴上不重不漏，在 (tp, pp) 内部是复制的。"
    );
    logln!(
        "  读法：rank 编号把 tp 放在最低位（Megatron 约定）——TP 队友的编号是连着的，可以放进\n  \
         同一个 NVLink 域。通信最密的正是 TP（每层两次 allreduce），pp 只传边界激活，\n  \
         dp 每步一次、可以走最慢的链路；三个轴的网络优先级完全不同，这就是「怎么摆卡」的全部内容。"
    );

    runlog::finish();
}

/// 两个等长缓冲的最大逐元素差（TP / PP 的对拍用）
fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(
        a.len(),
        b.len(),
        "对拍的两个缓冲长度不一致：{} vs {}",
        a.len(),
        b.len()
    );
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

/// Xavier 正态初始化（与 [`Linear::new`] 同一套口径）
fn xavier_data(rows: usize, cols: usize, rng: &mut Rng) -> Vec<f32> {
    let std = (2.0 / (rows + cols) as f32).sqrt();
    (0..rows * cols).map(|_| rng.randn() * std).collect()
}

/// 把 loss 序列压成一行（均匀取 `points` 个采样点）
fn loss_curve(losses: &[f32], points: usize) -> String {
    let n = losses.len();
    let k = points.min(n).max(1);
    (0..k)
        .map(|i| {
            let idx = if k == 1 { 0 } else { i * (n - 1) / (k - 1) };
            format!("{:.4}", losses[idx])
        })
        .collect::<Vec<_>>()
        .join(" → ")
}

// ==================== 第 36 课：RLHF 与对齐实验 ====================

/// 对齐实验（第 36 课）：奖励模型 / DPO / GRPO / PPO 四件套。
///
/// 全部跑在内置语料 + 字符分词器上（自包含、可复现），四节各自带**可对拍的自检**：
/// 奖励模型看没见过的长度上的排序准确率，DPO 看 chosen/rejected 的对数概率边界与
/// 参考模型是否真的冻结，GRPO 看组内优势的两条数学性质与裁剪分支的梯度，
/// PPO 看 KL(k3) 的两个"本该如此"与它把策略拴在原地的效果。
///
/// 四件套共用同一套小 Transformer 结构，只在初始化种子上分开：结构一样，差异才能全归给算法。
fn cmd_align(a: &AlignArgs) {
    let log_path = runlog::start("align");
    println!("运行日志：{log_path}");

    assert!(
        a.n_embd % 4 == 0,
        "n_embd 必须能被 n_head=4 整除（实际 {}）",
        a.n_embd
    );
    assert!(a.rm_steps >= 1, "奖励模型至少要训 1 步（实际 {}）", a.rm_steps);
    assert!(a.steps >= 1, "DPO 至少要训 1 步（实际 {}）", a.steps);
    assert!(a.beta > 0.0, "DPO 的 β 必须为正（实际 {}）", a.beta);
    assert!(a.clip_eps > 0.0, "裁剪范围 ε 必须为正（实际 {}）", a.clip_eps);
    assert!(a.kl_coef > 0.0, "KL 系数必须为正（实际 {}）", a.kl_coef);
    assert!(a.group_size >= 2, "组内相对优势至少要 2 条回答（实际 {}）", a.group_size);

    let tokenizer = Tokenizer::char(CORPUS);
    let vocab = tokenizer.vocab_size();
    assert!(vocab >= 4, "字符词表至少要有 4 个不同字符（实际 {vocab}）");
    let corpus_ids = tokenizer.encode(CORPUS);
    let t = a.block_size;

    let backbone = |s: u64| {
        let cfg = TransformerConfig {
            n_embd: a.n_embd,
            n_head: 4,
            n_layer: a.n_layer,
            block_size: t,
            ..TransformerConfig::tiny(vocab)
        };
        Transformer::new(cfg, &mut Rng::new(s))
    };
    let n_params: usize = backbone(a.seed).parameters().iter().map(|p| p.numel()).sum();

    runlog::fields(
        "本次运行参数",
        &[
            ("奖励模型步数 / 学习率", format!("{} / {}", a.rm_steps, a.rm_lr)),
            ("DPO 步数 / 学习率", format!("{} / {}", a.steps, a.lr)),
            ("DPO β", a.beta.to_string()),
            ("裁剪范围 ε", a.clip_eps.to_string()),
            ("PPO KL 系数", a.kl_coef.to_string()),
            ("GRPO 组大小", a.group_size.to_string()),
            ("批大小 / 上下文", format!("{} / {}", a.batch_size, t)),
            (
                "模型",
                format!("n_layer={} n_embd={} n_head=4（GELU FFN）", a.n_layer, a.n_embd),
            ),
            ("参数量", n_params.to_string()),
            ("分词器", format!("{}（词表 {vocab}）", tokenizer.kind())),
        ],
    );

    // ---------- 一、序列对数概率：对齐只算「模型要说的话」 ----------
    logln!("=== 一、序列对数概率与掩码：算 loss 的位置必须与推理时说的话一致 ===");
    let take = 24.min(corpus_ids.len() - 1);
    assert!(take >= 12, "内置语料太短，凑不出「提问 + 回答」（实际只有 {take} 个 token）");
    let mut with_bos: Vec<usize> = tokenizer.bos_id().into_iter().collect();
    with_bos.extend_from_slice(&corpus_ids[..take]);
    let prompt_len = 1 + 8; // BOS + 前 8 个 token 当作"提问"
    let full = align::MaskedSequence::full(with_bos.clone());
    let ans = align::MaskedSequence::answer_only(with_bos.clone(), prompt_len);
    let probe = backbone(a.seed);
    let full_lp = align::sequence_logprob_value(&probe, &full);
    let ans_lp = align::sequence_logprob_value(&probe, &ans);

    // 掩码口径自检：整条序列除首位外全参与；只算回答时恰好少掉 prompt 那一段
    assert_eq!(full.supervised(), full.len() - 1, "整条序列应除首位外全部参与");
    assert_eq!(ans.supervised(), ans.len() - prompt_len, "只有回答部分该参与");
    // 少算若干项 log π ≤ 0 的和，对数概率只会更"接近 0"（即更大）
    assert!(full_lp <= ans_lp, "整条序列的对数概率不可能比只算回答的更大");

    logln!(
        "  序列长度 {}（BOS + 语料前 {} 个 token），前 {} 个位置当作「提问」（含 BOS）。",
        full.len(),
        take,
        prompt_len
    );
    logln!("  整条都算：监督位 {} ｜ log π = {:.4}", full.supervised(), full_lp);
    logln!("  只算回答：监督位 {} ｜ log π = {:.4}", ans.supervised(), ans_lp);
    logln!(
        "  两者相差 {:.4}，正是提问那 {} 项的和。",
        ans_lp - full_lp,
        full.supervised() - ans.supervised()
    );
    logln!(
        "  读法：DPO / PPO 的目标是**序列对数概率之和**（不是平均）——每一步 log 概率都是独立\n  \
         贡献，取平均会让长短回答的尺度不一致，训练出来表现为「偏爱短回答」。\n  \
         而把提问也计进去，等于在奖励「复述用户的话」：模型很快学会照抄 prompt 就能拿分。"
    );

    // ---------- 二、奖励模型：成对比较怎么变成梯度 ----------
    logln!("=== 二、奖励模型：Bradley-Terry 成对损失 + 排序准确率 ===");
    // 偏好样本的构造口径：两条回答**长度完全相同**，唯一差别是最后一个 token——
    // 把「长的更好」这条捷径堵死，模型只能从内容里学；评测用训练时没出现过的长度。
    let pat = [corpus_ids[0], corpus_ids[1]];
    let good = corpus_ids[2];
    let bad = corpus_ids[3];
    let make_seq = |m: usize, last: usize| -> Vec<usize> {
        let mut v: Vec<usize> = (0..2 * m).map(|i| pat[i % 2]).collect();
        v.push(last);
        v
    };
    let pairs_at = |ms: &[usize]| -> Vec<(Vec<usize>, Vec<usize>)> {
        ms.iter()
            .map(|&m| (make_seq(m, good), make_seq(m, bad)))
            .collect()
    };
    let train_pairs = pairs_at(&[2, 3, 4]);
    let eval_pairs = pairs_at(&[5, 6, 7]);

    let reward = align::RewardModel::new(backbone(a.seed + 7), &mut Rng::new(a.seed + 8));
    let mean_margin = |m: &align::RewardModel, set: &[(Vec<usize>, Vec<usize>)]| -> f32 {
        set.iter()
            .map(|(c, r)| m.score_value(c) - m.score_value(r))
            .sum::<f32>()
            / set.len() as f32
    };
    let eval_scores = |m: &align::RewardModel| -> Vec<(f32, f32)> {
        eval_pairs.iter().map(|(c, r)| (m.score_value(c), m.score_value(r))).collect()
    };

    // Δ = 0 时 BT 损失取到最大值 ln 2：两条回答不分伯仲，正是最该被推开的时刻
    let tie_a = Tensor::param(vec![0.5], vec![1]);
    let tie_b = Tensor::param(vec![0.5], vec![1]);
    let tie_loss = align::bradley_terry_loss(&tie_a, &tie_b).item();
    assert!(
        (tie_loss - std::f32::consts::LN_2).abs() < 1e-6,
        "分数相同时 BT 损失应为 ln 2 = 0.6931，实际 {tie_loss}"
    );

    let acc_before = align::ranking_accuracy(&eval_scores(&reward));
    let margin_before = mean_margin(&reward, &train_pairs);

    let mut rm_opt = AdamW::new(a.rm_lr, reward.parameters(), 0.0);
    let mut rm_losses = Vec::with_capacity(a.rm_steps);
    for step in 0..a.rm_steps {
        let (c, r) = &train_pairs[step % train_pairs.len()];
        rm_opt.zero_grad();
        let loss = reward.pairwise_loss(c, r, false);
        rm_losses.push(loss.item());
        loss.backward();
        rm_opt.step();
    }

    let acc_after = align::ranking_accuracy(&eval_scores(&reward));
    let margin_after = mean_margin(&reward, &train_pairs);
    assert!(
        margin_after > margin_before,
        "训练后偏好边界应变大：{margin_before:.4} → {margin_after:.4}"
    );
    assert!(
        acc_after > 0.75,
        "排序准确率应明显高于随机 0.5：训练前 {acc_before:.3} → 训练后 {acc_after:.3}"
    );

    logln!("  每条回答的长度：训练用 {:?} 个 token，评测用 {:?}（评测长度没在训练里出现过）。",
        train_pairs.iter().map(|(c, _)| c.len()).collect::<Vec<_>>(),
        eval_pairs.iter().map(|(c, _)| c.len()).collect::<Vec<_>>());
    logln!("  Δ = 0 时损失 = ln 2 = {:.4}（最大值，训练把它推大）", tie_loss);
    logln!(
        "  {} 步成对训练（lr = {}）：平均偏好边界 {:.4} → {:.4}",
        a.rm_steps,
        a.rm_lr,
        margin_before,
        margin_after
    );
    logln!(
        "  排序准确率（chosen 分数严格高于 rejected 的比例）：训练前 {:.3} → 训练后 {:.3}",
        acc_before,
        acc_after
    );
    logln!("  训练损失：{}", loss_curve(&rm_losses, 6));
    logln!(
        "  读法：奖励模型的分数是**无界实数**，只有相对大小有意义——BT 只看两个分数之差，\n  \
         不存在标定问题。训练好之后它就成了裁判：PPO 与拒答采样都靠它打分，\n  \
         而 DPO 恰恰是**跳过**这一步、直接把偏好写进策略里。"
    );

    // ---------- 三、DPO：跳过奖励模型，直接在偏好对上优化策略 ----------
    logln!("=== 三、DPO：隐式奖励 + 参考模型全程冻结 ===");
    let reference = backbone(a.seed + 21);
    let policy = backbone(a.seed + 22);
    let mut prompt_ids: Vec<usize> = tokenizer.bos_id().into_iter().collect();
    prompt_ids.extend_from_slice(&corpus_ids[..8]);
    let answer = |m: usize, last: usize| -> align::MaskedSequence {
        let mut ids = prompt_ids.clone();
        ids.extend(make_seq(m, last));
        align::MaskedSequence::answer_only(ids, prompt_ids.len())
    };
    let dpo_pairs: Vec<align::PreferencePair> = [2usize, 3, 4]
        .iter()
        .map(|&m| align::PreferencePair::new(answer(m, good), answer(m, bad)))
        .collect();
    let flat: Vec<align::MaskedSequence> = dpo_pairs
        .iter()
        .flat_map(|p| [p.chosen.clone(), p.rejected.clone()])
        .collect();
    // 参考模型全程冻结，它的 logprob 一个数都不会变：提前算成 f32 常数，训练循环里只跑策略
    let ref_logprobs: Vec<(f32, f32)> = align::precompute_reference_logprobs(&reference, &flat)
        .chunks(2)
        .map(|c| (c[0], c[1]))
        .collect();

    let snapshot = |m: &Transformer| -> Vec<f32> { m.parameters().iter().flat_map(|p| p.data()).collect() };
    let ref_snapshot = snapshot(&reference);
    let policy_margin = |m: &Transformer| -> f32 {
        dpo_pairs
            .iter()
            .map(|p| {
                align::sequence_logprob_value(m, &p.chosen)
                    - align::sequence_logprob_value(m, &p.rejected)
            })
            .sum::<f32>()
    };
    let dpo_batch = |m: &Transformer| -> f32 {
        tensor::no_grad(|| align::dpo_batch_loss(m, &dpo_pairs, &ref_logprobs, a.beta, false).item())
    };

    // 策略与参考模型同一个模型时，隐式奖励之差为 0 ⇒ DPO 损失 = ln 2
    let tied_reference: Vec<(f32, f32)> = align::precompute_reference_logprobs(&policy, &flat)
        .chunks(2)
        .map(|c| (c[0], c[1]))
        .collect();
    let tied_loss = tensor::no_grad(|| {
        align::dpo_batch_loss(&policy, &dpo_pairs, &tied_reference, a.beta, false).item()
    });
    assert!(
        (tied_loss - std::f32::consts::LN_2).abs() < 1e-5,
        "策略与参考模型相同时 DPO 损失应为 ln 2，实际 {tied_loss}"
    );

    let margin_before = policy_margin(&policy);
    let loss_before = dpo_batch(&policy);
    let mut opt = AdamW::new(a.lr, policy.parameters(), 0.0);
    let mut losses = Vec::with_capacity(a.steps);
    for _ in 0..a.steps {
        opt.zero_grad();
        let loss = align::dpo_batch_loss(&policy, &dpo_pairs, &ref_logprobs, a.beta, false);
        losses.push(loss.item());
        loss.backward();
        opt.step();
    }
    let margin_after = policy_margin(&policy);
    let loss_after = dpo_batch(&policy);

    assert_eq!(snapshot(&reference), ref_snapshot, "参考模型必须全程冻结（一个参数都不能动）");
    assert!(
        margin_after > margin_before,
        "训练后 chosen/rejected 的对数概率边界应严格变大：{margin_before:.6} → {margin_after:.6}"
    );
    assert!(
        loss_after < loss_before,
        "训练后 DPO 损失应下降：{loss_before:.6} → {loss_after:.6}"
    );

    logln!("  {} 对偏好样本（同一 prompt 下好 / 坏回答，长度相同只差最后一个 token）", dpo_pairs.len());
    logln!(
        "  策略 = 参考模型时损失 = ln 2 = {:.4}（隐式奖励之差为 0，与奖励模型的 Δ = 0 同一个位置）",
        tied_loss
    );
    logln!(
        "  {} 步 DPO（lr = {} β = {}）：chosen/rejected 对数概率边界 {:.4} → {:.4}",
        a.steps,
        a.lr,
        a.beta,
        margin_before,
        margin_after
    );
    logln!("  DPO 损失：{:.4} → {:.4}｜{}", loss_before, loss_after, loss_curve(&losses, 6));
    logln!(
        "  读法：DPO 把「奖励」取成 β·(log π_θ − log π_ref)，于是 RLHF 的「奖励最大化 + KL 约束」\n  \
         有闭式最优解——不需要单独训一个奖励模型。β 是「允许偏离参考模型多远」的温度：\n  \
         越小越保守。参考模型的 logprob 是预先算好的常数，全程不进计算图（上面已逐参数核对）。"
    );

    // ---------- 四、GRPO：组内相对优势（不需要价值网络） ----------
    logln!("=== 四、GRPO：组内相对优势 + 裁剪代理目标 ===");
    let demo_rewards = vec![-1.0f32, 0.5, 2.0, 0.25];
    let demo_adv = align::group_advantages(&demo_rewards);
    let adv_mean = demo_adv.iter().sum::<f32>() / demo_adv.len() as f32;
    // 总体标准差（除以 n，不是 n−1）
    let adv_var = demo_adv.iter().map(|x| x * x).sum::<f32>() / demo_adv.len() as f32;
    assert!(adv_mean.abs() < 1e-6, "组内优势的均值必须为 0，实际 {adv_mean}");
    assert!(
        (adv_var.sqrt() - 1.0).abs() < 1e-6,
        "组内优势的总体标准差必须为 1，实际 {}",
        adv_var.sqrt()
    );
    // 组内分数全相同时 std = 0，除法会得到 NaN；NaN 流进梯度会把整份参数污染成 NaN
    let flat_adv = align::group_advantages(&[0.7f32; 4]);
    assert!(
        flat_adv.iter().all(|&x| x == 0.0),
        "组内分数全相同时优势必须是 0（短路），实际 {flat_adv:?}"
    );
    logln!("  构造一组奖励 {:?}：", demo_rewards);
    logln!(
        "    优势 = (r − mean) / std(population) = {:?}（均值 {:.2e}、标准差 {:.6}）",
        demo_adv
            .iter()
            .map(|x| format!("{x:.3}"))
            .collect::<Vec<_>>(),
        adv_mean,
        adv_var.sqrt()
    );
    logln!("  组内分数全相同（std = 0）时短路返回全 0：否则 NaN 会污染整份参数（不报错，只训出哑巴模型）");

    // 用训练好的奖励模型给一组真实回答打分 → 组内相对优势 → GRPO 损失
    let group: Vec<align::MaskedSequence> = (0..a.group_size)
        .map(|i| answer(2 + i % 3, if i % 2 == 0 { good } else { bad }))
        .collect();
    let group_rewards: Vec<f32> = group.iter().map(|s| reward.score_value(&s.ids)).collect();
    let group_adv = align::group_advantages(&group_rewards);
    let logp_old: Vec<f32> = group
        .iter()
        .map(|s| align::sequence_logprob_value(&policy, s))
        .collect();
    let logp_policy: Vec<Tensor> = group
        .iter()
        .map(|s| align::sequence_logprob(&policy, s, false))
        .collect();
    let grpo_value = align::grpo_loss(&logp_policy, &logp_old, &group_adv, a.clip_eps).item();
    assert!(grpo_value.is_finite(), "GRPO 损失不该是 NaN / inf：{grpo_value}");
    logln!(
        "  奖励模型给 {} 条回答打分（含好 / 坏两种末尾 token）：",
        a.group_size
    );
    for (i, (r, av)) in group_rewards.iter().zip(&group_adv).enumerate() {
        logln!("    第 {} 条 ｜ 奖励 {:>8.4} ｜ 优势 {:>8.4}", i, r, av);
    }
    logln!("  GRPO 损失（当前策略 = 采样时的策略，重要性比 ρ ≡ 1）= {:.6}", grpo_value);
    logln!(
        "    —— 恰好为 0：ρ ≡ 1 时代理目标就是组内优势本身，而组内优势的均值恒为 0。\n  \
         所以首次更新前 GRPO 的总梯度只来自「谁比组内平均好」，不来自「绝对分有多高」。"
    );

    // 裁剪分支的梯度：ρ 超出 1±ε 的部分被截断，梯度恰为 0
    let clip_old = vec![0.0f32, 0.0f32];
    let clip_adv = vec![1.0f32, 1.0f32];
    let lp_new = vec![
        Tensor::param(vec![(2.0f32).ln()], vec![1]), // ρ = 2.0 > 1 + ε ⇒ 被裁剪
        Tensor::param(vec![0.0f32], vec![1]),        // ρ = 1.0 ⇒ 未裁剪
    ];
    let clip_loss = align::grpo_loss(&lp_new, &clip_old, &clip_adv, a.clip_eps);
    clip_loss.backward();
    let g_clip = lp_new[0].grad()[0];
    let g_open = lp_new[1].grad()[0];
    // 被裁剪的样本：clip 在区间外是常数，梯度为 0；未裁剪的：dL/dlog π = −A·ρ/n = −1·1/2
    assert!(g_clip.abs() < 1e-6, "被裁剪样本的梯度必须为 0，实际 {g_clip}");
    assert!(
        (g_open + 0.5).abs() < 1e-5,
        "未裁剪样本的 dL/dlog π 应为 −A·ρ/n = −0.5，实际 {g_open}"
    );
    logln!(
        "  裁剪（ε = {}）：ρ = 2.0 与 ρ = 1.0 两条，损失 = {:.6}；",
        a.clip_eps,
        clip_loss.item()
    );
    logln!(
        "    被裁剪那条的 dL/dlog π = {:.2e}（clip 在区间外是常数，梯度恰为 0）",
        g_clip
    );
    logln!(
        "    未裁剪那条的 dL/dlog π = {:.4}（= −A·ρ/n）",
        g_open
    );
    logln!(
        "  读法：GRPO 省掉价值网络的关键就是组内平均分——同一 prompt 采一组回答，均值天然是基线，\n  \
         顺手把「这道题本身难不难」这个与策略无关的偏移减掉了。裁剪则是「一步别走太远」：\n  \
         超过 1±ε 的部分不再给梯度，避免几个高优势样本把策略一把带偏。"
    );

    // ---------- 五、PPO：裁剪 + 参考模型 KL 约束 ----------
    logln!("=== 五、PPO：KL(k3) 估计与「把策略拴在参考模型附近」 ===");
    let deltas = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0];
    let mut kl_max = 0.0f32;
    logln!("  delta = log π_ref − log π_θ 的正负两侧都扫一遍：");
    for &d in &deltas {
        let kl = align::kl_k3(d);
        assert!(kl >= 0.0, "k3 估计恒非负，实际 delta = {d} → {kl}");
        kl_max = kl_max.max(kl);
        // √(2KL)/|delta| 在 |delta| → 0 时趋于 1，说明小偏离下 KL ≈ delta²/2
        let ratio = if d == 0.0 {
            "—".to_string()
        } else {
            format!("{:.4}", (2.0 * kl).sqrt() / d.abs())
        };
        logln!("    delta = {:>5} ｜ KL = {:.6} ｜ √(2KL)/|delta| = {}", d, kl, ratio);
    }
    assert_eq!(align::kl_k3(0.0), 0.0, "π_θ = π_ref 时 KL 必须恰为 0");
    logln!(
        "  最大 KL = {:.4}。√(2KL)/|delta| 在 |delta| 变小时趋于 1，即小偏离下 KL ≈ delta²/2：",
        kl_max
    );
    logln!(
        "  一阶项为 0 ⇒ 参考模型所在处不会有「把它推走」的力（否则约束会把模型从参考点推开）；\n  \
         二阶系数 1/2 ⇒ 偏离越远罚得越狠。"
    );

    // kl_coef = 0 时 PPO 必须退化成 GRPO；加上 KL 后损失被抬高
    let ppo_old = vec![0.0f32, 0.0f32];
    let ppo_adv = vec![1.0f32, -1.0f32];
    let ppo_ref = vec![-1.0f32, -4.0f32];
    let mk = |v: Vec<f32>| -> Vec<Tensor> {
        v.into_iter().map(|x| Tensor::param(vec![x], vec![1])).collect()
    };
    let lp_for_grpo = mk(vec![-1.0, -2.0]);
    let lp_ppo0 = mk(vec![-1.0, -2.0]);
    let lp_ppo1 = mk(vec![-1.0, -2.0]);
    let grpo_ref = align::grpo_loss(&lp_for_grpo, &ppo_old, &ppo_adv, a.clip_eps).item();
    let ppo0 = align::ppo_loss(&lp_ppo0, &ppo_old, &ppo_adv, &ppo_ref, a.clip_eps, 0.0).item();
    let ppo1 = align::ppo_loss(&lp_ppo1, &ppo_old, &ppo_adv, &ppo_ref, a.clip_eps, a.kl_coef).item();
    assert!(
        (grpo_ref - ppo0).abs() < 1e-6,
        "kl_coef = 0 时 PPO 必须与 GRPO 完全一致：{grpo_ref} vs {ppo0}"
    );
    assert!(ppo1 > ppo0, "KL 惩罚必须把损失抬高：{ppo0} → {ppo1}");
    logln!("  kl_coef = 0 时 PPO 与 GRPO 逐位一致：{:.6} vs {:.6}", grpo_ref, ppo0);
    logln!(
        "  同一个代理目标加 KL 罚（kl_coef = {}）：{:.6} → {:.6}",
        a.kl_coef,
        ppo0,
        ppo1
    );
    logln!("  策略离参考模型越远，KL 罚得越狠（固定 logp_ref = −1.0，扫描当前策略的 log π）：");
    for &lp in &[-1.0f32, -2.0, -3.0, -5.0, -9.0] {
        let d = -1.0f32 - lp; // delta = log π_ref − log π_θ
        let one = mk(vec![lp]);
        let value = align::ppo_loss(&one, &[0.0], &[1.0], &[-1.0], a.clip_eps, a.kl_coef).item();
        logln!(
            "    log π = {:>5} ｜ delta = {:>5} ｜ KL = {:>9.4} ｜ PPO 损失 = {:.6}",
            lp,
            d,
            align::kl_k3(d),
            value
        );
    }
    logln!(
        "  读法：为什么不用最常见的 log π_θ − log π_ref——那个估计虽无偏，但方差可能极大，\n  \
         而且可以为负（KL 在数学上不可能为负，负值纯属估计噪声）。k3 无偏、低方差、恒非负，\n  \
         代价是多算一次 exp。KL 惩罚的存在理由：一路最大化奖励会把策略推到奖励模型没见过的\n  \
         区域，那里它的打分毫无意义（reward hacking）——KL 就是那根拴绳。"
    );

    runlog::finish();
}

// ==================== 第 37 课：检索增强生成实验 ====================

/// 检索增强生成实验（第 37 课）：分块 → 向量化 → 检索 → 提示组装，四步各带自检。
///
/// 检索质量的上限由**切块**决定（切错了后面再精巧也补不回来），所以第一节先把分块的
/// 三条不变量逐条钉住；第二、三节用字符级向量（TF-IDF / 特征哈希）跑检索与 MMR 重排，
/// 第四节看提示裁剪是否守住预算，第五节接上模型稠密向量做对照（含"为什么它必须先训练"）。
fn cmd_rag(a: &RagArgs) {
    let log_path = runlog::start("rag");
    println!("运行日志：{log_path}");

    assert!(
        a.chunk_size > a.overlap,
        "块大小 {} 必须大于重叠 {}（否则窗口不前进）",
        a.chunk_size,
        a.overlap
    );
    assert!(a.top_k >= 1, "至少要返回 1 条（实际 {}）", a.top_k);
    assert!(a.hash_dim >= 1, "哈希维度必须为正（实际 {}）", a.hash_dim);
    assert!(a.context_chars >= 1, "字符预算必须为正（实际 {}）", a.context_chars);
    assert!(
        (0.0..=1.0).contains(&a.mmr_lambda),
        "λ 必须落在 [0, 1]（实际 {}）",
        a.mmr_lambda
    );
    assert!(a.n_embd % 4 == 0, "n_embd 必须能被 n_head=4 整除（实际 {}）", a.n_embd);

    let opts = rag::ChunkOpts::new(a.chunk_size, a.overlap);
    let aligned = rag::ChunkOpts::aligned(a.chunk_size, a.overlap);
    let n_chars = CORPUS.chars().count();

    runlog::fields(
        "本次运行参数",
        &[
            ("切块（上限 / 重叠）", format!("{} / {} 字符", a.chunk_size, a.overlap)),
            ("返回条数 k", a.top_k.to_string()),
            (
                "MMR（λ / 池大小）",
                format!("{} / {}", a.mmr_lambda, a.mmr_pool),
            ),
            ("特征哈希维度", a.hash_dim.to_string()),
            ("提示资料预算", format!("{} 字符", a.context_chars)),
            ("查询", a.query.clone()),
            (
                "稠密向量小模型",
                format!("n_layer={} n_embd={} n_head=4", a.n_layer, a.n_embd),
            ),
        ],
    );

    // ---------- 一、分块：字符域切分 + 恰好 overlap 的重叠 ----------
    logln!("=== 一、分块：按字符切、相邻块重叠恰为 overlap、拼回来不丢字 ===");
    let mut chunks = rag::chunk_text(CORPUS, &opts);
    let snapped = rag::chunk_text(CORPUS, &aligned);
    assert!(!chunks.is_empty(), "非空文本应至少切出一块");

    // 与 `rag::tests::rebuild_and_check` 同一套不变量，这里独立再验一遍（交叉校验）
    let mut rebuilt = String::new();
    for (i, c) in chunks.iter().enumerate() {
        assert_eq!(
            c.text.chars().count(),
            c.end - c.start,
            "块 {}-{} 的字符数与区间不符（多字节字符被切开了）",
            c.start,
            c.end
        );
        if i + 1 < chunks.len() {
            assert_eq!(
                chunks[i + 1].start + a.overlap,
                c.end,
                "相邻块重叠必须恰为 {} 个字符",
                a.overlap
            );
            rebuilt.extend(c.text.chars().take(c.len() - a.overlap));
        } else {
            rebuilt.extend(c.text.chars());
        }
    }
    assert_eq!(rebuilt, CORPUS, "去掉与后块重叠的部分后拼接，必须恰好还原原文");

    // 对齐版：块尾尽量落在句子/段落结束符上（找不到断点时才保持原边界）
    let is_break = |c: char| {
        matches!(c, '。' | '！' | '？' | '；' | '…' | '.' | '!' | '?' | ';' | '\n')
    };
    let on_break = |cs: &[rag::Chunk]| -> usize {
        cs.iter()
            .take(cs.len().saturating_sub(1))
            .filter(|c| c.text.chars().last().map_or(false, is_break))
            .count()
    };
    for c in &snapped {
        assert!(
            c.len() <= a.chunk_size + a.chunk_size / 4,
            "对齐后块长最多多出 size/4：{} > {}",
            c.len(),
            a.chunk_size + a.chunk_size / 4
        );
    }
    let breaks_plain = on_break(&chunks);
    let breaks_snapped = on_break(&snapped);
    assert!(
        breaks_snapped >= breaks_plain,
        "对齐分块落在句子断点上的边界数不该变少：{breaks_snapped} vs {breaks_plain}"
    );

    logln!(
        "  原文 {} 字符 → 固定窗口 {} 块（块长 {:?}）、句子对齐 {} 块（块长 {:?}）",
        n_chars,
        chunks.len(),
        chunks.iter().map(|c| c.len()).collect::<Vec<_>>(),
        snapped.len(),
        snapped.iter().map(|c| c.len()).collect::<Vec<_>>()
    );
    logln!(
        "  边界落在句子/段落结束符上：固定窗口 {}/{}，句子对齐 {}/{}",
        breaks_plain,
        chunks.len().saturating_sub(1),
        breaks_snapped,
        snapped.len().saturating_sub(1)
    );
    logln!(
        "  前两块：字符 {}-{}「{}…」/ 字符 {}-{}「{}…」",
        chunks[0].start,
        chunks[0].end,
        rag::truncate_chars(chunks[0].text.trim(), 24),
        chunks[1].start,
        chunks[1].end,
        rag::truncate_chars(chunks[1].text.trim(), 24)
    );
    logln!(
        "  读法：三条不变量都验过了——①在**字符**边界上切（`&s[0..n]` 按字节切中文会 panic）；\n  \
         ②相邻块重叠恰为 {} 个字符，关键句正好落在边界时不会两头都缺；\n  \
         ③去掉重复部分后拼回来恰好等于原文，一个字不丢、一个字不重。\n  \
         对齐分块把边界推到句末，块读起来更完整，代价是块长最多多出 size/4。",
        a.overlap
    );

    // ---------- 二、向量化：TF-IDF vs 特征哈希 ----------
    logln!("=== 二、向量化：TF-IDF（有词表）vs 特征哈希（无词表） ===");
    // 块带上来源标记：组装提示时会写成「来源: corpus.txt」，回答可溯源
    for c in chunks.iter_mut() {
        c.source = "corpus.txt".to_string();
    }
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
    let tfidf = rag::TfIdf::fit(&texts);
    let df_of = |term: &str| {
        texts
            .iter()
            .filter(|c| rag::terms(c).iter().any(|x| x == term))
            .count()
    };
    logln!(
        "  词表（= 向量维度）{}，块数 {}。词项口径：单字 unigram + 相邻两字 bigram（中文没空格，不能按空格切词）",
        tfidf.vocab().len(),
        texts.len()
    );
    for term in ["e", "ke", "q"] {
        logln!(
            "    词项 {:>3} ｜ 出现在 {} / {} 个块里 ｜ IDF = {:.4}",
            term,
            df_of(term),
            texts.len(),
            tfidf.idf_of(term)
        );
    }
    logln!("  IDF 让「每个块都有」的字符权重趋近最低、「只此一处」的字符权重最高——这正是区分性。");

    // 未登录词：TF-IDF 里根本没有这些词项，向量全零；特征哈希没有词表，照样给出特征
    let oov = "熊猫";
    let oov_lex = rag::Embedder::embed(&tfidf, oov);
    let hasher = rag::HashingEmbedder::new(a.hash_dim);
    let oov_hash = rag::Embedder::embed(&hasher, oov);
    assert!(
        oov_lex.iter().all(|&x| x == 0.0),
        "词项全部未登录时，TF-IDF 向量应为全零"
    );
    assert!(
        oov_hash.iter().any(|&x| x != 0.0),
        "特征哈希没有词表，未登录词照样映射到若干桶上"
    );
    assert_eq!(
        rag::cosine_similarity(&oov_lex, &rag::Embedder::embed(&tfidf, &texts[0])),
        0.0,
        "全零向量的余弦相似度恒为 0（不是 NaN，否则排序会乱）"
    );
    logln!(
        "  未登录查询「{}」：TF-IDF 的 {} 维全为 0、所有相似度恒为 0，这一路彻底失效；",
        oov,
        oov_lex.len()
    );
    logln!(
        "  特征哈希（{} 维）非零维 {} 个——没有词表就不会有未登录词问题，代价是哈希冲突带来的噪声。",
        a.hash_dim,
        oov_hash.iter().filter(|&&x| x != 0.0).count()
    );
    logln!(
        "  读法：哈希向量**固定维度、内存不随语料增长、可流式增量更新**，适合流式场景；\n  \
         TF-IDF 的维度随语料词表增长，但每一维都有明确含义（某个字/字组），可解释性更好。"
    );

    // ---------- 三、检索：top-k 与 MMR 重排 ----------
    logln!("=== 三、检索：top-k 与 MMR 重排 ===");
    // 故意把语料接两遍：制造「同一条信息的多个副本」——MMR 的价值只有在重复语料上才看得出来
    let doc = format!("{CORPUS}\n\n{CORPUS}");
    let mut dup_chunks = rag::chunk_text(&doc, &opts);
    for c in dup_chunks.iter_mut() {
        c.source = "corpus.txt".to_string();
    }
    let retriever = rag::Retriever::build(Box::new(tfidf), dup_chunks);

    let hits = retriever.search(&a.query, a.top_k);
    let mmr = retriever.search_mmr(&a.query, a.top_k, a.mmr_lambda, a.mmr_pool);
    assert_eq!(hits.len(), a.top_k.min(retriever.len()), "top-k 应返回 min(k, 块数) 条");
    assert_eq!(mmr.len(), hits.len(), "MMR 与普通 top-k 返回条数应相同");
    for w in hits.windows(2) {
        assert!(w[0].score >= w[1].score - 1e-6, "top-k 必须按相关度降序（提示裁剪依赖这一点）");
    }
    assert!(hits[0].score > 0.0, "查询与语料有共同词项，最高相关度应大于 0");

    let plain_red = max_pairwise_similarity(&retriever, &hits);
    let mmr_red = max_pairwise_similarity(&retriever, &mmr);
    logln!(
        "  语料接两遍 → {} 块（原来 {} 块）、维度 {}；查询「{}」",
        retriever.len(),
        texts.len(),
        retriever.dim(),
        a.query
    );
    print_hits("普通 top-k", &hits, plain_red);
    print_hits(
        &format!("MMR 重排（λ = {}，池 {}）", a.mmr_lambda, a.mmr_pool),
        &mmr,
        mmr_red,
    );
    assert!(
        mmr_red <= plain_red + 1e-6,
        "MMR 的冗余度不该高于普通 top-k：{mmr_red:.4} vs {plain_red:.4}"
    );

    // λ = 1 时 MMR 退化成普通 top-k：选出的块应逐条一致
    let degenerate = retriever.search_mmr(&a.query, a.top_k, 1.0, a.mmr_pool);
    assert_eq!(
        degenerate.iter().map(|h| h.index).collect::<Vec<_>>(),
        hits.iter().map(|h| h.index).collect::<Vec<_>>(),
        "λ = 1 时 MMR 必须与普通 top-k 选出同一组块"
    );
    logln!("  λ = 1 时选出的块与普通 top-k 完全一致（逐条核对过）；λ 越小越看重多样性。");

    let oov_hits = retriever.search(oov, a.top_k);
    assert!(
        oov_hits.iter().all(|h| h.score == 0.0),
        "查询词项全部未登录时所有相关度都该是 0（而不是 NaN）"
    );
    logln!(
        "  未登录查询「{}」：{} 条结果相关度全为 0——排序退化成「按块下标」，说明这一路对\n  \
         语料里没有的字毫无办法（换稠密向量或特征哈希才有输出）。",
        oov,
        oov_hits.len()
    );
    logln!(
        "  读法：纯按相关度取 top-k 的问题在于语料里常有整段重复——k 个名额全给了同一条信息的\n  \
         副本，互补的信息反而挤不进来。MMR 每选一条就按「与已选的最大相似度」扣分，\n  \
         用 λ 调「要相关」还是「要多样」。"
    );

    // ---------- 四、提示组装：预算裁剪 + 来源标注 ----------
    logln!("=== 四、提示组装：按预算裁剪 + 来源标注 + 「只依据资料回答」 ===");
    let prompt_opts = rag::PromptOpts {
        max_context_chars: a.context_chars,
        ..Default::default()
    };
    let prompt = rag::build_rag_prompt(&a.query, &mmr, &prompt_opts);
    assert!(
        prompt.used_chars <= a.context_chars,
        "资料部分不能超预算：{} > {}",
        prompt.used_chars,
        a.context_chars
    );
    assert!(prompt.used_chunks >= 1, "非空检索结果至少该收下 1 条");
    assert!(
        prompt.text.contains("corpus.txt"),
        "块带了来源标记，提示里就应写出来源（回答可溯源）"
    );

    // 三个预算档位：充足 / 默认 / 连第一条都装不下
    let roomy = rag::PromptOpts {
        max_context_chars: 100_000,
        ..Default::default()
    };
    let roomy_prompt = rag::build_rag_prompt(&a.query, &mmr, &roomy);
    assert_eq!(
        roomy_prompt.used_chunks,
        mmr.len(),
        "预算充足时所有检索结果都该进来"
    );
    let tight_budget = 40.min(a.context_chars);
    let tight = rag::PromptOpts {
        max_context_chars: tight_budget,
        ..Default::default()
    };
    let tight_prompt = rag::build_rag_prompt(&a.query, &mmr, &tight);
    assert_eq!(
        tight_prompt.used_chunks, 1,
        "预算极小时仍应保住第一条（按字符截断收下，否则检索白做）"
    );
    assert!(tight_prompt.used_chars <= tight_budget, "截断后仍不能超预算");

    logln!(
        "  预算充足（10 万字符）→ 收下 {} / {} 条、用掉 {} 字符",
        roomy_prompt.used_chunks,
        mmr.len(),
        roomy_prompt.used_chars
    );
    logln!(
        "  本次预算（{} 字符）→ 收下 {} / {} 条、用掉 {} 字符",
        a.context_chars,
        prompt.used_chunks,
        mmr.len(),
        prompt.used_chars
    );
    logln!(
        "  预算压到 {} 字符（连第一条都装不下）→ 仍收下 {} 条、用掉 {} 字符（第一条按字符截断）",
        tight_budget,
        tight_prompt.used_chunks,
        tight_prompt.used_chars
    );
    logln!("  组装出的提示（截断显示）：");
    for line in prompt.text.lines() {
        logln!("    ｜ {}", rag::truncate_chars(line, 96));
    }
    logln!(
        "  读法：超预算时按分数从高到低放，放不下就**跳过并继续试后面更短的块**——一个大块装不下，\n  \
         不代表后面的小块也不该进。唯独第一条如果本身就超预算，就截断收下：\n  \
         否则预算小于最小块时会一条都进不去，检索白做、提示里只剩「我无法从提供的资料中找到答案」。"
    );

    // ---------- 五、模型稠密向量：唯一"懂语义"的一路 ----------
    logln!("=== 五、模型稠密向量：语义检索与它的两个代价 ===");
    let tokenizer = Tokenizer::char(CORPUS);
    let vocab = tokenizer.vocab_size();
    let t = (a.chunk_size + 8).max(64);
    let model = Transformer::new(
        TransformerConfig {
            n_embd: a.n_embd,
            n_head: 4,
            n_layer: a.n_layer,
            block_size: t,
            ..TransformerConfig::tiny(vocab)
        },
        &mut Rng::new(a.seed),
    );

    let t0 = std::time::Instant::now();
    let lex = rag::Retriever::build(Box::new(rag::TfIdf::fit(&texts)), chunks.clone());
    let lex_time = t0.elapsed();
    let t1 = std::time::Instant::now();
    let dense = rag::Retriever::build(
        Box::new(rag::ModelEmbedder::new(std::sync::Arc::new(model), tokenizer)),
        chunks.clone(),
    );
    let dense_time = t1.elapsed();

    assert_eq!(dense.dim(), a.n_embd, "稠密向量维度应等于 n_embd");
    let v1 = dense.embed_query(&a.query);
    let v2 = dense.embed_query(&a.query);
    assert!(
        (rag::l2_norm(&v1) - 1.0).abs() < 1e-5,
        "稠密向量必须已 L2 归一化，实际 ||v|| = {}",
        rag::l2_norm(&v1)
    );
    assert_eq!(v1, v2, "同一文本两次嵌入必须完全一致（推理无随机性）");

    let dense_hits = dense.search(&a.query, a.top_k);
    let (lex_lo, lex_hi) = score_range(&lex, &a.query);
    let (dense_lo, dense_hi) = score_range(&dense, &a.query);
    logln!(
        "  维度：字面（TF-IDF）= 词表大小 {} ｜ 稠密 = n_embd {}",
        lex.dim(),
        dense.dim()
    );
    logln!(
        "  建索引耗时：字面 {:?} → 稠密 {:?}（慢 {:.0} 倍：稠密要给每个块跑一次前向）",
        lex_time,
        dense_time,
        dense_time.as_secs_f64() / lex_time.as_secs_f64().max(1e-9)
    );
    logln!(
        "  同一查询的全量相关度区间：字面 [{:.4}, {:.4}]（跨度 {:.4}）｜ 稠密 [{:.4}, {:.4}]（跨度 {:.4}）",
        lex_lo,
        lex_hi,
        lex_hi - lex_lo,
        dense_lo,
        dense_hi,
        dense_hi - dense_lo
    );
    print_hits(
        "稠密向量 top-k（随机初始化的模型）",
        &dense_hits,
        max_pairwise_similarity(&dense, &dense_hits),
    );
    logln!(
        "  读法：稠密向量是三种里唯一「懂语义」的——问「怎么退款」也能检索到写着「申请退货流程」的段落，\n  \
         而 TF-IDF 与特征哈希只看字面重合。代价有两个：①每个块都要过一次前向（上面耗时已对照）；\n  \
         ②它依赖权重质量——这里用的是**随机初始化**的模型，相关度跨度比字面检索窄、且选出的几条\n  \
         彼此高度相似，几乎谈不上区分度。正式用法是接 `train` 出来的 checkpoint。"
    );

    runlog::finish();
}

/// 打印一组检索结果：块下标、相关度、字符区间、片段。
fn print_hits(title: &str, hits: &[rag::ScoredChunk], redundancy: f32) {
    logln!("  {}：最大两两相似度（冗余度）= {:.4}", title, redundancy);
    for (i, h) in hits.iter().enumerate() {
        logln!(
            "    [{}] 块 #{:<3} ｜ score {:.4} ｜ 字符 {}-{} ｜ {}",
            i + 1,
            h.index,
            h.score,
            h.chunk.start,
            h.chunk.end,
            rag::truncate_chars(&h.chunk.text.trim().replace('\n', " "), 36)
        );
    }
}

/// 一组检索结果里**两两相似度的最大值**（少于 2 条时为 0）——MMR 要压的就是它
fn max_pairwise_similarity(retriever: &rag::Retriever, hits: &[rag::ScoredChunk]) -> f32 {
    let vecs: Vec<Vec<f32>> = hits
        .iter()
        .map(|h| retriever.embed_query(&h.chunk.text))
        .collect();
    let mut worst = 0.0f32;
    for i in 0..vecs.len() {
        for j in i + 1..vecs.len() {
            worst = worst.max(rag::cosine_similarity(&vecs[i], &vecs[j]));
        }
    }
    worst
}

/// 全量相关度的区间 `(min, max)`：用来对照"这一路检索到底有没有区分度"
fn score_range(retriever: &rag::Retriever, query: &str) -> (f32, f32) {
    let all = retriever.search(query, retriever.len());
    let lo = all.iter().map(|h| h.score).fold(f32::MAX, f32::min);
    let hi = all.iter().map(|h| h.score).fold(f32::MIN, f32::max);
    (lo, hi)
}

// ==================== 推测解码实验（speculative 子命令） ====================

/// 某个 token 在 `(ids, probs)` 给出的分布下的概率（不在支撑集里就是 0）。
/// 与 `speculative` 模块内部的口径一致，分布检验要拿经验频率和它比。
fn token_prob(ids: &[usize], probs: &[f32], token: usize) -> f32 {
    ids.iter()
        .position(|&i| i == token)
        .map_or(0.0, |i| probs[i])
}

/// 与任何模型无关的固定草稿分布：`Drafter` 对 q 没有任何要求（q 再差只影响速度，
/// 不影响正确性），所以这里干脆不接模型——分布检验要做上千次，草稿侧的开销必须是 0。
struct FixedDrafter {
    ids: Vec<usize>,
    probs: Vec<f32>,
}

impl speculative::Drafter for FixedDrafter {
    fn propose(
        &mut self,
        _seq: &[usize],
        gamma: usize,
        _opts: &SampleOpts,
        _vocab: Option<&[Vec<u8>]>,
        rng: &mut Rng,
    ) -> Vec<speculative::Proposal> {
        (0..gamma)
            .map(|_| speculative::Proposal {
                // 候选必须真的按 q 抽：接受/拒绝能抵消成 p，靠的就是"x ~ q"这一条
                token: sample_from_probs(&self.ids, &self.probs, rng),
                ids: self.ids.clone(),
                probs: self.probs.clone(),
            })
            .collect()
    }

    fn commit(&mut self, _seq: &[usize]) {}

    fn reset(&mut self) {}
}

/// 推测解码（第 34 课）与多 Token 预测（第 35 课）的可运行实验：
/// 无损性核对 → 接受率与加速比 → 缓存账目 → 首 token 分布等价 → MTP 头当草稿。
fn cmd_speculative(s: &SpecArgs) {
    let log_path = runlog::start("speculative");
    println!("运行日志：{log_path}");

    assert!(
        s.n_embd % 4 == 0,
        "n_embd 必须能被 n_head=4 整除（实际 {}）",
        s.n_embd
    );
    assert!(s.gamma >= 1, "草稿长度 γ 至少为 1（实际 {}）", s.gamma);
    assert!(s.mtp_heads >= 1, "MTP 至少要有 1 个头（实际 {}）", s.mtp_heads);
    assert!(s.mtp_steps >= 1, "MTP 至少要训 1 步（实际 {}）", s.mtp_steps);
    assert!(s.max_new >= 1, "max_new 至少为 1（实际 {}）", s.max_new);
    assert!(s.trials >= 200, "分布检验至少 200 次试验（实际 {}）", s.trials);
    // 一轮前向要一次喂进 γ+1 个 token（序列最后一个 + γ 个草稿），必须塞得进窗口
    assert!(
        s.gamma + 1 <= s.block_size,
        "一轮前向要喂 γ+1 = {} 个 token，block_size 至少要 {}（实际 {}）",
        s.gamma + 1,
        s.gamma + 1,
        s.block_size
    );

    // 去掉特殊 token（BOS/EOS）：采样口径的实验里 EOS 只会添乱——一旦被采到就提前收工，
    // 几组对照生成的长度都不一样、统计没法比。去掉后 bos_id/eos_id 都是 None，生成严格跑满 max_new。
    let tokenizer = Tokenizer::char(CORPUS).without_specials();
    let vocab = tokenizer.vocab_size();
    // char 分词器的每个 token 都是完整字符，没有"半个汉字"的问题 → vocab_bytes 为 None
    let vocab_bytes: Option<Vec<Vec<u8>>> = tokenizer.vocab_bytes().map(|v| v.to_vec());

    let build = |seed: u64| {
        Transformer::new(
            TransformerConfig {
                n_embd: s.n_embd,
                n_head: 4,
                n_layer: s.n_layer,
                block_size: s.block_size,
                ..TransformerConfig::tiny(vocab)
            },
            &mut Rng::new(seed),
        )
    };
    let target = build(s.seed);
    let draft = build(s.seed + 1);
    let n_params: usize = target.parameters().iter().map(|p| p.numel()).sum();
    let kv = KvOpts::on(0, None);

    // 贪心：top-k=1 + 极低温度 → p、q 都退化成单点分布，输出有唯一答案，可以逐位对照
    let greedy = SampleOpts {
        temperature: 0.05,
        top_k: 1,
        top_p: 1.0,
        repetition_penalty: 1.0,
        repetition_window: 0,
        stop: &[],
    };
    // 采样：不截断（top-k=0、top-p=1），p 与 q 都保留完整支撑集，接受/残差两条路都能走到
    let sampled = SampleOpts {
        temperature: s.temperature,
        top_k: 0,
        top_p: 1.0,
        repetition_penalty: 1.0,
        repetition_window: 0,
        stop: &[],
    };
    // 分布检验专用的"尖一点"的采样：温度太低会退化成单点（检验没意义），太高又接近均匀
    // （分布在几百个字符上摊平，2000 次试验的统计噪声就会盖过任何真实偏差）
    let sharp = SampleOpts {
        temperature: 0.2,
        ..sampled
    };

    runlog::fields(
        "本次运行参数",
        &[
            ("草稿长度 γ", s.gamma.to_string()),
            ("每次生成 token 上限", s.max_new.to_string()),
            ("MTP 头数 / 训练步数", format!("{} / {}", s.mtp_heads, s.mtp_steps)),
            ("分布检验试验次数", s.trials.to_string()),
            ("采样温度", s.temperature.to_string()),
            ("批大小 / 上下文", format!("{} / {}", s.batch_size, s.block_size)),
            (
                "模型",
                format!("n_layer={} n_embd={} n_head=4（GELU FFN）", s.n_layer, s.n_embd),
            ),
            ("参数量", n_params.to_string()),
            ("分词器", format!("{}（词表 {vocab}）", tokenizer.kind())),
            ("提示词", s.prompt.clone()),
        ],
    );

    // 统计口径的一次性打印（后面几节反复用）
    let report = |label: &str, st: &speculative::SpecStats| {
        logln!(
            "  {}：{} 轮 ｜ 提出 {} ｜ 采纳 {} ｜ 拒绝修正 {} ｜ 白拿 {}",
            label,
            st.rounds,
            st.drafted,
            st.accepted,
            st.corrected,
            st.bonus
        );
        logln!(
            "    目标前向 {} 次 ｜ 共产出 {} token ｜ 接受率 {:.4} ｜ 每次前向 {:.3} 个 token",
            st.target_forwards,
            st.emitted,
            st.acceptance_rate(),
            st.tokens_per_forward()
        );
    };

    // ---------- 一、无损性：草稿再差也不改变输出 ----------
    logln!("=== 一、无损性：与目标模型的逐 token 贪心解码逐位对照 ===");
    let mut draft_model = speculative::ModelDrafter::new(&draft, kv);
    let (spec_text, spec_stats) = speculative::speculative_generate(
        &target,
        &tokenizer,
        &s.prompt,
        s.max_new,
        s.gamma,
        &greedy,
        kv,
        &mut draft_model,
        &mut Rng::new(s.seed),
    );
    let plain_text = generate(
        &target,
        &tokenizer,
        &s.prompt,
        s.max_new,
        &greedy,
        kv,
        &mut Rng::new(s.seed),
    );

    assert_eq!(
        spec_text, plain_text,
        "推测解码必须与目标模型的逐 token 解码逐位一致"
    );
    assert_eq!(
        spec_stats.target_forwards, spec_stats.rounds,
        "每轮只该有一次目标前向"
    );
    assert_eq!(
        spec_stats.emitted,
        spec_stats.accepted + spec_stats.corrected + spec_stats.bonus,
        "账目：产出的 token = 采纳数 + 拒绝修正 + 白拿"
    );
    assert!(
        spec_stats.emitted >= s.max_new,
        "至少要产出 {} 个 token（实际 {}）",
        s.max_new,
        spec_stats.emitted
    );

    logln!("  提示词「{}」，目标模型生成 {} 个 token。", s.prompt, s.max_new);
    logln!("  逐 token 贪心解码：{}", rag::truncate_chars(&plain_text, 44));
    logln!("  推测解码（γ = {}）：    {}", s.gamma, rag::truncate_chars(&spec_text, 44));
    report("本次统计", &spec_stats);
    logln!(
        "  读法：上面两行文本**逐位相同**。本次 {} 个候选里只有 {} 个被采纳（接受率 {:.4}），输出\n  \
         基本靠残差修正一步步顶出来，结果依然逐位正确——这就是『无损』的含义：草稿只影响速度，\n  \
         不影响质量。接受率为什么这么低？贪心采样把 p、q 都变成了单点分布，两个模型的 argmax\n  \
         只要不一样，草稿 token 在 p 下的概率就是 0、接受概率 min(1, p/q) 也随之归零；上面那几个\n  \
         被采纳的候选，就是两个模型偶尔撞上同一个 argmax 的情况。",
        spec_stats.drafted,
        spec_stats.accepted,
        spec_stats.acceptance_rate()
    );

    // ---------- 二、接受率与加速比：草稿质量决定收益 ----------
    logln!("=== 二、接受率与加速比：草稿越像目标，省下的目标前向越多 ===");
    // (a) 逐 token 采样的基线：每个 token 一次前向
    let t0 = std::time::Instant::now();
    let _ = generate(
        &target,
        &tokenizer,
        &s.prompt,
        s.max_new,
        &sampled,
        kv,
        &mut Rng::new(s.seed + 2),
    );
    let plain_time = t0.elapsed();
    // (b) 草稿 = 目标模型自身：q ≡ p，理论上每个候选都该被接受
    let mut self_draft = speculative::ModelDrafter::new(&target, kv);
    let t1 = std::time::Instant::now();
    let (_, self_stats) = speculative::speculative_generate(
        &target,
        &tokenizer,
        &s.prompt,
        s.max_new,
        s.gamma,
        &sampled,
        kv,
        &mut self_draft,
        &mut Rng::new(s.seed + 2),
    );
    let self_time = t1.elapsed();
    // (c) 草稿 = 另起种子的无关模型：分布几乎不重合
    let mut other_draft = speculative::ModelDrafter::new(&draft, kv);
    let t2 = std::time::Instant::now();
    let (_, other_stats) = speculative::speculative_generate(
        &target,
        &tokenizer,
        &s.prompt,
        s.max_new,
        s.gamma,
        &sampled,
        kv,
        &mut other_draft,
        &mut Rng::new(s.seed + 2),
    );
    let other_time = t2.elapsed();

    assert!(
        self_stats.acceptance_rate() > 0.99,
        "草稿就是目标模型时 q ≡ p，接受率应恒为 1（实际 {:.4}）",
        self_stats.acceptance_rate()
    );
    assert!(
        self_stats.tokens_per_forward() > 1.0,
        "自草稿每次前向该产出多于 1 个 token（实际 {:.3}）",
        self_stats.tokens_per_forward()
    );

    report("草稿 = 目标模型自身（q ≡ p）", &self_stats);
    report("草稿 = 另起种子的无关模型", &other_stats);
    logln!(
        "  墙钟：逐 token 采样 {:?} ｜ 自草稿推测 {:?} ｜ 无关草稿推测 {:?}",
        plain_time,
        self_time,
        other_time
    );

    // γ 扫描：草稿固定为目标模型自身，把"接受率"这个变量冻住，只看 γ 怎么决定收益
    logln!("  γ 的取舍（草稿 = 目标模型自身，q ≡ p，接受率恒为 1）：");
    let mut prev: Option<usize> = None;
    for g in [1usize, 2, 4, 8] {
        if g + 1 > s.block_size {
            continue; // 一轮要喂 γ+1 个 token，塞不进窗口的跳过
        }
        let mut d = speculative::ModelDrafter::new(&target, kv);
        let tg = std::time::Instant::now();
        let (_, st) = speculative::speculative_generate(
            &target,
            &tokenizer,
            &s.prompt,
            s.max_new,
            g,
            &sampled,
            kv,
            &mut d,
            &mut Rng::new(s.seed + 3),
        );
        let dt = tg.elapsed();
        assert!(
            st.tokens_per_forward() > 1.0,
            "γ = {g} 时每次前向应产出多于 1 个 token（实际 {:.3}）",
            st.tokens_per_forward()
        );
        assert!(
            st.target_forwards <= s.max_new,
            "推测解码的目标前向不该多于逐 token 基线"
        );
        if let Some(p) = prev {
            assert!(
                st.target_forwards <= p,
                "γ 变大时目标前向次数不该增加（{p} → {}）",
                st.target_forwards
            );
        }
        prev = Some(st.target_forwards);
        logln!(
            "    γ = {} → 目标前向 {:>3} 次（逐 token 基线 {} 次）｜ 每次前向 {:.2} 个 token ｜ 耗时 {:?}",
            g,
            st.target_forwards,
            s.max_new,
            st.tokens_per_forward(),
            dt
        );
    }

    logln!(
        "  读法：接受率完全由『草稿分布与目标分布的重合度』决定。自草稿（q ≡ p）时每个候选都被\n  \
         接受，接受率恒为 1，一轮稳定产出 γ+1 个 token——γ 扫描里目标前向次数就按这个规律往下降。\n  \
         而上面那个无关草稿的接受率居然也有 {:.4}：这是**随机初始化**造成的假象，模型 logits 幅度\n  \
         很小、softmax 接近均匀（每个字符约 1/{}），于是任意两个模型的 p、q 都长得差不多，比值\n  \
         p/q ≈ 1。它提醒我们：接受率不是『草稿有多好』的绝对度量，只有在 p 本身有区分度时才有意义\n  \
         （真训过的模型上才会看到 0.6~0.8 这种数字）。\n  \
         墙钟那行还说明另一件事：这里草稿与目标同尺寸，每轮它自己就要跑 γ 次前向，所以 γ 越大单轮\n  \
         越贵，墙钟时间的变化远不如目标前向次数那么漂亮。真实部署里草稿要比目标小一两个数量级\n  \
         （或干脆用第五节那种 MTP 头），省下的目标前向才换算得出真实加速。",
        other_stats.acceptance_rate(),
        vocab
    );

    // ---------- 三、缓存账目：推测解码最容易出错的地方 ----------
    logln!("=== 三、缓存账目：喂进缓存的 token 数恒等于「序列长度 − 1」 ===");
    let prompt_ids = tokenizer.encode(&s.prompt);
    assert!(!prompt_ids.is_empty(), "提示词至少要能编码出 1 个 token");
    let mut ids = prompt_ids.clone();
    let mut drafter = speculative::ModelDrafter::new(&draft, kv);
    let mut dec = speculative::SpecDecoder::new(
        &target,
        kv,
        s.gamma,
        sampled,
        vocab_bytes.as_deref(),
    );
    let mut rng = Rng::new(s.seed + 5);
    let mut lens: Vec<usize> = Vec::new();
    for round in 0..6 {
        let emitted = dec.step(&ids, &mut drafter, &mut rng);
        assert!(!emitted.is_empty(), "第 {round} 轮至少要产出 1 个 token");
        assert!(
            emitted.len() <= dec.gamma() + 1,
            "第 {round} 轮最多产出 γ+1 = {} 个 token（实际 {}）",
            dec.gamma() + 1,
            emitted.len()
        );
        ids.extend_from_slice(&emitted);
        assert_eq!(
            dec.cache_fed(),
            ids.len() - 1,
            "第 {round} 轮后缓存应恰好落后序列一格"
        );
        lens.push(emitted.len());
    }
    assert_eq!(
        dec.stats().target_forwards,
        dec.stats().rounds,
        "每轮只该有一次目标前向"
    );
    logln!("  提示词编码 {} 个 token，连跑 6 轮（γ = {}）。", prompt_ids.len(), dec.gamma());
    logln!("  每轮产出的 token 数：{:?}（γ + 1 = {} 是上限；被拒的轮次会短一些，靠 rollback 对齐缓存）", lens, dec.gamma() + 1);
    logln!(
        "  序列长度：{} → {} ｜ 缓存已喂入位置：{}",
        prompt_ids.len(),
        ids.len(),
        dec.cache_fed()
    );
    logln!(
        "  读法：一轮要一次喂进 γ+1 个 token（序列最后一个 + γ 个草稿），因此缓存必须**恰好**\n  \
         落后序列一格：多一位，注意力就会看到一批从未存在过的 token；少一位，位置号整体错开、\n  \
         RoPE 的相对距离全错。被拒的轮次里缓存里还留着多余的草稿，靠 rollback 掉回『新序列长度 − 1』，\n  \
         上面的断言逐轮核对的就是这条不变量。"
    );

    // ---------- 四、采样路径的硬指标：输出分布等于目标分布 ----------
    logln!("=== 四、采样路径的硬指标：首 token 的经验分布必须等于目标分布 ===");
    let seq = prompt_ids.clone();
    let (p_ids, p_probs) = {
        let mut stream = speculative::TargetStream::new(&target, kv);
        let rows = stream.feed(&seq);
        assert_eq!(rows.len(), seq.len() * vocab, "前向应返回每行 logits");
        probs_from_logits(&rows[(seq.len() - 1) * vocab..], &sharp, &[])
    };

    let t3 = std::time::Instant::now();
    let mut counts = vec![0usize; vocab];
    let mut accepted = 0usize;
    let mut corrected = 0usize;
    let mut rng = Rng::new(s.seed + 11);
    // 草稿直接给一组固定分布（支撑集只有 3 个 token，概率 0.5 / 0.3 / 0.2）：
    // 它与 p 差得很远，正好把"接受"与"拒绝后残差修正"两条路都逼出来
    let mut fixed = FixedDrafter {
        ids: vec![0, 1, 2],
        probs: vec![0.5, 0.3, 0.2],
    };
    for _ in 0..s.trials {
        let mut one = speculative::SpecDecoder::new(
            &target,
            kv,
            s.gamma,
            sharp,
            vocab_bytes.as_deref(),
        );
        let emitted = one.step(&seq, &mut fixed, &mut rng);
        counts[emitted[0]] += 1;
        accepted += one.stats().accepted;
        corrected += one.stats().corrected;
    }
    let trial_time = t3.elapsed();
    let tv: f32 = 0.5
        * (0..vocab)
            .map(|i| (counts[i] as f32 / s.trials as f32 - token_prob(&p_ids, &p_probs, i)).abs())
            .sum::<f32>();

    assert!(
        accepted > 0 && corrected > 0,
        "接受路径与残差修正路径都该被覆盖（采纳 {accepted} ｜ 修正 {corrected}）"
    );

    let mut top: Vec<(usize, f32)> = p_ids
        .iter()
        .copied()
        .zip(p_probs.iter().copied())
        .collect();
    top.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    top.truncate(3);
    logln!(
        "  温度 {:.2}（调低是为了让 p 有区分度），{} 次独立试验，耗时 {:?}。",
        sharp.temperature,
        s.trials,
        trial_time
    );
    logln!(
        "  草稿 = 一组与模型无关的固定分布（3 个 token，0.5 / 0.3 / 0.2）：被验证的位置 {} 个，\n  \
         其中采纳 {} 个、被拒后靠残差修正 {} 个。",
        accepted + corrected,
        accepted,
        corrected
    );
    let n = s.trials as f32;
    for (id, p) in &top {
        // 经验频率的标准差 σ = √(p(1−p)/n)：偏差落在 4σ 内就等于"统计噪声范围内一致"
        let sigma = (p * (1.0 - p) / n).sqrt();
        let dev = (counts[*id] as f32 / n - p).abs();
        logln!(
            "    字符 '{}'：目标 p = {:.4} ｜ 实测频率 = {:.4}（{} / {}）｜ 偏差 {:.4}，4σ = {:.4}",
            tokenizer.decode(&[*id]),
            p,
            counts[*id] as f32 / n,
            counts[*id],
            s.trials,
            dev,
            4.0 * sigma
        );
        assert!(
            dev <= 4.0 * sigma + 1e-3,
            "字符 '{}' 的实测频率偏离目标概率超过 4σ（{:.4} > {:.4}）",
            tokenizer.decode(&[*id]),
            dev,
            4.0 * sigma
        );
    }
    logln!(
        "  总变差距离（经验分布 vs 目标分布）= {:.4}（含统计噪声，只作参考；判据是上面的 4σ）",
        tv
    );
    logln!(
        "  读法：这是第 34 课的**硬指标**。贪心模式下『输出一致』只能说明两者都取了 argmax，\n  \
         而采样模式下推测解码的输出必须严格服从 p——接受路径贡献 min(p, q)，残差路径把差出来的\n  \
         max(0, p − q) 原样补回，两条路加起来正好是 p。上表每个字符的实测频率都落在目标概率的\n  \
         4σ 置信带里（σ = √(p(1−p)/n) 是经验频率的固有抖动），说明剩下的偏差就是 {} 次独立试验\n  \
         的统计噪声，而不是采样规则带来的系统性偏移。总变差距离有个 √(K/n) 量级的下限（K 是\n  \
         「有效字符数」），所以它只能当参考，不能当判据。",
        s.trials
    );

    // ---------- 五、MTP：一次前向给出往后 K 个候选 ----------
    logln!("=== 五、多 Token 预测（MTP）：一次前向给出往后 K 个位置的分布 ===");
    let corpus_ids = tokenizer.encode(CORPUS);
    let (b, t) = (s.batch_size, s.block_size);
    let need = b * t + 1;
    assert!(
        corpus_ids.len() >= need,
        "内置语料需要至少 {need} 个 token 才能凑出一个训练批（实际 {}）",
        corpus_ids.len()
    );

    let mtp = speculative::MtpDrafter::new(&target, s.mtp_heads, kv, &mut Rng::new(s.seed + 21));
    assert_eq!(mtp.heads().n_heads(), s.mtp_heads);
    assert_eq!(mtp.heads().n_embd(), target.cfg.n_embd);
    assert_eq!(mtp.heads().vocab(), vocab);

    // 只训预测头（主干冻结）：这一节要验证的是「头能不能学会往后看」，不是再训一次主干
    let params = mtp.heads().parameters();
    let mut opt = AdamW::new(3e-3, params, 0.0);
    let mut history = Vec::with_capacity(s.mtp_steps);
    for _ in 0..s.mtp_steps {
        opt.zero_grad();
        let hidden = target.forward_hidden(&corpus_ids[..need - 1], b, t, true);
        let loss = mtp.heads().loss(&hidden, &corpus_ids[..need - 1], b, t);
        loss.backward();
        opt.step();
        history.push(loss.item());
    }
    let first: f32 = history[..5].iter().sum::<f32>() / 5.0;
    let last: f32 = history[history.len() - 5..].iter().sum::<f32>() / 5.0;
    assert!(
        last < first * 0.95,
        "MTP 损失应随训练下降：首 {first:.4} → 末 {last:.4}"
    );

    let mut mtp_draft = mtp;
    let (mtp_text, mtp_stats) = speculative::speculative_generate(
        &target,
        &tokenizer,
        &s.prompt,
        s.max_new,
        s.gamma,
        &greedy,
        kv,
        &mut mtp_draft,
        &mut Rng::new(s.seed),
    );
    assert_eq!(mtp_text, plain_text, "MTP 草稿路径同样必须无损");

    logln!(
        "  预测头 {} 个（每个头预测往后第 k+1 个 token），训练 {} 步：交叉熵 {:.4} → {:.4}",
        s.mtp_heads,
        s.mtp_steps,
        first,
        last
    );
    logln!("  用 MTP 头当草稿生成 {} 个 token：{}", s.max_new, rag::truncate_chars(&mtp_text, 44));
    report("MTP 草稿", &mtp_stats);
    assert_eq!(mtp_stats.target_forwards, mtp_stats.rounds, "每轮只该有一次目标前向");
    logln!(
        "  对照：MTP 草稿接受率 {:.4}、每次前向 {:.2} 个 token ｜ 第二节的自回归草稿 {:.4}、{:.2} 个 token",
        mtp_stats.acceptance_rate(),
        mtp_stats.tokens_per_forward(),
        other_stats.acceptance_rate(),
        other_stats.tokens_per_forward()
    );
    logln!(
        "  读法：MTP 头挂在主干的隐状态上，**一次前向**就能给出往后 K 个位置的候选分布，\n  \
         不必像小模型草稿那样自回归跑 γ 次前向——草稿侧的算力几乎为零。代价是这 γ 个分布\n  \
         彼此条件独立（都只条件于同一个隐状态），比自回归草稿『糊』，接受率通常更低，\n  \
         这正是「草稿算力 vs 接受率」的取舍。作为训练信号，它还迫使主干把更长的未来信息\n  \
         编码进隐状态里——上面交叉熵的下降就是各头真的在学。"
    );

    runlog::finish();
}

// ==================== 端到端演示（demo 子命令） ====================

fn run_demo() {
    demo_xor();
    demo_bpe();
    demo_transformer();
    #[cfg(feature = "gpu")]
    demo_gpu();
}

/// 演示 1（第 7 课）：用 MLP 学会 XOR 异或
///
/// XOR 是经典的"神经网络必须非线性"案例：
/// 单层线性模型学不会（数据线性不可分），加一层 Tanh 就能学会。
fn demo_xor() {
    println!("=== 演示 1：MLP 学习 XOR（第 7 课）===");
    let mut rng = Rng::new(42);

    // 数据集：4 个样本
    let x_data = Tensor::from_vec(vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], vec![4, 2]);
    let y_targets = vec![0usize, 1, 1, 0]; // XOR 真值表

    // 网络：2 -> 4 (Tanh) -> 2（两个输出：0 和 1 的分数）
    let fc1 = Linear::new(2, 4, &mut rng);
    let fc2 = Linear::new(4, 2, &mut rng);
    let params: Vec<Tensor> = {
        let mut ps = fc1.parameters();
        ps.extend(fc2.parameters());
        ps
    };
    let mut opt = SGD::new(0.5, params);

    for step in 0..1000 {
        // 前向：tanh(x @ W1 + b1) @ W2 + b2
        let h = tanh(&fc1.forward(&x_data));
        let logits = fc2.forward(&h);
        let loss = cross_entropy_loss(&logits, &y_targets);

        loss.backward();
        opt.step();
        opt.zero_grad();

        if step % 200 == 0 {
            println!("  step {:>4} | loss {:.4}", step, loss.item());
        }
    }

    // 验证正确率
    let h = tanh(&fc1.forward(&x_data));
    let logits = fc2.forward(&h);
    let data = logits.data();
    let mut correct = 0;
    for i in 0..4 {
        let pred = if data[i * 2] > data[i * 2 + 1] { 0 } else { 1 };
        if pred == y_targets[i] {
            correct += 1;
        }
    }
    println!("  训练后正确率：{}/4（100% 说明反向传播正确）\n", correct);
}

/// 演示 2（第 8 课）：BPE 分词器
fn demo_bpe() {
    println!("=== 演示 2：BPE 分词器（第 8 课）===");

    let tok = BPETokenizer::train(CORPUS, 400);
    println!(
        "  BPE 词表大小：{}（初始 256 字节 + {} 次合并）",
        tok.vocab_size(),
        tok.vocab_size() - 256
    );

    println!("  \"Red\" 的 token：{:?}", tok.encode("Red"));
    let full = tok.encode("the garden");
    println!(
        "  \"the garden\" -> {} 个 token（高频子词被压缩）",
        full.len()
    );
    let decoded = tok.decode(&full);
    println!("  解码验证：\"{}\"", decoded);

    let char_tok = CharTokenizer::new(CORPUS);
    let ids = char_tok.encode("fox");
    println!(
        "  字符级词表：{}，\"fox\" -> {:?}\n",
        char_tok.vocab_size(),
        ids
    );
}

/// 演示 3（第 12-20、25 课）：训练小 Transformer 并生成文本
fn demo_transformer() {
    println!("=== 演示 3：训练小 Transformer 并生成文本 ===");

    let mut rng = Rng::new(1234);
    let tokenizer = Tokenizer::char(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    println!("  语料 {} 字符，字符词表 {} 个", CORPUS.len(), vocab_size);

    let model = Transformer::new(TransformerConfig::tiny(vocab_size), &mut rng);

    // 训练（第 13、17-18 课：训练循环 + AdamW + warmup/cosine 调度）
    let loader = DataLoader::new(CORPUS, &tokenizer, model.cfg.block_size, 8);
    let tcfg = config::TrainConfig {
        seed: 42,
        batch_size: 8,
        steps: 600,
        max_lr: 3e-3,
        warmup_steps: 50,
        eval_every: 100,
        log_file: None, // 演示不落盘
        ..config::TrainConfig::default()
    };
    train::train_transformer(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);

    // 生成 1（无 cache）：全量前向用滑动窗口，可以生成超过 block_size 的长文本
    // 三次生成共用同一套采样参数（含重复惩罚），否则差异分不清是 cache 还是采样造成的
    let opts = SampleOpts {
        top_k: 10,
        repetition_penalty: 1.1,
        repetition_window: 64,
        ..SampleOpts::default()
    };
    println!("\n  —— 生成 1（prompt=Once upon a, 无 KV cache，全量前向）——");
    let mut rng_full = Rng::new(2024);
    let out_full = generate(&model, &tokenizer, "Once upon a", 80, &opts, KvOpts::off(), &mut rng_full);
    println!("  {out_full}");

    // 生成 2（带 KV cache，第 25 课）：**必须用同一个 prompt 和同一个种子**，否则两次输出
    // 不同只是采样不同，证明不了 cache 的正确性。
    // 缓存带滑动窗口（第 25 课第 8 节），生成长度同样不受缓存容量限制，不会提前停。
    println!("\n  —— 生成 2（同 prompt、同种子，带 KV cache）——");
    let mut rng_kv = Rng::new(2024);
    let out_kv = generate(&model, &tokenizer, "Once upon a", 80, &opts, KvOpts::on(0, None), &mut rng_kv);
    println!("  {out_kv}");
    println!(
        "\n  KV cache 带滑动窗口，生成长度不再受缓存容量限制：两种模式都生成满 80 个 token —— {}",
        if out_kv.chars().count() == out_full.chars().count() {
            "一致 ✓"
        } else {
            "不一致 ✗"
        }
    );

    // 窗口内的严格等价自检（第 25 课第 7 节）：prompt + 新 token 不超 `block_size` 时，
    // 缓存里存的 K/V 与全量重算逐位相同，两条路径必须**逐 token 完全相等**。
    //
    // 超出窗口后两条路径本就不等价，也不是 bug：增量推理里每个位置当时能看到自己的完整窗口，
    // 而"截断重算"会把窗口内各位置在浅层可见的上下文一并砍掉，深层 K/V 随之不同。
    // 带 KV cache 的推理才是真实 LLM 的标准推断语义，所以这里只在窗口内做严格断言。
    let prompt_len = tokenizer.encode("Once upon a").len();
    let in_window = model.cfg.block_size.saturating_sub(prompt_len);
    let mut rng_a = Rng::new(2024);
    let a = generate(&model, &tokenizer, "Once upon a", in_window, &opts, KvOpts::off(), &mut rng_a);
    let mut rng_b = Rng::new(2024);
    let b = generate(&model, &tokenizer, "Once upon a", in_window, &opts, KvOpts::on(0, None), &mut rng_b);
    println!(
        "\n  窗口内等价自检（prompt {prompt_len} + 新 {in_window} = {} ≤ block_size {}）：\
         缓存模式与全量模式逐 token {}",
        prompt_len + in_window,
        model.cfg.block_size,
        if a == b { "完全一致 ✓" } else { "不一致 ✗" }
    );
    if a != b {
        println!("    全量：{a}\n    缓存：{b}");
    }

    // 生成 3：换一个训练语料里没出现过的开头，观察小模型的真实水平（第 16 课第 7 节）。
    // 用全量前向，这样输出的毛病都归模型自己，不被缓存路径的任何细节搅混。
    println!("\n  —— 生成 3（prompt=The fox，换开头看泛化）——");
    let mut rng_fox = Rng::new(2024);
    let out_fox = generate(&model, &tokenizer, "The fox", 80, &opts, KvOpts::off(), &mut rng_fox);
    println!("  {out_fox}");
}

/// 演示 4（第 27 课）：GPU 加速（wgpu 计算着色器）
///
/// 仅 `--features gpu` 时编译。验证 GPU 算子正确性并对比性能；
/// 训练/推理中的矩阵乘已自动走 GPU，失败时静默回退 CPU。
#[cfg(feature = "gpu")]
fn demo_gpu() {
    println!("=== 演示 4：GPU 加速（wgpu 计算着色器）===");
    if !gpu::is_available() {
        println!("  未检测到可用 GPU，已回退 CPU（训练/推理不受影响）\n");
        return;
    }
    println!("  GPU: {}（{}）", gpu::name(), gpu::backend());

    let mut rng = Rng::new(7);

    // 1. 正确性：批量矩阵乘 CPU vs GPU
    let (m, k, n, batch) = (32usize, 24, 40, 8);
    let a: Vec<f32> = (0..batch * m * k).map(|_| rng.randn()).collect();
    let b: Vec<f32> = (0..batch * k * n).map(|_| rng.randn()).collect();
    let cpu = naive_matmul(&a, &b, m, k, n, batch);
    let gpu_out = gpu::matmul(&a, &b, m, k, n, batch, false, false).unwrap();
    let max_err = cpu
        .iter()
        .zip(&gpu_out)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    println!(
        "  批量矩阵乘 [{}x{}]@[{}x{}] x{}：CPU vs GPU 最大误差 {:.2e}",
        m, k, k, n, batch, max_err
    );

    // 2. 性能对比：512x512 矩阵乘
    let (m, k, n) = (512usize, 512, 512);
    let a: Vec<f32> = (0..m * k).map(|_| rng.randn()).collect();
    let b: Vec<f32> = (0..k * n).map(|_| rng.randn()).collect();
    let t0 = std::time::Instant::now();
    let _ = naive_matmul(&a, &b, m, k, n, 1);
    let t_cpu = t0.elapsed();
    let t1 = std::time::Instant::now();
    let _ = gpu::matmul(&a, &b, m, k, n, 1, false, false).unwrap();
    let t_gpu = t1.elapsed();
    let speedup = t_cpu.as_secs_f64() / t_gpu.as_secs_f64().max(1e-9);
    println!(
        "  512x512 矩阵乘：CPU {:.1}ms vs GPU {:.1}ms（快 {:.1}x）",
        t_cpu.as_secs_f64() * 1000.0,
        t_gpu.as_secs_f64() * 1000.0,
        speedup
    );

    // 3. 逐元素算子（scale / relu / add）验证
    let x: Vec<f32> = (0..1024).map(|_| rng.randn()).collect();
    let y: Vec<f32> = (0..1024).map(|_| rng.randn()).collect();
    let s = gpu::scale(&x, 2.0).unwrap();
    let r = gpu::relu(&x).unwrap();
    let z = gpu::add(&x, &y).unwrap();
    let ok = s.iter().zip(&x).all(|(a, b)| (a - b * 2.0).abs() < 1e-4)
        && r.iter().zip(&x).all(|(a, b)| (*a - b.max(0.0)).abs() < 1e-5)
        && z.iter().zip(&x).zip(&y).all(|((a, b), c)| (a - (b + c)).abs() < 1e-4);
    println!(
        "  逐元素算子（scale/relu/add）验证：{}",
        if ok { "通过" } else { "失败" }
    );
    println!("  （训练/推理中 matmul 已自动走 GPU，失败自动回退 CPU）\n");
}

/// 朴素 CPU 批量矩阵乘（仅用于 GPU 正确性/性能对比）
#[cfg(feature = "gpu")]
fn naive_matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, batch: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a[(bi * m + i) * k + kk] * b[(bi * k + kk) * n + j];
                }
                out[(bi * m + i) * n + j] = s;
            }
        }
    }
    out
}
