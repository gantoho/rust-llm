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
//! - `cargo run --release -- demo`    # 教学演示（XOR / BPE / 内置语料小 GPT）
//!
//! 配套教程文档见 `docs/` 目录。
//!
//! 每次训练 / 推理都会自动在 `logs/` 下生成一份运行日志（文件名含操作名与毫秒级时间），
//! 内容包含完整命令行、完整配置与全部过程输出，见 [`runlog`]。

// `runlog` 里的 `logln!` 宏要覆盖后面所有模块，必须最先声明并带 #[macro_use]
#[macro_use]
mod runlog;

mod attention;
mod autograd;
mod checkpoint;
mod cli;
mod config;
mod data;
#[cfg(feature = "gpu")]
mod gpu;
mod layers;
mod loss;
mod model;
mod module;
mod optim;
mod rng;
mod rope;
mod sample;
mod tensor;
mod tokenizer;
mod train;

use cli::{Cli, Cmd};
use config::Config;
use data::{
    BatchSource, CORPUS, DataLoader, SFT_ASSISTANT, SFT_END, SFT_USER, SftLoader, load_text,
    load_texts,
};
use layers::{Linear, tanh};
use loss::cross_entropy_loss;
use model::{GPT, GPTConfig};
use module::Module;
use optim::{Optimizer, SGD};
use rng::Rng;
use sample::{SampleOpts, generate};
use tensor::Tensor;
use tokenizer::{BPETokenizer, CharTokenizer, Tokenizer};

fn main() {
    runlog::mark_start(); // 记下命令开始执行的时刻，作为运行日志总耗时的基准
    init_console_utf8();
    #[cfg(feature = "gpu")]
    gpu::init();
    let cli = Cli::parse_args();
    match cli.cmd {
        Cmd::Train { config, resume } => cmd_train(&config, resume.as_deref()),
        Cmd::Eval { config, ckpt, tokenizer } => cmd_eval(&config, ckpt.as_deref(), tokenizer.as_deref()),
        Cmd::Generate {
            config,
            ckpt,
            tokenizer,
            prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_window,
            seed,
            no_kv_cache,
            beam,
            length_penalty,
        } => cmd_generate(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
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
            no_kv_cache,
            beam,
            length_penalty,
        ),
        Cmd::Chat {
            config,
            ckpt,
            tokenizer,
            system,
            temperature,
            top_k,
            top_p,
            repetition_penalty,
            repetition_window,
            max_new,
            seed,
            prompt_format,
        } => cmd_chat(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
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
            seed,
            &prompt_format,
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
            steps,
            lr,
        } => cmd_finetune(&config, &pretrained, lora_rank, lora_alpha, steps, lr),
        Cmd::Preset { name, output } => cmd_preset(&name, &output),
        Cmd::Demo => run_demo(),
        Cmd::Bench { steps, gen_tokens } => cmd_bench(steps, gen_tokens),
    }
}

/// 读取文本文件
fn read_text(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("无法读取数据文件 {path}: {e}"))
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

/// 从 checkpoint 加载模型和分词器的通用辅助函数（推理用）
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
) -> (GPT, Tokenizer, checkpoint::Checkpoint) {
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
    let model = GPT::new(ckpt.model.clone(), &mut rng);
    checkpoint::load_params(ckpt_path, &model);
    (model, tokenizer, ckpt)
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

    let train_text = load_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    let tokenizer = build_tokenizer(tcfg, &train_text, 0); // 训练时词表由分词器决定

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
    let model = GPT::new(model_cfg.clone(), &mut rng);
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
    let loader = DataLoader::from_texts(
        &train_text,
        val_text.as_deref(),
        &tokenizer,
        model_cfg.block_size,
        tcfg.batch_size,
    );
    let best = train::train_gpt(
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
fn cmd_eval(config_path: &str, ckpt_path: Option<&str>, tokenizer_path: Option<&str>) {
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
    let (model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, tcfg.seed, tokenizer_path);

    let train_text = load_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    let loader = DataLoader::from_texts(
        &train_text,
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

/// 生成：`generate --config config/config.json --ckpt ckpt --prompt "..."`
#[allow(clippy::too_many_arguments)]
fn cmd_generate(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    prompt: &str,
    max_new: usize,
    opts: SampleOpts,
    seed: u64,
    no_kv_cache: bool,
    beam: Option<usize>,
    length_penalty: f32,
) {
    let log_path = runlog::start("generate");
    println!("运行日志：{log_path}");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let (model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    let mut rng = Rng::new(seed);
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
            (
                "KV cache",
                match beam {
                    Some(_) => "不适用（Beam Search 走全量前向）".to_string(),
                    None => if no_kv_cache { "关（每次全量前向）" } else { "开" }.to_string(),
                },
            ),
        ],
    );

    let out = if let Some(beam_size) = beam {
        // Beam Search 生成
        logln!(
            "Beam Search 生成（beam_size={} length_penalty={}）：",
            beam_size, length_penalty
        );
        sample::beam_search(
            &model,
            &tokenizer,
            prompt,
            max_new,
            beam_size,
            length_penalty,
            &mut rng,
        )
    } else {
        // 采样生成
        let use_kv_cache = !no_kv_cache;
        logln!(
            "生成（temperature={} top-k={} top-p={} 重复惩罚={}（窗口 {}），KV cache {}）：",
            opts.temperature,
            opts.top_k,
            opts.top_p,
            opts.repetition_penalty,
            opts.repetition_window,
            if use_kv_cache { "开" } else { "关" }
        );
        generate(&model, &tokenizer, prompt, max_new, &opts, use_kv_cache, &mut rng)
    };
    logln!("{out}");
    runlog::append(&format!("[done] 输出总长 {} 字符（含 prompt）", out.chars().count()));
    runlog::finish();
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
    system: &str,
    opts: SampleOpts,
    max_new: usize,
    seed: u64,
    prompt_format: &str,
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
    let (model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    let mut rng = Rng::new(seed);
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
            ("KV cache", "开".to_string()),
        ],
    );

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
    //   2. 本轮生成 —— 预留 max_new 个 token，否则 prompt + 生成超过 block_size 时
    //      KV cache 写满，生成被提前截断（历史自己就超窗口时甚至只剩 1 个 token）
    //   3. 对话历史 —— 剩下的额度都给它，不够就按 token 数从最老的开始丢
    let block_size = model.cfg.block_size;
    if max_new >= block_size {
        logln!(
            "[warn] --max-new={max_new} 不小于上下文窗口 {block_size}，本轮生成最多 {block_size} 个 token 就会停"
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

        let out = generate(
            &model,
            &tokenizer,
            &prompt,
            max_new,
            &opts,
            true, // 始终使用 KV cache 加速
            &mut rng,
        );

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

    let best = train::train_gpt(
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

/// LoRA 微调：加载预训练模型，冻结主参数，只训练 LoRA 层
fn cmd_finetune(
    config_path: &str,
    pretrained_path: &str,
    lora_rank: usize,
    lora_alpha: f32,
    steps: usize,
    lr: f32,
) {
    let log_path = runlog::start("finetune");
    println!("运行日志：{log_path}");
    let mut cfg = Config::load(config_path);
    // 设置 LoRA 配置
    cfg.train.lora = Some(config::LoRAConfig {
        rank: lora_rank,
        alpha: lora_alpha,
    });
    cfg.train.steps = steps;
    cfg.train.max_lr = lr;
    cfg.train.min_lr = lr * 0.1;

    // 日志记的是**覆盖 CLI 参数之后**的最终配置（这才是本次实际使用的配置）
    runlog::json(&format!("完整配置（{config_path} + CLI 覆盖后）"), &cfg);
    let tcfg = &cfg.train;
    logln!(
        "微调（全参更新）：steps={steps} lr={lr}｜--lora-rank/--lora-alpha 传了 {} / {}，\
         但 LoRA 尚未接入训练循环，本次不冻结主参数",
        lora_rank,
        lora_alpha
    );

    let (model, tokenizer, ckpt) = load_model_and_tokenizer(pretrained_path, tcfg, tcfg.seed, None);
    let train_text = load_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);

    // 打印模型信息
    //
    // ⚠️ 如实说明：LoRA 尚未接入训练循环。`--lora-rank/--lora-alpha` 只做校验与预估，
    // 既不冻结主参数、也不注入适配层，本次训练的**就是全部参数**。这里绝不能打印
    // "冻结 / 可训练比例"——那会让人以为显存和可训练参数量真的降下来了。
    // LoRA 原理与完整实现见 `docs/29-LoRA低秩适配.md`。
    let total_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    // 预估口径：Q/K/V 各一个 LoRA，两个低秩矩阵（r×d 与 d×r），逐层累加。
    // 未考虑 GQA 下 K/V 的输出维更小，所以只是量级参考。
    let lora_params = 2 * lora_rank * ckpt.model.n_embd * 3 * ckpt.model.n_layer;
    logln!(
        "模型参数：{}（本次**全部**参与训练）| LoRA rank={lora_rank} 未接入，\
         接入后约 {} 个（{:.1}%）",
        total_params,
        lora_params,
        100.0 * lora_params as f32 / total_params as f32
    );
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
                "LoRA rank / alpha",
                format!("{lora_rank} / {lora_alpha}（未接入训练循环，仅校验与预估）"),
            ),
            ("总参数", total_params.to_string()),
            ("LoRA 预估参数", lora_params.to_string()),
            ("训练语料", tcfg.train_file.clone()),
            ("输出目录", tcfg.out_dir.clone()),
        ],
    );

    let mut rng = Rng::new(tcfg.seed);
    let loader = DataLoader::from_texts(
        &train_text,
        val_text.as_deref(),
        &tokenizer,
        ckpt.model.block_size,
        tcfg.batch_size,
    );
    let best = train::train_gpt(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&tcfg.out_dir),
        None,
        &mut rng,
    );
    runlog::append(&format!("微调结束：best val loss = {best:.4}"));
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
        println!("  架构：GPT-2 风格（LayerNorm + GELU + MHA）");
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
    let gcfg = GPTConfig {
        vocab_size,
        n_embd: 128,
        n_head: 4,
        n_layer: 2,
        block_size: 64,
        ..GPTConfig::default()
    };
    let model = GPT::new(gcfg.clone(), &mut rng);
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
    train::train_gpt(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);
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
        model: &GPT,
        tokenizer: &Tokenizer,
        prompt: &str,
        n: usize,
        use_kv: bool,
        reps: usize,
    ) -> f64 {
        let mut rng = Rng::new(42);
        let opts = SampleOpts::default();
        let _ = generate(model, tokenizer, prompt, n, &opts, use_kv, &mut rng);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = generate(model, tokenizer, prompt, n, &opts, use_kv, &mut rng);
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    }

    let prompt = "Once upon a time";
    let prompt_len = tokenizer.encode(prompt).len();
    // KV cache 模式下上下文总长达到 block_size 就会停，这里取不超过该上限
    let kv_new = gen_tokens.min(gcfg.block_size.saturating_sub(prompt_len + 1));
    let kv_secs = bench_generate(&model, &tokenizer, prompt, kv_new, true, 5);
    println!(
        "[bench] infer/kv  : {} tok | {:.4}s | {:.1} tok/s",
        kv_new,
        kv_secs,
        kv_new as f64 / kv_secs
    );

    let full_new = gen_tokens.min(24);
    let full_secs = bench_generate(&model, &tokenizer, prompt, full_new, false, 3);
    println!(
        "[bench] infer/full: {} tok | {:.4}s | {:.1} tok/s",
        full_new,
        full_secs,
        full_new as f64 / full_secs
    );
    println!("（以上 tok/s 越高越好；优化前后同机对比即可看出收益）");
}

// ==================== 教学演示（demo 子命令） ====================

fn run_demo() {
    demo_xor();
    demo_bpe();
    demo_gpt();
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

/// 演示 3（第 12-20、25 课）：训练小 GPT 并生成文本
fn demo_gpt() {
    println!("=== 演示 3：训练小 GPT 并生成文本 ===");

    let mut rng = Rng::new(1234);
    let tokenizer = Tokenizer::char(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    println!("  语料 {} 字符，字符词表 {} 个", CORPUS.len(), vocab_size);

    let model = GPT::new(GPTConfig::tiny(vocab_size), &mut rng);

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
    train::train_gpt(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);

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
    let out_full = generate(&model, &tokenizer, "Once upon a", 80, &opts, false, &mut rng_full);
    println!("  {out_full}");

    // 生成 2（带 KV cache，第 25 课）：**必须用同一个 prompt 和同一个种子**，否则两次输出
    // 不同只是采样不同，证明不了 cache 的正确性。
    // 注意 cache 模式受 block_size 限制：缓存填满后无法像全量模式那样滑动窗口，
    // 会在窗口边界提前结束并打印 [warn]（这是当前实现的有意限制，见第 25 课第 8 节）。
    println!("\n  —— 生成 2（同 prompt、同种子，带 KV cache）——");
    let mut rng_kv = Rng::new(2024);
    let out_kv = generate(&model, &tokenizer, "Once upon a", 80, &opts, true, &mut rng_kv);
    println!("  {out_kv}");

    let consistent = out_full.starts_with(&out_kv);
    println!(
        "\n  KV cache 只改计算方式、不改生成分布：cache 输出应恰为全量输出的前缀 —— {}",
        if consistent { "一致 ✓" } else { "不一致 ✗" }
    );

    // 生成 3：换一个训练语料里没出现过的开头，观察小模型的真实水平（第 16 课第 7 节）。
    // 用全量前向，这样输出的毛病都归模型自己，不会被缓存窗口截断搅混。
    println!("\n  —— 生成 3（prompt=The fox，换开头看泛化）——");
    let mut rng_fox = Rng::new(2024);
    let out_fox = generate(&model, &tokenizer, "The fox", 80, &opts, false, &mut rng_fox);
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
