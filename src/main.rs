//! 从零实现大语言模型（纯 Rust，不依赖深度学习框架）
//!
//! 用法（cli 子命令）：
//! - `cargo run --release -- train    --config config.json [--resume checkpoints/latest.ckpt]`
//! - `cargo run --release -- eval     --config config.json [--ckpt checkpoints/latest.ckpt]`
//! - `cargo run --release -- generate --config config.json [--ckpt ...] --prompt "Once" --max-new 100`
//! - `cargo run --release -- demo`    # 教学演示（XOR / BPE / 内置语料小 GPT）
//!
//! 配套教程文档见 `docs/` 目录。

mod attention;
mod autograd;
mod checkpoint;
mod cli;
mod config;
mod data;
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
use data::{CORPUS, DataLoader};
use layers::{Linear, tanh};
use loss::cross_entropy_loss;
use model::{GPT, GPTConfig};
use module::Module;
use optim::SGD;
use rng::Rng;
use sample::generate;
use tensor::Tensor;
use tokenizer::{BPETokenizer, CharTokenizer, Tokenizer};

fn main() {
    init_console_utf8();
    let cli = Cli::parse_args();
    match cli.cmd {
        Cmd::Train { config, resume } => cmd_train(&config, resume.as_deref()),
        Cmd::Eval { config, ckpt } => cmd_eval(&config, ckpt.as_deref()),
        Cmd::Generate {
            config,
            ckpt,
            prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            seed,
            no_kv_cache,
        } => cmd_generate(
            &config,
            ckpt.as_deref(),
            &prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            seed,
            no_kv_cache,
        ),
        Cmd::Demo => run_demo(),
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

#[cfg(not(windows))]
fn init_console_utf8() {}

/// 按配置重建分词器（char / bpe）。
/// `expect_vocab = 0` 表示不校验（训练时词表由分词器决定）。
fn build_tokenizer(tcfg: &config::TrainConfig, train_text: &str, expect_vocab: usize) -> Tokenizer {
    let tok = Tokenizer::from_name(&tcfg.tokenizer, train_text, tcfg.bpe_vocab);
    if expect_vocab != 0 {
        assert_eq!(
            tok.vocab_size(),
            expect_vocab,
            "分词器词表（{}）与模型/checkpoint（{}）不一致：请确认 config.json 与训练时保持一致",
            tok.vocab_size(),
            expect_vocab
        );
    }
    tok
}

/// 训练：`train --config config.json [--resume ckpt]`
fn cmd_train(config_path: &str, resume: Option<&str>) {
    let cfg = Config::load(config_path);
    let tcfg = &cfg.train;

    let train_text = read_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    let tokenizer = build_tokenizer(tcfg, &train_text, 0); // 训练时词表由分词器决定

    // 词表大小 0 表示"由分词器决定"
    let mut model_cfg = cfg.model.clone();
    if model_cfg.vocab_size == 0 {
        model_cfg.vocab_size = tokenizer.vocab_size();
    }

    let mut rng = Rng::new(tcfg.seed);
    let model = GPT::new(model_cfg.clone(), &mut rng);
    let loader = DataLoader::from_texts(
        &train_text,
        val_text.as_deref(),
        &tokenizer,
        model_cfg.block_size,
        tcfg.batch_size,
    );
    train::train_gpt(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&tcfg.out_dir),
        resume,
        &mut rng,
    );
}

/// 评估：在验证集上计算 loss 与困惑度
fn cmd_eval(config_path: &str, ckpt_path: Option<&str>) {
    let cfg = Config::load(config_path);
    let tcfg = &cfg.train;
    let ckpt_path = match ckpt_path {
        Some(p) => p.to_string(),
        None => format!("{}/latest.ckpt", tcfg.out_dir),
    };

    let ckpt = checkpoint::load_header(&ckpt_path);
    let train_text = read_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    let tokenizer = build_tokenizer(tcfg, &train_text, ckpt.model.vocab_size);

    let mut rng = Rng::new(tcfg.seed);
    let model = GPT::new(ckpt.model.clone(), &mut rng);
    checkpoint::load_params(&ckpt_path, &model);
    let loader = DataLoader::from_texts(
        &train_text,
        val_text.as_deref(),
        &tokenizer,
        ckpt.model.block_size,
        tcfg.batch_size,
    );

    let mut eval_rng = Rng::new(tcfg.seed); // 固定种子，结果可复现
    let loss = train::eval_loss(&model, &loader, tcfg.eval_iters, &mut eval_rng);
    println!(
        "评估 step {}：val_loss {:.4} | perplexity {:.2}（{} 个 token 的验证集上采 {} 批）",
        ckpt.step,
        loss,
        loss.exp(),
        loader.num_val_tokens(),
        tcfg.eval_iters
    );
}

/// 生成：`generate --config config.json --ckpt ckpt --prompt "..."`
#[allow(clippy::too_many_arguments)]
fn cmd_generate(
    config_path: &str,
    ckpt_path: Option<&str>,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    no_kv_cache: bool,
) {
    let cfg = Config::load(config_path);
    let tcfg = &cfg.train;
    let ckpt_path = match ckpt_path {
        Some(p) => p.to_string(),
        None => format!("{}/latest.ckpt", tcfg.out_dir),
    };

    let ckpt = checkpoint::load_header(&ckpt_path);
    let train_text = read_text(&tcfg.train_file);
    let tokenizer = build_tokenizer(tcfg, &train_text, ckpt.model.vocab_size);

    let mut rng = Rng::new(seed);
    let model = GPT::new(ckpt.model.clone(), &mut rng);
    checkpoint::load_params(&ckpt_path, &model);

    let use_kv_cache = !no_kv_cache;
    println!(
        "生成（temperature={} top-k={} top-p={}，KV cache {}）：",
        temperature,
        top_k,
        top_p,
        if use_kv_cache { "开" } else { "关" }
    );
    let out = generate(
        &model,
        &tokenizer,
        prompt,
        max_new,
        temperature,
        top_k,
        top_p,
        use_kv_cache,
        &mut rng,
    );
    println!("{}", out);
}

// ==================== 教学演示（demo 子命令） ====================

fn run_demo() {
    demo_xor();
    demo_bpe();
    demo_gpt();
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
    let opt = SGD::new(0.5, params);

    for step in 0..1000 {
        // 前向：tanh(x @ W1 + b1) @ W2 + b2
        let h = tanh(&fc1.forward(&x_data));
        let logits = fc2.forward(&h);
        let loss = cross_entropy_loss(&logits, &y_targets);

        loss.backward();
        opt.step();
        opt.zero_grad();

        if step % 200 == 0 {
            println!("  step {:>4} | loss {:.4}", step, loss.data()[0]);
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

/// 演示 3（第 12-16、17-20 课）：训练小 GPT 并生成文本
fn demo_gpt() {
    println!("=== 演示 3：训练小 GPT 并生成文本 ===");

    let mut rng = Rng::new(1234);
    let tokenizer = Tokenizer::char(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    println!("  语料 {} 字符，字符词表 {} 个", CORPUS.len(), vocab_size);

    let model = GPT::new(GPTConfig::tiny(vocab_size), &mut rng);

    // 训练（第 13、17、20 课：训练循环 + AdamW + warmup/cosine 调度）
    let loader = DataLoader::new(CORPUS, &tokenizer, model.cfg.block_size, 8);
    let tcfg = config::TrainConfig {
        seed: 42,
        batch_size: 8,
        steps: 600,
        max_lr: 3e-3,
        warmup_steps: 50,
        eval_every: 100,
        ..config::TrainConfig::default()
    };
    train::train_gpt(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);

    // 生成（无 cache）
    println!("\n  —— 生成 1（temperature=0.8, top-k=10, top-p=0.9, 无 KV cache）——");
    let out1 = generate(
        &model,
        &tokenizer,
        "Once upon a",
        80,
        0.8,
        10,
        0.9,
        false,
        &mut rng,
    );
    println!("  {}", out1);

    // 生成（带 KV cache，第 18 课）
    println!("\n  —— 生成 2（temperature=0.8, top-k=10, top-p=0.9, 带 KV cache）——");
    let out2 = generate(
        &model, &tokenizer, "The fox", 80, 0.8, 10, 0.9, true, &mut rng,
    );
    println!("  {}", out2);
    println!("\n  （KV cache 只改计算方式、不改生成分布，两者应高度一致）");
}
