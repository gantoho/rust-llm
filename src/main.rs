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
    #[cfg(feature = "gpu")]
    gpu::init();
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

/// 演示 4（第 21 课）：GPU 加速（wgpu 计算着色器）
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
    let gpu_out = gpu::matmul(&a, &b, m, k, n, batch).unwrap();
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
    let _ = gpu::matmul(&a, &b, m, k, n, 1).unwrap();
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
