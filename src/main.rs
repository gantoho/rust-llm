//! 从零实现大语言模型（纯 Rust，不依赖深度学习框架）
//!
//! 用法（cli 子命令）：
//! - `cargo run --release -- train    --config config.json [--resume checkpoints/latest.ckpt]`
//! - `cargo run --release -- eval     --config config.json [--ckpt checkpoints/latest.ckpt]`
//! - `cargo run --release -- generate --config config.json [--ckpt ...] --prompt "Once" --max-new 100`
//! - `cargo run --release -- chat     --config config.json [--ckpt ...] [--system "..."]`
//! - `cargo run --release -- finetune --config config.json --pretrained ckpt [--lora-rank 16]`
//! - `cargo run --release -- preset   [--name small] [--output config.json]`
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
use data::{CORPUS, DataLoader, load_text};
use layers::{Linear, tanh};
use loss::cross_entropy_loss;
use model::{GPT, GPTConfig};
use module::Module;
use optim::{Optimizer, SGD};
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
            temperature,
            top_k,
            top_p,
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
            max_new,
            seed,
        } => cmd_chat(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
            &system,
            temperature,
            top_k,
            top_p,
            max_new,
            seed,
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
/// 2. `tcfg.tokenizer_file`（config.json 中配置的路径）
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
        println!("分词器：从 {} 加载（{}，词表 {}）", path, loaded.kind(), loaded.vocab_size());
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
        println!("分词器：从 {} 加载（{}，词表 {}）", path, loaded.kind(), loaded.vocab_size());
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
        println!("分词器：从 {} 加载（{}，词表 {}）", auto_path, loaded.kind(), loaded.vocab_size());
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
        let loaded = Tokenizer::load(path);
        println!("已从 {} 加载分词器（{}，词表 {}）", path, loaded.kind(), loaded.vocab_size());
        loaded
    } else {
        Tokenizer::from_name(&tcfg.tokenizer, train_text, tcfg.bpe_vocab)
    };
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
    #[cfg(feature = "gpu")]
    if gpu::is_available() {
        println!("GPU: {}（{}）", gpu::name(), gpu::backend());
    } else {
        println!("未检测到可用 GPU，本次训练走 CPU");
    }

    let train_text = load_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    let tokenizer = build_tokenizer(tcfg, &train_text, 0); // 训练时词表由分词器决定

    // 训练完成后保存分词器
    if tcfg.tokenizer_file.is_none() {
        let tok_path = format!("{}/tokenizer.json", tcfg.out_dir);
        std::fs::create_dir_all(&tcfg.out_dir).expect("创建 checkpoint 目录失败");
        tokenizer.save(&tok_path);
        println!("分词器已保存到 {tok_path}");
    }

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
fn cmd_eval(config_path: &str, ckpt_path: Option<&str>, tokenizer_path: Option<&str>) {
    let cfg = Config::load(config_path);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
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
    tokenizer_path: Option<&str>,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    seed: u64,
    no_kv_cache: bool,
    beam: Option<usize>,
    length_penalty: f32,
) {
    let cfg = Config::load(config_path);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let (model, tokenizer, _ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    let mut rng = Rng::new(seed);

    if let Some(beam_size) = beam {
        // Beam Search 生成
        println!(
            "Beam Search 生成（beam_size={} length_penalty={}）：",
            beam_size, length_penalty
        );
        let out = sample::beam_search(
            &model,
            &tokenizer,
            prompt,
            max_new,
            beam_size,
            length_penalty,
            &mut rng,
        );
        println!("{}", out);
    } else {
        // 采样生成
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
}

/// 交互式对话模式
#[allow(clippy::too_many_arguments)]
fn cmd_chat(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    system: &str,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    max_new: usize,
    seed: u64,
) {
    let cfg = Config::load(config_path);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let (model, tokenizer, _ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, seed, tokenizer_path);
    let mut rng = Rng::new(seed);

    println!("交互式对话模式（输入文本后按回车生成，输入 :quit 退出）");
    println!("参数：temperature={} top-k={} top-p={}", temperature, top_k, top_p);
    if !system.is_empty() {
        println!("系统提示：{}", system);
    }
    println!("---");

    let mut context_history = if !system.is_empty() {
        system.to_string()
    } else {
        String::new()
    };

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

        // 构造 prompt：历史 + 当前输入
        let prompt = if context_history.is_empty() {
            input.to_string()
        } else {
            format!("{}\n{}", context_history, input)
        };

        let out = generate(
            &model,
            &tokenizer,
            &prompt,
            max_new,
            temperature,
            top_k,
            top_p,
            true, // 始终使用 KV cache 加速
            &mut rng,
        );

        // 只打印新生成的部分（去掉 prompt 前缀）
        // 用 is_char_boundary 确保不在多字节字符中间截断（UTF-8 安全）
        let response = if out.len() > prompt.len() {
            let start = prompt.len();
            let start = if out.is_char_boundary(start) { start } else { start + 1 };
            out[start..].trim()
        } else {
            &out
        };
        println!("{}", response);

        // 更新历史上下文（截断到 block_size 以内的字符数）
        context_history = format!("{}\n{}\n{}", prompt, input, response);
        let max_chars = model.cfg.block_size * 4; // 粗略估计：平均每个 token ~4 字符
        if context_history.len() > max_chars {
            let skip = context_history.len() - max_chars;
            if let Some(pos) = context_history[skip..].find('\n') {
                context_history = context_history[skip + pos + 1..].to_string();
            }
        }
    }
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
    let mut cfg = Config::load(config_path);
    // 设置 LoRA 配置
    cfg.train.lora = Some(config::LoRAConfig {
        rank: lora_rank,
        alpha: lora_alpha,
    });
    cfg.train.steps = steps;
    cfg.train.max_lr = lr;
    cfg.train.min_lr = lr * 0.1;

    let tcfg = &cfg.train;
    println!(
        "LoRA 微调：rank={} alpha={} steps={} lr={}",
        lora_rank, lora_alpha, steps, lr
    );

    let (model, tokenizer, ckpt) = load_model_and_tokenizer(pretrained_path, tcfg, tcfg.seed, None);
    let train_text = load_text(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);

    // 打印模型信息
    let total_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    let lora_params = 2 * lora_rank * ckpt.model.n_embd * 3; // Q/K/V 各一个 LoRA
    println!(
        "模型参数：{}（冻结）| LoRA 参数：{}（可训练，占 {:.1}%）",
        total_params,
        lora_params,
        100.0 * lora_params as f32 / total_params as f32
    );

    let mut rng = Rng::new(tcfg.seed);
    let loader = DataLoader::from_texts(
        &train_text,
        val_text.as_deref(),
        &tokenizer,
        ckpt.model.block_size,
        tcfg.batch_size,
    );
    train::train_gpt(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&tcfg.out_dir),
        None,
        &mut rng,
    );
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
        let _ = generate(model, tokenizer, prompt, n, 0.8, 40, 0.9, use_kv, &mut rng);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = generate(model, tokenizer, prompt, n, 0.8, 40, 0.9, use_kv, &mut rng);
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
