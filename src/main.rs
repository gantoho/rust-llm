//! 从零实现大语言模型（纯 Rust，不依赖深度学习框架）
//!
//! 用法（cli 子命令）：
//! - `cargo run --release -- train    --config config/config.json [--resume checkpoints/latest.ckpt]`
//! - `cargo run --release -- eval     --config config/config.json [--ckpt checkpoints/latest.ckpt]`
//! - `cargo run --release -- generate --config config/config.json [--ckpt ...] --prompt "Once" --max-new 100`
//! - `cargo run --release -- chat     --config config/config.json [--ckpt ...] [--system "..."]`
//! - `cargo run --release -- serve    --config config/config.json [--ckpt ...] [--host 127.0.0.1 --port 8080]`
//! - `cargo run --release -- sft      --config config/config.json --pretrained ckpt [--sft-file "..."]`
//! - `cargo run --release -- finetune --config config/config.json --pretrained ckpt [--lora-rank 16]`
//! - `cargo run --release -- preset   [--name small] [--output config/config.json]`
//! - `cargo run --release -- demo`    # 端到端演示（XOR / BPE / 内置语料小 Transformer）
//!
//! 配套教程文档见 `docs/` 目录。
//!
//! 每次训练 / 推理都会自动在 `logs/` 下生成一份运行日志（文件名含操作名与毫秒级时间），
//! 内容包含完整命令行、完整配置与全部过程输出，见 [`runlog`]。

// serve 里的 OpenAPI 规范是一整个巨大的 `json!{...}`，宏展开层数超过默认上限
#![recursion_limit = "512"]

// `runlog` 里的 `logln!` 宏要覆盖后面所有模块，必须最先声明并带 #[macro_use]
#[macro_use]
mod runlog;

mod align;
mod attention;
mod autograd;
mod checkpoint;
mod cli;
mod cmd;
mod comm;
mod config;
mod csa;
mod data;
mod distill;
mod distributed;
mod fp8;
mod fp4;
#[cfg(feature = "gpu")]
mod gpu;
mod hf;
mod layers;
mod loss;
mod mhc;
mod model;
mod moe;
mod module;
mod optim;
mod prompt;
mod quant;
mod rag;
mod rng;
mod rope;
mod sample;
mod scaling;
mod serve;
mod speculative;
mod tensor;
mod tokenizer;
mod train;
mod vision;
mod vqvae;

use cli::{Cli, Cmd};
use sample::{KvOpts, SampleOpts};

use cmd::core::{cmd_bench, cmd_eval, cmd_finetune, cmd_generate, cmd_import_hf, cmd_preset, cmd_quant, cmd_sft, cmd_train};
use cmd::deploy::{cmd_chat, cmd_serve};
use cmd::experiments::{cmd_align, cmd_csa, cmd_distill, cmd_distributed, cmd_fp4, cmd_fp8, cmd_mla, cmd_moe, cmd_rag, cmd_scaling, cmd_speculative, run_demo};
use cmd::{init_console_utf8, kv_bits_from_name, stop_label};

fn main() {
    runlog::mark_start(); // 记下命令开始执行的时刻，作为运行日志总耗时的基准
    init_console_utf8();
    #[cfg(feature = "gpu")]
    gpu::init();
    let cli = Cli::parse_args();
    match cli.cmd {
        Cmd::Train { config, resume } => cmd_train(&config, resume.as_deref()),
        Cmd::Eval {
            config,
            ckpt,
            tokenizer,
            merge_lora,
            template,
            sft_file,
            mask_response,
        } => cmd_eval(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
            merge_lora,
            &template,
            sft_file.as_deref(),
            mask_response,
        ),
        Cmd::Generate {
            config,
            ckpt,
            tokenizer,
            merge_lora,
            prompt,
            image,
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
            image.as_deref(),
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
        Cmd::Serve {
            config,
            ckpt,
            tokenizer,
            merge_lora,
            host,
            port,
            api_key,
            cors,
            no_web,
            web_port,
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
        } => cmd_serve(
            &config,
            ckpt.as_deref(),
            tokenizer.as_deref(),
            merge_lora,
            &host,
            port,
            api_key,
            cors,
            no_web,
            web_port,
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
        Cmd::ImportHf { model_dir, output, tokenizer, tokenizer_out } => cmd_import_hf(
            &model_dir,
            &output,
            tokenizer.as_deref(),
            tokenizer_out.as_deref(),
        ),
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
            shared_experts,
            bias_balance,
            bias_lr,
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
            shared_experts,
            bias_balance,
            bias_lr,
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
        Cmd::Distill { distill } => cmd_distill(&distill),
        Cmd::Fp8 { fp8 } => cmd_fp8(&fp8),
        Cmd::Fp4 { fp4 } => cmd_fp4(&fp4),
        Cmd::Csa { csa } => cmd_csa(&csa),
        Cmd::Mla { mla } => cmd_mla(&mla),
    }
}
