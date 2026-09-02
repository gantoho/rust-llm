//! 训练循环与学习率调度（第 13、20 课）
//!
//! 训练 GPT 的完整骨架：
//! 1. 采样一个 batch
//! 2. 前向算损失
//! 3. 反向算梯度
//! 4. 梯度裁剪（防止梯度爆炸）
//! 5. 优化器更新参数
//! 6. 清零梯度
//!
//! 学习率调度（第 20 课）：
//! - warmup：前若干步学习率从 0 线性升到最大值（让训练稳定起步）
//! - cosine decay：之后按余弦曲线衰减到最小值（后期精细收敛）
//!
//! 工程化支持：
//! - 每 `eval_every` 步在验证集上评估 loss / 困惑度（perplexity）
//! - 周期性保存 checkpoint（latest / best），支持 `--resume` 断点续训

use crate::config::TrainConfig;
use crate::data::DataLoader;
use crate::loss::cross_entropy_loss;
use crate::model::GPT;
use crate::module::{Module, zero_grad_all};
use crate::optim::{AdamW, Optimizer};
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use crate::{checkpoint, checkpoint::Checkpoint};

/// 学习率调度器：warmup + cosine decay
pub struct LRScheduler {
    warmup_steps: usize,
    total_steps: usize,
    max_lr: f32,
    min_lr: f32,
    step: usize,
}

impl LRScheduler {
    pub fn new(warmup_steps: usize, total_steps: usize, max_lr: f32, min_lr: f32) -> Self {
        LRScheduler {
            warmup_steps,
            total_steps,
            max_lr,
            min_lr,
            step: 0,
        }
    }

    /// 当前学习率
    pub fn lr(&self) -> f32 {
        if self.step < self.warmup_steps {
            // 线性 warmup
            self.max_lr * (self.step as f32 + 1.0) / self.warmup_steps.max(1) as f32
        } else {
            // cosine 衰减：从 max_lr 平滑降到 min_lr
            let progress = (self.step - self.warmup_steps) as f32
                / (self.total_steps - self.warmup_steps).max(1) as f32;
            let progress = progress.min(1.0);
            let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
            self.min_lr + (self.max_lr - self.min_lr) * cosine
        }
    }

    pub fn step(&mut self) {
        self.step += 1;
    }

    /// 从 checkpoint 恢复时直接跳到对应步数
    pub fn set_step(&mut self, step: usize) {
        self.step = step;
    }
}

/// 梯度裁剪：如果所有参数梯度的总范数超过 max_norm，就整体等比缩放。
/// 防止个别大梯度把参数"推飞"，这是训练 LLM 的标准防爆措施。
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) {
    let mut total = 0.0f32;
    for p in params {
        let g = p.grad.borrow();
        total += g.iter().map(|v| v * v).sum::<f32>();
    }
    let norm = total.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for p in params {
            for v in p.grad.borrow_mut().iter_mut() {
                *v *= scale;
            }
        }
    }
}

/// 在验证集上评估：平均 loss（perplexity = e^loss）
///
/// `eval_iters` 批的平均，调用方用固定种子的 Rng 可保证结果可复现。
pub fn eval_loss(model: &GPT, loader: &DataLoader, eval_iters: usize, rng: &mut Rng) -> f32 {
    zero_grad_all(model); // 评估前清零梯度，避免残留影响
    let mut total = 0.0f32;
    for _ in 0..eval_iters {
        let (x, y) = loader.eval_batch(rng);
        let logits = model.forward(&x, loader.batch_size(), loader.block_size(), None);
        let loss = cross_entropy_loss(&logits, &y);
        total += loss.item();
    }
    total / eval_iters as f32
}

/// 训练函数（支持验证评估与 checkpoint）
///
/// - `out_dir = None` 时不保存 checkpoint（demo 用）
/// - `resume_from = Some(path)` 时从 checkpoint 续训（恢复参数、优化器状态与步数）
///
/// 返回最终的最优验证 loss（无验证集时为训练 loss 近似值）。
pub fn train_gpt(
    model: &GPT,
    tokenizer: &Tokenizer,
    loader: &DataLoader,
    cfg: &TrainConfig,
    out_dir: Option<&str>,
    resume_from: Option<&str>,
    rng: &mut Rng,
) -> f32 {
    let params = model.parameters();
    let mut opt = AdamW::new(cfg.max_lr, params.clone(), cfg.weight_decay);
    let mut scheduler = LRScheduler::new(cfg.warmup_steps, cfg.steps, cfg.max_lr, cfg.min_lr);

    // 断点续训：恢复参数 / 优化器 / 步数 / best loss
    let (mut start_step, mut best_val_loss) = (0usize, f32::INFINITY);
    if let Some(path) = resume_from {
        let ckpt: Checkpoint = checkpoint::load_with_opt(path, model, &mut opt);
        start_step = ckpt.step;
        best_val_loss = ckpt.best_val_loss;
        scheduler.set_step(start_step);
        println!(
            "已从 {path} 恢复：step={}，best_val_loss={:.4}",
            start_step, best_val_loss
        );
    }

    let block_size = loader.block_size();
    let batch_size = loader.batch_size();
    let param_count: usize = params.iter().map(|p| p.numel()).sum();
    println!(
        "开始训练：{}（vocab={}）模型参数 {} | 语料 {} tokens（训练 {} / 验证 {}）| batch={} block={}",
        tokenizer.kind(),
        model.cfg.vocab_size,
        param_count,
        loader.num_tokens(),
        loader.num_train_tokens(),
        loader.num_val_tokens(),
        batch_size,
        block_size,
    );

    let mut eval_rng = Rng::new(cfg.seed); // 固定种子，评估结果可复现
    let mut final_loss = f32::INFINITY;
    let diag_t0 = std::time::Instant::now(); // [诊断] 每步耗时
    // [诊断] 分段计时累计（forward / backward / opt / 采样+杂项）
    let (mut s_fw, mut s_bw, mut s_opt, mut s_misc) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    for step in start_step..cfg.steps {
        let s_t0 = std::time::Instant::now();
        // 1. 采样 batch
        let (x, y) = loader.sample_batch(rng);

        // 2. 前向 + 损失
        let logits = model.forward(&x, batch_size, block_size, None);
        let loss = cross_entropy_loss(&logits, &y);
        s_fw += s_t0.elapsed().as_secs_f64();

        // 3. 反向
        loss.backward();
        s_bw += s_t0.elapsed().as_secs_f64();

        // 4. 梯度裁剪
        clip_grad_norm(&params, cfg.grad_clip);

        // 5. 更新参数（设置当前学习率）
        let cur_lr = scheduler.lr(); // 先取当前步的学习率（scheduler.step() 之后会变成下一步的）
        opt.lr = cur_lr;
        opt.step();
        s_opt += s_t0.elapsed().as_secs_f64();

        // 6. 清零梯度
        opt.zero_grad();
        scheduler.step();
        s_misc += s_t0.elapsed().as_secs_f64() - (s_fw + s_bw + s_opt);

        // [诊断] 每步打印耗时与 GPU/CPU 分流
        if (step + 1) % 1 == 0 {
            let n = (step + 1) as f64;
            println!(
                "[diag] step {} | 平均 {:.3}s/步 | fw {:.2}s | bw {:.2}s | opt {:.2}s | 其他 {:.2}s",
                step + 1,
                diag_t0.elapsed().as_secs_f64() / n,
                s_fw,
                s_bw,
                s_opt,
                s_misc
            );
            #[cfg(feature = "gpu")]
            {
                let (g, c) = crate::gpu::stats();
                println!(
                    "[diag]   matmul GPU {} / CPU {} | GPU {:.1} 次/步",
                    g,
                    c,
                    g as f64 / n
                );
            }
        }

        // 周期性评估 + 存 checkpoint
        let last = step + 1 == cfg.steps;
        if (step + 1) % cfg.eval_every == 0 || last {
            let val_loss = if loader.has_val() {
                Some(eval_loss(model, loader, cfg.eval_iters, &mut eval_rng))
            } else {
                None
            };
            // 仅在本次验证 loss 严格更优时刷新 best（同时避免用 f32 相等比较）
            let is_best = val_loss.is_some_and(|v| v < best_val_loss);
            if is_best {
                best_val_loss = val_loss.unwrap();
            }
            if let Some(dir) = out_dir {
                std::fs::create_dir_all(dir).expect("创建 checkpoint 目录失败");
                checkpoint::save(
                    &format!("{dir}/latest.ckpt"),
                    model,
                    &opt,
                    step + 1,
                    best_val_loss,
                );
                if is_best && best_val_loss.is_finite() {
                    checkpoint::save(
                        &format!("{dir}/best.ckpt"),
                        model,
                        &opt,
                        step + 1,
                        best_val_loss,
                    );
                }
            }
            match val_loss {
                Some(v) => println!(
                    "step {:>5} | lr {:.6} | train_loss {:.4} | val_loss {:.4} | ppl {:.2}",
                    step + 1,
                    cur_lr,
                    loss.item(),
                    v,
                    v.exp()
                ),
                None => println!(
                    "step {:>5} | lr {:.6} | train_loss {:.4}",
                    step + 1,
                    cur_lr,
                    loss.item()
                ),
            }
        }
        final_loss = loss.item();
    }

    if let Some(dir) = out_dir {
        checkpoint::save(
            &format!("{dir}/final.ckpt"),
            model,
            &opt,
            cfg.steps,
            best_val_loss,
        );
        println!("训练完成，checkpoint 已保存到 {dir}/（latest / best / final）");
    }
    #[cfg(feature = "gpu")]
    {
        let (gpu_calls, cpu_calls) = crate::gpu::stats();
        println!(
            "matmul 分流统计：GPU {gpu_calls} 次 / CPU {cpu_calls} 次（小矩阵走 CPU 更划算，GPU 只负责足够大的矩阵乘）"
        );
    }
    if best_val_loss.is_finite() {
        best_val_loss
    } else {
        final_loss
    }
}
