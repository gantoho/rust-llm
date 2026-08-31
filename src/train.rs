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

use crate::data::DataLoader;
use crate::loss::cross_entropy_loss;
use crate::model::GPT;
use crate::module::Module;
use crate::optim::AdamW;
use crate::rng::Rng;
use crate::tokenizer::CharTokenizer;
use crate::tensor::Tensor;

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
}

/// 梯度裁剪：如果所有参数梯度的总范数超过 max_norm，就整体等比缩放。
/// 防止个别大梯度把参数"推飞"，这是训练 LLM 的标准防爆措施。
pub fn clip_grad_norm(params: &[Tensor], max_norm: f32) {
    let mut total = 0.0f32;
    for p in params {
        let g = p.grad();
        for &v in &g {
            total += v * v;
        }
    }
    let norm = total.sqrt();
    if norm > max_norm {
        let scale = max_norm / norm;
        for p in params {
            let g = p.grad();
            let scaled: Vec<f32> = g.iter().map(|&v| v * scale).collect();
            p.grad_set(scaled);
        }
    }
}

/// 训练函数：在语料上训练一个小 GPT
///
/// 返回训练好的模型（其实直接原地修改了）。
pub fn train_gpt(
    model: &GPT,
    tokenizer: &CharTokenizer,
    loader: &DataLoader,
    steps: usize,
    batch_size: usize,
    max_lr: f32,
    warmup_steps: usize,
    eval_every: usize,
    rng: &mut Rng,
) {
    let params = model.parameters();
    let mut opt = AdamW::new(max_lr, params.clone(), 0.01);
    let mut scheduler = LRScheduler::new(warmup_steps, steps, max_lr, max_lr * 0.1);

    println!("训练参数数量：{}", params.iter().map(|p| p.numel()).sum::<usize>());

    let block_size = model.cfg.block_size;
    for step in 0..steps {
        // 1. 采样 batch
        let (x, y) = loader.sample_batch(rng);
        let b = batch_size;
        let t = block_size;

        // 2. 前向 + 损失
        let logits = model.forward(&x, b, t, None);
        let loss = cross_entropy_loss(&logits, &y);

        // 3. 反向
        loss.backward();

        // 4. 梯度裁剪
        clip_grad_norm(&params, 1.0);

        // 5. 更新参数（设置当前学习率）
        opt.lr = scheduler.lr();
        opt.step();

        // 6. 清零梯度
        opt.zero_grad();
        scheduler.step();

        if step % eval_every == 0 || step == steps - 1 {
            println!(
                "step {:>5} | lr {:.5} | loss {:.4}",
                step,
                scheduler.lr(),
                loss.data()[0]
            );
        }
    }
}
