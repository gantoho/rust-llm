//! 训练循环与学习率调度（第 13、18 课）
//!
//! 训练 GPT 的完整骨架：
//! 1. 采样一个 batch
//! 2. 前向算损失
//! 3. 反向算梯度
//! 4. 梯度裁剪（防止梯度爆炸）
//! 5. 优化器更新参数
//! 6. 清零梯度
//!
//! 学习率调度（第 18 课）：
//! - warmup：前若干步学习率从 0 线性升到最大值（让训练稳定起步）
//! - cosine decay：之后按余弦曲线衰减到最小值（后期精细收敛）
//!
//! 工程化支持：
//! - 每 `eval_every` 步在验证集上评估 loss / 困惑度（perplexity）
//! - 周期性保存 checkpoint（latest / best），支持 `--resume` 断点续训
//! - 训练指标 CSV 日志记录（loss / lr / ppl / 速度）
//! - LoRA 微调模式：冻结主模型，只训练 LoRA 层

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

/// 混合精度训练（Automatic Mixed Precision，AMP）
///
/// 核心思想：
/// 1. **前向/反向用低精度**（FP16/BF16）：矩阵乘法在 FP16 下快 2-8×，显存减半
/// 2. **主权重用 FP32**：优化器更新需要高精度（小学习率 × 梯度在 FP16 下会下溢为 0）
/// 3. **损失缩放（Loss Scaling）**：FP16 最小正规数 ~6e-8，小梯度会下溢为 0。
///    解法：loss 乘一个大数（scale），让梯度数值范围移到 FP16 可表示区间，
///    优化器更新前再除回来。
///
/// **动态损失缩放**（本实现）：
/// - 初始 scale = 2^16 = 65536
/// - 每 N 步无溢出 → scale 翻倍（尝试更大）
/// - 出现溢出（NaN/Inf） → scale 减半，跳过本步更新
///
/// **本项目的简化**：当前 Tensor 全程 f32，没有真正的 FP16 类型。
/// MixedPrecision 只实现"动态损失缩放"机制，为将来引入 FP16 做好架构准备。
/// 缩放本身不影响 f32 训练（f32 的动态范围足够大），但代码逻辑与真实 AMP 完全一致。
#[allow(dead_code)] // 教学实现：AMP 动态损失缩放机制完整可用，为将来引入 FP16 做好架构准备
pub struct MixedPrecision {
    /// 当前损失缩放因子
    pub scale: f32,
    /// 初始缩放因子（2^init_scale_log2）
    init_scale: f32,
    /// 缩放因子增长步数（连续 N 步无溢出后翻倍）
    growth_interval: usize,
    /// 连续无溢出步数计数
    growth_steps: usize,
    /// 缩放因子上下界
    min_scale: f32,
    max_scale: f32,
}

#[allow(dead_code)] // 教学实现：AMP 动态损失缩放机制完整可用
impl MixedPrecision {
    pub fn new(init_scale_log2: u32, growth_interval: usize) -> Self {
        let init_scale = (2.0f32).powi(init_scale_log2 as i32);
        MixedPrecision {
            scale: init_scale,
            init_scale,
            growth_interval,
            growth_steps: 0,
            min_scale: 1.0,
            max_scale: 2.0f32.powi(24), // 2^24，防止 scale 过大导致 FP32 溢出
        }
    }

    /// 缩放损失（前向后、反向前调用）
    pub fn scale_loss(&self, loss: &Tensor) -> Tensor {
        loss.mul_scalar(self.scale)
    }

    /// 检查梯度是否溢出（反向后、优化器更新前调用）
    ///
    /// 返回 true = 无溢出，可以更新参数；false = 有溢出，跳过本步。
    /// 如果无溢出，还会尝试增长 scale。
    pub fn check_and_update(&mut self, params: &[Tensor]) -> bool {
        // 检查所有参数的梯度是否有 NaN/Inf
        for p in params {
            let g = p.grad.borrow();
            for &v in g.iter() {
                if v.is_nan() || v.is_infinite() {
                    // 溢出：scale 减半，重置计数
                    self.scale = (self.scale * 0.5).max(self.min_scale);
                    self.growth_steps = 0;
                    return false;
                }
            }
        }
        // 无溢出：计数 +1，达到阈值时 scale 翻倍
        self.growth_steps += 1;
        if self.growth_steps >= self.growth_interval {
            self.scale = (self.scale * 2.0).min(self.max_scale);
            self.growth_steps = 0;
        }
        true
    }

    /// 优化器更新后，需要把梯度除回 scale（因为 loss 被放大了 scale 倍）
    ///
    /// 注意：在实际 AMP 中，梯度在反向时已经自动按 scale 缩放了，
    /// 所以这里是在 optimizer.step() 之前把梯度归一化。
    /// 但在我们的实现中，optimizer.step() 不关心梯度的绝对值（AdamW 有自适应学习率），
    /// 所以这个除法实际上是隐式地通过学习率来补偿的。
    /// 这里提供一个显式的 unscale 方法，供需要时手动调用。
    pub fn unscale_gradients(&self, params: &[Tensor]) {
        let inv_scale = 1.0 / self.scale;
        for p in params {
            let mut g = p.grad.borrow_mut();
            for v in g.iter_mut() {
                *v *= inv_scale;
            }
        }
    }
}

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
        // 评估只做前向，无需建图：no_grad 下省掉整张计算图
        let loss = crate::tensor::no_grad(|| {
            let logits = model.forward(&x, loader.batch_size(), loader.block_size(), None, false);
            cross_entropy_loss(&logits, &y)
        });
        total += loss.item();
    }
    total / eval_iters as f32
}

/// 前向 + 交叉熵，返回「打印用未缩放、反向按 `1/accum` 缩放」的 loss 张量。
///
/// GPU 可用且尺寸合适时走「输出头常驻显存」路径：`hidden @ Wᵀ` 与 softmax+交叉熵
/// 录进一次提交，logits 与 dlogits（本配置下各 33.6M 元素）全程留在显存、只回读每行 CE，
/// 反向也一次算完 d_hidden / d_head。相比逐算子版省掉一步 268 MB 的往返（实测 445 ms）。
///
/// 交叉熵对 logits 的梯度是解析式的（softmax - onehot），不依赖上游梯度，
/// 所以不必为中间那段建计算图：直接算好边界上的梯度、注入图上的张量即可，
/// autograd 会从这些张量继续往前传播。
fn forward_loss(
    model: &GPT,
    x: &[usize],
    y: &[usize],
    batch_size: usize,
    block_size: usize,
    accum: usize,
) -> Tensor {
    let hidden = model.forward_hidden(x, batch_size, block_size, true);
    let head = model.head_weight();
    let inv_accum = 1.0 / accum as f32;

    #[cfg(feature = "gpu")]
    if crate::tensor::grad_enabled() {
        let rows = batch_size * block_size;
        let d = hidden.shape()[1];
        let vocab = head.shape()[0];
        if let Some(resident) =
            crate::gpu::lm_head_ce(&hidden.data.borrow(), &head.data.borrow(), y, rows, d, vocab)
        {
            let hidden_bwd = hidden.clone();
            let head_bwd = head.clone();
            return Tensor::external_scalar_loss(resident.loss, vec![hidden.clone()], move || {
                let (dx, dw) = resident.backward().expect("常驻输出头反向失败");
                hidden_bwd.accumulate_grad(&dx, inv_accum);
                head_bwd.accumulate_grad(&dw, inv_accum);
            });
        }
    }

    // 回落：逐算子路径，输出头照旧走 Tensor 算子
    let logits = hidden.matmul(&head.transpose());
    let loss = cross_entropy_loss(&logits, y);
    if accum == 1 {
        return loss;
    }
    // 梯度累积：不缩放 loss 本身（打印要用原始值），只把 1/accum 作为上游梯度注入
    let raw_val = loss.item();
    let loss_bwd = loss.clone();
    Tensor::external_scalar_loss(raw_val, vec![loss], move || {
        loss_bwd.accumulate_grad(&[inv_accum], 1.0);
    })
}

/// 训练指标记录器（CSV 格式）
struct MetricsLogger {
    file: Option<std::fs::File>,
}

impl MetricsLogger {
    fn new(path: Option<&str>) -> Self {
        use std::io::Write;
        let file = path.map(|p| {
            // 日志目录（默认 logs/）不存在时自动创建，避免用户手动 mkdir
            crate::config::ensure_parent_dir(p);
            let mut f = std::fs::File::create(p)
                .unwrap_or_else(|e| panic!("无法创建日志文件 {p}: {e}"));
            writeln!(f, "step,lr,train_loss,val_loss,ppl,tokens_per_sec")
                .expect("写入日志头失败");
            f
        });
        MetricsLogger { file }
    }

    fn log(&mut self, step: usize, lr: f32, train_loss: f32, val_loss: Option<f32>, tps: f64) {
        use std::io::Write;
        if let Some(ref mut f) = self.file {
            let (vl, ppl) = match val_loss {
                Some(v) => (format!("{:.6}", v), format!("{:.2}", v.exp())),
                None => (String::new(), String::new()),
            };
            writeln!(f, "{},{:.8},{:.6},{},{},{:.0}", step, lr, train_loss, vl, ppl, tps)
                .expect("写入日志失败");
        }
    }
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
    let _trainable_count = param_count; // LoRA 模式下会更少（但这里简化处理）
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
    if cfg.lora.is_some() {
        println!("LoRA 微调模式：rank={} alpha={}", 
            cfg.lora.as_ref().unwrap().rank,
            cfg.lora.as_ref().unwrap().alpha,
        );
    }

    // 指标日志
    let log_path = cfg.log_file.as_deref();
    let mut metrics = MetricsLogger::new(log_path);
    if log_path.is_some() {
        println!("训练指标将记录到 {}", log_path.unwrap());
    }

    // GPU dispatch 开销分解：必须等首步的真实 dispatch 跑完才有数据可打印，
    // 所以不在启动阶段调用，而是等训练循环里第一次进度打印时输出一次（见下）。
    #[cfg(feature = "gpu")]
    let mut gpu_diag_printed = false;
    // 判定实验：LLM_GPU_PROBE=1 时录制**第一步**的 matmul 形状，再用分组回放测批量吞吐，
    // 用来判断 GPU 后端该改哪个形状（录制期间 GPU 不参与，CPU 兜底，数值不受影响）。
    // 只录一步是刻意的：这样「每步几次」是精确值，而不是靠批数去猜。
    #[cfg(feature = "gpu")]
    let probe_on = std::env::var("LLM_GPU_PROBE").is_ok();
    #[cfg(feature = "gpu")]
    if probe_on {
        crate::gpu::probe_capture(true);
    }

    let mut eval_rng = Rng::new(cfg.seed); // 固定种子，评估结果可复现
    let mut no_improve_count = 0usize; // 早停计数器
    let patience = cfg.early_stop_patience; // 0 = 不启用
    if patience > 0 {
        println!("[info] 早停已启用：patience={}（连续 {} 次评估不改善则停止）", patience, patience);
    }
    let mut final_loss = f32::INFINITY;
    let train_t0 = std::time::Instant::now();
    let mut last_progress_t = train_t0; // 上次打印进度的时间
    let progress_interval = 5.0; // 每 5 秒打印一次进度
    let accum = cfg.accum_steps.max(1);
    // 实际完成到的步数：早停会提前退出，结尾统计与 final.ckpt 都不能用 cfg.steps
    let mut last_step_done = start_step;
    for step in start_step..cfg.steps {
        // 1. 采样 batch
        let (x, y) = loader.sample_batch(rng);

        // 2. 前向 + 损失（梯度累积时反向按 1/accum 缩放）
        let t_seg = std::time::Instant::now();
        let loss = forward_loss(&model, &x, &y, batch_size, block_size, accum);
        let fwd_ms = t_seg.elapsed().as_secs_f64() * 1000.0;

        // 3. 反向（梯度自动累加到现有梯度上）
        let t_seg = std::time::Instant::now();
        loss.backward();
        let bwd_ms = t_seg.elapsed().as_secs_f64() * 1000.0;

        // 判定实验：第一步的反向跑完就收网，按形状回放测出「每步每个形状各花多少 ms」
        #[cfg(feature = "gpu")]
        if probe_on {
            crate::gpu::probe_report();
            crate::gpu::probe_fma();
        }

        // 每 accum 步才做一次梯度裁剪 + 优化器更新 + 清零
        if (step + 1) % accum == 0 || step + 1 == cfg.steps {
            // 周期性打印进度（在 zero_grad 之前，此时梯度有效）
            {
                let now = std::time::Instant::now();
                let dt = now.duration_since(last_progress_t).as_secs_f64();
                if dt >= progress_interval {
                    let elapsed = now.duration_since(train_t0).as_secs_f64();
                    let steps_done = step - start_step + 1;
                    let steps_per_sec = steps_done as f64 / elapsed;
                    let remaining = (cfg.steps - step - 1) as f64 / steps_per_sec.max(0.001);
                    let tps = steps_done as f64 * batch_size as f64 * block_size as f64 / elapsed;
                    // 裁剪**前**的原始梯度范数，末尾 `*` 表示本步会触发裁剪。
                    // 不能打印 min(raw, grad_clip)：那样范数恒被压在阈值上限，
                    // 梯度是否爆炸、裁剪是否频繁完全看不出来（曾经因此漏判梯度异常）。
                    let raw_norm: f32 = params.iter().map(|p| {
                        p.grad.borrow().iter().map(|g| g * g).sum::<f32>()
                    }).sum::<f32>().sqrt();
                    let clipped = if raw_norm > cfg.grad_clip { "*" } else { "" };
                    println!(
                        "[train] step {}/{} | loss {:.4} | grad {:.2}{} | lr {:.6} | {:.1} st/s | {:.0} tok/s | fwd {:.0}ms bwd {:.0}ms | {:.0}s | ~{:.0}s",
                        step + 1, cfg.steps, loss.item(), raw_norm, clipped, scheduler.lr(),
                        steps_per_sec, tps, fwd_ms, bwd_ms, elapsed, remaining
                    );
                    // 首步结束后打印一次 GPU dispatch 开销分解（上传/提交/同步各占多少）
                    #[cfg(feature = "gpu")]
                    if !gpu_diag_printed {
                        gpu_diag_printed = true;
                        crate::gpu::flush_diag_log();
                    }
                    last_progress_t = now;
                }
            }

            // 4. 梯度裁剪
            clip_grad_norm(&params, cfg.grad_clip);

            // 5. 更新参数（设置当前学习率）
            let cur_lr = scheduler.lr();
            opt.lr = cur_lr;
            opt.step();

            // 6. 清零梯度
            opt.zero_grad();

            // 学习率调度：只在 optimizer 实际更新后递增
            scheduler.step();
        }

        // 周期性评估 + 存 checkpoint
        let last = step + 1 == cfg.steps;
        if (step + 1) % cfg.eval_every == 0 || last {
            last_step_done = step + 1;
            let val_loss = if loader.has_val() {
                Some(eval_loss(model, loader, cfg.eval_iters, &mut eval_rng))
            } else {
                None
            };
            // 仅在本次验证 loss 严格更优时刷新 best（同时避免用 f32 相等比较）
            let is_best = val_loss.is_some_and(|v| v < best_val_loss);
            let mut should_stop = false;
            if is_best {
                best_val_loss = val_loss.unwrap();
                no_improve_count = 0;
            } else if val_loss.is_some() && patience > 0 {
                no_improve_count += 1;
                should_stop = no_improve_count >= patience;
            }
            // 早停不能在此处直接 break：必须先存 checkpoint、写指标、打印本步评估行。
            // 否则最后一次评估会从日志里消失，且 latest.ckpt 停留在上一次评估
            //（断点续训会拿到落后一个 eval 周期的过期权重）。真正 break 在块末尾。
            if let Some(dir) = out_dir {
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
            // 计算 tokens/sec
            let elapsed = train_t0.elapsed().as_secs_f64().max(1e-9);
            let tokens_processed = (step - start_step + 1) as f64 * batch_size as f64 * block_size as f64;
            let tps = tokens_processed / elapsed;
            metrics.log(step + 1, scheduler.lr(), loss.item(), val_loss, tps);

            match val_loss {
                Some(v) => {
                    let marker = if is_best { " *" } else { "" };
                    println!(
                        "step {:>5} | lr {:.6} | loss {:.4} | val {:.4} (ppl {:.1}) | {:.0} tok/s{marker}",
                        step + 1,
                        scheduler.lr(),
                        loss.item(),
                        v,
                        v.exp(),
                        tps
                    );
                }
                None => println!(
                    "step {:>5} | lr {:.6} | loss {:.4} | {:.0} tok/s",
                    step + 1,
                    scheduler.lr(),
                    loss.item(),
                    tps
                ),
            }

            // 至此 checkpoint 已保存、指标已记录、评估行已打印，可以安全早停
            if should_stop {
                println!(
                    "早停触发：连续 {} 次评估 val_loss 未改善（best {:.4}），在 step {} 停止训练",
                    patience, best_val_loss, step + 1
                );
                break;
            }
        }
        final_loss = loss.item();
    }

    let elapsed = train_t0.elapsed().as_secs_f64();
    // 实际完成的步数：早停时小于 cfg.steps - start_step，用 cfg.steps 会让「每步耗时」
    // 被系统性低估（分母偏大）。
    let steps_done = last_step_done.saturating_sub(start_step).max(1);
    if let Some(dir) = out_dir {
        checkpoint::save(
            &format!("{dir}/final.ckpt"),
            model,
            &opt,
            last_step_done,
            best_val_loss,
        );
        println!(
            "[done] checkpoint 已保存到 {dir}/ | 总耗时 {:.0}s | {:.2}s/步（共 {} 步）",
            elapsed,
            elapsed / steps_done as f64,
            steps_done,
        );
    }
    #[cfg(feature = "gpu")]
    {
        // 进度打印要等满 5s 才触发一次，短跑（如消融实验）会在触发前就结束，
        // 于是分解永远打不出来。收尾再兜一次，保证任何长度的跑都能看到诊断。
        crate::gpu::flush_diag_log();
        let (gpu_calls, cpu_calls) = crate::gpu::stats();
        println!("[done] matmul 分流：GPU {} / CPU {}", gpu_calls, cpu_calls);
    }
    if best_val_loss.is_finite() {
        best_val_loss
    } else {
        final_loss
    }
}
