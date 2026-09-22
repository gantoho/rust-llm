//! 训练循环与学习率调度（第 13、18 课）
//!
//! 训练 GPT 的完整骨架：
//! 1. 采样一个 batch
//! 2. 前向算损失
//! 3. 反向算梯度（开启 AMP 时先把 loss 乘上 scale）
//! 4. 梯度裁剪（防止梯度爆炸）
//! 5. 优化器更新参数
//! 6. 清零梯度
//!
//! 开启 `train.amp` 后，第 3 步与第 4 步之间会插入动态损失缩放的两步：**溢出检查**
//! （梯度含 Inf/NaN 就丢弃本步、不更新参数）与**梯度反缩放**（把梯度除回真实尺度，
//! 保证裁剪阈值仍然作用在真实梯度上），见 [`MixedPrecision`]。
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
use crate::data::BatchSource;
use crate::loss::cross_entropy_loss_masked;
use crate::model::GPT;
use crate::module::{Module, zero_grad_all};
use crate::optim::{AdamW, Optimizer};
use crate::rng::Rng;
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use crate::{checkpoint, checkpoint::Checkpoint};

/// 混合精度训练（Automatic Mixed Precision，AMP）的损失缩放机制
///
/// 核心思想：
/// 1. **前向/反向用低精度**（FP16/BF16）：矩阵乘法在 FP16 下快 2-8×，显存减半
/// 2. **主权重用 FP32**：优化器更新需要高精度（小学习率 × 梯度在 FP16 下会下溢为 0）
/// 3. **损失缩放（Loss Scaling）**：FP16 最小正规数 ~6e-8，小梯度会下溢为 0。
///    解法：loss 乘一个大数（scale），让梯度数值范围移到可表示区间，
///    优化器更新前再除回来。
///
/// **动态损失缩放**（本实现）：
/// - 初始 scale = 2^init_scale_log2（默认 2^16 = 65536）
/// - 每 `growth_interval` 步无溢出 → scale 翻倍（试探更大的可表示区间）
/// - 出现溢出（NaN/Inf） → scale 减半，跳过本步更新
///
/// **在 Tensor 全程 f32 的本项目里，它起什么作用**：
/// - scale 恒为 2 的幂，f32 下乘以/除以 2^k 是精确的指数移位，整条反向链路逐位
///   等于未缩放的结果——所以损失缩放本身不改变数值，不会污染可复现性。
/// - 真正被改变的是**溢出保护**：某一步梯度爆成 Inf/NaN 时跳过本步更新
///   （否则 AdamW 的一阶/二阶矩被 Inf 污染，之后整个训练永久变成 NaN），
///   并让 scale 自动收缩去适配当前的数值范围。
/// - **反缩放是正确性的刚需**：梯度裁剪必须作用在真实尺度的梯度上。若带着 scale
///   去裁剪，阈值会被放大 2^16 倍而形同虚设，裁剪日志也会读出无意义的范数。
///
/// FP16/BF16 的存储与算力收益要求 `Tensor` 引入 dtype 维度（每个算子都要有半精度
/// 实现），那是独立工程（见 docs/26-混合精度训练.md）；而损失缩放、溢出跳步、
/// 梯度反缩放这三件事并不依赖半精度类型，本模块把它们完整落在训练循环里。
pub struct MixedPrecision {
    /// 当前损失缩放因子
    pub scale: f32,
    /// 缩放因子增长步数（连续 N 步无溢出后翻倍）
    growth_interval: usize,
    /// 连续无溢出步数计数
    growth_steps: usize,
    /// 缩放因子上下界
    min_scale: f32,
    max_scale: f32,
}

impl MixedPrecision {
    pub fn new(init_scale_log2: u32, growth_interval: usize) -> Self {
        MixedPrecision {
            scale: (2.0f32).powi(init_scale_log2 as i32),
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

    /// 把梯度除回真实尺度：溢出检查通过后、梯度裁剪**之前**调用
    ///
    /// 前向时 loss 被乘了 `scale`，反向出来的所有梯度都带着同一个因子，必须在更新参数前
    /// 除掉。少这一步不会让 AdamW 的更新方向出错（它按梯度自身的尺度自适应调步长），
    /// 但会有两处实打实的错误：
    /// - 梯度裁剪的阈值被整体放大 `scale` 倍，裁剪等于失效（本项目默认 grad_clip 很大，
    ///   看着"没坏"，换个正常阈值立刻暴露）；
    /// - 进度日志里的梯度范数是假的，看不出真实量级，梯度异常时无法察觉。
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
pub fn eval_loss(model: &GPT, loader: &dyn BatchSource, eval_iters: usize, rng: &mut Rng) -> f32 {
    zero_grad_all(model); // 评估前清零梯度，避免残留影响
    let mut total = 0.0f32;
    for _ in 0..eval_iters {
        let (x, y, mask) = loader.eval_batch(rng);
        // 评估只做前向，无需建图：no_grad 下省掉整张计算图
        let loss = crate::tensor::no_grad(|| {
            let logits = model.forward(&x, loader.batch_size(), loader.block_size(), None, false);
            cross_entropy_loss_masked(&logits, &y, mask.as_deref())
        });
        total += loss.item();
    }
    total / eval_iters as f32
}

/// 前向 + 交叉熵 + MoE 均衡辅助损失，返回「打印用未缩放、反向按 `1/accum` 缩放」的
/// 总损失张量。
///
/// 总损失 = 交叉熵 + Σ_layers α·L_aux（最后一项只在 MoE 配置下存在，见第 32 课）。
/// 辅助损失必须**跟着主损失一起反向**，否则各层路由器的梯度只能是 0——它唯一的职责
/// 就是把负载推平，主损失本身对路由偏斜是无感的（谁被选中都行，只要选中的专家算得准）。
///
/// 两条路径返回的 loss 在数值上都是**未缩放**的原始值，缩放只作用在梯度上：
/// 逐算子路径靠 [`Tensor::external_scalar_loss`] 注入上游梯度，常驻显存路径
/// 在同一个节点里把 `1/accum` 与上游梯度相乘，因此梯度累积与 AMP 的 `scale`
/// 可以任意叠加，不会丢比例。辅助损失同样只缩放梯度、不改数值，
/// 否则打印出来的 loss 会随 `accum` 变小。
///
/// GPU 可用且尺寸合适时走「输出头常驻显存」路径：`hidden @ Wᵀ` 与 softmax+交叉熵
/// 录进一次提交，logits 与 dlogits（本配置下各 33.6M 元素）全程留在显存、只回读每行 CE，
/// 反向也一次算完 d_hidden / d_head。相比逐算子版省掉一步 268 MB 的往返（实测 445 ms）。
///
/// 交叉熵对 logits 的梯度是解析式的（softmax - onehot），不依赖上游梯度，
/// 所以不必为中间那段建计算图：直接算好边界上的梯度、注入图上的张量即可，
/// autograd 会从这些张量继续往前传播。
///
/// `mask` 为 SFT 的 loss 掩码（`None` = 全位置参与）。带掩码时**不走**常驻显存路径：
/// 那条路径的 GPU 交叉熵核不接受逐位置权重，硬走会悄悄把掩码丢掉，
/// 变成"以为在按回答算 loss、其实整个窗口都在算"。宁可慢一点，也不能错。
fn forward_loss(
    model: &GPT,
    x: &[usize],
    y: &[usize],
    mask: Option<&[bool]>,
    batch_size: usize,
    block_size: usize,
    accum: usize,
) -> Tensor {
    let hidden = model.forward_hidden(x, batch_size, block_size, true);
    // 必须在 forward 之后**立刻**取走：每个 Block 只保留"最近一次前向"的那份辅助损失，
    // 下一次前向会覆盖它（见 [`crate::model::GPT::aux_loss`]）。
    let aux = model.aux_loss();
    let base = cross_entropy_with_head(
        model,
        &hidden,
        y,
        mask,
        batch_size,
        block_size,
        accum,
    );
    match aux {
        None => base,
        Some(a) => base.add(&scale_grad_only(a, 1.0 / accum as f32)),
    }
}

/// 把标量张量的**梯度**乘以 `factor`（值保持不变，打印用）。
///
/// 梯度累积要求 `(1/accum)·Σ_m (L_ce,m + α·L_aux,m)`，两项都得除 `accum`；
/// 而返回值又要保持未缩放供打印。于是与 [`forward_loss`] 里处理交叉熵的手法一致：
/// 新节点只改上游梯度的注入比例，不动数值。
fn scale_grad_only(t: Tensor, factor: f32) -> Tensor {
    if factor == 1.0 {
        return t;
    }
    let v = t.item();
    let inner = t.clone();
    Tensor::external_scalar_loss(v, vec![t], move |upstream| {
        inner.accumulate_grad(&[factor * upstream], 1.0);
    })
}

/// 交叉熵部分（`hidden @ Wᵀ` + 掩码交叉熵），MoE 辅助损失由 [`forward_loss`] 另加。
///
/// `batch_size` / `block_size` 只被「输出头常驻显存」路径用来算行数，不带 GPU 时用不上。
#[cfg_attr(not(feature = "gpu"), allow(unused_variables))]
fn cross_entropy_with_head(
    model: &GPT,
    hidden: &Tensor,
    y: &[usize],
    mask: Option<&[bool]>,
    batch_size: usize,
    block_size: usize,
    accum: usize,
) -> Tensor {
    let head = model.head_weight();
    let inv_accum = 1.0 / accum as f32;
    let hidden = hidden.clone();

    #[cfg(feature = "gpu")]
    if mask.is_none() && crate::tensor::grad_enabled() {
        let rows = batch_size * block_size;
        let d = hidden.shape()[1];
        let vocab = head.shape()[0];
        if let Some(resident) =
            crate::gpu::lm_head_ce(&hidden.data.borrow(), &head.data.borrow(), y, rows, d, vocab)
        {
            let hidden_bwd = hidden.clone();
            let head_bwd = head.clone();
            return Tensor::external_scalar_loss(
                resident.loss,
                vec![hidden.clone()],
                move |upstream| {
                    let (dx, dw) = resident.backward().expect("常驻输出头反向失败");
                    // 融合核的反向按「上游梯度 = 1」算好，比例在这里补上：
                    // AMP 的 scale_loss 就是靠这一项才没被丢掉
                    hidden_bwd.accumulate_grad(&dx, upstream * inv_accum);
                    head_bwd.accumulate_grad(&dw, upstream * inv_accum);
                },
            );
        }
    }

    // 回落：逐算子路径，输出头照旧走 Tensor 算子
    let logits = hidden.matmul(&head.transpose());
    let loss = cross_entropy_loss_masked(&logits, y, mask);
    if accum == 1 {
        return loss;
    }
    // 梯度累积：不缩放 loss 本身（打印要用原始值），只把 1/accum 作为上游梯度注入
    let raw_val = loss.item();
    let loss_bwd = loss.clone();
    Tensor::external_scalar_loss(raw_val, vec![loss], move |upstream| {
        loss_bwd.accumulate_grad(&[inv_accum * upstream], 1.0);
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
/// `loader` 用 [`BatchSource`] 抽象，预训练（[`crate::data::DataLoader`]）与
/// SFT（[`crate::data::SftLoader`]）共用这段循环——差别只在采样出的批次带不带 loss 掩码。
///
/// - `out_dir = None` 时不保存 checkpoint（demo 用）
/// - `resume_from = Some(path)` 时从 checkpoint 续训（恢复参数、优化器状态与步数）
///
/// 返回最终的最优验证 loss（无验证集时为训练 loss 近似值）。
pub fn train_gpt(
    model: &GPT,
    tokenizer: &Tokenizer,
    loader: &dyn BatchSource,
    cfg: &TrainConfig,
    out_dir: Option<&str>,
    resume_from: Option<&str>,
    rng: &mut Rng,
) -> f32 {
    let params = model.parameters();
    // 真正参与训练的参数（LoRA 形态下 = 各层的适配层；普通训练 = 全部）。
    // 梯度裁剪与范数统计只该看这一组：冻结参数不产生梯度（前向走 matmul_frozen），
    // 但 GPU 常驻输出头快路仍会往共享词嵌入上注回梯度，那些值不该影响裁剪系数。
    let trainable = model.trainable_parameters();
    let mut opt = AdamW::new(cfg.max_lr, params.clone(), cfg.weight_decay);
    let mut scheduler = LRScheduler::new(cfg.warmup_steps, cfg.steps, cfg.max_lr, cfg.min_lr);

    // 断点续训：恢复参数 / 优化器 / 步数 / best loss
    let (mut start_step, mut best_val_loss) = (0usize, f32::INFINITY);
    if let Some(path) = resume_from {
        let ckpt: Checkpoint = checkpoint::load_with_opt(path, model, &mut opt);
        start_step = ckpt.step;
        best_val_loss = ckpt.best_val_loss;
        scheduler.set_step(start_step);
        logln!(
            "已从 {path} 恢复：step={}，best_val_loss={:.4}",
            start_step, best_val_loss
        );
    }

    let block_size = loader.block_size();
    let batch_size = loader.batch_size();
    let param_count: usize = params.iter().map(|p| p.numel()).sum();
    let trainable_count: usize = trainable.iter().map(|p| p.numel()).sum();
    logln!(
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
    // LoRA 形态：主干已冻结，只有适配层在学。这里报出真实占比，别让人靠猜。
    if let Some(lora) = model.lora.as_ref() {
        logln!(
            "LoRA：rank={} alpha={} 挂载={}｜可训练参数 {} / {}（{:.2}%）｜主干冻结：反向不算 dW、\
             优化器不更新、不吃权重衰减",
            lora.rank,
            lora.alpha,
            lora.targets,
            trainable_count,
            param_count,
            100.0 * trainable_count as f32 / param_count.max(1) as f32,
        );
    } else if let Some(lora) = cfg.lora.as_ref() {
        // 配置里带了 LoRA 段只说明"想用 LoRA"。`train` 子命令不注入适配层（没有基座可挂），
        // 真正干活的是 `finetune`。这里如实说清楚，别让人以为可训练参数量降下来了。
        logln!(
            "[warn] 配置里声明了 LoRA（rank={} alpha={}），但本次模型未注入适配层：\
             `train` 子命令不做注入，LoRA 微调请用 `finetune`（见 docs/29-LoRA低秩适配.md）。\
             下面训练的是**全部参数**",
            lora.rank,
            lora.alpha,
        );
    }

    // 指标日志
    let log_path = cfg.log_file.as_deref();
    let mut metrics = MetricsLogger::new(log_path);
    if log_path.is_some() {
        logln!("训练指标将记录到 {}", log_path.unwrap());
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
        logln!("[info] 早停已启用：patience={}（连续 {} 次评估不改善则停止）", patience, patience);
    }
    let mut final_loss = f32::INFINITY;
    let train_t0 = std::time::Instant::now();
    let mut last_progress_t = train_t0; // 上次打印进度的时间
    let progress_interval = 5.0; // 每 5 秒打印一次进度
    let accum = cfg.accum_steps.max(1);
    // AMP：动态损失缩放。None = 关闭（不缩放、不检查溢出）。
    let mut amp = if cfg.amp {
        let mp = MixedPrecision::new(cfg.amp_init_scale_log2, cfg.amp_growth_interval);
        logln!(
            "[info] AMP 已启用：初始 scale = 2^{} = {:.0}，每 {} 次无溢出翻倍，\
             溢出则跳过本步并把 scale 减半（梯度裁剪前会反缩放回真实尺度）",
            cfg.amp_init_scale_log2,
            mp.scale,
            cfg.amp_growth_interval
        );
        Some(mp)
    } else {
        None
    };
    let mut amp_skipped = 0usize; // 因梯度溢出被跳过的参数更新次数
    // 实际完成到的步数：早停会提前退出，结尾统计与 final.ckpt 都不能用 cfg.steps
    let mut last_step_done = start_step;
    for step in start_step..cfg.steps {
        // 1. 采样 batch（SFT 语料会额外带回 loss 掩码）
        let (x, y, mask) = loader.sample_batch(rng);

        // 2. 前向 + 损失（梯度累积时反向按 1/accum 缩放）
        let t_seg = std::time::Instant::now();
        let loss = forward_loss(&model, &x, &y, mask.as_deref(), batch_size, block_size, accum);
        let fwd_ms = t_seg.elapsed().as_secs_f64() * 1000.0;

        // 3. 反向（梯度自动累加到现有梯度上）
        let t_seg = std::time::Instant::now();
        match amp.as_ref() {
            // AMP：把 loss 乘上 scale 再反向，让整条反向链路的梯度落在更安全的数值区间
            Some(mp) => mp.scale_loss(&loss).backward(),
            None => loss.backward(),
        }
        let bwd_ms = t_seg.elapsed().as_secs_f64() * 1000.0;

        // 判定实验：第一步的反向跑完就收网，按形状回放测出「每步每个形状各花多少 ms」
        #[cfg(feature = "gpu")]
        if probe_on {
            crate::gpu::probe_report();
            crate::gpu::probe_fma();
        }

        // 每 accum 步才做一次梯度裁剪 + 优化器更新 + 清零
        if (step + 1) % accum == 0 || step + 1 == cfg.steps {
            // 4. AMP 溢出检查（此时梯度还带着 scale）
            // 梯度里出现 Inf/NaN 就跳过本次参数更新：不跳的话 AdamW 的一阶/二阶矩会被
            // Inf 污染，之后每一步都是 NaN，训练再也回不来。scale 同时自动收缩。
            let overflow = match amp.as_mut() {
                Some(mp) => !mp.check_and_update(&trainable),
                None => false,
            };
            if overflow {
                amp_skipped += 1;
                let scale = amp.as_ref().map(|m| m.scale).unwrap_or(0.0);
                logln!(
                    "[amp] step {} 梯度溢出（Inf/NaN）：跳过本次参数更新，scale 收缩到 {:.0}（累计跳过 {} 次）",
                    step + 1,
                    scale,
                    amp_skipped
                );
            } else if let Some(mp) = amp.as_ref() {
                // 5. AMP 反缩放：把梯度除回真实尺度，必须在裁剪之前
                mp.unscale_gradients(&trainable);
            }

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
                    let raw_norm: f32 = trainable.iter().map(|p| {
                        p.grad.borrow().iter().map(|g| g * g).sum::<f32>()
                    }).sum::<f32>().sqrt();
                    let clipped = if raw_norm > cfg.grad_clip { "*" } else { "" };
                    logln!(
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

            if overflow {
                // 坏梯度整批丢弃：不更新参数，也不推进学习率（本步没有真正发生）
                opt.zero_grad();
            } else {
                // 6. 梯度裁剪
                clip_grad_norm(&trainable, cfg.grad_clip);

                // 7. 更新参数（设置当前学习率）
                let cur_lr = scheduler.lr();
                opt.lr = cur_lr;
                opt.step();

                // 8. 清零梯度
                opt.zero_grad();

                // 学习率调度：只在 optimizer 实际更新后递增
                scheduler.step();
            }
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
                    logln!(
                        "step {:>5} | lr {:.6} | loss {:.4} | val {:.4} (ppl {:.1}) | {:.0} tok/s{marker}",
                        step + 1,
                        scheduler.lr(),
                        loss.item(),
                        v,
                        v.exp(),
                        tps
                    );
                }
                None => logln!(
                    "step {:>5} | lr {:.6} | loss {:.4} | {:.0} tok/s",
                    step + 1,
                    scheduler.lr(),
                    loss.item(),
                    tps
                ),
            }

            // 至此 checkpoint 已保存、指标已记录、评估行已打印，可以安全早停
            if should_stop {
                logln!(
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
        logln!(
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
        logln!("[done] matmul 分流：GPU {} / CPU {}", gpu_calls, cpu_calls);
    }
    if best_val_loss.is_finite() {
        if let Some(mp) = amp.as_ref() {
            logln!(
                "[done] AMP：最终 scale = {:.0}，累计因梯度溢出跳过 {} 次参数更新",
                mp.scale,
                amp_skipped
            );
        }
        best_val_loss
    } else {
        final_loss
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::model::{GPT, GPTConfig};
    use crate::tokenizer::Tokenizer;

    /// 跑一次极小的端到端训练（无验证集、不存 checkpoint），返回最终训练 loss。
    fn run_tiny_training(amp: bool) -> f32 {
        let corpus = "the quick brown fox jumps over the lazy dog, and then the dog jumps \
                      back over the quick brown fox again and again and again.";
        let tokenizer = Tokenizer::char(corpus);
        let mut rng = Rng::new(11);
        let model = GPT::new(GPTConfig::tiny(tokenizer.vocab_size()), &mut rng);
        let loader = DataLoader::new(corpus, &tokenizer, 16, 4);
        let tcfg = TrainConfig {
            seed: 11,
            steps: 12,
            batch_size: 4,
            warmup_steps: 2,
            max_lr: 1e-3,
            min_lr: 1e-3,
            // 正常量级的裁剪阈值：一旦漏掉反缩放，阈值会被整体放大 2^16 倍，裁剪失效，
            // 结果会立刻偏离——这条断言因此同时覆盖了「反缩放」这一步。
            grad_clip: 1.0,
            eval_every: 12, // 只在最后一步评估
            eval_iters: 1,
            log_file: None,
            amp,
            ..TrainConfig::default()
        };
        let mut train_rng = Rng::new(11);
        train_gpt(&model, &tokenizer, &loader, &tcfg, None, None, &mut train_rng)
    }

    /// AMP 的损失缩放对 f32 训练是数值透明的：`scale` 恒为 2 的幂，乘/除 2^k 在 f32 下
    /// 只是指数移位（精确），整条反向链路逐位等于未缩放时的结果。所以开与不开启 AMP
    /// 必须得到完全相同的最终 loss。
    #[test]
    fn test_amp_loss_scaling_is_numerically_transparent() {
        let plain = run_tiny_training(false);
        let with_amp = run_tiny_training(true);
        assert_eq!(
            plain, with_amp,
            "开/关 AMP 的最终 loss 应完全相同（无 AMP {plain}，有 AMP {with_amp}）"
        );
    }

    /// 动态损失缩放的两个方向 + 反缩放：无溢出时按间隔翻倍，检测到 Inf/NaN 时减半并要求
    /// 跳过本步；反缩放把梯度除回真实尺度。
    #[test]
    fn test_mixed_precision_grows_shrinks_and_unscales() {
        let mut mp = MixedPrecision::new(16, 3);
        assert_eq!(mp.scale, 65536.0, "初始 scale 应为 2^16");

        let clean = Tensor::from_vec(vec![1.0, 2.0], vec![2]);
        assert!(mp.check_and_update(&[clean.clone()]));
        assert!(mp.check_and_update(&[clean.clone()]));
        assert_eq!(mp.scale, 65536.0, "未到增长间隔（3 次）不应翻倍");
        assert!(mp.check_and_update(&[clean]));
        assert_eq!(mp.scale, 131072.0, "满 3 次无溢出应翻倍");

        let broken = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        *broken.grad.borrow_mut() = vec![1.0, f32::NAN];
        assert!(!mp.check_and_update(&[broken]), "含 NaN 的梯度应判为溢出");
        assert_eq!(mp.scale, 65536.0, "溢出后 scale 应减半");

        // 反缩放：梯度此刻带着 scale 倍，除回来才是真实梯度
        let g = Tensor::from_vec(vec![0.0, 0.0], vec![2]);
        *g.grad.borrow_mut() = vec![65536.0, -32768.0];
        mp.scale = 65536.0;
        mp.unscale_gradients(&[g.clone()]);
        assert_eq!(*g.grad.borrow(), vec![1.0, -0.5]);
    }
}
