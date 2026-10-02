//! 训练循环与学习率调度（第 13、18 课）
//!
//! 训练 Transformer 的完整骨架：
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
use crate::model::Transformer;
use crate::module::{Module, zero_grad_all};
use crate::optim::{AdamW, Muon, OptimizerState};
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
    /// - 梯度裁剪的阈值被整体放大 `scale` 倍，裁剪等于失效（若不 unscale，
    ///   默认 grad_clip = 1.0 形同虚设，看着"没坏"，换个更小阈值立刻暴露）；
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

/// 按 `train.optimizer` 构造优化器（`"adamw"` / `"muon"`），统一成 `Box<dyn OptimizerState>`。
///
/// - `"adamw"`（默认）：`AdamW::new(max_lr, params, weight_decay)`，与加 Muon 之前**完全一致**；
/// - `"muon"`：二维权重走 Muon，一维参数回退 AdamW（见 [`crate::optim::Muon`]）。
///   学习率仍由同一个调度器按 `max_lr` / `min_lr` 驱动，只是 Muon 需要更大的数值
///   （每元素更新量约 `lr/√cols`），拿 AdamW 的学习率直接套会几乎不动。
///
/// 抽出工厂是为了让 `train` / VQ-VAE 两条训练路径拿到同一套选择逻辑。
pub fn make_optimizer(cfg: &TrainConfig, params: Vec<Tensor>) -> Box<dyn OptimizerState> {
    match cfg.optimizer.as_str() {
        "muon" => Box::new(Muon::new(
            cfg.max_lr,
            params,
            cfg.weight_decay,
            cfg.muon_momentum,
            cfg.muon_ns_steps,
        )),
        _ => Box::new(AdamW::new(cfg.max_lr, params, cfg.weight_decay)),
    }
}

/// 学习率调度器。
///
/// 两条曲线（由 [`TrainConfig::lr_schedule`] 选择）共用同一段线性 warmup：
///
/// - **cosine**（默认）：warmup 之后全程按余弦曲线从 `max_lr` 降到 `min_lr`。
/// - **WSD**（warmup-stable-decay，MiniCPM / DeepSeek 式）：warmup 之后先**恒定**在 `max_lr`
///   走完稳定段，最后 `decay_steps` 步**线性**退火到 `min_lr`。
///
/// 为什么要 WSD：cosine 要求"总步数"从一开始就确定（曲线形状取决于 `total_steps`），
/// 中途想多训一会儿就得重排整条曲线；WSD 的稳定段是一段平线，任何时刻切出来退火都成立，
/// 于是"先训一个稳定版、需要时再快速退火收尾"成了常规操作（这也是 DeepSeek-V2/V3
/// 能在稳定段中途换数据配比继续训的原因）。
pub struct LRScheduler {
    warmup_steps: usize,
    total_steps: usize,
    max_lr: f32,
    min_lr: f32,
    /// `None` = cosine 调度；`Some(d)` = WSD，最后 `d` 步线性退火
    wsd_decay_steps: Option<usize>,
    step: usize,
}

impl LRScheduler {
    /// cosine 调度：warmup + 余弦衰减到 `min_lr`
    pub fn new(warmup_steps: usize, total_steps: usize, max_lr: f32, min_lr: f32) -> Self {
        Self::with_wsd(warmup_steps, total_steps, max_lr, min_lr, None)
    }

    /// WSD 调度：warmup + 恒定 + 末段 `decay_steps` 步线性退火
    pub fn new_wsd(
        warmup_steps: usize,
        total_steps: usize,
        max_lr: f32,
        min_lr: f32,
        decay_steps: usize,
    ) -> Self {
        Self::with_wsd(warmup_steps, total_steps, max_lr, min_lr, Some(decay_steps.max(1)))
    }

    fn with_wsd(
        warmup_steps: usize,
        total_steps: usize,
        max_lr: f32,
        min_lr: f32,
        wsd_decay_steps: Option<usize>,
    ) -> Self {
        LRScheduler {
            warmup_steps,
            total_steps,
            max_lr,
            min_lr,
            wsd_decay_steps,
            step: 0,
        }
    }

    /// 当前学习率
    pub fn lr(&self) -> f32 {
        if self.step < self.warmup_steps {
            // 线性 warmup（两条曲线共用，保证切到 WSD 不动前段）
            self.max_lr * (self.step as f32 + 1.0) / self.warmup_steps.max(1) as f32
        } else {
            match self.wsd_decay_steps {
                // cosine 衰减：从 max_lr 平滑降到 min_lr
                None => {
                    let progress = (self.step - self.warmup_steps) as f32
                        / (self.total_steps - self.warmup_steps).max(1) as f32;
                    let progress = progress.min(1.0);
                    let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
                    self.min_lr + (self.max_lr - self.min_lr) * cosine
                }
                // WSD：稳定段恒定在 max_lr，末段线性退火到 min_lr
                Some(decay_steps) => {
                    // 退火起点不能早于 warmup 结束（否则 warmup 一结束就在退火）
                    let decay_start = self
                        .total_steps
                        .saturating_sub(decay_steps)
                        .max(self.warmup_steps);
                    if self.step < decay_start {
                        self.max_lr
                    } else {
                        let progress = (self.step - decay_start) as f32
                            / (self.total_steps - decay_start).max(1) as f32;
                        let progress = progress.min(1.0);
                        self.max_lr + (self.min_lr - self.max_lr) * progress
                    }
                }
            }
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

/// 按 `train.lr_schedule` 构造调度器（`"cosine"` 或 `"wsd"`）。
///
/// WSD 的退火步数 = `round(wsd_decay_frac × (steps - warmup_steps))`，至少 1 步。
/// 抽出这个工厂是为了让 `train` / `sft` / `finetune` 几条训练入口拿到**同一条曲线**，
/// 不会出现"主训练用 WSD、微调悄悄退回 cosine"这种不一致。
pub fn lr_scheduler(cfg: &TrainConfig, total_steps: usize) -> LRScheduler {
    if cfg.lr_schedule == "wsd" {
        let span = total_steps.saturating_sub(cfg.warmup_steps);
        let decay = ((cfg.wsd_decay_frac * span as f32).round() as usize).max(1);
        LRScheduler::new_wsd(
            cfg.warmup_steps,
            total_steps,
            cfg.max_lr,
            cfg.min_lr,
            decay,
        )
    } else {
        LRScheduler::new(cfg.warmup_steps, total_steps, cfg.max_lr, cfg.min_lr)
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
pub fn eval_loss(model: &Transformer, loader: &dyn BatchSource, eval_iters: usize, rng: &mut Rng) -> f32 {
    zero_grad_all(model); // 评估前清零梯度，避免残留影响
    let mut total = 0.0f32;
    for _ in 0..eval_iters {
        let (x, y, mask, pixels) = loader.eval_batch_mm(rng);
        // 评估只做前向，无需建图：no_grad 下省掉整张计算图
        let loss = crate::tensor::no_grad(|| {
            let logits = model.forward_mm(
                &x,
                loader.batch_size(),
                loader.block_size(),
                pixels.as_deref(),
                None,
                false,
            );
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
    model: &Transformer,
    x: &[usize],
    y: &[usize],
    mask: Option<&[bool]>,
    pixels: Option<&[f32]>,
    batch_size: usize,
    block_size: usize,
    accum: usize,
) -> Tensor {
    let hidden = model.forward_hidden_mm(x, batch_size, block_size, pixels, true);
    // 必须在 forward 之后**立刻**取走：每个 Block 只保留"最近一次前向"的那份辅助损失，
    // 下一次前向会覆盖它（见 [`crate::model::Transformer::aux_loss`]）。
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
    model: &Transformer,
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
            crate::gpu::lm_head_ce(&hidden.decode(), &head.decode(), y, rows, d, vocab)
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
pub fn train_transformer(
    model: &Transformer,
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
    let mut opt = make_optimizer(cfg, params.clone());
    let mut scheduler = lr_scheduler(cfg, cfg.steps);

    // 断点续训：恢复参数 / 优化器 / 步数 / best loss
    let (mut start_step, mut best_val_loss) = (0usize, f32::INFINITY);
    if let Some(path) = resume_from {
        let ckpt: Checkpoint = checkpoint::load_with_opt(path, model, &mut *opt);
        start_step = ckpt.step;
        best_val_loss = ckpt.best_val_loss;
        scheduler.set_step(start_step);
        logln!(
            "已从 {path} 恢复：step={}，best_val_loss={:.4}",
            start_step, best_val_loss
        );
    }

    // bf16 混合精度（批次 10b）：建模 / 断点恢复完成后统一把参数缓冲原地转为
    // 真 u16 bf16 存储（内存减半）。原地换 Buffer 变体对所有克隆句柄可见——
    // 模型各层与优化器参数表拿的是同一批 Arc，无需逐个更新。
    // 必须放在 resume 之后：checkpoint 恢复按 f32 写入，顺序反了会盖掉转换。
    if cfg.bf16 {
        for p in &params {
            p.to_bf16();
        }
        logln!(
            "bf16 混合精度：{} 个参数张量转 u16 存储（内存减半）｜计算恒 f32｜AMP loss scaling {}",
            params.len(),
            if cfg.amp { "开启" } else { "关闭" },
        );
    }

    // FP8 权重存储模拟（第 43 课）：只做数值模拟，跑的还是 f32 算子，**没有加速**
    if cfg.fp8 {
        logln!(
            "FP8 权重存储模拟：每步更新后按 {} + 每 {} 个元素一个 scale 往返量化二维权重（仅数值模拟，无加速）",
            crate::fp8::TRAIN_FORMAT.name(),
            crate::fp8::TRAIN_BLOCK
        );
    }

    // 优化器提示：Muon 与 AdamW 的"等效步长"口径不同，混用学习率是最容易踩的坑。
    if cfg.optimizer == "muon" {
        logln!(
            "优化器：Muon（μ={}，Newton–Schulz {} 步）｜二维权重走 Muon，一维参数回退 AdamW｜\
             ⚠️ Muon 每元素更新量约 lr/√cols，学习率要比 AdamW 大 √cols 倍才等效；\
             本步的 lr 仍由 train.max_lr / min_lr 调度",
            cfg.muon_momentum,
            cfg.muon_ns_steps
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
    // —— 预取：让采样与前向/反向重叠 ——
    // 第一批在进入 scope 前先采（否则 spawn 后立刻 recv 会白等一次采样），工作线程
    // 从「第一批之后」的 rng 状态继续克隆采样，超前备好后续批次；每批附带采样后的
    // rng 状态，主线程消费时回写——采样序列与串行完全一致（可复现训练依赖这点）。
    let mut first_batch = (start_step < cfg.steps).then(|| loader.sample_batch_mm(rng));
    let worker_rng = rng.clone();
    std::thread::scope(|s| {
        // 有界通道：最多超前 2 批，采样快于计算时不会无限囤积内存；
        // rx 在本闭包结束时丢弃（含早停 break），工作线程下一次 send 失败即自然退出
        let (tx, rx) = std::sync::mpsc::sync_channel::<(
            Vec<usize>,
            Vec<usize>,
            Option<Vec<bool>>,
            Option<Vec<f32>>,
            u64,
        )>(2);
        s.spawn(move || {
            let mut wrng = worker_rng;
            loop {
                // 1. 采样 batch（SFT 带 loss 掩码，VLM 另带像素）
                let (x, y, mask, pixels) = loader.sample_batch_mm(&mut wrng);
                if tx.send((x, y, mask, pixels, wrng.state())).is_err() {
                    break; // 接收端已丢弃（训练循环结束）→ 本线程退出
                }
            }
        });
        for step in start_step..cfg.steps {
            // 取下一批：第一批已在 scope 外采好，其余等工作线程预取的结果
            let (x, y, mask, pixels, worker_state) = match first_batch.take() {
                Some(b) => (b.0, b.1, b.2, b.3, rng.state()),
                None => rx.recv().expect("预取线程异常退出"),
            };
            // 回写 rng：调用方看到的状态与串行采样逐位一致（断点续训/测试依赖）
            rng.set_state(worker_state);

            // 2. 前向 + 损失（梯度累积时反向按 1/accum 缩放）
            let t_seg = std::time::Instant::now();
            let loss = forward_loss(
                &model,
                &x,
                &y,
                mask.as_deref(),
                pixels.as_deref(),
                batch_size,
                block_size,
                accum,
            );
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
                    opt.set_lr(cur_lr);
                    opt.step();

                    // 7b. FP8 权重存储模拟：把刚更新的二维权重按 E4M3 往返量化回写。
                    // 放在 opt.step() 之后、下次前向之前 —— 等价于"权重以 FP8 精度存放，
                    // 梯度与优化器状态仍是 f32"。**只做数值模拟，不带来任何加速**（见 crate::fp8）。
                    if cfg.fp8 {
                        crate::fp8::roundtrip_params_in_place(
                            &params,
                            crate::fp8::TRAIN_FORMAT,
                            crate::fp8::TRAIN_BLOCK,
                        );
                    }

                    // 8. 清零梯度
                    opt.zero_grad();

                    // 8b. aux-loss-free 均衡偏置更新（DeepSeek-V3 式）：用本步前向的路由负载
                    //     推进各 MoE 层的专家偏置。**必须在 opt.step() 之后、下一次前向之前**——
                    //     路由统计只保留最近一次前向，中间的评估前向会把它覆盖成验证集的负载。
                    //     未开 `moe_bias_balance`（默认）或非 MoE 配置时是空操作。
                    model.update_moe_balance_bias();

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
                        &*opt,
                        step + 1,
                        best_val_loss,
                    );
                    if is_best && best_val_loss.is_finite() {
                        checkpoint::save(
                            &format!("{dir}/best.ckpt"),
                            model,
                            &*opt,
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
    }); // rx 在此丢弃 → 预取线程 send 失败自然退出，scope 负责 join

    let elapsed = train_t0.elapsed().as_secs_f64();
    // 实际完成的步数：早停时小于 cfg.steps - start_step，用 cfg.steps 会让「每步耗时」
    // 被系统性低估（分母偏大）。
    let steps_done = last_step_done.saturating_sub(start_step).max(1);
    if let Some(dir) = out_dir {
        checkpoint::save(
            &format!("{dir}/final.ckpt"),
            model,
            &*opt,
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

/// VQ-VAE 预训练：在图像像素上重建 + 码本量化，得到可供生成任务使用的码本。
///
/// 数据就是 `images`（每张 `[3·S·S]`，CHW [-1,1]），每步有放回采 `batch_size` 张；
/// 骨架与 [`train_transformer`] 相同（AdamW + 梯度裁剪 + cosine 学习率），
/// 但没有验证集/checkpoint（由调用方在训练后自行 [`crate::vqvae::Vqvae::save`]）。
/// 返回末步 loss。
pub fn train_vqvae(
    vq: &crate::vqvae::Vqvae,
    images: &[Vec<f32>],
    cfg: &TrainConfig,
    rng: &mut Rng,
) -> f32 {
    assert!(!images.is_empty(), "VQ-VAE 训练需要至少 1 张图像");
    let steps = cfg.vq_steps;
    let batch = cfg.batch_size;
    let params = vq.parameters();
    let mut opt = make_optimizer(cfg, params.clone());
    let mut scheduler = lr_scheduler(cfg, steps);
    let dim = images[0].len();
    let mut px = Vec::with_capacity(batch * dim);
    let mut last = 0.0f32;
    for step in 0..steps {
        px.clear();
        for _ in 0..batch {
            let i = rng.choice(images.len());
            px.extend_from_slice(&images[i]);
        }
        let loss = vq.forward_loss(&px, batch);
        loss.backward();
        clip_grad_norm(&params, cfg.grad_clip);
        opt.set_lr(scheduler.lr());
        opt.step();
        opt.zero_grad();
        scheduler.step();
        last = loss.item();
        if (step + 1) % cfg.eval_every == 0 || step + 1 == steps {
            logln!(
                "vq step {:>5}/{steps} | lr {:.6} | loss {:.4}",
                step + 1,
                scheduler.lr(),
                last
            );
        }
    }
    last
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data::DataLoader;
    use crate::model::{Transformer, TransformerConfig};
    use crate::tokenizer::Tokenizer;

    /// 同种子对照实验的**通用底座**：固定语料、种子、批大小、步数与数据顺序，
    /// 只有模型结构（`model_cfg`）与训练流程（`train_cfg`）由调用方决定。
    /// 于是两次调用之间的 loss 差异只能归因于被替换的那一项（调度 / 优化器 / QK-Norm / 路由）。
    fn run_tiny_ablation(mut model_cfg: TransformerConfig, train_cfg: TrainConfig) -> f32 {
        let corpus = "the quick brown fox jumps over the lazy dog, and then the dog jumps \
                      back over the quick brown fox again and again and again.";
        let tokenizer = Tokenizer::char(corpus);
        model_cfg.vocab_size = tokenizer.vocab_size();
        let mut rng = Rng::new(train_cfg.seed);
        let model = Transformer::new(model_cfg, &mut rng);
        let loader = DataLoader::new(corpus, &tokenizer, 16, 4);
        let mut train_rng = Rng::new(train_cfg.seed);
        train_transformer(&model, &tokenizer, &loader, &train_cfg, None, None, &mut train_rng)
    }

    /// 极小训练的基准训练配置（所有对照实验共用；调用方按需覆盖少数字段）。
    fn tiny_train_cfg() -> TrainConfig {
        TrainConfig {
            seed: 11,
            steps: 12,
            batch_size: 4,
            warmup_steps: 2,
            // 正常量级的裁剪阈值：一旦漏掉 AMP 的反缩放，阈值会被整体放大 2^16 倍，
            // 裁剪失效、结果立刻偏离——这条设置因此同时覆盖了「反缩放」这一步。
            grad_clip: 1.0,
            eval_every: 12, // 只在最后一步评估
            eval_iters: 1,
            log_file: None,
            ..TrainConfig::default()
        }
    }

    /// 只换**学习率调度**的极小训练（其余口径见 [`run_tiny_ablation`]）。
    fn run_tiny_training_with(
        amp: bool,
        lr_schedule: &str,
        decay_frac: f32,
        max_lr: f32,
        min_lr: f32,
    ) -> f32 {
        let tcfg = TrainConfig {
            amp,
            max_lr,
            min_lr,
            lr_schedule: lr_schedule.to_string(),
            wsd_decay_frac: decay_frac,
            ..tiny_train_cfg()
        };
        run_tiny_ablation(TransformerConfig::tiny(0), tcfg)
    }

    /// 收集调度器从头到尾完整的学习率曲线。
    fn lr_curve(s: &mut LRScheduler, total: usize) -> Vec<f32> {
        let mut out = Vec::with_capacity(total);
        for _ in 0..total {
            out.push(s.lr());
            s.step();
        }
        out
    }

    /// AMP 的损失缩放对 f32 训练是数值透明的：`scale` 恒为 2 的幂，乘/除 2^k 在 f32 下
    /// 只是指数移位（精确），整条反向链路逐位等于未缩放时的结果。所以开与不开启 AMP
    /// 必须得到完全相同的最终 loss。
    #[test]
    fn test_amp_loss_scaling_is_numerically_transparent() {
        let plain = run_tiny_training_with(false, "cosine", 0.1, 1e-3, 1e-3);
        let with_amp = run_tiny_training_with(true, "cosine", 0.1, 1e-3, 1e-3);
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

    /// VQ-VAE 预训练冒烟：几步训练后 loss 有限且参数确实被更新（走通 AdamW + 裁剪 + cosine 骨架）。
    #[test]
    fn test_train_vqvae_smoke() {
        use crate::vqvae::{VqConfig, Vqvae};
        let cfg = VqConfig {
            image_size: 8,
            patch_size: 4,
            latent_dim: 8,
            codebook_size: 8,
            hidden: 16,
            ..VqConfig::default()
        };
        let mut rng = Rng::new(7);
        let vq = Vqvae::new(cfg, &mut rng);
        let before: Vec<Vec<f32>> = vq.parameters().iter().map(|p| p.data()).collect();
        let mut images = Vec::new();
        for i in 0..4 {
            let mut px = Vec::with_capacity(3 * 8 * 8);
            for c in 0..3 {
                for y in 0..8 {
                    for x in 0..8 {
                        px.push(((x + y + i + c) % 8) as f32 / 4.0 - 1.0);
                    }
                }
            }
            images.push(px);
        }
        let tcfg = TrainConfig {
            vq_steps: 3,
            batch_size: 2,
            eval_every: 3, // 只在最后一步打日志
            ..TrainConfig::default()
        };
        let mut train_rng = Rng::new(7);
        let loss = train_vqvae(&vq, &images, &tcfg, &mut train_rng);
        assert!(loss.is_finite() && loss > 0.0, "末步 loss 应为正有限值，实际 {loss}");
        let moved = vq
            .parameters()
            .iter()
            .zip(&before)
            .any(|(p, b)| p.data().iter().zip(b).any(|(a, c)| a != c));
        assert!(moved, "训练后至少一个参数应发生变化");
    }

    /// WSD 的三段结构：warmup 线性递增 → 稳定段恒等于 `max_lr` → 末段单调不增退火，
    /// 且整条曲线始终落在 `[min_lr, max_lr]` 内。
    #[test]
    fn test_wsd_schedule_has_three_phases() {
        let warmup = 4;
        let total = 20;
        let (max_lr, min_lr, decay) = (1.0f32, 0.1f32, 5usize);
        let mut s = LRScheduler::new_wsd(warmup, total, max_lr, min_lr, decay);
        let curve = lr_curve(&mut s, total);
        assert_eq!(curve.len(), total);

        // 阶段一：warmup 严格递增，末点正好踩到峰值
        for i in 0..warmup - 1 {
            assert!(curve[i] < curve[i + 1], "warmup 段应严格递增，第 {i} 步失败");
        }
        assert_eq!(curve[warmup - 1], max_lr, "warmup 结束应正好是 max_lr");

        // 阶段二：稳定段恒等于峰值（退火起点 = total - decay = 15）
        let decay_start = total - decay;
        for (step, &lr) in curve.iter().enumerate().take(decay_start).skip(warmup) {
            assert_eq!(lr, max_lr, "稳定段第 {step} 步应恒为 max_lr，实际 {lr}");
        }

        // 阶段三：退火段单调不增，且不小于 min_lr
        for step in decay_start..total - 1 {
            assert!(
                curve[step] >= curve[step + 1],
                "退火段应单调不增，第 {step}→{} 步失败（{} → {}）",
                step + 1,
                curve[step],
                curve[step + 1]
            );
        }
        assert_eq!(curve[decay_start], max_lr, "退火起点应从 max_lr 开始");
        assert!(
            curve[total - 1] >= min_lr && curve[total - 1] < max_lr,
            "末步应落在 [min_lr, max_lr) 内，实际 {}",
            curve[total - 1]
        );

        // 全段范围检查
        for (step, &lr) in curve.iter().enumerate() {
            assert!(
                lr >= min_lr && lr <= max_lr,
                "第 {step} 步 lr {lr} 越界 [{min_lr}, {max_lr}]"
            );
        }
    }

    /// 两条曲线共用同一段 warmup：切到 WSD 不应该改变前 `warmup_steps` 步的任何一位。
    #[test]
    fn test_wsd_warmup_matches_cosine() {
        let (warmup, total, max_lr, min_lr) = (4, 20, 1.0f32, 0.1f32);
        let mut cosine = LRScheduler::new(warmup, total, max_lr, min_lr);
        let mut wsd = LRScheduler::new_wsd(warmup, total, max_lr, min_lr, 5);
        let (a, b) = (lr_curve(&mut cosine, total), lr_curve(&mut wsd, total));
        for step in 0..warmup {
            assert_eq!(a[step], b[step], "warmup 第 {step} 步两条曲线应逐位相同");
        }
        assert_ne!(a[total - 1], b[total - 1], "末步两条曲线应确实不同（否则没测到差别）");
    }

    /// cosine 默认曲线的数值快照：加 WSD 不允许改动默认路径的行为。
    ///
    /// 逐点比对 `warmup=2, total=6, max_lr=1, min_lr=0` 的手算结果，
    /// 一旦有人改动 cosine 公式，这里会立刻红。
    #[test]
    fn test_cosine_schedule_default_unchanged() {
        let mut s = LRScheduler::new(2, 6, 1.0, 0.0);
        let curve = lr_curve(&mut s, 6);
        let expected = [
            0.5, // warmup 1/2
            1.0, // warmup 2/2
            1.0, // cosine progress 0
            0.853_553_4, // progress 0.25
            0.5, // progress 0.5
            0.146_446_6, // progress 0.75
        ];
        for (step, (&got, &want)) in curve.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "cosine 第 {step} 步应为 {want}，实际 {got}（默认行为被改动）"
            );
        }
    }

    /// `lr_scheduler` 工厂按 `lr_schedule` 分派：`"wsd"` 的退火起点 = `round(frac × (steps - warmup))`，
    /// 且稳定段是一段真正的平台；`"cosine"` 则没有平台。
    #[test]
    fn test_lr_scheduler_factory_respects_schedule() {
        let base = TrainConfig {
            steps: 20,
            warmup_steps: 4,
            max_lr: 1.0,
            min_lr: 0.1,
            wsd_decay_frac: 0.5,
            log_file: None,
            ..TrainConfig::default()
        };

        let mut wsd = lr_scheduler(
            &TrainConfig {
                lr_schedule: "wsd".to_string(),
                ..base.clone()
            },
            20,
        );
        let curve = lr_curve(&mut wsd, 20);
        let decay = (0.5 * (20 - 4) as f32).round() as usize; // 8
        let decay_start = 20 - decay; // 12
        let plateau = curve[4..decay_start]
            .iter()
            .filter(|&&lr| lr == 1.0)
            .count();
        assert_eq!(plateau, decay_start - 4, "稳定段应整段都是 max_lr");
        assert!(curve[decay_start + 1] < 1.0, "退火应从 decay_start 之后开始下降");
        assert!(
            (curve[19] - (1.0 + (0.1 - 1.0) * (7.0 / 8.0))).abs() < 1e-6,
            "末步应为线性退火到 7/8 处的值，实际 {}",
            curve[19]
        );

        let mut cos = lr_scheduler(&base, 20);
        let cos_curve = lr_curve(&mut cos, 20);
        assert!(
            cos_curve[5] < 1.0,
            "cosine 从 warmup 结束就该开始衰减，不应有平台，实际第 5 步 {}",
            cos_curve[5]
        );
    }

    /// 病态参数（退火步数几乎吃掉整个训练）也不能让曲线非单调：退火起点被 `warmup` 夹住，
    /// warmup 一结束就退火，只保证「不早于 warmup 结束」。
    #[test]
    fn test_wsd_decay_clamped_to_warmup() {
        let warmup = 8;
        let total = 10;
        let mut s = LRScheduler::new_wsd(warmup, total, 1.0, 0.0, 9);
        let curve = lr_curve(&mut s, total);
        // warmup 段严格递增
        for step in 0..warmup - 1 {
            assert!(curve[step] < curve[step + 1], "warmup 第 {step} 步应递增");
        }
        // warmup 结束之后（稳定 + 退火）单调不增
        for step in warmup - 1..total - 1 {
            assert!(
                curve[step] >= curve[step + 1],
                "warmup 之后曲线应单调不增，第 {step}→{} 步失败（{} → {}）",
                step + 1,
                curve[step],
                curve[step + 1]
            );
        }
        // 退火起点被夹到 warmup 结尾（第 8 步），第 8 步仍是峰值
        assert_eq!(curve[8], 1.0, "退火起点应被夹到 warmup 结束处");
    }

    /// 同种子对照（**隔离性**）：`max_lr == min_lr` 时两条曲线逐位恒等（都是常值），
    /// 于是同种子的整轮训练必须得到**逐位相同**的最终 loss。这证明调度差异被正确隔离、
    /// 没有偷偷改动别的路径，也顺手证明训练本身可复现。
    #[test]
    fn test_wsd_and_cosine_identical_when_lr_flat() {
        let cosine = run_tiny_training_with(true, "cosine", 0.1, 1e-3, 1e-3);
        let wsd = run_tiny_training_with(true, "wsd", 0.5, 1e-3, 1e-3);
        assert_eq!(
            cosine, wsd,
            "max_lr == min_lr 时两条曲线都是常值，同种子 loss 应逐位相同（cosine {cosine}，wsd {wsd}）"
        );
    }

    /// 同种子对照（**有效性**）：把学习率区间拉成 `max_lr > min_lr` 后，两条曲线确实不同，
    /// 同seed训练出来的 loss 也应当不同——否则说明调度只写在配置里、根本没作用到优化器。
    /// 断言只要求「都在正常量级且不相等」，不比较谁更优（12 步的极小模型上谁赢是噪声）。
    #[test]
    fn test_wsd_vs_cosine_same_seed_differs() {
        let cosine = run_tiny_training_with(true, "cosine", 0.1, 1e-3, 1e-4);
        let wsd = run_tiny_training_with(true, "wsd", 0.5, 1e-3, 1e-4);
        assert!(
            cosine.is_finite() && cosine > 0.0,
            "cosine 调度的 loss 应为正有限值，实际 {cosine}"
        );
        assert!(
            wsd.is_finite() && wsd > 0.0,
            "WSD 调度的 loss 应为正有限值，实际 {wsd}"
        );
        assert_ne!(
            cosine, wsd,
            "两条曲线不同，同种子 loss 也应有差异（cosine {cosine}，wsd {wsd}）"
        );
    }

    /// 同种子对照：QK-Norm 只改注意力子层的结构，其余配置与数据完全相同。
    /// 两条要求——都训得出有限 loss（新参数不破坏训练），且结果**确实不同**
    /// （否则说明开关没接进前向，只是配置里多了一个字段）。
    #[test]
    fn test_qk_norm_changes_training_same_seed() {
        let plain = run_tiny_ablation(TransformerConfig::tiny(0), tiny_train_cfg());
        let mut mcfg = TransformerConfig::tiny(0);
        mcfg.qk_norm = true;
        let qk = run_tiny_ablation(mcfg, tiny_train_cfg());

        assert!(
            plain.is_finite() && plain > 0.0,
            "未开 QK-Norm 的 loss 应为正有限值，实际 {plain}"
        );
        assert!(
            qk.is_finite() && qk > 0.0,
            "开启 QK-Norm 的 loss 应为正有限值，实际 {qk}"
        );
        assert_ne!(
            plain, qk,
            "开了 QK-Norm 却没有改变训练结果，说明它没接进前向（{plain} vs {qk}）"
        );
    }

    /// 同种子对照：只把优化器从 AdamW 换成 Muon，其余（结构 / 数据 / 种子 / 学习率曲线）全同。
    /// 断言「都有限且不为同一值」——有限说明 Muon 的 Newton–Schulz / 缩放链路没把参数推成 NaN，
    /// 不相等说明 `train.optimizer` 真的作用到了更新步骤上，而不只是配置里多了个字段。
    ///
    /// 学习率取 `0.05`（比默认 `3e-3` 大）：Muon 每元素更新量约 `lr/√cols`，
    /// 沿用 AdamW 的学习率会让它几乎不动，两次结果都被初始化主导而看不出差异。
    #[test]
    fn test_muon_changes_training_same_seed() {
        let base = TrainConfig {
            max_lr: 0.05,
            min_lr: 0.005,
            ..tiny_train_cfg()
        };
        let adam = run_tiny_ablation(TransformerConfig::tiny(0), base.clone());
        let muon = run_tiny_ablation(
            TransformerConfig::tiny(0),
            TrainConfig {
                optimizer: "muon".to_string(),
                ..base
            },
        );
        assert!(
            adam.is_finite() && adam > 0.0,
            "AdamW 的 loss 应为正有限值，实际 {adam}"
        );
        assert!(
            muon.is_finite() && muon > 0.0,
            "Muon 的 loss 应为正有限值（NaN 说明正交化/缩放有问题），实际 {muon}"
        );
        assert_ne!(
            adam, muon,
            "换优化器却没有改变训练结果，说明它没接进更新步骤（{adam} vs {muon}）"
        );
    }

    /// MoE 的 aux-loss-free 均衡偏置接进了训练步：γ > 0 时偏置会挪动硬路由、进而改 loss；
    /// γ = 0 时是恒等操作 ⇒ 必须与「完全关闭」逐位相同（默认路径不变的回归）。
    #[test]
    fn test_moe_bias_balance_changes_training_same_seed() {
        let moe_cfg = |bias: bool, lr: f32| TransformerConfig {
            n_expert: 2,
            moe_top_k: 1,
            // K = 1 时重归一化口径给不了路由器梯度，统一走 Switch 口径（与 main.rs 的 moe 子命令一致）
            moe_switch_gate: true,
            moe_bias_balance: bias,
            moe_bias_lr: lr,
            ..TransformerConfig::default()
        };
        let off = run_tiny_ablation(moe_cfg(false, 0.0), tiny_train_cfg());
        let zero = run_tiny_ablation(moe_cfg(true, 0.0), tiny_train_cfg());
        assert_eq!(off, zero, "γ = 0 应与关闭逐位相同（{off} vs {zero}）");

        let on = run_tiny_ablation(moe_cfg(true, 0.5), tiny_train_cfg());
        assert!(
            (on - off).abs() > 1e-9,
            "开启偏置均衡后 loss 应与关闭不同，说明它没接进训练循环（{off} vs {on}）"
        );
    }

    /// 回归保护：默认配置下走的就是 AdamW，且 `make_optimizer` 对未知取值也回退 AdamW
    /// （`validate` 会先拦下非法值，这里是双保险：工厂本身不会 panic）。
    #[test]
    fn test_optimizer_factory_defaults_to_adamw() {
        assert_eq!(TrainConfig::default().optimizer, "adamw", "默认优化器必须是 adamw");
        let mut rng = Rng::new(3);
        let model = Transformer::new(TransformerConfig::tiny(32), &mut rng);
        let params = model.parameters();
        // 未知取值（正常路径下 validate 会拒绝）不能 panic，退回 AdamW
        let cfg = TrainConfig {
            optimizer: "unknown".to_string(),
            ..tiny_train_cfg()
        };
        let mut opt = make_optimizer(&cfg, params.clone());
        assert_eq!(opt.params().len(), params.len());
        opt.step(); // 梯度全零：只验证能正常走一步
    }

    /// 回归保护：`qk_norm = false`（默认）时，模型结构与参数量与加该特性之前**完全一致**
    /// ——四个投影的参数一个不多一个不少，QKV 的所有权重逐位等于"没有 QK-Norm 这条路"。
    #[test]
    fn test_qk_norm_off_keeps_default_structure() {
        let mut rng = Rng::new(11);
        let cfg = TransformerConfig::tiny(32);
        assert!(!cfg.qk_norm, "默认必须是关闭");
        let model = Transformer::new(cfg.clone(), &mut rng);
        for (name, _) in model.named_parameters() {
            assert!(
                !name.contains("q_norm") && !name.contains("k_norm"),
                "关闭 QK-Norm 时不该出现归一化参数：{name}"
            );
        }

        // 开启后必须出现且形状为 head_dim
        let mut mcfg = cfg;
        mcfg.qk_norm = true;
        let mut rng2 = Rng::new(11);
        let model = Transformer::new(mcfg, &mut rng2);
        let hd = model.cfg.n_embd / model.cfg.n_head;
        let names: Vec<String> = model.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert_eq!(
            names.iter().filter(|n| n.ends_with(".attn.q_norm.gamma")).count(),
            model.cfg.n_layer
        );
        assert_eq!(
            names.iter().filter(|n| n.ends_with(".attn.k_norm.gamma")).count(),
            model.cfg.n_layer
        );
        for (n, p) in model.named_parameters() {
            if n.ends_with(".attn.q_norm.gamma") || n.ends_with(".attn.k_norm.gamma") {
                assert_eq!(p.numel(), hd, "{n} 的 gamma 长度应为 head_dim={hd}");
            }
        }
    }
}
