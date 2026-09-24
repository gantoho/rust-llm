//! 第 31 课：Scaling Laws（缩放定律）—— 幂律拟合、算力预算与 Chinchilla 最优配比
//!
//! 把「loss 与参数量 / 数据量 / 算力之间的幂律关系」从公式落成**可运行、可验证**的代码。
//! 全程手写，不引拟合库：
//!
//! 1. **幂律拟合** [`fit_power_law`]：`L(x) = a·x^(-α) + b`，其中 `b` 是不可约损失
//!    （irreducible loss，语料本身的熵，规模再大也压不下去的那部分）。直接对 `log L`
//!    做最小二乘会把 `b` 摊进斜率，拟合出的 α 系统性偏小——所以这里把 `b` 当**待定参数**
//!    搜：固定 `b` 后模型对 `log a` / `α` 是线性的（闭式最小二乘），外层再用网格 +
//!    黄金分割把 `b` 找出来。
//! 2. **口径统一** [`params_non_embedding`] / [`flops_train`]：`C ≈ 6ND` 里的 `N` 是
//!    **非嵌入**参数量（Kaplan / Chinchilla 口径），不是 `model.parameters()` 的总和。
//!    参数公式与真实建层共用同一个隐藏维度公式，扫描时还有断言逐项核对。
//! 3. **最优配比**：经验 20:1 法则 [`ratio20_optimal`]（Chinchilla 头条结论，来自
//!    IsoFLOP 实验）与参数化损失的闭式最优解 [`parametric_optimal`]（Approach 3）。
//!    两者在同等算力下给出的最优规模**并不相同**，本模块把分歧显式暴露出来而不是抹平
//!    ——这正是后续文献（Besiroglu et al., 2024）质疑的焦点，也是本课值得记住的一课。
//! 4. **实测扫描** [`params_scan`] / [`tokens_scan`]：真的训一组不同规模的 tiny 模型，
//!    用实测 loss 拟合幂律，而不是把论文数字抄一遍。
//!
//! 对应文档：`docs/31-Scaling-Laws.md`；命令行入口：`cargo run --release -- scaling`。

use crate::config::TrainConfig;
use crate::data::DataLoader;
use crate::model::{Transformer, TransformerConfig};
use crate::module::Module;
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;
use crate::train::train_transformer;

// ==================== 幂律拟合 ====================

/// 幂律拟合结果：`L(x) = a·x^(-α) + b`
///
/// - `alpha` 就是 Kaplan / Chinchilla 论文里的幂律指数（loss 对规模的敏感度）
/// - `b` 是不可约损失：`x → ∞` 时 loss 的下界
/// - `r2` 是**线性空间**（nats）上的决定系数，衡量拟合对真实 loss 的解释力
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PowerLaw {
    pub a: f64,
    pub alpha: f64,
    pub b: f64,
    pub r2: f64,
    pub n_points: usize,
}

impl PowerLaw {
    /// 用拟合出的幂律外推 `x` 处的 loss
    pub fn predict(&self, x: f64) -> f64 {
        self.a * x.powf(-self.alpha) + self.b
    }

    /// 规模翻倍能换来多少 loss 下降（nats）。幂律的"回报递减"就体现在它是个常数：
    /// 任何规模点上翻倍都只能拿到这么多。
    pub fn loss_drop_per_doubling(&self, x: f64) -> f64 {
        self.predict(x) - self.predict(2.0 * x)
    }
}

/// 固定 `b` 时对 `(log x, log(y-b))` 做闭式最小二乘，返回 `(a, α, 绝对平方误差和)`。
///
/// 为什么固定 `b` 后是闭式的：`log(y-b) = log a - α·log x`，对 `(log a, α)` 是线性的。
/// `b` 是唯一的非线性参数，留给外层搜索。
///
/// 选取 `b` 用的是**绝对**平方误差（nats²）而不是 log 空间的误差：预测 loss 时关心的
/// 是 nats 上的绝对偏差。
fn fit_linear_at_b(xs: &[f64], ys: &[f64], b: f64) -> Option<(f64, f64, f64)> {
    let n = xs.len() as f64;
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (&x, &y) in xs.iter().zip(ys) {
        let z = y - b;
        if z <= 0.0 {
            return None; // 不可约损失不可能不小于最小 loss
        }
        let (lx, lz) = (x.ln(), z.ln());
        sx += lx;
        sy += lz;
        sxx += lx * lx;
        sxy += lx * lz;
    }
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-12 {
        return None; // 所有 x 相同，斜率无解
    }
    let slope = (n * sxy - sx * sy) / denom;
    let intercept = (sy - slope * sx) / n;
    let alpha = -slope;
    let a = intercept.exp();
    if !(a.is_finite() && a > 0.0 && alpha.is_finite()) {
        return None;
    }
    let sse: f64 = xs
        .iter()
        .zip(ys)
        .map(|(&x, &y)| {
            let r = y - (a * x.powf(-alpha) + b);
            r * r
        })
        .sum();
    Some((a, alpha, sse))
}

/// 把 `(规模, loss)` 点拟合成 `L(x) = a·x^(-α) + b`。
///
/// 至少 3 个点（3 个待定参数）；点数越少，`α` 与 `b` 的分离越依赖数据跨度——
/// 想要稳定的指数，规模之间最好跨一个数量级（这也是论文里"跨多个数量级都成立"的意思）。
pub fn fit_power_law(xs: &[f64], ys: &[f64]) -> PowerLaw {
    assert_eq!(
        xs.len(),
        ys.len(),
        "x 与 y 的点数必须一致（{} vs {}）",
        xs.len(),
        ys.len()
    );
    assert!(
        xs.len() >= 3,
        "至少需要 3 个点才能同时定出 a / α / b（实际 {} 个）",
        xs.len()
    );
    for (&x, &y) in xs.iter().zip(ys) {
        assert!(x > 0.0 && x.is_finite(), "幂律拟合要求 x > 0，实际 {x}");
        assert!(y > 0.0 && y.is_finite(), "loss 必须是正有限值，实际 {y}");
    }

    let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
    // b 必须严格小于最小 loss：等于它时最小那个点的 z = y - b = 0，log 无解。
    // 留一点余量避免数值上贴着边界。
    let b_max = y_min * (1.0 - 1e-9);

    const GRID: usize = 512;
    let mut b_best = 0.0;
    let mut best: Option<(f64, f64, f64)> = None; // (sse, a, alpha)
    for i in 0..=GRID {
        let b = b_max * i as f64 / GRID as f64;
        if let Some((a, alpha, sse)) = fit_linear_at_b(xs, ys, b) {
            if best.is_none_or(|(bs, _, _)| sse < bs) {
                best = Some((sse, a, alpha));
                b_best = b;
            }
        }
    }
    let (_, a_fallback, alpha_fallback) =
        best.expect("网格内没有任何可行的 b（loss 是否全为常数？）");

    // 黄金分割细化：α 对 b 极其敏感（点数少时尤其明显），hi/512 的网格步长只是粗定位
    let step = b_max / GRID as f64;
    let mut lo = (b_best - step).max(0.0);
    let mut hi = (b_best + step).min(b_max);
    let phi = (5.0f64.sqrt() - 1.0) / 2.0;
    for _ in 0..200 {
        let m1 = hi - phi * (hi - lo);
        let m2 = lo + phi * (hi - lo);
        let s1 = fit_linear_at_b(xs, ys, m1).map_or(f64::INFINITY, |r| r.2);
        let s2 = fit_linear_at_b(xs, ys, m2).map_or(f64::INFINITY, |r| r.2);
        if s1 < s2 {
            hi = m2;
        } else {
            lo = m1;
        }
        if (hi - lo).abs() < b_max * 1e-12 {
            break;
        }
    }
    let b = 0.5 * (lo + hi);
    let (a, alpha, sse) = fit_linear_at_b(xs, ys, b).unwrap_or((a_fallback, alpha_fallback, 0.0));

    let mean = ys.iter().sum::<f64>() / ys.len() as f64;
    let sst: f64 = ys.iter().map(|y| (y - mean) * (y - mean)).sum();
    let r2 = if sst > 1e-12 { 1.0 - sse / sst } else { 1.0 };

    PowerLaw {
        a,
        alpha,
        b,
        r2,
        n_points: xs.len(),
    }
}

// ==================== 参数量 / 算力口径 ====================

/// 训练总浮点运算量的近似：`C ≈ 6ND`。
///
/// 6 = 前向 2（每个参数一次乘一次加）+ 反向 4（对输入的梯度 2、对权重的梯度 2）。
/// `n_params` 必须是**非嵌入**参数量（见 [`params_non_embedding`]），否则嵌入层
/// （词表 × 维度，动辄上亿）会把 `C` 抬高一截。
pub const FLOPS_PER_PARAM_TOKEN: f64 = 6.0;

/// `C ≈ 6ND`
pub fn flops_train(n_params: f64, n_tokens: f64) -> f64 {
    FLOPS_PER_PARAM_TOKEN * n_params * n_tokens
}

/// 一层 Transformer Block 的参数量（不含嵌入与最终归一化）
pub fn params_per_layer(cfg: &TransformerConfig) -> usize {
    let d = cfg.n_embd;
    let n_kv = if cfg.n_kv_head == 0 {
        cfg.n_head
    } else {
        cfg.n_kv_head
    };
    let hd = d / cfg.n_head;
    // 两个归一化子层：LayerNorm 有 γ/β，RMSNorm 只有 γ
    let norm = if cfg.use_rmsnorm { d } else { 2 * d };
    // 注意力四个投影：权重 [in, out] + 长度 out 的 bias。
    // GQA 下 K/V 只投影到 n_kv 个头，参数量随之下降
    let attn = (d * d + d)                        // c_q
        + 2 * (d * (n_kv * hd) + n_kv * hd)       // c_k, c_v
        + (d * d + d);                            // c_proj
    // MLP：GELU 两个投影（隐层 4d），SwiGLU 三个投影（隐层 ≈ 2.67d）
    let mlp = if cfg.use_swiglu {
        let h = crate::layers::swiglu_hidden(d);
        3 * d * h + 2 * h + d // w_gate / w_up: d×h + h；w_down: h×d + d
    } else {
        (d * 4 * d + 4 * d) + (4 * d * d + d)
    };
    2 * norm + attn + mlp
}

/// 非嵌入参数量：`C = 6ND` 里的 `N`（Kaplan / Chinchilla 口径）。
///
/// 减去嵌入层是有道理的：词嵌入的规模由**词表**决定，与"模型有多深多宽"无关，
/// 把它算进去会让小模型的 N 虚高（本项目的 512 词表下嵌入约占 2 层小模型的一半）。
pub fn params_non_embedding(cfg: &TransformerConfig) -> usize {
    let d = cfg.n_embd;
    let final_norm = if cfg.use_rmsnorm { d } else { 2 * d };
    cfg.n_layer * params_per_layer(cfg) + final_norm
}

/// 全部参数量（含词嵌入）。本项目输出头与词嵌入**共享同一张表**，所以只算一份。
pub fn params_total(cfg: &TransformerConfig) -> usize {
    params_non_embedding(cfg) + cfg.vocab_size * cfg.n_embd
}

// ==================== Chinchilla 最优配比 ====================

/// Chinchilla 参数化损失模型的拟合常数（Hoffmann et al., 2022, Approach 3）
///
/// `L(N, D) = E + A/N^α + B/D^β`：`E` 是不可约损失，`A/N^α` 是模型容量不足的惩罚，
/// `B/D^β` 是数据不足的惩罚。
pub const CHINCHILLA_E: f64 = 1.69;
pub const CHINCHILLA_A: f64 = 406.4;
pub const CHINCHILLA_B: f64 = 410.7;
pub const CHINCHILLA_ALPHA: f64 = 0.34;
pub const CHINCHILLA_BETA: f64 = 0.28;

/// 经验最优 token/参数比：Chinchilla 的"20:1 法则"
///
/// 注意它**不是**从上面那组参数化常数推出来的（推出来是另一个数，见
/// [`parametric_optimal`] 的说明），而是论文用 IsoFLOP 剖面直接量出来的头条结论。
pub const TOKENS_PER_PARAM_OPTIMAL: f64 = 20.0;

/// 一次算力预算下的最优资源分配
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Allocation {
    /// 算力预算（FLOPs）
    pub compute: f64,
    /// 模型参数量（非嵌入）
    pub params: f64,
    /// 训练 token 数
    pub tokens: f64,
    /// 按 [`chinchilla_loss`] 预测的 loss
    pub loss: f64,
}

impl Allocation {
    /// token/参数比：Chinchilla 最优配比下应接近 20
    pub fn tokens_per_param(&self) -> f64 {
        self.tokens / self.params
    }
}

/// Chinchilla 参数化损失模型 `L(N, D) = E + A/N^α + B/D^β`
pub fn chinchilla_loss(params: f64, tokens: f64) -> f64 {
    CHINCHILLA_E
        + CHINCHILLA_A / params.powf(CHINCHILLA_ALPHA)
        + CHINCHILLA_B / tokens.powf(CHINCHILLA_BETA)
}

/// 经验 20:1 法则下的最优分配（本项目的缺省口径，也是 [`crate::scaling`] 报告用的那个）。
///
/// 联立 `C = 6ND` 与 `D = 20N` 得 `120·N² = C`，即 `N = √(C/120)`，
/// `D = 20N`。两个量都 ∝ C^0.5：**算力翻倍，参数量与数据量各涨约 41%**
/// （√2 ≈ 1.41）——这就是 Chinchilla 相对 OpenAI（`N ∝ C^0.73`）的核心修正。
pub fn ratio20_optimal(c: f64) -> Allocation {
    assert!(c > 0.0 && c.is_finite(), "算力预算必须是正有限值，实际 {c}");
    let n = (c / (FLOPS_PER_PARAM_TOKEN * TOKENS_PER_PARAM_OPTIMAL)).sqrt();
    let d = TOKENS_PER_PARAM_OPTIMAL * n;
    Allocation {
        compute: c,
        params: n,
        tokens: d,
        loss: chinchilla_loss(n, d),
    }
}

/// 参数化损失模型在约束 `C = 6ND` 下的**闭式**最优解。
///
/// 把 `D = C/(6N)` 代回损失，令 `dL/dN = 0`：
///
/// ```text
/// -αA·N^(-α-1) + βB·(6/C)^β·N^(β-1) = 0
/// ⇒ N^(α+β) = (αA/βB)·(C/6)^β
/// ⇒ N* = [ (αA/βB)·(C/6)^β ]^(1/(α+β)),  D* = C/(6N*)
/// ```
///
/// 用论文正文的这组常数代入，`N* ∝ C^0.4516`、`D*/N* ≈ 26`（C = 1e18）并随 C 缓慢上升，
/// 与 20:1 法则**不一致**。这不是本项目的实现误差：论文正文里的常数经过四舍五入，
/// 会让 Approach 3 的最优解偏离 Approaches 1/2（Besiroglu et al., 2024 复现了这一点）。
/// 因此本模块同时提供两条路径，并在 `scaling` 子命令里把两者的差距打印出来。
pub fn parametric_optimal(c: f64) -> Allocation {
    assert!(c > 0.0 && c.is_finite(), "算力预算必须是正有限值，实际 {c}");
    let (a, b) = (CHINCHILLA_A, CHINCHILLA_B);
    let (alpha, beta) = (CHINCHILLA_ALPHA, CHINCHILLA_BETA);
    let n = ((alpha * a / (beta * b)) * (c / FLOPS_PER_PARAM_TOKEN).powf(beta))
        .powf(1.0 / (alpha + beta));
    let d = c / (FLOPS_PER_PARAM_TOKEN * n);
    Allocation {
        compute: c,
        params: n,
        tokens: d,
        loss: chinchilla_loss(n, d),
    }
}

/// 给定模型规模，Chinchilla 最优的数据量与所需算力（20:1 口径）。
///
/// 也就是 [`ratio20_optimal`] 的反解：`D = 20N`、`C = 6ND = 120N²`。
/// 用来回答"我要训一个 13B 的模型，该配多少数据"。
pub fn optimal_for_params(params: f64) -> Allocation {
    assert!(params > 0.0 && params.is_finite(), "参数量必须是正有限值，实际 {params}");
    let d = TOKENS_PER_PARAM_OPTIMAL * params;
    Allocation {
        compute: flops_train(params, d),
        params,
        tokens: d,
        loss: chinchilla_loss(params, d),
    }
}

/// 过训练 / 欠训练：固定算力预算 `C`，把数据量放大到最优值的 `k` 倍（模型相应变小）。
///
/// `k = 1` 就是 20:1 最优点；`k > 1` 是"小模型喂更多数据"（LLaMA-3 8B 训 15T token
/// 就是 k ≈ 94），`k < 1` 是"大模型数据不够"（175B 级模型的 k ≈ 0.09）。
///
/// 注意这里固定的是**训练算力**。实际部署要看**总成本**（训练 + 推理）：推理量大时，
/// 过训练的小模型总账更划算——这就是"为什么故意偏离 Chinchilla"的答案。
pub fn overtrain(c: f64, k: f64) -> Allocation {
    assert!(k > 0.0 && k.is_finite(), "数据量倍数必须是正有限值，实际 {k}");
    let base = ratio20_optimal(c);
    let d = base.tokens * k;
    // C = 6ND 固定，数据量涨了就只能缩小模型
    let n = c / (FLOPS_PER_PARAM_TOKEN * d);
    Allocation {
        compute: c,
        params: n,
        tokens: d,
        loss: chinchilla_loss(n, d),
    }
}

/// Chinchilla 论文的算力-最优配比表：`(算力 FLOPs, 最优参数量, 最优数据量 token)`。
///
/// 这三列在每一行都满足 `C = 6ND` 与 `D/N = 20`，测试会逐行核对；文档
/// `docs/31-Scaling-Laws.md` 里的那张表就是这六行。
pub const CHINCHILLA_TABLE: [(f64, f64, f64); 6] = [
    (1.92e19, 4.0e8, 8.0e9),
    (2.03e20, 1.3e9, 2.6e10),
    (1.92e21, 4.0e9, 8.0e10),
    (2.03e22, 1.3e10, 2.6e11),
    (1.92e23, 4.0e10, 8.0e11),
    (2.03e24, 1.3e11, 2.6e12),
];

// ==================== 训练时长 / 成本估算 ====================

/// 硬件与价格参数（估算训练时长与电费用）
#[derive(Clone, Copy, Debug)]
pub struct Hardware {
    /// 单卡理论峰值（TFLOPS，按训练所用精度，如 A100 FP16 = 312）
    pub gpu_tflops: f64,
    pub n_gpu: usize,
    /// MFU（Model FLOPs Utilization）：实际算力利用率，常见 0.3 ~ 0.6
    pub mfu: f64,
    /// 单卡功耗（W）
    pub gpu_watts: f64,
    /// 电价（美元 / kWh）
    pub usd_per_kwh: f64,
}

impl Default for Hardware {
    fn default() -> Self {
        Hardware {
            gpu_tflops: 312.0, // A100 80GB FP16
            n_gpu: 64,
            mfu: 0.4,
            gpu_watts: 400.0,
            usd_per_kwh: 0.1,
        }
    }
}

impl Hardware {
    /// 集群有效算力（FLOPS）：单卡峰值 × MFU × 卡数
    pub fn effective_flops_per_sec(&self) -> f64 {
        self.gpu_tflops * 1e12 * self.mfu * self.n_gpu as f64
    }

    /// 估算训练 `c` FLOPs 所需的时间与电费
    pub fn estimate(&self, c: f64) -> WallClock {
        let eff = self.effective_flops_per_sec();
        let seconds = c / eff;
        let kwh = self.n_gpu as f64 * self.gpu_watts / 1000.0 * (seconds / 3600.0);
        WallClock {
            seconds,
            days: seconds / 86400.0,
            effective_flops_per_sec: eff,
            energy_kwh: kwh,
            cost_usd: kwh * self.usd_per_kwh,
        }
    }
}

/// 训练时长与成本估算结果
#[derive(Clone, Copy, Debug)]
pub struct WallClock {
    pub seconds: f64,
    pub days: f64,
    /// 集群有效算力（FLOPS）
    pub effective_flops_per_sec: f64,
    pub energy_kwh: f64,
    pub cost_usd: f64,
}

// ==================== 实测扫描 ====================

/// 实测扫描的公共设置。
///
/// 扫描要的是**同一口径下的可比数字**：固定 token 预算 / 固定评估批数 / 不落盘 / 不早停，
/// 否则不同规模各自的曲线会停在不同位置，横向没法比。
#[derive(Clone, Copy, Debug)]
pub struct ScanConfig {
    /// 每个点训练多少步
    pub steps: usize,
    pub batch_size: usize,
    /// 上下文长度（决定每步的 token 数 = batch × block）
    pub block_size: usize,
    pub max_lr: f32,
    pub seed: u64,
}

impl Default for ScanConfig {
    fn default() -> Self {
        ScanConfig {
            steps: 600,
            batch_size: 8,
            block_size: 64,
            max_lr: 3e-3,
            seed: 42,
        }
    }
}

impl ScanConfig {
    /// 每个点实际见到的 token 数（固定 token 预算的关键：它不随规模变化）
    pub fn tokens_per_point(&self, steps: usize) -> f64 {
        (steps * self.batch_size * self.block_size) as f64
    }
}

/// 实测扫描的一个数据点
#[derive(Clone, Debug)]
pub struct ScanPoint {
    /// 人类可读的标签（如 `4x128`）
    pub label: String,
    pub n_layer: usize,
    pub n_embd: usize,
    /// 全部参数量（含嵌入，实测值）
    pub params: usize,
    /// 非嵌入参数量（`C = 6ND` 里的 N）
    pub params_non_embedding: usize,
    /// 训练步数
    pub steps: usize,
    /// 训练见到的 token 数
    pub tokens: f64,
    /// 测得的验证 loss（无验证集时为训练 loss）
    pub loss: f64,
}

impl ScanPoint {
    /// 该点的训练算力 `C ≈ 6ND`
    pub fn compute(&self) -> f64 {
        flops_train(self.params_non_embedding as f64, self.tokens)
    }

    /// token / 参数比
    pub fn tokens_per_param(&self) -> f64 {
        self.tokens / self.params_non_embedding as f64
    }
}

/// 为给定隐藏维度挑一个能整除的头数：优先 `want`，否则退到不超过它的最大因子。
///
/// 扫描时模型宽度是变量，`n_embd % n_head` 这个约束不能靠调用方手工维护
/// （手写一串能整除的组合，改个尺寸就 panic）。
fn pick_n_head(d: usize, want: usize) -> usize {
    if want >= 1 && d % want == 0 {
        return want;
    }
    let mut h = want.min(d).max(1);
    while h > 1 && d % h != 0 {
        h -= 1;
    }
    h
}

/// 扫描用的训练配置：口径固定、不落盘、不早停
fn scan_train_config(sc: &ScanConfig, steps: usize) -> TrainConfig {
    TrainConfig {
        seed: sc.seed,
        batch_size: sc.batch_size,
        steps,
        max_lr: sc.max_lr,
        min_lr: sc.max_lr * 0.1,
        warmup_steps: (steps / 10).max(1),
        // 正常量级的裁剪阈值：扫描里模型很小，梯度偶发尖峰，不裁会把某个点跑飞
        grad_clip: 1.0,
        // 只在最后评估一次：多个评估点会让"取最优"这件事在不同规模间引入不可比的偏置
        eval_every: steps.max(1),
        eval_iters: 8,
        log_file: None,
        early_stop_patience: 0,
        ..TrainConfig::default()
    }
}

/// 按 `(层数, 宽度)` 造出扫描用的模型配置（块大小与头数按约束修正）
pub fn scan_model_config(
    base: &TransformerConfig,
    n_layer: usize,
    n_embd: usize,
    block_size: usize,
) -> TransformerConfig {
    let n_head = pick_n_head(n_embd, base.n_head.max(1));
    let mut cfg = base.clone();
    cfg.vocab_size = base.vocab_size;
    cfg.n_layer = n_layer;
    cfg.n_embd = n_embd;
    cfg.n_head = n_head;
    cfg.block_size = block_size;
    // GQA 的 KV 头数不能超过 Q 头数
    if cfg.n_kv_head > n_head || n_head % cfg.n_kv_head.max(1) != 0 {
        cfg.n_kv_head = 0; // 退回标准 MHA
    }
    cfg
}

/// **参数量扫描**：固定 token 预算，逐级放大模型，测 loss 随 N 的幂律关系。
///
/// 这是 IsoFLOP 剖面的"同 token 预算"版本（严格来说 IsoFLOP 要求 `6ND` 相等，即
/// 模型变大时步数要减少）。这里固定 token 数、不固定算力，是因为在本项目的算力尺度上
/// 固定算力会让最小模型只跑几步、噪声盖过信号。差异在文档里写明。
///
/// `sizes` 是 `(层数, 宽度)` 列表。返回按给定顺序排列的实测点。
pub fn params_scan(
    base: &TransformerConfig,
    sizes: &[(usize, usize)],
    docs: &[String],
    val_text: Option<&str>,
    tokenizer: &Tokenizer,
    sc: &ScanConfig,
) -> Vec<ScanPoint> {
    assert!(!sizes.is_empty(), "扫描至少要给一个规模");
    // 语料只分词一次：DataLoader 只取决于 block/batch 与文档，与模型规模无关，
    // 放在循环里会让每个规模都重新分词整份语料（这一步比训练本身还慢）。
    let loader =
        DataLoader::from_documents(docs, val_text, tokenizer, sc.block_size, sc.batch_size);
    let mut points = Vec::with_capacity(sizes.len());
    for &(n_layer, n_embd) in sizes {
        let cfg = scan_model_config(base, n_layer, n_embd, sc.block_size);
        let mut rng = Rng::new(sc.seed);
        let model = Transformer::new(cfg.clone(), &mut rng);
        // 参数口径必须与真实建层逐位一致：两处公式一旦漂移，6ND 与 Chinchilla 表就全错了。
        // 放在这里而不是只放测试里，是因为扫描本身就要用实测参数量。
        let measured: usize = model.parameters().iter().map(|p| p.numel()).sum();
        assert_eq!(
            measured,
            params_total(&cfg),
            "参数口径公式与真实建层不一致（实测 {measured} vs 公式 {}）",
            params_total(&cfg)
        );
        let tcfg = scan_train_config(sc, sc.steps);
        logln!(
            "[scan] N 扫描 {n_layer}x{n_embd}：非嵌入参数 {} | {} 步 × {} token = {:.3}M token",
            params_non_embedding(&cfg),
            sc.steps,
            sc.batch_size * sc.block_size,
            sc.tokens_per_point(sc.steps) / 1e6,
        );
        let loss = train_transformer(&model, tokenizer, &loader, &tcfg, None, None, &mut rng);
        points.push(ScanPoint {
            label: format!("{n_layer}x{n_embd}"),
            n_layer,
            n_embd,
            params: measured,
            params_non_embedding: params_non_embedding(&cfg),
            steps: sc.steps,
            tokens: sc.tokens_per_point(sc.steps),
            loss: loss as f64,
        });
    }
    points
}

/// **数据量扫描**：固定模型，按 `multiples` 放大训练 token 数，测 loss 随 D 的幂律关系。
///
/// 对应文档的"过训练分析"练习：数据量翻倍带来的 loss 下降是递减的（幂律），
/// 从曲线上就能看出"再堆数据还值不值"。
pub fn tokens_scan(
    cfg: &TransformerConfig,
    multiples: &[usize],
    docs: &[String],
    val_text: Option<&str>,
    tokenizer: &Tokenizer,
    sc: &ScanConfig,
) -> Vec<ScanPoint> {
    assert!(!multiples.is_empty(), "数据量扫描至少要给一个倍数");
    assert!(multiples.iter().all(|&k| k >= 1), "数据量倍数必须 >= 1");
    let mut rng = Rng::new(sc.seed);
    let model = Transformer::new(cfg.clone(), &mut rng);
    let measured: usize = model.parameters().iter().map(|p| p.numel()).sum();
    assert_eq!(measured, params_total(cfg));
    let loader =
        DataLoader::from_documents(docs, val_text, tokenizer, sc.block_size, sc.batch_size);
    let mut points = Vec::with_capacity(multiples.len());
    for &k in multiples {
        let steps = sc.steps * k;
        let tcfg = scan_train_config(sc, steps);
        logln!(
            "[scan] D 扫描 ×{k}：{} 步 × {} token = {:.3}M token（模型 {}(非嵌入)）",
            steps,
            sc.batch_size * sc.block_size,
            sc.tokens_per_point(steps) / 1e6,
            params_non_embedding(cfg),
        );
        let loss = train_transformer(&model, tokenizer, &loader, &tcfg, None, None, &mut rng);
        points.push(ScanPoint {
            label: format!("x{k}"),
            n_layer: cfg.n_layer,
            n_embd: cfg.n_embd,
            params: measured,
            params_non_embedding: params_non_embedding(cfg),
            steps,
            tokens: sc.tokens_per_point(steps),
            loss: loss as f64,
        });
    }
    points
}

/// 用扫描点拟合 `loss vs 非嵌入参数量` 的幂律
pub fn fit_over_params(points: &[ScanPoint]) -> PowerLaw {
    let xs: Vec<f64> = points.iter().map(|p| p.params_non_embedding as f64).collect();
    let ys: Vec<f64> = points.iter().map(|p| p.loss as f64).collect();
    fit_power_law(&xs, &ys)
}

/// 用扫描点拟合 `loss vs 训练 token 数` 的幂律
pub fn fit_over_tokens(points: &[ScanPoint]) -> PowerLaw {
    let xs: Vec<f64> = points.iter().map(|p| p.tokens).collect();
    let ys: Vec<f64> = points.iter().map(|p| p.loss as f64).collect();
    fit_power_law(&xs, &ys)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 幂律拟合要能把合成数据的 `α` 与不可约损失 `b` 反解回来。
    ///
    /// 用 `L = 2.5·N^-0.34 + 1.2` 造点（指数取 Chinchilla 的 α），跨两个数量级。
    /// 若实现里漏掉 `b`（直接对 `log L` 做最小二乘），斜率会被 `b` 稀释，
    /// `α` 会明显偏小——这条断言就是钉住这一点的。
    #[test]
    fn test_fit_power_law_recovers_known_exponent() {
        let truth = |n: f64| 2.5 * n.powf(-0.34) + 1.2;
        let xs: Vec<f64> = vec![1e6, 3e6, 1e7, 3e7, 1e8, 3e8, 1e9];
        let ys: Vec<f64> = xs.iter().map(|&n| truth(n)).collect();
        let fit = fit_power_law(&xs, &ys);

        assert_eq!(fit.n_points, 7);
        assert!(
            (fit.alpha - 0.34).abs() < 0.01,
            "α 应恢复到 0.34 附近，实际 {:.4}",
            fit.alpha
        );
        assert!(
            (fit.b - 1.2).abs() < 0.02,
            "不可约损失应恢复到 1.2 附近，实际 {:.4}",
            fit.b
        );
        assert!(fit.r2 > 0.9999, "完美幂律数据的 r² 应接近 1，实际 {:.6}", fit.r2);
        // 外推：在没给过的规模上也要准
        let pred = fit.predict(5e8);
        assert!(
            (pred - truth(5e8)).abs() < 1e-3,
            "外推误差过大：{pred:.4} vs {:.4}",
            truth(5e8)
        );
    }

    /// 不可约损失为 0 的纯幂律（`L = 8·D^-0.28`）也要拟合得出来，
    /// 且 `b` 不能被硬凑成一个正数（网格搜索的下界就是 b = 0）。
    #[test]
    fn test_fit_power_law_with_zero_irreducible_loss() {
        let xs: Vec<f64> = vec![1e9, 2e9, 4e9, 8e9, 1.6e10];
        let ys: Vec<f64> = xs.iter().map(|&d| 8.0 * d.powf(-0.28)).collect();
        let fit = fit_power_law(&xs, &ys);
        assert!(
            (fit.alpha - 0.28).abs() < 0.02,
            "α 应恢复到 0.28 附近，实际 {:.4}",
            fit.alpha
        );
        assert!(fit.b >= 0.0 && fit.b < 0.01, "b 应贴住 0，实际 {:.6}", fit.b);
        assert!(fit.r2 > 0.999, "r² 应接近 1，实际 {:.6}", fit.r2);
    }

    /// 参数口径：公式算出的参数量必须与真实建层的 `parameters()` 逐位相等
    /// （含 GQA、RMSNorm、SwiGLU 三种现代配置；嵌入只算一份因为输出头与之共享）。
    #[test]
    fn test_param_accounting_matches_real_model() {
        let configs = [
            TransformerConfig {
                vocab_size: 512,
                n_embd: 64,
                n_head: 4,
                n_layer: 2,
                block_size: 32,
                ..TransformerConfig::default()
            },
            // LLaMA 风格：RMSNorm + SwiGLU + GQA
            TransformerConfig {
                vocab_size: 512,
                n_embd: 128,
                n_head: 8,
                n_kv_head: 2,
                n_layer: 3,
                block_size: 64,
                use_rmsnorm: true,
                use_swiglu: true,
                dropout: 0.1,
                ..TransformerConfig::default()
            },
        ];
        for cfg in configs {
            let mut rng = Rng::new(7);
            let model = Transformer::new(cfg.clone(), &mut rng);
            let measured: usize = model.parameters().iter().map(|p| p.numel()).sum();
            assert_eq!(
                measured,
                params_total(&cfg),
                "配置 {cfg:?} 的参数口径不一致：实测 {measured} vs 公式 {}",
                params_total(&cfg)
            );
            // 非嵌入 = 全部 - 词嵌入（输出头与之共享，不再另算）
            assert_eq!(
                params_non_embedding(&cfg),
                measured - cfg.vocab_size * cfg.n_embd,
                "非嵌入口径不一致"
            );
        }
    }

    /// `C ≈ 6ND`、20:1 法则、以及"算力翻倍 ⇒ 规模涨 √2 倍"这三条要同时成立。
    #[test]
    fn test_ratio20_allocation_invariants() {
        for &c in &[1e18f64, 1e20, 1e22, 5.88e23] {
            let a = ratio20_optimal(c);
            let c_back = flops_train(a.params, a.tokens);
            assert!(
                (c_back - c).abs() / c < 1e-9,
                "C = 6ND 必须精确回代：{c_back} vs {c}"
            );
            assert!(
                (a.tokens_per_param() - 20.0).abs() < 1e-9,
                "20:1 法则被破坏：{:.4}",
                a.tokens_per_param()
            );
        }
        // 论文那条 70B / 1.4T 的算力预算，反解回来的最优配比同样落在 20:1 上
        let chinchilla = ratio20_optimal(flops_train(7.0e10, 1.4e12));
        assert!(
            (chinchilla.tokens_per_param() - 20.0).abs() < 1e-9,
            "Chinchilla 70B/1.4T 应落在 20:1 线上"
        );
        assert!(
            (chinchilla.tokens / 1.4e12 - 1.0).abs() < 1e-9,
            "由算力反解出的数据量应就是 1.4T，实际 {:.3e}",
            chinchilla.tokens
        );
        // 算力翻倍：参数量与数据量各涨 √2 ≈ 1.414（Chinchilla 的核心修正）
        let a1 = ratio20_optimal(1e22);
        let a2 = ratio20_optimal(2e22);
        let ratio = a2.params / a1.params;
        assert!(
            (ratio - 2.0f64.sqrt()).abs() < 1e-6,
            "算力翻倍时规模应涨 √2，实际 {ratio:.6}"
        );
        assert!((a2.tokens / a1.tokens - 2.0f64.sqrt()).abs() < 1e-6);
    }

    /// 文档 `docs/31-Scaling-Laws.md` 那张 Chinchilla 表必须逐行自洽：
    /// `C = 6ND`（±3%）且 `D/N = 20`。
    #[test]
    fn test_chinchilla_table_rows_are_consistent() {
        for &(c, n, d) in &CHINCHILLA_TABLE {
            let c_calc = flops_train(n, d);
            assert!(
                (c_calc - c).abs() / c < 0.03,
                "表中算力列与 6ND 不符：{c:.3e} vs {c_calc:.3e}"
            );
            assert!(
                (d / n - 20.0).abs() < 0.05,
                "表中 D/N 应为 20，实际 {:.3}",
                d / n
            );
        }
        // 表是单调的：算力/规模/数据量都逐行递增
        for w in CHINCHILLA_TABLE.windows(2) {
            assert!(w[1].1 > w[0].1 && w[1].2 > w[0].2, "表必须逐行递增");
        }
    }

    /// 参数化最优解是**真的**最小：把闭式解与网格暴力搜索对比，并且它必须不比 20:1 配比更差。
    ///
    /// 这条同时把"两条路线给出的最优规模不一样"这件事钉下来——不是 bug，是常数本身的性质。
    #[test]
    fn test_parametric_optimal_matches_grid_search() {
        let c = 1e22f64;
        let closed = parametric_optimal(c);
        // 网格暴力搜索：在 [1e7, 1e12] 的参数量上取对数均匀网格
        let mut best = (f64::INFINITY, 0.0f64);
        for i in 0..=20000 {
            let n = 1e7 * (1e12f64 / 1e7).powf(i as f64 / 20000.0);
            let d = c / (FLOPS_PER_PARAM_TOKEN * n);
            let l = chinchilla_loss(n, d);
            if l < best.0 {
                best = (l, n);
            }
        }
        assert!(
            closed.loss <= best.0 + 1e-4,
            "闭式解应不劣于网格搜索：{:.6} vs {:.6}",
            closed.loss,
            best.0
        );
        // 闭式解与网格解的规模一致（同一个极小点）
        assert!(
            (closed.params / best.1).ln().abs() < 0.01,
            "闭式解与网格解的 N 应一致：{:.4e} vs {:.4e}",
            closed.params,
            best.1
        );
        // 闭式解严格优于（或等于）20:1 的配比：它是同一个损失函数的最小值
        let ratio20 = ratio20_optimal(c);
        assert!(
            closed.loss <= ratio20.loss + 1e-9,
            "参数化最优（{:.6}）不该比 20:1（{:.6}）更差",
            closed.loss,
            ratio20.loss
        );
        // 但两者的最优规模确实不同：这就是「三类方法不一致」在数字上的体现
        assert!(
            (closed.params / ratio20.params - 1.0).abs() > 0.2,
            "两条路线的最优规模本应明显不同（{:.3e} vs {:.3e}）",
            closed.params,
            ratio20.params
        );
        // 且闭式解满足 C = 6ND
        assert!((flops_train(closed.params, closed.tokens) / c - 1.0).abs() < 1e-9);
    }

    /// 文档里的算力估算示例（LLaMA-7B：7B 参数 × 1T token，A100×64 / MFU 0.4）
    /// 必须复现出"约 61 天、约 3700 美元"。
    #[test]
    fn test_wall_clock_matches_doc_example() {
        let c = flops_train(7.0e9, 1.0e12);
        assert!((c - 4.2e22).abs() / 4.2e22 < 1e-9, "6ND = {c:.3e}，应为 4.2e22");
        let hw = Hardware::default();
        let wc = hw.estimate(c);
        assert!(
            (wc.days - 61.0).abs() < 1.0,
            "训练天数应约 61 天，实际 {:.2}",
            wc.days
        );
        assert!(
            (wc.cost_usd - 3700.0).abs() < 200.0,
            "电费应约 3700 美元，实际 {:.0}",
            wc.cost_usd
        );
        // MFU 越高越快：这条比例关系是估算公式的唯一自由度
        let faster = Hardware { mfu: 0.8, ..hw }.estimate(c);
        assert!((faster.days - wc.days / 2.0).abs() < 1e-6);
    }

    /// 过训练：数据量放大 k 倍（模型相应缩小）时，loss 先降后升 ——
    /// 极小点不在 20:1 上，而在参数化模型给出的更靠"过训练"的一侧。
    /// 这正是实践里"故意过训练小模型"能把同等算力用得更值的区间。
    #[test]
    fn test_overtrain_curve_has_interior_minimum() {
        let c = 1e22f64;
        let base = ratio20_optimal(c);
        // k = 1 就是 20:1 本身
        let k1 = overtrain(c, 1.0);
        assert!((k1.params / base.params - 1.0).abs() < 1e-9);
        assert!((k1.loss - base.loss).abs() < 1e-9);
        // 数据量放大 ⇒ 模型缩小（C = 6ND 固定）
        let k4 = overtrain(c, 4.0);
        assert!((k4.params - base.params / 4.0).abs() / base.params < 1e-9);
        assert!((k4.tokens - base.tokens * 4.0).abs() / base.tokens < 1e-9);
        // 极小点在区间内部：两头（k=0.25 与 k=16）都不如中间
        let inner = overtrain(c, 3.0).loss;
        assert!(
            inner < overtrain(c, 0.25).loss && inner < overtrain(c, 16.0).loss,
            "过训练曲线应有内部极小点：k=0.25 {:.4} / k=3 {:.4} / k=16 {:.4}",
            overtrain(c, 0.25).loss,
            inner,
            overtrain(c, 16.0).loss
        );
        // 论文的过训练案例：LLaMA-3 8B 训 15T token，Chinchilla 最优只要约 1.6e11，
        // 也就是 94 倍过训练
        let llama3 = optimal_for_params(8.0e9);
        assert!(
            (llama3.tokens / 1.6e11 - 1.0).abs() < 0.02,
            "8B 的 Chinchilla 最优数据量应在 1.6e11 量级，实际 {:.3e}",
            llama3.tokens
        );
        let ratio = 1.5e13 / llama3.tokens;
        assert!(
            (ratio - 94.0).abs() < 2.0,
            "LLaMA-3 的过训练倍数应约 94×，实际 {ratio:.1}×"
        );
    }

    /// 实测扫描：更大的模型在同一 token 预算下必须拿到更低的 loss，
    /// 且拟合出的幂律指数为正、不可约损失低于所有实测点。
    #[test]
    fn test_params_scan_larger_models_reach_lower_loss() {
        let corpus = "the quick brown fox jumps over the lazy dog. \
                      a fast brown fox leaps across the sleepy hound. \
                      the lazy dog sleeps while the quick fox runs away. \
                      quick foxes and lazy dogs are friends in this little tale. "
            .repeat(12);
        let tokenizer = Tokenizer::char(&corpus);
        let sc = ScanConfig {
            steps: 150,
            batch_size: 4,
            block_size: 32,
            max_lr: 5e-3,
            seed: 3,
        };
        let base = TransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            // 固定经典风格（LayerNorm + GELU）：本测验证的是 scaling 律本身，
            // 不让归一化/激活的架构变量干扰 150 步 toy 训练下的 loss 排序
            use_rmsnorm: false,
            use_swiglu: false,
            ..TransformerConfig::default()
        };
        let sizes = [(1usize, 32usize), (2, 64), (4, 128)];
        let points = params_scan(&base, &sizes, &[corpus.clone()], None, &tokenizer, &sc);

        assert_eq!(points.len(), 3);
        // 固定 token 预算：每个点见到的 token 数必须完全相同
        let tokens = points[0].tokens;
        assert!(
            points.iter().all(|p| (p.tokens - tokens).abs() < 1e-9),
            "固定 token 预算被破坏：{tokens} vs {:?}",
            points.iter().map(|p| p.tokens).collect::<Vec<_>>()
        );
        // 参数量逐级上升
        for w in points.windows(2) {
            assert!(
                w[1].params_non_embedding > w[0].params_non_embedding,
                "规模没有递增：{} vs {}",
                w[1].params_non_embedding,
                w[0].params_non_embedding
            );
            assert!(
                w[1].loss < w[0].loss,
                "更大的模型 {} 的 loss（{:.4}）应低于 {}（{:.4}）",
                w[1].label,
                w[1].loss,
                w[0].label,
                w[0].loss
            );
        }
        // 幂律拟合：指数为正、不可约损失不超过最小实测 loss
        let fit = fit_over_params(&points);
        assert!(fit.alpha > 0.0, "拟合出的 α 必须为正，实际 {:.4}", fit.alpha);
        let min_loss = points.iter().map(|p| p.loss).fold(f64::INFINITY, f64::min);
        assert!(
            fit.b <= min_loss + 1e-6,
            "不可约损失（{:.4}）不能高于最小实测 loss（{:.4}）",
            fit.b,
            min_loss
        );
        assert!(fit.r2 > 0.99, "r² 应接近 1，实际 {:.4}", fit.r2);
        // 规模翻倍换来的 loss 下降为正（回报递减但仍在下降）
        assert!(fit.loss_drop_per_doubling(points[0].params_non_embedding as f64) > 0.0);
    }

    /// 数据量扫描：token 数单调递增、算力按 6ND 跟着涨，loss 单调下降。
    #[test]
    fn test_tokens_scan_is_monotone() {
        let corpus = "red green blue yellow black white purple orange. \
                      the garden is full of flowers and the river runs nearby. "
            .repeat(16);
        let tokenizer = Tokenizer::char(&corpus);
        let sc = ScanConfig {
            steps: 20,
            batch_size: 4,
            block_size: 32,
            max_lr: 5e-3,
            seed: 5,
        };
        let cfg = TransformerConfig {
            vocab_size: tokenizer.vocab_size(),
            ..TransformerConfig::default()
        };
        let points = tokens_scan(&cfg, &[1, 2, 4], &[corpus.clone()], None, &tokenizer, &sc);
        assert_eq!(points.len(), 3);
        for w in points.windows(2) {
            assert!(
                w[1].tokens > w[0].tokens,
                "token 数应递增：{} vs {}",
                w[1].tokens,
                w[0].tokens
            );
            assert!(w[1].loss < w[0].loss, "数据更多时 loss 应更低：{:.4} vs {:.4}", w[1].loss, w[0].loss);
        }
        // 同一模型的 6ND：token 翻倍 ⇒ 算力翻倍
        assert!((points[1].compute() / points[0].compute() - 2.0).abs() < 1e-9);
        let fit = fit_over_tokens(&points);
        assert!(fit.alpha > 0.0, "数据量的幂律指数必须为正，实际 {:.4}", fit.alpha);
    }
}
