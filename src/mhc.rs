//! mHC：流形约束超连接（Manifold-Constrained Hyper-Connections，第 47 课）
//!
//! ## 从"一条残差流"到"多条残差流"
//!
//! 从 ResNet 到 Transformer，主干一直是**一条**残差流：
//!
//! ```text
//! x ← x + Sublayer(Norm(x))
//! ```
//!
//! 信息沿着唯一的一条通道往前流，层与层之间只有"加法"这一种耦合。**超连接
//! （Hyper-Connections, HC）** 把这条通道拓宽成 `n` 条**并行残差流** `X₀..X_{n-1}`，
//! 每个子层不再直接读写同一条流，而是：
//!
//! ```text
//! 读出：h = Σ_s α_s · X_s                  （α 在单纯形上，Σα = 1）
//! 计算：y = Sublayer(Norm(h))
//! 写入：X_s' = Σ_t A_{s,t} · X_t + β_s · y   （A 是 n×n 混合矩阵，β 在单纯形上）
//! ```
//!
//! 读出用 `α` 把多条流压成一条喂给子层；写入用矩阵 `A` 在**流之间**重新混合，
//! 再用 `β` 把子层的输出按不同强度注入各条流。`n = 1` 时 `α = A = β = 1`，
//! 上面三行就退化成 `x ← x + Sublayer(Norm(x))`——也就是普通残差。
//!
//! ## mHC 约束了什么：双随机矩阵
//!
//! 放开矩阵 `A` 去学，很快会失控：`A` 的元素可以任意大，前向一遍遍乘上去，
//! 信号指数爆炸（或塌缩）。**mHC（DeepSeek-V4）** 的做法是把 `A` 约束到
//! **双随机矩阵**（Birkhoff 多面体）里：
//!
//! ```text
//! A ≥ 0,  A·1 = 1（每行和为 1）,  1ᵀ·A = 1（每列和为 1）
//! ```
//!
//! 这个约束有两个直接好处：
//!
//! 1. **信号不放大**。双随机矩阵是置换矩阵的凸组合（Birkhoff–von Neumann 定理），
//!    而谱范数 ‖·‖₂ 是凸函数、置换矩阵的谱范数恰好为 1，于是
//!    `‖A‖₂ ≤ Σ λᵢ‖Pᵢ‖₂ = 1`；又因为 `A·1 = 1` 给出特征值 1，所以 `‖A‖₂ = 1`。
//!    每一步混合都**恰好保范**，既不放也不缩。这就是"流形约束"的来历——
//!    把可学参数限制在一个能保证数值稳定的**流形**（多面体）上。
//! 2. **保均值**。`Σ_s X_s'` 里的混合部分等于 `1ᵀ A X = 1ᵀ X`，列的归一化让
//!    "所有流的和"在混合前后不变，残差流的整体尺度不会漂移。
//!
//! 参数化上，我们让 `A` 由一份**自由矩阵**经 **Sinkhorn–Knopp 迭代**投影得到
//! （行、列交替归一化，正矩阵一定收敛到双随机矩阵），因此 `A` 永远合法，
//! 可学参数却不受约束——这是"约束流形 + 自由参数化"的常用技巧。
//!
//! ## 为什么这里全用矩阵乘
//!
//! `read` / `write` 把"多条流"看成一个大矩阵 `S ∈ R^{n×D}`（`D = b·t·d`）：
//!
//! - 读出 `αᵀ S` 是一次 `[1,n]×[n,D]` 的矩阵乘；
//! - 写入 `A S + β⊗y` 是一次 `[n,n]×[n,D]` 的矩阵乘加一次 `[n,1]×[1,D]` 的广播乘。
//!
//! 这样写的好处是**全部走已有的、带反向的算子**：`α`、`β`、`A` 都能拿到梯度，
//! 不必为"逐流相加"另写一套广播/求和内核。
//!
//! ## 默认关闭，逐位不变
//!
//! `n_stream = 1`（默认）时 [`crate::model::TransformerBlock`] **根本不构造**本模块，
//! 残差路径仍是原来那一行 `x.add(&h)`——不消耗随机数、不新增参数、行为与前向逐位一致。
//! 即便手动构造 `HyperConnection::new(1, _)`，softmax(单元素) 与 Sinkhorn(1×1)
//! 都精确等于 1，读出/写入也与普通残差**逐位相同**（见本文件单测）。

use crate::tensor::Tensor;

/// 初始化时给"读出 / 写入 / 深度对角"三者的 logit 偏置。
///
/// `softmax([4, 0, 0, …])` 的首元素约为 0.982、其余约 0.018/个：让 `n > 1` 起步时
/// 首条流是"主残差流"、其余流只是轻微耦合，整体行为接近普通残差，训练更稳。
/// 直接用 0（softmax 后均匀分布）会把残差项摊薄成 `y/n`，起步阶段信号明显偏弱。
const INIT_BIAS: f32 = 4.0;

/// 输入展开时**每条流**的静态增益步长：流 `s` 的增益是 `1 + s·STEP`（第 0 条恒为 1）。
///
/// 为什么需要它：如果 `n` 条流在入口处逐位相同，第一个子层的读出
/// `h = Σ α_s X_s` 就只依赖 `Σα_s = 1` 而与 `α` 的具体取值无关 —— 于是
/// `hc_attn.alpha_raw` / `depth_raw` 在**第一层永远拿不到梯度**，成了死参数。
/// 给每条流一个互不相同的确定性增益（不加参数、不耗随机数）即可立刻打破对称性。
/// 增益是"静态宽度连接"的最简形态；`n = 1` 时增益为 1，行为逐位不变。
const STREAM_GAIN_STEP: f32 = 0.01;

/// 单侧集中的 logit 初始化：首位 [`INIT_BIAS`]、其余为 0。
///
/// softmax 后约为 `[0.982, 0.018, …]`——让首条流充当"主残差流"，其余流轻微耦合，
/// 起步时整体行为接近普通残差。用于读出 / 写入 / 收敛三处门控。
pub fn concentrated_logits(n: usize) -> Vec<f32> {
    let mut v = vec![0.0f32; n];
    if n > 0 {
        v[0] = INIT_BIAS;
    }
    v
}

/// Sinkhorn–Knopp 迭代：把非负方阵投影到**双随机矩阵**（行、列和都为 1）。
///
/// 输入按行另存一份（`raw.len() == n*n`，行主序），先逐行归一到 1，再交替
/// "按列归一化 → 按行归一化" `iters` 轮。正矩阵在这个迭代下必收敛到双随机矩阵
/// （Sinkhorn–Knopp 定理）；对角阵、置换矩阵这类本来就双随机的输入是不动点。
///
/// 这是给文档与单测用的 **f32 参考实现**；训练路径走张量版的
/// [`HyperConnection::depth`]（同样交替归一化，但每一步都在计算图里，可反向）。
#[allow(dead_code)] // 生产路径无调用点：作为分步参考实现与数值口径基准保留给单测/文档
pub fn sinkhorn(raw: &[f32], n: usize, iters: usize) -> Vec<f32> {
    assert_eq!(raw.len(), n * n, "sinkhorn 需要 n×n 输入（n={n}）");
    assert!(n >= 1);
    // 先按行归一化，保证矩阵非负且每行和为 1
    let mut m = vec![0.0f32; n * n];
    for i in 0..n {
        let mut s = 0.0;
        for j in 0..n {
            s += raw[i * n + j].max(0.0);
        }
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for j in 0..n {
            m[i * n + j] = raw[i * n + j].max(0.0) * inv;
        }
    }
    for _ in 0..iters {
        // 列归一化
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += m[i * n + j];
            }
            if s > 0.0 {
                for i in 0..n {
                    m[i * n + j] /= s;
                }
            }
        }
        // 行归一化
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += m[i * n + j];
            }
            if s > 0.0 {
                for j in 0..n {
                    m[i * n + j] /= s;
                }
            }
        }
    }
    m
}

/// 矩阵离"双随机"多远：返回 max(所有 |行和 − 1|、|列和 − 1| 的最大值)。
#[allow(dead_code)] // 数值口径基准，供单测/文档核对，不进训练路径
pub fn doubly_stochastic_error(mat: &[f32], n: usize) -> f32 {
    assert_eq!(mat.len(), n * n);
    let mut worst = 0.0f32;
    for i in 0..n {
        let s: f32 = (0..n).map(|j| mat[i * n + j]).sum();
        worst = worst.max((s - 1.0).abs());
    }
    for j in 0..n {
        let s: f32 = (0..n).map(|i| mat[i * n + j]).sum();
        worst = worst.max((s - 1.0).abs());
    }
    worst
}

/// 幂迭代估计谱范数 `‖A‖₂`（对 `AᵀA` 做幂迭代，取最大特征值的平方根）。
///
/// 双随机矩阵的这个值应当恰好是 1：`≤ 1` 由 Birkhoff 凸性给出、`≥ 1` 由
/// `A·1 = 1` 给出。单测用它验证 Sinkhorn 真的把混合矩阵钉在了保范流形上。
#[allow(dead_code)] // 数值口径基准，供单测/文档核对，不进训练路径
pub fn spectral_norm(mat: &[f32], n: usize, iters: usize) -> f32 {
    assert_eq!(mat.len(), n * n);
    if n == 0 {
        return 0.0;
    }
    // 初始向量取 1/√n —— 它已经是双随机矩阵对应 σ=1 的右奇异向量，
    // 迭代会立刻稳定；若真实 σ_max > 1，幂迭代会把它放大出来。
    let mut v = vec![1.0f32 / (n as f32).sqrt(); n];
    let mut sigma = 1.0f32;
    for _ in 0..iters {
        // w = A v
        let mut w = vec![0.0f32; n];
        for i in 0..n {
            let mut s = 0.0;
            for j in 0..n {
                s += mat[i * n + j] * v[j];
            }
            w[i] = s;
        }
        // u = Aᵀ w
        let mut u = vec![0.0f32; n];
        for j in 0..n {
            let mut s = 0.0;
            for i in 0..n {
                s += mat[i * n + j] * w[i];
            }
            u[j] = s;
        }
        // Rayleigh 商 vᵀ(AᵀA)v / vᵀv，开方即 σ 的估计
        let num: f32 = (0..n).map(|j| v[j] * u[j]).sum();
        let den: f32 = (0..n).map(|j| v[j] * v[j]).sum();
        sigma = (num / den).max(0.0).sqrt();
        let norm: f32 = u.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm <= 1e-12 {
            break;
        }
        for j in 0..n {
            v[j] = u[j] / norm;
        }
    }
    sigma
}

/// 把 `[b, t, d]` 的输入复制成 `n` 条并行流（每条再乘一个静态增益），
/// 堆成一个 `[n, b·t·d]` 的矩阵。
///
/// 用一次 `[n,1] × [1,D]` 的矩阵乘完成复制（每行都是同一份展平输入），
/// 好处是反向时各流梯度会**自动加和**回输入——正是"多条流同源"该有的梯度语义。
/// 增益见 [`STREAM_GAIN_STEP`]：`n = 1` 时是 1，逐位不变。
pub fn expand_streams(x: &Tensor, n: usize) -> Tensor {
    let d_tot = x.numel();
    let flat = x.reshape(vec![1, d_tot]);
    let ones = Tensor::from_vec(vec![1.0; n], vec![n, 1]);
    let gains: Vec<f32> = (0..n)
        .map(|s| 1.0 + s as f32 * STREAM_GAIN_STEP)
        .collect();
    let g = Tensor::from_vec(gains, vec![n, 1]);
    ones.matmul(&flat).mul(&g)
}

/// 收敛：用**可学**的宽度连接 `γ`（softmax → 单纯形）把 `[n, b·t·d]` 的多条流
/// 合成一条 `[b, t, d]`。
///
/// 为什么是学出来的而不是简单取平均：均匀平均会让最后一个子层的写入参数
/// （`β`、`A`）**永久失去梯度**——`mean_s(Σ_t A_{s,t}X_t) = Σ_t X_t/n` 只用到
/// "列和为 1"，`mean_s(β_s y) = y/n` 只用到 "β 和为 1"，两者都与具体取值无关。
/// 用一个非均匀的 `γ` 加权求和，`Σ_s γ_s β_s` 就重新依赖 `β`，参数才能被训练。
/// `γ` 的初始化同样用 [`concentrated_logits`]（首条流为主）。
pub fn collapse_streams(
    streams: &Tensor,
    gamma_raw: &Tensor,
    n: usize,
    b: usize,
    t: usize,
    d: usize,
) -> Tensor {
    let gamma = gamma_raw.softmax_last_dim().reshape(vec![1, n]);
    gamma.matmul(streams).reshape(vec![b, t, d])
}

/// 一个子层配套的超连接参数（读出门控 `α`、写入强度 `β`、深度混合矩阵 `A`）。
///
/// 每个 `TransformerBlock` 的两个子层（注意力、前馈）各持一份，互不共享。
pub struct HyperConnection {
    /// 并行流数量 `n`
    n: usize,
    /// Sinkhorn 迭代次数
    iters: usize,
    /// 读出 logits，softmax 后是单纯形上的 `α`
    alpha_raw: Tensor,
    /// 写入 logits，softmax 后是单纯形上的 `β`
    beta_raw: Tensor,
    /// 深度混合矩阵的自由参数，行 softmax + Sinkhorn 后是双随机的 `A`
    depth_raw: Tensor,
}

impl HyperConnection {
    /// 构造一份超连接参数。初始化是**确定性**的（不消耗随机数），这样同一随机种子下
    /// 开/关 mHC 的对照实验里，其余所有参数的随机数消耗完全一致，唯一变量就是结构本身。
    pub fn new(n: usize, iters: usize) -> Self {
        assert!(n >= 1, "mHC 至少要有 1 条流");
        let mut depth = vec![0.0f32; n * n];
        for i in 0..n {
            depth[i * n + i] = INIT_BIAS;
        }
        HyperConnection {
            n,
            iters,
            alpha_raw: Tensor::param(concentrated_logits(n), vec![n]),
            beta_raw: Tensor::param(concentrated_logits(n), vec![n]),
            depth_raw: Tensor::param(depth, vec![n, n]),
        }
    }

    /// 读出门控 `α = softmax(alpha_raw)`，形状 `[n]`。
    pub fn alpha(&self) -> Tensor {
        self.alpha_raw.softmax_last_dim()
    }

    /// 写入强度 `β = softmax(beta_raw)`，形状 `[n]`。
    pub fn beta(&self) -> Tensor {
        self.beta_raw.softmax_last_dim()
    }

    /// 深度混合矩阵 `A`：行 softmax 保证非负，再 Sinkhorn 投影到双随机。
    ///
    /// 每一步（`[1,n]×[n,n]` 求列和、`[n,n]×[n,1]` 求行和、广播相除）都在计算图里，
    /// 所以 `depth_raw` 能拿到真实梯度，Sinkhorn 不是"前向才有的装饰"。
    pub fn depth(&self) -> Tensor {
        let ones_row = Tensor::from_vec(vec![1.0; self.n], vec![1, self.n]);
        let ones_col = Tensor::from_vec(vec![1.0; self.n], vec![self.n, 1]);
        let mut a = self.depth_raw.softmax_last_dim();
        for _ in 0..self.iters {
            let col = ones_row.matmul(&a); // [1, n] 列和
            a = a.div(&col);
            let row = a.matmul(&ones_col); // [n, 1] 行和
            a = a.div(&row);
        }
        a
    }

    /// 读出：`h = Σ_s α_s · X_s`。`streams` 形状 `[n, b·t·d]`，返回 `[b, t, d]`。
    pub fn read(&self, streams: &Tensor, b: usize, t: usize, d: usize) -> Tensor {
        let alpha = self.alpha().reshape(vec![1, self.n]);
        alpha.matmul(streams).reshape(vec![b, t, d])
    }

    /// 写入：`X_s' = Σ_t A_{s,t}·X_t + β_s·y`。`streams` 形状 `[n, b·t·d]`，
    /// `y` 形状 `[b, t, d]`，返回新的 `[n, b·t·d]`。
    pub fn write(&self, streams: &Tensor, y: &Tensor) -> Tensor {
        let mixed = self.depth().matmul(streams);
        let beta = self.beta().reshape(vec![self.n, 1]);
        let y_flat = y.reshape(vec![1, y.numel()]);
        mixed.add(&beta.mul(&y_flat))
    }

    /// 本模块的参数（checkpoint 用）
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}.alpha_raw"), self.alpha_raw.clone()),
            (format!("{prefix}.beta_raw"), self.beta_raw.clone()),
            (format!("{prefix}.depth_raw"), self.depth_raw.clone()),
        ]
    }

    /// 本模块的参数（优化器用）
    pub fn parameters(&self) -> Vec<Tensor> {
        vec![
            self.alpha_raw.clone(),
            self.beta_raw.clone(),
            self.depth_raw.clone(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 就在双随机流形上的输入必须是 Sinkhorn 的不动点：单位阵投影后还是单位阵。
    #[test]
    fn test_sinkhorn_identity_fixed_point() {
        let n = 4;
        let mut id = vec![0.0f32; n * n];
        for i in 0..n {
            id[i * n + i] = 1.0;
        }
        let out = sinkhorn(&id, n, 20);
        for (a, b) in out.iter().zip(id.iter()) {
            assert!((a - b).abs() < 1e-6, "单位阵不是不动点：{out:?}");
        }
    }

    /// 任意正矩阵经 Sinkhorn 后行列和都要收敛到 1。
    #[test]
    fn test_sinkhorn_rows_and_cols_sum_to_one() {
        let n = 5;
        // 用一个不对称的正矩阵，确保列归一化真的起了作用
        let raw: Vec<f32> = (0..n * n).map(|k| 0.5 + (k % 7) as f32 * 0.3).collect();
        let out = sinkhorn(&raw, n, 50);
        let err = doubly_stochastic_error(&out, n);
        assert!(err < 1e-4, "双随机误差过大：{err}");
        assert!(out.iter().all(|&v| v >= -1e-6), "出现负元素：{out:?}");
    }

    /// 双随机矩阵的谱范数恰为 1：这正是 mHC"信号不放大"的理论保证。
    #[test]
    fn test_sinkhorn_spectral_norm_is_one() {
        let n = 4;
        let raw: Vec<f32> = (0..n * n).map(|k| 0.2 + ((k * 13) % 11) as f32 * 0.5).collect();
        let out = sinkhorn(&raw, n, 50);
        let sigma = spectral_norm(&out, n, 200);
        assert!(
            (sigma - 1.0).abs() < 1e-3,
            "双随机矩阵谱范数应恰为 1，实测 {sigma}"
        );
    }

    /// 一行独大的自由矩阵，Sinkhorn 会把行/列都拉平；谱范数仍被钉在 1。
    #[test]
    fn test_sinkhorn_mildly_anisotropic_spectral_norm_bounded() {
        let n = 3;
        // 明显非双随机的正矩阵
        let raw = vec![5.0, 0.5, 0.5, 0.5, 5.0, 0.5, 0.5, 0.5, 5.0];
        let out = sinkhorn(&raw, n, 100);
        assert!(doubly_stochastic_error(&out, n) < 1e-4);
        assert!(spectral_norm(&out, n, 300) < 1.0 + 1e-3);
    }

    /// 张量版深度矩阵（走 softmax + Sinkhorn）在初始化处也应是双随机、且谱范数 ≤ 1。
    #[test]
    fn test_depth_tensor_is_doubly_stochastic() {
        let hc = HyperConnection::new(3, 30);
        let a = hc.depth();
        assert_eq!(a.shape(), &[3, 3]);
        let d = a.data_ref();
        let err = doubly_stochastic_error(&d, 3);
        assert!(err < 1e-3, "张量版 Sinkhorn 双随机误差过大：{err}");
        assert!(spectral_norm(&d, 3, 200) < 1.0 + 1e-3);
    }

    /// n = 1 时读出/写入必须**逐位**退化为普通残差。
    #[test]
    fn test_single_stream_degenerates_to_residual_bitwise() {
        let hc = HyperConnection::new(1, 8);
        let x = Tensor::from_vec(vec![0.3, -1.2, 4.0, 0.0, 2.5, -0.7], vec![1, 2, 3]);
        let streams = expand_streams(&x, 1);
        let h = hc.read(&streams, 1, 2, 3);
        // 读出 = 原输入
        for (a, b) in h.data_ref().iter().zip(x.data_ref().iter()) {
            assert_eq!(a, b, "n=1 的读出应与输入逐位相同");
        }
        let y = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 2, 3]);
        let out = hc.write(&streams, &y);
        let gamma = Tensor::from_vec(vec![0.0], vec![1]); // softmax([0]) = [1]，逐位
        let out = collapse_streams(&out, &gamma, 1, 1, 2, 3);
        for (o, (a, b)) in out
            .data_ref()
            .iter()
            .zip(x.data_ref().iter().zip(y.data_ref().iter()))
        {
            assert_eq!(*o, a + b, "n=1 的写入应逐位等于 x + y");
        }
    }

    /// 展开会给每条流乘上互不相同的静态增益（打破对称性的关键）；
    /// 收敛则按可学权重 γ 加权求和。
    #[test]
    fn test_expand_gain_and_collapse_weighted_sum() {
        let x = Tensor::from_vec(vec![1.5, -2.25, 0.75, 8.0, -0.125, 3.5], vec![1, 2, 3]);
        for n in [2usize, 3, 5] {
            let s = expand_streams(&x, n);
            assert_eq!(s.shape(), &[n, 6]);
            {
                let d = s.data_ref();
                for st in 0..n {
                    let gain = 1.0 + st as f32 * STREAM_GAIN_STEP;
                    for (j, xv) in x.data_ref().iter().enumerate() {
                        assert!(
                            (d[st * 6 + j] - gain * xv).abs() < 1e-6,
                            "第 {st} 条流的静态增益不对"
                        );
                    }
                }
            }
            // 均匀 γ（全 0 logits）→ 各流等权相加；因增益不同，结果是"平均增益 × 输入"
            let gamma = Tensor::from_vec(vec![0.0; n], vec![n]);
            let back = collapse_streams(&s, &gamma, n, 1, 2, 3);
            let mean_gain = (0..n)
                .map(|st| 1.0 + st as f32 * STREAM_GAIN_STEP)
                .sum::<f32>()
                / n as f32;
            for (a, b) in back.data_ref().iter().zip(x.data_ref().iter()) {
                assert!(
                    (a - b * mean_gain).abs() < 1e-5,
                    "均匀 γ 的收敛结果不对：{a} vs {}",
                    b * mean_gain
                );
            }
        }
    }

    /// α、β 落在单纯形上（非负、和为 1）。
    #[test]
    fn test_alpha_beta_on_simplex() {
        let hc = HyperConnection::new(4, 20);
        for t in [hc.alpha(), hc.beta()] {
            let d = t.data_ref();
            assert!(d.iter().all(|&v| v >= 0.0), "出现负权重：{:?}", &d[..]);
            let s: f32 = d.iter().sum();
            assert!((s - 1.0).abs() < 1e-6, "权重和不为 1：{s}");
        }
    }

    /// 混合矩阵与门控必须能拿到梯度——否则 Sinkhorn 只是"前向装饰"，参数永远不动。
    ///
    /// 注意损失不能简单取 `out.sum()`：因为 `A` 的列和恒为 1、`β` 的和恒为 1，
    /// 全量求和恰好与两者的具体取值无关，梯度会精确为 0（这本身是个好性质：
    /// 双随机约束让"总能量"守恒）。所以这里用一份非均匀权重逐元素加权后再求和。
    #[test]
    fn test_mixing_params_receive_gradient() {
        let hc = HyperConnection::new(3, 10);
        let x = Tensor::from_vec((0..12).map(|i| (i as f32) * 0.1 - 0.5).collect(), vec![1, 4, 3]);
        let streams = expand_streams(&x, 3);
        let h = hc.read(&streams, 1, 4, 3);
        let y = h.mul_scalar(2.0);
        let out = hc.write(&streams, &y); // [3, 12]
        let w: Vec<f32> = (0..36).map(|i| ((i * 7) % 11) as f32 / 11.0 - 0.5).collect();
        let coef = Tensor::from_vec(w, vec![3, 12]);
        out.mul(&coef).sum().backward();

        let g = hc.depth_raw.grad();
        assert!(
            g.iter().any(|&v| v.abs() > 1e-9),
            "depth_raw 没有收到梯度：{g:?}"
        );
        assert!(hc.alpha_raw.grad().iter().any(|&v| v.abs() > 1e-9));
        assert!(hc.beta_raw.grad().iter().any(|&v| v.abs() > 1e-9));
    }

    /// "保能量"不变式：`A` 的列和为 1，所以整段流经一次混合后，元素总和不变。
    /// 这是双随机约束最直接的数值体现（也是"信号既不放大也不缩小"的账本口径）。
    #[test]
    fn test_doubly_stochastic_preserves_total_energy() {
        let hc = HyperConnection::new(3, 30);
        let x = Tensor::from_vec((0..12).map(|i| (i as f32) * 0.1 - 0.5).collect(), vec![1, 4, 3]);
        let streams = expand_streams(&x, 3);
        let before: f32 = streams.data_ref().iter().sum();
        let after: f32 = hc.depth().matmul(&streams).data_ref().iter().sum();
        let tol = 1e-4 * before.abs().max(1.0);
        assert!(
            (after - before).abs() < tol,
            "双随机混合应保持总和不变：{before} -> {after}"
        );
    }
}
