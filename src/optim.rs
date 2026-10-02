//! 优化器（第 6、17 课）
//!
//! 优化器负责"怎么更新参数"：
//! - SGD（第 6 课）：θ = θ - lr·g，最朴素
//! - AdamW（第 17 课）：自适应学习率 + 动量 + 权重衰减，现代 LLM 标配
//! - Muon（第 46 课）：二维权重的动量矩阵做 Newton–Schulz 正交化，一维参数回退 AdamW

use crate::tensor::Tensor;
use rayon::prelude::*;

/// 优化器统一接口（第 17 课）
///
/// - `step()`：用当前梯度更新一次参数（SGD / AdamW / Muon 各自实现）
/// - `zero_grad()`：清零所有参数梯度（默认实现相同，这里只写一份）
pub trait Optimizer {
    /// 返回被优化的参数列表
    fn params(&self) -> &[Tensor];

    /// 用当前梯度更新一次参数（SGD / AdamW / Muon 各自实现）
    fn step(&mut self);

    /// 清零所有参数的梯度（各优化器共用同一实现）
    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }
}

/// 可存档优化器：在 [`Optimizer`] 之上补齐 checkpoint 续训必需的能力
/// （学习率写入 + 状态导出/恢复）。
///
/// 状态统一用 `(t, m, v)` 三段布局承载，与 [`crate::checkpoint`] 的磁盘格式
/// （参数 / 一阶动量 / 二阶动量三段数据块）一一对应：
/// - [`AdamW`]：`m` / `v` 就是一阶 / 二阶动量；
/// - [`Muon`]：二维参数用 `m` 存正交化**之前**的动量缓冲、`v` 恒为 0（占位以保持三段等长）；
///   一维参数回退 AdamW，`m` / `v` 语义与 AdamW 完全相同。
///
/// 学习率也放进 trait 是因为训练循环每步要用调度器改写它——`dyn` 抽象下访问不到
/// 结构体字段（`train_vqvae` 原本写的是 `opt.lr = ...`）。
pub trait OptimizerState: Optimizer {
    /// 按学习率调度器改写学习率（训练循环每步调用）
    fn set_lr(&mut self, lr: f32);
    /// 导出 `(步数 t, 一阶动量 m, 二阶动量 v)`（借用，不克隆）
    fn state(&self) -> (usize, &[Vec<f32>], &[Vec<f32>]);
    /// 恢复状态（resume 用），长度必须与参数一致
    fn restore_state(&mut self, t: usize, m: Vec<Vec<f32>>, v: Vec<Vec<f32>>);
}

/// 随机梯度下降（SGD，第 6 课）
pub struct SGD {
    lr: f32,
    params: Vec<Tensor>,
}

impl SGD {
    pub fn new(lr: f32, params: Vec<Tensor>) -> Self {
        SGD { lr, params }
    }
}

impl Optimizer for SGD {
    fn params(&self) -> &[Tensor] {
        &self.params
    }

    /// 更新一步：θ = θ - lr * g（原位更新，避免每次克隆整份数据）
    ///
    /// 冻结参数（`requires_grad = false`，LoRA 的主干）直接跳过：它们的梯度槽可能被
    /// GPU 常驻显存路径注入了非零值，不跳过就会被"顺手更新"，冻结就名存实亡了。
    fn step(&mut self) {
        let lr = self.lr;
        // 参数间串行、参数内按元素并行——
        // 大参数（embedding、输出头）占更新量的绝大头
        for p in &self.params {
            if !p.requires_grad() {
                continue;
            }
            let g = p.grad.borrow();
            // decode_mut：bf16 参数按 f32 视图就地更新，退出作用域时统一 encode 回 u16
            //（master weights 语义——更新在 f32 精度下进行，存储仍是 bf16）
            let mut d = p.decode_mut();
            let g_ref: &[f32] = &g;
            d.par_iter_mut()
                .zip(g_ref.par_iter())
                .for_each(|(dv, &gv)| *dv -= lr * gv);
        }
    }
}

/// AdamW（Adam + 权重衰减解耦，第 17 课）
///
/// 核心思想：
/// 1. 一阶动量 m：梯度的指数移动平均（记住"方向"，像小球下坡的惯性）
/// 2. 二阶动量 v：梯度平方的指数移动平均（感知"坡度陡缓"，陡的地方步子小）
/// 3. 偏差修正：训练初期 m、v 从 0 起步，除以 (1-β^t) 修正
/// 4. 权重衰减：每步额外把参数往 0 拉一点（正则化，防止过拟合）
///
/// 冻结参数（`requires_grad = false`）不参与更新：既不做梯度步，也不吃权重衰减。
/// 后者尤其重要——AdamW 的衰减项是 `lr·wd·θ`，与梯度无关，若不跳过，
/// 被冻结的主干权重会每步朝 0 缩一点，LoRA 的"冻结"就只是名义上的。
/// 冻结参数的动量槽仍然分配（保持 `params` / `state()` 与模型参数一一对应，
/// checkpoint 的三段数据块才能等长），只是始终为 0。
pub struct AdamW {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    /// 数值稳定常数：√v_hat + eps 防止除以 0
    eps: f32,
    weight_decay: f32,
    t: usize,
    params: Vec<Tensor>,
    m: Vec<Vec<f32>>, // 一阶动量
    v: Vec<Vec<f32>>, // 二阶动量
}

impl AdamW {
    pub fn new(lr: f32, params: Vec<Tensor>, weight_decay: f32) -> Self {
        Self::new_with_betas(lr, params, weight_decay, 0.9, 0.999)
    }

    /// 可自定义 beta1 / beta2 的构造器（小 batch 或特殊场景需要调优时使用）
    fn new_with_betas(
        lr: f32,
        params: Vec<Tensor>,
        weight_decay: f32,
        beta1: f32,
        beta2: f32,
    ) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        AdamW {
            lr,
            beta1,
            beta2,
            eps: 1e-8,
            weight_decay,
            t: 0,
            params,
            m,
            v,
        }
    }

    /// 导出优化器状态（checkpoint 用）：(步数 t, 一阶动量 m, 二阶动量 v)。
    /// 返回借用而非克隆：动量与参数同量级，保存时没必要再复制一份到内存里。
    pub fn state(&self) -> (usize, &[Vec<f32>], &[Vec<f32>]) {
        (self.t, &self.m, &self.v)
    }

    /// 恢复优化器状态（resume 用），长度必须与参数一致
    pub fn restore_state(&mut self, t: usize, m: Vec<Vec<f32>>, v: Vec<Vec<f32>>) {
        assert_eq!(m.len(), self.params.len(), "动量 m 的参数数量不匹配");
        assert_eq!(v.len(), self.params.len(), "动量 v 的参数数量不匹配");
        for (i, p) in self.params.iter().enumerate() {
            assert_eq!(m[i].len(), p.numel(), "参数 {} 的动量长度不匹配", i);
            assert_eq!(v[i].len(), p.numel(), "参数 {} 的二阶动量长度不匹配", i);
        }
        self.t = t;
        self.m = m;
        self.v = v;
    }
}

impl Optimizer for AdamW {
    fn params(&self) -> &[Tensor] {
        &self.params
    }

    fn step(&mut self) {
        self.t += 1;
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);
        let lr = self.lr;
        let beta1 = self.beta1;
        let beta2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        for i in 0..self.params.len() {
            if !self.params[i].requires_grad() {
                continue; // 冻结参数：不更新、也不做权重衰减（见结构体注释）
            }
            let g = self.params[i].grad.borrow();
            // decode_mut：bf16 参数按 f32 视图就地更新，退出作用域时统一 encode 回 u16
            let mut d = self.params[i].decode_mut();
            let g_ref: &[f32] = &g;
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            // 参数内按元素并行；单元素内的运算顺序与串行版一致，数值不变
            mi.par_iter_mut()
                .zip(vi.par_iter_mut())
                .zip(d.par_iter_mut())
                .zip(g_ref.par_iter())
                .for_each(|(((mv, vv), dv), &gv)| {
                    *mv = beta1 * *mv + (1.0 - beta1) * gv;
                    *vv = beta2 * *vv + (1.0 - beta2) * gv * gv;
                    let m_hat = *mv / bc1;
                    let v_hat = *vv / bc2;
                    let step = lr * m_hat / (v_hat.sqrt() + eps);
                    let decay = lr * wd * *dv;
                    *dv = *dv - step - decay;
                });
        }
    }
}

impl OptimizerState for AdamW {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn state(&self) -> (usize, &[Vec<f32>], &[Vec<f32>]) {
        AdamW::state(self)
    }

    fn restore_state(&mut self, t: usize, m: Vec<Vec<f32>>, v: Vec<Vec<f32>>) {
        AdamW::restore_state(self, t, m, v)
    }
}

// ==================== Muon ====================

/// 行主序稠密矩阵乘 `C[m,n] = A[m,k] · B[k,n]`。
///
/// 只用在本模块内部的 Newton–Schulz 迭代上（不参与自动求导），所以不引张量算子：
/// 那里每个元素都是 `f32` 裸切片，走一次 `Tensor` 反而要额外分配缓存、还不好并行。
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        let arow = &a[i * k..(i + 1) * k];
        let crow = &mut c[i * n..(i + 1) * n];
        for (p, &av) in arow.iter().enumerate() {
            let brow = &b[p * n..(p + 1) * n];
            for (cv, &bv) in crow.iter_mut().zip(brow) {
                *cv += av * bv;
            }
        }
    }
    c
}

/// 行主序转置 `[r, c] → [c, r]`
fn transpose(a: &[f32], r: usize, c: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; r * c];
    for i in 0..r {
        for j in 0..c {
            t[j * r + i] = a[i * c + j];
        }
    }
    t
}

/// Newton–Schulz 迭代：求 `g`（形状 `[rows, cols]`，行主序）的近似**正交极因子**。
///
/// 迭代式 `X ← aX + (bA + cA²)X`（`A = XXᵀ`），常数 `(3.4445, -4.7750, 2.0315)`
/// 是在"不要发散"的前提下收敛最快的 5 阶组合（Keller Jordan 的 Muon 用的就是它）：
/// 它把 `B` 的所有奇异值往 1 推，收敛完 `X ≈ UVᵀ`（`B = UΣVᵀ` 的极因子）。
///
/// 两个关键细节：
/// - **迭代前必须按 Frobenius 范数归一化**：初值最大奇异值 > 1 时迭代会发散成 NaN，
///   归一化把谱范数压到 ≤ 1（`‖X‖₂ ≤ ‖X‖_F = 1`），迭代才是收缩的；
/// - `rows > cols` 时先转置成"矮胖"矩阵：中间量 `A = XXᵀ` 的规模由**较小那一维**决定，
///   转置能让它从 `[rows, rows]` 缩到 `[cols, cols]`，省掉一大块算力。
fn zeropower_newton_schulz(g: &[f32], rows: usize, cols: usize, steps: usize) -> Vec<f32> {
    const A: f32 = 3.4445;
    const B: f32 = -4.7750;
    const C: f32 = 2.0315;
    let transposed = rows > cols;
    let (r, c) = if transposed { (cols, rows) } else { (rows, cols) };
    let mut x = if transposed {
        transpose(g, rows, cols)
    } else {
        g.to_vec()
    };
    // +1e-7 是防止全零矩阵除零（梯度全零时正则会返回零矩阵，而不是 NaN）
    let norm = x.iter().map(|v| v * v).sum::<f32>().sqrt() + 1e-7;
    for v in x.iter_mut() {
        *v /= norm;
    }
    for _ in 0..steps {
        let a = matmul(&x, &transpose(&x, r, c), r, c, r); // A = X Xᵀ
        let a2 = matmul(&a, &a, r, r, r); // A²
        let b: Vec<f32> = a
            .iter()
            .zip(&a2)
            .map(|(&av, &a2v)| B * av + C * a2v)
            .collect();
        let bx = matmul(&b, &x, r, r, c); // (bA + cA²) X
        x = x
            .iter()
            .zip(&bx)
            .map(|(&xv, &bxv)| A * xv + bxv)
            .collect();
    }
    if transposed {
        transpose(&x, r, c)
    } else {
        x
    }
}

/// Muon 优化器（**M**oment**u**m **o**rthogonalized by **N**ewton–Schulz）
///
/// 与 AdamW 的差别只在一处：**二维权重矩阵的更新方向不是逐元素自适应的一阶动量，
/// 而是把动量矩阵正交化后的正交极因子**。每步对二维参数 `W ∈ R^{r×c}`：
///
/// 1. 累积动量 `B ← μB + (1-μ)G`（`μ` 默认 0.95）
/// 2. Newton–Schulz 把 `B` 逼近到"奇异值全为 1"的正交极因子 `O ≈ UVᵀ`
/// 3. 缩放 `O ← O·√(max(1, r/c))`，再 `W ← W - lr·(O + wd·W)`
///
/// 为什么更快：AdamW 的更新在**每个坐标**上独立缩放，得到的更新矩阵奇异值谱很宽
/// （个别方向步子特别大）；Muon 把整个更新矩阵按谱范数归一化，等于"所有方向以同样的
/// 步长前进"，避免了被个别大奇异方向带偏。实测在同等算力下收敛显著更快
/// （Keller Jordan et al., 2024）。
///
/// 范围与回退：
/// - 形状二维且两维都 > 1 的参数走 Muon；
/// - **一维参数（bias、RMSNorm 的 γ）仍走 AdamW**——它们没有"行/列"的概念，
///   强行正交化没有意义；
/// - 冻结参数（`requires_grad = false`）整体跳过，既不做梯度步也不吃权重衰减
///   （原因同 [`AdamW`]）。
///
/// 两点使用提醒：
/// - **学习率要调大**：`O` 的行是单位向量，每个元素量级约 `1/√c`，于是每元素更新量
///   约 `lr/√c`；AdamW 每元素约 `·lr`。要拿到相近的有效步长，Muon 的 lr 通常要大
///   `√c` 倍（`c` 为列数）。
/// - 本实现按**形状**分流，而嵌入矩阵也是二维，因此也会走 Muon。工程上更常见的做法是
///   把嵌入与输出头留在 AdamW（它们的梯度按行统计更有意义）；本项目优化器层拿不到
///   参数名，故统一按形状分流。
pub struct Muon {
    lr: f32,
    /// 动量系数 μ（默认 0.95）
    momentum: f32,
    /// Newton–Schulz 迭代步数（默认 5）
    ns_steps: usize,
    weight_decay: f32,
    /// 一维参数回退 AdamW 用的超参（与 [`AdamW`] 默认一致）
    beta1: f32,
    beta2: f32,
    eps: f32,
    t: usize,
    params: Vec<Tensor>,
    /// 二维参数：正交化**之前**的动量缓冲 B；一维参数：AdamW 一阶动量 m
    m: Vec<Vec<f32>>,
    /// 二维参数：恒为 0（占位）；一维参数：AdamW 二阶动量 v
    v: Vec<Vec<f32>>,
}

impl Muon {
    pub fn new(
        lr: f32,
        params: Vec<Tensor>,
        weight_decay: f32,
        momentum: f32,
        ns_steps: usize,
    ) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        Muon {
            lr,
            momentum,
            ns_steps: ns_steps.max(1),
            weight_decay,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            t: 0,
            params,
            m,
            v,
        }
    }

    /// 该参数是否走 Muon 分支：二维且两维都 > 1。
    /// 形如 `[n, 1]` / `[1, n]` 的退化二维权重等价于一维，正交化只会把它缩成 ±1，
    /// 因此也归回 AdamW。
    fn is_matrix(shape: &[usize]) -> bool {
        shape.len() == 2 && shape[0] > 1 && shape[1] > 1
    }
}

impl Optimizer for Muon {
    fn params(&self) -> &[Tensor] {
        &self.params
    }

    fn step(&mut self) {
        self.t += 1;
        let (lr, mu, wd, ns) = (self.lr, self.momentum, self.weight_decay, self.ns_steps);
        let (beta1, beta2, eps) = (self.beta1, self.beta2, self.eps);
        let bc1 = 1.0 - beta1.powi(self.t as i32);
        let bc2 = 1.0 - beta2.powi(self.t as i32);
        for i in 0..self.params.len() {
            if !self.params[i].requires_grad() {
                continue; // 冻结参数：不更新、也不做权重衰减
            }
            let shape = self.params[i].shape().to_vec();
            if Self::is_matrix(&shape) {
                let (rows, cols) = (shape[0], shape[1]);
                // 1. 动量缓冲 B ← μB + (1-μ)G（Muon 的动量不做偏差校正）
                {
                    let g = self.params[i].grad.borrow();
                    let mi = &mut self.m[i];
                    let g_ref: &[f32] = &g;
                    mi.par_iter_mut()
                        .zip(g_ref.par_iter())
                        .for_each(|(mv, &gv)| *mv = mu * *mv + (1.0 - mu) * gv);
                }
                // 2. 正交极因子 O ≈ UVᵀ；3. 缩放并更新 W
                let o = zeropower_newton_schulz(&self.m[i], rows, cols, ns);
                let scale = (1.0f32).max(rows as f32 / cols as f32).sqrt();
                let mut d = self.params[i].decode_mut();
                d.par_iter_mut().zip(o.par_iter()).for_each(|(dv, &ov)| {
                    *dv -= lr * (scale * ov + wd * *dv);
                });
            } else {
                // 一维参数回退 AdamW（公式与 [`AdamW::step`] 逐行相同，t/beta/eps 也一致）
                let g = self.params[i].grad.borrow();
                let mut d = self.params[i].decode_mut();
                let g_ref: &[f32] = &g;
                let mi = &mut self.m[i];
                let vi = &mut self.v[i];
                mi.par_iter_mut()
                    .zip(vi.par_iter_mut())
                    .zip(d.par_iter_mut())
                    .zip(g_ref.par_iter())
                    .for_each(|(((mv, vv), dv), &gv)| {
                        *mv = beta1 * *mv + (1.0 - beta1) * gv;
                        *vv = beta2 * *vv + (1.0 - beta2) * gv * gv;
                        let m_hat = *mv / bc1;
                        let v_hat = *vv / bc2;
                        let step = lr * m_hat / (v_hat.sqrt() + eps);
                        let decay = lr * wd * *dv;
                        *dv = *dv - step - decay;
                    });
            }
        }
    }
}

impl OptimizerState for Muon {
    fn set_lr(&mut self, lr: f32) {
        self.lr = lr;
    }

    fn state(&self) -> (usize, &[Vec<f32>], &[Vec<f32>]) {
        (self.t, &self.m, &self.v)
    }

    fn restore_state(&mut self, t: usize, m: Vec<Vec<f32>>, v: Vec<Vec<f32>>) {
        assert_eq!(m.len(), self.params.len(), "动量 m 的参数数量不匹配");
        assert_eq!(v.len(), self.params.len(), "动量 v 的参数数量不匹配");
        for (i, p) in self.params.iter().enumerate() {
            assert_eq!(m[i].len(), p.numel(), "参数 {} 的动量长度不匹配", i);
            assert_eq!(v[i].len(), p.numel(), "参数 {} 的动量长度不匹配", i);
        }
        self.t = t;
        self.m = m;
        self.v = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor {
        Tensor::param(data, shape)
    }

    /// 用确定性公式给参数填梯度（不用随机数，失败可复现）
    fn fill_grad(p: &Tensor, f: impl Fn(usize) -> f32) {
        let mut g = p.grad.borrow_mut();
        for (i, x) in g.iter_mut().enumerate() {
            *x = f(i);
        }
    }

    /// 求值：`O`（`[r,c]` 行主序）是否等于 `factor · Q`（`Q` 为给定的正交结构矩阵）
    fn assert_scaled_q(o: &[f32], q: &[f32], factor: f32, what: &str) {
        for (i, (&ov, &qv)) in o.iter().zip(q).enumerate() {
            assert!(
                (ov - factor * qv).abs() < 1e-3,
                "{what}[{i}] = {ov}，期望 {}（factor {factor}）",
                factor * qv
            );
        }
    }

    /// NS 迭代的多项式 `p(x) = 3.4445x - 4.775x³ + 2.0315x⁵`。
    /// 输入是「奇异值全相同」的矩阵时，迭代精确退化成对这一个标量反复作用 `p`。
    fn ns_poly(x: f32) -> f32 {
        3.4445 * x - 4.775 * x.powi(3) + 2.0315 * x.powi(5)
    }

    /// 反复迭代 `p`（模拟 `steps` 步 Newton–Schulz）
    fn ns_iterate(x: f32, steps: usize) -> f32 {
        let mut v = x;
        for _ in 0..steps {
            v = ns_poly(v);
        }
        v
    }

    /// 构造 `[rows, cols]` 的「最小维单位阵」：`Q[i][i] = 1`（`i < min(rows, cols)`），其余为 0。
    /// 它的**较小子集方向**是正交单位的，`‖Q‖_F = √min(rows, cols)`——
    /// 正是能让 NS 迭代精确退化成标量多项式的输入。
    fn min_identity(rows: usize, cols: usize) -> Vec<f32> {
        let mut q = vec![0.0f32; rows * cols];
        for i in 0..rows.min(cols) {
            q[i * cols + i] = 1.0;
        }
        q
    }

    /// Newton–Schulz 对**正交结构输入**是精确的：`X = Q`（行/列正交单位）时，
    /// 归一化后所有奇异值都等于 `s = 1/‖Q‖_F`，迭代把每个奇异值同乘 `p⁵(s)`，
    /// 于是输出恰为 `p⁵(s)·Q`。
    ///
    /// 注意它**不是**严格的 `UVᵀ`：五次迭代的常数刻意选成"零点斜率最大"而非"处处收敛到 1"，
    /// 输出是 `US'Vᵀ`（`S'` 分散在 0.5~1.5）。所以这里断言的是上面那条精确代数性质，
    /// 而不是"行正交单位"（那会被常量设计上的行为差异误判为 bug）。
    #[test]
    fn test_newton_schulz_matches_polynomial_on_orthogonal_input() {
        for &(r, c) in &[(4usize, 4usize), (3, 5), (5, 3)] {
            let q = min_identity(r, c);
            let o = zeropower_newton_schulz(&q, r, c, 5);
            assert!(o.iter().all(|v| v.is_finite()), "[{r},{c}] 出现非有限值");
            let s = 1.0 / (q.iter().map(|v| v * v).sum::<f32>().sqrt());
            let factor = ns_iterate(s, 5);
            assert_scaled_q(&o, &q, factor, &format!("NS[{r},{c}]"));
        }
    }

    /// 全零动量：正则化前有 +1e-7，除出来仍是零矩阵，**不能**变成 NaN；
    /// 关掉权重衰减后参数应一动不动。
    #[test]
    fn test_muon_zero_gradient_yields_zero_update() {
        let p = tensor(vec![0.0; 9], vec![3, 3]);
        let mut opt = Muon::new(0.1, vec![p.clone()], 0.0, 0.95, 5);
        opt.step();
        let d = p.data_ref();
        assert!(d.iter().all(|v| v.is_finite()), "零梯度步产生了非有限参数");
        assert!(d.iter().all(|&v| v == 0.0), "零梯度 + 零权重衰减不应改动参数");
    }

    /// 二维参数的更新量应为 `-lr·scale·NS(B)`。取梯度为单位阵时动量是 `(1-μ)·I`（奇异值全同），
    /// 可以精确算出 `Δ = -lr·scale·p⁵(s)·I`，逐元素核对整条更新链路（动量 → 归一化 → 迭代 → 缩放）。
    #[test]
    fn test_muon_matrix_update_matches_closed_form() {
        let lr = 0.25f32;
        let mu = 0.95f32;
        let (r, c) = (4, 4);
        let p = tensor(vec![0.0; r * c], vec![r, c]);
        // 梯度 = 单位阵
        fill_grad(&p, |i| if i / c == i % c { 1.0 } else { 0.0 });
        let mut opt = Muon::new(lr, vec![p.clone()], 0.0, mu, 5);
        opt.step();
        // 动量 B = (1-μ)I，归一化后奇异值 = (1-μ)/‖B‖_F = 1/2
        let s = 1.0 / (r as f32).sqrt();
        let factor = lr * 1.0 * ns_iterate(s, 5); // scale = max(1, 4/4)^0.5 = 1
        let expect: Vec<f32> = (0..r * c)
            .map(|i| if i / c == i % c { -factor } else { 0.0 })
            .collect();
        let got = p.data_ref().to_vec();
        for (i, (&g, &e)) in got.iter().zip(&expect).enumerate() {
            assert!((g - e).abs() < 1e-4, "Δ[{i}] = {g}，期望 {e}");
        }
    }

    /// 缩放因子应为 `√(max(1, rows/cols))`：`[8,2]` 的更新范数应是 `[2,8]` 的 2 倍
    /// （前者 rows/cols = 4 ⇒ scale = 2，后者 = 0.25 ⇒ scale = 1，而 NS 输出范数同为 `p⁵(1/√2)·√2`）。
    #[test]
    fn test_muon_scaling_uses_dimension_ratio() {
        let lr = 0.1f32;
        let norm_of = |rows: usize, cols: usize| -> f32 {
            let q = min_identity(rows, cols);
            let p = tensor(vec![0.0; rows * cols], vec![rows, cols]);
            fill_grad(&p, |i| q[i]);
            let mut opt = Muon::new(lr, vec![p.clone()], 0.0, 0.95, 5);
            opt.step();
            p.data_ref().iter().map(|v| v * v).sum::<f32>().sqrt()
        };
        let wide = norm_of(2, 8);
        let tall = norm_of(8, 2);
        assert!((tall / wide - 2.0).abs() < 1e-2, "缩放比 = {}，期望 2", tall / wide);
    }

    /// 一维参数必须与 AdamW 走出**逐位相同**的结果（同样的 betas / eps / t / lr / wd）。
    #[test]
    fn test_muon_falls_back_to_adamw_on_1d_params() {
        let init = vec![0.7f32, -1.3, 2.1, 0.05];
        let grad = vec![0.2f32, -0.4, 0.9, 0.15];
        let (lr, wd) = (0.05f32, 0.1f32);

        let a = tensor(init.clone(), vec![4]);
        fill_grad(&a, |i| grad[i]);
        let mut adam = AdamW::new(lr, vec![a.clone()], wd);
        adam.step();

        let b = tensor(init, vec![4]);
        fill_grad(&b, |i| grad[i]);
        let mut muon = Muon::new(lr, vec![b.clone()], wd, 0.95, 5);
        muon.step();

        let av = a.data_ref().to_vec();
        let bv = b.data_ref().to_vec();
        for (i, (x, y)) in av.iter().zip(&bv).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "一维参数[{i}]与 AdamW 不是逐位一致：{x} vs {y}");
        }
    }

    /// 冻结参数（`requires_grad = false`）整体跳过：既不更新也不吃权重衰减。
    /// 这里特意开权重衰减 —— 若跳过逻辑失效，参数会每步朝 0 缩，测试立刻发现。
    #[test]
    fn test_muon_skips_frozen_params() {
        let frozen = tensor(vec![1.0; 4], vec![2, 2]);
        frozen.set_requires_grad(false);
        fill_grad(&frozen, |_| 10.0);
        let mut opt = Muon::new(0.5, vec![frozen.clone()], 0.9, 0.95, 5);
        opt.step();
        let d = frozen.data_ref().to_vec();
        assert!(d.iter().all(|&v| v == 1.0), "冻结参数被改动了：{d:?}");
    }

    /// 状态导出 / 恢复往返一致（checkpoint 续训依赖这个三段布局）。
    #[test]
    fn test_muon_state_roundtrip() {
        let m2 = tensor(vec![0.1; 6], vec![2, 3]);
        let b1 = tensor(vec![0.0; 3], vec![3]);
        fill_grad(&m2, |i| (i as f32 * 0.3).sin());
        fill_grad(&b1, |i| (i as f32 * 0.5).cos());
        let mut opt = Muon::new(0.1, vec![m2.clone(), b1.clone()], 0.0, 0.9, 5);
        opt.step();
        opt.step();
        let (t, m, v) = opt.state();
        let (t, m, v) = (t, m.to_vec(), v.to_vec());

        let mut opt2 = Muon::new(0.1, vec![m2.clone(), b1.clone()], 0.0, 0.9, 5);
        opt2.restore_state(t, m.clone(), v.clone());
        let (t2, m2v, v2v) = opt2.state();
        assert_eq!(t2, t, "步数未恢复");
        assert_eq!(m2v, &m[..], "一阶动量未恢复");
        assert_eq!(v2v, &v[..], "二阶动量（占位）未恢复");
    }
}
