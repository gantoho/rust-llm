//! 优化器（第 6、17 课）
//!
//! 优化器负责"怎么更新参数"：
//! - SGD（第 6 课）：θ = θ - lr·g，最朴素
//! - AdamW（第 17 课）：自适应学习率 + 动量 + 权重衰减，现代 LLM 标配

use crate::tensor::Tensor;

/// 优化器统一接口（第 17 课）
///
/// - `step()`：用当前梯度更新一次参数（SGD / AdamW 各自实现）
/// - `zero_grad()`：清零所有参数梯度（默认实现相同，这里只写一份）
pub trait Optimizer {
    /// 返回被优化的参数列表
    fn params(&self) -> &[Tensor];

    /// 用当前梯度更新一次参数（SGD / AdamW 各自实现）
    fn step(&mut self);

    /// 清零所有参数的梯度（两种优化器共用同一实现）
    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }
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
        for p in &self.params {
            if !p.requires_grad() {
                continue;
            }
            let g = p.grad.borrow();
            let mut d = p.data.borrow_mut();
            for j in 0..d.len() {
                d[j] -= self.lr * g[j];
            }
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
    pub fn new_with_betas(
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
            let mut d = self.params[i].data.borrow_mut();
            let mi = &mut self.m[i];
            let vi = &mut self.v[i];
            for j in 0..d.len() {
                let gv = g[j];
                mi[j] = beta1 * mi[j] + (1.0 - beta1) * gv;
                vi[j] = beta2 * vi[j] + (1.0 - beta2) * gv * gv;
                let m_hat = mi[j] / bc1;
                let v_hat = vi[j] / bc2;
                let step = lr * m_hat / (v_hat.sqrt() + eps);
                let decay = lr * wd * d[j];
                d[j] = d[j] - step - decay;
            }
        }
    }
}
