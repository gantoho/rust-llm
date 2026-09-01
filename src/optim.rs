//! 优化器（第 6、17 课）
//!
//! 优化器负责"怎么更新参数"：
//! - SGD（第 6 课）：θ = θ - lr·g，最朴素
//! - AdamW（第 17 课）：自适应学习率 + 动量 + 权重衰减，现代 LLM 标配

use crate::tensor::Tensor;

/// 随机梯度下降（SGD）
pub struct SGD {
    lr: f32,
    params: Vec<Tensor>,
}

impl SGD {
    pub fn new(lr: f32, params: Vec<Tensor>) -> Self {
        SGD { lr, params }
    }

    /// 更新一步：θ = θ - lr * g
    pub fn step(&self) {
        for p in &self.params {
            let g = p.grad();
            let d = p.data();
            let updated: Vec<f32> = d.iter().zip(&g).map(|(v, g)| v - self.lr * g).collect();
            p.set_data(updated);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }
}

/// SGD（第 2 课）AdamW（Adam + 权重衰减解耦）
///
/// 核心思想：
/// 1. 一阶动量 m：梯度的指数移动平均（记住"方向"，像小球下坡的惯性）
/// 2. 二阶动量 v：梯度平方的指数移动平均（感知"坡度陡缓"，陡的地方步子小）
/// 3. 偏差修正：训练初期 m、v 从 0 起步，除以 (1-β^t) 修正
/// 4. 权重衰减：每步额外把参数往 0 拉一点（正则化，防止过拟合）
pub struct AdamW {
    pub lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    t: usize,
    params: Vec<Tensor>,
    m: Vec<Vec<f32>>, // 一阶动量
    v: Vec<Vec<f32>>, // 二阶动量
}

impl AdamW {
    pub fn new(lr: f32, params: Vec<Tensor>, weight_decay: f32) -> Self {
        let m = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        let v = params.iter().map(|p| vec![0.0f32; p.numel()]).collect();
        AdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
            t: 0,
            params,
            m,
            v,
        }
    }

    pub fn step(&mut self) {
        self.t += 1;
        // 偏差修正系数（训练初期 t 小，修正大）
        let bc1 = 1.0 - self.beta1.powi(self.t as i32);
        let bc2 = 1.0 - self.beta2.powi(self.t as i32);

        for (i, p) in self.params.iter().enumerate() {
            let g = p.grad();
            let d = p.data();
            let mut updated = vec![0.0f32; d.len()];
            for j in 0..d.len() {
                let gv = g[j];
                // 1. 更新动量
                self.m[i][j] = self.beta1 * self.m[i][j] + (1.0 - self.beta1) * gv;
                self.v[i][j] = self.beta2 * self.v[i][j] + (1.0 - self.beta2) * gv * gv;
                // 2. 偏差修正
                let m_hat = self.m[i][j] / bc1;
                let v_hat = self.v[i][j] / bc2;
                // 3. 更新：θ -= lr * m_hat/(√v_hat + eps) + lr * wd * θ（权重衰减解耦）
                let step = self.lr * m_hat / (v_hat.sqrt() + self.eps);
                let decay = self.lr * self.weight_decay * d[j];
                updated[j] = d[j] - step - decay;
            }
            p.set_data(updated);
        }
    }

    pub fn zero_grad(&self) {
        for p in &self.params {
            p.zero_grad();
        }
    }

    /// 导出优化器状态（checkpoint 用）：(步数 t, 一阶动量 m, 二阶动量 v)
    pub fn state(&self) -> (usize, Vec<Vec<f32>>, Vec<Vec<f32>>) {
        (self.t, self.m.clone(), self.v.clone())
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
