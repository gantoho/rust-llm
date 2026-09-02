//! 神经网络层（第 5 课）
//!
//! 这一课把"层"抽象出来，它们是神经网络的基本积木：
//! - Linear：y = xW + b，全连接层
//! - LayerNorm：层归一化（第 11 课）
//! - Embedding：把 token id 变成向量（第 12 课）
//! - 激活函数：ReLU / GELU / Tanh

use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;

/// 线性层：y = x @ W + b
///
/// - weight: [in_features, out_features]
/// - bias:   [out_features]
///
/// 输入可以是 [B, in] 或 [B, T, in]（会自动展平处理）
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
}

impl Linear {
    /// 创建线性层。
    /// 权重用 Xavier 正态分布初始化（std = √(2/(in+out))，保证前向/反向方差稳定）。
    pub fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Self {
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.randn() * std)
            .collect();
        Linear {
            weight: Tensor::param(w, vec![in_features, out_features]),
            bias: Tensor::param(vec![0.0; out_features], vec![out_features]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 支持 [B, in] 和 [B, T, in]；3D 输入在内部展平计算，输出保持 3D
        let is_3d = x.rank() == 3;
        let orig_shape = x.shape().to_vec();
        let x = match x.rank() {
            2 => x.clone(),
            3 => x.reshape(vec![x.shape()[0] * x.shape()[1], x.shape()[2]]),
            // 公开库 API：维度不合法给可读错误（带实际维度信息）
            r => panic!("Linear 输入必须为 2D 或 3D，实际是 {r}D（形状 {:?}）", x.shape()),
        };
        // y = x @ W + b（b 是 [out]，与 [B, out] 广播相加）
        let y = x.matmul(&self.weight).add(&self.bias);
        if is_3d {
            // 3D 输入 [B, T, in] -> 输出 [B, T, out]（最后一维换成 out_features）
            let mut out_shape = orig_shape;
            let n = out_shape.len();
            out_shape[n - 1] = self.weight.shape()[1];
            y.reshape(out_shape)
        } else {
            y
        }
    }

    /// 带名字的参数（checkpoint 保存/恢复用）：`{prefix}.weight` / `{prefix}.bias`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}.weight"), self.weight.clone()),
            (format!("{prefix}.bias"), self.bias.clone()),
        ]
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.bias.clone()]
    }
}

/// 层归一化（LayerNorm）：
/// 对最后一维做归一化（均值 0、方差 1），再缩放平移。
///
/// y = (x - μ) / √(σ² + ε) * γ + β
///
/// 为什么需要它？（第 11 课详解）
/// - 稳定训练：避免层输出数值范围过大导致梯度爆炸/消失
/// - 加快收敛：每层输入分布一致
pub struct LayerNorm {
    pub gamma: Tensor, // [d] 可学习缩放
    pub beta: Tensor,  // [d] 可学习平移
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(d: usize, eps: f32) -> Self {
        LayerNorm {
            gamma: Tensor::param(vec![1.0; d], vec![d]),
            beta: Tensor::param(vec![0.0; d], vec![d]),
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 融合实现（一个算子完成 11 个基础算子的前向+反向），见 Tensor::layernorm
        x.layernorm(&self.gamma, &self.beta, self.eps)
    }

    /// 带名字的参数：`{prefix}.gamma` / `{prefix}.beta`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}.gamma"), self.gamma.clone()),
            (format!("{prefix}.beta"), self.beta.clone()),
        ]
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

/// 嵌入层：把 token id 查表变成向量。
/// table: [vocab_size, d_model]
pub struct Embedding {
    pub table: Tensor,
}

impl Embedding {
    /// 创建嵌入表，用正态分布 N(0, 0.02) 初始化
    pub fn new(vocab_size: usize, d_model: usize, rng: &mut Rng) -> Self {
        let std = 0.02;
        let data: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| rng.randn() * std)
            .collect();
        Embedding {
            table: Tensor::param(data, vec![vocab_size, d_model]),
        }
    }

    /// 前向：ids [N] -> out [N, d_model]（每一行取表里的对应向量）
    pub fn forward(&self, ids: &[usize]) -> Tensor {
        self.table.gather_rows(ids)
    }

    /// 带名字的参数：`{prefix}.table`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{prefix}.table"), self.table.clone())]
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.table.clone()]
    }
}

// ---------- 激活函数（第 5 课） ----------

/// GELU：GPT 系列的默认激活，用 tanh 近似，比 ReLU 更平滑
pub fn gelu(x: &Tensor) -> Tensor {
    x.gelu()
}

/// Tanh：S 型，输出 (-1, 1)
pub fn tanh(x: &Tensor) -> Tensor {
    x.tanh()
}
