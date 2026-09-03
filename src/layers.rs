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

/// MLP 隐藏层放大系数（GPT-2 风格：输入维度的 4 倍）
const MLP_RATIO: usize = 4;

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

/// RMSNorm（Root Mean Square Layer Normalization）：
/// y = x / √(mean(x²) + ε) * γ
///
/// 比 LayerNorm 更高效（不减均值、无 β），现代 LLM 标配：
/// - LLaMA / LLaMA 2 / LLaMA 3
/// - Mistral / Mixtral
/// - Qwen / Qwen2
/// - Gemma
pub struct RMSNorm {
    pub gamma: Tensor, // [d] 可学习缩放
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(d: usize, eps: f32) -> Self {
        RMSNorm {
            gamma: Tensor::param(vec![1.0; d], vec![d]),
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rmsnorm(&self.gamma, self.eps)
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{prefix}.gamma"), self.gamma.clone())]
    }
}

impl Module for RMSNorm {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone()]
    }
}

/// 统一的归一化层枚举：支持 LayerNorm 和 RMSNorm 两种选择
pub enum NormLayer {
    LN(LayerNorm),
    RMS(RMSNorm),
}

impl NormLayer {
    pub fn new(d: usize, eps: f32, use_rmsnorm: bool) -> Self {
        if use_rmsnorm {
            NormLayer::RMS(RMSNorm::new(d, eps))
        } else {
            NormLayer::LN(LayerNorm::new(d, eps))
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            NormLayer::LN(ln) => ln.forward(x),
            NormLayer::RMS(rms) => rms.forward(x),
        }
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            NormLayer::LN(ln) => ln.named_parameters(prefix),
            NormLayer::RMS(rms) => rms.named_parameters(prefix),
        }
    }
}

impl Module for NormLayer {
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            NormLayer::LN(ln) => ln.parameters(),
            NormLayer::RMS(rms) => rms.parameters(),
        }
    }
}

/// SwiGLU MLP 层（LLaMA 风格）：
///
/// ```text
/// hidden = SiLU(x @ W_gate) ⊙ (x @ W_up)
/// out = hidden @ W_down
/// ```
///
/// 与 GPT-2 风格 MLP（GELU(x @ W1) @ W2）的区别：
/// - 用 SiLU(x) ⊙ gate 替代 GELU（表达力更强）
/// - 多一个 W_gate 矩阵（门控分支）
/// - hidden_dim 通常设为 (2/3) * 4d（保持参数量相近）
///
/// LLaMA 的 hidden_dim = 11008（2/3 * 4 * 4096 ≈ 10922，取 256 的倍数）
pub struct SwiGLUMLP {
    pub w_gate: Linear, // [D, hidden] 门控分支
    pub w_up: Linear,   // [D, hidden] 上投影
    pub w_down: Linear, // [hidden, D] 下投影
}

impl SwiGLUMLP {
    pub fn new(d: usize, hidden: usize, rng: &mut Rng) -> Self {
        SwiGLUMLP {
            w_gate: Linear::new(d, hidden, rng),
            w_up: Linear::new(d, hidden, rng),
            w_down: Linear::new(hidden, d, rng),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.w_gate.forward(x);
        let up = self.w_up.forward(x);
        let hidden = gate.swiglu(&up);
        self.w_down.forward(&hidden)
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps = self.w_gate.named_parameters(&format!("{prefix}.w_gate"));
        ps.extend(self.w_up.named_parameters(&format!("{prefix}.w_up")));
        ps.extend(self.w_down.named_parameters(&format!("{prefix}.w_down")));
        ps
    }
}

impl Module for SwiGLUMLP {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.w_gate.parameters();
        ps.extend(self.w_up.parameters());
        ps.extend(self.w_down.parameters());
        ps
    }
}

/// 统一的 MLP 层枚举：支持 GPT-2 风格 GELU MLP 和 LLaMA 风格 SwiGLU MLP
pub enum MLPEnum {
    GELU {
        linear1: Linear,
        linear2: Linear,
    },
    SwiGLU(SwiGLUMLP),
}

impl MLPEnum {
    pub fn new_gelu(d: usize, rng: &mut Rng) -> Self {
        MLPEnum::GELU {
            linear1: Linear::new(d, MLP_RATIO * d, rng),
            linear2: Linear::new(MLP_RATIO * d, d, rng),
        }
    }

    pub fn new_swiglu(d: usize, rng: &mut Rng) -> Self {
        // SwiGLU hidden_dim = (2/3) * 4d ≈ 2.67d，取 256 的倍数
        let hidden = ((2 * MLP_RATIO * d + 2) / 3 + 255) & !255;
        MLPEnum::SwiGLU(SwiGLUMLP::new(d, hidden, rng))
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            MLPEnum::GELU { linear1, linear2 } => linear2.forward(&gelu(&linear1.forward(x))),
            MLPEnum::SwiGLU(swiglu) => swiglu.forward(x),
        }
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => {
                let mut ps = linear1.named_parameters(&format!("{prefix}.mlp_linear1"));
                ps.extend(linear2.named_parameters(&format!("{prefix}.mlp_linear2")));
                ps
            }
            MLPEnum::SwiGLU(s) => s.named_parameters(prefix),
        }
    }
}

impl Module for MLPEnum {
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => {
                let mut ps = linear1.parameters();
                ps.extend(linear2.parameters());
                ps
            }
            MLPEnum::SwiGLU(s) => s.parameters(),
        }
    }
}

/// LoRA（Low-Rank Adaptation，低秩适配）层
///
/// 在冻结的预训练权重旁边插入低秩矩阵：
/// ```text
/// y = x @ (W + ΔW) = x @ W + x @ (B @ A) = x @ W + (x @ B) @ A
/// ```
///
/// 其中：
/// - W ∈ R^{out × in}：原始冻结权重（requires_grad = false）
/// - B ∈ R^{out × r}：上投影矩阵
/// - A ∈ R^{r × in}：下投影矩阵
/// - r ≪ min(in, out)：秩（通常 4-64）
///
/// 可训练参数量：r × (in + out)，远小于 in × out。
/// 例如 in=4096, out=4096, r=16 时：16×(4096+4096) = 131072（0.8% 的原始参数量）。
///
/// 初始化：
/// - A：正态分布 N(0, σ²)，σ 通常很小（如 1/sqrt(r)）
/// - B：全零（保证 ΔW = BA = 0，训练开始时模型行为不变）
/// - α 缩放因子：ΔW = (α/r) × BA，α 通常 = r（即不缩放）
///
/// 使用方式：
/// ```rust
/// let lora = LoRA::new(in_dim, out_dim, rank, alpha, &mut rng);
/// let y = lora.forward(&x); // y = x @ W_frozen + (x @ B) @ A * (alpha/rank)
/// ```
#[allow(dead_code)] // 教学实现：LoRA 层完整可用，通过 inject_lora 注入到现有模型
pub struct LoRA {
    /// 原始权重（冻结）
    pub weight: Tensor,
    /// 低秩下投影 [rank, in]
    pub a: Tensor,
    /// 低秩上投影 [out, rank]
    pub b: Tensor,
    /// 缩放因子 α/r
    scaling: f32,
    pub rank: usize,
}

#[allow(dead_code)] // 教学实现：LoRA 层完整可用，通过 inject_lora 注入到现有模型
impl LoRA {
    /// 创建 LoRA 层
    ///
    /// - `in_dim`：输入维度
    /// - `out_dim`：输出维度
    /// - `rank`：低秩维度 r（越小越省参数，越大表达力越强）
    /// - `alpha`：缩放因子 α（通常 = rank）
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, alpha: f32, rng: &mut Rng) -> Self {
        let w_scale = 1.0 / (in_dim as f32).sqrt();
        let w_data: Vec<f32> = (0..out_dim * in_dim).map(|_| rng.randn() * w_scale).collect();
        let mut w = Tensor::param(w_data, vec![out_dim, in_dim]);
        w.requires_grad = false; // 冻结

        // A: [rank, in]，正态 N(0, 1/sqrt(rank))
        let a_scale = 1.0 / (rank as f32).sqrt();
        let a_data: Vec<f32> = (0..rank * in_dim).map(|_| rng.randn() * a_scale).collect();
        let a = Tensor::param(a_data, vec![rank, in_dim]);
        // B: [out, rank]，全零（初始 ΔW = 0）
        let b = Tensor::param(vec![0.0; out_dim * rank], vec![out_dim, rank]);

        LoRA {
            weight: w,
            a,
            b,
            scaling: alpha / rank as f32,
            rank,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // y = x @ W^T + (x @ A^T @ B^T) * scaling
        // 先用原始权重
        let base = x.matmul(&self.weight.transpose());
        // 再加 LoRA 增量：(x @ A^T) @ B^T * scaling
        let lora_out = x
            .matmul(&self.a.transpose())
            .matmul(&self.b.transpose())
            .mul_scalar(self.scaling);
        base.add(&lora_out)
    }

    /// 只返回可训练参数（A 和 B）
    pub fn trainable_parameters(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }

    /// 全部参数（含冻结的 W）
    pub fn all_parameters(&self) -> Vec<Tensor> {
        vec![self.weight.clone(), self.a.clone(), self.b.clone()]
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}.weight"), self.weight.clone()),
            (format!("{prefix}.lora_a"), self.a.clone()),
            (format!("{prefix}.lora_b"), self.b.clone()),
        ]
    }
}

impl Module for LoRA {
    fn parameters(&self) -> Vec<Tensor> {
        self.all_parameters()
    }
}

/// 给任意 Linear 层注入 LoRA 的辅助函数
///
/// 把原始 Linear 的权重移到 LoRA 中（冻结），返回 LoRA 层。
/// 用法：
/// ```rust
/// let lora_attn = inject_lora(&attn_layer, 16, 16.0, &mut rng);
/// ```
#[allow(dead_code)] // 教学实现：给 Linear 层注入 LoRA
pub fn inject_lora(linear: &Linear, rank: usize, alpha: f32, rng: &mut Rng) -> LoRA {
    let w = &linear.weight;
    let in_dim = w.shape[1];
    let out_dim = w.shape[0];
    let mut lora = LoRA::new(in_dim, out_dim, rank, alpha, rng);
    // 用原始权重替换随机初始化的 W
    lora.weight = w.clone();
    lora.weight.requires_grad = false;
    lora
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
