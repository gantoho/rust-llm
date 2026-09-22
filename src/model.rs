//! GPT 模型（第 9-12、19 课）
//!
//! 结构（从下到上）：
//! 1. token embedding：每个 token id -> 向量
//! 2. N 层 Transformer Block：注意力（找相关性）+ 前馈网络（加工信息）
//! 3. 最终 LayerNorm + 输出头（预测下一个 token）
//!
//! 位置信息由 RoPE（第 20 课）提供：在注意力内部对 Q/K 做旋转，不再向输入加位置向量。
//!
//! 注意点：
//! - 因果掩码（causal mask）：模型只能看到过去，不能看到未来
//! - KV Cache（第 25 课）：推理时缓存历史的 K/V，避免重复计算
//! - RoPE（第 20 课）：只旋转 Q/K、不旋转 V；旋转发生在 KV cache append 之前，
//!   缓存里存的是"已旋转的 K"，历史 K 直接复用

use crate::attention::{KVCache, KvCacheOpts, MultiHeadAttention};
use crate::config::LoRAConfig;
use crate::layers::{Embedding, Linear, MLPEnum, NormLayer};
use crate::moe::MoELayer;
use crate::module::Module;
use crate::quant::{
    CalibOpts, CalibStats, HessianKind, LayerCalib, QBits, QuantMeta, QuantMethod, QuantOpts,
    QuantReport, QuantSummary,
};
use crate::rng::Rng;
use crate::rope::{self, RopeScaling, RopeSpec};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::rc::Rc;

/// LayerNorm 数值稳定常数（防止方差为 0 时除零）
const LN_EPS: f32 = 1e-5;

/// 模型配置（`config/config.json` 里可调，缺省字段用 [`GPTConfig::default`]）
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GPTConfig {
    /// 词表大小；0 表示"由分词器决定"（训练时自动填入）
    pub vocab_size: usize,
    pub n_embd: usize,     // 隐藏维度
    pub n_head: usize,     // 注意力头数（Q 的头数）
    pub n_layer: usize,    // Transformer 层数
    pub block_size: usize, // 最大上下文长度
    // ---- 现代 LLM 扩展 ----
    /// KV 头数（GQA）：n_kv_head < n_head 时启用 Grouped Query Attention。
    /// 0 表示与 n_head 相同（标准 Multi-Head Attention）。
    /// LLaMA 2 70B 用 n_head=64, n_kv_head=8；Mistral 7B 用 n_head=32, n_kv_head=8。
    pub n_kv_head: usize,
    /// 是否使用 RMSNorm（true = LLaMA 风格，false = GPT-2 风格 LayerNorm）
    pub use_rmsnorm: bool,
    /// 是否使用 SwiGLU MLP（true = LLaMA 风格，false = GPT-2 风格 GELU MLP）
    pub use_swiglu: bool,
    /// Dropout 概率（0 = 不丢弃）。用于注意力权重和残差连接。
    pub dropout: f32,
    // ---- MoE 稀疏专家（第 32 课）----
    /// 每个 MoE 层的专家数。**1 = 稠密 FFN**（默认，行为与以前完全一致），
    /// ≥ 2 时该前馈子层换成 [`crate::moe::MoELayer`]。
    pub n_expert: usize,
    /// 每个 token 激活几个专家（Top-K），须满足 `1 ≤ moe_top_k ≤ n_expert`。
    /// 1 = Switch Transformer 式，2 = Mixtral / Grok-1 式。
    pub moe_top_k: usize,
    /// 专家容量因子：每个专家最多接收 `cf × n·K/E` 个 token，超出的分配按 token
    /// 顺序先到先得地丢弃。**0 表示不限容量**（默认）——推理必须用 0，
    /// 否则同一个 token 的输出会取决于同批次别的 token。
    pub moe_capacity_factor: f32,
    /// 负载均衡辅助损失系数 α（`L_aux = α·E·Σ f_i·p_i`）。0 = 不加（默认）。
    pub moe_aux_coef: f32,
    /// 门控权重的求和口径（详见 [`crate::moe`] 文件头 §2）：
    /// - `false`（默认）：Top-K 内部**重归一化**（Mixtral / DeepSeek / Qwen 式），`Σw = 1`。
    ///   ⚠️ `moe_top_k = 1` 时权重恒等于 1，主损失给不了路由器任何梯度——
    ///   K = 1 请置 `true`，否则路由器只能靠辅助损失训练。
    /// - `true`：用**全部专家**上的 softmax 原概率（Switch Transformer 式），`Σw < 1`。
    pub moe_switch_gate: bool,
    // ---- RoPE 频率（第 20 课长度外推）----
    /// RoPE 的频率底数：10 000 = 原始 RoPE / GPT-NeoX；LLaMA-3 用 500 000。
    /// 底数本身就是一种静态的频率缩放，改它必须与训练时保持一致。
    pub rope_base: f32,
    /// RoPE 的长度外推方式（[`crate::rope::RopeScaling`]）：
    /// `None` = 不外推；`Linear` / `Ntk` / `Yarn` = 把频率表拉长以便超出
    /// `block_size` 后仍能工作。默认 `None`（训练窗口内不需要外推）。
    ///
    /// ⚠️ 与训练时不一致会导致输出完全错乱：位置被旋转到了不同角度。
    pub rope_scaling: RopeScaling,
    /// **原始训练窗口**：YaRN 的分段边界（哪些维度算"高频"）必须相对它来算。
    /// 0 = 取 [`Self::block_size`]（训练时两者本来就相同）。
    ///
    /// 只有一种情况需要显式填它：拿 `block_size = 512` 训出来的权重、推理时想把
    /// `block_size` 改成 4096 来跑长文——此时窗口放大了，但"模型见过的位置差最大到 512"
    /// 这个事实没变，YaRN 必须按 512 分段才对。
    pub rope_train_ctx: usize,
}

impl Default for GPTConfig {
    fn default() -> Self {
        GPTConfig {
            vocab_size: 0,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            block_size: 32,
            n_kv_head: 0,
            use_rmsnorm: false,
            use_swiglu: false,
            dropout: 0.0,
            n_expert: 1,
            moe_top_k: 1,
            moe_capacity_factor: 0.0,
            moe_aux_coef: 0.0,
            moe_switch_gate: false,
            rope_base: rope::ROPE_BASE,
            rope_scaling: RopeScaling::None,
            rope_train_ctx: 0,
        }
    }
}

impl GPTConfig {
    /// YaRN 分段边界要用的训练窗口（`rope_train_ctx = 0` 时即 `block_size`）
    pub fn rope_train_ctx(&self) -> usize {
        if self.rope_train_ctx == 0 {
            self.block_size
        } else {
            self.rope_train_ctx
        }
    }

    /// 本配置的完整 RoPE 频率参数
    pub fn rope_spec(&self) -> RopeSpec {
        RopeSpec::new(self.rope_base, self.rope_scaling, self.rope_train_ctx())
    }

    /// 一个小配置，适合学习演示（其余字段与 Default 一致）
    pub fn tiny(vocab_size: usize) -> Self {
        GPTConfig {
            vocab_size,
            ..Default::default()
        }
    }
}

/// 前馈子层：稠密 FFN 或稀疏 MoE（第 32 课）。
///
/// 两者对外接口一致（`forward` / `parameters` / `named_parameters`），`TransformerBlock`
/// 只需在构造时二选一，后面的残差、dropout、GPU 常驻快路全都不用改。
/// `cfg.n_expert <= 1` 时就是普通的 [`MLPEnum`]，行为与加 MoE 之前**逐位相同**。
enum Ffn {
    Dense(MLPEnum),
    Moe(MoELayer),
}

impl Ffn {
    /// 前向 + 本子层的均衡辅助损失（稠密分支恒为 `None`）。
    ///
    /// 辅助损失必须**从子层里带出来**：它是 `L = L_ce + Σ_layers α·L_aux` 的一部分，
    /// 得跟着主损失一起 `backward()` 才能把梯度送到各层的路由器上。
    fn forward_with_aux(&self, x: &Tensor) -> (Tensor, Option<Tensor>) {
        match self {
            Ffn::Dense(m) => (m.forward(x), None),
            Ffn::Moe(m) => m.forward_with_aux(x),
        }
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            Ffn::Dense(m) => m.named_parameters(&format!("{prefix}.mlp")),
            Ffn::Moe(m) => m.named_parameters(&format!("{prefix}.moe")),
        }
    }

    /// LoRA 只挂在稠密 FFN 上：MoE 的专家是"本就稀疏激活"的一层，
    /// 再叠低秩增量既没有通用做法，也会破坏"专家各学各的"这一前提。
    /// 于是 `apply_lora` 对 MoE 是空操作，`has_lora` 恒为 false。
    fn apply_lora(&mut self, lora: &LoRAConfig, rng: &mut Rng) {
        if let Ffn::Dense(m) = self {
            m.apply_lora(lora, rng);
        }
    }

    fn merge_lora(&mut self) {
        if let Ffn::Dense(m) = self {
            m.merge_lora();
        }
    }

    fn has_lora(&self) -> bool {
        match self {
            Ffn::Dense(m) => m.has_lora(),
            Ffn::Moe(_) => false,
        }
    }

    /// 本子层是否有投影被量化（GPU 常驻显存快路据此让路，见
    /// [`TransformerBlock::attn_resident`] 的说明）
    fn has_quant(&self) -> bool {
        match self {
            Ffn::Dense(m) => m.has_quant(),
            Ffn::Moe(_) => false,
        }
    }

    /// 把各投影的量化状态烘焙回 f32（checkpoint 里写的始终是 f32 权重）
    fn dequantize_weights(&mut self) {
        if let Ffn::Dense(m) = self {
            m.dequantize_weights();
        }
    }

    /// 本子层的全部投影 + 参数名前缀（MoE 为空：专家个多而小，量化收益被每组 scale
    /// 的开销吃掉，且没有统一的校准口径，与 LoRA 同理不覆盖）。
    fn named_linears(&self, prefix: &str) -> Vec<(String, &Linear)> {
        match self {
            Ffn::Dense(m) => m.named_linears(&format!("{prefix}.mlp")),
            Ffn::Moe(_) => Vec::new(),
        }
    }

    fn named_linears_mut(&mut self, prefix: &str) -> Vec<(String, &mut Linear)> {
        match self {
            Ffn::Dense(m) => m.named_linears_mut(&format!("{prefix}.mlp")),
            Ffn::Moe(_) => Vec::new(),
        }
    }

    /// 只有稠密 GELU 分支能走 GPU 常驻显存路径（SwiGLU 与 MoE 都返回 None → 逐算子回退）
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    fn gelu_weights(&self) -> Option<(&Tensor, &Tensor, &Tensor, &Tensor)> {
        match self {
            Ffn::Dense(m) => m.gelu_weights(),
            Ffn::Moe(_) => None,
        }
    }
}

impl Module for Ffn {
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            Ffn::Dense(m) => m.parameters(),
            Ffn::Moe(m) => m.parameters(),
        }
    }
}

/// Transformer Block（第 11 课）
///
/// 结构（GPT-2 风格，pre-norm）：
///   x -> LayerNorm -> Attention -> 残差 +
///   x -> LayerNorm -> MLP(GELU)  -> 残差 +
///
/// 可通过配置切换为 LLaMA 风格：
///   x -> RMSNorm -> Attention(GQA) -> Dropout -> 残差 +
///   x -> RMSNorm -> SwiGLU MLP     -> Dropout -> 残差 +
///
/// 前馈子层还可切换为稀疏 MoE（第 32 课，`cfg.n_expert > 1` 时）。
struct TransformerBlock {
    ln1: NormLayer,
    attn: MultiHeadAttention,
    ln2: NormLayer,
    ffn: Ffn,
    dropout: f32,
    /// 最近一次前向里 MoE 子层的均衡辅助损失（稠密配置恒为 `None`）。
    /// `forward` 只拿到 `&self`，所以用 `RefCell`；训练循环在 `forward` 之后把它取走。
    aux: RefCell<Option<Tensor>>,
}

impl TransformerBlock {
    fn new(cfg: &GPTConfig, rng: &mut Rng) -> Self {
        let ffn = if cfg.n_expert > 1 {
            Ffn::Moe(MoELayer::new(cfg, rng))
        } else if cfg.use_swiglu {
            Ffn::Dense(MLPEnum::new_swiglu(cfg.n_embd, rng))
        } else {
            Ffn::Dense(MLPEnum::new_gelu(cfg.n_embd, rng))
        };
        TransformerBlock {
            ln1: NormLayer::new(cfg.n_embd, LN_EPS, cfg.use_rmsnorm),
            attn: MultiHeadAttention::new(
                cfg.n_embd,
                cfg.n_head,
                cfg.n_kv_head,
                // 训练窗口取 block_size：YaRN 的分段边界与 NTK 的 factor 都是相对它而言
                cfg.rope_spec(),
                rng,
            ),
            ln2: NormLayer::new(cfg.n_embd, LN_EPS, cfg.use_rmsnorm),
            ffn,
            dropout: cfg.dropout,
            aux: RefCell::new(None),
        }
    }

    /// 本层 MoE 子层的路由统计（稠密配置返回 None）
    fn route_stats(&self) -> Option<crate::moe::RouteStats> {
        match &self.ffn {
            Ffn::Moe(m) => Some(m.stats()),
            Ffn::Dense(_) => None,
        }
    }

    /// 带名字的参数（checkpoint 用）
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps = self.ln1.named_parameters(&format!("{prefix}.ln1"));
        ps.extend(self.attn.named_parameters(&format!("{prefix}.attn")));
        ps.extend(self.ln2.named_parameters(&format!("{prefix}.ln2")));
        ps.extend(self.ffn.named_parameters(prefix));
        ps
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        kv_cache: Option<&mut KVCache>,
        base: usize,
        training: bool,
    ) -> Tensor {
        // 注意力子层 + 残差连接：优先走 GPU 常驻显存路径
        // （ln1 → QKV 投影 → RoPE → attn → c_proj → dropout → 残差 整段录进一次提交）
        // 只在训练（无 KV cache、base=0）时可用 —— RoPE 的位置直接取序列内下标
        //
        // 注意：常驻路径只替换**注意力子层**，不能在这里提前 return —— 后面还有前馈子层，
        // 提前返回会把整叠 Block 的 MLP 全部丢掉。
        #[cfg(feature = "gpu")]
        let attn_resident_out = if kv_cache.is_none() && base == 0 {
            self.attn_resident(x, mask, training)
        } else {
            None
        };
        #[cfg(not(feature = "gpu"))]
        let attn_resident_out: Option<Tensor> = None;

        let x = match attn_resident_out {
            Some(out) => out,
            None => {
                let ln1_out = self.ln1.forward(x);
                let h = self.attn.forward(&ln1_out, mask, kv_cache, base);
                let h = if self.dropout > 0.0 { h.dropout(self.dropout, training) } else { h };
                x.add(&h)
            }
        };
        // 前馈子层 + 残差连接：优先走 GPU 常驻显存路径
        // （ln2 → linear1 → GELU → linear2 → dropout → 残差 整段录进一次提交）
        #[cfg(feature = "gpu")]
        if let Some(out) = self.mlp_resident(&x, training) {
            return out;
        }
        let h = self.ln2.forward(&x);
        let (h, aux) = self.ffn.forward_with_aux(&h);
        *self.aux.borrow_mut() = aux;
        let h = if self.dropout > 0.0 { h.dropout(self.dropout, training) } else { h };
        let out = x.add(&h);
        out
    }

    /// 本层是否有任一子层挂了 LoRA 适配器（两个子层的 GPU 常驻路径各自据此让路）
    fn has_lora(&self) -> bool {
        self.attn.has_lora() || self.ffn.has_lora()
    }

    /// 注意力子层的 GPU 常驻显存前向（见 [`crate::gpu::attn_layer_forward`]）。
    ///
    /// 返回的张量前向数据已在显存里算好，反向闭包直接调 `AttnLayerResident::backward`，
    /// 把 11 项边界梯度（对 x、Q/K/V/输出四个投影的权重与偏置、归一化的 γ/β）
    /// 注回计算图 —— 于是这一段不必再建一张「LN + 四个投影 + RoPE + 注意力 + dropout + 残差」
    /// 的计算图，中间量（Q/K/V、旋转后的 Q/K、注意力输出）也不会流回 CPU。
    ///
    /// 归一化用 LayerNorm 还是 RMSNorm、K/V 是不是 GQA 的头数，在这里都**归一化掉**：
    /// · RMSNorm：没有 β，给内核喂一块零张量占住 β 槽位（值会被内核丢弃），
    ///   反向回传的 `dβ` 也无人接收；
    /// · GQA：把 K/V 权重与偏置按头展开成 `n_head` 个头（[`expand_kv_head`]），
    ///   展开后与标准 MHA 同形，常驻路径整段原样复用；反向再把梯度折回去
    ///   （[`fold_kv_head_grad`]）。
    ///
    /// 不适用时返回 None，调用方回退逐算子路径：推理模式、形状或规模不合适、
    /// GPU 不可用、SwiGLU 前馈，以及**本层挂了 LoRA 或被量化**（见下）。
    ///
    /// 挂了 LoRA 必须整体让路：这条路径直接读 `c_q/c_k/c_v` 的 `weight.data` 显存，
    /// 不走 [`Linear::forward`]，适配器的增量会被静默丢掉——前向照跑、loss 照降，
    /// 但适配层根本没有参与计算（梯度只流到主干，而主干又被优化器跳过）。
    ///
    /// 量化过的层同理必须让路：`weight` 里存的仍是原始 f32 权重，真正参与前向的是
    /// [`Linear::quant`] 里的整数码（还要被反量化、AWQ 还要先把输入除回缩放）。
    /// 内核直接读 `weight` 等于**绕过量化**：数值上倒是不亏（f32 更准），
    /// 但"量化后的模型到底什么效果"就测不出来了，报告与实测会互相矛盾。
    #[cfg(feature = "gpu")]
    fn attn_resident(&self, x: &Tensor, mask: &Tensor, training: bool) -> Option<Tensor> {
        use crate::attention::{expand_kv_head, fold_kv_head_grad};

        if !crate::tensor::grad_enabled() || self.attn.has_lora() || self.attn.has_quant() {
            return None;
        }
        // GPU 内核里的 RoPE 角度按 `base = 10000`、不缩放写死（见 `gpu.rs` 的
        // `ref_rope_theta`）。频率表一旦被缩放或换底数，这条路算出的旋转就与 CPU
        // 参考实现不同 —— 让路，回退逐算子路径。
        if !self.attn.rope.is_plain() {
            return None;
        }
        let shape = x.shape();
        if shape.len() != 3 {
            return None;
        }
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let (gamma, beta_opt, eps, is_rms) = self.ln1.norm_params();
        // RMSNorm 没有 β：内核的 β 绑定槽位仍要一块长度合法的显存，值为 0（内核不读它的值）
        let zero_beta;
        let beta: &Tensor = match beta_opt {
            Some(bt) => bt,
            None => {
                zero_beta = Tensor::from_vec(vec![0.0f32; d], vec![d]);
                &zero_beta
            }
        };
        let (cq, ck, cv, cp) = (
            &self.attn.c_q,
            &self.attn.c_k,
            &self.attn.c_v,
            &self.attn.c_proj,
        );
        // K/V 的头数可能少于 Q（GQA）：把参数按头展开成 n_head 份再喂给常驻路径
        let n_kv = self.attn.n_kv_head;
        let n_rep = self.attn.n_head / n_kv;
        let hd = d / self.attn.n_head;
        let wk_d = ck.weight.data.borrow();
        let bk_d = ck.bias.data.borrow();
        let wv_d = cv.weight.data.borrow();
        let bv_d = cv.bias.data.borrow();
        let (wk_exp, bk_exp, wv_exp, bv_exp) = if n_rep > 1 {
            (
                expand_kv_head(&wk_d[..], d, n_kv, hd, n_rep),
                expand_kv_head(&bk_d[..], 1, n_kv, hd, n_rep),
                expand_kv_head(&wv_d[..], d, n_kv, hd, n_rep),
                expand_kv_head(&bv_d[..], 1, n_kv, hd, n_rep),
            )
        } else {
            (Vec::new(), Vec::new(), Vec::new(), Vec::new())
        };
        let (wk, bk, wv, bv): (&[f32], &[f32], &[f32], &[f32]) = if n_rep > 1 {
            (&wk_exp[..], &bk_exp[..], &wv_exp[..], &bv_exp[..])
        } else {
            (&wk_d[..], &bk_d[..], &wv_d[..], &bv_d[..])
        };
        let res = crate::gpu::attn_layer_forward(&crate::gpu::AttnLayerArgs {
            x: &x.data.borrow(),
            gamma: &gamma.data.borrow(),
            beta: &beta.data.borrow(),
            wq: &cq.weight.data.borrow(),
            bq: &cq.bias.data.borrow(),
            wk,
            bk,
            wv,
            bv,
            wproj: &cp.weight.data.borrow(),
            bproj: &cp.bias.data.borrow(),
            mask: &mask.data.borrow(),
            b,
            t,
            d,
            n_head: self.attn.n_head,
            eps,
            is_rms,
            dropout: self.dropout,
            training,
        })?;
        let out_data = res.out.clone();
        let x_bwd = x.clone();
        // 参数张量不在 parents 里，autograd 不会为它们建图反向 → 由闭包手动注入
        let wq_bwd = cq.weight.clone();
        let bq_bwd = cq.bias.clone();
        let wk_bwd = ck.weight.clone();
        let bk_bwd = ck.bias.clone();
        let wv_bwd = cv.weight.clone();
        let bv_bwd = cv.bias.clone();
        let wp_bwd = cp.weight.clone();
        let bp_bwd = cp.bias.clone();
        let gamma_bwd = Tensor::clone(gamma);
        // RMSNorm 没有 β，不接收 dβ（常驻路径仍会算出来，那份残值无处可去）
        let beta_bwd = beta_opt.map(Tensor::clone);
        Some(Tensor::external(
            out_data,
            vec![b, t, d],
            vec![x.clone()],
            move |grad| {
                Box::new(move || {
                    // 上游梯度此刻已经攒齐（autograd 按逆拓扑序执行）
                    let dout = grad.borrow().clone();
                    let g = res.backward(&dout).expect("注意力常驻显存反向失败");
                    x_bwd.accumulate_grad(&g.dx, 1.0);
                    wq_bwd.accumulate_grad(&g.dwq, 1.0);
                    bq_bwd.accumulate_grad(&g.dbq, 1.0);
                    // K/V 的梯度是**展开空间**（n_head 个头）的，GQA 时折回 n_kv_head 个头
                    if n_rep > 1 {
                        let (dwk, dbk) = (
                            fold_kv_head_grad(&g.dwk, d, n_kv, hd, n_rep),
                            fold_kv_head_grad(&g.dbk, 1, n_kv, hd, n_rep),
                        );
                        let (dwv, dbv) = (
                            fold_kv_head_grad(&g.dwv, d, n_kv, hd, n_rep),
                            fold_kv_head_grad(&g.dbv, 1, n_kv, hd, n_rep),
                        );
                        wk_bwd.accumulate_grad(&dwk, 1.0);
                        bk_bwd.accumulate_grad(&dbk, 1.0);
                        wv_bwd.accumulate_grad(&dwv, 1.0);
                        bv_bwd.accumulate_grad(&dbv, 1.0);
                    } else {
                        wk_bwd.accumulate_grad(&g.dwk, 1.0);
                        bk_bwd.accumulate_grad(&g.dbk, 1.0);
                        wv_bwd.accumulate_grad(&g.dwv, 1.0);
                        bv_bwd.accumulate_grad(&g.dbv, 1.0);
                    }
                    wp_bwd.accumulate_grad(&g.dwproj, 1.0);
                    bp_bwd.accumulate_grad(&g.dbproj, 1.0);
                    gamma_bwd.accumulate_grad(&g.dgamma, 1.0);
                    if let Some(bb) = &beta_bwd {
                        bb.accumulate_grad(&g.dbeta, 1.0);
                    }
                })
            },
        ))
    }

    /// 前馈子层的 GPU 常驻显存前向（见 [`crate::gpu::mlp_forward`]）。
    ///
    /// 返回的张量前向数据已在显存里算好，反向闭包直接调 `MlpResident::backward`，
    /// 把 7 项边界梯度（对 x、两个线性层、LayerNorm 的 γ/β）注回计算图 ——
    /// 于是这一段不必再建一张「LN + 两次矩阵乘 + GELU + dropout + 残差」的计算图，
    /// 中间量（LN 输出、GELU 前后、两层投影结果）也不会流回 CPU。
    ///
    /// 归一化用 LayerNorm 还是 RMSNorm 在这里同样被抹平：RMSNorm 没有 β，
    /// 给内核喂一块零张量占住 β 槽位（值会被内核丢弃），反向回传的 `dβ` 也无人接收。
    ///
    /// 不适用时返回 None，调用方回退逐算子路径：SwiGLU 配置、推理模式
    /// （`no_grad`）、形状或规模不合适、GPU 不可用，以及**本 MLP 挂了 LoRA 或被量化**
    /// （理由同 [`TransformerBlock::attn_resident`]：这条路径直接读权重显存，绕过
    /// [`Linear::forward`]，适配器增量会被静默丢掉、量化也会被绕过）。
    #[cfg(feature = "gpu")]
    fn mlp_resident(&self, x: &Tensor, training: bool) -> Option<Tensor> {
        if !crate::tensor::grad_enabled() || self.ffn.has_lora() || self.ffn.has_quant() {
            return None;
        }
        let shape = x.shape();
        if shape.len() != 3 {
            return None;
        }
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let (gamma, beta_opt, eps, is_rms) = self.ln2.norm_params();
        // RMSNorm 没有 β：内核的 β 绑定槽位仍要一块长度合法的显存，值为 0（内核不读它的值）
        let zero_beta;
        let beta: &Tensor = match beta_opt {
            Some(bt) => bt,
            None => {
                zero_beta = Tensor::from_vec(vec![0.0f32; d], vec![d]);
                &zero_beta
            }
        };
        let (w1, b1, w2, b2) = self.ffn.gelu_weights()?;
        let hid = w1.shape()[1];
        let res = crate::gpu::mlp_forward(
            &x.data.borrow(),
            &gamma.data.borrow(),
            &beta.data.borrow(),
            &w1.data.borrow(),
            &b1.data.borrow(),
            &w2.data.borrow(),
            &b2.data.borrow(),
            b * t,
            d,
            hid,
            eps,
            is_rms,
            self.dropout,
            training,
        )?;
        let out_data = res.out.clone();
        let x_bwd = x.clone();
        let gamma_bwd = Tensor::clone(gamma);
        // RMSNorm 没有 β，不接收 dβ（常驻路径仍会算出来，那份残值无处可去）
        let beta_bwd = beta_opt.map(Tensor::clone);
        let (w1_bwd, b1_bwd) = (Tensor::clone(w1), Tensor::clone(b1));
        let (w2_bwd, b2_bwd) = (Tensor::clone(w2), Tensor::clone(b2));
        Some(Tensor::external(
            out_data,
            vec![b, t, d],
            vec![x.clone()],
            move |grad| {
                Box::new(move || {
                    // 上游梯度此刻已经攒齐（autograd 按逆拓扑序执行）
                    let dout = grad.borrow().clone();
                    let g = res.backward(&dout).expect("MLP 常驻显存反向失败");
                    x_bwd.accumulate_grad(&g.dx, 1.0);
                    w1_bwd.accumulate_grad(&g.dw1, 1.0);
                    b1_bwd.accumulate_grad(&g.db1, 1.0);
                    w2_bwd.accumulate_grad(&g.dw2, 1.0);
                    b2_bwd.accumulate_grad(&g.db2, 1.0);
                    gamma_bwd.accumulate_grad(&g.dgamma, 1.0);
                    if let Some(bb) = &beta_bwd {
                        bb.accumulate_grad(&g.dbeta, 1.0);
                    }
                })
            },
        ))
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.ln1.parameters();
        ps.extend(self.attn.parameters());
        ps.extend(self.ln2.parameters());
        ps.extend(self.ffn.parameters());
        ps
    }
}

/// 完整的 GPT 模型
pub struct GPT {
    pub cfg: GPTConfig,
    tok_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: NormLayer,
    /// Dropout 概率（残差/嵌入层用）
    dropout: f32,
    /// LoRA 微调状态：`Some` 表示主干已冻结、每层 Q/K/V 都挂了适配器（见 [`GPT::apply_lora`]）。
    /// checkpoint 头也记录它，加载时据此重放同样的注入，参数名才能对上。
    pub lora: Option<LoRAConfig>,
    /// 最近一次部署量化的记录（[`GPT::quantize_weights`] 写入，checkpoint 头读出/写入）。
    ///
    /// 它只记**参数**（位宽 / 算法 / 字节口径），不记整数码：存档里的权重始终是 f32
    /// （见 [`crate::checkpoint::save`]），任何现有加载路径都能直接读；真要在部署时省显存，
    /// 加载端按这份记录重放一次量化即可（见 [`crate::checkpoint::requantize_after_load`]）。
    pub(crate) quant: Option<QuantMeta>,
}

/// 校准期一层的统计累加器（[`GPT::calibrate`] 内部用）。
///
/// 之所以先"累加原始和"、最后才除以 token 数：`H = Σ xᵀx` 是**唯一**需要
/// 逐 token 累加的量，中途做除法会白费 `n` 次乘法/除法，而且浮点误差更差
/// （先累加再一次性折算只有一次舍入）。
///
/// `dim` 记的是 `(rows, cols)` = `(输入维度, 输出通道)`：本项目 `Linear.weight`
/// 是 `[in_features, out_features]`，于是 `H` 必须落在**输入维度**上
/// （`[rows, rows]`），`mean|x|` 也按**输入通道**排列——这与
/// [`crate::quant::gptq_quantize`] / [`crate::quant::awq_scales`] 的约定严格一致。
struct Acc {
    /// 该层的参数名前缀（与 checkpoint / 日志里的名字同源）
    name: String,
    /// 前向钩子：`forward` 每次都会把本次输入写进来（`Tensor` 内部是 `Rc`，只加引用计数）
    hook: Rc<RefCell<Tensor>>,
    /// `(输入维度, 输出通道)`
    dim: (usize, usize),
    /// `Σ xᵀx`：`full_hessian` 时是行优先 `[rows, rows]`（对称，两个三角都填，
    /// 读的时候不必分方向），否则只有 `Σ diag(xᵀx)`（长度 `rows`）
    hessian: Vec<f32>,
    /// 是否在存全矩阵。由 [`CalibOpts::full_hessian_max_dim`] 决定：全矩阵是 `4·in²`
    /// 字节/层，`in` 大起来能吃掉几十 MB，超预算时只留对角——
    /// 代价见 [`HessianKind::Diagonal`]（GPTQ 随之退化成 RTN）。
    full_hessian: bool,
    /// 每个输入通道的 `Σ|x|`（长度 = `rows`）
    abs_sum: Vec<f32>,
    /// 已累计的样本 token 数
    rows_seen: usize,
}

impl GPT {
    pub fn new(cfg: GPTConfig, rng: &mut Rng) -> Self {
        // GQA 校验
        let n_kv = if cfg.n_kv_head == 0 { cfg.n_head } else { cfg.n_kv_head };
        assert!(
            cfg.n_head % n_kv == 0,
            "n_head（{}）必须能被 n_kv_head（{}）整除",
            cfg.n_head,
            n_kv
        );
        let n_embd = cfg.n_embd;
        let vocab_size = cfg.vocab_size;
        let blocks = (0..cfg.n_layer)
            .map(|_| TransformerBlock::new(&cfg, rng))
            .collect();
        GPT {
            cfg: cfg.clone(),
            tok_emb: Embedding::new(vocab_size, n_embd, rng),
            blocks,
            ln_f: NormLayer::new(n_embd, LN_EPS, cfg.use_rmsnorm),
            dropout: cfg.dropout,
            lora: None,
            quant: None,
        }
    }

    /// 扩大词表：在词嵌入（= 输出头，权重绑定）表尾追加若干行，新行按 N(0, 0.02) 初始化。
    ///
    /// 用于"给已经训过的模型加新 token"：新增特殊 token（BOS / EOS / PAD）、
    /// 领域词、新语言都靠它。旧行的数值**一行都不动**，新行从零开始学；
    /// 训练侧只需继续训练，新增行就会自己收敛。
    ///
    /// 只支持**变大**：缩小词表要重排索引、还要丢弃对应的输出头行，语义上
    /// 是另一个操作（"裁词表"），不在这里混着做。
    ///
    /// 旧 checkpoint 的 `tok_emb.table` 行数比模型少时，
    /// [`crate::checkpoint::load_params`] 按**前缀行**恢复，新行保留这里的初始化值，
    /// 因此"加 EOS 就要重训"这件事不会发生。
    pub fn resize_vocab(&mut self, new_vocab: usize, rng: &mut Rng) {
        let old_vocab = self.cfg.vocab_size;
        assert!(
            new_vocab >= old_vocab,
            "resize_vocab 只支持扩大词表：{} -> {}",
            old_vocab,
            new_vocab
        );
        if new_vocab == old_vocab {
            return;
        }
        let d = self.cfg.n_embd;
        let std = 0.02; // 与 Embedding::new 的初始化一致，新增行与前缀行同尺度
        let mut data = Vec::with_capacity(new_vocab * d);
        {
            data.extend_from_slice(&self.tok_emb.table.data_ref());
        }
        data.extend((old_vocab * d..new_vocab * d).map(|_| rng.randn() * std));
        self.tok_emb.table = Tensor::param(data, vec![new_vocab, d]);
        self.cfg.vocab_size = new_vocab;
    }

    /// 把模型切成 LoRA 微调形态：**冻结全部主干**，再按 `lora.targets` 给每层挂上低秩适配器。
    ///
    /// 顺序很关键：先冻结（`parameters()` 此时还只有主干参数），再注入——新生的
    /// `lora_a` / `lora_b` 保持可训练，于是 [`Module::trainable_parameters`]
    /// 返回的就恰好是适配层。`requires_grad` 是共享标志，因此哪怕优化器已经按
    /// 主干参数建好了，冻结也会同步生效（不会出现"优化器眼里还是可训练的"）。
    ///
    /// 冻结是**真的**冻结，不只是在优化器里跳过：
    /// - 前向：冻结权重走 [`Tensor::matmul_frozen`]，反向只求 `dx`、不算 `dW`
    /// - 优化器：AdamW / SGD 跳过冻结参数，连带不吃权重衰减
    /// - GPU 常驻显存快路（直接读权重显存、绕过 [`Linear::forward`]）整体让路
    ///
    /// 本方法是**覆盖式**的：已有的适配层会被换成一整套全新的（B = 0），上一轮学到的
    /// 增量随之丢弃。要接着训旧适配层，用 [`GPT::resume_lora`]。
    ///
    /// `rng` 只用于初始化 A（B 恒为 0），随后会被 checkpoint 的真实值覆盖。
    pub fn apply_lora(&mut self, lora: &LoRAConfig, rng: &mut Rng) {
        for p in self.parameters() {
            p.set_requires_grad(false);
        }
        for block in &mut self.blocks {
            block.attn.apply_lora(lora, rng);
            if lora.targets.mlp {
                block.ffn.apply_lora(lora, rng);
            }
        }
        self.lora = Some(lora.clone());
    }

    /// 链式续训：沿用**已经注入**的适配层（数值全部保留），只冻结主干、把 A/B 解冻。
    ///
    /// 与 [`GPT::apply_lora`] 的区别：后者会重挂一套全新的 A/B，把上一轮学到的增量丢掉。
    /// 前提是模型上已有适配层——通常来自 `load_model_and_tokenizer` 按存档头重放注入
    /// （见 [`crate::checkpoint::Checkpoint::lora`]），此时 `self.lora` 里已经是存档的
    /// rank / alpha / targets，不需要也不应该再由命令行指定。
    ///
    /// 顺序与 `apply_lora` 一致：先全冻结（`parameters()` 此刻已含适配层），再把适配层
    /// 逐个解冻——"可训练 = 适配层"这个不变式在两条路径上完全一样。
    pub fn resume_lora(&mut self) {
        let adapters = self.lora_parameters();
        assert!(
            !adapters.is_empty(),
            "resume_lora 要求模型上已有适配层：基座得是 LoRA 存档"
        );
        for p in self.parameters() {
            p.set_requires_grad(false);
        }
        for p in adapters {
            p.set_requires_grad(true);
        }
    }

    /// 把全部适配层的增量合并进主干，并丢弃适配层（推理加速用，见 [`Linear::merge_lora`]）。
    ///
    /// 合并后 [`GPT::has_lora`] 为 false、`self.lora` 为 `None`：模型回到普通形态，
    /// 前向不再有每层那两次小矩阵乘与一次相加。**不可逆**——合并后的存档再也分不出
    /// 主干与适配层，也就无法再链式续训；所以只在推理命令上显式开启（`--merge-lora`），
    /// 训练路径不碰它。
    pub fn merge_lora(&mut self) {
        if !self.has_lora() {
            return;
        }
        for block in &mut self.blocks {
            block.attn.merge_lora();
            block.ffn.merge_lora();
        }
        self.lora = None;
    }

    /// 全部适配层参数（A/B），续训时用来把可训练集合挑出来。
    ///
    /// 按名字筛（`{prefix}.lora_a` / `.lora_b`）而不是另写一套递归遍历：这些名字本就是
    /// checkpoint 对齐参数的依据，用处唯一，再维护一份平行的遍历只会多一处"两本账"。
    pub fn lora_parameters(&self) -> Vec<Tensor> {
        self.named_parameters()
            .into_iter()
            .filter(|(name, _)| name.contains(".lora_"))
            .map(|(_, t)| t)
            .collect()
    }

    /// 是否处于 LoRA 微调形态（直接看各层有没有适配器，而不是读缓存的配置字段——
    /// 少一处"两本账"）
    pub fn has_lora(&self) -> bool {
        self.blocks.iter().any(|b| b.has_lora())
    }

    // ==================== 第 33 课：模型级量化（weight-only） ====================
    //
    // 流程：`calibrate`（采激活统计）→ `quantize_weights`（逐层量化并出报告）→
    // 推理时走 `Linear::forward` 的反量化路径 → 存档前 `dequantize_weights` 烘焙回 f32。
    // 词嵌入表始终不动（理由见 `quantize_weights`）。

    /// 本模型是否有任何投影被量化（直接看各层，不读缓存的记录字段——少一处"两本账"）
    pub fn has_quant(&self) -> bool {
        self.blocks
            .iter()
            .any(|b| b.attn.has_quant() || b.ffn.has_quant())
    }

    /// 与 [`GPT::has_quant`] 同义的可读别名，供部署侧做"是否还要再量化一次"的判断。
    pub fn is_quantized(&self) -> bool {
        self.has_quant()
    }

    /// 可量化投影的 `(参数名前缀, &Linear)`（MoE 除外，见 [`Ffn::named_linears`]）。
    ///
    /// 名字与 [`GPT::named_parameters`] 里 `.weight` 的前缀严格一致——校准统计、
    /// 量化报告、checkpoint 参数表三处共用同一套名字，任何一处对不上都会被立刻发现，
    /// 而不是变成"某层悄悄退回 RTN"。
    fn named_linears(&self) -> Vec<(String, &Linear)> {
        let mut out = Vec::new();
        for (i, block) in self.blocks.iter().enumerate() {
            let prefix = format!("blocks.{i}");
            out.extend(block.attn.named_linears(&format!("{prefix}.attn")));
            out.extend(block.ffn.named_linears(&prefix));
        }
        out
    }

    fn named_linears_mut(&mut self) -> Vec<(String, &mut Linear)> {
        let mut out = Vec::new();
        for (i, block) in self.blocks.iter_mut().enumerate() {
            let prefix = format!("blocks.{i}");
            out.extend(block.attn.named_linears_mut(&format!("{prefix}.attn")));
            out.extend(block.ffn.named_linears_mut(&prefix));
        }
        out
    }

    /// `(原始 f32 字节数, 当前实际占用字节数)`：全部投影 + 词嵌入表。
    /// 词嵌入表参与了分子分母两边（它不量化），所以这里给出的压缩率是**整模型的真实口径**，
    /// 而不是只挑被量化那部分自我感觉良好。
    ///
    /// 注意分子始终是"不量化会占多少"：量化层的 `weight` 已经被换成占位张量，
    /// 口径必须由 [`Linear::weight_bytes`] 按逻辑维度给出，不能去数实际字节。
    pub fn weight_bytes(&self) -> (usize, usize) {
        let emb = self.tok_emb.table.numel() * 4;
        let mut orig = emb;
        let mut cur = emb;
        for (_, lin) in self.named_linears() {
            let (o, c) = lin.weight_bytes();
            orig += o;
            cur += c;
        }
        (orig, cur)
    }

    /// 量化前整模型的 f32 权重字节数（投影 + 词嵌入表）。
    /// 与 [`GPT::quant_bytes`] 配对使用，两者之比才是真实的压缩率。
    pub fn f32_bytes(&self) -> usize {
        self.weight_bytes().0
    }

    /// 量化后整模型权重实际占用的字节数（整数码 + 分组 scale + AWQ 输入缩放 + 未量化的词嵌入表）
    pub fn quant_bytes(&self) -> usize {
        self.weight_bytes().1
    }

    /// 这份模型当前的量化记录（没量化过则为 `None`）
    pub fn quant_meta(&self) -> Option<QuantMeta> {
        self.quant
    }

    /// 用一批文本做**激活校准**：跑一遍前向，把每层输入的二阶统计采下来。
    ///
    /// - GPTQ 要 `H = XᵀX`（输入通道之间的相关性），逐列补偿误差全靠它；
    /// - AWQ 要每通道的 `mean|x|`（哪些通道"重要"），据此决定把动态范围让给谁。
    ///
    /// 全程 `no_grad`：校准只读激活、不碰梯度（这些统计量不需要求导，建图纯属浪费内存）。
    /// 前向走 [`GPT::forward_hidden`] 而不是 [`GPT::forward`]——输出头那次
    /// `[tokens, vocab]` 的大矩阵乘对任何一层的输入统计都没有贡献。
    ///
    /// 选项见 [`CalibOpts`]：
    /// - `max_tokens`：校准集最多喂多少 token。统计量按 token 累加，越多越准；
    ///   几百个 token 就足以让 `H` 摆脱采样噪声（样本数远大于通道数），
    ///   而 `H⁻¹` 会把噪声放大，所以**宁可多喂一点**也别省这一步。
    /// - `max_seq`：一次前向喂多长（会被 `block_size` 截断）。用整段连续文本比拆成
    ///   单 token 更好：同一通道的长程相关只有在长序列里才显现得出来。
    /// - `full_hessian_max_dim`：**按 `in_features` 逐层决定**存不存全矩阵。校准统计
    ///   逐层独立，所以"宽层退化成对角、窄层保留全矩阵"是完全合法的混合策略——
    ///   显存开销由最宽的那一层决定，而不是由"最宽那层的全矩阵"决定。之所以要有这个
    ///   旋钮：全矩阵是 `4·in²` 字节/层，`in = 4096` 时单层 64 MB，比很多层的权重还大；
    ///   而退化成对角后 `H⁻¹` 也是对角阵，补偿量恒为 0，GPTQ 随之变成 RTN。
    ///   代价的完整说明见 [`CalibOpts::full_hessian_max_dim`] 与 [`HessianKind`]。
    ///
    /// 结束后钩子（[`Linear::capture`]）会被拆掉：它只在采集期有意义，
    /// 留着会让之后每次前向都白存一份输入。
    pub fn calibrate(
        &mut self,
        tokenizer: &Tokenizer,
        texts: &[String],
        opts: CalibOpts,
    ) -> CalibStats {
        // 1) 编码成一条 token 流并按预算截断
        let mut ids: Vec<usize> = Vec::new();
        for t in texts {
            ids.extend(tokenizer.encode(t));
            if ids.len() >= opts.max_tokens {
                break;
            }
        }
        ids.truncate(opts.max_tokens);
        let seq = opts.max_seq.clamp(1, self.cfg.block_size.max(1));
        if ids.is_empty() {
            return CalibStats::default();
        }

        // 2) 逐投影装钩子，并准备累加器。
        //    维度走 `lin.dims()`（而不是 `weight.shape()`）：对已量化的层来说
        //    `weight` 只是 1 元素占位张量，直接读形状会把 Hessian 的轴算错。
        let mut accs: Vec<Acc> = Vec::with_capacity(self.blocks.len() * 8);
        for (name, lin) in self.named_linears_mut() {
            let (rows, cols) = lin.dims();
            let full_hessian = rows <= opts.full_hessian_max_dim;
            let hook = Rc::new(RefCell::new(Tensor::from_vec(vec![0.0], vec![1])));
            lin.capture = Some(hook.clone());
            accs.push(Acc {
                name,
                hook,
                dim: (rows, cols),
                hessian: vec![0.0; if full_hessian { rows * rows } else { rows }],
                full_hessian,
                abs_sum: vec![0.0; rows],
                rows_seen: 0,
            });
        }

        // 3) 逐段前向并累加。位置 t 的输入是真实数据流里的第 t 个 token（因果掩码只影响
        //    它能"看到"什么，不影响它自己的输入激活），所以整段的每个位置都是合法样本。
        for chunk in ids.chunks(seq) {
            crate::tensor::no_grad(|| {
                let _ = self.forward_hidden(chunk, 1, chunk.len(), false);
            });
            for acc in accs.iter_mut() {
                let t = acc.hook.borrow();
                let x = t.data_ref();
                // 钩子里存的是这一层的输入，形状 `[tokens, 输入维度]`——累加的宽度必须用
                // `dim.0`（输入维度）而不是 `dim.1`（输出通道），否则要么切片越界，
                // 要么把统计算到错误的轴上（Hessian 落错轴 = GPTQ 的补偿方向全反）。
                let (inp, n) = (acc.dim.0, t.shape()[0]);
                if t.rank() != 2 || t.shape()[1] != inp {
                    continue; // 钩子没被这次前向碰到（例如该层不存在）——跳过即可
                }
                for r in 0..n {
                    let row = &x[r * inp..(r + 1) * inp];
                    if acc.full_hessian {
                        for i in 0..inp {
                            // 对称累加：H 只用下三角做 Cholesky，但两个三角都填更省事也更不容易错
                            for j in i..inp {
                                let v = row[i] * row[j];
                                acc.hessian[i * inp + j] += v;
                                if j != i {
                                    acc.hessian[j * inp + i] += v;
                                }
                            }
                        }
                    } else {
                        // 对角模式：只累 `Σ x_i²`，省掉全部非对角项的计算与存储。
                        // 注意这是与 `abs_sum` 独立的一份平方和——不能用 `Σ|x|`
                        // 去近似（两者差着量级），否则 act-order 的重要性排序会整体失真。
                        for i in 0..inp {
                            acc.hessian[i] += row[i] * row[i];
                        }
                    }
                    for i in 0..inp {
                        acc.abs_sum[i] += row[i].abs();
                    }
                }
                acc.rows_seen += n;
            }
        }

        // 4) 拆钩子，把"累加和"折算成"均值"（H = E[xᵀx]、激活 = E|x|）
        for (_, lin) in self.named_linears_mut() {
            lin.capture = None;
        }
        let mut stats = CalibStats::default();
        for acc in accs {
            if acc.rows_seen == 0 {
                continue;
            }
            let inv = 1.0 / acc.rows_seen as f32;
            // 累加用的是裸和 `Σ xᵀx`，而 GPTQ 的阻尼、act-order 的重要性都按
            // "均值"的量级来定（`trace(H)/n`），所以这里必须一次性折算；
            // 折算因子对全矩阵与对角是同一个，两条路径的统计口径完全一致。
            let scaled: Vec<f32> = acc.hessian.iter().map(|v| v * inv).collect();
            let hessian = if acc.full_hessian {
                HessianKind::Full(scaled)
            } else {
                HessianKind::Diagonal(scaled)
            };
            stats.per_layer.push((
                acc.name,
                LayerCalib {
                    hessian: Some(hessian),
                    act_abs_mean: Some(acc.abs_sum.iter().map(|v| v * inv).collect()),
                    n_tokens: acc.rows_seen,
                },
            ));
        }
        stats
    }

    /// 逐层量化全部投影，返回本次量化的报告（同时把记录写进 [`GPT::quant_meta`]）。
    ///
    /// - `opts` 带全了算法参数：分组方向（默认 [`QAxis::Col`] = **每个输出通道一个
    ///   scale**，同一列的元素共同决定一个输出通道的贡献，数值尺度最接近，是权重量化的
    ///   通行口径；[`QAxis::Row`] 是"每个输入通道一个 scale"，本项目留给 KV cache 的 K）、
    ///   GPTQ 的 `act_order`/`damp`/`block`、AWQ 的 α（`None` = 逐层在网格上搜索）。
    /// - `calib` 为 `None` 或某层缺统计时，该层**退回 RTN**：量化是部署前的最后一步，
    ///   在这里失败会卡死整条流水线，而"这层没吃到补偿"只是精度略降。
    /// - 词嵌入表不量化：它的行由词表决定、前向时每个 token 只用一行，
    ///   省下来的字节远不如"每个 token 都要整块参与矩阵乘"的投影；而它的输出直接
    ///   决定采样分布，是数值上最敏感的地方。收益小、代价大，留 f32。
    /// - 已挂 LoRA 适配器的层会被跳过（见 [`Linear::quantize_weight`]）：量化会只看
    ///   主干权重，适配器带来的那点增量会被静默丢掉。调用方应先 [`GPT::merge_lora`]。
    ///
    /// 返回**逐层**报告（`Vec<QuantReport>`），要整模型口径的摘要（总压缩比、最差层、
    /// 跳过层数）再调一次 [`GPT::quant_summary`]。
    pub fn quantize_weights(
        &mut self,
        bits: QBits,
        method: QuantMethod,
        calib: Option<&CalibStats>,
        opts: QuantOpts,
    ) -> Vec<QuantReport> {
        let mut reports = Vec::new();
        for (name, lin) in self.named_linears_mut() {
            let stats = calib.and_then(|c| c.get(&name));
            if let Some(r) = lin.quantize_weight(&name, bits, method, &opts, stats) {
                reports.push(r);
            }
        }
        // 记录用**量化完成后**的字节口径：`weight_bytes` 里未量化的词嵌入表
        // 参与分子分母两边，所以压缩率是整模型的真实数字，而不是只挑被量化那部分
        // 自我感觉良好。
        let (orig_bytes, quant_bytes) = self.weight_bytes();
        self.quant = Some(QuantMeta {
            bits,
            method,
            orig_bytes,
            quant_bytes,
        });
        reports
    }

    /// 把逐层报告汇总成**整模型口径**的摘要（`quant` 子命令的打印入口）。
    ///
    /// 总字节在这里现算（而不是抄 [`QuantMeta`] 里的快照）：摘要在
    /// [`GPT::dequantize_weights`] 之后再调用，就该如实反映"已经烘焙回 f32"这个事实。
    /// 跳过的层数由"可量化投影总数 − 实到的报告数"推出——[`Linear::quantize_weight`]
    /// 返回 `None` 的唯一原因就是"这层不该量化"（LoRA 或形状不合法），
    /// 所以这个减法不会把别的原因算成"跳过"。
    pub fn quant_summary(&self, reports: Vec<QuantReport>) -> QuantSummary {
        let meta = self.quant.expect(
            "quant_summary 必须在 quantize_weights 之后调用：没有 QuantMeta 就不知道算法与位宽",
        );
        let (f32_bytes, quant_bytes) = (self.f32_bytes(), self.quant_bytes());
        QuantSummary {
            method: meta.method,
            bits: meta.bits,
            f32_bytes,
            quant_bytes,
            skipped: self.named_linears().len().saturating_sub(reports.len()),
            layers: reports,
        }
    }

    /// 把所有量化层**烘焙**回 f32 权重（AWQ 的输入缩放也一并除掉），但保留
    /// [`GPT::quant_meta`] 的记录。
    ///
    /// 存档里写的始终是 f32（见 [`crate::checkpoint::save`]）：任何现有加载路径都能直接读，
    /// 不必让 checkpoint 格式长出第二种参数编码。记录留住是为了让加载端知道
    /// "这份存档当初是怎么量化的"，需要时重放一次即可。
    pub fn dequantize_weights(&mut self) {
        for block in &mut self.blocks {
            block.attn.dequantize_weights();
            block.ffn.dequantize_weights();
        }
    }

    /// 最近一次 `forward` 里各层 MoE 均衡辅助损失之和，稠密配置恒为 `None`。
    ///
    /// 训练循环必须把它加到主损失上再 `backward()`：
    ///
    /// ```text
    /// L = L_ce + Σ_layers α · L_aux
    /// ```
    ///
    /// 辅助损失**只回流到路由器**（`f_i` 来自 argmax，是常数；梯度只经平均概率 `p_i`），
    /// 所以它不会干扰专家本身的拟合——它的唯一职责是别让路由器赢者通吃。
    /// 不加它也不会报错，只是几十步后少数几个专家会吃掉几乎全部 token，
    /// 其余专家的参数拿不到任何梯度，等于白占显存（`moe` 子命令的对照实验会展示这一点）。
    pub fn aux_loss(&self) -> Option<Tensor> {
        let mut acc: Option<Tensor> = None;
        for b in &self.blocks {
            if let Some(a) = b.aux.borrow().as_ref() {
                acc = Some(match acc {
                    None => a.clone(),
                    Some(prev) => prev.add(a),
                });
            }
        }
        acc
    }

    /// 各层 MoE 的路由统计（合并成一份），稠密配置返回 `None`。
    /// 训练日志用来看负载是否被均衡、有多少 token 因容量被丢弃。
    pub fn route_stats(&self) -> Option<crate::moe::RouteStats> {
        let stats: Vec<_> = self.blocks.iter().filter_map(|b| b.route_stats()).collect();
        if stats.is_empty() {
            None
        } else {
            Some(crate::moe::merge_stats(&stats))
        }
    }

    /// 设置所有 MoE 层的容量因子，返回是否真的改到了（稠密配置返回 false）。
    ///
    /// **推理前应当设 0（不限容量）**。容量是训练时给显存与 All-to-All 通信量封顶才引入的：
    /// 它让同一个 token 的输出取决于同批次里别的 token 挤没挤占容量——训练时这只是噪声，
    /// 推理时却变成"同一句话换个批大小就换个答案"。
    pub fn set_moe_capacity_factor(&mut self, cf: f32) -> bool {
        let mut any = false;
        for b in &mut self.blocks {
            if let Ffn::Moe(m) = &mut b.ffn {
                m.capacity_factor = cf;
                any = true;
            }
        }
        any
    }

    /// 推理期改写 RoPE 频率参数与上下文长度（第 20 课长度外推），返回改前的窗口。
    ///
    /// 可以直接套在**已加载的权重**上：`rope_base` / `rope_scaling` / `block_size`
    /// 三者都不参与任何参数的形状，权重是按"相对位置"训练的，换一张频率表只是把
    /// 这套相对位置关系引到更长的位置区间上。这正是各家"免训练扩上下文"的做法。
    ///
    /// `train_ctx` 必须填**训练时**的窗口（`rope_train_ctx`），而不是 `new_block_size`：
    /// YaRN 按"该维度相对训练窗口转了多少圈"分段，用放大后的窗口去算会把分段整体推偏。
    /// 传 0 表示"与 `new_block_size` 相同"。
    pub fn set_rope(&mut self, base: f32, scaling: RopeScaling, train_ctx: usize, new_block_size: usize) -> usize {
        let old = self.cfg.block_size;
        self.cfg.block_size = new_block_size;
        self.cfg.rope_base = base;
        self.cfg.rope_scaling = scaling;
        self.cfg.rope_train_ctx = if train_ctx == 0 { new_block_size } else { train_ctx };
        let spec = self.cfg.rope_spec();
        for b in &mut self.blocks {
            b.attn.rope = spec;
        }
        old
    }

    /// 前向传播
    ///
    /// - idx: [B*T] 展平的 token id
    /// - b / t：batch 与序列长度
    /// - kv_cache: Some(每层一个缓存) 时启用 KV cache（推理模式）
    /// - training: 是否在训练模式（影响 dropout）
    ///
    /// 返回 logits：[B*T, vocab_size]（每个位置预测"下一个 token"的分数）
    pub fn forward(
        &self,
        idx: &[usize],
        b: usize,
        t: usize,
        kv_cache: Option<&mut Vec<KVCache>>,
        training: bool,
    ) -> Tensor {
        // 权重绑定：lm_head 复用 tok_emb.table 的转置
        self.forward_core(idx, b, t, kv_cache, training)
            .matmul(&self.tok_emb.table.transpose())
    }

    /// 前向到 ln_f 之后的 hidden `[B*T, d]`，不做输出头。
    ///
    /// 供 GPU 常驻输出头路径使用：那条路径把 `hidden @ Wᵀ` 与交叉熵一起放进显存算，
    /// 不需要中间那份 `[B*T, vocab]` 的 logits 被拉回 CPU（本配置下 33.6M 元素）。
    pub fn forward_hidden(&self, idx: &[usize], b: usize, t: usize, training: bool) -> Tensor {
        self.forward_core(idx, b, t, None, training)
    }

    /// 同 [`GPT::forward_hidden`]，但**带 KV cache**。
    ///
    /// 推测解码的多 Token 预测草稿（见 [`crate::speculative::MtpDrafter`]）需要
    /// "增量缓存 + 隐状态"两者兼得：只走 [`GPT::forward`] 拿不到隐状态，
    /// 只走 [`GPT::forward_hidden`] 则每轮都要把整段上下文重算一遍，
    /// 草稿那点省下来的时间又全还回去了。
    pub fn forward_hidden_cached(
        &self,
        idx: &[usize],
        b: usize,
        t: usize,
        kv_cache: Option<&mut Vec<KVCache>>,
        training: bool,
    ) -> Tensor {
        self.forward_core(idx, b, t, kv_cache, training)
    }

    /// 输出头权重 `[vocab, d]`（与词嵌入共享同一份参数）
    pub fn head_weight(&self) -> &Tensor {
        &self.tok_emb.table
    }

    /// 整叠 Transformer Block 的 GPU 常驻显存前向（见 [`crate::gpu::stack_forward`]）。
    ///
    /// 与逐子层常驻路径（[`TransformerBlock::attn_resident`] / [`TransformerBlock::mlp_resident`]）
    /// 的区别在于**子层边界也留在显存**：那两条路径每个子层都要把输出回读 2MB 给
    /// `Tensor::external` 当前向数据、下一层再原样传回 2MB；这里整叠一次提交，
    /// 只回读整叠的最终输出，反向同理只上传一次上游梯度，层与层之间的 dX 在显存接力。
    ///
    /// 不适用时返回 None，调用方回退逐 Block 路径（逐子层常驻 → 逐算子），数值行为不变。
    /// 与前两条常驻路径同理，LoRA 形态下整段让路（适配器增量不在显存里算）、
    /// 量化过的模型也整段让路（内核直接读 `weight` 的 f32 原始权重，等于绕过量化）。
    ///
    /// 归一化用 LayerNorm 还是 RMSNorm、K/V 是不是 GQA 的头数，同样在**参数层面**抹平
    /// （做法与 [`TransformerBlock::attn_resident`] 一致）：RMSNorm 的 β 槽位填零张量，
    /// GQA 的 K/V 参数按头展开成 `n_head` 份；反向再把 K/V 的梯度折回。
    #[cfg(feature = "gpu")]
    fn blocks_resident(
        &self,
        x: &Tensor,
        mask: &Tensor,
        b: usize,
        t: usize,
        training: bool,
    ) -> Option<Tensor> {
        use crate::attention::{expand_kv_head, fold_kv_head_grad};
        use crate::gpu::{STACK_PARAMS_PER_LAYER, StackArgs, StackLayerArgs};
        if !crate::tensor::grad_enabled() || self.blocks.is_empty() || self.has_lora() || self.has_quant()
        {
            return None;
        }
        let d = self.cfg.n_embd;
        let (_, _, eps, is_rms) = self.blocks[0].ln1.norm_params();
        // K/V 的头数可能少于 Q（GQA）：整叠同样把参数按头展开成 n_head 份
        let n_head = self.cfg.n_head;
        let n_kv = self.blocks[0].attn.n_kv_head;
        let n_rep = n_head / n_kv;
        let hd = d / n_head;
        // 每层的 16 个参数张量，顺序必须与 `StackLayerArgs` 的字段顺序一致
        // （`gpu::StackGrads::grads` 也按这个顺序回来）
        let mut params: Vec<Tensor> =
            Vec::with_capacity(STACK_PARAMS_PER_LAYER * self.blocks.len());
        // GQA 展开后的 K/V 参数（顺序 Wk, bk, Wv, bv），按层对齐；n_rep == 1 时为空
        let mut kv_expanded: Vec<[Vec<f32>; 4]> = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (gamma1, beta1_opt, _, _) = block.ln1.norm_params();
            let (gamma2, beta2_opt, _, _) = block.ln2.norm_params();
            let (w1, b1, w2, b2) = block.ffn.gelu_weights()?;
            let (cq, ck, cv, cp) = (
                &block.attn.c_q,
                &block.attn.c_k,
                &block.attn.c_v,
                &block.attn.c_proj,
            );
            // RMSNorm 没有 β：槽位仍要一块长度合法的张量顶上（内核在 RMS 模式下一律丢弃）
            let zero_beta = || Tensor::from_vec(vec![0.0f32; d], vec![d]);
            params.extend([
                gamma1.clone(),
                beta1_opt.cloned().unwrap_or_else(zero_beta),
                cq.weight.clone(),
                cq.bias.clone(),
                ck.weight.clone(),
                ck.bias.clone(),
                cv.weight.clone(),
                cv.bias.clone(),
                cp.weight.clone(),
                cp.bias.clone(),
                gamma2.clone(),
                beta2_opt.cloned().unwrap_or_else(zero_beta),
                w1.clone(),
                b1.clone(),
                w2.clone(),
                b2.clone(),
            ]);
            if n_rep > 1 {
                let (wk, bk) = (ck.weight.data.borrow(), ck.bias.data.borrow());
                let (wv, bv) = (cv.weight.data.borrow(), cv.bias.data.borrow());
                kv_expanded.push([
                    expand_kv_head(&wk[..], d, n_kv, hd, n_rep),
                    expand_kv_head(&bk[..], 1, n_kv, hd, n_rep),
                    expand_kv_head(&wv[..], d, n_kv, hd, n_rep),
                    expand_kv_head(&bv[..], 1, n_kv, hd, n_rep),
                ]);
            }
        }
        // 借用只在这一段里存活：`guards` 借用了 `params`，出块后 `params` 才能进反向闭包
        let res = {
            let guards: Vec<std::cell::Ref<'_, Vec<f32>>> =
                params.iter().map(|p| p.data.borrow()).collect();
            let layers: Vec<StackLayerArgs> = (0..self.blocks.len())
                .map(|i| {
                    let s = &guards[i * STACK_PARAMS_PER_LAYER..][..STACK_PARAMS_PER_LAYER];
                    // GQA 时 K/V 用展开后的参数，其余槽位一律取原始张量
                    let (wk, bk, wv, bv): (&[f32], &[f32], &[f32], &[f32]) = if n_rep > 1 {
                        let e = &kv_expanded[i];
                        (&e[0][..], &e[1][..], &e[2][..], &e[3][..])
                    } else {
                        (&s[4][..], &s[5][..], &s[6][..], &s[7][..])
                    };
                    StackLayerArgs {
                        gamma1: &s[0][..],
                        beta1: &s[1][..],
                        wq: &s[2][..],
                        bq: &s[3][..],
                        wk,
                        bk,
                        wv,
                        bv,
                        wproj: &s[8][..],
                        bproj: &s[9][..],
                        gamma2: &s[10][..],
                        beta2: &s[11][..],
                        w1: &s[12][..],
                        b1: &s[13][..],
                        w2: &s[14][..],
                        b2: &s[15][..],
                    }
                })
                .collect();
            let xd = x.data.borrow();
            let md = mask.data.borrow();
            crate::gpu::stack_forward(&StackArgs {
                x: &xd,
                mask: &md,
                layers: &layers,
                b,
                t,
                d,
                n_head,
                eps,
                is_rms,
                dropout: self.dropout,
                training,
            })?
        };
        let out_data = res.out.clone();
        let x_bwd = x.clone();
        Some(Tensor::external(
            out_data,
            vec![b, t, d],
            vec![x.clone()],
            move |grad| {
                Box::new(move || {
                    // 上游梯度此刻已经攒齐（autograd 按逆拓扑序执行）
                    let dout = grad.borrow().clone();
                    let g = res.backward(&dout).expect("整叠 Block 常驻显存反向失败");
                    x_bwd.accumulate_grad(&g.dx, 1.0);
                    // 参数张量不在 parents 里，autograd 不会为它们建图反向 → 由闭包手动注入
                    for (k, pg) in g.grads.iter().enumerate() {
                        let slot = k % STACK_PARAMS_PER_LAYER;
                        // RMSNorm 没有 β：写回的 dβ 是「Σ dy」的残值，无人接收
                        if is_rms && (slot == 1 || slot == 11) {
                            continue;
                        }
                        // K/V 的梯度是**展开空间**（n_head 个头）的，GQA 时折回 n_kv_head 个头
                        if n_rep > 1 && (4..8).contains(&slot) {
                            // 4=Wk, 5=bk, 6=Wv, 7=bv：偶数槽位是权重（rows=d），奇数是偏置（rows=1）
                            let rows = if slot % 2 == 0 { d } else { 1 };
                            let folded = fold_kv_head_grad(pg, rows, n_kv, hd, n_rep);
                            params[k].accumulate_grad(&folded, 1.0);
                        } else {
                            params[k].accumulate_grad(pg, 1.0);
                        }
                    }
                })
            },
        ))
    }

    /// 前向主体：embedding → 逐层 Block → 最终归一化，返回 `[B*T, d]`
    fn forward_core(
        &self,
        idx: &[usize],
        b: usize,
        t: usize,
        mut kv_cache: Option<&mut Vec<KVCache>>,
        training: bool,
    ) -> Tensor {
        let d = self.cfg.n_embd;
        assert_eq!(idx.len(), b * t, "输入 id 数量必须等于 b*t");

        // 1. token embedding
        let x = self.tok_emb.forward(idx).reshape(vec![b, t, d]);
        let x = if self.dropout > 0.0 { x.dropout(self.dropout, training) } else { x };

        // 2. 位置信息由 RoPE 提供（在注意力内部旋转 Q/K，见 MultiHeadAttention::forward）。
        //    RoPE 的基准是**绝对位置**：新 token 的绝对位置 = 缓存已见位置总数 + 窗口内下标 j。
        //    滑动窗口只丢弃最旧的 K/V，绝对位置始终连续，因此相对距离与全量重算一致。
        let (seen, window) = kv_cache
            .as_ref()
            .and_then(|c| c.first())
            .map(|k| (k.positions_seen(), k.window()))
            .unwrap_or((0, 0));
        let base = seen;

        // 3. 因果掩码：scores 形状 [B*H, T, T_total]，广播 mask [T, T_total]
        //    缓存本次之后保留 min(seen + t, window) 个位置（window = 0 表示不丢弃）
        assert!(
            window == 0 || t <= window,
            "单次前向的 token 数（{t}）不能超过 KV cache 窗口（{window}）"
        );
        let t_total = if window == 0 {
            seen + t
        } else {
            (seen + t).min(window)
        };
        // 缓存里位于"本次新增的第一个 token"左侧（含自身）的位置数：
        // query i 只能看到下标 j <= visible_before + i 的 key
        let visible_before = t_total - t;
        let mut mask_data = vec![0.0f32; t * t_total];
        for i in 0..t {
            for j in 0..t_total {
                if j > i + visible_before {
                    mask_data[i * t_total + j] = f32::NEG_INFINITY;
                }
            }
        }
        let mask = Tensor::from_vec(mask_data, vec![t, t_total]);

        // 4. 整叠 Block 的 GPU 常驻快路：子层边界也留在显存（前向/反向各一次提交）
        //
        //    **默认关闭**：数值上与逐子层路径 bit-exact，实测也更快
        //    （batch=8/block=512/n_layer=4 的 ABBA 对照：0.59/0.60 vs 0.67/0.70 s/步，约快 12%），
        //    因为逐子层路径按子层粒度提交（每层的注意力/前馈、前向/反向各一次），
        //    整叠把每步压成 2 次提交，省掉的是每次提交后的 poll + 回读等待。
        //
        //    仍然默认关闭：整叠的准入条件更严——要求**所有** Block 都满足常驻条件
        //    （LayerNorm + GELU MLP），任一层不满足就整条路径放弃；逐子层则是哪个子层
        //    不满足就只回退那一个。想复现对照实验或追求吞吐时设环境变量
        //    `LLM_GPU_STACK`（**只要设置了就生效**，值本身不参与判断，`is_some`）。
        #[cfg(feature = "gpu")]
        if kv_cache.is_none() && base == 0 && std::env::var_os("LLM_GPU_STACK").is_some() {
            if let Some(out) = self.blocks_resident(&x, &mask, b, t, training) {
                return self.ln_f.forward(&out).reshape(vec![b * t, d]);
            }
        }

        // 5. 逐层过 Transformer Block
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = kv_cache.as_mut().map(|c| &mut c[i]);
            x = block.forward(&x, &mask, cache, base, training);
        }

        // 6. 最终归一化（输出头由 forward / 常驻显存路径各自完成）
        let x = self.ln_f.forward(&x);
        x.reshape(vec![b * t, d])
    }

    /// 推理用的缓存集合：每层一个。
    ///
    /// 每层都带 `block_size` 的滑动窗口：模型能看多远由 `block_size` 决定，
    /// 缓存保留更多位置也没有用，反而让显存随生成长度无界增长。设成窗口后
    /// 生成长度不再受缓存容量限制，KV cache 模式与全量重算模式行为一致。
    pub fn new_kv_cache(&self) -> Vec<KVCache> {
        self.new_kv_cache_with(0, None)
    }

    /// 同上，但可以指定 Attention Sink 个数与缓存量化位宽（第 33 课）。
    ///
    /// - `sink > 0`：超窗丢弃时永久保留最前面的 `sink` 个位置（StreamingLLM）。
    ///   流式生成想看到"丢开头也不崩"就必须开它，见 [`KvCacheOpts::sink`]。
    /// - `bits = Some(_)`：缓存以 int8/int4 存放（KIVI：K 逐通道、V 逐 token），
    ///   显存占用降到 1/4 或 1/8，代价是每次读回时反量化。
    pub fn new_kv_cache_with(&self, sink: usize, bits: Option<QBits>) -> Vec<KVCache> {
        let opts = KvCacheOpts {
            window: self.cfg.block_size,
            sink,
            bits,
        };
        (0..self.cfg.n_layer)
            .map(|_| KVCache::new(opts))
            .collect()
    }

    /// 带名字的参数列表（checkpoint 保存/恢复用）。
    /// 名字形如 `blocks.0.attn.c_q.weight`。
    /// 名字由各层的 `named_parameters(prefix)` 递归生成，与 `Module::parameters` 的
    /// 结构保持一致（同一层只枚举一次，避免两处手工维护失同步）。
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut ps = self.tok_emb.named_parameters("tok_emb");
        for (i, block) in self.blocks.iter().enumerate() {
            ps.extend(block.named_parameters(&format!("blocks.{i}")));
        }
        ps.extend(self.ln_f.named_parameters("ln_f"));
        ps
    }
}

impl Module for GPT {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.tok_emb.parameters();
        for block in &self.blocks {
            ps.extend(block.parameters());
        }
        ps.extend(self.ln_f.parameters());
        ps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checkpoint;
    use crate::loss::cross_entropy_loss;
    use crate::module::{Module, zero_grad_all};
    use crate::optim::{AdamW, Optimizer};

    /// RoPE + KV cache 一致性（第 18-19 课）：
    /// 用 KV cache 分步推理得到"位置 10 的 logits"，应与全量前向 11 个 token 的最后一行一致。
    /// 这同时验证了：RoPE 的旋转位置（base + j）与因果掩码在两种模式下行为一致。
    #[test]
    fn test_kv_cache_matches_full_forward() {
        let mut rng = Rng::new(42);
        let model = GPT::new(GPTConfig::tiny(32), &mut rng);
        let v = model.cfg.vocab_size;
        let seq = vec![1, 5, 7, 3, 9, 2, 8, 4, 6, 0]; // 10 个 token，都小于词表 32

        // KV cache 模式：先喂完整序列填缓存，再只前向 1 个新 token（位置 10）
        let mut cache = model.new_kv_cache();
        let _ = model.forward(&seq, 1, seq.len(), Some(&mut cache), false);
        let new_id = 3;
        let one = model.forward(&[new_id], 1, 1, Some(&mut cache), false);
        let last_one = one.data()[one.numel() - v..].to_vec();

        // 全量模式：一次前向 [seq..., new_id]（11 个 token），取最后一个位置（位置 10）
        let mut seq2 = seq;
        seq2.push(new_id);
        let full = model.forward(&seq2, 1, seq2.len(), None, false);
        let last_full = full.data()[full.numel() - v..].to_vec();

        assert!(
            last_one
                .iter()
                .zip(&last_full)
                .all(|(a, b)| (a - b).abs() < 1e-4),
            "KV cache 推理与全量前向的同一位置 logits 应一致"
        );
    }

    /// 推理期换频率表 + 扩窗口（第 20 课长度外推）：
    /// `set_rope` 只改配置与各层的频率参数、不动任何权重，改完之后模型必须能在
    /// **超过训练窗口**的位置上照常前向，KV cache 也按新窗口保留全部行、不提前丢弃。
    ///
    /// 这里顺带守住一个容易漏的地方：`set_rope` 必须把新 spec 写到**每一层**上。
    /// 只改 `cfg` 不改层的话，前向仍按老频率旋转，而日志里显示的却是新设置——
    /// 这种"配置与计算不一致"是最难查的一类 bug。
    #[test]
    fn test_set_rope_extends_window_without_touching_weights() {
        let mut rng = Rng::new(7);
        let mut model = GPT::new(GPTConfig::tiny(32), &mut rng);
        let train_ctx = model.cfg.rope_train_ctx();
        let new_ctx = train_ctx * 4;

        let old = model.set_rope(rope::ROPE_BASE, RopeScaling::Ntk { factor: 4.0 }, train_ctx, new_ctx);
        assert_eq!(old, train_ctx, "返回值应是改前的窗口");
        assert_eq!(model.cfg.block_size, new_ctx);
        assert_eq!(model.cfg.rope_train_ctx(), train_ctx, "训练窗口不该被外推窗口覆盖");

        let spec = model.cfg.rope_spec();
        assert!(!spec.is_plain(), "NTK 缩放后频率表已与 GPU 内核写死的式子不同");
        for (i, b) in model.blocks.iter().enumerate() {
            assert_eq!(b.attn.rope, spec, "第 {i} 层的频率参数没被更新");
        }

        // 位置一路排到新窗口末尾：KV cache 必须全部留下（容量就是新窗口）
        let seq: Vec<usize> = (0..new_ctx).map(|i| i % 32).collect();
        let mut cache = model.new_kv_cache();
        let logits = model.forward(&seq, 1, seq.len(), Some(&mut cache), false);
        assert_eq!(cache[0].seq_len(), new_ctx, "扩窗后缓存不该再按训练窗口截断");
        assert_eq!(logits.shape(), vec![new_ctx, 32]);
        assert!(logits.data().iter().all(|v| v.is_finite()), "超出训练窗口的位置应算出有限值");
    }

    /// 滑动窗口 + KV cache 一致性（第 25 课）：
    /// 序列长度超过窗口后，缓存的每层都只保留最近 `block_size` 行；增量推理得到的
    /// 最后一行 logits，应与"只取最近 `block_size` 个 token 做一次全量前向"的最后一行吻合。
    ///
    /// **必须用单层模型**：注意力只在第 0 层，而第 0 层的 K/V 只由 token 嵌入决定、与上下文无关，
    /// 于是两条路径看到的是同一组 K/V，只差一个整体的位置平移（RoPE 打分只依赖相对距离）→ 严格等价。
    ///
    /// 层数 ≥ 2 时两者**本来就不是同一个函数**，差异不是 bug：增量路径里位置 p 的第 l 层 K/V 是
    /// p 当时算出来的（当时能看到它自己的窗口），而"截断重算"会把窗口内每个位置在第 0 层可见的
    /// 上下文一并砍掉，深层 K/V 随之不同。任何带 KV cache 的 LM 都是如此，缓存路径才是标准推理语义。
    #[test]
    fn test_kv_cache_sliding_window_matches_full_window_forward() {
        let mut rng = Rng::new(42);
        let mut cfg = GPTConfig::tiny(32);
        cfg.block_size = 8; // 小窗口：3 倍长度就要滑动多次
        cfg.n_layer = 1; // 见上方说明：多层时两条路径不等价
        let model = GPT::new(cfg, &mut rng);
        let w = model.cfg.block_size;
        let v = model.cfg.vocab_size;
        let seq: Vec<usize> = (0..w * 3).map(|i| (i * 7 + 1) % v).collect();

        // 增量路径：先整段喂入填满窗口，之后每次只喂 1 个 token
        let mut cache = model.new_kv_cache();
        let _ = model.forward(&seq[..w], 1, w, Some(&mut cache), false);
        let mut last_one = Vec::new();
        for &id in &seq[w..] {
            let out = model.forward(&[id], 1, 1, Some(&mut cache), false);
            last_one = out.data()[out.numel() - v..].to_vec();
        }
        assert_eq!(cache[0].seq_len(), w, "滑动窗口应把缓存控制在 window 行");
        assert_eq!(
            cache[0].positions_seen(),
            seq.len(),
            "绝对位置计数应累计全部喂入的 token（RoPE 基准靠它）"
        );

        // 全量路径：只取最近 w 个 token 重算
        let tail = &seq[seq.len() - w..];
        let full = model.forward(tail, 1, w, None, false);
        let last_full = full.data()[full.numel() - v..].to_vec();

        let max_diff = last_one
            .iter()
            .zip(&last_full)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-3,
            "滑动窗口增量推理与全量窗口前向的 logits 应吻合，实测最大偏差 {max_diff}"
        );
    }

    /// LoRA 接入训练循环的验收测试：冻结范围、可训练集合、以及"真的训起来了"。
    ///
    /// 一次真实训练步走完全链路（前向 → backward → AdamW.step），断言：
    /// 1. 可训练集合恰好是各层 Q/K/V 的 `lora_a` / `lora_b`，一个主干参数都不许混进来
    /// 2. 主干参数在这一步之后**逐位**不变（权重衰减也拉不动它）
    /// 3. 适配层确实被更新了（否则就是"挂着 LoRA 却没在学"）
    #[test]
    fn test_apply_lora_freezes_backbone_and_trains_adapters_only() {
        let mut rng = Rng::new(7);
        let mut model = GPT::new(GPTConfig::tiny(32), &mut rng);
        let total: usize = model.parameters().iter().map(|p| p.numel()).sum();
        let lora = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        model.apply_lora(&lora, &mut rng);

        // 1. 可训练集合 = 每层 Q/K/V 各一对低秩矩阵（本配置 n_kv_head = 0，K/V 输出维同 Q）
        let names: Vec<(String, Tensor)> = model
            .named_parameters()
            .into_iter()
            .filter(|(_, t)| t.requires_grad())
            .collect();
        assert_eq!(
            names.len(),
            3 * 2 * model.cfg.n_layer,
            "适配参数张量数不对：{}（应为 3 个投影 × 2 个矩阵 × {} 层）",
            names.len(),
            model.cfg.n_layer
        );
        let bad: Vec<&String> = names
            .iter()
            .map(|(n, _)| n)
            .filter(|n| !(n.ends_with(".lora_a") || n.ends_with(".lora_b")))
            .collect();
        assert!(
            bad.is_empty(),
            "可训练集合里混进了主干参数：{bad:?}"
        );
        let d = model.cfg.n_embd;
        let trainable_count: usize = model.trainable_parameters().iter().map(|p| p.numel()).sum();
        assert_eq!(trainable_count, 3 * 2 * lora.rank * d * model.cfg.n_layer);
        assert!(
            trainable_count * 10 < total,
            "适配层应远小于主干：{trainable_count} vs {total}"
        );
        for (name, t) in model.named_parameters() {
            if !(name.ends_with(".lora_a") || name.ends_with(".lora_b")) {
                assert!(!t.requires_grad(), "{name} 未被冻结");
            }
        }

        // 快照：冻结参数的句柄（共享底层数据）+ 训练前的数值
        let frozen_params: Vec<(String, Tensor)> = model
            .named_parameters()
            .into_iter()
            .filter(|(_, t)| !t.requires_grad())
            .collect();
        let frozen_before: Vec<Vec<f32>> = frozen_params.iter().map(|(_, t)| t.data()).collect();
        let trainable_before: Vec<Vec<f32>> =
            model.trainable_parameters().iter().map(|p| p.data()).collect();

        // 2./3. 一次真实训练步（权重衰减给到 0.1：冻结参数若没被优化器跳过，会立刻被拉走）
        let idx = vec![1usize, 2, 3, 4, 5, 6, 7, 8];
        let targets = vec![2usize, 3, 4, 5, 6, 7, 8, 9];
        let mut opt = AdamW::new(1e-2, model.parameters(), 0.1);
        zero_grad_all(&model);
        let logits = model.forward(&idx, 1, idx.len(), None, false);
        let loss = cross_entropy_loss(&logits, &targets);
        assert!(loss.item().is_finite(), "loss 不是有限值：{}", loss.item());
        loss.backward();
        opt.step();

        // 主干：逐位未变（句柄共享底层数据，这里读到的就是 step 之后的数值）
        for ((name, t), before) in frozen_params.iter().zip(&frozen_before) {
            let after = t.data();
            assert_eq!(before.len(), after.len(), "{name} 长度变了");
            for (i, (x, y)) in before.iter().zip(&after).enumerate() {
                assert_eq!(
                    x.to_bits(),
                    y.to_bits(),
                    "冻结参数 {name}[{i}] 被改动了：{x} -> {y}"
                );
            }
        }
        // 适配层：确实动了（B 初始为 0，第一步只有 B 拿到梯度，A 要等 B 非零后才动）
        let moved = model
            .trainable_parameters()
            .iter()
            .zip(&trainable_before)
            .any(|(p, before)| p.data().iter().zip(before).any(|(a, b)| a != b));
        assert!(moved, "适配层一步之后没有任何变化，LoRA 没接上");
    }

    /// `targets` 决定适配层挂在哪：只开 v 与 mlp 时，Q/K/输出投影必须一个适配参数都没有。
    /// 这条同时守住 `has_lora`——它必须看全部四个投影加 MLP，只看 `c_q` 会误判成"非 LoRA"，
    /// 于是 GPU 常驻快路不让路、增量被静默丢掉。
    #[test]
    fn test_lora_targets_control_where_adapters_land() {
        let mut rng = Rng::new(3);
        let mut model = GPT::new(GPTConfig::tiny(32), &mut rng);
        let lora = LoRAConfig {
            rank: 2,
            alpha: 2.0,
            targets: crate::config::LoRATargets {
                q: false,
                k: false,
                v: true,
                o: false,
                mlp: true,
            },
        };
        model.apply_lora(&lora, &mut rng);

        assert!(model.has_lora(), "只挂 v/mlp 同样是 LoRA 形态");

        let lora_names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .filter(|(n, _)| n.contains(".lora_"))
            .map(|(n, _)| n)
            .collect();
        let misplaced: Vec<&String> = lora_names
            .iter()
            .filter(|n| !(n.contains(".c_v.") || n.contains(".mlp")))
            .collect();
        assert!(misplaced.is_empty(), "适配层挂到了没打开的位置：{misplaced:?}");

        // 每层：c_v 一对 + MLP 每个投影一对（GELU 是 2 个，SwiGLU 是 3 个）
        let mlp_linears = if model.cfg.use_swiglu { 3 } else { 2 };
        assert_eq!(
            lora_names.len(),
            (1 + mlp_linears) * 2 * model.cfg.n_layer,
            "适配参数张量数不对"
        );
        // 没挂适配器的投影依旧被冻结（冻结动作与挂载位置无关）
        assert!(!model.blocks[0].attn.c_q.weight.requires_grad());
        assert!(!model.blocks[0].attn.c_proj.weight.requires_grad());
    }

    /// 合并前后前向必须一致：`W + ΔW`（一条支路）与 `W`、`ΔW`（两条支路相加）
    /// 在数学上是同一个结果，差别只有浮点累加顺序。
    #[test]
    fn test_merge_lora_matches_two_branch_forward() {
        let mut rng = Rng::new(9);
        let mut model = GPT::new(GPTConfig::tiny(32), &mut rng);
        model.apply_lora(
            &LoRAConfig {
                rank: 4,
                alpha: 8.0,
                ..Default::default()
            },
            &mut rng,
        );
        // B 初始全零 ⇒ ΔW = 0，那样合并前后本来就相同，等于没测。先灌成非平凡值。
        for (i, p) in model.lora_parameters().iter().enumerate() {
            let n = p.numel();
            p.set_data(
                (0..n)
                    .map(|j| ((i * 7 + j) as f32 * 0.13).sin() * 0.5)
                    .collect(),
            );
        }

        let idx: Vec<usize> = (0..8).collect();
        let before = model.forward(&idx, 1, idx.len(), None, false).data();
        let w_before = model.blocks[0].attn.c_q.weight.data();

        model.merge_lora();

        assert!(!model.has_lora(), "合并后不该仍是 LoRA 形态");
        assert!(model.lora.is_none(), "合并后 model.lora 应清空");
        assert!(
            model.lora_parameters().is_empty(),
            "合并后适配层参数应从参数表里消失"
        );
        // 主干确实被改写过，否则说明合并根本没落到权重上
        let w_after = model.blocks[0].attn.c_q.weight.data();
        assert!(
            w_before.iter().zip(&w_after).any(|(a, b)| a != b),
            "主干权重没变，合并没生效"
        );

        let after = model.forward(&idx, 1, idx.len(), None, false).data();
        let diff = before
            .iter()
            .zip(&after)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-4, "合并前后前向偏差过大：{diff}");
    }

    /// 链式续训：加载 LoRA 存档后 `resume_lora` 必须**保留**存档里的 A/B 数值并解冻它们；
    /// 与此相对，`apply_lora` 会重挂一套全新的（B 全零）——这就是"续训"与"重挂"的差别。
    #[test]
    fn test_resume_lora_keeps_adapter_values_and_unfreezes() {
        let mut rng = Rng::new(21);
        let lora = LoRAConfig {
            rank: 4,
            alpha: 8.0,
            ..Default::default()
        };
        let mut model = GPT::new(GPTConfig::tiny(32), &mut rng);
        model.apply_lora(&lora, &mut rng);
        // 灌成非零：B 初始为零，全零的存档"保没保住"看不出来
        for (i, p) in model.lora_parameters().iter().enumerate() {
            let n = p.numel();
            p.set_data(
                (0..n)
                    .map(|j| ((i * 5 + j) as f32 * 0.11).cos() * 0.3)
                    .collect(),
            );
        }
        let saved: Vec<Vec<f32>> = model.lora_parameters().iter().map(|p| p.data()).collect();

        let opt = AdamW::new(1e-3, model.parameters(), 0.0);
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("llm_lora_resume_test_{}.ckpt", std::process::id()));
        let path = tmp.to_string_lossy().into_owned();
        checkpoint::save(&path, &model, &opt, 10, 1.0);

        // 加载端：按存档头重放注入，再灌参数（与 `load_model_and_tokenizer` 同序）
        let ckpt = checkpoint::load_header(&path);
        let mut rng2 = Rng::new(999);
        let mut restored = GPT::new(ckpt.model.clone(), &mut rng2);
        restored.apply_lora(
            ckpt.lora.as_ref().expect("存档应带 LoRA 形态"),
            &mut rng2,
        );
        checkpoint::load_params(&path, &restored);
        let _ = std::fs::remove_file(&path);

        for (i, (a, b)) in saved
            .iter()
            .zip(&restored.lora_parameters().iter().map(|p| p.data()).collect::<Vec<_>>())
            .enumerate()
        {
            assert_eq!(a, b, "第 {i} 个适配参数没被还原");
        }

        restored.resume_lora();

        let after: Vec<Vec<f32>> = restored.lora_parameters().iter().map(|p| p.data()).collect();
        for (i, (a, b)) in saved.iter().zip(&after).enumerate() {
            assert_eq!(a, b, "resume_lora 重新初始化了第 {i} 个适配参数");
        }
        assert!(
            restored.lora_parameters().iter().all(|p| p.requires_grad()),
            "续训的适配层必须是可训练的"
        );
        assert_eq!(
            restored.trainable_parameters().len(),
            restored.lora_parameters().len(),
            "可训练集合应恰好是适配层"
        );
        for (name, t) in restored.named_parameters() {
            if !name.contains(".lora_") {
                assert!(!t.requires_grad(), "{name} 应保持冻结");
            }
        }
    }

    /// GPU 常驻显存快路与逐算子路径的一致性：同一份参数、同一批数据，
    /// loss 与**全部**参数梯度必须逐项吻合。覆盖 LayerNorm / RMSNorm / GQA 三种形态。
    ///
    /// 关闭常驻路径的办法是打开形状录制（[`crate::gpu::probe_capture`]）：录制模式下
    /// `recorder()` 返回 `None`，三条常驻入口（注意力子层、MLP 子层、整叠 Block）全部让路，
    /// 走的正是同一份计算图的逐算子实现。
    #[cfg(feature = "gpu")]
    #[test]
    fn test_gpu_resident_path_matches_loop_path() {
        use std::sync::atomic::Ordering;
        if !crate::gpu::is_available() {
            return;
        }
        // 小配置的矩阵乘达不到真实训练的规模门槛，这里把门槛降到 0 强制走常驻路径
        crate::gpu::MATMUL_MIN_FLOPS.store(0, Ordering::Relaxed);

        let (b, t) = (1usize, 24usize);
        let idx: Vec<usize> = (0..b * t).map(|i| (i * 7 + 3) % 32).collect();
        let targets: Vec<usize> = (0..b * t).map(|i| (i * 5 + 1) % 32).collect();

        for (name, use_rmsnorm, n_kv_head) in [
            ("LayerNorm + MHA", false, 0usize),
            ("RMSNorm + MHA", true, 0),
            ("LayerNorm + GQA", false, 1),
            ("RMSNorm + GQA", true, 2),
        ] {
            let cfg = GPTConfig {
                vocab_size: 32,
                n_embd: 32,
                n_head: 4,
                n_layer: 2,
                block_size: 64,
                n_kv_head,
                use_rmsnorm,
                use_swiglu: false,
                dropout: 0.0,
                ..Default::default()
            };
            // 同一种子、同一批数据跑两遍：`resident = true` 走常驻显存，`false` 走逐算子
            let run = |resident: bool| {
                let mut rng = Rng::new(11);
                let model = GPT::new(cfg.clone(), &mut rng);
                crate::gpu::probe_capture(!resident);
                let logits = model.forward(&idx, b, t, None, true);
                let loss = cross_entropy_loss(&logits, &targets);
                zero_grad_all(&model);
                loss.backward();
                crate::gpu::probe_capture(false);
                let grads: Vec<Vec<f32>> =
                    model.parameters().iter().map(|p| p.grad.borrow().clone()).collect();
                (loss.item(), grads)
            };
            let (loss_res, g_res) = run(true);
            let (loss_loop, g_loop) = run(false);
            assert!(
                (loss_res - loss_loop).abs() < 1e-4 * loss_loop.abs().max(1.0),
                "{name}：常驻路径 loss {loss_res} 与逐算子 {loss_loop} 不一致"
            );
            assert_eq!(g_res.len(), g_loop.len(), "{name}：参数梯度数量不一致");
            for (i, (a, c)) in g_res.iter().zip(&g_loop).enumerate() {
                assert_eq!(a.len(), c.len(), "{name}：第 {i} 个参数梯度长度不一致");
                let max_err = a.iter().zip(c).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
                let mag = c.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                assert!(
                    max_err / mag.max(1e-6) < 2e-3,
                    "{name}：第 {i} 个参数梯度不一致（最大误差 {max_err}，参考量级 {mag}）"
                );
            }
        }
    }

    /// 模型级量化（第 33 课）的端到端闭环，一次把五件事串起来验证：
    ///
    /// 1. **校准真的采到了统计**——`n_tokens` 与 H 的对角（`E[x²] > 0`）都要立得住；
    ///    采不到就会静默退回 RTN，而 RTN 与 GPTQ 的报告长得一模一样，不查这一项
    ///    根本发现不了"补偿压根没生效"。
    /// 2. **两种 Hessian 模式都由阈值真实切换**——默认（阈值 1024）在 tiny 配置下是
    ///    全矩阵，把阈值压到 0 就必须整片退化成对角；这是"大层省内存"这条路的开关，
    ///    静默失效会让显存预估完全失真。
    /// 3. **量化真的省了字节**——口径是"投影 + 词嵌入表"的整模型真实值（词表不量化
    ///    会把压缩率拉低，所以这里仍然要求 2 倍以上）。
    /// 4. **量化后的 loss 不崩**——int8 + 逐通道 scale 的误差远小于这个容差。
    /// 5. **反量化前后前向逐位一致**——`quantize_weights` 之后的前向走"反量化 + f32 乘",
    ///    `dequantize_weights` 之后走"f32 权重直接乘"，两条路径必须给出同一个数，
    ///    否则说明"量化表示"与"烘焙回 f32 的权重"是两份不同的东西。
    #[test]
    fn quantize_weights_reduces_bytes_and_keeps_loss_close() {
        let text = "abcdefghij".repeat(16);
        let tok = Tokenizer::from_name("char", &text, 0);
        let mut rng = Rng::new(2024);
        let mut model = GPT::new(GPTConfig::tiny(64), &mut rng);
        let ids = tok.encode(&text);
        let seq = ids.len().min(model.cfg.block_size);
        // 量化层是推理专用的（见 [`Linear::forward`] 的断言），所以这里整段前向
        // 都跑在 `no_grad` 下；否则量化后的那次 `loss_of` 会直接 panic 在断言上。
        let loss_of = |m: &GPT| {
            crate::tensor::no_grad(|| {
                let logits = m.forward(&ids[..seq], 1, seq, None, false);
                cross_entropy_loss(&logits, &ids[1..=seq]).item()
            })
        };
        let before = loss_of(&model);
        assert!(before.is_finite(), "量化前的 loss 就不是有限值：{before}");

        // 1) 校准：每个投影层都要采到统计
        let stats = model.calibrate(&tok, &[text.clone()], CalibOpts::default());
        assert!(!stats.per_layer.is_empty(), "校准应覆盖到各投影层");
        for (name, c) in &stats.per_layer {
            assert_eq!(c.n_tokens, ids.len(), "{name} 采到的 token 数不对");
            let h = c.hessian.as_ref().expect("每层都应有 Hessian");
            let a = c.act_abs_mean.as_ref().expect("每层都应有激活均值");
            assert!(a.iter().all(|v| v.is_finite() && *v >= 0.0), "{name} 的均值不合法");
            assert!(a.iter().any(|v| *v > 0.0), "{name} 的激活均值恒为零 = 没采到");
            assert_eq!(
                h.dim(),
                a.len(),
                "{name} 的 Hessian 阶数必须等于输入维度（落错轴 = GPTQ 的补偿方向全反）"
            );
            let d = h.diagonal();
            assert!(d.iter().all(|v| v.is_finite()), "{name} 的 Hessian 有非有限值");
            assert!(d.iter().any(|v| *v > 0.0), "{name} 的 Hessian 恒为零 = 没采到");
        }

        // 2) 阈值 1024 ≫ tiny 的隐层宽度 ⇒ 必须走全矩阵（GPTQ 的补偿信息才有来源）
        assert!(
            stats.per_layer.iter().all(|(_, c)| c.hessian.as_ref().unwrap().is_full()),
            "默认阈值下 tiny 模型应存全矩阵 Hessian"
        );
        // 把阈值压到 0：整片退化成对角，且对角就是全矩阵的对角（同一套统计，只是留不留非对角）
        let diag_stats = model.calibrate(
            &tok,
            &[text.clone()],
            CalibOpts {
                full_hessian_max_dim: 0,
                ..CalibOpts::default()
            },
        );
        assert!(
            diag_stats.per_layer.iter().all(|(_, c)| !c.hessian.as_ref().unwrap().is_full()),
            "阈值 0 之下不该有任何层还存着全矩阵"
        );
        for ((name, full), (_, diag)) in stats.per_layer.iter().zip(diag_stats.per_layer.iter()) {
            let (f, d) = (full.hessian.as_ref().unwrap(), diag.hessian.as_ref().unwrap());
            assert_eq!(f.dim(), d.dim(), "{name} 两种模式的对角长度应一致");
            // 对角模式省掉的正是非对角项：内存口径必须严格更小（否则这个旋钮等于没有）
            assert!(
                d.byte_len() < f.byte_len(),
                "{name} 的对角统计并不比全矩阵省内存"
            );
        }

        // 3) 量化（GPTQ：校准统计真的被用上）+ 4) loss 对比
        let reports = model.quantize_weights(
            QBits::Int8,
            QuantMethod::Gptq,
            Some(&stats),
            QuantOpts::default(),
        );
        assert_eq!(
            reports.len(),
            stats.per_layer.len(),
            "每一层都该被量化（本配置没有 LoRA / MoE）"
        );
        assert!(model.has_quant() && model.is_quantized());
        let summary = model.quant_summary(reports);
        assert_eq!(summary.skipped, 0, "本配置没有该跳过的层");
        assert!(summary.quant_bytes < summary.f32_bytes);
        assert_eq!(summary.f32_bytes, model.f32_bytes());
        assert_eq!(summary.quant_bytes, model.quant_bytes());
        assert!(
            summary.ratio() > 2.0,
            "int8 之下整模型压缩率应在 2 倍以上（词嵌入表不量化会拉低它）：{}",
            summary.describe()
        );
        assert!(
            summary.worst_layer().is_some_and(|w| w.rel_err.is_finite()),
            "最差层的相对误差必须是有限值：{}",
            summary.describe()
        );
        let after = loss_of(&model);
        println!("[模型级量化] {}｜loss {before:.4} → {after:.4}", summary.describe());
        assert!(
            (after - before).abs() < 0.5,
            "int8 量化后验证 loss 崩了：{before} → {after}（{}）",
            summary.describe()
        );

        // 记录必须与实测口径一致（存档头写的就是这一份）
        let meta = model.quant_meta().expect("量化后应留下记录");
        assert_eq!(meta.bits, QBits::Int8);
        assert_eq!(meta.method, QuantMethod::Gptq);
        assert_eq!(
            (meta.orig_bytes, meta.quant_bytes),
            (summary.f32_bytes, summary.quant_bytes)
        );

        // 5) 反量化只把"量化后的数值"写回 weight：前后前向必须给出同一个数
        model.dequantize_weights();
        assert!(!model.has_quant(), "烘焙后不应再留着量化表示");
        assert!(model.quant_meta().is_some(), "烘焙不应丢掉量化记录");
        let baked = loss_of(&model);
        assert!(
            (baked - after).abs() < 1e-5,
            "量化前向与反量化后的 f32 前向不一致：{after} vs {baked}"
        );
    }

    /// 对角 Hessian（`full_hessian_max_dim = 0`）之下 GPTQ 必须表现得和 RTN 一样：
    /// 这不是"实现退化"，而是**数学结论**——`H` 是对角阵时 `H⁻¹` 也是对角阵，
    /// 加权误差 `tr((W-Ŵ)ᵀH(W-Ŵ))` 变成逐列可分的二次型，跨通道补偿系数恒为 0。
    ///
    /// 为什么要专门测它：全矩阵与对角两条路走的是同一段 `gptq_quantize`，一旦哪天
    /// "对角"被当成"全矩阵的一半"来实现（例如把非对角位置读成 0 却仍走 Cholesky
    /// 求逆再乘），两者就会悄悄分叉，而单看报告里的误差是发现不了的。
    #[test]
    fn diagonal_calibration_makes_gptq_degenerate_to_rtn() {
        let text = "abcdefghij".repeat(16);
        let tok = Tokenizer::from_name("char", &text, 0);
        let mut rng = Rng::new(7);
        let mut model = GPT::new(GPTConfig::tiny(64), &mut rng);
        let diag = CalibOpts {
            full_hessian_max_dim: 0,
            ..CalibOpts::default()
        };
        let stats = model.calibrate(&tok, &[text.clone()], diag);
        // 每层的 stats 都必须是"对角"形态
        assert!(stats.per_layer.iter().all(|(_, c)| !c.hessian.as_ref().unwrap().is_full()));

        let reports = model.quantize_weights(
            QBits::Int8,
            QuantMethod::Gptq,
            Some(&stats),
            QuantOpts::default(),
        );
        // 所有层都跑通、误差有限（退化 ≠ 失败）
        assert!(!reports.is_empty());
        assert!(reports.iter().all(|r| r.rel_err.is_finite()));
        // 逐位等于 RTN：把同一份权重用 RTN 再量化一遍，码值必须一模一样
        let mut rtn_model = GPT::new(GPTConfig::tiny(64), &mut Rng::new(7));
        let rtn = rtn_model.quantize_weights(QBits::Int8, QuantMethod::Rtn, None, QuantOpts::default());
        let gptq_codes: Vec<_> = model
            .named_linears()
            .iter()
            .map(|(_, l)| l.quant.as_ref().unwrap().q.bytes().to_vec())
            .collect();
        let rtn_codes: Vec<_> = rtn_model
            .named_linears()
            .iter()
            .map(|(_, l)| l.quant.as_ref().unwrap().q.bytes().to_vec())
            .collect();
        assert_eq!(gptq_codes.len(), rtn_codes.len());
        assert_eq!(gptq_codes, rtn_codes, "对角 Hessian 下 GPTQ 应与 RTN 逐位一致");
        assert_eq!(rtn.len(), gptq_codes.len());
    }
}
