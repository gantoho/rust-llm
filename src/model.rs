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

use crate::attention::{KVCache, MultiHeadAttention};
use crate::layers::{Embedding, MLPEnum, NormLayer};
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

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
        }
    }
}

impl GPTConfig {
    /// 一个小配置，适合学习演示（其余字段与 Default 一致）
    pub fn tiny(vocab_size: usize) -> Self {
        GPTConfig {
            vocab_size,
            ..Default::default()
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
struct TransformerBlock {
    ln1: NormLayer,
    attn: MultiHeadAttention,
    ln2: NormLayer,
    mlp: MLPEnum,
    dropout: f32,
}

impl TransformerBlock {
    fn new(cfg: &GPTConfig, rng: &mut Rng) -> Self {
        let mlp = if cfg.use_swiglu {
            MLPEnum::new_swiglu(cfg.n_embd, rng)
        } else {
            MLPEnum::new_gelu(cfg.n_embd, rng)
        };
        TransformerBlock {
            ln1: NormLayer::new(cfg.n_embd, LN_EPS, cfg.use_rmsnorm),
            attn: MultiHeadAttention::new(cfg.n_embd, cfg.n_head, cfg.n_kv_head, rng),
            ln2: NormLayer::new(cfg.n_embd, LN_EPS, cfg.use_rmsnorm),
            mlp,
            dropout: cfg.dropout,
        }
    }

    /// 带名字的参数（checkpoint 用）
    fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps = self.ln1.named_parameters(&format!("{prefix}.ln1"));
        ps.extend(self.attn.named_parameters(&format!("{prefix}.attn")));
        ps.extend(self.ln2.named_parameters(&format!("{prefix}.ln2")));
        ps.extend(self.mlp.named_parameters(&format!("{prefix}.mlp")));
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
        let h = self.mlp.forward(&h);
        let h = if self.dropout > 0.0 { h.dropout(self.dropout, training) } else { h };
        let out = x.add(&h);
        out
    }

    /// 注意力子层的 GPU 常驻显存前向（见 [`crate::gpu::attn_layer_forward`]）。
    ///
    /// 返回的张量前向数据已在显存里算好，反向闭包直接调 `AttnLayerResident::backward`，
    /// 把 11 项边界梯度（对 x、Q/K/V/输出四个投影的权重与偏置、LayerNorm 的 γ/β）
    /// 注回计算图 —— 于是这一段不必再建一张「LN + 四个投影 + RoPE + 注意力 + dropout + 残差」
    /// 的计算图，中间量（Q/K/V、旋转后的 Q/K、注意力输出）也不会流回 CPU。
    ///
    /// 不适用时返回 None，调用方回退逐算子路径：RMSNorm / GQA（n_kv_head != n_head）配置、
    /// 推理模式、形状或规模不合适、GPU 不可用。
    #[cfg(feature = "gpu")]
    fn attn_resident(&self, x: &Tensor, mask: &Tensor, training: bool) -> Option<Tensor> {
        if !crate::tensor::grad_enabled() {
            return None;
        }
        let shape = x.shape();
        if shape.len() != 3 {
            return None;
        }
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let (gamma, beta, eps) = self.ln1.ln_params()?;
        if self.attn.n_kv_head != self.attn.n_head {
            return None; // GQA 需要头复制，常驻路径未实现
        }
        let (cq, ck, cv, cp) = (
            &self.attn.c_q,
            &self.attn.c_k,
            &self.attn.c_v,
            &self.attn.c_proj,
        );
        let res = crate::gpu::attn_layer_forward(&crate::gpu::AttnLayerArgs {
            x: &x.data.borrow(),
            gamma: &gamma.data.borrow(),
            beta: &beta.data.borrow(),
            wq: &cq.weight.data.borrow(),
            bq: &cq.bias.data.borrow(),
            wk: &ck.weight.data.borrow(),
            bk: &ck.bias.data.borrow(),
            wv: &cv.weight.data.borrow(),
            bv: &cv.bias.data.borrow(),
            wproj: &cp.weight.data.borrow(),
            bproj: &cp.bias.data.borrow(),
            mask: &mask.data.borrow(),
            b,
            t,
            d,
            n_head: self.attn.n_head,
            eps,
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
        let (gamma_bwd, beta_bwd) = (Tensor::clone(gamma), Tensor::clone(beta));
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
                    wk_bwd.accumulate_grad(&g.dwk, 1.0);
                    bk_bwd.accumulate_grad(&g.dbk, 1.0);
                    wv_bwd.accumulate_grad(&g.dwv, 1.0);
                    bv_bwd.accumulate_grad(&g.dbv, 1.0);
                    wp_bwd.accumulate_grad(&g.dwproj, 1.0);
                    bp_bwd.accumulate_grad(&g.dbproj, 1.0);
                    gamma_bwd.accumulate_grad(&g.dgamma, 1.0);
                    beta_bwd.accumulate_grad(&g.dbeta, 1.0);
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
    /// 不适用时返回 None，调用方回退逐算子路径：SwiGLU / RMSNorm 配置、推理模式
    /// （`no_grad`）、形状或规模不合适、GPU 不可用。
    #[cfg(feature = "gpu")]
    fn mlp_resident(&self, x: &Tensor, training: bool) -> Option<Tensor> {
        if !crate::tensor::grad_enabled() {
            return None;
        }
        let shape = x.shape();
        if shape.len() != 3 {
            return None;
        }
        let (b, t, d) = (shape[0], shape[1], shape[2]);
        let (gamma, beta, eps) = self.ln2.ln_params()?;
        let (w1, b1, w2, b2) = self.mlp.gelu_weights()?;
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
            self.dropout,
            training,
        )?;
        let out_data = res.out.clone();
        let x_bwd = x.clone();
        let (gamma_bwd, beta_bwd) = (Tensor::clone(gamma), Tensor::clone(beta));
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
                    beta_bwd.accumulate_grad(&g.dbeta, 1.0);
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
        ps.extend(self.mlp.parameters());
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
        }
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
    #[cfg(feature = "gpu")]
    fn blocks_resident(
        &self,
        x: &Tensor,
        mask: &Tensor,
        b: usize,
        t: usize,
        training: bool,
    ) -> Option<Tensor> {
        use crate::gpu::{STACK_PARAMS_PER_LAYER, StackArgs, StackLayerArgs};
        if !crate::tensor::grad_enabled() || self.blocks.is_empty() {
            return None;
        }
        let d = self.cfg.n_embd;
        let (_, _, eps) = self.blocks[0].ln1.ln_params()?;
        // 每层的 16 个参数张量，顺序必须与 `StackLayerArgs` 的字段顺序一致
        // （`gpu::StackGrads::grads` 也按这个顺序回来）
        let mut params: Vec<Tensor> =
            Vec::with_capacity(STACK_PARAMS_PER_LAYER * self.blocks.len());
        for block in &self.blocks {
            let (gamma1, beta1, _) = block.ln1.ln_params()?;
            let (gamma2, beta2, _) = block.ln2.ln_params()?;
            let (w1, b1, w2, b2) = block.mlp.gelu_weights()?;
            let (cq, ck, cv, cp) = (
                &block.attn.c_q,
                &block.attn.c_k,
                &block.attn.c_v,
                &block.attn.c_proj,
            );
            params.extend([
                gamma1.clone(),
                beta1.clone(),
                cq.weight.clone(),
                cq.bias.clone(),
                ck.weight.clone(),
                ck.bias.clone(),
                cv.weight.clone(),
                cv.bias.clone(),
                cp.weight.clone(),
                cp.bias.clone(),
                gamma2.clone(),
                beta2.clone(),
                w1.clone(),
                b1.clone(),
                w2.clone(),
                b2.clone(),
            ]);
        }
        // 借用只在这一段里存活：`guards` 借用了 `params`，出块后 `params` 才能进反向闭包
        let res = {
            let guards: Vec<std::cell::Ref<'_, Vec<f32>>> =
                params.iter().map(|p| p.data.borrow()).collect();
            let layers: Vec<StackLayerArgs> = (0..self.blocks.len())
                .map(|i| {
                    let s = &guards[i * STACK_PARAMS_PER_LAYER..][..STACK_PARAMS_PER_LAYER];
                    StackLayerArgs {
                        gamma1: &s[0][..],
                        beta1: &s[1][..],
                        wq: &s[2][..],
                        bq: &s[3][..],
                        wk: &s[4][..],
                        bk: &s[5][..],
                        wv: &s[6][..],
                        bv: &s[7][..],
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
                n_head: self.cfg.n_head,
                eps,
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
                    for (p, pg) in params.iter().zip(g.grads.iter()) {
                        p.accumulate_grad(pg, 1.0);
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
        //    base = KV cache 模式下已缓存的位置数：新 token 的绝对位置 = base + 窗口内下标 j。
        let base = kv_cache
            .as_ref()
            .map(|c| c.first().map(|k| k.seq_len()).unwrap_or(0))
            .unwrap_or(0);

        // 3. 因果掩码：scores 形状 [B*H, T, T_total]，广播 mask [T, T_total]
        let t_total = t + base;
        let mut mask_data = vec![0.0f32; t * t_total];
        for i in 0..t {
            for j in 0..t_total {
                if j > i + base {
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
        //    不满足就只回退那一个。想复现对照实验或追求吞吐时设 `LLM_GPU_STACK=1` 打开。
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

    /// 推理用的缓存集合：每层一个
    pub fn new_kv_cache(&self) -> Vec<KVCache> {
        (0..self.cfg.n_layer).map(|_| KVCache::new()).collect()
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
}
