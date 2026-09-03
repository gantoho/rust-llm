//! GPT 模型（第 9-12、18-19 课）
//!
//! 结构（从下到上）：
//! 1. token embedding：每个 token id -> 向量
//! 2. N 层 Transformer Block：注意力（找相关性）+ 前馈网络（加工信息）
//! 3. 最终 LayerNorm + 输出头（预测下一个 token）
//!
//! 位置信息由 RoPE（第 19 课）提供：在注意力内部对 Q/K 做旋转，不再向输入加位置向量。
//!
//! 注意点：
//! - 因果掩码（causal mask）：模型只能看到过去，不能看到未来
//! - KV Cache（第 18 课）：推理时缓存历史的 K/V，避免重复计算
//! - RoPE（第 19 课）：只旋转 Q/K、不旋转 V；旋转发生在 KV cache append 之前，
//!   缓存里存的是"已旋转的 K"，历史 K 直接复用

use crate::attention::{KVCache, MultiHeadAttention};
use crate::layers::{Embedding, MLPEnum, NormLayer};
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// LayerNorm 数值稳定常数（防止方差为 0 时除零）
const LN_EPS: f32 = 1e-5;

/// 模型配置（`config.json` 里可调，缺省字段用 [`GPTConfig::default`]）
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
        // [诊断] block 内部分段计时（仅前 2 次调用）
        static BLK_DIAG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let blk_diag = BLK_DIAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2;
        let blk_t0 = std::time::Instant::now();
        // 注意力子层 + 残差连接
        let ln1_out = self.ln1.forward(x);
        let t_ln1 = blk_t0.elapsed();
        let h = self
            .attn
            .forward(&ln1_out, mask, kv_cache, base);
        let t_attn = blk_t0.elapsed();
        let h = if self.dropout > 0.0 { h.dropout(self.dropout, training) } else { h };
        let x = x.add(&h);
        let t_add1 = blk_t0.elapsed();
        // 前馈子层 + 残差连接
        let h = self.ln2.forward(&x);
        let t_ln2 = blk_t0.elapsed();
        let h = self.mlp.forward(&h);
        let t_mlp1 = blk_t0.elapsed();
        let h = if self.dropout > 0.0 { h.dropout(self.dropout, training) } else { h };
        let out = x.add(&h);
        if blk_diag {
            println!(
                "[diag-blk] ln1 {:.1} | attn {:.1} | res1 {:.1} | ln2 {:.1} | mlp {:.1} | 总 {:.1} ms",
                t_ln1.as_secs_f64() * 1000.0,
                (t_attn - t_ln1).as_secs_f64() * 1000.0,
                (t_add1 - t_attn).as_secs_f64() * 1000.0,
                (t_ln2 - t_add1).as_secs_f64() * 1000.0,
                (t_mlp1 - t_ln2).as_secs_f64() * 1000.0,
                blk_t0.elapsed().as_secs_f64() * 1000.0,
            );
        }
        out
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
        mut kv_cache: Option<&mut Vec<KVCache>>,
        training: bool,
    ) -> Tensor {
        let d = self.cfg.n_embd;
        assert_eq!(idx.len(), b * t, "输入 id 数量必须等于 b*t");

        // [诊断] forward 分段计时（仅前 2 次调用）
        static FW_DIAG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let fw_diag = FW_DIAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2;
        let fw_t0 = std::time::Instant::now();

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
        let t_emb_mask = fw_t0.elapsed();

        // 4. 逐层过 Transformer Block
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = kv_cache.as_mut().map(|c| &mut c[i]);
            let t_blk = std::time::Instant::now();
            x = block.forward(&x, &mask, cache, base, training);
            if fw_diag {
                println!(
                    "[diag-fw] block {i}: {:.1}ms",
                    t_blk.elapsed().as_secs_f64() * 1000.0
                );
            }
        }
        let t_after_blocks = fw_t0.elapsed();

        // 5. 最终归一化 + 输出头（权重绑定：lm_head 复用 tok_emb.table 的转置）
        let x = self.ln_f.forward(&x);
        let x = x.reshape(vec![b * t, d]);
        let out = x.matmul(&self.tok_emb.table.transpose());
        if fw_diag {
            println!(
                "[diag-fw] emb+mask {:.1}ms | blocks 共 {:.1}ms | ln_f+lm_head {:.1}ms | forward 总 {:.1}ms",
                t_emb_mask.as_secs_f64() * 1000.0,
                (t_after_blocks - t_emb_mask).as_secs_f64() * 1000.0,
                (fw_t0.elapsed() - t_after_blocks).as_secs_f64() * 1000.0,
                fw_t0.elapsed().as_secs_f64() * 1000.0
            );
        }
        out
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
