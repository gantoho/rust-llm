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
use crate::layers::{Embedding, LayerNorm, Linear, gelu};
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// 模型配置（`config.json` 里可调，缺省字段用 [`GPTConfig::default`]）
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct GPTConfig {
    /// 词表大小；0 表示"由分词器决定"（训练时自动填入）
    pub vocab_size: usize,
    pub n_embd: usize,     // 隐藏维度
    pub n_head: usize,     // 注意力头数
    pub n_layer: usize,    // Transformer 层数
    pub block_size: usize, // 最大上下文长度
}

impl Default for GPTConfig {
    fn default() -> Self {
        GPTConfig {
            vocab_size: 0,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            block_size: 32,
        }
    }
}

impl GPTConfig {
    /// 一个小配置，适合学习演示
    pub fn tiny(vocab_size: usize) -> Self {
        GPTConfig {
            vocab_size,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            block_size: 32,
        }
    }
}

/// Transformer Block（第 11 课）
///
/// 结构（GPT-2 风格，pre-norm）：
///   x -> LayerNorm -> Attention -> 残差 +
///   x -> LayerNorm -> MLP(GELU)  -> 残差 +
struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    mlp_linear1: Linear, // [D, 4D]
    mlp_linear2: Linear, // [4D, D]
}

impl TransformerBlock {
    fn new(cfg: &GPTConfig, rng: &mut Rng) -> Self {
        TransformerBlock {
            ln1: LayerNorm::new(cfg.n_embd, 1e-5),
            attn: MultiHeadAttention::new(cfg.n_embd, cfg.n_head, rng),
            ln2: LayerNorm::new(cfg.n_embd, 1e-5),
            mlp_linear1: Linear::new(cfg.n_embd, 4 * cfg.n_embd, rng),
            mlp_linear2: Linear::new(4 * cfg.n_embd, cfg.n_embd, rng),
        }
    }

    fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        kv_cache: Option<&mut KVCache>,
        base: usize,
    ) -> Tensor {
        // 注意力子层 + 残差连接
        let h = self
            .attn
            .forward(&self.ln1.forward(x), mask, kv_cache, base);
        let x = x.add(&h);
        // 前馈子层 + 残差连接
        let h = self.ln2.forward(&x);
        let h = gelu(&self.mlp_linear1.forward(&h));
        let h = self.mlp_linear2.forward(&h);
        x.add(&h)
    }
}

impl Module for TransformerBlock {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.ln1.parameters();
        ps.extend(self.attn.parameters());
        ps.extend(self.ln2.parameters());
        ps.extend(self.mlp_linear1.parameters());
        ps.extend(self.mlp_linear2.parameters());
        ps
    }
}

/// 完整的 GPT 模型
pub struct GPT {
    pub cfg: GPTConfig,
    tok_emb: Embedding,
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
}

impl GPT {
    pub fn new(cfg: GPTConfig, rng: &mut Rng) -> Self {
        let n_embd = cfg.n_embd;
        let vocab_size = cfg.vocab_size;
        let blocks = (0..cfg.n_layer)
            .map(|_| TransformerBlock::new(&cfg, rng))
            .collect();
        GPT {
            cfg,
            tok_emb: Embedding::new(vocab_size, n_embd, rng),
            blocks,
            ln_f: LayerNorm::new(n_embd, 1e-5),
        }
    }

    /// 前向传播
    ///
    /// - idx: [B*T] 展平的 token id
    /// - b / t：batch 与序列长度
    /// - kv_cache: Some(每层一个缓存) 时启用 KV cache（推理模式）
    ///
    /// 返回 logits：[B*T, vocab_size]（每个位置预测"下一个 token"的分数）
    pub fn forward(
        &self,
        idx: &[usize],
        b: usize,
        t: usize,
        mut kv_cache: Option<&mut Vec<KVCache>>,
    ) -> Tensor {
        let d = self.cfg.n_embd;
        assert_eq!(idx.len(), b * t, "输入 id 数量必须等于 b*t");

        // 1. token embedding
        let x = self.tok_emb.forward(idx).reshape(vec![b, t, d]);

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

        // 4. 逐层过 Transformer Block
        let mut x = x;
        for (i, block) in self.blocks.iter().enumerate() {
            let cache = kv_cache.as_mut().map(|c| &mut c[i]);
            x = block.forward(&x, &mask, cache, base);
        }

        // 5. 最终归一化 + 输出头（权重绑定：lm_head 复用 tok_emb.table 的转置）
        let x = self.ln_f.forward(&x);
        let x = x.reshape(vec![b * t, d]);
        x.matmul(&self.tok_emb.table.transpose())
    }

    /// 推理用的缓存集合：每层一个
    pub fn new_kv_cache(&self) -> Vec<KVCache> {
        (0..self.cfg.n_layer).map(|_| KVCache::new()).collect()
    }

    /// 带名字的参数列表（checkpoint 保存/恢复用）。
    /// 名字形如 `blocks.0.attn.c_q.weight`，顺序与 [`Module::parameters`] 完全一致。
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut ps = vec![("tok_emb.table".to_string(), self.tok_emb.table.clone())];
        for (i, block) in self.blocks.iter().enumerate() {
            let p = format!("blocks.{i}");
            ps.push((format!("{p}.ln1.gamma"), block.ln1.gamma.clone()));
            ps.push((format!("{p}.ln1.beta"), block.ln1.beta.clone()));
            ps.push((
                format!("{p}.attn.c_q.weight"),
                block.attn.c_q.weight.clone(),
            ));
            ps.push((format!("{p}.attn.c_q.bias"), block.attn.c_q.bias.clone()));
            ps.push((
                format!("{p}.attn.c_k.weight"),
                block.attn.c_k.weight.clone(),
            ));
            ps.push((format!("{p}.attn.c_k.bias"), block.attn.c_k.bias.clone()));
            ps.push((
                format!("{p}.attn.c_v.weight"),
                block.attn.c_v.weight.clone(),
            ));
            ps.push((format!("{p}.attn.c_v.bias"), block.attn.c_v.bias.clone()));
            ps.push((
                format!("{p}.attn.c_proj.weight"),
                block.attn.c_proj.weight.clone(),
            ));
            ps.push((
                format!("{p}.attn.c_proj.bias"),
                block.attn.c_proj.bias.clone(),
            ));
            ps.push((format!("{p}.ln2.gamma"), block.ln2.gamma.clone()));
            ps.push((format!("{p}.ln2.beta"), block.ln2.beta.clone()));
            ps.push((
                format!("{p}.mlp_linear1.weight"),
                block.mlp_linear1.weight.clone(),
            ));
            ps.push((
                format!("{p}.mlp_linear1.bias"),
                block.mlp_linear1.bias.clone(),
            ));
            ps.push((
                format!("{p}.mlp_linear2.weight"),
                block.mlp_linear2.weight.clone(),
            ));
            ps.push((
                format!("{p}.mlp_linear2.bias"),
                block.mlp_linear2.bias.clone(),
            ));
        }
        ps.push(("ln_f.gamma".to_string(), self.ln_f.gamma.clone()));
        ps.push(("ln_f.beta".to_string(), self.ln_f.beta.clone()));
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
        let _ = model.forward(&seq, 1, seq.len(), Some(&mut cache));
        let new_id = 3;
        let one = model.forward(&[new_id], 1, 1, Some(&mut cache));
        let last_one = one.data()[one.numel() - v..].to_vec();

        // 全量模式：一次前向 [seq..., new_id]（11 个 token），取最后一个位置（位置 10）
        let mut seq2 = seq;
        seq2.push(new_id);
        let full = model.forward(&seq2, 1, seq2.len(), None);
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
