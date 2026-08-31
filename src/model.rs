//! GPT 模型（第 9-12、18 课）
//!
//! 结构（从下到上）：
//! 1. token embedding：每个 token id -> 向量
//! 2. 位置编码：告诉模型"每个 token 在序列里的位置"
//! 3. N 层 Transformer Block：注意力（找相关性）+ 前馈网络（加工信息）
//! 4. 最终 LayerNorm + 输出头（预测下一个 token）
//!
//! 注意点：
//! - 因果掩码（causal mask）：模型只能看到过去，不能看到未来
//! - KV Cache（第 18 课）：推理时缓存历史的 K/V，避免重复计算

use crate::layers::{Embedding, LayerNorm, Linear, gelu};
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;

/// 模型配置
pub struct GPTConfig {
    pub vocab_size: usize,
    pub n_embd: usize,     // 隐藏维度
    pub n_head: usize,     // 注意力头数
    pub n_layer: usize,    // Transformer 层数
    pub block_size: usize, // 最大上下文长度
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

/// KV 缓存（第 18 课）：
/// 生成第 N 个 token 时，前 N-1 个 token 的 K、V 不需要重算。
/// 把每个注意力层的 K、V 存起来，每次只算新 token 的 K、V 并拼接。
pub struct KVCache {
    k: Option<Tensor>, // [1, T, D]
    v: Option<Tensor>,
}

impl KVCache {
    pub fn new() -> Self {
        KVCache { k: None, v: None }
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
    }

    /// 当前已缓存的位置数
    pub fn seq_len(&self) -> usize {
        self.k.as_ref().map(|t| t.shape()[1]).unwrap_or(0)
    }

    /// 把新的 k/v 拼到缓存后面（纯数据拼接，推理时无梯度）
    fn append_data(prev: &Option<Tensor>, cur: &Tensor) -> Tensor {
        match prev {
            Some(p) => {
                let mut all = p.data();
                all.extend(cur.data());
                let d = cur.shape()[2];
                Tensor::from_vec(all, vec![1, p.shape()[1] + 1, d])
            }
            None => cur.clone(),
        }
    }

    pub fn append(&mut self, k: &Tensor, v: &Tensor) {
        self.k = Some(Self::append_data(&self.k, k));
        self.v = Some(Self::append_data(&self.v, v));
    }

    pub fn k(&self) -> Option<Tensor> {
        self.k.clone()
    }

    pub fn v(&self) -> Option<Tensor> {
        self.v.clone()
    }
}

/// 多头注意力（第 9-10 课）
struct MultiHeadAttention {
    c_q: Linear, // [D, D]
    c_k: Linear, // [D, D]
    c_v: Linear, // [D, D]
    c_proj: Linear,
    n_head: usize,
}

impl MultiHeadAttention {
    fn new(cfg: &GPTConfig, rng: &mut Rng) -> Self {
        MultiHeadAttention {
            c_q: Linear::new(cfg.n_embd, cfg.n_embd, rng),
            c_k: Linear::new(cfg.n_embd, cfg.n_embd, rng),
            c_v: Linear::new(cfg.n_embd, cfg.n_embd, rng),
            c_proj: Linear::new(cfg.n_embd, cfg.n_embd, rng),
            n_head: cfg.n_head,
        }
    }

    /// 前向
    /// - x: [B, T, D]
    /// - mask: [T, T_total] 因果掩码（-inf 的位置不能看）
    /// - kv_cache: Some(缓存) 时走推理模式（只算新 token）
    fn forward(&self, x: &Tensor, mask: &Tensor, kv_cache: Option<&mut KVCache>) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let head_dim = d / self.n_head;
        assert_eq!(head_dim * self.n_head, d, "n_embd 必须能被 n_head 整除");

        // 1. 投影得到 Q、K、V（Linear 输出是 2D [B*T, D]，恢复成 3D）
        let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
        let k = self.c_k.forward(x).reshape(vec![b, t, d]);
        let v = self.c_v.forward(x).reshape(vec![b, t, d]);

        // 2. KV cache：拼接历史的 K/V（只影响 K、V 的长度）
        let (k, v) = match kv_cache {
            Some(cache) => {
                cache.append(&k, &v);
                (cache.k().unwrap(), cache.v().unwrap())
            }
            None => (k, v),
        };
        let t_total = k.shape()[1];

        // 3. 拆头：[B, T, D] -> [B*H, T, head_dim]
        //    （先 reshape 出 H 维，再 permute 把 H 提到第 2 维）
        let q = q
            .reshape(vec![b, t, self.n_head, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b * self.n_head, t, head_dim]);
        let k = k
            .reshape(vec![b, t_total, self.n_head, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b * self.n_head, t_total, head_dim]);
        let v = v
            .reshape(vec![b, t_total, self.n_head, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b * self.n_head, t_total, head_dim]);

        // 4. 注意力分数：scores = Q·Kᵀ / √d_k
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kt = k.permute(&[0, 2, 1]); // [B*H, head_dim, T_total]
        let scores = q.matmul(&kt).mul_scalar(scale); // [B*H, T, T_total]

        // 5. 因果掩码：把"未来位置"变成 -inf，softmax 后概率为 0
        let scores = scores.add(mask);

        // 6. softmax 得到注意力权重，加权求和
        let attn = scores.softmax_last_dim(); // [B*H, T, T_total]
        let out = attn.matmul(&v); // [B*H, T, head_dim]

        // 7. 合并头回 [B, T, D]
        let out = out
            .reshape(vec![b, self.n_head, t, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b, t, d]);

        // 8. 输出投影
        self.c_proj.forward(&out)
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.c_q.parameters();
        ps.extend(self.c_k.parameters());
        ps.extend(self.c_v.parameters());
        ps.extend(self.c_proj.parameters());
        ps
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
            attn: MultiHeadAttention::new(cfg, rng),
            ln2: LayerNorm::new(cfg.n_embd, 1e-5),
            mlp_linear1: Linear::new(cfg.n_embd, 4 * cfg.n_embd, rng),
            mlp_linear2: Linear::new(4 * cfg.n_embd, cfg.n_embd, rng),
        }
    }

    fn forward(&self, x: &Tensor, mask: &Tensor, kv_cache: Option<&mut KVCache>) -> Tensor {
        // 注意力子层 + 残差连接
        let h = self.attn.forward(&self.ln1.forward(x), mask, kv_cache);
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
    pos_emb: Tensor, // 正弦位置编码 [block_size, D]（常数，不参与训练）
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear, // [D, vocab]
}

/// 正弦位置编码（第 11 课）
///
/// PE(pos, 2i)   = sin(pos / 10000^(2i/D))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/D))
///
/// 用不同频率的正弦波编码位置，让模型能区分不同位置、捕捉相对距离。
fn sinusoidal_positions(max_len: usize, d: usize) -> Vec<f32> {
    let mut data = vec![0.0f32; max_len * d];
    for pos in 0..max_len {
        for i in 0..d {
            let freq = 10000f32.powf((2 * (i / 2)) as f32 / d as f32);
            let angle = pos as f32 / freq;
            data[pos * d + i] = if i % 2 == 0 { angle.sin() } else { angle.cos() };
        }
    }
    data
}

impl GPT {
    pub fn new(cfg: GPTConfig, rng: &mut Rng) -> Self {
        let n_embd = cfg.n_embd;
        let vocab_size = cfg.vocab_size;
        let pos_emb = Tensor::from_vec(
            sinusoidal_positions(cfg.block_size, cfg.n_embd),
            vec![cfg.block_size, cfg.n_embd],
        );
        let blocks = (0..cfg.n_layer).map(|_| TransformerBlock::new(&cfg, rng)).collect();
        GPT {
            cfg,
            tok_emb: Embedding::new(vocab_size, n_embd, rng),
            pos_emb,
            blocks,
            ln_f: LayerNorm::new(n_embd, 1e-5),
            lm_head: Linear::new(n_embd, vocab_size, rng),
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
        let tok = self.tok_emb.forward(idx).reshape(vec![b, t, d]);

        // 2. 位置编码：KV cache 推理时，当前位置从缓存长度开始
        let base = kv_cache
            .as_ref()
            .map(|c| c.first().map(|k| k.seq_len()).unwrap_or(0))
            .unwrap_or(0);
        let mut positions = Vec::with_capacity(b * t);
        for _ in 0..b {
            for j in 0..t {
                positions.push(base + j);
            }
        }
        let pos_emb = self
            .pos_emb
            .gather_rows(&positions)
            .reshape(vec![b, t, d]);
        let x = tok.add(&pos_emb);

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
            x = block.forward(&x, &mask, cache);
        }

        // 5. 最终归一化 + 输出头
        let x = self.ln_f.forward(&x);
        let x = x.reshape(vec![b * t, d]);
        self.lm_head.forward(&x)
    }

    /// 推理用的缓存集合：每层一个
    pub fn new_kv_cache(&self) -> Vec<KVCache> {
        (0..self.cfg.n_layer).map(|_| KVCache::new()).collect()
    }
}

impl Module for GPT {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.tok_emb.parameters();
        for block in &self.blocks {
            ps.extend(block.parameters());
        }
        ps.extend(self.ln_f.parameters());
        ps.extend(self.lm_head.parameters());
        ps
    }
}
