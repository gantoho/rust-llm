//! 多头注意力（第 9-10 课）与 KV Cache（第 18 课）
//!
//! 注意力是 Transformer 的核心：让每个 token "关注"序列中其他 token，提取相关性。
//!
//! 本模块包含：
//! - [`KVCache`]：推理时缓存历史 K/V，避免重复计算
//! - [`MultiHeadAttention`]：多头自注意力 + RoPE 位置编码（第 19 课）

use crate::layers::Linear;
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;

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
///
/// 流程：
/// 1. 输入 x 经过 Q/K/V 三个线性投影
/// 2. Q/K 做 RoPE 旋转（第 19 课），V 不转
/// 3. 拆成多个头，计算 scores = Q·Kᵀ / √d_k
/// 4. 加因果掩码（屏蔽未来位置），softmax 得到注意力权重
/// 5. 加权求和 V，合并头，输出投影
pub struct MultiHeadAttention {
    pub c_q: Linear,
    pub c_k: Linear,
    pub c_v: Linear,
    pub c_proj: Linear,
    pub n_head: usize,
}

impl MultiHeadAttention {
    pub fn new(n_embd: usize, n_head: usize, rng: &mut Rng) -> Self {
        MultiHeadAttention {
            c_q: Linear::new(n_embd, n_embd, rng),
            c_k: Linear::new(n_embd, n_embd, rng),
            c_v: Linear::new(n_embd, n_embd, rng),
            c_proj: Linear::new(n_embd, n_embd, rng),
            n_head,
        }
    }

    /// 前向
    /// - x: [B, T, D]
    /// - mask: [T, T_total] 因果掩码（-inf 的位置不能看）
    /// - kv_cache: Some(缓存) 时走推理模式（只算新 token）
    /// - base: 当前窗口在绝对序列中的起始位置（训练时 = 0，KV cache 推理时 = 缓存长度）
    pub fn forward(
        &self,
        x: &Tensor,
        mask: &Tensor,
        kv_cache: Option<&mut KVCache>,
        base: usize,
    ) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let head_dim = d / self.n_head;
        assert_eq!(head_dim * self.n_head, d, "n_embd 必须能被 n_head 整除");

        // 1. 投影得到 Q、K、V（Linear 输出是 2D [B*T, D]，恢复成 3D）
        let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
        let k = self.c_k.forward(x).reshape(vec![b, t, d]);
        let v = self.c_v.forward(x).reshape(vec![b, t, d]);

        // 2. RoPE（第 19 课）：按绝对位置旋转 Q/K，V 保持原样。
        //    - 位置信息只参与打分 q·k，所以只转 Q/K 不转 V
        //    - 必须在 KV cache append 之前旋转：缓存里存的是"已旋转的 K"，历史 K 直接复用
        //    - 新 token 的绝对位置 = base + 窗口内下标 j（batch 内每个样本位置相同，重复 b 次）
        let mut positions = Vec::with_capacity(b * t);
        for _ in 0..b {
            positions.extend(base..base + t);
        }
        let q = q
            .reshape(vec![b * t, d])
            .rotary(&positions)
            .reshape(vec![b, t, d]);
        let k = k
            .reshape(vec![b * t, d])
            .rotary(&positions)
            .reshape(vec![b, t, d]);

        // 3. KV cache：拼接历史的 K/V（只影响 K、V 的长度）
        let (k, v) = match kv_cache {
            Some(cache) => {
                cache.append(&k, &v);
                (cache.k().unwrap(), cache.v().unwrap())
            }
            None => (k, v),
        };
        let t_total = k.shape()[1];

        // 4. 拆头：[B, T, D] -> [B*H, T, head_dim]
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

        // 5. 注意力分数：scores = Q·Kᵀ / √d_k
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kt = k.permute(&[0, 2, 1]); // [B*H, head_dim, T_total]
        let scores = q.matmul(&kt).mul_scalar(scale); // [B*H, T, T_total]

        // 6. 因果掩码：把"未来位置"变成 -inf，softmax 后概率为 0
        let scores = scores.add(mask);

        // 7. softmax 得到注意力权重，加权求和
        let attn = scores.softmax_last_dim(); // [B*H, T, T_total]
        let out = attn.matmul(&v); // [B*H, T, head_dim]

        // 8. 合并头回 [B, T, D]
        let out = out
            .reshape(vec![b, self.n_head, t, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b, t, d]);

        // 9. 输出投影
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
