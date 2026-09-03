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
use std::cell::RefCell;
use std::rc::Rc;

/// KV 缓存（第 18 课）：
/// 生成第 N 个 token 时，前 N-1 个 token 的 K、V 不需要重算。
/// 把每个注意力层的 K、V 存起来，每次只算新 token 的 K、V 并追加。
///
/// 内部直接持有 `Vec<f32>` 缓存，append 时只把新块 extend 到末尾，
/// 避免"每步克隆整段历史再拼接"的 O(T²) 开销。
pub struct KVCache {
    k: Rc<RefCell<Vec<f32>>>, // 行优先 [1, T, D] 展平
    v: Rc<RefCell<Vec<f32>>>,
    len: usize, // 已缓存的位置数 T
    d: usize,   // 隐藏维 D，第一次 append 时确定
}

impl KVCache {
    pub fn new() -> Self {
        KVCache {
            k: Rc::new(RefCell::new(Vec::new())),
            v: Rc::new(RefCell::new(Vec::new())),
            len: 0,
            d: 0,
        }
    }

    #[allow(dead_code)] // 多轮生成时重置缓存
    pub fn reset(&mut self) {
        self.k.borrow_mut().clear();
        self.v.borrow_mut().clear();
        self.len = 0;
        self.d = 0;
    }

    /// 当前已缓存的位置数
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// 把新的 k/v 追加到缓存末尾（只拷贝新块，不复制历史数据）
    pub fn append(&mut self, k: &Tensor, v: &Tensor) {
        assert_eq!(k.shape(), v.shape(), "K/V 形状必须一致");
        assert_eq!(k.rank(), 3, "K/V 必须为 3D [1, T, D]，实际 {:?}", k.shape());
        self.d = k.shape()[2];
        self.k.borrow_mut().extend(k.data());
        self.v.borrow_mut().extend(v.data());
        self.len += k.shape()[1];
    }

    /// 返回完整缓存张量 [1, T, D]（注意力打分需要读全量历史，这里克隆一次）
    pub fn k(&self) -> Tensor {
        Tensor::from_vec(self.k.borrow().clone(), vec![1, self.len, self.d])
    }

    pub fn v(&self) -> Tensor {
        Tensor::from_vec(self.v.borrow().clone(), vec![1, self.len, self.d])
    }
}

/// 多头注意力（第 9-10 课）+ Grouped Query Attention（GQA）
///
/// 流程：
/// 1. 输入 x 经过 Q/K/V 三个线性投影
/// 2. Q/K 做 RoPE 旋转（第 19 课），V 不转
/// 3. 拆成多个头，计算 scores = Q·Kᵀ / √d_k
/// 4. 加因果掩码（屏蔽未来位置），softmax 得到注意力权重
/// 5. 加权求和 V，合并头，输出投影
///
/// GQA（Grouped Query Attention）：n_kv_head < n_head 时，多个 Q head 共享 K/V head。
/// - n_kv_head = n_head：标准 MHA
/// - n_kv_head = 1：Multi-Query Attention（MQA）
/// - 1 < n_kv_head < n_head：GQA（LLaMA 2/3、Mistral 使用）
pub struct MultiHeadAttention {
    pub c_q: Linear,
    pub c_k: Linear,
    pub c_v: Linear,
    pub c_proj: Linear,
    pub n_head: usize,
    pub n_kv_head: usize,
    n_rep: usize, // n_head / n_kv_head
}

impl MultiHeadAttention {
    pub fn new(n_embd: usize, n_head: usize, n_kv_head: usize, rng: &mut Rng) -> Self {
        let n_kv = if n_kv_head == 0 { n_head } else { n_kv_head };
        assert!(n_head % n_kv == 0, "n_head 必须能被 n_kv_head 整除");
        let head_dim = n_embd / n_head;
        let kv_dim = n_kv * head_dim;
        MultiHeadAttention {
            c_q: Linear::new(n_embd, n_embd, rng),
            c_k: Linear::new(n_embd, kv_dim, rng),
            c_v: Linear::new(n_embd, kv_dim, rng),
            c_proj: Linear::new(n_embd, n_embd, rng),
            n_head,
            n_kv_head: n_kv,
            n_rep: n_head / n_kv,
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

        // [诊断] attention 内部分段计时（仅前 2 次调用）
        static ATT_DIAG: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let att_diag = ATT_DIAG.fetch_add(1, std::sync::atomic::Ordering::Relaxed) < 2;
        let att_t0 = std::time::Instant::now();

        // 1. 投影得到 Q、K、V
        let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
        let kv_dim = self.n_kv_head * head_dim;
        let k = self.c_k.forward(x).reshape(vec![b, t, kv_dim]); // [B, T, kv_dim]
        let v = self.c_v.forward(x).reshape(vec![b, t, kv_dim]);
        let t_proj = att_t0.elapsed();

        // 2. RoPE：Q/K 按 head_dim 旋转（GQA 时 K 只有 n_kv_head 个头）
        let mut positions = Vec::with_capacity(b * t);
        for _ in 0..b {
            positions.extend(base..base + t);
        }
        let (q, k) = q
            .reshape(vec![b * t, d])
            .rotary_pair(&k.reshape(vec![b * t, kv_dim]), &positions);
        let (q, k) = (
            q.reshape(vec![b, t, d]),
            k.reshape(vec![b, t, kv_dim]),
        );
        let t_rope = att_t0.elapsed();

        // 3. KV cache
        let (k, v) = match kv_cache {
            Some(cache) => {
                cache.append(&k, &v);
                (cache.k(), cache.v())
            }
            None => (k, v),
        };
        let t_total = k.shape()[1];

        // 4. 拆头 + GQA repeat
        //    Q: [B, T, n_head, head_dim] -> [B*n_head, T, head_dim]
        //    K/V: [B, T_total, n_kv_head, head_dim] -> repeat -> [B*n_head, T_total, head_dim]
        let q = q
            .reshape(vec![b, t, self.n_head, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b * self.n_head, t, head_dim]);

        let k = k
            .reshape(vec![b, t_total, self.n_kv_head, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b * self.n_kv_head, t_total, head_dim]);
        let v = v
            .reshape(vec![b, t_total, self.n_kv_head, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b * self.n_kv_head, t_total, head_dim]);

        // GQA：如果 n_kv_head < n_head，把 K/V 的每个头重复 n_rep 次
        let (k, v) = if self.n_rep > 1 {
            (repeat_kv(&k, self.n_rep), repeat_kv(&v, self.n_rep))
        } else {
            (k, v)
        };

        // 5. 注意力分数：scores = Q·Kᵀ / √d_k
        //    把缩放提前到 q（[B*H, T, Dh]）而不是 scores（[B*H, T, T_total]）：
        //    元素数少 T/Dh 倍，前向与反向都省一次大数组逐元素扫描。
        let scale = 1.0 / (head_dim as f32).sqrt();
        let kt = k.permute(&[0, 2, 1]); // [B*H, head_dim, T_total]
        let scores = q.mul_scalar(scale).matmul(&kt); // [B*H, T, T_total]
        let t_scores = att_t0.elapsed();

        // 6+7. 因果掩码 + softmax（融合实现：一个算子替代 add+softmax 两个算子）
        let attn = scores.masked_softmax(mask); // [B*H, T, T_total]
        let t_softmax = att_t0.elapsed();
        let out = attn.matmul(&v); // [B*H, T, head_dim]
        let t_attnv = att_t0.elapsed();

        // 8. 合并头回 [B, T, D]
        let out = out
            .reshape(vec![b, self.n_head, t, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b, t, d]);
        let t_merge = att_t0.elapsed();

        // 9. 输出投影
        let out = self.c_proj.forward(&out);
        if att_diag {
            let t_total = att_t0.elapsed();
            println!(
                "[diag-att] 投影 {:.1} | rope {:.1} | 拆头+scores {:.1} | mask+softmax {:.1} | attn·v {:.1} | 合头 {:.1} | c_proj {:.1} | 总 {:.1} ms",
                t_proj.as_secs_f64() * 1000.0,
                (t_rope - t_proj).as_secs_f64() * 1000.0,
                (t_scores - t_rope).as_secs_f64() * 1000.0,
                (t_softmax - t_scores).as_secs_f64() * 1000.0,
                (t_attnv - t_softmax).as_secs_f64() * 1000.0,
                (t_merge - t_attnv).as_secs_f64() * 1000.0,
                (t_total - t_merge).as_secs_f64() * 1000.0,
                t_total.as_secs_f64() * 1000.0,
            );
        }
        out
    }

    /// 带名字的参数（checkpoint 用）：`{prefix}.c_q/c_k/c_v/c_proj.*`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps = self.c_q.named_parameters(&format!("{prefix}.c_q"));
        ps.extend(self.c_k.named_parameters(&format!("{prefix}.c_k")));
        ps.extend(self.c_v.named_parameters(&format!("{prefix}.c_v")));
        ps.extend(self.c_proj.named_parameters(&format!("{prefix}.c_proj")));
        ps
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

/// GQA 辅助函数：把 KV 头重复 n_rep 次。
///
/// 输入 x: [B*n_kv_head, T, head_dim]
/// 输出:   [B*n_head, T, head_dim]
///
/// 例如 n_kv_head=2, n_rep=4 时：
/// [head0, head1] -> [head0, head0, head0, head0, head1, head1, head1, head1]
fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        return x.clone();
    }
    let shape = x.shape();
    assert_eq!(shape.len(), 3, "repeat_kv 输入必须为 3D");
    let (batch_kv, t, head_dim) = (shape[0], shape[1], shape[2]);
    let batch = batch_kv * n_rep;
    let xd = x.data();
    let mut out = vec![0.0f32; batch * t * head_dim];
    for b in 0..batch_kv {
        let src = &xd[b * t * head_dim..(b + 1) * t * head_dim];
        for r in 0..n_rep {
            let dst_start = (b * n_rep + r) * t * head_dim;
            out[dst_start..dst_start + t * head_dim].copy_from_slice(src);
        }
    }
    Tensor::from_vec(out, vec![batch, t, head_dim])
}
