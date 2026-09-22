//! 多头注意力（第 9-10 课）与 KV Cache（第 25 课）
//!
//! 注意力是 Transformer 的核心：让每个 token "关注"序列中其他 token，提取相关性。
//!
//! 本模块包含：
//! - [`KVCache`]：推理时缓存历史 K/V，避免重复计算
//! - [`MultiHeadAttention`]：多头自注意力 + RoPE 位置编码（第 20 课）

use crate::layers::Linear;
use crate::module::Module;
use crate::quant::{QAxis, QBits, QMatrix};
use crate::rng::Rng;
use crate::rope::RopeSpec;
use crate::tensor::Tensor;
use std::rc::Rc;

/// KV 缓存的配置（第 25 课滑动窗口 + 第 33 课量化 + Attention Sink）
#[derive(Clone, Copy, Debug, Default)]
pub struct KvCacheOpts {
    /// 保留上限（位置数）；0 = 不丢弃
    pub window: usize,
    /// **永久保留最前面的 `sink` 个位置**（Attention Sink / StreamingLLM）。
    ///
    /// 流式推理里丢掉最早的几个 token 会让质量断崖式下跌——不是因为那几句话重要，
    /// 而是因为 softmax 必须有地方"放"多余的注意力：序列开头那几个位置在任何模型里
    /// 都承担着"注意力汇（attention sink）"的角色，一旦被窗口丢掉，剩余位置的
    /// 概率质量就被强行摊到少数几个 token 上，输出立刻退化成一堆重复词。
    /// 保留它们（哪怕只有 4 个）就能让模型在几十万 token 的流式输入上稳定生成。
    ///
    /// 必须满足 `0 ≤ sink < window`（`window = 0` 时无意义，整体不丢弃）。
    pub sink: usize,
    /// KV cache 的量化位宽：`None` = f32，`Some(Int8/Int4)` = 按 KIVI 的方式压缩
    /// （K 逐通道、V 逐 token，见 [`KVCache`] 的文档）
    pub bits: Option<QBits>,
}

/// KV 缓存的存储块：f32 或量化。
///
/// 两者对外只暴露"读成 `Vec<f32>` / 追加若干行 / 丢掉最前面若干行"，
/// 于是窗口与 Attention Sink 的裁剪逻辑不必关心底层是哪种表示。
enum KvBlock {
    F32(Vec<f32>),
    Quant(QMatrix),
}

impl KvBlock {
    fn new(bits: Option<QBits>, axis: QAxis) -> Self {
        match bits {
            None => KvBlock::F32(Vec::new()),
            // 列数要等第一次 append 才知道，这里先占位（rows=0 的合法空矩阵）
            Some(b) => KvBlock::Quant(QMatrix::zeros(0, 0, b, axis)),
        }
    }

    /// 还原成 `[rows, d]` 的 f32 行优先数据
    fn to_vec(&self, d: usize) -> Vec<f32> {
        match self {
            KvBlock::F32(v) => v.clone(),
            KvBlock::Quant(q) => {
                debug_assert_eq!(q.cols(), d);
                q.dequantize()
            }
        }
    }

    /// 追加 `rows` 行（每行 `d` 个数）
    fn push(&mut self, x: &[f32], rows: usize, d: usize) {
        match self {
            KvBlock::F32(v) => v.extend_from_slice(x),
            KvBlock::Quant(q) => {
                if q.rows() == 0 && q.cols() == 0 {
                    // 第一批数据：用它的数值统计出分组 scale
                    *q = QMatrix::quantize(x, rows, d, q.bits(), q.axis());
                } else {
                    q.push_rows(x, rows);
                }
            }
        }
    }

    /// 丢掉最前面的 `n` 行
    fn drop_front(&mut self, n: usize, d: usize) {
        if n == 0 {
            return;
        }
        match self {
            KvBlock::F32(v) => {
                v.drain(..n * d);
            }
            KvBlock::Quant(q) => q.drop_front_rows(n),
        }
    }

    /// 当前占用的字节数（用于打印量化收益）
    fn byte_len(&self) -> usize {
        match self {
            KvBlock::F32(v) => v.len() * 4,
            KvBlock::Quant(q) => q.byte_len(),
        }
    }
}

/// KV 缓存（第 25 课；滑动窗口、Attention Sink、量化见 [`KvCacheOpts`]）：
/// 生成第 N 个 token 时，前 N-1 个 token 的 K、V 不需要重算。
/// 把每个注意力层的 K、V 存起来，每次只算新 token 的 K、V 并追加。
///
/// 内部直接持有展平缓存，append 时只把新块 extend 到末尾，
/// 避免"每步克隆整段历史再拼接"的 O(T²) 开销。
///
/// **滑动窗口**（`window > 0`）：缓存超过 `window` 个位置时丢弃最旧的若干行，
/// 于是生成长度不再受缓存容量限制，可以一直生成下去。这是真实长上下文推理的
/// 常规做法（Mistral 的 sliding window attention 即此），也是"KV cache 模式"
/// 与"全量重算模式"行为对齐的前提——两者都必须只让 query 看最近 `window` 个位置。
///
/// **Attention Sink**（`sink > 0`）：丢弃时**跳过最前面的 `sink` 个位置**，
/// 只丢中间那些。原因见 [`KvCacheOpts::sink`]。
///
/// 丢弃旧行不破坏 RoPE 的相对位置语义：绝对位置由 [`KVCache::positions_seen`]
/// 统一计数，缓存里保留的 K/V 仍带着各自真实的绝对位置旋转，任意 query/key 对的
/// 相对距离与"全量重算"完全一致（RoPE 的注意力打分只依赖相对距离）。
///
/// **量化（KIVI）**：`opts.bits = Some(_)` 时缓存里的 K/V 以整数码存放
/// （见 [`crate::quant::QMatrix`]），读出时才还原成 f32。长上下文推理的显存瓶颈
/// 恰恰是这个缓存（`2 × 层数 × 头数 × 上下文 × head_dim × 4` 字节），压到 int8/int4
/// 是唯一能把上下文继续加长的办法。两个方向刻意取不同的分组：
/// - **K 逐通道（[`QAxis::Col`]）**：K 的每个通道在同一个头内尺度稳定，
///   逐通道 scale 能把量化误差压到最低；代价是 scale 必须**冻结**在第一批数据上
///   （列 scale 是共享的，随追加重算会让已写入的历史码值失效）。
/// - **V 逐 token（[`QAxis::Row`]）**：V 的每个位置是一行，行内数值同尺度、
///   行间差异大，逐行 scale 既准确又不影响历史数据（每行自带一个 scale）。
pub struct KVCache {
    k: KvBlock,
    v: KvBlock,
    len: usize,    // 当前**保留**的位置数 T（滑动窗口下不会超过 window）
    seen: usize,   // 累计喂进来的位置总数，只增不减（RoPE 绝对位置基准）
    window: usize, // 保留上限；0 = 不丢弃（缓存只拼不丢）
    sink: usize,   // 永久保留的最前面若干位置
    d: usize,      // 隐藏维 D，第一次 append 时确定
}

impl KVCache {
    /// 按配置构造
    pub fn new(opts: KvCacheOpts) -> Self {
        assert!(
            opts.window == 0 || opts.sink < opts.window,
            "Attention Sink 个数（{}）必须小于窗口（{}），否则缓存里没有可丢的位置",
            opts.sink,
            opts.window
        );
        KVCache {
            k: KvBlock::new(opts.bits, QAxis::Col),
            v: KvBlock::new(opts.bits, QAxis::Row),
            len: 0,
            seen: 0,
            window: opts.window,
            sink: opts.sink,
            d: 0,
        }
    }

    /// 带滑动窗口的缓存：最多保留 `window` 个位置，超出则丢最旧的；`window = 0` 表示不丢弃
    pub fn with_window(window: usize) -> Self {
        Self::new(KvCacheOpts {
            window,
            sink: 0,
            bits: None,
        })
    }

    /// 深拷贝一份（Beam Search 的每条候选路径都要有自己独立的缓存）。
    ///
    /// 不能靠 `#[derive(Clone)]`：缓存内部是 `Rc<RefCell<..>>`，派生的 clone 会共享
    /// 同一份数据，一条路径的 append 会污染其它路径。这里逐字节复制。
    pub fn fork(&self) -> Self {
        KVCache {
            k: match &self.k {
                KvBlock::F32(v) => KvBlock::F32(v.clone()),
                KvBlock::Quant(q) => KvBlock::Quant(q.clone()),
            },
            v: match &self.v {
                KvBlock::F32(v) => KvBlock::F32(v.clone()),
                KvBlock::Quant(q) => KvBlock::Quant(q.clone()),
            },
            len: self.len,
            seen: self.seen,
            window: self.window,
            sink: self.sink,
            d: self.d,
        }
    }

    pub fn reset(&mut self) {
        self.k = KvBlock::new(self.bits(), QAxis::Col);
        self.v = KvBlock::new(self.bits(), QAxis::Row);
        self.len = 0;
        self.seen = 0;
        self.d = 0;
    }

    /// 当前保留的位置数（= 注意力里 K/V 的序列长度）
    pub fn seq_len(&self) -> usize {
        self.len
    }

    /// 累计喂进来的位置总数：新 token 的 RoPE 绝对位置基准（滑动窗口下与 `seq_len` 不同）
    pub fn positions_seen(&self) -> usize {
        self.seen
    }

    /// 保留上限（0 = 不丢弃）
    pub fn window(&self) -> usize {
        self.window
    }

    /// 永久保留的位置个数（Attention Sink）
    pub fn sink(&self) -> usize {
        self.sink
    }

    /// 量化位宽（`None` = f32）
    pub fn bits(&self) -> Option<QBits> {
        match &self.k {
            KvBlock::F32(_) => None,
            KvBlock::Quant(q) => Some(q.bits()),
        }
    }

    /// 当前缓存占用的字节数（含分组 scale）。量化收益就是拿它与 f32 版本对比
    pub fn byte_len(&self) -> usize {
        self.k.byte_len() + self.v.byte_len()
    }

    /// 把新的 k/v 追加到缓存末尾（只拷贝新块，不复制历史数据），
    /// 并在超过窗口时丢弃最旧的若干行（跳过最前面的 `sink` 个位置）。
    pub fn append(&mut self, k: &Tensor, v: &Tensor) {
        assert_eq!(k.shape(), v.shape(), "K/V 形状必须一致");
        assert_eq!(k.rank(), 3, "K/V 必须为 3D [1, T, D]，实际 {:?}", k.shape());
        assert_eq!(k.shape()[0], 1, "KV cache 只支持 batch = 1");
        let t = k.shape()[1];
        let d = k.shape()[2];
        if self.d == 0 {
            self.d = d;
        } else {
            assert_eq!(self.d, d, "KV cache 的隐藏维不能中途改变");
        }
        self.k.push(&k.data_ref(), t, d);
        self.v.push(&v.data_ref(), t, d);
        self.len += t;
        self.seen += t;
        // 滑动窗口：丢掉中间那些旧行，让 len 回到 window。
        // 丢的行是 `[sink, len - (window - sink))`：最前面的 sink 个（Attention Sink）
        // 与最近的 `window - sink` 个都保留。
        if self.window > 0 && self.len > self.window {
            let drop = self.len - self.window;
            self.k.drop_front_from(self.sink, drop, self.d);
            self.v.drop_front_from(self.sink, drop, self.d);
            self.len = self.window;
        }
    }

    /// 返回完整缓存张量 [1, T, D]（注意力打分需要读全量历史，这里克隆一次）
    pub fn k(&self) -> Tensor {
        Tensor::from_vec(self.k.to_vec(self.d), vec![1, self.len, self.d])
    }

    pub fn v(&self) -> Tensor {
        Tensor::from_vec(self.v.to_vec(self.d), vec![1, self.len, self.d])
    }

    /// 回滚最近追加的 `n` 个位置（推测解码用：草稿 token 被拒后必须从缓存里撤掉，
    /// 否则下一步的注意力会把这批"从未被采纳"的 token 当成真实历史）。
    ///
    /// `positions_seen` 也一起回退，于是被重新喂进来的 token 拿到的绝对位置与第一次相同，
    /// RoPE 的相对距离语义不受影响。
    ///
    /// **滑动窗口下不是严格可逆**：如果刚才那几次 [`KVCache::append`] 已经触发了超窗丢弃，
    /// 被丢掉的旧行不在缓存里，回滚找不回来。所以推测解码里 `gamma` 要远小于 `window`
    /// （常规用法本就如此），此时差异可以忽略。
    pub fn rollback(&mut self, n: usize) {
        assert!(n <= self.len, "回滚 {n} 个位置超过缓存里的 {} 个", self.len);
        if n == 0 {
            return;
        }
        self.k.drop_back(n, self.d);
        self.v.drop_back(n, self.d);
        self.len -= n;
        self.seen -= n;
    }
}

impl KvBlock {
    /// 丢掉最末尾的 `n` 行（推测解码回滚用）
    fn drop_back(&mut self, n: usize, d: usize) {
        if n == 0 {
            return;
        }
        match self {
            KvBlock::F32(v) => {
                v.truncate(v.len() - n * d);
            }
            KvBlock::Quant(q) => q.drop_back_rows(n),
        }
    }

    /// 从 `offset` 行开始丢掉 `n` 行（Attention Sink 用：跳过最前面的若干行）
    fn drop_front_from(&mut self, offset: usize, n: usize, d: usize) {
        if n == 0 {
            return;
        }
        if offset == 0 {
            return self.drop_front(n, d);
        }
        // 保留 [0, offset) 与 [offset + n, 末尾)：重建一次数据。
        // 只有在"超窗 + 开了 sink"时才会走到这里，且每次只重建一次，
        // 与 f32 路径的 `drain` 同为 O(len·d)，不改变整体复杂度。
        let all = self.to_vec(d);
        let rows = all.len() / d;
        let mut kept: Vec<f32> = Vec::with_capacity((rows - n) * d);
        kept.extend_from_slice(&all[..offset * d]);
        kept.extend_from_slice(&all[(offset + n) * d..]);
        match self {
            KvBlock::F32(v) => *v = kept,
            KvBlock::Quant(q) => {
                let bits = q.bits();
                let axis = q.axis();
                // 逐通道（K）的 scale 全列共享、必须沿用旧的（重建不改列的定义）；
                // 逐 token（V）的 scale 随着行一起搬，跳掉被丢弃的那一段。
                let scales: Vec<f32> = match axis {
                    QAxis::Col => q.scales().to_vec(),
                    QAxis::Row => q.scales()[..offset]
                        .iter()
                        .chain(&q.scales()[offset + n..])
                        .copied()
                        .collect(),
                };
                *q = QMatrix::quantize_with_scales(&kept, rows - n, d, bits, axis, &scales);
            }
        }
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
    /// RoPE 的频率参数（底数 + 长度外推方式）。**结构的一部分**：训练与推理、
    /// checkpoint 加载与续训必须一致，否则同一段文本会被旋转到不同角度。
    pub rope: RopeSpec,
}

impl MultiHeadAttention {
    pub fn new(
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        rope: RopeSpec,
        rng: &mut Rng,
    ) -> Self {
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
            rope,
        }
    }

    /// 前向
    /// - x: [B, T, D]
    /// - mask: [T, T_total] 因果掩码（-inf 的位置不能看）
    /// - kv_cache: Some(缓存) 时走推理模式（只算新 token）
    /// - base: RoPE 的绝对位置基准（训练时 = 0，KV cache 推理时 = 缓存已见位置总数）
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

        // 1. 投影得到 Q、K、V
        let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
        let kv_dim = self.n_kv_head * head_dim;
        let k = self.c_k.forward(x).reshape(vec![b, t, kv_dim]); // [B, T, kv_dim]
        let v = self.c_v.forward(x).reshape(vec![b, t, kv_dim]);

        // 2. RoPE：Q/K 按 head_dim 旋转（GQA 时 K 只有 n_kv_head 个头）
        let mut positions = Vec::with_capacity(b * t);
        for _ in 0..b {
            positions.extend(base..base + t);
        }
        let (q, k) = q
            .reshape(vec![b * t, d])
            .rotary_pair(&k.reshape(vec![b * t, kv_dim]), &positions, &self.rope);
        let (q, k) = (
            q.reshape(vec![b, t, d]),
            k.reshape(vec![b, t, kv_dim]),
        );

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

        // 5-7. Flash Attention 融合算子：Q'·Kᵀ → softmax(+mask) → ·V 全走矩阵乘内核
        //      （2026-09-16 重写，见 `Tensor::flash_attention`：数学等价，但不再做分块在线 softmax）
        //      `block_size` 参数已失效；显存仍是 O(T²)（保留 P 供反向用）
        let out = Tensor::flash_attention(&q, &k, &v, mask, 32);

        // 8. 合并头回 [B, T, D]
        let out = out
            .reshape(vec![b, self.n_head, t, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b, t, d]);

        // 9. 输出投影
        let out = self.c_proj.forward(&out);
        out
    }

    /// 按 `targets` 给 Q/K/V/输出投影挂上 LoRA 适配器。
    ///
    /// 缺省只挂 Q/K/V：注意力里"该去看哪里"（Q/K）和"看到了取什么"（V）最需要随下游任务
    /// 调整，是 LoRA 论文与社区实践里性价比最高的一组。输出投影 `c_proj` 只做一次线性汇总，
    /// 加适配器收益最小而参数与 Q 一样多，所以默认关闭（`--lora-targets` 可打开）。
    ///
    /// 主干冻结不在这里做，由 [`crate::model::GPT::apply_lora`] 统一处理。
    pub fn apply_lora(&mut self, lora: &crate::config::LoRAConfig, rng: &mut Rng) {
        let t = lora.targets;
        for (on, lin) in [
            (t.q, &mut self.c_q),
            (t.k, &mut self.c_k),
            (t.v, &mut self.c_v),
            (t.o, &mut self.c_proj),
        ] {
            if on {
                lin.attach_lora(lora.rank, lora.alpha, rng);
            }
        }
    }

    /// 把本层各投影的适配器合并进主干（推理用，见 [`Linear::merge_lora`]）
    pub fn merge_lora(&mut self) {
        for lin in [
            &mut self.c_q,
            &mut self.c_k,
            &mut self.c_v,
            &mut self.c_proj,
        ] {
            lin.merge_lora();
        }
    }

    /// 本层是否有任一投影挂了适配器
    /// （GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_lora(&self) -> bool {
        [&self.c_q, &self.c_k, &self.c_v, &self.c_proj]
            .iter()
            .any(|l| l.lora.is_some())
    }

    /// 本层是否有任一投影被量化（GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_quant(&self) -> bool {
        [&self.c_q, &self.c_k, &self.c_v, &self.c_proj]
            .iter()
            .any(|l| l.has_quant())
    }

    /// 把各投影的量化状态烘焙回 f32（checkpoint 里写的始终是 f32 权重）
    pub fn dequantize_weights(&mut self) {
        for lin in [
            &mut self.c_q,
            &mut self.c_k,
            &mut self.c_v,
            &mut self.c_proj,
        ] {
            lin.dequantize_weight();
        }
    }

    /// 带名字的参数（checkpoint 用）：`{prefix}.c_q/c_k/c_v/c_proj.*`
    /// （挂了 LoRA 时各投影下还有 `.lora_a` / `.lora_b`，由 [`Linear::named_parameters`] 递归带出）
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_linears(prefix)
            .into_iter()
            .flat_map(|(p, lin)| lin.named_parameters(&p))
            .collect()
    }

    /// 四个投影 + 各自的参数名前缀。`named_parameters` 由它派生，
    /// 名字因此只有一处定义——量化要按同一套名字取校准统计，见 [`crate::quant::CalibStats`]。
    pub fn named_linears(&self, prefix: &str) -> Vec<(String, &Linear)> {
        vec![
            (format!("{prefix}.c_q"), &self.c_q),
            (format!("{prefix}.c_k"), &self.c_k),
            (format!("{prefix}.c_v"), &self.c_v),
            (format!("{prefix}.c_proj"), &self.c_proj),
        ]
    }

    pub fn named_linears_mut(&mut self, prefix: &str) -> Vec<(String, &mut Linear)> {
        vec![
            (format!("{prefix}.c_q"), &mut self.c_q),
            (format!("{prefix}.c_k"), &mut self.c_k),
            (format!("{prefix}.c_v"), &mut self.c_v),
            (format!("{prefix}.c_proj"), &mut self.c_proj),
        ]
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
///
/// 反向：将 n_rep 个重复副本的梯度求和回原始 KV 头。
fn repeat_kv(x: &Tensor, n_rep: usize) -> Tensor {
    if n_rep == 1 {
        return x.clone();
    }
    let shape = x.shape();
    assert_eq!(shape.len(), 3, "repeat_kv 输入必须为 3D");
    let (batch_kv, t, head_dim) = (shape[0], shape[1], shape[2]);
    let batch = batch_kv * n_rep;
    let xd = x.data.borrow();
    let mut out = vec![0.0f32; batch * t * head_dim];
    for b in 0..batch_kv {
        let src = &xd[b * t * head_dim..(b + 1) * t * head_dim];
        for r in 0..n_rep {
            let dst_start = (b * n_rep + r) * t * head_dim;
            out[dst_start..dst_start + t * head_dim].copy_from_slice(src);
        }
    }
    drop(xd);

    let mut result = Tensor::new(out, vec![batch, t, head_dim], x.req());
    if x.req() {
        let rg = result.grad.clone();
        let sg = x.grad.clone();
        let n_rep2 = n_rep;
        let elems = t * head_dim;
        result.parents = Rc::new(vec![x.clone()]);
        result.backward = Some(Rc::new(move || {
            let g = rg.borrow();
            let mut sgm = sg.borrow_mut();
            // 反向：将 n_rep 个副本的梯度求和回原始头
            for b in 0..batch_kv {
                let dst_base = b * elems;
                for r in 0..n_rep2 {
                    let src_base = (b * n_rep2 + r) * elems;
                    for i in 0..elems {
                        sgm[dst_base + i] += g[src_base + i];
                    }
                }
            }
        }));
    }
    result
}

/// GQA 辅助函数（权重空间版）：把 K/V 投影的**参数**按 [`repeat_kv`] 的同一套头顺序展开。
///
/// `src` 是 `[rows, n_kv_head * head_dim]` 的行主序数据（权重取 `rows = d`，
/// 偏置取 `rows = 1`），返回 `[rows, n_head * head_dim]`：
/// 输出第 `hh` 个头直接复制自源头 `hh / n_rep` —— 与 `repeat_kv` 把
/// `[head0, head1]` 变成 `[head0, head0, head0, head0, head1, …]` 完全一致。
///
/// 存在的理由：GPU 常驻显存路径的「按头重排」内核只认 `n_head` 个头，
/// 与其在三条内核里各加一套头复制（前向复制、反向跨副本求和），不如在**参数**上
/// 做一次等价变换——展开后 K/V 与 Q 同形，整条常驻路径原样复用，一行内核都不用改。
/// 多做的工作只是每步每层多上传 `(n_rep-1)` 份 K/V 权重（默认配置下 64KB 量级）。
///
/// 梯度的折回见 [`fold_kv_head_grad`]。
///
/// 唯一调用点在 `model.rs` 的 `#[cfg(feature = "gpu")]` 函数里，
/// 不带 gpu feature 编译时它是"死代码"——这不是真死代码，别删（单测也在用它）。
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn expand_kv_head(src: &[f32], rows: usize, n_kv: usize, hd: usize, n_rep: usize) -> Vec<f32> {
    let n_head = n_kv * n_rep;
    let mut out = vec![0.0f32; rows * n_head * hd];
    for r in 0..rows {
        let src_row = &src[r * n_kv * hd..(r + 1) * n_kv * hd];
        let dst_row = &mut out[r * n_head * hd..(r + 1) * n_head * hd];
        for hh in 0..n_head {
            let s = (hh / n_rep) * hd;
            dst_row[hh * hd..(hh + 1) * hd].copy_from_slice(&src_row[s..s + hd]);
        }
    }
    out
}

/// [`expand_kv_head`] 的反向：把展开空间里的参数梯度折回原始 `n_kv` 个头。
///
/// 同一源头的 `n_rep` 个副本各自攒到一份梯度，折回时按列相加 ——
/// 对应 `repeat_kv` 反向「把 n_rep 个副本的梯度求和回原始头」。
/// 数学上与「先求和再乘」是同一个结果：展开空间里 `dWk[:, hh] = xnᵀ·dk_pre'[:, hh]`，
/// 按 `hh / n_rep` 分组相加后正是 `xnᵀ·(Σ_r dk_pre'[:, …])`，也就是未展开时的 `dWk`。
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub(crate) fn fold_kv_head_grad(g: &[f32], rows: usize, n_kv: usize, hd: usize, n_rep: usize) -> Vec<f32> {
    let n_head = n_kv * n_rep;
    let mut out = vec![0.0f32; rows * n_kv * hd];
    for r in 0..rows {
        let src_row = &g[r * n_head * hd..(r + 1) * n_head * hd];
        let dst_row = &mut out[r * n_kv * hd..(r + 1) * n_kv * hd];
        for hh in 0..n_head {
            let d = (hh / n_rep) * hd;
            for j in 0..hd {
                dst_row[d + j] += src_row[hh * hd + j];
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 滑动窗口的机械行为：超窗后只丢最旧的行、`seq_len` 封顶而 `positions_seen` 继续累加。
    #[test]
    fn test_kv_cache_sliding_window_drops_oldest_rows() {
        let mut cache = KVCache::with_window(3);
        // 每次追加 1 行，行内容就是该位置的编号（d = 2，方便肉眼核对）
        let row = |i: usize| Tensor::from_vec(vec![i as f32, i as f32 + 0.5], vec![1, 1, 2]);

        for i in 0..6 {
            cache.append(&row(i), &row(i));
            assert!(cache.seq_len() <= 3, "缓存不应超过窗口");
        }
        assert_eq!(cache.seq_len(), 3, "窗口应保持在 3 行");
        assert_eq!(cache.positions_seen(), 6, "绝对位置应累计全部喂入的行");

        // 留下的必须是最近 3 行（位置 3/4/5），最旧的 0/1/2 被丢掉
        assert_eq!(cache.k().data(), vec![3.0, 3.5, 4.0, 4.5, 5.0, 5.5]);
        assert_eq!(cache.v().data(), vec![3.0, 3.5, 4.0, 4.5, 5.0, 5.5]);

        // 一次追加多行、且一次就超出窗口：同样只保留最后 window 行
        let mut cache = KVCache::with_window(2);
        let big = Tensor::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0], vec![1, 4, 2]);
        cache.append(&big, &big);
        assert_eq!(cache.seq_len(), 2);
        assert_eq!(cache.positions_seen(), 4);
        assert_eq!(cache.k().data(), vec![2.0, 2.0, 3.0, 3.0]);

        // window = 0：不丢弃，缓存只拼不丢
        let mut cache = KVCache::with_window(0);
        for i in 0..5 {
            cache.append(&row(i), &row(i));
        }
        assert_eq!(cache.seq_len(), 5);
        assert_eq!(cache.positions_seen(), 5);
    }

    /// Attention Sink：超窗时被丢的是**中间**那些行，最前面的 sink 行必须留下。
    #[test]
    fn test_kv_cache_attention_sink_keeps_leading_rows() {
        // 窗口 3、sink 1：永久保留位置 0，其余两个名额给最近的行
        let mut cache = KVCache::new(KvCacheOpts {
            window: 3,
            sink: 1,
            bits: None,
        });
        let row = |i: usize| Tensor::from_vec(vec![i as f32, i as f32 + 0.5], vec![1, 1, 2]);
        for i in 0..6 {
            cache.append(&row(i), &row(i));
        }
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.positions_seen(), 6);
        assert_eq!(cache.sink(), 1);
        // 0（注意力汇）+ 4 + 5：中间被丢掉的正是 1、2、3
        assert_eq!(cache.k().data(), vec![0.0, 0.5, 4.0, 4.5, 5.0, 5.5]);
        assert_eq!(cache.v().data(), vec![0.0, 0.5, 4.0, 4.5, 5.0, 5.5]);
    }

    /// sink 必须严格小于窗口，否则缓存里根本没有可丢的位置
    #[test]
    #[should_panic(expected = "Attention Sink")]
    fn test_attention_sink_must_be_smaller_than_window() {
        let _ = KVCache::new(KvCacheOpts {
            window: 4,
            sink: 4,
            bits: None,
        });
    }

    /// 量化缓存：读回的行要贴着 f32 版本（KIVI 的两个方向各自成立），且占用字节更少
    #[test]
    fn test_quantized_kv_cache_matches_f32_rows_and_saves_bytes() {
        let (rows, d) = (8usize, 16usize);
        let data: Vec<f32> = (0..rows * d).map(|i| (i as f32 * 0.37).sin()).collect();
        let kv = Tensor::from_vec(data.clone(), vec![1, rows, d]);

        let mut plain = KVCache::with_window(0);
        plain.append(&kv, &kv);

        for (bits, tol) in [(QBits::Int8, 0.01f32), (QBits::Int4, 0.08f32)] {
            let mut c = KVCache::new(KvCacheOpts {
                window: 0,
                sink: 0,
                bits: Some(bits),
            });
            c.append(&kv, &kv);
            assert_eq!(c.seq_len(), rows);
            assert_eq!(c.positions_seen(), rows);
            assert_eq!(c.bits(), Some(bits));

            let max_err = |a: &[f32], b: &[f32]| {
                a.iter()
                    .zip(b)
                    .fold(0.0f32, |m, (x, y)| m.max((x - y).abs()))
            };
            // K 是逐通道分组，V 是逐 token 分组，两条路径都要落在误差界内
            let ek = max_err(&plain.k().data(), &c.k().data());
            let ev = max_err(&plain.v().data(), &c.v().data());
            assert!(ek <= tol, "{bits:?} 的 K 最大误差 {ek} 超过 {tol}");
            assert!(ev <= tol, "{bits:?} 的 V 最大误差 {ev} 超过 {tol}");
            assert!(
                c.byte_len() < plain.byte_len(),
                "{bits:?} 的缓存 {} 字节应少于 f32 的 {} 字节",
                c.byte_len(),
                plain.byte_len()
            );
        }

        // int4 必须比 int8 更省
        let mk = |bits| {
            let mut c = KVCache::new(KvCacheOpts {
                window: 0,
                sink: 0,
                bits: Some(bits),
            });
            c.append(&kv, &kv);
            c.byte_len()
        };
        assert!(mk(QBits::Int4) < mk(QBits::Int8));
    }

    /// 量化缓存与滑动窗口 / Attention Sink 共存：裁剪后行数、首行、最近行都要对。
    ///
    /// 数值刻意设成"第 0 行全零、第 1 行给出最大量级、之后逐行递减"：
    /// 第 0 行让列 scale 停在**未定**状态，第 1 行才完成定标，之后的行都在范围内不被裁剪。
    #[test]
    fn test_quantized_kv_cache_coexists_with_sink_and_window() {
        let d = 4;
        let values = [0.0f32, 0.9, 0.8, 0.7, 0.6, 0.5];
        let row = |v: f32| Tensor::from_vec(vec![v; d], vec![1, 1, d]);

        let mut c = KVCache::new(KvCacheOpts {
            window: 3,
            sink: 1,
            bits: Some(QBits::Int8),
        });
        for &v in &values {
            c.append(&row(v), &row(v));
        }
        assert_eq!(c.seq_len(), 3);
        assert_eq!(c.positions_seen(), values.len());
        let k = c.k().data();
        // 留下的是位置 0（sink）、4、5：中间 1、2、3 被丢
        for (i, want) in [values[0], values[4], values[5]].iter().enumerate() {
            for j in 0..d {
                let got = k[i * d + j];
                assert!(
                    (got - want).abs() < 0.01,
                    "第 {i} 行第 {j} 列应为 {want}，实际 {got}：{k:?}"
                );
            }
        }
    }

    /// [`KVCache::fork`] 必须是真正的深拷贝：一条路径的追加不得污染另一条
    #[test]
    fn test_kv_cache_fork_is_independent() {
        let d = 2;
        let row = |i: usize| Tensor::from_vec(vec![i as f32, i as f32], vec![1, 1, d]);
        let mut base = KVCache::new(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: Some(QBits::Int8),
        });
        base.append(&row(1), &row(1));
        base.append(&row(2), &row(2));

        let mut a = base.fork();
        let b = base.fork();
        a.append(&row(3), &row(3));

        assert_eq!(base.seq_len(), 2, "fork 之后原缓存不应被改动");
        assert_eq!(b.seq_len(), 2, "两条分叉互不影响");
        assert_eq!(a.seq_len(), 3);
        assert_eq!(a.positions_seen(), 3);
        assert_eq!(base.positions_seen(), 2);
        assert!(b.k().data().iter().all(|v| *v < 3.0), "b 不应看到 a 追加的行");
    }

    /// [`KVCache::rollback`] 撤掉末尾若干行后，缓存必须与"一开始就没喂过这些行"逐位一致，
    /// 且 `positions_seen` 一起回退——于是重新喂进来的 token 绝对位置不变。
    ///
    /// 这是推测解码能成立的前提：草稿被拒的那一刻，target 缓存不能留着那批 token 的痕迹。
    #[test]
    fn test_kv_cache_rollback_is_exact() {
        let d = 3;
        let row = |i: usize| {
            Tensor::from_vec(
                (0..d).map(|j| ((i * 5 + j * 3) % 11) as f32 * 0.13 - 0.7).collect(),
                vec![1, 1, d],
            )
        };
        for bits in [None, Some(QBits::Int8), Some(QBits::Int4)] {
            let mut c = KVCache::new(KvCacheOpts { window: 0, sink: 0, bits });
            for i in 0..4 {
                c.append(&row(i), &row(i));
            }
            c.append(&row(4), &row(4)); // 两条"草稿"
            c.append(&row(5), &row(5));
            c.rollback(2);

            let mut fresh = KVCache::new(KvCacheOpts { window: 0, sink: 0, bits });
            for i in 0..4 {
                fresh.append(&row(i), &row(i));
            }
            assert_eq!(c.seq_len(), 4, "{bits:?}：回滚后的行数");
            assert_eq!(c.positions_seen(), 4, "{bits:?}：位置计数要一起回退");
            for (a, b) in c.k().data().iter().zip(fresh.k().data()) {
                assert!((a - b).abs() < 1e-6, "{bits:?}：回滚后的 K 应与全新缓存一致");
            }
            for (a, b) in c.v().data().iter().zip(fresh.v().data()) {
                assert!((a - b).abs() < 1e-6, "{bits:?}：回滚后的 V 应与全新缓存一致");
            }
            // 回滚之后继续追加：位置接在 4 上，与从未草稿过一样
            c.append(&row(9), &row(9));
            fresh.append(&row(9), &row(9));
            assert_eq!(c.positions_seen(), 5);
            for (a, b) in c.k().data().iter().zip(fresh.k().data()) {
                assert!((a - b).abs() < 1e-6, "{bits:?}：回滚后再追加仍要一致");
            }
        }
    }

    /// GQA 的**权重空间展开**必须与张量空间的 [`repeat_kv`] 同序，且折回是展开的共轭。
    /// 这两条就是「参数空间做头复制」能替代「张量空间头复制」的全部依据：
    /// 同序保证前向算的是同一个函数，共轭保证反向梯度折回后与未展开时逐位一致
    /// （`⟨expand(a), g⟩ == ⟨a, fold(g)⟩` 即「先求和再乘」= 「分别乘再求和」）。
    #[test]
    fn test_expand_kv_head_matches_repeat_kv_and_fold_is_adjoint() {
        let (rows, n_kv, hd, n_rep) = (3usize, 2usize, 2usize, 3usize);
        let n_head = n_kv * n_rep;
        let src: Vec<f32> = (0..rows * n_kv * hd)
            .map(|i| ((i * 37 % 19) as f32 * 0.21).sin())
            .collect();
        let exp = expand_kv_head(&src, rows, n_kv, hd, n_rep);
        assert_eq!(exp.len(), rows * n_head * hd);

        // 1) 头顺序：`repeat_kv` 吃 `[B*n_kv, T, hd]`（令 B=1、T=rows），
        //    把第 b 个 KV 头复制成 n_rep 份；展开结果的第 hh 个头应取自源头的 `hh / n_rep`。
        //    两者的数据布局不同（源是「行内多头连续」，repeat_kv 是「头在外、行长在内」），
        //    这里把源转置成 repeat_kv 认的布局再比。
        let mut kv_in = vec![0.0f32; n_kv * rows * hd];
        for r in 0..rows {
            for kv in 0..n_kv {
                for j in 0..hd {
                    kv_in[kv * rows * hd + r * hd + j] = src[r * n_kv * hd + kv * hd + j];
                }
            }
        }
        let rep = repeat_kv(&Tensor::from_vec(kv_in, vec![n_kv, rows, hd]), n_rep);
        assert_eq!(rep.shape(), &[n_head, rows, hd]);
        for hh in 0..n_head {
            for r in 0..rows {
                for j in 0..hd {
                    let got = exp[r * n_head * hd + hh * hd + j];
                    let want = rep.data()[(hh * rows + r) * hd + j];
                    assert!(
                        (got - want).abs() < 1e-6,
                        "第 {hh} 个头第 {r} 行第 {j} 列不一致：{got} vs {want}"
                    );
                }
            }
        }

        // 2) 共轭性：对任意 g 都有 ⟨expand(src), g⟩ == ⟨src, fold(g)⟩
        let g: Vec<f32> = (0..rows * n_head * hd)
            .map(|i| ((i * 53 % 23) as f32 * 0.17).cos())
            .collect();
        let lhs: f32 = exp.iter().zip(&g).map(|(a, b)| a * b).sum();
        let folded = fold_kv_head_grad(&g, rows, n_kv, hd, n_rep);
        assert_eq!(folded.len(), src.len());
        let rhs: f32 = src.iter().zip(&folded).map(|(a, b)| a * b).sum();
        assert!(
            (lhs - rhs).abs() < 1e-4 * lhs.abs().max(1.0),
            "折回不是展开的共轭：{lhs} vs {rhs}"
        );
    }
}
