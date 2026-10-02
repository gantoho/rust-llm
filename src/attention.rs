//! 多头注意力（第 9-10 课）与 KV Cache（第 25 课）
//!
//! 注意力是 Transformer 的核心：让每个 token "关注"序列中其他 token，提取相关性。
//!
//! 本模块包含：
//! - [`KVCache`]：推理时缓存历史 K/V，避免重复计算
//! - [`MultiHeadAttention`]：多头自注意力 + RoPE 位置编码（第 20 课）

use crate::layers::{Linear, RMSNorm};
use crate::model::LN_EPS;
use crate::module::Module;
use crate::quant::{QAxis, QBits, QMatrix};
use crate::rng::Rng;
use crate::rope::RopeSpec;
use crate::tensor::{Buffer, Shared, Tensor};

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
    /// f32 数据挂在 `Shared<..>`（`Arc<Mutex<..>>` 薄封装）上：[`KVCache::k`] / [`KVCache::v`]
    /// 直接共享这个句柄构造零拷贝视图（见那两个方法的文档），历史数据不再每步克隆。
    /// 缓冲本体是 [`Buffer::F32`]（KV 缓存恒为 f32 存储，量化走 `Quant` 变体）。
    F32(Shared<Buffer>),
    Quant(QMatrix),
}

impl KvBlock {
    fn new(bits: Option<QBits>, axis: QAxis) -> Self {
        match bits {
            None => KvBlock::F32(Shared::new(Buffer::F32(Vec::new()))),
            // 列数要等第一次 append 才知道，这里先占位（rows=0 的合法空矩阵）
            Some(b) => KvBlock::Quant(QMatrix::zeros(0, 0, b, axis)),
        }
    }

    /// 还原成 `[rows, d]` 的 f32 行优先数据
    fn to_vec(&self, d: usize) -> Vec<f32> {
        match self {
            KvBlock::F32(v) => v.decode().to_vec(),
            KvBlock::Quant(q) => {
                debug_assert_eq!(q.cols(), d);
                q.dequantize()
            }
        }
    }

    /// 追加 `rows` 行（每行 `d` 个数）
    fn push(&mut self, x: &[f32], rows: usize, d: usize) {
        match self {
            KvBlock::F32(v) => match &mut *v.borrow_mut() {
                Buffer::F32(buf) => buf.extend_from_slice(x),
                Buffer::Bf16(_) => unreachable!("KV 缓存 f32 路径不会中途变 bf16"),
            },
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
            KvBlock::F32(v) => match &mut *v.borrow_mut() {
                Buffer::F32(buf) => {
                    buf.drain(..n * d);
                }
                Buffer::Bf16(_) => unreachable!("KV 缓存 f32 路径不会中途变 bf16"),
            },
            KvBlock::Quant(q) => q.drop_front_rows(n),
        }
    }

    /// 当前占用的字节数（用于打印量化收益）
    fn byte_len(&self) -> usize {
        match self {
            KvBlock::F32(v) => v.borrow().len() * 4,
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
/// **MLA 模式**（[`KVCache::new_latent`]）下缓存里存的是**压缩后的 latent**：
/// 一份 `[T, kv_lora_rank]` 就同时蕴含了 K 和 V 两条路径的信息，
/// 而普通模式下要存两份 `[T, n_kv_head·head_dim]`。
pub struct KVCache {
    k: KvBlock,
    v: KvBlock,
    len: usize,    // 当前**保留**的位置数 T（滑动窗口下不会超过 window）
    seen: usize,   // 累计喂进来的位置总数，只增不减（RoPE 绝对位置基准）
    window: usize, // 保留上限；0 = 不丢弃（缓存只拼不丢）
    sink: usize,   // 永久保留的最前面若干位置
    d: usize,      // 隐藏维 D（MLA 模式下是 kv_lora_rank），第一次 append 时确定
    /// MLA（低秩压缩）模式：缓存里只有一个流（latent），`v` 流保持为空。
    /// 见 [`KVCache::new_latent`]。
    latent: bool,
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
            latent: false,
        }
    }

    /// **MLA 模式**的缓存：只存一份低秩 latent（见 [`crate::attention::MultiHeadAttention`]
    /// 的 MLA 说明），而不是 K、V 两份。
    ///
    /// 长上下文推理的显存瓶颈就是这份缓存：普通 GQA 要存
    /// `2 × 层数 × T × n_kv_head × head_dim × 4` 字节，MLA 只存
    /// `层数 × T × kv_lora_rank × 4` 字节——DeepSeek-V2 里
    /// `kv_lora_rank = 512` 对比 `n_head·head_dim = 128 × 192`，缓存小了约 96 倍。
    /// 代价是**算力换显存**：每次生成都要把整段 latent 升维回 K/V（见 `forward_mla`），
    /// 而不是像普通模式那样只算新 token 的 K/V。
    ///
    /// 量化口径：latent 逐 token 一行、行内同尺度，所以逐行（[`QAxis::Row`]）定标，
    /// 与 V 同理；滑动窗口 / Attention Sink / 推测解码回滚 / Beam 分叉全部照旧可用。
    pub fn new_latent(opts: KvCacheOpts) -> Self {
        let mut c = Self::new(opts);
        c.latent = true;
        c.k = KvBlock::new(opts.bits, QAxis::Row);
        c
    }

    /// 带滑动窗口的缓存：最多保留 `window` 个位置，超出则丢最旧的；`window = 0` 表示不丢弃
    pub fn with_window(window: usize) -> Self {
        Self::new(KvCacheOpts {
            window,
            sink: 0,
            bits: None,
        })
    }

    /// 是否 MLA（低秩压缩）模式
    pub fn is_latent(&self) -> bool {
        self.latent
    }

    /// 深拷贝一份（Beam Search 的每条候选路径都要有自己独立的缓存）。
    ///
    /// 不能靠 `#[derive(Clone)]`：缓存内部是 `Shared<..>`，派生的 clone 会共享
    /// 同一份数据，一条路径的 append 会污染其它路径。这里逐字节复制。
    pub fn fork(&self) -> Self {
        KVCache {
            // F32：取出内容后逐字节复制——若共享句柄，一条路径的 append 会污染其它路径
            k: match &self.k {
                KvBlock::F32(v) => KvBlock::F32(Shared::new(v.borrow().clone())),
                KvBlock::Quant(q) => KvBlock::Quant(q.clone()),
            },
            v: match &self.v {
                KvBlock::F32(v) => KvBlock::F32(Shared::new(v.borrow().clone())),
                KvBlock::Quant(q) => KvBlock::Quant(q.clone()),
            },
            len: self.len,
            seen: self.seen,
            window: self.window,
            sink: self.sink,
            d: self.d,
            latent: self.latent,
        }
    }

    pub fn reset(&mut self) {
        // MLA 模式下 `k` 流存的是 latent，逐 token 定标（与 `new_latent` 一致）
        let axis = if self.latent { QAxis::Row } else { QAxis::Col };
        self.k = KvBlock::new(self.bits(), axis);
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
        assert!(
            !self.latent,
            "MLA 模式的缓存只接收 latent（请用 KVCache::append_latent），它不存 K/V 两份"
        );
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

    /// **MLA 模式**：追加新的 latent（`[1, t, kv_lora_rank]`），返回缓存里**全部** latent
    /// `[1, len, kv_lora_rank]`（f32 路径零拷贝）。
    ///
    /// 与 [`KVCache::append`] 的滑动窗口 / Attention Sink 记账完全一致，只是只有一个流。
    /// 调用方拿全量 latent 后再升维出 K/V——MLA 的就是这么用算力换显存的。
    pub fn append_latent(&mut self, latent: &Tensor) -> Tensor {
        assert!(
            self.latent,
            "当前缓存不是 MLA 模式（构造时用 KVCache::new_latent），收不到 latent"
        );
        assert_eq!(latent.rank(), 3, "latent 必须为 3D [1, T, r]，实际 {:?}", latent.shape());
        assert_eq!(latent.shape()[0], 1, "KV cache 只支持 batch = 1");
        let t = latent.shape()[1];
        let d = latent.shape()[2];
        if self.d == 0 {
            self.d = d;
        } else {
            assert_eq!(self.d, d, "latent 维度不能中途改变");
        }
        self.k.push(&latent.data_ref(), t, d);
        self.len += t;
        self.seen += t;
        if self.window > 0 && self.len > self.window {
            let drop = self.len - self.window;
            self.k.drop_front_from(self.sink, drop, self.d);
            self.len = self.window;
        }
        self.k()
    }

    /// 缓存里每一行对应的**绝对位置**（RoPE 用）。
    ///
    /// 未触发丢弃时就是 `0..seen`；一旦滑动窗口开始丢行，留下的行是
    /// `[0, sink) ∪ [seen - (len - sink), seen)` ——**绝对位置不连续**。
    ///
    /// 普通模式用不到它（K 在写进缓存之前就已经旋转好了，缓存里存的是"已旋转的 K"）；
    /// MLA 模式必须用：缓存里存的是**未旋转的 latent**，每次推理都要按每行各自的
    /// 绝对位置重新旋转升维出来的 K，否则滑动窗口下的相对距离会整体错位。
    pub fn positions(&self) -> Vec<usize> {
        if self.len >= self.seen {
            return (0..self.seen).collect();
        }
        let mut p: Vec<usize> = (0..self.sink).collect();
        p.extend(self.seen - (self.len - self.sink)..self.seen);
        p
    }

    /// 返回完整缓存张量 [1, T, D]。
    ///
    /// **f32 路径零拷贝**：直接共享缓存底层的 `Shared` 句柄（张量是只读叶子：
    /// req = false、不挂 backward），解码每步省掉整段历史 O(T·D) 的克隆。
    /// 缓存的变更（append / rollback / 窗口裁剪）只发生在两次前向之间，
    /// 不会与正在读它的注意力前向重叠。
    ///
    /// 量化路径仍需反量化出 f32——「量化布局直接打分」要动 flash 内核、
    /// 反向重算与 GPU 常驻路径三处，推迟到 strides 视图重构批（批次 10）。
    pub fn k(&self) -> Tensor {
        match &self.k {
            KvBlock::F32(v) => Tensor::shared(v.clone(), vec![1, self.len, self.d]),
            KvBlock::Quant(_) => Tensor::from_vec(self.k.to_vec(self.d), vec![1, self.len, self.d]),
        }
    }

    pub fn v(&self) -> Tensor {
        match &self.v {
            KvBlock::F32(v) => Tensor::shared(v.clone(), vec![1, self.len, self.d]),
            KvBlock::Quant(_) => Tensor::from_vec(self.v.to_vec(self.d), vec![1, self.len, self.d]),
        }
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
        // MLA 模式只有 latent 一个流（`v` 是空的，回滚它会让长度下溢）
        if !self.latent {
            self.v.drop_back(n, self.d);
        }
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
                // 单次借用同一把锁（Mutex 不可重入）：先借出来再取长度与截断
                let mut g = v.borrow_mut();
                match &mut *g {
                    Buffer::F32(buf) => {
                        let len = buf.len();
                        buf.truncate(len - n * d);
                    }
                    Buffer::Bf16(_) => unreachable!("KV 缓存 f32 路径不会中途变 bf16"),
                }
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
            KvBlock::F32(v) => *v.borrow_mut() = Buffer::F32(kept),
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
///
/// # MLA（Multi-head Latent Attention，DeepSeek-V2/V3）
///
/// `kv_lora_rank > 0` 时切换到 MLA：**K 和 V 不再各自从输入投影，而是先压到一份
/// 低秩 latent，再从 latent 升维出来**：
///
/// ```text
///     GQA:   x ──c_k──> K ┐                       缓存：K、V 两份
///            x ──c_v──> V ┘
///
///     MLA:   x ──c_kv──> c (r 维)  ─┬─c_k──> K   缓存：**只有 c**
///                                   └─c_v──> V
/// ```
///
/// 为什么省显存：推理时的缓存是 `2 × 层数 × T × n_kv_head × head_dim × 4` 字节，
/// 长上下文下它比权重还大。MLA 只缓存 `层数 × T × r × 4`，而 `r` 可以远小于
/// `n_kv_head × head_dim`（DeepSeek-V2：r = 512，对比 128 头 × 192 维）。
/// 代价写在明处：**每次推理都要把整段 latent 升维回 K/V**（用算力换显存），
/// 而不是只算新 token 的 K/V。
///
/// 本实现的取舍（教学版，与论文的差异写在这里而不是藏起来）：
/// - 保留论文的**核心**：KV 联合低秩压缩 + 缓存只存 latent。
/// - 省略论文的 **decoupled RoPE**（把 Q/K 的 rope 子维单独拎出来、只对那部分旋转，
///   让压缩后的 latent 不必带位置信息）。因此这里的 RoPE 作用在完整的 head_dim 上，
///   缓存里的 latent 是**未旋转**的，靠 [`KVCache::positions`] 每步按各自绝对位置补旋。
///   效果上的差别：压缩率略低、每步多一次旋转（`T` 行的 O(T·d) 计算，与升维同量级）。
pub struct MultiHeadAttention {
    pub c_q: Linear,
    /// K 的投影。普通模式下是 `d -> n_kv_head·head_dim`；MLA 模式下是
    /// `kv_lora_rank -> n_kv_head·head_dim`（"升维"那一半）。
    pub c_k: Linear,
    /// V 的投影，输入维口径同 [`MultiHeadAttention::c_k`]。
    pub c_v: Linear,
    pub c_proj: Linear,
    /// MLA 的**压缩投影** `d -> kv_lora_rank`；`None` = 不走 MLA（`kv_lora_rank = 0`）。
    pub c_kv: Option<Linear>,
    /// MLA 的低秩维 `r`；0 = 关闭（普通 MHA / GQA）。
    pub kv_lora_rank: usize,
    pub n_head: usize,
    pub n_kv_head: usize,
    /// **QK-Norm**（Query-Key Normalization）：`Some` 时在 Q/K 投影之后、RoPE 之前，
    /// 对**每个头**的 `head_dim` 向量做一次 RMSNorm。`None` = 关闭（默认，逐位不变）。
    ///
    /// 为什么有用：注意力打分 `q·k/√d` 的幅度不受约束，训练中 Q/K 的尺度会一起漂移，
    /// 大 logit 把 softmax 推到饱和区（梯度趋零），这是长训练里突然发散的最常见原因之一。
    /// 在 Q/K 上各加一层归一化把尺度钉死，Gemma 2 / Chameleon / ViT-22B 都靠它换来了
    /// 更稳的训练（可以用更大的学习率、更少的 warmup）。注意它归一化的是 **head_dim
    /// 方向**（每个头内部），不是隐藏维——所以 `gamma` 长度是 `head_dim` 而不是 `n_embd`。
    pub q_norm: Option<RMSNorm>,
    /// K 侧的 QK-Norm（GQA 下 `gamma` 长度仍是 `head_dim`，各 KV 头共用一套参数）。
    pub k_norm: Option<RMSNorm>,
    /// RoPE 的频率参数（底数 + 长度外推方式）。**结构的一部分**：训练与推理、
    /// checkpoint 加载与续训必须一致，否则同一段文本会被旋转到不同角度。
    pub rope: RopeSpec,
}

impl MultiHeadAttention {
    /// `kv_lora_rank = 0`：普通 MHA / GQA（行为与加 MLA 之前**逐位一致**，包括初始化
    /// 消耗的随机数顺序）；`> 0`：走 MLA，见结构体文档。
    ///
    /// `qk_norm = false` 同样逐位不变：RMSNorm 的 `gamma` 初始化为全 1，不消耗随机数，
    /// 所以老配置（`config.json` 里没有这个字段、serde 取默认值 `false`）训出来的权重
    /// 与加了 QK-Norm 之前的代码完全一致。
    pub fn new(
        n_embd: usize,
        n_head: usize,
        n_kv_head: usize,
        kv_lora_rank: usize,
        qk_norm: bool,
        rope: RopeSpec,
        rng: &mut Rng,
    ) -> Self {
        let n_kv = if n_kv_head == 0 { n_head } else { n_kv_head };
        assert!(n_head % n_kv == 0, "n_head 必须能被 n_kv_head 整除");
        let head_dim = n_embd / n_head;
        let kv_dim = n_kv * head_dim;
        assert!(
            kv_lora_rank < kv_dim,
            "MLA 的低秩维（{kv_lora_rank}）必须小于 K/V 的完整维度（{kv_dim} = n_kv_head × head_dim），否则谈不上压缩"
        );
        // 先按**普通模式**的随机数顺序建 c_q/c_k/c_v/c_proj，再建 MLA 的压缩投影：
        // `kv_lora_rank = 0` 时不消耗任何额外随机数，老配置的初始化逐位不变。
        let c_q = Linear::new(n_embd, n_embd, rng);
        let (c_k, c_v) = if kv_lora_rank > 0 {
            (
                Linear::new(kv_lora_rank, kv_dim, rng),
                Linear::new(kv_lora_rank, kv_dim, rng),
            )
        } else {
            (
                Linear::new(n_embd, kv_dim, rng),
                Linear::new(n_embd, kv_dim, rng),
            )
        };
        let c_proj = Linear::new(n_embd, n_embd, rng);
        let c_kv = (kv_lora_rank > 0).then(|| Linear::new(n_embd, kv_lora_rank, rng));
        // QK-Norm 放在所有投影之后构造：它不消耗随机数，但保持"老路径的随机数顺序在前"这条规矩
        let (q_norm, k_norm) = if qk_norm {
            (
                Some(RMSNorm::new(head_dim, LN_EPS)),
                Some(RMSNorm::new(head_dim, LN_EPS)),
            )
        } else {
            (None, None)
        };
        MultiHeadAttention {
            c_q,
            c_k,
            c_v,
            c_proj,
            c_kv,
            kv_lora_rank,
            n_head,
            n_kv_head: n_kv,
            q_norm,
            k_norm,
            rope,
        }
    }

    /// 前向
    /// - x: [B, T, D]
    /// - mask: 可选的 `[T, T_total]` 后缀因果掩码；CPU 分块核在核内屏蔽不需要它，
    ///   传 `None` 即可；只有 GPU 常驻 / probe 录制路径会消费真实掩码 buffer
    /// - kv_cache: Some(缓存) 时走推理模式（只算新 token）
    /// - base: RoPE 的绝对位置基准（训练时 = 0，KV cache 推理时 = 缓存已见位置总数）
    pub fn forward(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        kv_cache: Option<&mut KVCache>,
        base: usize,
    ) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let head_dim = d / self.n_head;
        assert_eq!(head_dim * self.n_head, d, "n_embd 必须能被 n_head 整除");

        // MLA 走另一条前向：K/V 由 latent 升维而来，缓存里存的是 latent
        if self.c_kv.is_some() {
            return self.forward_mla(x, mask, kv_cache, base);
        }

        // 1. 投影得到 Q、K、V
        let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
        let kv_dim = self.n_kv_head * head_dim;
        let k = self.c_k.forward(x).reshape(vec![b, t, kv_dim]); // [B, T, kv_dim]
        let v = self.c_v.forward(x).reshape(vec![b, t, kv_dim]);

        // 1.5 QK-Norm（若开启）：投影之后、RoPE 之前，对每个头做 RMSNorm。
        //     放在 RoPE 前是因为旋转是正交变换、不改变向量范数，先归一化即"旋转前钉死尺度"；
        //     放在缓存之前，于是缓存里存的仍是归一化+旋转后的 K，历史复用不受影响。
        let q = self.apply_qk_norm(q, &self.q_norm, self.n_head);
        let k = self.apply_qk_norm(k, &self.k_norm, self.n_kv_head);

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

        // 3. KV cache（缓存里存的是**已旋转的** K，历史直接复用，新 token 只算新块）
        let (k, v) = match kv_cache {
            Some(cache) => {
                cache.append(&k, &v);
                (cache.k(), cache.v())
            }
            None => (k, v),
        };

        // 4-8. 拆头 → flash attention → 合并头 → 输出投影
        self.attend(&q, &k, &v, mask, b, t)
    }

    /// MLA 前向：`x → latent →（缓存）→ K/V → 注意力`。
    ///
    /// 与普通前向的三处不同：
    /// 1. K/V 不是从 `x` 直接投影，而是从**压缩 latent** 升维（`c_kv` → `c_k`/`c_v`）；
    /// 2. 缓存里存的是 latent（[`KVCache::append_latent`]），每次要**全量**升维回 K/V；
    /// 3. latent 是**未旋转**的，所以 K 必须按缓存里每一行各自的绝对位置补旋
    ///    （[`KVCache::positions`]；滑动窗口丢行后这些位置并不连续）。
    fn forward_mla(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        mut kv_cache: Option<&mut KVCache>,
        base: usize,
    ) -> Tensor {
        let (b, t, d) = (x.shape()[0], x.shape()[1], x.shape()[2]);
        let head_dim = d / self.n_head;
        let kv_dim = self.n_kv_head * head_dim;
        let c_kv = self.c_kv.as_ref().expect("forward_mla 只应在 MLA 模式下调用");

        // 1. Q 投影 + latent 压缩投影
        let q = self.c_q.forward(x).reshape(vec![b, t, d]);
        let latent = c_kv.forward(x).reshape(vec![b, t, self.kv_lora_rank]);

        // 2. latent 进缓存（MLA 省显存的地方就在这一步：只存这一份）
        let (latent, k_pos) = match kv_cache.as_deref_mut() {
            Some(cache) => {
                assert!(
                    cache.is_latent(),
                    "MLA 模型必须配 MLA 模式的缓存（Transformer::new_kv_cache 会按配置自动选）"
                );
                let all = cache.append_latent(&latent);
                let pos = cache.positions();
                (all, pos)
            }
            None => {
                let pos: Vec<usize> = (0..b).flat_map(|_| base..base + t).collect();
                (latent, pos)
            }
        };
        let tk = latent.shape()[1];

        // 3. 全量升维回 K/V（用算力换显存）
        let k = self.c_k.forward(&latent).reshape(vec![b, tk, kv_dim]);
        let v = self.c_v.forward(&latent).reshape(vec![b, tk, kv_dim]);

        // 3.5 QK-Norm（若开启）：口径与普通路径一致——升维出来的 K 按头归一化后再旋转。
        //     注意缓存里存的是**未归一化**的 latent，归一化每步在升维之后重做，
        //     因此"全量重算"与"latent 缓存增量"两条路的结果仍然一致。
        let q = self.apply_qk_norm(q, &self.q_norm, self.n_head);
        let k = self.apply_qk_norm(k, &self.k_norm, self.n_kv_head);

        // 4. RoPE：Q 用本次新 token 的绝对位置；K 用它自己每一行的绝对位置
        let q_pos: Vec<usize> = (0..b).flat_map(|_| base..base + t).collect();
        let q = q.reshape(vec![b * t, d]).rotary(&q_pos, &self.rope).reshape(vec![b, t, d]);
        let k = k
            .reshape(vec![b * tk, kv_dim])
            .rotary(&k_pos, &self.rope)
            .reshape(vec![b, tk, kv_dim]);

        self.attend(&q, &k, &v, mask, b, t)
    }

    /// QK-Norm 的具体动作：把 `[…, n_head·head_dim]` 摊成 `[…·n_head, head_dim]`，
    /// 逐头做 RMSNorm，再**还原成原来的形状**。`norm = None` 时原样返回
    /// （**不产生任何算子**，默认配置下这条路径的开销是一次所有权转移）。
    ///
    /// GQA 下 K 传 `n_kv_head`，每行仍是一个头的 `head_dim` 向量，逐头归一化口径不变；
    /// 所有 KV 头共用同一套 `gamma`（与 Gemma 2 一致——它也是每个注意力层一套 QK 归一化）。
    fn apply_qk_norm(&self, x: Tensor, norm: &Option<RMSNorm>, n_head: usize) -> Tensor {
        match norm {
            None => x,
            Some(n) => {
                let shape = x.shape().to_vec();
                let rows: usize = shape[..shape.len() - 1].iter().product();
                let dim = shape[shape.len() - 1];
                let head_dim = dim / n_head;
                x.reshape(vec![rows * n_head, head_dim])
                    .rmsnorm(&n.gamma, n.eps)
                    .reshape(shape)
            }
        }
    }

    /// 注意力核心（两条前向路径共用）：
    /// 拆头 → Flash Attention → 合并头 → 输出投影。
    ///
    /// GQA 不再物化 repeat_kv：flash_attention 的 CPU 分块核按 Q 头核内索引共享 KV 头，
    /// GPU 路径在 `Tensor::flash_attention` 内核外展开（见 `crate::attention::expand_kv_head`）。
    /// 缓存路径下 `k`/`v` 的行数是 `T_total`（≥ 本次的 `t`），据此推出序列长度。
    fn attend(&self, q: &Tensor, k: &Tensor, v: &Tensor, mask: Option<&Tensor>, b: usize, t: usize) -> Tensor {
        let d = q.shape()[2];
        let head_dim = d / self.n_head;
        let t_total = k.shape()[1];
        let split = |x: &Tensor, n_head: usize, rows: usize| {
            x.reshape(vec![b, rows, n_head, head_dim])
                .permute(&[0, 2, 1, 3])
                .reshape(vec![b * n_head, rows, head_dim])
        };
        // 1. 拆头
        let q = split(q, self.n_head, t);
        let k = split(k, self.n_kv_head, t_total);
        let v = split(v, self.n_kv_head, t_total);
        // 2. Flash Attention 融合算子：CPU 走分块在线 softmax（不物化掩码、不落地 P，
        //    `block_size` 真实生效）；GPU 常驻/probe 路径才消费 mask
        let out = Tensor::flash_attention(&q, &k, &v, mask, 32);
        // 3. 合并头回 [B, T, D]
        let out = out
            .reshape(vec![b, self.n_head, t, head_dim])
            .permute(&[0, 2, 1, 3])
            .reshape(vec![b, t, d]);
        // 4. 输出投影
        self.c_proj.forward(&out)
    }

    /// 按 `targets` 给 Q/K/V/输出投影挂上 LoRA 适配器。
    ///
    /// 缺省只挂 Q/K/V：注意力里"该去看哪里"（Q/K）和"看到了取什么"（V）最需要随下游任务
    /// 调整，是 LoRA 论文与社区实践里性价比最高的一组。输出投影 `c_proj` 只做一次线性汇总，
    /// 加适配器收益最小而参数与 Q 一样多，所以默认关闭（`--lora-targets` 可打开）。
    ///
    /// 主干冻结不在这里做，由 [`crate::model::Transformer::apply_lora`] 统一处理。
    /// MLA 的压缩投影 `c_kv` **不挂适配器**：它同时供给 K 和 V 两条路，
    /// 低秩增量在那里既破坏"压缩"的口径（增量本身可能与主干同维），
    /// 也没有现成的社区做法；要调 MLA 的 KV 表示，直接调 `kv_lora_rank` 重训更干净。
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

    /// 本层的全部投影：`c_q / c_k / c_v / c_proj`，MLA 模式下再加压缩投影 `c_kv`。
    /// 量化、反量化、冻结这些"扫一遍所有权重"的操作都走它，避免每加一个投影就要改三处。
    fn linears(&self) -> Vec<&Linear> {
        let mut v = vec![&self.c_q, &self.c_k, &self.c_v, &self.c_proj];
        if let Some(c) = &self.c_kv {
            v.push(c);
        }
        v
    }

    fn linears_mut(&mut self) -> Vec<&mut Linear> {
        let mut v = vec![&mut self.c_q, &mut self.c_k, &mut self.c_v, &mut self.c_proj];
        if let Some(c) = self.c_kv.as_mut() {
            v.push(c);
        }
        v
    }

    /// 把本层各投影的适配器合并进主干（推理用，见 [`Linear::merge_lora`]）
    pub fn merge_lora(&mut self) {
        for lin in self.linears_mut() {
            lin.merge_lora();
        }
    }

    /// 本层是否有任一投影挂了适配器
    /// （GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_lora(&self) -> bool {
        self.linears().iter().any(|l| l.lora.is_some())
    }

    /// 本层是否有任一投影被量化（GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_quant(&self) -> bool {
        self.linears().iter().any(|l| l.has_quant())
    }

    /// 把各投影的量化状态烘焙回 f32（checkpoint 里写的始终是 f32 权重）
    pub fn dequantize_weights(&mut self) {
        for lin in self.linears_mut() {
            lin.dequantize_weight();
        }
    }

    /// 带名字的参数（checkpoint 用）：`{prefix}.c_q/c_k/c_v/c_proj.*`
    /// （挂了 LoRA 时各投影下还有 `.lora_a` / `.lora_b`，由 [`Linear::named_parameters`] 递归带出）
    ///
    /// QK-Norm 开启时额外带 `{prefix}.q_norm.gamma` / `{prefix}.k_norm.gamma`；
    /// 关闭时这两个名字不出现，于是**老 checkpoint 与老配置仍然逐位兼容**。
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps: Vec<(String, Tensor)> = self
            .named_linears(prefix)
            .into_iter()
            .flat_map(|(p, lin)| lin.named_parameters(&p))
            .collect();
        if let Some(n) = &self.q_norm {
            ps.extend(n.named_parameters(&format!("{prefix}.q_norm")));
        }
        if let Some(n) = &self.k_norm {
            ps.extend(n.named_parameters(&format!("{prefix}.k_norm")));
        }
        ps
    }

    /// 各投影 + 各自的参数名前缀（MLA 模式下 `c_kv` 排在 `c_q` 之后）。
    /// `named_parameters` 由它派生，名字因此只有一处定义——
    /// 量化要按同一套名字取校准统计，见 [`crate::quant::CalibStats`]。
    pub fn named_linears(&self, prefix: &str) -> Vec<(String, &Linear)> {
        let mut v = Vec::with_capacity(5);
        v.push((format!("{prefix}.c_q"), &self.c_q));
        if let Some(c) = &self.c_kv {
            v.push((format!("{prefix}.c_kv"), c));
        }
        v.push((format!("{prefix}.c_k"), &self.c_k));
        v.push((format!("{prefix}.c_v"), &self.c_v));
        v.push((format!("{prefix}.c_proj"), &self.c_proj));
        v
    }

    pub fn named_linears_mut(&mut self, prefix: &str) -> Vec<(String, &mut Linear)> {
        let mut v = Vec::with_capacity(5);
        v.push((format!("{prefix}.c_q"), &mut self.c_q));
        if let Some(c) = self.c_kv.as_mut() {
            v.push((format!("{prefix}.c_kv"), c));
        }
        v.push((format!("{prefix}.c_k"), &mut self.c_k));
        v.push((format!("{prefix}.c_v"), &mut self.c_v));
        v.push((format!("{prefix}.c_proj"), &mut self.c_proj));
        v
    }
}

impl Module for MultiHeadAttention {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.c_q.parameters();
        if let Some(c) = &self.c_kv {
            ps.extend(c.parameters());
        }
        ps.extend(self.c_k.parameters());
        ps.extend(self.c_v.parameters());
        ps.extend(self.c_proj.parameters());
        if let Some(n) = &self.q_norm {
            ps.extend(n.parameters());
        }
        if let Some(n) = &self.k_norm {
            ps.extend(n.parameters());
        }
        ps
    }
}

/// GQA 辅助函数（权重空间版）：把 K/V 投影的**参数**按 [`crate::tensor::repeat_flat`] 的同一套头顺序展开。
///
/// `src` 是 `[rows, n_kv_head * head_dim]` 的行主序数据（权重取 `rows = d`，
/// 偏置取 `rows = 1`），返回 `[rows, n_head * head_dim]`：
/// 输出第 `hh` 个头直接复制自源头 `hh / n_rep` —— 与 [`crate::tensor::repeat_flat`] 把
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
/// 对应 `crate::tensor::fold_repeat_grad` 反向「把 n_rep 个副本的梯度求和回原始头」。
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

    /// GQA 的**权重空间展开**必须与张量空间的 [`crate::tensor::repeat_flat`] 同序，且折回是展开的共轭。
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

        // 1) 头顺序：`repeat_flat` 吃 `[B*n_kv, T, hd]` 展平数据（令 B=1、T=rows），
        //    把第 b 个 KV 头复制成 n_rep 份；展开结果的第 hh 个头应取自源头的 `hh / n_rep`。
        //    两者的数据布局不同（源是「行内多头连续」，repeat_flat 是「头在外、行长在内」），
        //    这里把源转置成 repeat_flat 认的布局再比。
        let mut kv_in = vec![0.0f32; n_kv * rows * hd];
        for r in 0..rows {
            for kv in 0..n_kv {
                for j in 0..hd {
                    kv_in[kv * rows * hd + r * hd + j] = src[r * n_kv * hd + kv * hd + j];
                }
            }
        }
        let rep = crate::tensor::repeat_flat(&kv_in, n_kv, n_rep);
        assert_eq!(rep.len(), n_head * rows * hd);
        for hh in 0..n_head {
            for r in 0..rows {
                for j in 0..hd {
                    let got = exp[r * n_head * hd + hh * hd + j];
                    let want = rep[(hh * rows + r) * hd + j];
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

    // ==================== MLA（低秩压缩 KV cache） ====================

    /// 测试用的普通 RoPE 参数
    fn spec() -> RopeSpec {
        RopeSpec::new(crate::rope::ROPE_BASE, crate::rope::RopeScaling::None, 64)
    }

    /// 造一个 MLA 注意力 + 一段输入（`[1, t, d]` 展平数据，确定性伪随机）
    fn mla_fixture(
        d: usize,
        n_head: usize,
        n_kv: usize,
        r: usize,
        t: usize,
        seed: u64,
    ) -> (MultiHeadAttention, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let attn = MultiHeadAttention::new(d, n_head, n_kv, r, false, spec(), &mut rng);
        let data: Vec<f32> = (0..t * d).map(|i| ((i * 7 % 23) as f32 * 0.31).sin()).collect();
        (attn, data)
    }

    /// MLA：缓存里只存**一份 latent**，显存随低秩维 r 走，而不是随 K/V 的完整维度走。
    #[test]
    fn test_mla_cache_stores_only_latent_and_saves_bytes() {
        let (d, n_head, n_kv, r) = (32usize, 4usize, 2usize, 4usize);
        let (attn, data) = mla_fixture(d, n_head, n_kv, r, 8, 7);
        let hd = d / n_head;
        let kv_dim = n_kv * hd;
        // 结构：压缩投影 d -> r，升维投影 r -> kv_dim（K、V 各一个）
        assert_eq!(attn.kv_lora_rank, r);
        assert_eq!(attn.c_kv.as_ref().unwrap().dims(), (d, r));
        assert_eq!(attn.c_k.dims(), (r, kv_dim));
        assert_eq!(attn.c_v.dims(), (r, kv_dim));

        let t = 8;
        let x = Tensor::from_vec(data, vec![1, t, d]);
        let mut cache = KVCache::new_latent(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: None,
        });
        let _ = attn.forward(&x, None, Some(&mut cache), 0);
        assert!(cache.is_latent());
        assert_eq!(cache.seq_len(), t);
        assert_eq!(cache.byte_len(), t * r * 4, "MLA 缓存只该占 T×r 个 f32");

        // 同样长度的普通缓存要存 K、V 两份（每份 kv_dim）
        let mut plain = KVCache::new(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: None,
        });
        let kv = Tensor::from_vec(vec![0.0f32; t * kv_dim], vec![1, t, kv_dim]);
        plain.append(&kv, &kv);
        assert_eq!(plain.byte_len(), 2 * t * kv_dim * 4);
        assert!(
            cache.byte_len() * 4 <= plain.byte_len(),
            "r={r} 对 kv_dim={kv_dim}：MLA 应省下 4 倍以上（{} vs {} 字节）",
            cache.byte_len(),
            plain.byte_len()
        );
        // 参数上也省：c_k/c_v 的输入维从 d 降到 r
        assert!(attn.c_k.dims().0 < d);
    }

    /// MLA 增量推理（latent 缓存）与全量重算必须给出同一个结果。
    ///
    /// 这条测试盯的是最容易被写错的地方：缓存里存的是**未旋转**的 latent，
    /// 所以升维出来的 K 必须按缓存里每一行各自的绝对位置补旋（[`KVCache::positions`]）。
    #[test]
    fn test_mla_latent_cache_matches_full_recompute() {
        let (d, n_head, n_kv, r) = (24usize, 3usize, 3usize, 5usize);
        let t = 6;
        let (attn, rows) = mla_fixture(d, n_head, n_kv, r, t, 13);
        let x = Tensor::from_vec(rows.clone(), vec![1, t, d]);
        let full = attn.forward(&x, None, None, 0).data();

        let mut cache = KVCache::new_latent(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: None,
        });
        let mut inc = Vec::new();
        for i in 0..t {
            let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
            let o = attn.forward(&xi, None, Some(&mut cache), i);
            inc.extend_from_slice(&o.data());
        }
        for (i, (a, b)) in full.iter().zip(&inc).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * (1.0 + a.abs()),
                "第 {i} 个元素不一致：全量 {a} vs 增量 {b}（缓存里的 latent 补旋口径错了？）"
            );
        }
    }

    /// MLA + 滑动窗口：留下的是最近 `window` 行，`positions()` 给出的绝对位置必须让
    /// 增量推理**等价于只喂这几行的全量前向**（相对距离一致）。
    #[test]
    fn test_mla_cache_window_matches_subsequence() {
        let (d, n_head, n_kv, r) = (16usize, 2usize, 1usize, 4usize);
        let t = 6;
        let window = 3;
        let (attn, rows) = mla_fixture(d, n_head, n_kv, r, t, 23);

        let mut cache = KVCache::new_latent(KvCacheOpts {
            window,
            sink: 0,
            bits: None,
        });
        let mut last = Vec::new();
        for i in 0..t {
            let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
            last = attn.forward(&xi, None, Some(&mut cache), i).data();
        }
        assert_eq!(cache.seq_len(), window);
        assert_eq!(cache.positions_seen(), t);
        assert_eq!(cache.positions(), vec![3, 4, 5], "留下的正是位置 3/4/5");

        // 全量重算：只喂第 3..6 个 token、绝对位置从 3 开始
        let sub = Tensor::from_vec(rows[3 * d..].to_vec(), vec![1, window, d]);
        let full = attn.forward(&sub, None, None, 3).data();
        for (i, (a, b)) in last.iter().zip(&full[2 * d..]).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * (1.0 + a.abs()),
                "第 {i} 个元素不一致：窗口推理 {a} vs 子序列全量 {b}"
            );
        }
    }

    /// MLA + Attention Sink：被丢的是**中间**的行，最前面 `sink` 行留下，
    /// `positions()` 因此必须输出**不连续**的绝对位置；留下的 latent 行就是那几行的压缩结果。
    #[test]
    fn test_mla_cache_sink_positions_and_rows() {
        let (d, n_head, n_kv, r) = (16usize, 2usize, 2usize, 6usize);
        let t = 6;
        let (attn, rows) = mla_fixture(d, n_head, n_kv, r, t, 29);
        let mut cache = KVCache::new_latent(KvCacheOpts {
            window: 3,
            sink: 1,
            bits: None,
        });
        let per_token: Vec<Vec<f32>> = (0..t)
            .map(|i| {
                let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
                attn.c_kv.as_ref().unwrap().forward(&xi).data()
            })
            .collect();
        for i in 0..t {
            let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
            let _ = attn.forward(&xi, None, Some(&mut cache), i);
        }
        assert_eq!(cache.seq_len(), 3);
        assert_eq!(cache.positions(), vec![0, 4, 5], "0 是注意力汇，另两个名额给最近的行");
        // 缓存里的三行 = 位置 0/4/5 各自的 latent（latent 逐 token 独立，不跨位置混合）
        let got = cache.k().data();
        for (k, src) in [0usize, 4, 5].iter().enumerate() {
            for j in 0..r {
                assert!(
                    (got[k * r + j] - per_token[*src][j]).abs() < 1e-6,
                    "第 {k} 行第 {j} 列应为位置 {src} 的 latent"
                );
            }
        }
    }

    /// 量化 + MLA：缓存压到 int8/int4 后读回的 latent 要贴着 f32 版本（逐 token 定标）。
    #[test]
    fn test_mla_quantized_cache_matches_f32() {
        let (d, n_head, n_kv, r) = (16usize, 2usize, 2usize, 8usize);
        let t = 6;
        let (attn, rows) = mla_fixture(d, n_head, n_kv, r, t, 31);
        let mut plain = KVCache::new_latent(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: None,
        });
        for i in 0..t {
            let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
            let _ = attn.forward(&xi, None, Some(&mut plain), i);
        }
        for (bits, tol) in [(QBits::Int8, 0.02f32), (QBits::Int4, 0.15f32)] {
            let mut q = KVCache::new_latent(KvCacheOpts {
                window: 0,
                sink: 0,
                bits: Some(bits),
            });
            for i in 0..t {
                let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
                let _ = attn.forward(&xi, None, Some(&mut q), i);
            }
            assert_eq!(q.bits(), Some(bits));
            assert!(q.byte_len() < plain.byte_len(), "{bits:?} 应更省字节");
            let err = plain
                .k()
                .data()
                .iter()
                .zip(q.k().data())
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            assert!(err <= tol, "{bits:?} 的 latent 最大误差 {err} 超过 {tol}");
        }
    }

    /// 训练侧：梯度必须穿过 latent —— 压缩投影 `c_kv` 拿得到梯度，
    /// 否则"低秩压缩"就只是个前向技巧，根本训不起来。
    #[test]
    fn test_mla_gradients_reach_compression_projection() {
        let (attn, data) = mla_fixture(16, 2, 1, 4, 3, 37);
        let x = Tensor::from_vec(data, vec![1, 3, 16]);
        let loss = attn.forward(&x, None, None, 0).sum();
        loss.backward();
        let peak = |t: &Tensor| t.grad().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak(&attn.c_kv.as_ref().unwrap().weight) > 0.0, "压缩投影 c_kv 没拿到梯度");
        assert!(peak(&attn.c_k.weight) > 0.0, "升维投影 c_k 没拿到梯度");
        assert!(peak(&attn.c_v.weight) > 0.0, "升维投影 c_v 没拿到梯度");
        assert!(peak(&attn.c_q.weight) > 0.0, "Q 投影没拿到梯度");
    }

    /// `kv_lora_rank = 0` 时必须是**普通 MHA/GQA**：没有压缩投影、参数形状与老代码一致，
    /// 且 MLA 的局部缓存/位置接口不会被误用。
    #[test]
    fn test_mla_disabled_keeps_plain_gqa_shape() {
        let mut rng = Rng::new(41);
        let attn = MultiHeadAttention::new(32, 4, 2, 0, false, spec(), &mut rng);
        assert!(attn.c_kv.is_none());
        assert_eq!(attn.kv_lora_rank, 0);
        assert_eq!(attn.c_k.dims(), (32, 16));
        assert_eq!(attn.c_v.dims(), (32, 16));
        assert_eq!(attn.named_linears("a").len(), 4);
    }

    // ==================== QK-Norm（Q/K 逐头归一化） ====================

    /// 造一个开了 QK-Norm 的普通注意力 + 一段输入（`[1, t, d]` 展平数据）
    fn qk_fixture(d: usize, n_head: usize, n_kv: usize, t: usize, seed: u64) -> (MultiHeadAttention, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let attn = MultiHeadAttention::new(d, n_head, n_kv, 0, true, spec(), &mut rng);
        let data: Vec<f32> = (0..t * d).map(|i| ((i * 11 % 29) as f32 * 0.43).sin()).collect();
        (attn, data)
    }

    /// QK-Norm 的核心数学：把 `[B, T, n_head·head_dim]` 摊成逐头的 `head_dim` 向量后，
    /// 每个头的向量都落到**单位 RMS**（`gamma = 1` 时）；且对任意大的输入尺度都成立
    /// （这正是它抑制大 logit 的原理）。关闭时（`None`）必须逐位原样返回。
    #[test]
    fn test_qk_norm_makes_each_head_unit_rms_and_is_scale_invariant() {
        let (d, n_head, hd) = (16usize, 4usize, 4usize);
        let mut rng = Rng::new(5);
        let attn = MultiHeadAttention::new(d, n_head, n_head, 0, true, spec(), &mut rng);
        assert_eq!(attn.q_norm.as_ref().unwrap().gamma.shape(), vec![hd]);

        // 输入放大 1000 倍：RMSNorm 对正尺度不变，归一化后的范数不该变
        for scale in [1.0f32, 1e3] {
            let data: Vec<f32> = (0..2 * d).map(|i| ((i as f32 * 0.7).sin()) * scale).collect();
            let x = Tensor::from_vec(data, vec![1, 2, d]);
            let out = attn.apply_qk_norm(x, &attn.q_norm, n_head);
            assert_eq!(out.shape(), vec![1, 2, d]);
            for head in 0..2 * n_head {
                let v = &out.data()[head * hd..(head + 1) * hd];
                let rms = (v.iter().map(|a| a * a).sum::<f32>() / hd as f32).sqrt();
                assert!(
                    (rms - 1.0).abs() < 1e-4,
                    "scale={scale}：第 {head} 个头的 RMS 应为 1，实际 {rms}"
                );
            }
        }

        // 关闭 QK-Norm：原样返回，逐位相同
        let data: Vec<f32> = (0..2 * d).map(|i| (i as f32 * 0.31).cos()).collect();
        let x = Tensor::from_vec(data, vec![1, 2, d]);
        let same = attn.apply_qk_norm(x.clone(), &None, n_head);
        assert_eq!(same.data(), x.data(), "关闭 QK-Norm 时必须逐位原样返回");
    }

    /// 开启 QK-Norm **不能扰动老权重的初始化**：它不消耗随机数，所以同 seed 下
    /// 四个投影的权重与偏置必须与关闭时**逐位相同**；多出来的只有每层两个 `gamma`。
    #[test]
    fn test_qk_norm_does_not_change_weight_initialization() {
        let (d, n_head, n_kv) = (32usize, 4usize, 2usize);
        let mut r1 = Rng::new(17);
        let plain = MultiHeadAttention::new(d, n_head, n_kv, 0, false, spec(), &mut r1);
        let mut r2 = Rng::new(17);
        let qk = MultiHeadAttention::new(d, n_head, n_kv, 0, true, spec(), &mut r2);

        assert_eq!(plain.c_q.weight.data(), qk.c_q.weight.data());
        assert_eq!(plain.c_k.weight.data(), qk.c_k.weight.data());
        assert_eq!(plain.c_v.weight.data(), qk.c_v.weight.data());
        assert_eq!(plain.c_proj.weight.data(), qk.c_proj.weight.data());
        assert_eq!(plain.c_proj.bias.data(), qk.c_proj.bias.data());

        assert!(plain.q_norm.is_none() && plain.k_norm.is_none());
        // 4 个 Linear × (weight + bias) = 8；开启后每层多 Q、K 两个 gamma
        assert_eq!(plain.parameters().len(), 8, "关闭时不应多出任何参数");
        assert_eq!(qk.parameters().len(), 10, "开启时应多出 q_norm/k_norm 两个 gamma");
        assert_eq!(qk.q_norm.as_ref().unwrap().gamma.numel(), d / n_head);
        assert_eq!(qk.k_norm.as_ref().unwrap().gamma.numel(), d / n_head);
    }

    /// 梯度必须流到 QK-Norm 的 `gamma` 上，否则"归一化"就只是个前向技巧、根本学不动。
    #[test]
    fn test_qk_norm_gradients_reach_gamma() {
        let (attn, data) = qk_fixture(16, 2, 2, 3, 23);
        let x = Tensor::from_vec(data, vec![1, 3, 16]);
        attn.forward(&x, None, None, 0).sum().backward();
        let peak = |t: &Tensor| t.grad().iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(peak(&attn.q_norm.as_ref().unwrap().gamma) > 0.0, "q_norm.gamma 没拿到梯度");
        assert!(peak(&attn.k_norm.as_ref().unwrap().gamma) > 0.0, "k_norm.gamma 没拿到梯度");
    }

    /// QK-Norm 与增量推理共存：普通（已旋转 K 进缓存）路径下，逐 token 增量前向
    /// 必须与全量重算逐位一致 —— 归一化发生在入缓存**之前**，所以历史 K 仍然是
    /// "归一化 + 旋转"后的结果，缓存复用不受影响。
    #[test]
    fn test_qk_norm_incremental_cache_matches_full_recompute() {
        let (d, n_head, n_kv, t) = (16usize, 2usize, 2usize, 5usize);
        let (attn, rows) = qk_fixture(d, n_head, n_kv, t, 29);
        let x = Tensor::from_vec(rows.clone(), vec![1, t, d]);
        let full = attn.forward(&x, None, None, 0).data();

        let mut cache = KVCache::new(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: None,
        });
        let mut inc = Vec::new();
        for i in 0..t {
            let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
            inc.extend_from_slice(&attn.forward(&xi, None, Some(&mut cache), i).data());
        }
        for (i, (a, b)) in full.iter().zip(&inc).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * (1.0 + a.abs()),
                "第 {i} 个元素不一致：全量 {a} vs 增量 {b}"
            );
        }
    }

    /// QK-Norm + MLA：latent 缓存里存的是**未归一化**的 latent，归一化每步在升维后重做，
    /// 因此两条路径仍然一致（这是 MLA + QK-Norm 能组合使用的前提）。
    #[test]
    fn test_qk_norm_with_mla_latent_cache_matches_full_recompute() {
        let (d, n_head, n_kv, r, t) = (24usize, 3usize, 3usize, 6usize, 5usize);
        let mut rng = Rng::new(31);
        let attn = MultiHeadAttention::new(d, n_head, n_kv, r, true, spec(), &mut rng);
        let rows: Vec<f32> = (0..t * d).map(|i| ((i * 13 % 31) as f32 * 0.29).sin()).collect();

        let x = Tensor::from_vec(rows.clone(), vec![1, t, d]);
        let full = attn.forward(&x, None, None, 0).data();

        let mut cache = KVCache::new_latent(KvCacheOpts {
            window: 0,
            sink: 0,
            bits: None,
        });
        let mut inc = Vec::new();
        for i in 0..t {
            let xi = Tensor::from_vec(rows[i * d..(i + 1) * d].to_vec(), vec![1, 1, d]);
            inc.extend_from_slice(&attn.forward(&xi, None, Some(&mut cache), i).data());
        }
        for (i, (a, b)) in full.iter().zip(&inc).enumerate() {
            assert!(
                (a - b).abs() < 1e-4 * (1.0 + a.abs()),
                "第 {i} 个元素不一致：全量 {a} vs latent 增量 {b}"
            );
        }
    }
}
