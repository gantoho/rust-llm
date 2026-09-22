//! 推测解码（第 34 课）与多 Token 预测（第 35 课）
//!
//! ## 为什么能更快
//!
//! 自回归生成的每一步只是"一次矩阵乘 + 一次采样"，却要把整个模型跑一遍。
//! 瓶颈不在算力，而在**访存**：每个权重从显存读进来只用它做几次浮点运算。
//! 于是"把串行的 N 次前向换成 1 次前向验证 N 个候选"就是纯赚——同一次权重读取
//! 摊到了 N 个位置上。
//!
//! 做法是让一个便宜的**草稿**（draft）先猜 γ 个 token，再让目标模型**一次前向**
//! 并行验证它们。
//!
//! ## 为什么是无损的
//!
//! 草稿按自己的分布 q 提出候选 x，目标分布为 p。逐个位置按下面的规则判：
//!
//! - 以 `min(1, p(x) / q(x))` 的概率**接受** x；
//! - 一旦被拒，就从**残差分布** `norm(max(0, p − q))` 重新采一个 token 顶上，
//!   并丢弃这一位之后的全部草稿。
//!
//! 这两步合起来保证输出**严格服从 p**（标准的拒绝采样论证：接受路径贡献
//! `min(p, q)`，残差路径把差出来的那一块 `max(0, p − q)` 原样补回）。所以它
//! 不是"近似目标模型"，而是与目标模型自己的采样**分布等价**——第 34 课的
//! 硬指标就是这一条。
//!
//! γ 个候选**全被接受**时还能白拿一个 token：目标模型这一次前向本来就在最后一个
//! 草稿位置算出了"下一个位置的分布"，直接采一个就行（记在 [`SpecStats::bonus`] 里）。
//!
//! ## 与 KV cache 的账目（踩坑重灾区）
//!
//! "一次前向喂 γ+1 个 token"意味着缓存里的历史必须**精确**对应"已经被采纳的序列"：
//! 多一位，注意力就会看到一批从未存在过的 token（表现为输出莫名跑偏）；少一位，
//! 缓存里的位置号就整体错开（RoPE 的相对距离全错）。本模块用一条不变量把账目钉死：
//!
//! > **不变量**：喂进缓存的 token 数恒等于「当前序列长度 − 1」，即缓存永远落后序列一格。
//!
//! 落后一格正是为了"一次前向拿到 p₁..p_{γ+1}"：把「序列最后一个 token + γ 个草稿」
//! 一起喂进去，返回的 γ+1 行 logits 恰好就是这 γ+1 个位置的分布（第 j 行 = 预测第 j 个
//! 草稿的分布，最后一行 = 预测"白拿的那个 token"的分布）。被拒时按实际采纳数
//! [`KVCache::rollback`] 掉多余的草稿，不变量自然恢复。
//!
//! 注意 `rollback` 在**滑动窗口已溢出**时不是严格可逆的（见 [`KVCache::rollback`]）：
//! 所以 `gamma` 要远小于窗口，这也是推测解码的常规用法。
//!
//! ## 多 Token 预测（MTP，第 35 课）
//!
//! 在主干的隐状态上挂 K 个预测头，第 k 个头直接预测"往后第 k+1 个 token"，
//! 训练损失是各头交叉熵之和（见 [`MtpHeads::loss`]）。它既是**训练信号**
//! （迫使隐状态多装一些未来信息），也是**天然的草稿**：一次前向就给出 K 个位置的
//! 候选分布，不必像小模型草稿那样自回归跑 γ 次前向（见 [`MtpDrafter`]）。

use crate::attention::KVCache;
use crate::layers::Linear;
use crate::loss::cross_entropy_loss;
use crate::model::GPT;
use crate::module::Module;
use crate::rng::Rng;
use crate::sample::{KvOpts, SampleOpts, probs_from_logits, sample_from_probs};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;

// ==================== 统计 ====================

/// 推测解码的统计口径。三项核心指标：
/// - 目标模型**前向次数**（加速比的分母）
/// - **接受率** = 被采纳的草稿数 / 提出的草稿数（草稿质量，决定实际加速）
/// - **平均每次前向产出的 token 数**（> 1 才说明真的加速了）
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// 跑过多少轮
    pub rounds: usize,
    /// 草稿一共提出多少候选
    pub drafted: usize,
    /// 其中被采纳多少
    pub accepted: usize,
    /// 被拒的次数（每次拒绝都会走一次残差修正）
    pub corrected: usize,
    /// 白拿的 token 数（γ 个候选全被接受的那些轮）
    pub bonus: usize,
    /// 目标模型前向次数
    pub target_forwards: usize,
    /// 一共产出多少 token
    pub emitted: usize,
}

impl SpecStats {
    /// 接受率：被采纳的草稿 / 提出的草稿。没提出过草稿时为 0。
    pub fn acceptance_rate(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }

    /// 平均每次目标前向产出多少 token（> 1 = 相对逐 token 解码有加速）
    pub fn tokens_per_forward(&self) -> f64 {
        if self.target_forwards == 0 {
            0.0
        } else {
            self.emitted as f64 / self.target_forwards as f64
        }
    }
}

// ==================== 草稿来源 ====================

/// 一个草稿候选：token 本身 + **它被提出时的完整分布 q**。
///
/// 必须把 q 一起带回来：接受概率 `min(1, p/q)` 与残差分布 `max(0, p − q)` 都要用到它。
/// 只带 token 而"事后重新算一遍 q"是行不通的——q 必须是**当初真正用来抽样**的那一份，
/// 任何一处预处理（top-k 边界、UTF-8 掩码、重复惩罚窗口）不一致都会让输出偏离 p。
pub struct Proposal {
    /// 候选 token
    pub token: usize,
    /// 该分布的支撑集（按概率降序）
    pub ids: Vec<usize>,
    /// 与 `ids` 一一对应的概率（和为 1）
    pub probs: Vec<f32>,
}

impl Proposal {
    /// q(token)：候选在草稿分布下的概率
    pub fn q(&self) -> f32 {
        prob_of(&self.ids, &self.probs, self.token)
    }
}

/// 草稿来源的统一接口。
///
/// 只要实现它，任何"猜未来 token"的东西都能接进推测解码：一个更小的自回归模型
/// （[`ModelDrafter`]）、MTP 头（[`MtpDrafter`]）、甚至测试里手写的固定分布。
/// 算法对 q 没有任何要求——q 再差也只是接受率低，**不会**影响输出的正确性。
pub trait Drafter {
    /// 提出至多 `gamma` 个候选（每个都带自己的分布 q）。
    ///
    /// 实现者内部维护自己的缓存/状态，并且要自己保证"从 `seq` 出发"：
    /// 第一个候选的分布必须是 `q(· | seq)`。
    fn propose(
        &mut self,
        seq: &[usize],
        gamma: usize,
        opts: &SampleOpts,
        vocab: Option<&[Vec<u8>]>,
        rng: &mut Rng,
    ) -> Vec<Proposal>;

    /// 本轮序列已确定：把内部状态对齐到 `seq[..seq.len()-1]`。
    ///
    /// 被采纳的草稿保留、被拒的丢弃——这就是草稿侧的"rollback"。
    fn commit(&mut self, seq: &[usize]);

    /// 丢弃全部内部状态（换一段新 prompt 时调用）
    fn reset(&mut self);
}

// ==================== 目标模型的增量前向 ====================

/// 单序列的增量前向句柄：维护一份 KV cache，并维持"**缓存落后序列一格**"的不变量。
///
/// 抽成独立类型是因为推测解码与草稿模型都要这套账目：`sync` 负责把缓存对齐、
/// `feed` 负责一次吃进若干 token 并把 logits 行交回来。
pub struct TargetStream<'a> {
    model: &'a GPT,
    cache: Vec<KVCache>,
}

impl<'a> TargetStream<'a> {
    /// 按推理侧设置构造（`kv.enable` 被忽略：推测解码必须配缓存，否则
    /// "一次前向验证 γ+1 个位置"这件事本身就不成立）
    pub fn new(model: &'a GPT, kv: KvOpts) -> Self {
        TargetStream {
            model,
            cache: model.new_kv_cache_with(kv.sink, kv.bits),
        }
    }

    /// 缓存里已喂入的位置数（不变量成立时 = 序列长度 − 1）。
    ///
    /// 用 `positions_seen` 而不是 `seq_len`：滑动窗口开始丢行之后两者不再相等，
    /// 而账目要按"总共喂过多少"来算。
    pub fn fed(&self) -> usize {
        self.cache.first().map_or(0, |c| c.positions_seen())
    }

    /// 缓存窗口（= `block_size`），也是单次前向能吃下的最大 token 数
    pub fn window(&self) -> usize {
        self.cache.first().map_or(0, |c| c.window())
    }

    /// 词表大小
    pub fn vocab_size(&self) -> usize {
        self.model.cfg.vocab_size
    }

    /// 一次前向吃掉 `block`，返回全部行的 logits（`[block.len(), vocab]` 展平）。
    pub fn feed(&mut self, block: &[usize]) -> Vec<f32> {
        assert!(!block.is_empty(), "一次前向至少要喂一个 token");
        let model = self.model;
        let logits = crate::tensor::no_grad(|| {
            model.forward(block, 1, block.len(), Some(&mut self.cache), false)
        });
        logits.data()
    }

    /// 一次前向吃掉 `block`，返回**最后一行的隐状态**（长度 = `n_embd`）。
    ///
    /// 给 MTP 草稿用：它只需要"最后一个位置的隐状态"，再由各预测头一次给出
    /// 往后 γ 个位置的分布，不必逐 token 自回归。
    pub fn feed_hidden(&mut self, block: &[usize]) -> Vec<f32> {
        assert!(!block.is_empty(), "一次前向至少要喂一个 token");
        let model = self.model;
        let d = self.model.cfg.n_embd;
        let hidden = crate::tensor::no_grad(|| {
            model.forward_hidden_cached(block, 1, block.len(), Some(&mut self.cache), false)
        });
        let data = hidden.data();
        data[data.len() - d..].to_vec()
    }

    /// 把缓存对齐到 `seq[..seq.len()-1]`：多了回滚、少了补喂。
    ///
    /// 首次调用（缓存为空）就是整段 prompt 的 prefill——按窗口大小分块喂，
    /// 因为单次前向的 token 数不能超过窗口。
    pub fn sync(&mut self, seq: &[usize]) {
        assert!(!seq.is_empty(), "序列不能为空");
        let target = seq.len() - 1;
        let fed = self.fed();
        if fed == target {
            return;
        }
        if fed > target {
            self.rollback_to(target);
            return;
        }
        let window = self.window().max(1);
        let mut i = fed;
        while i < target {
            let end = (i + window).min(target);
            self.feed(&seq[i..end]);
            i = end;
        }
    }

    /// 把缓存回滚到"喂过 `fed` 个位置"的状态
    pub fn rollback_to(&mut self, fed: usize) {
        let cur = self.fed();
        assert!(fed <= cur, "回滚目标 {fed} 大于当前已喂入的 {cur}");
        let n = cur - fed;
        if n == 0 {
            return;
        }
        for c in self.cache.iter_mut() {
            c.rollback(n);
        }
    }

    /// 清空缓存（换一段新序列时调用）
    pub fn reset(&mut self) {
        for c in self.cache.iter_mut() {
            c.reset();
        }
    }
}

// ==================== 一行 logits → 分布 ====================

/// 取某个 token 在给定分布下的概率（不在支撑集里就是 0）。
fn prob_of(ids: &[usize], probs: &[f32], token: usize) -> f32 {
    ids.iter()
        .position(|&i| i == token)
        .map_or(0.0, |i| probs[i])
}

/// 一行 logits → `(按概率降序的候选, 概率)`，预处理与 [`crate::sample::generate`] **完全一致**：
/// 先按 UTF-8 合法性掩码（字节级词表），再做重复惩罚 / temperature / top-k / top-p。
///
/// `prefix` 是该行预测位置**之前**的完整 token 序列，用来算重复惩罚的回看窗口与
/// 待拼字节。推测解码里每一行的 prefix 都不同（后面的行要算上前面的草稿），
/// 所以必须逐行传进来，不能共用一份。
fn dist_row(
    logits: &[f32],
    opts: &SampleOpts,
    vocab: Option<&[Vec<u8>]>,
    prefix: &[usize],
) -> (Vec<usize>, Vec<f32>) {
    let recent: &[usize] = if opts.repetition_window == 0 {
        &[]
    } else {
        let start = prefix.len().saturating_sub(opts.repetition_window);
        &prefix[start..]
    };
    match vocab {
        // 非字节级词表（char 分词器）：没有"半个字符"的问题，直接采样
        None => probs_from_logits(logits, opts, recent),
        Some(v) => {
            let mut masked = logits.to_vec();
            crate::sample::mask_illegal_utf8(
                &mut masked,
                v,
                &crate::sample::pending_tail(v, prefix),
            );
            probs_from_logits(&masked, opts, recent)
        }
    }
}

/// 在残差分布 `norm(max(0, p − q))` 上采样（草稿被拒时的修正）。
///
/// 残差只在 p 的支撑集上非零（`r > 0` 要求 `p > q ≥ 0`，故 `p > 0`），
/// 所以遍历 p 的支撑集就够了。总质量理论上恒为 `1 − Σ min(p, q) > 0`；
/// 真为 0 只可能是 p 与 q 逐位相等（那种情形下不该走到这里），
/// 兜底退回 p 的最高概率项，绝不返回非法 token。
fn residual_sample(
    p_ids: &[usize],
    p_probs: &[f32],
    q_ids: &[usize],
    q_probs: &[f32],
    rng: &mut Rng,
) -> usize {
    let mut total = 0.0f32;
    for (j, &id) in p_ids.iter().enumerate() {
        total += (p_probs[j] - prob_of(q_ids, q_probs, id)).max(0.0);
    }
    if total <= 0.0 {
        return p_ids[0];
    }
    let mut u = rng.next_f32() * total;
    for (j, &id) in p_ids.iter().enumerate() {
        let r = (p_probs[j] - prob_of(q_ids, q_probs, id)).max(0.0);
        if u < r {
            return id;
        }
        u -= r;
    }
    p_ids[0]
}

// ==================== 一轮推测解码 ====================

/// 推测解码的执行体：持有目标模型的缓存与统计，可以连续调用 [`SpecDecoder::step`]。
///
/// 用法见 [`speculative_generate`]；想自己控制生成循环（比如接进 `chat`）就直接
/// 反复调 `step`，每轮把返回的 token 追加到序列末尾即可。
pub struct SpecDecoder<'a> {
    target: TargetStream<'a>,
    gamma: usize,
    opts: SampleOpts,
    vocab: Option<&'a [Vec<u8>]>,
    stats: SpecStats,
}

impl<'a> SpecDecoder<'a> {
    /// - `gamma`：每轮草稿长度。会被夹到 `1..=block_size-1`——单次前向要喂
    ///   γ+1 个 token，超过窗口会直接报错。
    /// - `vocab`：字节级词表（[`crate::tokenizer::Tokenizer::vocab_bytes`]），
    ///   用来做 UTF-8 掩码；char 分词器传 `None`。
    pub fn new(
        model: &'a GPT,
        kv: KvOpts,
        gamma: usize,
        opts: SampleOpts,
        vocab: Option<&'a [Vec<u8>]>,
    ) -> Self {
        let gamma = gamma.max(1).min(model.cfg.block_size.saturating_sub(1).max(1));
        SpecDecoder {
            target: TargetStream::new(model, kv),
            gamma,
            opts,
            vocab,
            stats: SpecStats::default(),
        }
    }

    /// 实际生效的草稿长度
    pub fn gamma(&self) -> usize {
        self.gamma
    }

    pub fn stats(&self) -> &SpecStats {
        &self.stats
    }

    /// 目标缓存已喂入的位置数（不变量成立时应等于「序列长度 − 1」）
    pub fn cache_fed(&self) -> usize {
        self.target.fed()
    }

    /// 跑一轮：草稿提出 γ 个候选，目标模型**一次**前向并行验证。
    ///
    /// 返回本轮产出的 token（`1..=γ+1` 个）：至少一个（被拒时是残差修正出来的那个），
    /// 全被接受时是 γ 个草稿 + 白拿的一个。
    pub fn step(&mut self, seq: &[usize], drafter: &mut dyn Drafter, rng: &mut Rng) -> Vec<usize> {
        assert!(!seq.is_empty(), "推测解码需要非空上下文");
        let proposals = drafter.propose(seq, self.gamma, &self.opts, self.vocab, rng);
        assert!(!proposals.is_empty(), "草稿至少要给出一个候选");
        let gamma = proposals.len();
        assert!(
            gamma + 1 <= self.target.window().max(1),
            "一轮要喂 γ+1 = {} 个 token，超过缓存窗口 {}；请调小 gamma",
            gamma + 1,
            self.target.window()
        );

        // 一次前向：序列最后一个 token + 全部草稿（共 γ+1 个）。
        // 缓存先对齐到 `seq[..len-1]`（不变量），于是本次返回的 γ+1 行正是
        // p₁..p_{γ+1}（第 j 行 = 预测第 j 个草稿的分布）。
        self.target.sync(seq);
        let mut block: Vec<usize> = Vec::with_capacity(gamma + 1);
        block.push(seq[seq.len() - 1]);
        block.extend(proposals.iter().map(|p| p.token));
        let rows = self.target.feed(&block);
        let v = self.target.vocab_size();

        let mut prefix: Vec<usize> = seq.to_vec();
        let mut emitted: Vec<usize> = Vec::with_capacity(gamma + 1);
        let mut accepted = 0usize;
        let mut rejected = false;

        for (j, prop) in proposals.iter().enumerate() {
            let row = &rows[j * v..(j + 1) * v];
            let (p_ids, p_probs) = dist_row(row, &self.opts, self.vocab, &prefix);
            let p = prob_of(&p_ids, &p_probs, prop.token);
            let q = prop.q();
            // min(1, p/q)。q = 0 说明"草稿不可能提出这个 token"（正常流程不会发生，
            // 因为候选就是从 q 里抽出来的）——按数学极限取 1，即无条件接受。
            let accept_p = if q <= 0.0 { 1.0 } else { (p / q).min(1.0) };
            if rng.next_f32() < accept_p {
                emitted.push(prop.token);
                prefix.push(prop.token);
                accepted += 1;
            } else {
                // 拒绝：残差分布采样顶上，并丢弃这一位之后的全部草稿
                let token = residual_sample(&p_ids, &p_probs, &prop.ids, &prop.probs, rng);
                emitted.push(token);
                prefix.push(token);
                rejected = true;
                break;
            }
        }

        if !rejected {
            // 全被接受：最后一个草稿位置的分布（第 γ 行）就是"白拿"的那个 token 的分布
            let row = &rows[gamma * v..(gamma + 1) * v];
            let (ids, probs) = dist_row(row, &self.opts, self.vocab, &prefix);
            let bonus = sample_from_probs(&ids, &probs, rng);
            emitted.push(bonus);
        }

        // 账目对齐：缓存应停留在"新序列长度 − 1"。被拒时缓存里留着
        // `seq[len-1] + 草稿[..γ-1]`，比目标多出 `γ − 采纳数` 个位置，回滚掉。
        let mut new_seq: Vec<usize> = seq.to_vec();
        new_seq.extend_from_slice(&emitted);
        self.target.rollback_to(new_seq.len() - 1);

        self.stats.rounds += 1;
        self.stats.drafted += gamma;
        self.stats.accepted += accepted;
        self.stats.bonus += usize::from(!rejected);
        self.stats.corrected += usize::from(rejected);
        self.stats.target_forwards += 1;
        self.stats.emitted += emitted.len();

        drafter.commit(&new_seq);
        emitted
    }
}

// ==================== 草稿来源一：小模型自回归 ====================

/// 用另一个（通常更小的）模型做草稿：它按自己的分布自回归猜 γ 个 token。
///
/// 草稿模型**不要求**与目标模型同源。它猜得准就收益大，猜不准就多拒绝几次，
/// 但无论多不准，输出分布都不会变（这点由拒绝采样保证）。
pub struct ModelDrafter<'a> {
    stream: TargetStream<'a>,
    /// 草稿是否用贪心（argmax）而不是按 q 采样。
    ///
    /// 贪心草稿的 q 退化成单点分布：接受率完全由目标分布在该 token 上的概率决定。
    /// 小模型草稿的常规做法就是贪心——它的"错误答案"往往也是目标模型的高概率项。
    pub greedy: bool,
}

impl<'a> ModelDrafter<'a> {
    pub fn new(model: &'a GPT, kv: KvOpts) -> Self {
        ModelDrafter {
            stream: TargetStream::new(model, kv),
            greedy: false,
        }
    }

    /// 用贪心草稿的构造器
    pub fn greedy(model: &'a GPT, kv: KvOpts) -> Self {
        ModelDrafter {
            stream: TargetStream::new(model, kv),
            greedy: true,
        }
    }
}

impl Drafter for ModelDrafter<'_> {
    fn propose(
        &mut self,
        seq: &[usize],
        gamma: usize,
        opts: &SampleOpts,
        vocab: Option<&[Vec<u8>]>,
        rng: &mut Rng,
    ) -> Vec<Proposal> {
        self.stream.sync(seq);
        // 先把序列最后一个 token 喂进去 → q₁ = q(·|seq)
        let mut prefix: Vec<usize> = seq.to_vec();
        let mut row = self.stream.feed(&prefix[prefix.len() - 1..]);

        let mut out = Vec::with_capacity(gamma);
        for k in 0..gamma {
            let (ids, probs) = dist_row(&row, opts, vocab, &prefix);
            let token = if self.greedy {
                ids[0] // 已按概率降序，第一项就是 argmax
            } else {
                sample_from_probs(&ids, &probs, rng)
            };
            out.push(Proposal { token, ids, probs });
            prefix.push(token);
            if k + 1 < gamma {
                // 把刚猜的 token 喂进去 → q_{k+1}；最后一个不必喂，
                // 这样 propose 结束时缓存正好落后一轮（commit 直接沿用）
                row = self.stream.feed(&prefix[prefix.len() - 1..]);
            }
        }
        out
    }

    fn commit(&mut self, seq: &[usize]) {
        self.stream.sync(seq);
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

// ==================== 多 Token 预测（第 35 课）====================

/// 多 Token 预测头：K 个 `Linear`，第 k 个把**位置 i 的隐状态**映射到
/// 「位置 `i + k + 1` 的 token」的 logits。
///
/// 输入是"逐层位移"的：第 k 个头用的还是同一份隐状态，只是目标往后挪 k+1 位。
/// 于是同一份隐状态要同时支撑"下一个 token"和"往后第 K 个 token"的预测，
/// 这迫使主干把更长的未来信息编码进去——这是 MTP 作为训练信号的价值所在。
///
/// 与 DeepSeek-V3 的做法有一处刻意的差别：那里 MTP 头与主干的输出头共享词嵌入，
/// 这里各头独立成小矩阵。共享权重需要改 `GPT` 的输出头结构（牵动 checkpoint 布局），
/// 而本课要验证的是"多未来 token 的预测与它的草稿用途"，独立小头已经足够表达。
pub struct MtpHeads {
    heads: Vec<Linear>,
    n_embd: usize,
    vocab: usize,
}

impl MtpHeads {
    /// `n_heads` = 往后预测多少个 token（K=1 就是普通的"预测下一个 token"）
    pub fn new(n_embd: usize, vocab: usize, n_heads: usize, rng: &mut Rng) -> Self {
        assert!(n_heads >= 1, "至少要有一个预测头");
        MtpHeads {
            heads: (0..n_heads).map(|_| Linear::new(n_embd, vocab, rng)).collect(),
            n_embd,
            vocab,
        }
    }

    pub fn n_heads(&self) -> usize {
        self.heads.len()
    }

    pub fn n_embd(&self) -> usize {
        self.n_embd
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// 暴露各头，便于取梯度/权重做诊断
    pub fn heads(&self) -> &[Linear] {
        &self.heads
    }

    /// 第 k 个头（0 = 预测下一个 token）对每一行隐状态出 logits：`[n, n_embd] -> [n, vocab]`
    pub fn logits(&self, hidden: &Tensor, k: usize) -> Tensor {
        assert!(k < self.heads.len(), "第 {k} 个头不存在（共 {} 个）", self.heads.len());
        self.heads[k].forward(hidden)
    }

    /// MTP 训练损失：各头交叉熵之和 / 头数。
    ///
    /// - `hidden`：主干的隐状态 `[b*t, n_embd]`（一般来自 [`GPT::forward_hidden`]）
    /// - `tokens`：同一次前向对应的输入 token `[b*t]`（展平）
    /// - `b` / `t`：batch 与序列长度。第 k 个头的目标是 `tokens[i + k + 1]`。
    ///
    /// 只取**同一段内不越界**的位置（`i % t + k + 1 < t`）：展平后的数组里，
    /// 一段的末尾与下一段的开头在内存上相邻，不排除的话第 k 个头会把"下一段的第 k 个
    /// token"当成正确答案——那是个纯粹由 batch 拼接方式决定的假目标。
    pub fn loss(&self, hidden: &Tensor, tokens: &[usize], b: usize, t: usize) -> Tensor {
        assert_eq!(
            hidden.shape(),
            &[b * t, self.n_embd],
            "隐状态形状应为 [b*t, n_embd]"
        );
        assert_eq!(tokens.len(), b * t, "token 数应为 b*t");
        let mut acc: Option<Tensor> = None;
        for k in 0..self.heads.len() {
            let idx: Vec<usize> = (0..b * t).filter(|i| i % t + k + 1 < t).collect();
            if idx.is_empty() {
                continue; // 序列太短，这个头没有可用位置
            }
            let h = hidden.gather_rows(&idx);
            let targets: Vec<usize> = idx.iter().map(|&i| tokens[i + k + 1]).collect();
            let l = cross_entropy_loss(&self.heads[k].forward(&h), &targets);
            acc = Some(match acc {
                Some(a) => a.add(&l),
                None => l,
            });
        }
        acc.expect("至少要有一个头能取到有效位置（需要 t ≥ 2）")
            .mul_scalar(1.0 / self.heads.len() as f32)
    }
}

impl Module for MtpHeads {
    fn parameters(&self) -> Vec<Tensor> {
        self.heads.iter().flat_map(|h| h.parameters()).collect()
    }
}

/// 草稿来源二：拿 MTP 头当草稿——**一次前向**给出往后 γ 个位置的候选分布。
///
/// 与小模型草稿的区别：小模型草稿要自回归跑 γ 次前向才能凑出 γ 个候选，MTP 草稿
/// 只需要 1 次（拿最后一个位置的隐状态），剩下的全靠预测头并行出。代价是这 γ 个
/// 分布彼此**条件独立**（都只条件于同一个隐状态），比自回归草稿"糊"一些，
/// 所以接受率通常更低——这正是"草稿算力 vs 接受率"的取舍。
pub struct MtpDrafter<'a> {
    backbone: &'a GPT,
    stream: TargetStream<'a>,
    heads: MtpHeads,
}

impl<'a> MtpDrafter<'a> {
    /// `backbone` 一般就是目标模型（MTP 头本来就挂在它的隐状态上），
    /// 也可以传一个更小的模型当草稿底座。
    pub fn new(backbone: &'a GPT, n_heads: usize, kv: KvOpts, rng: &mut Rng) -> Self {
        let heads = MtpHeads::new(
            backbone.cfg.n_embd,
            backbone.cfg.vocab_size,
            n_heads,
            rng,
        );
        MtpDrafter {
            backbone,
            stream: TargetStream::new(backbone, kv),
            heads,
        }
    }

    /// MTP 头（要接着训练就把它的参数交给优化器）
    pub fn heads(&self) -> &MtpHeads {
        &self.heads
    }

    pub fn heads_mut(&mut self) -> &mut MtpHeads {
        &mut self.heads
    }
}

impl Drafter for MtpDrafter<'_> {
    fn propose(
        &mut self,
        seq: &[usize],
        gamma: usize,
        opts: &SampleOpts,
        vocab: Option<&[Vec<u8>]>,
        rng: &mut Rng,
    ) -> Vec<Proposal> {
        // 头数不够就给几个算几个：调用方（SpecDecoder）用的是返回值的实际长度
        let gamma = gamma.min(self.heads.n_heads());
        self.stream.sync(seq);
        // 一次前向拿最后一个位置的隐状态，往后 γ 个位置的分布全部由它推出
        let row = self.stream.feed_hidden(&seq[seq.len() - 1..]);
        let d = self.backbone.cfg.n_embd;
        assert_eq!(row.len(), d);
        let h = Tensor::from_vec(row, vec![1, d]);

        let mut prefix: Vec<usize> = seq.to_vec();
        let mut out = Vec::with_capacity(gamma);
        for k in 0..gamma {
            let logits = self.heads.logits(&h, k).data();
            let (ids, probs) = dist_row(&logits, opts, vocab, &prefix);
            let token = sample_from_probs(&ids, &probs, rng);
            out.push(Proposal { token, ids, probs });
            // 后续候选的重复惩罚窗口要把前面猜的 token 算进去
            prefix.push(token);
        }
        out
    }

    fn commit(&mut self, seq: &[usize]) {
        self.stream.sync(seq);
    }

    fn reset(&mut self) {
        self.stream.reset();
    }
}

// ==================== 文本生成 ====================

/// 用推测解码生成文本，返回 `(文本, 统计)`。
///
/// 结束条件、prompt 前补 BOS、停止标记的处理都与 [`crate::sample::generate`] 对齐，
/// 唯一的差别是多了一个草稿模型与 `gamma`。`drafter` 的内部缓存会被 [`Drafter::reset`]
/// 清空，所以同一个 drafter 可以反复传给不同的 prompt。
#[allow(clippy::too_many_arguments)]
pub fn speculative_generate(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    gamma: usize,
    opts: &SampleOpts,
    kv: KvOpts,
    drafter: &mut dyn Drafter,
    rng: &mut Rng,
) -> (String, SpecStats) {
    // prompt 前补 BOS：训练时每篇文档都以 BOS 开头（与 sample::generate 一致）
    let mut ids: Vec<usize> = tokenizer.bos_id().into_iter().collect();
    ids.extend(tokenizer.encode(prompt));
    if ids.is_empty() {
        ids.push(0);
    }

    let mut dec = SpecDecoder::new(model, kv, gamma, *opts, tokenizer.vocab_bytes());
    drafter.reset();
    let eos = tokenizer.eos_id();
    let mut produced = 0usize;

    while produced < max_new {
        for &tok in dec.step(&ids, drafter, rng).iter() {
            ids.push(tok);
            // 采到 EOS 就收（EOS 本身不进结果）
            if eos == Some(tok) {
                return (tokenizer.decode(&ids[..ids.len() - 1]), *dec.stats());
            }
            // 命中停止标记就收：只在**新生成的部分**里查找（prompt 自己就含标记）
            if !opts.stop.is_empty() {
                let text = tokenizer.decode(&ids);
                if let Some(generated) = text.get(prompt.len()..) {
                    if let Some(rel) = opts.stop.iter().filter_map(|s| generated.find(s)).min() {
                        return (text[..prompt.len() + rel].trim_end().to_string(), *dec.stats());
                    }
                }
            }
            produced += 1;
            if produced >= max_new {
                break;
            }
        }
    }

    (tokenizer.decode(&ids), *dec.stats())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GPTConfig;
    use crate::optim::Optimizer;
    use crate::sample::generate;

    /// 极小模型：推测解码的统计检验要跑几千轮，模型必须便宜到"每轮几乎免费"。
    /// `block_size` 给足，避免滑动窗口溢出（溢出后 `rollback` 不是严格可逆，
    /// 单测里只验证窗口内的严格等价）。
    fn tiny_model(vocab: usize, seed: u64) -> GPT {
        let cfg = GPTConfig {
            n_embd: 8,
            n_head: 2,
            n_layer: 1,
            block_size: 64,
            ..GPTConfig::tiny(vocab)
        };
        GPT::new(cfg, &mut Rng::new(seed))
    }

    /// 近似贪心：top_k = 1 + 极低温度 → 分布退化成单点，采样结果与 argmax 等价
    fn greedy_opts() -> SampleOpts {
        SampleOpts {
            temperature: 0.05,
            top_k: 1,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            stop: &[],
        }
    }

    /// 不截断的采样（用于分布检验：真值分布就是完整的 softmax）
    fn sampled_opts(temperature: f32) -> SampleOpts {
        SampleOpts {
            temperature,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            repetition_window: 0,
            stop: &[],
        }
    }

    /// 测试用草稿：完全脱离模型的固定分布 q。
    ///
    /// 候选必须**真的按 q 抽**（而不是恒定返回 q 的 argmax）：接受/拒绝能抵消成 p，
    /// 靠的正是"x ~ q 且接受概率 min(1, p(x)/q(x))"这一对配合。把 x 定死会破坏这个关系。
    struct FixedDrafter {
        ids: Vec<usize>,
        probs: Vec<f32>,
    }

    impl Drafter for FixedDrafter {
        fn propose(
            &mut self,
            _seq: &[usize],
            gamma: usize,
            _opts: &SampleOpts,
            _vocab: Option<&[Vec<u8>]>,
            rng: &mut Rng,
        ) -> Vec<Proposal> {
            (0..gamma)
                .map(|_| Proposal {
                    token: sample_from_probs(&self.ids, &self.probs, rng),
                    ids: self.ids.clone(),
                    probs: self.probs.clone(),
                })
                .collect()
        }

        fn commit(&mut self, _seq: &[usize]) {}

        fn reset(&mut self) {}
    }

    /// 贪心路径：草稿水平再差，输出也必须与目标模型的逐个 token 贪心解码**完全一致**。
    ///
    /// 草稿故意换成一个另起随机种子的模型（预测大概率不同），用来逼出拒绝分支。
    #[test]
    fn test_spec_greedy_matches_target_greedy() {
        let corpus = "the quick brown fox jumps over the lazy dog";
        let tokenizer = Tokenizer::char(corpus);
        let target = tiny_model(tokenizer.vocab_size(), 1);
        let draft = tiny_model(tokenizer.vocab_size(), 2);

        let opts = greedy_opts();
        let mut drafter = ModelDrafter::new(&draft, KvOpts::on(0, None));
        let (spec, stats) = speculative_generate(
            &target,
            &tokenizer,
            "the",
            24,
            4,
            &opts,
            KvOpts::on(0, None),
            &mut drafter,
            &mut Rng::new(7),
        );
        let plain = generate(
            &target,
            &tokenizer,
            "the",
            24,
            &opts,
            KvOpts::on(0, None),
            &mut Rng::new(7),
        );

        assert_eq!(spec, plain, "贪心推测解码必须与目标模型贪心解码逐 token 一致");
        assert_eq!(stats.target_forwards, stats.rounds, "每轮只该有一次目标前向");
        assert!(stats.emitted >= 24);
    }

    /// 采样路径的正确性（硬指标）：输出的**分布**必须等于目标分布。
    ///
    /// 做法：固定上下文，草稿用一个与模型无关的固定分布 q，统计几千次独立推测
    /// 产出的**第一个** token 的经验频率，与目标模型在该上下文上的真实分布比对
    /// （总变差距离）。被拒时走的是残差修正，所以这个检验同时覆盖了接受与修正两条路径。
    fn check_first_token_distribution(ids: Vec<usize>, probs: Vec<f32>, label: &str) {
        let tokenizer = Tokenizer::char("abcd");
        let vocab = tokenizer.vocab_size();
        let target = tiny_model(vocab, 11);
        let opts = sampled_opts(0.7);
        // 单 token 上下文：每轮只花"一次 γ+1 个 token 的前向"，几千轮也很快
        let seq = vec![0usize];

        // 真值分布 p(·|seq)：喂 seq 拿最后一行
        let (p_ids, p_probs) = {
            let mut stream = TargetStream::new(&target, KvOpts::on(0, None));
            let rows = stream.feed(&seq);
            assert_eq!(rows.len(), vocab);
            dist_row(&rows, &opts, None, &seq)
        };

        let n = 4000;
        let mut counts = vec![0usize; vocab];
        let mut corrected = 0usize;
        // 4000 次独立试验共用一条随机流，而不是"每次换一个种子"：
        // `Rng::next_f32` 取的是状态的高 24 位，小种子（0,1,2,…）前几次输出恒为 ~0，
        // 拿它们当随机源会让草稿每次都提出 q 的首个 token——那就退化成了单点草稿，
        // 一般分布 q 的接受/残差路径根本走不到。
        let mut rng = Rng::new(20240918);
        for _ in 0..n {
            let mut drafter = FixedDrafter {
                ids: ids.clone(),
                probs: probs.clone(),
            };
            let mut dec = SpecDecoder::new(&target, KvOpts::on(0, None), 3, opts, None);
            let emitted = dec.step(&seq, &mut drafter, &mut rng);
            assert!(!emitted.is_empty());
            counts[emitted[0]] += 1;
            corrected += dec.stats().corrected;
        }

        let tv: f32 = 0.5
            * (0..vocab)
                .map(|i| (counts[i] as f32 / n as f32 - prob_of(&p_ids, &p_probs, i)).abs())
                .sum::<f32>();
        assert!(
            tv < 0.05,
            "{label}：推测解码的首 token 分布应等于目标分布，总变差 {tv:.4}\n\
             经验频率 {:?}\n目标概率 {:?}",
            counts.iter().map(|c| *c as f32 / n as f32).collect::<Vec<_>>(),
            (0..vocab)
                .map(|i| prob_of(&p_ids, &p_probs, i))
                .collect::<Vec<_>>()
        );
        assert!(corrected > 0, "{label}：应当有被拒的轮次（否则没覆盖残差修正）");
    }

    /// 退化草稿（q 是单点分布）：接受概率就是 p(x)，被拒时残差把剩下的概率质量补回来
    #[test]
    fn test_spec_distribution_matches_target_with_degenerate_draft() {
        // q = δ₀：草稿只会提 token 0
        check_first_token_distribution(vec![0], vec![1.0], "单点草稿");
    }

    /// 非退化草稿：q 在 4 个 token 上有分布 → 覆盖 `min(1, p/q)` 与残差的一般情形
    #[test]
    fn test_spec_distribution_matches_target_with_general_draft() {
        check_first_token_distribution(vec![0, 1, 2, 3], vec![0.4, 0.3, 0.2, 0.1], "一般草稿");
    }

    /// 全被接受时，一次目标前向要产出 γ+1 个 token（加速确实发生）。
    /// 草稿用**同一个模型**的贪心输出：目标分布是单点，草稿必然命中。
    #[test]
    fn test_all_accepted_emits_gamma_plus_one_per_forward() {
        let tokenizer = Tokenizer::char("abcd");
        let model = tiny_model(tokenizer.vocab_size(), 3);
        let opts = greedy_opts();

        let mut drafter = ModelDrafter::greedy(&model, KvOpts::on(0, None));
        let mut dec = SpecDecoder::new(&model, KvOpts::on(0, None), 4, opts, None);
        let seq = tokenizer.encode("ab");
        let emitted = dec.step(&seq, &mut drafter, &mut Rng::new(1));

        assert_eq!(emitted.len(), 5, "γ=4 全部被接受 → 4 + 1 个 token");
        assert_eq!(dec.stats().target_forwards, 1, "整轮只该有一次目标前向");
        assert!(dec.stats().tokens_per_forward() > 1.0, "一次前向产出应大于 1 个 token");
        assert_eq!(dec.stats().bonus, 1, "白拿了一个 token");
        assert_eq!(dec.stats().corrected, 0);
        assert_eq!(dec.stats().acceptance_rate(), 1.0);
        assert_eq!(
            dec.cache_fed(),
            seq.len() + emitted.len() - 1,
            "缓存应恒等于「序列长度 − 1」"
        );
    }

    /// 拒绝路径的账目：连着跑多轮故意被拒的推测，每轮结束后缓存都必须正好落后一格；
    /// 最终文本仍要与逐个 token 的贪心解码一致（缓存错一格就会立刻跑偏）。
    #[test]
    fn test_rejection_rolls_back_and_generation_stays_identical() {
        let tokenizer = Tokenizer::char("the quick brown fox jumps over the lazy dog");
        let model = tiny_model(tokenizer.vocab_size(), 5);
        let opts = greedy_opts();

        // 起点必须与 `generate` 完全一致：生成入口都会在 prompt 前补 BOS
        // （char 分词器也有 BOS/EOS/PAD 三个特殊 token），少补一个 BOS
        // 两边就是两条不同的上下文，后面的对比也就毫无意义。
        let mut ids: Vec<usize> = tokenizer.bos_id().into_iter().collect();
        ids.extend(tokenizer.encode("the"));
        let prompt_len = ids.len();
        let mut drafter = FixedDrafter {
            ids: vec![0],
            probs: vec![1.0],
        };
        let mut dec = SpecDecoder::new(&model, KvOpts::on(0, None), 3, opts, tokenizer.vocab_bytes());
        let mut rng = Rng::new(1);

        for round in 0..8 {
            let emitted = dec.step(&ids, &mut drafter, &mut rng);
            assert!(emitted.len() <= 4, "γ=3 时一轮最多产出 4 个 token");
            ids.extend_from_slice(&emitted);
            assert_eq!(
                dec.cache_fed(),
                ids.len() - 1,
                "第 {round} 轮后缓存应落后序列一格"
            );
        }

        let plain = generate(
            &model,
            &tokenizer,
            "the",
            ids.len() - prompt_len,
            &opts,
            KvOpts::on(0, None),
            &mut Rng::new(1),
        );
        assert_eq!(
            tokenizer.decode(&ids),
            plain,
            "经历多次回滚后，贪心结果仍应与逐 token 解码一致"
        );
    }

    /// MTP：头数与形状正确，反传后每个头的权重都拿到非零梯度
    #[test]
    fn test_mtp_head_shapes_and_gradients() {
        let tokenizer = Tokenizer::char("abcdabcd");
        let model = tiny_model(tokenizer.vocab_size(), 1);
        let mut rng = Rng::new(2);
        let k = 3;
        let heads = MtpHeads::new(model.cfg.n_embd, model.cfg.vocab_size, k, &mut rng);
        assert_eq!(heads.n_heads(), k);

        let (b, t) = (2usize, 4usize);
        let tokens = tokenizer.encode("abcdabcdabcdabcd");
        let hidden = model.forward_hidden(&tokens[..b * t], b, t, true);
        assert_eq!(hidden.shape(), &[b * t, model.cfg.n_embd]);

        // 单头 logits 形状
        let h0 = hidden.gather_rows(&[0, 1]);
        assert_eq!(heads.logits(&h0, k - 1).shape(), &[2, model.cfg.vocab_size]);

        let loss = heads.loss(&hidden, &tokens[..b * t], b, t);
        assert_eq!(loss.rank(), 0, "MTP 损失应是标量");
        assert!(loss.item().is_finite() && loss.item() > 0.0);
        loss.backward();

        for (i, head) in heads.heads().iter().enumerate() {
            let gw = head.weight.grad();
            let gb = head.bias.grad();
            assert!(
                gw.iter().any(|&v| v != 0.0),
                "第 {i} 个头的权重梯度应非零"
            );
            assert!(
                gb.iter().any(|&v| v != 0.0),
                "第 {i} 个头的偏置梯度应非零"
            );
        }
    }

    /// MTP：多步训练后总损失下降（各头都在真的学）
    #[test]
    fn test_mtp_training_reduces_loss() {
        let tokenizer = Tokenizer::char("abcd");
        let model = tiny_model(tokenizer.vocab_size(), 3);
        let mut rng = Rng::new(4);
        let heads = MtpHeads::new(model.cfg.n_embd, model.cfg.vocab_size, 2, &mut rng);
        let (b, t) = (2usize, 6usize);
        let tokens = tokenizer.encode("abcdabcdabcd");

        // 主干与预测头一起训（MTP 的梯度要能回传到隐状态上）
        let mut params = Module::parameters(&model);
        params.extend(heads.parameters());
        let mut opt = crate::optim::AdamW::new(3e-3, params, 0.0);

        let mut history = Vec::new();
        for _ in 0..60 {
            opt.zero_grad();
            let hidden = model.forward_hidden(&tokens[..b * t], b, t, true);
            let loss = heads.loss(&hidden, &tokens[..b * t], b, t);
            loss.backward();
            opt.step();
            history.push(loss.item());
        }

        let first: f32 = history[..5].iter().sum::<f32>() / 5.0;
        let last: f32 = history[history.len() - 5..].iter().sum::<f32>() / 5.0;
        assert!(
            last < first * 0.8,
            "多步训练后 MTP 损失应明显下降：首 {first:.4} → 末 {last:.4}"
        );
    }

    /// MTP 草稿接入推测解码：贪心目标 + MTP 草稿，结果仍与逐 token 贪心一致
    #[test]
    fn test_mtp_drafter_serves_as_draft_path() {
        let tokenizer = Tokenizer::char("abcd");
        let model = tiny_model(tokenizer.vocab_size(), 4);
        let opts = greedy_opts();
        let mut rng = Rng::new(9);
        let mut drafter = MtpDrafter::new(&model, 4, KvOpts::on(0, None), &mut rng);

        let (spec, stats) = speculative_generate(
            &model,
            &tokenizer,
            "ab",
            20,
            4,
            &opts,
            KvOpts::on(0, None),
            &mut drafter,
            &mut rng,
        );
        let plain = generate(
            &model,
            &tokenizer,
            "ab",
            20,
            &opts,
            KvOpts::on(0, None),
            &mut Rng::new(9),
        );

        assert_eq!(spec, plain, "MTP 草稿路径同样必须无损");
        assert_eq!(stats.emitted, 20);
        assert_eq!(stats.target_forwards, stats.rounds);
        assert!(stats.rounds > 1, "应当跑了多轮");
    }
}
