//! 采样与文本生成（第 15 课）
//!
//! 模型输出的是"每个 token 的概率分布"，怎么从分布里选一个 token？
//! - 贪心：总是选概率最大的（容易重复、呆板）
//! - temperature：缩放概率分布的"锐度"（<1 更确定，>1 更随机）
//! - top-k：只在前 k 个概率最高的 token 里选
//! - top-p（nucleus）：在累积概率达到 p 的最小集合里选
//! - 重复惩罚（repetition penalty）：把最近出现过的 token 的 logit 压低，
//!   抑制"同一 token 反复出现"的自我强化循环
//!
//! 结合使用：重复惩罚 -> temperature 调整锐度 -> top-k/top-p 截断 -> 按概率随机抽样。

use crate::attention::KVCache;
use crate::model::Transformer;
use crate::quant::QBits;
use crate::rng::Rng;
use crate::tokenizer::{ImageTokens, Tokenizer};

/// 推理侧 KV cache 的开关与压缩设置（第 25 / 33 课）。
///
/// 打包成结构体而不是给 `generate` 再加三个标量参数：`generate` 的入参本来就多，
/// 而这三项是**同进退**的一组（不用缓存时后两项无意义）。
#[derive(Clone, Copy, Debug, Default)]
pub struct KvOpts {
    /// 是否使用 KV cache。关掉 = 每生成一个 token 都全量重算一次上下文
    /// （慢，但没有缓存内存；用来做数值对照）。
    pub enable: bool,
    /// Attention Sink：超窗丢弃时**永久保留最前面的 `sink` 个位置**（StreamingLLM）。
    /// 流式长文本生成必须开，否则丢掉序列开头后质量断崖下跌，见 [`crate::attention::KvCacheOpts::sink`]。
    pub sink: usize,
    /// 缓存量化位宽：`None` = f32，`Some(Int8/Int4)` = KIVI 式压缩（K 逐通道、V 逐 token）
    pub bits: Option<QBits>,
}

impl KvOpts {
    /// 不开缓存（等价于全量前向）
    pub fn off() -> Self {
        KvOpts {
            enable: false,
            ..KvOpts::default()
        }
    }

    /// 开缓存 + 指定 Attention Sink 与量化位宽
    pub fn on(sink: usize, bits: Option<QBits>) -> Self {
        KvOpts {
            enable: true,
            sink,
            bits,
        }
    }

    /// 按模型结构造出这一轮推理要用的缓存集合
    fn build(&self, model: &Transformer) -> Option<Vec<KVCache>> {
        self.enable
            .then(|| model.new_kv_cache_with(self.sink, self.bits))
    }
}

/// 采样超参数（temperature / top-k / top-p / 重复惩罚）
///
/// 打包成一个结构体而不是一长串函数参数：`generate` 的入参本来就多，
/// 再加两个标量会到 11 个，调用点也全部要跟着改。
#[derive(Clone, Copy, Debug)]
pub struct SampleOpts {
    /// 采样温度（>1 更随机，<1 更确定）
    pub temperature: f32,
    /// top-k：只从概率最高的 k 个里选（0 = 不限制）
    pub top_k: usize,
    /// top-p：累计概率到 p 的最小集合（1.0 = 不限制）
    pub top_p: f32,
    /// 重复惩罚系数：> 1.0 生效（1.0 = 关闭）
    pub repetition_penalty: f32,
    /// 重复惩罚回看窗口：只看最近 N 个 token（0 = 关闭）
    pub repetition_window: usize,
    /// 生成到其中任意一个字符串出现就停下，结果**不含**该标记。
    ///
    /// 用 `&'static [&'static str]` 而不是 `Vec<String>`：停止标记就是模板常量
    ///（见 [`crate::data::SFT_END`]），编译期就知道；这样 `SampleOpts` 仍是 `Copy`，
    /// 每次生成也不必为字符串分配内存。
    pub stop: &'static [&'static str],
}

impl Default for SampleOpts {
    fn default() -> Self {
        SampleOpts {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            repetition_window: 64,
            stop: &[],
        }
    }
}

/// 从 logits 分布中采样一个 token
///
/// - `recent`：参与重复惩罚的 token（调用方按窗口截好，通常是"最近 N 个已生成的 token"）。
///   传空切片即跳过惩罚。
pub fn sample_token(
    logits: &[f32],
    opts: &SampleOpts,
    recent: &[usize],
    rng: &mut Rng,
) -> usize {
    let (ids, probs) = probs_from_logits(logits, opts, recent);
    let mut u = rng.next_f32();
    for (i, p) in probs.iter().enumerate() {
        if u < *p {
            return ids[i];
        }
        u -= p;
    }
    ids.last().copied().unwrap_or(0)
}

/// 把一行 logits 变成「候选 token + 归一化概率」，即采样流程的全部预处理：
/// 重复惩罚 → temperature → top-k → top-p → softmax 归一化。
///
/// 抽成公开函数是为了让**任何**需要"这个分布长什么样"的地方都用同一份实现：
/// 推测解码要在草稿分布 q 与目标分布 p 之间做接受/拒绝校正（见
/// [`crate::speculative`]），校正的正确性完全建立在"p、q 是同一套预处理算出来的"
/// 之上。两边各写一遍预处理，一旦有一项（比如 top-p 的边界取法）不一致，
/// 拒绝采样就不再收敛到目标分布，而这种偏差在输出里表现为"偶尔串味"，极难定位。
///
/// 返回值按概率从高到低排列；`ids` 与 `probs` 等长，`probs` 之和为 1。
pub fn probs_from_logits(
    logits: &[f32],
    opts: &SampleOpts,
    recent: &[usize],
) -> (Vec<usize>, Vec<f32>) {
    // 1. 重复惩罚：压低最近出现过的 token。
    //    正值除以系数、负值乘以系数 —— 两种情况下数值都变小，因此更难被抽中。
    //    作用在**原始 logit** 上、再统一做 temperature 缩放，与 HuggingFace 的处理顺序一致。
    let mut penalized: Vec<f32> = logits.to_vec();
    if opts.repetition_penalty > 1.0 {
        for &id in recent {
            if let Some(l) = penalized.get_mut(id) {
                *l = if *l > 0.0 {
                    *l / opts.repetition_penalty
                } else {
                    *l * opts.repetition_penalty
                };
            }
        }
    }

    // 2. 除以 temperature 缩放（1e-5 是温度下限，防止 0 除）
    let inv_temp = 1.0 / opts.temperature.max(1e-5);
    for l in penalized.iter_mut() {
        *l *= inv_temp;
    }

    // 3. 按分数从高到低排序
    let mut items: Vec<(usize, f32)> = penalized.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 4. top-k：只保留前 k 个
    if opts.top_k > 0 && items.len() > opts.top_k {
        items.truncate(opts.top_k);
    }

    // 5. softmax 得到概率
    let max = items
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = items.iter().map(|(_, v)| (*v - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // 6. top-p：从高到低累加概率，直到超过 p，后面的全部丢弃
    if opts.top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = items.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= opts.top_p {
                keep = i + 1;
                break;
            }
        }
        items.truncate(keep);
        probs.truncate(keep);
        let s: f32 = probs.iter().sum();
        for p in probs.iter_mut() {
            *p /= s;
        }
    }

    (items.into_iter().map(|(i, _)| i).collect(), probs)
}

/// 在**给定的离散分布**上采样（`probs` 不必有序，内部不重排）。
///
/// 与 [`sample_token`] 的区别：那个从 logits 出发、含整套预处理；这个只做"掷一次骰子"。
/// 推测解码的拒绝采样要把"残差分布 `max(0, p − q)`"归一化后再抽一次，
/// 那一步没有 logits 可言，只能直接给概率。
pub fn sample_from_probs(ids: &[usize], probs: &[f32], rng: &mut Rng) -> usize {
    debug_assert_eq!(ids.len(), probs.len());
    let mut u = rng.next_f32();
    for (i, p) in probs.iter().enumerate() {
        if u < *p {
            return ids[i];
        }
        u -= p;
    }
    ids.last().copied().unwrap_or(0)
}

/// 这一轮生成是怎么停下来的
///
/// 三个结束条件（见 [`generate_with_reason`]）各对应一个变体。它存在的意义：
/// 「模型一个字都没说」与「程序根本没跑」在终端上都表现为一行空白，只有把收尾原因
/// 记下来，空白输出才可解释（聊天模式把原因写进日志，见 `cmd_chat`）。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    /// 采到 EOS：模型自己学出来的结束符号
    Eos,
    /// 命中 [`SampleOpts::stop`] 里的停止标记（模板标记兜底）
    StopMark(&'static str),
    /// 到达 `max_new` 上限，不是提前收尾
    MaxNew,
}

/// 生成结果：全文、本轮新生成的部分、收尾原因
#[derive(Clone, Debug)]
pub struct GenOutput {
    /// 完整文本（prompt + 新生成部分），与 [`generate`] 的返回值一致
    pub text: String,
    /// 本轮新生成的部分：已丢掉特殊 token、已按停止标记截断、**不含** prompt
    pub generated: String,
    /// 收尾原因
    pub reason: StopReason,
}

/// 生成文本，只取全文（prompt + 新生成部分）。
///
/// 需要区分"这一轮生成了什么、为什么停下"时用 [`generate_with_reason`]。
pub fn generate(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    opts: &SampleOpts,
    kv: KvOpts,
    rng: &mut Rng,
) -> String {
    generate_with_reason(model, tokenizer, prompt, max_new, opts, kv, rng).text
}

/// 生成状态机：把生成循环拆成可单步推进的结构，供**流式输出**（HTTP SSE）使用。
///
/// 非流式路径 [`generate_with_reason`] 就是 `new + run + into_output` 的封装，
/// 两者执行完全相同的前向/采样序列，同种子下结果逐字节一致。
///
/// 流式用法：循环 `step()`，每步之后用 [`Generator::next_text`] 取"可安全发出"的增量，
/// 收尾后（`step()` 返回 `false`）用 [`Generator::flush`] 补发尾巴，最后 [`Generator::into_output`] 拿结果。
///
/// 流式不能把 `decode` 出来的字节直接全发出去——有两处会被后续"追回"：
/// 1. 命中停止标记时最终结果是 `raw[..rel].trim_end()`，尾部空白与标记本身都不出现，
///    先发出去就收不回来；
/// 2. 停止标记可能**横跨两次采样**（这次 decode 出「用户」，下次补上「：」），
///    已发出的字节必须保证不超过标记的起始位置。
///
/// 因此 [`Generator::next_text`] 按 [`safe_len`] 扣掉"尾部空白 + 停止标记的真前缀"再发，
/// 被扣住的部分在收尾时由 [`Generator::flush`] 随最终结果一次性补齐（此刻结果已定，不再变动）。
pub struct Generator<'m, 'r> {
    model: &'m Transformer,
    tokenizer: &'m Tokenizer,
    opts: SampleOpts,
    eos: Option<usize>,
    /// 字节级 BPE 的 UTF-8 约束词表（char 分词器为 `None`）
    vocab_bytes: Option<&'m [Vec<u8>]>,
    /// BOS + prompt + 已生成的 token
    ids: Vec<usize>,
    /// 生成段的起点：这之前的 token 全是「BOS + prompt」
    n_prompt: usize,
    prompt_text: String,
    cache: Option<Vec<KVCache>>,
    rng: &'r mut Rng,
    masked: Vec<f32>,
    /// 还能生成多少个新 token（到 0 即以 [`StopReason::MaxNew`] 收尾）
    remaining: usize,
    /// 已通过 [`Generator::next_text`] 发出的字节数（生成段内偏移）
    emitted: usize,
    /// 收尾原因；`Some` 表示已收尾
    reason: Option<StopReason>,
    /// 收尾时定稿的生成段（收尾前为空，[`Generator::into_output`] 兜底补算）
    generated: String,
    /// 图片 token 区间（分词器未挂载图片区间时为 `None`；K=0 时采样不受约束）
    image: Option<ImageTokens>,
    /// 图内状态机：`None` = 图外，`Some(n)` = 图内已采 n 个码本 token（n < P）
    img_seg: Option<usize>,
    /// 图片理解的输入像素（CHW、[-1,1]、按 `vision.image_size` 加载）。
    /// 只在**带 KV cache 的首步（prefill）**与**全量无 cache 模式**下注入——
    /// 增量步没有新图像（占位符的 K/V 已进缓存），传 None 即可
    pixels: Option<Vec<f32>>,
}

impl<'m, 'r> Generator<'m, 'r> {
    /// 建立生成状态：编码 prompt（前补 BOS）、建 KV 缓存、准备采样状态。
    ///
    /// 参数与 [`generate_with_reason`] 一致。
    pub fn new(
        model: &'m Transformer,
        tokenizer: &'m Tokenizer,
        prompt: &str,
        max_new: usize,
        opts: &SampleOpts,
        kv: KvOpts,
        rng: &'r mut Rng,
    ) -> Self {
        // prompt 前面补 BOS：训练时每篇文档都以 BOS 开头（见 `data::encode_document`），
        // 推理从 BOS 起头才与训练分布一致。老分词器没有 BOS，行为不变。
        let mut ids = match tokenizer.bos_id() {
            Some(bos) => vec![bos],
            None => Vec::new(),
        };
        ids.extend(tokenizer.encode(prompt));
        if ids.is_empty() {
            ids.push(0); // 空 prompt 且无 BOS：先喂一个 token，避免 0 长度上下文导致下标下溢
        }
        // 生成段的起点：这之前的 token 全是「BOS + prompt」。`decode` 会在特殊 token（BOS）
        // 处断开，所以前后两段分别 decode 再拼起来与整体 decode 结果一致，而调用方拿到的
        // `generated` 就是纯粹的"本轮生成"，不必再按 `prompt.len()` 的字节偏移去切全文。
        let n_prompt = ids.len();
        let prompt_text = tokenizer.decode(&ids[..n_prompt]);
        // 图片区间：只有挂载了码本（K>0）且有占位符（P>0）才启用受约束采样
        let image = tokenizer
            .image_tokens()
            .filter(|i| i.cb_count > 0 && i.count > 0);
        // prompt 可能含 <|image|> 展开段：扫一遍状态机，确定起点处是否在图内
        let mut img_seg = None;
        for &id in &ids {
            img_seg = advance_img_seg(img_seg, image, id);
        }
        Generator {
            model,
            tokenizer,
            opts: *opts,
            eos: tokenizer.eos_id(),
            vocab_bytes: tokenizer.vocab_bytes(),
            ids,
            n_prompt,
            prompt_text,
            cache: kv.build(model),
            rng,
            masked: Vec::new(),
            remaining: max_new,
            emitted: 0,
            reason: None,
            generated: String::new(),
            image,
            img_seg,
            pixels: None,
        }
    }

    /// 注入图片理解的输入像素（须在**首次 [`step`] 之前**调用）。
    ///
    /// prompt 必须含 `<|image|>`（展开为 P 个占位符，见 [`crate::tokenizer::IMAGE_LITERAL`]）：
    /// 前向时视觉塔的 patch 特征会原位覆写这些占位符（[`Transformer::forward_mm`]）。
    /// 没配视觉塔的 checkpoint 调用会在前向时断言失败。
    ///
    /// [`Transformer::forward_mm`]: crate::model::Transformer::forward_mm
    pub fn with_pixels(mut self, pixels: Vec<f32>) -> Self {
        self.pixels = Some(pixels);
        self
    }

    /// 推进一步：一次前向 + 一次采样。
    ///
    /// 返回 `true` = 还在生成中（可接着调 [`Generator::next_text`] 取增量）；
    /// 返回 `false` = 已收尾（原因见内部 `reason`，最终结果用 [`Generator::into_output`] 取）。
    /// 已收尾后再调用恒返回 `false`。
    pub fn step(&mut self) -> bool {
        if self.reason.is_some() {
            return false;
        }
        // 上限用尽：与原循环跑满 `0..max_new` 后的出口一致（生成段不 trim，原样交出）
        if self.remaining == 0 {
            self.generated = self.tokenizer.decode(&self.ids[self.n_prompt..]);
            self.reason = Some(StopReason::MaxNew);
            return false;
        }
        self.remaining -= 1;
        let block_size = self.model.cfg.block_size;

        // 只保留最近的 block_size 个 token（两种模式都必须遵守的上下文上限）
        let start = self.ids.len().saturating_sub(block_size);
        let ctx = &self.ids[start..];

        // 推理不需要反向：no_grad 下不挂计算图、不分配梯度缓冲
        // （把 cache 提成局部变量：闭包按字段精确捕获，避免与下面的 `self.ids` 打架）
        let model = self.model;
        let cache = self.cache.as_mut();
        // 图片理解：带 cache 时只在首步（prefill）注入像素——占位符的 K/V 从此进缓存；
        // 无 cache 全量模式每步都重算整个上下文，占位符每次都要重新覆写。
        // 只有当 ctx 里**恰好凑齐 P 个占位符**才注入：滑窗把图挤出窗口时 forward_mm
        // 的数量断言会炸，此时退化成普通 embedding（图片已看不见了，与纯文本路径一致）
        let (ph_first, ph_count) = match &model.cfg.vision {
            Some(v) => (v.ph_first, v.patch_count()),
            None => (0, 0),
        };
        let n_ph = ctx
            .iter()
            .filter(|&&id| id >= ph_first && id < ph_first + ph_count)
            .count();
        let inject = self.pixels.is_some() && ph_count > 0 && n_ph == ph_count;
        let px = if inject { self.pixels.as_deref() } else { None };
        let logits = crate::tensor::no_grad(|| {
            if let Some(c) = cache {
                // 首次：缓存为空，把整个 prompt 喂进去（顺便填充缓存）
                // 之后：每步只前向最新 1 个 token，历史 K/V 从缓存取
                if c[0].seq_len() == 0 {
                    model.forward_mm(ctx, 1, ctx.len(), if inject { px } else { None }, Some(c), false)
                } else {
                    model.forward(&self.ids[self.ids.len() - 1..], 1, 1, Some(c), false)
                }
            } else {
                // 全量模式：每次把整个上下文重新算一遍（慢，但没有 cache 内存）
                model.forward_mm(ctx, 1, ctx.len(), if inject { px } else { None }, None, false)
            }
        });

        // 取最后一个位置的 logits
        let v = model.cfg.vocab_size;
        let n = logits.numel();
        let last_row = &logits.data()[n - v..];
        // 只允许"接上后仍是合法 UTF-8 前缀"的 token，否则会拼出半个字符；
        // 图片段（start→码本×P→end）还要按状态机收紧采样域：两套掩码叠加
        // （UTF-8 掩码只作用于内容词表，图片 token 落在 `get` 的 None 分支不受影响）。
        let im = self.image;
        let need_mask = self.vocab_bytes.is_some() || im.is_some();
        let row: &[f32] = if need_mask {
            self.masked.clear();
            self.masked.extend_from_slice(last_row);
            if let Some(vocab) = self.vocab_bytes {
                let pending = pending_tail(vocab, &self.ids);
                mask_illegal_utf8(&mut self.masked, vocab, &pending);
            }
            if let Some(im) = im {
                mask_image(&mut self.masked, im, self.img_seg);
            }
            &self.masked
        } else {
            last_row
        };
        // 重复惩罚的回看窗口：只看最近 N 个 token（含 prompt），更早的不再计入。
        // 窗口过大时高频 token 会被持续压低，可能影响语句的连贯性。
        // 图段内跳过惩罚：VQ 码本里重复是常态，按出现次数累乘的惩罚会把常见码打崩。
        let recent = if self.opts.repetition_window == 0 || self.img_seg.is_some() {
            &[][..]
        } else {
            let start = self.ids.len().saturating_sub(self.opts.repetition_window);
            &self.ids[start..]
        };
        let next = sample_token(row, &self.opts, recent, self.rng);
        self.ids.push(next);
        // 推进图内状态机：图外遇 start 进图，图内计数，遇 end 出图
        self.img_seg = advance_img_seg(self.img_seg, self.image, next);

        // 采到 EOS 就收：这是模型自己学出来的结束符号（训练时每段序列末尾都带它），
        // 比"等某个字符组合出现"可靠得多。EOS 本身不进结果（`decode` 跳过特殊 token）。
        if self.eos == Some(next) {
            self.generated = self.tokenizer.decode(&self.ids[self.n_prompt..self.ids.len() - 1]);
            self.reason = Some(StopReason::Eos);
            return false;
        }

        // 命中停止标记就收：SFT 模板下模型答完会自己吐「。。」，
        // 不截断的话它会顺着模板继续编下一轮提问（"回答后面跟提问"在训练语料里到处都是）。
        // 这一条是给**没学过 EOS 的老权重**兜底的路径。
        //
        // 只在**新生成的部分**里查找：prompt 自己就含「用户：」（模板的一部分），
        // 对整个字符串搜索会立刻在 prompt 里命中，结果返回空白。
        // 每步解码一次生成段：长度上限就是 block_size，这点开销远小于一次前向。
        if !self.opts.stop.is_empty() {
            let raw = self.tokenizer.decode(&self.ids[self.n_prompt..]);
            if let Some((rel, mark)) = self
                .opts
                .stop
                .iter()
                .filter_map(|s| raw.find(s).map(|i| (i, *s)))
                .min_by_key(|(i, _)| *i)
            {
                // `rel` 来自 `find`，一定落在字符边界上
                self.generated = raw[..rel].trim_end().to_string();
                self.reason = Some(StopReason::StopMark(mark));
                return false;
            }
        }
        true
    }

    /// 取一段**可以安全发给客户端**的增量（上次发出之后的新内容）。
    ///
    /// 返回 `None` = 这一步还没有可发的字节（例如尾部正卡着半个停止标记）。
    /// 已收尾后恒返回 `None`，尾巴请用 [`Generator::flush`] 补。
    pub fn next_text(&mut self) -> Option<String> {
        if self.reason.is_some() {
            return None;
        }
        let raw = self.tokenizer.decode(&self.ids[self.n_prompt..]);
        let n = safe_len(&raw, self.opts.stop);
        // 安全长度按构造不会倒退，但守住这道闸：宁可这步不发，也绝不回吐已发字节
        if n <= self.emitted {
            return None;
        }
        let delta = raw[self.emitted..n].to_string();
        self.emitted = n;
        Some(delta)
    }

    /// 收尾后补发剩余尾巴（最终结果里还没发出的部分）。
    ///
    /// 必须在 `step()` 返回 `false` **之后**调用；收尾前调用返回空串。
    pub fn flush(&mut self) -> String {
        if self.reason.is_none() {
            return String::new();
        }
        let g = &self.generated;
        let mut start = self.emitted.min(g.len());
        while start > 0 && !g.is_char_boundary(start) {
            start -= 1; // 兜底：decode 产物按字节切出的偏移必须落在字符边界
        }
        let tail = g[start..].to_string();
        self.emitted = g.len();
        tail
    }

    /// 一直推进到收尾（非流式用法）。
    pub fn run(&mut self) {
        while self.step() {}
    }

    /// 已采样出的新 token 数（含刚收尾时补上的那个 EOS/停止标记）。
    ///
    /// 服务端要把它当 `usage.completion_tokens` 回给客户端，
    /// 这里的计数与 `prompt_tokens` 同源（都是 `ids` 的长度差），两者相加即总长。
    pub fn generated_tokens(&self) -> usize {
        self.ids.len().saturating_sub(self.n_prompt)
    }

    /// 输入侧 token 数（BOS + prompt），生成前生成后都不变。
    ///
    /// 服务端把它当 `usage.prompt_tokens` 回给客户端。
    pub fn prompt_tokens(&self) -> usize {
        self.n_prompt
    }

    /// 提取生成段中**完整的图片码本序列**（每张图 = P 个码本下标），供 VQ-VAE 解码成像素。
    ///
    /// 只认「start + P 个码本 token + end」的完整段：残缺段（截断在图中间）丢弃。
    /// 分词器没挂载码本（K=0）时恒返回空。
    pub fn generated_images(&self) -> Vec<Vec<usize>> {
        let Some(im) = self.image else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let segs = &self.ids[self.n_prompt..];
        let mut i = 0;
        while i < segs.len() {
            if segs[i] == im.start && segs.len() > i + im.count + 1 && segs[i + im.count + 1] == im.end {
                let seg = &segs[i + 1..=i + im.count];
                if let Some(codes) = seg.iter().map(|&id| im.cb_index(id)).collect::<Option<Vec<_>>>()
                {
                    out.push(codes);
                    i += im.count + 2; // start + P 个码本 + end
                    continue;
                }
            }
            i += 1;
        }
        out
    }

    /// 消费状态机，取出最终结果（[`GenOutput`]）。
    pub fn into_output(self) -> GenOutput {
        let Generator {
            tokenizer,
            ids,
            n_prompt,
            prompt_text,
            reason,
            generated,
            ..
        } = self;
        // 未收尾就取结果（调用方没跑完）：按当前已生成的 token 算，原因记 MaxNew
        let (reason, generated) = match reason {
            Some(r) => (r, generated),
            None => (StopReason::MaxNew, tokenizer.decode(&ids[n_prompt..])),
        };
        GenOutput {
            text: format!("{prompt_text}{generated}"),
            generated,
            reason,
        }
    }
}

/// 流式安全发出的字节长度：见 [`Generator`] 的文档。返回值落在 UTF-8 字符边界上。
///
/// - `stops` 为空：直接全发（此路径的收尾结果从不 trim，也没有标记可截断）；
/// - 否则扣两样：① 尾部可能成为某标记**开头**的片段（标记可横跨两次采样），
///   ② 扣完之后的尾部空白（命中标记时最终结果要 `trim_end`，空白会消失）。
///
/// 关键不变式：`safe_len ≤ 标记起始位置`。完整标记在 `step()` 里就已检出并收尾，
/// 所以这里只会遇到"真前缀"；而任何未来才补全的标记，其起始位置之前的部分
/// 必然是当前 `raw` 的尾部真前缀（生成是**只追加**的），于是被 ① 扣住——
/// 因此已发出的字节永远不会越过最终的 `rel`，`flush` 的尾巴永不为空差。
fn safe_len(raw: &str, stops: &[&str]) -> usize {
    if stops.is_empty() {
        return raw.len();
    }
    let mut cut = raw.len();
    for mark in stops {
        // 只看真前缀（1..len）：完整标记意味着 step() 已收尾，走不到这里
        for k in 1..mark.len() {
            if !mark.is_char_boundary(k) {
                continue;
            }
            if raw.len() >= k && raw.ends_with(&mark[..k]) {
                cut = (raw.len() - k).min(cut);
            }
        }
    }
    // ② 标记起始位置之前的空白同样会被最终的 trim_end 掉，跟在前缀后面一起扣
    raw[..cut].trim_end().len()
}

/// 生成文本，并给出收尾原因与本轮新生成的部分。
///
/// - prompt: 起始文本
/// - max_new: 最多生成多少个新 token
/// - opts: 采样超参数（含重复惩罚与停止标记 `opts.stop`）
/// - kv: KV cache 开关与压缩设置（第 25 / 33 课）
///
/// 结束条件有三个，任何一个命中就收：
/// 1. 采到 EOS token（分词器带特殊 token 时；这是模型真正学过的结束符号）
/// 2. `opts.stop` 里的字符串出现（模板停止标记，给没学 EOS 的老权重兜底）
/// 3. 生成到 `max_new` 个 token
///
/// [`GenOutput::generated`] 单独给出生成段，调用方不必再用 `prompt.len()` 去切全文：
/// 那个字节偏移在"生成段为空"时会把整段切成空串，于是"模型没说话"与"程序出错"
/// 长得一模一样。偏移本身还与"`decode` 丢掉特殊 token"这一实现细节耦合。
///
/// 本函数是 [`Generator`]（状态机）的非流式封装：同种子下与手动
/// `new + run + into_output` 逐字节一致。需要边生成边输出（HTTP SSE）时直接用 [`Generator`]。
pub fn generate_with_reason(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    opts: &SampleOpts,
    kv: KvOpts,
    rng: &mut Rng,
) -> GenOutput {
    let mut machine = Generator::new(model, tokenizer, prompt, max_new, opts, kv, rng);
    machine.run();
    machine.into_output()
}

/// Beam Search 生成：维护 `beam_size` 个候选序列，每步扩展后保留 top-k。
///
/// 与采样（temperature + top-k/top-p）的区别：
/// - 采样是随机的，每次生成不同
/// - Beam Search 是确定性的（给定 seed），总选择全局最优的 k 条路径
/// - 生成质量更高，但多样性更低
///
/// 返回按长度惩罚选出的最佳序列。
///
/// - beam_size: 束宽（通常 4-10），越大搜索越充分，但越慢
/// - length_penalty: 长度惩罚指数 α（0 = 不惩罚，>0 偏好长序列，<0 偏好短序列）
///   最终分数 = log_prob / len^α（Google NMT 的公式）
/// - kv: KV cache 设置。开启后**每条 beam 持有自己的缓存**：束内所有路径共享同一段
///   前缀，扩展时克隆父路径的缓存（[`KVCache::fork`]），于是每步每条路径只前向 1 个
///   token，而不再是"每条路径重算整段上下文"。缓存窗口就是 `block_size`，所以它与
///   全量重算的可见范围完全一致（RoPE 只看相对距离，位置整体平移不影响打分）。
///
/// 累加的必须是 **log 概率**而不是原始 logit：log_softmax 会把 logit 归一化成一个
/// 合法分布的对数（各项 ≤ 0，序列越长和越小），除以 `len^α` 才有"平均每 token 的对数概率"
/// 的含义。直接累加原始 logit 得到的数没有归一化，剪枝会系统性偏向"logit 整体偏大"的
/// 路径，长度惩罚也会得到与注释相反的效果。
pub fn beam_search(
    model: &Transformer,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    beam_size: usize,
    length_penalty: f32,
    kv: KvOpts,
) -> String {
    assert!(beam_size >= 1, "beam_size 必须 >= 1");
    let vocab_size = model.cfg.vocab_size;
    // 与 generate 一致：prompt 前补 BOS（训练时文档以 BOS 开头）
    let mut prompt_ids: Vec<usize> = tokenizer.bos_id().into_iter().collect();
    prompt_ids.extend(tokenizer.encode(prompt));
    if prompt_ids.is_empty() {
        prompt_ids.push(0);
    }

    // 每个候选路径：token 序列 + 累计对数概率 + 它自己的缓存 + 是否已收尾（吐过 EOS）
    struct Beam {
        ids: Vec<usize>,
        score: f64,
        cache: Option<Vec<KVCache>>,
        finished: bool,
    }

    let mut base_cache = kv.build(model);
    // 每条路径的缓存都要从"同一个 prompt 前缀"出发，所以先做一次 prefill 再克隆。
    // prefill 只喂到**倒数第二个** token 为止：剩下那一个交给主循环统一处理
    // （循环里"把最后一个 token 喂进缓存换出下一个 token 的分布"是同一套动作，
    // 少一个特例就少一处出错的地方）。prompt 只有一个 token 时不用 prefill。
    if let Some(cache) = base_cache.as_mut() {
        if prompt_ids.len() >= 2 {
            let start = prompt_ids.len().saturating_sub(model.cfg.block_size);
            let prefill = &prompt_ids[start..prompt_ids.len() - 1];
            crate::tensor::no_grad(|| model.forward(prefill, 1, prefill.len(), Some(cache), false));
        }
    }
    let only_cache = || base_cache.as_ref().map(|c| c.iter().map(|x| x.fork()).collect());

    let mut beams: Vec<Beam> = vec![Beam {
        ids: prompt_ids,
        score: 0.0,
        cache: only_cache(),
        finished: false,
    }];
    let eos = tokenizer.eos_id();
    // 字节级 BPE 的 UTF-8 约束（同 generate：排除"半个汉字"的 token）
    let vocab_bytes = tokenizer.vocab_bytes();
    let mut masked: Vec<f32> = Vec::new();

    for _step in 0..max_new {
        let mut candidates: Vec<Beam> = Vec::new();

        for mut beam in beams {
            // 已收尾的路径不再扩展，但保留下来参与最终比较
            if beam.finished {
                candidates.push(beam);
                continue;
            }

            // 拿到"下一个 token"的分布：把这条路径的最后一个 token 喂进缓存。
            // 缓存覆盖 `ids[..len-1]`，喂完正好补成 `ids[..len]`，分布就是"接在 ids 之后"。
            let row: Vec<f32> = {
                let last = *beam.ids.last().expect("beam 至少含 prompt 一个 token");
                let logits = match beam.cache.as_mut() {
                    Some(cache) => crate::tensor::no_grad(|| {
                        model.forward(&[last], 1, 1, Some(cache), false)
                    }),
                    // 不用缓存：整段上下文重算（慢，但没有缓存内存）
                    None => {
                        let start = beam.ids.len().saturating_sub(model.cfg.block_size);
                        let ctx = &beam.ids[start..];
                        crate::tensor::no_grad(|| model.forward(ctx, 1, ctx.len(), None, false))
                    }
                };
                let n = logits.numel();
                let last_row = logits.data()[n - vocab_size..].to_vec();
                match vocab_bytes {
                    Some(vocab) => {
                        masked.clear();
                        masked.extend_from_slice(&last_row);
                        mask_illegal_utf8(&mut masked, vocab, &pending_tail(vocab, &beam.ids));
                        masked.clone()
                    }
                    None => last_row,
                }
            };

            // 先 log_softmax 成对数概率再累加（用 max 减去最大值做数值稳定化）。
            // 被屏蔽的 token 是 -inf，`exp(-inf - max) = 0`，不参与配分函数也不影响结果。
            let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let log_z: f64 = if max_logit.is_finite() {
                max_logit as f64
                    + row
                        .iter()
                        .map(|&l| ((l - max_logit) as f64).exp())
                        .sum::<f64>()
                        .ln()
            } else {
                // 整个词表都被屏蔽（理论上不会发生）：下面 scored 必为空，
                // 这里给 0 只是避免 `-inf - -inf` 算出 NaN。
                0.0
            };

            // 找 top-beam_size 个候选 token
            let mut scored: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, l)| l.is_finite()) // 被屏蔽的 -inf 直接跳过
                .map(|(i, &l)| (i, beam.score + (l as f64 - log_z)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(beam_size);

            for (tok, new_score) in scored {
                let mut ids = beam.ids.clone();
                ids.push(tok);
                // 克隆父路径的缓存：兄弟路径的前缀完全相同，缓存自然也一样
                let cache = beam.cache.as_ref().map(|c| c.iter().map(|x| x.fork()).collect());
                candidates.push(Beam {
                    ids,
                    score: new_score,
                    cache,
                    finished: eos == Some(tok),
                });
            }
        }

        // 按分数排序，保留 top-beam_size
        candidates.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(beam_size);
        beams = candidates;

        // 所有 beam 都收尾了就提前停
        if beams.iter().all(|b| b.finished) {
            break;
        }
    }

    // 按长度惩罚后的分数选最佳：`log_prob / len^α`。
    // log_prob 是负数、且序列越长越负，除以 `len^α`（α>0）相当于取"平均每 token 的对数概率"，
    // 因此 α>0 会偏好长序列——这与参数文档一致（之前的实现累加原始 logit，符号是正的，
    // 除以 len^α 反而偏好短序列，方向是反的）。
    let best = beams
        .iter()
        .max_by(|a, b| {
            let sa = a.score / (a.ids.len() as f64).powf(length_penalty as f64);
            let sb = b.score / (b.ids.len() as f64).powf(length_penalty as f64);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("beams 至少有一条（prompt 本身）");

    // EOS 不进结果：它在 id 序列里是"结束符"，不是内容
    let ids: &[usize] = match best.ids.last() {
        Some(&last) if eos == Some(last) => &best.ids[..best.ids.len() - 1],
        _ => &best.ids,
    };
    tokenizer.decode(ids)
}

// ==================== 生成时的 UTF-8 约束 ====================

/// 把"接上后不再是合法 UTF-8 前缀"的 token 的 logit 置为 `-inf`，采样时自然抽不到。
///
/// 字节级 BPE 的词表里有"半个汉字"（如只含 `E4` 或 `B8`）。训练时它们能拼回原文，
/// 但推理是模型自由采样的，碎片一旦乱序就会拼出非法 UTF-8（表现为丢字、乱码）。
/// 这里在采样前剪掉这些 token：数学上等价于把这些 token 的概率设为 0 后重新归一化。
///
/// `pending` 是当前字节流末尾未拼完的字节（见 [`pending_tail`]）。
///
/// 公开到 crate 内是给推测解码用的（见 [`crate::speculative`]）：它逐位验证草稿时
/// 必须用**同一套**掩码，否则目标分布 p 与草稿分布 q 的定义域不一致，
/// 拒绝采样就不再收敛到目标分布。
pub(crate) fn mask_illegal_utf8(logits: &mut [f32], vocab: &[Vec<u8>], pending: &[u8]) {
    let mut buf: Vec<u8> = Vec::with_capacity(pending.len() + 4);
    for (id, logit) in logits.iter_mut().enumerate() {
        let Some(tok) = vocab.get(id) else { continue };
        buf.clear();
        buf.extend_from_slice(pending);
        buf.extend_from_slice(tok);
        if crate::tokenizer::utf8_pending(&buf).is_none() {
            *logit = f32::NEG_INFINITY;
        }
    }
}

/// 算出已生成 token 序列末尾"未拼完的字节"（最多 3 字节，UTF-8 单字符最长 4 字节）。
///
/// 只需回看末尾几个 token：取满 8 个（每个至少 1 字节）必然覆盖最近的字符边界。
///
/// 同样公开到 crate 内给推测解码复用（见 [`mask_illegal_utf8`] 的说明）。
pub(crate) fn pending_tail(vocab: &[Vec<u8>], ids: &[usize]) -> Vec<u8> {
    let mut buf: Vec<u8> = Vec::new();
    for &id in ids.iter().rev().take(8) {
        let Some(tok) = vocab.get(id) else { continue };
        let mut head = tok.clone();
        head.extend_from_slice(&buf);
        buf = head;
    }
    // buf 的开头可能落在上一个字符中间（token 本身可以从非字符边界开始），
    // 因此从前往后找第一个能解析成合法前缀的位置。
    for j in 0..buf.len() {
        if let Some(rest) = crate::tokenizer::utf8_pending(&buf[j..]) {
            return rest.to_vec();
        }
    }
    Vec::new()
}

/// 图片段状态机推进：图外遇 start 进图（计 0），图内每遇占位/码本 token 计数 +1，
/// 遇 end 出图；图内出现其他 token 保持状态不变（掩码已挡住，不会真的发生）。
fn advance_img_seg(seg: Option<usize>, im: Option<ImageTokens>, id: usize) -> Option<usize> {
    let im = im?;
    match seg {
        None => (id == im.start).then_some(0),
        Some(n) => {
            if id == im.end {
                None
            } else if (im.ph_first..im.ph_first + im.count).contains(&id)
                || (im.cb_first..im.cb_first + im.cb_count).contains(&id)
            {
                Some(n + 1)
            } else {
                Some(n)
            }
        }
    }
}

/// 图片段的采样域约束（叠加在 UTF-8 掩码之上）：
/// - 图外：屏蔽占位符段 / 码本段 / end——模型不能凭空吐图片 token，必须先出 start；
/// - 图内未满 P 个：只留码本段（EOS、文本全屏蔽，图必须完整采完）；
/// - 图内已满 P 个：只留 end 收尾。
fn mask_image(logits: &mut [f32], im: ImageTokens, seg: Option<usize>) {
    let neg = f32::NEG_INFINITY;
    match seg {
        None => {
            for id in im.ph_first..im.ph_first + im.count {
                if let Some(l) = logits.get_mut(id) {
                    *l = neg;
                }
            }
            for id in im.cb_first..im.cb_first + im.cb_count {
                if let Some(l) = logits.get_mut(id) {
                    *l = neg;
                }
            }
            if let Some(l) = logits.get_mut(im.end) {
                *l = neg;
            }
        }
        Some(n) if n >= im.count => {
            for (id, l) in logits.iter_mut().enumerate() {
                if id != im.end {
                    *l = neg;
                }
            }
        }
        Some(_) => {
            for (id, l) in logits.iter_mut().enumerate() {
                if !(im.cb_first..im.cb_first + im.cb_count).contains(&id) {
                    *l = neg;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TransformerConfig;

    fn tiny_setup() -> (Transformer, Tokenizer) {
        let corpus = "the quick brown fox jumps over the lazy dog, and then runs away.";
        let tokenizer = Tokenizer::char(corpus);
        let mut rng = Rng::new(7);
        let model = Transformer::new(TransformerConfig::tiny(tokenizer.vocab_size()), &mut rng);
        (model, tokenizer)
    }

    fn opts() -> SampleOpts {
        SampleOpts {
            temperature: 0.8,
            top_k: 10,
            top_p: 0.9,
            repetition_penalty: 1.1,
            repetition_window: 64,
            stop: &[],
        }
    }

    /// KV cache 只改注意力的计算方式（增量 vs 全量），不改生成分布：
    /// 同 prompt + 同种子、且总长不超 `block_size` 时，两种模式应逐 token 一致。
    #[test]
    fn test_kv_cache_generate_matches_full() {
        let (model, tokenizer) = tiny_setup();
        // tiny 的 block_size = 32：prompt 3 + 生成 20 = 23 ≤ 32，全程在缓存窗口内
        let mut rng_full = Rng::new(1234);
        let full = generate(&model, &tokenizer, "the", 20, &opts(), KvOpts::off(), &mut rng_full);
        let mut rng_cache = Rng::new(1234);
        let cached = generate(&model, &tokenizer, "the", 20, &opts(), KvOpts::on(0, None), &mut rng_cache);

        assert_eq!(full, cached, "窗口内的 KV cache 生成应与全量前向逐 token 一致");
    }

    /// 超出缓存窗口后，KV cache 模式靠滑动窗口继续生成，不再提前结束：
    /// 同一个 prompt 加不加 `--no-kv-cache` 都应产出同样多的 token。
    ///
    /// 这里只断言长度，不断言内容：超过上下文窗口后，"增量 + 滑动窗口"（标准推理语义）
    /// 与"截断后重算"本就不是同一个函数——重算会把窗口内各位置在浅层可见的上下文一并砍掉，
    /// 深层 K/V 随之不同。机制层面的严格等价由
    /// `model::tests::test_kv_cache_sliding_window_matches_full_window_forward` 验证。
    #[test]
    fn test_kv_cache_generate_beyond_window_is_not_truncated() {
        let (model, tokenizer) = tiny_setup();
        let mut rng_full = Rng::new(1234);
        let full = generate(&model, &tokenizer, "the", 60, &opts(), KvOpts::off(), &mut rng_full);
        let mut rng_cache = Rng::new(1234);
        let cached = generate(&model, &tokenizer, "the", 60, &opts(), KvOpts::on(0, None), &mut rng_cache);

        assert_eq!(
            full.chars().count(),
            cached.chars().count(),
            "滑动窗口下 cache 模式不应再提前结束：\nfull   = {full}\ncached = {cached}"
        );
    }

    /// 流式（`step + next_text + flush`）与非流式（`generate_with_reason`）同种子下
    /// 必须逐字节一致：HTTP SSE "边聊边出的字"与非流式"一次性返回的字"是同一段，
    /// 客户端无论用哪种方式拿到的最终文本都相同（覆盖 EOS / 停止标记 / 上限三种收尾）。
    #[test]
    fn test_streaming_equals_non_streaming() {
        let (model, tokenizer) = tiny_setup();
        fn run_case(
            model: &Transformer,
            tokenizer: &Tokenizer,
            prompt: &str,
            max_new: usize,
            stop: &'static [&'static str],
        ) {
            let mut o = opts();
            o.stop = stop;

            let mut rng_plain = Rng::new(77);
            let plain = generate_with_reason(model, tokenizer, prompt, max_new, &o, KvOpts::off(), &mut rng_plain);

            let mut rng_s = Rng::new(77);
            let mut g = Generator::new(model, tokenizer, prompt, max_new, &o, KvOpts::off(), &mut rng_s);
            let mut stream = String::new();
            while g.step() {
                if let Some(t) = g.next_text() {
                    stream.push_str(&t);
                }
            }
            stream.push_str(&g.flush());
            let out = g.into_output();

            let tag = format!("prompt={prompt:?} max_new={max_new} stop={stop:?}");
            assert_eq!(stream, out.generated, "流式拼接必须等于最终生成段（{tag}）");
            assert_eq!(out.generated, plain.generated, "同种子下流式与非流式生成段一致（{tag}）");
            assert_eq!(out.reason, plain.reason, "收尾原因一致（{tag}）");
            assert_eq!(out.text, plain.text, "全文一致（{tag}）");
        }

        run_case(&model, &tokenizer, "the", 40, &[]); // 走 EOS 或上限
        run_case(&model, &tokenizer, "the", 40, &["brown"]); // 走停止标记
        run_case(&model, &tokenizer, "the", 40, &["qz"]); // 标记可能整段都不出现
        run_case(&model, &tokenizer, "the", 0, &[]); // 上限为 0：立即 MaxNew
    }

    /// 流式安全发出的扣留规则：尾部空白与停止标记的真前缀都不能先发出去。
    #[test]
    fn test_safe_len_holds_back_stop_prefix_and_trailing_space() {
        assert_eq!(safe_len("abc", &[]), 3, "无停止标记时全发（收尾结果不 trim）");
        assert_eq!(safe_len("abc ", &["用户："]), 3, "尾部空白要扣：命中标记时最终结果要 trim_end");
        assert_eq!(safe_len("abc用", &["用户："]), 3, "标记的真前缀要扣：标记可能横跨两次采样");
        assert_eq!(safe_len("abc用户", &["用户："]), 3, "多字节前缀整段扣住");
        assert_eq!(safe_len("abc 用户", &["用户："]), 3, "前缀之前的空白也会被最终的 trim_end 掉");
        assert_eq!(safe_len("abc用 ", &["用户："]), 6, "空格已隔开前缀，标记拼不上，只扣尾部空白");
        assert_eq!(safe_len("abc", &["xyz"]), 3, "没有前缀命中就不扣");
    }

    /// `generate_with_reason` 要把「生成段」单独交出来，并把收尾原因说清楚：
    /// 聊天模式下生成段为空就是终端上的一行空白，只有 `reason` 能解释它。
    #[test]
    fn test_generate_with_reason_separates_generated_text_and_reports_stop() {
        let (model, tokenizer) = tiny_setup();
        // prompt 里就含停止标记 "qz"：若搜索落到 prompt 上，开场就会命中并返回空回答
        let prompt = "qzthe";
        let mut o = opts();
        o.stop = &["qz"];
        let mut rng = Rng::new(99);
        // 生成 1 个 token：生成段只有一个字符，拼不出两字符的停止标记，断言是确定的
        let out = generate_with_reason(&model, &tokenizer, prompt, 1, &o, KvOpts::off(), &mut rng);
        assert_ne!(out.reason, StopReason::StopMark("qz"), "停止标记只在生成段里查找");
        assert_eq!(out.text, format!("{prompt}{}", out.generated), "全文 = prompt + 生成段");

        // 停止标记落在生成段第 0 位（空标记的 find 恒为 0）：生成段为空、全文退回 prompt
        o.stop = &[""];
        let mut rng = Rng::new(99);
        let out = generate_with_reason(&model, &tokenizer, prompt, 8, &o, KvOpts::off(), &mut rng);
        assert_eq!(out.reason, StopReason::StopMark(""));
        assert_eq!(out.generated, "", "生成段为空");
        assert_eq!(out.text, prompt, "生成段为空时全文就是 prompt（旧写法在这里会切出空串）");
    }

    /// 重复惩罚要把"最近出现过"的 token 压下去。
    /// temperature 取极小值让采样退化成近似贪心，断言才是确定性的。
    /// （回看窗口是 `generate` 按 `repetition_window` 截好 `recent` 后传进来的，
    ///   所以 `sample_token` 只认拿到的 `recent`。）
    #[test]
    fn test_repetition_penalty_suppresses_recent_token() {
        let near_greedy = |penalty: f32| SampleOpts {
            temperature: 0.05,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: penalty,
            repetition_window: 8,
            stop: &[],
        };

        // id=1 领先 id=2 一点点；惩罚 2.0 之后 4.0/2.0 = 2.0 < 3.9，id=2 反超
        let logits = vec![0.0, 4.0, 3.9];
        let mut rng = Rng::new(1);
        for _ in 0..10 {
            assert_eq!(
                sample_token(&logits, &near_greedy(1.0), &[1], &mut rng),
                1,
                "关闭惩罚时应选 argmax"
            );
            assert_eq!(
                sample_token(&logits, &near_greedy(2.0), &[1], &mut rng),
                2,
                "惩罚后应由 id=2 反超"
            );
        }

        // 负 logit 要**乘小**（-3.0 -> -6.0）而不是变大：于是 id=0 的 -4.0 反超
        let neg = vec![-4.0, -3.0];
        let mut rng = Rng::new(2);
        for _ in 0..10 {
            assert_eq!(sample_token(&neg, &near_greedy(1.0), &[1], &mut rng), 1);
            assert_eq!(
                sample_token(&neg, &near_greedy(2.0), &[1], &mut rng),
                0,
                "负 logit 也应被压低"
            );
        }
    }

    /// 字节级 BPE 的"半个汉字"token 必须在采样前屏蔽，否则会拼出非法 UTF-8。
    #[test]
    fn test_utf8_mask_blocks_half_char_tokens() {
        let tok = Tokenizer::bpe("中文测试中文测试中文测试", 300);
        let vocab = tok.vocab_bytes().expect("BPE 是字节级词表");
        let a = b'A' as usize; // 单字节 token id 0..255 恒等于字节值

        // 停在"中"的首字节 E4 上：此时接 ASCII 会立刻拼出非法序列
        let mut logits = vec![0.0f32; tok.vocab_size()];
        mask_illegal_utf8(&mut logits, vocab, &[0xE4]);
        assert!(logits[a].is_infinite(), "ASCII 接在多字节字符中间应被屏蔽");
        assert!(logits[0xB8].is_finite(), "合法的续字节应保留");

        // 停在字符边界上：ASCII 合法，孤立的续字节非法
        let mut logits = vec![0.0f32; tok.vocab_size()];
        mask_illegal_utf8(&mut logits, vocab, &[]);
        assert!(logits[a].is_finite(), "字符边界上 ASCII 合法");
        assert!(logits[0x80].is_infinite(), "孤立的续字节应被屏蔽");
    }

    /// 从已生成 token 反推"未拼完的字节"：token 可以从非字符边界开始，需从前往后找合法的起点。
    #[test]
    fn test_pending_tail_detects_incomplete_char() {
        let tok = Tokenizer::bpe("中文测试中文测试中文测试", 300);
        let vocab = tok.vocab_bytes().unwrap();
        let zhong = "中".as_bytes(); // E4 B8 AD

        assert_eq!(pending_tail(vocab, &[0x41]), Vec::<u8>::new(), "ASCII 后无未完成字节");
        assert_eq!(pending_tail(vocab, &[0xE4]), zhong[..1].to_vec());
        assert_eq!(pending_tail(vocab, &[0xE4, 0xB8]), zhong[..2].to_vec());
        assert_eq!(
            pending_tail(vocab, &[0xE4, 0xB8, 0xAD]),
            Vec::<u8>::new(),
            "完整字符不应残留 pending"
        );
        // 序列从"中"的续字节开始（真实场景里前面还有 E4），也要能定位到 E4
        assert_eq!(pending_tail(vocab, &[0x41, 0xB8, 0xAD, 0xE4]), zhong[..1].to_vec());
    }

    /// 图片段受约束采样的两条腿：状态机推进（start→码本×P→end）与采样域掩码。
    #[test]
    fn test_image_segment_state_machine_and_mask() {
        let im = ImageTokens {
            start: 10,
            ph_first: 11,
            count: 4,
            end: 15,
            cb_first: 16,
            cb_count: 8,
        };
        // 状态机：图外 → start 进图 → 码本逐个计数 → end 出图；
        // 图内出现普通 token 不改变状态（掩码已挡住，防御性保持）
        let mut seg = advance_img_seg(None, Some(im), 5);
        assert_eq!(seg, None, "图外遇到普通 token 不进图");
        seg = advance_img_seg(seg, Some(im), im.start);
        assert_eq!(seg, Some(0));
        seg = advance_img_seg(seg, Some(im), im.cb_id(3));
        seg = advance_img_seg(seg, Some(im), im.cb_id(7));
        assert_eq!(seg, Some(2));
        seg = advance_img_seg(seg, Some(im), 99);
        assert_eq!(seg, Some(2), "图内普通 token 不改变状态");
        seg = advance_img_seg(seg, Some(im), im.end);
        assert_eq!(seg, None);

        // 图外：占位符段 / 码本段 / end 全屏蔽，start 与普通 token 保留
        let mut l = vec![1.0f32; 32];
        mask_image(&mut l, im, None);
        for id in 11..=23 {
            assert!(l[id].is_sign_negative() && l[id].is_infinite(), "图外应屏蔽 {id}");
        }
        assert_eq!(l[im.start], 1.0, "start 必须可采样（生成图片的唯一入口）");
        assert_eq!(l[0], 1.0, "普通文本 token 不受影响");

        // 图内未满 P 个：只留码本段（EOS / 文本全屏蔽，图必须完整采完）
        let mut l = vec![1.0f32; 32];
        mask_image(&mut l, im, Some(2));
        for id in 0..32 {
            let alive = !l[id].is_infinite();
            assert_eq!(alive, (16..24).contains(&id), "图内(2) 应只留码本区，id={id}");
        }

        // 图内已满 P 个：只留 end 收尾
        let mut l = vec![1.0f32; 32];
        mask_image(&mut l, im, Some(4));
        for id in 0..32 {
            let alive = !l[id].is_infinite();
            assert_eq!(alive, id == im.end, "图内(满) 应只留 end，id={id}");
        }
    }

    /// 图片理解的采样通路（`with_pixels`）：
    /// 1) 注入像素会覆写占位符 → 同种子下生成结果与纯文本路径不同；
    /// 2) 无 cache 全量模式生成超过 block_size 时，滑窗把占位符挤出窗口，
    ///    守卫自动退回纯文本路径——不触发 `forward_mm` 的数量断言，且照常出字。
    #[test]
    fn test_generator_with_pixels_injects_vision_and_survives_sliding_window() {
        let corpus = "the quick brown fox jumps over the lazy dog, and then runs away.";
        let mut tokenizer = Tokenizer::char(corpus);
        let im = tokenizer
            .attach_image_tokens(4, 0)
            .expect("char 分词器自带 specials");
        let vcfg = crate::vision::VisionConfig {
            image_size: 8,
            patch_size: 4,
            n_embd: 16,
            n_head: 2,
            n_layer: 1,
            dropout: 0.0,
            ph_first: im.ph_first, // 占位符 id 必须与分词器展开的区间一致
        };
        let mut rng = Rng::new(5);
        let cfg = TransformerConfig {
            vision: Some(vcfg),
            ..TransformerConfig::tiny(tokenizer.vocab_size())
        };
        let model = Transformer::new(cfg, &mut rng);

        // prompt 里恰好一个 <|image|>，展开成 4 个占位符（P = (8/4)² = 4）
        let prompt = format!("the{}", crate::tokenizer::IMAGE_LITERAL);
        let pixels: Vec<f32> = (0..3 * 8 * 8)
            .map(|i| ((i % 13) as f32 / 12.0) * 2.0 - 1.0)
            .collect();

        let run = |inject: bool, kv: KvOpts, max_new: usize| -> GenOutput {
            let mut r = Rng::new(9);
            let mut g = Generator::new(&model, &tokenizer, &prompt, max_new, &opts(), kv, &mut r);
            if inject {
                g = g.with_pixels(pixels.clone());
            }
            g.run();
            g.into_output()
        };

        // 1) 同种子：注入像素 vs 纯文本占位符，生成应分道扬镳
        let with_px = run(true, KvOpts::on(0, None), 12);
        let no_px = run(false, KvOpts::on(0, None), 12);
        assert_ne!(
            with_px.text, no_px.text,
            "注入像素应改变生成结果（视觉特征没进前向？）"
        );

        // 2) 生成 40 > block_size 32：占位符必被滑窗挤出，守卫须退回纯文本路径
        //    （decode 不还原 <|image|> 字面量，所以只看生成段而非全文长度）
        let long_run = run(true, KvOpts::off(), 40);
        assert!(
            !long_run.generated.is_empty(),
            "滑窗后应照常出字：{long_run:?}"
        );
    }
}
