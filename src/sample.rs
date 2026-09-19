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

use crate::model::GPT;
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;

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
}

impl Default for SampleOpts {
    fn default() -> Self {
        SampleOpts {
            temperature: 0.8,
            top_k: 40,
            top_p: 0.9,
            repetition_penalty: 1.1,
            repetition_window: 64,
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

    // 7. 按概率随机抽样
    let mut u = rng.next_f32();
    for (i, p) in probs.iter().enumerate() {
        if u < *p {
            return items[i].0;
        }
        u -= p;
    }
    items.last().map(|(i, _)| *i).unwrap_or(0)
}

/// 生成文本
///
/// - prompt: 起始文本
/// - max_new: 最多生成多少个新 token
/// - opts: 采样超参数（含重复惩罚）
/// - use_kv_cache: 是否使用 KV cache 加速（第 18 课）
pub fn generate(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    opts: &SampleOpts,
    use_kv_cache: bool,
    rng: &mut Rng,
) -> String {
    let block_size = model.cfg.block_size;
    let mut ids = tokenizer.encode(prompt);
    if ids.is_empty() {
        ids.push(0); // 空 prompt：先喂一个 token，避免 0 长度上下文导致下标下溢
    }
    // 仅在使用 KV cache 时才分配缓存，全量模式不浪费内存
    let mut cache = use_kv_cache.then(|| model.new_kv_cache());
    // 缓存窗口写满时是否提前结束（全量模式会滑动窗口继续，KV cache 做不到）
    let mut hit_window_limit = false;
    // 字节级 BPE 的 UTF-8 约束：词表里有"半个汉字"，采样前要把它们排除（char 分词器返回 None）
    let vocab_bytes = tokenizer.vocab_bytes();
    let mut masked: Vec<f32> = Vec::new();

    for _ in 0..max_new {
        // KV cache 模式：上下文总长达到 block_size 就停（缓存无法像全量模式那样截断历史）
        if cache.as_ref().is_some_and(|c| c[0].seq_len() >= block_size) {
            hit_window_limit = true;
            break;
        }
        // 只保留最近的 block_size 个 token（全量模式需要）
        let start = ids.len().saturating_sub(block_size);
        let ctx = &ids[start..];

        // 推理不需要反向：no_grad 下不挂计算图、不分配梯度缓冲
        let logits = crate::tensor::no_grad(|| {
            if use_kv_cache {
                // 首次：缓存为空，把整个 prompt 喂进去（顺便填充缓存）
                // 之后：每步只前向最新 1 个 token，历史 K/V 从缓存取
                let c = cache.as_mut().unwrap();
                if c[0].seq_len() == 0 {
                    model.forward(ctx, 1, ctx.len(), Some(c), false)
                } else {
                    model.forward(&ids[ids.len() - 1..], 1, 1, Some(c), false)
                }
            } else {
                // 全量模式：每次把整个上下文重新算一遍（慢，但没有 cache 内存）
                model.forward(ctx, 1, ctx.len(), None, false)
            }
        });

        // 取最后一个位置的 logits
        let v = model.cfg.vocab_size;
        let n = logits.numel();
        let last_row = &logits.data()[n - v..];
        // 只允许"接上后仍是合法 UTF-8 前缀"的 token，否则会拼出半个字符
        let row: &[f32] = match vocab_bytes {
            Some(vocab) => {
                masked.clear();
                masked.extend_from_slice(last_row);
                mask_illegal_utf8(&mut masked, vocab, &pending_tail(vocab, &ids));
                &masked
            }
            None => last_row,
        };
        // 重复惩罚的回看窗口：只看最近 N 个 token（含 prompt），更早的不再计入。
        // 窗口过大时高频 token 会被持续压低，可能影响语句的连贯性。
        let recent = if opts.repetition_window == 0 {
            &[][..]
        } else {
            let start = ids.len().saturating_sub(opts.repetition_window);
            &ids[start..]
        };
        let next = sample_token(row, opts, recent, rng);
        ids.push(next);
    }

    if hit_window_limit {
        // 不能静默变短：同一个 prompt 加不加 KV cache 会得到不同长度，用户必须知情。
        // 打到 stderr 而不是 stdout —— stdout 是生成结果（`generate` 子命令可能被重定向到文件），
        // 但同时也写进运行日志，否则事后复盘看不到"这次生成为什么变短了"。
        let msg = format!(
            "[warn] KV cache 窗口已满（block_size={block_size}），生成在 {} 个 token 处提前结束；\
             需要更长输出请缩短 prompt，或加 --no-kv-cache 改用全量前向（滑动窗口可继续生成）",
            ids.len()
        );
        eprintln!("{msg}");
        crate::runlog::append(&msg);
    }

    tokenizer.decode(&ids)
}

/// Beam Search 生成：维护 `beam_size` 个候选序列，每步扩展后保留 top-k。
///
/// 与采样（temperature + top-k/top-p）的区别：
/// - 采样是随机的，每次生成不同
/// - Beam Search 是确定性的（给定 seed），总选择全局最优的 k 条路径
/// - 生成质量更高，但多样性更低
///
/// 返回 top-1 序列（log 概率最高的完整序列）。
///
/// - beam_size: 束宽（通常 4-10），越大搜索越充分，但越慢
/// - length_penalty: 长度惩罚指数 α（0 = 不惩罚，>0 偏好长序列，<0 偏好短序列）
///   最终分数 = log_prob / len^α（Google NMT 的公式）
#[allow(dead_code)] // 教学实现：Beam Search 完整可用，通过 sample::beam_search 调用
pub fn beam_search(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    beam_size: usize,
    length_penalty: f32,
    _rng: &mut Rng,
) -> String {
    assert!(beam_size >= 1, "beam_size 必须 >= 1");
    let block_size = model.cfg.block_size;
    let vocab_size = model.cfg.vocab_size;
    let prompt_ids = tokenizer.encode(prompt);
    let prompt_len = prompt_ids.len();
    let prompt_ids = if prompt_ids.is_empty() {
        vec![0]
    } else {
        prompt_ids
    };

    // 每个 beam: (token_ids, cumulative_log_prob)
    let mut beams: Vec<(Vec<usize>, f64)> = vec![(prompt_ids, 0.0)];
    // 字节级 BPE 的 UTF-8 约束（同 generate：排除"半个汉字"的 token）
    let vocab_bytes = tokenizer.vocab_bytes();
    let mut masked: Vec<f32> = Vec::new();

    for _step in 0..max_new {
        let mut candidates: Vec<(Vec<usize>, f64)> = Vec::new();

        for (ids, score) in &beams {
            // 已经生成 EOS（这里用 id=0 简化）就不扩展
            if ids.last() == Some(&0) && ids.len() > prompt_len {
                candidates.push((ids.clone(), *score));
                continue;
            }
            // 上下文截断
            let start = ids.len().saturating_sub(block_size);
            let ctx = &ids[start..];

            // 全量前向（beam search 通常是离线的，不用 KV cache）
            // 推理无需反向，包在 no_grad 里避免建图开销
            let logits = crate::tensor::no_grad(|| model.forward(ctx, 1, ctx.len(), None, false));
            let n = logits.numel();
            let last_row = &logits.data()[n - vocab_size..];
            // 只允许"接上后仍是合法 UTF-8 前缀"的 token
            let row: &[f32] = match vocab_bytes {
                Some(vocab) => {
                    masked.clear();
                    masked.extend_from_slice(last_row);
                    mask_illegal_utf8(&mut masked, vocab, &pending_tail(vocab, ids));
                    &masked
                }
                None => last_row,
            };

            // 找 top-beam_size 个候选 token
            let mut scored: Vec<(usize, f64)> = row
                .iter()
                .enumerate()
                .filter(|(_, l)| l.is_finite()) // 被屏蔽的 -inf 直接跳过
                .map(|(i, &l)| (i, *score + l as f64))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(beam_size);

            for (tok, new_score) in scored {
                let mut new_ids = ids.clone();
                new_ids.push(tok);
                candidates.push((new_ids, new_score));
            }
        }

        // 按分数排序，保留 top-beam_size
        candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        candidates.truncate(beam_size);
        beams = candidates;

        // 所有 beam 都结束了就提前停
        if beams.iter().all(|(ids, _)| {
            ids.len() > prompt_len && ids.last() == Some(&0)
        }) {
            break;
        }
    }

    // 按长度惩罚后的分数选最佳
    // 不过我们这里简单返回 log_prob 最高的
    // length_penalty: score / len^alpha
    let best = beams
        .iter()
        .max_by(|a, b| {
            let sa = a.1 / (a.0.len() as f64).powf(length_penalty as f64);
            let sb = b.1 / (b.0.len() as f64).powf(length_penalty as f64);
            sa.partial_cmp(&sb).unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap();

    tokenizer.decode(&best.0)
}

// ==================== 生成时的 UTF-8 约束 ====================

/// 把"接上后不再是合法 UTF-8 前缀"的 token 的 logit 置为 `-inf`，采样时自然抽不到。
///
/// 字节级 BPE 的词表里有"半个汉字"（如只含 `E4` 或 `B8`）。训练时它们能拼回原文，
/// 但推理是模型自由采样的，碎片一旦乱序就会拼出非法 UTF-8（表现为丢字、乱码）。
/// 这里在采样前剪掉这些 token：数学上等价于把这些 token 的概率设为 0 后重新归一化。
///
/// `pending` 是当前字节流末尾未拼完的字节（见 [`pending_tail`]）。
fn mask_illegal_utf8(logits: &mut [f32], vocab: &[Vec<u8>], pending: &[u8]) {
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
fn pending_tail(vocab: &[Vec<u8>], ids: &[usize]) -> Vec<u8> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GPTConfig;

    fn tiny_setup() -> (GPT, Tokenizer) {
        let corpus = "the quick brown fox jumps over the lazy dog, and then runs away.";
        let tokenizer = Tokenizer::char(corpus);
        let mut rng = Rng::new(7);
        let model = GPT::new(GPTConfig::tiny(tokenizer.vocab_size()), &mut rng);
        (model, tokenizer)
    }

    fn opts() -> SampleOpts {
        SampleOpts {
            temperature: 0.8,
            top_k: 10,
            top_p: 0.9,
            repetition_penalty: 1.1,
            repetition_window: 64,
        }
    }

    /// KV cache 只改注意力的计算方式（增量 vs 全量），不改生成分布：
    /// 同 prompt + 同种子、且总长不超 `block_size` 时，两种模式应逐 token 一致。
    #[test]
    fn test_kv_cache_generate_matches_full() {
        let (model, tokenizer) = tiny_setup();
        // tiny 的 block_size = 32：prompt 3 + 生成 20 = 23 ≤ 32，全程在缓存窗口内
        let mut rng_full = Rng::new(1234);
        let full = generate(&model, &tokenizer, "the", 20, &opts(), false, &mut rng_full);
        let mut rng_cache = Rng::new(1234);
        let cached = generate(&model, &tokenizer, "the", 20, &opts(), true, &mut rng_cache);

        assert_eq!(full, cached, "窗口内的 KV cache 生成应与全量前向逐 token 一致");
    }

    /// 缓存窗口写满后 KV cache 模式会提前结束，全量模式靠滑动窗口继续生成。
    /// 两者长度不同是有意为之（见 `generate` 里的 `hit_window_limit` 警告），
    /// 但 cache 的输出应是全量输出的**前缀**：只变短，内容不变。
    #[test]
    fn test_kv_cache_output_is_prefix_of_full_beyond_window() {
        let (model, tokenizer) = tiny_setup();
        let mut rng_full = Rng::new(1234);
        let full = generate(&model, &tokenizer, "the", 60, &opts(), false, &mut rng_full);
        let mut rng_cache = Rng::new(1234);
        let cached = generate(&model, &tokenizer, "the", 60, &opts(), true, &mut rng_cache);

        assert!(
            cached.chars().count() < full.chars().count(),
            "超出窗口时 cache 模式应提前结束，输出比全量短"
        );
        assert!(
            full.starts_with(&cached),
            "超出窗口时 cache 输出应仍是全量输出的前缀：\nfull   = {full}\ncached = {cached}"
        );
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
}
