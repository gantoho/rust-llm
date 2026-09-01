//! 采样与文本生成（第 15 课）
//!
//! 模型输出的是"每个 token 的概率分布"，怎么从分布里选一个 token？
//! - 贪心：总是选概率最大的（容易重复、呆板）
//! - temperature：缩放概率分布的"锐度"（<1 更确定，>1 更随机）
//! - top-k：只在前 k 个概率最高的 token 里选
//! - top-p（nucleus）：在累积概率达到 p 的最小集合里选
//!
//! 结合使用：temperature 调整锐度 -> top-k/top-p 截断 -> 按概率随机抽样。

use crate::model::GPT;
use crate::rng::Rng;
use crate::tokenizer::Tokenizer;

/// 从 logits 分布中采样一个 token
pub fn sample_token(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    rng: &mut Rng,
) -> usize {
    // 1. 除以 temperature 缩放
    let scaled: Vec<f32> = logits.iter().map(|&l| l / temperature.max(1e-5)).collect();

    // 2. 按分数从高到低排序
    let mut items: Vec<(usize, f32)> = scaled.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    items.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // 3. top-k：只保留前 k 个
    if top_k > 0 && items.len() > top_k {
        items.truncate(top_k);
    }

    // 4. softmax 得到概率
    let max = items
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = items.iter().map(|(_, v)| (*v - max).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }

    // 5. top-p：从高到低累加概率，直到超过 p，后面的全部丢弃
    if top_p < 1.0 {
        let mut cum = 0.0;
        let mut keep = items.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum >= top_p {
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

    // 6. 按概率随机抽样
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
/// - use_kv_cache: 是否使用 KV cache 加速（第 18 课）
pub fn generate(
    model: &GPT,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new: usize,
    temperature: f32,
    top_k: usize,
    top_p: f32,
    use_kv_cache: bool,
    rng: &mut Rng,
) -> String {
    let block_size = model.cfg.block_size;
    let mut ids = tokenizer.encode(prompt);
    let mut cache = model.new_kv_cache();

    for _ in 0..max_new {
        // KV cache 模式：上下文总长达到 block_size 就停（缓存无法像全量模式那样截断历史）
        if use_kv_cache && cache[0].seq_len() >= block_size {
            break;
        }
        // 只保留最近的 block_size 个 token（全量模式需要）
        let start = ids.len().saturating_sub(block_size);
        let ctx = &ids[start..];

        let logits = if use_kv_cache {
            // 首次：缓存为空，把整个 prompt 喂进去（顺便填充缓存）
            // 之后：每步只前向最新 1 个 token，历史 K/V 从缓存取
            if cache[0].seq_len() == 0 {
                model.forward(ctx, 1, ctx.len(), Some(&mut cache))
            } else {
                model.forward(&ids[ids.len() - 1..], 1, 1, Some(&mut cache))
            }
        } else {
            // 全量模式：每次把整个上下文重新算一遍（慢，但没有 cache 内存）
            model.forward(ctx, 1, ctx.len(), None)
        };

        // 取最后一个位置的 logits
        let v = model.cfg.vocab_size;
        let n = logits.numel();
        let last_row = &logits.data()[n - v..];
        let next = sample_token(last_row, temperature, top_k, top_p, rng);
        ids.push(next);
    }

    tokenizer.decode(&ids)
}
