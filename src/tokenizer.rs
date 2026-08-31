//! 分词器（第 8 课）
//!
//! 大语言模型处理的是数字，不是文字。分词器负责"文字 <-> 数字"的转换。
//!
//! 本模块实现两种：
//! - `CharTokenizer`：按字符切分（简单直观，适合小模型学习）
//! - `BPETokenizer`：字节对编码（现代 GPT 的实际方案，能压缩常见词/子词）

use std::collections::HashMap;

// ==================== 字符级分词器 ====================

/// 字符级分词器：词表就是语料中出现过的所有字符
pub struct CharTokenizer {
    chars: Vec<char>,
    stoi: HashMap<char, usize>,
}

impl CharTokenizer {
    /// 从语料构建词表（按字符首次出现的顺序）
    pub fn new(text: &str) -> Self {
        let mut chars: Vec<char> = Vec::new();
        let mut seen = std::collections::HashSet::new();
        for c in text.chars() {
            if seen.insert(c) {
                chars.push(c);
            }
        }
        let stoi = chars
            .iter()
            .enumerate()
            .map(|(i, &c)| (c, i))
            .collect();
        CharTokenizer { chars, stoi }
    }

    pub fn vocab_size(&self) -> usize {
        self.chars.len()
    }

    /// 文本 -> id 序列
    pub fn encode(&self, text: &str) -> Vec<usize> {
        text.chars()
            .map(|c| {
                *self
                    .stoi
                    .get(&c)
                    .unwrap_or_else(|| panic!("词表中没有字符 '{}'", c))
            })
            .collect()
    }

    /// id 序列 -> 文本
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter().map(|&i| self.chars[i]).collect()
    }
}

// ==================== BPE 分词器 ====================

/// 字节级 BPE 分词器
///
/// 思想（第 8 课详解）：
/// 1. 初始词表 = 256 个字节
/// 2. 统计文本中相邻"符号对"的出现频率，把最高频的一对**合并**成一个新符号
/// 3. 重复合并，直到词表达到目标大小
/// 4. 高频子词（如 "the"、"ing"）逐渐成为独立符号，实现"用更少的 token 表示更多文本"
pub struct BPETokenizer {
    /// 合并规则，按下标顺序（越早合并优先级越高）
    merges: Vec<(u16, u16)>,
    /// token id -> 它代表的字节序列
    vocab: Vec<Vec<u8>>,
}

impl BPETokenizer {
    /// 在语料上训练 BPE，目标词表大小 = 256 + 合并次数
    pub fn train(corpus: &str, target_vocab: usize) -> Self {
        assert!(target_vocab >= 256, "BPE 词表至少 256（字节级）");
        // 初始：每个 token 就是一个字节
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
        let mut merges: Vec<(u16, u16)> = Vec::new();

        // 语料 -> 字节 -> id 序列
        let mut ids: Vec<u16> = corpus.as_bytes().iter().map(|&b| b as u16).collect();

        while vocab.len() < target_vocab {
            // 1. 统计相邻 pair 频率
            let mut pair_freq: HashMap<(u16, u16), usize> = HashMap::new();
            for pair in ids.windows(2) {
                *pair_freq.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
            // 2. 找最高频的 pair（频率相同取 pair 值小者，保证确定性）
            let Some(&best) = pair_freq
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(k, _)| k)
            else {
                break; // 没有可合并的 pair 了
            };
            // 3. 合并：新符号 = 两个符号的字节拼接
            let new_id = vocab.len() as u16;
            let mut new_bytes = vocab[best.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best.1 as usize]);
            vocab.push(new_bytes);
            merges.push(best);

            // 4. 替换 ids 中所有该 pair
            let mut new_ids: Vec<u16> = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == best.0 && ids[i + 1] == best.1 {
                    new_ids.push(new_id);
                    i += 2;
                } else {
                    new_ids.push(ids[i]);
                    i += 1;
                }
            }
            ids = new_ids;
        }

        BPETokenizer { merges, vocab }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 文本 -> token id 序列
    ///
    /// 贪心合并：反复寻找"当前 ids 中存在且合并规则下标最小"的 pair 进行合并。
    /// 越早合并的规则优先级越高（它对应的新符号 id 更小）。
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<u16> = text.as_bytes().iter().map(|&b| b as u16).collect();
        loop {
            // 找到 ids 中优先级最高（merge 下标最小）的可合并 pair
            let mut best_merge: Option<usize> = None; // merge 下标
            let mut pos = 0;
            for i in 0..ids.len().saturating_sub(1) {
                let pair = (ids[i], ids[i + 1]);
                if let Some(idx) = self.merges.iter().position(|&m| m == pair) {
                    if best_merge.map_or(true, |b| idx < b) {
                        best_merge = Some(idx);
                        pos = i;
                    }
                }
            }
            let Some(idx) = best_merge else {
                break;
            };
            let new_id = (256 + idx) as u16;
            ids[pos] = new_id;
            ids.remove(pos + 1);
        }
        ids.into_iter().map(|x| x as usize).collect()
    }

    /// token id 序列 -> 文本
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            bytes.extend_from_slice(&self.vocab[id]);
        }
        String::from_utf8_lossy(&bytes).to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_char_tokenizer_roundtrip() {
        let text = "hello world hello";
        let tok = CharTokenizer::new(text);
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
        assert_eq!(tok.vocab_size(), 8); // h,e,l,o,' ',w,r,d
    }

    #[test]
    fn test_bpe_roundtrip() {
        let corpus = "low low low low low lowest lowest newest newest newest";
        let tok = BPETokenizer::train(corpus, 300);
        let text = "lowest new";
        let ids = tok.encode(text);
        assert_eq!(tok.decode(&ids), text);
        // "low" 应该被合并成一个 token（最高频）
        let ids_low = tok.encode("low");
        assert!(ids_low.len() <= 3, "high-frequency 子词应被压缩，实际 {} 个 token", ids_low.len());
    }
}
