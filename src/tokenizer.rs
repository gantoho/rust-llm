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
        let stoi = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
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
    /// 贪心合并（GPT-2 的标准实现）：按优先级从高到低，对每条合并规则在序列上
    /// 做一趟扫描替换。复杂度 O(len × 合并数)，大语料也能秒级完成。
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<u16> = text.as_bytes().iter().map(|&b| b as u16).collect();
        for (idx, &(a, b)) in self.merges.iter().enumerate() {
            let new_id = (256 + idx) as u16;
            let mut out: Vec<u16> = Vec::with_capacity(ids.len());
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == a && ids[i + 1] == b {
                    out.push(new_id);
                    i += 2;
                } else {
                    out.push(ids[i]);
                    i += 1;
                }
            }
            ids = out;
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

// ==================== 统一分词器接口（配置可切换） ====================

/// 统一的分词器：`"char"` 用 [`CharTokenizer`]，`"bpe"` 用 [`BPETokenizer`]。
///
/// 上层（数据加载 / 训练 / 采样）只依赖这一个接口，不关心具体实现。
/// 注意：`encode` 出来的 id 空间由具体实现决定，两者互不通用。
pub enum Tokenizer {
    Char(CharTokenizer),
    Bpe(BPETokenizer),
}

impl Tokenizer {
    /// 字符级分词器：词表来自语料中出现的所有字符
    pub fn char(text: &str) -> Self {
        Tokenizer::Char(CharTokenizer::new(text))
    }

    /// BPE 分词器：在语料上训练，目标词表大小 = 256 + 合并次数
    pub fn bpe(text: &str, target_vocab: usize) -> Self {
        Tokenizer::Bpe(BPETokenizer::train(text, target_vocab))
    }

    /// 按名称构造：`"char"` / `"bpe"`，其余报错
    pub fn from_name(name: &str, corpus: &str, bpe_vocab: usize) -> Self {
        match name {
            "char" => Tokenizer::char(corpus),
            "bpe" => Tokenizer::bpe(corpus, bpe_vocab),
            other => panic!("未知分词器类型 '{}'（可选：char / bpe）", other),
        }
    }

    pub fn vocab_size(&self) -> usize {
        match self {
            Tokenizer::Char(t) => t.vocab_size(),
            Tokenizer::Bpe(t) => t.vocab_size(),
        }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Tokenizer::Char(t) => t.encode(text),
            Tokenizer::Bpe(t) => t.encode(text),
        }
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        match self {
            Tokenizer::Char(t) => t.decode(ids),
            Tokenizer::Bpe(t) => t.decode(ids),
        }
    }

    /// 类型名（打印用）："char" / "bpe"
    pub fn kind(&self) -> &'static str {
        match self {
            Tokenizer::Char(_) => "char",
            Tokenizer::Bpe(_) => "bpe",
        }
    }
}

// ==================== 测试 ====================

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
        assert!(
            ids_low.len() <= 3,
            "high-frequency 子词应被压缩，实际 {} 个 token",
            ids_low.len()
        );
    }
}
