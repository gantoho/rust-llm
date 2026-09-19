//! 分词器（第 8 课）
//!
//! 大语言模型处理的是数字，不是文字。分词器负责"文字 <-> 数字"的转换。
//!
//! 本模块实现两种：
//! - `CharTokenizer`：按字符切分（简单直观，适合小模型学习）
//! - `BPETokenizer`：字节对编码（现代 GPT 的实际方案，能压缩常见词/子词）
//!
//! 两种分词器都支持序列化/反序列化（save/load），训练后可持久化，推理时直接加载。

use std::collections::HashMap;
use std::io::{Read, Write};

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
        ids.iter()
            .map(|&i| {
                self.chars
                    .get(i)
                    .copied()
                    .unwrap_or_else(|| {
                        panic!("decode 遇到越界 token id {i}（词表大小 {}）", self.chars.len())
                    })
            })
            .collect()
    }

    /// 保存到文件（JSON 格式）
    pub fn save(&self, path: &str) {
        let json = serde_json::json!({
            "type": "char",
            "chars": self.chars.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
        });
        let mut f = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("无法创建分词器文件 {path}: {e}"));
        f.write_all(serde_json::to_string_pretty(&json).unwrap().as_bytes())
            .unwrap_or_else(|e| panic!("写入分词器文件 {path} 失败: {e}"));
    }

    /// 从文件加载（独立使用，Tokenizer::load 会自动调用 from_json 避免重复读文件）
    #[allow(dead_code)]
    pub fn load(path: &str) -> Self {
        let mut f = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("无法打开分词器文件 {path}: {e}"));
        let mut text = String::new();
        f.read_to_string(&mut text)
            .unwrap_or_else(|e| panic!("读取分词器文件 {path} 失败: {e}"));
        let json: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("解析分词器文件 {path} 失败: {e}"));
        Self::from_json(&json)
    }

    /// 从已解析的 JSON 值构造
    pub fn from_json(json: &serde_json::Value) -> Self {
        let chars: Vec<char> = json["chars"]
            .as_array()
            .expect("分词器文件格式错误：缺少 chars 字段")
            .iter()
            .map(|v| {
                let s = v.as_str().expect("chars 元素应为字符串");
                assert_eq!(s.len(), s.chars().count(), "chars 元素应为单个字符");
                s.chars().next().unwrap()
            })
            .collect();
        let stoi = chars.iter().enumerate().map(|(i, &c)| (c, i)).collect();
        CharTokenizer { chars, stoi }
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
        // 语料过长时采样，避免 BPE 训练耗时过长
        let max_train_bytes: usize = 1_000_000; // 1MB 足以学到良好的合并规则
        let train_bytes = if corpus.len() > max_train_bytes {
            println!("  语料 {} 字节，采样前 {} 字节用于 BPE 训练", corpus.len(), max_train_bytes);
            &corpus.as_bytes()[..max_train_bytes]
        } else {
            corpus.as_bytes()
        };
        Self::train_bytes(train_bytes, target_vocab)
    }

    fn train_bytes(data: &[u8], target_vocab: usize) -> Self {
        assert!(target_vocab >= 256, "BPE 词表至少 256（字节级）");
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
        let mut merges: Vec<(u16, u16)> = Vec::new();
        let mut ids: Vec<u16> = data.iter().map(|&b| b as u16).collect();

        let target_merges = target_vocab - 256;
        let log_interval = if target_merges >= 20 { target_merges / 20 } else { 1 };

        while vocab.len() < target_vocab {
            // 统计相邻 pair 频率（每次重新扫描，但语料已采样到1MB）
            let mut pair_freq: HashMap<(u16, u16), usize> = HashMap::new();
            for pair in ids.windows(2) {
                *pair_freq.entry((pair[0], pair[1])).or_insert(0) += 1;
            }
            // 找最高频 pair（频率相同取 pair 值小者，保证确定性）
            let best = match pair_freq
                .iter()
                .max_by(|a, b| a.1.cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .map(|(&k, _)| k)
            {
                Some(pair) => pair,
                None => break,
            };

            // 创建新 token
            let new_id = vocab.len() as u16;
            let mut new_bytes = vocab[best.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best.1 as usize]);
            vocab.push(new_bytes);
            merges.push(best);

            // 合并 ids 中所有该 pair（单次扫描重建）
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

            let merge_count = merges.len();
            if merge_count % log_interval == 0 || merge_count <= 5 {
                println!("  BPE 进度: {}/{} 次合并, 词表 {}, 序列长度 {}",
                    merge_count, target_merges, vocab.len(), ids.len());
            }
        }
        println!("  BPE 训练完成: {} 次合并, 词表 {}", merges.len(), vocab.len());

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
    ///
    /// 生成被 token 数（`max_new`）截断时，最后一个 token 可能只覆盖某个字符的一部分，
    /// 此时末尾几个字节凑不成字符：**丢掉它们**返回前面完整的部分。
    /// 这是"按 token 截断"的固有结果，不是错误，因此不打印警告。
    /// 真正的非法字节（采样阶段已由 UTF-8 约束排除，见 `crate::sample`）才会报警。
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        for &id in ids {
            let tok = self
                .vocab
                .get(id)
                .unwrap_or_else(|| panic!("decode 遇到越界 token id {id}（词表大小 {}）", self.vocab.len()));
            bytes.extend_from_slice(tok);
        }
        match String::from_utf8(bytes) {
            Ok(s) => s,
            Err(e) => {
                let bytes = e.as_bytes();
                match utf8_pending(bytes) {
                    // 合法前缀：只是末尾停在字符中间（半个汉字），丢掉不完整的尾部字节
                    Some(tail) => {
                        let keep = bytes.len() - tail.len();
                        String::from_utf8(bytes[..keep].to_vec())
                            .expect("前缀已被 utf8_pending 判定为完整 UTF-8")
                    }
                    // 非法字节：不该出现，保留警告便于定位
                    None => {
                        eprintln!("[decode] 警告：拼接出非法 UTF-8（{}），已跳过无效字节", e.utf8_error());
                        skip_invalid_utf8(bytes)
                    }
                }
            }
        }
    }

    /// 保存到文件（JSON 格式）
    pub fn save(&self, path: &str) {
        let json = serde_json::json!({
            "type": "bpe",
            "merges": self.merges.iter().map(|(a, b)| vec![*a, *b]).collect::<Vec<_>>(),
            "vocab": self.vocab.iter().map(|v| v.clone()).collect::<Vec<_>>(),
        });
        let mut f = std::fs::File::create(path)
            .unwrap_or_else(|e| panic!("无法创建分词器文件 {path}: {e}"));
        f.write_all(serde_json::to_string_pretty(&json).unwrap().as_bytes())
            .unwrap_or_else(|e| panic!("写入分词器文件 {path} 失败: {e}"));
    }

    /// 从文件加载（独立使用，Tokenizer::load 会自动调用 from_json 避免重复读文件）
    #[allow(dead_code)]
    pub fn load(path: &str) -> Self {
        let mut f = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("无法打开分词器文件 {path}: {e}"));
        let mut text = String::new();
        f.read_to_string(&mut text)
            .unwrap_or_else(|e| panic!("读取分词器文件 {path} 失败: {e}"));
        let json: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("解析分词器文件 {path} 失败: {e}"));
        Self::from_json(&json)
    }

    /// 从已解析的 JSON 值构造
    pub fn from_json(json: &serde_json::Value) -> Self {
        let merges: Vec<(u16, u16)> = json["merges"]
            .as_array()
            .expect("分词器文件格式错误：缺少 merges 字段")
            .iter()
            .map(|v| {
                let arr = v.as_array().expect("merge 应为数组");
                (arr[0].as_u64().unwrap() as u16, arr[1].as_u64().unwrap() as u16)
            })
            .collect();
        let vocab: Vec<Vec<u8>> = json["vocab"]
            .as_array()
            .expect("分词器文件格式错误：缺少 vocab 字段")
            .iter()
            .map(|v| {
                v.as_array()
                    .expect("vocab 元素应为数组")
                    .iter()
                    .map(|x| x.as_u64().unwrap() as u8)
                    .collect()
            })
            .collect();
        BPETokenizer { merges, vocab }
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

    /// 词表中每个 token 的字节序列，供生成时做 UTF-8 约束（见 [`utf8_pending`]）。
    ///
    /// `char` 分词器每个 token 都是完整字符，不可能拼出非法 UTF-8，因此返回 `None`。
    pub fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        match self {
            Tokenizer::Char(_) => None,
            Tokenizer::Bpe(t) => Some(&t.vocab),
        }
    }

    /// 类型名（打印用）："char" / "bpe"
    pub fn kind(&self) -> &'static str {
        match self {
            Tokenizer::Char(_) => "char",
            Tokenizer::Bpe(_) => "bpe",
        }
    }

    /// 保存分词器到文件（父目录不存在时自动创建）
    pub fn save(&self, path: &str) {
        crate::config::ensure_parent_dir(path);
        match self {
            Tokenizer::Char(t) => t.save(path),
            Tokenizer::Bpe(t) => t.save(path),
        }
    }

    /// 从文件加载分词器（自动识别 char/bpe 类型，只读一次文件）
    pub fn load(path: &str) -> Self {
        let mut f = std::fs::File::open(path)
            .unwrap_or_else(|e| panic!("无法打开分词器文件 {path}: {e}"));
        let mut text = String::new();
        f.read_to_string(&mut text)
            .unwrap_or_else(|e| panic!("读取分词器文件 {path} 失败: {e}"));
        let json: serde_json::Value = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("解析分词器文件 {path} 失败: {e}"));
        let typ = json["type"]
            .as_str()
            .expect("分词器文件格式错误：缺少 type 字段");
        match typ {
            "char" => Tokenizer::Char(CharTokenizer::from_json(&json)),
            "bpe" => Tokenizer::Bpe(BPETokenizer::from_json(&json)),
            other => panic!("未知分词器类型 '{}'（可选：char / bpe）", other),
        }
    }
}

/// 跳过非法 UTF-8 字节，只保留合法部分（不用无意义的替换字符）
fn skip_invalid_utf8(bytes: &[u8]) -> String {
    let mut result = String::new();
    let mut i = 0;
    while i < bytes.len() {
        // 尝试从位置 i 开始解码一个合法的 UTF-8 字符
        if let Some((ch, len)) = decode_utf8_char(&bytes[i..]) {
            result.push(ch);
            i += len;
        } else {
            // 跳过这个非法字节（不插入任何字符）
            i += 1;
        }
    }
    result
}

/// 尝试从字节切片开头解码一个 UTF-8 字符，返回 (字符, 字节数)
fn decode_utf8_char(bytes: &[u8]) -> Option<(char, usize)> {
    if bytes.is_empty() {
        return None;
    }
    let b = bytes[0];
    let (code, len) = if b < 0x80 {
        (b as u32, 1)
    } else if b & 0xE0 == 0xC0 {
        (b as u32 & 0x1F, 2)
    } else if b & 0xF0 == 0xE0 {
        (b as u32 & 0x0F, 3)
    } else if b & 0xF8 == 0xF0 {
        (b as u32 & 0x07, 4)
    } else {
        return None; // 非法起始字节
    };
    if bytes.len() < len {
        return None;
    }
    let mut cp = code;
    for &b in &bytes[1..len] {
        if b & 0xC0 != 0x80 {
            return None; // 非法续字节
        }
        cp = (cp << 6) | (b as u32 & 0x3F);
    }
    // 每种字节长度都有码点下限，低于下限的是"过长编码"（如 C0 80 表示 U+0000），同样非法
    let min = match len {
        1 => 0,
        2 => 0x80,
        3 => 0x800,
        _ => 0x1_0000,
    };
    if cp < min {
        return None;
    }
    char::from_u32(cp).map(|c| (c, len))
}

/// 判断字节串是否是**合法的 UTF-8 前缀**（允许停在某个字符中间，如 `E4 B8` 之于"中"）。
///
/// - `Some(尾部字节)`：合法前缀，`尾部字节` 是尚未拼完的部分（空切片 = 正好停在字符边界）
/// - `None`：已经出现非法字节
///
/// 字节级 BPE 生成时用它做约束：只有结果仍返回 `Some` 的 token 才允许被采样到。
pub fn utf8_pending(bytes: &[u8]) -> Option<&[u8]> {
    let mut i = 0;
    while i < bytes.len() {
        let rest = &bytes[i..];
        if let Some((_, len)) = decode_utf8_char(rest) {
            i += len; // 完整字符，继续往后
            continue;
        }
        // 解不出完整字符：只有"字节数还不够 + 已有的续字节都合法"才是合法前缀，
        // 长度够了还解不出来（含过长编码、代理区码点）就是非法字节。
        let b = rest[0];
        let expect = if b & 0xE0 == 0xC0 {
            2
        } else if b & 0xF0 == 0xE0 {
            3
        } else if b & 0xF8 == 0xF0 {
            4
        } else {
            return None; // 续字节或非法起始字节开头
        };
        if rest.len() >= expect || rest[1..].iter().any(|&x| x & 0xC0 != 0x80) {
            return None;
        }
        return Some(rest);
    }
    Some(&[])
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

    #[test]
    fn test_utf8_pending_accepts_only_legal_prefix() {
        let zhong = "中".as_bytes(); // E4 B8 AD
        // 完整字符 / 纯 ASCII：停在字符边界，尾部为空
        assert_eq!(utf8_pending(b"abc").map(<[u8]>::len), Some(0));
        assert_eq!(utf8_pending(zhong).map(<[u8]>::len), Some(0));
        // 少一字节：合法的不完整前缀，尾部就是这两个字节
        assert_eq!(utf8_pending(&zhong[..2]).map(<[u8]>::len), Some(2));
        assert_eq!(utf8_pending(&zhong[..1]).map(<[u8]>::len), Some(1));
        // ASCII 跟在未完成的字符后面 → 非法
        assert_eq!(utf8_pending(&[zhong[0], b'A']).map(<[u8]>::len), None);
        // 孤立的续字节 → 非法
        assert_eq!(utf8_pending(&[0x80]).map(<[u8]>::len), None);
        // 过长编码（C0 80 表示 U+0000）→ 非法
        assert_eq!(utf8_pending(&[0xC0, 0x80]).map(<[u8]>::len), None);
        // 非法起始字节（F8 以上）→ 非法
        assert_eq!(utf8_pending(&[0xF8]).map(<[u8]>::len), None);
        // 三字节编码落在代理区（ED A0 80）→ 非法
        assert_eq!(utf8_pending(&[0xED, 0xA0, 0x80]).map(<[u8]>::len), None);
    }

    /// 生成被 token 数截断时，末尾可能只有"半个汉字"，decode 应丢掉它而不是报错。
    #[test]
    fn test_decode_drops_incomplete_tail() {
        let tok = BPETokenizer::train("中文测试中文测试", 300);
        let mut ids = tok.encode("中");
        ids.push(0xE4); // 再补一个"中"的首字节：凑不成字符
        assert_eq!(tok.decode(&ids), "中", "末尾不完整的字节应被丢弃");

        // 末尾凑成"半个字符"（E4 B8 是"中"的前两字节）时同样处理
        let mut ids = tok.encode("测试");
        ids.extend_from_slice(&[0xE4, 0xB8]);
        assert_eq!(tok.decode(&ids), "测试");
    }
}
