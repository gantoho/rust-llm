//! 分词器（第 8 课）
//!
//! 大语言模型处理的是数字，不是文字。分词器负责"文字 <-> 数字"的转换。
//!
//! 本模块实现两种：
//! - `CharTokenizer`：按字符切分（简单直观，适合小模型学习）
//! - `BPETokenizer`：字节对编码（现代 Transformer 的实际方案，能压缩常见词/子词）
//!
//! 两种分词器都支持序列化/反序列化（save/load），训练后可持久化，推理时直接加载。
//!
//! 统一入口 [`Tokenizer`] 在内容词表之上再挂一组**特殊 token**
//! （[`SpecialTokens`]：BOS / EOS / PAD，见该结构体的文档）。它们让"序列在哪里结束"
//! 成为一个可以学习的符号，而不是靠语料里凑巧高频的字符组合来暗示。

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::io::{Read, Write};

/// 把 JSON 美化成文本写到文件（父目录不存在时自动创建）。
/// 分词器的三种保存路径（char / bpe / 带特殊 token 的统一入口）共用同一份写法，
/// 避免三处各写一遍 `File::create` + `to_string_pretty`。
fn write_json(path: &str, json: &serde_json::Value) {
    crate::config::ensure_parent_dir(path);
    let text = serde_json::to_string_pretty(json).expect("序列化分词器 JSON 失败");
    std::fs::File::create(path)
        .and_then(|mut f| f.write_all(text.as_bytes()))
        .unwrap_or_else(|e| panic!("写入分词器文件 {path} 失败: {e}"));
}

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

    /// 序列化成 JSON（[`Tokenizer::save`] 会往这份 JSON 上再补特殊 token 字段）
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "char",
            "chars": self.chars.iter().map(|c| c.to_string()).collect::<Vec<_>>(),
        })
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

/// 增量调整 pair 频次（delta = ±1），并把新值推进优先队列。
///
/// 队列里同一 pair 可能积累多个条目，pop 时按 `pair_freq` 当前值校验，
/// 不一致的视为过期直接丢弃（懒删除）——避免每轮合并后重建整个堆。
fn bump_pair(
    pair_freq: &mut HashMap<(u16, u16), usize>,
    heap: &mut BinaryHeap<(usize, Reverse<(u16, u16)>)>,
    pair: (u16, u16),
    delta: i32,
) {
    let e = pair_freq.entry(pair).or_insert(0);
    let new = *e as i32 + delta;
    debug_assert!(new >= 0, "pair {pair:?} 频次不应减为负数");
    *e = new as usize;
    if *e > 0 {
        heap.push((*e, Reverse(pair)));
    }
}

impl BPETokenizer {
    /// 在语料上训练 BPE，目标词表大小 = 256 + 合并次数
    ///
    /// **全量语料参与统计**（历史实现只取前 1MB，大语料采样无代表性）；
    /// 训练用「增量频次 + 优先队列」，不再每次合并都全量重扫重建统计。
    pub fn train(corpus: &str, target_vocab: usize) -> Self {
        Self::train_bytes(corpus.as_bytes(), target_vocab)
    }

    fn train_bytes(data: &[u8], target_vocab: usize) -> Self {
        assert!(target_vocab >= 256, "BPE 词表至少 256（字节级）");
        let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
        let mut merges: Vec<(u16, u16)> = Vec::new();
        let mut ids: Vec<u16> = data.iter().map(|&b| b as u16).collect();

        // pair -> 当前频次：初始扫一遍，之后随每次合并增量增减，
        // 不再像旧实现那样每轮全量重扫语料重建 HashMap（O(merges × corpus) 的主项之一）
        let mut pair_freq: HashMap<(u16, u16), usize> = HashMap::new();
        for w in ids.windows(2) {
            *pair_freq.entry((w[0], w[1])).or_insert(0) += 1;
        }
        // 最大堆：键 (频次, Reverse(pair))，频次最高者在顶、平手取 pair 值小者
        // （确定性 tie-break，与旧实现的 max_by 规则一致）。
        // 频次每次变化都 push 新条目，pop 时按 pair_freq 校验，过期条目直接丢弃（懒删除）。
        let mut heap: BinaryHeap<(usize, Reverse<(u16, u16)>)> = BinaryHeap::new();
        for (&p, &f) in &pair_freq {
            if f > 0 {
                heap.push((f, Reverse(p)));
            }
        }

        let target_merges = target_vocab - 256;
        let log_interval = if target_merges >= 20 { target_merges / 20 } else { 1 };
        // 双缓冲交替复用：每轮合并结果写进 buf 再 swap，避免逐轮重新分配大 Vec
        let mut buf: Vec<u16> = Vec::with_capacity(ids.len());

        'train: while vocab.len() < target_vocab {
            // 弹出所有过期条目，取当前真实最高频 pair；堆空说明没有可合并的 pair
            let best = loop {
                match heap.pop() {
                    Some((f, Reverse(p))) if f > 0 && pair_freq.get(&p).copied() == Some(f) => {
                        break p
                    }
                    Some(_) => continue, // 频次已过期（该 pair 后来被合并改动过）
                    None => break 'train, // 语料已无可合并 pair（如单字节语料）
                }
            };

            // 创建新 token
            let new_id = vocab.len() as u16;
            let mut new_bytes = vocab[best.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best.1 as usize]);
            vocab.push(new_bytes);
            merges.push(best);

            // 单次扫描合并该 pair，同时**增量**更新受影响 pair 的频次。
            // 每处合并只影响三个旧 pair：左邻 (prev, a)、本身 (a, b)、右邻 (b, next)，
            // 对应两个新 pair：(prev, N) 与 (N, next) —— 不需要重扫全语料重新统计。
            buf.clear();
            let mut i = 0;
            while i < ids.len() {
                if i + 1 < ids.len() && ids[i] == best.0 && ids[i + 1] == best.1 {
                    if let Some(&prev) = buf.last() {
                        bump_pair(&mut pair_freq, &mut heap, (prev, best.0), -1);
                        bump_pair(&mut pair_freq, &mut heap, (prev, new_id), 1);
                    }
                    bump_pair(&mut pair_freq, &mut heap, best, -1);
                    if let Some(&next) = ids.get(i + 2) {
                        bump_pair(&mut pair_freq, &mut heap, (best.1, next), -1);
                        bump_pair(&mut pair_freq, &mut heap, (new_id, next), 1);
                    }
                    buf.push(new_id);
                    i += 2;
                } else {
                    buf.push(ids[i]);
                    i += 1;
                }
            }
            std::mem::swap(&mut ids, &mut buf);

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
    /// 贪心合并（通行的标准实现）：按优先级从高到低，对每条合并规则在序列上
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

    /// 序列化成 JSON（[`Tokenizer::save`] 会往这份 JSON 上再补特殊 token 字段）
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "bpe",
            "merges": self.merges.iter().map(|(a, b)| vec![*a, *b]).collect::<Vec<_>>(),
            "vocab": self.vocab.iter().map(|v| v.clone()).collect::<Vec<_>>(),
        })
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

// ==================== 特殊 token ====================

/// 三个特殊 token 的字面写法。语料与 prompt 里可以直接写这三个字符串，
/// [`Tokenizer::encode`] 会把它们识别成对应的特殊 id（不会被 BPE 拆成字节）。
pub const BOS_LITERAL: &str = "<|bos|>";
pub const EOS_LITERAL: &str = "<|eos|>";
pub const PAD_LITERAL: &str = "<|pad|>";

/// 特殊 token 的 id 分配（BOS / EOS / PAD）
///
/// 三个 id 紧跟在**内容词表之后**（`base` = `CharTokenizer` / `BPETokenizer` 的词表大小），
/// 这样分配有两个好处：
/// 1. 内容 token 的 id 一个都没动，旧的 `tokenizer.json` 读进来还是同一套编码；
/// 2. 词表大小 = `base + COUNT`，模型只需在 embedding 表尾追加三行即可容纳——
///    旧 checkpoint 靠 [`crate::checkpoint`] 的"前缀行恢复 + 优化器状态补零"继续用，
///    不必因为加了一个 EOS 就重训。
///
/// 有了真正的 EOS，生成（[`crate::sample`]）与 SFT（[`crate::data`]）不再需要用
/// "语料里高频出现的字符组合"来冒充结束标记：那个做法要求标记的每个字符都被充分训练过，
/// 一旦低频就会把模型带进乱码模式。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SpecialTokens {
    pub bos: usize,
    pub eos: usize,
    pub pad: usize,
}

impl SpecialTokens {
    /// 特殊 token 的个数：词表大小 = 内容词表 + `COUNT`
    pub const COUNT: usize = 3;

    /// 紧跟在内容词表 `base` 之后分配三个 id
    pub fn new(base: usize) -> Self {
        SpecialTokens {
            bos: base,
            eos: base + 1,
            pad: base + 2,
        }
    }

    /// 是否属于这三个特殊 token
    pub fn contains(&self, id: usize) -> bool {
        id >= self.bos && id <= self.pad
    }

    /// id -> 字面写法；不是特殊 token 时返回 `None`
    pub fn name(&self, id: usize) -> Option<&'static str> {
        if id == self.bos {
            Some(BOS_LITERAL)
        } else if id == self.eos {
            Some(EOS_LITERAL)
        } else if id == self.pad {
            Some(PAD_LITERAL)
        } else {
            None
        }
    }

    /// 三个字面量与 id 的对应表（`encode` 切分时用）
    fn literals(&self) -> [(&'static str, usize); Self::COUNT] {
        [
            (BOS_LITERAL, self.bos),
            (EOS_LITERAL, self.eos),
            (PAD_LITERAL, self.pad),
        ]
    }
}

// ==================== 统一分词器接口（配置可切换） ====================

/// 内容分词器的两种实现（只负责"文本 <-> 内容 token"，不认识特殊 token）
enum Inner {
    Char(CharTokenizer),
    Bpe(BPETokenizer),
}

impl Inner {
    fn vocab_size(&self) -> usize {
        match self {
            Inner::Char(t) => t.vocab_size(),
            Inner::Bpe(t) => t.vocab_size(),
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        match self {
            Inner::Char(t) => t.encode(text),
            Inner::Bpe(t) => t.encode(text),
        }
    }

    fn decode(&self, ids: &[usize]) -> String {
        match self {
            Inner::Char(t) => t.decode(ids),
            Inner::Bpe(t) => t.decode(ids),
        }
    }

    fn to_json(&self) -> serde_json::Value {
        match self {
            Inner::Char(t) => t.to_json(),
            Inner::Bpe(t) => t.to_json(),
        }
    }
}

/// 统一的分词器：`"char"` 用 [`CharTokenizer`]，`"bpe"` 用 [`BPETokenizer`]。
///
/// 上层（数据加载 / 训练 / 采样）只依赖这一个接口，不关心具体实现。
/// 注意：`encode` 出来的 id 空间由具体实现决定，两者互不通用。
///
/// 在内容词表之上再挂一组特殊 token（[`SpecialTokens`]）：新建的分词器一律带，
/// 从旧文件读入的（JSON 里没有 `specials` 字段）则保持原样、`vocab_size` 不变，
/// 于是旧 checkpoint 不受影响。
pub struct Tokenizer {
    inner: Inner,
    specials: Option<SpecialTokens>,
}

impl Tokenizer {
    /// 字符级分词器：词表 = 语料中出现的所有字符 + 3 个特殊 token
    pub fn char(text: &str) -> Self {
        Self::wrap(Inner::Char(CharTokenizer::new(text)))
    }

    /// BPE 分词器：在语料上训练，内容词表大小 = 256 + 合并次数（再加 3 个特殊 token）
    pub fn bpe(text: &str, target_vocab: usize) -> Self {
        Self::wrap(Inner::Bpe(BPETokenizer::train(text, target_vocab)))
    }

    /// 按名称构造：`"char"` / `"bpe"`，其余报错
    pub fn from_name(name: &str, corpus: &str, bpe_vocab: usize) -> Self {
        match name {
            "char" => Tokenizer::char(corpus),
            "bpe" => Tokenizer::bpe(corpus, bpe_vocab),
            other => panic!("未知分词器类型 '{}'（可选：char / bpe）", other),
        }
    }

    /// 新建分词器都要挂上特殊 token：id 从内容词表的末尾接着排
    fn wrap(inner: Inner) -> Self {
        let specials = Some(SpecialTokens::new(inner.vocab_size()));
        Tokenizer { inner, specials }
    }

    /// 内容词表大小（不含特殊 token）
    pub fn content_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// 完整词表大小 = 内容词表 + 特殊 token 个数
    pub fn vocab_size(&self) -> usize {
        self.content_vocab_size()
            + self.specials.map_or(0, |_| SpecialTokens::COUNT)
    }

    pub fn specials(&self) -> Option<SpecialTokens> {
        self.specials
    }

    /// 是否带特殊 token（旧文件读入的分词器没有）
    pub fn has_specials(&self) -> bool {
        self.specials.is_some()
    }

    /// 去掉特殊 token，退回"纯内容词表"。
    ///
    /// 只在一个场景用得上：分词器文件丢了、从语料重建时（[`Self::from_name`] 一律带
    /// 特殊 token），而要加载的 checkpoint 是老格式（嵌入表只有内容词表那么多行）。
    /// 模型词表是硬约束——多出来的 3 行没有对应的嵌入行，只能按老格式对齐。
    pub fn without_specials(mut self) -> Self {
        self.specials = None;
        self
    }

    /// 结束标记 id；旧分词器返回 `None`（此时调用方需回退到字符串停止标记）
    pub fn eos_id(&self) -> Option<usize> {
        self.specials.map(|s| s.eos)
    }

    /// 序列起始标记 id；旧分词器返回 `None`
    pub fn bos_id(&self) -> Option<usize> {
        self.specials.map(|s| s.bos)
    }

    fn is_special(&self, id: usize) -> bool {
        self.specials.map_or(false, |s| s.contains(id))
    }

    /// 文本 -> id 序列。
    ///
    /// 带特殊 token 时，文本里的 `<|bos|>` / `<|eos|>` / `<|pad|>` 会被识别成对应的
    /// 特殊 id，其余部分照常交给内容分词器（**不跨字面量做 BPE 合并**——否则字面量
    /// 可能被并进相邻 token，边界就错位了）。
    pub fn encode(&self, text: &str) -> Vec<usize> {
        match self.specials {
            None => self.inner.encode(text),
            Some(sp) => self.encode_with_specials(text, &sp),
        }
    }

    fn encode_with_specials(&self, text: &str, sp: &SpecialTokens) -> Vec<usize> {
        let mut ids: Vec<usize> = Vec::new();
        let mut rest = text;
        loop {
            // 找"最靠前"的那个字面量；同一位置不可能同时命中两个（字面量互不为前缀）
            let mut hit: Option<(usize, usize, usize)> = None; // (位置, id, 字面量字节数)
            for (lit, id) in sp.literals() {
                if let Some(p) = rest.find(lit) {
                    if hit.map_or(true, |(best, _, _)| p < best) {
                        hit = Some((p, id, lit.len()));
                    }
                }
            }
            match hit {
                Some((p, id, len)) => {
                    if p > 0 {
                        ids.extend(self.inner.encode(&rest[..p]));
                    }
                    ids.push(id);
                    rest = &rest[p + len..];
                }
                None => {
                    ids.extend(self.inner.encode(rest));
                    return ids;
                }
            }
        }
    }

    /// id 序列 -> 文本（**丢掉特殊 token**，用于给用户看生成结果）
    pub fn decode(&self, ids: &[usize]) -> String {
        if self.specials.is_none() {
            return self.inner.decode(ids);
        }
        let mut out = String::new();
        let mut run: Vec<usize> = Vec::new();
        for &id in ids {
            if self.is_special(id) {
                if !run.is_empty() {
                    out.push_str(&self.inner.decode(&run));
                    run.clear();
                }
            } else {
                run.push(id);
            }
        }
        if !run.is_empty() {
            out.push_str(&self.inner.decode(&run));
        }
        out
    }

    /// id 序列 -> 文本，特殊 token 保留成 `<|eos|>` 这样的字面写法。
    ///
    /// 调试 / 日志用（等价于 HuggingFace 的 `skip_special_tokens=False`）：
    /// [`Tokenizer::decode`] 会把 EOS 抹掉，想看"模型到底在哪一步收的尾"就得用这个。
    pub fn decode_verbose(&self, ids: &[usize]) -> String {
        let Some(sp) = self.specials else {
            return self.inner.decode(ids);
        };
        let mut out = String::new();
        let mut run: Vec<usize> = Vec::new();
        for &id in ids {
            match sp.name(id) {
                Some(name) => {
                    if !run.is_empty() {
                        out.push_str(&self.inner.decode(&run));
                        run.clear();
                    }
                    out.push_str(name);
                }
                None => run.push(id),
            }
        }
        if !run.is_empty() {
            out.push_str(&self.inner.decode(&run));
        }
        out
    }

    /// 内容词表中每个 token 的字节序列，供生成时做 UTF-8 约束（见 [`utf8_pending`]）。
    ///
    /// `char` 分词器每个 token 都是完整字符，不可能拼出非法 UTF-8，因此返回 `None`。
    /// 返回的切片**不含**特殊 token（长度为 [`Tokenizer::content_vocab_size`]）：
    /// UTF-8 约束的循环用 `get(id)` 取值，特殊 token 自然落到 `None` 分支——它们没有
    /// 字节语义，任何时候都允许被采样到（正是 EOS 需要的）。
    pub fn vocab_bytes(&self) -> Option<&[Vec<u8>]> {
        match &self.inner {
            Inner::Char(_) => None,
            Inner::Bpe(t) => Some(&t.vocab),
        }
    }

    /// 类型名（打印用）："char" / "bpe"
    pub fn kind(&self) -> &'static str {
        match self.inner {
            Inner::Char(_) => "char",
            Inner::Bpe(_) => "bpe",
        }
    }

    /// 保存分词器到文件（父目录不存在时自动创建）。
    ///
    /// 内容部分与旧格式**逐字段一致**，只在末尾追加一个 `specials` 字段——
    /// 因此新版分词器文件可以被旧版代码读出内容词表（旧版忽略未知字段），
    /// 而新版代码读旧文件时 `specials` 缺席，就按"没有特殊 token"处理。
    pub fn save(&self, path: &str) {
        let mut json = self.inner.to_json();
        if let Some(sp) = self.specials {
            json["specials"] = serde_json::json!({
                "bos": sp.bos, "eos": sp.eos, "pad": sp.pad,
            });
        }
        write_json(path, &json);
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
        Self::from_json(&json)
    }

    /// 从已解析的 JSON 构造（`load` 与测试共用）
    pub fn from_json(json: &serde_json::Value) -> Self {
        let typ = json["type"]
            .as_str()
            .expect("分词器文件格式错误：缺少 type 字段");
        let inner = match typ {
            "char" => Inner::Char(CharTokenizer::from_json(json)),
            "bpe" => Inner::Bpe(BPETokenizer::from_json(json)),
            other => panic!("未知分词器类型 '{}'（可选：char / bpe）", other),
        };
        let specials = if json["specials"].is_object() {
            let s = &json["specials"];
            let read = |k: &str| {
                s[k].as_u64()
                    .unwrap_or_else(|| panic!("分词器 specials.{k} 缺失或不是整数")) as usize
            };
            let sp = SpecialTokens {
                bos: read("bos"),
                eos: read("eos"),
                pad: read("pad"),
            };
            // 特殊 token 必须紧跟在内容词表之后，否则 id 空间有洞
            assert_eq!(
                sp.bos,
                inner.vocab_size(),
                "分词器文件损坏：specials.bos（{}）不等于内容词表大小（{}）",
                sp.bos,
                inner.vocab_size()
            );
            Some(sp)
        } else {
            None
        };
        Tokenizer { inner, specials }
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

    /// 特殊 token 紧跟在内容词表之后：内容 token 的 id 一个都没变。
    #[test]
    fn test_special_tokens_are_appended_after_content_vocab() {
        let corpus = "hello world hello";
        let tok = Tokenizer::char(corpus);
        let sp = tok.specials().expect("新建分词器应带特殊 token");

        assert_eq!(tok.content_vocab_size(), 8); // h,e,l,o,' ',w,r,d
        assert_eq!(sp.bos, 8);
        assert_eq!(sp.eos, 9);
        assert_eq!(sp.pad, 10);
        assert_eq!(tok.vocab_size(), 11);
        assert_eq!(tok.eos_id(), Some(9));

        // 内容字符的编码与不带特殊 token 的字符级分词器完全一致
        let plain = CharTokenizer::new(corpus);
        assert_eq!(tok.encode("hello"), plain.encode("hello"));
    }

    /// 字面量 `<|eos|>` 要被识别成特殊 id，而不是被 BPE 拆成一串字节 token；
    /// 它旁边的正文照常编码，且**不会**跨字面量做合并。
    #[test]
    fn test_encode_recognizes_special_literals() {
        let tok = Tokenizer::bpe("你好世界你好世界", 300);
        let eos = tok.eos_id().unwrap();
        let bos = tok.bos_id().unwrap();

        let ids = tok.encode(&format!("{BOS_LITERAL}你好{EOS_LITERAL}"));
        assert_eq!(ids[0], bos);
        assert_eq!(*ids.last().unwrap(), eos);
        // 中间部分与单独编码"你好"一致（没有跨字面量合并）
        assert_eq!(&ids[1..ids.len() - 1], &tok.encode("你好")[..]);

        // 空文本、纯字面量、开头就是字面量都不出错
        assert!(tok.encode("").is_empty());
        assert_eq!(tok.encode(EOS_LITERAL), vec![eos]);
        assert_eq!(tok.encode(&format!("{EOS_LITERAL}abc"))[0], eos);
    }

    /// `decode` 丢掉特殊 token（给用户看的文本），`decode_verbose` 保留字面写法（调试用）。
    #[test]
    fn test_decode_skips_specials_and_verbose_keeps_them() {
        let tok = Tokenizer::bpe("你好世界你好世界", 300);
        let ids = tok.encode(&format!("{BOS_LITERAL}你好{EOS_LITERAL}{PAD_LITERAL}"));
        assert_eq!(tok.decode(&ids), "你好");
        assert_eq!(
            tok.decode_verbose(&ids),
            format!("{BOS_LITERAL}你好{EOS_LITERAL}{PAD_LITERAL}")
        );
    }

    /// 特殊 token 必须能存活 save/load 往返；内容词表部分与不带特殊 token 的写法一致。
    #[test]
    fn test_specials_survive_save_load() {
        let tok = Tokenizer::bpe("你好世界你好世界", 300);
        let mut path = std::env::temp_dir();
        path.push(format!("llm_tok_test_{}.json", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        tok.save(&path);
        let back = Tokenizer::load(&path);
        let _ = std::fs::remove_file(&path);

        assert_eq!(back.vocab_size(), tok.vocab_size());
        assert_eq!(back.eos_id(), tok.eos_id());
        assert_eq!(back.kind(), "bpe");
        assert_eq!(back.encode("你好世界"), tok.encode("你好世界"));
    }

    /// 旧格式分词器文件（没有 `specials` 字段）照常读入，词表大小不变——
    /// 这样旧 checkpoint 不会因为本次改动而失效。
    #[test]
    fn test_legacy_tokenizer_file_loads_without_specials() {
        let legacy = serde_json::json!({
            "type": "char",
            "chars": ["h", "i"],
        });
        let tok = Tokenizer::from_json(&legacy);
        assert!(!tok.has_specials());
        assert_eq!(tok.vocab_size(), 2);
        assert_eq!(tok.eos_id(), None);
        assert_eq!(tok.encode("hi"), vec![0, 1]);
        assert_eq!(tok.decode(&[1, 0]), "ih");
    }
}
