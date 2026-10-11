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
    /// 合并规则，按下标顺序（越早合并优先级越高）。
    ///
    /// 三元组是 `(左 id, 右 id, 合并后的 id)`——**合并结果 id 必须显式存**：
    /// 自研训练时它恰好是 `256 + 合并序号`，但 HF 的词表把特殊 token 排在前面
    /// （LLaMA-2 的 `<unk>/<s></s>` 占 0/1/2），"合并结果 = 256+序号"从此不成立。
    /// 老存档里的二元组按 `256 + 序号` 还原（见 [`Self::from_json`]），行为逐位不变。
    merges: Vec<(u32, u32, u32)>,
    /// token id -> 它代表的字节序列
    vocab: Vec<Vec<u8>>,
    /// 每个**单字节**对应的初始 token id（长度 256）。
    ///
    /// 自研训练时恒为 `byte_id[b] = b`（字节 token 就是 0..256），但 HF 的词表常把
    /// 特殊 token 排在前面（LLaMA-2 的 `<unk>/<s>/</s>` 占 0/1/2，字节 token 从 3 开始），
    /// 那时"字节 b 的 id"必须查表。老存档没有这个字段 ⇒ 按恒等映射还原（逐位不变）。
    byte_id: Vec<u32>,
    /// 编码前是否先按 **GPT-2 正则**切分（HF 的 `pre_tokenizer: ByteLevel` 口径）。
    ///
    /// `false`（默认）= 本仓库自研分词器的历史行为：整段文本一起做 BPE 合并。
    /// HF 的词表是在"先切分、再逐片合并"的前提下训出来的，不切分就会跨边界合并出
    /// 词表里不存在的 token 序列，与 HF 的输出对不上，所以导入时必须置 `true`。
    pretok: bool,
}

/// 增量调整 pair 频次（delta = ±1），并把新值推进优先队列。
///
/// 队列里同一 pair 可能积累多个条目，pop 时按 `pair_freq` 当前值校验，
/// 不一致的视为过期直接丢弃（懒删除）——避免每轮合并后重建整个堆。
fn bump_pair(
    pair_freq: &mut HashMap<(u32, u32), usize>,
    heap: &mut BinaryHeap<(usize, Reverse<(u32, u32)>)>,
    pair: (u32, u32),
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
        let mut vocab: Vec<Vec<u8>> = (0u32..=255).map(|b| vec![b as u8]).collect();
        let mut merges: Vec<(u32, u32, u32)> = Vec::new();
        let mut ids: Vec<u32> = data.iter().map(|&b| b as u32).collect();

        // pair -> 当前频次：初始扫一遍，之后随每次合并增量增减，
        // 不再像旧实现那样每轮全量重扫语料重建 HashMap（O(merges × corpus) 的主项之一）
        let mut pair_freq: HashMap<(u32, u32), usize> = HashMap::new();
        for w in ids.windows(2) {
            *pair_freq.entry((w[0], w[1])).or_insert(0) += 1;
        }
        // 最大堆：键 (频次, Reverse(pair))，频次最高者在顶、平手取 pair 值小者
        // （确定性 tie-break，与旧实现的 max_by 规则一致）。
        // 频次每次变化都 push 新条目，pop 时按 pair_freq 校验，过期条目直接丢弃（懒删除）。
        let mut heap: BinaryHeap<(usize, Reverse<(u32, u32)>)> = BinaryHeap::new();
        for (&p, &f) in &pair_freq {
            if f > 0 {
                heap.push((f, Reverse(p)));
            }
        }

        let target_merges = target_vocab - 256;
        let log_interval = if target_merges >= 20 { target_merges / 20 } else { 1 };
        // 双缓冲交替复用：每轮合并结果写进 buf 再 swap，避免逐轮重新分配大 Vec
        let mut buf: Vec<u32> = Vec::with_capacity(ids.len());

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
            let new_id = vocab.len() as u32;
            let mut new_bytes = vocab[best.0 as usize].clone();
            new_bytes.extend_from_slice(&vocab[best.1 as usize]);
            vocab.push(new_bytes);
            merges.push((best.0, best.1, new_id));

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

        BPETokenizer { merges, vocab, byte_id: (0u32..256).collect(), pretok: false }
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 文本 -> token id 序列
    ///
    /// 贪心合并（通行的标准实现）：按优先级从高到低，对每条合并规则在序列上
    /// 做一趟扫描替换。复杂度 O(len × 合并数)，大语料也能秒级完成。
    ///
    /// `pretok = true`（HF 口径）时**先按 GPT-2 正则切成片、再逐片合并**：
    /// HF 的词表就是在切分后的片段上训出来的，整段一起合并会跨词/跨标点边界拼出
    /// 词表里不存在的序列，与 HF 的分词结果对不上。
    pub fn encode(&self, text: &str) -> Vec<usize> {
        if self.pretok {
            let mut out = Vec::new();
            for piece in gpt2_pretokenize(text) {
                out.extend(self.merge_piece(&piece));
            }
            return out;
        }
        self.merge_piece(text.as_bytes())
    }

    /// 对一段字节做 BPE 合并（`encode` 的单片实现，两种口径共用）
    fn merge_piece(&self, bytes: &[u8]) -> Vec<usize> {
        let mut ids: Vec<u32> = bytes.iter().map(|&b| self.byte_id[b as usize]).collect();
        for &(a, b, new_id) in self.merges.iter() {
            let mut out: Vec<u32> = Vec::with_capacity(ids.len());
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
            "pretok": self.pretok,
            "byte_ids": self.byte_id,
            "merges": self.merges.iter().map(|(a, b, r)| vec![*a, *b, *r]).collect::<Vec<_>>(),
            "vocab": self.vocab.iter().map(|v| v.clone()).collect::<Vec<_>>(),
        })
    }

    /// 从已解析的 JSON 值构造。
    ///
    /// `merges` 兼容两种写法：三元组 `[a, b, 结果 id]`（新，HF 导入必须用这种，
    /// 因为 HF 的合并结果 id 未必等于 `256 + 序号`）与二元组 `[a, b]`
    /// （旧存档，按 `256 + 序号` 还原——自研训练的口径，逐位不变）。
    pub fn from_json(json: &serde_json::Value) -> Self {
        let merges: Vec<(u32, u32, u32)> = json["merges"]
            .as_array()
            .expect("分词器文件格式错误：缺少 merges 字段")
            .iter()
            .enumerate()
            .map(|(idx, v)| {
                let arr = v.as_array().expect("merge 应为数组");
                let a = arr[0].as_u64().unwrap() as u32;
                let b = arr[1].as_u64().unwrap() as u32;
                let r = arr
                    .get(2)
                    .and_then(|x| x.as_u64())
                    .map(|x| x as u32)
                    .unwrap_or(256 + idx as u32);
                (a, b, r)
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
        let pretok = json["pretok"].as_bool().unwrap_or(false);
        // 老存档没有 byte_ids ⇒ 恒等映射（自研训练的口径，逐位不变）
        let byte_id: Vec<u32> = match json["byte_ids"].as_array() {
            Some(a) => a.iter().map(|x| x.as_u64().unwrap_or(0) as u32).collect(),
            None => (0u32..256).collect(),
        };
        assert_eq!(byte_id.len(), 256, "byte_ids 必须恰好 256 项");
        BPETokenizer { merges, vocab, byte_id, pretok }
    }
}

/// GPT-2 的**预切分正则**（HF `pre_tokenizer: ByteLevel` 的实际口径）的手写实现。
///
/// 原正则（HuggingFace / OpenAI GPT-2 用的那一版）：
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
/// 本仓库刻意不引 `regex` 依赖（依赖表保持精简），所以按分支手写。各分支的优先级
/// 就是正则里从左到右的顺序，且**最先匹配者胜**（这是 `|` 的语义）。
///
/// 为什么要它：HF 的字节级 BPE 是"先切分、再逐片合并"。片段边界上永远不会有 token 跨越，
/// 所以"the" 与 "  the" 是不同的 token、数字最多按 3 位一组切、标点自成一类。
/// 少了这一步，`encode` 会在边界处拼出 HF 词表里不存在的序列，输出与 HF 不一致。
pub fn gpt2_pretokenize(text: &str) -> Vec<Vec<u8>> {
    let b = text.as_bytes();
    let mut out: Vec<Vec<u8>> = Vec::new();
    let mut i = 0usize;
    // 判断某个字节是否算"字母"：以 ASCII 字母 + 任何非 ASCII 字节（≈ Unicode 字母）近似。
    // 与 `\p{L}` 的差别只出现在非 ASCII 的标点/数字上，而那些字符的字节都在 0x80 以上，
    // 会落进"非空白非字母数字"分支——对本仓库支持的模型（LLaMA/Qwen 的中英混合语料）
    // 结果一致；真要严格对齐需要 `regex` 的 Unicode 属性支持（见模块测试的说明）。
    let is_letter = |c: u8| c.is_ascii_alphabetic() || c >= 0x80;
    let is_num = |c: u8| c.is_ascii_digit();
    let is_space = |c: u8| c == b' ' || c == b'\t' || c == b'\n' || c == b'\r' || c == 0x0b || c == 0x0c;
    while i < b.len() {
        let start = i;
        // 分支 1：英文缩写后缀（'s / 't / 're / 've / 'm / 'll / 'd）
        if b[i] == b'\'' && i + 1 < b.len() {
            let rest = &b[i + 1..];
            let hit = [
                (b"s".as_slice(), 1usize),
                (b"t".as_slice(), 1),
                (b"re".as_slice(), 2),
                (b"ve".as_slice(), 2),
                (b"m".as_slice(), 1),
                (b"ll".as_slice(), 2),
                (b"d".as_slice(), 1),
            ]
            .into_iter()
            .find(|(s, n)| rest.len() >= *n && &rest[..*n] == *s);
            if let Some((_, n)) = hit {
                i += 1 + n;
                out.push(b[start..i].to_vec());
                continue;
            }
        }
        // 分支 2/3：可选的**单个前导空格** + 一串字母（或数字）
        let mut j = i;
        if b[j] == b' ' && j + 1 < b.len() && (is_letter(b[j + 1]) || is_num(b[j + 1])) {
            j += 1;
        }
        if j < b.len() && is_letter(b[j]) {
            while j < b.len() && is_letter(b[j]) {
                j += 1;
            }
            i = j;
            out.push(b[start..i].to_vec());
            continue;
        }
        if j < b.len() && is_num(b[j]) {
            // 数字：原正则用 `\p{N}+`（≥1 位连续）。这里按连续 ASCII 数字切，
            // 与 LLaMA/Qwen 词表的实际训练口径一致（它们的分词器把数字整串交给 BPE）。
            while j < b.len() && is_num(b[j]) {
                j += 1;
            }
            i = j;
            out.push(b[start..i].to_vec());
            continue;
        }
        // 分支 4：可选的单个前导空格 + 一串标点/符号（非空白、非字母、非数字）
        let mut j = i;
        if b[j] == b' ' && j + 1 < b.len() && !is_space(b[j + 1]) && !is_letter(b[j + 1]) && !is_num(b[j + 1]) {
            j += 1;
        }
        if j < b.len() && !is_space(b[j]) && !is_letter(b[j]) && !is_num(b[j]) {
            while j < b.len() && !is_space(b[j]) && !is_letter(b[j]) && !is_num(b[j]) {
                j += 1;
            }
            i = j;
            out.push(b[start..i].to_vec());
            continue;
        }
        // 分支 5：空白串（原正则的 `\s+(?!\S)` / `\s+`）。
        // 关键细节：**末尾那个空白要留给下一个词**（`\s+(?!\S)` 的负向前瞻会让贪婪匹配
        // 回退一格），于是 "  x" 切成 `" "` + `" x"` 而不是 `"  "` + `"x"`。
        if is_space(b[i]) {
            let mut e = i;
            while e < b.len() && is_space(b[e]) {
                e += 1;
            }
            // 后面还有内容 ⇒ 只吃到倒数第二个空白；到末尾 ⇒ 整段收下
            let cut = if e < b.len() { (e - 1).max(i + 1) } else { e };
            i = cut;
            out.push(b[start..i].to_vec());
            continue;
        }
        // 兜底：单个字节（正常情况下走不到这里）
        i += 1;
        out.push(b[start..i].to_vec());
    }
    out
}

/// GPT-2 的**字节 ↔ Unicode** 映射反查表（char → 原始字节）。
///
/// HF 的 `ByteLevel` 预处理器先把 256 个任意字节装进可打印 Unicode 码位
/// （`!`..`~`、`¡`..`¬`、`®`..`ÿ` 原样保留，其余 68 个映射到 `U+0100` 起），
/// 这样词表在 JSON 里就是合法字符串。导入时要把它**反查回字节**，
/// 否则 `vocab` 里存的是码位而不是原始字节，decode 会输出乱码。
fn gpt2_byte_decoder() -> HashMap<char, u8> {
    let mut direct: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
    direct.extend(0xA1u32..=0xAC);
    direct.extend(0xAEu32..=0xFF);
    let mut map = HashMap::new();
    for &c in &direct {
        map.insert(char::from_u32(c).expect("可打印区码位必合法"), c as u8);
    }
    let mut n = 0u32;
    for b in 0u32..256 {
        if !direct.contains(&b) {
            map.insert(
                char::from_u32(256 + n).expect("U+0100 起连续码位必合法"),
                b as u8,
            );
            n += 1;
        }
    }
    map
}

impl Tokenizer {
    /// 从 **HuggingFace 的 `tokenizer.json`** 构造分词器（LLaMA / Qwen2 系的字节级 BPE）。
    ///
    /// 转换与校验（每一步都宁可报错，不静默降级）：
    /// 1. `model.vocab` 的 token 字符串按 GPT-2 字节表还原成字节 → 我们的 `vocab`，
    ///    并要求 id **稠密**（0..N 无空洞）；
    /// 2. `model.merges`（`"a b"`）按字符串查 id，**结果 id 取 `a+b` 拼接串的 id**
    ///    （HF 的合并结果 id 未必是 `256+序号`）；并校验
    ///    `bytes(a+b) == bytes(a) ++ bytes(b)`——这条能一举抓住字节表反查错、vocab 读错；
    /// 3. `added_tokens` 参与词表与特殊 token 识别：`<s>`/`<|begin_of_text|>` → BOS，
    ///    `</s>`/`<|end_of_text|>`/`<|im_end|>` → EOS，其余 `special: true` 的进
    ///    `extra_literals`（编码时整体命中，不被 BPE 拆开）；
    /// 4. `pre_tokenizer` 必须是 ByteLevel（GPT-2 正则）口径 ⇒ 打开 `pretok`；
    ///    非空 `normalizer` 只放行 `NFC`（对已是 NFC 的文本是恒等变换），其余报错。
    pub fn load_hf(path: &str) -> Result<Self, String> {
        let text = std::fs::read_to_string(path).map_err(|e| format!("读不到 {path}: {e}"))?;
        let j: serde_json::Value =
            serde_json::from_str(&text).map_err(|e| format!("解析 {path} 失败: {e}"))?;
        let model = j
            .get("model")
            .ok_or_else(|| "tokenizer.json 缺 model 字段".to_string())?;
        let mtype = model.get("type").and_then(|x| x.as_str()).unwrap_or("");
        if mtype != "BPE" {
            return Err(format!(
                "只支持字节级 BPE 分词器（HF 的 model.type = `{mtype}`）：\
                 SentencePiece / WordPiece / Unigram 需要另做转换"
            ));
        }
        if let Some(nrm) = j.get("normalizer") {
            if !nrm.is_null() {
                // 只放行 **NFC**：Qwen2.5 系列带 `{"type":"NFC"}`（Qwen2 是 null，所以以前能过）。
                // NFC = Unicode 规范合成（`e` + U+0301 → `é`），对**已经处于 NFC 的文本是
                // 恒等变换**——中文、常规英文、正常输入的文本都已是 NFC，分词结果与 HF
                // 完全一致；只有"组合字符序列"这类输入才会不同，所以这里放行并打印提示。
                // 其余归一化形式（NFD / NFKC / NFKD / Sequence）会真正改变字节序列，仍然拒绝。
                let t = nrm.get("type").and_then(|x| x.as_str()).unwrap_or("");
                if t != "NFC" {
                    return Err(format!(
                        "tokenizer.json 带 normalizer（{nrm}）：本仓库只放行 `NFC`\
                         （对已是 NFC 的文本是恒等变换），`{t}` 会改变字节序列、与 HF 分词结果不一致"
                    ));
                }
                println!(
                    "  提示：该分词器声明了 NFC 归一化。本仓库不实现 Unicode 归一化，\
                     对已是 NFC 的文本（中文/常规英文）分词结果与 HF 一致；\
                     含组合字符序列的输入会有差异（如 `e`+U+0301 不会被合成 `é`）。"
                );
            }
        }
        let pt = serde_json::to_string(&j["pre_tokenizer"]).unwrap_or_default();
        if !pt.contains("ByteLevel") {
            return Err(format!(
                "pre_tokenizer 不是 ByteLevel 口径（实为 {pt}）：缺了 GPT-2 正则切分，\
                 分词结果会与 HF 不一致"
            ));
        }
        let enc = gpt2_byte_decoder();
        let dec = |s: &str| -> Result<Vec<u8>, String> {
            s.chars()
                .map(|c| {
                    enc.get(&c).copied().ok_or_else(|| {
                        format!("token `{s}` 含无法映射的字符 U+{:04X}", c as u32)
                    })
                })
                .collect()
        };
        // 字符串 → id
        let mut str2id: HashMap<String, u32> = HashMap::new();
        let vocab_obj = model
            .get("vocab")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "tokenizer.json 缺 model.vocab".to_string())?;
        for (tok, id) in vocab_obj {
            let id = id
                .as_u64()
                .ok_or_else(|| format!("model.vocab 的 `{tok}` 不是整数 id"))? as u32;
            str2id.insert(tok.clone(), id);
        }
        let mut specials_raw: Vec<(String, u32)> = Vec::new();
        if let Some(arr) = j.get("added_tokens").and_then(|x| x.as_array()) {
            for t in arr {
                let content = t.get("content").and_then(|c| c.as_str()).unwrap_or("");
                let id = t.get("id").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
                if content.is_empty() {
                    continue;
                }
                str2id.entry(content.to_string()).or_insert(id);
                if t.get("special").and_then(|x| x.as_bool()).unwrap_or(false) {
                    specials_raw.push((content.to_string(), id));
                }
            }
        }
        // id → 字节
        let n = str2id.values().copied().max().map(|m| m as usize + 1).unwrap_or(0);
        let mut slots: Vec<Option<Vec<u8>>> = vec![None; n];
        for (tok, id) in &str2id {
            let bytes = dec(tok)?;
            match &slots[*id as usize] {
                Some(prev) if *prev != bytes => {
                    return Err(format!("词表 id {id} 同时对应两组字节：{prev:?} vs {bytes:?}"))
                }
                _ => slots[*id as usize] = Some(bytes),
            }
        }
        let vocab: Vec<Vec<u8>> = slots
            .into_iter()
            .enumerate()
            .map(|(i, v)| v.ok_or_else(|| format!("词表 id {i} 没有对应 token（HF 词表应稠密）")))
            .collect::<Result<_, _>>()?;
        // 字节 → id
        let mut byte_id = vec![u32::MAX; 256];
        for (id, b) in vocab.iter().enumerate() {
            if b.len() == 1 && byte_id[b[0] as usize] == u32::MAX {
                byte_id[b[0] as usize] = id as u32;
            }
        }
        if let Some(miss) = byte_id.iter().position(|x| *x == u32::MAX) {
            return Err(format!(
                "词表里找不到字节 {miss} 的单字节 token：不是字节级 BPE 词表，无法导入"
            ));
        }
        // merges
        let merges_arr = model
            .get("merges")
            .and_then(|x| x.as_array())
            .ok_or_else(|| "tokenizer.json 缺 model.merges".to_string())?;
        let mut merges: Vec<(u32, u32, u32)> = Vec::with_capacity(merges_arr.len());
        for (idx, m) in merges_arr.iter().enumerate() {
            // 老导出是 `"a b"` 字符串，新导出也有 `["a","b"]` 数组形式
            let (sa, sb) = match m {
                serde_json::Value::String(s) => {
                    let mut it = s.splitn(2, ' ');
                    (
                        it.next().unwrap_or("").to_string(),
                        it.next().unwrap_or("").to_string(),
                    )
                }
                serde_json::Value::Array(a) if a.len() == 2 => (
                    a[0].as_str().unwrap_or("").to_string(),
                    a[1].as_str().unwrap_or("").to_string(),
                ),
                _ => return Err(format!("第 {idx} 条 merges 既不是 \"a b\" 也不是二元数组")),
            };
            let ia = *str2id
                .get(&sa)
                .ok_or_else(|| format!("merge 第 {idx} 条的左半 `{sa}` 不在词表里"))?;
            let ib = *str2id
                .get(&sb)
                .ok_or_else(|| format!("merge 第 {idx} 条的右半 `{sb}` 不在词表里"))?;
            let joined = format!("{sa}{sb}");
            let ir = *str2id
                .get(&joined)
                .ok_or_else(|| format!("merge 第 {idx} 条的结果 `{joined}` 不在词表里"))?;
            let mut cat = dec(&sa)?;
            cat.extend_from_slice(&dec(&sb)?);
            if cat != dec(&joined)? {
                return Err(format!(
                    "merge 第 {idx} 条（`{sa}` + `{sb}`）的字节拼接与结果 token 不一致：{cat:?} vs {:?}",
                    dec(&joined)?
                ));
            }
            merges.push((ia, ib, ir));
        }
        // 特殊 token：按常见写法挑
        let pick = |cands: &[&str], who: &str| -> Result<usize, String> {
            cands
                .iter()
                .find_map(|c| str2id.get(*c).copied())
                .map(|x| x as usize)
                .ok_or_else(|| {
                    format!(
                        "找不到 {who}（试过 {cands:?}）：请确认 tokenizer.json 完整（含 added_tokens）"
                    )
                })
        };
        let bos = pick(&["<s>", "<|begin_of_text|>", "<|im_start|>"], "BOS")?;
        let eos = pick(
            &["</s>", "<|end_of_text|>", "<|im_end|>", "<|endoftext|>"],
            "EOS",
        )?;
        let pad = pick(&["<pad>", "<|endoftext|>", "<unk>"], "PAD").unwrap_or(eos);
        // 额外字面量：所有 special 里除了已认领的 BOS/EOS/PAD 之外的
        let extra_literals: Vec<(String, usize)> = specials_raw
            .into_iter()
            .filter(|(_, id)| (*id as usize) != bos && (*id as usize) != eos && (*id as usize) != pad)
            .map(|(t, id)| (t, id as usize))
            .collect();
        let inner = Inner::Bpe(BPETokenizer { merges, vocab, byte_id, pretok: true });
        Ok(Tokenizer {
            inner,
            specials: Some(SpecialTokens { bos, eos, pad }),
            image: None,
            extra_literals,
        })
    }
}

// ==================== 特殊 token ====================

/// 三个特殊 token 的字面写法。语料与 prompt 里可以直接写这三个字符串，
/// [`Tokenizer::encode`] 会把它们识别成对应的特殊 id（不会被 BPE 拆成字节）。
pub const BOS_LITERAL: &str = "<|bos|>";
pub const EOS_LITERAL: &str = "<|eos|>";
pub const PAD_LITERAL: &str = "<|pad|>";
/// 图片占位符字面写法。挂在分词器上（[`ImageTokens`]）后，
/// [`Tokenizer::encode`] 会把它展开成 `[start, ph₀..ph_{P-1}, end]` ——
/// P = (image_size / patch_size)² 个与 ViT patch 数一一对应的占位 token。
pub const IMAGE_LITERAL: &str = "<|image|>";

/// 图片相关 token 的 id 分配。
///
/// 排在三个常规特殊 token（BOS/EOS/PAD）之后，形成一段连续区间：
///
/// ```text
/// [内容词表][bos][eos][pad][start][ph₀..ph_{P-1}][end][vq₀..vq_{K-1}]
/// ```
///
/// - `start` / `ph*` / `end`：**图片理解**用。`<|image|>` 展开成这一整段，
///   占位符个数 P 与 patch 数相同，于是视觉 embedding 可以**原位覆写**
///   占位符位置（序列长度不变，loss 对齐与 KV cache 都不受影响）。
/// - `cb*`：**图片生成**用，是 VQ-VAE 码本 id 的直接偏移
///   （真实 code = `cb_first + 码本下标`），训练序列里它们就是普通 token。
///
/// 这一段 id 同样排在内容词表之后，旧 checkpoint 靠前缀行恢复继续可用。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ImageTokens {
    /// 图片段起始标记
    pub start: usize,
    /// 占位符区间首 id（共 [`Self::count`] 个）
    pub ph_first: usize,
    /// 占位符个数 P = (image_size / patch_size)²
    pub count: usize,
    /// 图片段结束标记
    pub end: usize,
    /// VQ 码本 token 区间首 id（共 [`Self::cb_count`] 个）
    pub cb_first: usize,
    /// VQ 码本大小 K
    pub cb_count: usize,
}

impl ImageTokens {
    /// 追加到词表里的 id 总数：start + P 个占位符 + end + K 个码本 token
    pub fn len(&self) -> usize {
        2 + self.count + self.cb_count
    }

    /// id 是否属于图片区间（占位符 / 起止标记 / 码本）
    pub fn contains(&self, id: usize) -> bool {
        id >= self.start && id < self.start + self.len()
    }

    /// 真实 VQ 码本下标 -> 词表 id
    pub fn cb_id(&self, code: usize) -> usize {
        assert!(code < self.cb_count, "VQ 码 {code} 超出码本大小 {}", self.cb_count);
        self.cb_first + code
    }

    /// 词表 id -> 真实 VQ 码本下标（不属于码本区间时返回 None）
    pub fn cb_index(&self, id: usize) -> Option<usize> {
        let i = id.checked_sub(self.cb_first)?;
        (i < self.cb_count).then_some(i)
    }

    /// 展开 `<|image|>` 得到的完整 id 段：`[start, ph₀..ph_{P-1}, end]`
    pub fn expand(&self) -> Vec<usize> {
        let mut ids = Vec::with_capacity(self.count + 2);
        ids.push(self.start);
        ids.extend(self.ph_first..self.ph_first + self.count);
        ids.push(self.end);
        ids
    }
}

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
    /// 图片 token 区间；`None` = 该分词器不支持多模态（旧文件读入的一律为 None）
    image: Option<ImageTokens>,
    /// **额外**的特殊 token 字面量 `(写法, id)`：HF 导入时从 `added_tokens` 抄过来
    /// （`<|im_start|>` / `<|endoftext|>` 这类）。自研分词器为空 ⇒ 只认固定的三个字面量，
    /// 行为逐位不变。`encode` 会先按这些字面量把文本切开、原样映射成对应 id，
    /// 而不是让 BPE 把它们拆成字节。
    extra_literals: Vec<(String, usize)>,
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
        Tokenizer { inner, specials, image: None, extra_literals: Vec::new() }
    }

    /// 挂上图片 token 区间（id 紧跟在 PAD 之后），供图片理解/生成使用。
    ///
    /// - `patch_count`：一张图的 patch 数 P = (image_size / patch_size)²
    /// - `codebook_size`：VQ-VAE 码本大小 K（生成任务用；纯理解可传 0）
    ///
    /// 幂等：重复调用以最后一次参数为准。未带特殊 token 的旧分词器返回 `None`
    /// （id 空间没有 PAD 可以接续，无法安置图片区间）。
    pub fn attach_image_tokens(
        &mut self,
        patch_count: usize,
        codebook_size: usize,
    ) -> Option<ImageTokens> {
        let sp = self.specials?;
        let im = ImageTokens {
            start: sp.pad + 1,
            ph_first: sp.pad + 2,
            count: patch_count,
            end: sp.pad + 2 + patch_count,
            cb_first: sp.pad + 3 + patch_count,
            cb_count: codebook_size,
        };
        self.image = Some(im);
        Some(im)
    }

    /// 图片 token 区间；`None` = 该分词器不支持多模态
    pub fn image_tokens(&self) -> Option<ImageTokens> {
        self.image
    }

    /// 内容词表大小（不含特殊 token）
    pub fn content_vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// 完整词表大小 = 内容词表 + 特殊 token 个数 + 图片 token 个数
    pub fn vocab_size(&self) -> usize {
        // 自研分词器：内容词表 + 3 个特殊 token（+ 图片段），特殊 token 紧跟在内容词表之后。
        // HF 导入的分词器（`pretok = true`）：特殊 token **本来就在内容词表里**
        // （LLaMA 的 `<unk>/<s></s>` 占 0/1/2），再 +3 就会凭空多出 3 个 id、
        // 与模型存档里的 `vocab_size` 对不上。
        let content = self.content_vocab_size();
        if matches!(&self.inner, Inner::Bpe(b) if b.pretok) {
            let top = self.specials.map_or(0, |s| s.bos.max(s.eos).max(s.pad) + 1);
            content.max(top) + self.image.map_or(0, |im| im.len())
        } else {
            content
                + self.specials.map_or(0, |_| SpecialTokens::COUNT)
                + self.image.map_or(0, |im| im.len())
        }
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
        self.image = None;
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

    /// 填充标记 id；旧分词器返回 `None`（多模态序列对齐填充必须用它，见 `data::VlmLoader`）
    pub fn pad_id(&self) -> Option<usize> {
        self.specials.map(|s| s.pad)
    }

    fn is_special(&self, id: usize) -> bool {
        self.specials.map_or(false, |s| s.contains(id)) || self.image.map_or(false, |im| im.contains(id))
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
            // 命中结果三选一：普通特殊 token 的 id、图片占位段（展开）、没有命中
            #[derive(Clone, Copy)]
            enum Hit {
                Id(usize),
                Image,
            }
            let mut hit: Option<(usize, Hit, usize)> = None; // (位置, 命中, 字面量字节数)
            let note = |p: usize, h: Hit, len: usize, hit: &mut Option<(usize, Hit, usize)>| {
                if hit.map_or(true, |(best, _, _)| p < best) {
                    *hit = Some((p, h, len));
                }
            };
            for (lit, id) in sp.literals() {
                if let Some(p) = rest.find(lit) {
                    note(p, Hit::Id(id), lit.len(), &mut hit);
                }
            }
            // HF 导入的额外特殊 token（`<|im_start|>` / `<|endoftext|>` …）：
            // 同样按字面量整体命中，绝不让 BPE 把它们拆成字节
            for (lit, id) in &self.extra_literals {
                if let Some(p) = rest.find(lit.as_str()) {
                    note(p, Hit::Id(*id), lit.len(), &mut hit);
                }
            }
            if self.image.is_some() {
                if let Some(p) = rest.find(IMAGE_LITERAL) {
                    note(p, Hit::Image, IMAGE_LITERAL.len(), &mut hit);
                }
            }
            match hit {
                Some((p, what, len)) => {
                    if p > 0 {
                        ids.extend(self.inner.encode(&rest[..p]));
                    }
                    match what {
                        Hit::Id(id) => ids.push(id),
                        // 展开成 [start, ph₀..ph_{P-1}, end]：与 ViT 的 patch 一一对应，
                        // 视觉 embedding 会在前向时原位覆写这些占位符
                        Hit::Image => ids.extend(self.image.unwrap().expand()),
                    }
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
        let mut i = 0;
        while i < ids.len() {
            let id = ids[i];
            // 图片段折叠成单个 <|image|>：占位符逐个打印没有可读性，
            // 而折叠后重新 encode 会得到同一段 id（往返稳定）
            if let Some(im) = self.image {
                if id == im.start {
                    if !run.is_empty() {
                        out.push_str(&self.inner.decode(&run));
                        run.clear();
                    }
                    out.push_str(IMAGE_LITERAL);
                    // 跳到 end（缺失时只跳一个，避免越界死循环）
                    while i + 1 < ids.len() && ids[i + 1] != im.end {
                        i += 1;
                    }
                    if i + 1 < ids.len() {
                        i += 1; // 落在 end 上，一并吃掉
                    }
                    i += 1;
                    continue;
                }
                if let Some(code) = im.cb_index(id) {
                    if !run.is_empty() {
                        out.push_str(&self.inner.decode(&run));
                        run.clear();
                    }
                    out.push_str(&format!("<|vq:{code}|>"));
                    i += 1;
                    continue;
                }
            }
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
            i += 1;
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
        if let Some(im) = self.image {
            json["image"] = serde_json::json!({
                "start": im.start, "ph_first": im.ph_first, "count": im.count,
                "end": im.end, "cb_first": im.cb_first, "cb_count": im.cb_count,
            });
        }
        if !self.extra_literals.is_empty() {
            json["extra_literals"] = serde_json::Value::Array(
                self.extra_literals
                    .iter()
                    .map(|(t, id)| serde_json::json!({"text": t, "id": id}))
                    .collect(),
            );
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
            // 特殊 token 必须紧跟在内容词表之后，否则 id 空间有洞。
            //
            // ⚠️ 这条只对**自研分词器**成立（它们就是这样构造的）。HF 导入的词表把
            // `<unk>/<s></s>` 排在 id 0/1/2，字节 token 反而从 3 起——那时 id 空间
            // 依然稠密，只是特殊 token 不在末尾。判据用 **`pretok`**（HF 档必为 true、
            // 自研档必为 false，且它是持久化字段）来区分；注意**不能**用
            // `extra_literals` 是否为空来判断：HF 档可能压根没有额外字面量
            // （`<unk>/<s></s>` 都被认领成 BOS/EOS/PAD 了），那会让存档往返后改变行为。
            let hf_style = json["pretok"].as_bool().unwrap_or(false);
            if hf_style {
                for (name, id) in [("bos", sp.bos), ("eos", sp.eos), ("pad", sp.pad)] {
                    assert!(
                        id < inner.vocab_size(),
                        "分词器文件损坏：specials.{name}（{id}）越出词表（{}）",
                        inner.vocab_size()
                    );
                }
            } else {
                assert_eq!(
                    sp.bos,
                    inner.vocab_size(),
                    "分词器文件损坏：specials.bos（{}）不等于内容词表大小（{}）",
                    sp.bos,
                    inner.vocab_size()
                );
            }
            Some(sp)
        } else {
            None
        };
        // 图片 token 区间：可选字段，缺席 = 非多模态分词器（旧文件）
        let image = if json["image"].is_object() {
            let s = &json["image"];
            let read = |k: &str| {
                s[k].as_u64()
                    .unwrap_or_else(|| panic!("分词器 image.{k} 缺失或不是整数")) as usize
            };
            let im = ImageTokens {
                start: read("start"),
                ph_first: read("ph_first"),
                count: read("count"),
                end: read("end"),
                cb_first: read("cb_first"),
                cb_count: read("cb_count"),
            };
            // 区间必须紧接 PAD 且内部连续，否则词表大小与模型嵌入表对不上
            let pad = specials.expect("分词器带 image 字段时必须也有 specials").pad;
            assert_eq!(
                im.start,
                pad + 1,
                "分词器文件损坏：image.start（{}）应为 pad+1（{}）",
                im.start,
                pad + 1
            );
            assert_eq!(im.ph_first, im.start + 1, "image.ph_first 必须紧跟 start");
            assert_eq!(im.end, im.ph_first + im.count, "image.end 必须紧跟占位符区间");
            assert_eq!(im.cb_first, im.end + 1, "image.cb_first 必须紧跟 end");
            Some(im)
        } else {
            None
        };
        // 额外特殊 token 字面量（HF 导入的存档才有；老存档缺这个字段 ⇒ 空）
        let extra_literals: Vec<(String, usize)> = json["extra_literals"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|v| {
                        (
                            v["text"].as_str().unwrap_or("").to_string(),
                            v["id"].as_u64().unwrap_or(0) as usize,
                        )
                    })
                    .collect()
            })
            .unwrap_or_default();
        Tokenizer { inner, specials, image, extra_literals }
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

    /// `<|image|>` 展开成 `[start, ph₀.., end]`，且 id 区间紧跟特殊 token、
    /// 往返 save/load 后完全一致。
    #[test]
    fn test_image_tokens_expand_and_survive_save_load() {
        let mut tok = Tokenizer::bpe("你好世界你好世界", 300);
        let content = tok.content_vocab_size();
        let im = tok
            .attach_image_tokens(4 /* P=4 个占位符 */, 8 /* K=8 个码本 token */)
            .expect("带 specials 的新分词器应能挂图片 token");

        // id 排布：[内容][bos,eos,pad][start][ph×4][end][vq×8]
        let sp = tok.specials().unwrap();
        assert_eq!(im.start, sp.pad + 1);
        assert_eq!(im.ph_first, content + 4); // [内容][bos,eos,pad][start] → ph_first
        assert_eq!(im.end, im.ph_first + 4);
        assert_eq!(im.cb_first, im.end + 1);
        assert_eq!(tok.vocab_size(), im.cb_first + 8);
        assert_eq!(im.expand(), vec![im.start, im.ph_first, im.ph_first + 1, im.ph_first + 2, im.ph_first + 3, im.end]);

        // 编码时字面量展开；decode 丢弃、verbose 折叠回字面量
        let plain = tok.encode("图:");
        let ids = tok.encode(&format!("图:{IMAGE_LITERAL}"));
        let k = plain.len();
        assert_eq!(&ids[..k], &plain[..], "图片字面量之前的正文编码不变");
        assert_eq!(ids[k], im.start, "字面量应展开成 start 打头");
        assert_eq!(ids[k + 1], im.ph_first, "start 之后是第一个占位符");
        // expand 只含 [start, ph×P, end]（码本 token 是生成时才用的，不参与图片占位）
        assert_eq!(im.expand().len(), 2 + im.count);
        assert_eq!(ids.len(), k + im.expand().len());
        assert_eq!(tok.decode(&ids), "图:");
        assert!(tok.decode_verbose(&ids).contains(IMAGE_LITERAL));

        // 码本 id 映射与 is_special 判定
        assert_eq!(im.cb_id(0), im.cb_first);
        assert_eq!(im.cb_index(im.cb_first + 3), Some(3));
        assert_eq!(im.cb_index(im.start), None);
        assert!(tok.is_special(im.ph_first) && tok.is_special(im.cb_first + 7));
        assert!(!tok.is_special(content - 1));

        // save/load 往返
        let mut path = std::env::temp_dir();
        path.push(format!("llm_tok_img_test_{}.json", std::process::id()));
        let path = path.to_string_lossy().into_owned();
        tok.save(&path);
        let back = Tokenizer::load(&path);
        let _ = std::fs::remove_file(&path);
        assert_eq!(back.image_tokens(), Some(im));
        assert_eq!(back.vocab_size(), tok.vocab_size());
        assert_eq!(back.encode(&format!("图:{IMAGE_LITERAL}")), ids);
    }

    /// 没有 specials 的旧分词器挂不上图片 token（返回 None，词表不变）。
    #[test]
    fn test_image_tokens_require_specials() {
        let mut tok = Tokenizer::from_json(&serde_json::json!({
            "type": "char",
            "chars": ["h", "i"],
        }));
        assert!(tok.attach_image_tokens(4, 8).is_none());
        assert_eq!(tok.vocab_size(), 2);
    }
}

#[cfg(test)]
mod hf_import_tests {
    use super::*;

    /// GPT-2 字节 → Unicode 的正向表（与 `gpt2_byte_decoder` 互逆），测试里用来造 token 字符串
    fn gpt2_byte_encoder() -> Vec<char> {
        let mut direct: Vec<u32> = (b'!' as u32..=b'~' as u32).collect();
        direct.extend(0xA1u32..=0xAC);
        direct.extend(0xAEu32..=0xFF);
        let mut map: Vec<Option<char>> = vec![None; 256];
        for &c in &direct {
            map[c as usize] = Some(char::from_u32(c).unwrap());
        }
        let mut n = 0u32;
        for m in map.iter_mut() {
            if m.is_none() {
                *m = Some(char::from_u32(256 + n).unwrap());
                n += 1;
            }
        }
        map.into_iter().map(|x| x.unwrap()).collect()
    }

    /// 造一份 HF 口味的 `tokenizer.json`：`<unk>/<s>/</s>` 占 0/1/2（字节 token 从 3 起，
    /// 这正是"合并结果 ≠ 256+序号"的场景），外加三条合并规则。
    fn write_hf_tokenizer(path: &str, bad_merge: bool) {
        let enc = gpt2_byte_encoder();
        let mut vocab = serde_json::Map::new();
        vocab.insert("<unk>".into(), serde_json::json!(0));
        vocab.insert("<s>".into(), serde_json::json!(1));
        vocab.insert("</s>".into(), serde_json::json!(2));
        for (b, c) in enc.iter().enumerate() {
            vocab.insert(c.to_string(), serde_json::json!(3 + b));
        }
        vocab.insert("ab".into(), serde_json::json!(259));
        vocab.insert("abc".into(), serde_json::json!(260));
        vocab.insert("1a".into(), serde_json::json!(261));
        let a = enc[b'a' as usize];
        let b = enc[b'b' as usize];
        let c = enc[b'c' as usize];
        let one = enc[b'1' as usize];
        // bad_merge：结果 token 写成一个字节拼接对不上的串（验证结构性校验）
        let merges = if bad_merge {
            serde_json::json!([format!("{a} {c}")])
        } else {
            serde_json::json!([format!("{a} {b}"), format!("ab {c}"), format!("{one} {a}")])
        };
        let j = serde_json::json!({
            "version": "1.0",
            "added_tokens": [
                {"id": 0, "content": "<unk>", "special": true},
                {"id": 1, "content": "<s>", "special": true},
                {"id": 2, "content": "</s>", "special": true},
                {"id": 262, "content": "<|im_start|>", "special": true},
            ],
            "normalizer": null,
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "ByteLevel", "add_prefix_space": false, "use_regex": true}
            ]},
            "model": {"type": "BPE", "unk_token": "<unk>", "vocab": vocab, "merges": merges},
            "vocab_size": 263,
        });
        std::fs::write(path, serde_json::to_string(&j).unwrap()).unwrap();
    }

    #[test]
    fn test_load_hf_tokenizer_and_encode() {
        let p = std::env::temp_dir().join("rust_llm_tok_hf.json");
        let ps = p.to_string_lossy().to_string();
        write_hf_tokenizer(&ps, false);
        let tok = Tokenizer::load_hf(&ps).expect("应能导入 HF 分词器");
        // 特殊 token：<s>/</s> 被认成 BOS/EOS
        assert_eq!(tok.bos_id(), Some(1));
        assert_eq!(tok.eos_id(), Some(2));
        // 合并链：a+b=ab(259)、ab+c=abc(260)
        assert_eq!(tok.encode("abc"), vec![260]);
        assert_eq!(tok.encode("ab"), vec![259]);
        // **pre_tokenizer 生效**：GPT-2 正则把数字与字母切成两片，所以 "1a" 不合并
        assert_eq!(tok.encode("1a"), vec![3 + b'1' as usize, 3 + b'a' as usize]);
        // 额外特殊 token 按字面量整体命中（不被 BPE 拆开）
        assert_eq!(tok.encode("<|im_start|>"), vec![262]);
        // decode 往返
        assert_eq!(tok.decode(&[259]), "ab");
        assert_eq!(tok.decode(&tok.encode("abc")), "abc");
        // 存档往返：写出来再读回来，行为一致（pretok / byte_ids / merges 三元组都要保住）
        let p2 = std::env::temp_dir().join("rust_llm_tok_hf_roundtrip.json");
        let p2s = p2.to_string_lossy().to_string();
        tok.save(&p2s);
        let tok2 = Tokenizer::load(&p2s);
        assert_eq!(tok2.encode("abc"), tok.encode("abc"));
        assert_eq!(tok2.encode("<|im_start|>"), vec![262]);
        assert_eq!(tok2.eos_id(), tok.eos_id());
        let _ = std::fs::remove_file(&ps);
        let _ = std::fs::remove_file(&p2s);
    }

    /// 结构不变量必须真的在把关：结果 token 的字节拼接对不上时报错
    #[test]
    fn test_load_hf_tokenizer_rejects_broken_merge() {
        let p = std::env::temp_dir().join("rust_llm_tok_hf_bad.json");
        let ps = p.to_string_lossy().to_string();
        write_hf_tokenizer(&ps, true);
        let e = match Tokenizer::load_hf(&ps) {
            Ok(_) => panic!("拼接对不上的 merge 必须被拒绝"),
            Err(e) => e,
        };
        assert!(e.contains("不在词表里") || e.contains("拼接"), "{e}");
        let _ = std::fs::remove_file(&ps);
    }

    /// GPT-2 预切分：数字与字母分开、缩写后缀成一片、空白独立
    #[test]
    fn test_gpt2_pretokenize_branches() {
        let p = |s: &str| {
            gpt2_pretokenize(s)
                .into_iter()
                .map(|b| String::from_utf8(b).unwrap())
                .collect::<Vec<_>>()
        };
        // GPT-2 会把**一个前导空格**粘到后面的词上（` ?\p{L}+`），所以是 " def" 而不是 " " + "def"
        assert_eq!(p("abc def"), vec!["abc", " def"]);
        assert_eq!(p("1a"), vec!["1", "a"]);
        assert_eq!(p("it's"), vec!["it", "'s"]);
        assert_eq!(p("a!!!"), vec!["a", "!!!"]);
        // 多个空格：只有最后一个粘给后面的词（`\s+(?!\S)` 的负向前瞻）
        assert_eq!(p("  x"), vec![" ", " x"]);
        assert_eq!(p("a   b"), vec!["a", "  ", " b"]);
        // 到末尾的空白整段成片
        assert_eq!(p("a b "), vec!["a", " b", " "]);
    }
}
