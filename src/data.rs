//! 数据加载（第 14 课）
//!
//! 训练 GPT 的自监督方式：给模型一段文本，让它预测"下一个 token"。
//! 不需要人工标注——文本本身就是标签（这就是"自监督学习"）。
//!
//! 支持：
//! - 内置小语料（`CORPUS`，demo 用）或外部文本文件（正式训练用）
//! - 训练 / 验证划分：显式提供验证文件，或自动从训练文本末尾切 10%
//! - `sample_batch`（训练区随机采样）与 `eval_batch`（验证区随机采样）
//! - 多文件加载：路径以 `*` 通配符或目录时，自动合并所有 .txt 文件
//!
//! 两种训练目标各有一个加载器，共用一个 [`BatchSource`] 接口：
//! - [`DataLoader`]：**预训练**。连续文本切片，「预测下一个 token」，每个位置都是目标
//! - [`SftLoader`]：**监督微调**。对话样本，只回答段参与 loss（提问与角色标记被掩码屏蔽）

use crate::rng::Rng;
use crate::tokenizer::Tokenizer;

/// 内置小语料（一个英文小故事），用于演示训练
pub const CORPUS: &str = "\
Once upon a time in a small village, there lived a curious little fox named Red. \
Every morning, Red would wake up early and explore the forest. He loved to watch \
the birds fly and the rivers flow. One day, Red found a golden key under an old \
oak tree. What could it open? Red wondered. He ran to his friend, the wise old owl. \
The owl said, the key opens the door to the hidden garden, where flowers bloom all \
year round. Red was excited! He followed the path to the garden and turned the key. \
The door creaked open, revealing a world of colors and light. From that day on, Red \
visited the garden every day, and he learned that every adventure begins with a \
single step.";

/// 从路径加载文本，支持：
/// - 单文件路径
/// - 目录路径（加载目录下所有 .txt 文件）
/// - 通配符路径（如 `data/*.txt`）
pub fn load_text(path: &str) -> String {
    read_all(&resolve_files(path))
}

/// 加载以 `,` 分隔的多个路径，**每个文件一份文本**（不拼接）。每项可以是文件、目录或以
/// `*` 通配的路径。
///
/// 保留文件边界是为了 SFT 解析：说话人的角色是按"该文件里谁先说"判定的
///（见 [`parse_dialogues`]），拼成一份文本会让后面文件里的说话人角色弄反。
pub fn load_texts(paths: &str) -> Vec<String> {
    let mut files = Vec::new();
    for raw in paths.split(',') {
        let p = raw.trim();
        if p.is_empty() {
            continue;
        }
        files.extend(resolve_files(p));
    }
    files.iter().map(|f| read_one(f)).collect()
}

fn read_one(path: &str) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("无法读取数据文件 {path}: {e}"))
}

fn read_all(files: &[String]) -> String {
    files
        .iter()
        .map(|f| read_one(f))
        .collect::<Vec<_>>()
        .join("\n")
}

/// 把一个路径项展开成若干**文件路径**：文件 → 它自己；目录 → 目录下所有 `.txt`；
/// 含 `*` → 通配匹配结果。目录与通配符的结果都排序，保证不同机器上顺序一致（采样可复现）。
fn resolve_files(path: &str) -> Vec<String> {
    if path.contains('*') {
        let files = expand_glob(path);
        assert!(!files.is_empty(), "通配符 {path} 没有匹配到任何文件");
        println!("通配符 {path} 匹配到 {} 个文件", files.len());
        return files;
    }
    if std::fs::metadata(path).map(|m| m.is_dir()).unwrap_or(false) {
        let mut entries: Vec<_> = std::fs::read_dir(path)
            .unwrap_or_else(|e| panic!("无法读取目录 {path}: {e}"))
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map_or(false, |ext| ext == "txt"))
            .collect();
        entries.sort_by_key(|e| e.path());
        let files: Vec<String> = entries
            .iter()
            .map(|e| e.path().to_string_lossy().into_owned())
            .collect();
        assert!(!files.is_empty(), "目录 {path} 下没有 .txt 文件");
        println!("从目录 {path} 加载了 {} 个文本文件", files.len());
        return files;
    }
    vec![path.to_string()]
}

/// 极简通配符展开：只支持路径最后一段里的**单个** `*`（如 `data/corpus/zh_dialogue_*.txt`）。
/// 为这一个用途引入 glob 依赖不划算，手写二十行足够。
fn expand_glob(pattern: &str) -> Vec<String> {
    let (dir, name) = match pattern.rfind(|c| c == '/' || c == '\\') {
        Some(i) => (&pattern[..i], &pattern[i + 1..]),
        None => (".", pattern),
    };
    let (prefix, suffix) = match name.find('*') {
        Some(i) => (&name[..i], &name[i + 1..]),
        None => return vec![pattern.to_string()],
    };
    let mut out: Vec<String> = std::fs::read_dir(dir)
        .unwrap_or_else(|e| panic!("无法读取目录 {dir}: {e}"))
        .filter_map(|e| e.ok())
        .filter_map(|e| e.file_name().into_string().ok())
        .filter(|n| n.starts_with(prefix) && n.ends_with(suffix))
        .map(|n| format!("{dir}/{n}"))
        .collect();
    out.sort(); // 保证不同机器上顺序一致（采样可复现）
    out
}

/// 数据加载器
///
/// 输入 x 是一段长度为 block_size 的 token 序列；
/// 目标 y 是 x 右移一位（x[i] 的下一个 token 是 y[i]）。
///
/// token 序列被切为两块：
/// - `tokens[..val_start]`：训练区（`sample_batch` 在这里随机采样）
/// - `tokens[val_start..]`：验证区（`eval_batch` 在这里采样，用于评估）
pub struct DataLoader {
    tokens: Vec<usize>,
    block_size: usize,
    batch_size: usize,
    val_start: usize,
}

impl DataLoader {
    /// 整个文本都作为训练数据（demo 用，无验证集）。
    /// 需要 `tokens.len() > block_size`，否则无法切出完整序列。
    pub fn new(text: &str, tokenizer: &Tokenizer, block_size: usize, batch_size: usize) -> Self {
        let tokens = tokenizer.encode(text);
        assert!(
            tokens.len() > block_size,
            "语料太短，无法切出完整序列（{} <= {}，必须严格大于 block_size）",
            tokens.len(),
            block_size
        );
        let len = tokens.len();
        DataLoader {
            tokens,
            block_size,
            batch_size,
            val_start: len,
        }
    }

    /// 从训练/验证文本构造加载器。
    /// - `val_text = Some(..)`：使用独立的验证文本；
    /// - `val_text = None`：自动从训练文本末尾切出约 10%（至少 block+1 个 token）作验证集。
    pub fn from_texts(
        train_text: &str,
        val_text: Option<&str>,
        tokenizer: &Tokenizer,
        block_size: usize,
        batch_size: usize,
    ) -> Self {
        let mut tokens = tokenizer.encode(train_text);
        assert!(
            tokens.len() > block_size,
            "训练语料太短，无法切出完整序列（{} <= {}，必须严格大于 block_size）",
            tokens.len(),
            block_size
        );
        let val_start = match val_text {
            Some(v) => {
                let split = tokens.len();
                tokens.extend(tokenizer.encode(v));
                assert!(
                    tokens.len() - split > block_size,
                    "验证文本太短，无法切出完整序列（{} token，需要 > {}）",
                    tokens.len() - split,
                    block_size
                );
                split
            }
            None => {
                // 末尾留出至少 block+1 个 token 作验证集，再按 10% 切分
                (tokens.len() as f64 * 0.9) as usize
            }
        };
        let val_start = val_start.min(tokens.len() - (block_size + 1));
        assert!(
            tokens.len() - val_start > block_size,
            "验证语料太短，无法切出完整序列"
        );
        DataLoader {
            tokens,
            block_size,
            batch_size,
            val_start,
        }
    }

    /// 在 [lo, hi) 区间内随机选起点采样
    fn sample_region(
        &self,
        rng: &mut Rng,
        lo: usize,
        hi: usize,
        tag: &str,
    ) -> (Vec<usize>, Vec<usize>) {
        assert!(
            hi > lo + self.block_size,
            "{}区数据不足，无法采样（{} token，需要 > {}）",
            tag,
            hi - lo,
            self.block_size
        );
        let max_start = hi - lo - self.block_size - 1;
        let mut x = Vec::with_capacity(self.batch_size * self.block_size);
        let mut y = Vec::with_capacity(self.batch_size * self.block_size);
        for _ in 0..self.batch_size {
            let start = lo + rng.choice(max_start);
            for j in 0..self.block_size {
                x.push(self.tokens[start + j]);
                y.push(self.tokens[start + j + 1]);
            }
        }
        (x, y)
    }
}

/// 训练 / 评估的批次来源。
///
/// 预训练与 SFT 的采样方式不同：前者每个位置都是预测目标，后者只有回答段是。
/// 把这点差异收在一个 trait 后面，`train::train_gpt` 那段训练循环就能原样复用，
/// 不必为 SFT 再抄一遍（早停、断点续训、指标记录这些逻辑抄第二份必然会走样）。
pub trait BatchSource {
    fn block_size(&self) -> usize;
    fn batch_size(&self) -> usize;
    fn num_tokens(&self) -> usize;
    fn num_train_tokens(&self) -> usize;
    fn num_val_tokens(&self) -> usize;
    fn has_val(&self) -> bool;
    /// 返回 `(x, y, mask)`。`mask[i] == false` 表示第 i 个位置不计 loss；
    /// `None` 表示全部位置参与（预训练）。
    fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>);
    fn eval_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>);
}

impl BatchSource for DataLoader {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    fn num_train_tokens(&self) -> usize {
        self.val_start
    }

    fn num_val_tokens(&self) -> usize {
        self.tokens.len() - self.val_start
    }

    fn has_val(&self) -> bool {
        self.val_start < self.tokens.len()
    }

    /// 采样一批训练数据（训练区随机）：随机选 B 个起点，每个起点截取 block_size+1 个 token。
    fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>) {
        let (x, y) = self.sample_region(rng, 0, self.val_start, "训练");
        (x, y, None)
    }

    /// 采样一批验证数据（验证区随机，调用方用固定种子的 Rng 保证可复现）。
    fn eval_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>) {
        assert!(self.has_val(), "没有验证数据，无法采样 eval batch");
        let (x, y) = self.sample_region(rng, self.val_start, self.tokens.len(), "验证");
        (x, y, None)
    }
}

// ==================== SFT（监督微调）语料 ====================

/// SFT 对话模板的角色标记。
///
/// 用**纯文本标记**而不是新增特殊 token：字节级 BPE 本就能把任意 UTF-8 字符串编码成
/// 已有 token 的序列，所以模板不必动词表、不必给 embedding 扩容，也就不会出现
/// "加了 token 之后旧 checkpoint 的 embedding 行数对不上"这类兼容问题。
///
/// 标记只用**预训练语料里出现过的字**。这一条不是洁癖：`#` 在 4.7M 字的语料里只出现
/// 2 次，它的 embedding 基本没被训过，模板一带上 `### ` 就会把模型推进 ASCII 乱码模式
///（实测基座模型输出 `'何人？」621.-----2..3E8188`）。同理不能用 `【】`（只出现 2 次），
/// 用 `（）`（出现 129 次）。
pub const SFT_USER: &str = "用户：";
/// 回答方标记。模型在推理时永远由我们喂给它，所以它本身不需要被监督。
pub const SFT_ASSISTANT: &str = "助手：";
/// 回答结束标记。
///
/// 必须有：单轮样本如果只是"提问+回答"就结束，模型没地方学"该收尾了"，
/// 推理时它只好像预训练那样一路续写下去——这正是"只续写不回复"的成因之一。
pub const SFT_END: &str = "（结束）";

/// 一段对话：若干「(提问, 回答)」轮次
pub type SftTurns = Vec<(String, String)>;

/// 从行首解析角色标记 `<标签>：` / `<标签>:`，返回 `(标签, 标记之后的内容)`。
///
/// 标签限 1~8 个字符、只由字母数字（含汉字）组成——这样既能认 `A` / `用户` / `面试官` / `陈教授`，
/// 又不会把「2024年数据：……」这类正文当成角色行。
fn split_label(line: &str) -> Option<(&str, &str)> {
    let (idx, colon) = line.char_indices().find(|(_, c)| *c == '：' || *c == ':')?;
    let label = &line[..idx];
    if label.is_empty() || label.chars().count() > 8 {
        return None;
    }
    if !label.chars().all(|c| c.is_alphanumeric()) {
        return None;
    }
    Some((label, &line[idx + colon.len_utf8()..]))
}

/// 说话人标签 → 「是不是提问方」。
///
/// 已知标签（`用户` / `助手` / `A` / `B`）有固定含义，直接查表。
/// 人名标签（`陈教授` / `面试官` / `王芳`）没有固定含义，按**该文件里首次出现的顺序**判定：
/// 第一个出现的算提问方，其余算回答方——对话语料里总是提问方先说。
///
/// 角色是**文件级**属性（换一份语料，先说的人可能换成另一个人），
/// 所以每解析一个文件都要重建一次；跨文件共用会把后面文件里的角色弄反。
struct Speakers {
    names: Vec<String>,
}

impl Speakers {
    fn new() -> Self {
        Speakers { names: Vec::new() }
    }

    fn role(&mut self, label: &str) -> bool {
        match label {
            "用户" | "A" => return true,
            "助手" | "B" => return false,
            _ => {}
        }
        match self.names.iter().position(|n| n == label) {
            Some(0) => true,
            Some(_) => false,
            None => {
                self.names.push(label.to_string());
                self.names.len() == 1
            }
        }
    }
}

/// 从**一个文件**的文本解析出若干段对话。
///
/// 按行解析：能识别角色标记（`用户：` / `A:` / `陈教授：`）的行开启新一轮；
/// 其余非空行续到当前角色上（多行回答不会被腰斩）；空行在「已进入回答方」时
/// 作为一段对话的结束。
///
/// 人名标签（`陈教授：`）只在**一眼能看出这是对话语料**时才接受：非空行里 ≥90% 是角色行、
/// 且全程只有两个说话人。小说正文里也有 `秦琼道：……` 这种行，但旁白、描写、心理
/// 会把它稀释到 90% 以下，于是把 `sft_file` 指向混杂目录仍然不会把小说正文当成对话喂进来。
pub fn parse_dialogues(text: &str) -> Vec<SftTurns> {
    /// 把累积的 `buf` 落到当前角色上
    fn flush(cur: &mut SftTurns, role: Option<bool>, buf: &mut String) {
        let line = buf.trim();
        if line.is_empty() {
            buf.clear();
            return;
        }
        match role {
            Some(true) => cur.push((line.to_string(), String::new())),
            Some(false) => match cur.last_mut() {
                // 提问与回答在数据里可能跨多行，续行接在同一个字段里
                Some(last) if !last.1.is_empty() => {
                    last.1.push('\n');
                    last.1.push_str(line);
                }
                Some(last) => last.1 = line.to_string(),
                // 数据以回答开头（没有对应提问）：也收下，交给下面的完整性检查处理
                None => cur.push((String::new(), line.to_string())),
            },
            None => {}
        }
        buf.clear();
    }

    let non_empty: Vec<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty())
        .collect();
    let mut labels: Vec<&str> = non_empty
        .iter()
        .filter_map(|l| split_label(l).map(|(label, _)| label))
        .collect();
    labels.sort_unstable();
    labels.dedup();
    let n_role = non_empty.iter().filter(|l| split_label(l).is_some()).count();
    let allow_names = labels.len() == 2 && n_role * 10 >= non_empty.len() * 9;

    let mut out: Vec<SftTurns> = Vec::new();
    let mut cur: SftTurns = Vec::new();
    let mut role: Option<bool> = None;
    let mut buf = String::new();
    let mut speakers = Speakers::new();

    for raw in text.lines() {
        let line = raw.trim();
        if let Some((label, rest)) = split_label(line) {
            // 固定标签永远认；人名标签要 `allow_names` 放行
            if matches!(label, "用户" | "助手" | "A" | "B") || allow_names {
                flush(&mut cur, role, &mut buf);
                role = Some(speakers.role(label));
                // 标记之后的内容才是这一轮的话；不接上就等于把整句话丢了
                buf.push_str(rest.trim());
                continue;
            }
        }
        if line.is_empty() {
            flush(&mut cur, role, &mut buf);
            // 只有"答案已经出现"时空行才算一段对话结束，否则提问与回答之间的空行会把它们拆散
            if role == Some(false) {
                let complete: SftTurns = std::mem::take(&mut cur)
                    .into_iter()
                    .filter(|(u, a)| !u.is_empty() && !a.is_empty())
                    .collect();
                if !complete.is_empty() {
                    out.push(complete);
                }
                role = None;
            }
            continue;
        }
        if !buf.is_empty() {
            buf.push('\n');
        }
        buf.push_str(line);
    }
    flush(&mut cur, role, &mut buf);
    let complete: SftTurns = cur
        .into_iter()
        .filter(|(u, a)| !u.is_empty() && !a.is_empty())
        .collect();
    if !complete.is_empty() {
        out.push(complete);
    }
    out
}

/// 从**多个文件**的文本解析对话。逐文件解析并合并结果——
/// 说话人角色是文件级属性，不能跨文件共用一张表（见 [`Speakers`]）。
pub fn parse_dialogue_files(texts: &[String]) -> Vec<SftTurns> {
    texts.iter().flat_map(|t| parse_dialogues(t)).collect()
}

/// 把若干段对话渲染成 token 流，并标出每个位置上「预测该 token」是不是监督目标。
///
/// `sup[p] == true` 表示「以第 p 个 token 为目标」要算 loss。监督区只有三块：
/// 回答正文、回答后的换行、以及结尾的 [`SFT_END`]；提问与角色标记全部屏蔽——
/// 它们只是给模型的上下文，不是它该学会生成的东西。
///
/// 逐段编码再拼接，而不是先拼成一个大字符串整体编码：BPE 的合并会跨边界进行，
/// 整串编码有可能把"标记末尾"和"正文开头"并成一个 token，那样监督区的起点
/// 就不落在 token 边界上了，掩码会错位一个 token。
fn build_sft_stream(tokenizer: &Tokenizer, convs: &[SftTurns]) -> (Vec<usize>, Vec<bool>) {
    let mut tokens: Vec<usize> = Vec::new();
    let mut sup: Vec<bool> = Vec::new();
    let mut push = |text: &str, supervised: bool| {
        let ids = tokenizer.encode(text);
        sup.extend(std::iter::repeat(supervised).take(ids.len()));
        tokens.extend(ids);
    };
    for conv in convs {
        for (user, assistant) in conv {
            push(&format!("{SFT_USER}\n{user}\n"), false);
            push(&format!("{SFT_ASSISTANT}\n"), false);
            push(&format!("{assistant}\n"), true);
        }
        push(&format!("{SFT_END}\n"), true);
    }
    (tokens, sup)
}

/// 取一个训练窗口的 loss 掩码。
///
/// 关键在**右移一位**：窗口里 `y[i] = tokens[start+1+i]`，所以第 i 个位置的掩码
/// 要看目标 token 所在的位置 `start+1+i`，而不是输入位置 `start+i`。
/// 少移这一位，整个批次的监督信号会整体错开一个 token——loss 照降，但学的是
/// "用前一个字的答案预测后一个字"，形式上完全说得通，极难从结果看出来。
fn window_mask(sup: &[bool], start: usize, block_size: usize) -> Vec<bool> {
    (0..block_size).map(|j| sup[start + j + 1]).collect()
}

/// SFT 数据加载器。
///
/// 与 [`DataLoader`] 的区别只有一点：每批多带一个 loss 掩码。样本是**打包**的——
/// 所有对话首尾相接成一条 token 流，训练时按 `block_size` 随机切窗口，
/// 因此一个窗口里可能横跨几段对话、也可能从提问中途开始，掩码会照实
/// 把这些位置标成"不计 loss"。
pub struct SftLoader {
    tokens: Vec<usize>,
    /// 与 `tokens` 等长：第 p 个位置是否为监督目标
    sup: Vec<bool>,
    /// 解析出的对话段数（仅用于日志）
    num_conversations: usize,
    block_size: usize,
    batch_size: usize,
    val_start: usize,
}

impl SftLoader {
    /// 从对话语料构造加载器，`texts` 是**逐文件**的文本（见 [`load_texts`]）。末尾自动切约 10% 作验证区。
    pub fn from_texts(
        texts: &[String],
        tokenizer: &Tokenizer,
        block_size: usize,
        batch_size: usize,
    ) -> Self {
        let convs = parse_dialogue_files(texts);
        assert!(
            !convs.is_empty(),
            "SFT 语料里没解析出任何对话——检查文件是否带角色标记（{} 或 A: / B:）",
            SFT_USER
        );
        let (tokens, sup) = build_sft_stream(tokenizer, &convs);
        assert!(
            tokens.len() > block_size,
            "SFT 语料太短，无法切出完整序列（{} <= {}，必须严格大于 block_size）",
            tokens.len(),
            block_size
        );
        // 末尾留出至少 block+1 个 token 作验证区，再按 10% 切分
        let val_start = ((tokens.len() as f64 * 0.9) as usize).min(tokens.len() - (block_size + 1));
        assert!(
            tokens.len() - val_start > block_size,
            "SFT 验证语料太短，无法切出完整序列"
        );
        let n_val_sup = sup[val_start..].iter().filter(|b| **b).count();
        assert!(
            n_val_sup > 0,
            "SFT 验证区里没有任何监督位置，val_loss 会恒为 0；请增加对话语料或调小验证比例"
        );
        SftLoader {
            tokens,
            sup,
            num_conversations: convs.len(),
            block_size,
            batch_size,
            val_start,
        }
    }

    pub fn num_conversations(&self) -> usize {
        self.num_conversations
    }

    /// 监督位置占总 token 的比例。太低（比如 <5%）说明打包进来的语料里
    /// 提问/标记远多于回答，训练信号会很稀。
    pub fn supervised_ratio(&self) -> f64 {
        if self.sup.is_empty() {
            return 0.0;
        }
        self.sup.iter().filter(|b| **b).count() as f64 / self.sup.len() as f64
    }

    /// 在 [lo, hi) 区间内随机选起点采样；掩码取 `sup[start+1 .. start+1+T]`
    ///（因为 `y[i] = tokens[start+1+i]`）。
    fn sample_region(
        &self,
        rng: &mut Rng,
        lo: usize,
        hi: usize,
        tag: &str,
    ) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>) {
        assert!(
            hi > lo + self.block_size,
            "{}区数据不足，无法采样（{} token，需要 > {}）",
            tag,
            hi - lo,
            self.block_size
        );
        let max_start = hi - lo - self.block_size - 1;
        let n = self.batch_size * self.block_size;
        let mut x = Vec::with_capacity(n);
        let mut y = Vec::with_capacity(n);
        let mut mask = Vec::with_capacity(n);
        for _ in 0..self.batch_size {
            let start = lo + rng.choice(max_start);
            for j in 0..self.block_size {
                x.push(self.tokens[start + j]);
                y.push(self.tokens[start + j + 1]);
            }
            mask.extend(window_mask(&self.sup, start, self.block_size));
        }
        (x, y, Some(mask))
    }
}

impl BatchSource for SftLoader {
    fn block_size(&self) -> usize {
        self.block_size
    }

    fn batch_size(&self) -> usize {
        self.batch_size
    }

    fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    fn num_train_tokens(&self) -> usize {
        self.val_start
    }

    fn num_val_tokens(&self) -> usize {
        self.tokens.len() - self.val_start
    }

    fn has_val(&self) -> bool {
        self.val_start < self.tokens.len()
    }

    fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>) {
        self.sample_region(rng, 0, self.val_start, "训练")
    }

    fn eval_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>) {
        assert!(self.has_val(), "没有验证数据，无法采样 eval batch");
        self.sample_region(rng, self.val_start, self.tokens.len(), "验证")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_accepts_both_prefix_styles() {
        let text = "用户：你好\n助手：你也好\n\nA: 吃了吗\nB: 吃了\n";
        let convs = parse_dialogues(text);
        assert_eq!(convs.len(), 2);
        assert_eq!(convs[0], vec![("你好".into(), "你也好".into())]);
        assert_eq!(convs[1], vec![("吃了吗".into(), "吃了".into())]);
    }

    #[test]
    fn parse_keeps_multiline_answer_and_drops_incomplete() {
        // 多行回答不能被腰斩；只有提问、没有回答的残段要丢掉
        let text = "用户：讲讲\n助手：第一行\n第二行\n\n用户：没回答的\n";
        let convs = parse_dialogues(text);
        assert_eq!(convs.len(), 1);
        assert_eq!(convs[0][0].1, "第一行\n第二行");
    }

    #[test]
    fn parse_skips_text_without_role_markers() {
        // 小说正文没有角色标记，必须整体跳过，不能被当成对话
        let text = "话说天下大势，分久必合，合久必分。\n周末七国分争，并入于秦。";
        assert!(parse_dialogues(text).is_empty());
    }

    #[test]
    fn parse_accepts_named_speakers_in_dialogue_corpus() {
        // 人名说话的对话语料（仓库里的 zh_dialogue_ai_research/interview/technology 就是这个形式）
        let text = "面试官：请介绍一下你自己。\n应聘者：我是一名后端工程师。\n\n\
                    面试官：你最擅长什么语言？\n应聘者：Python 和 Rust。\n";
        let convs = parse_dialogues(text);
        assert_eq!(convs.len(), 2);
        assert_eq!(
            convs[0],
            vec![("请介绍一下你自己。".into(), "我是一名后端工程师。".into())]
        );
    }

    #[test]
    fn parse_rejects_named_speakers_in_prose() {
        // 小说里也有「秦琼道：……」这种行，但角色行被旁白稀释到 90% 以下，
        // 所以认了人名也不会把正文当成对话喂进来
        let text = "秦琼道：来者何人？\n叔宝默默不语，只是把缰绳握得更紧了些。\n\
                    那汉子又喝道：问你话呢！\n秦琼心下暗想，此人武艺不弱。\n\
                    天色渐晚，两人在林中僵持了半个时辰。\n秦琼道：罢了，你走吧。\n";
        assert!(parse_dialogues(text).is_empty());
    }

    #[test]
    fn parse_resets_speakers_per_file() {
        // 说话人角色是文件级属性：第二份语料里先说的人换成了「李教授」，
        // 不能因为第一份里「张伟」先说就把后面这份的角色弄反
        let a = "张伟：你好。\n王芳：你好呀。\n".to_string();
        let b = "李教授：最近在忙什么？\n陈教授：在做 AI 伦理方面的研究。\n".to_string();
        let convs = parse_dialogue_files(&[a, b]);
        assert_eq!(convs.len(), 2);
        assert_eq!(convs[1][0].0, "最近在忙什么？");
        assert_eq!(convs[1][0].1, "在做 AI 伦理方面的研究。");
    }

    #[test]
    fn build_stream_masks_question_and_marks_answer() {
        let tok = Tokenizer::char("用户：你好\n助手：你也好\n（结束）\n");
        let convs = vec![vec![("你好".to_string(), "你也好".to_string())]];
        let (tokens, sup) = build_sft_stream(&tok, &convs);
        assert_eq!(tokens.len(), sup.len());
        assert!(sup.iter().any(|b| !*b), "提问与角色标记必须被屏蔽");

        // 监督区必须紧接着「助手：」标记开始（按 token 下标切，不能按字节切：
        // 一个汉字占多个字节，用 token 下标当字节下标会切在字符中间）
        let first = sup.iter().position(|b| *b).unwrap();
        let prefix = tok.decode(&tokens[..first]);
        assert_eq!(
            prefix,
            format!("{SFT_USER}\n你好\n{SFT_ASSISTANT}\n"),
            "监督区之前只应是提问与角色标记"
        );

        // 监督区内容 = 回答正文 + 换行 + 结束标记
        let last = sup.iter().rposition(|b| *b).unwrap();
        let supervised = tok.decode(&tokens[first..=last]);
        assert_eq!(supervised, format!("你也好\n{SFT_END}\n"));
    }

    #[test]
    fn window_mask_is_shifted_by_one() {
        // 掩码必须取「目标位置」的监督标记：这一位错了，学出来的东西形式对但内容是错的
        let sup = vec![false, true, false, true, true, false];
        assert_eq!(window_mask(&sup, 1, 3), vec![false, true, true]);
        assert_eq!(window_mask(&sup, 0, 5), sup.iter().skip(1).copied().collect::<Vec<_>>());
    }

    #[test]
    fn sft_loader_batch_shapes_match() {
        let tok = Tokenizer::char("用户：你好\n助手：你也好\n（结束）\n");
        let convs = vec![
            vec![("你好".to_string(), "你也好".to_string())],
            vec![("你好".to_string(), "你也好".to_string())],
        ];
        // 直接构造（绕过 from_texts 的比例切分），只验证批次形状与掩码确实带上了。
        // 窗口起点是随机选的，某个窗口完全落在提问段上很正常，所以多抽几次看有没有
        // 窗口带上监督位置——掩码要是没透传出去，抽多少次都会全是 false
        let (tokens, sup) = build_sft_stream(&tok, &convs);
        let n = tokens.len();
        let loader = SftLoader {
            tokens,
            sup,
            num_conversations: convs.len(),
            block_size: 4,
            batch_size: 2,
            val_start: n / 2,
        };
        let mut rng = Rng::new(42);
        let mut saw_sup = false;
        for _ in 0..20 {
            let (x, y, mask) = loader.sample_batch(&mut rng);
            let mask = mask.expect("SFT 必须返回掩码");
            assert_eq!(x.len(), 8);
            assert_eq!(y.len(), 8);
            assert_eq!(mask.len(), 8);
            saw_sup |= mask.iter().any(|b| *b);
        }
        assert!(saw_sup, "掩码里至少要有一个监督位置");
        assert!(loader.supervised_ratio() > 0.0);
    }
}
