//! RAG 检索增强生成（第 37 课）：分块、向量化、向量检索、提示组装
//!
//! 模型的知识有两个硬边界：训练数据的**时间截止点**，以及**它压根没见过的私有文档**。
//! 微调也救不了后者——每来一份新文档就重训一遍不现实。RAG 换了个思路：
//! 不要求模型记住，而是**先把相关片段检索出来塞进上下文**，让"知识"变成推理时的输入。
//!
//! 本模块把这条流水线拆成四段，每段都是可单独使用、可单独测的：
//!
//! 1. **分块**（[`chunk_text`]）：长文档按字符窗口切块并保留重叠，避免关键句正好被
//!    边界劈开；切分按**字符**而不是字节，中文不会被切成半个字。
//! 2. **向量化**（[`Embedder`]）：三种真正可用的检索器——
//!    [`TfIdf`]（真实 IDF 统计）、[`HashingEmbedder`]（无词表、固定维度）、
//!    [`ModelEmbedder`]（用模型隐状态做均值池化的稠密向量）。
//! 3. **检索**（[`Retriever`]）：余弦相似度暴力 top-k，外加 [`Retriever::search_mmr`]
//!    的最大边际相关去冗余重排。
//! 4. **提示组装**（[`build_rag_prompt`]）：带来源标注与预算裁剪，并强制"只依据资料回答"。

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::model::Transformer;
use crate::tokenizer::Tokenizer;

// ==================== 通用向量工具 ====================

/// L2 范数
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

/// 就地 L2 归一化。零向量原样返回（不产生 NaN）。
///
/// 归一化的意义：余弦相似度只关心方向，先把长度除掉，检索时就退化成一次点积，
/// 而且不同长度的文本块之间不再有"长块天然占优"的偏置。
pub fn l2_normalize(v: &mut [f32]) {
    let n = l2_norm(v);
    if n > 0.0 {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

/// 余弦相似度。任一向量为零时返回 `0.0`——那是"没有共同词项"，不是 NaN。
///
/// 直接算 `dot / (|a||b|)` 的话，查询里全是语料没见过的词时向量全零，
/// `0/0` 会变成 NaN，而 NaN 参与排序会让整个结果顺序变成未定义的乱序。
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "余弦相似度要求两向量同维：{} vs {}", a.len(), b.len());
    let na = l2_norm(a);
    let nb = l2_norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    dot / (na * nb)
}

/// 按**字符**截断（不是字节），中文不会被截出半个字。
pub fn truncate_chars(s: &str, max_chars: usize) -> String {
    s.chars().take(max_chars).collect()
}

// ==================== 检索用的词项切分 ====================

/// 把文本切成检索用的"词项"：逐字符取 unigram，再取相邻字符的 bigram。
///
/// 为什么不用空格切词：中文没有空格，按空格切会把整句当成一个"词"，检索直接失效。
/// 字 unigram 保证召回（任何出现过的字都能匹配上），字 bigram 补一点局部语序信息
/// ——"上海"和"海上"的 unigram 集合完全相同，bigram 不同。这是没有分词器时最可靠的折中。
///
/// 空白字符被整体丢掉，因此 bigram 可能跨过词间空格（"hello world" 会产生 "ow" 这类
/// 组合）。这只会**增加**一些噪声特征，不会丢信息，对检索排序的影响可以忽略。
pub fn terms(text: &str) -> Vec<String> {
    let chars: Vec<char> = text
        .chars()
        .filter(|c| !c.is_whitespace())
        .map(|c| c.to_ascii_lowercase())
        .collect();
    let mut out = Vec::with_capacity(chars.len() * 2);
    for i in 0..chars.len() {
        out.push(chars[i].to_string());
        if i + 1 < chars.len() {
            out.push(format!("{}{}", chars[i], chars[i + 1]));
        }
    }
    out
}

// ==================== 1. 分块 ====================

/// 分块参数
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ChunkOpts {
    /// 每块的**字符**数上限
    pub size: usize,
    /// 相邻块重叠的字符数（避免关键句正好落在边界上被劈成两半）
    pub overlap: usize,
    /// 是否把窗口边界微调到最近的句子/段落结束符
    pub align: bool,
}

impl ChunkOpts {
    /// 固定窗口分块（最通用）
    pub fn new(size: usize, overlap: usize) -> Self {
        assert!(size > 0, "块大小必须为正");
        assert!(overlap < size, "重叠 {overlap} 必须小于块大小 {size}，否则窗口不前进");
        ChunkOpts { size, overlap, align: false }
    }

    /// 句子边界对齐分块（适合有标点的正文，块读起来更完整）
    pub fn aligned(size: usize, overlap: usize) -> Self {
        ChunkOpts { align: true, ..Self::new(size, overlap) }
    }
}

/// 一个文本块。`start`/`end` 是**字符**下标（不是字节），便于人工核对与二次切分。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    pub text: String,
    pub start: usize,
    pub end: usize,
    /// 来源标记（文件名或自定的标签），用于在提示里标注出处
    pub source: String,
}

impl Chunk {
    /// 字符数（`text.chars().count()` 的等价物，但不必重算）
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end == self.start
    }
}

/// 句子/段落结束符
fn is_break(c: char) -> bool {
    matches!(c, '。' | '！' | '？' | '；' | '…' | '.' | '!' | '?' | ';' | '\n')
}

/// 在窗口边界 `end` 附近找最近的断点（返回"断点之后"的下标）。
///
/// 先试向后（块更长、更完整），再试向前；`d` 从小到大枚举即"距离最近优先"。
/// - `lo` 是向前回退的下限，保证新块的起点仍在推进；
/// - `lookahead` 限制向后最多多看多少字符，防止为了一个句号把块拉得过长。
fn nearest_break(chars: &[char], end: usize, lo: usize, lookahead: usize) -> Option<usize> {
    let hi = (end + lookahead).min(chars.len());
    for d in 1..=lookahead {
        let fwd = end + d;
        if fwd <= hi && is_break(chars[fwd - 1]) {
            return Some(fwd);
        }
        if d <= end {
            let bwd = end - d;
            if bwd >= lo && is_break(chars[bwd - 1]) {
                return Some(bwd);
            }
        }
    }
    None
}

/// 把文本切成带重叠的块。
///
/// **按字符切**而不是按字节：`&s[0..n]` 这种字节切片遇到中文会 panic（切在多字节字符中间），
/// 所以先转成 `Vec<char>`，整个流程都在字符域里做，中文、emoji 都安全。
///
/// 不变量（单测逐条钉住）：
/// - 每块不超过 `size` 个字符（对齐时最多多出 `size/4`）；
/// - 相邻块重叠**恰为** `overlap` 个字符；
/// - 把每块去掉与后块重叠的部分后拼起来，**恰好等于原文**（不丢字、不重复）。
pub fn chunk_text(text: &str, opts: &ChunkOpts) -> Vec<Chunk> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    if n == 0 {
        return Vec::new();
    }
    let size = opts.size.min(n.max(1));
    let overlap = opts.overlap.min(size - 1);
    let mut out = Vec::new();
    let mut start = 0usize;
    loop {
        let mut end = (start + size).min(n);
        if opts.align && end < n {
            // 向前回退的下限：既保证块不会缩得太短，也保证 `end - overlap > start`
            let lo = start + (size / 2).max(overlap + 1);
            if let Some(p) = nearest_break(&chars, end, lo.min(end), (size / 4).max(1)) {
                end = p;
            }
        }
        out.push(Chunk {
            text: chars[start..end].iter().collect(),
            start,
            end,
            source: String::new(),
        });
        if end >= n {
            break;
        }
        start = end - overlap;
    }
    out
}

// ==================== 2. 向量化 ====================

/// 向量化器：把文本变成定长向量，供余弦相似度检索。
pub trait Embedder {
    /// 输出维度
    fn dim(&self) -> usize;
    /// 文本 → 向量
    fn embed(&self, text: &str) -> Vec<f32>;
}

/// TF-IDF 向量化器：词表来自语料，权重 = 次线性词频 × 真实 IDF。
///
/// 频次统计是**语料级**的，所以必须先 `fit` 语料再 `embed`——这也是"检索器"与
/// "随便算个向量"的分界：IDF 让"人人都有"的字（如"的"）权重低、"只此一处"的字权重高。
#[derive(Clone, Debug)]
pub struct TfIdf {
    index: HashMap<String, usize>,
    vocab: Vec<String>,
    idf: Vec<f32>,
}

impl TfIdf {
    /// 在语料上统计文档频率（`df`）并定下词表。
    ///
    /// `df` 用的是"**出现过该词项的文档数**"，同一文档里出现 10 次也只算 1——
    /// 这正是 IDF 想要的信息（区分性），而不是频次。
    pub fn fit(corpus: &[String]) -> Self {
        let n_docs = corpus.len();
        let mut df: HashMap<String, usize> = HashMap::new();
        for doc in corpus {
            let mut uniq = terms(doc);
            uniq.sort();
            uniq.dedup();
            for t in uniq {
                *df.entry(t).or_insert(0) += 1;
            }
        }
        // 词表必须排序：HashMap 的迭代顺序每进程随机，不排序会让"同一语料两次 fit"
        // 得到不同的下标，索引与查询之间就对不上了。
        let mut vocab: Vec<String> = df.keys().cloned().collect();
        vocab.sort();
        let idf = vocab
            .iter()
            .map(|t| {
                let d = df[t] as f32;
                // 平滑口径：语料为空时也不会出现 ln(0)
                ((1.0 + n_docs as f32) / (1.0 + d)).ln() + 1.0
            })
            .collect();
        let index = vocab.iter().enumerate().map(|(i, t)| (t.clone(), i)).collect();
        TfIdf { index, vocab, idf }
    }

    /// 词表大小 = 向量维度
    pub fn vocab(&self) -> &[String] {
        &self.vocab
    }

    /// 某个词项的 IDF（未登录词返回 0）
    pub fn idf_of(&self, term: &str) -> f32 {
        self.index.get(term).map_or(0.0, |&i| self.idf[i])
    }
}

impl Embedder for TfIdf {
    fn dim(&self) -> usize {
        self.vocab.len()
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.vocab.len()];
        let mut tf: HashMap<usize, f32> = HashMap::new();
        for t in terms(text) {
            if let Some(&i) = self.index.get(&t) {
                *tf.entry(i).or_insert(0.0) += 1.0;
            }
        }
        for (i, c) in tf {
            // 次线性词频 1 + ln(c)：一个词出现 100 次不该比出现 1 次重要 100 倍
            v[i] = (1.0 + c.ln()) * self.idf[i];
        }
        l2_normalize(&mut v);
        v
    }
}

/// 特征哈希向量化器：用哈希把词项直接映射到固定维度，**不需要词表**。
///
/// 优点是没有"未登录词"问题（不像 TF-IDF，查询里的新词直接权重为 0），
/// 内存固定、可以流式增量更新；代价是哈希冲突带来的噪声。
///
/// 冲突用**符号哈希**缓解：同一个桶让正负特征互相抵消，而不是一味相加——
/// 否则任何词项都会把该桶的值推高，冲突就变成了系统性的正偏。
#[derive(Clone, Copy, Debug)]
pub struct HashingEmbedder {
    dim: usize,
}

impl HashingEmbedder {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "哈希维度必须为正");
        HashingEmbedder { dim }
    }
}

/// FNV-1a 64 位哈希：实现只有几行、无依赖、分布够均匀，适合当特征哈希。
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

impl Embedder for HashingEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; self.dim];
        for t in terms(text) {
            let h = fnv1a(t.as_bytes());
            let idx = (h % self.dim as u64) as usize;
            let sign = if h >> 63 == 0 { 1.0 } else { -1.0 };
            v[idx] += sign;
        }
        l2_normalize(&mut v);
        v
    }
}

/// 模型稠密向量：把文本喂进 Transformer，取**最后一层隐状态按位置求均值**再归一化。
///
/// 这是三种检索器里唯一"懂语义"的：TF-IDF 与哈希只看字面重合，
/// 问"怎么退款"可能检索不到写着"申请退货流程"的段落；稠密向量可以。
/// 代价是慢（每个块都要过一次前向）且依赖模型质量——随机初始化的模型给出的
/// 向量基本是噪声，必须用**训练过**的 checkpoint。
///
/// 用 `Arc<Transformer>` 而不是借用：`Transformer` 不可克隆、也不实现 `Clone`，而检索器需要
/// 和别的组件一起长期持有模型。`Arc` 只加一次引用计数，不拷贝权重。
pub struct ModelEmbedder {
    model: Arc<Transformer>,
    tokenizer: Tokenizer,
}

impl ModelEmbedder {
    pub fn new(model: Arc<Transformer>, tokenizer: Tokenizer) -> Self {
        ModelEmbedder { model, tokenizer }
    }
}

impl Embedder for ModelEmbedder {
    fn dim(&self) -> usize {
        self.model.cfg.n_embd
    }

    fn embed(&self, text: &str) -> Vec<f32> {
        let mut ids: Vec<usize> = self.tokenizer.bos_id().into_iter().collect();
        ids.extend(self.tokenizer.encode(text));
        assert!(
            !ids.is_empty(),
            "文本为空且分词器没有 BOS，无法生成嵌入（换成带 BOS 的分词器即可）"
        );
        // 超过上下文窗口就截断：这里要的是"这段文本大概在说什么"，
        // 前面的内容足以代表，没必要（也不能）全塞进去。
        ids.truncate(self.model.cfg.block_size.max(1));
        let t = ids.len();
        let d = self.model.cfg.n_embd;
        let hidden = crate::tensor::no_grad(|| self.model.forward_hidden(&ids, 1, t, false));
        let data = hidden.data_ref();
        let mut v = vec![0.0f32; d];
        for i in 0..t {
            for (j, x) in v.iter_mut().enumerate() {
                *x += data[i * d + j];
            }
        }
        for x in v.iter_mut() {
            *x /= t as f32;
        }
        l2_normalize(&mut v);
        v
    }
}

// ==================== 3. 检索 ====================

/// 一条检索结果
#[derive(Clone, Debug)]
pub struct ScoredChunk {
    /// 在索引中的块下标（用于定位与去重）
    pub index: usize,
    pub chunk: Chunk,
    /// 与查询的余弦相似度。MMR 重排时这里仍存**相关性**（而不是 MMR 值），
    /// 因为下游（提示裁剪、展示）需要的是"有多相关"，不是"有多被选中"。
    pub score: f32,
}

/// 向量检索器：块 + 预计算好的块向量。
pub struct Retriever {
    embedder: Box<dyn Embedder>,
    chunks: Vec<Chunk>,
    vectors: Vec<Vec<f32>>,
}

impl Retriever {
    /// 直接用现成的块建索引（逐块算向量，只算一次，查询时不再碰语料）
    pub fn build(embedder: Box<dyn Embedder>, chunks: Vec<Chunk>) -> Self {
        let vectors = chunks.iter().map(|c| embedder.embed(&c.text)).collect();
        Retriever { embedder, chunks, vectors }
    }

    /// 从一段文本建索引
    pub fn from_text(embedder: Box<dyn Embedder>, text: &str, opts: &ChunkOpts) -> Self {
        Self::build(embedder, chunk_text(text, opts))
    }

    /// 从目录下的所有 `.txt` 建索引，块自动带上文件名作为来源。
    ///
    /// 文件按路径排序后处理：`read_dir` 的返回顺序依文件系统而定，
    /// 不排序的话同一目录两次建索引会得到不同的块编号，检索结果无法复现。
    pub fn from_corpus_dir(
        embedder: Box<dyn Embedder>,
        dir: &Path,
        opts: &ChunkOpts,
    ) -> std::io::Result<Self> {
        let mut files: Vec<PathBuf> = std::fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|e| e.to_str()).map_or(false, |e| e == "txt"))
            .collect();
        files.sort();
        let mut chunks = Vec::new();
        for p in files {
            let text = std::fs::read_to_string(&p)?;
            let source = p
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();
            for mut c in chunk_text(&text, opts) {
                c.source = source.clone();
                chunks.push(c);
            }
        }
        Ok(Self::build(embedder, chunks))
    }

    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    pub fn dim(&self) -> usize {
        self.embedder.dim()
    }

    pub fn chunks(&self) -> &[Chunk] {
        &self.chunks
    }

    /// 查询向量（供调试与自查）
    pub fn embed_query(&self, query: &str) -> Vec<f32> {
        self.embedder.embed(query)
    }

    /// 暴力 top-k：余弦相似度降序。同分时按块下标升序，保证结果可复现。
    ///
    /// 没做 ANN（HNSW/IVF，见第 37 课文档）：块数上万之前，一次全量点积是
    /// 几毫秒的事，而 ANN 要引入图结构、参数与召回率损失。等真的卡在延迟上再换。
    pub fn search(&self, query: &str, k: usize) -> Vec<ScoredChunk> {
        let mut scored = self.relevance(query);
        scored.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        scored.truncate(k);
        self.collect(scored)
    }

    /// MMR（最大边际相关）重排：`λ·相关度 − (1−λ)·与已选结果的最大相似度`。
    ///
    /// 为什么需要它：语料里常有整段重复或高度相似的块，纯按相关度取 top-k 会
    /// 把 k 个名额全给同一条信息的多个副本，真正互补的信息反而挤不进来。
    /// MMR 每选一个就抑制"与已选相似"的候选，用 `λ` 调"要相关"还是"要多样"：
    /// `λ = 1` 退化成普通 top-k，`λ` 越小越看重多样性。
    ///
    /// `pool` 是重排池大小：先按相关度取 `pool` 个候选再重排，避免为了多样性
    /// 去考察一大堆毫不相关的块。
    pub fn search_mmr(&self, query: &str, k: usize, lambda: f32, pool: usize) -> Vec<ScoredChunk> {
        assert!(
            (0.0..=1.0).contains(&lambda),
            "λ 必须落在 [0, 1]，实际 {lambda}"
        );
        let mut cand = self.relevance(query);
        cand.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        cand.truncate(pool.max(k));

        let mut picked: Vec<usize> = Vec::new();
        let mut used = vec![false; cand.len()];
        while picked.len() < k.min(cand.len()) {
            let mut best: Option<(usize, f32)> = None;
            for (pos, &(ci, rel)) in cand.iter().enumerate() {
                if used[pos] {
                    continue;
                }
                // 与"已选集合"的最大相似度就是这条候选的冗余度；空集合时冗余为 0
                let redundancy = picked
                    .iter()
                    .map(|&j| cosine_similarity(&self.vectors[ci], &self.vectors[j]))
                    .fold(0.0f32, f32::max);
                let mmr = lambda * rel - (1.0 - lambda) * redundancy;
                if best.map_or(true, |(_, b)| mmr > b) {
                    best = Some((pos, mmr));
                }
            }
            match best {
                Some((pos, _)) => {
                    used[pos] = true;
                    picked.push(cand[pos].0);
                }
                None => break,
            }
        }
        // 输出按相关度降序：调用方（提示裁剪）依赖"分数高的先放"，与选择顺序无关
        let mut hits: Vec<(usize, f32)> = cand
            .into_iter()
            .filter(|(i, _)| picked.contains(i))
            .collect();
        hits.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        self.collect(hits)
    }

    /// 全量相关度（未排序）
    fn relevance(&self, query: &str) -> Vec<(usize, f32)> {
        let q = self.embedder.embed(query);
        self.vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_similarity(&q, v)))
            .collect()
    }

    fn collect(&self, scored: Vec<(usize, f32)>) -> Vec<ScoredChunk> {
        scored
            .into_iter()
            .map(|(i, s)| ScoredChunk {
                index: i,
                chunk: self.chunks[i].clone(),
                score: s,
            })
            .collect()
    }
}

// ==================== 4. 提示组装 ====================

/// 提示组装参数
#[derive(Clone, Debug)]
pub struct PromptOpts {
    /// 参考资料部分允许占用的**字符**预算（不含模板文字）
    pub max_context_chars: usize,
    /// 系统约束（默认要求"只依据资料回答"，见 [`PromptOpts::default`]）
    pub instruction: String,
}

impl Default for PromptOpts {
    fn default() -> Self {
        PromptOpts {
            max_context_chars: 1200,
            instruction: "请只依据下面提供的参考资料回答问题；\
                          如果资料中没有相关信息，请直接回答“我无法从提供的资料中找到答案”，\
                          不要凭自己的知识补充。"
                .to_string(),
        }
    }
}

/// 组装好的提示：正文 + 实际用掉的字符数（用于验证预算确实被遵守）
#[derive(Clone, Debug)]
pub struct RagPrompt {
    pub text: String,
    pub used_chars: usize,
    pub used_chunks: usize,
}

/// 把检索结果拼成 RAG 提示。
///
/// 两条约束直接写进模板：**标注来源**（回答可溯源，出问题能定位到具体文档）与
/// **只依据资料回答**（否则模型会把自己的记忆和检索内容混在一起，
/// 检索反而成了幻觉的帮凶）。
///
/// 超预算时的策略：按分数从高到低放，放不下就跳过并继续试后面更短的块
/// （而不是直接停下——一个大块装不下，不代表后面的小块也不该进）。
/// 但**第一条**如果本身就超预算，就按字符截断后收下：否则预算小于最小块时
/// 会一条都放不进去，检索白做、提示里只剩"我无法找到答案"。
pub fn build_rag_prompt(question: &str, hits: &[ScoredChunk], opts: &PromptOpts) -> RagPrompt {
    let mut body = String::new();
    let mut used_chars = 0usize;
    let mut used_chunks = 0usize;
    for h in hits {
        let text = h.chunk.text.trim();
        if text.is_empty() {
            continue;
        }
        let remain = opts.max_context_chars.saturating_sub(used_chars);
        if remain == 0 {
            break;
        }
        let len = text.chars().count();
        let piece = if len <= remain {
            text.to_string()
        } else if used_chunks == 0 {
            truncate_chars(text, remain)
        } else {
            continue;
        };
        used_chunks += 1;
        let piece_len = piece.chars().count();
        used_chars += piece_len;
        if h.chunk.source.is_empty() {
            body.push_str(&format!(
                "[{used_chunks}] 字符 {}-{}\n{piece}\n\n",
                h.chunk.start, h.chunk.end
            ));
        } else {
            body.push_str(&format!(
                "[{used_chunks}] 来源: {}（字符 {}-{}）\n{piece}\n\n",
                h.chunk.source, h.chunk.start, h.chunk.end
            ));
        }
    }

    let text = format!(
        "{}\n\n参考资料:\n{}\n问题: {}\n\n回答:",
        opts.instruction,
        if body.is_empty() { "(无)\n" } else { body.as_str() },
        question
    );
    RagPrompt { text, used_chars, used_chunks }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::TransformerConfig;
    use crate::rng::Rng;

    /// 检查分块的通用不变量：全在字符边界上切、相邻块重叠恰为 `opts.overlap`、
    /// 去掉重叠后拼接恰好还原原文；通过后返回各块。
    fn rebuild_and_check(text: &str, opts: &ChunkOpts) -> Vec<Chunk> {
        let chunks = chunk_text(text, opts);
        assert!(!chunks.is_empty(), "非空文本应至少切出一块");
        let chars: Vec<char> = text.chars().collect();
        for c in &chunks {
            // 中文安全：字符数必须与下标区间一致（按字节切就会在这里露馅）
            assert_eq!(
                c.text.chars().count(),
                c.end - c.start,
                "块 {}-{} 的字符数与区间不符（多字节字符被切开了）：{:?}",
                c.start,
                c.end,
                c.text
            );
            assert_eq!(c.text, chars[c.start..c.end].iter().collect::<String>());
        }
        for w in chunks.windows(2) {
            assert_eq!(
                w[0].end - w[1].start,
                opts.overlap,
                "相邻块重叠应恰为 {} 个字符：{:?} 与 {:?}",
                opts.overlap,
                w[0].text,
                w[1].text
            );
        }
        // 除最后一块外，去掉与后块的重叠部分后拼接，应恰为原文
        let mut rebuilt = String::new();
        for (i, c) in chunks.iter().enumerate() {
            // 重叠区留给后一块，这里只取到后一块的起点
            let keep = chunks.get(i + 1).map_or(c.end, |n| n.start);
            assert!(keep >= c.start && keep <= c.end, "重叠区超出了块自身范围");
            rebuilt.extend(&chars[c.start..keep]);
        }
        assert_eq!(rebuilt, text, "去掉重叠后应恰好还原原文");
        chunks
    }

    /// 分块：覆盖不丢不重、相邻块重叠恰为 overlap、每块不超长
    #[test]
    fn test_chunking_is_lossless_and_overlap_exact() {
        let text = "第一章开篇。故事从这里开始，主角走进小镇。\
                    第二章发展。情节逐渐展开，冲突一点点浮出水面。\
                    第三章收尾。一切尘埃落定，主角离开小镇。";
        let opts = ChunkOpts::new(20, 5);
        let chunks = rebuild_and_check(text, &opts);
        assert!(chunks.len() > 1, "这么长的文本应切成多块");

        for c in &chunks {
            assert!(c.len() <= opts.size, "每块不超过 {} 字符，实际 {}", opts.size, c.len());
        }

        // 空文本与"比块还短的文本"都要能正确处理
        assert!(chunk_text("", &opts).is_empty());
        let short = chunk_text("短短的", &opts);
        assert_eq!(short.len(), 1);
        assert_eq!(short[0].text, "短短的");
    }

    /// 对齐分块：非末块都收在句子/段落结束符上，且仍然无损
    #[test]
    fn test_aligned_chunking_ends_on_sentence_boundary() {
        let text = "甲甲甲甲甲。乙乙乙乙乙。丙丙丙丙丙。丁丁丁丁丁。";
        let opts = ChunkOpts::aligned(10, 2);
        let chunks = rebuild_and_check(text, &opts);
        assert!(chunks.len() >= 2, "应切出多块，实际 {chunks:?}");
        for c in &chunks[..chunks.len() - 1] {
            assert!(
                c.text.ends_with('。'),
                "对齐后除末块外都应以句末标点收尾，实际 {:?}",
                c.text
            );
        }
    }

    /// 余弦相似度：自比为 1、正交为 0、零向量不产生 NaN
    #[test]
    fn test_cosine_similarity_basics() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6, "自比为 1");
        assert!(cosine_similarity(&a, &b) > 0.9, "同向向量应接近 1");
        assert!(cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6, "正交为 0");
        assert!(cosine_similarity(&[1.0, 0.0], &[-1.0, 0.0]) < -0.99, "反向为 -1");
        let z = vec![0.0; a.len()];
        assert_eq!(cosine_similarity(&z, &a), 0.0, "零向量不得产生 NaN");
        assert!(cosine_similarity(&z, &z).is_finite());
    }

    /// TF-IDF：IDF 随文档频率单调下降；查询能命中正确块并排第一
    #[test]
    fn test_tfidf_retrieval_ranks_correct_chunk_first() {
        let docs = vec![
            "退款的流程是：在订单页点击申请退款，填写原因后等待审核".to_string(),
            "密码的找回需要绑定手机号，通过短信验证码重置".to_string(),
            "配送的范围覆盖全国，偏远地区可能需要额外时间".to_string(),
        ];
        let corpus: Vec<String> = docs.clone();
        let tf = TfIdf::fit(&corpus);
        // "的" 三篇文档里都有（df = 3），"退款" 只在一篇里（df = 1）⇒ 后者更具区分性、IDF 更高
        assert!(
            tf.idf_of("退款") > tf.idf_of("的"),
            "稀有词 IDF 应更高：退款 {} vs 的 {}",
            tf.idf_of("退款"),
            tf.idf_of("的")
        );
        assert_eq!(tf.idf_of("不存在的词"), 0.0, "未登录词 IDF 为 0");
        assert!(tf.vocab().contains(&"退款".to_string()), "词表应含语料里的字词");
        assert_eq!(tf.vocab().len(), tf.dim(), "词表大小即向量维度");

        let emb = Box::new(tf);
        assert!(emb.dim() > 0);
        let r = Retriever::from_text(emb, &docs.join("\n"), &ChunkOpts::new(30, 5));
        let hits = r.search("怎么申请退款", 3);
        assert!(!hits.is_empty());
        assert!(
            hits[0].chunk.text.contains("退款"),
            "最相关的块应排第一，实际 {:?}",
            hits[0].chunk.text
        );
        assert!(hits[0].score > 0.0);

        // 查询里的字全都没在语料出现过 ⇒ 向量为零 ⇒ 分数全 0，而不是 NaN
        let miss = r.search("量子纠缠", 2);
        assert!(miss.iter().all(|h| h.score == 0.0), "未登录查询应为 0 分：{miss:?}");
    }

    /// 特征哈希：不需要词表即可检索，维度固定，自比相似度为 1
    #[test]
    fn test_hashing_embedder_retrieves_and_has_fixed_dim() {
        let emb = HashingEmbedder::new(256);
        assert_eq!(emb.dim(), 256);
        let v = emb.embed("退款流程");
        assert_eq!(v.len(), 256);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-5, "应做 L2 归一化");
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6);

        let docs = "退款的流程是在订单页申请。密码找回需要绑定手机号。配送覆盖全国大部分地区。";
        let r = Retriever::from_text(Box::new(emb), docs, &ChunkOpts::new(14, 2));
        assert_eq!(r.dim(), 256, "检索器维度应等于向量化器维度");
        assert_eq!(r.embed_query("密码").len(), 256);
        let hits = r.search("如何重置密码", 3);
        assert!(
            hits[0].chunk.text.contains("密码"),
            "哈希检索应命中含'密码'的块，实际 {:?}",
            hits[0].chunk.text
        );
    }

    /// MMR：语料里有近重复块时，λ < 1 会选多样化块而不是第二条重复块
    #[test]
    fn test_mmr_prefers_diverse_chunk_over_duplicate() {
        let a = "退款流程是在订单页面提交申请并填写退款原因等待审核";
        // 前两篇只差一个标点，向量几乎相同；第三篇也提到"退款"但与查询只沾一点边
        let texts = vec![
            format!("{a}。"),
            format!("{a}！"),
            "退款的到账时间一般是三到五个工作日。".to_string(),
        ];
        let tf = TfIdf::fit(&texts);
        // 一条文档一个块：分块由别的测试负责，这里只想看重排行为
        let chunks: Vec<Chunk> = texts
            .iter()
            .enumerate()
            .map(|(i, t)| Chunk {
                text: t.clone(),
                start: 0,
                end: t.chars().count(),
                source: format!("doc{i}.txt"),
            })
            .collect();
        let r = Retriever::build(Box::new(tf), chunks);

        // λ = 1 就是纯相关度：前两名必然是那一对近乎重复的"退款"文档
        let pure = r.search_mmr("退款流程怎么走", 2, 1.0, 3);
        assert_eq!(pure.len(), 2);
        assert!(
            pure[0].chunk.text.contains("退款") && pure[1].chunk.text.contains("退款"),
            "λ = 1 时应退化成纯相关度排序：{pure:?}"
        );

        // λ = 0.3：第二名应换成"到账时间"那条（多样化），而不是第二条近乎一样的"退款"块
        let mmr = r.search_mmr("退款流程怎么走", 2, 0.3, 3);
        assert_eq!(mmr.len(), 2);
        assert_eq!(mmr[0].index, 0, "最相关的一条仍应排第一");
        assert!(
            mmr[1].chunk.text.contains("到账时间"),
            "λ < 1 时第二名应为多样化块，实际 {:?}",
            mmr[1].chunk.text
        );
        assert_ne!(mmr[0].index, mmr[1].index, "同一条块不应被重复选中");
        // 输出按相关度降序，便于下游按分数裁剪
        assert!(mmr[0].score >= mmr[1].score);
    }

    /// 提示组装：含资料与问题、标注来源、预算内不超长
    #[test]
    fn test_prompt_assembly_respects_budget() {
        let text = "甲甲甲甲甲甲甲甲甲甲。乙乙乙乙乙乙乙乙乙乙。丙丙丙丙丙丙丙丙丙丙。";
        let r = Retriever::from_text(
            Box::new(HashingEmbedder::new(64)),
            text,
            &ChunkOpts::new(11, 0),
        );
        let hits = r.search("甲甲甲", 3);

        // 预算充足：全部资料都在提示里
        let roomy = PromptOpts { max_context_chars: 200, ..PromptOpts::default() };
        let full = build_rag_prompt("这段在讲什么？", &hits, &roomy);
        assert!(full.text.contains("这段在讲什么？"), "提示必须包含问题");
        for h in &hits {
            assert!(
                full.text.contains(h.chunk.text.trim()),
                "预算充足时资料应完整出现：{:?}",
                h.chunk.text
            );
        }
        assert!(full.text.contains("参考资料"), "应有资料区块标题");
        assert!(full.text.contains("只依据") || full.text.contains("无法从提供的资料"), "应带上约束语句");
        assert!(full.used_chars <= roomy.max_context_chars);
        assert_eq!(full.used_chunks, hits.len());

        // 预算很小：按分数保留并截断，实际用量不超预算
        let tight = PromptOpts { max_context_chars: 7, ..PromptOpts::default() };
        let cut = build_rag_prompt("这段在讲什么？", &hits, &tight);
        assert!(
            cut.used_chars <= tight.max_context_chars,
            "实际用量 {} 不应超过预算 {}",
            cut.used_chars,
            tight.max_context_chars
        );
        assert_eq!(cut.used_chunks, 1, "预算只够第一条时不该硬塞更多");
        assert!(cut.text.contains(&truncate_chars(hits[0].chunk.text.trim(), 7)));
        assert!(cut.text.contains("这段在讲什么？"), "裁剪资料不能把问题也裁掉");

        // 空检索结果也要给出结构完整的提示（否则下游会拿到半个模板）
        let none = build_rag_prompt("这段在讲什么？", &[], &roomy);
        assert!(none.text.contains("(无)"));
        assert_eq!(none.used_chunks, 0);
    }

    /// 目录建索引：块带来源文件名，且两次建索引结果一致
    #[test]
    fn test_retriever_from_corpus_dir_tags_source() {
        let dir = std::path::Path::new("data/corpus");
        if !dir.is_dir() {
            eprintln!("跳过：本地没有 data/corpus 目录");
            return;
        }
        let opts = ChunkOpts::new(200, 40);
        let r = Retriever::from_corpus_dir(Box::new(HashingEmbedder::new(512)), dir, &opts)
            .expect("读取 data/corpus 失败");
        assert!(r.len() > 10, "语料目录应切出足够多的块，实际 {}", r.len());
        assert!(!r.is_empty());
        assert!(r.chunks().iter().all(|c| !c.source.is_empty()), "每块都应有来源");
        assert!(r.chunks().iter().all(|c| !c.is_empty()), "块不应为空");

        // 同一目录两次建索引：块序列必须完全一致（依赖文件排序）
        let r2 = Retriever::from_corpus_dir(Box::new(HashingEmbedder::new(512)), dir, &opts)
            .expect("读取 data/corpus 失败");
        let keys: Vec<(String, usize, usize)> = r
            .chunks()
            .iter()
            .map(|c| (c.source.clone(), c.start, c.end))
            .collect();
        let keys2: Vec<(String, usize, usize)> = r2
            .chunks()
            .iter()
            .map(|c| (c.source.clone(), c.start, c.end))
            .collect();
        assert_eq!(keys, keys2, "同一目录两次建索引的结果应一致");
    }

    /// 稠密向量：维度 = n_embd、自比为 1、同一文本两次嵌入一致
    #[test]
    fn test_model_embedder_dim_and_determinism() {
        let tok_text = "甲乙丙丁戊己庚辛壬癸";
        let tok = Tokenizer::char(tok_text);
        let cfg = TransformerConfig {
            n_embd: 16,
            n_head: 2,
            n_layer: 2,
            block_size: 32,
            ..TransformerConfig::tiny(tok.vocab_size())
        };
        let model = Arc::new(Transformer::new(cfg, &mut Rng::new(7)));
        let emb = ModelEmbedder::new(model, tok);

        assert_eq!(emb.dim(), 16, "池化输出维度应等于 n_embd");
        let v = emb.embed("甲乙丙");
        assert_eq!(v.len(), 16);
        assert!((l2_norm(&v) - 1.0).abs() < 1e-5, "稠密向量应 L2 归一化");
        assert!((cosine_similarity(&v, &v) - 1.0).abs() < 1e-6, "同一文本自比应为 1");
        // 推理路径没有 dropout，同一文本两次嵌入必须逐位相同
        assert_eq!(v, emb.embed("甲乙丙"), "同一文本的嵌入应可复现");

        // 超长文本按 block_size 截断，不应 panic
        let long: String = tok_text.chars().cycle().take(200).collect();
        let vl = emb.embed(&long);
        assert_eq!(vl.len(), 16);
        assert!((l2_norm(&vl) - 1.0).abs() < 1e-5);
    }
}
