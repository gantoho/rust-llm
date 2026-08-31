# 第 8 课：BPE 分词器 —— 让模型"读懂"文字

> 代码位置：[src/tokenizer.rs](file:///d:/Code/Rust/llm_from_scratch/src/tokenizer.rs)
> 演示入口：[src/main.rs](file:///d:/Code/Rust/llm_from_scratch/src/main.rs)（`demo_bpe`）
> 语料：[src/data.rs](file:///d:/Code/Rust/llm_from_scratch/src/data.rs)（`CORPUS`）

---

## 1. 本课要搞懂的问题

1. 模型只能吃数字，文字怎么变成数字？
2. 按词切、按字符切，各有什么问题？有没有更好的方案？
3. BPE 到底是什么算法？训练、编码、解码分别怎么做？

---

## 2. 为什么需要分词

大语言模型的输入输出都是**数字**：输入是一串 token id（`usize`），输出是对每个 token id 的预测分数。所以第一步要解决的问题是：**把人类语言变成一串整数**。

| 方案 | 词表大小 | 优点 | 缺点 |
|------|---------|------|------|
| 按词切分 | 数十万 | 语义单元完整 | 词表巨大；遇到没见过的词（OOV）直接抓瞎；run / ran / running 是 3 个互不相干的 token |
| 按字符切分 | 几十 | 词表极小、无 OOV | 序列变长 5~10 倍；"lowest" 被拆成 6 个字符，丢失"这是 low 的最高级"这种结构信息 |
| **子词切分（BPE）** | 几千~几万 | 常见词 1 个 token，罕见词拆成子词 | 算法比前两者复杂（本课重点） |

核心思想一句话：**高频出现的片段合并成一个 token，低频内容用更小的片段表示。**

---

## 3. 字符级分词 CharTokenizer（对照组）

### 3.1 结构

```rust
pub struct CharTokenizer {
    chars: Vec<char>,            // 词表：语料中出现过的所有字符
    stoi: HashMap<char, usize>,  // 字符 -> id
}
```

### 3.2 构建词表

`new(text)` 扫描语料，**按字符首次出现的顺序**收集去重：

```rust
for c in text.chars() {
    if seen.insert(c) {
        chars.push(c);
    }
}
```

比如对 `"hello world hello"`：

- 从左到右第一次遇到的字符依次是 h, e, l, o, ' ', w, r, d → 词表就是这 8 个字符
- `stoi = {h:0, e:1, l:2, o:3, ' ':4, w:5, r:6, d:7}`，`vocab_size() == 8`

### 3.3 编码 / 解码

```rust
// 文本 -> id 序列（每个字符查表）
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

// id 序列 -> 文本（按 id 反查字符）
pub fn decode(&self, ids: &[usize]) -> String {
    ids.iter().map(|&i| self.chars[i]).collect()
}
```

编码和解码互为逆运算：`decode(encode(text)) == text`（单元测试 `test_char_tokenizer_roundtrip` 验证了这一点）。

> 注意：字符级编码遇到词表外的字符会**直接 panic**——这是它最大的短板。
> 演示程序里 `CharTokenizer::new(CORPUS)` 得到 **35** 个字符的词表，`"fox"` 编码为 `[20, 7, 21]`。

---

## 4. BPE 的直觉：把高频 pair "焊"在一起

BPE（Byte Pair Encoding，字节对编码）源自数据压缩算法，规则很简单：**反复找到出现次数最多的相邻符号对，把它们合并成一个新符号。**

以单元测试的语料 `"low low low low low lowest lowest newest newest newest"` 为例：

| 轮次 | 最高频相邻对 | 出现次数 | 合并成 | 效果 |
|------|-------------|---------|--------|------|
| 1 | `(l, o)` | 7（5 个 low + 2 个 lowest） | id 256 | 每个 `low` 变成 `[256, w]` |
| 2 | `(256, w)` | 7（同上） | id 257 | `low` 被压成一个 token `[257]` |
| 3 | `(' ', 257)` | 5（4 个词间空格 + 1 个 lowest 前空格） | id 258 | 连"空格 + low"这样的跨词片段也能合并 |

经过两轮合并，出现 7 次的常用子词 "low" 从 3 个字节压缩成了 **1 个 token**——这就是"用更少的 token 表示更多文本"。

---

## 5. BPE 训练：train()

### 5.1 初始化：字节级词表

```rust
assert!(target_vocab >= 256, "BPE 词表至少 256（字节级）");
// 初始：每个 token 就是一个字节
let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
let mut merges: Vec<(u16, u16)> = Vec::new();

// 语料 -> 字节 -> id 序列
let mut ids: Vec<u16> = corpus.as_bytes().iter().map(|&b| b as u16).collect();
```

初始词表 = **256 个字节**（0~255），每个 token 恰好是一个字节。为什么用字节而不是字符？

- 任何 UTF-8 文本都可以拆成字节，**不存在"词表外"字符**（OOV = 0）
- 中文等多语言文本也能直接编码（一个汉字是 3 个字节）
- GPT-2 等真实模型用的就是字节级 BPE

### 5.2 训练主循环：统计 → 选择 → 合并 → 替换

```rust
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
            i += 2;               // 一次吞掉两个符号
        } else {
            new_ids.push(ids[i]);
            i += 1;
        }
    }
    ids = new_ids;
}
```

| 步骤 | 代码 | 说明 |
|------|------|------|
| ① 统计 | `ids.windows(2)` 滑窗 | 每相邻两个 id 组成 pair，用 HashMap 计数 |
| ② 选择 | `pair_freq.iter().max_by(...)` | 频率最高者；**频率相同取 pair 数值更小的**（保证结果确定）|
| ③ 合并 | `vocab.push` + `merges.push` | 新符号的内容 = 两个旧符号内容拼接；id = 当前词表长度（从 256 起）|
| ④ 替换 | 单遍 while 扫描 | 把序列里所有该 pair 替换成新 id，再进入下一轮 |

训练结束后得到两份"产物"：

- **`vocab: Vec<Vec<u8>>`**：token id → 它代表的字节序列（解码要用）
- **`merges: Vec<(u16, u16)>`**：合并规则表，按下标顺序排列（**越早合并优先级越高**，编码要用）

词表大小 = 256 + 合并次数。演示里 `BPETokenizer::train(CORPUS, 400)` 得到 **400 = 256 字节 + 144 次合并**。

---

## 6. BPE 编码：encode()

编码 = 对新文本执行**同样的贪心合并**。但合并顺序必须和训练时一致：训练时越早合并的规则优先级越高（它对应的 token id 更小）。

```rust
pub fn encode(&self, text: &str) -> Vec<usize> {
    let mut ids: Vec<u16> = text.as_bytes().iter().map(|&b| b as u16).collect();
    loop {
        // 找到 ids 中优先级最高（merge 下标最小）的可合并 pair
        let mut best_merge: Option<usize> = None;
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
        let Some(idx) = best_merge else { break; };
        let new_id = (256 + idx) as u16;
        ids[pos] = new_id;
        ids.remove(pos + 1);
    }
    ids.into_iter().map(|x| x as usize).collect()
}
```

要点：

- 每次找**当前序列里可合并且 merge 下标最小**的 pair，替换后重新找，直到没有可合并的 pair
- `new_id = 256 + idx`：merge 下标 idx 直接映射成 token id——因为训练时第 idx 次合并恰好产生 id `256 + idx`
- 字节 id（0~255）直接复用训练时的字节 → id 映射
- 演示里 `"the garden"` 编码后只有 **2 个 token**（"the" 和 " garden" 都被压缩成了单个 token）

以 `"lowest"` 为例走一遍：字节 `[l,o,w,e,s,t]` → 合并 `(l,o)` → `[256,w,e,s,t]` → 合并 `(256,w)` → `[257,e,s,t]` → 继续合并 `(e,s)`、`(s,t)`……最终 6 个字节被压成 3 个 token（单元测试要求 `"low"` 编码后不超过 3 个 token）。

---

## 7. BPE 解码：decode()

解码是查表 + 拼接：每个 id 查到它代表的字节序列，拼起来还原文本。

```rust
pub fn decode(&self, ids: &[usize]) -> String {
    let mut bytes: Vec<u8> = Vec::new();
    for &id in ids {
        bytes.extend_from_slice(&self.vocab[id]);
    }
    String::from_utf8_lossy(&bytes).to_string()
}
```

- `vocab[id]`：id → 字节序列（0~255 是单字节，256+ 是合并出来的多字节序列）
- `String::from_utf8_lossy`：万一拼出非法 UTF-8，用替换字符 `�` 顶替而不是 panic——**decode 永不失败**

> 完整闭环：`decode(encode("lowest new")) == "lowest new"`（单元测试 `test_bpe_roundtrip` 验证）。

---

## 8. 训练 / 编码 / 解码 对照总结

| 操作 | 一句话 | 关键代码 | 产物/结果 |
|------|--------|---------|-----------|
| 训练 train | 从语料学合并规则 | 统计 pair → 合并最高频 → 替换（循环至目标词表大小） | `merges`（规则）+ `vocab`（字节序列）|
| 编码 encode | 对新文本按规则贪心合并 | 反复找 merge 下标最小的可合并 pair 并替换 | 一串 token id |
| 解码 decode | id → 字节序列拼接 | `vocab[id]` 逐个拼接 + `from_utf8_lossy` | 还原的文本 |

三者关系：**编码必须复现训练时的合并顺序**，解码只是查表，所以编码、解码天然互逆，`decode(encode(x)) == x`。

---

## 9. 运行与测试

```bash
cargo test   # 15 个测试全部通过，其中 tokenizer 的 2 个验证了编解码互逆
cargo run    # 演示 2（BPE）：词表 400（256 + 144 次合并）；"Red" -> [82, 101, 100]；"the garden" -> 2 个 token
```

---

## 10. 动手练习

1. 把语料换成中文句子（如 `"机器学习机器学习深度学习"`），跑一遍 `BPETokenizer::train`，观察"机器""学习"是否会被合并成单个 token（提示：中文 UTF-8 每字 3 字节，BPE 依然适用）。
2. 修改 `train` 的 `target_vocab`，分别用 256 / 300 / 500，比较 `encode("lowest")` 的 token 数变化。
3. 思考：`encode` 为什么必须按"merge 下标最小"合并，而不是按"频率最高"合并？（提示：训练时的合并顺序决定了 id 分配，编码必须复现同一顺序才能保证解码还原）
4. 思考：`CharTokenizer` 遇到词表外字符会 panic，BPE 为什么永远不会有这个问题？

---

## 11. 本课总结

- 模型只认数字，分词器负责"文字 ↔ id"的转换
- 字符级分词：简单直观，但序列长、有词表外字符；词级分词：词表大、仍有 OOV
- BPE：字节级底座（256 个字节起步，零 OOV）+ 反复合并最高频相邻 pair
- 训练产出 `merges`（合并规则）和 `vocab`（id → 字节序列）；编码贪心复现合并；解码查表拼接
- 下一课：token 变成向量之后，怎么让它们"互相看"？——注意力机制！
