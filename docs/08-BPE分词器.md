# 第 8 课：BPE 分词器 —— 让模型"读懂"文字

> 代码位置：[src/tokenizer.rs](../src/tokenizer.rs)
> 演示入口：[src/main.rs](../src/main.rs)（`demo_bpe`）
> 语料：[src/data.rs](../src/data.rs)（`CORPUS`）

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

// id 序列 -> 文本（按 id 反查字符，越界 id 带下标 panic 提示）
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

### 5.1 全量语料：不再截断

早期实现的训练主循环每次合并都要**重新扫描整个语料**统计 pair 频率，语料一大就慢得离谱，所以曾按 1MB 上限截断。现在训练改成了**增量更新**（见 5.3）：初始只全量扫一遍建频次表，之后每轮合并只调整受影响的几个 pair，代价与语料大小脱钩——因此截断已删除，**全量语料都参与统计**：

```rust
pub fn train(corpus: &str, target_vocab: usize) -> Self {
    Self::train_bytes(corpus.as_bytes(), target_vocab)
}
```

### 5.2 初始化：字节级词表

```rust
assert!(target_vocab >= 256, "BPE 词表至少 256（字节级）");
// 初始：每个 token 就是一个字节
let mut vocab: Vec<Vec<u8>> = (0u16..=255).map(|b| vec![b as u8]).collect();
let mut merges: Vec<(u16, u16)> = Vec::new();

// 语料 -> 字节 -> id 序列
let mut ids: Vec<u16> = data.iter().map(|&b| b as u16).collect();
```

初始词表 = **256 个字节**（0~255），每个 token 恰好是一个字节。为什么用字节而不是字符？

- 任何 UTF-8 文本都可以拆成字节，**不存在"词表外"字符**（OOV = 0）
- 中文等多语言文本也能直接编码（一个汉字是 3 个字节）
- 早期真实大模型用的就是字节级 BPE

### 5.3 训练主循环：初始统计 → 堆选优 → 单遍合并 + 增量更新

训练分三段：**初始只全量扫一遍**建立 pair 频次表和最大堆；之后**每轮合并只单遍扫一遍序列**，顺手增量调整受影响的几个 pair 频次——不再每轮重建统计（旧实现是 O(merges × 语料)，现在是 O(语料 + merges × 合并命中数)）。

**第一步：初始统计**

```rust
// pair -> 当前频次：初始扫一遍，之后随每次合并增量增减
let mut pair_freq: HashMap<(u16, u16), usize> = HashMap::new();
for w in ids.windows(2) {
    *pair_freq.entry((w[0], w[1])).or_insert(0) += 1;
}
// 最大堆：键 (频次, Reverse(pair))，频次最高者在顶、平手取 pair 值小者
// （确定性 tie-break，保证结果与旧实现的 max_by 规则一致）
let mut heap: BinaryHeap<(usize, Reverse<(u16, u16)>)> = BinaryHeap::new();
for (&p, &f) in &pair_freq {
    if f > 0 {
        heap.push((f, Reverse(p)));
    }
}
```

**第二步：每轮选优 —— 堆 + 懒删除**

频次每次变化都往堆里 push 新条目，旧条目不删（懒删除）；pop 时拿 `pair_freq` 的当前值校验，对不上就说明过期、丢弃继续弹：

```rust
let best = loop {
    match heap.pop() {
        Some((f, Reverse(p))) if f > 0 && pair_freq.get(&p).copied() == Some(f) => break p,
        Some(_) => continue, // 频次已过期（该 pair 后来被合并改动过）
        None => break 'train, // 语料已无可合并 pair（如单字节语料）
    }
};
```

**第三步：单遍合并 + 增量更新频次**

创建新 token 的逻辑不变（`vocab.push` + `merges.push`）。关键是替换阶段：每处合并只影响**三个旧 pair**——左邻 `(prev, a)`、本身 `(a, b)`、右邻 `(b, next)`——对应**两个新 pair** `(prev, N)` 和 `(N, next)`，用 `bump_pair` 增量 ±1 即可，不需要重扫全语料重新统计：

```rust
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
        heap.push((*e, Reverse(pair))); // 新值入堆，旧条目靠懒删除失效
    }
}
```

```rust
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
std::mem::swap(&mut ids, &mut buf); // 双缓冲交替复用，避免逐轮重新分配大 Vec
```

| 步骤 | 代码 | 说明 |
|------|------|------|
| ① 初始统计 | `ids.windows(2)` 滑窗 | 只在开头扫一遍，建 `pair_freq` + 堆 |
| ② 选择 | `heap.pop()` + 频次校验 | 频率最高者；**频率相同取 pair 数值更小的**（`Reverse` 保证结果确定）；过期条目懒删除 |
| ③ 合并 | `vocab.push` + `merges.push` | 新符号的内容 = 两个旧符号内容拼接；id = 当前词表长度（从 256 起）|
| ④ 替换 | 单遍 while 扫描 + `bump_pair` | 把序列里所有该 pair 换成新 id，顺带增量更新受影响 pair 的频次，再进入下一轮 |

训练结束后得到两份"产物"：

- **`vocab: Vec<Vec<u8>>`**：token id → 它代表的字节序列（解码要用）
- **`merges: Vec<(u16, u16)>`**：合并规则表，按下标顺序排列（**越早合并优先级越高**，编码要用）

词表大小 = 256 + 合并次数。演示里 `BPETokenizer::train(CORPUS, 400)` 得到 **400 = 256 字节 + 144 次合并**。

---

## 6. BPE 编码：encode()

编码 = 对新文本执行**同样的合并**。但合并顺序必须和训练时一致：训练时越早合并的规则优先级越高（它对应的 token id 更小）。

```rust
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
```

要点：

- **按规则优先级单趟扫描**（通行的标准实现）：从 `merges[0]` 到 `merges[最后]`，每条规则在序列上扫一遍，能合并就替换成它对应的新 token。复杂度 O(len × 合并数)，大语料也能秒级完成（若"每次只合并一个 pair 并全量重扫"是 O(n²×m)，174KB 语料会卡死）
- `new_id = 256 + idx`：merge 下标 idx 直接映射成 token id——因为训练时第 idx 次合并恰好产生 id `256 + idx`
- 字节 id（0~255）直接复用训练时的字节 → id 映射
- 演示里 `"the garden"` 编码后只有 **2 个 token**（"the" 和 " garden" 都被压缩成了单个 token）

以 `"lowest"` 为例走一遍：字节 `[l,o,w,e,s,t]` → 应用规则 0（假设是 `(l,o)`）→ `[256,w,e,s,t]` → 应用规则 1（`(256,w)`）→ `[257,e,s,t]` → 继续应用 `(e,s)`、`(s,t)` 对应规则……最终 6 个字节被压成 3 个 token（单元测试要求 `"low"` 编码后不超过 3 个 token）。

---

## 7. BPE 解码：decode()

解码是查表 + 拼接：每个 id 查到它代表的字节序列，拼起来还原文本。

```rust
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
                // 合法前缀：末尾停在字符中间（半个汉字），丢掉不完整的尾部字节
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
```

- `vocab[id]`：id → 字节序列（0~255 是单字节，256+ 是合并出来的多字节序列）
- 越界 id 用 `vocab.get(id)` 拦截：带下标信息的 panic 提示（decode 也会因传入非法 id 报错，不再是"永不失败"）
- **末尾半个字符**：生成按 token 数（`max_new`）截断时，最后一个 token 可能只覆盖某个字符的一部分，此时末尾几个字节凑不成字符——丢掉它们返回前面完整的部分。这是"按 token 截断"的固有结果（中文一个字 3 字节，停半个字的概率并不低），不是错误，所以**不打印警告**
- `skip_invalid_utf8`：真正的非法字节（如 [E4] 后面直接跟 ASCII）会跳过无效字节只保留合法部分，并在 stderr 打印警告；中间位置的乱码已由 7.1 的采样约束排除，走到这里就说明有 bug

> 完整闭环：`decode(encode("lowest new")) == "lowest new"`（单元测试 `test_bpe_roundtrip` 验证）。

### 7.1 生成时的 UTF-8 约束（推理专用）

训练时 `decode(encode(x)) == x` 永远成立，但**推理不一样**：模型是从 logits 里自由采样的，可能挑中"半个汉字"的 token。

字节级词表里 0~255 是单字节，所以必然存在这些 token：

```text
id 228 -> [E4]        "中"（E4 B8 AD）的首字节
id 184 -> [B8]        中间字节
id 173 -> [AD]        末尾字节
```

碎片一旦乱序（比如 [E4] 后面直接跟 ASCII），拼出来的字节流就不是合法 UTF-8，`decode` 只能丢弃无效字节并打印警告——**文字会缺字、乱码，且已经无法补救**。

正确做法是在**采样前**就把它们排除，而不是等拼完再擦屁股：

| 位置 | 函数 | 作用 |
|------|------|------|
| `src/tokenizer.rs` | `utf8_pending(bytes)` | 判断字节串是否仍是**合法的 UTF-8 前缀**（允许停在字符中间，如 `E4 B8`）；返回未拼完的尾部字节，`None` 表示已非法 |
| `src/sample.rs` | `pending_tail(vocab, ids)` | 从已生成的 token 反推末尾未拼完的字节（最多 3 字节，需回看末尾几个 token） |
| `src/sample.rs` | `mask_illegal_utf8(logits, vocab, pending)` | 把"接上后不再合法"的 token 的 logit 置为 `-inf`，采样时自然抽不到 |

```rust
// src/sample.rs（generate / beam_search 每步采样前）
let row: &[f32] = match vocab_bytes {
    Some(vocab) => {
        masked.clear();
        masked.extend_from_slice(last_row);
        mask_illegal_utf8(&mut masked, vocab, &pending_tail(vocab, &ids));
        &masked
    }
    None => last_row, // char 分词器每个 token 都是完整字符，无需约束
};
```

等价于"把这些 token 的概率设为 0 后重新归一化"：温度、top-k / top-p、重复惩罚的顺序都不受影响，只是候选集变小了。代价是每步多一次 O(词表大小) 的字节检查（8192 词表约几十微秒），换来**生成的字节流在任何位置都不会出现乱码**。

约束保证的是"每一步接上后仍是合法前缀"，所以还剩最后一种情况：生成到 `max_new` 停下来时，末尾可能正好是"半个字符"（前缀合法，但整个字节流不是完整的 UTF-8）。这不是乱码，只是被 token 数截断了，`decode` 会把尾部这几个字节丢掉（见第 7 节）。

因此 `decode` 的两种兜底各司其职：

| 情况 | 现象 | 处理 |
|------|------|------|
| 末尾停在半个字符 | `from_utf8` 报 `incomplete ... from index N`，N = 完整部分长度 | 丢掉尾部字节，**不报警**（正常现象） |
| 中间出现非法字节 | `from_utf8` 报 `invalid ... from index N`，N < 长度 | 跳过无效字节 + stderr 警告（采样约束已排除，出现即 bug） |

> `char` 分词器不受影响：它的每个 token 就是一个完整字符，`vocab_bytes()` 返回 `None`，跳过整个约束。

---

## 8. 训练 / 编码 / 解码 对照总结

| 操作 | 一句话 | 关键代码 | 产物/结果 |
|------|--------|---------|-----------|
| 训练 train | 从语料学合并规则 | 统计 pair → 合并最高频 → 替换（循环至目标词表大小） | `merges`（规则）+ `vocab`（字节序列）|
| 编码 encode | 对新文本按规则贪心合并 | 按规则优先级（merges 顺序）单趟扫描替换 | 一串 token id |
| 解码 decode | id → 字节序列拼接 | `vocab[id]` 逐个拼接；末尾半个字符丢掉，非法字节跳过（见第 7 节） | 还原的文本 |
| 生成约束 | 采样时剔除"半个字符" | `utf8_pending` + `mask_illegal_utf8`（见 7.1） | 生成的字节流中间不会出现乱码 |

三者关系：**编码必须复现训练时的合并顺序**，解码只是查表，所以编码、解码天然互逆，`decode(encode(x)) == x`。

---

## 9. 运行与测试

```bash
cargo test   # 全部测试通过
cargo run --release -- demo    # 演示 2（BPE）：词表 400（256 + 144 次合并）；"Red" -> [82, 101, 100]；"the garden" -> 2 个 token
```

---

## 10. 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

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

## 12. 扩展：分词器序列化（第 39 课补充）

训练 BPE 分词器需要遍历整个语料统计频率，大语料可能耗时数十秒。序列化后可以：

```rust
// 训练后保存（统一 Tokenizer 入口）
let tok = Tokenizer::bpe(corpus, 2048);
tok.save("tokenizer.json");

// 下次直接加载（秒级完成）
let tok = Tokenizer::load("tokenizer.json");
```

`config/config.json` 中通过 `tokenizer_file` 字段控制：

```jsonc
{
  "train": {
    "tokenizer_file": null  // null = 从语料训练并自动保存到 {out_dir}/tokenizer.json
  }
}
```

这保证了训练、评估、生成、对话都使用完全相同的词表。
