# 第 18 课：KV Cache —— 让逐 token 生成不再重复计算

> 代码位置：[src/model.rs](file:///d:/Code/Rust/llm_from_scratch/src/model.rs)（`KVCache` / `MultiHeadAttention` / `GPT::forward`）
> 代码位置：[src/sample.rs](file:///d:/Code/Rust/llm_from_scratch/src/sample.rs)（`generate`）
> 演示入口：[src/main.rs](file:///d:/Code/Rust/llm_from_scratch/src/main.rs)（演示 3：生成 1 / 生成 2）

---

## 1. 本课要搞懂的问题

1. 推理时为什么"历史 token 的 K/V"会被一遍遍重复计算？
2. `KVCache` 的数据结构长什么样？`append` / `seq_len` 各做了什么？
3. cache 模式与全量模式在 `generate` 里的流程有什么不同（首次 vs 之后每步）？
4. 用了缓存之后，为什么生成的概率分布和全量模式**完全一样**？
5. 为什么缓存模式下上下文达到 `block_size` 就必须停止生成？

---

## 2. 问题：为什么历史 K/V 会被重复计算

生成是**逐 token**的：每产生一个新 token，都要把它拼到上下文末尾，再前向一次，从输出的 logits 里采样下一个 token。

全量模式（第 16 课 `generate` 的 `use_kv_cache=false` 分支）每步都把**整个上下文**重新喂给模型：

```
第 1 步：输入 [t0]               → 前向 1 个位置
第 2 步：输入 [t0, t1]           → 前向 2 个位置
第 3 步：输入 [t0, t1, t2]       → 前向 3 个位置
...
第 T 步：输入 [t0, t1, ..., tT]  → 前向 T 个位置
```

关键观察：**第 k 步算出来的前 k-1 个位置的 K/V，和第 k-1 步算出来的一模一样**——推理时权重冻结、输入前缀相同，同一个 Linear 层（`c_k`、`c_v`）对相同输入必然给出相同输出。

既然如此，为什么要重算？直接记住不就好了？这就是 KV Cache 的动机：

| | 全量模式（每步重算） | KV Cache 模式 |
|---|---|---|
| 每步前向的位置数 | T（整个上下文，越来越大） | 只有新来的 1 个位置 |
| K/V 的计算量 | O(T)，累计 O(T²) | 每个位置只算一次，累计 O(T) |
| 额外内存 | 无 | 存所有历史 K/V（O(T·D)） |

> 注意：**只有 K 和 V 需要缓存，Q 不用**。因为预测"下一个 token"只关心新位置上的注意力输出，而它只需要新位置的 Q 去和所有历史位置的 K、V 做注意力。历史位置自己的注意力输出（以及它们的 Q）在生成中根本用不上。

---

## 3. KVCache 的结构

`src/model.rs` 里的定义：

```rust
/// KV 缓存（第 18 课）：
/// 生成第 N 个 token 时，前 N-1 个 token 的 K、V 不需要重算。
/// 把每个注意力层的 K、V 存起来，每次只算新 token 的 K、V 并拼接。
pub struct KVCache {
    k: Option<Tensor>, // [1, T, D]
    v: Option<Tensor>,
}
```

| 字段 | 形状 | 含义 |
|------|------|------|
| `k` | `[1, T, D]` | 该层已缓存的所有位置的 Key（T = 已缓存位置数） |
| `v` | `[1, T, D]` | 该层已缓存的所有位置的 Value |
| `Option` | —— | 空缓存 = `None`；一旦 append 过就一直是 `Some` |

注意：**每个注意力层各有一个 `KVCache`**。`GPT::new_kv_cache` 返回 `Vec<KVCache>`，长度 = `n_layer`（本项目 2 层）：

```rust
pub fn new_kv_cache(&self) -> Vec<KVCache> {
    (0..self.cfg.n_layer).map(|_| KVCache::new()).collect()
}
```

### 3.1 append：把新 K/V 拼到缓存尾部

```rust
fn append_data(prev: &Option<Tensor>, cur: &Tensor) -> Tensor {
    match prev {
        Some(p) => {
            let mut all = p.data();
            all.extend(cur.data());
            let d = cur.shape()[2];
            Tensor::from_vec(all, vec![1, p.shape()[1] + 1, d])
        }
        None => cur.clone(),
    }
}

pub fn append(&mut self, k: &Tensor, v: &Tensor) {
    self.k = Some(Self::append_data(&self.k, k));
    self.v = Some(Self::append_data(&self.v, v));
}
```

以 `[1, T, D]` 为例，`append_data` 做的事：

1. 取旧缓存 `p` 的**数据**（`p.data()`，一维展平数组）；
2. 把新 K/V 的数据 `cur.data()` 拼到末尾；
3. 按 `[1, 旧长度+1, D]` 重新包成张量。

> 细节：`cur` 在推理模式下形状是 `[1, 1, D]`（只算 1 个新位置），所以长度 +1；`d` 从 `cur.shape()[2]` 取。纯数据拼接，推理时无梯度，所以没有走任何 autograd 路径。

### 3.2 seq_len：已缓存了多少位置

```rust
pub fn seq_len(&self) -> usize {
    self.k.as_ref().map(|t| t.shape()[1]).unwrap_or(0)
}
```

- 缓存为空（`None`）→ 0；
- 否则返回 `k` 张量的第 1 维大小，即已缓存的位置数。

它有两个用途（后面会看到）：一是 `GPT::forward` 用它算位置偏移 `base`；二是 `generate` 用它判断要不要停止。

---

## 4. MultiHeadAttention 怎么用缓存

`MultiHeadAttention::forward` 中与缓存相关的部分：

```rust
// 1. 投影得到 Q、K、V（Linear 输出是 2D [B*T, D]，恢复成 3D）
let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
let k = self.c_k.forward(x).reshape(vec![b, t, d]);
let v = self.c_v.forward(x).reshape(vec![b, t, d]);

// 2. KV cache：拼接历史的 K/V（只影响 K、V 的长度）
let (k, v) = match kv_cache {
    Some(cache) => {
        cache.append(&k, &v);
        (cache.k().unwrap(), cache.v().unwrap())
    }
    None => (k, v),
};
let t_total = k.shape()[1];
```

变化只有一处：**K、V 变长**，Q 保持 `[B, T, D]` 不动：

| 变量 | 无缓存 | 有缓存（推理） |
|------|--------|----------------|
| `q` | `[B, T, D]` | `[B, 1, D]`（只算新位置） |
| `k` | `[B, T, D]` | `[B, t_total, D]` = 新 `[B,1,D]` 拼上缓存 |
| `v` | `[B, T, D]` | `[B, t_total, D]` |
| `t_total` | = T | = 缓存长度 + 本次新增（本项目每次 +1） |

后续的拆头、注意力分数、softmax 等代码**一行都不用改**，因为它们是按 `t_total` 写的通用代码：

- 拆头时 k/v 用 `t_total` 做 reshape（`vec![b, t_total, self.n_head, head_dim]`），q 仍用 `t`；
- 分数 `scores = q.matmul(&kt).mul_scalar(scale)` 形状 `[B*H, t, t_total]`；
- 因果掩码 mask 是 `[t, t_total]`，广播相加后 `softmax_last_dim()`，最后 `attn.matmul(&v)`。

这就是 KV Cache 优雅的地方：**模型代码零侵入，只把"输入 K/V 的来源"从"当场算"换成"缓存里取"**。

---

## 5. GPT::forward 里的 base 偏移

推理模式下，新 token 的位置不再是"序列内的第 j 个"，而是"全局的第 base + j 个"。`GPT::forward` 这样处理：

```rust
// 2. 位置编码：KV cache 推理时，当前位置从缓存长度开始
let base = kv_cache
    .as_ref()
    .map(|c| c.first().map(|k| k.seq_len()).unwrap_or(0))
    .unwrap_or(0);
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    for j in 0..t {
        positions.push(base + j);
    }
}
...
// 3. 因果掩码：scores 形状 [B*H, T, T_total]，广播 mask [T, T_total]
let t_total = t + base;
let mut mask_data = vec![0.0f32; t * t_total];
for i in 0..t {
    for j in 0..t_total {
        if j > i + base {
            mask_data[i * t_total + j] = f32::NEG_INFINITY;
        }
    }
}
```

| 量 | 全量模式（base=0） | 缓存模式 |
|----|-------------------|---------|
| 位置编码行号 | `j`（0..t） | `base + j`（从缓存长度继续往后数） |
| 掩码总宽 | `t` | `t_total = t + base` |
| 掩码规则 | `j > i` 禁止（只能看自己及之前） | `j > i + base` 禁止（新 token 只能看缓存里的历史 + 自己） |

> 为什么掩码的下界是 `base`：新 token 在全局序列里的下标从 `base` 开始（`i=0` 对应全局 `base`），所以它能看全局 `0..=base`（全是缓存里的历史）+ 自己，不能看 `base+1` 之后（未来）。这和全量模式的因果性完全一致。

训练时 `kv_cache` 传 `None`（`src/train.rs` 里 `model.forward(&x, b, t, None)`），因为训练时权重每步都在变、历史 K/V 没有复用价值，缓存反而白占内存。

---

## 6. generate：cache 模式 vs 全量模式的流程对比

`src/sample.rs` 的 `generate`：

```rust
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
```

两种模式逐项对比：

| | 全量模式（无缓存） | KV cache 模式 |
|---|---|---|
| 首次前向 | `forward(ctx, 1, ctx.len(), None)` | `forward(ctx, 1, ctx.len(), Some(&mut cache))`：同样前向整个 prompt，但**把每层 K/V 顺手存进缓存** |
| 之后每步 | `forward(ctx, 1, ctx.len(), None)`：整个上下文（截断到最近 32 个）重算 | `forward(&ids[ids.len()-1..], 1, 1, Some(&mut cache))`：**只喂最后一个 token**，K/V 从缓存取 |
| 上下文处理 | `ids.len().saturating_sub(block_size)` 截断，窗口可滑动 | 不截断，全量积累在缓存里 |
| 停止条件 | 生成满 `max_new` 个 | 生成满 `max_new` 个，**或缓存长度达到 `block_size`** |
| 每次前向的位置数 | 32（封顶后固定） | 首次 prompt 长度，之后恒为 1 |

用流程图看第一次生成（prompt = "The fox"，7 个 token，max_new=80）：

```
cache 模式：                             全量模式：
───────────                             ───────────
第 1 步：前向 ["The fox"(7个)]          第 1 步：前向 ["The fox"(7个)]
         ↓ 填充缓存（7 个位置）                    ↓ 结果只取最后一行，丢弃其余
         采样出第 1 个新 token
第 2 步：前向 [最新 1 个]               第 2 步：前向 ["The fox" + 1个]（8 个）
         ↓ 缓存 = 8 个位置                       ↓ 又从头算了一遍 7 个历史 K/V
第 3 步：前向 [最新 1 个]               第 3 步：前向 [9 个]
         ↓ 缓存 = 9 个位置                       ↓ 重复劳动越来越多
...                                    ...
第 26 步：前向 [最新 1 个]              第 80 步：前向 [窗口内 32 个]
         ↓ 缓存 = 32 个位置                      ↓ 80 个新 token 全部生成
         采样出第 26 个新 token
第 27 步：开头检查缓存 = 32 ≥ block_size → 停止
```

> 取 logits 的细节：`generate` 只取输出张量的**最后一行**（`logits.data()[n - v..]`，v = vocab_size）。全量模式算了一整段序列，但生成只需要最后一个位置的预测——前半部分的计算全部是"浪费"；缓存模式干脆只算最后一行需要的东西，正是这种浪费的反面。

---

## 7. 为什么输出分布不变

这是 KV Cache 正确性的核心论证，分三步：

1. **K/V 值相同**：推理时权重冻结。缓存里存的 K/V，与全量模式下同一批输入算出来的 K/V，数值**逐位相同**（都是同一份代码算的）。
2. **注意力计算相同**：新位置的注意力输出 = `softmax(Q_k·Kᵀ/√d + mask) · V`。其中 K、V 是"全部历史"（缓存模式从缓存取、全量模式当场算），数值相同；Q 是新位置的投影，也相同。
3. **softmax 结果相同**：mask 规则一致（第 5 节已证），同一组分数经过同样的 softmax → 同样的概率分布 → 同样的采样分布。

用一句话概括：**缓存只是把"这次算完就扔"的中间结果留了下来，计算路径和数值一个都没变，所以分布必然不变。**

代码注释也点明了这一点（`src/main.rs`）：

```rust
println!("\n  （KV cache 只改计算方式、不改生成分布，两者应高度一致）");
```

> 演示里的"验证"其实是间接的：demo 用了两个不同的 prompt（"Once upon a" vs "The fox"）和同一个 rng 序列，所以两段输出文本不同是正常的。想严格验证"分布一致"，应该**用相同 prompt + 相同 rng 种子**分别跑 `use_kv_cache=false` 和 `true`，对比逐 token 输出是否逐位一致——这是动手练习 1。

---

## 8. 上下文达到 block_size 后停止生成

真实输出里能直接看到这个机制（第 16 课跑出来的）：

```
  —— 生成 2（temperature=0.8, top-k=10, top-p=0.9, 带 KV cache）——
  The fox the hidden garden, whe li
```

数一下："The fox" = 7 个 token，续写 ` the hidden garden, whe li` = 26 个 token，**输出共 7 + 26 = 33 个字符**。逐迭代看缓存怎么涨的：

| 迭代 | 前向内容 | 前向之后缓存长度 | 采样出新 token |
|------|---------|----------------|----------------|
| 第 1 步 | 整个 prompt（7 个） | 7 | 第 1 个 |
| 第 2 步 | 最新 1 个 | 8 | 第 2 个 |
| ... | ... | ... | ... |
| 第 26 步 | 最新 1 个 | 32 | 第 26 个 |
| 第 27 步 | ——（循环开头检查） | 32 ≥ 32 → **break** | —— |

也就是说，生成到第 26 个新 token 后，下一次循环开头检查：

```rust
if use_kv_cache && cache[0].seq_len() >= block_size {
    break;
}
```

此时缓存（prompt 7 个 + 已前向的 25 个新 token）恰好等于 32 = block_size，直接跳出——所以 `max_new=80` 根本没跑完，输出戛然而止。注意最后采样的第 26 个 token 甚至**没有参与前向、也没进缓存**（它只是被采样并 push 进 `ids`，下一次循环就 break 了）。

**为什么必须停？** 三个原因，都指向同一个根：

| 原因 | 说明 |
|------|------|
| 位置编码表不够长 | `pos_emb` 是 `[block_size=32, D]` 的常数表，第 33 个位置没有编码可用 |
| 缓存无法"截断" | 全量模式可以用 `ids.len().saturating_sub(block_size)` 把窗口滑到最近 32 个 token；而 `KVCache` 只会 append、不会丢弃最早的位置（当前实现没有"弹掉开头"的操作） |
| 因果掩码越界 | `t_total = t + base` 一旦超过 block_size，位置编码 gather_rows 就会越界 |

对比全量模式的生成 1：prompt "Once upon a" = 12 个 token，每步窗口都滑到最近 32 个，所以 80 个新 token 全部生成完（`Once upon a to his friend, the wise old owl. One day, Red found a wold of colors and turned`）。

> 真实 LLM 的 KV cache 比这复杂得多：支持"滑动窗口 + 丢弃最旧块"（如 Mistral 的 sliding window）、对缓存做量化压缩等。本项目的 `KVCache` 是最简版——**只拼不丢**，因此一旦填满就必须停止。把"丢了也能继续"留作动手练习 5。

---

## 9. 动手练习

1. **严格验证分布不变**：在 `demo_gpt` 里用**相同的 prompt**（如都传 `"The fox"`）和**相同的 rng** 分别调 `generate(..., false, ...)` 与 `generate(..., true, ...)`，对比逐 token 输出是否一致。
2. **打印缓存形状**：在 `MultiHeadAttention::forward` 的 `cache.append` 之后加一行 `println!("cache seq_len = {}", cache.seq_len());`，观察它从 7 一路涨到 32 的过程。
3. **把 `break` 条件去掉**：临时注释掉 `generate` 里的 `if use_kv_cache && cache[0].seq_len() >= block_size { break; }`，运行看会发生什么——体会位置编码表 `[32, D]` 的下界约束。
4. **对比计算量**：全量模式第 k 步前向 k 个位置、缓存模式每步只前向 1 个位置。对 `block_size=32`、`max_new=80`，估算两种模式累计前向的位置总数各是多少。
5. **（进阶）给 KVCache 加"截断"**：仿照全量模式的窗口滑动，给 `KVCache` 加一个 `truncate(len)` 方法（把 `k.data()` 裁到最近 `len` 个位置再包回张量），并在 `generate` 的缓存分支里每步调用它，让缓存模式也能像全量模式一样持续生成——对比改动前后的输出。

---

## 10. 本课总结

- 逐 token 生成时，历史位置的 K/V 每步都在被重复计算——全量模式累计 O(T²)，这是 KV Cache 要消灭的浪费
- `KVCache` = 每层一份的 `(k, v)` 张量（`[1, T, D]`），`append` 纯数据拼接、`seq_len` 读已缓存长度
- `MultiHeadAttention` 用缓存后只有 K/V 变长，Q 只算新位置，后续代码零改动；`GPT::forward` 用 `base` 修正位置编码与因果掩码
- 流程对比：首次前向整个 prompt 填缓存 → 之后每步只前向 1 个 token；全量模式则是每步重算整个窗口
- 分布不变的原因：缓存里的 K/V 与全量模式算出的数值相同，注意力、softmax 计算路径一致
- 缓存模式只拼不丢，上下文达到 `block_size=32` 必须停止（位置编码表到头 + 无法截断历史），真实输出里生成 2 止步于 32 个 token

- 下一课：换掉正弦位置编码，用 RoPE（旋转位置编码）让位置信息融入注意力计算。
