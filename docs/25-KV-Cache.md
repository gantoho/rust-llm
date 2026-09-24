# 第 25 课：KV Cache —— 让逐 token 生成不再重复计算

> 代码位置：[src/attention.rs](../src/attention.rs)（`KVCache` / `MultiHeadAttention`）
> 代码位置：[src/model.rs](../src/model.rs)（`Transformer::forward` / `forward_core`）
> 代码位置：[src/sample.rs](../src/sample.rs)（`generate`）
> 演示入口：[src/main.rs](../src/main.rs)（演示 3：生成 1 / 生成 2 / 生成 3）

---

## 1. 本课要搞懂的问题

1. 推理时为什么"历史 token 的 K/V"会被一遍遍重复计算？
2. `KVCache` 的数据结构长什么样？`append` / `seq_len` / `positions_seen` 各做了什么？
3. cache 模式与全量模式在 `generate` 里的流程有什么不同（首次 vs 之后每步）？
4. 用了缓存之后，为什么生成的概率分布和全量模式**完全一样**？
5. 生成长度超过 `block_size` 之后，缓存靠什么继续生成下去？

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

`src/attention.rs` 里的定义：

```rust
pub struct KVCache {
    k: Shared<Vec<f32>>, // 行优先 [1, T, D] 展平
    v: Shared<Vec<f32>>,
    len: usize,    // 当前**保留**的位置数 T（滑动窗口下不会超过 window）
    seen: usize,   // 累计喂进来的位置总数，只增不减（RoPE 绝对位置基准）
    window: usize, // 保留上限；0 = 不丢弃
    d: usize,      // 隐藏维 D，第一次 append 时确定
}
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `k` / `v` | `Shared<Vec<f32>>` | 该层**保留**的 Key / Value，行优先展平成 `[1, T, D]` |
| `len` | `usize` | 当前保留的位置数 T（不再靠 `shape()[1]` 反推） |
| `seen` | `usize` | 累计喂进来的位置数，**只增不减**；新 token 的 RoPE 绝对位置基准（第 8 节） |
| `window` | `usize` | 保留上限，超出就丢最旧的行；`0` = 不丢弃 |
| `d` | `usize` | 隐藏维 D，第一次 `append` 时从 `k.shape()[2]` 确定 |

> 为什么不用 `Option<Tensor>` 直接存张量？因为推理时"追加一个新位置"如果走「取旧数据 → 拼新数据 → 重新包成张量」，
> 每步都要把整段历史复制一遍，T 步累计 O(T²) 拷贝。把 `Vec<f32>` 放进 `Shared`（`Arc<Mutex>`）里就地 `extend`，
> 历史数据一次都不用动。`Shared` 是为了让 `Transformer::forward` 这类只读者也能共享同一块缓存。

注意：**每个注意力层各有一个 `KVCache`**。`Transformer::new_kv_cache` 返回 `Vec<KVCache>`，长度 = `n_layer`，并且每个都带上 `block_size` 大小的滑动窗口：

```rust
pub fn new_kv_cache(&self) -> Vec<KVCache> {
    let window = self.cfg.block_size;
    (0..self.cfg.n_layer).map(|_| KVCache::with_window(window)).collect()
}
```

### 3.1 append：把新 K/V 追加到缓存尾部，并丢弃超窗的最旧行

```rust
pub fn append(&mut self, k: &Tensor, v: &Tensor) {
    assert_eq!(k.shape(), v.shape(), "K/V 形状必须一致");
    assert_eq!(k.rank(), 3, "K/V 必须为 3D [1, T, D]，实际 {:?}", k.shape());
    let t = k.shape()[1];
    self.d = k.shape()[2];
    self.k.borrow_mut().extend(k.data());
    self.v.borrow_mut().extend(v.data());
    self.len += t;
    self.seen += t;
    // 滑动窗口：丢掉最旧的行，让 len 回到 window（drain 从头删，尾部整体左移）
    if self.window > 0 && self.len > self.window {
        let drop_elems = (self.len - self.window) * self.d;
        self.k.borrow_mut().drain(..drop_elems);
        self.v.borrow_mut().drain(..drop_elems);
        self.len = self.window;
    }
}
```

做的事：

1. 校验 K/V 形状一致且是 3D；
2. 把新 K/V 的数据 `extend` 到各自的 `Vec<f32>` 末尾——**历史数据原地不动**；
3. `len` 与 `seen` 各加上本次新增的位置数；
4. 超过窗口时 `drain` 掉最旧的 `(len - window) * d` 个元素（一次 memmove，不是 O(T²) 的拷贝链）。

> 反面教材（本项目**曾经**的写法）：把缓存存成 `Option<Tensor>`，每次 append 时
> 「取旧数据 → `all.extend(cur.data())` → 重新包成张量」——每步都把整段历史复制一遍，
> T 步累计 O(T²) 拷贝。改成在 `Vec<f32>` 上就地 `extend` 后，历史数据一次都不用动。

> 细节：`cur` 在推理模式下形状是 `[1, 1, D]`（只算 1 个新位置），所以 `len` / `seen` 每次 +1，
> `d` 从 `k.shape()[2]` 取。纯数据追加，推理时无梯度，所以没有走任何 autograd 路径。

### 3.2 seq_len / positions_seen / window / k() / v()

```rust
/// 当前保留的位置数（= 注意力里 K/V 的序列长度）
pub fn seq_len(&self) -> usize { self.len }

/// 累计喂进来的位置总数：新 token 的 RoPE 绝对位置基准（滑动窗口下与 `seq_len` 不同）
pub fn positions_seen(&self) -> usize { self.seen }

/// 保留上限（0 = 不丢弃）
pub fn window(&self) -> usize { self.window }

/// 返回完整缓存张量 [1, T, D]（注意力打分需要读全量历史，这里克隆一次）
pub fn k(&self) -> Tensor {
    Tensor::from_vec(self.k.borrow().clone(), vec![1, self.len, self.d])
}
// v() 同理
```

- `seq_len()` 返回**保留**的位置数。它是注意力里 K/V 的序列长度，也是 `generate` 判断"首次前向"的依据。
- `positions_seen()` 返回**累计喂进来**的位置数，只增不减。它是 RoPE 的绝对位置基准（第 8 节），在滑动窗口生效后 `positions_seen > seq_len`。
- `k()` / `v()` 把展平缓存重新包成 `[1, T, D]` 张量供注意力打分使用。
  这里**仍会克隆一次整段缓存**——因为打分算子是按「拥有所有权的 `Tensor`」写的；
  想再省掉这一步，需要让算子支持借用视图，属于后续优化（见 README 的"还能压的地方"）。

---

## 4. MultiHeadAttention 怎么用缓存

`MultiHeadAttention::forward` 中与缓存相关的部分：

```rust
// 1. 投影得到 Q、K、V（Linear 输出是 2D [B*T, D]，恢复成 3D）
let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
let kv_dim = self.n_kv_head * head_dim;
let k = self.c_k.forward(x).reshape(vec![b, t, kv_dim]); // [B, T, kv_dim]
let v = self.c_v.forward(x).reshape(vec![b, t, kv_dim]);

// 2. RoPE：Q/K 按 head_dim 旋转（GQA 时 K 只有 n_kv_head 个头）；
//    旋转发生在 append 之前，所以缓存里存的是"已旋转的 K"，历史位置直接复用、不再重算
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
let (q, k) = q
    .reshape(vec![b * t, d])
    .rotary_pair(&k.reshape(vec![b * t, kv_dim]), &positions);
let (q, k) = (q.reshape(vec![b, t, d]), k.reshape(vec![b, t, kv_dim]));

// 3. KV cache：把本次新算的 K/V 追加到缓存（超窗自动丢最旧行），再取回保留的全部历史
let (k, v) = match kv_cache {
    Some(cache) => {
        cache.append(&k, &v);
        (cache.k(), cache.v()) // [1, seq_len, kv_dim]
    }
    None => (k, v),
};
let t_total = k.shape()[1];
```

变化只有一处：**K、V 变长**，Q 保持 `[B, T, D]` 不动：

| 变量 | 无缓存 | 有缓存（推理） |
|------|--------|----------------|
| `q` | `[B, T, D]` | `[B, 1, D]`（只算新位置） |
| `k` | `[B, T, kv_dim]` | `[B, t_total, kv_dim]` = 新 `[B,1,kv_dim]` 追加到缓存后的结果 |
| `v` | `[B, T, kv_dim]` | `[B, t_total, kv_dim]` |
| `t_total` | = T | = `cache.seq_len()`（本项目每次 +1，封顶窗口大小） |

> `kv_dim = n_kv_head × head_dim`（第 23 课 GQA）：`n_kv_head == n_head` 时就是标准 MHA 的 `D`。

后续的拆头、注意力分数、softmax 等代码**一行都不用改**，因为它们是按 `t_total` 写的通用代码：

- 拆头时 k/v 用 `t_total` 做 reshape（`vec![b, t_total, self.n_kv_head, head_dim]`，再按 `n_rep` 复制成 `n_head` 个头），q 仍用 `t`；
- 分数 `scores = q.matmul(&kt).mul_scalar(scale)` 形状 `[B*H, t, t_total]`；
- 因果掩码 mask 是 `[t, t_total]`，广播相加后 `softmax_last_dim()`，最后 `attn.matmul(&v)`。

这就是 KV Cache 优雅的地方：**模型代码零侵入，只把"输入 K/V 的来源"从"当场算"换成"缓存里取"**。

---

## 5. 位置与掩码：`seen` 作为 RoPE 基准

推理模式下，新 token 的位置不再是"窗口内的第 j 个"，而是"全局的第 seen + j 个"。`forward_core` 里这样算（`src/model.rs:630-661`）：

```rust
// RoPE 的基准是**绝对位置**：新 token 的绝对位置 = 累计已喂入的位置数 + 窗口内下标 j
let (seen, window) = kv_cache
    .as_ref()
    .and_then(|c| c.first())
    .map(|k| (k.positions_seen(), k.window()))
    .unwrap_or((0, 0));
let base = seen;

// 因果掩码：scores 形状 [B*H, T, T_total]，广播 mask [T, T_total]
// 缓存本次之后保留 min(seen + t, window) 个位置（window = 0 表示不丢弃）
assert!(window == 0 || t <= window, "单次前向的 token 数（{t}）不能超过 KV cache 窗口（{window}）");
let t_total = if window == 0 { seen + t } else { (seen + t).min(window) };
// 缓存里位于"本次新增的第一个 token"左侧（含自身）的位置数
let visible_before = t_total - t;
let mut mask_data = vec![0.0f32; t * t_total];
for i in 0..t {
    for j in 0..t_total {
        if j > i + visible_before {
            mask_data[i * t_total + j] = f32::NEG_INFINITY;
        }
    }
}
```

它有两个用途。

**用途 ①：传给注意力内部的 RoPE**（`src/attention.rs:170-173`）。位置序列在这里生成，`rotary_pair` 据此旋转 Q/K：

```rust
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
```

**用途 ②：构造因果掩码**（上面的代码）。注意掩码宽度是"本次前向之后缓存里会有多少行"，而不是 `seen + t` 这个朴素值——滑动窗口打开时，超过 `window` 的部分会立刻被丢掉，掩码必须与之对齐，否则会对着不存在的列求和。

| 量 | 全量模式（无缓存） | 缓存模式（不超窗） | 缓存模式（本轮触发滑动） |
|----|-------------------|-------------------|------------------------|
| 位置编码行号 | `j`（0..t） | `seen + j` | `seen + j`（继续递增） |
| 掩码总宽 `t_total` | `t` | `seen + t` | `window` |
| `visible_before` | 0 | `seen` | `window - t` |
| 掩码规则 | `j > i` 禁止 | `j > i + seen` 禁止 | `j > i + (window - t)` 禁止 |
| 每行 query 能看多远 | 自己及之前全部 | 自己及之前全部 | 最多 `window` 个 key |

> 为什么掩码的下界是 `visible_before`：新 token 在全局序列里的下标从 `seen` 开始（`i=0` 对应全局 `seen`），所以它能看全局 `0..=seen`（全是缓存里的历史）+ 自己，不能看 `seen+1` 之后（未来）。这和全量模式的因果性完全一致。
>
> 触发滑动时，缓存里最老的行已经是对应"全局 `seen + t - window`"的那一行，所以可看的起点整体右移。

训练时 `kv_cache` 传 `None`，`base = 0`，掩码退化成标准的 `j > i`——因为训练时权重每步都在变、历史 K/V 没有复用价值，缓存反而白占内存。

---

## 6. generate：cache 模式 vs 全量模式的流程对比

`src/sample.rs` 的 `generate`：

```rust
let block_size = model.cfg.block_size;
let mut ids = tokenizer.encode(prompt);
if ids.is_empty() { ids.push(0); }                             // 空 prompt 兜底
let mut cache = use_kv_cache.then(|| model.new_kv_cache());    // 只有用 cache 才分配

for _ in 0..max_new {
    // 只保留最近的 block_size 个 token（两种模式都必须遵守的上下文上限）
    let start = ids.len().saturating_sub(block_size);
    let ctx = &ids[start..];

    // 推理不需要反向：no_grad 下不挂计算图、不分配梯度缓冲
    let logits = crate::tensor::no_grad(|| {
        if use_kv_cache {
            // 首次：缓存为空，把整个 prompt 喂进去（顺便填充缓存）
            // 之后：每步只前向最新 1 个 token，历史 K/V 从缓存取
            let c = cache.as_mut().unwrap();
            if c[0].seq_len() == 0 {
                model.forward(ctx, 1, ctx.len(), Some(c), false)
            } else {
                model.forward(&ids[ids.len() - 1..], 1, 1, Some(c), false)
            }
        } else {
            // 全量模式：每次把整个上下文重新算一遍（慢，但没有 cache 内存）
            model.forward(ctx, 1, ctx.len(), None, false)
        }
    });

    // 取最后一个位置的 logits
    let v = model.cfg.vocab_size;
    let n = logits.numel();
    let last_row = &logits.data()[n - v..];
    // 重复惩罚的回看窗口（见第 15 课第 7 节）；窗口 0 = 不惩罚
    let recent = if opts.repetition_window == 0 {
        &[][..]
    } else {
        &ids[ids.len().saturating_sub(opts.repetition_window)..]
    };
    let next = sample_token(last_row, opts, recent, rng);
    ids.push(next);
}
```

两种模式逐项对比：

| | 全量模式（无缓存） | KV cache 模式 |
|---|---|---|
| 首次前向 | `forward(ctx, 1, ctx.len(), None, false)` | `forward(ctx, 1, ctx.len(), Some(c), false)`：同样前向整个 prompt，但**把每层 K/V 顺手存进缓存** |
| 之后每步 | `forward(ctx, 1, ctx.len(), None, false)`：整个上下文（截断到最近 32 个）重算 | `forward(&ids[ids.len()-1..], 1, 1, Some(c), false)`：**只喂最后一个 token**，K/V 从缓存取 |
| 上下文处理 | 每步 `ids.len().saturating_sub(block_size)` 截断，窗口滑到最近 32 个 | 缓存在 `append` 里自己丢最旧行（第 8 节），每步 +1 行 |
| 停止条件 | 生成满 `max_new` 个 | 生成满 `max_new` 个（缓存容量不再是上限） |
| 每次前向的位置数 | 32（封顶后固定） | 首次 prompt 长度，之后恒为 1 |
| 生成 80 个新 token 的累计前向位置数 | 441 + 32×59 = 2329（前 21 步按 11..31 递增） | 11 + 80 = 91 |

用流程图看 demo 的生成 1 / 生成 2（prompt = "Once upon a"，11 个 token，max_new=80，block_size=32）：

```
cache 模式：                             全量模式：
───────────                             ───────────
第 1 步：前向 ["Once upon a"(11个)]     第 1 步：前向 ["Once upon a"(11个)]
         ↓ 填充缓存（11 个位置）                   ↓ 结果只取最后一行，丢弃其余
         采样出第 1 个新 token
第 2 步：前向 [最新 1 个]               第 2 步：前向 ["Once upon a" + 1个]（12 个）
         ↓ 缓存 = 12 个位置                      ↓ 又从头算了一遍 11 个历史 K/V
第 3 步：前向 [最新 1 个]               第 3 步：前向 [13 个]
         ↓ 缓存 = 13 个位置                      ↓ 重复劳动越来越多
...                                    ...
第 22 步：前向 [最新 1 个]              第 22 步：前向 [窗口内 32 个]
         ↓ 缓存 = 32 个位置（封顶）              ↓ 从这里起每步都是固定 32 个位置
第 23 步：前向 [最新 1 个]              第 23 步：前向 [窗口内 32 个]
         ↓ 缓存仍 32 行，seen = 33              ↓ 窗口整体右移一格
         （丢弃全局位置 0，位置编码继续用 33）
...
第 80 步：缓存仍 32 行，两种模式都生成满 80 个新 token
```

> 取 logits 的细节：`generate` 只取输出张量的**最后一行**（`logits.data()[n - v..]`，v = vocab_size）。全量模式算了一整段序列，但生成只需要最后一个位置的预测——前半部分的计算全部是"浪费"；缓存模式干脆只算最后一行需要的东西，正是这种浪费的反面。

---

## 7. 为什么输出分布不变

**在窗口内**这是严格结论，分三步：

1. **K/V 值相同**：推理时权重冻结。缓存里存的 K/V，与全量模式下同一批输入算出来的 K/V，数值**逐位相同**（都是同一份代码算的，缓存模式下也确实一次都没重算过）。
2. **注意力计算相同**：新位置的注意力输出 = `softmax(Q_k·Kᵀ/√d + mask) · V`。其中 K、V 是"全部历史"（缓存模式从缓存取、全量模式当场算），数值相同；Q 是新位置的投影，也相同。
3. **softmax 结果相同**：mask 规则一致（第 5 节已证），同一组分数经过同样的 softmax → 同样的概率分布 → 同样的采样分布。

用一句话概括：**缓存只是把"这次算完就扔"的中间结果留了下来，计算路径和数值一个都没变，所以分布必然不变。**

demo 现在**自己做这个验证**（`src/main.rs`）：同 prompt、同种子、生成长度控制在窗口内，两种模式必须逐 token 完全相等。

```rust
println!(
    "\n  窗口内等价自检（prompt {prompt_len} + 新 {in_window} = {} ≤ block_size {}）：\
     缓存模式与全量模式逐 token {}",
    prompt_len + in_window,
    model.cfg.block_size,
    if a == b { "完全一致 ✓" } else { "不一致 ✗" }
);
```

> 两次生成必须用**相同 prompt + 相同种子**（demo 里都是 `"Once upon a"` 和 `Rng::new(2024)`）。早先的 demo 用了两个不同 prompt（"Once upon a" vs "The fox"）再共用同一个 rng 序列，两段输出不同只是采样不同，**证明不了任何事**，属于反面教材。
>
> 为什么要把生成长度**控制在窗口内**才做严格断言？因为一旦触发滑动窗口，"增量推理"与"截断重算"就**不是同一个函数**了（下一节给出证明与反例），此时两者不相等不是 bug，而是推理语义的正常结果。窗口内的严格相等由 `sample::tests::test_kv_cache_generate_matches_full` 守住，窗口外的机制等价由 `model::tests::test_kv_cache_sliding_window_matches_full_window_forward` 守住。

---

## 8. 滑动窗口：超出 block_size 之后继续生成

早期实现里 `generate` 会在缓存填满时 `break`，导致同一个 prompt 加不加 KV cache 会得到**长度不同**的输出（而且当时只能靠一行 `[warn]` 提示，用户依然会困惑）。现在缓存自带滑动窗口，这个问题从根上消失了：**生成长度只由 `max_new` 决定，与缓存容量无关**。

机制全在 `KVCache::append`（第 3.1 节）与 `forward_core`（第 5 节）里：

| 位置 | 行为 |
|------|------|
| `append` | `len` 超过 `window` 时 `drain` 掉最旧的 `(len - window)` 行 |
| `seq_len()` | 封顶在 `window`（注意力里 K/V 的序列长度就是它） |
| `positions_seen()` | **继续累加**，不受窗口影响（RoPE 的绝对位置基准） |
| `forward_core` | 掩码宽度取 `min(seen + t, window)`，与丢弃后的实际行数对齐 |

以 `block_size = 32`、prompt 11 个 token 为例，缓存长度随生成步数变化：

| 迭代 | 前向内容 | `seq_len()` | `positions_seen()` | 采样出新 token |
|------|---------|------------|-------------------|----------------|
| 第 1 步 | 整个 prompt（11 个） | 11 | 11 | 第 1 个 |
| 第 2 步 | 最新 1 个 | 12 | 12 | 第 2 个 |
| ... | ... | ... | ... | ... |
| 第 22 步 | 最新 1 个 | 32 | 32 | 第 22 个 |
| 第 23 步 | 最新 1 个 | **32**（丢掉全局位置 0） | **33** | 第 23 个 |
| 第 80 步 | 最新 1 个 | 32 | 80 | 第 80 个 |

两个计数器必须分开，原因在 RoPE：**缓存里存的是已旋转的 K**，每条 K 都带着它产生时的绝对位置。如果为了"把窗口下标重新编号成 0..32"而改动位置基准，就得把缓存里每一行重新旋转一遍——那是每步 O(T) 的额外工作，正好把缓存省下的计算量又还回去。让绝对位置一路递增、缓存行原地不动，是唯一自洽的做法；而 RoPE 打分只依赖**相对距离**，同一 query 与同一 key 之间的相对距离在两种编号下完全一致，所以递增编号不会改变任何分数。

### 窗口外两条路径为什么不再逐位相等

| 层数 | 增量（缓存 + 滑动窗口） vs 截断重算 | 原因 |
|------|-----------------------------------|------|
| 1 层 | **严格等价** | 第 0 层的 K/V 只由 token 嵌入决定、与上下文无关，两条路径看到的是同一组 K/V，只差一个整体位置平移（RoPE 打分只依赖相对距离） |
| ≥ 2 层 | 不相等，且**不是 bug** | 增量路径里位置 p 的第 l 层 K/V 是 p 当时算出来的（当时能看到它自己的完整窗口）；"截断重算"会把窗口内每个位置在第 0 层可见的上下文一并砍掉，深层 K/V 随之不同 |

最后一行值得强调：**带 KV cache 的增量推理才是真实 LLM 的标准推断语义**（所有生产级推理引擎都是这么跑的），"截断重算"只是全量前向在上下文超限时的一种近似。所以两条路径在窗口外不相等时，以增量路径为准。这也正是 demo 把严格等价断言限制在窗口内的原因。

单层下的严格等价由 `model::tests::test_kv_cache_sliding_window_matches_full_window_forward` 验证：`block_size = 8`、序列长度 24（滑动 2 次），断言缓存增量推理与"只取最近 8 个 token 全量前向"的最后一行 logits 偏差 < 1e-3。

### 与真实长上下文方案的关系

| 方案 | 思路 | 本项目 |
|------|------|--------|
| Sliding Window Attention（Mistral） | 只让 query 看最近 W 个 token，缓存只保留 W 行 | **已实现**，就是本节这套 `window` |
| Attention Sink / StreamingLLM | 滑动窗口之外**永远保留最前面几行**（首 token 是注意力的"锚点"，丢了会让整条分布失稳） | **已实现**：`KvCacheOpts::sink`（丢弃时跳过最前面的 `sink` 行），CLI 开关 `--kv-sink` |
| KV cache 量化（KIVI 等） | 把缓存的 K/V 压到 int8/int4，显存换精度 | **已实现**：K 逐通道、V 逐 token 的 int8/int4 缓存（CLI 开关 `--kv-bits`），原理见第 33 课 |

---

## 9. 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. **在窗口内验证"完全相等"**：demo 的窗口内自检已经是 `a == b` 的严格比较，把它换成 `assert_eq!`、再把 prompt 与 `max_new` 调到刚好占满窗口（11 + 21 = 32），确认仍然相等——边界值最容易暴露 off-by-one。
2. **打印两个计数器**：在 `MultiHeadAttention::forward` 的 `cache.append` 之后加一行 `println!("seq_len = {}, seen = {}", cache.seq_len(), cache.positions_seen());`，跑 demo 的生成 2，观察一个在第 23 步停住、另一个继续涨到 80。
3. **关掉滑动窗口看后果**：把 `Transformer::new_kv_cache` 里的 `window` 改成 `0`（不丢弃），重新跑 demo 的生成 2——缓存长度会一路上涨到 91，模型被迫在 RoPE 外推区（训练只见过 `0..32`）做注意力，生成质量会明显劣化。这是"窗口不是可选项"最直观的证据。
4. **对比计算量**：对 `block_size=32`、prompt 11、`max_new=80`，按第 6 节的表分别估算两种模式累计前向的位置数，再和 demo 里两种模式的实测耗时比一比。
5. **（进阶）Attention Sink 与量化的叠加**：sink 已实现在 `KvCacheOpts::sink`（CLI `--kv-sink`），
   缓存量化已实现在 `--kv-bits`。把两个开关一起打开跑长文本生成，对比"只开 sink""只量化""都开"
   三种配置下首 token 附近的分布稳定性——sink 抗的是"锚点丢失"，量化引入的是数值噪声，两者叠加会不会互相放大。

---

## 10. 本课总结

- 逐 token 生成时，历史位置的 K/V 每步都在被重复计算——全量模式累计 O(T²)，这是 KV Cache 要消灭的浪费
- `KVCache` = 每层一份的 `Vec<f32>` 缓存（行优先展平的 `[1, T, D]`，外加 `len` / `seen` / `window` / `d`），
  `append` 就地 extend（不复制历史）+ 超窗 `drain` 丢最旧行，`seq_len` 读保留行数、`positions_seen` 读累计位置数
- `MultiHeadAttention` 用缓存后只有 K/V 变长，Q 只算新位置，后续代码零改动；`Transformer::forward_core` 用 `seen` 修正位置编码、并用 `min(seen + t, window)` 定掩码宽度
- 流程对比：首次前向整个 prompt 填缓存 → 之后每步只前向 1 个 token；全量模式则是每步重算整个窗口
- 分布不变的原因：缓存里的 K/V 与全量模式算出的数值相同，注意力、softmax 计算路径一致；demo 用**同 prompt + 同种子**自验证，单元测试守住"窗口内逐 token 相同"
- 滑动窗口让生成长度不再受缓存容量限制：`seq_len` 封顶 `window`、`positions_seen` 继续累加，缓存里存的是已旋转的 K，绝对位置递增才能避免每步重旋转整段缓存
- 窗口外"增量推理"与"截断重算"不是同一个函数（单层严格等价、多层不等价且以增量路径为准），这是标准推理语义，不是实现缺陷

- 下一课：换掉正弦位置编码，用 RoPE（旋转位置编码）让位置信息融入注意力计算。
