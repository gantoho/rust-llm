# 第 25 课：KV Cache —— 让逐 token 生成不再重复计算

> 代码位置：[src/attention.rs](../src/attention.rs)（`KVCache` / `MultiHeadAttention`）
> 代码位置：[src/model.rs](../src/model.rs)（`GPT::forward` / `forward_core`）
> 代码位置：[src/sample.rs](../src/sample.rs)（`generate`）
> 演示入口：[src/main.rs](../src/main.rs)（演示 3：生成 1 / 生成 2）

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

`src/attention.rs` 里的定义：

```rust
/// KV 缓存（第 25 课）：
/// 生成第 N 个 token 时，前 N-1 个 token 的 K、V 不需要重算。
/// 把每个注意力层的 K、V 存起来，每次只算新 token 的 K、V 并追加。
///
/// 内部直接持有 `Vec<f32>` 缓存，append 时只把新块 extend 到末尾，
/// 避免"每步克隆整段历史再拼接"的 O(T²) 开销。
pub struct KVCache {
    k: Rc<RefCell<Vec<f32>>>, // 行优先 [1, T, D] 展平
    v: Rc<RefCell<Vec<f32>>>,
    len: usize, // 已缓存的位置数 T
    d: usize,   // 隐藏维 D，第一次 append 时确定
}
```

| 字段 | 类型 | 含义 |
|------|------|------|
| `k` / `v` | `Rc<RefCell<Vec<f32>>>` | 该层已缓存的所有位置的 Key / Value，行优先展平成 `[1, T, D]` |
| `len` | `usize` | 已缓存的位置数 T（不再靠 `shape()[1]` 反推） |
| `d` | `usize` | 隐藏维 D，第一次 `append` 时从 `k.shape()[2]` 确定 |

> 为什么不用 `Option<Tensor>` 直接存张量？因为推理时"追加一个新位置"如果走「取旧数据 → 拼新数据 → 重新包成张量」，
> 每步都要把整段历史复制一遍，T 步累计 O(T²) 拷贝。把 `Vec<f32>` 放进 `RefCell` 里就地 `extend`，
> 历史数据一次都不用动。`Rc` 是为了让 `GPT::forward` 这类只读者也能共享同一块缓存。

注意：**每个注意力层各有一个 `KVCache`**。`GPT::new_kv_cache` 返回 `Vec<KVCache>`，长度 = `n_layer`：

```rust
pub fn new_kv_cache(&self) -> Vec<KVCache> {
    (0..self.cfg.n_layer).map(|_| KVCache::new()).collect()
}
```

### 3.1 append：把新 K/V 追加到缓存尾部（当前实现）

```rust
/// 把新的 k/v 追加到缓存末尾（只拷贝新块，不复制历史数据）
pub fn append(&mut self, k: &Tensor, v: &Tensor) {
    assert_eq!(k.shape(), v.shape(), "K/V 形状必须一致");
    assert_eq!(k.rank(), 3, "K/V 必须为 3D [1, T, D]，实际 {:?}", k.shape());
    self.d = k.shape()[2];
    self.k.borrow_mut().extend(k.data());
    self.v.borrow_mut().extend(v.data());
    self.len += k.shape()[1];
}
```

做的事：

1. 校验 K/V 形状一致且是 3D；
2. 把新 K/V 的数据 `extend` 到各自的 `Vec<f32>` 末尾——**历史数据原地不动**；
3. `len` 加上本次新增的位置数。

> 反面教材（本项目**曾经**的写法）：把缓存存成 `Option<Tensor>`，每次 append 时
> 「取旧数据 → `all.extend(cur.data())` → 重新包成张量」——每步都把整段历史复制一遍，
> T 步累计 O(T²) 拷贝。改成在 `Vec<f32>` 上就地 `extend` 后，历史数据一次都不用动。

> 细节：`cur` 在推理模式下形状是 `[1, 1, D]`（只算 1 个新位置），所以 `len` 每次 +1，
> `d` 从 `k.shape()[2]` 取。纯数据追加，推理时无梯度，所以没有走任何 autograd 路径。

### 3.2 seq_len 与 k()/v()

```rust
/// 当前已缓存的位置数
pub fn seq_len(&self) -> usize {
    self.len
}

/// 返回完整缓存张量 [1, T, D]（注意力打分需要读全量历史，这里克隆一次）
pub fn k(&self) -> Tensor {
    Tensor::from_vec(self.k.borrow().clone(), vec![1, self.len, self.d])
}
// v() 同理
```

- `seq_len()` 直接返回 `len`。它有两个用途（后面会看到）：一是 `forward_core` 用它算位置偏移 `base`；二是 `generate` 用它判断要不要停止。
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

// 3. KV cache：把本次新算的 K/V 追加到缓存，再取回全量历史
let (k, v) = match kv_cache {
    Some(cache) => {
        cache.append(&k, &v);
        (cache.k(), cache.v()) // [1, t_total, kv_dim]
    }
    None => (k, v),
};
let t_total = k.shape()[1];
```

变化只有一处：**K、V 变长**，Q 保持 `[B, T, D]` 不动：

| 变量 | 无缓存 | 有缓存（推理） |
|------|--------|----------------|
| `q` | `[B, T, D]` | `[B, 1, D]`（只算新位置） |
| `k` | `[B, T, kv_dim]` | `[B, t_total, kv_dim]` = 新 `[B,1,kv_dim]` 追加到缓存 |
| `v` | `[B, T, kv_dim]` | `[B, t_total, kv_dim]` |
| `t_total` | = T | = 缓存长度 + 本次新增（本项目每次 +1） |

> `kv_dim = n_kv_head × head_dim`（第 23 课 GQA）：`n_kv_head == n_head` 时就是标准 MHA 的 `D`。

后续的拆头、注意力分数、softmax 等代码**一行都不用改**，因为它们是按 `t_total` 写的通用代码：

- 拆头时 k/v 用 `t_total` 做 reshape（`vec![b, t_total, self.n_kv_head, head_dim]`，再按 `n_rep` 复制成 `n_head` 个头），q 仍用 `t`；
- 分数 `scores = q.matmul(&kt).mul_scalar(scale)` 形状 `[B*H, t, t_total]`；
- 因果掩码 mask 是 `[t, t_total]`，广播相加后 `softmax_last_dim()`，最后 `attn.matmul(&v)`。

这就是 KV Cache 优雅的地方：**模型代码零侵入，只把"输入 K/V 的来源"从"当场算"换成"缓存里取"**。

---

## 5. `base` 偏移：位置与掩码

推理模式下，新 token 的位置不再是"序列内的第 j 个"，而是"全局的第 base + j 个"。`base` 在 `GPT::forward_core` 里算出来（`src/model.rs:528-531`）：

```rust
// base = KV cache 模式下已缓存的位置数：新 token 的绝对位置 = base + 窗口内下标 j
let base = kv_cache
    .as_ref()
    .map(|c| c.first().map(|k| k.seq_len()).unwrap_or(0))
    .unwrap_or(0);
```

它随后有两个用途。

**用途 ①：传给注意力内部的 RoPE**（`src/attention.rs:135-138`）。位置序列在这里生成，`rotary_pair` 据此旋转 Q/K：

```rust
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
```

**用途 ②：构造因果掩码**（`forward_core` 内）：

```rust
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

训练时 `kv_cache` 传 `None`（`src/train.rs:201` 里 `model.forward(&x, b, t, None, false)`），因为训练时权重每步都在变、历史 K/V 没有复用价值，缓存反而白占内存。

---

## 6. generate：cache 模式 vs 全量模式的流程对比

`src/sample.rs` 的 `generate`：

```rust
let block_size = model.cfg.block_size;
let mut ids = tokenizer.encode(prompt);
if ids.is_empty() { ids.push(0); }                             // 空 prompt 兜底
let mut cache = use_kv_cache.then(|| model.new_kv_cache());    // 只有用 cache 才分配
let mut hit_window_limit = false;

for _ in 0..max_new {
    // KV cache 模式：上下文总长达到 block_size 就停（缓存无法像全量模式那样截断历史）
    if cache.as_ref().is_some_and(|c| c[0].seq_len() >= block_size) {
        hit_window_limit = true;
        break;
    }
    // 只保留最近的 block_size 个 token（全量模式需要）
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

// 窗口写满导致提前结束时，向 stderr 打印 [warn]（见第 8 节）
if hit_window_limit { /* eprintln!(...) */ }
```

两种模式逐项对比：

| | 全量模式（无缓存） | KV cache 模式 |
|---|---|---|
| 首次前向 | `forward(ctx, 1, ctx.len(), None, false)` | `forward(ctx, 1, ctx.len(), Some(c), false)`：同样前向整个 prompt，但**把每层 K/V 顺手存进缓存** |
| 之后每步 | `forward(ctx, 1, ctx.len(), None, false)`：整个上下文（截断到最近 32 个）重算 | `forward(&ids[ids.len()-1..], 1, 1, Some(c), false)`：**只喂最后一个 token**，K/V 从缓存取 |
| 上下文处理 | `ids.len().saturating_sub(block_size)` 截断，窗口可滑动 | 不截断，全量积累在缓存里 |
| 停止条件 | 生成满 `max_new` 个 | 生成满 `max_new` 个，**或缓存长度达到 `block_size`** |
| 每次前向的位置数 | 32（封顶后固定） | 首次 prompt 长度，之后恒为 1 |

用流程图看 demo 的生成 1 / 生成 2（prompt = "Once upon a"，11 个 token，max_new=80）：

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
第 22 步：前向 [最新 1 个]              第 80 步：前向 [窗口内 32 个]
         ↓ 缓存 = 32 个位置                      ↓ 80 个新 token 全部生成
         采样出第 22 个新 token
第 23 步：开头检查缓存 = 32 ≥ block_size → 停止
```

（第 1～22 步两个模式的输出逐 token 相同，所以 demo 里生成 2 是生成 1 的前缀；差别只在第 23 步 cache 模式停下、全量模式继续滑窗口。）

> 取 logits 的细节：`generate` 只取输出张量的**最后一行**（`logits.data()[n - v..]`，v = vocab_size）。全量模式算了一整段序列，但生成只需要最后一个位置的预测——前半部分的计算全部是"浪费"；缓存模式干脆只算最后一行需要的东西，正是这种浪费的反面。

---

## 7. 为什么输出分布不变

这是 KV Cache 正确性的核心论证，分三步：

1. **K/V 值相同**：推理时权重冻结。缓存里存的 K/V，与全量模式下同一批输入算出来的 K/V，数值**逐位相同**（都是同一份代码算的）。
2. **注意力计算相同**：新位置的注意力输出 = `softmax(Q_k·Kᵀ/√d + mask) · V`。其中 K、V 是"全部历史"（缓存模式从缓存取、全量模式当场算），数值相同；Q 是新位置的投影，也相同。
3. **softmax 结果相同**：mask 规则一致（第 5 节已证），同一组分数经过同样的 softmax → 同样的概率分布 → 同样的采样分布。

用一句话概括：**缓存只是把"这次算完就扔"的中间结果留了下来，计算路径和数值一个都没变，所以分布必然不变。**

代码注释也点明了这一点（`src/main.rs`），而且 demo 现在**自己做这个验证**：

```rust
let consistent = out_full.starts_with(&out_kv);
println!(
    "\n  KV cache 只改计算方式、不改生成分布：cache 输出应恰为全量输出的前缀 —— {}",
    if consistent { "一致 ✓" } else { "不一致 ✗" }
);
```

> 两次生成必须用**相同 prompt + 相同种子**（demo 里都是 `"Once upon a"` 和 `Rng::new(2024)`）。早先的 demo 用了两个不同 prompt（"Once upon a" vs "The fox"）再共用同一个 rng 序列，两段输出不同只是采样不同，**证明不了任何事**，属于反面教材。
>
> 为什么断言是"**前缀**"而不是"完全相等"？因为 cache 模式受 `block_size` 限制会提前结束（下一节），输出比全量模式短；但在它生成的那段里必须逐 token 相同。窗口内的严格相等由单元测试 `sample::tests::test_kv_cache_generate_matches_full` 守住，超窗口的"前缀"关系由 `test_kv_cache_output_is_prefix_of_full_beyond_window` 守住。

---

## 8. 上下文达到 block_size 后停止生成

真实输出里能直接看到这个机制（`cargo run --release -- demo`）：

```
  —— 生成 2（同 prompt、同种子，带 KV cache）——
  Once upon a time in a small villa

  KV cache 只改计算方式、不改生成分布：cache 输出应恰为全量输出的前缀 —— 一致 ✓
```

同时在 stderr 会打印一行警告，把"提前结束"这件事**说出来**（见本节末）：

```
[warn] KV cache 窗口已满（block_size=32），生成在 33 个 token 处提前结束；需要更长输出请缩短 prompt，或加 --no-kv-cache 改用全量前向（滑动窗口可继续生成）
```

数一下："Once upon a" = 11 个 token，续写 ` time in a small villa` = 22 个 token，**输出共 11 + 22 = 33 个字符**。逐迭代看缓存怎么涨的：

| 迭代 | 前向内容 | 前向之后缓存长度 | 采样出新 token |
|------|---------|----------------|----------------|
| 第 1 步 | 整个 prompt（11 个） | 11 | 第 1 个 |
| 第 2 步 | 最新 1 个 | 12 | 第 2 个 |
| ... | ... | ... | ... |
| 第 22 步 | 最新 1 个 | 32 | 第 22 个 |
| 第 23 步 | ——（循环开头检查） | 32 ≥ 32 → **break** | —— |

也就是说，生成到第 22 个新 token 后，下一次循环开头检查：

```rust
if cache.as_ref().is_some_and(|c| c[0].seq_len() >= block_size) {
    hit_window_limit = true;
    break;
}
```

此时缓存（prompt 11 个 + 已前向的 21 个新 token）恰好等于 32 = block_size，直接跳出——所以 `max_new=80` 根本没跑完，输出戛然而止。注意最后采样的第 22 个 token 甚至**没有参与前向、也没进缓存**（它只是被采样并 push 进 `ids`，下一次循环就 break 了）。

> 这种"静默变短"曾经是个坑：同一个 prompt 加不加 KV cache 会得到**不同长度**的输出，而使用者看不出原因。现在 `generate` 会在窗口写满时向 stderr 打印 `[warn]`，把限制显式暴露出来（`hit_window_limit` 标志 + `eprintln!`）。

**为什么必须停？** 三个原因，都指向同一个根：

| 原因 | 说明 |
|------|------|
| 训练长度之外是外推区 | RoPE 对任意位置都能算出旋转角（没有"位置表"可言），但模型训练时只见过位置 `0..32`，超出后是**外推区**（第 20 课讲过），注意力分数可能畸变，输出质量断崖下跌 |
| 缓存无法"截断" | 全量模式可以用 `ids.len().saturating_sub(block_size)` 把窗口滑到最近 32 个 token；而 `KVCache` 只会 append、不会丢弃最早的位置（当前实现没有"弹掉开头"的操作） |
| 上下文窗口硬上限 | `block_size` 是模型的设计上下文长度（每个训练样本最长 32 个位置），`generate` 用 `cache[0].seq_len() >= block_size` 把生成长度锁在训练见过的最长窗口内，不越界 |

对比全量模式的生成 1：同一个 prompt "Once upon a" = 11 个 token，每步窗口都滑到最近 32 个，所以 80 个新 token 全部生成完（`Once upon a time in a small village, there lived a curious little fox named Red. Every morn`）。

> 真实 LLM 的 KV cache 比这复杂得多：支持"滑动窗口 + 丢弃最旧块"（如 Mistral 的 sliding window）、对缓存做量化压缩等。本项目的 `KVCache` 是最简版——**只拼不丢**，因此一旦填满就必须停止。把"丢了也能继续"留作动手练习 5。

---

## 9. 动手练习

1. **在窗口内验证"完全相等"**：把 demo 的生成 2 改成 `max_new=20`（prompt 11 + 20 = 31 ≤ 32，全程不碰窗口上限），此时两次输出应**完全相等**而不只是前缀——把 `starts_with` 换成 `assert_eq!` 验证一下。
2. **打印缓存形状**：在 `MultiHeadAttention::forward` 的 `cache.append` 之后加一行 `println!("cache seq_len = {}", cache.seq_len());`，观察它从 11 一路涨到 32 的过程。
3. **把 `break` 条件去掉**：临时注释掉 `generate` 里的 `if cache.as_ref().is_some_and(|c| c[0].seq_len() >= block_size) { break; }`，运行看会发生什么——体会 RoPE 外推区（位置远超训练见过的 `0..32`）对生成质量的影响。
4. **对比计算量**：全量模式第 k 步前向 k 个位置、缓存模式每步只前向 1 个位置。对 `block_size=32`、`max_new=80`，估算两种模式累计前向的位置总数各是多少。
5. **（进阶）给 KVCache 加"截断"**：仿照全量模式的窗口滑动，给 `KVCache` 加一个 `truncate(keep: usize)` 方法（把 `Vec<f32>` 前部多出来的 `(len - keep) * d` 个元素 `drain` 掉，并同步更新 `len`），并在 `generate` 的缓存分支里每步调用它，让缓存模式也能像全量模式一样持续生成——对比改动前后的输出。

---

## 10. 本课总结

- 逐 token 生成时，历史位置的 K/V 每步都在被重复计算——全量模式累计 O(T²)，这是 KV Cache 要消灭的浪费
- `KVCache` = 每层一份的 `Vec<f32>` 缓存（行优先展平的 `[1, T, D]`，外加 `len` / `d`），
  `append` 就地 extend（不复制历史）、`seq_len` 读 `len`、`k()`/`v()` 包成张量（会克隆一次）
- `MultiHeadAttention` 用缓存后只有 K/V 变长，Q 只算新位置，后续代码零改动；`GPT::forward_core` 用 `base` 修正位置编码与因果掩码
- 流程对比：首次前向整个 prompt 填缓存 → 之后每步只前向 1 个 token；全量模式则是每步重算整个窗口
- 分布不变的原因：缓存里的 K/V 与全量模式算出的数值相同，注意力、softmax 计算路径一致；demo 用**同 prompt + 同种子**自验证，单元测试守住"窗口内完全一致 / 超窗口仅是前缀"
- 缓存模式只拼不丢，上下文达到 `block_size=32` 必须停止（训练长度外是 RoPE 外推区 + 缓存无法截断历史），真实输出里生成 2 止步于 33 个字符，并向 stderr 打印 `[warn]` 提示提前结束

- 下一课：换掉正弦位置编码，用 RoPE（旋转位置编码）让位置信息融入注意力计算。
