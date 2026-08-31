# 第 12 课：完整 GPT 模型 —— 把积木拼成能预测下一个词的模型

> 代码位置：[src/model.rs](file:///d:/Code/Rust/llm_from_scratch/src/model.rs)（`GPTConfig` / `GPT` / `TransformerBlock` / `MultiHeadAttention`）
> 代码位置：[src/layers.rs](file:///d:/Code/Rust/llm_from_scratch/src/layers.rs)（`Embedding` / `Linear` / `LayerNorm` / `gelu`）
> 演示入口：[src/main.rs](file:///d:/Code/Rust/llm_from_scratch/src/main.rs)（演示 3：训练小 GPT 并生成文本）

---

## 1. 本课要搞懂的问题

1. 一个完整的 GPT 由哪几大块组成？各自负责什么？
2. Transformer Block 内部的数据是怎么流动的？pre-norm 到底"pre"在哪？
3. 输入一串 token id，经过 `GPT::forward` 后形状怎么一步步变成 logits？
4. `GPTConfig::tiny` 里每个数字（64 / 4 / 2 / 32）分别代表什么？

---

## 2. GPT 架构总览

把前几课的积木全部拼起来，就是完整的 GPT：

```
              ┌───────────────────────────────┐
 token id     │  tok_emb: Embedding [V, D]    │  ← 每个 token 查表成向量
              └───────────────────────────────┘
              ┌───────────────────────────────┐
              │  pos_emb: 正弦位置编码 [T, D]   │  ← 常数，告诉模型位置（第 11 课）
              └───────────────────────────────┘
              │        x = tok + pos_emb       │
              ┌───────────────────────────────┐
              │  blocks: N 层 Transformer      │  ← 注意力找相关性 + MLP 加工信息
              │  Block（本课第 3 节）           │
              └───────────────────────────────┘
              ┌───────────────────────────────┐
              │  ln_f: 最终 LayerNorm          │  ← 输出前再归一化一次
              └───────────────────────────────┘
              ┌───────────────────────────────┐
              │  lm_head: Linear [D, V]        │  ← 映射成对每个词的打分
              └───────────────────────────────┘
                      logits [B*T, V]（预测下一个 token）
```

对应的 Rust 结构体（`src/model.rs`）：

```rust
/// 完整的 GPT 模型
pub struct GPT {
    pub cfg: GPTConfig,
    tok_emb: Embedding,
    pos_emb: Tensor, // 正弦位置编码 [block_size, D]（常数，不参与训练）
    blocks: Vec<TransformerBlock>,
    ln_f: LayerNorm,
    lm_head: Linear, // [D, vocab]
}
```

| 字段 | 类型 | 作用 | 对应积木（第几课） |
|------|------|------|------------------|
| `cfg` | `GPTConfig` | 保存模型配置（维度、层数……） | —— |
| `tok_emb` | `Embedding` | token id → 向量，查表 `[V, D]` | 第 12 课（`layers.rs`） |
| `pos_emb` | `Tensor` | 正弦位置编码，`[block_size, D]`，**不参与训练** | 第 11 课 |
| `blocks` | `Vec<TransformerBlock>` | N 层 Transformer Block，重复堆叠 | 第 9-11 课 |
| `ln_f` | `LayerNorm` | 输出前的最后归一化 | 第 11 课 |
| `lm_head` | `Linear` | `[D, V]`，把向量映射成对每个词的分数 | 第 5 课 |

在 `GPT::new` 里把它们创建出来：

```rust
pub fn new(cfg: GPTConfig, rng: &mut Rng) -> Self {
    let n_embd = cfg.n_embd;
    let vocab_size = cfg.vocab_size;
    let pos_emb = Tensor::from_vec(
        sinusoidal_positions(cfg.block_size, cfg.n_embd),
        vec![cfg.block_size, cfg.n_embd],
    );
    let blocks = (0..cfg.n_layer).map(|_| TransformerBlock::new(&cfg, rng)).collect();
    GPT {
        cfg,
        tok_emb: Embedding::new(vocab_size, n_embd, rng),
        pos_emb,
        blocks,
        ln_f: LayerNorm::new(n_embd, 1e-5),
        lm_head: Linear::new(n_embd, vocab_size, rng),
    }
}
```

注意 `blocks` 是 `(0..cfg.n_layer).map(...).collect()`——**同一份代码复制 N 份**，层数完全由配置决定，想加深网络只要改一个数字。

---

## 3. TransformerBlock 内部结构

### 3.1 pre-norm 结构

每个 Block 内部是"两个子层 + 各自残差"的 pre-norm 结构（第 11 课讲过 pre-norm 的好处）：

```
x ──► LN1 ──► MultiHeadAttention ──► (+残差) ──► LN2 ──► MLP(GELU) ──► (+残差) ──► 输出
      │                                ▲          │                        ▲
      └────────────────────────────────┘          └────────────────────────┘
            加回输入 x（残差连接）                         加回输入 x（残差连接）
```

对应的结构体与 forward：

```rust
/// Transformer Block（第 11 课）
///
/// 结构（GPT-2 风格，pre-norm）：
///   x -> LayerNorm -> Attention -> 残差 +
///   x -> LayerNorm -> MLP(GELU)  -> 残差 +
struct TransformerBlock {
    ln1: LayerNorm,
    attn: MultiHeadAttention,
    ln2: LayerNorm,
    mlp_linear1: Linear, // [D, 4D]
    mlp_linear2: Linear, // [4D, D]
}

impl TransformerBlock {
    fn forward(&self, x: &Tensor, mask: &Tensor, kv_cache: Option<&mut KVCache>) -> Tensor {
        // 注意力子层 + 残差连接
        let h = self.attn.forward(&self.ln1.forward(x), mask, kv_cache);
        let x = x.add(&h);
        // 前馈子层 + 残差连接
        let h = self.ln2.forward(&x);
        let h = gelu(&self.mlp_linear1.forward(&h));
        let h = self.mlp_linear2.forward(&h);
        x.add(&h)
    }
}
```

逐行拆解：

| 代码 | 在做什么 | 对应结构 |
|------|---------|---------|
| `self.ln1.forward(x)` | 先归一化（pre-norm 的"pre"） | `x → LN1` |
| `self.attn.forward(..., mask, kv_cache)` | 多头注意力（第 10 课），`mask` 保证只能看过去 | `→ Attention` |
| `let x = x.add(&h);` | 注意力输出 + 输入（残差连接） | `+ 残差` |
| `self.ln2.forward(&x)` | 再归一化 | `→ LN2` |
| `self.mlp_linear1.forward(&h)` | 升维到 4D：`[D] → [4D]`（每层 MLP 把维度先放大 4 倍） | `→ MLP 第一层` |
| `gelu(...)` | GELU 激活（第 5 课，GPT 系列默认激活，比 ReLU 平滑） | `→ 激活` |
| `self.mlp_linear2.forward(&h)` | 降维回 D：`[4D] → [D]` | `→ MLP 第二层` |
| `x.add(&h)` | 第二个残差连接 | `+ 残差` |

### 3.2 为什么 MLP 要"先升维再降维"

前馈子层 `MLP(D → 4D → D)` 是 Transformer 里唯一"逐位置"加工信息的部件（注意力负责"跨位置"交换信息，MLP 负责"在每个位置上"独立加工）：

```
MLP:  D → 4D → D
       │      │
   先放大   再压缩
  （特征空间更丰富，    （恢复原维度，方便残差相加）
   更容易学非线性）
```

`4D` 是 GPT-2 论文里的惯例比例。MLP 和注意力形成互补：**注意力负责"找谁相关"，MLP 负责"想清楚该怎么表达"。**

---

## 4. GPT::forward 数据流

### 4.1 输入与输出

```rust
pub fn forward(
    &self,
    idx: &[usize],          // [B*T] 展平的 token id
    b: usize,               // batch 大小
    t: usize,               // 序列长度
    mut kv_cache: Option<&mut Vec<KVCache>>,  // 推理缓存（第 18 课，训练时传 None）
) -> Tensor {
```

- **输入**：`idx` 是一维数组，长度 `b * t`，里面是 token id（比如 `[2, 5, 9, 3]` 表示一条样本 4 个词）。
- **输出**：`logits`，形状 `[B*T, vocab_size]`——每个位置对"下一个词是谁"的打分。

### 4.2 五步数据流

| 步骤 | 代码 | 形状变化 |
|------|------|---------|
| 0. 输入 | `idx: &[usize]` | `[B*T]` |
| 1. token embedding | `self.tok_emb.forward(idx).reshape(vec![b, t, d])` | `[B*T] → [B*T, D] → [B, T, D]` |
| 2. 位置编码 | `self.pos_emb.gather_rows(&positions).reshape(vec![b, t, d])` | `[B, T, D]`（与 tok 同形状） |
| —— 相加 | `let x = tok.add(&pos_emb);` | `[B, T, D]`（第 11 课：语义+位置） |
| 3. 构造因果掩码 | `Tensor::from_vec(mask_data, vec![t, t_total])` | `[T, T_total]`（未来位置为 -inf） |
| 4. 逐层 Transformer Block | `x = block.forward(&x, &mask, cache);` | `[B, T, D]` → `[B, T, D]`（层内拆头又合并，形状不变） |
| 5. 最终归一化 + 输出头 | `ln_f.forward(&x)` 然后 `reshape(vec![b * t, d])`，`lm_head.forward(&x)` | `[B, T, D] → [B*T, D] → [B*T, V]` |

完整代码：

```rust
let d = self.cfg.n_embd;
assert_eq!(idx.len(), b * t, "输入 id 数量必须等于 b*t");

// 1. token embedding
let tok = self.tok_emb.forward(idx).reshape(vec![b, t, d]);

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
let pos_emb = self.pos_emb.gather_rows(&positions).reshape(vec![b, t, d]);
let x = tok.add(&pos_emb);

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
let mask = Tensor::from_vec(mask_data, vec![t, t_total]);

// 4. 逐层过 Transformer Block
let mut x = x;
for (i, block) in self.blocks.iter().enumerate() {
    let cache = kv_cache.as_mut().map(|c| &mut c[i]);
    x = block.forward(&x, &mask, cache);
}

// 5. 最终归一化 + 输出头
let x = self.ln_f.forward(&x);
let x = x.reshape(vec![b * t, d]);
self.lm_head.forward(&x)
```

几个容易忽略的细节：

1. **`gather_rows` 取位置行**：`pos_emb` 是 `[block_size, D]`，`positions` 里存的是每个 token 的位置（`base + j`），`gather_rows` 按行取出对应的位置向量（第 1 课学的查表）。
2. **因果掩码的构造**：`j > i + base` 的位置设为 `-inf`。也就是说第 i 个 token 只能看到"它自己和它前面的"（含 KV cache 里的历史位置），未来位置在 softmax 后概率为 0——保证模型只能预测下一个词、不能偷看答案。
3. **最终形状**：`lm_head` 是 `Linear [D, V]`，把每个位置的 `[D]` 向量打分成 `[V]` 个词的分数，所以输出是 `[B*T, V]`。

### 4.3 一张形状变化总表

以 `b=1, t=4, n_embd=64, vocab=100` 为例：

| 阶段 | 形状 | 含义 |
|------|------|------|
| `idx` | `[4]` | 4 个 token id |
| `tok_emb.forward(idx)` | `[4, 64]` → reshape `[1, 4, 64]` | 每个 token 的语义向量 |
| `pos_emb` | `[1, 4, 64]` | 每个 token 的位置向量 |
| `x = tok + pos_emb` | `[1, 4, 64]` | 语义 + 位置 |
| 注意力内部 | scores `[4, 4]`（1 个 batch、4 头时 `[4, 4, 4]`） | 相关性打分 |
| 过完 2 个 Block | `[1, 4, 64]` | 形状不变，信息被加工 |
| `ln_f` 后 | `[1, 4, 64]` | 归一化 |
| reshape | `[4, 64]` | 展平 B、T 两维 |
| `lm_head` | `[4, 100]` | 每个位置对 100 个词的分数 = logits |

> 全程只有两处形状变化：`tok_emb` 之后从 `[B*T, D]` 变成 `[B, T, D]`（为了层内处理），`lm_head` 之前从 `[B, T, D]` 变回 `[B*T, D]`（因为 `Linear` 把 3D 输入自动展平，输出再还原）。中间的注意力拆头/合头都在 Block 内部完成，外部看到的一直是 `[B, T, D]`。

---

## 5. GPTConfig::tiny 配置解读

```rust
/// 一个小配置，适合学习演示
pub fn tiny(vocab_size: usize) -> Self {
    GPTConfig {
        vocab_size,
        n_embd: 64,
        n_head: 4,
        n_layer: 2,
        block_size: 32,
    }
}
```

| 字段 | 值 | 含义 | 影响 |
|------|----|------|------|
| `vocab_size` | 由调用者传入 | 词表大小（有多少种 token） | 决定 `tok_emb` 表行数和 `lm_head` 输出维度 |
| `n_embd` | 64 | 隐藏维度 D：每个 token 的向量长度 | 所有层的宽度，模型"容量"的核心 |
| `n_head` | 4 | 注意力头数 | `head_dim = D / H = 64 / 4 = 16`，每个头在 16 维子空间找相关性 |
| `n_layer` | 2 | Transformer Block 层数 | 网络深度（原版 GPT-2 是 12~48 层） |
| `block_size` | 32 | 最大上下文长度 | 位置编码表 `[32, 64]` 的宽度；序列超过 32 个 token 就没位置编码可用 |

### 5.1 由配置推导出的关键数字

- **每个头处理多少维**：`head_dim = n_embd / n_head = 64 / 4 = 16`（`MultiHeadAttention::forward` 里有断言 `head_dim * n_head == d`，配置必须能整除）。
- **每层的形状**：注意力 4 个 Linear 都是 `[64, 64]`；MLP 是 `[64, 256]` 和 `[256, 64]`（4 倍升维）。
- **位置编码表**：`[block_size=32, n_embd=64]`，一共 32×64 个常数。

### 5.2 粗略参数量估算

设词表大小 = V，参数量约：

| 部件 | 参数量 | tiny（V=100）时 |
|------|--------|----------------|
| `tok_emb` | V × 64 | 6,400 |
| 每个 Block 注意力（q/k/v/proj） | 4 × (64×64 + 64) | 16,640 |
| 每个 Block MLP（两个 Linear） | 64×256+256 + 256×64+64 | 33,088 |
| 每个 Block 两个 LayerNorm | 2 × (64+64) | 256 |
| 每层合计 | —— | ≈ 49,984 |
| 2 层 Block | —— | ≈ 99,968 |
| `ln_f` | 64 + 64 | 128 |
| `lm_head` | 64×V + V | 6,500 |
| **总计** | —— | **≈ 113,000（约 11 万参数）** |

> 真实 GPT-2 Small 有 1.17 亿参数（n_embd=768、n_layer=12、n_head=12）。我们的 tiny 把它缩小了约 1000 倍，纯粹是为了**能在普通 CPU 上几秒钟跑一步训练**，把原理讲清楚。

---

## 6. 动手练习

1. **手推数据流**：设 `vocab=50, b=2, t=3`，用 tiny 配置，写出 `GPT::forward` 里每一步张量的形状（从 `idx` 到 `logits`），对照第 4.3 节的表检查。
2. **改配置**：自己加一个 `GPTConfig::small`，比如 `n_embd=128, n_head=8, n_layer=4, block_size=64`。注意 `head_dim = 128/8 = 16` 仍成立；再按 5.2 节的表估一下参数量。
3. **验证 logits 形状**：在 `main.rs` 演示 3 里，`model.forward(...)` 之后加一行打印 `logits.shape()`，确认是 `[B*T, V]`。
4. **看 Block 内部分工**：把 `mlp_linear1` 的维度改成 `2 * cfg.n_embd`（2 倍而不是 4 倍），训练看 loss 变化——体会 MLP 宽度对模型能力的影响。
5. **思考**：为什么 `lm_head` 和 `tok_emb` 可以"共享权重"（很多实现里是同一个矩阵的转置）？我们的代码里是两份独立参数，如果共享会省多少参数？（提示：`[V, D]` 和 `[D, V]` 恰好互为转置。）

---

## 7. 本课总结

- GPT 由 **token embedding + 正弦位置编码 + N 层 Transformer Block + 最终 LayerNorm + lm_head** 五大部分组成
- Transformer Block 是 **pre-norm** 结构：`LN → Attention → 残差`，再 `LN → MLP(GELU) → 残差`；MLP 做 `D → 4D → D` 的升维降维
- `GPT::forward` 数据流：`idx [B*T] → [B, T, D] → ... → [B*T, V]`，只有进出输出头时形状变化，层内形状始终 `[B, T, D]`
- 因果掩码保证"只能看过去"，位置编码保证"知道在哪"
- `tiny` 配置：`n_embd=64`、`n_head=4`（head_dim=16）、`n_layer=2`、`block_size=32`，约 11 万参数，CPU 上几分钟就能跑一轮训练

- 下一课：写训练循环（前向 → 算损失 → 反向传播 → 更新参数），让这个 GPT 真的学会生成文本！
