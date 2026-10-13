# 第 49 课：CSA + HCA 混合注意力 —— 把长序列注意力分层压缩

> **本课已落地为可运行代码**：[`src/csa.rs`](../src/csa.rs)（三层覆盖 + 覆盖不变式 + 解析账本），
> 接入 [`src/attention.rs`](../src/attention.rs) 的注意力分派与 [`src/model.rs`](../src/model.rs) 的 `TransformerBlock`，
> 演示命令 [`src/cmd/experiments.rs`](../src/cmd/experiments.rs) 的 `cmd_csa`。
> 配置开关：`model.csa_block` / `csa_top_k` / `hca_block`（默认全 `0` = 关闭）/ 命令 `cargo run -- csa`。
>
> 与第 23 课（MLA）/ 第 25 课（Flash Attention）的关系：Flash Attention 解决的是**怎么把
> 全量注意力算得省内存**，但它每个 query 仍然要看 `O(T)` 个 key；MLA 压的是 KV 的**维度**。
> 本课压的是 KV 的**个数**——把"每个 query 能看多少 key"从 `T` 降到 `m + k + T/(m·h)`，
> 这是长上下文（几十万 token）里真正卡脖子的那一维。

---

## 1. 本课要搞懂的问题

1. 长序列注意力的瓶颈到底是"内存"还是"算力"？为什么 Flash Attention 不够？
2. 为什么"把序列按块做均值"能压缩？压缩之后一个 query 还能不能看到**全部历史**？
3. 最重要的一个问题：一个压缩块**什么时候**才允许被看见？为什么是"块末位置 ≤ i"而不是"块起点 ≤ i"？
4. 三层（局部 / 细压缩 / 粗压缩）各自解决什么？为什么它们能并进**一次** softmax？
5. 压缩换来了什么、代价是什么？（本课的账本：KV 字节 + 每 query 槽位数）

---

## 2. 为什么要"压缩"而不是"更快的全量注意力"

Flash Attention 的在线 softmax 让中间矩阵 `P` 不落地，显存从 `O(T²)` 降到 `O(T)`。
但**计算量没变**：每个 query 仍要跟 `T` 个 key 打分，总 FLOPs 还是 `O(T²·d)`。
当 `T = 256K` 时，即便显存够，算力与访存也扛不住。

要真正压下去，只有一条路：**让每个 query 不必看全部 key**。这就是稀疏 / 压缩注意力。
本课采用的分层做法与 DeepSeek 系列的长上下文思路一致：

```text
   Level 0 局部全分辨率：最近 m 个位置，一个不漏          —— 保证"眼前看得清"
   Level 1 细压缩（CSA）：已完成的 m 长块做块内均值压缩，
                          再按 q·Kc 打分做 top-k 稀疏选择 —— 保证"远处有重点"
   Level 2 粗压缩（HCA）：把 Level 1 的块再压 hca_block 倍，
                          稠密注意力，提供廉价全局覆盖     —— 保证"再远也不瞎"
```

三层覆盖的位置**合并**起来，才是这个 query 真正"看到"的上下文。

---

## 3. 块均值压缩：CSA 的基石

把序列按 `m` 切成块，每块用**块内均值**当它的代表向量：

```text
  K_c[j] = (1/m) · Σ_{s=0}^{m-1} K[j·m + s]        // 第 j 个块的代表 key
  V_c[j] = (1/m) · Σ_{s=0}^{m-1} V[j·m + s]        // 第 j 个块的代表 value
```

块均值是一个**线性**算子（求和后乘 `1/m`），所以它对 `K` / `V` 是**(反)可微的**——
`block_mean` 的梯度会平均地摊回块内每个原始位置。这条性质很重要：CSA 不是"先离线压好、
再当常数用"，而是就长在计算图里、端到端训练。

### 3.1 关键判据：块**末位置** ≤ i 才算完成

这是本课最容易写错、也最危险的一处。query `i` 只能看**已经完全落在它之前**的块：

```text
  B_i = floor((i+1) / m)          // 严格正确的"已完成块数"
  可见块 = 第 0..B_i-1 块          // 覆盖原始位置 [0, B_i·m)
```

**绝对不是** `块起点 ≤ i`。举个例子，`m = 4`、`i = 2`：

- 第 0 块的起点是 `0 ≤ 2`，但它的成员是位置 `0,1,2,3`——位置 `3` 还没生成。
- 若按"块起点 ≤ i"判定，就把未来的 token `3` 平均进了 `K_c[0]`，query `i=2` 于是"偷看"到了位置 3。
- 训练时这会让 loss 虚假地变好，推理时又根本复现不了——是所有因果注意力里最典型的信息泄漏。

正确的 `B_i = floor((i+1)/m)` 在 `i = 2, m = 4` 时给出 `B_i = 0`：第 0 块不可见，因为位置 3 还没到。
到 `i = 3` 时 `B_i = 1`，第 0 块才可见。这条判据由 [`fine_blocks`](../src/csa.rs) 实现，并被
`covered_positions` 与三个因果性单测死死钉住。

### 3.2 局部窗口补上"最近的一截"

只压缩是不够的——块均值会把最近 `m-1` 个位置的信息糊掉，而"眼前"恰恰最需要清晰。
所以 Level 0 用**全分辨率**的滑动窗口覆盖最近 `m` 个位置 `[i-m+1, i]`。

细压缩覆盖 `[0, B_i·m)`，局部覆盖 `[i-m+1, i]`。二者**必然相接**：

```text
  B_i·m = floor((i+1)/m)·m > i + 1 - m     ⇒     两块区间重叠或相邻
```

所以并集恰好是 `[0, i]`，一个不多、一个不少。这就是**覆盖不变式**，由
[`covered_positions`](../src/csa.rs) 计算并在 `csa` 子命令第一节里对 `m ∈ {2,4,8,16}`、
`i ∈ [0, T)` 全部核对。

---

## 4. HCA：给"再远"补一层廉价全局覆盖

CSA 的 top-k 稀疏有个缺点：被 top-k 丢掉的远距离块就**彻底看不到了**。
HCA（Hierarchical Compressed Attention）在细块之上**再平均 `hca_block` 倍**，
得到一个更粗的表示，然后对它做**稠密**（不稀疏）注意力：

```text
  K_c2[g] = (1/h) · Σ_{t=0}^{h-1} K_c[g·h + t]      // 粗块 = h 个细块的平均
```

粗块数量只有 `floor(T / (m·h))`，非常少，所以"稠密看一遍"的代价很低。
它带来的是**冗余覆盖**：粗块的成员细块必然也是已完成的，因此不会引入任何新位置——
覆盖不变式不被破坏，但 top-k 漏掉的远距离信息有了兜底。

三层合并后的每 query 槽位数：

```text
  slots = m              // 局部窗口
        + k              // 细压缩保留的块数（top_k，0 = 全部已完成块）
        + floor(T/(m·h))  // 粗压缩块数（hca_block = h ≥ 2 时才 > 0）
```

---

## 5. 三层怎么并进**一次** softmax

朴素做法是分三次注意力再加权合并，但那要引入额外的合并权重、也不等价于一次归一化。
本课的做法是：**三层只是把不同范围的 key 放进同一行分数**，拼接后**一次 softmax**：

```text
  scores  = [ q·K_loc (m 个) | q·Kc_sel (k 个) | q·Kc2 (T/(m·h) 个) ]     // [g, t, S]
  mask    = [ 局部掩码      | 细压缩掩码      | 粗压缩掩码 ]              // 同形、逐行同序
  P       = masked_softmax(scores, mask)                                  // 一次归一化
  out     = P · [V_loc | Vc_sel | Vc2]                                    // 加权求和
```

这样做的好处：

1. **语义干净**：等价于"在并集 `[0, i]` 上做一次普通因果注意力"，只是把并集拆成三段、各自用合适的粒度表示。
2. **实现省事**：一次 softmax 的反向就是标准 softmax 反向，不用为三个分支各写一套合并逻辑。
3. **数值稳定**：复用张量库已有的 `masked_softmax`（减去行最大值再 exp）。

⚠️ **坑**：拼接后的掩码必须与拼接后的分数**逐行同序**。第一版把局部掩码和压缩掩码各自
按"整块"拼起来（`[全部局部][全部压缩]`）再 `reshape` 成 `[g,t,S]`，结果每行的掩码与分数错位，
某些行被整行 `-inf` 淹没（那时候还会直接算成 `NaN`；现在 `masked_softmax` 对整行全屏蔽的行
会输出全 0 而不是 `NaN`，但**掩码错位依旧会让那一行丢光真正的 key、结果照样是错的**，
只是不再以 NaN 的形式报警）。正确做法是**先算好总槽数 `S`，再把每层的掩码
写到该行的对应偏移** `row + 0 / row + m / row + m + k`。这条已作为注释写死在
[`attention`](../src/csa.rs) 里。

---

## 6. 本仓库实现要点

| 位置 | 内容 |
|------|------|
| [`src/csa.rs`](../src/csa.rs) `CsaOpts` | 纯数据开关（`block` / `top_k` / `hca_block`），`From<&TransformerConfig>` 从配置读入；`derive(Default)` 即"全关" |
| [`src/csa.rs`](../src/csa.rs) `fine_blocks` / `coarse_blocks` / `covered_positions` | 因果判据与覆盖不变式的**整数版**实现，单测与 `csa` 子命令共用；把"因果性"从几百行张量代码里抽出来单独核对 |
| [`src/csa.rs`](../src/csa.rs) `select_topk` | 分数降序取前 k，**同分时小下标优先**，保证可复现 |
| [`src/csa.rs`](../src/csa.rs) `block_mean` / `expand_gqa` / `gather_seq` / `concat_last` | 张量辅助：块均值压缩、GQA 头展开、按索引取行、沿最后一维拼接（`Tensor::external` 自定义反向） |
| [`src/csa.rs`](../src/csa.rs) `attention` | 三层前向：局部窗口 → 块均值 → top-k → HCA → `concat_last` 拼分数 → 一次 `masked_softmax` → 加权值 |
| [`src/attention.rs`](../src/attention.rs) | `MultiHeadAttention` 新增 `csa: CsaOpts` 字段；`attend` 按 `use_csa` 分派到 `crate::csa::attention` 或原 `Tensor::flash_attention` |
| [`src/model.rs`](../src/model.rs) | `TransformerConfig` 新增 `csa_block` / `csa_top_k` / `hca_block`；`TransformerBlock::new` 把配置写进 `a.csa` |
| [`src/cmd/experiments.rs`](../src/cmd/experiments.rs) `cmd_csa` | `csa` 子命令：覆盖不变式 → KV 字节账本 → top-k 稀疏算力账本 |

**两个实现细节**：

- `sd`（用于 top-k 选择的稠密块分数）包在 `crate::tensor::no_grad` 里。它只用来**选块**、
  从不进入最终输出/反向；若不包，它会在 tape 上留下永远不可达的节点，每步堆积、内存只涨不降。
- GQA 下 KV 头要按 `n_rep = n_head / n_kv_head` 展开成 Q 头的粒度再做注意力；梯度经
  `gather_rows` 的反向自动折回共享的 KV 头（单测 `test_gqa_kv_gradient_accumulates_across_heads` 验证）。

### 默认路径逐位不变

- 默认 `csa_block = 0`（`CsaOpts::enabled()` 为假）：`attend` 走原 `Tensor::flash_attention`，
  与加本课之前**逐位一致**；
- CSA 是**纯数据开关**，`TransformerBlock::new` 只是把配置写进 `a.csa`，**不构造任何参数、
  不消耗任何随机数**——所以同种子下开关前后的参数名与参数值逐位相同（单测
  `test_csa_changes_training_same_seed` 断言参数表一致、前向改变、数值有限）；
- 只对**无 KV cache 的全量前向**（训练 / 预填）生效；带 cache 的自回归解码回退原路径
  （`use_csa = self.csa.enabled() && kv_cache.is_none()`）。

### 代码地图

| 位置 | 内容 |
|------|------|
| [`src/csa.rs`](../src/csa.rs) | `CsaOpts`、因果判据（`fine_blocks` / `coarse_blocks` / `covered_positions`）、`select_topk`、张量辅助、三层 `attention`、字节 / 算力账本 |
| [`src/attention.rs`](../src/attention.rs) | `attend` 里按 `use_csa` 在 CSA 与 `flash_attention` 之间分派 |
| [`src/model.rs`](../src/model.rs) | `TransformerConfig` 的三个开关字段与 `TransformerBlock` 的接线 |
| [`src/cmd/experiments.rs`](../src/cmd/experiments.rs) | `cmd_csa` 的三节账本打印 |

---

## 7. 运行与验证

```bash
# CSA 单测（覆盖不变式 / 因果性 / 块均值 / top-k / GQA / 稠密等价 / 梯度有限差分 / 账本）
cargo test csa

# 同种子对照：csa_block = 0 vs 4 走完整前向
cargo test csa_changes_training_same_seed

# 端到端账本：三节演示（覆盖不变式 → KV 字节 → 稀疏算力）
cargo run --release -- csa
```

配置示例：

```jsonc
{
  "model": {
    "csa_block": 16,   // 局部窗口 = 压缩块宽 m = 16（≥ 2 才启用；默认 0 关闭）
    "csa_top_k": 8,    // 细压缩每 query 保留 8 个已完成块（0 = 稠密）
    "hca_block": 4     // 粗压缩在细块之上再平均 4 倍（< 2 = 关闭 HCA）
  }
}
```

### 7.1 实测账本（`cargo run --release -- csa`，默认参数 `T=512`、`n_kv=8`、`hd=64`、`hca=4`、`top_k=8`）

**第一节：覆盖不变式**（`m ∈ {2,4,8,16}`、HCA 开 / 关，检查全部 `T` 个 query）

| m | HCA | 结论 |
|---|-----|------|
| 2 / 4 / 8 / 16 | 关 / 开 | 并集 == `[0,i]` ✓（全部通过） |

**第二节：KV 字节账本**（f32，K + V 两份；全量缓存 2.00 MB）

| 块宽 m | 全量字节 | 压缩后字节 | 压缩比 |
|--------|---------|-----------|--------|
| 2 | 2,097,152 | 1,056,768 | 1.98x |
| 4 | 2,097,152 | 540,672 | 3.88x |
| 8 | 2,097,152 | 294,912 | 7.11x |
| 16 | 2,097,152 | 196,608 | **10.67x** |
| 32 | 2,097,152 | 196,608 | **10.67x** |
| 64 | 2,097,152 | 294,912 | 7.11x |

压缩缓存 = `2·(m + T/m)·n_kv·hd·4`，长度约 `m + T/m`，在 `m ≈ √T` 处取最小。
本题 `√512 ≈ 22.6`，实测最优在 `m = 16~32` 之间（同为 10.67x）。
**注意**：这是**存储口径**——同样 `T` 个历史，缓存从 `T` 行降到约 `m + T/m` 行。
但 `m` 接近 `T` 时局部窗口本身就等于全长，压缩无从谈起（`m = T` 时账本反而更大），
所以压缩只在 `m ≪ T` 时成立。

**第三节：top-k 稀疏算力账本**（每 query 参与打分的槽位数；全量每 query 131,072 FLOPs）

| m | top-k | 局部 | 细压缩 | 粗压缩 | 总槽位 | 稀疏度 | 算力比 |
|---|-------|------|--------|--------|--------|--------|--------|
| 4 | 8 | 4 | 8 | 32 | 44 | 0.086 | 11.64x |
| 8 | 8 | 8 | 8 | 16 | 32 | 0.063 | **16.00x** |
| 16 | 8 | 16 | 8 | 8 | 32 | 0.063 | **16.00x** |
| 32 | 8 | 32 | 8 | 4 | 44 | 0.086 | 11.64x |

`m` 调大能省细压缩块数、但局部窗口变长；`top-k` 调小能省算力、但更依赖块均值选得准。
两者都要与覆盖不变式一起看：省下来的槽位必须真的落在 `[0, i]` 内。

| 测试 | 验证内容 |
|------|---------|
| `test_coverage_invariant_exact_range` | 任意 `m/h/i`，三层可见位置并集恰为 `[0,i]`，一个不多一个不少 |
| `test_no_future_leak_in_coverage` | 任何可见位置都 ≤ `i`（不泄漏未来） |
| `test_fine_block_causal_boundary` | 细压缩只在"块末位置 ≤ i"时可见（`i=2,m=4` → 0 块；`i=3` → 1 块） |
| `test_coarse_block_causal_boundary` | 粗压缩块数同样满足因果边界 |
| `test_block_mean_matches_manual_and_drops_tail` | 块均值与手工对照，且尾部不足一块被丢弃 |
| `test_select_topk_picks_highest_and_ties_by_index` | top-k 取最高分，同分小下标优先，候选不足时只返回候选 |
| `test_attention_shape_and_differs_from_flash` | 输出与 `flash_attention` 同形；开启后数值确实改变；全部有限 |
| `test_forward_has_no_future_leak` | 改掉 `i` 之后的 K/V，query `i` 的输出不变（前向口径的无泄漏） |
| `test_gqa_heads_share_kv_but_score_independently` | GQA 下同组 Q 头共享 KV 但各自打分 |
| `test_dense_topk_equals_full_block_attention` | `top_k = 0` 与 `top_k = 块数` 等价（softmax 对槽位顺序不敏感） |
| `test_gradient_matches_finite_difference` | 稠密配置下的解析反向与有限差分一致 |
| `test_gqa_kv_gradient_accumulates_across_heads` | 共享 KV 头的梯度跨 Q 头正确累加 |
| `test_ledger_is_monotone_and_sane` | 账本在 `m ≪ T` 时确实更省；`m = T` 时如实更大 |
| `test_csa_changes_training_same_seed`（model.rs） | 同种子不新增参数、前向改变、数值有限 |

### 常见坑（实测踩出来的）

#### 1. 掩码必须与分数**逐行同序**，否则整行 `-inf` → `NaN`

见 §5 的 ⚠️。这是本次唯一一个"测试直接报 NaN"的 bug：`mask_flat` 按"层"拼接再 `reshape`，
与按"行"拼接的分数错位。修法是先算 `total_s = m + k + nb2`，再把每层掩码写到 `row + 偏移`。
（`masked_softmax` 现在对整行全屏蔽会输出全 0、不再产生 `NaN`，所以同类错误今后可能**静默**
给出错结果而不是当场报错——掩码与分数的对齐仍必须逐行核对。）

#### 2. `gather_seq` 里源序列长度 ≠ 输出序列长度

局部窗口取的是 `[gk, t, hd]` 的 K，输出却是 `[g, t, m, hd]`（GQA 下 `g ≠ gk`）；
细压缩取的是 `[g, nb, hd]` 的块均值，输出 `[g, t, k, hd]`（`nb ≠ t`）。
第一版把源的 `(g, t)` 与输出的 `(g, t)` 混用同一个参数，`reshape` 直接断言失败。
正确做法是把源的 `(gx, nx)` 与输出的 `(g_out, t_out)` **分开传**。

#### 3. 别把解析账本当成实测性能

`csa` 子命令打印的字节与 FLOPs 都是**纸面账本**——本仓库没有为 CSA 写融合 kernel，
算子仍是 CPU 上的一堆 `gather`/`mul`/`softmax`。这份账本能说明"压缩在原理上省了多少"，
**说明不了"跑得快了多少"**。能实测到的是正确性（覆盖不变式、无未来泄漏、梯度有限差分）。
真实收益取决于算子融合与内存带宽，这需要真正的 kernel 实现。

---

## 8. 小结与下一课

### 本课小结

CSA + HCA 把"每个 query 能看多少 key"从 `T` 压到 `m + k + T/(m·h)`，并用覆盖不变式保证可见位置的并集恰为 `[0, i]`；账本上的字节与算力收益要等融合 kernel 才能兑现。

### 拓展方向

- **融合 kernel**：把"块均值 + 打分 + top-k + 加权求和"写成一个内核，才能真正兑现账本里的算力收益。
- **压缩块缓存**：推理时把已完成块的 `K_c` / `V_c` 常驻，新的 token 只更新最近一个块；
  这样 `m + T/m` 的存储账本才在推理里落地。
- **top-k 的近似**：现在是精确选 top-k（`O(nb log nb)`）。可换成"按块均值范数预筛 + 精确重排"，
  把选择本身也降到亚线性。
- **与 MLA 组合**：MLA 压 KV 的**维度**、CSA 压 KV 的**个数**，两者正交，可以叠加。
- **可学习的压缩**：块均值是最朴素的压缩。可以换成每块一个小的线性投影，让模型自己决定"怎么压"。

> **下一课**：[第 50 课 MLA 解耦 RoPE + 权重吸收](50-MLA解耦RoPE与权重吸收.md) —— 压 KV 的维度：MLA 解耦 RoPE 与权重吸收。
