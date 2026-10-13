# 第 47 课：mHC 流形约束超连接 —— 把残差流拓宽成多条并行通道

> **本课已落地为可运行代码**：[`src/mhc.rs`](../src/mhc.rs)（`HyperConnection` +
> `expand_streams` / `collapse_streams` / `sinkhorn` / `spectral_norm`），
> 接入点 [`src/model.rs`](../src/model.rs) 的 `TransformerBlock::forward_streams`
> 与 `Transformer::forward_core`。
> 配置开关：`model.n_stream`（默认 `1` = 关闭）/ `model.mhc_sinkhorn_iters`（默认 `20`）。
>
> 与前面各课的衔接：第 4 课起的残差连接一直是**一条**流（`x ← x + Sublayer(Norm(x))`）；
> 本课把它拓宽成 `n` 条**并行残差流**，层间用一份经 Sinkhorn 投影到**双随机矩阵**的
> 混合矩阵耦合。`n_stream = 1` 时行为与加 mHC 之前**逐位一致**。

---

## 1. 本课要搞懂的问题

1. 残差流从"一条"变成"多条"，读出 / 写入到底在做什么？
2. 为什么放开混合矩阵去学会失控？mHC 把它约束到哪个集合里？
3. 为什么"双随机"能同时给出"信号不放大"和"保均值"两个保证？
4. Sinkhorn–Knopp 迭代为什么能当"投影算子"用，还能反向？
5. `n_stream = 1` 为什么能**逐位**退化成普通残差？

---

## 2. 动机：单条残差流的信息瓶颈

从 ResNet 到 Transformer，主干骨架一直是：

```text
x ← x + Sublayer(Norm(x))
```

信息沿唯一的一条通道往前流，层与层之间只有"加法"这一种耦合。想让某一层"少传一点、
下一层多传一点"，除了改参数没有别的旋钮——**通路的宽度是固定的 1**。

**超连接（Hyper-Connections, HC）** 把这条通道拓宽成 `n` 条并行残差流
`X₀ … X_{n-1}`（本文按 `S ∈ R^{n×D}` 看，`D = b·t·d`）。每个子层不再直接读写同一条流：

```text
读出：h     = Σ_s α_s · X_s                  （α 在单纯形上，Σα = 1）
计算：y     = Sublayer(Norm(h))
写入：X_s'  = Σ_t A_{s,t} · X_t + β_s · y     （A 是 n×n 混合矩阵，β 在单纯形上）
```

读出用 `α` 把多条流压成一条喂给子层；写入先用 `A` 在**流之间**重新混合，再用 `β`
把子层输出按不同强度注入各条流。`n = 1` 时 `α = A = β = 1`，上面三行退化成普通残差。

---

## 3. 原理：为什么必须约束到双随机矩阵

### 3.1 不约束会怎样

放开 `A` 去学，它很快失控：元素可以任意大，前向一遍遍乘上去，信号指数爆炸（或塌缩）。
**mHC（Manifold-Constrained Hyper-Connections，DeepSeek-V4）** 的做法是把 `A` 约束到
**双随机矩阵**的集合（Birkhoff 多面体）：

```text
A ≥ 0,    A·1 = 1（每行和为 1）,    1ᵀ·A = 1（每列和为 1）
```

"流形约束"这个词的来历就在这：把可学参数限制在一个能保证数值稳定的**流形**（多面体）上。

### 3.2 好处一：信号不放大（谱范数恰为 1）

双随机矩阵是置换矩阵的凸组合（**Birkhoff–von Neumann 定理**）。谱范数 `‖·‖₂` 是凸函数、
置换矩阵的谱范数恰好是 1，于是

```text
‖A‖₂ = ‖Σ λᵢ Pᵢ‖₂ ≤ Σ λᵢ ‖Pᵢ‖₂ = 1
```

而 `A·1 = 1` 说明 `1` 是 `A` 的特征值，故 `‖A‖₂ ≥ 1`。两头一夹：

```text
‖A‖₂ = 1     ——每一步混合都恰好保范，既不放也不缩
```

这正是单测 `test_sinkhorn_spectral_norm_is_one` 用幂迭代核对的性质。

### 3.3 好处二：保均值（列和为 1）

`Σ_s X_s'` 里的混合部分是 `1ᵀ A X`。列和为 1 即 `1ᵀ A = 1ᵀ`，于是

```text
Σ_s (A X)_s = Σ_t X_t     ——"所有流的和"在混合前后不变
```

残差流的整体尺度不会随层数漂移。这条对应单测 `test_doubly_stochastic_preserves_total_energy`。

### 3.4 参数化：自由参数 + Sinkhorn 投影

直接把 `A` 参数化成双随机很难（要自定义梯度）。常用技巧是"**约束流形 + 自由参数化**"：
让 `A` 由一份自由矩阵经 **Sinkhorn–Knopp 迭代**投影得到。

```text
A⁰ = softmax_rows(depth_raw)        // 行 softmax 先保证非负、行和为 1
重复 iters 轮：
    列归一化  A ← A / (1ᵀ A + ε)     // 每列除以列和（ε = 1e-12 防除零）
    行归一化  A ← A / (A 1 + ε)      // 每行除以行和（同上）
```

> 两个 `+ ε` 不是装饰：softmax 在数值上可以把某一列压成**精确的 0**（例如行 softmax 的
> logits 极差很大时），此时列和 = 0，裸除法会得到 `0/0 = NaN` 并顺着反向污染整个
> `depth_raw` 的梯度。加 `ε` 之后这种列只会被压到 0（而不是 NaN），迭代语义不变。

正矩阵在这个迭代下必收敛到双随机矩阵（Sinkhorn–Knopp 定理）；本来就在流形上的输入
（单位阵、置换阵）是不动点。因此 `A` **永远合法**，而可学参数 `depth_raw` 不受约束。

---

## 4. 本仓库实现要点

| 位置 | 内容 |
|------|------|
| [`src/mhc.rs`](../src/mhc.rs) `expand_streams` | `[b,t,d] → [n, b·t·d]`，用一次 `[n,1]×[1,D]` 矩阵乘复制各流，再乘静态增益 `1 + s·0.01` |
| [`src/mhc.rs`](../src/mhc.rs) `collapse_streams` | 用**可学** `γ = softmax(gamma_raw)` 把 `[n, D] → [b,t,d]` |
| [`src/mhc.rs`](../src/mhc.rs) `HyperConnection::depth` | 行 softmax + `iters` 轮 Sinkhorn，**全在计算图里**，`depth_raw` 拿得到真梯度 |
| [`src/mhc.rs`](../src/mhc.rs) `read` / `write` | `αᵀS` 与 `A S + β⊗y`，各一次 `matmul` + 一次广播乘加 |
| [`src/mhc.rs`](../src/mhc.rs) `sinkhorn` / `doubly_stochastic_error` / `spectral_norm` | f32 参考实现，供单测与文档核对数值口径（不进训练路径） |
| [`src/model.rs`](../src/model.rs) `TransformerBlock::forward_streams` | 每个子层：读出 → 子层 → 写入，两个子层各持一份 `HyperConnection` |
| [`src/model.rs`](../src/model.rs) `Transformer::forward_core` | `n_stream > 1` 时走多流分支，否则走原来的单流循环 |
| [`src/config.rs`](../src/config.rs) → [`src/model.rs`](../src/model.rs) `TransformerConfig` | `n_stream`（默认 1）/ `mhc_sinkhorn_iters`（默认 20） |

**为什么全用矩阵乘**：把多条流看成一个大矩阵后，读出是一次 `[1,n]×[n,D]`、写入是一次
`[n,n]×[n,D]` 加一次 `[n,1]×[1,D]` 广播乘。全部走已有的、带反向的算子，`α` / `β` / `A`
都能自动拿到梯度，不必为"逐流相加"另写一套广播 / 求和内核。

**收敛为什么是学出来的 `γ` 而不是取平均**：均匀平均会让最后一个子层的写入参数
（`β`、`A`）**永久失去梯度**——`mean_s(Σ_t A_{s,t}X_t) = Σ_t X_t / n` 只用到"列和为 1"，
`mean_s(β_s y) = y / n` 只用到"β 和为 1"，两者都与具体取值无关。换成非均匀的 `γ` 加权求和后
`Σ_s γ_s β_s` 重新依赖 `β`，参数才能被训练（详见 §5 常见坑第 1 条）。

### 默认路径逐位不变

- `n_stream = 1`（默认）时 `TransformerBlock` **根本不构造** `HyperConnection`，
  `Transformer` 也不构造 `hc_collapse`：不新增参数、不消耗随机数；
- `forward_core` 的 `n_stream > 1` 分支整段不执行，残差路径仍是 `x = x.add(&h)`；
- GPU 常驻整叠快路的守卫里加了 `&& n_stream == 1`，多流时自动回退逐层路径；
- 单测 `test_mhc_n_stream_one_is_bitwise_identical` 钉住这一点：参数个数与名字完全一致、
  前向输出**逐位相同**；
- 即便手动构造 `HyperConnection::new(1, _)`，`softmax(单元素)` 与 `Sinkhorn(1×1)` 都精确等于 1，
  读出 / 写入也与普通残差逐位相同（`test_single_stream_degenerates_to_residual_bitwise`）。

### 代码地图

| 位置 | 内容 |
|------|------|
| [`src/mhc.rs`](../src/mhc.rs) | `HyperConnection`、`expand_streams` / `collapse_streams`、Sinkhorn 参考实现 |
| [`src/model.rs`](../src/model.rs) | `TransformerBlock::forward_streams`、`Transformer::forward_core` 的多流分支 |
| [`src/train.rs`](../src/train.rs) | `test_mhc_changes_training_same_seed`（同种子账本） |

---

## 5. 运行与验证

```bash
# mHC 单测（Sinkhorn 不动点 / 行列和 / 谱范数 / 双随机 / n=1 逐位退化 / 参数字典 / 梯度）
cargo test mhc

# 同种子对照：n_stream = 1 vs 2 走完整训练循环
cargo test mhc_changes_training_same_seed
```

配置示例：

```jsonc
{
  "model": {
    "n_stream": 2,              // 默认 1 = 关闭；2 即启用双残差流
    "mhc_sinkhorn_iters": 20    // 默认 20，只对 n_stream > 1 生效
  }
}
```

| 测试 | 验证内容 |
|------|---------|
| `test_sinkhorn_identity_fixed_point` | 单位阵（已在流形上）是不动点 |
| `test_sinkhorn_rows_and_cols_sum_to_one` | 任意正矩阵投影后行列和都收敛到 1 |
| `test_sinkhorn_spectral_norm_is_one` | 谱范数恰为 1（§3.2 的理论保证） |
| `test_sinkhorn_mildly_anisotropic_spectral_norm_bounded` | 强各向异性输入也被钉在 1 |
| `test_depth_tensor_is_doubly_stochastic` | 张量版 `depth()`（走计算图）同样是双随机 |
| `test_single_stream_degenerates_to_residual_bitwise` | `n = 1` 逐位退化成普通残差 |
| `test_expand_gain_and_collapse_weighted_sum` | 展开的静态增益与收敛的加权求和口径 |
| `test_alpha_beta_on_simplex` | `α` / `β` 落在单纯形上（非负、和为 1） |
| `test_mixing_params_receive_gradient` | `α` / `β` / `A` 都拿得到非零梯度 |
| `test_doubly_stochastic_preserves_total_energy` | 双随机 ⇒ 混合前后"所有流的和"不变 |
| `test_mhc_n_stream_one_is_bitwise_identical` | 默认关闭时参数表与前向逐位不变 |
| `test_mhc_two_streams_changes_forward_and_receives_grad` | `n_stream = 2` 改变前向且 mHC 参数拿到梯度 |
| `test_mhc_changes_training_same_seed` | 同种子完整训练循环下 loss 确实改变且有限 |

### 常见坑（实测踩出来的）

#### 1. 均匀平均收敛 ⇒ 最后一层的混合参数是"死参数"

第一版把收敛写成"对 `n` 条流取平均"。结果 `test_mhc_two_streams_changes_forward_and_receives_grad`
失败：缺梯度的恰好是 `blocks.0.hc_attn.alpha_raw`、`blocks.0.hc_attn.depth_raw`、
以及**最后一层**的 `hc_mlp.beta_raw`。两个独立原因：

- **入口对称**：展开时各条流逐位相同（都是同一份输入的复制），于是第一层的读出
  `h = Σ α_s X_s` 只依赖 `Σα_s = 1`，与 `α` 的取值无关；`Σ_t A_{s,t}X_t` 同理与 `A` 无关。
- **均匀收敛**：最后一层之后立刻取平均，见 §4 末尾的推导，`β` 与 `A` 都不可见。

修法是两处对称性破除，都不引入可学参数、不消耗随机数：

1. `expand_streams` 给第 `s` 条流乘静态增益 `1 + s·0.01`，各流互不相同；
2. 收敛改成可学的 `γ = softmax(gamma_raw)`（`concentrated_logits` 初始化，首条流为主）。

#### 2. "均匀损失对混合参数梯度为 0"是错的直觉

一度想用"均匀损失下 `depth_raw` 梯度恒为 0"当不变量断言，实测失败。原因是 Sinkhorn 的
Jacobian 不是正交投影：`A` 被推离某个非均匀点后，行列归一化的两次除法对自由参数的依赖
并不是严格抵消的。**不要用"梯度恒为 0"当不变量**；改用**前向值不变量**——
`sum(A @ S) == sum(S)`（列和为 1 的直接推论），这就是现在的
`test_doubly_stochastic_preserves_total_energy`。

---

## 6. 小结与下一课

### 本课小结

mHC 把单条残差流拓宽成 `n` 条并行流，用 Sinkhorn 投影到双随机矩阵，换来"步长保范 + 保均值"两个保证；`n_stream = 1` 时逐位退化成普通残差。

### 拓展方向

- **`n_stream` 扫描**：`2 / 4 / 8` 对照，看"通路变宽"的收益拐点与显存代价（多流会让
  中间激活按 `n` 倍放大）。
- **每层独立的 `n`**：现在全模型统一 `n_stream`，工程上可以让浅层少、深层多。
- **只读不写 / 只写不读**：把 `α` 或 `β` 固定成均匀，看哪个方向更关键。
- **与 Muon（第 46 课）组合**：`A` 是二维自由矩阵，Muon 会按谱范数正交化它——
  而双随机约束已经钉住了谱范数，两者的交互值得实测。
- **GPU 常驻整叠路径**：多流目前自动回退逐层路径，要吃到整叠快路需要把 `forward_streams`
  也接进常驻批次（见第 27 课）。

> **下一课**：[第 48 课 FP4 量化感知训练](48-FP4量化感知训练.md) —— 把位宽再砍一半：FP4（E2M1）量化感知训练与位宽账本。
