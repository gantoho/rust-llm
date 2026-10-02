# 第 45 课：QK-Norm —— 把注意力的 Q/K 尺度钉死

> **本课已落地为可运行代码**：[`src/attention.rs`](../src/attention.rs) 的
> `MultiHeadAttention`（字段 `q_norm` / `k_norm: Option<RMSNorm>` + 辅助方法
> `apply_qk_norm`）+ 配置开关 `model.qk_norm`（默认 `false`）。
> 参数量口径在 [`src/scaling.rs`](../src/scaling.rs) 的 `params_per_layer` 里同步（`2 × head_dim`）。
>
> 与第 21/23 课的衔接：RMSNorm 是第 21 课的主角，GQA 是第 23 课的主角；QK-Norm 把 RMSNorm
> **逐头**用到 Q/K 上，并且与 GQA / MLA 都能组合。

---

## 1. 本课要搞懂的问题

1. 注意力打分 `q·k/√d` 的幅度为什么需要约束？它会怎么伤害训练？
2. QK-Norm 归一化的到底是哪一维？为什么是 `head_dim` 而不是 `n_embd`？
3. 它加在流水线的哪一步？放在 RoPE 之前还是之后、KV 缓存之前还是之后？
4. 默认关闭时，为什么能保证与加它之前逐位一致？

---

## 2. 动机：不受约束的 logit 幅度

单头注意力的打分是 `s = q·k / √head_dim`，再做 softmax。问题在于 **Q/K 的尺度没有天然约束**：

- 训练中 Q/K 投影的权重会漂移，`‖q‖`、`‖k‖` 一起变大或变小；
- `s` 的幅度随之漂移，softmax 被推到**饱和区**——最大项接近 1、其余接近 0，
  反向梯度 `∝ p_i(δ − p_j)` 趋近 0，这一层"学不动了"；
- 这是长训练里"训练到一半突然发散 / 变成平台"最常见的原因之一。

解法很直接：在 Q/K 进入注意力之前，把每个头的向量归一化到**单位 RMS**，让 `s` 的尺度
只由 `head_dim` 和 `gamma` 决定，不再随投影权重漂移。Gemma 2 / Chameleon / ViT-22B
都用它换来了更稳的训练（允许更大学习率、更短 warmup）。

---

## 3. 原理与公式

RMSNorm（第 21 课）对向量 `x` 做：

```text
RMSNorm(x) = x / sqrt(mean(x²) + eps) · γ
```

QK-Norm 就是**对每个头的 `head_dim` 向量**各做一次 RMSNorm：

```text
q_head = RMSNorm(q_head)，   k_head = RMSNorm(k_head)
```

以 4 头、`head_dim = 4`、`gamma = 1` 为例，归一化后每个头的 RMS 恒为 1，且对任意正尺度输入
都不变（`RMSNorm(c·x) = RMSNorm(x)`）——这正是它"钉死尺度"的原理。

GQA 下 Q 有 `n_head` 个头、K 只有 `n_kv_head` 个头：Q 侧逐头归一化 `n_head` 次，
K 侧逐头归一化 `n_kv_head` 次，**所有 KV 头共用同一套 `gamma`**（与 Gemma 2 一致——
它也是每个注意力层一套 QK 归一化）。

---

## 4. 本仓库实现要点

| 位置 | 内容 |
|------|------|
| [`src/attention.rs`](../src/attention.rs) `MultiHeadAttention::q_norm` / `k_norm` | `Option<RMSNorm>`；`Some` 时各是一根长度 **`head_dim`** 的 `gamma` |
| [`src/attention.rs`](../src/attention.rs) `MultiHeadAttention::new` | 参数 `qk_norm: bool`；为 `true` 时在**所有投影之后**构造 `RMSNorm::new(head_dim, LN_EPS)` |
| [`src/attention.rs`](../src/attention.rs) `apply_qk_norm(q_or_k, &norm, n_head)` | 把 `[..., n_head·head_dim]` 摊成 `[rows·n_head, head_dim]` → `rmsnorm` → 还原形状；`norm = None` 时**原样返回，不产生任何算子** |
| [`src/attention.rs`](../src/attention.rs) `forward` | 第 1.5 步：投影之后、RoPE 之前调 `apply_qk_norm`；之后才做 RoPE、才能进 KV 缓存 |
| [`src/attention.rs`](../src/attention.rs) `forward_mla` | MLA 路径同样在第 3.5 步（升维出 K/V 之后、旋转之前）调 `apply_qk_norm` |
| [`src/model.rs`](../src/model.rs) `TransformerConfig::qk_norm` | 配置字段，默认 `false` |
| [`src/scaling.rs`](../src/scaling.rs) `params_per_layer` | `let qk = if cfg.qk_norm { 2 * hd } else { 0 };`——每层多 Q、K 两根长度 `head_dim` 的 `gamma` |

流水线位置（普通路径）：

```text
x ──c_q──▶ q ──QK-Norm──▶ q ──RoPE──▶ q ┐
x ──c_k──▶ k ──QK-Norm──▶ k ──RoPE──▶ k ┤─▶ KV cache ──▶ attention
x ──c_v──▶ v ────────────────────────▶ v ┘
                 ↑
        归一化在 RoPE 与缓存**之前**
```

两个位置选择都有理由：

- **在 RoPE 之前**：RoPE 是正交变换、不改变向量范数，"旋转前归一化"就等于"旋转后归一化"，
  先做少一次旋转开销。
- **在 KV 缓存之前**：缓存里存的仍是"归一化 + 旋转"后的 K，历史复用不受影响——
  逐 token 增量前向与全量重算结果一致（`test_qk_norm_incremental_cache_matches_full_recompute`）。

MLA 路径的细节不同：缓存里存的是**未归一化**的 latent，所以归一化每步在"升维出 K/V 之后"
重做一次，保证 latent 增量与全量重算一致
（`test_qk_norm_with_mla_latent_cache_matches_full_recompute`）。

---

## 5. 默认路径逐位不变

- 默认 `qk_norm = false`：`q_norm` / `k_norm` 都是 `None`；
- `apply_qk_norm` 在 `None` 时**直接返回输入张量**，不产生任何算子（只是一次所有权转移）；
- `RMSNorm` 的 `gamma` 初始化为全 1，**不消耗随机数**；而且 QK-Norm 在所有投影之后才构造，
  所以老路径的随机数顺序一字未动；
- 因此老配置（`config.json` 里没有 `qk_norm` 字段、serde 取默认 `false`）训出来的权重与
  加 QK-Norm 之前**逐位一致**。

单测 `test_qk_norm_does_not_change_weight_initialization` 直接钉住了这一点：同 seed 下
`c_q/c_k/c_v/c_proj` 的权重与偏置逐位相同，多出来的只有每层 Q、K 两个 `gamma`
（`parameters().len()` 从 8 变 10，`gamma.numel() == d / n_head`）。

---

## 6. 验证方式

```bash
# QK-Norm 单测（逐头单位 RMS 与尺度不变、不扰动初始化、梯度到 gamma、与增量缓存一致）
cargo test qk_norm

# 同种子对照：开 QK-Norm 确实改变训练（否则说明开关没接进前向），且 loss 仍有限
cargo test test_qk_norm_changes_training_same_seed
```

配置示例：

```jsonc
{
  "model": {
    "qk_norm": true   // 默认 false；开启后每层多 2 × head_dim 个 gamma
  }
}
```

| 测试 | 验证内容 |
|------|---------|
| `test_qk_norm_makes_each_head_unit_rms_and_is_scale_invariant` | 每个头的 RMS 落到 1；输入放大 1000 倍结果不变；`None` 时逐位原样返回 |
| `test_qk_norm_does_not_change_weight_initialization` | 开启后四个投影权重/偏置与关闭时逐位相同，只多两根 `gamma`（长度 `head_dim`） |
| `test_qk_norm_gradients_reach_gamma` | `q_norm.gamma` / `k_norm.gamma` 都拿得到非零梯度（否则归一化学不动） |
| `test_qk_norm_incremental_cache_matches_full_recompute` | 普通路径下逐 token 增量前向与全量重算一致 |
| `test_qk_norm_with_mla_latent_cache_matches_full_recompute` | MLA latent 缓存下两条路径仍一致 |
| `test_qk_norm_changes_training_same_seed` | 同种子对照：开了确实改变 loss，且 loss 有限（新参数不破坏训练） |

---

## 7. 两个坑（实测踩出来的）

### 7.1 归一化的是 `head_dim`，不是 `n_embd`

QK-Norm 的口径是**每个头内部**的 `head_dim` 向量，所以 `gamma` 长度是 `head_dim = n_embd / n_head`，
不是 `n_embd`。实现里必须先把 `[B, T, n_head·head_dim]` 摊成 `[B·T·n_head, head_dim]` 再归一化、
再还原形状（`apply_qk_norm` 做的就是这个）。

写成对 `n_embd` 整体归一化会同时犯两个错：参数形状不对（`scaling.rs` 里 `2 * hd` 会对不上），
以及**跨头**求均值——那会把不同头的信息混在一起，语义完全变了。GQA 下 K 侧尤其容易写错：
它只有 `n_kv_head` 个头，但 `gamma` 仍然只有一套，各 KV 头共用。

### 7.2 开 QK-Norm 会让 GPU 常驻快路失效（性能取舍，不是数值问题）

[`src/model.rs`](../src/model.rs) 里判断是否走"GPU 常驻快路"的条件包含 `|| self.cfg.qk_norm`：

```rust
|| self.cfg.kv_lora_rank > 0
|| self.cfg.qk_norm      // QK-Norm 让路：内核把 Q/K 投影与 RoPE 融成一段
```

原因是常驻内核把 Q/K 投影与 RoPE 融成一段，中间插不进"逐头 RMSNorm"这一步。
**强行走常驻路径会静默丢掉归一化**（前向照跑、数值却与逐算子路径不一致），所以这里直接
让常驻快路返回 `None`、退回逐算子路径。代价是开了 QK-Norm 后 GPU 加速吃不到那条快路——
这是明确的性能取舍，别把它误当成数值 bug。

---

## 8. 代码地图与拓展方向

| 位置 | 内容 |
|------|------|
| [`src/attention.rs`](../src/attention.rs) | `q_norm` / `k_norm` 字段、`apply_qk_norm`、两条前向里的调用点 |
| [`src/model.rs`](../src/model.rs) | `TransformerConfig::qk_norm` 字段、常驻快路的绕行条件 |
| [`src/scaling.rs`](../src/scaling.rs) | `params_per_layer` 的 `2 * hd` 参数量口径 |

拓展方向：

- **与 warmup 的关系**：QK-Norm 的卖点之一是"允许更短 warmup、更大学习率"，
  可以做一个"开/关 QK-Norm × 不同 warmup 长度"的对照，看谁在短 warmup 下先发散。
- **可学习 vs 不可学习缩放**：本实现是标准 RMSNorm（`gamma` 可学习、初始化全 1）；
  也可以试"归一化后乘一个固定温度 `τ`"的变体，显式调 logit 温度。
- **只归一化一侧**：本实现 Q、K 两侧都归一化；也可以只归一化 Q（参数减半）看稳定性差多少。
- **与 MLA 组合**：见 `test_qk_norm_with_mla_latent_cache_matches_full_recompute`，
  两者正交、可同时开启。
