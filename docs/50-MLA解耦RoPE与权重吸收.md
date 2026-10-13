# 第 50 课：MLA 解耦 RoPE + 权重吸收 —— 让 KV 缓存只存一份 latent

> **本课已落地为可运行代码**：[`src/attention.rs`](../src/attention.rs) 的
> `forward_mla_decoupled` / `absorb_attention`（解耦 RoPE 与 W_UK/W_UV 吸收），
> 接入 [`src/model.rs`](../src/model.rs) 的 `TransformerConfig` 与 [`src/config.rs`](../src/config.rs) 的校验，
> 演示命令 [`src/cmd/experiments.rs`](../src/cmd/experiments.rs) 的 `cmd_mla`。
> 配置开关：`model.mla_rope_dim` / `model.mla_absorb`（默认 `0` / `false` = 关闭）/ 命令 `cargo run -- mla`。
>
> 与第 23 课（MLA 低秩压缩 KV cache）的关系：第 23 课的 MLA **只做了低秩压缩**——
> 把 K/V 先压到 `r` 维 latent、再从 latent 升维，缓存只存一份 latent。但它留了一个教学取舍：
> **RoPE 作用在完整 `head_dim` 上**。本课补上 DeepSeek-V2/V3 真正的做法——**解耦 RoPE**，
> 并顺带解锁它带来的重量级优化：**W_UK/W_UV 权重吸收**（解码时 K/V 全程不物化）。

---

## 1. 本课要搞懂的问题

1. 第 23 课的 MLA 为什么"不完整"？把 RoPE 作用在完整 `head_dim` 上有什么问题？
2. 什么是 **decoupled RoPE（解耦 RoPE）**？为什么它要额外投影一份 `q_rope` / `k_rope`？
3. 解耦之后，**权重吸收**（absorption）为什么才成立？它省下的到底是什么？
4. 吸收路径里，`c_k` / `c_v` 的 bias 各去哪了？为什么一个能忽略、一个必须补回？
5. 解耦 RoPE 给缓存加了什么？这份代价值与不值？（本课账本：缓存字节 + 吸收算力）

---

## 2. 第 23 课遗留的问题：RoPE 与低秩压缩的冲突

第 23 课的 MLA 结构是：

```text
  c_kv = x · W_DKV                       // [b, t, r]   一份低秩 latent
  K    = c_kv · W_UK                     // [b, t, n_head·head_dim]   从 latent 升维
  V    = c_kv · W_UV                     // [b, t, n_head·head_dim]
  K    = RoPE(K)                         // ⚠️ 旋转在升维之后、作用在整个 head_dim
```

问题出在最后一步：**RoPE 是位置相关的旋转，而 K 是从 latent 升维来的**。
一旦旋转作用在"升维之后"的 K 上，K 就**不再是 latent 的纯线性函数**了
（旋转矩阵依赖位置）。这直接堵死了权重吸收——第 3 节的恒等式要求 `K` 对 `latent` 是线性的。

DeepSeek 的解法是：**把 RoPE 从 K 里摘出来，单独走一条低维通路**。

---

## 3. 解耦 RoPE：把"内容"和"位置"分开走

把每个头的维度切成两段：`head_dim = hd_nope + hd_rope`。

```text
  ── 内容通路（不带位置）──────────────
  c_kv   = x · W_DKV                     // [b, t, r]       共享 latent
  k_nope = c_kv · W_UK                   // [b, t, n_head·hd_nope]
  v      = c_kv · W_UV                   // [b, t, n_head·hd_nope]

  ── 位置通路（直接由输入投影，不经过 latent）──
  q_rope = x · W_QR  →  RoPE             // [b, t, n_head·hd_rope]
  k_rope = x · W_KR  →  RoPE             // [b, t, n_head·hd_rope]

  ── 拼接 ──
  q = [ q_nope ‖ q_rope ]                // q_nope = x · W_Q
  k = [ k_nope ‖ k_rope ]
```

关键点：**`k_rope` 完全绕开了 latent**，由输入 `x` 直接投影得到。于是

- `k_nope` 仍是 `latent` 的**纯线性**函数（`W_UK`）⇒ 可以吸收；
- `k_rope` 带着位置信息，但它本来就只有一个很窄的 `hd_rope` 维（V3 里 64），
  直接**按位置旋转后缓存起来**即可，历史 token 不需要重算。

注意力打分变成两段之和：

```text
  q·k = q_nope·k_nope  +  q_rope·k_rope
```

第一段走 latent（可吸收），第二段是普通小 RoPE 点积。

### 3.1 缓存里存什么

解耦后，每个位置需要缓存的"键侧"信息是：

```text
  缓存一行 = [ latent(r)  ‖  rotated_k_rope(n_head · hd_rope) ]
```

`latent` 各头共享（一份），`k_rope` 逐头（`n_head` 份）。本仓库把两者**拼成一条流**
存进既有的 latent 缓存（[`KVCache::append_latent`](../src/attention.rs)），
读取时用 `split_last_cols` 无损切分——所以滑动窗口 / Attention Sink / rollback 全部原样复用，
**不用改任何缓存逻辑**。

---

## 4. 权重吸收：解码时 K/V 全程不物化

有了"`k_nope` 是 latent 的线性函数"这个前提，就能把 `W_UK` 从 K 侧**搬到 Q 侧**：

```text
  q_nope[h] · k_nope[h]
    = q_nope[h] · (c · W_UK[h])              // k_nope[h] = c · W_UK[h]
    = (q_nope[h] · W_UK[h]ᵀ) · c             // 结合律：把 W_UK 折进 q
    = qa[h] · c                              // qa[h] ≜ q_nope[h] · W_UK[h]ᵀ  （一次预计算）
```

于是打分不再需要"把 `c` 升维成 `k_nope`"，而是**在 `r` 维 latent 空间直接点积**。
V 侧同理，利用"加权求和是线性"：

```text
  Σ_j p[h,j] · v_j
    = Σ_j p[h,j] · (c_j · W_UV[h])           // v_j = c_j · W_UV[h]
    = (Σ_j p[h,j] · c_j) · W_UV[h]           // 先对 latent 加权，再一次性升维
```

结论：**K 与 V 全程不物化**——注意力完全在 `r + hd_rope` 维空间里完成。

### 4.1 两处 bias 的不同命运

`c_k` / `c_v` 都带 bias，它们在吸收里的处置**不一样**：

- `b_k`（`c_k.bias`）加到 `k_nope` 上，对每个 query 都是一个**逐 head 的常数**。
  常数加在 softmax 的输入上，分子分母同乘 `e^{const}`，**自动抵消** ⇒ 可以忽略。
- `b_v`（`c_v.bias`）加到 `v` 上。因为 `Σ_j p[h,j] = 1`，输出会整体平移 `b_v` ⇒
  **必须补回**（[`absorbed_v_bias`](../src/attention.rs)）。

这条差异由单测 `test_mla_absorb_matches_naive_same_weights` 钉死：它特意给 `c_k`/`c_v`
设了**非零偏置**，再核对吸收输出 == 朴素输出。

### 4.2 吸收的复用边界

吸收用到的恒等式只对**线性**的 K/V 通路成立，所以：

- 吸收前向**只在无梯度推理**（`!grad_enabled()`）时启用——训练仍走朴素路径（可求导、正确）；
- 吸收与 **QK-Norm 不兼容**：RMSNorm 是逐头非线性缩放，无法被线性吸收进 `W_UK`；
- 解耦要求 `n_kv_head == 0 || n_kv_head == n_head`（不允许真正的 GQA 头分组），
  这样"每个 Q 头对应一个 K 头"，吸收的 `W_UK[h]` 分块才一一对应。

---

## 5. 朴素路径怎么实现（可求导）

训练 / 需要梯度时，走朴素解耦路径 [`forward_mla_decoupled`](../src/attention.rs)：

1. `q_nope`（可 QK-Norm）、`q_rope`（RoPE）分别投影；
2. `k_rope` 投影 + RoPE，与 `latent` 拼成一条流进缓存；
3. 从缓存切出 `latent` / `k_rope`，`k_nope = c_k(latent)`、`v = c_v(latent)`；
4. **拼接**：`q = [q_nope ‖ q_rope]`、`k = [k_nope ‖ k_rope]`（用可求导的
   [`concat_last`](../src/csa.rs)）；
5. `v` 的**后 `hd_rope` 维补 0**，凑齐 `flash_attention` 要求的同维 K/V；
6. `flash_attention` → reshape → `c_proj`。

> ⚠️ 第 5 步的"补零"是本实现的一个简化：真实 MLA 的 V 只有 `hd_nope` 维（没有 RoPE 段），
> 而本仓库的 `flash_attention` 要求 K/V 同 `head_dim`，所以把 V 的 rope 段填 0，
> 由 `c_proj` 对应的那几列"吸收"掉（那几列会一直是初始值，且拿不到梯度——无害）。

---

## 6. 本仓库实现要点

| 位置 | 内容 |
|------|------|
| [`src/attention.rs`](../src/attention.rs) `MultiHeadAttention` | 新增字段 `mla_rope_dim` / `mla_absorb` / `c_qr: Option<Linear>` / `c_kr: Option<Linear>`；`mla_rope_dim = 0` 时不构造 `c_qr`/`c_kr`、**不消耗随机数** |
| [`src/attention.rs`](../src/attention.rs) `forward_mla` | 顶部按 `mla_rope_dim > 0` 分派到 `forward_mla_decoupled`；否则走原路径，**逐位不变** |
| [`src/attention.rs`](../src/attention.rs) `forward_mla_decoupled` | 解耦前向：Q/K 两段拼接 → 缓存 `[latent‖k_rope]` → 朴素 `flash_attention` 或吸收分支 |
| [`src/attention.rs`](../src/attention.rs) `absorb_attention` | 吸收前向：`qa = q_nope·W_UKᵀ`，在 `r + hd_rope` 维做 `matmul + masked_softmax`，最后补 `b_v` |
| [`src/attention.rs`](../src/attention.rs) 辅助函数 | `join_head_parts` / `split_last_cols` / `repeat_batch_heads` / `absorbed_k_weight` / `absorbed_v_weight` / `absorbed_v_bias` |
| [`src/attention.rs`](../src/attention.rs) 账本函数 | `full_kv_cache_bytes_per_token` / `mla_cache_bytes_per_token` / `naive_mla_macs` / `absorbed_mla_macs` |
| [`src/model.rs`](../src/model.rs) | `TransformerConfig` 新增 `mla_rope_dim` / `mla_absorb` |
| [`src/config.rs`](../src/config.rs) | 校验：`r > 0`、`n_kv_head ∈ {0, n_head}`、`hd_rope < head_dim` 且为偶数、`mla_absorb ⇒ !qk_norm` |
| [`src/cmd/experiments.rs`](../src/cmd/experiments.rs) `cmd_mla` | `mla` 子命令：缓存字节账本 → 吸收算力账本 → 约束自检 |

**两个实现细节**：

- 解耦模式下 `c_q`/`c_k`/`c_v` 的输出维变成 `n_head·hd_nope`，`q_norm`/`k_norm` 的
  `gamma` 长度也变成 `hd_nope`（QK-Norm 只作用在**不旋转**的那一段上）。
- `c_qr` / `c_kr` **最后构造**：`mla_rope_dim = 0` 时根本不创建它们，
  保证默认配置下的随机数消耗顺序与加本课之前**逐位一致**。

### 默认路径逐位不变

- 默认 `mla_rope_dim = 0`：`forward_mla` 走原 MLA 路径，`c_qr`/`c_kr` 为 `None`、
  不消耗任何随机数 ⇒ 同种子下参数名与参数值逐位相同
  （单测 `test_mla_decoupled_off_is_bitwise_identical`）；
- `mla_rope_dim = 0` 时 `mla_absorb` 必须为 `false`（`config.rs` 校验），
  所以"关"的语义是干净的；
- 解耦开启后，`mla_rope_dim` 仍然只在 **MLA 模式**（`kv_lora_rank > 0`）下有意义。

### 代码地图

| 位置 | 内容 |
|------|------|
| [`src/attention.rs`](../src/attention.rs) | `forward_mla_decoupled`（朴素可求导）/ `absorb_attention`（无梯度吸收）/ 6 个辅助函数 / 4 个账本函数 |
| [`src/model.rs`](../src/model.rs) | `TransformerConfig` 的 `mla_rope_dim` / `mla_absorb` |
| [`src/config.rs`](../src/config.rs) | 解耦与吸收的约束校验 |
| [`src/cmd/experiments.rs`](../src/cmd/experiments.rs) | `cmd_mla` 三节账本 |
| [`docs/23-GQA分组查询注意力.md`](23-GQA分组查询注意力.md) | MLA 低秩压缩 KV Cache 的基础（第 23 课「6. 延伸：MLA 低秩压缩 KV Cache（DeepSeek-V2/V3）」一节；本课补上它留的取舍——解耦 RoPE 与权重吸收） |

---

## 7. 运行与验证

```bash
# 解耦 + 吸收的单测（逐位不变 / 吸收等价 / 缓存一致 / 字节账本）
cargo test mla_decoupled
cargo test mla_absorb

# 端到端账本：三节演示（缓存字节 → 吸收算力 → 约束自检）
cargo run --release -- mla
```

配置示例：

```jsonc
{
  "model": {
    "kv_lora_rank": 512,   // 先开 MLA 低秩压缩（第 23 课）
    "mla_rope_dim": 64,    // 解耦 RoPE 的旋转维 hd_rope（> 0 开启；必须为偶数、< head_dim）
    "mla_absorb": true     // 权重吸收（仅无梯度解码时生效；与 qk_norm 互斥）
  }
}
```

| 测试 | 验证内容 |
|------|---------|
| `test_mla_decoupled_off_is_bitwise_identical` | `mla_rope_dim = 0` 时不构造 `c_qr/c_kr`、不消耗随机数，与同种子普通 MLA 逐位一致；开启后 `head_dim` 才拆成 `hd_nope + hd_rope` |
| `test_mla_absorb_matches_naive_same_weights` | 同权重、非零 `c_k`/`c_v` bias 下，无梯度吸收输出 == 朴素输出（`b_k` 抵消、`b_v` 补回） |
| `test_mla_decoupled_cache_matches_full_recompute` | 带 latent 缓存的逐 token 增量 == 全量重算（拼接/切分无损） |
| `test_mla_decoupled_cache_bytes` | 缓存字节 `== T·(r + n_head·hd_rope)·4`，且与解析账本同口径 |

### 7.1 实测账本（`cargo run --release -- mla`，默认 `n_head=128`、`head_dim=192=128+64`、`r=512`、`T=4096`、`L=60`）

**第一节：增量缓存字节**（每层每 token，f32，K + V 两份）

| latent r | MLA 字节 | 全量字节 | 压缩比 |
|----------|---------|---------|--------|
| 64 | 33,024 | 196,608 | 5.95x |
| 128 | 33,280 | 196,608 | 5.91x |
| 256 | 33,792 | 196,608 | 5.82x |
| 512 | 34,816 | 196,608 | **5.65x** |
| 1024 | 36,864 | 196,608 | 5.33x |
| 2048 | 40,960 | 196,608 | 4.80x |

放大到 `T = 4096 × 60` 层：MLA ≈ **7.97 GB** vs 全量 ≈ **45.00 GB**。
注意压缩比随 `r` 变化**不大**——因为这里 `hd_rope` 让 `k_rope` 占了 `128×64 = 8192` 维，
已经是瓶颈；`r` 只在分母里占一小块。这正是解耦的代价（见 §7 常见坑第 3 条）。

**第二节：权重吸收的解码算力**（每层、全历史 `T` 的乘加次数）

| T | 朴素 MACs | 吸收 MACs | 算力比 |
|---|-----------|-----------|--------|
| 128 | 2,151,677,952 | 33,554,432 | 64.12x |
| 512 | 8,606,711,808 | 83,886,080 | 102.60x |
| 2048 | 34,426,847,232 | 285,212,672 | 120.71x |
| 8192 | 137,707,388,928 | 1,090,519,040 | 126.28x |

比值随 `T` 增长趋近 **`hd_nope = 128`**——T 越长越接近（短序列时 `qa` 预计算与投回 `W_UV`
的常数项还占一定比例，所以 `T = 128` 时只有 64x）。

### 常见坑（实测踩出来的）

#### 1. 解耦的前提是"K 对 latent 线性"

吸收恒等式 `q_nope·(c·W_UK) = (q_nope·W_UKᵀ)·c` 要求 `k_nope` 是 `latent` 的**线性**函数。
任何加在 K 上的**非线性**（比如 QK-Norm 的 RMSNorm）都会破坏它。所以
`mla_absorb` 与 `qk_norm` 在 `config.rs` 里被显式判为互斥。

#### 2. `flash_attention` 要求 K/V 同维，而 MLA 的 V 少一段

真实 MLA 里 `v` 只有 `hd_nope` 维，`k` 有 `head_dim = hd_nope + hd_rope` 维。
本仓库的 `flash_attention` 要求两者同 `head_dim`，所以朴素路径给 `v` 的 rope 段**补 0**，
吸收路径则在输出后补 0（两者口径一致，由 `test_mla_absorb_matches_naive_same_weights` 保证）。

#### 3. 解耦 RoPE **不是免费的**

`k_rope` 由输入直接投影、逐头缓存，所以缓存里多出 `n_head · hd_rope` 一整段。
在 V3 超参下它是 `128 × 64 = 8192` 维，**比 latent 的 512 维还大一个量级**——
这就是 §7.1 里"压缩比对 `r` 不敏感"的原因。解耦换来的是**权重吸收的能力**
（以及更标准的 MLA 语义），代价是缓存里多一份逐头的位置分量。选 `hd_rope` 时要在
"位置分辨率"与"缓存体积"之间权衡，DeepSeek-V3 取的是 `head_dim` 的三分之一。

---

## 8. 小结与下一课

### 本课小结

解耦 RoPE 让 `k_nope` 对 latent 保持线性，从而解锁权重吸收（解码时 K/V 全程不物化，算力比随 `T` 趋近 `hd_nope = 128`）；代价是缓存里多一份逐头的 `k_rope`。

### 拓展方向

- **吸收 + 量化**：吸收后注意力在 latent 空间完成，而 latent 正是 KV 量化（第 33 课）最该量化的对象；
  两者可叠加做"低秩 + 低比特"的 KV 缓存。
- **融合 kernel**：把 `qa = q_nope·W_UKᵀ` 与 `s1 = qa·cᵀ` 写成一个内核，才能真正兑现账本里的算力收益。
- **解耦 RoPE 的变体**：`hd_rope` 可以按层变化（浅层大、深层小），或让 `q_rope`/`k_rope`
  共享一个更小的投影。这是 V3 之后不少工作继续调的方向。
- **与 CSA 组合**：MLA 压 KV 的**维度**、CSA（第 49 课）压 KV 的**个数**，两者正交，可叠加。

> **下一课**：本课是当前路线的最后一课（十二阶段共 50 课，路线图见 [`00-学习计划.md`](00-学习计划.md)）；再往前的方向见上面的拓展方向。
