# 第 23 课：GQA 分组查询注意力 —— 推理时省显存的利器

> 代码位置：[src/attention.rs](../src/attention.rs)（`MultiHeadAttention` 支持 `n_kv_head`）
>
> 配置开关：`config/config.json` → `model.n_kv_head: 2`（示例值；仓库当前 `config/config.json` 里是 `0` = 标准 MHA，需手动改为 `2` 才能启用 GQA）

---

## 1. 本课要搞懂的问题

1. 标准多头注意力（MHA）的 KV Cache 为什么在推理时是瓶颈？
2. GQA 是怎么通过"共享 K/V 头"来减少 KV Cache 的？
3. MHA、GQA、MQA 三者的关系是什么？

---

## 2. KV Cache 回顾（第 25 课）

推理时每生成一个新 token 都要用到之前所有 token 的 K 和 V，标准 MHA 每个头都有自己独立的 K 和 V，
于是缓存大小是：

```text
KV Cache 大小 = 2 * n_layer * n_head * T * head_dim * sizeof(f32)
```

例如 LLaMA-7B（n_layer=32, n_head=32, head_dim=128, T=2048）：

```text
2 * 32 * 32 * 2048 * 128 * 4B = 2GB
```

这 2GB 全部是 K/V 缓存，推理时必须常驻显存——本课只关心"省多少份"，
缓存本身的结构、`append` / 滑动窗口 / Attention Sink 等细节**归属课是 [第 25 课 KV Cache](25-KV-Cache.md)**，这里不再展开。

---

## 3. GQA 的核心思想

**问题**：每个 Q 头都需要独立的 K/V 吗？

研究发现，训练好的 MHA 模型中，很多 K/V 头的输出高度相似（冗余）。

**解决**：让多个 Q 头共享同一组 K/V 头。

```
MHA:  8 个 Q 头，8 个 K/V 头    → KV Cache = 8 份
GQA:  8 个 Q 头，2 个 K/V 头    → KV Cache = 2 份（省 4 倍！）
MQA:  8 个 Q 头，1 个 K/V 头    → KV Cache = 1 份（省 8 倍！）
```

### 3.1 三者的关系

| 方法 | Q 头数 | K/V 头数 | 关系 |
|------|--------|---------|------|
| MHA | n_head | n_head | 每头独立 |
| GQA | n_head | n_kv_head | 多个 Q 共享一组 KV |
| MQA | n_head | 1 | 所有 Q 共享一组 KV |

GQA 是 MHA 和 MQA 的**折中**：比 MQA 效果好（保留了多样性），比 MHA 省显存。

---

## 4. 实现细节

### 4.1 K/V 投影维度变化

```
MHA:  c_k = Linear(n_embd, n_embd)        # n_head * head_dim
GQA:  c_k = Linear(n_embd, n_kv_head * head_dim)   # 更小！
```

### 4.2 共享头（核内索引，不物化）

GQA 前向时，Q 有 `n_head` 个头、K/V 只有 `n_kv_head` 个头。早期实现先用 `repeat_kv`
把每个 KV 头复制 `n_rep = n_head / n_kv_head` 次再算注意力；现在**不再物化**：
`flash_attention` 的 CPU 分块核按 Q 头 `hh / n_rep` 直接索引共享的 KV 头
（GPU 路径在算子入口核外展开），省掉整块拷贝与反向的跨副本求和节点。

```
Q 头 hh → KV 头 hh / n_rep   （核内索引，零物化）
```

### 4.3 计算流程

```
Q: [B, T, n_head, head_dim] → 拆头 [B*n_head, T, head_dim]
K: [B, T, n_kv_head, head_dim] → 拆头 [B*n_kv_head, T, head_dim]   # 不复制！
V: [B, T, n_kv_head, head_dim] → 拆头 [B*n_kv_head, T, head_dim]   # 不复制！
flash_attention 内部：Q 头 hh 按 hh / n_rep 取共享 KV 头计算
```

---

## 5. 推理时的显存节省

```
MHA KV Cache = 2 * n_layer * n_head * T * head_dim
GQA KV Cache = 2 * n_layer * n_kv_head * T * head_dim
节省比例    = n_head / n_kv_head
```

| 模型 | n_head | n_kv_head | KV Cache 节省 |
|------|--------|---------|-------------|
| LLaMA-2 7B | 32 | 32 | 1×（标准 MHA） |
| LLaMA-2 70B | 64 | 8 | 8× |
| Mistral 7B | 32 | 8 | 4× |

---

## 6. 延伸：MLA 低秩压缩 KV Cache（DeepSeek-V2/V3）

> **已落地代码**：[`src/attention.rs`](../src/attention.rs) 的 `MultiHeadAttention`
> （`kv_lora_rank > 0` 时走 `forward_mla`）+ `KVCache::new_latent` / `append_latent` /
> `positions`。配置开关：`model.kv_lora_rank`（默认 `0` = 关闭）。

### 6.1 GQA 之后，为什么还要 MLA

GQA 的思路是"**少存几份**"：让多个 Q 头共享 K/V 头，缓存缩小 `n_head / n_kv_head` 倍。

MLA（Multi-head Latent Attention）换了个方向："**每份压扁**"——K 和 V 不再各自从输入
投影，而是先压到一份低秩 latent，再从 latent 升维出来：

```text
    GQA:   x ──c_k──> K ┐                       缓存：K、V 两份
           x ──c_v──> V ┘

    MLA:   x ──c_kv──> c (r 维)  ─┬─c_k──> K   缓存：只有 c
                                  └─c_v──> V
```

两种思路可以叠加（MLA 里同样有 `n_kv_head`），但收益来源不同：GQA 省的是"份数"，
MLA 省的是"每份的体积"。

### 6.2 省了多少

```text
MHA/GQA 缓存 = 2 * n_layer * T * n_kv_head * head_dim * 4 字节
MLA   缓存 =     n_layer * T * kv_lora_rank          * 4 字节
```

DeepSeek-V2：`kv_lora_rank = 512`，对比 `n_kv_head × head_dim = 128 × 192`，
缓存小了约 96 倍。长上下文下 KV Cache 常常比权重还大，这一项直接决定能开多长的上下文。

### 6.3 代价写在明处：算力换显存

普通注意力每次只算**新 token** 的 K/V；MLA 每次推理都要把**整段 latent 升维回 K/V**
（`forward_mla` 第 3 步是全量 `c_k`/`c_v` 投影）。省的是显存，付的是算力——
这不是"免费的压缩"，而是一个明确的取舍。

### 6.4 本实现的取舍（教学版）

- **保留论文的核心**：KV 联合低秩压缩 + 缓存只存 latent。
- **decoupled RoPE 单独成课**：论文把 Q/K 的 rope 子维单独拎出来只对那部分旋转，
  让压缩后的 latent 不必带位置信息。本课（第 23 课）的 MLA 是**最简形态**——让 RoPE 作用在完整的
  `head_dim` 上，因此缓存里的 latent 是**未旋转**的，靠 `KVCache::positions` 每步按各自绝对位置补旋
  （滑动窗口丢行后位置并不连续，所以必须存每个位置的绝对下标）。
  效果上的差别：压缩率略低、每步多一次旋转（`T` 行的 `O(T·d)` 计算，与升维同量级）。
  **论文式的完整版（decoupled RoPE + `W_UK`/`W_UV` 权重吸收）已在第 50 课落地**：
  开 `model.mla_rope_dim > 0`（`head_dim = hd_nope + hd_rope`，只转后 `hd_rope` 维、
  `k_rope` 由输入直接投影）与 `model.mla_absorb = true`（解码在 `r + n_head·hd_rope` 维 latent 空间完成、
  K/V 不物化），详见 [第 50 课](50-MLA解耦RoPE与权重吸收.md)。

### 6.5 配置与约束

```json
{
  "model": {
    "n_head": 8,
    "n_kv_head": 2,
    "kv_lora_rank": 32
  }
}
```

- `kv_lora_rank: 0` 或不设 = 普通 MHA / GQA（行为与加 MLA 之前**逐位一致**，
  包括初始化消耗的随机数顺序）。
- **约束**：`kv_lora_rank` 必须**小于** `n_kv_head × head_dim`，否则谈不上压缩（构造时断言）。
- MLA 模型必须配 MLA 模式的缓存：`Transformer::new_kv_cache` 会按配置自动选
  `KVCache::new_latent`，前向里也会断言缓存类型匹配。
- MLA 的压缩投影 `c_kv` **不挂 LoRA 适配器**：它同时供给 K 和 V 两条路，低秩增量在那里
  既破坏"压缩"的口径，也没有现成的社区做法；要调 MLA 的 KV 表示，直接调
  `kv_lora_rank` 重训更干净。

---

## 7. 配置示例

```json
{
  "model": {
    "n_head": 8,
    "n_kv_head": 2
  }
}
```

`n_kv_head: 0` 或不设 = 标准 MHA（n_kv_head = n_head）。

**约束**：`n_head` 必须能被 `n_kv_head` 整除。

启用 MLA 时另加 `kv_lora_rank`（见 6.5 节）。

---

## 8. 小结与下一课

### 本课小结

- GQA = 多个 Q 头共享一组 K/V 头，是 MHA 和 MQA 的折中
- KV Cache 缩小 `n_head / n_kv_head` 倍，推理显存大幅降低
- 训练时几乎不影响效果（LLaMA 2 验证）
- 实现：K/V 投影维度变小 + flash_attention 核内按 `hh / n_rep` 索引共享头（零物化）

### 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. **扫 `n_kv_head`**：把 `config/config.json` 的 `model.n_kv_head` 在 `1 / 2 / 4 / 8`（配合 `n_head = 8`）各训一遍，对比 loss 与推理显存——`1` 就是 MQA，看它与 GQA 的差距。
2. **确认零物化**：在 `flash_attention` 的 CPU 分块核里加一行计数，确认 `n_rep > 1` 时按 `hh / n_rep` 索引共享头、而不是先 `repeat_kv` 复制一份。
3. **验整除约束**：把 `n_kv_head` 设成不整除 `n_head` 的值（如 `n_head = 8`、`n_kv_head = 3`），确认构造时直接断言失败而不是静默算错。
4. **算一遍 KV Cache 的字节数**：用第 5 节的三行公式，按本仓库的实际 `n_layer` / `n_head` / `block_size` 估算 MHA 与 GQA 的缓存字节数，和进程实际驻留对比。

> **下一课**：[第 24 课 Flash Attention](24-Flash-Attention.md) —— GQA 省的是缓存的"份数"，下一课解决注意力打分矩阵本身的显存与 IO 瓶颈。
