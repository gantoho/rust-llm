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

推理时，每生成一个新 token，需要用到之前所有 token 的 K 和 V。

标准 MHA：每个头都有独立的 K 和 V。

```
KV Cache 大小 = 2 * n_layer * n_head * T * head_dim * sizeof(f32)
```

例如 LLaMA-7B（n_layer=32, n_head=32, head_dim=128, T=2048）：

```
2 * 32 * 32 * 2048 * 128 * 4B = 2GB
```

这 2GB 全部是 K/V 缓存，推理时必须常驻显存。

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

## 6. 配置示例

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

---

## 7. 关键要点

- GQA = 多个 Q 头共享一组 K/V 头，是 MHA 和 MQA 的折中
- KV Cache 缩小 `n_head / n_kv_head` 倍，推理显存大幅降低
- 训练时几乎不影响效果（LLaMA 2 验证）
- 实现：K/V 投影维度变小 + flash_attention 核内按 `hh / n_rep` 索引共享头（零物化）
