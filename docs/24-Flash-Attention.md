# 第 24 课：Flash Attention —— 分块在线 softmax，GPU 显存救星

> 代码位置：[src/tensor.rs](src/tensor.rs)（`Tensor::flash_attention` 融合算子）
>
> 算法论文：*FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness* (Tri Dao, 2022)

---

## 1. 本课要搞懂的问题

1. 标准 attention 的显存瓶颈在哪里？
2. Flash Attention 怎么通过"分块"避免构建完整的 T×T 矩阵？
3. "在线 softmax"是什么？为什么可以分块计算 softmax？
4. Flash Attention 对训练速度有什么影响？

---

## 2. 标准 Attention 的显存问题

```
scores = Q · Kᵀ        # [B*H, T, T]    ← 这个矩阵是瓶颈！
attn   = softmax(scores) # [B*H, T, T]
out    = attn · V        # [B*H, T, D]
```

当 T=2048, B*H=32 时：

```
scores 矩阵 = 32 * 2048 * 2048 * 4B = 512MB
```

这个 T×T 矩阵必须完整存在显存里（反向传播需要），是 attention 的主要显存瓶颈。

---

## 3. Flash Attention 的核心思想

**问题**：能不能不构建完整的 T×T 矩阵，也能算出正确的结果？

**答案**：可以！用"分块 + 在线 softmax"。

### 3.1 分块计算

把 Q 按行分块（Br 行），K/V 按列分块（Bc 列），逐块计算：

```
for each Q block (Br rows):
    for each K/V block (Bc cols):
        S_ij = Q_i · K_j^T / √d    # [Br, Bc] 小矩阵，不保存！
        P_ij = softmax(S_ij)        # 局部 softmax
        O_i += P_ij · V_j           # 累加输出
```

**关键**：每个 `[Br, Bc]` 的 scores 矩阵算完就丢，只保留累加后的输出。

### 3.2 在线 softmax

问题：softmax 需要知道整行的最大值和总和，但分块计算时一次只能看到一部分。

解决：用"在线"算法，逐块更新最大值和总和：

```
初始：m = -inf, l = 0, O = 0

for each K/V block:
    S_ij = Q_i · K_j^T / √d
    m_new = max(m, rowmax(S_ij))
    P_ij = exp(S_ij - m_new)
    l_new = exp(m - m_new) * l + rowsum(P_ij)
    O = exp(m - m_new) * O + P_ij · V_j
    m = m_new, l = l_new

O = O / l    # 最终归一化
```

每块只需要更新三个标量（m, l, O），不需要保存完整 scores。

---

## 4. IO 复杂度分析

| | 标准 Attention | Flash Attention |
|---|--------------|----------------|
| HBM 读写 | O(N² + Nd) | O(N²d²/M) |
| 显存占用 | O(N² + Nd) | O(N + d) |

M = SRAM 大小（GPU 的 L2 cache，通常几 MB）。当 d² ≪ M 时，Flash Attention 的 IO 复杂度接近 O(Nd)。

---

## 5. 反向传播

Flash Attention 的反向需要重新计算 P（注意力权重），但这次有保存的 (m, l) 统计量：

```
P_ij = exp(S_ij - m_i) / l_i
```

然后用 P 计算 dQ、dK、dV（和标准 attention 反向一样）。

**权衡**：反向时重新算 P（多一次前向计算），但省掉了存储完整 P 的显存。

---

## 6. 本项目的实现

本项目用纯 Rust 实现了 Flash Attention 的**前向 + 反向**（`Tensor::flash_attention`，`tensor.rs:1362`）：

- 前向：分块计算，在线 softmax，不构建完整 scores 矩阵
- 反向：保存 P（归一化后的注意力权重），用块级循环计算 dQ/dK/dV
- 单元测试：`test_flash_attention_matches_standard` 验证与标准注意力的数值一致性

**集成状态**：`Tensor::flash_attention` 是独立的融合算子，当前**未接入**模型的注意力层
（`MultiHeadAttention::forward` 使用的是 `Tensor::masked_softmax` 路径）。
接入方式：在 `attention.rs` 的 scores 计算后，将 `scores.masked_softmax(mask).matmul(&v)` 替换为
`scores.flash_attention(&v, mask)`。本课程重点是理解算法原理，接入留作练习。

**注意**：本项目保存了完整的 P 矩阵（O(T²)），这在 GPU 实现中是可以避免的
（反向时重新计算 P），但纯 CPU 实现中重算的开销太大，所以保留了 P。

---

## 7. 关键要点

- Flash Attention 通过分块计算避免构建完整的 T×T scores 矩阵
- 在线 softmax 技巧：逐块更新最大值和总和，不需要全局信息
- GPU 上速度提升 2-4×，显存从 O(N²) 降到 O(N)
- 反向时重新计算 P（用保存的统计量），不需要存储完整 scores
- Tri Dao 的论文是现代 LLM 训练的基石之一
