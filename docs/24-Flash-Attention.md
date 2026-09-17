# 第 24 课：Flash Attention —— 分块在线 softmax，GPU 显存救星

> 代码位置：[../src/tensor.rs](../src/tensor.rs)（`Tensor::flash_attention` 融合算子）
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

> ⚠️ 本项目**没有**实现这一套在线 softmax —— 参见 [§6](#6-本项目的实现)：
> 代码里的 `Tensor::flash_attention` 已经重写成「三次矩阵乘」的等价形式，数学结果相同，工程路线不同。

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

**先看清一件事**：本项目的注意力**数学上等价于**上面描述的 Flash Attention，但**工程实现走的是完全不同的路线** ——
没有分块、没有在线 softmax，而是「把三个算子都换成矩阵乘内核」。

### 6.1 2026-09-16 的重写

最初的实现就是标准的「分块 + 在线 softmax」标量三重循环版。逐算子插桩后发现它是**单步最大的瓶颈**：

| 算子 | 耗时 | 有效算力 |
|------|------|---------|
| flash 前向 | 4.6s | 0.12~0.24 GFLOP/s |
| flash 反向 | 8.7s | （同上） |
| 单步合计 | 17.1s，其中注意力占 **78%** | 比 `matmul_data` 慢约 100 倍 |

瓶颈既不是访存也不是算法，而是**没走上已经分块 / 向量化 / 可走 GPU 的矩阵乘内核**。
于是重写为（[src/tensor.rs](../src/tensor.rs) `Tensor::flash_attention`）：

```text
Q' = Q / √d          // 缩放挪到 Q 上，只要 524K 次乘法
S  = Q'·Kᵀ           // matmul_data [BH,T,T_total]，一次算完
P  = softmax(S + M)  // 融合内核，每行一遍过
O  = P·V             // matmul_data
```

反向同样全用矩阵乘：`dV = Pᵀ·dO`、`dP = dO·Vᵀ`、`dS = P⊙(dP - ΣdP·P)`、`dQ = dS·K·scale`、`dK = dSᵀ·Q'`。

**代价与收益**：

| | 旧实现（分块在线 softmax） | 现实现（matmul 重写） |
|---|---|---|
| 显存 | 存 P，O(T²) | 存 P 与 dP，仍是 O(T²) |
| 单步注意力耗时 | ~13.3s | **亚秒级** |

也就是说：**这是「用显存换速度」的反向操作** —— 省下来的显存本来就没被真正省掉（旧实现同样保存 P），
所以直接换回矩阵乘内核。

### 6.2 `block_size` 参数已失效

`Tensor::flash_attention(q, k, v, mask, _block_size)` 的第 5 个参数**现在被忽略**（签名保留是为了兼容旧调用），
分块交给 `matmul_data` 内部处理。`attention.rs:181` 传的 `32` 不再有任何含义。

### 6.3 集成状态：已接入

`MultiHeadAttention::forward`（[src/attention.rs](../src/attention.rs)）已经调用：

```rust
let out = Tensor::flash_attention(&q, &k, &v, mask, 32);
```

**不是**留作练习的独立算子 —— 训练与推理走的都是它。

### 6.4 常驻显存路径（`--features gpu`）

前向的 `S = Q'·Kᵀ → P = softmax(S+mask) → O = P·V` 三个算子会录进**一次提交**，
S（33.6MB）与 P（33.6MB）全程留在显存，只把 O（4.2MB）回读给 CPU，P 的显存句柄留到反向用。
反向的 `dV/dP/dS/dQ/dK` 同理一次提交，只回读 dQ/dK/dV（各 4.2MB）。

不这么做的代价很直观：逐算子路径要把 P 回读 33.6MB、下一步再原样传回，一来一回 67MB/层/次纯属白跑
（实测单步回读 1.46GB，91% 的时间花在等回读）。GPU 不可用或形状太小时自动回退逐算子 / CPU，数值行为不变。

### 6.5 测试

| 测试 | 验证内容 |
|------|---------|
| `test_flash_attention_matches_standard` | 前向 vs 标准注意力（`masked_softmax` + matmul）一致性 |
| `test_flash_attention_backward_matches_standard` | 反向 dQ/dK/dV vs 标准注意力反向 |
| `test_flash_attention_resident_path_matches_loop_reference` | GPU 常驻显存路径 vs 逐算子参考（仅 `--features gpu`） |

---

## 7. 关键要点

**论文侧（§1-5，Flash Attention 的原意）**

- 通过分块计算避免构建完整的 T×T scores 矩阵，把 attention 的显存从 O(T²) 降到 O(T·d)
- 在线 softmax 技巧：逐块更新 running max 与 running sum，不需要一次性看到整行
- 反向时用保存的统计量（max、sum）重新计算 P，不需要存储完整 scores
- Tri Dao 的论文是现代 LLM 训练的基石之一

**本项目侧（§6，实际实现）**

- 本项目**没有**实现分块与在线 softmax（[src/tensor.rs](../src/tensor.rs) `Tensor::flash_attention`），
  参数 `_block_size` 已被忽略；显存仍是 **O(T²)**（P 与 dP 都完整保留给反向用）
- 走的是「数学等价 + 换成矩阵乘内核」的路线：`Q'=Q/√d → S=Q'·Kᵀ → P=softmax(S+M) → O=P·V`，
  反向同样全用矩阵乘；单步注意力耗时 ~13.3s → 亚秒级，这是本项目**最值得记住的一条经验**：
  优化前先插桩，瓶颈常常不在算法而在「没走上已优化好的内核」
- `--features gpu` 下前向/反向各录成一次提交，S（33.6MB）与 P（33.6MB）留在显存不回读，
  只回读 O（4.2MB）；GPU 不可用或形状过小时自动回退，数值行为不变
- 正确性由 3 个测试守住：前向、反向各对拍标准注意力，常驻显存路径对拍逐算子参考
