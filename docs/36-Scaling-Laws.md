# 第 36 课：Scaling Laws（缩放定律）

> **本课为前沿技术教程（纯文档，无配套代码实现）。**
> 与本项目已有代码的关联：可通过修改 `config.json` 的 `n_embd` / `n_layer` / `block_size` /
> `batch_size` / `steps` 来验证 Scaling Laws 的幂律关系。

---

## 为什么 Scaling Laws 重要？

Scaling Laws 是指导 LLM 发展的**第一性原理**——它告诉你：

- 增加多少参数、多少数据、多少算力，模型性能会提升多少
- 在有限预算下，如何最优地分配资源
- 模型性能的"天花板"在哪里

> **核心洞察**：模型的 loss（困惑度）与参数量、数据量、算力之间存在**幂律关系**（Power Law），且这种关系非常稳定，跨越多个数量级都成立。

## OpenAI Scaling Laws（2020）

Kaplan et al. 在论文《Scaling Laws for Neural Language Models》中发现：

### 幂律关系

$$
L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}, \quad \alpha_N \approx 0.076
$$

$$
L(D) = \left(\frac{D_c}{D}\right)^{\alpha_D}, \quad \alpha_D \approx 0.095
$$

$$
L(C) = \left(\frac{C_c}{C}\right)^{\alpha_C}, \quad \alpha_C \approx 0.050
$$

其中：
- $L$ 是测试 loss（交叉熵）
- $N$ 是模型参数量（非嵌入层）
- $D$ 是训练数据量（token 数）
- $C$ 是训练算力（FLOPs）

### 关键发现

1. **参数量最重要**：loss 对参数量的幂律指数最大（$\alpha_N > \alpha_D > \alpha_C$）
2. **架构细节不重要**：层数、宽度、注意力头数的具体配置对 loss 影响很小
3. **平滑可预测**：loss 随 N/D/C 的增长非常平滑，没有明显的"相变"点
4. **大模型更高效**：同样的算力，训练一个更大的模型（即使没训完）比训练一个小模型到收敛更好

## Chinchilla Scaling Laws（2022）

Hoffmann et al.（DeepMind）在《Training Compute-Optimal Large Language Models》中修正了 OpenAI 的结论：

### 核心修正

**数据量和参数量应该同步增长**。

OpenAI 的结论是"优先增大模型"，但实际上他们的实验**数据量不够大**，模型处于"欠训练"状态。

Chinchilla 的最优配比：

$$
N_{\text{opt}} \propto C^{0.5}, \quad D_{\text{opt}} \propto C^{0.5}
$$

即：**算力翻倍时，参数量和数据量应该各增加约 41%（$2^{0.5} \approx 1.41$）**。

### Chinchilla 最优

| 算力 (FLOPs) | 最优参数量 | 最优数据量 (tokens) |
|-------------|-----------|-------------------|
| $10^{18}$ | 400M | 8B |
| $10^{19}$ | 1.3B | 26B |
| $10^{20}$ | 4B | 80B |
| $10^{21}$ | 13B | 260B |
| $10^{22}$ | 40B | 800B |
| $10^{23}$ | 130B | 2.6T |

> **Chinchilla 的影响**：Chinchilla 70B 用 1.4T tokens 训练，性能超过了 Gopher 280B（用 300B tokens 训练）——参数量只有 1/4，但因为数据量更充足，效果更好。

### 实际应用中的偏离

实际训练往往**偏离 Chinchilla 最优**：

| 模型 | 参数 | 训练 tokens | Chinchilla 最优 tokens | 偏离程度 |
|------|------|-------------|----------------------|---------|
| LLaMA-1 7B | 7B | 1T | ~140B | 7× 过训练 |
| LLaMA-2 7B | 7B | 2T | ~140B | 14× 过训练 |
| LLaMA-3 8B | 8B | 15T | ~160B | 94× 过训练 |

**为什么故意过训练？**
- 推理成本：小模型推理更便宜，过训练小模型可以在推理时节省成本
- 数据充足：互联网数据量远超 Chinchilla 最优所需
- 小模型过训练的边际收益递减很慢

## 计算最优训练

### 给定算力预算，如何分配？

设总算力预算为 $C$ FLOPs，单位 FLOPs 的价格为 $p_C$，token 的价格为 $p_D$：

$$
\text{总成本} = p_C \cdot C + p_D \cdot D
$$

在算力约束 $C \approx 6ND$ 下（前向+反向传播的 FLOPs 近似为 $6 \times$ 参数量 $\times$ 数据量）：

$$
D_{\text{opt}} = \sqrt{\frac{C}{6N_{\text{opt}}}}
$$

### 算力估算

**训练 FLOPs 估算**（前向 + 反向）：

$$
C \approx 6 \cdot N \cdot D
$$

其中：
- $N$ = 模型参数量
- $D$ = 训练 token 数
- 6 = 前向约 2× + 反向约 4×（反向是前向的 2 倍）

**GPU 时间估算**：

$$
T = \frac{C}{\text{GPU\_FLOPS} \cdot \text{MFU} \cdot n_{\text{GPU}}}
$$

其中 MFU（Model FLOPs Utilization）是模型算力利用率，通常 30-60%。

### 实际计算示例

训练 LLaMA-7B（7B 参数，1T tokens）：

```
FLOPs = 6 × 7×10^9 × 10^12 = 4.2 × 10^22

假设 A100 80GB (312 TFLOPS FP16), MFU = 40%, 64 张 GPU:
每秒 FLOPs = 312×10^12 × 0.4 × 64 = 7.99 × 10^15

训练时间 = 4.2×10^22 / 7.99×10^15 ≈ 5.26×10^6 秒 ≈ 61 天

电费 (A100 ~400W): 64 × 0.4kW × 61天 × 24h × 0.1$/kWh ≈ $3,700
```

## Emergent Abilities（涌现能力）

### 什么是涌现？

某些能力（如思维链推理、多步算术）在小模型上完全不存在，但当模型规模超过某个阈值时**突然出现**。

```
性能
 ^
 |                    ╱ ← 大模型突然学会
 |                   ╱
 |    ──────────────╱ ← 看似"涌现"
 |   小模型表现随机
 +─────────────────────→ 模型规模
```

### 争议

Schaeffer et al.（2023）提出：涌现可能是**评估指标的假象**——如果用连续指标（如 Brier score）替代离散指标（如 exact match），涌现现象会消失，变成平滑的提升曲线。

## Scaling Laws 对实践的指导

| 决策 | Scaling Laws 的建议 |
|------|-------------------|
| 训练预算有限 | 优先增大模型，适当减少数据（但不要差太远） |
| 推理预算有限 | 过训练小模型（LLaMA-3 8B 训了 15T tokens） |
| 选择模型规模 | 用 $N_{\text{opt}} \approx 0.3 \cdot C^{0.5}$ 估算 |
| 预测最终性能 | 用幂律曲线外推（准确度高） |
| 何时停止训练 | loss 下降速度低于阈值，或达到 Chinchilla 最优 tokens |

## 动手练习

1. **幂律拟合**：在不同规模的模型上记录 loss，用对数坐标拟合 loss vs N 的幂律关系。
2. **算力估算**：给定 GPU 型号和数量，估算训练一个 13B 模型到 Chinchilla 最优需要多少天。
3. **最优配比计算**：给定 1e22 FLOPs 的算力预算，计算最优的参数量和数据量。
4. **过训练分析**：在小模型上分别用 1×、2×、4×、8× Chinchilla 最优数据量训练，观察 loss 曲线和生成质量的变化。
