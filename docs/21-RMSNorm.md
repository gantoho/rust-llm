# 第 21 课：RMSNorm —— 现代 LLM 的归一化标配

> 代码位置：[src/tensor.rs](src/tensor.rs)（`Tensor::rmsnorm` 融合算子）
> 代码位置：[src/layers.rs](src/layers.rs)（`RMSNorm` 层、`NormLayer` 枚举）
> 代码位置：[src/model.rs](src/model.rs)（`TransformerBlock` / `GPT` 使用 `NormLayer`）
>
> 配置开关：`config.json` → `model.use_rmsnorm: true` 启用

---

## 1. 本课要搞懂的问题

1. RMSNorm 和 LayerNorm 有什么区别？为什么现代 LLM 全部换成了 RMSNorm？
2. 去掉均值减法和 β 偏置，模型效果会变差吗？
3. RMSNorm 的反向传播公式怎么推导？

---

## 2. LayerNorm 回顾

第 11 课实现的 LayerNorm：

```
μ = mean(x)
σ² = var(x)
y = (x - μ) / √(σ² + ε) * γ + β
```

每行需要两次 reduction：一次算均值 μ，一次算方差 σ²。

---

## 3. RMSNorm 公式

```
RMS = √(mean(x²) + ε)
y = x / RMS * γ
```

**和 LayerNorm 的三个区别**：

| | LayerNorm | RMSNorm |
|---|-----------|---------|
| 减均值 | ✅ `x - μ` | ❌ 不减 |
| β 偏置 | ✅ 有 | ❌ 没有 |
| Reduction 次数 | 2 次（mean + var） | 1 次（mean of squares） |

---

## 4. 为什么 RMSNorm 能替代 LayerNorm

### 4.1 直觉解释

LayerNorm 的核心作用是"把每层的输入拉回标准分布"。但研究发现：

- **减均值不是必需的**：Transformer 的输入通常已经接近零均值（经过 Xavier 初始化和多层传播）
- **β 偏置不是必需的**：γ 的缩放已经足够调整分布，β 增加的表达力微乎其微
- **方差信息就够了**：RMS（均方根）本身就是方差的平方根，足以做归一化

### 4.2 实验证据

论文《Root Mean Square Layer Normalization》(Zhang & Sennrich, 2019) 在多个任务上验证：
- RMSNorm 和 LayerNorm 效果相当（甚至略好）
- 训练速度快 7%~64%（省了一次 reduction）

### 4.3 现代 LLM 的选择

| 模型 | 归一化 |
|------|--------|
| GPT-2/3 | LayerNorm |
| LLaMA / LLaMA 2 / LLaMA 3 | **RMSNorm** |
| Mistral / Mixtral | **RMSNorm** |
| Qwen / Qwen2 | **RMSNorm** |
| Gemma | **RMSNorm** |

---

## 5. 反向传播推导

设 `is = 1/RMS`（每行一个标量），则 `y_i = x_i * is * γ_i`。

### 5.1 对 x 的梯度

```
d_y_γ_i = d_y_i * γ_i           // 先把 γ 吸收进来
Σxg = Σ_j(x_j * d_y_γ_j)       // 每行一个标量
d_x_i = is * (d_y_γ_i - (Σxg / d) * is² * x_i)
```

### 5.2 对 γ 的梯度

```
d_γ_j = Σ_r d_y[r,j] * (x[r,j] * is_r)
```

### 5.3 和 LayerNorm 反向的区别

- 没有 `d_β`（因为没有 β）
- 没有 mean 相关的项（因为没有减均值）
- 公式更简洁，实现更高效

---

## 6. 实现要点

### 6.1 融合算子

和 LayerNorm 一样，RMSNorm 实现为单个融合算子（`Tensor::rmsnorm`），一个函数完成前向+反向，
比拆成 `square → mean → add_eps → sqrt → div → mul` 6 个基础算子快得多。

### 6.2 `NormLayer` 枚举

为了兼容 GPT-2（LayerNorm）和 LLaMA（RMSNorm），用枚举统一接口：

```rust
pub enum NormLayer {
    LN(LayerNorm),
    RMS(RMSNorm),
}
```

调用方不需要关心具体用哪种归一化，`config.json` 里的 `use_rmsnorm` 控制选择。

### 6.3 并行化

每行独立计算，用 rayon `par_chunks_mut` 并行（和 LayerNorm 一样）。

---

## 7. 配置示例

```json
{
  "model": {
    "use_rmsnorm": true
  }
}
```

启用后，模型的所有归一化层（每层的 ln1、ln2、最终 ln_f）都会切换为 RMSNorm。

---

## 8. 关键要点

- RMSNorm = LayerNorm 去掉减均值和 β，只保留方差归一化
- 效果相当，速度更快（少一次 reduction，少一组参数）
- 现代 LLM（LLaMA/Mistral/Qwen/Gemma）全部使用 RMSNorm
- 反向公式比 LayerNorm 更简洁
