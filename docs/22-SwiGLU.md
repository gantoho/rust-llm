# 第 22 课：SwiGLU 激活函数 —— LLaMA 的"门控" MLP

> 代码位置：[src/tensor.rs](src/tensor.rs)（`Tensor::swiglu` 融合算子）
> 代码位置：[src/layers.rs](src/layers.rs)（`SwiGLUMLP` 层、`MLPEnum` 枚举）
> 代码位置：[src/model.rs](src/model.rs)（`TransformerBlock` 使用 `MLPEnum`）
>
> 配置开关：`config.json` → `model.use_swiglu: true` 启用

---

## 1. 本课要搞懂的问题

1. SwiGLU 和 GELU 有什么区别？为什么 LLaMA 用 SwiGLU 替代 GELU？
2. 门控机制（GLU）是怎么工作的？
3. SwiGLU MLP 有三个权重矩阵，参数量怎么控制？

---

## 2. GPT-2 风格 MLP（回顾）

第 11 课的 MLP 结构：

```
hidden = GELU(x @ W1)     # [B*T, 4D]
out    = hidden @ W2       # [B*T, D]
```

两个权重矩阵，隐藏层维度 = 4D。

---

## 3. SwiGLU MLP 结构

```
gate   = x @ W_gate        # [B*T, hidden] 门控分支
up     = x @ W_up          # [B*T, hidden] 上投影
hidden = SiLU(gate) ⊙ up   # [B*T, hidden] 门控融合
out    = hidden @ W_down    # [B*T, D]      下投影
```

三个权重矩阵：`W_gate`、`W_up`、`W_down`。

### 3.1 SiLU 激活函数

```
SiLU(x) = x * sigmoid(x)
```

也叫 Swish（Google 命名）。和 GELU 类似，但更平滑：

```
GELU(x) ≈ x * Φ(x)          # Φ 是标准正态 CDF
SiLU(x) = x * σ(x)          # σ 是 sigmoid
```

### 3.2 门控机制（GLU）

```
hidden = SiLU(gate) ⊙ up
```

- `SiLU(gate)` 决定"哪些信息应该通过"（门控值 0~1）
- `up` 是"要通过的信息"
- `⊙` 是逐元素乘法（门控）

**直觉**：就像一个阀门，SiLU(gate) 控制每个维度的"开度"，up 是"水流"。
模型可以学会"这个维度重要，开大一点；那个维度不重要，关小一点"。

---

## 4. 为什么 SwiGLU 比 GELU 好

### 4.1 表达力更强

GELU MLP：`GELU(x @ W1) @ W2` —— 激活函数是固定的，没有门控。

SwiGLU MLP：`SiLU(x @ W_gate) ⊙ (x @ W_up) @ W_down` —— 有两个"视角"：
- `W_gate` 学习"哪些特征重要"
- `W_up` 学习"特征的值"
- 两者融合后，模型可以更精细地控制信息流

### 4.2 实验结果

论文《GLU Variants Improve Transformer》(Shazeer, 2020) 在多个基准上验证：
- SwiGLU > GELU > ReLU（在同等参数量下）
- 效果提升约 0.5-1.0 个困惑度点

### 4.3 现代 LLM 的选择

| 模型 | MLP 激活 |
|------|---------|
| GPT-2/3 | GELU |
| LLaMA / LLaMA 2 / LLaMA 3 | **SwiGLU** |
| Mistral / Mixtral | **SwiGLU** |
| Qwen / Qwen2 | **SwiGLU** |
| PaLM | **SwiGLU** |

---

## 5. 参数量控制

SwiGLU 有三个矩阵（vs GELU 的两个），但通过调整隐藏层维度保持总参数量相近：

```
GPT-2:   W1[D, 4D] + W2[4D, D] = 8D² 参数
SwiGLU:  W_gate[D, h] + W_up[D, h] + W_down[h, D] = 3Dh 参数
```

令 `3Dh ≈ 8D²` → `h ≈ 2.67D`

LLaMA 取 `h = 2/3 * 4D`，再向上取 256 的倍数（方便 GPU 对齐）。

例如 LLaMA-7B：`D = 4096, h = 11008`（`2/3 * 4 * 4096 ≈ 10922`，取 256 的倍数）。

---

## 6. 反向传播

### 6.1 SiLU 的导数

```
d_SiLU/d_x = σ(x) * (1 + x * (1 - σ(x)))
```

推导：`SiLU(x) = x * σ(x)`，用乘积法则：

```
d/dx = σ(x) + x * σ'(x)
     = σ(x) + x * σ(x) * (1 - σ(x))
     = σ(x) * (1 + x * (1 - σ(x)))
```

### 6.2 SwiGLU 的梯度

设 `out = SiLU(gate) ⊙ up`，则：

```
d_out/d_gate = d_out * up * d_SiLU/d_gate
d_out/d_up   = d_out * SiLU(gate)
```

---

## 7. 配置示例

```json
{
  "model": {
    "use_swiglu": true
  }
}
```

---

## 8. 关键要点

- SwiGLU = SiLU 激活 + 门控机制，比 GELU 表达力更强
- 门控让模型学会"哪些特征重要、哪些可以丢弃"
- 三个矩阵（gate/up/down），隐藏维度取 `2/3 * 4D` 保持参数量相近
- LLaMA/Mistral/Qwen/PaLM 全部使用 SwiGLU
