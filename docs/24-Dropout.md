# 第 24 课：Dropout —— 训练时的"随机失忆"

> 代码位置：[src/tensor.rs](src/tensor.rs)（`Tensor::dropout` 算子）
> 代码位置：[src/model.rs](src/model.rs)（`TransformerBlock` / `GPT` 中的 dropout 调用）
>
> 配置开关：`config.json` → `model.dropout: 0.1`（推荐值 0.1~0.3）

---

## 1. 本课要搞懂的问题

1. 什么是 Dropout？为什么训练时要随机"丢掉"一些神经元？
2. 为什么推理时不需要 Dropout？
3. 什么是"反转 Dropout"（inverted Dropout）？为什么所有框架都用它？
4. Dropout 应用在 Transformer 的哪些位置？

---

## 2. 过拟合问题

训练时 loss 很低，验证时 loss 很高 —— 模型"记住"了训练数据，而不是"学会"了规律。

**原因**：模型参数太多，训练数据太少，模型可以靠"死记硬背"来拟合训练集。

**Dropout 的解决思路**：训练时随机让一部分神经元"失忆"（输出置零），强迫其他神经元独立工作，
不能过度依赖某几个"明星神经元"。

---

## 3. Dropout 的工作原理

### 3.1 训练时

对每个元素，以概率 `p` 将其置零：

```
mask = Bernoulli(1 - p)    # 每个元素独立采样，1=保留，0=丢弃
out  = x * mask / (1 - p)  # 缩放保证期望不变
```

### 3.2 推理时

不做任何操作，直接输出：

```
out = x
```

### 3.3 为什么要除以 `(1-p)`？

假设 `p=0.5`，训练时一半元素被丢弃。如果不缩放，输出的期望只有原来的一半。
除以 `(1-p)` 保证 `E[out] = E[x]`，训练和推理的数值范围一致。

这就是"反转 Dropout"（inverted Dropout）：训练时缩放，推理时不缩放。

---

## 4. 反向传播

Dropout 的反向非常简单：梯度同样乘以 mask（和前向一样的缩放）。

```
d_x = d_out * mask / (1 - p)
```

被丢弃的位置（mask=0），梯度也是 0 —— 这些神经元在本步不参与学习。

---

## 5. Dropout 在 Transformer 中的位置

GPT-2 论文中 Dropout 应用在两个地方：

```
x = x + Dropout(Attention(LN(x)))    # 注意力输出后
x = x + Dropout(MLP(LN(x)))          # MLP 输出后
```

另外，嵌入层也可以加 Dropout：

```
x = Dropout(TokenEmbedding(tokens))
```

### 注意事项

- **残差连接之后**：Dropout 在残差加法之前（不是之后）
- **不在 LN 内部**：LayerNorm/RMSNorm 本身不加 Dropout
- **推理时关闭**：`model.eval()` 模式下 Dropout 自动关闭
- **预训练 vs 微调**：大模型预训练通常用 Dropout=0.1；微调时可能关掉（数据够多时）

---

## 6. 实现要点

### 6.1 mask 生成

用项目自带的 xorshift64* RNG 生成随机 mask，保证可复现（固定种子）。

### 6.2 训练/推理切换

`Tensor::dropout` 接受 `training: bool` 参数：
- `training = true`：生成随机 mask，应用 Dropout
- `training = false`：直接返回输入的克隆（恒等操作）

`model.forward()` 调用时传入 `training` 参数，由训练循环控制。

### 6.3 形状保持

Dropout 不改变张量形状，输入和输出形状完全一致。

---

## 7. 配置示例

```json
{
  "model": {
    "dropout": 0.1
  }
}
```

`dropout: 0` 表示不使用 Dropout（默认值）。

---

## 8. 关键要点

- Dropout 训练时随机置零 + 缩放，推理时恒等
- 反转 Dropout：训练时除以 (1-p)，推理时不用缩放
- 在 Transformer 中用于嵌入层和每个子层的输出
- 防止过拟合，但数据量足够大时可以关掉
- 反向传播：梯度同样乘以 mask
