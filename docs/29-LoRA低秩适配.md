# 第 29 课：LoRA 低秩适配 —— 用 0.1% 的参数微调大模型

> 代码位置：[src/layers.rs](src/layers.rs)（`LoRA` 层、`inject_lora` 辅助函数）
>
> 算法论文：*LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)

---

## 1. 本课要搞懂的问题

1. 为什么大模型微调不需要更新全部参数？
2. LoRA 是怎么用低秩矩阵近似权重更新的？
3. 为什么 B 初始化为全零、A 初始化为正态？
4. α 缩放因子是干什么的？

---

## 2. 全量微调的问题

LLaMA-7B 有 70 亿参数，全量微调需要：

```
参数显存：7B * 4B = 28GB（FP32）
优化器状态：7B * 8B = 56GB（AdamW 的 m 和 v）
梯度：7B * 4B = 28GB
总计：~112GB
```

单卡放不下，而且微调数据通常很少（几千条），全量更新容易过拟合。

---

## 3. LoRA 的核心思想

**关键假设**：微调时的权重变化 ΔW 是**低秩**的（信息集中在少数方向）。

```
ΔW = B · A
```

其中：
- B ∈ R^{out × r}：上投影矩阵
- A ∈ R^{r × in}：下投影矩阵
- r ≪ min(in, out)：秩（通常 4-64）

### 3.1 参数量对比

```
全量微调：in × out 参数
LoRA：    r × (in + out) 参数

例：in=4096, out=4096, r=16
全量：4096 × 4096 = 16,777,216（16M）
LoRA：16 × (4096 + 4096) = 131,072（131K）→ 0.78%
```

---

## 4. 前向计算

```
y = x @ (W + ΔW) = x @ W + x @ (B · A) = x @ W + (x @ B) @ A
```

- W 是预训练权重（**冻结**，不更新）
- B 和 A 是可训练参数
- 推理时可以把 ΔW 合并到 W 里：`W' = W + B·A`，没有额外开销

---

## 5. 初始化策略

### 5.1 A：正态分布 N(0, σ²)

```
σ = 1 / √r
```

保证初始 ΔW = BA 的方差不会太大。

### 5.2 B：全零

```
B = 0 → ΔW = BA = 0
```

**关键**：训练开始时 ΔW = 0，模型行为和预训练模型完全一致。
这保证了微调不会"破坏"预训练学到的知识。

### 5.3 为什么不能反过来？

如果 A = 0, B = 正态，则 ΔW = B·0 = 0（一样）。
但如果 A = 正态, B = 正态，则 ΔW ≠ 0，模型一开始就会偏离预训练，训练不稳定。

---

## 6. α 缩放因子

```
ΔW = (α / r) · B · A
```

- α 通常设为 r（即不缩放，scaling = 1）
- 增大 α = 放大 LoRA 的影响（学习率等效变大）
- 减小 α = 缩小 LoRA 的影响（更保守的微调）

**使用建议**：先用 α = r，如果效果不好再调整。

---

## 7. 应用场景

### 7.1 LoRA 的典型用法

```rust
// 1. 加载预训练模型
let model = load_pretrained("llama-7b");

// 2. 给 Q/K/V 投影注入 LoRA
let lora_q = inject_lora(&model.attn.c_q, rank=16, alpha=16.0, &mut rng);
let lora_k = inject_lora(&model.attn.c_k, rank=16, alpha=16.0, &mut rng);
let lora_v = inject_lora(&model.attn.c_v, rank=16, alpha=16.0, &mut rng);

// 3. 只训练 LoRA 参数（冻结的 W 不参与梯度计算）
let trainable = [lora_q.a, lora_q.b, lora_k.a, lora_k.b, ...];
let optimizer = AdamW::new(trainable, lr=1e-4);

// 4. 推理时合并：W' = W + (α/r) · B · A
```

### 7.2 通常给哪些层加 LoRA

| 层 | 是否加 LoRA | 原因 |
|---|-----------|------|
| Q/K/V 投影 | ✅ | 注意力是微调的核心 |
| 输出投影 | ✅ | 影响注意力输出 |
| MLP | 可选 | 效果有限 |
| Embedding | ❌ | 词表变化少 |
| LayerNorm | ❌ | 参数太少 |

---

## 8. 集成状态

`LoRA` 层和 `inject_lora` 辅助函数已完整实现（`layers.rs`），并通过 `finetune` 子命令接入 CLI。

### 8.1 命令行用法

```bash
# 基础 LoRA 微调（rank=16, alpha=16, 1000 步）
cargo run --release -- finetune --config config.json --pretrained checkpoints/best.ckpt

# 自定义 LoRA 参数
cargo run --release -- finetune --config config.json --pretrained checkpoints/best.ckpt \
    --lora-rank 32 --lora-alpha 32 --steps 2000 --lr 5e-5
```

### 8.2 配置文件方式

也可以在 `config.json` 中配置 LoRA：

```jsonc
{
  "train": {
    "lora": {
      "rank": 16,
      "alpha": 16.0
    }
  }
}
```

### 8.3 工作流程

1. 加载预训练 checkpoint（`--pretrained` 参数）
2. 冻结主模型所有参数
3. 注入 LoRA 适配层（Q/K/V 投影）
4. 只训练 LoRA 参数（A 和 B），学习率通常为 1e-4
5. 保存 checkpoint（包含 LoRA 参数）
6. 推理时正常使用（LoRA 增量已融入前向计算）

---

## 9. 关键要点

- LoRA 用低秩矩阵 ΔW = BA 近似权重更新，可训练参数量降到 0.1%~1%
- 冻结预训练权重，只训练 A 和 B
- B 初始化为全零，保证训练开始时模型行为不变
- 推理时可以把 ΔW 合并到 W，没有额外开销
- 适合"在已有大模型上微调特定任务"的场景
