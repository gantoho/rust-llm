# 第 35 课：多 Token 预测（Multi-Token Prediction）

> **本课已落地为可运行代码**：`src/speculative.rs` 的 `MtpHeads`（K 个 `Linear` 头 + 各头交叉熵损失）
> 与 `MtpDrafter`（把 MTP 头当推测解码的草稿）+ `speculative` 子命令的第五节实验。
> 与已有代码的衔接：预测头复用 `src/layers.rs` 的 `Linear` 层，主干隐状态由 `GPT::forward_hidden_cached`
> 提供；训练时只更新各头参数（主干可冻结），推理时一次前向即可给出往后 γ 个位置的分布。

---

## 标准自回归的局限

标准 LLM 训练是 **Next-Token Prediction（NTP）**：给定前缀，预测下一个 token。

$$
\mathcal{L}_{NTP} = -\sum_{t=1}^{T} \log P(x_t | x_{<t})
$$

这有一个根本问题：**每一步只看到"下一步"的监督信号**，模型无法学到"为了未来 5 步都正确，当前这一步应该怎么选"。

> **类比**：下棋时，如果你只能看到下一步的胜负，很难学到深谋远虑的策略；但如果能看到未来 5 步的结果，你就能学到更好的走法。

## Multi-Token Prediction（MTP）

Meta 在 2024 年的论文《Better & Faster Large Language Models via Multi-token Prediction》中提出：

**训练时同时预测未来 K 个 token，而不是只预测 1 个。**

```
输入: "The cat sat on"
       │
       ├──→ Head 1: 预测 "cat"     (next-1)
       ├──→ Head 2: 预测 "sat"     (next-2)
       ├──→ Head 3: 预测 "on"      (next-3)
       └──→ Head 4: 预测 "the"     (next-4)
       │
       Loss = L_1 + L_2 + L_3 + L_4
```

### 架构

```
                共享的 Transformer Backbone
                         │
            ┌────────────┼────────────┐
            │            │            │
       ┌────▼────┐ ┌────▼────┐ ┌────▼────┐
       │ Head 1  │ │ Head 2  │ │ Head 3  │
       │(预测+1) │ │(预测+2) │ │(预测+3) │
       └─────────┘ └─────────┘ └─────────┘
```

关键设计：
- **共享 Backbone**：所有预测头共用同一个 Transformer 主干
- **独立预测头**：每个头是一个独立的线性层（+ 可选的小 Transformer 层）
- **自回归依赖**：Head K 的输入包含了 Head K-1 的预测结果（通过 embedding 层传递）

### 损失函数

$$
\mathcal{L}_{MTP} = \sum_{k=1}^{K} \mathcal{L}_{NTP}^{(k)} = -\sum_{k=1}^{K} \sum_{t=1}^{T-k} \log P_k(x_{t+k} | x_{\leq t})
$$

其中 $P_k$ 是第 $k$ 个预测头的输出分布。

### 训练 vs 推理

| 阶段 | 使用方式 |
|------|---------|
| **训练** | 所有 K 个头同时训练，损失相加 |
| **标准推理** | 只用 Head 1（和标准 NTP 一样） |
| **推测解码** | Head 2~K 作为"草稿模型"，Head 1 做验证 |

> **核心价值**：MTP 训练出的 Backbone 表征质量更高（因为要同时预测多个未来 token，被迫学到更好的"计划"能力），推理时可以和推测解码结合实现加速。

## DeepSeek 的 MTP 实现

DeepSeek-V3 的 MTP 实现更加精细，引入了 **顺序预测模块（Sequential Prediction Module）**：

```
位置 t 的表示 h_t
  │
  ├──→ Head 1 → 预测 x_{t+1}
  │         │
  │         └──→ Embed(x_{t+1}) → 拼接 h_t → MTP Module 2 → Head 2 → 预测 x_{t+2}
  │                                                                    │
  │                                                                    └──→ ...
```

### 与标准 MTP 的区别

| 方面 | Meta MTP | DeepSeek MTP |
|------|----------|-------------|
| 头的输入 | 独立（都来自 backbone） | 串行（后一个头依赖前一个头的输出） |
| 训练时 | 只用于辅助损失 | 同时用于训练和推测解码 |
| 推理时 | 丢弃额外头 | 用作推测解码的草稿头 |

## 为什么 MTP 能提升质量？

### 1. 更好的表征学习

同时预测多个未来 token，迫使模型学到的内部表征必须编码更丰富的语义信息。

类比：
- **NTP**：只看一步 → 学到"这个词后面最常见的词是什么"
- **MTP**：看多步 → 学到"这个句子接下来要表达什么意思"

### 2. 计划能力

模型必须在当前位置就"计划"好接下来几个 token 的输出，类似于人类写作时的"提纲"能力。

### 3. 推理加速

多个预测头可以直接用于推测解码（无需额外的小模型），实现推理加速。

## 实验结果

### Meta 论文结果

在代码生成任务上（HumanEval），MTP 的提升尤为显著：

| 模型 | 标准 NTP | MTP (K=4) | 提升 |
|------|---------|-----------|------|
| 7B | 28.7% | 34.2% | +5.5% |
| 13B | 35.3% | 40.7% | +5.4% |

> **有趣发现**：MTP 对代码生成的提升大于自然语言，可能因为代码有更强的长程依赖和结构化模式。

### DeepSeek-V3

DeepSeek-V3 使用 MTP 作为辅助训练目标，并在推理时用 MTP 头做推测解码，实现约 1.8× 的推理加速。

## 与其他方法的对比

| 方法 | 训练时 | 推理时 | 是否需要额外参数 |
|------|--------|--------|----------------|
| 标准 NTP | 预测 1 个 token | 自回归 | 无 |
| MTP | 预测 K 个 token | 自回归 或 推测解码 | K-1 个预测头 |
| 推测解码 | 无变化 | 小模型猜 + 大模型验 | 需要额外小模型 |
| Medusa | 加多头 | 多头同时猜 + 验证 | 多个预测头 |

## 实现要点

### 预测头的参数效率

预测头不应太重——如果每个头都是一个完整 Transformer 层，参数量会暴增。

推荐配置：
```
Head K = Linear(d_model → d_model) → TransformerLayer(1层) → Linear(d_model → vocab_size)
```

或更轻量：
```
Head K = Linear(d_model → vocab_size)  // 最简单，效果也不错
```

### 梯度累积

MTP 的 K 个损失会累加，总梯度比 NTP 大 K 倍。需要：
- 调低学习率，或
- 对每个头的损失做 $1/K$ 的缩放

## 本项目的实际实现

上面的原理已全部落地为可运行代码。MTP 相关组件与第 34 课共用 [`src/speculative.rs`](../src/speculative.rs)（8 个单测），
CLI 入口是 [`speculative`](../README.md#12-quant--distributed--align--rag--speculative--第-3338-课实验) 子命令（通过 `--mtp-heads` / `--mtp-steps` 控制）。

### 代码结构

| 组件 | 位置 | 说明 |
|------|------|------|
| `MtpHeads` | `speculative.rs` | K 个 `Linear` 预测头：第 k 个头把位置 i 的隐状态映射到位置 i+k+1 的 token logits。提供 `new(k, n_embd, vocab)`、`logits(hidden)`（返回 K+1 个 logits 张量）、`loss(hidden, targets)`（各头交叉熵之和的均值） |
| `MtpDrafter` | `speculative.rs` | 拿 MTP 头当推测解码的草稿来源：实现 `Drafter` trait，一次前向即可给出往后 γ 个位置的候选分布（不必自回归），但各分布彼此条件独立，接受率通常低于小模型草稿 |
| `Drafter` trait | `speculative.rs` | 草稿来源统一接口：`propose()`/`commit()`/`reset()`，`MtpDrafter` 和 `ModelDrafter` 均实现此 trait |
| `SpecDecoder` | `speculative.rs` | 推测解码执行体，可接受任意 `Drafter` 实现（包括 `MtpDrafter`） |

### CLI 用法

```bash
# 训练 MTP 头并用于推测解码（K=4 头，80 步训练）
cargo run --release -- speculative --mtp-heads 4 --mtp-steps 80

# 调整 MTP 头数（K=2 头，更轻量）
cargo run --release -- speculative --mtp-heads 2 --mtp-steps 80

# 对比：用小模型草稿 vs MTP 头草稿（不传 --mtp-heads 即用 ModelDrafter）
cargo run --release -- speculative --gamma 4 --max-new 48
```

---

## 拓展方向

> 核心内容已全部实现（`src/speculative.rs` 的 `MtpHeads` / `MtpDrafter`），这里是进阶拓展。

1. **实现 MTP 训练**：在现有 GPT 模型上添加 3 个额外预测头，实现 K=4 的多 token 预测。
2. **对比实验**：在相同数据和步数下，分别用 NTP 和 MTP 训练小模型，对比 loss 曲线和生成质量。
3. **MTP + 推测解码**：用 MTP 的 Head 2~K 作为草稿头，Head 1 做验证，实现推测解码。
4. **代码 vs 文本**：分别在代码和自然语言数据上训练 MTP，观察哪种数据获益更大。
