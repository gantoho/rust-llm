# 第 36 课：RLHF 与人类对齐（Alignment）

> **本课为前沿技术教程（纯文档，无配套代码实现）。**
> 与本项目已有代码的关联：SFT 阶段可复用 `src/train.rs` 的训练循环和 `src/loss.rs` 的交叉熵损失；
> DPO 损失可基于 `src/tensor.rs` 的现有算子（log_softmax、sigmoid）实现。

---

## 为什么需要对齐？

预训练模型的目标是**预测下一个 token**，但这个目标和"对人类有用、无害、诚实"之间存在巨大鸿沟：

- 预训练后，模型可能会生成有害内容、编造事实（幻觉）、不遵循指令
- 模型学到的是"互联网上什么样的文本最常见"，而不是"什么样的回答最好"

**对齐（Alignment）** 就是让模型的行为符合人类的价值观和偏好。

## 三阶段训练流程

现代 LLM 的训练通常分为三个阶段：

```
阶段 1: 预训练 (Pre-training)
  目标: 学习语言能力（预测下一个 token）
  数据: 互联网文本（万亿 token）
  ↓

阶段 2: 监督微调 (Supervised Fine-Tuning, SFT)
  目标: 学习"如何回答问题"的格式
  数据: 人工标注的 (问题, 回答) 对（万~十万条）
  ↓

阶段 3: 对齐 (Alignment)
  目标: 学习"什么是好回答"
  方法: RLHF / DPO 等
  数据: 人工排序的 (好回答, 差回答) 对
```

## 监督微调（SFT）

SFT 是最简单的对齐方式：直接用高质量的 (指令, 回答) 数据微调模型。

```
训练样本:
  Input:  "请解释什么是量子计算"
  Target: "量子计算是利用量子力学原理进行计算的技术..."

Loss: CrossEntropy(模型输出, Target tokens)
```

**关键难点**：
- 数据质量 > 数据数量：少量高质量数据 >> 大量低质量数据（LIMA 论文：仅 1000 条精心标注的数据就能达到很好的 SFT 效果）
- 回答风格需要一致（不能有时详细、有时简略）

## RLHF（Reinforcement Learning from Human Feedback）

### 核心思想

1. 训练一个**奖励模型（Reward Model, RM）**，学习人类对回答质量的偏好
2. 用强化学习（PPO）优化 LLM，使其生成奖励模型打高分的回答

### 步骤详解

#### 步骤 1：收集偏好数据

对同一个问题，让 SFT 模型生成多个回答，由人工排序：

```
问题: "如何做番茄炒蛋？"

回答 A: "先打鸡蛋，然后..."（详细、正确）
回答 B: "把东西放锅里炒"（过于简略）
回答 C: "番茄炒蛋的量子力学原理是..."（离谱）

人类排序: A > B > C
```

#### 步骤 2：训练奖励模型

奖励模型的目标：对好回答打高分，对差回答打低分。

使用 **Bradley-Terry 模型**，损失函数：

$$
\mathcal{L}_{RM} = -\log \sigma(r_\theta(x, y_w) - r_\theta(x, y_l))
$$

其中：
- $y_w$ 是人类偏好的"更好"回答
- $y_l$ 是"更差"回答
- $r_\theta(x, y)$ 是奖励模型对 (问题 x, 回答 y) 的打分
- $\sigma$ 是 sigmoid 函数

**直觉**：让好回答的分数比差回答的分数高出足够多（sigmoid 的输入越大，loss 越小）。

#### 步骤 3：PPO 强化学习优化

用 Proximal Policy Optimization（PPO）算法优化 LLM 的生成策略：

$$
\mathcal{L}_{PPO} = \mathbb{E}_{(x,y) \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot D_{KL}(\pi_\theta \| \pi_{\text{ref}}) \right]
$$

其中：
- $\pi_\theta$ 是当前 LLM 策略
- $\pi_{\text{ref}}$ 是 SFT 后的参考模型（防止模型偏离太远）
- $r_\phi$ 是奖励模型的打分
- $\beta$ 是 KL 惩罚系数（控制"不要太偏离原始模型"）

**KL 惩罚的作用**：防止模型"hack"奖励模型——如果不加 KL 约束，LLM 会找到奖励模型的漏洞（如反复说"我非常确定"来得高分），而不是真正提高回答质量（这叫 reward hacking）。

### PPO 算法核心

PPO 是一种**策略梯度**方法，核心是 clipped surrogate objective：

$$
L^{CLIP} = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]
$$

其中：
- $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)}$ 是新旧策略的概率比
- $\hat{A}_t$ 是优势函数估计（advantage）
- $\epsilon$ 是裁剪范围（通常 0.1~0.2）

**直觉**：不要让策略更新太大——如果新策略和旧策略差异超过 $\epsilon$，就用 clip 截断梯度。

### RLHF 的完整训练流程

```
                ┌─────────────┐
                │  预训练 LLM  │
                └──────┬──────┘
                       │ SFT
                ┌──────▼──────┐
                │   SFT 模型   │──────────────────┐
                └──────┬──────┘                   │
                       │                          │ (作为参考模型 π_ref)
                       │ 生成多个回答              │
                       │ + 人工排序                │
                ┌──────▼──────┐                   │
                │  奖励模型 RM │                   │
                └──────┬──────┘                   │
                       │ 打分                     │
                ┌──────▼──────┐                   │
                │ PPO 优化 LLM │◄──────────────────┘
                └──────┬──────┘     KL 惩罚
                       │
                ┌──────▼──────┐
                │ 对齐后 LLM   │
                └─────────────┘
```

## DPO（Direct Preference Optimization，2023）

### 为什么需要 DPO？

RLHF 的 PPO 阶段非常复杂：
- 需要训练单独的奖励模型
- PPO 训练不稳定，超参数敏感
- 需要同时维护 4 个模型（策略、参考、奖励、价值），显存开销大

DPO 的核心思想：**跳过奖励模型，直接用偏好数据优化 LLM**。

### 数学推导

DPO 从 RLHF 的目标函数出发，经过数学推导，得到一个**闭式解**：

$$
\mathcal{L}_{DPO} = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{ref}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{ref}(y_l|x)} \right) \right]
$$

其中：
- $\pi_\theta$ 是当前模型
- $\pi_{ref}$ 是参考模型（通常是 SFT 后的模型）
- $y_w$ 是偏好回答（chosen）
- $y_l$ 是非偏好回答（rejected）
- $\beta$ 是温度参数（通常 0.1~0.5）

**直觉**：增大好回答相对于参考模型的概率比，同时减小差回答的概率比。

### DPO vs RLHF 对比

| 方面 | RLHF (PPO) | DPO |
|------|-----------|-----|
| 是否需要奖励模型 | 是 | 否 |
| 训练稳定性 | 不稳定，超参敏感 | 稳定，类似 SFT |
| 显存占用 | 4 个模型 | 2 个模型（策略 + 参考） |
| 实现复杂度 | 高 | 低 |
| 性能 | 被认为更好（但差距在缩小） | 接近 RLHF |

## 其他对齐方法

### RLAIF（RL from AI Feedback）

用 AI 代替人类标注偏好数据（如用 GPT-4 判断哪个回答更好），大幅降低标注成本。

### GRPO（Group Relative Policy Optimization，DeepSeek）

DeepSeek 提出的简化 PPO 方法：
- 不需要单独的价值模型（Critic）
- 用同一组采样的相对奖励作为 baseline
- 训练更稳定，资源消耗更少

$$
\mathcal{L}_{GRPO} = \mathbb{E}_{q, \{o_i\}} \left[ \frac{1}{G} \sum_{i=1}^{G} \min\left( r_i(\theta) \hat{A}_i, \text{clip}(r_i(\theta)) \hat{A}_i \right) - \beta D_{KL} \right]
$$

其中 $\hat{A}_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\})}$ 是组内相对优势。

### Constitutional AI（Anthropic）

让 AI 自己根据一组"宪法"原则来修订和改进回答，减少人工参与。

## 动手练习

1. **理解 SFT Loss**：在已有的 CrossEntropy 基础上，实现一个简单的 SFT 训练循环（用 (指令, 回答) 数据）。
2. **实现 Bradley-Terry Loss**：给定 (好回答分数, 差回答分数)，实现奖励模型的损失函数。
3. **实现 DPO Loss**：给定 (策略模型概率, 参考模型概率, 好回答, 差回答)，实现 DPO 损失函数。
4. **对比实验**：在小模型上分别用 SFT 和 DPO 训练，观察生成质量的变化。
