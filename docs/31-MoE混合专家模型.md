# 第 31 课：MoE 混合专家模型（Mixture of Experts）

> **本课为前沿技术教程（纯文档，无配套代码实现）。**
> 每课末尾的"动手练习"可作为代码实现的指引。
> 与本项目已有代码的关联：MoE 层可复用 `src/layers.rs` 中的 `MLPEnum`（SwiGLU）作为专家网络。

---

## 为什么需要 MoE？

当模型参数量从 7B 增长到 175B（GPT-3）乃至万亿级别时，**每个 token 都经过全部参数**的 Dense 模型面临两个根本瓶颈：

1. **计算成本**：FLOPs 与参数量成正比，训练一个 175B 模型需要数千 GPU 运行数月。
2. **推理延迟**：每生成一个 token 都要跑完所有参数，延迟不可接受。

MoE 的核心思想是**稀疏激活**——模型有很多参数（专家），但每次只激活其中一小部分。

> **类比**：Dense 模型像一个"全能医生"什么都要看；MoE 像一家"医院"——有内科、外科、眼科等专家，门急诊（Router）根据症状把你分到对应科室，你只占用其中一个专家的时间。

## 数学定义

设模型有 $N$ 个专家网络 $\{E_1, E_2, ..., E_N\}$，输入为 $x$：

$$
y = \sum_{i=1}^{N} g_i(x) \cdot E_i(x)
$$

其中 $g(x)$ 是**门控网络（Router / Gate）**，输出一个概率分布，决定每个专家的权重。

### 稀疏 MoE（Switch Transformer / Mixtral 风格）

不是所有专家都参与——只选 Top-K 个（通常 K=1 或 K=2）：

$$
\text{TopK}(g(x)) = \text{softmax}(\text{TopK}(W_g \cdot x))
$$

- **K=1**：每个 token 只走一个专家（Switch Transformer）
- **K=2**：每个 token 走两个专家（Mixtral 8x7B、Grok-1）

## 架构详解

```
输入 x
  │
  ├──→ Router (门控网络): W_g @ x → logits → TopK → weights
  │         │
  │         ├──→ Expert 0 (FFN): W_up / W_gate / W_down
  │         ├──→ Expert 1 (FFN)
  │         ├──→ ...
  │         └──→ Expert N-1 (FFN)
  │
  └──→ 加权求和 → y = Σ weight_i * Expert_i(x)
```

### 关键组件

| 组件 | 说明 |
|------|------|
| **Router / Gate** | 一个线性层：`W_g ∈ R^{d_model × n_experts}`，输入 token 表示，输出各专家的得分 |
| **Expert** | 通常是 FFN（SwiGLU），每个专家是独立的 MLP |
| **Auxiliary Loss** | 辅助损失，防止路由器把所有 token 都送到同一个专家（负载均衡） |

## 负载均衡（Load Balancing）

MoE 最大的工程挑战是**专家负载不均**——路由器倾向于反复选同一个"好"专家（赢者通吃），导致其他专家闲置。

### Switch Transformer 的辅助损失

$$
\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot p_i
$$

其中：
- $f_i$ = 专家 $i$ 被分配到的 token 比例（实际负载）
- $p_i$ = 专家 $i$ 的平均路由概率
- $\alpha$ = 系数（通常 0.01）
- $N$ = 专家数

**直觉**：当所有专家的 $f_i$ 和 $p_i$ 都相等时，$\mathcal{L}_{aux}$ 最小（完美均衡）。

### Mixtral 的实现

Mixtral 8x7B 没有显式辅助损失，而是靠 **Top-2 + softmax** 的隐式均衡（两个专家分担一个 token 的权重，自然比 Top-1 更分散）。

## 真实 MoE 模型对比

| 模型 | 专家数 | 激活专家 | 总参数 | 激活参数 | 说明 |
|------|--------|----------|--------|----------|------|
| Switch Transformer | 128 | 1 | 1.6T | ~1/128 | Google，2021 |
| Mixtral 8x7B | 8 | 2 | 46.7B | 12.9B | Mistral AI，2023 |
| Grok-1 | 8 | 2 | 314B | ~86B | xAI，2024 |
| DeepSeek-V2 | 160 | 6 | 236B | 21B | DeepSeek，2024 |
| Qwen2-57B-A14B | 64 | 8 | 57B | 14B | Alibaba，2024 |

> **核心洞察**：Mixtral 8x7B 总参数 46.7B，但推理时只激活 12.9B（约 27%），性能接近 LLaMA-2 70B，但推理成本只有其约 1/5。

## 实现要点

### 1. Router 实现

```rust
pub struct Router {
    pub gate: Linear,  // [n_experts, d_model] 或 [d_model, n_experts]
    pub n_experts: usize,
    pub top_k: usize,
}

impl Router {
    pub fn forward(&self, x: &Tensor) -> (Tensor, Vec<usize>) {
        // logits = x @ W_g
        let logits = self.gate.forward(x);
        // TopK 选择
        let (weights, indices) = topk(logits, self.top_k);
        // 在被选中的专家上做 softmax
        let weights = softmax(weights);
        (weights, indices)
    }
}
```

### 2. MoE Layer

```rust
pub struct MoELayer {
    pub router: Router,
    pub experts: Vec<FFN>,  // N 个独立的 FFN
}

impl MoELayer {
    pub fn forward(&self, x: &Tensor) -> Tensor {
        let (weights, indices) = self.router.forward(x);
        let mut y = Tensor::zeros(x.shape());
        for (k, &expert_idx) in indices.iter().enumerate() {
            let expert_out = self.experts[expert_idx].forward(x);
            y = y + weights[k] * expert_out;
        }
        y
    }
}
```

### 3. Token 路由的工程挑战

实际实现中，MoE 面临的关键工程问题是**并行效率**：

- **Expert Parallelism**：不同专家放在不同 GPU 上，token 通过 All-to-All 通信发送到对应专家
- **Token Dropping**：超过专家容量的 token 被丢弃（Switch Transformer 的做法）
- **Capacity Factor**：每个专家处理 token 的上限，通常设为平均负载的 1.25× ～ 2×

## MoE 的优缺点

| 优点 | 缺点 |
|------|------|
| 训练/推理 FLOPs 远小于同参数量 Dense 模型 | 显存占用仍是全量参数（所有专家都要加载） |
| 可以在不增加推理成本的前提下扩大模型容量 | 负载不均导致部分专家"浪费" |
| 不同专家可自发学到不同"专长" | 需要 All-to-All 通信，对网络带宽要求高 |
| 适合大规模预训练 | 微调时容易过拟合（只更新部分专家） |

## 动手练习

1. **实现 Router**：用 `Linear` 层实现门控网络，输入 token 向量，输出 Top-K 专家索引和权重。
2. **实现 MoE Layer**：将 Router 和 N 个 FFN 组合，实现稀疏前向传播。
3. **负载均衡实验**：训练一个简单的 MoE-MLP，观察各专家的被选频率，加入辅助损失后观察均衡效果。
4. **专家分化观察**：在小数据集上训练 MoE，检查不同专家是否学到了不同的模式。
