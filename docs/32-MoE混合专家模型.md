# 第 32 课：MoE 混合专家模型（Mixture of Experts）

> **本课已落地为可运行代码**：核心实现在 [`src/moe.rs`](../src/moe.rs)（1395 行，含 18 个单测），
> 接线在 [`src/model.rs`](../src/model.rs)（`TransformerConfig.n_expert` / `moe_top_k` 等 9 个字段 +
> `Ffn` 枚举二选一），演示在 `cargo run --release -- moe`（见文末「本项目实现」）。
> 专家网络直接复用 `src/layers.rs` 的 `MLPEnum`（GELU 或 SwiGLU）。

---

## 为什么需要 MoE？

当模型参数量从 7B 增长到 175B乃至万亿级别时，**每个 token 都经过全部参数**的 Dense 模型面临两个根本瓶颈：

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

> **⚠️ 一个常见误读（本课实测修正过）**：$f_i$ 来自 argmax，是**常数、不可导**；$p_i$ 是可导的
> softmax 平均概率。所以**当 $p_i$ 均匀时，无论 $f$ 长什么样都有 $\mathcal{L}_{aux} = 1$**
> （$\sum f_i p_i = \frac{1}{N}\sum f_i = \frac{1}{N}$）。也就是说「$\mathcal{L}_{aux}$ 降到 1」
> **不是**负载均衡的证书——硬路由可以照样压在少数专家上。另外 $\mathcal{L}_{aux}$ 的**下界不是 1**：
> $f$ 与 $p$ 支撑集不交时 $\sum f_i p_i = 0$，$\mathcal{L}_{aux}$ 可以低到 0。
> 详见文末「本项目实现」里的隔离实验。

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

## 本项目实现

核心代码在 [`src/moe.rs`](../src/moe.rs)，接线在 [`src/model.rs`](../src/model.rs)，
演示命令是 `cargo run --release -- moe`（[`src/main.rs`](../src/main.rs) 的 `cmd_moe`）。
18 个单测全部与 CPU 参考实现对齐（其中 4 个专测 aux-loss-free 均衡偏置与共享专家）。

### 公开 API

| 项目 | 说明 |
|------|------|
| `top_k_gate` | 纯函数路由：`logits → Top-K`（并列按下标、**确定性**），返回 `RoutePlan`；选择用 `select_nth_unstable_by` O(E) 部分选择 + 对前 K 排序，不做全排序 |
| `RoutePlan` | 一次路由的完整结果：`sel`（各专家实际处理的 token 行号，已按容量截断）、`penalty`（未选中 −∞）、`mask`（选中 1 / 其余 0）、`f`（分配比例）、`dropped` / `routed` |
| `expert_capacity` | `ceil(cf · n·K/E)`；`cf ≤ 0` → `usize::MAX`（不限） |
| `MoELayer` | 专家集合 + 路由器（+ 可选的共享专家 `shared_experts`）；`forward` / `forward_with_aux` / `route_plan` / `stats` / `update_balance_bias`（推进 aux-loss-free 偏置）/ `balance_bias`（只读诊断） |
| `RouteStats` | 诊断：各专家被路由到的次数、`aux`、`imbalance()`、`dropped_ratio()` |
| `sparse_stats` | 参数 / 激活量口径：`total_params` / `active_params` / `param_ratio` / `flops_saving`；末位参数 `n_shared` 决定共享专家 `shared_params()`（同时计入总参数与激活参数，见下文「共享专家」） |

### 稀疏前向：gather → expert → weighted → scatter

逐专家跑，而不是「全跑一遍再用掩码筛掉」：

1. `gather_rows(x2, &sel[e])` 取出分到专家 `e` 的 token（`[n_e, d]`）；
2. 过该专家的 FFN（`MLPEnum`，GELU / SwiGLU 与 `TransformerConfig` 保持一致）；
3. 取门控权重矩阵 `w` 的第 `e` 列——实现上是 `w.gather_rows(sel).select_col(e)`，
   两步都是**带梯度的索引算子**（直接读显存会把送回路由器的那条梯度掐断，
   早先的 `w.matmul(one_hot_col(e, E))` 语义等价但要多算 n×E 次乘加）；
4. `we.mul(&ye)` 加权（`[n_e,1]` 广播到 `[n_e,d]`）；
5. `scatter_add_rows(sel, n)` 散射回原位累加。

`sel[e]` 为空就 `continue` 整段跳过——稀疏激活省下的正是这一段。
全部 token 的分配都被容量丢掉时输出是全零张量，这正是「被丢弃」的正确语义（该层贡献为 0、
残差照常直通）。

### 门控口径：K = 1 的梯度陷阱

由 `TransformerConfig.moe_switch_gate` 切换：

| 口径 | 做法 | `Σw` | K = 1 时 | 代表模型 |
|------|------|------|----------|----------|
| **重归一化**（默认） | 未选中置 −∞ 后 softmax | 1 | `w ≡ 1`，**对 logits 的雅可比恒为 0** | Mixtral / DeepSeek / Qwen |
| **原概率** | 全部专家上的 softmax 原概率，未选中置 0 | < 1 | `w = p_0`（非 0 非 1），路由器拿得到梯度 | Switch Transformer |

第一条看似天经地义（`w_j = 0` 时 `∂w_j/∂logit_i = w_j(δ_ij − w_i) ≡ 0`，梯度天然干净），
但 K = 1 时只剩一个非零权重、而它必然等于 1，雅可比整体是 0——**主损失给不了路由器任何梯度**。
单测 `test_k1_renorm_gate_has_no_router_gradient` 直接对比了两条口径下路由器的梯度范数：
同一输入下重归一化 < 1e-7，原概率 > 1e-4。所以 K = 1 必须配 `switch_gate = true`，
否则路由器只能靠辅助损失训练。

### 负载均衡辅助损失：实测到的退化解

`L_aux = α·E·Σ f_i·p_i` 的性质用 `moe` 子命令第二节的**隔离实验**验证过。这里有**三组对照**，
从**同一个塌缩起点**（前 K 个专家的门控偏置 +3.0 ⇒ 负载全压在它们身上）出发，
分别 ① 什么都不做 ② 只优化 `α·L_aux` ③ 只推进 aux-loss-free 均衡偏置：

```text
K=1（E=8）
① 不均衡（塌缩起点）：不均衡度 8.00（用 1/8），L_aux 5.932
② α·L_aux（300 步梯度）：不均衡度 3.12（用 7/8，比 ① 摊平 69.8%），L_aux 5.932 → 1.053
③ sign 偏置（2000 步、γ = 0.001、不算梯度）：不均衡度 1.32（用 8/8，比 ① 摊平 95.5%），L_aux 0.929
K=2（E=8）
① 不均衡（塌缩起点）：不均衡度 4.00（用 2/8），L_aux 3.114
② α·L_aux（300 步梯度）：不均衡度 3.81（用 7/8，比 ① 摊平 59.8%），L_aux 3.114 → 1.025
③ sign 偏置（2000 步、γ = 0.001、不算梯度）：不均衡度 1.11（用 8/8，比 ① 摊平 98.4%），L_aux 1.036
```

`L_aux` 确实被压到 1.0，但**硬路由最多只被摊平一部分**：K = 2 组 4.00 → 3.81 几乎原地不动；
K = 1 组 8.00 → 3.12 看着动了，其实是梯度顺手把路由器权重从 0 抬了起来（② 组的权重是被清零的
——见下），而**一次梯度都不算**的 ③ 组反而压到了 1.32 / 1.11。原因就是软/硬错配：

- `f` 由 argmax 给出，不可导；`p` 由 softmax 给出，可导。
- 梯度 `∂L/∂logit_j ∝ p_j·(f_j − Σ f_i p_i)` 的**大小正比于 `p_j`**：`p` 越平梯度越小。
  于是最快的下降路径是**先把 `p` 摊平**（`L_aux` 一步到位到 1），而不是把硬路由摊平。
- 而 `p` 均匀时 `Σ f_i p_i = (1/E)·Σ f_i = 1/E` ⇒ `L_aux = 1`，**与 `f` 无关**。

所以「`L_aux` 掉到 1」**不是**负载均衡的证书；并且 1 也**不是下界**——`f` 与 `p` 支撑集不交时
`Σ f_i p_i = 0`，`L_aux` 可以低到 0（`test_aux_loss_minimum_at_uniform` 覆盖了这三种情形：
`p = f` 时 `≥ α`、支撑集不交时 `= 0`、`p` 均匀时 `= 1`）。

这正是后续工作（DeepSeek-V3 的 loss-free 均衡偏置、expert-choice 路由）要绕开的软/硬错配，
也是「辅助损失系数要调小」的真实原因：它压的是概率分布，不是分配结果。

### aux-loss-free 均衡偏置（DeepSeek-V3）

软辅助损失的病根是"软/硬错配"：`L_aux` 压的是可导的**路由概率** `p`，而真正决定负载的是
不可导的**硬路由** argmax。DeepSeek-V3 换了个思路：**不动损失、不动梯度，直接给每个专家加一个
只用于选路的偏置 `b_i`**。选 Top-K 时看 `logits + b`，门控权重、`L_aux`、z-loss 一律仍看原始
`logits`——于是偏置既不进梯度也不进损失，均衡"免费"。

**更新规则**（目标负载取平均 `Σcount_i / E`，按**符号**更新，固定步长 γ）：

```text
target = Σ_i count_i / E
count_i > target  ⇒  b_i ← b_i − γ      # 过载的专家：压低它的选路得分
count_i < target  ⇒  b_i ← b_i + γ      # 欠载的专家：抬高它的选路得分
```

用**符号**而不是按失衡幅度成比例，是因为偏置的职责只是"把负载推平、推平了就停住"；
按幅度更新会在接近均衡时来回过冲、长期震荡（DeepSeek-V3 用的就是 sign）。

**与 `α·L_aux` 的对照**：

| 维度 | `α·L_aux`（软辅助损失） | aux-loss-free 偏置 |
|------|------------------------|--------------------|
| 作用对象 | 路由**概率** `p`（可导） | **硬路由** argmax（`logits + b` 的排序） |
| 是否进损失 / 梯度 | 进，占训练目标里的一项 | **不进**，偏置不是可导路径上的任何一环 |
| 代价 | 拿主损失做交易（α 要调小） | 免费；但只能压 argmax，管不了概率形状 |
| 与主损失的关系 | 会扰动路由器梯度 | 完全不碰主损失 |

**作用前提：`logits` 必须对 token 有区分度**。偏置改的是 `logits + b` 的**排序**，所以只有当
`logits` 本身随 token 变化时，偏置才可能把不同的 token 分给不同的专家。极端例子是上面的隔离实验
② 组：路由器权重被清零后所有 token 的 `logits` 完全相同，argmax 只由偏置决定，于是"赢家"只会
在专家之间**轮转**、任一时刻的负载仍全压在那 K 个专家上——偏置再推也摊不平（③ 组因此改用
"权重整体缩小"而不是清零）。

反过来，把权重**原样留着**也不行：实测 `logits` 的极差 > 3.0，会盖过 `+3.0` 的偏置，K = 1 时
8 个专家里 7 个都能抢到 token，起点根本没塌缩。所以"起点塌缩"与"区分度还在"必须同时成立——
这也是为什么真实训练里偏置是从 0 开始、伴随训练长期推进的，而不是像 demo 里那样一次摆到塌缩点
再往回推。

**代码接线**：

- [`src/moe.rs`](../src/moe.rs) `MoELayer` 的 `bias: Shared<Vec<f32>>`（长度 `E`）、
  `bias_balance: bool`、`bias_lr: f32`；`forward_with_aux` 里"选路用 `logits + bias` 的副本、
  门控权重仍用原始 `logits`"那段就是题眼。
- [`src/moe.rs`](../src/moe.rs) `MoELayer::update_balance_bias()`：读**最近一次前向**的路由统计
  （`RouteStats::counts`）按符号推进偏置；未开 `moe_bias_balance` 或没跑过前向时返回 `false`
  （空操作）。`MoELayer::balance_bias()` 是只读诊断入口。
- [`src/model.rs`](../src/model.rs) `Transformer::update_moe_balance_bias()`：遍历所有 MoE 层调用。
- [`src/train.rs`](../src/train.rs)：训练循环在 `opt.step()` **之后、下一次前向之前**调用
  `model.update_moe_balance_bias()`（注释标为 `8b.`）。
- 配置：`moe_bias_balance`（默认 `false`）/ `moe_bias_lr`（默认 `0.001`，DeepSeek-V3 的取值）。

**两个使用约束**：

- **必须在 `opt.step()` 之后、下一次前向（含评估前向）之前调用**：偏置靠的是"上一步实际把 token
  分给了谁"这份统计，而统计只保留最近一次前向；中间夹一次评估前向会把它覆盖成验证集的负载。
- **梯度累积下只反映最后一个微批**：每步只前向一次并记录统计，所以偏置看到的是最后那个微批的负载。
- 偏置**不参与 checkpoint**：它不在 `named_parameters` 里（不是被梯度训练的量、也不该进优化器），
  续训时从 0 重新平衡，几百万 token 的语料下可忽略。

**默认路径逐位不变**：`moe_bias_balance = false`（默认）时 `update_balance_bias` 是空操作；
偏置全 0 时"选路看 `logits + 0`"与直接看 `logits` 完全等价。单测
`test_balance_bias_is_free_of_params_and_graph` 断言：偏置全 0 时，开与不开的参数表、输出、
梯度**逐位相同**。`test_balance_bias_update_direction` 验证更新方向（过载 −γ、欠载 +γ），
`test_balance_bias_converges_load_without_gradients` 验证"只做前向 + 偏置更新、完全不碰梯度"
也能把不均衡度显著推平。

### 共享专家（DeepSeek-V2/V3）

路由专家有个隐含问题：总有一些**所有 token 都要用的常识**（高频语法、常见搭配），
如果每个专家都各学一遍，就是在浪费容量。共享专家的做法是：额外放 `n_shared` 个专家，
**全部 token 都过它们**，输出以权重 1 与路由部分**并行相加**，且**不参与路由**：

```text
y = Σ_{e ∈ TopK} w_e · Expert_e(x)   +   Σ_{s ∈ shared} SharedExpert_s(x)
                      ↑ 路由部分（稀疏）              ↑ 共享部分（稠密，每 token 必算）
```

- **作用**：让共享专家承接"常识"，路由专家腾出来专心分工，减少专家冗余。
- **代价**：这部分参数**没有稀疏性可言**——每 token 都要算，所以它**同时进总参数与激活参数**，
  会按比例吃掉 MoE 省下的 FLOPs、拉低稀疏比（总参数 / 激活参数）。
  用 [`src/moe.rs`](../src/moe.rs) 的 `SparseStats::shared_params()` / `param_ratio()` 可以精确量化：
  `shared_params = n_shared × expert_params`，同时计入 `total_params()` 与 `active_params()`。

**代码接线**：

- [`src/moe.rs`](../src/moe.rs) `MoELayer::shared_experts: Vec<MLPEnum>`，与路由专家同规模
  （DeepSeek-V2 口径：共享专家的中间维 = 路由专家之一倍）；`forward_with_aux` 第 3b 步把它们的
  输出直接累加到路由输出上（即使路由把某些 token 全丢了，共享专家仍照常贡献——容量只管路由专家）。
- 参数命名前缀为 `{prefix}.shared_experts.{i}`；`SparseStats` 的 `n_shared` 字段承载个数。
- 配置：`moe_shared_experts`（默认 `0` = 不启用，行为与加它之前逐位相同）。

单测 `test_shared_experts_add_parallel_and_count_params` 断言：摘掉共享专家后的输出，
加上共享专家输出之和，恰好等于完整前向（证明是"并行相加"）；且两个共享专家各贡献一个路由专家的
参数量，`param_ratio()` 相对纯路由版本下降。

### Router z-loss：压 logits 的幅度

与 `L_aux` 互补的第二项：`L_z = β·mean(logsumexp(logits)²)`
（`TransformerConfig.moe_z_loss_coef`，默认 0，CLI `--z-loss`）。

- `L_aux` 压的是**概率**有多平（α），`L_z` 压的是 **logits 幅度**有多小（β）——
  logits 越大 `logsumexp` 越大，z-loss 直接罚它；
- 数值上走 `Tensor::logsumexp_last_dim()`（减行 max 稳定化），反向即 `g·softmax`；
- 对 K = 1 重归一化口径尤其有意义：那里主损失给不了路由器梯度，
  z-loss 是不依赖路由结果的额外训练信号（单测 `test_z_loss_value_and_router_gradient`
  手算对拍数值并断言路由器拿得到非零梯度）；
- 与 α 项合并进 `forward_with_aux` 的第二个返回值，任一为 0 就不加。

### 容量因子与 Token Dropping

`capacity = ceil(cf · n·K/E)`，超出的分配按 token 顺序**先到先得**地丢弃（Switch 的做法：
不退而求其次——否则同一 token 的去向会依赖批次里其他 token，路由变得不可预测）。
实测（每批 512 token，K = 2，平均每专家 128）：

```text
cf = 0     → 每专家容量 不限         丢弃 0/32768 = 0.00% ｜读到 token 的专家 8/8
cf = 1     → 每专家容量 128        丢弃 9257/32768 = 28.25% ｜读到 token 的专家 8/8
cf = 1.25  → 每专家容量 160        丢弃 6016/32768 = 18.36% ｜读到 token 的专家 8/8
cf = 2     → 每专家容量 256        丢弃 1059/32768 = 3.23% ｜读到 token 的专家 8/8
```

`cf = 1.0` 在偏斜路由上照样丢近 28%——容量按**平均负载**算，而负载根本不均。
**推理时必须 `cf = 0`**，否则同一句话换个批大小就换个答案（`TransformerConfig` 默认就是 0）。

### 参数 / 计算量口径

```text
每个专家就是一个普通 FFN（SwiGLU 版 3dh+2h+d = 49728 参数，d = 64）
稠密基线（n_expert=1）实测参数：135488
E=8   K=2 | 单层 总 398344 / 激活 99976（3.98× / 省 74.9% FLOPs）
          | 模型 总 832720 / 激活 235984（3.53× / 省 71.7% FLOPs）
```

`total = E·expert + router`（全部专家驻显存），`active = K·expert + router`（每 token 只算 K 个）。
启用共享专家后，`total` 与 `active` 都要再加 `n_shared × expert`——共享部分每 token 必算、不稀疏，
所以它等量地吃掉"省下的 FLOPs"（`SparseStats::shared_params()` 单独列出这一项）：
实测 `--shared-experts 2` 时单层 `总 497800 / 激活 199432`（2.50× / 省 59.9%），
稀疏比从 3.98× 掉到 2.50×——这就是"用稀疏性换质量"的量化形式。
单层口径 GELU 版 `8d²+5d`、SwiGLU 版 `3dh+2h+d`（`h = swiglu_hidden(d)`）；`moe` 子命令
**显式钉死 SwiGLU**（`moe_model_full` 里写死的 `use_swiglu: true`），免得默认值一变就静默漂移。
CLI 里有一处 `assert_eq!` 把「公式算出的参数」与「建层实测的参数」对账，防止公式随代码漂移。
**MoE 省的是 FLOPs 不是显存**——卖点是「同样的 FLOPs 预算下能塞进更多参数」。

### 接线方式

`TransformerConfig` 新增 9 个 MoE 字段（都带 `#[serde(default)]`，旧 `config.json` 无需改动）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `n_expert` | 1 | 专家数；**1 = 稠密 FFN**，行为与加 MoE 之前逐位相同 |
| `moe_top_k` | 1 | Top-K（须 `1 ≤ K ≤ E`） |
| `moe_capacity_factor` | 0.0 | 容量因子，0 = 不限（推理必须 0） |
| `moe_aux_coef` | 0.0 | 辅助损失系数 α |
| `moe_z_loss_coef` | 0.0 | router z-loss 系数 β（`β·mean(logsumexp(logits)²)`） |
| `moe_switch_gate` | false | 门控口径（见上表；`moe_top_k = 1` 时应置 `true`） |
| `moe_shared_experts` | 0 | 共享专家数 `E_s`（DeepSeek-V2/V3 式）；0 = 关闭，>0 时每个 token 都过这 `E_s` 个专家、输出与路由部分并行相加（不参与路由） |
| `moe_bias_balance` | false | aux-loss-free 均衡偏置开关（DeepSeek-V3 式）；true 时选 Top-K 看 `logits + b`，按 **sign** 以 `moe_bias_lr` 推进偏置 |
| `moe_bias_lr` | 0.001 | 均衡偏置的更新步长 γ，只对 `moe_bias_balance = true` 生效 |

> 后三个字段是 DeepSeek-V2/V3 的两项改进，默认值下与加它们之前**逐位相同**——细节见下文
> 「aux-loss-free 均衡偏置」与「共享专家」两节。它们也已开成 `moe` 子命令的命令行开关
> （`--shared-experts` / `--bias-balance` / `--bias-lr`，默认值与上表一致），不必改 `config.json` 就能做实验。

`TransformerBlock` 的前馈子层是一个 `Ffn` 枚举（`MLPEnum` 或 `MoELayer`）。两者接口一致
（`forward` / `parameters` / `named_parameters`），所以残差、dropout、GPU 常驻快路全都不用改。
`Transformer::aux_loss()` 把各层 `L_aux` 累加（每层已在 `MoELayer` 内乘过 α），训练侧用
`scale_grad_only(a, 1/accum)` 只缩放梯度、不改数值——与交叉熵在梯度累积下的处理手法一致。
另有 `Transformer::route_stats()`（合并各层统计，训练日志用）与 `Transformer::set_moe_capacity_factor()`。

### 运行方式

```bash
cargo run --release -- moe                        # 默认 E=8 K=2，端到端 300 步
cargo run --release -- moe --steps 60             # 快速跑通
cargo run --release -- moe --experts 4,8,16 --aux-coef 0.01
cargo run --release -- moe --shared-experts 2 --bias-balance --bias-lr 0.001   # 共享专家 + 偏置臂
```

| 参数 | 默认 | 说明 |
|------|------|------|
| `--experts` | `8` | 专家数网格（逗号分隔，逐个跑对照实验） |
| `--top-k` | `2` | 每个 token 激活的专家数 |
| `--steps` | `300` | 端到端对照的训练步数 |
| `--batch-size` / `--block-size` / `--lr` | `8` / `64` / `3e-3` | 训练超参 |
| `--n-embd` / `--n-layer` | `64` / `2` | 模型规模 |
| `--aux-coef` | `0.01` | 辅助损失系数 α（对照组固定为 0） |
| `--z-loss` | `0.0` | router z-loss 系数 β（0 = 不加） |
| `--capacity-factors` | `0,1.0,1.25,2.0` | 容量因子扫描（0 = 不限） |
| `--shared-experts` | `0` | 共享专家数 `E_s`（对应 `model.moe_shared_experts`）；作用于第一节口径与第三节端到端 |
| `--bias-balance` | 关 | 给第三节**追加**一条 sign 偏置均衡臂（对应 `model.moe_bias_balance`）；第二节隔离实验恒含该臂 |
| `--bias-lr` | `0.001` | 偏置更新步长 γ（对应 `model.moe_bias_lr`）；只被偏置臂使用 |
| `--seed` | `42` | 各组共用，保证初始权重逐位一致、可比 |

三个新开关的默认值与 `TransformerConfig` 的默认**逐位一致**，所以不传它们时建出的模型与旧版本完全相同
（单测 `test_moe_switches_default_to_config_defaults` 守住这条）。

输出分四节：① 参数 / 计算量口径；② 负载均衡**三组对照**隔离实验（① 不均衡 vs ② α·L_aux vs ③ sign 偏置）；
③ 端到端对照（α = 0 vs α > 0，`--bias-balance` 时再加偏置臂）；④ 容量因子与 Token Dropping。

> **关于端到端对照的诚实读法**：这个规模（2 层 × 64 维、几十步）下 α = 0 那一份**不会**塌缩——
> 随机初始化的路由器在几百步内大体保持对称，路由塌缩是「富者愈富」的**长期**动力学。
> 所以第二节才改用隔离实验（把路由器直接摆到塌缩点）。但塌缩的**代价**在任何规模下都一样：
> 主损失只关心算得准不准、完全不关心是谁在算，未被选中的专家拿不到任何梯度（等于白占显存），
> 而 loss 曲线看不出异常——所以必须有别的东西盯着路由分布。

## 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. **实现 Router**：用 `Linear` 层实现门控网络，输入 token 向量，输出 Top-K 专家索引和权重。
   —— 已实现：`src/moe.rs` 的 `top_k_gate` + `MoELayer::router`。
2. **实现 MoE Layer**：将 Router 和 N 个 FFN 组合，实现稀疏前向传播。
   —— 已实现：`MoELayer::forward_with_aux`（gather → expert → weighted → scatter，逐专家跳过空集）。
3. **负载均衡实验**：训练一个简单的 MoE-MLP，观察各专家的被选频率，加入辅助损失后观察均衡效果。
   —— 已实现：`moe` 子命令第二、三节。**注意结论与直觉相反**（见上文「实测到的退化解」）。
4. **专家分化观察**：在小数据集上训练 MoE，检查不同专家是否学到了不同的模式。
   —— 已实现（路由侧）：`RouteStats::imbalance` / `dropped_ratio` / `merge_stats`，
   `moe` 子命令的 `print_route_report` 会逐专家打印被选次数直方图与均衡度。
   进一步的「语义功能分化」（统计每个专家高激活 token 的分布、看是否各管一类模式）仍可自己加。
