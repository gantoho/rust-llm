# 第 32 课：MoE 混合专家模型（Mixture of Experts）

> **本课已落地为可运行代码**：核心实现在 [`src/moe.rs`](../src/moe.rs)（1174 行，含 14 个单测），
> 接线在 [`src/model.rs`](../src/model.rs)（`TransformerConfig.n_expert` / `moe_top_k` 等 6 个字段 +
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
14 个单测全部与 CPU 参考实现对齐。

### 公开 API

| 项目 | 说明 |
|------|------|
| `top_k_gate` | 纯函数路由：`logits → Top-K`（并列按下标、**确定性**），返回 `RoutePlan`；选择用 `select_nth_unstable_by` O(E) 部分选择 + 对前 K 排序，不做全排序 |
| `RoutePlan` | 一次路由的完整结果：`sel`（各专家实际处理的 token 行号，已按容量截断）、`penalty`（未选中 −∞）、`mask`（选中 1 / 其余 0）、`f`（分配比例）、`dropped` / `routed` |
| `expert_capacity` | `ceil(cf · n·K/E)`；`cf ≤ 0` → `usize::MAX`（不限） |
| `MoELayer` | 专家集合 + 路由器；`forward` / `forward_with_aux` / `route_plan` / `stats` |
| `RouteStats` | 诊断：各专家被路由到的次数、`aux`、`imbalance()`、`dropped_ratio()` |
| `sparse_stats` | 参数 / 激活量口径：`total_params` / `active_params` / `param_ratio` / `flops_saving` |

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

`L_aux = α·E·Σ f_i·p_i` 的性质用 `moe` 子命令第二节的**隔离实验**验证过
（人为把路由器摆到塌缩点，然后不跑主损失、只优化 `L_aux`）：

```text
汇总：K=1：L_aux 5.932 → 1.039（p 摊平 ⇒ 1.0，不是下界），不均衡度 8.00 → 6.32，用到的专家 1/8 → 4/8
     ｜K=2：L_aux 3.114 → 1.010（p 摊平 ⇒ 1.0，不是下界），不均衡度 4.00 → 3.92，用到的专家 2/8 → 6/8
```

`L_aux` 确实被压到 1.0，但**硬路由几乎没动**（K=2 组 4.00 → 3.92）。原因就是上面说的软/硬错配：

- `f` 由 argmax 给出，不可导；`p` 由 softmax 给出，可导。
- 梯度 `∂L/∂logit_j ∝ p_j·(f_j − Σ f_i p_i)` 的**大小正比于 `p_j`**：`p` 越平梯度越小。
  于是最快的下降路径是**先把 `p` 摊平**（`L_aux` 一步到位到 1），而不是把硬路由摊平。
- 而 `p` 均匀时 `Σ f_i p_i = (1/E)·Σ f_i = 1/E` ⇒ `L_aux = 1`，**与 `f` 无关**。

所以「`L_aux` 掉到 1」**不是**负载均衡的证书；并且 1 也**不是下界**——`f` 与 `p` 支撑集不交时
`Σ f_i p_i = 0`，`L_aux` 可以低到 0（`test_aux_loss_minimum_at_uniform` 覆盖了这三种情形：
`p = f` 时 `≥ α`、支撑集不交时 `= 0`、`p` 均匀时 `= 1`）。

这正是后续工作（DeepSeek-V3 的 loss-free 均衡偏置、expert-choice 路由）要绕开的软/硬错配，
也是「辅助损失系数要调小」的真实原因：它压的是概率分布，不是分配结果。

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
cf = 1     → 每专家容量 128        丢弃 8171/32768 = 24.94% ｜读到 token 的专家 8/8
cf = 1.25  → 每专家容量 160        丢弃 5317/32768 = 16.23% ｜读到 token 的专家 8/8
cf = 2     → 每专家容量 256        丢弃 862/32768 = 2.63% ｜读到 token 的专家 8/8
```

`cf = 1.0` 在偏斜路由上照样丢近 25%——容量按**平均负载**算，而负载根本不均。
**推理时必须 `cf = 0`**，否则同一句话换个批大小就换个答案（`TransformerConfig` 默认就是 0）。

### 参数 / 计算量口径

```text
E=8   K=2 | 单层 总 265224 / 激活 66696（3.98× / 省 74.9% FLOPs）
          | 模型 总 566608 / 激活 169552（3.34× / 省 70.1% FLOPs）
```

`total = E·expert + router`（全部专家驻显存），`active = K·expert + router`（每 token 只算 K 个）。
单层口径 GELU 版 `8d²+5d`、SwiGLU 版 `3dh+2h+d`（`h = swiglu_hidden(d)`）。
CLI 里有一处 `assert_eq!` 把「公式算出的参数」与「建层实测的参数」对账，防止公式随代码漂移。
**MoE 省的是 FLOPs 不是显存**——卖点是「同样的 FLOPs 预算下能塞进更多参数」。

### 接线方式

`TransformerConfig` 新增 6 个 MoE 字段（都带 `#[serde(default)]`，旧 `config.json` 无需改动）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `n_expert` | 1 | 专家数；**1 = 稠密 FFN**，行为与加 MoE 之前逐位相同 |
| `moe_top_k` | 1 | Top-K（须 `1 ≤ K ≤ E`） |
| `moe_capacity_factor` | 0.0 | 容量因子，0 = 不限（推理必须 0） |
| `moe_aux_coef` | 0.0 | 辅助损失系数 α |
| `moe_z_loss_coef` | 0.0 | router z-loss 系数 β（`β·mean(logsumexp(logits)²)`） |
| `moe_switch_gate` | false | 门控口径（见上表；`moe_top_k = 1` 时应置 `true`） |

`TransformerBlock` 的前馈子层是一个 `Ffn` 枚举（`MLPEnum` 或 `MoELayer`）。两者接口一致
（`forward` / `parameters` / `named_parameters`），所以残差、dropout、GPU 常驻快路全都不用改。
`Transformer::aux_loss()` 把各层 `L_aux` 累加（每层已在 `MoELayer` 内乘过 α），训练侧用
`scale_grad_only(a, 1/accum)` 只缩放梯度、不改数值——与交叉熵在梯度累积下的处理手法一致。
另有 `Transformer::route_stats()`（合并各层统计，训练日志用）与 `Transformer::set_moe_capacity_factor()`。

### 运行方式

```bash
cargo run --release -- moe                        # 默认 E=8 K=2，端到端 300 步
cargo run --release -- moe --steps 60             # 快速跑通（上面所有数字的来源）
cargo run --release -- moe --experts 4,8,16 --aux-coef 0.01
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
| `--seed` | `42` | 各组共用，保证初始权重逐位一致、可比 |

输出分四节：① 参数 / 计算量口径；② 负载均衡辅助损失隔离实验；③ 端到端对照（α = 0 vs α > 0）；
④ 容量因子与 Token Dropping。

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
