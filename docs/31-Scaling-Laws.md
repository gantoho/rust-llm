# 第 31 课：Scaling Laws（缩放定律）

> **配套代码**：`src/scaling.rs`；**命令行**：`cargo run --release -- scaling`。
> 本课不满足于"把论文数字抄一遍"：幂律指数、Chinchilla 最优配比、训练时长/电费都能算，
> 而且能**真的训一组不同规模的模型**、用实测 loss 拟合出幂律（见文末
> [§本项目实现](#本项目实现)）。

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

## 本项目实现

代码：[src/scaling.rs](../src/scaling.rs)。命令行入口：

```bash
# 预算规划 + 实测扫描 + 幂律拟合（默认三个规模 2x64,4x128,6x192，各 600 步）
cargo run --release -- scaling

# 快速看一眼（规模小、步数少）
cargo run --release -- scaling --steps 200 --sizes 2x64,4x128,6x192 --data-multiples 1,2,4

# 换算法力预算与硬件假设（如 H100 ×1024、MFU 45%），并把扫描点导成 CSV
cargo run --release -- scaling --budget 1e23 --gpu-tflops 989 --n-gpu 1024 --mfu 0.45 --out logs/scaling.csv
```

子命令分两半：**上半场只算不训**（解析解，毫秒级），**下半场真训**（实测拟合，分钟级）。

### 1. 幂律拟合：为什么不能直接对 `log L` 做最小二乘

$$
L(x) = a \cdot x^{-\alpha} + b
$$

`b` 是不可约损失（语料本身的熵）。若直接对 `log L` 与 `log x` 做线性回归，`b` 会被
**摊进斜率**，拟合出的 `α` 系统性偏小——规模跨度越窄、`b` 占比越大，偏得越厉害。
本项目的做法（`fit_power_law` / `fit_linear_at_b`）：

1. 固定 `b` 后 `log(y-b) = log a - α·log x` 对 `(log a, α)` 是**线性**的 → 闭式最小二乘；
2. `b` 是唯一的非线性参数 → 外层 512 点网格定位 + 黄金分割 200 次细化；
3. 选 `b` 的判据用 **nats² 绝对误差**，而不是 log 空间的相对误差——因为我们要预测的是
   nats 上的 loss，关心的就是绝对偏差。

单测 `test_fit_power_law_recovers_known_exponent` 用已知 `(a, α, b)` 造点再拟合回来，
核对 `α` 能回到真值；`test_fit_power_law_with_zero_irreducible_loss` 验证 `b = 0` 的
退化工况下不会把 `α` 也一起带偏。

### 2. 口径统一：`C = 6ND` 里的 `N` 是**非嵌入**参数量

`6 = 前向 2 + 反向 4`（每个参数一次乘加；反向对输入、对权重各 2）。而 `N` 必须取
**非嵌入**参数量（[`params_non_embedding`](../src/scaling.rs)）：词嵌入的规模由词表决定，
与"模型有多深多宽"无关，算进去会让小模型的 `N` 虚高。本项目词表 8192 维、
模型只有几十万参数时，嵌入能占掉一半以上。

这带来一个静默风险：**参数量公式与真实建层是两份代码**，改一处忘另一处就会让 `C`、
Chinchilla 表、最优配比整体偏掉，且没有任何编译错误。本项目的对策有两条：

- SwiGLU 隐藏维抽成共用函数 `layers::swiglu_hidden(d)`，`new_swiglu` 与参数公式都调它；
- 扫描时用实测参数量逐位核对公式（`params_scan` 里的 `assert_eq!(measured, params_total(cfg))`），
  公式一旦漂移，跑扫描的第一步就会 panic，而不是给出一个"看起来合理"的错数字。

单测 `test_param_accounting_matches_real_model` 把各结构开关（RMSNorm / SwiGLU / GQA）
的组合都真实建一遍模型，逐项核对。

### 3. 两条最优配比路线，以及它们的分歧

| 路线 | 依据 | 结果 |
|------|------|------|
| `ratio20_optimal` | Chinchilla 头条结论（IsoFLOP 实测）$D = 20N$ | 联立 $C=6ND$ 得 $N = \sqrt{C/120}$，$N, D \propto C^{0.5}$ |
| `parametric_optimal` | 参数化损失 $L(N,D)$ 在 $C=6ND$ 下的闭式解 | $N^* = [(\alpha A/\beta B)(C/6)^\beta]^{1/(\alpha+\beta)}$ |

同一份算力下这两条路给出的模型规模**不一样**，本项目把它**显式打印出来**而不是抹平：

```
20:1 法则（论文头条结论，IsoFLOP 实测）：N 9.129e9 参数 | D 1.826e11 token | 20.0 token/参数
参数化损失闭式解（Approach 3）：          N 5.160e9 参数 | D 3.230e11 token | 62.6 token/参数
```

差别不是本项目的实现误差：论文正文的拟合常数经过四舍五入，会让 Approach 3 偏离
Approaches 1/2（Besiroglu et al., 2024 复现了这一点）。所以实践里只把参数化解当
**量级**指导，不要当精确解。

同理，`overtrain(C, k)` 沿 IsoFLOP 曲线移动（固定算力、数据量 ×k、模型相应变小）时，
曲线上参数化损失的最小值**不在** `k = 1` 而在 $k^* = \sqrt{(D^*/N^*)/20}$——子命令会把
`k*` 一起打出来，让两个口径的分歧摆在同一张表里。

### 4. 训练时长与电费

`Hardware::estimate` 按 `T = C / (单卡峰值 × MFU × 卡数)`、`电费 = 卡数 × 功耗 × 时长 × 电价`
估算。默认假设 A100 80GB FP16（312 TFLOPS）、64 卡、MFU 40%、400 W/卡、\$0.1/kWh，
单测 `test_wall_clock_matches_doc_example` 复现本文档上方的例子（7B × 1T token →
约 61 天、约 \$3700）。

### 5. 实测扫描：真的训一组模型

`params_scan` 固定 token 预算、逐级放大模型（`--sizes 2x64,4x128,6x192`），
`tokens_scan` 固定最小模型、逐级放大数据量（`--data-multiples 1,2,4,8`），
两者都用 `fit_power_law` 拟合出实测指数。

扫描的所有点共用同一份语料与分词器、同一 token 预算、同一随机种子，否则不同规模的
loss 之间没有可比性；训练配置由 `scan_train_config` 统一固定为"不落盘、不早停"。

> **诚实说明**：这里的指数与本课论文数字（$\alpha_N \approx 0.076$、$\alpha_D \approx 0.095$）
> **不会接近**。原因是本项目根本不在论文的规模区间内：模型 10⁵ 量级参数 vs 论文 10⁸~10¹⁰，
> 语料 10⁶ token vs 论文 10¹¹。规模区间差三到五个数量级，再加上"固定 token 预算"下模型很容易
> 掉进数据受限区间（`D/N < 1`：容量收益被过拟合抵消，val loss 可能不降反升），
> 测出来的指数一定与论文不同。
> 有价值的是**方法**：口径统一（非嵌入参数、6ND）、预算固定、可复现，把它放大到论文的
> 规模上就是论文的实验。反过来，把论文指数硬套到本项目规模上，才是真正没意义的做法。

#### 一次真实运行的输出（`--steps 90 --sizes 2x64,4x128,6x192 --data-multiples 1,2,4`）

参数扫描（固定 46,080 token 预算，每点 90 步，`D/N` 是 token/非嵌入参数）：

```
  2x64     | 非嵌参    100096 | token    4.608e4 | D/N    0.5 | 实测 val loss 7.8099
  4x128    | 非嵌参    793344 | token    4.608e4 | D/N    0.1 | 实测 val loss 7.7438
  6x192    | 非嵌参   2669568 | token    4.608e4 | D/N    0.0 | 实测 val loss 7.7621
  幂律拟合：L(N) = 1.0683·N^(-0.2128) + 7.7035｜r² = 0.688676
```

这里第三个点 **7.7621 > 7.7438**，曲线非单调——正是上面说的数据受限区间：2.7M 参数的模型
只喂 46K token（`D/N ≈ 0.017`），训练 loss 更低（7.2213 < 7.4317）但验证 loss 反而更高。
子命令会对这种情况打印 `⚠ 读表提示`，而不是把它当噪声藏起来。
**要测纯 `N` 的幂律，得让 token 预算跟得上参数**（缩小 `--sizes` 或加大 `--steps`）。

数据量扫描（固定 2x64 模型，基准 22 步 ×4）：

```
  x1       | token    1.126e4 | 实测 val loss 8.0102
  x2       | token    2.253e4 | 实测 val loss 7.8461
  x4       | token    4.506e4 | 实测 val loss 7.6291
  幂律拟合：L(D) = 11.1333·D^(-0.0352) + 0.0000｜r² = 0.992432
```

数据量那一侧就干净得多（r² = 0.99，单调下降），因为固定模型放大数据不会引入过拟合反向效应。
`b = 0` 贴在搜索下界上，说明在这段跨度里还看不出不可约损失——子命令同样会把这一点说明白。

对应下面四条拓展方向在仓库里的落地方式：

| 拓展方向 | 落地 |
|------|------|
| 1. 幂律拟合 | `params_scan` + `fit_over_params`（实测点与拟合曲线逐点打印偏差） |
| 2. 算力估算 | `scaling --gpu-tflops ... --n-gpu ... --mfu ...`（打印天数与电费） |
| 3. 最优配比计算 | `scaling --budget 1e22`（同时给出 20:1 与参数化闭式解两条路线） |
| 4. 过训练分析 | `tokens_scan`（真训）+ `overtrain`（解析预测） |

## 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. **幂律拟合**：在不同规模的模型上记录 loss，用对数坐标拟合 loss vs N 的幂律关系。
2. **算力估算**：给定 GPU 型号和数量，估算训练一个 13B 模型到 Chinchilla 最优需要多少天。
3. **最优配比计算**：给定 1e22 FLOPs 的算力预算，计算最优的参数量和数据量。
4. **过训练分析**：在小模型上分别用 1×、2×、4×、8× Chinchilla 最优数据量训练，观察 loss 曲线和生成质量的变化。
