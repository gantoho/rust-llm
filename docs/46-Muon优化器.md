# 第 46 课：Muon 优化器 —— 动量矩阵的 Newton–Schulz 正交化

> **本课已落地为可运行代码**：[`src/optim.rs`](../src/optim.rs) 的 `Muon`
> （含私有辅助 `zeropower_newton_schulz` / `matmul` / `transpose`）+ 新增 trait
> `OptimizerState` + 工厂 [`src/train.rs`](../src/train.rs) 的 `make_optimizer`。
> 配置开关：`train.optimizer`（默认 `"adamw"`）/ `train.muon_momentum`（默认 0.95）/
> `train.muon_ns_steps`（默认 5）。
>
> 与第 17 课的衔接：第 17 课的 AdamW 在**每个坐标**上独立自适应缩放；Muon 换了个方向——
> 把整个更新矩阵按**谱范数**归一化。两者在同一份代码里共存，一维参数自动回退 AdamW。

---

## 1. 本课要搞懂的问题

1. AdamW 的"逐坐标自适应"在矩阵参数上有什么副作用？
2. Muon 的两步（动量矩阵 + Newton–Schulz 正交化）各自在做什么？
3. Newton–Schulz 的迭代式与常数是怎么定的？为什么迭代前要先按 Frobenius 范数归一化？
4. 为什么 Muon 的学习率要比 AdamW 大 `√cols` 倍才"等效"？
5. 一个优化器里怎么同时容纳 Muon 与 AdamW？

---

## 2. 动机：逐坐标缩放的更新矩阵谱太宽

AdamW 的更新是 `lr · m̂ / (√v̂ + ε)`，**逐个元素**独立缩放。对矩阵权重 `W ∈ R^{r×c}` 来说，
这意味着更新矩阵 `ΔW` 的各方向步长差异可以非常大：个别方向的奇异值特别大，其余很小。

后果是"被个别大奇异方向带偏"——训练像是一部分方向猛走、一部分几乎不动。Muon
（**M**oment**u**m **o**rthogonalized by **N**ewton–Schulz，Keller Jordan et al., 2024）
的想法是：把更新矩阵**正交化**，让它的所有奇异值都等于 1，"所有方向以同样的步长前进"。

---

## 3. 原理与公式

每步对二维参数 `W ∈ R^{r×c}`：

```text
1. 累积动量      B ← μ·B + (1−μ)·G          （μ 默认 0.95，不做偏差校正）
2. 正交化        O ← NewtonSchulz(B) ≈ U Vᵀ  （B = U Σ Vᵀ 的极因子）
3. 缩放并更新    O ← O · √(max(1, r/c))
                 W ← W − lr·(O + wd·W)
```

### 3.1 Newton–Schulz 迭代

求近似正交极因子用的是 5 阶多项式迭代（`A = X Xᵀ`）：

```text
X ← a·X + (b·A + c·A²)·X
    a = 3.4445,  b = −4.7750,  c = 2.0315
```

这组常数（Keller Jordan 的 Muon 用的就是它）是在"不要发散"约束下收敛最快的 5 阶组合，
作用是把 `B` 的**所有奇异值往 1 推**。两个关键细节：

- **迭代前必须按 Frobenius 范数归一化**：初值的最大奇异值 > 1 时迭代会发散成 NaN；
  归一化把谱范数压到 `‖X‖₂ ≤ ‖X‖_F = 1`，迭代才是收缩的。实现里还加了 `+1e-7`
  防止全零矩阵除零（梯度全零时返回零矩阵而不是 NaN）。
- **`rows > cols` 时先转置成"矮胖"矩阵**：中间量 `A = X Xᵀ` 的规模由**较小那一维**决定，
  转置能让它从 `[rows, rows]` 缩到 `[cols, cols]`，省掉一大块算力。

### 3.2 缩放与"有效步长"

`O ≈ UVᵀ` 的行是单位向量，每个元素的量级约 `1/√c`（`c` 是列数），于是：

```text
Muon 每元素更新量 ≈ lr / √c
AdamW 每元素更新量 ≈ lr
```

要拿到相近的有效步长，Muon 的学习率通常要大 `√c` 倍。这也是 `scale = √(max(1, r/c))` 的来历：
`r/c > 1`（矮胖）时把整体尺度抬起来，使不同形状矩阵的更新范数可比。

---

## 4. 本仓库实现要点

| 位置 | 内容 |
|------|------|
| [`src/optim.rs`](../src/optim.rs) `Muon` | 字段：`momentum`(0.95)、`ns_steps`(5)、`weight_decay`，以及一维回退用的 `beta1`(0.9) / `beta2`(0.999) / `eps`(1e-8)；`m` / `v` 三段状态 |
| [`src/optim.rs`](../src/optim.rs) `Muon::is_matrix` | `shape.len() == 2 && shape[0] > 1 && shape[1] > 1`；`[n,1]`/`[1,n]` 的退化二维等价于一维，归回 AdamW |
| [`src/optim.rs`](../src/optim.rs) `Muon::step` | 二维走 Muon，一维走 AdamW（公式与 `AdamW::step` 逐行相同，`t`/`beta`/`eps` 一致） |
| [`src/optim.rs`](../src/optim.rs) `zeropower_newton_schulz` | 归一化 → `steps` 次迭代 →（必要时）转置回原形状；常数 `3.4445 / -4.7750 / 2.0315` |
| [`src/optim.rs`](../src/optim.rs) `OptimizerState` trait | 在 `Optimizer` 之上补齐 `set_lr` / `state` / `restore_state`，供训练循环与 checkpoint 使用 |
| [`src/train.rs`](../src/train.rs) `make_optimizer` | `"muon"` → `Muon::new(max_lr, params, weight_decay, muon_momentum, muon_ns_steps)`；其余 → `AdamW::new(...)` |
| [`src/config.rs`](../src/config.rs) | `optimizer`（默认 `"adamw"`）/ `muon_momentum`（0.95）/ `muon_ns_steps`（5）字段与校验 |

**状态布局**（checkpoint 的三段数据块）：二维参数用 `m` 存正交化**之前**的动量缓冲、
`v` 恒为 0 占位；一维参数的 `m` / `v` 语义与 AdamW 完全相同。这样 `Muon` 的状态导出/恢复
与 [`crate::checkpoint`](../src/checkpoint.rs) 的三段格式（参数 / 一阶动量 / 二阶动量）一一对应。

**为什么要新增 `OptimizerState`**：训练循环每步要用调度器改写学习率——`Box<dyn Optimizer>`
抽象下访问不到结构体字段（`train_vqvae` 原本写的是 `opt.lr = ...`）。把 `set_lr` /
`state` / `restore_state` 提到 trait 上，`AdamW` 与 `Muon` 才能被同一套训练循环与 checkpoint
代码统一处理。

---

## 5. 默认路径逐位不变

- 默认 `optimizer = "adamw"`：`make_optimizer` 走 `_` 分支，`AdamW::new(max_lr, params, weight_decay)`
  与加 Muon 之前**完全一致**；
- `AdamW` 的实现没变，只是把 `set_lr` / `state` / `restore_state` 也实现到 `OptimizerState` 上——
  这与它原来就有的同名方法是同一套逻辑，行为逐位不变；
- 单测 `test_optimizer_factory_defaults_to_adamw` 钉住两点：默认配置就是 `"adamw"`；
  对未知取值也回退 `AdamW`（`validate` 会先拦下非法值，工厂这里是不 panic 的双保险）。

---

## 6. 验证方式

```bash
# Muon 单测（NS 精确性、零梯度、闭式更新、缩放比、一维回退、冻结跳过、状态往返）
cargo test muon

# Newton–Schulz 对正交结构输入的精确代数性质
cargo test newton_schulz

# 工厂默认行为
cargo test optimizer_factory
```

配置示例：

```jsonc
{
  "train": {
    "optimizer": "muon",     // 默认 "adamw"
    "max_lr": 0.05,          // ⚠️ Muon 每元素更新量约 lr/√cols，要比 AdamW 大 √cols 倍
    "muon_momentum": 0.95,   // 默认 0.95
    "muon_ns_steps": 5       // 默认 5
  }
}
```

| 测试 | 验证内容 |
|------|---------|
| `test_newton_schulz_matches_polynomial_on_orthogonal_input` | 对"最小维单位阵"输入，输出精确等于 `p⁵(1/‖Q‖_F)·Q`（见 §7.2） |
| `test_muon_zero_gradient_yields_zero_update` | 全零动量 + 零权重衰减 ⇒ 参数一动不动，且不产生 NaN（`+1e-7` 的作用） |
| `test_muon_matrix_update_matches_closed_form` | 梯度取单位阵时，逐元素核对整条链路 `Δ = −lr·scale·p⁵(s)·I` |
| `test_muon_scaling_uses_dimension_ratio` | 缩放因子是 `√(max(1, r/c))`：`[8,2]` 的更新范数是 `[2,8]` 的 2 倍 |
| `test_muon_falls_back_to_adamw_on_1d_params` | 一维参数与 AdamW 走出**逐位相同**的结果 |
| `test_muon_skips_frozen_params` | 冻结参数整体跳过（连权重衰减都不吃） |
| `test_muon_state_roundtrip` | `state` / `restore_state` 往返一致（checkpoint 续训依赖） |
| `test_muon_changes_training_same_seed` | 同种子对照：换 Muon 确实改变 loss，且 loss 有限 |
| `test_optimizer_factory_defaults_to_adamw` | 默认走 AdamW；未知取值不 panic |

---

## 7. 两个坑（实测踩出来的）

### 7.1 沿用 AdamW 的学习率，Muon 会"几乎不动"

Muon 每元素更新量约 `lr/√cols`，比 AdamW 小 `√cols` 倍。直接套默认 `max_lr = 3e-3`，
参数几乎不动，看起来像"没生效"。`test_muon_changes_training_same_seed` 特意把学习率设成
`0.05`（而不是默认的 `3e-3`）并注释了原因——这正是配置注释里 `⚠️` 提醒的那一条。

配置里给了 `optimizer = "muon"` 却忘了调大 `max_lr`，是一条很容易踩的暗坑：loss 曲线不会报错，
只是学得很慢。

### 7.2 5 阶 Newton–Schulz 输出不是严格 `UVᵀ`，而是 `US'Vᵀ`

这组 5 阶常数的设计目标是"零点斜率最大、不保证处处精确收敛到 1"，
所以迭代完的奇异值**分散在 0.5~1.5**，输出其实写作 `US'Vᵀ`，而不是严格的 `UVᵀ`。

只有在"**奇异值全相同**"的特殊输入上，迭代才精确退化成对一个标量的多项式反复作用：

```text
Q 的奇异值全为 s = 1/‖Q‖_F  ⇒  输出 = p⁵(s) · Q
p(x) = 3.4445·x − 4.775·x³ + 2.0315·x⁵
```

这正是单测 `test_newton_schulz_matches_polynomial_on_orthogonal_input` 的断言口径
（用"最小维单位阵"当输入，逐元素核对 `p⁵(s)·Q`）。它**不是**在断言"输出是行正交单位"——
用后者当口径会被常量设计上的行为差异误判成 bug。

另一个容易混的细节：**二维参数的动量不做偏差校正**（直接用 `B`，不除 `1−μᵗ`），
而**一维回退的 AdamW 走完整偏差校正**。同一个优化器实例里两套动量语义并存，别混。

---

## 8. 代码地图与拓展方向

| 位置 | 内容 |
|------|------|
| [`src/optim.rs`](../src/optim.rs) | `Muon`、`zeropower_newton_schulz`、`matmul` / `transpose`、`OptimizerState` |
| [`src/train.rs`](../src/train.rs) | `make_optimizer` 工厂 |
| [`src/config.rs`](../src/config.rs) | `optimizer` / `muon_momentum` / `muon_ns_steps` 字段与校验 |

拓展方向：

- **嵌入/输出头单独走 AdamW**：本实现按**形状**分流，嵌入矩阵也是二维、也会走 Muon。
  工程上更常见的做法是把嵌入与输出头留在 AdamW（它们的梯度按行统计更有意义）；
  本项目优化器层拿不到参数名，故统一按形状分流——要改需要把参数名传进优化器。
- **`ns_steps` 扫描**：`muon_ns_steps = 1/3/5/10` 对照，看"正交化不足 vs 太慢"的拐点。
- **与 WSD 组合**：换了有效步长口径后，学习率调度曲线的峰值需要重新标定（见第 44 课）。
- **扩展到三维以上**：本实现只对二维权重正交化，其余全部回退 AdamW。
