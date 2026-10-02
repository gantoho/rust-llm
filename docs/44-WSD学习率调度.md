# 第 44 课：WSD 学习率调度（warmup-stable-decay）

> **本课已落地为可运行代码**：[`src/train.rs`](../src/train.rs) 的 `LRScheduler`
> （`new` / `new_wsd` / 私有 `with_wsd`，以及工厂 `lr_scheduler(cfg, total_steps)`）+
> 配置开关 `train.lr_schedule` / `train.wsd_decay_frac`。
>
> 与第 18 课的衔接：第 18 课实现的是 warmup + cosine 衰减；本课在**同一个** `LRScheduler`
> 上加一条"warmup → 恒定 → 线性退火"的三段曲线，默认仍是 cosine——`lr_schedule = "cosine"`
> 时调用链与加 WSD 之前**逐位一致**。

---

## 1. 本课要搞懂的问题

1. cosine 衰减为什么要"总步数一开始就定死"？这带来什么工程约束？
2. WSD 的三段各是什么？为什么"稳定段 + 末段退火"能替代余弦收尾？
3. 退火段该占多少比例？退火起点被谁夹住？
4. 怎么保证引入 WSD 不会偷偷改动默认的 cosine 路径？

---

## 2. 动机：cosine 的"总步数前置"问题

cosine 衰减的形状由 `total_steps` 直接决定：

```text
progress = (step - warmup_steps) / (total_steps - warmup_steps)
lr       = min_lr + (max_lr - min_lr) · 0.5·(1 + cos(π·progress))
```

只要 `total_steps` 变动，**整条曲线就要重排**——中途想"再多训一会儿"，前一段已经走过的
学习率就再也对不上了。这在实际训练里很别扭：

- 大规模预训练常常"先跑一个稳定版本，看情况再决定何时收尾"，而 cosine 逼你把收尾点一开始就定死；
- 想用同一份数据从不同 checkpoint 接着训，cosine 的曲线形状对不上。

WSD（warmup-stable-decay，MiniCPM / DeepSeek 等采用）把曲线拆成三段解决这个问题：
**稳定段是一段平线（恒等于峰值学习率）**，任何时刻从平线上切出来退火都成立，于是
"稳定段随便训多久、要收尾时再退火"成了常规操作。DeepSeek-V2/V3 能在稳定段中途换数据配比
继续训，靠的就是这条性质。

---

## 3. 三段的公式

`LRScheduler` 的两条曲线**共用同一段线性 warmup**（这是刻意设计，见 §5）：

```text
warmup（step < warmup_steps）：
    lr = max_lr · (step + 1) / max(warmup_steps, 1)
```

warmup 之后按 `wsd_decay_steps` 分流：

```text
cosine（wsd_decay_steps = None）：
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1)
    lr       = min_lr + (max_lr - min_lr) · 0.5·(1 + cos(π·progress))

WSD（wsd_decay_steps = Some(d)）：
    decay_start = max(total_steps - d, warmup_steps)     # 退火起点不能早于 warmup 结束
    step < decay_start:  lr = max_lr                     # 稳定段：一段平线
    否则:
        progress = min((step - decay_start) / max(total_steps - decay_start, 1), 1)
        lr       = max_lr + (min_lr - max_lr) · progress  # 线性退火
```

三段直观形状（`warmup = 4, total = 20, decay = 5`，`max_lr = 1.0, min_lr = 0.1`）：

```text
lr
1.0 ┤      ┌──────────────────┐
    │    ┌─┘                  └──┐
    │  ┌─┘                      └──┐
0.1 ┤ ─┘                          └─
    └───┬────┬─────────────────┬────┬──▶ step
      warmup  stable(平台)      decay
        4                 15     20
```

### 3.1 退火步数怎么算

工厂 `lr_scheduler(cfg, total_steps)` 把比例换算成绝对步数：

```text
span   = max(total_steps - warmup_steps, 0)
decay  = max(round(wsd_decay_frac × span), 1)     # 至少 1 步
```

默认 `wsd_decay_frac = 0.1`，即退火吃掉"扣除 warmup 后剩余步数"的 10%。

---

## 4. 本仓库实现要点

| 位置 | 内容 |
|------|------|
| [`src/train.rs`](../src/train.rs) `LRScheduler` | 字段 `wsd_decay_steps: Option<usize>`：`None` = cosine，`Some(d)` = WSD；`lr()` 按上面公式取当前步学习率 |
| [`src/train.rs`](../src/train.rs) `LRScheduler::new` | cosine 构造：`with_wsd(..., None)` |
| [`src/train.rs`](../src/train.rs) `LRScheduler::new_wsd` | WSD 构造：`with_wsd(..., Some(decay_steps.max(1)))` |
| [`src/train.rs`](../src/train.rs) `with_wsd` | 私有公共构造器，两条曲线只在这里分流 |
| [`src/train.rs`](../src/train.rs) `lr_scheduler` | 按 `cfg.lr_schedule` 分派；`"wsd"` 时算 `decay_steps`，否则退回 `new` |
| [`src/config.rs`](../src/config.rs) `TrainConfig::lr_schedule` | 默认 `"cosine"`；`validate` 断言只允许 `"cosine"` / `"wsd"` |
| [`src/config.rs`](../src/config.rs) `TrainConfig::wsd_decay_frac` | 默认 `0.1`；`validate` 断言必须落在 `(0, 1]` |

训练循环里每步先 `scheduler.lr()` 取当前学习率写进优化器（`opt.set_lr(cur_lr)`），
**只有在优化器真正更新之后**才 `scheduler.step()`——AMP 溢出跳步时不推进学习率
（那一步没有真正发生）。

> 设计取舍：抽出工厂 `lr_scheduler` 是为了让 `train` / `sft` / `finetune` 几条训练入口
> 拿到**同一条曲线**，不会出现"主训练用 WSD、微调悄悄退回 cosine"这种不一致。

---

## 5. 默认路径逐位不变

- 默认 `lr_schedule = "cosine"`：`lr_scheduler` 走进 `else` 分支，调用
  `LRScheduler::new(warmup_steps, total_steps, max_lr, min_lr)`——它与加 WSD 之前**完全同一条路径**。
- `wsd_decay_steps` 为 `None` 时 `lr()` 里的 `match` 走 cosine 分支，公式一字未动。
- 单测 `test_cosine_schedule_default_unchanged` 逐点钉住了默认曲线的数值快照
  （`warmup=2, total=6, max_lr=1, min_lr=0` → `[0.5, 1.0, 1.0, 0.8535534, 0.5, 0.14644664]`），
  谁改动 cosine 公式这里会立刻红。
- `test_wsd_warmup_matches_cosine` 进一步证明：切到 WSD 后，前 `warmup_steps` 步与 cosine
  **逐位相同**——WSD 只改 warmup 之后的段。

---

## 6. 验证方式

```bash
# WSD 相关单测（三段结构、warmup 对齐、工厂分派、退火夹取、同种子对照）
cargo test wsd

# 默认 cosine 行为未变
cargo test cosine_schedule_default_unchanged

# 工厂分派
cargo test lr_scheduler
```

配置示例：

```jsonc
{
  "train": {
    "lr_schedule": "wsd",   // 默认 "cosine"
    "wsd_decay_frac": 0.1   // 退火占"剩余步数"的比例，默认 0.1
  }
}
```

几个关键单测及它们钉住的性质：

| 测试 | 验证内容 |
|------|---------|
| `test_wsd_schedule_has_three_phases` | warmup 严格递增且末点踩到 `max_lr`；稳定段恒等于 `max_lr`；退火段单调不增、落在 `[min_lr, max_lr)` |
| `test_wsd_warmup_matches_cosine` | warmup 段两条曲线逐位相同，末步确实不同（否则没测到差别） |
| `test_cosine_schedule_default_unchanged` | 默认 cosine 曲线数值快照不变 |
| `test_lr_scheduler_factory_respects_schedule` | 工厂按 `lr_schedule` 分派；`decay_start = total − round(frac·(total−warmup))`，稳定段是真正的平台；cosine 无平台 |
| `test_wsd_decay_clamped_to_warmup` | 病态参数下退火起点被 `warmup` 夹住，曲线仍单调 |
| `test_wsd_and_cosine_identical_when_lr_flat` | `max_lr == min_lr` 时两曲线都是常值，同种子 loss 逐位相同（隔离性证明） |
| `test_wsd_vs_cosine_same_seed_differs` | `max_lr > min_lr` 时两曲线不同，同种子 loss 也确实不同（有效性证明） |

---

## 7. 两个坑（实测踩出来的）

### 7.1 退火步数过大会把稳定段吃掉

`decay_start = max(total_steps − decay_steps, warmup_steps)` 里的 `max(...)` 是必需的：
当 `wsd_decay_frac` 配得很大（`decay_steps ≥ total_steps − warmup_steps`）时，退火起点会被
**夹到 warmup 结束处**，稳定段长度为 0——warmup 一结束就开始退火。

`test_wsd_decay_clamped_to_warmup`（`warmup=8, total=10, decay=9`）验证了这条边界：
退火起点落在第 8 步（warmup 结尾），且曲线仍保持"warmup 之后单调不增"。
如果这里不夹，曲线会出现先降后升的折返，训练反而更不稳。

### 7.2 `max_lr == min_lr` 时两条曲线逐位相同，别拿它当"WSD 生效"的证据

当 `max_lr == min_lr`，cosine 与 WSD 在 warmup 之后都是**常值**，同种子的整轮训练
必然逐位相同（`test_wsd_and_cosine_identical_when_lr_flat`）。

这条测试本身的用途是**隔离性证明**（说明调度差异没有偷偷改动别的路径、训练可复现），
但它也提醒：要验证"WSD 真的接进了优化器"，必须把学习率区间拉成 `max_lr > min_lr`——
`test_wsd_vs_cosine_same_seed_differs` 就是这么做的（断言两条曲线的 loss 都有限且不相等，
不比较谁更优：十几步的极小模型上谁赢是噪声）。

---

## 8. 代码地图与拓展方向

| 位置 | 内容 |
|------|------|
| [`src/train.rs`](../src/train.rs) `LRScheduler` | 两条曲线的统一实现 + `step` / `set_step`（续训跳步） |
| [`src/train.rs`](../src/train.rs) `lr_scheduler` | 配置 → 调度器的工厂 |
| [`src/config.rs`](../src/config.rs) | `lr_schedule` / `wsd_decay_frac` 字段与校验 |

拓展方向：

- **稳定段续训**：利用"平线任意时刻可切"的性质，写一个"从稳定段 checkpoint 起、只跑退火段"
  的实验，对比同算力下 cosine 从头训的效果。
- **退火形状**：本实现用线性退火；可以试 `1 − √progress`、cosine 退火等变体，看末段损失曲线。
- **退火比例扫描**：`wsd_decay_frac` 在 `{0.05, 0.1, 0.2, 0.5}` 上扫一遍，量"退火太短来不及收敛、
  太长浪费算力"的拐点。
- **与 Muon 组合**：Muon 的有效步长口径与 AdamW 不同（见第 46 课），换优化器时调度曲线的峰值
  需要重新标定。
