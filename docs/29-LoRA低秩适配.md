# 第 29 课：LoRA 低秩适配 —— 用 1% 的参数微调大模型

> 代码位置：[src/layers.rs](../src/layers.rs)（`LoraAdapter` 适配层）、
> [src/attention.rs](../src/attention.rs)（`MultiHeadAttention::apply_lora`）、
> [src/model.rs](../src/model.rs)（`GPT::apply_lora` 冻结 + 注入）、
> [src/tensor.rs](../src/tensor.rs)（`Tensor::matmul_frozen` 冻结权重的反向）
>
> 算法论文：*LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
>
> 上手命令：`cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt --steps 60 --lr 5e-5`

---

## 1. 本课要搞懂的问题

1. 为什么大模型微调不需要更新全部参数？
2. LoRA 是怎么用低秩矩阵近似权重更新的？
3. 为什么 B 初始化为全零、A 初始化为正态？
4. α 缩放因子是干什么的？
5. 把 LoRA 接进训练循环时，"冻结主干"要在哪几处一起做实才不出错？

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
ΔW = (α / r) · B · A
```

其中：
- A ∈ R^{r × in}：下投影矩阵，把输入压到低秩空间
- B ∈ R^{out × r}：上投影矩阵，再从低秩空间升回输出维
- r ≪ min(in, out)：秩（通常 4-64）
- α：缩放因子（见第 6 节）

### 3.1 参数量对比

```
全量微调：in × out 参数
LoRA：    r × (in + out) 参数

例：in=4096, out=4096, r=16
全量：4096 × 4096 = 16,777,216（16M）
LoRA：16 × (4096 + 4096) = 131,072（131K）→ 0.78%
```

本项目实测（[config/config.json](../config/config.json)：d=128、4 层、Q/K/V 三处、r=8）：

```
全量：1,866,496
LoRA：3 个投影 × 2 个矩阵 × 8 × 128 × 4 层 = 24,576（1.32%）
```

### 3.2 为什么在「层内」而不是「权重上」加适配器

本项目把适配器挂成 [`Linear`](../src/layers.rs) 的一个字段（`lora: Option<LoraAdapter>`），
**不持有、也不复制 W**——`LoraAdapter` 里只有 A 和 B 两个张量。
好处：
- 冻结主干不用克隆权重，显存里只有一份 W；
- `Linear::forward` 里主干与增量分开算，主干那条路走 `matmul_frozen`（不产生 `dW`）；
- checkpoint 只需额外存 A/B，主干仍按原来的名字和形状存，旧存档逻辑不用改。

---

## 4. 前向计算

```
y = x @ (W + ΔW) = x @ W + (x @ B) @ A = x @ W + (α/r) · (x @ B) @ A
```

- W 是预训练权重（**冻结**，不更新）
- B 和 A 是可训练参数
- 推理时可以把 ΔW 合并到 W 里：`W' = W + (α/r)·B·A`，没有额外开销

本项目按上面的形式**分开算两条路**（[`Linear::forward`](../src/layers.rs)）：

```rust
let base = if self.weight.requires_grad() {
    x.matmul(&self.weight)          // 可训练：反向要 dW
} else {
    x.matmul_frozen(&self.weight)   // 冻结：反向只算 dx，不算 dW
};
let mut y = base.add(&self.bias);
if let Some(lora) = &self.lora {
    y = y.add(&lora.forward(&x));   // (α/r)·(x @ Aᵀ) @ Bᵀ
}
```

> 注意：`Tensor::matmul_frozen` 不是"优化技巧"，而是**语义正确性**的一部分——
> 冻结参数不该产生梯度，也就没必要为它保留 `dW` 的计算图和显存。

---

## 5. 初始化策略

### 5.1 A：正态分布 N(0, σ²)

```
σ = 1 / √r
```

保证初始 ΔW = BA 的方差不会太大。

### 5.2 B：全零

```
B = 0 → ΔW = (α/r)·B·A = 0
```

**关键**：训练开始时 ΔW = 0，模型行为和预训练模型完全一致。
这保证了微调不会"破坏"预训练学到的知识。

本项目的 `LoraAdapter::new` 就是这么做的：

```rust
let a_scale = 1.0 / (rank as f32).sqrt();                 // σ = 1/√r
let a = Tensor::param(rng.randn() * a_scale, [r, in]);    // A ~ N(0, 1/r)
let b = Tensor::param(vec![0.0; out * r], [out, r]);      // B = 0
```

单测 `test_lora_zero_init_keeps_forward_identical` 断言：挂上适配器前后，同一输入的前向输出**逐位相同**。

### 5.3 为什么"A 随机、B 为零"，而不是反过来或都随机

看两个梯度（`h = x · Aᵀ`，`g = ∂L/∂y`）：

```
∂L/∂B = (α/r) · hᵀ · g      ∂L/∂A = (α/r) · xᵀ · (g · B)
```

| 初始化 | ΔW(初始) | 梯度是否为零 | 结果 |
|---|---|---|---|
| A=0，B=0 | 0 | **两者都是 0** | 永远不动，适配层是死的 |
| A=随机，B=随机 | ≠ 0 | 都不为 0 | 一上来就偏离预训练，r 大时更明显 |
| **A=随机，B=0** | **0** | ∂L/∂B ∝ h ≠ 0（A 非零，h 才有值） | **起点等价于原模型，又能正常学** |

第三行才是对的：B 为零保证起点不动，而 A 非零让 `h` 有值、`∂L/∂B` 非零，于是 B 先动、
A 随后跟上（B 一动，`∂L/∂A` 也就不再为零）。若两个都是零，两条梯度同时为零，训练直接卡死。

---

## 6. α 缩放因子

```
ΔW = (α / r) · B · A
```

- α 通常设为 r（即不缩放，scaling = 1）
- 增大 α = 放大 LoRA 的影响（学习率等效变大）
- 减小 α = 缩小 LoRA 的影响（更保守的微调）

**使用建议**：先用 α = r，如果效果不好再调整。

实现上就是 `LoraAdapter` 里的一个乘数——`(α/r)` 只在最后乘一次，不参与 A/B 的初始化：

```rust
pub fn scaling(&self) -> f32 { self.alpha / self.rank as f32 }
// forward: (x @ Aᵀ) @ Bᵀ 再 mul_scalar(scaling())
```

所以改 α **不会**改变适配层的初始状态（ΔW 恒为 0），只是把增量的尺度整体放大/缩小——
单测 `test_lora_forward_matches_formula` 断言了"α 翻倍 ⇒ 增量翻倍"。

---

## 7. 应用场景

### 7.1 本项目里的调用链

不用手写注入逻辑，`finetune` 子命令一条命令就走完整条路（见 [§8](#8-接入系统finetune-子命令)）。
直接调 API 的话是这样：

```rust
// 1. 加载预训练模型（checkpoint 的头里若记了 LoRA 形态，会先按它重建适配层再灌参数）
let (mut model, tokenizer, _) = load_model_and_tokenizer("checkpoints/zh/latest.ckpt", ...)?;

// 2. 先冻结整网，再按 targets 给对应投影挂上适配器（顺序不能反：见 GPT::apply_lora 的注释）
let lora = LoRAConfig { rank: 8, alpha: 8.0, targets: LoRATargets::default() };  // 缺省 q,k,v
model.apply_lora(&lora, &mut rng);

// 2'. 若基座本身就是 LoRA 存档，想接着训存档里那套就改用 resume_lora（保留旧 A/B 并解冻）
// model.resume_lora();

// 3. 只有适配层是可训练的——trainable_parameters() 就是唯一该喂给优化器的那组
let trainable = model.trainable_parameters();          // 缺省 24 个张量：3 投影 × 2 矩阵 × 4 层
let opt = AdamW::new(5e-5, model.parameters(), 0.1);   // 传全集也行，step() 会自动跳过冻结参数

// 4. 反向前向都无需改动：Linear::forward 自己会把增量加上，checkpoint 头会记下 LoRA 形态

// 5. 只想推理的话，可以把增量就地并进主干：model.merge_lora()（不可逆，之后不能再续训）
```

### 7.2 给哪些层加 LoRA

通用的经验分布：

| 层 | 是否加 LoRA | 原因 |
|---|-----------|------|
| Q/K/V 投影 | ✅ | 注意力是微调的核心 |
| 输出投影 | ✅ | 影响注意力输出，但收益相对小 |
| MLP | 可选 | 效果有限 |
| Embedding | ❌ | 词表变化少 |
| LayerNorm | ❌ | 参数太少 |

**本项目缺省只给 Q/K/V 加，但挂载位置是可配的**（[`LoRATargets`](../src/config.rs)，由 `--lora-targets` 决定）：
Q/K/V 是注意力里信息量最大的一组，缺省选它是因为参数量占比最"划算"；`o`（`c_proj`）与 `mlp`
缺省关闭，想扩就写 `--lora-targets q,k,v,o` 或直接 `all`。

| `--lora-targets` | 挂哪几个 `Linear` | 每层对数 | 实测可训练参数（rank=8） |
|---|---|---|---|
| `q,k,v`（缺省） | `c_q` / `c_k` / `c_v` | 3 | 24,576（1.32%） |
| `q,k,v,o` | 再加 `c_proj` | 4 | 32,768（1.75%） |
| `all` | 再加 MLP 的两个线性层 | 6 | 73,728（3.85%） |

（上表数字实跑自 [config/config.json](../config/config.json) 的 1.84M 小模型；GPT-2 风格 MLP 是 2 个线性层，
换成 SwiGLU 时是 3 个，`all` 的每层对数相应变成 8。）
别名：`proj` / `c_proj` → `o`，`ffn` / `mlp` → `mlp`；大小写与空格随意、重复项自动去重、没命中的词直接报错。
**挂载结构写进 checkpoint 头部**（`lora.targets`），推理端按头部记录重建，不需要再传一遍——
所以训练与推理的挂载位置永远一致，忘了传参也不会静默错配。旧档没有 `targets` 字段，
`#[serde(default)]` 让它回落成 `q,k,v`，读起来与从前完全一样。

---

## 8. 接入系统：`finetune` 子命令

LoRA **已完整接入**：`finetune` 就是"LoRA 版的 SFT"——同一套训练循环、同一份对话语料
（`train.sft_file`，可用 `--sft-file` 覆盖），唯一差别是**参数集合**：

| | `sft` | `finetune` |
|---|---|---|
| 可训练参数 | 全部（1,866,496） | 只训适配层（24,576，1.32%） |
| 冻结主干 | ❌ | ✅（三处一起做实，见下） |
| 输出目录 | `checkpoints/zh-sft/` | `checkpoints/zh-lora/` |
| 指标 CSV | `sft.csv` | `lora.csv` |

### 8.1 冻结是"真的"冻结，三处一起做实

缺任何一处都会出现"以为冻结了、其实没冻"的隐性错误：

| 位置 | 做法 | 不做会怎样 |
|---|---|---|
| 前向 | 冻结权重走 [`Tensor::matmul_frozen`](../src/tensor.rs)：反向只求 `dx`、不算 `dW` | `dW` 白算白存一遍（4 字节/参数 × 1.87M） |
| 优化器 | `AdamW::step` / `SGD::step` 跳过 `!requires_grad()`，**连带不做权重衰减** | 衰减项 `lr·wd·θ` 与梯度无关，冻结主干会每步朝 0 缩 |
| GPU 常驻快路 | `attn_resident` / `mlp_resident` / `blocks_resident` 整体让路 | 它们直接读权重显存、绕过 `Linear::forward`，适配层增量被**静默丢弃** |

还有一个容易踩的坑：`requires_grad` 必须是**共享标志**。本项目早期它是普通 `bool`，
而 `Tensor` 派生了 `Clone`——在模型上冻结、优化器手里那份仍是 `true`，冻结就失效了。
现在它是 `Rc<Cell<bool>>`，模型与优化器看的是同一个开关。

### 8.2 命令行用法

```bash
# LoRA 微调（缺省 1000 步、lr 1e-4、rank 16、alpha=rank、挂 q,k,v，语料取 config 的 train.sft_file）
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt

# 本项目实测用的配方：rank=8、alpha=8、60 步、lr=5e-5（与 SFT 的最优配方同量级）
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt \
    --lora-rank 8 --lora-alpha 8 --steps 60 --lr 5e-5

# 扩挂载：O 投影 → 再加 MLP（实测 1.32% → 1.75% → 3.85%）
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt --lora-targets all

# 链式续训：基座是 LoRA 存档时，接着训存档里那套适配层（结构照存档，不能再传 rank/alpha/targets）
cargo run --release -- finetune --pretrained checkpoints/zh-lora/best.ckpt --resume-lora --steps 60

# 换一份语料 / 指定输出目录
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt \
    --sft-file data/sft/ --out-dir checkpoints/zh-lora2
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config <路径>` | `config/config.json` | 配置文件路径 |
| `--pretrained <路径>` | 必填 | 主干来源 checkpoint |
| `--lora-rank <秩>` | `16` | 秩 r；越小适配层越少、越不容易过拟合 |
| `--lora-alpha <系数>` | `= rank` | 缩放因子 α（缺省 = rank，即增量不额外缩放） |
| `--lora-targets <位置>` | `q,k,v` | 挂载位置，逗号分隔：`q` / `k` / `v` / `o` / `mlp` / `all`（见 §7.2） |
| `--resume-lora` | 关 | **链式续训**：接着训存档里那套适配层（保留已学增量），与 rank/alpha/targets 互斥 |
| `--sft-file <路径>` | config 的 `train.sft_file` | 对话语料，逗号分隔、可含 `*` 通配 |
| `--steps <步数>` | `1000` | 微调步数 |
| `--lr <学习率>` | `1e-4` | 微调学习率 |
| `--out-dir <目录>` | `{out_dir}-lora` | 输出目录（缺省加 `-lora` 后缀，绝不覆盖预训练权重） |

### 8.3 实测：一次真实的 LoRA 微调

命令：`finetune --pretrained checkpoints/zh/latest.ckpt --lora-rank 8 --lora-alpha 8 --steps 60 --lr 5e-5`

```
LoRA 微调：rank=8 alpha=8 挂载=q,k,v steps=60 lr=0.00005｜冻结主干，只训适配层
模型参数：1866496（含冻结主干）｜可训练：24576（1.317%）｜12 对低秩矩阵（挂载 q,k,v），共 24 个适配参数张量
SFT 语料：89 段对话，打包 17898 token | 监督位置（回答段）占 56.1%
LoRA：rank=8 alpha=8 挂载=q,k,v｜可训练参数 24576 / 1866496（1.32%）｜主干冻结：反向不算 dW、优化器不更新、不吃权重衰减
...
step    36 | lr 0.000024 | loss 6.1391 | val 6.5640 (ppl 709.1)
step    54 | lr 0.000006 | loss 6.2499 | val 6.5596 (ppl 706.0) *
step    60 | lr 0.000005 | loss 6.3406 | val 6.5813 (ppl 721.5) | 515 tok/s
[done] checkpoint 已保存到 checkpoints/zh-lora/ | 总耗时 478s | 7.96s/步（共 60 步）
```

产物（目录自包含，含冻结主干）：

```
checkpoints/zh-lora/{best,final,latest}.ckpt   tokenizer.json   lora.csv
```

**存档变大了多少**：22,106,657 → 22,402,843 字节，差 296,186 = `24,576 个适配参数 × 4 字节 × 3 段数据块
（参数 / m / v）` + 1,274 字节（JSON 头里多了 24 个参数条目和 `lora` 字段）——主干那部分的字节数一个没变。

> 这只是"体积账"。**数值**真的没动，由两条测试合起来保证：
> `model.rs` 的 `test_apply_lora_freezes_backbone_and_trains_adapters_only`（一步真实 AdamW
> 训练后，全部冻结参数逐位不变）+ `checkpoint.rs` 的 `test_save_load_roundtrip_is_bit_exact`
> （存/读逐位还原）。链路上任何一环有损，这两个断言就先挂。

> 别把这个 val 当成 LoRA 效果好不好：SFT 语料只有 89 段、验证集是切出来的 9 段，
> 这个数字**没有判别力**（[第 39 课](39-工程化完善.md) 的 SFT 也是这个坑）。真正要看的是
> ① 预训练 val 有没有退化（主干冻结 ⇒ 不会）② `chat` 聊两句有没有怪字。

### 8.4 配置文件方式

也可以在 `config/config.json` 里写好 LoRA 形态（CLI 参数会覆盖它；注意 `train` 子命令
**不**注入适配层，它只读这份配置来打印提示）：

```jsonc
{
  "train": {
    "lora": {
      "rank": 16,
      "alpha": 16.0,
      "targets": { "q": true, "k": true, "v": true, "o": false, "mlp": false }
    }
  }
}
```

`targets` 可以整块省略（回落 `q,k,v`）；块内也可以只写想改的那几个（缺的字段各自回落缺省）。
所以旧配置**一个字都不用改**，读进来的行为与从前逐位相同。

**谁覆盖谁**（三个字段各自独立判定，CLI 没传才看配置）：

| 字段 | 判定顺序 |
|---|---|
| rank | `--lora-rank` → `config.lora.rank` → `16` |
| alpha | `--lora-alpha` → `config.lora.alpha`（**仅当配置文件里写了 `lora` 块**）→ `rank` |
| targets | `--lora-targets` → `config.lora.targets` → `q,k,v` |

alpha 这一档多一个条件，是为了让"没写配置 + 只传 `--lora-rank 8`"仍然得到 α=8（缺省 α=rank 的直觉），
而不是被 `LoRAConfig::default()` 里的 16.0 顶掉。实测：配置写 `"rank": 4, "alpha": 12.0, "targets": { "v": true, "mlp": true }`，
不传任何 LoRA 参数跑 `finetune`，日志是 `rank=4 alpha=12 挂载=q,k,v,mlp`（`q,k` 由缺省补上）。

### 8.5 工作流程（`finetune` 实际执行的顺序）

1. 加载预训练 checkpoint；若该档本身是 LoRA 形态，先按档案头重建适配层再灌参数
2. 决定怎么挂：
   - 默认路径 —— 冻结整网 `for p in model.parameters() { p.set_requires_grad(false) }`，
     再按 `targets` 给对应投影各挂一对 A/B（**顺序不能反**：冻结要发生在适配层出生之前）
   - `--resume-lora` —— 跳过"重挂"，改为解冻档案里已有的 A/B（数值原样保留，不重随机）
3. 只优化 `trainable_parameters()`，日志如实打印可训练占比
4. 保存 checkpoint：头部记下 `lora: {rank, alpha, targets}`，参数块按名字顺序带上 `*.lora_a` / `*.lora_b`
5. 推理时 `load_model_and_tokenizer` 按头部记录**先注入再恢复参数**，名字与形状才能对齐

> 第 5 步是设计上的一个约束：LoRA 存档**能独立加载**（主干也在文件里，不必再找基座），
> 但加载端必须知道"这是个 LoRA 档案"。忘了注入就直接 `load_params` 会立刻报错，
> 而不是悄悄装进去一半——`test_lora_checkpoint_roundtrip` 专门断言了这个 panic。

### 8.6 链式续训：`--resume-lora`

把一个 LoRA 存档当基座有两种截然不同的语义，`finetune` 默认做的是**前者**：

| | 默认（重挂一套） | `--resume-lora`（链式续训） |
|---|---|---|
| 旧 A/B | **丢弃**，只继承主干 | **保留**，从现有数值继续 |
| rank / alpha / targets | 由 CLI 或 config 决定 | 一律照存档，再传就报错 |
| 适用 | 换新任务、想清掉旧增量 | 同一任务接着磨、多轮迭代 |

缺省打"重挂一套"是因为它更符合直觉（"给我在新数据上微调"），但对已经微调过的存档来说，
不说明就悄悄丢掉旧增量是危险的，所以这条路径会额外打一行 `[warn]` 提示，
并把 `--resume-lora` 的写法直接印出来。

实现上只有三步差异（[`GPT::resume_lora`](../src/model.rs)）：**不覆盖**（沿用已注入的 A/B）、
**解冻**（把 `lora_parameters()` 逐个 `set_requires_grad(true)`，其余仍是 `false`）、
**rank 必须一致**（结构来自存档，想换 rank 只能重挂）。

### 8.7 推理合并：`--merge-lora`

`eval` / `generate` / `chat` 都接受 `--merge-lora`，把增量**就地**并进主干权重：

```text
W ← W + (α/r)·AᵀBᵀ
```

（注意方向：本项目 `A` 是 `[r, in]`、`B` 是 `[out, r]`，所以是 `Aᵀ·Bᵀ` 而不是 `B·A`。）

两个必须说清的点：

- **数值不变**。合并只是把 `y = xW + (α/r)(xAᵀ)Bᵀ` 里的两项提前相加，前向数学等价。
  实测同一 seed 下合并前后生成结果**逐字相同**；单测 `test_merge_lora_matches_two_branch_forward`
  断言偏差 `< 1e-4`（浮点加法次序不同，不是逐位相等）。
- **必须就地改**。用 `Tensor::set_data` 改原有张量的数据，而不是把 `Linear.weight` 换成一个新张量——
  优化器和其他持有者手里拿的是旧句柄，换了之后它们会各算各的。
  `test_merge_lora_into_weight_is_in_place` 用旧句柄读新值来钉死这一点。

合并后适配器被摘除（`lora: None`），因为增量已经在 `W` 里了，留着会算两遍。
所以它是**不可逆的运行时操作**：合并过的模型对象不能再拿来续训，要续训请重新加载存档。
存档本身不受影响——`--merge-lora` 只作用于本次进程的内存副本，A/B 仍原样躺在文件里。

---

## 9. 关键要点

- LoRA 用低秩矩阵 ΔW = (α/r)·B·A 近似权重更新，可训练参数量降到 1% 左右
- 冻结预训练权重，只训练 A 和 B；冻结要在**前向 / 优化器 / GPU 快路**三处一起做实
- B 初始化为全零，保证训练开始时模型行为不变（前向逐位相同）
- 挂载位置可配（`--lora-targets`，缺省 Q/K/V，可扩到 O 与 MLP），结构写进存档头部
- 把 LoRA 存档当基座时要选语义：默认**重挂一套**（丢旧增量），`--resume-lora` 则**链式续训**（保留旧增量）
- 推理时可以把 ΔW 合并到 W（`--merge-lora`，切掉每层两次小矩阵乘）：
  合并**必须就地**改权重数据，且是**不可逆**的运行时操作（合并后就再也单独取不出增量了）。
  所以默认**不合并**——保留"主干 + 增量分开算"两条路，代价只是每层两次小矩阵乘
- 适配层要能被 checkpoint 记住形态（rank / alpha / targets），否则加载端无法对齐参数名
- 适合"在已有大模型上微调特定任务"的场景——本项目实测可训练参数 1.32%，主干零改动
