# 第 16 课：训练小 Transformer —— 看 loss 从 1.30 降到 0.16

> 代码位置：[src/main.rs](../src/main.rs)（`demo_transformer`）
> 代码位置：[src/train.rs](../src/train.rs)（`train_transformer` / `LRScheduler` / `clip_grad_norm`）
> 代码位置：[src/data.rs](../src/data.rs)（`CORPUS` / `DataLoader`）
> 代码位置：[src/sample.rs](../src/sample.rs)（`generate` / `sample_token`）

---

## 1. 本课要搞懂的问题

1. `demo_transformer` 从数据到生成文本，完整流程分哪几步？
2. 只有 669 个字符的小语料，训练日志里的 `step / lr / loss / tok/s` 四列怎么读？
3. 日志里为什么看不到 warmup 段？lr 从 `0.002945` 一路衰减到 `0.000300` 是怎么来的？
4. temperature、top-k、top-p 三个参数是怎么配合采样的？
5. 为什么 loss 已经降到 0.16，模型输出的文本依然只是"像样"而不是"正确"？

---

## 2. 训练全景：demo_transformer 做了什么

`src/main.rs` 的演示 3（第 12-21 课）是本节的主角：

```rust
fn demo_transformer() {
    println!("=== 演示 3：训练小 Transformer 并生成文本 ===");

    let mut rng = Rng::new(1234);
    let tokenizer = Tokenizer::char(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    println!("  语料 {} 字符，字符词表 {} 个", CORPUS.len(), vocab_size);

    let model = Transformer::new(TransformerConfig::tiny(vocab_size), &mut rng);

    // 训练（第 13、17-18 课：训练循环 + AdamW + warmup/cosine 调度）
    let loader = DataLoader::new(CORPUS, &tokenizer, model.cfg.block_size, 8);
    let tcfg = config::TrainConfig {
        seed: 42,
        batch_size: 8,
        steps: 600,
        max_lr: 3e-3,
        warmup_steps: 50,
        eval_every: 100,
        log_file: None, // 演示不落盘
        ..config::TrainConfig::default()
    };
    train::train_transformer(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);

    // 生成 1（无 cache）：全量前向用滑动窗口，可以生成超过 block_size 的长文本
    // 三次生成共用同一套采样参数（含重复惩罚），否则差异分不清是 cache 还是采样造成的
    let opts = SampleOpts {
        top_k: 10,
        repetition_penalty: 1.1,
        repetition_window: 64,
        ..SampleOpts::default()
    };
    println!("\n  —— 生成 1（prompt=Once upon a, 无 KV cache，全量前向）——");
    let mut rng_full = Rng::new(2024);
    let out_full = generate(&model, &tokenizer, "Once upon a", 80, &opts, KvOpts::off(), &mut rng_full);
    println!("  {out_full}");

    // 生成 2（带 KV cache，第 25 课）：**必须用同一个 prompt 和同一个种子**，否则两次输出
    // 不同只是采样不同，证明不了 cache 的正确性。
    // 缓存带滑动窗口（第 25 课第 8 节），生成长度同样不受缓存容量限制，不会提前停。
    println!("\n  —— 生成 2（同 prompt、同种子，带 KV cache）——");
    let mut rng_kv = Rng::new(2024);
    let out_kv = generate(&model, &tokenizer, "Once upon a", 80, &opts, KvOpts::on(0, None), &mut rng_kv);
    println!("  {out_kv}");
    println!(
        "\n  KV cache 带滑动窗口，生成长度不再受缓存容量限制：两种模式都生成满 80 个 token —— {}",
        if out_kv.chars().count() == out_full.chars().count() {
            "一致 ✓"
        } else {
            "不一致 ✗"
        }
    );

    // 窗口内的严格等价自检（第 25 课第 7 节）：prompt + 新 token 不超 `block_size` 时，
    // 缓存里存的 K/V 与全量重算逐位相同，两条路径必须**逐 token 完全相等**。
    //
    // 超出窗口后两条路径本就不等价，也不是 bug：增量推理里每个位置当时能看到自己的完整窗口，
    // 而"截断重算"会把窗口内各位置在浅层可见的上下文一并砍掉，深层 K/V 随之不同。
    // 带 KV cache 的推理才是真实 LLM 的标准推断语义，所以这里只在窗口内做严格断言。
    let prompt_len = tokenizer.encode("Once upon a").len();
    let in_window = model.cfg.block_size.saturating_sub(prompt_len);
    let mut rng_a = Rng::new(2024);
    let a = generate(&model, &tokenizer, "Once upon a", in_window, &opts, KvOpts::off(), &mut rng_a);
    let mut rng_b = Rng::new(2024);
    let b = generate(&model, &tokenizer, "Once upon a", in_window, &opts, KvOpts::on(0, None), &mut rng_b);
    println!(
        "\n  窗口内等价自检（prompt {prompt_len} + 新 {in_window} = {} ≤ block_size {}）：\
         缓存模式与全量模式逐 token {}",
        prompt_len + in_window,
        model.cfg.block_size,
        if a == b { "完全一致 ✓" } else { "不一致 ✗" }
    );
    if a != b {
        println!("    全量：{a}\n    缓存：{b}");
    }

    // 生成 3：换一个训练语料里没出现过的开头，观察小模型的真实水平（第 16 课第 7 节）。
    // 用全量前向，这样输出的毛病都归模型自己，不被缓存路径的任何细节搅混。
    println!("\n  —— 生成 3（prompt=The fox，换开头看泛化）——");
    let mut rng_fox = Rng::new(2024);
    let out_fox = generate(&model, &tokenizer, "The fox", 80, &opts, KvOpts::off(), &mut rng_fox);
    println!("  {out_fox}");
}
```

整个流程可以拆成 5 步：

| 步骤 | 代码 | 做了什么 |
|------|------|---------|
| 1. 分词 | `Tokenizer::char(CORPUS)` | 扫描语料，得到 38 个 token 的词表（35 个字符 + BOS / EOS / PAD 三个特殊 token） |
| 2. 建模型 | `Transformer::new(TransformerConfig::tiny(vocab_size), &mut rng)` | 用 tiny 配置（n_embd=64、n_head=4、n_layer=2、block_size=32、RMSNorm、SwiGLU）初始化模型 |
| 3. 造数据 | `DataLoader::new(CORPUS, &tokenizer, model.cfg.block_size, 8)` | 把 669 字符的语料切成 671 个 token（每篇文档首尾各加 BOS / EOS），按 block_size=32 切块、batch_size=8 |
| 4. 训练 | `train_transformer(&model, &tokenizer, &loader, &tcfg, None, None, ...)` | 600 步，峰值学习率 3e-3，前 50 步 warmup，每 100 步打印一次（其余参数取 `TrainConfig::default()`） |
| 5. 生成 | `generate(&model, &tokenizer, "Once upon a", 80, &opts, KvOpts::off(), &mut rng)` | 给定开头，最多续写 80 个字符（采样参数打包在 `opts: SampleOpts` 里，KV cache 开关在 `KvOpts` 里） |

> 注意：训练用的是字符级分词器，所以"1 个字符 = 1 个 token"，669 字符的语料编码后是 671 个 token（首尾各有 BOS / EOS）。这让后面的数字（32、80）可以直接按"字符数"理解。

---

## 3. 数据：669 字符的小语料

`src/data.rs` 里内置了一篇英文小故事（狐狸 Red 找金钥匙）：

```rust
pub const CORPUS: &str = "\
Once upon a time in a small village, there lived a curious little fox named Red. \
Every morning, Red would wake up early and explore the forest. ...";
```

训练数据是**自监督**的：输入 x 是一段 32 个 token 的序列，标签 y 是 x 右移一位——每个位置都预测"下一个字符是谁"，文本自己就是标签，不需要人工标注。

`DataLoader::sample_batch` 每次随机选 8 个起点，各截 33 个 token（前 32 个作 x，后 32 个作 y）。它本身只有几行，把区间交给 `sample_region`，切窗口的双循环在 `sample_region` 里：

```rust
fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>, Option<Vec<bool>>) {
    let (x, y) = self.sample_region(rng, 0, self.val_start, "训练"); // val_start = tokens.len()（整段都是训练数据）
    (x, y, None)   // 第三个值是 SFT 的 loss 掩码，预训练不需要
}

fn sample_region(&self, rng: &mut Rng, lo: usize, hi: usize, tag: &str) -> (Vec<usize>, Vec<usize>) {
    assert!(hi > lo + self.block_size, "{tag}区数据不足，无法采样");
    // 起点上界是 hi - block_size：窗口要取到 tokens[start + block_size]（`y` 右移一位）
    let max_start = hi - lo - self.block_size;
    let mut x = Vec::with_capacity(self.batch_size * self.block_size);
    let mut y = Vec::with_capacity(self.batch_size * self.block_size);
    for _ in 0..self.batch_size {
        let start = lo + rng.choice(max_start);
        for j in 0..self.block_size {
            x.push(self.tokens[start + j]);
            y.push(self.tokens[start + j + 1]);
        }
    }
    (x, y)
}
```

关键点：

- **随机采样而非顺序扫描**：每次 `sample_batch` 都在语料里随机挑起点。语料只有 671 token，但 600 步 × 8 个 batch 会反复"看到"语料的不同片段（有些片段会被重复看，有的可能一次都没被抽到）——小语料训练天然就是"背课文"。
- 返回的 x、y 都是 `[B*T] = [8×32] = [256]` 的展平数组，正好满足 `Transformer::forward(idx, b=8, t=32, kv_cache, training)` 的输入要求（训练时 `kv_cache` 传 `None`、`training` 传 `true`）。

---

## 4. 超参数一览

`train_transformer` 的调用参数与 `TransformerConfig::tiny` 汇总：

| 超参数 | 值 | 含义 |
|--------|----|------|
| `steps` | 600 | 总训练步数 |
| `batch_size` | 8 | 每步采样 8 条序列（每条 32 token） |
| `block_size` | 32 | 最大上下文长度，来自 `TransformerConfig::tiny` |
| `max_lr` | 3e-3 | 学习率峰值 |
| `warmup_steps` | 50 | 前 50 步学习率从 0 线性爬升到峰值 |
| `min_lr` | 3e-4 | cosine 衰减的终点，作为第 4 个参数传给 `LRScheduler::new`（demo 用 `TrainConfig::default()` 的值，恰好是 max_lr × 0.1） |
| `weight_decay` | 0.01 | AdamW 的权重衰减（第 17 课） |
| `grad_clip`（梯度裁剪） | 1.0 | 梯度范数上限，取自 `TrainConfig::default()`（demo 没覆盖） |
| `eval_every` | 100 | 每 100 步打印一次日志 |

模型参数量：`train_transformer` 开头会打印一行"开始训练"（真实数字就在其中）：

```
开始训练：char（vocab=38）模型参数 135488 | 语料 671 tokens（训练 671 / 验证 0）| batch=8 block=32
```

按第 12 课的方法验证一下：词表 V=38（35 个字符 + BOS / EOS / PAD，不是 100）时，

| 组成 | 计算 | 参数 |
|------|------|------|
| `tok_emb` | 38 × 64 | 2432 |
| 每层 Block | ln1 64（RMSNorm 只有 γ）+ attn 16640（c_q/c_k/c_v/c_proj 各 64×64+64）+ ln2 64 + SwiGLU FFN 49728（w_gate/w_up 各 64×256+256、w_down 256×64+64） | 66496 |
| `ln_f` | 64 | 64 |
| 输出头 | 权重绑定（复用 `tok_emb` 转置，无独立 lm_head 参数） | 0 |

总计 **2432 + 2×66496 + 64 = 135488** ✓（`n_kv_head=0` 走标准 MHA，Q/K/V 同宽；SwiGLU 的隐藏维 `swiglu_hidden(64) = 256`）。约 13.5 万参数，CPU 上几十秒就能跑完整个 demo。

---

## 5. 真实训练日志解读

运行 `cargo run --release -- demo`，演示 3 会打印（这是**真实运行输出**，不是编的）：

```
=== 演示 3：训练小 Transformer 并生成文本 ===
  语料 669 字符，字符词表 38 个
开始训练：char（vocab=38）模型参数 135488 | 语料 671 tokens（训练 671 / 验证 0）| batch=8 block=32
[info] AMP 已启用：初始 scale = 2^16 = 65536，每 2000 次无溢出翻倍，溢出则跳过本步并把 scale 减半（梯度裁剪前会反缩放回真实尺度）
step   100 | lr 0.002945 | loss 1.3001 | 1564 tok/s
step   200 | lr 0.002534 | loss 0.3962 | 1747 tok/s
step   300 | lr 0.001842 | loss 0.2236 | 1973 tok/s
step   400 | lr 0.001089 | loss 0.2401 | 2175 tok/s
step   500 | lr 0.000514 | loss 0.1552 | 2353 tok/s
step   600 | lr 0.000300 | loss 0.1586 | 2459 tok/s
```

> 日志按 5 秒节流还会插入 `[train] step 53/600 | loss … | grad … | lr … | … st/s | … tok/s | …` 形式的进度行，
> 上面只摘了每 100 步的评估行。`loss` 与 `lr` 因为固定种子（`seed = 42`）可复现，`tok/s` 随机器而变。
> 那个 `[info] AMP 已启用` 来自 `TrainConfig::default()` 的 `amp = true`（第 26 课）：loss 先乘动态 `scale` 再反向，
> 更新前检查溢出并反缩放。`scale` 恒为 2 的幂，f32 下乘除是精确的指数移位，所以开不开 AMP 的 loss 逐位相同。

### 5.1 四列日志分别是什么

| 列 | 含义 | 从哪来 |
|----|------|--------|
| `step` | 训练步数（从 1 开始数，日志显示 100、200、…、600） | `train_transformer` 打印的是 `step + 1` |
| `lr` | 打印时刻 `scheduler` 里的学习率 | `scheduler.lr()`，且是在本步 `scheduler.step()` **之后**读取的（`src/train.rs` 的进度行与评估行都在更新后打印）——即"下一步要用"的 lr，不是本步已用的 `cur_lr` |
| `loss` | 本步 batch 的平均交叉熵 | `forward_loss(&model, &x, &y, b, t, accum)`（内部调用 `cross_entropy_loss`，返回的是未缩放的原始 loss） |
| `tok/s` | 训练吞吐：已处理 token 数 ÷ 已耗时 | `tps = steps_done × batch_size × block_size / elapsed` |

> demo 没有验证集，所以日志里没有 `val` / `ppl` 两列；有验证集时 `train_transformer` 还会打印 `val {:.4} (ppl {:.1})`。

`train_transformer` 的循环分两档：**每步**做「采样 → 前向+损失 → 反向」；「裁剪 → 更新 → 清零 → 调度器前进」只在**每个累积窗口结束时**执行一次（`accum_steps = 1` 时才是每步一次，demo 就是这种默认情况）。日志打印也在累积窗口结束时，另有每 `eval_every` 步的评估：

```rust
for step in start_step..cfg.steps {
    let (x, y) = loader.sample_batch(rng);                  // 1. 采样 batch（每步）
    let loss = forward_loss(&model, &x, &y, b, t, accum);   // 2. 前向 + 交叉熵（每步）
    loss.backward();                                        // 3. 反向（每步，梯度累加到现有梯度上）

    // 4~6 只在累积窗口结束时执行；最后一步即使不满 accum 也强制收尾
    if (step + 1) % accum == 0 || step + 1 == cfg.steps {
        clip_grad_norm(&params, cfg.grad_clip);             // 4. 梯度裁剪
        let cur_lr = scheduler.lr();                        // 5. 取当前步 lr 喂给优化器
        opt.lr = cur_lr;
        opt.step();                                         //    更新参数
        opt.zero_grad();                                    // 6. 清零梯度
        scheduler.step();                                   //    调度器前进（只在真正更新后递增）
        // 进度日志（实际实现每 5 秒最多打印一次，格式见上文日志；这里是精简写法）
        // 打印的是 scheduler.step() **之后**的 lr，即"下一步要用"的值；step 打印 step + 1
        println!("step {:>5} | lr {:.6} | loss {:.4} | {:.0} tok/s", step + 1, scheduler.lr(), loss.item(), tps);
    }

    // 每 eval_every 步（或最后一步）：评估 + 存 checkpoint
    if (step + 1) % cfg.eval_every == 0 || step + 1 == cfg.steps { /* ... */ }
}
```

### 5.2 loss：1.30 → 0.16 说明了什么

- **第一个打印点 1.30**：日志只在 `step 100、200、…` 打印（`eval_every = 100`）。随机初始化时模型对 38 个 token 基本"一视同仁"，理论下界是均匀分布的交叉熵 `ln(38) ≈ 3.64`；训练 100 步后降到 1.30，说明已经开始学习。
- **先快后慢**：step 100→300 loss 从 1.30 掉到 0.22（降了约 83%），step 300→600 只在 0.16～0.24 之间小幅摆动。这是训练曲线的典型形态——早期梯度大、方向明确，后期接近收敛、只能精雕细琢。
- **相邻打印点会上下跳**：评估 loss 是**单个 batch** 的平均（不是整份语料），所以 step 400 的 0.2401 比 step 300 的 0.2236 略高是正常的统计波动，不代表模型变差。
- **终点 0.16**：交叉熵 0.16 意味着模型给"正确下一个字符"的平均概率约为 `exp(-0.16) ≈ 0.85`。对一篇 669 字符的"课文"来说，模型已经相当好地"背"下了其中的统计规律。

### 5.3 warmup 阶段：为什么日志里看不到

`LRScheduler` 的规则（`src/train.rs`）：

```rust
pub fn lr(&self) -> f32 {
    if self.step < self.warmup_steps {
        // 线性 warmup
        self.max_lr * (self.step as f32 + 1.0) / self.warmup_steps.max(1) as f32
    } else {
        // cosine 衰减
        let progress = (self.step - self.warmup_steps) as f32
            / (self.total_steps - self.warmup_steps).max(1) as f32;
        let progress = progress.min(1.0);
        let cosine = 0.5 * (1.0 + (std::f32::consts::PI * progress).cos());
        self.min_lr + (self.max_lr - self.min_lr) * cosine
    }
}
```

warmup 就是前 50 步让学习率**线性爬升**：

```
lr(step) = max_lr × (step + 1) / warmup_steps     （step < 50 时）
```

代入 `max_lr = 0.003`、`warmup_steps = 50`：

| scheduler.step（= 日志里的 step） | 计算 | lr |
|----------------|------|----|
| 0（真正用于第 1 步更新） | 0.003 × 1 / 50 | 0.00006 |
| 1 | 0.003 × 2 / 50 | 0.00012 |
| 25 | 0.003 × 26 / 50 | 0.00156 |
| 49 | 0.003 × 50 / 50 | 0.003（warmup 段峰值） |
| 50（`step < warmup_steps` 不再成立，切到 cosine 分支） | progress = 0 → cosine = 1 | 0.003 |

> 注意：demo 的 `eval_every = 100`，warmup 段（step 0-49）**没有打印点**，所以真实日志里看不到 0.00006 起步的爬升。
> 把 `eval_every` 改成 10，就能看到 step 10/20/30/40 的 lr = `0.00066 → 0.00126 → 0.00186 → 0.00246`
> （每步增加 `0.003/50 = 0.00006`，10 步就是 0.0006）。
>
> 为什么要 warmup？训练刚开始时参数是随机值，梯度方向噪声大、量级不可控。如果一上来就用 0.003 的大步长，很容易把参数"推飞"（loss 直接变成 NaN）。先用小步长稳住方向，再逐渐加力，是现代 LLM 训练的标准做法。

### 5.4 cosine 衰减：从峰值平滑降回 min_lr

第 50 步之后走 cosine 曲线，从 `max_lr = 0.003` 平滑降到 `min_lr = 0.0003`（`config::TrainConfig::default()`
里恰好等于 `max_lr × 0.1`，但 `train_transformer` 并不做这个换算，直接用 `cfg.min_lr`）：

```
lr = min_lr + (max_lr - min_lr) × 0.5 × (1 + cos(π × progress))
progress = (step - 50) / (600 - 50)，超过 1 就截断到 1
```

验证日志里的两个数字（日志里的 `lr` 读的是 `scheduler.step()` **之后**的值，所以 `scheduler` 计数就等于日志的 step 号）：

- `step 100`：scheduler 计数 = 100，`progress = (100-50)/550 ≈ 0.0909`，`cosine ≈ 0.9595`，`lr = 0.0003 + 0.0027×0.9595 ≈ 0.002945` ✓
- `step 600`：scheduler 计数 = 600，`progress = (600-50)/550 = 1`，`cosine = 0`，`lr = min_lr = 0.000300` ✓

学习率全程曲线：

```
lr
│
0.003 ┤        ╭╮
      │       ╭╯ ╰╮
0.002 ┤      ╭╯    ╰╮
      │     ╭╯      ╰╮
0.001 ┤    ╭╯        ╰╮
      │   ╭╯          ╰╮
0.0003┤──╯             ╰────── (min_lr)
      └──┬────┬────┬────┬────→ step
         0   100  200  300  400  500  600
         └warmup(50步)┘└─── cosine 衰减 ───┘
```

后期的"小步慢走"是为了在 loss 接近收敛时不震荡、精细地落到更优的参数点。

---

## 6. 生成文本与采样参数

训练 600 步后调用 `generate`（`src/sample.rs`），采样参数打包成 `SampleOpts { temperature: 0.8, top_k: 10, top_p: 0.9, repetition_penalty: 1.1, repetition_window: 64 }`（demo 里显式覆盖 `top_k`、`repetition_penalty`、`repetition_window`，其余取默认值；三次生成共用同一套参数，差异才只能归因于缓存）。

`sample_token` 内部的 7 步采样管线：

| 步骤 | 代码 | 作用 |
|------|------|------|
| 1. 重复惩罚 | 最近出现过的 token：正 logit 除以系数、负 logit 乘以系数 | 压低刚说过的字，防止卡在"太太太太……" |
| 2. 温度缩放 | `*l *= 1.0 / opts.temperature.max(1e-5)` | 除以 0.8：logits 变大 → softmax 更"锐利"，更敢选高概率 token |
| 3. 排序 | `items.sort_by(...)` | 按分数从高到低排 |
| 4. top-k | `items.truncate(opts.top_k)` | 只留前 10 个 |
| 5. softmax | `(*v - max).exp()` 再归一化 | 把截断后的分数变成概率 |
| 6. top-p | 累积概率到 0.9 截断 | 进一步砍掉长尾低概率 token，再归一化 |
| 7. 抽样 | `rng.next_f32()` 按概率累积选取 | 有随机性地选一个 token |

真实生成结果（`cargo run --release -- demo` 原样输出）：

```
  —— 生成 1（prompt=Once upon a, 无 KV cache，全量前向）——
  Once upon a time in a small village, there lived a curious litth., F was, Rere w.ni F re He

  —— 生成 2（同 prompt、同种子，带 KV cache）——
  Once upon a time in a small village, there lived a curious litt. Fradtlereom Red Redexcopat

  KV cache 带滑动窗口，生成长度不再受缓存容量限制：两种模式都生成满 80 个 token —— 一致 ✓

  窗口内等价自检（prompt 11 + 新 21 = 32 ≤ block_size 32）：缓存模式与全量模式逐 token 完全一致 ✓

  —— 生成 3（prompt=The fox，换开头看泛化）——
  The fox st. He loved to watch the birds fly and the rivers flow. OneWhe He,re diseriy f
```

读这段输出：

- **生成 1 vs 生成 2 是严格对照**：同一个 prompt、同一个种子（`Rng::new(2024)`），只切换 `KvOpts::off()` 与 `KvOpts::on(0, None)`。两段都是 80 个 token 长——早期实现里 `generate` 一旦把缓存填满 `block_size` 就 `break`，加不加 cache 会得到**长度不同**的输出；现在缓存自带滑动窗口（第 25 课第 8 节），生成长度只由 `max_new` 决定，与缓存容量无关。两段开头一大段逐字符相同，之后才分叉：出窗以后"增量推理"与"截断重算"本来就不是同一个函数（第 25 课第 8 节有证明与反例），这不是 bug。
- **窗口内等价自检才是严格证据**：把 `max_new` 压到 `block_size - prompt`（11 + 21 = 32），两种模式必须**逐 token 完全相等**——缓存里存的就是同一批 K/V，注意力与采样路径一个数都没变，所以分布必然相同。超过窗口后只断言"长度一致"，因为那时两条路径的语义已经不同。
- **生成 3 才暴露真实水平**：换一个语料里没出现过的开头，"The fox st. He loved to watch..." 语法不通、词都拼不成（生成 3 也走全量前向，不涉及缓存，所以这些毛病全是模型自己的）。

---

## 7. 为什么小模型输出只是"像样"而非"正确"

四个层面叠加，缺一不可：

| 原因 | 说明 |
|------|------|
| **语料太小** | 只有 669 字符、单一故事。模型只能"背"这篇课文里的统计规律，从未见过通用英语，谈不上泛化 |
| **模型太小** | 13.5 万参数 vs 真实 LLM 的数十亿～万亿参数。容量只够记住局部 n-gram 统计（"Red" 后常跟动词、名词前常有 the），装不下真正的语法规则 |
| **训练不足** | 600 步后 loss 仍为 0.16（正确概率约 85%），还没收敛到 0。模型对很多位置仍"没把握" |
| **采样带随机性** | temperature=0.8 + top-k/top-p 是有意引入随机性。即使模型 100% 会预测 "world"，采样也可能选到 "wold"——这是"创造性"的代价 |

用一句话总结：**"像样"来自学到了语料的高频统计规律；"不正确"来自语料/模型/训练都不足以学到完整语法，再加上采样本身的随机性。** 想要更"正确"，方向是加大语料、加大模型、多训几步（后面第 19、20、21 课还会继续优化），但永远不可能在 669 字符上学出真正的英语——这也侧面说明了为什么现代 LLM 需要 TB 级数据和千亿参数。

---

## 8. 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. **改种子观察差异**：把 `demo_transformer` 里 `Rng::new(1234)` 改成别的数字（如 42），重新 `cargo run --release -- demo`。loss 曲线和生成文本都会变——思考：为什么损失曲线也会变？（提示：采样 batch 的随机起点变了）
2. **改 warmup**：把 `train_transformer` 的 `warmup_steps` 从 50 改成 5 和 500，分别跑一次，对比前 100 步的 loss。体会"warmup 太短容易起飞、太长浪费步数"。
3. **改生成参数**：把 `generate` 的 `temperature` 改成 0.2 和 1.5 各跑一次。观察文本变得更"死板/重复"还是更"发散/乱"。
4. **数 token**：验证第 5.3 节末尾的 cosine 验算——打印 `scheduler.lr()` 在 step 100、600 的计算过程，对照日志里的 `0.002945` 和 `0.000300`。
5. **思考**：loss 从 1.30 降到 0.16，但为什么不能说"模型学会了英语"？模型"学会"的到底是什么？

---

## 9. 本课总结

- `demo_transformer` 五步走：分词 → 建模型 → 造数据 → `train_transformer` 训练 600 步 → `generate` 采样生成
- 数据是自监督的：x 是 32 个 token，y 是 x 右移一位，预测"下一个字符"
- 真实日志：loss `1.30 → 0.16`，前 300 步降得最快（后期在 0.16～0.24 之间摆动，因为评估 loss 只取单个 batch）；lr 从 `0.002945` 一路 cosine 衰减到 `0.000300`（warmup 段因 `eval_every=100` 没有打印点）
- 生成用 `temperature=0.8 + top-k=10 + top-p=0.9 + 重复惩罚 1.1`：先缩放、再截断、再按概率随机抽样
- 小模型输出"像样而非正确"：语料太小、模型太小、训练不足、采样随机，四者叠加

- 下一课：换掉朴素的 SGD，给优化器装上"动量 + 自适应步长 + 权重衰减"——AdamW。
