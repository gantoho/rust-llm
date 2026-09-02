# 第 16 课：训练小 GPT —— 看 loss 从 1.46 一路降到 0.16

> 代码位置：[src/main.rs](src/main.rs)（`demo_gpt`）
> 代码位置：[src/train.rs](src/train.rs)（`train_gpt` / `LRScheduler` / `clip_grad_norm`）
> 代码位置：[src/data.rs](src/data.rs)（`CORPUS` / `DataLoader`）
> 代码位置：[src/sample.rs](src/sample.rs)（`generate` / `sample_token`）

---

## 1. 本课要搞懂的问题

1. `demo_gpt` 从数据到生成文本，完整流程分哪几步？
2. 只有 669 个字符的小语料，训练日志里的 `step / lr / loss` 三列怎么读？
3. 日志里为什么看不到 warmup 段？lr 从 `0.002947` 一路衰减到 `0.000300` 是怎么来的？
4. temperature、top-k、top-p 三个参数是怎么配合采样的？
5. 为什么 loss 已经降到 0.16，模型输出的文本依然只是"像样"而不是"正确"？

---

## 2. 训练全景：demo_gpt 做了什么

`src/main.rs` 的演示 3（第 12-21 课）是本节的主角：

```rust
fn demo_gpt() {
    println!("=== 演示 3：训练小 GPT 并生成文本 ===");

    let mut rng = Rng::new(1234);
    let tokenizer = CharTokenizer::new(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    println!("  语料 {} 字符，字符词表 {} 个", CORPUS.len(), vocab_size);

    let model = GPT::new(GPTConfig::tiny(vocab_size), &mut rng);

    // 训练（第 13、17、20 课：训练循环 + AdamW + warmup/cosine 调度）
    let loader = DataLoader::new(CORPUS, &tokenizer, model.cfg.block_size, 8);
    let tcfg = config::TrainConfig {
        seed: 42,
        batch_size: 8,
        steps: 600,
        max_lr: 3e-3,
        warmup_steps: 50,
        eval_every: 100,
        ..config::TrainConfig::default()
    };
    train::train_gpt(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng);

    // 生成（无 cache）
    println!("\n  —— 生成 1（temperature=0.8, top-k=10, top-p=0.9, 无 KV cache）——");
    let out1 = generate(&model, &tokenizer, "Once upon a", 80, 0.8, 10, 0.9, false, &mut rng);
    println!("  {}", out1);

    // 生成（带 KV cache，第 18 课）
    println!("\n  —— 生成 2（temperature=0.8, top-k=10, top-p=0.9, 带 KV cache）——");
    let out2 = generate(&model, &tokenizer, "The fox", 80, 0.8, 10, 0.9, true, &mut rng);
    println!("  {}", out2);
    println!("\n  （KV cache 只改计算方式、不改生成分布，两者应高度一致）");
}
```

整个流程可以拆成 5 步：

| 步骤 | 代码 | 做了什么 |
|------|------|---------|
| 1. 分词 | `CharTokenizer::new(CORPUS)` | 扫描语料，得到 35 个字符的词表 |
| 2. 建模型 | `GPT::new(GPTConfig::tiny(vocab_size), &mut rng)` | 用 tiny 配置（n_embd=64、n_head=4、n_layer=2、block_size=32）初始化模型 |
| 3. 造数据 | `DataLoader::new(CORPUS, &tokenizer, 32, 8)` | 把 669 字符的语料切成 token 序列，按 block_size=32 切块、batch_size=8 |
| 4. 训练 | `train_gpt(&model, &tokenizer, &loader, &tcfg, None, None, ...)` | 600 步，峰值学习率 3e-3，前 50 步 warmup，每 100 步打印一次（其余参数取 `TrainConfig::default()`） |
| 5. 生成 | `generate(..., "Once upon a", 80, 0.8, 10, 0.9, false, ...)` | 给定开头，最多续写 80 个字符 |

> 注意：训练用的是字符级分词器，所以"1 个字符 = 1 个 token"，语料 669 个字符就是 669 个 token。这让后面的数字（32、80）可以直接按"字符数"理解。

---

## 3. 数据：669 字符的小语料

`src/data.rs` 里内置了一篇英文小故事（狐狸 Red 找金钥匙）：

```rust
pub const CORPUS: &str = "\
Once upon a time in a small village, there lived a curious little fox named Red. \
Every morning, Red would wake up early and explore the forest. ...";
```

训练数据是**自监督**的：输入 x 是一段 32 个 token 的序列，标签 y 是 x 右移一位——每个位置都预测"下一个字符是谁"，文本自己就是标签，不需要人工标注。

`DataLoader::sample_batch` 每次随机选 8 个起点，各截 33 个 token（前 32 个作 x，后 32 个作 y）：

```rust
pub fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
    let max_start = self.tokens.len() - self.block_size - 1;
    let mut x = Vec::with_capacity(self.batch_size * self.block_size);
    let mut y = Vec::with_capacity(self.batch_size * self.block_size);
    for _ in 0..self.batch_size {
        let start = rng.choice(max_start);
        for j in 0..self.block_size {
            x.push(self.tokens[start + j]);
            y.push(self.tokens[start + j + 1]);
        }
    }
    (x, y)
}
```

关键点：

- **随机采样而非顺序扫描**：每次 `sample_batch` 都在语料里随机挑起点。语料只有 669 token，但 600 步 × 8 个 batch 会反复"看到"语料的不同片段（有些片段会被重复看，有的可能一次都没被抽到）——小语料训练天然就是"背课文"。
- 返回的 x、y 都是 `[B*T] = [8×32] = [256]` 的展平数组，正好满足 `GPT::forward(idx, b=8, t=32, None)` 的输入要求（训练时 `kv_cache` 传 `None`）。

---

## 4. 超参数一览

`train_gpt` 的调用参数与 `GPTConfig::tiny` 汇总：

| 超参数 | 值 | 含义 |
|--------|----|------|
| `steps` | 600 | 总训练步数 |
| `batch_size` | 8 | 每步采样 8 条序列（每条 32 token） |
| `block_size` | 32 | 最大上下文长度，来自 `GPTConfig::tiny` |
| `max_lr` | 3e-3 | 学习率峰值 |
| `warmup_steps` | 50 | 前 50 步学习率从 0 线性爬升到峰值 |
| `min_lr` | max_lr × 0.1 = 3e-4 | cosine 衰减的终点（`LRScheduler::new` 里算的） |
| `weight_decay` | 0.01 | AdamW 的权重衰减（第 17 课） |
| `max_norm`（梯度裁剪） | 1.0 | 梯度范数上限（`clip_grad_norm`） |
| `eval_every` | 100 | 每 100 步打印一次日志 |

模型参数量：`train_gpt` 开头会打印一行"开始训练"（真实数字就在其中）：

```
开始训练：char（vocab=35）模型参数 102336 | 语料 669 tokens（训练 669 / 验证 0）| batch=8 block=32
```

按第 12 课的方法验证一下：词表 V=35（不是 100）时，`tok_emb = 35×64 = 2240`，每层 Block ≈ 49984，两层 ≈ 99968，`ln_f = 128`，输出头是权重绑定（复用 `tok_emb` 转置，无独立 lm_head 参数），总计 **2240 + 99968 + 128 = 102336** ✓。约 10 万参数，CPU 上几秒就能跑完整个 demo。

---

## 5. 真实训练日志解读

运行 `cargo run --release`，演示 3 会打印（这是**真实运行输出**，不是编的）：

```
=== 演示 3：训练小 GPT 并生成文本 ===
  语料 669 字符，字符词表 35 个
开始训练：char（vocab=35）模型参数 102336 | 语料 669 tokens（训练 669 / 验证 0）| batch=8 block=32
step   100 | lr 0.002947 | train_loss 1.4597
step   200 | lr 0.002540 | train_loss 0.5498
step   300 | lr 0.001850 | train_loss 0.3338
step   400 | lr 0.001096 | train_loss 0.2312
step   500 | lr 0.000518 | train_loss 0.1557
step   600 | lr 0.000300 | train_loss 0.1624
```

### 5.1 三列日志分别是什么

| 列 | 含义 | 从哪来 |
|----|------|--------|
| `step` | 训练步数（从 1 开始数，日志显示 100、200、…、600） | `train_gpt` 打印的是 `step + 1` |
| `lr` | 本步实际用于更新的学习率 | `cur_lr`（先取 `scheduler.lr()`，再 `scheduler.step()`） |
| `train_loss` | 本步 batch 的平均交叉熵 | `cross_entropy_loss(&logits, &y)` |

`train_gpt` 里每步做 6 件事，日志打印在最后：

```rust
for step in 0..steps {
    let (x, y) = loader.sample_batch(rng);          // 1. 采样 batch
    let logits = model.forward(&x, b, t, None);     // 2. 前向
    let loss = cross_entropy_loss(&logits, &y);     //    算损失
    loss.backward();                                // 3. 反向
    clip_grad_norm(&params, 1.0);                   // 4. 梯度裁剪
    let cur_lr = scheduler.lr();                    // 5. 取当前步 lr 喂给优化器
    opt.lr = cur_lr;
    opt.step();
    opt.zero_grad();                                // 6. 清零梯度
    scheduler.step();                               //    步数 +1（为下一步准备 lr）
    let last = step + 1 == steps;
    if (step + 1) % cfg.eval_every == 0 || last {   // 每 eval_every 步（或最后一步）打印
        // lr 打印的就是本步实际用的 cur_lr，没有错位；step 打印 step + 1
        println!("step {:>5} | lr {:.6} | train_loss {:.4}", step + 1, cur_lr, loss.item());
    }
}
```

### 5.2 loss：1.46 → 0.16 说明了什么

- **第一个打印点 1.46**：日志只在 `step 100、200、…` 打印（`eval_every = 100`）。随机初始化时模型对 35 个字符基本"一视同仁"，理论下界是均匀分布的交叉熵 `ln(35) ≈ 3.56`；训练 100 步后降到 1.46，说明已经开始学习。
- **先快后慢**：step 100→300 loss 从 1.46 掉到 0.33（降了 77%），step 300→600 只从 0.33 掉到 0.16。这是训练曲线的典型形态——早期梯度大、方向明确，后期接近收敛、只能精雕细琢。
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

| scheduler.step | 计算 | lr |
|----------------|------|----|
| 0（真正用于第 1 步更新） | 0.003 × 1 / 50 | 0.00006 |
| 1 | 0.003 × 2 / 50 | 0.00012 |
| 25 | 0.003 × 26 / 50 | 0.00156 |
| 50（warmup 结束） | 0.003 × 51 / 50 | ≈ 0.00306（峰值） |

> 注意：demo 的 `eval_every = 100`，warmup 段（step 0-49）**没有打印点**，所以真实日志里看不到 0.00006 起步的爬升。
> 把 `eval_every` 改成 10，就能看到 step 10/20/30/40 的 lr = `0.0006 → 0.0012 → 0.0018 → 0.0024`
> （每步增加 `0.003/50 = 0.00006`，10 步就是 0.0006）。
>
> 为什么要 warmup？训练刚开始时参数是随机值，梯度方向噪声大、量级不可控。如果一上来就用 0.003 的大步长，很容易把参数"推飞"（loss 直接变成 NaN）。先用小步长稳住方向，再逐渐加力，是现代 LLM 训练的标准做法。

### 5.4 cosine 衰减：从峰值平滑降回 min_lr

第 50 步之后走 cosine 曲线，从 `max_lr = 0.003` 平滑降到 `min_lr = 0.003 × 0.1 = 0.0003`：

```
lr = min_lr + (max_lr - min_lr) × 0.5 × (1 + cos(π × progress))
progress = (step - 50) / (600 - 50)，超过 1 就截断到 1
```

验证日志里的两个数字：

- `step 100`：scheduler 计数 = 99，`progress = (99-50)/550 ≈ 0.089`，`cosine ≈ 0.9805`，`lr = 0.0003 + 0.0027×0.9805 ≈ 0.002947` ✓
- `step 600`：scheduler 计数 = 599，`progress = (599-50)/550 ≈ 0.998`，`cosine ≈ 0`，`lr ≈ min_lr = 0.000300` ✓

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

训练 600 步后调用 `generate`（`src/sample.rs`），参数 `(prompt, max_new=80, temperature=0.8, top_k=10, top_p=0.9)`。

`sample_token` 内部的 6 步采样管线：

| 步骤 | 代码 | 作用 |
|------|------|------|
| 1. 温度缩放 | `l / temperature.max(1e-5)` | 除以 0.8：logits 变大 → softmax 更"锐利"，更敢选高概率 token |
| 2. 排序 | `items.sort_by(...)` | 按分数从高到低排 |
| 3. top-k | `items.truncate(top_k)` | 只留前 10 个 |
| 4. softmax | `(*v - max).exp()` 再归一化 | 把截断后的分数变成概率 |
| 5. top-p | 累积概率到 0.9 截断 | 进一步砍掉长尾低概率 token，再归一化 |
| 6. 抽样 | `rng.next_f32()` 按概率累积选取 | 有随机性地选一个 token |

真实生成结果（`cargo run --release` 原样输出）：

```
  —— 生成 1（temperature=0.8, top-k=10, top-p=0.9, 无 KV cache）——
  Once upon a time in a small village, there lived a curious little fox named Red. Every morn

  —— 生成 2（temperature=0.8, top-k=10, top-p=0.9, 带 KV cache）——
  The fox named all is s fend the g
```

（生成 2 用的是另一个 prompt "The fox"，且因缓存模式上下文达到 block_size=32 提前停止，第 18 课会专门讲；生成 1 在无缓存模式下把 80 个新字符完整生成完了。）

读这段输出：模型学会了故事的结构——"Once upon a time..." 开头、"in a small village, there lived a curious little fox named Red" 几乎完整复现语料原文、主谓宾、句号逗号。**字面上"像样"，但仔细读全是毛病**：生成 2 的 "named all is s fend the g" 语法不通、句子戛然而止（block_size 截断）。这就是下一节要回答的问题。

---

## 7. 为什么小模型输出只是"像样"而非"正确"

四个层面叠加，缺一不可：

| 原因 | 说明 |
|------|------|
| **语料太小** | 只有 669 字符、单一故事。模型只能"背"这篇课文里的统计规律，从未见过通用英语，谈不上泛化 |
| **模型太小** | 10 万参数 vs 真实 LLM 的数十亿～万亿参数。容量只够记住局部 n-gram 统计（"Red" 后常跟动词、名词前常有 the），装不下真正的语法规则 |
| **训练不足** | 600 步后 loss 仍为 0.16（正确概率约 85%），还没收敛到 0。模型对很多位置仍"没把握" |
| **采样带随机性** | temperature=0.8 + top-k/top-p 是有意引入随机性。即使模型 100% 会预测 "world"，采样也可能选到 "wold"——这是"创造性"的代价 |

用一句话总结：**"像样"来自学到了语料的高频统计规律；"不正确"来自语料/模型/训练都不足以学到完整语法，再加上采样本身的随机性。** 想要更"正确"，方向是加大语料、加大模型、多训几步（后面第 19、20、21 课还会继续优化），但永远不可能在 669 字符上学出真正的英语——这也侧面说明了为什么现代 LLM 需要 TB 级数据和千亿参数。

---

## 8. 动手练习

1. **改种子观察差异**：把 `demo_gpt` 里 `Rng::new(1234)` 改成别的数字（如 42），重新 `cargo run --release`。loss 曲线和生成文本都会变——思考：为什么损失曲线也会变？（提示：采样 batch 的随机起点变了）
2. **改 warmup**：把 `train_gpt` 的 `warmup_steps` 从 50 改成 5 和 500，分别跑一次，对比前 100 步的 loss。体会"warmup 太短容易起飞、太长浪费步数"。
3. **改生成参数**：把 `generate` 的 `temperature` 改成 0.2 和 1.5 各跑一次。观察文本变得更"死板/重复"还是更"发散/乱"。
4. **数 token**：验证第 5.4 节——打印 `scheduler.lr()` 在 step 100、600 的计算过程，对照日志里的 `0.002947` 和 `0.000300`。
5. **思考**：loss 从 1.46 降到 0.16，但为什么不能说"模型学会了英语"？模型"学会"的到底是什么？

---

## 9. 本课总结

- `demo_gpt` 五步走：分词 → 建模型 → 造数据 → `train_gpt` 训练 600 步 → `generate` 采样生成
- 数据是自监督的：x 是 32 个 token，y 是 x 右移一位，预测"下一个字符"
- 真实日志：loss `1.46 → 0.16`，前 300 步降得最快；lr 从 `0.002947` 一路 cosine 衰减到 `0.000300`（warmup 段因 `eval_every=100` 没有打印点）
- 生成用 `temperature=0.8 + top-k=10 + top-p=0.9`：先缩放、再截断、再按概率随机抽样
- 小模型输出"像样而非正确"：语料太小、模型太小、训练不足、采样随机，四者叠加

- 下一课：换掉朴素的 SGD，给优化器装上"动量 + 自适应步长 + 权重衰减"——AdamW。
