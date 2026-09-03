# 第 20 课：RoPE 旋转位置编码 —— 把"相对位置"揉进注意力

> 代码位置：[src/rope.rs](src/rope.rs)（`rotary_pair` 生产入口 + `rotary` 测试用 + `test_rotary` / `test_rotary_grad_exact` 测试）
> 配套代码：[src/attention.rs](src/attention.rs)（RoPE 接入 `MultiHeadAttention`）
> 配套文档：[docs/11-位置编码与归一化.md](docs/11-位置编码与归一化.md)

---

## 1. 本课要搞懂的问题

1. 第 11 课的正弦位置编码有什么缺陷？为什么现代 LLM（LLaMA、Qwen、Gemma……）几乎清一色用 RoPE？
2. "旋转"怎么把位置信息编码进向量？公式 `x'_{2i} = x_{2i}·cosθ - x_{2i+1}·sinθ` 是怎么来的？
3. 为什么说旋转是**正交变换**？范数不变这个性质重要在哪？
4. 为什么 RoPE 的点积只与"位置差"有关？相比正弦编码强在哪？
5. 为什么 RoPE 和 KV cache 是"天生一对"？推理时为什么只需要旋转新 token？
6. `src/rope.rs` 里 `rotary_pair` 的前向、反向具体怎么实现？`test_rotary` 测了什么？

---

## 2. 先回顾：正弦位置编码的两个局限

第 11 课我们实现了正弦位置编码，把位置向量**加到** token embedding 上：

```
x = token_embedding + pos_emb
```

它解决了"注意力看不见位置"的问题，但有两个不足：

### 2.1 编码的是"绝对位置"，模型得自己推断相对关系

正弦编码给每个绝对位置 `pos` 一个固定向量，加到输入上。模型面对的其实是"内容 + 绝对位置"的混合表示，
**"A 在第 3 位、B 在第 7 位" 这个信息是间接的**——模型需要自己学会"位置 7 减去位置 3 = 距离 4"。

原论文证明了 `PE(pos+k)` 可以表示为 `PE(pos)` 的线性组合（三角恒等式），即"理论上模型能学到相对距离"，
但**这只是给了模型一个机会，不是保证**——要靠训练把这种规律学出来。

> 直觉：正弦编码像给每个座位发一张写着"座位号"的号牌，模型得自己学会"6 号和 2 号的号牌之差 = 隔了 4 排"。
> RoPE 的做法是直接把"隔了几排"编码进打分公式里，模型不用学这一步。

### 2.2 外推（extrapolation）差

我们的模型 `block_size = 32`（`GPTConfig::tiny`），训练时位置只见过 `0..32`。如果推理时生成更长的序列：

- 位置 32、33…… 的 pos_emb 向量虽然能算出来（正弦函数对任意 pos 都有定义），
- 但模型**从没见过这种输入分布**，注意力分数可能畸变，输出质量断崖式下跌。

| 对比项 | 正弦编码（第 11 课） | RoPE（本课） |
|--------|---------------------|--------------|
| 注入方式 | 位置向量**加**到输入 embedding | 对 Q/K 向量做**旋转** |
| 编码的信息 | 绝对位置（相对关系靠模型自己学） | 相对位置（点积直接只依赖位置差） |
| 作用位置 | 模型输入（所有层共享一份） | 注意力内部（每层的 Q/K 各自旋转） |
| 是否改变范数 | 加一个固定向量，范数会变 | 旋转保范数，输入表示"只转向、不变长" |
| KV cache 配合 | 也兼容，但位置信息"粘"在输入上 | 只需旋转新 token 的 Q/K，天然契合 |

> 补充：RoPE 也不是"万能外推药"。角度随位置线性增长，训练长度之外同样会失效。
> 业界用 **NTK-aware scaling、YaRN** 等技巧做长度外推，那是后话，本课先把 RoPE 本身讲透。

---

## 3. RoPE 原理：把 d 维拆成 d/2 对来旋转

RoPE 出自论文 *RoFormer: Enhanced Transformer with Rotary Position Embedding*（Su et al., 2021）。

### 3.1 二维旋转回顾

在二维平面上，把向量 `(a, b)` 绕原点逆时针旋转角度 `θ`：

```
a' = a·cosθ - b·sinθ
b' = a·sinθ + b·cosθ
```

写成矩阵就是旋转矩阵 `R(θ) = [[cosθ, -sinθ], [sinθ, cosθ]]`。

### 3.2 把 d 维向量看成 d/2 个二维向量

一个 `d` 维向量（d 必须为偶数），按**相邻两两配对**：

```
(x₀, x₁), (x₂, x₃), (x₄, x₅), ..., (x_{d-2}, x_{d-1})
```

每一对看作平面上的一个点，第 `i` 对 `(x_{2i}, x_{2i+1})` 绕原点旋转角度：

```
θ_i = pos / 10000^(2i/d)      i = 0, 1, ..., d/2 - 1
```

旋转后：

```
x'_{2i}   = x_{2i}·cosθ_i - x_{2i+1}·sinθ_i
x'_{2i+1} = x_{2i}·sinθ_i + x_{2i+1}·cosθ_i
```

> 注意频率方案 `10000^(2i/d)` 和正弦编码一模一样（第 11 课）：`i` 越小频率越高（角度变化快，管近处的精细位置），
> `i` 越大频率越低（角度变化慢，管远处的粗略位置）。这样"近距离靠高维、远距离靠低维"的分工被保留了下来。

### 3.3 手算一个例子

设 `d = 4`、位置 `pos = 1`，则有两对：

| 对 | i | θ_i = 1 / 10000^(2i/4) | cos θ_i | sin θ_i |
|----|---|------------------------|---------|---------|
| (x₀, x₁) | 0 | 1 / 1 = 1.0 rad | 0.5403 | 0.8415 |
| (x₂, x₃) | 1 | 1 / 10000^0.5 = 1/100 = 0.01 rad | ≈ 0.99995 | ≈ 0.01 |

于是：

```
x'₀ = x₀·0.5403 - x₁·0.8415
x'₁ = x₀·0.8415 + x₁·0.5403
x'₂ ≈ x₂·0.99995 - x₃·0.01
x'₃ ≈ x₂·0.01 + x₃·0.99995
```

第一对被旋转了整整 1 弧度（幅度很大，专门区分相邻位置），第二对几乎没动（幅度极小，负责长距离）。
**同一个向量，位置不同旋转角度就不同**——位置信息就这样被"揉"进了向量本身。

---

## 4. 旋转是正交变换：范数不变

旋转矩阵满足 `R(θ)ᵀ·R(θ) = I`（转置乘自己等于单位阵），且 `|det R| = 1`，因此它是**正交矩阵**：

```
‖R(θ)·x‖² = (R(θ)x)ᵀ(R(θ)x) = xᵀR(θ)ᵀR(θ)x = xᵀx = ‖x‖²
```

也就是说**旋转只改变向量的方向，不改变长度（范数）**。这带来两个直接好处：

| 好处 | 解释 |
|------|------|
| 数值稳定 | 旋转前后的数值范围一模一样，不会像"加一个固定向量"那样撑大或压扁激活值 |
| 不破坏归一化 | 向量进注意力之前刚过 LayerNorm，旋转不改范数，归一化统计量不会被破坏 |

> 对比一下：正弦编码是"加法"，加完 `‖x + pos_emb‖` 和 `‖x‖` 一般不同，多多少少会扰动数值分布；
> RoPE 是"乘法"（乘以正交矩阵），**只转不拉**，范数严格不变。

更妙的是：正交矩阵的逆就是它的转置，而转置恰好等于**负角度旋转**：

```
R(θ)ᵀ = R(-θ)
```

这个性质在反向传播里极其好用——梯度回传时把角度取反再旋转一次就行（详见第 8 节）。

---

## 5. 核心性质：相对位置（点积只与位置差有关）

### 5.1 先看 d = 2 的情形

设查询在位置 `m`：`q_m = R(mθ)·q`；键在位置 `n`：`k_n = R(nθ)·k`。注意力分数是它们的点积：

```
q_m · k_n = (R(mθ)q)ᵀ · (R(nθ)k)
          = qᵀ · R(mθ)ᵀ · R(nθ) · k
          = qᵀ · R(-mθ) · R(nθ) · k      （Rᵀ = R⁻¹ = 负角度）
          = qᵀ · R((n - m)θ) · k          （旋转矩阵乘法 = 角度相加）
```

**结果只依赖 `n - m`，与 m、n 各自的值无关！**

### 5.2 推广到 d 维

d 维的旋转矩阵是**块对角矩阵**：

```
R(θ) = diag( R(θ₀), R(θ₁), ..., R(θ_{d/2-1}) )
```

块对角矩阵相乘等于各块分别相乘，所以 d 维下同样成立：

```
q_m · k_n = Σ_i  q_{2i}ᵀ · R((n-m)·θ_i) · k_{2i}    只依赖 n - m
```

### 5.3 为什么这比正弦编码强

| | 正弦编码 | RoPE |
|--|---------|------|
| 打分时依赖 | 绝对位置（模型要从 `pos_emb[m] + pos_emb[n]` 里反推相对关系） | 直接就是相对位置 `n - m` |
| 模型要学吗 | 要额外学"位置减法"这一课 | 数学上已经编死，无需学 |
| 训练样本外 | 外推差 | 同样有外推问题，但相对性质使配合插值技巧更容易 |

> 一句话总结：**正弦编码把位置塞进"表示"里，RoPE 把位置塞进"打分"里。**
> 而注意力唯一关心位置的地方就是打分（q·k），所以 RoPE 做到了"精确投放"。

---

## 6. 与 KV cache 天然兼容：只旋转新 token

回顾第 18 课的 KV cache 推理流程（`src/sample.rs` 的 `generate` + `src/attention.rs` 的 `KVCache`）：

- 第一次：把整个 prompt 喂给模型，算出所有位置的 K/V 存入缓存；
- 之后每步：**只前向最新 1 个 token**，历史的 K/V 直接从缓存取。

RoPE 和这个流程是无缝衔接的：

```
新 token 的绝对位置 = 缓存长度 base + 它在当前窗口里的下标 j
```

`base`（已缓存的位置数）由 `GPT::forward` 算出来传给每层（第 18 课的设计），`MultiHeadAttention::forward` 里 `positions` 就是这么构造的：

```rust
// src/model.rs（GPT::forward）：base = 缓存长度
let base = kv_cache
    .as_ref()
    .map(|c| c.first().map(|k| k.seq_len()).unwrap_or(0))
    .unwrap_or(0);

// src/attention.rs（MultiHeadAttention::forward）：batch 内每个样本位置相同，重复 b 次
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
```

于是接入 RoPE 时：

| 谁 | 怎么处理 |
|----|---------|
| 新 token 的 Q | 按当前位置 `base + j` 旋转 |
| 新 token 的 K | 按当前位置 `base + j` 旋转，再 append 进缓存 |
| 缓存里的历史 K | 早就旋转好了，**原样复用，绝不再动** |

因为旋转只依赖"位置号"，而每个 token 的位置号在生成时是唯一确定的，所以**历史上每个 K 只旋转一次、终身有效**。
相比"每次生成都要重新取一段 pos_emb 加进输入"的加法式编码，RoPE 是"随算随旋"，不需要任何历史重算。

> 另一个视角：RoPE 在注意力内部旋转 Q/K，位置信息不会污染 token 的"内容表示"（残差流上的 x 始终是纯语义），
> 这也让每一层的 Q/K 都能用"本层自己的位置感"去打分。

---

## 7. 前向实现讲解：cos/sin 表 + 查表旋转

`src/rope.rs` 把实现拆成三层：先一次性**预计算** cos/sin 表，再用表**查表旋转**，Q/K 共用同一张表：

```rust
/// 预计算每个 (位置, 对偶下标) 的 cos/sin 表，长度 rows × (D/2)。
/// 同一批 positions 的三角只算一次：前向、反向、Q/K 复用。
fn build_cos_sin_tab(positions: &[usize], d: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = positions.len();
    let mut c_tab = vec![0.0f32; rows * (d / 2)];
    let mut s_tab = vec![0.0f32; rows * (d / 2)];
    for r in 0..rows {
        let pos = positions[r] as f32;
        for i in 0..d / 2 {
            let theta = pos / 10000f32.powf((2 * i) as f32 / d as f32);
            c_tab[r * (d / 2) + i] = theta.cos();
            s_tab[r * (d / 2) + i] = theta.sin();
        }
    }
    (c_tab, s_tab)
}

/// 用现成的 cos/sin 表旋转一个张量（[rows, D]），每对元素按公式旋转。
fn rotate_with_tab(x: &Tensor, c_tab: &[f32], s_tab: &[f32]) -> Tensor {
    let (rows, d) = (x.shape[0], x.shape[1]);
    let sd = x.data.borrow();
    let mut out_data = vec![0.0f32; rows * d];
    for r in 0..rows {
        for i in 0..d / 2 {
            let (c, s) = (c_tab[r * (d / 2) + i], s_tab[r * (d / 2) + i]);
            let (a, b) = (sd[r * d + 2 * i], sd[r * d + 2 * i + 1]);
            out_data[r * d + 2 * i] = a * c - b * s;
            out_data[r * d + 2 * i + 1] = a * s + b * c;
        }
    }
    drop(sd);
    // ...（反向闭包查同一张表、按 R(θ)ᵀ 回传，见第 8 节）
    result
}
```

对外只暴露两个入口——`rotary`（旋转单个张量，仅供测试）和 `rotary_pair`（一次建表同时旋转 Q/K）：

```rust
pub fn rotary_pair(&self, other: &Tensor, positions: &[usize]) -> (Tensor, Tensor) {
    let d = self.shape[1];
    let (c_tab, s_tab) = build_cos_sin_tab(positions, d);
    (
        rotate_with_tab(self, &c_tab, &s_tab),
        rotate_with_tab(other, &c_tab, &s_tab),
    )
}
```

> 为什么要"先建表再查表"？注意力里 Q 和 K 的 `positions` 完全相同，三角函数的 `cos/sin` 只要算一遍；
> 前向算一遍、反向闭包再查一遍——相比"每次旋转现场重算三角"，批量训练能省下可观的重复计算。

逐行对照公式：

| 代码 | 对应公式 | 说明 |
|------|---------|------|
| `assert_eq!(self.rank(), 2, ...)` | — | 输入必须是 2 维 `[rows, D]`：每行是一个待旋转的向量 |
| `assert_eq!(self.shape[0], positions.len(), ...)` | — | `positions[r]` 就是第 `r` 行的位置，数量必须一一对应 |
| `assert_eq!(d % 2, 0, ...)` | — | 最后一维必须是偶数，才能两两配对 |
| `let theta = pos / 10000f32.powf((2 * i) as f32 / d as f32);` | `θ_i = pos / 10000^(2i/d)` | 第 `i` 对的旋转角度，注意 `i` 的范围是 `0..d/2` |
| `let (c, s) = (theta.cos(), theta.sin());` | `cosθ_i, sinθ_i` | 一次算出，避免重复调用三角函数 |
| `let (a, b) = (sd[r*d + 2*i], sd[r*d + 2*i + 1]);` | `(x_{2i}, x_{2i+1})` | 取出第 r 行第 i 对的两个元素 |
| `out_data[r*d + 2*i] = a * c - b * s;` | `x'_{2i} = x_{2i}·cosθ - x_{2i+1}·sinθ` | 旋转后的第一个分量 |
| `out_data[r*d + 2*i + 1] = a * s + b * c;` | `x'_{2i+1} = x_{2i}·sinθ + x_{2i+1}·cosθ` | 旋转后的第二个分量 |

几个值得注意的设计点：

1. **三重循环的次序**：建表时外层按行 `r`、内层按对 `i`，`cos/sin` 每个 `(r, i)` 只算一次；旋转时同样两层循环，但只做查表乘加，不再碰任何三角函数。
2. **`drop(sd)`**：读完输入数据后立刻释放借用，之后才创建结果张量和反向闭包——避免闭包捕获时和 `self.data` 的借用纠缠。
3. **`requires_grad` 分支**：如果输入不需要梯度（比如纯推理），就直接返回普通结果，不建 `parents`/`backward`，省下计算图的维护开销。这和第 18 课 KV cache 推理时"纯数据拼接、无梯度"的思路一致。
4. **反向闭包查同一张表**：反向闭包为了"自包含"（只捕获 `ct`/`st` 两份表拷贝 + `rg`/`sg` 两个 Rc），直接查表取值，**不再重算 `theta/cos/sin`**。代价是闭包多持有两张表，好处是前向建的三角表被完整复用。

---

## 8. 反向实现讲解：正交矩阵的梯度

### 8.1 理论：梯度用"转置"回传

设 `y = R(θ)·x`（旋转一对）。根据链式法则：

```
∂L/∂x = R(θ)ᵀ · ∂L/∂y
```

因为旋转矩阵正交，`R(θ)ᵀ = R(-θ)`，所以**梯度回传 = 把梯度按负角度再旋转一次**。这就是方法注释里写的：

> "每个 pair 的旋转矩阵正交，梯度用其转置（即负角度旋转）回传。"

展开来写，若记 `(ga, gb) = (∂L/∂y_{2i}, ∂L/∂y_{2i+1})`，则正确回传是：

```
∂L/∂x_{2i}   = ga·cosθ + gb·sinθ
∂L/∂x_{2i+1} = -ga·sinθ + gb·cosθ
```

> 验证（也可以硬算）：`y₀ = a·c - b·s`、`y₁ = a·s + b·c`，
> `∂L/∂a = ga·(∂y₀/∂a) + gb·(∂y₁/∂a) = ga·c + gb·s` ✓
> `∂L/∂b = ga·(∂y₀/∂b) + gb·(∂y₁/∂b) = -ga·s + gb·c` ✓

### 8.2 代码实际怎么写的（已按正确形式实现）

`src/rope.rs` 里 `rotate_with_tab` 反向闭包的循环体是（`c`/`s` 直接查 `ct`/`st` 表）：

```rust
let (ga, gb) = (g[r * d + 2 * i], g[r * d + 2 * i + 1]);
// 反向 = 前向旋转矩阵的转置 R(θ)ᵀ：grad = (ga·c + gb·s, -ga·s + gb·c)
sgm[r * d + 2 * i] += ga * c + gb * s;
sgm[r * d + 2 * i + 1] += -ga * s + gb * c;
```

与 8.1 的结论逐行对应：第一行 `ga·c + gb·s`、第二行 `-ga·s + gb·c`，正是 `R(θ)ᵀ·g`。

> 💡 **陷阱提示**：反向回传必须用**转置**（即负角度旋转）。写代码时容易手滑把符号写成正角度旋转
> `(ga·c - gb·s, ga·s + gb·c)`——方向错了但梯度**范数不变**（正交变换保范数），
> 只校验范数的测试发现不了。所以测试要"既测范数、又测方向"（见第 9 节 `test_rotary_grad_exact`）。
> 本仓库初版实现确实写反过，已修复并补了逐元素断言。

对训练的影响：本课已经把 RoPE 接进了 `MultiHeadAttention`（`src/attention.rs`），训练和 KV cache 推理都用它。
`GPT` 已不再有 `pos_emb` 字段，位置信息完全由注意力内部旋转 Q/K 提供（见第 10 节）。

---

## 9. `test_rotary` 测试讲解

测试在 `src/rope.rs` 的 `mod tests` 里，共三个断言，分别验证三件事：

```rust
#[test]
fn test_rotary() {
    // 1. 旋转是正交变换：范数不变
    let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
    let r = x.rotary(&[3]);
    let orig_norm: f32 = x.data().iter().map(|v| v * v).sum();
    let rot_norm: f32 = r.data().iter().map(|v| v * v).sum();
    assert!((orig_norm - rot_norm).abs() < 1e-3, "范数应守恒：{} vs {}", orig_norm, rot_norm);

    // 2. pos=0 时所有角度为 0，等于恒等变换
    let x2 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let r2 = x2.rotary(&[0]);
    assert!((r2.data()[0] - 1.0).abs() < 1e-5);
    assert!((r2.data()[3] - 4.0).abs() < 1e-5);

    // 3. 梯度：sum 的梯度是单位向量，经正交矩阵回传后范数不变（= 元素数）
    let x3 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
    let loss = x3.rotary(&[2]).sum();
    loss.backward();
    let g: Vec<f32> = x3.grad();
    assert!((g.iter().map(|v| v * v).sum::<f32>() - 6.0).abs() < 1e-3, "梯度范数应为 6");
}
```

| 测试项 | 输入 | 断言 | 原理 |
|--------|------|------|------|
| ① 范数不变 | `[1,2,3,4,5,6]`（1×6），位置 `[3]` | 旋转前后范数平方相等（容差 1e-3） | 旋转是正交变换，`‖Rx‖ = ‖x‖` |
| ② pos=0 恒等 | `[1,2,3,4]`（1×4），位置 `[0]` | 输出第 0、3 个元素不变 | `θ_i = 0/… = 0`，`cos0=1, sin0=0`，`R(0)=I` |
| ③ 梯度回传 | `[1,2,3,4,5,6]`，位置 `[2]`，`loss = sum` | 梯度范数平方 = 6（= 元素数） | `sum` 的梯度是 `[1,1,...,1]`（范数平方 6），经正交矩阵回传后范数不变 |

测试 ③ 的思路很巧妙：`∂sum/∂y` 是"全是 1"的单位向量，只要反向实现用的是正交矩阵（无论正角度还是负角度），
回传后梯度范数平方必然还是元素数 6。它验证了"**正交性**"，但正如 8.2 所说，验证不了"**方向**"。

仓库里还补了一个 `test_rotary_grad_exact`，专测**方向**：`d=2`、`pos=1` 时 `θ₀=1` rad，
对 `x=[1,2]`、`loss=sum`，期望梯度是 `R(θ)ᵀ·[1,1] = (cos1+sin1, -sin1+cos1) = (1.3818, -0.3012)`，
逐元素断言。有了它，反向符号写反会立刻红：

```rust
#[test]
fn test_rotary_grad_exact() {
    let x = Tensor::param(vec![1.0, 2.0], vec![1, 2]);
    let loss = x.rotary(&[1]).sum();
    loss.backward();
    let (c, s) = (1f32.cos(), 1f32.sin());
    let (ga, gb) = (c + s, -s + c);
    assert!((x.grad()[0] - ga).abs() < 1e-5, "grad[0] = {}", x.grad()[0]);
    assert!((x.grad()[1] - gb).abs() < 1e-5, "grad[1] = {}", x.grad()[1]);
}
```

运行测试：

```bash
cargo test test_rotary    # 跑全部 rotary 测试
```

---

## 10. 如何接入注意力：对 Q/K 应用

### 10.1 为什么只旋转 Q 和 K，不旋转 V

注意力分数是 `q·k`（点积），位置信息只有在这里才"有用武之地"。而输出 `out = softmax(scores)·V` 是**内容**的加权求和，
V 是"被加权的内容"，不需要带位置信息。所以标准做法是：**只旋转 Q 和 K，V 保持原样**。

### 10.2 接入点：`MultiHeadAttention::forward`

在 `src/attention.rs` 的 `MultiHeadAttention::forward` 里，Q/K/V 投影出来后是这样的：

```rust
let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
let k = self.c_k.forward(x).reshape(vec![b, t, d]);
let v = self.c_v.forward(x).reshape(vec![b, t, d]);
```

RoPE 有两个可选的接入时机：

| 时机 | 形状 | 说明 |
|------|------|------|
| 拆头之前 | `q/k` 是 `[B, T, D]` | 把整个 batch 展平成 `[B*T, D]` 一次旋转，代码最省事 |
| 拆头之后 | 每个头是 `[T, head_dim]` | 逐头旋转，更贴近"每头各转各的"的原始论文写法 |

我们的 `rotary_pair` 接口（输入 `[rows, D]` + 每行的 `positions`）两种都支持，只要最后一维是偶数即可——
`GPTConfig::tiny` 里 `n_embd=64`、`head_dim=16`，都满足。

### 10.3 实际接入方式（本仓库已接入）

`src/attention.rs` 里，Q/K 投影之后、KV cache 拼接**之前**旋转：

```rust
// 位置：Q/K 投影之后、KV cache append 之前
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
// Q/K 用同一批 positions，一次建表同时旋转（三角函数只算一遍）
let (q, k) = q
    .reshape(vec![b * t, d])
    .rotary_pair(&k.reshape(vec![b * t, d]), &positions);
let (q, k) = (
    q.reshape(vec![b, t, d]),
    k.reshape(vec![b, t, d]),
);
// 之后照旧：cache.append(&k, &v)（缓存里存的就是旋转后的 K）
```

要点：

1. **`positions` 长度必须等于行数**（`rotary_pair` 里 `assert_eq!(self.shape[0], positions.len())`），这里行数是 `b*t`，所以 batch 内每个样本的同一列位置相同、要重复 `b` 次。
2. **旋转发生在 append 之前**：缓存里存的是"已旋转的 K"，历史 K 直接复用，符合第 6 节的兼容性约定。
3. **Q 没有缓存**（推理时 Q 永远只有新 token 一个），所以 Q 总是用当前位置旋转，天然正确。
4. 由于 K/V 已旋转并缓存，**推理模式（KV cache）和训练模式行为一致**，不会像正弦编码那样需要区分两套位置逻辑。这一点有专门的测试守着（`src/model.rs` 的 `test_kv_cache_matches_full_forward`）。
5. `rotary_pair` 的反向按 `R(θ)ᵀ` 实现并通过 `test_rotary_grad_exact` 逐元素验证。

> 替换还是叠加？本仓库选了**替换**：`GPT` 结构体里没有 `pos_emb` 字段（第 11 课的正弦编码在接入 RoPE 后已删除），
> 位置信息完全由注意力内部旋转 Q/K 提供。叠加方案（保留 `pos_emb` 再加 RoPE）一般没必要，且会稀释 RoPE 的相对位置特性。

---

## 11. 动手练习

1. **手算一对旋转**：设 `d = 2`、`pos = 1`，对向量 `[1.0, 0.0]` 手算 `rotary` 的输出
   （提示：`θ₀ = 1/10000^0 = 1` rad，`cos1 ≈ 0.5403, sin1 ≈ 0.8415`），再用 `x.rotary(&[1])` 验证。
2. **验证梯度方向**：`rotary` 的反向已修复为 `R(θ)ᵀ`（见第 8.2 节）。自己动手把测试改成"正角度旋转"的错误版本，
   跑 `test_rotary` 和 `test_rotary_grad_exact`，观察只有后者能发现错误——体会"既测范数、又测方向"的价值。
3. **对照实际接入**：对照 10.3 节的代码，在 `src/attention.rs` 里找到 RoPE 的接入位置。
   试着把 `rotary_pair` 改成"先 `rotary` 转 Q、再 `rotary` 转 K"（各自建表），跑 `cargo test` 确认结果一致——体会 `rotary_pair` 一次建表省下的重复计算。
4. **验证相对位置性质**：写一个测试——取同一个 `q`，在位置 `m` 旋转；取同一个 `k`，在位置 `n` 旋转，
   断言 `q_m · k_n` 只和 `n - m` 有关（对固定的位置差 `d`，换不同的 `m` 结果应相同）。这正是第 5 节推导的代码验证。
5. **思考外推**：把 `block_size` 从 32 改成 64 重新训练，再让模型生成超过训练长度的文本，
   观察 RoPE 的输出质量在训练长度之外如何退化，想想为什么（角度随位置线性增长，训练长度之外照样失效）。
6. **思考**：为什么 `θ_i = pos / 10000^(2i/d)` 里的 `10000` 和 `d` 与正弦编码一模一样？
   如果把它改成 `10` 会怎样？（提示：所有维度频率变高，近距离区分更敏感、远距离更早"打满"一圈。）

---

## 12. 本课总结

- 正弦编码的两个局限：编码**绝对位置**（相对关系要模型自己学）、**外推差**（训练长度外没见过的输入分布）。
- RoPE 把 d 维拆成 **d/2 对**，第 i 对按 `θ_i = pos / 10000^(2i/d)` 旋转：
  `x'_{2i} = x_{2i}·cosθ - x_{2i+1}·sinθ`、`x'_{2i+1} = x_{2i}·sinθ + x_{2i+1}·cosθ`。
- 旋转是**正交变换**：`R(θ)ᵀR(θ) = I`，范数不变 → 数值稳定、不破坏 LayerNorm。
- **相对位置性质**：`q_m · k_n = qᵀR((n-m)θ)k`，点积只与位置差 `n - m` 有关——位置被"编死"进了打分公式。
- 与 **KV cache 天然兼容**：新 token 按 `base + j` 旋转自己的 Q/K，历史 K 已旋转、直接复用。
- `src/rope.rs` 的实现分三层：`build_cos_sin_tab` 预计算 cos/sin 表 → `rotate_with_tab` 查表旋转 → `rotary_pair` 一次建表同时旋转 Q/K；反向按正交矩阵的转置 `R(θ)ᵀ` 回传（已修复并通过 `test_rotary_grad_exact` 验证）。
- `test_rotary` 三个断言分别验证：正交性（范数不变）、`pos=0` 恒等、梯度范数守恒；`test_rotary_grad_exact` 逐元素验证梯度方向。
- 实际接入：在 `MultiHeadAttention` 拆头前、KV cache append 之前对 Q/K 旋转（只转 Q/K、不转 V），已完成接入；`GPT` 已无 `pos_emb` 字段（第 11 课正弦编码被替换）。

- 下一课（第 20 课）：收尾——学习率调度（warmup + cosine decay）与整个项目的总结回顾！
