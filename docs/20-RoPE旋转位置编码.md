# 第 20 课：RoPE 旋转位置编码 —— 把"相对位置"揉进注意力

> 代码位置：[src/rope.rs](../src/rope.rs)（`rotary_pair` 生产入口 + `rotary` 测试用 + `test_rotary` / `test_rotary_grad_exact` 测试）
> 配套代码：[src/attention.rs](../src/attention.rs)（RoPE 接入 `MultiHeadAttention`）
> 配套文档：[docs/11-位置编码与归一化.md](11-位置编码与归一化.md)

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

我们的模型 `block_size = 32`（`TransformerConfig::tiny`），训练时位置只见过 `0..32`。如果推理时生成更长的序列：

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
> 业界用 **NTK-aware scaling、YaRN** 等技巧把频率表拉长来做长度外推——本仓库已经实现，
> 见第 11 节「长度外推」（连 CLI 开关一并给出）。

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

> 这是本仓库默认的 `RopeStyle::Interleaved` 配对（也是 GPT-J / NeoX 的口径）。
> HuggingFace 的 LLaMA / Qwen 用另一种配对 `rotate_half`（`i` 与 `i + d/2` 配对），
> 本仓库也已支持（`RopeStyle::Half`）——两者频率表相同、旋转结果不同，详见第 12 节「实现更正」。

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
> 本仓库实现里这个 `d` 就是**单个头的宽度 `head_dim`**（不是 Q/K 拍平后的总宽）：每个头都拥有完整的频率谱，
> 行内逐头重复同一套角度，见第 7、12 节。

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

回顾第 25 课的 KV cache 推理流程（`src/sample.rs` 的 `generate` + `src/attention.rs` 的 `KVCache`）：

- 第一次：把整个 prompt 喂给模型，算出所有位置的 K/V 存入缓存；
- 之后每步：**只前向最新 1 个 token**，历史的 K/V 直接从缓存取。

RoPE 和这个流程是无缝衔接的：

```
新 token 的绝对位置 = 缓存长度 base + 它在当前窗口里的下标 j
```

`base`（已缓存的位置数）由 `Transformer::forward_core` 算出来传给每层（第 25 课的设计），`MultiHeadAttention::forward` 里 `positions` 就是这么构造的：

```rust
// src/model.rs（Transformer::forward_core）：base = 缓存**累计见过**的位置数
let (seen, _window) = kv_cache
    .as_ref()
    .and_then(|c| c.first())
    .map(|k| (k.positions_seen(), k.window()))
    .unwrap_or((0, 0));
let base = seen;

// src/attention.rs（MultiHeadAttention::forward）：batch 内每个样本位置相同，重复 b 次
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
```

> 注意用的是 `positions_seen()` 而不是缓存行数 `seq_len()`：滑动窗口丢行之后 `seq_len()`
> 会封顶在窗口长度上，绝对位置却要继续累加，用错就会让每个新 token 都拿到重复的位置号
> （两者的分工见第 25 课 3.2 节）。

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
/// 预计算每个 (位置, 头内对偶下标) 的 cos/sin 表，长度 rows × (head_dim/2)。
/// 同一批 positions 的三角只算一次：前向、反向、Q/K 复用。
///
/// `d` 就是 **head_dim**（单个注意力头的宽度，不是 Q/K 拍平后的总宽、也不是 n_embd）：
/// 表只为"一个头"建一套角度，行内每 head_dim 个元素重复使用它（见 rotate_with_tab）。
///
/// 记忆化（线程本地 4 槽 LRU，key = head_dim + spec + positions）：命中时只做两次
/// Arc 指针克隆，不再重算三角、不再分配 Vec；未命中时先查频率表缓存 FREQ_SLOT
/// （只与 (head_dim, spec) 有关）拿逆波长表，再逐位置建表。
fn build_cos_sin_tab(
    positions: &[usize],
    d: usize,
    spec: &RopeSpec,
) -> (Arc<Vec<f32>>, Arc<Vec<f32>>) {
    TAB_SLOT.with(|slot| {
        let mut slot = slot.borrow_mut();
        // 命中：直接返回共享的 Arc 表（零拷贝），并把该槽移到末尾（LRU）
        if let Some(i) = slot
            .iter()
            .position(|(cd, cs, cpos, _)| *cd == d && *cs == *spec && cpos.as_slice() == positions)
        {
            let hit = slot.remove(i);
            let tab = hit.3.clone();
            slot.push(hit); // 最近命中移到末尾
            return tab;
        }
        // 未命中：查频率表缓存（FREQ_SLOT），再逐位置建表
        let fm = freq_and_mscale(d, spec);
        let (freq, mscale) = (&fm.0, fm.1);
        let rows = positions.len();
        let half = d / 2;
        let mut c_tab = vec![0.0f32; rows * half];
        let mut s_tab = vec![0.0f32; rows * half];
        for r in 0..rows {
            let pos = positions[r] as f32;
            for i in 0..half {
                let theta = pos * freq[i]; // freq[i] = 1 / 10000^(2i/head_dim)
                c_tab[r * half + i] = theta.cos() * mscale;
                s_tab[r * half + i] = theta.sin() * mscale;
            }
        }
        let tab = (Arc::new(c_tab), Arc::new(s_tab));
        if slot.len() >= ROPE_CACHE_SLOTS {
            slot.remove(0); // 淘汰最久未用的那个
        }
        slot.push((d, *spec, positions.to_vec(), tab.clone()));
        tab
    })
}

/// 行内第 e 对元素的 (低半下标, 高半下标, 该对查表的列)——配对方式由 style 决定：
/// Interleaved 是相邻的 (2e, 2e+1)；Half（rotate_half）是同一个头内的 (j, j + head_dim/2)。
/// 两种方式**共用同一张频率表**（表列都是头内对下标），只有"哪两个元素构成一对"不同。
#[inline]
fn rope_pair_index(e: usize, head_dim: usize, style: RopeStyle) -> (usize, usize, usize) {
    let half = head_dim / 2;
    match style {
        RopeStyle::Interleaved => (2 * e, 2 * e + 1, e % half),
        RopeStyle::Half => {
            let (h, j) = (e / half, e % half); // 第几个头 / 头内第几对
            (h * head_dim + j, h * head_dim + j + half, j)
        }
    }
}

/// 用现成的 cos/sin 表旋转一个张量（[rows, D]；D 必须是 head_dim 的整数倍）。
/// 表按 head_dim 建，行内每 head_dim 个元素重复使用同一套角度（为什么必须如此见第 12 节）。
/// 反向用旋转矩阵的转置 R(θ)ᵀ 回传梯度，闭包直接查表。
fn rotate_with_tab(
    x: &Tensor,
    c_tab: &Arc<Vec<f32>>,
    s_tab: &Arc<Vec<f32>>,
    head_dim: usize,
    style: RopeStyle,
) -> Tensor {
    let (rows, d) = (x.shape[0], x.shape[1]);
    assert!(head_dim > 0 && head_dim % 2 == 0, "head_dim 必须是正偶数");
    assert_eq!(d % head_dim, 0, "最后一维必须是 head_dim 的整数倍");
    let sd = x.decode();
    let sd_ref: &[f32] = &sd;
    // Arc 表先解引用成只读切片再进并行闭包（&[f32] 跨线程共享，无需逐线程克隆表）
    let ct_f: &[f32] = c_tab;
    let st_f: &[f32] = s_tab;
    let mut out_data = vec![0.0f32; rows * d];
    let half = head_dim / 2; // 表每行的长度 = 头内对数
    let pairs = d / 2; // 一行里的对数（跨多个头）
    // 并行：每行旋转独立，行间无依赖
    out_data.par_chunks_mut(d).enumerate().for_each(|(r, out_row)| {
        let base = r * d;
        let ct_base = r * half; // 表按行建：第 r 行占 [r*half, (r+1)*half)
        for e in 0..pairs {
            let (lo, hi, i) = rope_pair_index(e, head_dim, style);
            let (c, s) = (ct_f[ct_base + i], st_f[ct_base + i]);
            let (a, b) = (sd_ref[base + lo], sd_ref[base + hi]);
            out_row[lo] = a * c - b * s;
            out_row[hi] = a * s + b * c;
        }
    });
    drop(sd);

    let result = x.new_like(out_data, x.shape.clone(), x.req());
    if x.req() {
        let rg = result.grad.clone();
        let sg = x.grad.clone();
        // cos/sin 表用 Arc 共享给反向闭包（闭包要求 'static），不 to_vec 克隆整张表
        let ct = Arc::clone(c_tab);
        let st = Arc::clone(s_tab);
        record(&result, vec![x.clone()], Arc::new(move || {
            // 梯度 borrow 一次转只读切片，不做 to_vec 拷贝；
            // 切片是 Sync，可安全带进下面的并行闭包（guard 本身不是）
            let g_b = rg.borrow();
            let g_ref: &[f32] = &g_b;
            let ct_ref: &[f32] = &ct;
            let st_ref: &[f32] = &st;
            let mut sgm = sg.borrow_mut();
            // 并行：每行独立计算梯度，行间无依赖（与前向一致）
            sgm.par_chunks_mut(d).enumerate().for_each(|(r, sgm_row)| {
                let g_base = r * d;
                let ct_base = r * half;
                for e in 0..pairs {
                    let (lo, hi, i) = rope_pair_index(e, head_dim, style);
                    let (c, s) = (ct_ref[ct_base + i], st_ref[ct_base + i]);
                    let (ga, gb) = (g_ref[g_base + lo], g_ref[g_base + hi]);
                    // 反向 = 前向旋转矩阵的转置 R(θ)ᵀ：grad = (ga·c + gb·s, -ga·s + gb·c)
                    sgm_row[lo] += ga * c + gb * s;
                    sgm_row[hi] += -ga * s + gb * c;
                }
            });
        }));
    }
    result
}
```

对外只暴露两个入口——`rotary`（旋转单个张量，仅供测试）和 `rotary_pair`（一次建表同时旋转 Q/K）。
两者现在都要求**显式传入 `head_dim`**（表按头建，不能从最后一维反推——GQA 下 Q/K 总宽本来就不同）：

```rust
/// 旋转单个张量 [rows, D]（仅测试用）
pub fn rotary(&self, positions: &[usize], head_dim: usize, spec: &RopeSpec) -> Tensor

/// 一次建表同时旋转 Q 和 K（两者 positions 相同、三角函数只算一遍），返回 (q, k)。
/// `head_dim` 两侧必须相同，但**允许 Q/K 的总宽不同**：GQA 下 `n_kv_head < n_head`，
/// K 只有 `n_kv_head·head_dim` 宽；MLA 的 latent 升维同理。
pub fn rotary_pair(
    &self,
    other: &Tensor,
    positions: &[usize],
    head_dim: usize,
    spec: &RopeSpec,
) -> (Tensor, Tensor) {
    let (c_tab, s_tab) = build_cos_sin_tab(positions, head_dim, spec); // 命中缓存则零拷贝
    (
        rotate_with_tab(self, &c_tab, &s_tab, head_dim, spec.style),
        rotate_with_tab(other, &c_tab, &s_tab, head_dim, spec.style),
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
| `assert!(head_dim > 0 && head_dim % 2 == 0, ...)` | — | `head_dim` 必须是正偶数，才能两两配对 |
| `assert_eq!(d % head_dim, 0, ...)` | — | 最后一维必须由整数个头拼成（GQA 下 K 比 Q 窄是正常的） |
| `let theta = pos * freq[i];` | `θ_i = pos · freq[i]`，`freq[i] = 1 / 10000^(2i/head_dim)` | 第 `i` 对的旋转角度（等价于 `pos / 10000^(2i/head_dim)`，`i` 的范围是 `0..head_dim/2`） |
| `let (lo, hi, i) = rope_pair_index(e, head_dim, style);` | — | 第 `e` 对的两个元素下标由**配对方式**决定（默认相邻 `(2e, 2e+1)`），`i` 是这一对查表的列 |
| `let (c, s) = (ct_f[ct_base + i], st_f[ct_base + i]);` | `cosθ_i, sinθ_i` | 直接查表取值，不再调用三角函数 |
| `let (a, b) = (sd_ref[base + lo], sd_ref[base + hi]);` | `(x_{lo}, x_{hi})` | 取出第 r 行第 e 对的两个元素（默认相邻配对时即 `(x_{2i}, x_{2i+1})`） |
| `out_row[lo] = a * c - b * s;` | `x'_{2i} = x_{2i}·cosθ - x_{2i+1}·sinθ` | 旋转后的第一个分量 |
| `out_row[hi] = a * s + b * c;` | `x'_{2i+1} = x_{2i}·sinθ + x_{2i+1}·cosθ` | 旋转后的第二个分量 |

几个值得注意的设计点：

1. **建表两层循环 + 旋转按行并行**：建表时外层按行 `r`、内层按头内对 `i`，`cos/sin` 每个 `(r, i)` 只算一次；旋转时用 `par_chunks_mut(d).enumerate().for_each(...)` 把每一行分给一个任务并行处理，行内只做查表乘加，不再碰任何三角函数（反向闭包同样按行并行）。
2. **两层线程本地缓存，各 4 槽 LRU**：`FREQ_SLOT` 按 `(head_dim, spec)` 记忆化频率表（`powf` 与 YaRN 温度只算一次），`TAB_SLOT` 按 `(head_dim, spec, positions)` 记忆化整张 cos/sin 表。key 用 **`head_dim`** 而不是"最后一维总宽"：GQA 下 Q 是 `n_head·head_dim`、K 只有 `n_kv_head·head_dim`，两者必须共用同一张表。同一次前向里各层注意力的 positions 完全相同，只有第 1 层真正建表，其余各层拿到共享的 `Arc` 表（零拷贝）。解码时 positions 逐 token 变化导致整表未命中，但频率表仍命中，每 token 只剩 rows×head_dim/2 次三角计算。**为什么是 4 槽而不是单槽**：多模态模型里视觉塔与主干交替前向，两者 `head_dim` 不同（MLA 的 `hd_rope` 又是一种），单槽会被对方顶掉、每层都未命中——纯性能问题，但会让"缓存"形同虚设。
3. **`drop(sd)`**：读完输入数据后立刻释放借用，之后才创建结果张量和反向闭包——避免闭包捕获时和输入数据的借用纠缠。
4. **`requires_grad` 分支**：如果输入不需要梯度（比如纯推理），就直接返回普通结果，不建 `parents`/`backward`，省下计算图的维护开销。这和第 25 课 KV cache 推理时"纯数据拼接、无梯度"的思路一致。
5. **反向闭包查同一张表**：反向闭包为了"自包含"（只捕获 `ct`/`st` 两份 `Arc` 克隆 + `rg`/`sg` 两个 Arc），直接查表取值，**不再重算 `theta/cos/sin`**。因为表是 `Arc` 共享的，闭包持有一份克隆几乎零成本，前向建的三角表被完整复用。注意反向闭包里 `half`/`pairs` 与配对下标都由 `head_dim`、`style` 决定，与前向逐位对称（见第 8 节）。

---

## 8. 反向实现讲解：正交矩阵的梯度

### 8.1 理论：梯度用"转置"回传

设 `y = R(θ)·x`（旋转一对）。根据链式法则：

```
∂L/∂x = R(θ)ᵀ · ∂L/∂y
```

因为旋转矩阵正交，`R(θ)ᵀ = R(-θ)`，所以**梯度回传 = 把梯度按负角度再旋转一次**。这就是方法注释里写的：

> "反向：旋转矩阵正交，梯度用其转置（负角度）回传。"（`Tensor::rotary` 的文档注释）

展开来写，若记 `(ga, gb) = (∂L/∂y_{2i}, ∂L/∂y_{2i+1})`，则正确回传是：

```
∂L/∂x_{2i}   = ga·cosθ + gb·sinθ
∂L/∂x_{2i+1} = -ga·sinθ + gb·cosθ
```

> 验证（也可以硬算）：`y₀ = a·c - b·s`、`y₁ = a·s + b·c`，
> `∂L/∂a = ga·(∂y₀/∂a) + gb·(∂y₁/∂a) = ga·c + gb·s` ✓
> `∂L/∂b = ga·(∂y₀/∂b) + gb·(∂y₁/∂b) = -ga·s + gb·c` ✓

### 8.2 代码实际怎么写的（已按正确形式实现）

`src/rope.rs` 里 `rotate_with_tab` 反向闭包的循环体是（`c`/`s` 直接查 `ct`/`st` 表；
`lo`/`hi` 是这一对的两个元素下标，由 `rope_pair_index(e, head_dim, style)` 给出，
默认相邻配对时就是 `2e` / `2e+1`）：

```rust
let (lo, hi, i) = rope_pair_index(e, head_dim, style);
let (c, s) = (ct_ref[ct_base + i], st_ref[ct_base + i]);
let (ga, gb) = (g_ref[g_base + lo], g_ref[g_base + hi]);
// 反向 = 前向旋转矩阵的转置 R(θ)ᵀ：grad = (ga·c + gb·s, -ga·s + gb·c)
sgm_row[lo] += ga * c + gb * s;
sgm_row[hi] += -ga * s + gb * c;
```

与 8.1 的结论逐行对应：第一行 `ga·c + gb·s`、第二行 `-ga·s + gb·c`，正是 `R(θ)ᵀ·g`。

> 💡 **陷阱提示**：反向回传必须用**转置**（即负角度旋转）。写代码时容易手滑把符号写成正角度旋转
> `(ga·c - gb·s, ga·s + gb·c)`——方向错了但梯度**范数不变**（正交变换保范数），
> 只校验范数的测试发现不了。所以测试要"既测范数、又测方向"（见第 9 节 `test_rotary_grad_exact`）。
> 本仓库初版实现确实写反过，已修复并补了逐元素断言。

对训练的影响：本课已经把 RoPE 接进了 `MultiHeadAttention`（`src/attention.rs`），训练和 KV cache 推理都用它。
`Transformer` 已不再有 `pos_emb` 字段，位置信息完全由注意力内部旋转 Q/K 提供（见第 10 节）。

---

## 9. `test_rotary` 测试讲解

测试在 `src/rope.rs` 的 `mod tests` 里，共三个断言，分别验证三件事（`plain()` 是测试里的辅助函数，
返回"底数 10000、不缩放、相邻配对"的 `RopeSpec`；`rotary` 现在要显式给 `head_dim`）：

```rust
#[test]
fn test_rotary() {
    // 1. 旋转是正交变换：范数不变
    let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
    let r = x.rotary(&[3], 6, &plain());
    let orig_norm: f32 = x.data().iter().map(|v| v * v).sum();
    let rot_norm: f32 = r.data().iter().map(|v| v * v).sum();
    assert!((orig_norm - rot_norm).abs() < 1e-3, "范数应守恒：{} vs {}", orig_norm, rot_norm);

    // 2. pos=0 时所有角度为 0，等于恒等变换
    let x2 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
    let r2 = x2.rotary(&[0], 4, &plain());
    assert!((r2.data()[0] - 1.0).abs() < 1e-5);
    assert!((r2.data()[3] - 4.0).abs() < 1e-5);

    // 3. 梯度：sum 的梯度是单位向量，经正交矩阵回传后范数不变（= 元素数）
    let x3 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
    let loss = x3.rotary(&[2], 6, &plain()).sum();
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
    let loss = x.rotary(&[1], 2, &plain()).sum();
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

在 `src/attention.rs` 的 `MultiHeadAttention::forward` 里，Q/K/V 投影出来后是这样的（GQA 下 K/V 只有 `n_kv_head` 个头，所以宽度是 `kv_dim`）：

```rust
let q = self.c_q.forward(x).reshape(vec![b, t, d]); // [B, T, D]
let kv_dim = self.n_kv_head * head_dim;
let k = self.c_k.forward(x).reshape(vec![b, t, kv_dim]);
let v = self.c_v.forward(x).reshape(vec![b, t, kv_dim]);
```

RoPE 有两个可选的接入时机：

| 时机 | 形状 | 说明 |
|------|------|------|
| 拆头之前 | `q/k` 是 `[B, T, D]` | 把整个 batch 展平成 `[B*T, D]` 一次旋转，代码最省事 |
| 拆头之后 | 每个头是 `[T, head_dim]` | 逐头旋转，更贴近"每头各转各的"的原始论文写法 |

我们的 `rotary_pair` 接口（输入 `[rows, D]` + 每行的 `positions` + **单个头的宽度 `head_dim`**）
两种都支持：只要"最后一维能被 `head_dim` 整除"即可。`TransformerConfig::tiny` 里 `n_embd=64`、
`n_head=4`，推出 `head_dim = 16`，拆头前后都满足。

### 10.3 实际接入方式（本仓库已接入）

`src/attention.rs` 里，Q/K 投影之后、KV cache 拼接**之前**旋转：

```rust
// 位置：Q/K 投影之后、KV cache append 之前（QK-Norm 若开启则在更前面）
let head_dim = d / self.n_head;
let kv_dim = self.n_kv_head * head_dim;
let mut positions = Vec::with_capacity(b * t);
for _ in 0..b {
    positions.extend(base..base + t);
}
// Q/K 用同一批 positions、同一个 head_dim，一次建表同时旋转（三角函数只算一遍）；
// `&self.rope` 是 RopeSpec（底数 + 外推方式 + 训练窗口 + 配对方式），随配置构造一次
let (q, k) = q
    .reshape(vec![b * t, d])
    .rotary_pair(&k.reshape(vec![b * t, kv_dim]), &positions, head_dim, &self.rope);
let (q, k) = (
    q.reshape(vec![b, t, d]),
    k.reshape(vec![b, t, kv_dim]),
);
// 之后照旧：cache.append(&k, &v)（缓存里存的就是旋转后的 K）
```

要点：

1. **`positions` 长度必须等于行数**（`rotary_pair` 里 `assert_eq!(self.shape[0], positions.len())`），这里行数是 `b*t`，所以 batch 内每个样本的同一列位置相同、要重复 `b` 次。
2. **旋转发生在 append 之前**：缓存里存的是"已旋转的 K"，历史 K 直接复用，符合第 6 节的兼容性约定。
3. **Q 没有缓存**（推理时 Q 永远只有新 token 一个），所以 Q 总是用当前位置旋转，天然正确。
4. 由于 K/V 已旋转并缓存，**推理模式（KV cache）和训练模式行为一致**，不会像正弦编码那样需要区分两套位置逻辑。这一点有专门的测试守着（`src/model.rs` 的 `test_kv_cache_matches_full_forward`）。
5. `rotary_pair` 的反向按 `R(θ)ᵀ` 实现并通过 `test_rotary_grad_exact` 逐元素验证。

> 替换还是叠加？本仓库选了**替换**：`Transformer` 结构体里没有 `pos_emb` 字段（第 11 课的正弦编码在接入 RoPE 后已删除），
> 位置信息完全由注意力内部旋转 Q/K 提供。叠加方案（保留 `pos_emb` 再加 RoPE）一般没必要，且会稀释 RoPE 的相对位置特性。

---

## 11. 长度外推：Linear（位置插值）/ NTK-aware / YaRN（本仓库已实现）

第 2.2 节说"外推是后话"——**那句话现在不成立了**：`src/rope.rs` 的 `RopeScaling`
已经把三种主流外推都实现了，并且开成了 CLI 开关（`RopeArgs`，已 flatten 进
`generate` / `chat` / `serve` 三个子命令）：

| 参数 | 含义 |
|------|------|
| `--rope-scaling linear\|ntk\|yarn` | 外推方式；留空 = 沿用 checkpoint 里的 `model.rope_scaling` |
| `--rope-factor 4.0` | 外推倍数（只在给了 `--rope-scaling` 时生效）：想跑 4 倍上下文就给 4 |
| `--rope-base 500000` | 频率底数（LLaMA-3 用 500000）；留空 = 沿用 checkpoint（默认 10000） |
| `--max-ctx 4096` | 推理上下文上限，**可以大于训练时的 `block_size`** |

```bash
# 训练窗口 512、想直接跑 2048 上下文：免训练扩上下文的标准用法
cargo run --release -- generate --config config/config.json --ckpt checkpoints/zh/latest.ckpt \
  --rope-scaling yarn --rope-factor 4 --max-ctx 2048 --prompt "..."
```

三种方式改的都只是**频率表**（旋转公式、反向公式、KV cache 存"已旋转的 K"这些结构一概不动）：

| 方式 | 干什么 | 代价 |
|------|--------|------|
| Linear（位置插值，PI） | 位置整体除以 factor，让没见过的长位置落回见过的区间 | 短距离分辨力一起被压缩 |
| NTK-aware | 只改底数 `base' = base·factor^(d/(d-2))`：高频维几乎不动、低频维整体拉长 | 短距离保住了，长距离仍略差 |
| YaRN | 按**波长**分段（`beta_fast` / `beta_slow`）：波长短于窗口的保持原频率、长于窗口的完全插值，中间线性过渡；再乘温度系数 `mscale` 补偿注意力被摊平 | 分段边界与 `train_ctx` 绑死，用错窗口会整体推偏 |

⚠️ 三条硬约束：

1. `model.rope_train_ctx` 记的是**训练时的窗口**，YaRN 的分段边界相对它计算；
   用放大后的窗口去算会把分段整体推偏。
2. `rope_base` 改了必须与训练一致——换底数就是换一张频率表。
3. 本仓库的 **GPU 常驻快路只支持原始 RoPE**（着色器里写死了底数 10000、不含任何缩放，
   而且只实现**相邻配对**），所以开了外推、或用了 `rope_style = half`（HuggingFace 口径）
   的权重，都会整段回落逐算子/CPU 路径：功能正确，但没有 GPU 加速
   （判据是 `RopeSpec::is_plain`：`scaling == None && base == 10000 && style == Interleaved`）。

## 12. 实现更正：频率表是**逐头重复**的

早期 `build_cos_sin_tab` 按**张量最后一维**（= `n_head × head_dim`）建表，等于把一整套
频率横跨所有头铺开：head 0 只拿到高频段、最后一个头只拿到低频段，每个头各覆盖 `1/n_head`
的频带。三个后果：

1. 与 GPU 着色器 `heads_split_main`（`pow(10000, -2p/hd)`，**逐头**）算的不是同一个旋转，
   CPU 与 GPU 两条路径训出来的模型不一致；
2. **GQA 直接崩溃**：`n_kv_head < n_head` 时 K 的行比 Q 短，一张表服务不了两种宽度，
   `rotary_pair` 的断言当场 panic（官方 `medium` / `large` 预设都带 GQA ⇒ **预设根本训不了**）；
3. MLA 的 latent 升维路径同样 Q/K 宽度不同，于是**静默地**用两套角度旋转 Q 与 K，
   相对位置性质失效——不报错，只是算错。

现在改成主流口径：**表按 `head_dim` 建，行内每 `head_dim` 个元素重复使用同一套角度**
（`rotate_with_tab`），于是"角度只由头内下标决定"，Q/K 宽度不同也永远一致。
**这一层与配对方式正交**：表永远是"头内对数"的完整频率谱，至于第 `e` 对取哪两个元素，
由 `rope_pair_index(e, head_dim, style)` 按 `RopeSpec.style` 决定——`Interleaved` 取相邻的
`(2e, 2e+1)`，`Half` 取同一头内的 `(j, j + head_dim/2)`；两种配对**共用同一张表**。
回归测试：`test_rope_schedule_repeats_within_each_head`、
`test_per_head_schedule_matches_gpu_shader_formula`、`test_gqa_rotary_pair_accepts_narrower_k`、
`test_gqa_relative_position_invariance_across_heads`，以及配置级的
`model::tests::test_every_preset_runs_forward_and_backward`。

> ⚠️ **这是一次行为变更**：逐头调度与旧的"横跨铺开"算出的角度不同，所以**本次改动之前**
> 训出来的 checkpoint 在数值上与新代码不一致（能加载、形状全对，但位置编码对不上，质量会掉）。
> 本仓库的权重是分钟~小时级就能重训的，重训一次即可——这也是与主流实现（以及本仓库自己的
> GPU 路径）对齐的代价。
>
> **配对方式也一并写进 checkpoint 头**：`TransformerConfig::rope_style` 会随存档序列化
> （头里的 `model.rope_style`），于是 `Interleaved`（默认，自研权重）与 `Half`
> （HuggingFace 的 `rotate_half`，`import-hf` 导入 LLaMA/Qwen 权重时写的值）都能被正确解释。
> 两者频率表完全相同、旋转结果不同，**权重不能直接互换**，所以这一位不能靠猜；旧存档缺该字段时
> 按 `Interleaved` 读入（与加这个字段之前逐位一致）。

---

## 13. 小结与下一课

### 本课小结

- 正弦编码的两个局限：编码**绝对位置**（相对关系要模型自己学）、**外推差**（训练长度外没见过的输入分布）。
- RoPE 把 d 维拆成 **d/2 对**，第 i 对按 `θ_i = pos / 10000^(2i/d)` 旋转：
  `x'_{2i} = x_{2i}·cosθ - x_{2i+1}·sinθ`、`x'_{2i+1} = x_{2i}·sinθ + x_{2i+1}·cosθ`。
- 旋转是**正交变换**：`R(θ)ᵀR(θ) = I`，范数不变 → 数值稳定、不破坏 LayerNorm。
- **相对位置性质**：`q_m · k_n = qᵀR((n-m)θ)k`，点积只与位置差 `n - m` 有关——位置被"编死"进了打分公式。
- 与 **KV cache 天然兼容**：新 token 按 `base + j` 旋转自己的 Q/K，历史 K 已旋转、直接复用。
- `src/rope.rs` 的实现分三层：`build_cos_sin_tab` 预计算 cos/sin 表（线程本地 4 槽缓存：`FREQ_SLOT` 按 `(head_dim, spec)` 记忆化频率表、`TAB_SLOT` 按 `(head_dim, spec, positions)` 记忆化整表，命中即 `Arc` 零拷贝；4 槽是为了让多模态下"视觉塔与主干交替前向"不互相顶掉）→ `rotate_with_tab` 查表旋转（角度按 **head_dim** 建、行内逐头重复，见 §12）→ `rotary_pair` 一次建表同时旋转 Q/K（**允许 Q/K 总宽不同**，GQA 与 MLA 靠这一点才成立）；反向按正交矩阵的转置 `R(θ)ᵀ` 回传（已修复并通过 `test_rotary_grad_exact` 验证）。
- `test_rotary` 三个断言分别验证：正交性（范数不变）、`pos=0` 恒等、梯度范数守恒；`test_rotary_grad_exact` 逐元素验证梯度方向。
- 两种**配对方式**（默认 `Interleaved` 相邻配对 `(2i, 2i+1)` / `Half` 的 `rotate_half`）**共用同一张频率表**，但配对下标不同 ⇒ 是不同的旋转，所以必须随权重一起记录：`TransformerConfig::rope_style` 会写进 checkpoint 头（`model.rope_style`），运行时不能靠猜；GPU 常驻快路只支持相邻配对（`RopeSpec::is_plain` 要求 `style == Interleaved`）。
- 实际接入：在 `MultiHeadAttention` 拆头前、KV cache append 之前对 Q/K 旋转（只转 Q/K、不转 V），已完成接入；`Transformer` 已无 `pos_emb` 字段（第 11 课正弦编码被替换）。

### 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. **手算一对旋转**：设 `d = 2`、`pos = 1`，对向量 `[1.0, 0.0]` 手算 `rotary` 的输出
   （提示：`θ₀ = 1/10000^0 = 1` rad，`cos1 ≈ 0.5403, sin1 ≈ 0.8415`），再用 `x.rotary(&[1], 2, &RopeSpec::default())` 验证。
2. **验证梯度方向**：`rotary` 的反向已修复为 `R(θ)ᵀ`（见第 8.2 节）。自己动手把测试改成"正角度旋转"的错误版本，
   跑 `test_rotary` 和 `test_rotary_grad_exact`，观察只有后者能发现错误——体会"既测范数、又测方向"的价值。
3. **对照实际接入**：对照 10.3 节的代码，在 `src/attention.rs` 里找到 RoPE 的接入位置。
   试着把 `rotary_pair` 改成"先 `rotary` 转 Q、再 `rotary` 转 K"（各自建表，两边都传同一个 `head_dim` / `spec`），
   跑 `cargo test` 确认结果一致——体会 `rotary_pair` 一次建表省下的重复计算。
4. **验证相对位置性质**：写一个测试——取同一个 `q`，在位置 `m` 旋转；取同一个 `k`，在位置 `n` 旋转，
   断言 `q_m · k_n` 只和 `n - m` 有关（对固定的位置差 `d`，换不同的 `m` 结果应相同）。这正是第 5 节推导的代码验证。
5. **思考外推**：把 `block_size` 从 32 改成 64 重新训练，再让模型生成超过训练长度的文本，
   观察 RoPE 的输出质量在训练长度之外如何退化，想想为什么（角度随位置线性增长，训练长度之外照样失效）。
6. **思考**：为什么 `θ_i = pos / 10000^(2i/d)` 里的 `10000` 和 `d` 与正弦编码一模一样？
   如果把它改成 `10` 会怎样？（提示：所有维度频率变高，近距离区分更敏感、远距离更早"打满"一圈。）

> **下一课**：[第 21 课 RMSNorm](21-RMSNorm.md) —— 归一化层换成只算均方根的 RMSNorm，省掉一次 reduction 与一组偏置。
