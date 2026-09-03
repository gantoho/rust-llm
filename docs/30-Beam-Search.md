# 第 30 课：Beam Search —— 贪心搜索的升级版

> 代码位置：[src/sample.rs](src/sample.rs)（`beam_search` 函数）
>
> 对比：第 15 课实现了 temperature + top-k + top-p 采样

---

## 1. 本课要搞懂的问题

1. Greedy Search、Beam Search、随机采样三者的区别是什么？
2. Beam Search 为什么能生成更"优质"的序列？
3. 什么是长度惩罚？为什么需要它？
4. Beam Search 适合什么场景？不适合什么场景？

---

## 2. 三种解码策略对比

### 2.1 Greedy Search（贪心搜索）

每步选概率最高的 token：

```
next_token = argmax(P(token | context))
```

- 优点：简单、快
- 缺点：容易陷入局部最优，生成重复内容

### 2.2 Random Sampling（随机采样）

按概率分布随机采样：

```
next_token ~ P(token | context) / temperature
```

第 15 课实现了 temperature + top-k + top-p。

- 优点：多样性高，有创意
- 缺点：可能生成不连贯的内容

### 2.3 Beam Search（束搜索）

维护 k 个候选序列（beam），每步扩展后保留 top-k：

```
初始：beam = [prompt]
每步：
    对每个 beam，扩展所有可能的 next token
    从所有候选中选得分最高的 k 个
返回：得分最高的完整序列
```

- 优点：全局最优（在束宽 k 内），生成质量高
- 缺点：慢（每步 k 倍计算量），确定性（没有创意）

---

## 3. Beam Search 算法

### 3.1 详细步骤

```
beam_size = 4
初始 beams = [(prompt, log_prob=0)]

Step 1:
    对每个 beam，计算所有 vocab 的 log_prob
    保留 top-4 候选：[(prompt+tok1, -0.1), (prompt+tok2, -0.3), ...]

Step 2:
    对每个 beam，扩展所有 vocab
    保留 top-4：[(prompt+tok1+tok5, -0.2), ...]

...直到生成 EOS 或达到 max_len
```

### 3.2 得分计算

每个 beam 的得分是**累积 log 概率**：

```
score = Σ log P(token_i | context)
```

用 log 概率而不是概率，避免数值下溢（很多小概率相乘会变成 0）。

---

## 4. 长度惩罚

**问题**：短序列天然得分高（累积的 log_prob 项数少）。

**解决**：用长度惩罚归一化：

```
final_score = log_prob / len^α
```

- α = 0：不惩罚（默认）
- α > 0：偏好长序列
- α < 0：偏好短序列

Google NMT 论文使用 α = 0.6。

---

## 5. Beam Search vs 采样

| | Beam Search | 采样（temperature/top-k/top-p） |
|---|-----------|-------------------------------|
| 确定性 | ✅ 确定性（给定 beam_size） | ❌ 随机性 |
| 多样性 | 低 | 高 |
| 质量 | 高（全局最优） | 中（局部随机） |
| 速度 | 慢（k 倍计算量） | 快 |
| 适用场景 | 翻译、摘要、问答 | 创意写作、对话 |

---

## 6. 实现要点

### 6.1 数据结构

```rust
// 每个 beam: (token_ids, cumulative_log_prob)
let mut beams: Vec<(Vec<usize>, f64)> = vec![(prompt_ids, 0.0)];
```

### 6.2 提前终止

如果所有 beam 都生成了 EOS token，提前结束。

### 6.3 KV Cache 兼容

Beam Search 中每个 beam 的 KV Cache 独立，需要为每个 beam 维护一份。
本项目实现中使用全量前向（不用 KV Cache），简化了实现。

---

## 7. 集成状态

`beam_search` 函数已完整实现（`sample.rs:146-229`），但当前**未接入** CLI 的 `generate` 子命令。
接入方式：在 `cli.rs` 中添加 `--beam-size` 参数，在 `cmd_generate()` 中当 `beam_size > 1` 时
调用 `sample::beam_search()` 替代 `sample::generate()`。

---

## 8. 关键要点

- Beam Search 维护 k 个候选，每步扩展后保留 top-k
- 得分用累积 log 概率，避免数值下溢
- 长度惩罚 α 解决短序列天然得分高的偏置
- 适合翻译、摘要等"精确"任务；不适合创意写作
- 速度是采样的 k 倍（k = beam_size）
