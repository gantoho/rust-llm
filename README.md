# llm_from_scratch —— 用纯 Rust 从零实现大语言模型

> **深度学习算法全部纯手写**：不使用任何深度学习框架（如 tch-rs / candle / burn），
> 从零手写张量、自动微分、神经网络层、Transformer 架构。
> 仅引入少量**工具库**（serde_json 做配置/序列化、clap 做命令行、windows-sys 修控制台编码、
> rayon 做 CPU 并行、可选的 wgpu 做 GPU 计算后端），它们都不参与任何算法实现。
>
> 每个实现步骤都配套一篇中文教程文档（见 `docs/`），边写代码边学原理。

## 项目简介

本项目是一个**教学性质**的深度学习项目，目标是让你理解大语言模型（LLM）的底层原理：

- **算法零依赖**：所有张量运算、自动微分、网络层全部手写，算法部分不用任何第三方库。
- **循序渐进**：按 [docs/00-学习计划.md](docs/00-学习计划.md) 划分 8 个阶段、38 课，从张量一路写到现代 LLM 架构，再到前沿技术（MoE、量化、RLHF、分布式训练等）。
- **工程化完整**：CLI 子命令（train / eval / generate / demo）、外部语料、train/val 划分、
  验证集评估与困惑度、checkpoint 保存/恢复、断点续训。
- **现代 LLM 技术栈**：RMSNorm、SwiGLU、GQA、Flash Attention、LoRA、混合精度、梯度累积、Beam Search。
- **透明度高**：训练过程中每一步的中间结果、梯度、损失都可以直接打印检查。

### 包含的功能（对应 38 课）

| 模块 | 文件 | 内容 |
|------|------|------|
| 张量运算 | `src/tensor.rs` | Tensor 结构体、广播、逐元素/标量运算、matmul、softmax、permute、gather |
| 自动微分 | `src/autograd.rs` | backward 反向传播、拓扑排序（计算图 → 梯度流） |
| RoPE 位置编码 | `src/rope.rs` | 旋转位置编码：把相对位置揉进 Q/K 向量 |
| 神经网络层 | `src/layers.rs` | Linear、LayerNorm、**RMSNorm**、Embedding、ReLU/GELU/Tanh、**SwiGLU**、**LoRA** |
| 损失与优化器 | `src/loss.rs` `src/optim.rs` | MSE、CrossEntropy、SGD、AdamW（动量 + 权重衰减） |
| 分词器 | `src/tokenizer.rs` | 字符级分词 + BPE（字节对编码），配置可切换 |
| 注意力机制 | `src/attention.rs` | 多头自注意力、因果掩码、RoPE、KV Cache、**GQA 分组查询注意力** |
| GPT 模型 | `src/model.rs` | Transformer Block 堆叠、GPT 整体前向、checkpoint 参数名、**Dropout** |
| 数据加载 | `src/data.rs` | 外部文本文件、train/val 划分、随机 batch 采样 |
| 训练与评估 | `src/train.rs` | 训练循环、梯度裁剪、warmup+cosine 学习率、验证集 loss / 困惑度、**梯度累积**、**混合精度 AMP** |
| 采样 | `src/sample.rs` | temperature / top-k / top-p 采样，KV cache 推理，**Beam Search** |
| 配置 | `src/config.rs` | `config.json`：模型超参 + 训练参数（serde 序列化） |
| Checkpoint | `src/checkpoint.rs` | 模型参数 + 优化器状态保存/恢复（latest / best / final） |
| 命令行 | `src/cli.rs` | clap 子命令：train / eval / generate / demo |
| 随机数 | `src/rng.rs` | 自实现 xorshift64 伪随机数发生器 |
| GPU 加速 | `src/gpu.rs` | 可选（`--features gpu`）：wgpu 计算着色器加速 matmul/scale/add/relu，失败自动回退 CPU |

### 前沿技术教程（第 31-38 课，纯文档）

| 主题 | 教程文档 | 内容 |
|------|---------|------|
| MoE 混合专家模型 | `docs/31-MoE混合专家模型.md` | 稀疏激活、Router 门控网络、负载均衡、Switch/Mixtral/DeepSeek 架构 |
| 量化技术 | `docs/32-量化技术.md` | INT8/INT4 量化、GPTQ、AWQ、GGUF、PTQ vs QAT、STE |
| RLHF 与对齐 | `docs/33-RLHF与对齐.md` | SFT、奖励模型（Bradley-Terry）、PPO、DPO、GRPO、Constitutional AI |
| 推测解码 | `docs/34-推测解码.md` | 草稿模型 + 验证、拒绝采样、无损保证、Medusa/EAGLE |
| RAG 检索增强生成 | `docs/35-RAG检索增强生成.md` | 文档分块、向量嵌入、相似度检索、重排序、HyDE、Self-RAG |
| Scaling Laws | `docs/36-Scaling-Laws.md` | 幂律关系、Chinchilla 最优配比、算力估算、涌现能力 |
| 多 Token 预测 | `docs/37-多token预测.md` | MTP 训练目标、DeepSeek 实现、与推测解码结合 |
| 分布式训练 | `docs/38-分布式训练.md` | 数据并行、ZeRO、张量并行、流水线并行、3D 并行、通信原语 |

## 快速开始

需要 **Rust 2024 edition** 工具链（Rust 1.85+，建议使用最新的 stable）。

```bash
# 最简方式：用默认 config.json 训练 + 生成
cargo run --release -- train --config config.json
cargo run --release -- generate --config config.json --prompt "Once upon a" --max-new 100
```

---

## 命令行完整参考

程序提供 4 个子命令：`train` / `eval` / `generate` / `demo`。

### 1. `train` —— 训练模型

```bash
cargo run --release -- train [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config.json` | 配置文件路径（模型超参 + 训练参数） |
| `--resume <路径>` | string | 无 | 从已有 checkpoint 续训（恢复参数、优化器状态、步数） |

**训练流程**：
1. 读取配置文件，构建分词器（char / bpe）
2. 构建 GPT 模型（参数量由 `config.json` 的 `model` 段决定）
3. 加载训练语料，自动切分训练集 / 验证集
4. 每 `eval_every` 步：在验证集上评估 loss / 困惑度，保存 `latest.ckpt`
5. 验证 loss 刷新最优时额外保存 `best.ckpt`
6. 训练结束时保存 `final.ckpt`

**输出文件**（在 `out_dir` 目录下）：
- `latest.ckpt` —— 最近一次评估的 checkpoint
- `best.ckpt` —— 验证 loss 最优的 checkpoint
- `final.ckpt` —— 训练结束时的 checkpoint

**示例**：

```bash
# ── 基础训练 ──
# 用默认 config.json 训练（BPE 分词、2000 步、batch=8）
cargo run --release -- train --config config.json

# ── 断点续训 ──
# 从最近的 checkpoint 继续（恢复参数、优化器状态、步数）
cargo run --release -- train --config config.json --resume checkpoints/latest.ckpt

# 从最优 checkpoint 续训（继续微调）
cargo run --release -- train --config config.json --resume checkpoints/best.ckpt

# ── GPU 加速训练 ──
# 开启 wgpu 计算着色器（NVIDIA / Intel 核显），失败自动回退 CPU
cargo run --release --features gpu -- train --config config.json

# GPU 加速 + 断点续训
cargo run --release --features gpu -- train --config config.json --resume checkpoints/latest.ckpt

# ── 不同模型规模的训练（修改 config.json）──
# 小模型（教学用，秒级完成）：n_embd=64, n_layer=2, block_size=32
# 中模型（几分钟）：n_embd=256, n_layer=4, block_size=128
# 大模型（需要耐心）：n_embd=512, n_layer=8, block_size=256

# ── 不同分词器（修改 config.json 的 tokenizer 字段）──
# 字符级分词（小数据集，词表小）
#   "tokenizer": "char"
# BPE 分词（大数据集，压缩率高）
#   "tokenizer": "bpe", "bpe_vocab": 512
# BPE 大词表（更好的子词覆盖）
#   "tokenizer": "bpe", "bpe_vocab": 1024

# ── 不同训练策略（修改 config.json）──
# 快速验证（50 步，确认代码能跑通）
#   "steps": 50, "eval_every": 10
# 标准训练（2000 步，loss 充分收敛）
#   "steps": 2000, "eval_every": 250
# 长训练（更高质量）
#   "steps": 10000, "eval_every": 500, "max_lr": 3e-4, "min_lr": 3e-5

# ── 梯度累积（小显存模拟大 batch）──
# 有效 batch = batch_size × accum_steps = 4 × 4 = 16
#   "batch_size": 4, "accum_steps": 4

# ── LLaMA 风格架构训练 ──
# 启用 RMSNorm + SwiGLU + GQA
#   "use_rmsnorm": true, "use_swiglu": true, "n_kv_head": 2
```

---

### 2. `eval` —— 评估模型

```bash
cargo run --release -- eval [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config.json` | 配置文件路径 |
| `--ckpt <路径>` | string | `checkpoints/latest.ckpt` | checkpoint 文件路径 |

**评估指标**：
- `val_loss` —— 验证集上的交叉熵损失
- `perplexity`（困惑度）—— `e^val_loss`，越低越好（理想值接近 1）

**示例**：

```bash
# ── 基础评估 ──
# 评估最新 checkpoint（默认 checkpoints/latest.ckpt）
cargo run --release -- eval --config config.json

# ── 评估不同 checkpoint ──
# 评估最优 checkpoint
cargo run --release -- eval --config config.json --ckpt checkpoints/best.ckpt

# 评估最终 checkpoint
cargo run --release -- eval --config config.json --ckpt checkpoints/final.ckpt

# 评估指定路径的 checkpoint
cargo run --release -- eval --config config.json --ckpt /path/to/my_model.ckpt

# ── 评估不同模型配置 ──
# 用不同的 config 评估（config 决定模型架构，必须与 checkpoint 训练时一致）
cargo run --release -- eval --config config_llama.json
cargo run --release -- eval --config config_large.json --ckpt checkpoints/best.ckpt

# ── GPU 加速评估 ──
cargo run --release --features gpu -- eval --config config.json
```

---

### 3. `generate` —— 生成文本

```bash
cargo run --release -- generate [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config.json` | 配置文件路径 |
| `--ckpt <路径>` | string | `checkpoints/latest.ckpt` | checkpoint 文件路径 |
| `--prompt <文本>` | string | `""`（空） | 初始提示词（模型从这里开始续写） |
| `--max-new <数量>` | int | `100` | 最多生成的新 token 数 |
| `--temperature <温度>` | float | `0.8` | 采样温度（>1 更随机，<1 更确定，0 = 贪心） |
| `--top-k <数量>` | int | `40` | top-k 采样：只从概率最高的 k 个 token 里选 |
| `--top-p <概率>` | float | `0.9` | top-p 采样：累积概率到 p 的最小集合 |
| `--seed <种子>` | int | `42` | 随机种子（相同种子 + 相同参数 = 相同输出） |
| `--no-kv-cache` | flag | 关闭 | 禁用 KV cache（每个新 token 都全量前向，慢但省内存） |

**采样策略**：temperature 调整 → top-k 截断 → top-p 截断 → 按概率随机抽样

**示例**：

```bash
# ═══════════════════════════════════════════
#  基础生成
# ═══════════════════════════════════════════

# 用默认参数生成（temperature=0.8, top-k=40, top-p=0.9）
cargo run --release -- generate --config config.json --prompt "Alice was" --max-new 100

# 指定 checkpoint 生成
cargo run --release -- generate --config config.json --ckpt checkpoints/best.ckpt --prompt "Once upon a" --max-new 200

# 空 prompt（模型自由发挥）
cargo run --release -- generate --config config.json --max-new 50

# ═══════════════════════════════════════════
#  采样温度控制（--temperature）
# ═══════════════════════════════════════════

# 贪心解码（temperature→0，每次输出完全相同，最高确定性）
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.01 --max-new 50

# 低温采样（保守，输出较确定，适合事实性文本）
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.3 --max-new 50

# 默认温度（平衡创造性和连贯性）
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.8 --max-new 50

# 高温采样（更随机、更有创造性，可能出现不通顺的文本）
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 1.5 --max-new 50

# ═══════════════════════════════════════════
#  Top-k 采样控制（--top-k）
# ═══════════════════════════════════════════

# top-k=1（等价于贪心，只选概率最高的 token）
cargo run --release -- generate --config config.json --prompt "The" --top-k 1 --max-new 30

# top-k=10（较保守，只从 top-10 候选中选）
cargo run --release -- generate --config config.json --prompt "The" --top-k 10 --max-new 30

# top-k=40（默认，较平衡）
cargo run --release -- generate --config config.json --prompt "The" --top-k 40 --max-new 30

# top-k=100（较开放，候选更多）
cargo run --release -- generate --config config.json --prompt "The" --top-k 100 --max-new 30

# top-k=0（禁用 top-k，不限制候选数量，完全依赖 top-p）
cargo run --release -- generate --config config.json --prompt "The" --top-k 0 --max-new 30

# ═══════════════════════════════════════════
#  Top-p (nucleus) 采样控制（--top-p）
# ═══════════════════════════════════════════

# top-p=0.5（较保守，只从累积概率前 50% 的 token 中选）
cargo run --release -- generate --config config.json --prompt "The" --top-p 0.5 --max-new 30

# top-p=0.9（默认，较平衡）
cargo run --release -- generate --config config.json --prompt "The" --top-p 0.9 --max-new 30

# top-p=1.0（禁用 top-p，不限制候选范围，完全依赖 top-k）
cargo run --release -- generate --config config.json --prompt "The" --top-p 1.0 --max-new 30

# ═══════════════════════════════════════════
#  组合使用（temperature + top-k + top-p）
# ═══════════════════════════════════════════

# 确定性输出（低温 + 小 top-k，适合代码/事实生成）
cargo run --release -- generate --config config.json --prompt "def" --temperature 0.2 --top-k 5 --top-p 0.8 --max-new 50

# 平衡输出（默认参数，适合一般文本续写）
cargo run --release -- generate --config config.json --prompt "Once" --temperature 0.8 --top-k 40 --top-p 0.9 --max-new 100

# 创意输出（高温 + 大 top-k + 大 top-p，适合创意写作）
cargo run --release -- generate --config config.json --prompt "Once" --temperature 1.2 --top-k 100 --top-p 0.95 --max-new 100

# 极端随机（高温 + 禁用截断，可能产生不通顺文本，用于观察模型分布）
cargo run --release -- generate --config config.json --prompt "The" --temperature 2.0 --top-k 0 --top-p 1.0 --max-new 30

# ═══════════════════════════════════════════
#  随机种子控制（--seed）
# ═══════════════════════════════════════════

# 相同种子 = 相同输出（可复现）
cargo run --release -- generate --config config.json --prompt "Hello" --seed 42 --max-new 30
cargo run --release -- generate --config config.json --prompt "Hello" --seed 42 --max-new 30  # 输出与上面完全相同

# 不同种子 = 不同输出（同一分布的不同采样）
cargo run --release -- generate --config config.json --prompt "Hello" --seed 1 --max-new 30
cargo run --release -- generate --config config.json --prompt "Hello" --seed 999 --max-new 30  # 输出不同

# ═══════════════════════════════════════════
#  可复现性说明（相同命令 = 相同输出，这是刻意设计）
# ═══════════════════════════════════════════

# 相同命令执行两次，输出完全一致（可复现）：
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 50
# ↑ 两次输出一模一样，因为：固定种子 + 确定性权重 + CPU f32 确定性计算

# 换种子 → 不同的采样路径 → 不同的文本：
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 99 --max-new 50
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 1234 --max-new 50
# ↑ 三次输出各不相同（同一分布的不同采样）

# 提高温度 → 更大的随机性 → 输出变化更大：
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.2 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.8 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 1.5 --top-k 20 --seed 7 --max-new 50
# ↑ 同一种子，温度从低到高，输出从保守到随机

# temperature→0 等价于贪心解码，无论什么种子输出都一样（不走随机采样）：
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.01 --top-k 1 --seed 7 --max-new 30
cargo run --release -- generate --config config.json --prompt "The fox" --temperature 0.01 --top-k 1 --seed 99 --max-new 30
# ↑ 两次输出完全相同（贪心模式下种子无效，总是选概率最高的 token）

# 与 ChatGPT 等商用模型的区别：
#   商用模型：每次请求随机生成种子 → 每次输出不同
#   本项目：  固定种子 → 每次输出相同 → 便于精确对比不同参数/模型的效果（学习项目的核心优势）
#

# ═══════════════════════════════════════════
#  KV Cache 控制（--no-kv-cache）
# ═══════════════════════════════════════════

# 默认开启 KV cache（推荐，推理速度快）
cargo run --release -- generate --config config.json --prompt "Once" --max-new 100

# 禁用 KV cache（每个 token 都全量前向，慢但结果一致，用于调试对比）
cargo run --release -- generate --config config.json --prompt "Once" --max-new 100 --no-kv-cache

# ═══════════════════════════════════════════
#  使用不同 checkpoint 生成
# ═══════════════════════════════════════════

# 使用最新 checkpoint（默认）
cargo run --release -- generate --config config.json --prompt "The key"

# 使用验证 loss 最优的 checkpoint
cargo run --release -- generate --config config.json --ckpt checkpoints/best.ckpt --prompt "The key" --max-new 100

# 使用训练结束时的 checkpoint
cargo run --release -- generate --config config.json --ckpt checkpoints/final.ckpt --prompt "The key" --max-new 100

# 使用自定义路径的 checkpoint
cargo run --release -- generate --config config.json --ckpt /path/to/custom.ckpt --prompt "The key" --max-new 100

# ═══════════════════════════════════════════
#  GPU 加速生成
# ═══════════════════════════════════════════

# GPU 加速推理（大模型时明显提速）
cargo run --release --features gpu -- generate --config config.json --prompt "Once upon a" --max-new 200

# GPU + 自定义参数
cargo run --release --features gpu -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --max-new 150 --seed 7
```

---

### 4. `demo` —— 教学演示

```bash
cargo run --release -- demo
```

无参数。依次运行 4 个教学演示：

1. **MLP 学习 XOR**（第 7 课）：验证神经网络 + 反向传播正确，训练后正确率 4/4（100%）
2. **BPE 分词器**（第 8 课）：在示例语料上训练 BPE 词表（400 个 token），演示编码/解码往返
3. **训练小 GPT 并生成文本**（第 12-21 课）：669 字符英文故事上训练 600 步，loss 4.14→0.16，
   然后用 temperature=0.8 / top-k=10 / top-p=0.9 生成文本，分别演示无 KV cache 和带 KV cache 两种推理
4. **GPU 加速对比**（第 21 课，仅 `--features gpu`）：验证 CPU vs GPU 数值一致性，实测加速比

**示例**：

```bash
# ── 标准演示（CPU）──
# 依次运行：XOR → BPE → 小 GPT 训练+生成 → （无 GPU 提示）
cargo run --release -- demo

# ── 带 GPU 加速的演示 ──
# 第 4 个演示会做 CPU vs GPU 数值一致性验证 + 性能对比
cargo run --release --features gpu -- demo

# ── Debug 模式（不加 --release，编译快但运行慢，适合调试）──
cargo run -- demo
```

---

### 5. `cargo test` —— 单元测试

```bash
# ── 运行全部测试 ──
cargo test

# ── 运行指定测试（按名称过滤）──
# 只运行自动微分相关测试
cargo test test_chain_rule

# 只运行矩阵乘法测试
cargo test test_matmul

# 只运行 RoPE 测试
cargo test test_rotary

# 只运行 KV cache 一致性测试
cargo test test_kv_cache

# 只运行分词器测试
cargo test tokenizer

# ── 显示测试输出（包括 println!）──
cargo test -- --nocapture

# ── 运行指定测试并显示输出 ──
cargo test test_softmax -- --nocapture
```

运行 26 个单元测试（零外部依赖，秒级完成）：

| 测试 | 验证内容 |
|------|---------|
| `test_chain_rule` | 自动微分链式法则 |
| `test_broadcast_add` | 广播加法正确性 |
| `test_softmax` / `test_log_softmax` | softmax 数值稳定性 |
| `test_matmul_2d` / `test_matmul_3d` | 2D / 3D 矩阵乘法 |
| `test_permute` | 任意维度重排 |
| `test_reshape_grad_flows` | reshape 梯度传递 |
| `test_relu_grad` | ReLU 反向传播 |
| `test_gather_rows` | Embedding 查表 |
| `test_layernorm_fused_matches_chain` | LayerNorm 融合算子 vs 分步实现 |
| `test_rmsnorm_fused_matches_chain` | RMSNorm 融合算子 vs 分步实现 |
| `test_swiglu_matches_elementwise` | SwiGLU 融合算子 vs 逐元素实现 |
| `test_masked_softmax_matches_chain` | 掩码 softmax 融合算子 vs 分步实现 |
| `test_flash_attention_matches_standard` | Flash Attention vs 标准注意力 |
| `test_dropout_p0_identity` / `test_dropout_eval_identity` | Dropout 正确性 |
| `test_rotary` / `test_rotary_grad_exact` | RoPE 正交性 + 梯度精确验证 |
| `test_kv_cache_matches_full_forward` | KV cache 推理 vs 全量前向一致性 |
| `test_char_tokenizer_roundtrip` / `test_bpe_roundtrip` | 分词器编码/解码往返 |
| `test_linear_regression_converges` | 线性回归收敛 |
| `test_rng_deterministic` / `test_rng_range` / `test_choice_range` | 随机数生成器 |

---

### 6. GPU 加速（可选 feature）

默认构建**不启用 GPU**，保持依赖轻量。通过 `--features gpu` 开启 wgpu 计算着色器加速：

```bash
# ── 所有子命令都支持 --features gpu ──

# GPU 加速训练
cargo run --release --features gpu -- train --config config.json

# GPU 加速训练 + 断点续训
cargo run --release --features gpu -- train --config config.json --resume checkpoints/latest.ckpt

# GPU 加速评估
cargo run --release --features gpu -- eval --config config.json

# GPU 加速生成
cargo run --release --features gpu -- generate --config config.json --prompt "Once upon a" --max-new 200

# GPU 加速生成 + 自定义采样参数
cargo run --release --features gpu -- generate --config config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 150

# GPU 加速演示（含 CPU vs GPU 正确性验证 + 性能对比）
cargo run --release --features gpu -- demo

# ── Debug 模式 GPU（编译快，适合开发调试）──
cargo run --features gpu -- demo
```

**支持的 GPU**：NVIDIA 独显、Intel 核显（Windows 走 DX12 / Vulkan，无需额外驱动）

**加速范围**：
- 批量矩阵乘（tiled 16×16 共享内存）—— QKV 投影、MLP 等大矩阵自动走 GPU
- 逐元素 scale / add / ReLU
- 微型矩阵（注意力 scores 等）自动回退 CPU（GPU dispatch 开销 > 计算本身）

**分流策略**：FLOPs < 200,000 的矩阵走 CPU，其余走 GPU。训练结束时打印 `matmul 分流统计：GPU x 次 / CPU y 次`

**自动回退**：GPU 初始化失败或任何调用出错时，自动回退 CPU，不影响正确性

---

## 配置文件完整参考（`config.json`）

配置文件分为 `model`（模型超参）和 `train`（训练参数）两个段。缺省字段自动取默认值。

### model 段 —— 模型超参数

```jsonc
{
  "model": {
    "vocab_size": 0,       // 词表大小。0 = 由分词器决定（训练时自动填入）
    "n_embd": 128,         // 隐藏维度（越大模型越强，显存和计算量越大）
    "n_head": 4,           // 注意力头数（Q 的头数）
    "n_layer": 4,          // Transformer 层数（越深越强）
    "block_size": 64,      // 最大上下文长度（能处理的最长序列）
    "n_kv_head": 0,        // KV 头数。0 = 标准 MHA；< n_head 时启用 GQA
    "use_rmsnorm": false,  // true = RMSNorm（LLaMA 风格），false = LayerNorm（GPT-2 风格）
    "use_swiglu": false,   // true = SwiGLU MLP（LLaMA 风格），false = GELU MLP（GPT-2 风格）
    "dropout": 0.0         // Dropout 概率。0 = 不丢弃，>0 时训练中随机丢弃
  }
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vocab_size` | int | `0` | 词表大小。`0` 表示训练时由分词器自动决定 |
| `n_embd` | int | `64` | 隐藏维度。GPT-2 用 768，LLaMA-7B 用 4096 |
| `n_head` | int | `4` | 注意力头数。`n_embd` 必须能被 `n_head` 整除 |
| `n_layer` | int | `2` | Transformer 层数 |
| `block_size` | int | `32` | 最大上下文长度（token 数） |
| `n_kv_head` | int | `0` | KV 头数。`0` = 与 `n_head` 相同（标准 MHA）；设为更小值启用 GQA（如 `n_head=8, n_kv_head=2`） |
| `use_rmsnorm` | bool | `false` | 是否使用 RMSNorm 替代 LayerNorm |
| `use_swiglu` | bool | `false` | 是否使用 SwiGLU MLP 替代 GELU MLP |
| `dropout` | float | `0.0` | Dropout 概率（0~1）。用于注意力权重和残差连接 |

### train 段 —— 训练参数

```jsonc
{
  "train": {
    "seed": 42,               // 随机种子（相同种子 = 可复现的实验）
    "batch_size": 8,          // 每批序列条数
    "steps": 2000,            // 总训练步数
    "max_lr": 6e-4,           // 峰值学习率（warmup 后达到）
    "min_lr": 6e-5,           // cosine 衰减的最低学习率
    "warmup_steps": 50,       // 线性预热步数（从 0 线性升到 max_lr）
    "weight_decay": 0.01,     // AdamW 权重衰减系数
    "grad_clip": 1.0,         // 梯度裁剪阈值（梯度总范数超过此值时等比缩放）
    "eval_every": 250,        // 每 N 步评估一次验证集（同时保存 latest checkpoint）
    "eval_iters": 20,         // 评估时采样的批数（取平均减少方差）
    "tokenizer": "bpe",       // 分词器类型："char"（字符级）或 "bpe"（字节对编码）
    "bpe_vocab": 512,         // BPE 目标词表大小（= 256 字节 + 合并数）
    "train_file": "data/alice.txt", // 训练语料文件路径
    "val_file": null,         // 验证语料文件路径。null = 自动从训练文本末尾切 10%
    "out_dir": "checkpoints", // checkpoint 输出目录
    "accum_steps": 1          // 梯度累积步数。有效 batch = batch_size × accum_steps
  }
}
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `seed` | int | `42` | 随机种子。相同种子产生相同初始化和数据采样顺序 |
| `batch_size` | int | `8` | 每批并行处理的序列条数 |
| `steps` | int | `1000` | 总训练步数。越大训练越久，效果通常越好 |
| `max_lr` | float | `0.003` | 峰值学习率。AdamW 的初始学习率 |
| `min_lr` | float | `0.0003` | cosine 衰减的最低学习率。训练后期学习率衰减到此值 |
| `warmup_steps` | int | `20` | 线性预热步数。前 N 步学习率从 0 线性升到 `max_lr`（≤ `steps`） |
| `weight_decay` | float | `0.01` | AdamW 权重衰减。正则化防过拟合 |
| `grad_clip` | float | `1.0` | 梯度裁剪。所有参数梯度的 L2 范数超过此值时等比缩放 |
| `eval_every` | int | `100` | 每 N 步在验证集上评估 loss / 困惑度，并保存 `latest.ckpt` |
| `eval_iters` | int | `20` | 评估时采样多少批取平均（减少随机波动） |
| `tokenizer` | string | `"bpe"` | `"char"` = 字符级分词；`"bpe"` = 字节对编码 |
| `bpe_vocab` | int | `512` | BPE 词表大小。仅当 `tokenizer = "bpe"` 时生效 |
| `train_file` | string | `"data/sample.txt"` | 训练语料文件路径（纯文本） |
| `val_file` | string/null | `null` | 验证语料文件。`null` = 自动从训练文本末尾切约 10% |
| `out_dir` | string | `"checkpoints"` | checkpoint 保存目录（自动创建） |
| `accum_steps` | int | `1` | 梯度累积步数。每 `accum_steps` 个小 batch 才做一次 optimizer.step()。有效 batch = `batch_size × accum_steps` |

### 完整配置示例

```json
{
  "model": {
    "vocab_size": 0,
    "n_embd": 256,
    "n_head": 8,
    "n_layer": 4,
    "block_size": 128,
    "n_kv_head": 0,
    "use_rmsnorm": false,
    "use_swiglu": false,
    "dropout": 0.0
  },
  "train": {
    "seed": 42,
    "batch_size": 16,
    "steps": 2000,
    "max_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 50,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "eval_every": 250,
    "eval_iters": 20,
    "tokenizer": "bpe",
    "bpe_vocab": 512,
    "train_file": "data/alice.txt",
    "val_file": null,
    "out_dir": "checkpoints",
    "accum_steps": 1
  }
}
```

### LLaMA 风格配置示例

```json
{
  "model": {
    "vocab_size": 0,
    "n_embd": 512,
    "n_head": 8,
    "n_kv_head": 2,
    "n_layer": 6,
    "block_size": 256,
    "use_rmsnorm": true,
    "use_swiglu": true,
    "dropout": 0.1
  },
  "train": {
    "steps": 5000,
    "batch_size": 32,
    "accum_steps": 2,
    "max_lr": 3e-4,
    "min_lr": 3e-5,
    "warmup_steps": 100,
    "tokenizer": "bpe",
    "bpe_vocab": 1024
  }
}
```

## 目录结构

```
llm_from_scratch/
├── Cargo.toml          # 依赖：仅工具库（serde_json / clap / windows-sys）+ 可选 wgpu
├── README.md           # 本文件
├── config.json         # 训练配置（模型超参 + 训练参数）
├── data/               # 示例语料（data/alice.txt：公版《爱丽丝梦游仙境》）
├── src/
│   ├── main.rs         # CLI 入口：train / eval / generate / demo
│   ├── cli.rs          # 命令行定义（clap）
│   ├── config.rs       # 配置加载（serde）
│   ├── checkpoint.rs   # checkpoint 保存 / 恢复
│   ├── attention.rs    # 多头注意力 + KV Cache（第 9-10、18-19 课）
│   ├── autograd.rs     # 自动微分：backward + 拓扑排序（第 2 课）
│   ├── tensor.rs       # 张量运算（第 1、3-4 课）
│   ├── gpu.rs          # GPU 计算后端（第 21 课，--features gpu）：WGSL 计算着色器
│   ├── rope.rs         # RoPE 旋转位置编码（第 19 课）
│   ├── rng.rs          # 随机数（第 5 课）
│   ├── layers.rs       # 网络层（第 5、11 课）
│   ├── loss.rs         # 损失函数（第 6 课）
│   ├── optim.rs        # 优化器（第 6、17 课）
│   ├── module.rs       # 参数管理 trait（第 5 课）
│   ├── tokenizer.rs    # 分词器（第 8 课）
│   ├── model.rs        # GPT 模型：Transformer Block + 前向（第 11-12 课）
│   ├── data.rs         # 数据集（第 14 课）
│   ├── train.rs        # 训练循环与学习率调度（第 13、20 课）
│   └── sample.rs       # 推理与采样（第 15 课）
└── docs/               # 38 课教程文档（00-学习计划 + 01~38 各课）
```

## 学习路线

1. 先读每课教程文档（`docs/XX-xxx.md`）理解原理；
2. 再看对应源码，对照实现；
3. 自己动手改代码、跑实验，验证理解；
4. 完成每课末尾的"动手练习"。

> 推荐从 [docs/00-学习计划.md](docs/00-学习计划.md) 开始，按阶段顺序阅读：
>
> | 阶段 | 课程 | 内容 | 代码 |
> |------|------|------|------|
> | 一、地基 | 01-04 | 张量、自动微分、广播、模块化 | ✅ |
> | 二、神经网络 | 05-07 | 线性层、激活函数、损失、MLP | ✅ |
> | 三、分词器 | 08 | BPE 字节对编码 | ✅ |
> | 四、Transformer | 09-12 | 注意力、多头、位置编码、GPT | ✅ |
> | 五、训练与推理 | 13-16 | 训练循环、数据、采样、训练小 GPT | ✅ |
> | 六、进阶组件 | 17-21 | AdamW、KV Cache、RoPE、学习率调度、GPU | ✅ |
> | 七、现代技术栈 | 22-30 | RMSNorm、SwiGLU、GQA、Flash Attention、LoRA、AMP 等 | ✅ |
> | 八、前沿技术 | 31-38 | Scaling Laws、MoE、量化、推测解码、RLHF、RAG、分布式 | 📖 |

## 代码验证状态

- **26 个单元测试全部通过**（`cargo test`），零警告
- 测试覆盖：自动微分、广播、softmax、BPE 编解码、RoPE 正交性与梯度、KV cache 与全量前向一致性、
  RMSNorm/SwiGLU 融合算子与分步实现一致性、Flash Attention 与标准注意力一致性、Dropout、线性回归收敛
- **Demo 端到端验证通过**（`cargo run --release -- demo`）：XOR 100%、BPE 往返、GPT 训练 loss 4.14→0.16、文本生成正常
- 已实现但未集成到 demo 的功能：LoRA 注入、Beam Search、AMP 动态损失缩放（均有完整代码和测试）

## 实现要点与踩坑记录

- **性能陷阱**：`Tensor` 的父节点列表必须用 `Rc<Vec<Tensor>>` 存储。若直接存 `Vec<Tensor>`，
  `#[derive(Clone)]` 会递归深拷贝整棵祖先计算图，深层图上每次建节点都是 O(图深) 开销，
  曾导致单步训练耗时 26 秒；改用 Rc 共享后降至约 150ms。
- **同张量特判**：`x * x`、`x + x` 这类同一张量参与运算的情况，反向传播时梯度要走两条路径
  合并（用 `Rc::ptr_eq` 判断），否则会 `RefCell` 双重借用报错或梯度算错。
- **数值稳定**：softmax 先减每行最大值再 exp，防止指数溢出。
- **RoPE 反向**：旋转矩阵正交，梯度回传要用其转置 `R(θ)ᵀ`（相当于负角度旋转），
  符号写反不会影响梯度范数，但方向会错——务必用逐元素断言测试校验。
- **RoPE 接入**：在注意力内部对 Q/K 旋转（只转 Q/K、不转 V），且旋转发生在 KV cache append 之前，
  缓存里存的是"已旋转的 K"，历史 K 直接复用；训练（base=0）与 KV cache 推理的绝对位置统一为 `base + j`。
- **BPE 编码复杂度**：`BPETokenizer::encode` 若"每次只合并一个 pair 并全量重扫"是 O(n²×m)，
  大语料会卡死。改为 GPT-2 风格的"按规则优先级单趟扫描替换"（O(len×合并数)），174KB 语料秒级编码。
- **Windows 控制台**：默认 GBK 代码页会让中文输出乱码，程序启动时用 `SetConsoleOutputCP(65001)`
  切到 UTF-8（通过 windows-sys 实现）。
- **WGSL 变量遮蔽**：matmul 着色器里 `let b = ...` 会把全局 storage 数组 `b` 遮蔽成 u32，
  再写 `b[...]` 就变成"对 u32 索引"，naga 报 `Invalid access into expression`——局部变量不要与全局资源同名。
- **WGSL uniform 数组对齐**：uniform 地址空间中数组 stride 必须 16 字节对齐，`array<u32,4>` 会被摊成 64 字节；
  改用 4 个独立 u32 字段（共 16 字节）传参即可。
- **wgpu 30 API 变更**：`PipelineLayoutDescriptor` 已无 `push_constant_ranges`（改 `immediate_size`）、
  `bind_group_layouts` 元素是 `Option<&BindGroupLayout>`、`PollType::Wait` 是带字段的 struct variant、
  `get_mapped_range()` 返回 `Result`、`ComputePipelineDescriptor` 需 `cache` 字段。
- **绑定编号**：多个计算入口共用同一 module 时，storage 绑定声明（binding 0/1/2/3）是全局的；
  scale/relu 只用其中 3 个，创建 bind group 时要显式指定与 layout 一致的 binding 编号（0/2/3），不能从 0 连续排。

## 后续方向

以下方向已在第 31-38 课教程文档中详细讲解，代码实现可作为进阶练习：

- **MoE 混合专家模型**（第 31 课）：Router 门控 + 多专家稀疏激活，Mixtral / DeepSeek 架构
- **量化部署**（第 32 课）：INT4/INT8 量化、GPTQ/AWQ，让大模型跑在消费级显卡上
- **人类对齐**（第 33 课）：RLHF (PPO) / DPO / GRPO，让模型"有用、无害、诚实"
- **推测解码**（第 34 课）：小模型猜 + 大模型验，无损加速推理 2-4×
- **RAG 检索增强生成**（第 35 课）：向量检索 + LLM 生成，解决知识截止和幻觉问题
- **Scaling Laws**（第 36 课）：指导训练资源分配的幂律公式，Chinchilla 最优配比
- **多 Token 预测**（第 37 课）：同时预测未来 K 个 token，提升表征质量
- **分布式训练**（第 38 课）：数据并行、ZeRO、张量/流水线并行，训练百亿级模型
- 长度外推：RoPE 配合 NTK-aware scaling、YaRN 等技巧（见第 19 课文档）
- 更大的语料与模型规模（`config.json` 可直接调大，CPU 训练需耐心）
