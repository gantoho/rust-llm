# llm_from_scratch —— 用纯 Rust 从零实现大语言模型

> **深度学习算法全部纯手写**：不使用任何深度学习框架（如 tch-rs / candle / burn），
> 从零手写张量、自动微分、神经网络层、Transformer 架构。
> 仅引入少量**工具库**（serde_json 做配置/序列化、clap 做命令行、windows-sys 修控制台编码、
> rayon 做 CPU 并行、可选的 wgpu 做 GPU 计算后端），它们都不参与任何算法实现。
>
> 每个实现步骤都配套一篇中文教程文档（见 `docs/`），边写代码边学原理。



## 项目简介

本项目是一个从零实现的 **GPT 大语言模型**项目，目标是让你理解大语言模型（LLM）的底层原理：

- **算法零依赖**：所有张量运算、自动微分、网络层全部手写，算法部分不用任何第三方库。
- **循序渐进**：按 [docs/00-学习计划.md](docs/00-学习计划.md) 划分 10 个阶段、39 课，从张量一路写到现代 LLM 架构，再到前沿技术（MoE、量化、RLHF、分布式训练等），最后工程化完善。
- **工程化完整**：CLI 子命令（train / eval / generate / chat / sft / finetune / preset / demo / bench）、
  外部语料、train/val 划分、验证集评估与困惑度、checkpoint 保存/恢复、断点续训。
- **性能可量化**：内置 `bench` 基准子命令，用固定小模型在秒级内测出训练/推理吞吐（tok/s），
  优化改动前后可同机对比（详见 [性能优化与基准测试](#性能优化与基准测试)）。
- **现代 LLM 技术栈**：RoPE、RMSNorm、SwiGLU、GQA、Flash Attention、梯度累积、Beam Search。
- **真实可用**：加载预训练权重微调、交互式对话、分词器持久化、训练指标日志、运行日志（每次训练/推理自动存档）、预设模型配置。
- **教学实现（尚未接入训练循环）**：LoRA 层与 `MixedPrecision` 动态损失缩放都有完整实现和文档，
  但都还没接进 `train_gpt`——`finetune` 目前是常规全参微调，见 [§6](#6-finetune--加载预训练权重微调) 与第 26 / 29 课。
- **透明度高**：训练过程中每一步的中间结果、梯度、损失都可以直接打印检查。

### 包含的功能（对应 39 课）

| 模块 | 文件 | 内容 |
|------|------|------|
| 张量运算 | `src/tensor.rs` | Tensor 结构体、广播、逐元素/标量运算、matmul、softmax、permute、gather |
| 自动微分 | `src/autograd.rs` | backward 反向传播、拓扑排序（计算图 → 梯度流） |
| 模块接口 | `src/module.rs` | `Module` trait：参数收集的统一接口（`parameters()`） |
| RoPE 位置编码 | `src/rope.rs` | 旋转位置编码：把相对位置揉进 Q/K 向量 |
| 神经网络层 | `src/layers.rs` | Linear、LayerNorm、**RMSNorm**、Embedding、ReLU/GELU/Tanh、**SwiGLU**、**LoRA** |
| 损失与优化器 | `src/loss.rs` `src/optim.rs` | MSE、CrossEntropy、SGD、AdamW（动量 + 权重衰减） |
| 分词器 | `src/tokenizer.rs` | 字符级分词 + BPE（字节对编码），**save/load 持久化**，配置可切换；生成时做 UTF-8 约束，不会拼出乱码字符 |
| 注意力机制 | `src/attention.rs` | 多头自注意力、因果掩码、RoPE、KV Cache、**GQA 分组查询注意力** |
| GPT 模型 | `src/model.rs` | Transformer Block 堆叠、GPT 整体前向、checkpoint 参数名、**Dropout** |
| 数据加载 | `src/data.rs` | 外部文本文件、**目录批量加载**、train/val 划分、随机 batch 采样；**SFT 对话语料解析 + loss 掩码** |
| 训练与评估 | `src/train.rs` | 训练循环、梯度裁剪、warmup+cosine 学习率、验证集 loss / 困惑度、**梯度累积**、早停、**CSV 指标日志**、**SFT 掩码透传**；另外实现了但未接入的 `MixedPrecision` 动态损失缩放 |
| 采样 | `src/sample.rs` | temperature / top-k / top-p 采样 + 重复惩罚，KV cache 推理，**Beam Search**，**停止标记** |
| 配置 | `src/config.rs` | `config/config.json`：模型超参 + 训练参数 + **预设配置**（small/medium/large）+ **LoRA 配置** + **SFT 语料** |
| Checkpoint | `src/checkpoint.rs` | 模型参数 + 优化器状态保存/恢复（latest / best / final），`LLMCP2` 二进制格式 |
| 命令行 | `src/cli.rs` | clap 子命令：train / eval / generate / **chat** / **sft** / **finetune** / **preset** / demo / **bench** |
| 随机数 | `src/rng.rs` | 自实现 xorshift64 伪随机数发生器 |
| GPU 加速 | `src/gpu.rs` | 可选（`--features gpu`）：wgpu 计算着色器加速 matmul/scale/add/relu，失败自动回退 CPU |

### 前沿技术教程（第 31-38 课，纯文档）

| 主题 | 教程文档 | 内容 |
|------|---------|------|
| Scaling Laws | `docs/31-Scaling-Laws.md` | 幂律关系、Chinchilla 最优配比、算力估算、涌现能力 |
| MoE 混合专家模型 | `docs/32-MoE混合专家模型.md` | 稀疏激活、Router 门控网络、负载均衡、Switch/Mixtral/DeepSeek 架构 |
| 量化技术 | `docs/33-量化技术.md` | INT8/INT4 量化、GPTQ、AWQ、GGUF、PTQ vs QAT、STE |
| 推测解码 | `docs/34-推测解码.md` | 草稿模型 + 验证、拒绝采样、无损保证、Medusa/EAGLE |
| 多 Token 预测 | `docs/35-多token预测.md` | MTP 训练目标、DeepSeek 实现、与推测解码结合 |
| RLHF 与对齐 | `docs/36-RLHF与对齐.md` | SFT、奖励模型（Bradley-Terry）、PPO、DPO、GRPO、Constitutional AI |
| RAG 检索增强生成 | `docs/37-RAG检索增强生成.md` | 文档分块、向量嵌入、相似度检索、重排序、HyDE、Self-RAG |
| 分布式训练 | `docs/38-分布式训练.md` | 数据并行、ZeRO、张量并行、流水线并行、3D 并行、通信原语 |

### 工程化完善教程（第 39 课，代码+文档）

| 主题 | 教程文档 | 内容 |
|------|---------|------|
| 工程化完善 | `docs/39-工程化完善.md` | 8 个 CLI 子命令、分词器序列化、预设配置、微调工作流、交互式对话、Beam Search CLI、多文件数据加载、CSV 指标日志 |

## 快速开始

需要 **Rust 2024 edition** 工具链（Rust 1.85+，建议使用最新的 stable）。

```bash
# ═══════════════════════════════════════════
#  最简方式：训练 + 生成（推理不需要语料）
# ═══════════════════════════════════════════
cargo run --release -- train --config config/config.json
# 训练完成后，推理只需 checkpoint，分词器自动加载
cargo run --release -- generate --ckpt checkpoints/best.ckpt --prompt "Once upon a" --max-new 100

cargo run --release -- generate --ckpt checkpoints/best.ckpt --prompt "The" --max-new 200

# ═══════════════════════════════════════════
#  使用预设配置（推荐）
# ═══════════════════════════════════════════
# 生成中等模型配置（LLaMA 风格，~15M 参数）
cargo run --release -- preset --name medium --output config/config_medium.json
cargo run --release -- train --config config/config_medium.json

# ═══════════════════════════════════════════
#  交互式对话（训练后直接对话，无需语料）
# ═══════════════════════════════════════════
cargo run --release -- chat --ckpt checkpoints/best.ckpt

# ═══════════════════════════════════════════
#  监督微调（把只会续写的预训练模型教会"应答"）
# ═══════════════════════════════════════════
cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt
cargo run --release -- chat --ckpt checkpoints/zh-sft/best.ckpt   # 之后用 SFT 权重对话

# ═══════════════════════════════════════════
#  微调（加载预训练模型，当前为全参微调；LoRA 尚未接入训练循环）
# ═══════════════════════════════════════════
cargo run --release -- finetune --config config/config.json --pretrained checkpoints/best.ckpt

# ═══════════════════════════════════════════
#  教学演示（验证所有算法正确性）
# ═══════════════════════════════════════════
cargo run --release -- demo

# ═══════════════════════════════════════════
#  性能基准（秒级测出训练/推理吞吐，用于优化前后对比）
# ═══════════════════════════════════════════
cargo run --release -- bench
cargo run --release -- bench --steps 30
```

---

## 命令行完整参考

程序提供 9 个子命令：`train` / `eval` / `generate` / `chat` / `sft` / `finetune` / `preset` / `demo` / `bench`。

### 1. `train` —— 训练模型

```bash
cargo run --release -- train [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径（模型超参 + 训练参数） |
| `--resume <路径>` | string | 无 | 从已有 checkpoint 续训（恢复参数、优化器状态、步数） |

**训练流程**：
1. 读取配置文件，构建分词器（char / bpe）
2. 构建 GPT 模型（参数量由 `config/config.json` 的 `model` 段决定）
3. 加载训练语料，自动切分训练集 / 验证集
4. 每 `eval_every` 步：在验证集上评估 loss / 困惑度，保存 `latest.ckpt`
5. 验证 loss 刷新最优时额外保存 `best.ckpt`
6. 训练结束时保存 `final.ckpt`

**输出文件**（在 `out_dir` 目录下）：
- `latest.ckpt` —— 最近一次评估的 checkpoint
- `best.ckpt` —— 验证 loss 最优的 checkpoint
- `final.ckpt` —— 训练结束时的 checkpoint
- `tokenizer.json` —— 训练好的分词器（推理时自动加载，无需语料）

**checkpoint 文件格式**（`LLMCP2`，自描述二进制）：

```text
魔数 "LLMCP2\n"（7 字节）
u32 小端：JSON 头长度
JSON 头：step、best_val_loss、模型配置、优化器步数 opt_t、参数元信息（名字 + 形状）
参数数据块：按参数顺序拼接的 f32 小端
一阶动量 m 数据块：与参数同样的顺序与形状（续训用）
二阶动量 v 数据块：与参数同样的顺序与形状（续训用）
```

三段数据块等长（都是「参数总元素数 × 4」字节），所以**文件大小只由模型结构决定、与数值无关**。JSON 头里只有标量和元信息，占比极小——例如 144 万参数的模型，17.3MB 的存档里 JSON 头只有 1961 字节。

注意 `.ckpt` 只是文件后缀，不代表内部格式。旧版（`LLMCP1`）的内容是「魔数 + **51.3MB JSON 文本头** + 7.4MB 二进制参数块」：**参数本来就是二进制**，被文本编码撑大的是**优化器状态**——`m` / `v` 当时是 JSON 头里的两个字段，一个 f32 平均要 13~14 字节（二进制只要 4 字节）。同一个 184 万参数的模型（`config/config.json`），58.6MB 的存档里 87% 都花在这上面，把 `m` / `v` 也改成数据块后降到 22.1MB。文本还有个隐患：JSON 没有 `NaN` / `Infinity` 字面量，`serde_json` 会把它写成 `null`，读回时直接反序列化失败——训练一发散，存档就变成读不回来的废文件。

> `LLMCP2` 与旧格式不兼容，也不提供转换脚本（改格式只为存得更小、更稳），旧 `.ckpt` 请重新训练。

**日志**：**每个评估点**（每 `eval_every` 步 + 最后一步）向 `logs/train.csv` 写一行（由 `train.log_file` 指定，默认 `logs/train.csv`，目录自动创建），列为 `step,lr,train_loss,val_loss,ppl,tokens_per_sec`；每次训练会覆盖该文件，要留档就一个实验配一个路径。

**运行日志**：`train` / `finetune` / `eval` / `generate` / `chat` 五个子命令**每次运行都会自动在 `logs/` 下写一份运行日志**，文件名是 `{操作}_{年-月-日_时-分-秒-毫秒}.log`（如 `logs/generate_2026-09-19_14-30-12-345.log`），操作名区分命令、毫秒时间戳区分同命令的多次运行，互不覆盖。每份日志包含：运行头部（操作名、开始时间（命令开始执行的时刻）、**完整命令行**、工作目录、版本 / 平台 / 线程数 / GPU）、**完整配置**（`--config` 解析后的全部字段，含被 CLI 覆盖后的最终值）、本次运行的关键参数（采样参数 / prompt / checkpoint 等）与全部过程输出（训练进度、评估点、生成文本、对话轮次），结尾附结束时间与总耗时。文件名时间戳、开始时间、总耗时同源，都取自 `main()` 入口记下的时刻，所以耗时覆盖参数解析、配置与模型加载在内的**全过程**。写入由 `src/runlog.rs` 统一负责，控制台与日志内容一致，不需要再手动重定向。

**推理不需要语料文件**：训练完成后，`eval` / `generate` / `chat` 命令自动从 checkpoint 目录加载 `tokenizer.json`，不再需要 `train_file` 或语料。只需指定 `--ckpt` 即可：

```bash
# 训练
cargo run --release -- train --config config/config.json
# 推理（只需 checkpoint，分词器自动加载）
cargo run --release -- generate --ckpt checkpoints/best.ckpt --prompt "Once upon a" --max-new 100
cargo run --release -- chat --ckpt checkpoints/best.ckpt
cargo run --release -- eval --ckpt checkpoints/best.ckpt
```

**分词器加载优先级**：
1. `--tokenizer` 参数（命令行显式指定）
2. `config/config.json` 的 `tokenizer_file` 字段
3. `{out_dir}/tokenizer.json`（训练时自动保存的，推荐）
4. 从 `train_file` 语料训练（兜底，不推荐）

**示例**：

```bash
# ── 基础训练 ──
# 用默认 config/config.json 训练（BPE 分词、2000 步、batch=8）
cargo run --release -- train --config config/config.json

# ── 断点续训 ──
# 从最近的 checkpoint 继续（恢复参数、优化器状态、步数）
cargo run --release -- train --config config/config.json --resume checkpoints/latest.ckpt

# 从最优 checkpoint 续训（继续微调）
cargo run --release -- train --config config/config.json --resume checkpoints/best.ckpt

# ── GPU 加速训练 ──
# 开启 wgpu 计算着色器（NVIDIA / Intel 核显），失败自动回退 CPU
cargo run --release --features gpu -- train --config config/config.json

# GPU 加速 + 断点续训
cargo run --release --features gpu -- train --config config/config.json --resume checkpoints/latest.ckpt

# ── 不同模型规模的训练（修改 config/config.json）──
# 小模型（教学用，秒级完成）：n_embd=64, n_layer=2, block_size=32
# 中模型（几分钟）：n_embd=256, n_layer=4, block_size=256
# 大模型（需要耐心）：n_embd=512, n_layer=8, block_size=256

# ── 不同分词器（修改 config/config.json 的 tokenizer 字段）──
# 字符级分词（小数据集，词表小）
#   "tokenizer": "char"
# BPE 分词（大数据集，压缩率高）
#   "tokenizer": "bpe", "bpe_vocab": 512
# BPE 大词表（更好的子词覆盖）
#   "tokenizer": "bpe", "bpe_vocab": 1024

# ── 不同训练策略（修改 config/config.json）──
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
| `--config <路径>` | string | `config/config.json` | 配置文件路径 |
| `--ckpt <路径>` | string | 无（缺省用 `{out_dir}/latest.ckpt`） | checkpoint 文件路径 |
| `--tokenizer <路径>` | string | 自动查找 | 分词器文件路径（缺省从 `{out_dir}/tokenizer.json` 加载） |

**评估指标**：
- `val_loss` —— 验证集上的交叉熵损失
- `perplexity`（困惑度）—— `e^val_loss`，越低越好（理想值接近 1）

**示例**：

```bash
# ── 最简评估（只需 checkpoint，分词器自动加载）──
cargo run --release -- eval --ckpt checkpoints/best.ckpt

# ── 基础评估 ──
# 不传 --ckpt 时用 {out_dir}/latest.ckpt（config/config.json 的 out_dir 是 checkpoints/zh）
cargo run --release -- eval --config config/config.json

# ── 评估不同 checkpoint ──
# 评估最优 checkpoint
cargo run --release -- eval --config config/config.json --ckpt checkpoints/best.ckpt

# 评估最终 checkpoint
cargo run --release -- eval --config config/config.json --ckpt checkpoints/final.ckpt

# 评估指定路径的 checkpoint
cargo run --release -- eval --config config/config.json --ckpt /path/to/my_model.ckpt

# ── 评估不同模型配置 ──
# 用不同的 config 评估（config 决定模型架构，必须与 checkpoint 训练时一致）
cargo run --release -- eval --config config/config_llama.json
cargo run --release -- eval --config config/config_large.json --ckpt checkpoints/best.ckpt

# ── GPU 加速评估 ──
cargo run --release --features gpu -- eval --config config/config.json
```

---

### 3. `generate` —— 生成文本

```bash
cargo run --release -- generate [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径 |
| `--ckpt <路径>` | string | 无（缺省用 `{out_dir}/latest.ckpt`） | checkpoint 文件路径 |
| `--tokenizer <路径>` | string | 自动查找 | 分词器文件路径（缺省从 `{out_dir}/tokenizer.json` 加载） |
| `--prompt <文本>` | string | `""`（空） | 初始提示词（模型从这里开始续写） |
| `--max-new <数量>` | int | `100` | 最多生成的新 token 数 |
| `--temperature <温度>` | float | `0.8` | 采样温度（>1 更随机，<1 更确定，0 = 贪心） |
| `--top-k <数量>` | int | `40` | top-k 采样：只从概率最高的 k 个 token 里选 |
| `--top-p <概率>` | float | `0.9` | top-p 采样：累积概率到 p 的最小集合 |
| `--repetition-penalty <系数>` | float | `1.1` | 重复惩罚：>1 压低最近出现过的 token（`1.0` = 关闭） |
| `--repetition-window <数量>` | int | `64` | 重复惩罚的回看窗口：只看最近 N 个 token（`0` = 关闭） |
| `--seed <种子>` | int | `42` | 随机种子（相同种子 + 相同参数 = 相同输出） |
| `--no-kv-cache` | flag | 关闭 | 禁用 KV cache（每个新 token 都全量前向，慢但省内存） |
| `--beam <束宽>` | int | 无（不用） | Beam Search 束宽（通常 4-10），指定后使用确定性搜索 |
| `--length-penalty <α>` | float | `0.6` | Beam Search 长度惩罚（0=不惩罚，>0 偏好长序列） |

**推理不需要语料**：训练时自动保存 `tokenizer.json` 到 checkpoint 目录，推理时自动加载。

**采样策略**：重复惩罚 → temperature 调整 → top-k 截断 → top-p 截断 → 按概率随机抽样

**重复惩罚（repetition penalty）**：小模型容易陷入"太太太太太太……"这种自我强化的重复循环——
一旦上下文里出现某个 token，注意力就让它成为下一个 token 的最优选择，采样怎么截断都跳不出去。
惩罚的做法是：把最近 `--repetition-window` 个 token 的 logit 压低（正值除以系数、负值乘以系数），
再走正常的 temperature/top-k/top-p。`1.0` 关闭，常用范围 `1.05~1.3`；调得太大会连"的""了"这类
高频虚词一起压掉，句子反而散架。

**示例**：

```bash
# ═══════════════════════════════════════════
#  最简推理（只需 checkpoint，无需 config 和语料）
# ═══════════════════════════════════════════
cargo run --release -- generate --ckpt checkpoints/best.ckpt --prompt "Alice was" --max-new 100

# ═══════════════════════════════════════════
#  基础生成
# ═══════════════════════════════════════════

# 用默认参数生成（temperature=0.8, top-k=40, top-p=0.9）
cargo run --release -- generate --config config/config.json --prompt "Alice was" --max-new 100

# 指定 checkpoint 生成
cargo run --release -- generate --config config/config.json --ckpt checkpoints/best.ckpt --prompt "Once upon a" --max-new 200

# 空 prompt（模型自由发挥）
cargo run --release -- generate --config config/config.json --max-new 50

# ═══════════════════════════════════════════
#  采样温度控制（--temperature）
# ═══════════════════════════════════════════

# 贪心解码（temperature→0，每次输出完全相同，最高确定性）
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.01 --max-new 50

# 低温采样（保守，输出较确定，适合事实性文本）
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.3 --max-new 50

# 默认温度（平衡创造性和连贯性）
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.8 --max-new 50

# 高温采样（更随机、更有创造性，可能出现不通顺的文本）
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 1.5 --max-new 50

# ═══════════════════════════════════════════
#  Top-k 采样控制（--top-k）
# ═══════════════════════════════════════════

# top-k=1（等价于贪心，只选概率最高的 token）
cargo run --release -- generate --config config/config.json --prompt "The" --top-k 1 --max-new 30

# top-k=10（较保守，只从 top-10 候选中选）
cargo run --release -- generate --config config/config.json --prompt "The" --top-k 10 --max-new 30

# top-k=40（默认，较平衡）
cargo run --release -- generate --config config/config.json --prompt "The" --top-k 40 --max-new 30

# top-k=100（较开放，候选更多）
cargo run --release -- generate --config config/config.json --prompt "The" --top-k 100 --max-new 30

# top-k=0（禁用 top-k，不限制候选数量，完全依赖 top-p）
cargo run --release -- generate --config config/config.json --prompt "The" --top-k 0 --max-new 30

# ═══════════════════════════════════════════
#  Top-p (nucleus) 采样控制（--top-p）
# ═══════════════════════════════════════════

# top-p=0.5（较保守，只从累积概率前 50% 的 token 中选）
cargo run --release -- generate --config config/config.json --prompt "The" --top-p 0.5 --max-new 30

# top-p=0.9（默认，较平衡）
cargo run --release -- generate --config config/config.json --prompt "The" --top-p 0.9 --max-new 30

# top-p=1.0（禁用 top-p，不限制候选范围，完全依赖 top-k）
cargo run --release -- generate --config config/config.json --prompt "The" --top-p 1.0 --max-new 30

# ═══════════════════════════════════════════
#  组合使用（temperature + top-k + top-p）
# ═══════════════════════════════════════════

# 确定性输出（低温 + 小 top-k，适合代码/事实生成）
cargo run --release -- generate --config config/config.json --prompt "def" --temperature 0.2 --top-k 5 --top-p 0.8 --max-new 50

# 平衡输出（默认参数，适合一般文本续写）
cargo run --release -- generate --config config/config.json --prompt "Once" --temperature 0.8 --top-k 40 --top-p 0.9 --max-new 100

# 创意输出（高温 + 大 top-k + 大 top-p，适合创意写作）
cargo run --release -- generate --config config/config.json --prompt "Once" --temperature 1.2 --top-k 100 --top-p 0.95 --max-new 100

# 极端随机（高温 + 禁用截断，可能产生不通顺文本，用于观察模型分布）
cargo run --release -- generate --config config/config.json --prompt "The" --temperature 2.0 --top-k 0 --top-p 1.0 --max-new 30

# ═══════════════════════════════════════════
#  重复惩罚（--repetition-penalty）
# ═══════════════════════════════════════════

# 默认开启（1.1）：小模型长文本生成最常见的毛病是"某某某某某某……"卡住，惩罚能直接打断循环
cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "我们" --max-new 200

# 关掉对照：同一个种子下会坍缩成一长串重复字
cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "我们" --max-new 200 --repetition-penalty 1.0

# 加重惩罚 + 放大回看窗口（重复更顽固时用，太大则句子散架）
cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "我们" --max-new 200 --repetition-penalty 1.3 --repetition-window 128

# ═══════════════════════════════════════════
#  随机种子控制（--seed）
# ═══════════════════════════════════════════

# 相同种子 = 相同输出（可复现）
cargo run --release -- generate --config config/config.json --prompt "Hello" --seed 42 --max-new 30
cargo run --release -- generate --config config/config.json --prompt "Hello" --seed 42 --max-new 30  # 输出与上面完全相同

# 不同种子 = 不同输出（同一分布的不同采样）
cargo run --release -- generate --config config/config.json --prompt "Hello" --seed 1 --max-new 30
cargo run --release -- generate --config config/config.json --prompt "Hello" --seed 999 --max-new 30  # 输出不同

# ═══════════════════════════════════════════
#  可复现性说明（相同命令 = 相同输出，这是刻意设计）
# ═══════════════════════════════════════════

# 相同命令执行两次，输出完全一致（可复现）：
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 50
# ↑ 两次输出一模一样，因为：固定种子 + 确定性权重 + CPU f32 确定性计算

# 换种子 → 不同的采样路径 → 不同的文本：
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 99 --max-new 50
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 1234 --max-new 50
# ↑ 三次输出各不相同（同一分布的不同采样）

# 提高温度 → 更大的随机性 → 输出变化更大：
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.2 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.8 --top-k 20 --seed 7 --max-new 50
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 1.5 --top-k 20 --seed 7 --max-new 50
# ↑ 同一种子，温度从低到高，输出从保守到随机

# temperature→0 等价于贪心解码，无论什么种子输出都一样（不走随机采样）：
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.01 --top-k 1 --seed 7 --max-new 30
cargo run --release -- generate --config config/config.json --prompt "The fox" --temperature 0.01 --top-k 1 --seed 99 --max-new 30
# ↑ 两次输出完全相同（贪心模式下种子无效，总是选概率最高的 token）

# 与 ChatGPT 等商用模型的区别：
#   商用模型：每次请求随机生成种子 → 每次输出不同
#   本项目：  固定种子 → 每次输出相同 → 便于精确对比不同参数/模型的效果（学习项目的核心优势）
#

# ═══════════════════════════════════════════
#  KV Cache 控制（--no-kv-cache）
# ═══════════════════════════════════════════

# 默认开启 KV cache（推荐，推理速度快）
cargo run --release -- generate --config config/config.json --prompt "Once" --max-new 100

# 禁用 KV cache（每个 token 都全量前向，慢但结果一致，用于调试对比）
cargo run --release -- generate --config config/config.json --prompt "Once" --max-new 100 --no-kv-cache

# ═══════════════════════════════════════════
#  使用不同 checkpoint 生成
# ═══════════════════════════════════════════

# 使用最新 checkpoint（默认）
cargo run --release -- generate --config config/config.json --prompt "The key"

# 使用验证 loss 最优的 checkpoint
cargo run --release -- generate --config config/config.json --ckpt checkpoints/best.ckpt --prompt "The key" --max-new 100

# 使用训练结束时的 checkpoint
cargo run --release -- generate --config config/config.json --ckpt checkpoints/final.ckpt --prompt "The key" --max-new 100

# 使用自定义路径的 checkpoint
cargo run --release -- generate --config config/config.json --ckpt /path/to/custom.ckpt --prompt "The key" --max-new 100

# ═══════════════════════════════════════════
#  Beam Search 生成（确定性，质量更高）
# ═══════════════════════════════════════════

# Beam Search（beam_size=5），确定性输出，质量优于随机采样
cargo run --release -- generate --config config/config.json --prompt "The fox" --beam 5 --max-new 100

# Beam Search + 长度惩罚（偏好长序列）
cargo run --release -- generate --config config/config.json --prompt "The fox" --beam 8 --length-penalty 0.8 --max-new 100

# ═══════════════════════════════════════════
#  GPU 加速生成
# ═══════════════════════════════════════════

# GPU 加速推理（大模型时明显提速）
cargo run --release --features gpu -- generate --config config/config.json --prompt "Once upon a" --max-new 200

# GPU + 自定义参数
cargo run --release --features gpu -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --max-new 150 --seed 7
```

---

### 4. `chat` —— 交互式对话

```bash
cargo run --release -- chat [参数]
```

交互式对话模式：持续输入文本，模型逐个生成回复。输入 `:quit` 退出。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径 |
| `--ckpt <路径>` | string | 无（缺省用 `{out_dir}/latest.ckpt`） | checkpoint 文件路径 |
| `--tokenizer <路径>` | string | 自动查找 | 分词器文件路径（缺省从 `{out_dir}/tokenizer.json` 加载） |
| `--system <文本>` | string | `""` | 系统提示词（在每次输入前附加） |
| `--temperature <温度>` | float | `0.8` | 采样温度 |
| `--top-k <数量>` | int | `40` | top-k 采样 |
| `--top-p <概率>` | float | `0.9` | top-p 采样 |
| `--repetition-penalty <系数>` | float | `1.1` | 重复惩罚（`1.0` = 关闭） |
| `--repetition-window <数量>` | int | `64` | 重复惩罚的回看窗口（`0` = 关闭） |
| `--max-new <数量>` | int | `200` | 每次生成的最大 token 数 |
| `--seed <种子>` | int | `42` | 随机种子 |
| `--prompt-format <模板>` | string | `sft` | `sft` = 用与 `sft` 子命令一致的对话模板拼 prompt（模型才会"回答"）；`raw` = 直接把历史拼给模型 |

**推理不需要语料**：训练时自动保存 `tokenizer.json` 到 checkpoint 目录，对话时自动加载。

**`--prompt-format`**：默认 `sft`，prompt 会被拼成训练时的模板形态（`用户：` / `助手：`，见 [§5](#5-sft--监督微调把续写变成应答)），
并在模型吐出 `（结束）` 或 `用户：` 时停下——这样它接的是"该我回答了"的位置，而且不会顺着模板继续编下一轮提问。
**未做过 SFT 的预训练权重请用 `--prompt-format raw`**：它没见过这套标记，套上模板只会更差。

**上下文预算**：`block_size` 是「system prompt + 对话历史 + 本轮生成」三者共用的窗口。分配优先级依次是：
system prompt 永远保留（它是序列开头的位置锚点），本轮生成预留 `--max-new` 个 token，剩下的额度给对话历史。
历史按真实 token 数裁剪，超出即从最老的轮次开始丢弃；窗口紧张时可调小 `--max-new` 换取更长记忆。

**示例**：

```bash
# ── 最简对话（只需 checkpoint，无需 config 和语料）──
cargo run --release -- chat --ckpt checkpoints/best.ckpt

# ── 带系统提示的对话 ──
cargo run --release -- chat --ckpt checkpoints/best.ckpt --system "You are a helpful assistant."

# ── 创意对话（高温采样）──
cargo run --release -- chat --ckpt checkpoints/best.ckpt --temperature 1.0 --max-new 300
```

---

### 5. `sft` —— 监督微调：把"续写"变成"应答"

```bash
cargo run --release -- sft [参数]
```

预训练的目标是"预测下一个 token"、语料是连续文本切片，所以模型学到的唯一行为就是**接着往下写**：
你问它「1+1 等于几」，它会把这句话当成小说开头继续编下去。要让它"回答"，就得给它
「提问 → 回答」的监督信号——这就是 SFT（监督微调）。

与 `train` 的区别只有**数据与 loss**，训练循环完全共用：

| | `train`（预训练） | `sft`（监督微调） |
|---|---|---|
| 样本 | 语料切片 | 对话（若干轮问答） |
| 参与 loss 的位置 | 全部 | **只有回答段**（提问与角色标记被掩码屏蔽） |

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径 |
| `--pretrained <路径>` | string | 必填 | 预训练 checkpoint（SFT 必须从预训练权重出发） |
| `--sft-file <路径>` | string | 用 config 的 `train.sft_file` | SFT 语料，逗号分隔，每项可为文件、目录或含 `*` 的路径 |
| `--steps <步数>` | int | 用 config 的 `train.steps` | 微调步数 |
| `--lr <学习率>` | float | config `train.max_lr` 的 **1/10** | 峰值学习率 |
| `--out-dir <目录>` | string | `{config 的 out_dir}-sft` | 输出目录 |

**输出目录默认带 `-sft` 后缀**：绝不写回 `out_dir`，否则会覆盖预训练攒下的 `latest.ckpt` / `best.ckpt`
和整条 loss 曲线，SFT 效果不好就再也回不去了。指标 CSV 同理（写到 `{out_dir}/sft.csv`）。

**学习率默认取预训练配置的 1/10**：SFT 是在已收敛的权重上继续训，用预训练那种步长会把预训练
攒下的语言能力一起冲掉。显式传 `--lr` 时以你给的为准。

**评估间隔会按步数自动收窄**：配置里的 `eval_every`（默认 200）是给预训练上万步用的，
直接套在几百步的 SFT 上会把整段训练压成"只在最后评估一次"——`best.ckpt` 退化成 `final.ckpt`，
早停也永远等不到第二次评估。所以实际取 `min(eval_every, steps/10)`。

**`best.ckpt` 与 `final.ckpt` 都要试**：SFT 语料通常很小（本项目 89 段对话 / 18432 token，
切 10% 当验证区），验证区与训练区同分布，于是 val 曲线往往**头几十步就见底**，之后
train_loss 继续降而 val_loss 回升——`best.ckpt` 恰好是那个"刚开始像样"的欠训练快照。
实测 `best.ckpt`（val 4.8569 @ step 50）聊出来的句子比 `final.ckpt`（train 2.5437 @ step 250）
更散；`final.ckpt` 反而把中文助手的应答句式学得更足。两个都在输出目录里，各聊几句再定。

**语料格式**：行首写角色标记，识别这些前缀（同一文件里混用两套也行）：

```text
用户：你好
助手：你好！有什么可以帮你的？
（空行分隔不同的对话）

A: 吃了吗
B: 吃了

面试官：请介绍一下你自己。
应聘者：我是一名后端工程师。
```

- 固定标签 `用户` / `助手` / `A` / `B` 有明确含义。
- **人名标签**（`陈教授：` / `面试官：` / `张伟：`）按「该文件里谁先说」判定：第一个出现的算提问方。
  它只在**整份文件 ≥90% 的非空行都是角色行、且说话人只有两个**时才启用——这样即使把
  `sft_file` 指向混杂目录，也不会把小说正文（`秦琼道：……`）当成对话喂进来。
- 不带角色标记的文本整体跳过。
- 每段对话末尾的 `（结束）` 由程序自动补，**不用自己写**。

**模板标记只用预训练语料里出现过的字**（自定义模板时同样适用）：`#` 在 4.7M 字的中文语料里
只出现 2 次，它的 embedding 基本没被训过，模板一带上 `### ` 就会把模型推进 ASCII 乱码模式
（实测基座模型输出 `'何人？」621.-----2..3E8188`）。所以模板用的是 `用户：` / `助手：` / `（结束）`，
不是 `### 用户：` 这种 Markdown 风格标记；`【】` 同理不能用在标记里（也只出现 2 次）。

**示例**：

```bash
# ── 基础 SFT（步数用 config 的，学习率用 config lr 的 1/10）──
cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt

# ── 指定语料与步数 ──
cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt \
    --sft-file "data/corpus/zh_dialogue_*.txt,data/sft/" --steps 300 --lr 1e-4

# ── 训练完对话（--prompt-format 默认就是 sft）──
cargo run --release -- chat --ckpt checkpoints/zh-sft/best.ckpt --max-new 60
cargo run --release -- chat --ckpt checkpoints/zh-sft/final.ckpt --max-new 60
```

效果对照（`chat --temperature 0.3`，同一条输入「你好」）：

| 权重 | 输出 | 行为 |
|---|---|---|
| 基座（`--prompt-format raw`） | `了？"那孩子們一碗。"` …接着往下写 | **续写**：把提问当小说开头 |
| SFT `final.ckpt` | `我要这个月费。我们有限。您的账户用。` | **应答**：短句、现代中文、答完收尾 |

> ⚠️ **能力上限**：SFT 只负责把"续写"扭成"应答"（学会角色位置、答完收尾），它教不出知识——
> 模型知不知道「1+1 等于几」是预训练阶段决定的，而本项目默认模型只有 ~1.8M 参数、语料 ~4.7M 字，
> 与真实 LLM 差了若干个数量级。要答得对，得先把预训练做够。
>
> 这不是猜测：基座自己的验证集 loss 从 step 4000 起就卡在 **6.2（ppl ≈ 500）**不再下降，
> 4000→9600 步几乎原地踏步——它的"流利"主要来自把训练语料背下来，换个领域（明清小说 → 现代汉语问答）
> 就没有可迁移能力。所以 SFT 后的句子虽然换成了助手的口吻（`当然可以帮…有什么？`、`您的账户`、
> `行李额` 都确实来自 SFT 语料里的 technical_support / banking / travel 三段对话），
> 但内容接不上问题本身。

---

### 6. `finetune` —— 加载预训练权重微调

```bash
cargo run --release -- finetune [参数]
```

从预训练 checkpoint 接着训练：加载权重后按指定的步数与学习率做**全参微调**。

> ⚠️ **LoRA 尚未接入训练循环**：`src/layers.rs` 的 `LoRA` / `inject_lora` 以及 `--lora-rank` / `--lora-alpha`
> 参数都已实现，但它们只会写进 `config.train.lora` 并打印一行提示，**不会冻结主参数、也不会注入适配层**。
> 也就是说当前 `finetune` 训练的是全部参数，可训练参数量并没有「降到 1% 以下」。
> LoRA 的原理与完整实现见第 29 课。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径 |
| `--pretrained <路径>` | string | 必填 | 预训练模型 checkpoint |
| `--lora-rank <秩>` | int | `16` | LoRA 秩（低秩维度，通常 4-64） |
| `--lora-alpha <系数>` | float | `16.0` | LoRA 缩放因子 α（通常 = rank） |
| `--steps <步数>` | int | `1000` | 微调步数 |
| `--lr <学习率>` | float | `1e-4` | 微调学习率 |

**示例**：

```bash
# ── 基础微调 ──
cargo run --release -- finetune --config config/config.json --pretrained checkpoints/best.ckpt

# ── 自定义步数与学习率 ──
cargo run --release -- finetune --config config/config.json --pretrained checkpoints/best.ckpt \
    --steps 2000 --lr 5e-5
```

---

### 7. `preset` —— 生成预设配置

```bash
cargo run --release -- preset [参数]
```

生成预设的模型配置文件，适合快速开始训练不同规模的模型。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--name <名称>` | string | `small` | 预设名称（small / medium / large） |
| `--output <路径>` | string | `config/config.json` | 输出配置文件路径 |

**预设说明**：

| 预设 | 参数量 | 架构 | 适合场景 |
|------|--------|------|----------|
| `small` | ~2M | GPT-2 风格（4层，256维） | 学习/演示，CPU 几分钟 |
| `medium` | ~15M | LLaMA 风格（8层，512维，GQA） | 中等语料，推荐 GPU |
| `large` | ~85M | LLaMA 风格（12层，768维，GQA） | 较大语料，需要 GPU |

**示例**：

```bash
# ── 生成小模型配置 ──
cargo run --release -- preset --name small --output config/config_small.json

# ── 生成中等模型配置（LLaMA 风格）──
cargo run --release -- preset --name medium --output config/config_medium.json

# ── 生成大模型配置 ──
cargo run --release -- preset --name large --output config/config_large.json

# ── 然后用生成的配置训练 ──
cargo run --release -- train --config config/config_medium.json
```

---

### 8. `demo` —— 教学演示

```bash
cargo run --release -- demo
```

无参数。依次运行 3 个教学演示（启用 `--features gpu` 时追加第 4 个 GPU 演示）：

1. **MLP 学习 XOR**（第 7 课）：验证神经网络 + 反向传播正确，训练后正确率 4/4（100%）
2. **BPE 分词器**（第 8 课）：在示例语料上训练 BPE 词表（400 个 token），演示编码/解码往返
3. **训练小 GPT 并生成文本**（第 12-20、25 课）：669 字符英文故事上训练 600 步，每 100 步记录一次
   （loss `1.63 → 0.15`），然后用 temperature=0.8 / top-k=10 / top-p=0.9 做三次生成：
   生成 1 全量前向、生成 2 同 prompt + 同种子的 KV cache（输出应是生成 1 的前缀，用于验证
   cache 不改生成分布）、生成 3 换个开头看小模型的泛化毛病
4. **GPU 加速对比**（第 27 课，仅 `--features gpu`）：验证 CPU vs GPU 数值一致性，实测加速比

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

### 9. `bench` —— 性能基准

```bash
cargo run --release -- bench [参数]
```

用**固定、可复现、短时**的任务测训练与推理吞吐（tok/s），专用于「优化改动前后」的同机对比，
不必每次都跑完整训练。模型结构、语料、随机种子全部写死，只受下面两个参数影响。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--steps <步数>` | int | `10` | 训练步数（越大越稳、越慢） |
| `--gen-tokens <数量>` | int | `64` | 生成 token 数（受 `block_size` 上限约束） |

**固定任务**：

- 模型：`n_layer=2 n_embd=128 n_head=4 block_size=64`，词表来自内置小语料（字符级，约 35 个 token）
- 训练：`batch_size=4`，走真实训练路径（前向 + 反向 + 梯度裁剪 + AdamW + 学习率调度）
- 推理：分别测 **KV cache** 与 **全量前向** 两种模式的生成吞吐

**抗噪处理**（否则数字没法比）：

- 推理先**预热一次**（排除线程池、首次内存分配的影响），再重复多次取**最短**耗时
- 训练用固定种子，`eval_every` 设到步数之外，避免评估干扰计时

**输出示例**：

```
=== 性能基准（bench）===
rayon 线程数：8
模型：n_layer=2 n_embd=128 n_head=4 block=64 vocab=35 | 参数 401280
开始训练：char（vocab=35）模型参数 401280 | 语料 669 tokens（训练 669 / 验证 0）| batch=4 block=64
step    30 | lr 0.000060 | loss 2.6326 | 1597 tok/s
[bench] train     : 30 steps | 4.823s | 0.1608s/step | 1592 tok/s
[bench] infer/kv  : 47 tok | 0.0388s | 1210.0 tok/s
[bench] infer/full: 24 tok | 0.1404s | 170.9 tok/s
（以上 tok/s 越高越好；优化前后同机对比即可看出收益）
```

> 同一 seed 下每步 loss 与训练步数完全一致（上例 step 30 的 loss 恒为 `2.6326`），
> 因此数值正确性可以直接用 loss 对比来验证。

**用法建议**：

```bash
# ── 快速冒烟（默认 10 步）──
cargo run --release -- bench

# ── 稳定对比（30 步，建议连跑 3 次取最优）──
cargo run --release -- bench --steps 30
cargo run --release -- bench --steps 30
cargo run --release -- bench --steps 30
```

> ⚠️ 同一二进制连跑也可能有 ±20% 波动（CPU 频率/后台负载），
> **务必跑多次取最优值**再下结论，别用单次结果判断优化是否有效。

---

### 10. `cargo test` —— 单元测试

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

默认构建运行 **35 个单元测试**（零外部依赖，秒级完成）；加 `--features gpu` 再跑 9 个 GPU 一致性 / 标定测试，
合计 44 个（其中 2 个是 `#[ignore]` 的性能探针，需手动运行）：

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
| `test_flash_attention_matches_standard` | Flash Attention vs 标准注意力（前向） |
| `test_flash_attention_backward_matches_standard` | Flash Attention 反向 vs 标准注意力反向 |
| `test_flash_attention_resident_path_matches_loop_reference` | 注意力常驻显存路径 vs 逐算子参考 |
| `test_dropout_p0_identity` / `test_dropout_eval_identity` | Dropout 正确性 |
| `test_dropout_masks_differ_across_calls` | Dropout 每次调用的掩码不同（种子确实在推进） |
| `test_rotary` / `test_rotary_grad_exact` | RoPE 正交性 + 梯度精确验证 |
| `test_kv_cache_matches_full_forward` | KV cache 推理 vs 全量前向一致性 |
| `test_kv_cache_generate_matches_full` | 同 prompt + 同种子下，KV cache 生成 vs 全量生成逐 token 一致（窗口内） |
| `test_kv_cache_output_is_prefix_of_full_beyond_window` | 超出缓存窗口时 KV cache 提前结束，输出仍是全量输出的前缀 |
| `test_char_tokenizer_roundtrip` / `test_bpe_roundtrip` | 分词器编码/解码往返 |
| `test_linear_regression_converges` | 线性回归收敛 |
| `test_repetition_penalty_suppresses_recent_token` | 重复惩罚确实压低最近出现过的 token（正负 logit 都验证） |
| `test_save_load_roundtrip_is_bit_exact` | checkpoint 保存/恢复后参数逐位一致 |
| `test_load_params_skips_optimizer_state` | 只加载参数时跳过优化器状态（`eval` / `generate` 路径） |
| `test_non_finite_values_survive_roundtrip` | NaN / ±Inf 能原样保存并读回（二进制格式的收益） |
| `test_rng_deterministic` / `test_rng_range` / `test_choice_range` | 随机数生成器 |

`--features gpu` 额外 9 个（都在 `src/gpu.rs`）：

| 测试 | 验证内容 |
|------|---------|
| `gpu_matmul_matches_cpu` | 14 组形状（含 3D 批量、非 tile 整数倍、跨 tile 边界、三种转置组合）vs CPU 三重循环 |
| `gpu_masked_softmax_matches_cpu` | 4 组 `(rows, d)` vs CPU，能暴露「归约数组复用少插一道 barrier」的竞态 |
| `gpu_lm_head_ce_matches_loop_reference` | 输出头融合 CE 的前向 loss + 反向 `d_hidden` / `d_weight` vs 纯循环参考 |
| `gpu_mlp_resident_matches_loop_reference` | 前馈常驻链路（含 dropout）前向输出 + 7 项梯度（第 2 组跨 256 归约边界） |
| `gpu_attn_layer_resident_matches_loop_reference` | 注意力常驻链路前向 + 11 项边界梯度 |
| `gpu_stack_resident_matches_sublayer_reference` | 整叠常驻路径前向 + `dx` + 每层 16 项参数梯度 vs 逐子层常驻路径 |
| `gpu_matmul_small_tile_matches_big_tile_bits` | 4 形状 × 4 转置组合下，64×64 与 128×128 两个 matmul 内核输出**逐位相同** |
| `matmul_throughput_probe`（`#[ignore]`） | 单次提交内连跑同一 matmul，测**纯内核**吞吐（排除回读造成的假象） |
| `mm_tile_ab_probe`（`#[ignore]`） | 同进程内按「大 → 小」交替各 3 轮，标定两种 tile 的 ms 与 GFLOP/s |

两个标定探针要手动跑（它们不是通过/失败型测试，而是打印数据）：

```bash
cargo test --release --features gpu matmul_throughput_probe -- --ignored --nocapture
cargo test --release --features gpu mm_tile_ab_probe -- --ignored --nocapture
```

---

### 11. GPU 加速（可选 feature）

默认构建**不启用 GPU**，保持依赖轻量。通过 `--features gpu` 开启 wgpu 计算着色器加速：

```bash
# ── 所有子命令都支持 --features gpu ──

# GPU 加速训练
cargo run --release --features gpu -- train --config config/config.json

# GPU 加速训练 + 断点续训
cargo run --release --features gpu -- train --config config/config.json --resume checkpoints/latest.ckpt

# GPU 加速评估
cargo run --release --features gpu -- eval --config config/config.json

# GPU 加速生成
cargo run --release --features gpu -- generate --config config/config.json --prompt "Once upon a" --max-new 200

# GPU 加速生成 + 自定义采样参数
cargo run --release --features gpu -- generate --config config/config.json --prompt "The fox" --temperature 0.5 --top-k 20 --seed 7 --max-new 150

# GPU 加速演示（含 CPU vs GPU 正确性验证 + 性能对比）
cargo run --release --features gpu -- demo

# ── Debug 模式 GPU（编译快，适合开发调试）──
cargo run --features gpu -- demo
```

**支持的 GPU**：NVIDIA 独显、Intel 核显（Windows 走 DX12 / Vulkan，无需额外驱动）

**加速范围**：GPU 后端有两条执行路径，粒度不同。

**① 逐算子路径（per-op）** —— 一个算子一次提交、一次回读：

- 批量矩阵乘（workgroup 8×8 = 64 线程、输出 tile 64×64、每线程 8×8 寄存器累加器）
- 逐元素 scale / add / ReLU、掩码 softmax 正反向

代价是**每个算子**都要付一次固定开销。本机（MX150）实测采样 50 次逐算子回读：
绑定+编码+提交 `0.8ms(6%)`、**poll+回读 `13.0ms(94%)`**（回读带宽约 646 MB/s）。
所以这条路只适合大形状，小矩阵自动回退 CPU（阈值默认 5000 万 FLOPs）。

**② 常驻 / 批量路径（resident）** —— 一个**子层**录进同一个 command encoder，最后只提交一次：

| 链路 | 入口 | 覆盖范围 | 回注的边界梯度 |
|------|------|---------|--------------|
| 输出头融合 CE | `gpu::lm_head_ce` | matmul → log_softmax → CE → dlogits | 2 项（d_hidden / d_weight） |
| 注意力子层 | `gpu::attn_layer_forward` | ln1 → QKV → RoPE → attn → c_proj → dropout → 残差 | 11 项 |
| 前馈子层 | `gpu::mlp_forward` | ln2 → linear1 → GELU → linear2 → dropout → 残差 | 7 项 |
| 整叠 Block | `gpu::stack_forward` | 上述两条链路 × n_layer，子层边界也留在显存 | 每层 16 项 |

走常驻路径时中间张量**全部留在显存不回读**，反向由 `Tensor::accumulate_grad` 把边界梯度注回计算图。
效果最直观的是输出头：logits 有 4096×8192 = 3355 万个数，逐算子版一步来回 **268MB（实测 355ms）**，
融合后只回读每行一个 f32 的 row_loss（约 16KB）。

常驻路径的**适用条件**（任一不满足就自动回退逐算子 → CPU）：训练模式（无 KV cache、`base = 0`）、
LayerNorm（不支持 RMSNorm）、GELU MLP（不支持 SwiGLU）、`n_kv_head == n_head`（不支持 GQA 头复制）、
形状与规模够大。整叠路径默认**关闭**，用 `LLM_GPU_STACK` 打开——它与逐子层路径数值逐位一致，
实测还**更快**（同配置 ABBA 两轮：0.59/0.60 s/步 vs 0.67/0.70 s/步，约快 12%）；默认关闭只是因为它的
准入条件更严（要求所有层都是 LayerNorm + GELU MLP，任一层不满足就整条路径放弃）。
详见 [docs/27-GPU加速.md](docs/27-GPU加速.md) 的 §4 与 §8。

**环境变量**：

| 变量 | 作用 | 默认 |
|------|------|------|
| `LLM_GPU_MATMUL_MIN_FLOPS` | 覆盖分流阈值（FLOPs 低于它走 CPU） | `50000000` |
| `LLM_GPU_MM_SMALL` | `0` = 强制 128×128 大 tile matmul 内核；其他值 = 64×64 小 tile | 小 tile（见 [标定](#gpu-matmul-内核从-128128-改到-6464)） |
| `LLM_GPU_STACK` | 存在即启用整叠 Block 常驻路径 | 关 |
| `LLM_GPU_PROBE` | 存在即录制第一步的全部 matmul 形状并逐形状单独回放，打印每形状吞吐 + 纯 FMA 峰值 | 关 |
| `LLM_GPU_ABLATE` | 逗号分隔算子类别名（`heads_split,heads_join,col_sum`），命中的类别只计数不提交，用步时差反推真实耗时 | 空 |
| `LLM_GPU_ABLATE_MM` | 逗号分隔形状谓词（`k<=32`、`n>=8192`、`m<=128`），命中的 matmul 只计数不提交 | 空 |

**GPU vs CPU 选择指南**：

| 模型规模 | n_embd | 推荐 | 原因 |
|----------|--------|------|------|
| small（~2M） | 256 | **GPU** | FLOPs 阈值已调低至 5000 万，QKV/MLP 投影走 GPU |
| medium（~15M） | 512 | **GPU** | 矩阵更大，GPU 加速明显 |
| large（~85M） | 768 | **GPU** | 矩阵够大，GPU 充分利用 |
| xlarge（~300M） | 1024+ | **GPU** | 矩阵足够大，GPU 加速显著 |

**分流策略**：FLOPs < 5000 万的矩阵走 CPU（如注意力头内积），其余走 GPU。训练结束时打印 `matmul 分流：GPU X / CPU Y`

**运行时诊断**：训练第一步结束后打印一行 `[gpu] dispatch 分解…`（把每个算子类别的录制/提交/同步耗时按批数归一），
结束时打印 `[gpu] 稳态（末尾 20 批）…`。这些是性能分析用的，不影响训练结果。

**自动回退**：GPU 初始化失败或任何调用出错时，自动回退 CPU，不影响正确性

---

## 配置文件完整参考（`config/config.json`）

配置文件分为 `model`（模型超参）和 `train`（训练参数）两个段。缺省字段自动取默认值。

### model 段 —— 模型超参数

```jsonc
{
  "model": {
    "vocab_size": 0,       // 词表大小。0 = 由分词器决定（训练时自动填入）
    "n_embd": 64,          // 隐藏维度（越大模型越强，显存和计算量越大）
    "n_head": 4,           // 注意力头数（Q 的头数）
    "n_layer": 2,          // Transformer 层数（越深越强）
    "block_size": 32,      // 最大上下文长度（能处理的最长序列）
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
    "steps": 1000,            // 总训练步数
    "max_lr": 3e-3,           // 峰值学习率（warmup 后达到）
    "min_lr": 3e-4,           // cosine 衰减的最低学习率
    "warmup_steps": 20,       // 线性预热步数（从 0 线性升到 max_lr）
    "weight_decay": 0.01,     // AdamW 权重衰减系数
    "grad_clip": 1000000.0,   // 梯度裁剪阈值（梯度总范数超过此值时等比缩放；默认极大值 = 实际上不裁剪）
    "eval_every": 100,        // 每 N 步评估一次验证集（同时保存 latest checkpoint）
    "eval_iters": 20,         // 评估时采样的批数（取平均减少方差）
    "tokenizer": "bpe",       // 分词器类型："char"（字符级）或 "bpe"（字节对编码）
    "bpe_vocab": 512,         // BPE 目标词表大小（= 256 字节 + 合并数）
    "train_file": "data/alice.txt", // 训练语料文件路径（支持目录路径，自动合并 .txt 文件）
    "val_file": null,         // 验证语料文件路径。null = 自动从训练文本末尾切 10%
    "out_dir": "checkpoints", // checkpoint 输出目录（权重 + tokenizer.json）
    "accum_steps": 1,         // 梯度累积步数。有效 batch = batch_size × accum_steps
    "tokenizer_file": null,   // 分词器文件路径。null = 从语料训练并保存；Some = 从文件加载
    "lora": null,             // LoRA 配置（目前未接入训练循环，见第 29 课）
    "log_file": "logs/train.csv", // 训练指标日志。null = 不记录；默认 logs/train.csv（自动建目录）
    "early_stop_patience": 0  // 早停耐心值。0 = 不启用；N = 连续 N 次评估不改善则停止
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
| `grad_clip` | float | `1000000.0` | 梯度裁剪。所有参数梯度的 L2 范数超过此值时等比缩放（默认值极大 ≈ 实际不裁剪，各预设配置会显式指定） |
| `eval_every` | int | `100` | 每 N 步在验证集上评估 loss / 困惑度，并保存 `latest.ckpt` |
| `eval_iters` | int | `20` | 评估时采样多少批取平均（减少随机波动） |
| `tokenizer` | string | `"bpe"` | `"char"` = 字符级分词；`"bpe"` = 字节对编码 |
| `bpe_vocab` | int | `512` | BPE 词表大小。仅当 `tokenizer = "bpe"` 时生效 |
| `train_file` | string | `"data/sample.txt"` | 训练语料文件路径（纯文本或目录路径）。**代码默认值指向的 `data/sample.txt` 已不在仓库中，请显式指定 `data/alice.txt` 或 `data/corpus/` 目录**（`config/config.json` 已配好） |
| `val_file` | string/null | `null` | 验证语料文件。`null` = 自动从训练文本末尾切约 10% |
| `out_dir` | string | `"checkpoints"` | 权重输出目录：checkpoint（latest / best / final）与 `tokenizer.json` 都写在这里。目录不存在时自动创建 |
| `accum_steps` | int | `1` | 梯度累积步数。有效 batch = `batch_size × accum_steps` |
| `tokenizer_file` | string/null | `null` | 分词器文件路径。`null` = 从语料训练并自动保存；指定路径 = 直接加载 |
| `lora` | object/null | `null` | LoRA 配置 `{ "rank": 16, "alpha": 16.0 }`。**目前只被校验并打印提示，尚未冻结主参数 / 注入适配层**（详见第 29 课） |
| `log_file` | string/null | `"logs/train.csv"` | 训练指标日志文件路径。默认 `logs/train.csv`（`logs/` 目录自动创建）；`null` = 不记录；指定路径 = **每个评估点**（每 `eval_every` 步 + 最后一步）写一行 CSV，列为 `step,lr,train_loss,val_loss,ppl,tokens_per_sec`。无验证集时 `val_loss` / `ppl` 两列留空。**每次训练覆盖该文件**，不是追加 |
| `early_stop_patience` | int | `0` | 早停耐心值。`0` = 不启用；`N` = 验证 loss 连续 N 次评估不改善就提前停止（停止前仍会保存 checkpoint 与日志） |

### 完整配置示例

```json
{
  "model": {
    "vocab_size": 0,
    "n_embd": 256,
    "n_head": 8,
    "n_layer": 4,
    "block_size": 256,
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
├── Cargo.toml          # 依赖：serde / serde_json / clap / rayon / windows-sys + 可选 wgpu / pollster
├── README.md           # 本文件
├── config/             # 配置文件目录
│   └── config.json     #   默认训练配置（模型超参 + 训练参数）
├── checkpoints/        # 权重目录（自动创建）：latest.ckpt / best.ckpt / final.ckpt / tokenizer.json
│   ├── perf/           #   各实验按 out_dir 分成子目录，如 checkpoints/zh、checkpoints/probe_b2
│   └── ...
├── logs/               # 日志目录（自动创建）：运行日志（每次 train/eval/generate/chat/finetune 各一份）
│                       #      与训练指标 CSV（train.csv，由 train.log_file 指定）
├── data/               # 语料：alice.txt（公版《爱丽丝梦游仙境》）、corpus/（中英文混合语料，含文章/代码/对话/新闻/诗歌）
│                       #      corpus_zh/（《红楼梦》《三国演义》等中文名著）、corpus_perf/（性能测试用节选）
├── src/
│   ├── main.rs         # CLI 入口：train / eval / generate / chat / finetune / preset / demo / bench
│   ├── cli.rs          # 命令行定义（clap）
│   ├── config.rs       # 配置加载（serde）+ 目录约定常量（config/、checkpoints/、logs/）
│   ├── runlog.rs       # 运行日志：每次训练 / 推理自动写 logs/{操作}_{时间戳}.log（命令行 + 完整配置 + 过程输出）
│   ├── checkpoint.rs   # checkpoint 保存 / 恢复
│   ├── attention.rs    # 多头注意力 + KV Cache（第 9-10、23、25 课）
│   ├── autograd.rs     # 自动微分：backward + 拓扑排序（第 2 课）
│   ├── tensor.rs       # 张量运算（第 1、3-4 课）
│   ├── gpu.rs          # GPU 计算后端（第 27 课，--features gpu）：WGSL 计算着色器
│   ├── rope.rs         # RoPE 旋转位置编码（第 20 课）
│   ├── rng.rs          # 随机数（第 5 课）
│   ├── layers.rs       # 网络层（第 5、11、19、21-22、29 课）
│   ├── loss.rs         # 损失函数（第 6 课）
│   ├── optim.rs        # 优化器（第 6、17 课）
│   ├── module.rs       # 参数管理 trait（第 5 课）
│   ├── tokenizer.rs    # 分词器（第 8 课）
│   ├── model.rs        # GPT 模型：Transformer Block + 前向（第 9-12、19 课）
│   ├── data.rs         # 数据集（第 14 课）
│   ├── train.rs        # 训练循环、学习率调度、梯度累积、早停、CSV 日志（第 13、18、28 课）
│   └── sample.rs       # 推理与采样（第 15、30 课）
└── docs/               # 39 课教程文档（00-学习计划 + 01~39 各课）
```

### 产物目录约定（自动创建，无需手动 mkdir）

| 产物 | 默认位置 | 由谁决定 | 说明 |
|------|----------|----------|------|
| 配置文件 | `config/config.json` | `--config` / `--output` | 所有子命令的配置默认路径；`preset --output` 写同类路径 |
| 权重 | `checkpoints/` | `train.out_dir` | `latest.ckpt` / `best.ckpt` / `final.ckpt` 与 `tokenizer.json` |
| 训练指标日志 | `logs/train.csv` | `train.log_file` | CSV：`step,lr,train_loss,val_loss,ppl,tokens_per_sec` |
| 运行日志 | `logs/{操作}_{时间戳}.log` | 程序自动生成 | 每次 `train` / `finetune` / `eval` / `generate` / `chat` 各写一份，文件名含操作名与毫秒级本地时间；内容 = 完整命令行 + 完整配置 + 该次运行的全部输出 |

实现方式：`src/config.rs` 提供 `ensure_parent_dir()` / `ensure_dir()`，并在**每个写盘出口**调用——
`Config::save()`（配置）、`checkpoint::save()`（权重）、`Tokenizer::save()`（分词器）、`MetricsLogger::new()`（指标 CSV）、
`runlog::start()`（运行日志）。因此把 `out_dir`、`log_file` 改成任意嵌套路径（如 `outputs/run1/logs/train.csv`）
也会自动建好目录，产物不会再散落到仓库根目录。

> Windows 上 PowerShell 的 `*>` 可以把控制台输出重定向到文件；但运行日志已由程序自动写入 `logs/`，
> 含完整命令与配置，通常不需要再手动重定向。

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
> | 六、训练进阶与正则化 | 17-19 | AdamW、学习率调度、Dropout | ✅ |
> | 七、现代 LLM 架构 | 20-24 | RoPE、RMSNorm、SwiGLU、GQA、Flash Attention | ✅ |
> | 八、工程优化 | 25-30 | KV Cache、混合精度、GPU、梯度累积、LoRA、Beam Search | ✅ |
> | 九、前沿技术 | 31-38 | Scaling Laws、MoE、量化、推测解码、RLHF、RAG、分布式 | 📖 |
> | 十、工程化完善 | 39 | CLI 工程、分词器持久化、预设配置、微调工作流、交互式对话 | ✅ |

## 代码验证状态

- **48 个单元测试全部通过**（`cargo test`）；加 `--features gpu` 再跑 9 个 GPU 测试，合计 57 个（详见 [§10 `cargo test`](#10-cargo-test--单元测试)）
- 已知提示（`never used` 警告，不影响功能）：
  - `cargo build`（非 gpu）报 3 条：`layers.rs` 的 `ln_params`、`gelu_weights`，`tensor.rs` 的 `external` / `mul` / `neg` / `sum_last_dim`
  - `cargo build --features gpu` 报 2 条：`gpu.rs` 的 `GpuDispatchDiag.n`，`tensor.rs` 的 `mul` / `neg` / `sum_last_dim`
  - 这些是给自动微分 / GPU 对照测试留的算子，非测试构建下未被调用；`cargo test` 构建里 `mul` / `sum_last_dim` 会被测试用到，只剩 `external` / `neg`
- 测试覆盖：自动微分、广播、softmax、BPE 编解码、RoPE 正交性与梯度、KV cache 与全量前向一致性、
  RMSNorm/SwiGLU 融合算子与分步实现一致性、Flash Attention 与标准注意力一致性、Dropout、线性回归收敛
- **Demo 端到端验证通过**（`cargo run --release -- demo`）：XOR 100%、BPE 往返、GPT 训练 loss 1.63→0.15、文本生成正常
- **工程化功能已全部集成**：微调（`finetune`，当前为**全参微调**，LoRA 只做参数校验与提示）、
  Beam Search（`generate --beam`）、交互式对话（`chat`）、分词器持久化（`tokenizer.json`）、
  预设配置（`preset`）、训练指标日志（CSV）、运行日志（`logs/{操作}_{时间戳}.log`）、早停（`early_stop_patience`）

## 性能优化与基准测试

### 怎么测：`bench` 子命令

性能优化最大的坑是「感觉快了」——所以本项目内置了 `bench`，把测量固化下来：

```bash
cargo run --release -- bench --steps 30
```

固定小模型 + 固定语料 + 固定种子，输出训练 / 推理吞吐（tok/s）。两个关键设计：

1. **抗噪**：推理先预热一次，再重复多次取**最短**耗时；训练把 `eval_every` 设到步数之外，不让评估干扰计时。
2. **可校验正确性**：固定种子下每步 loss 完全确定（如 step 30 恒为 `2.6326`），
   所以「性能是否提升」看 tok/s，「数值是否被改坏」看 loss —— 一跑就有答案，不用等长训练。

> 测出来的数字有 ±20% 波动（CPU 频率、后台负载），**务必跑 3 次取最优**再下结论。

### 优化一：CPU matmul 改为「按输出行并行 + cache 友好循环顺序」

**根因**：`matmul_data`（`src/tensor.rs`）原本只按 `batch` 维度并行，
但 `Linear` 会把 3D 输入展平成 2D 再调用（此时 `batch = 1`）——
于是注意力 QKV 投影、MLP、输出头这些**最重的矩阵乘只拿到 1 个并行任务**，
实际退化成单线程三重循环，多核完全用不上。

**改动**：把 `(batch × m)` 个输出行拉平后交给 rayon（行与行之间无依赖）；
循环顺序从 `i-j-k` 改为 `i-k-j`（axpy 累加），使 `b` 的一行、`out` 的一行都是**连续**访问，
对缓存和自动向量化友好（原顺序里 `b[k*n+j]` 每步跨 `n` 个元素，命中率极差）。

每个输出元素仍在 `k` 上按**相同顺序**累加，因此浮点结果与旧实现**逐位一致**，测试无需放宽容差。

### 优化二：推理路径引入 `no_grad`，不再构建计算图

**根因**：算子按 `requires_grad` 决定是否挂 `parents` / `backward` 闭包，而模型参数恒为 `true`，
导致**每次推理前向都在分配一堆永远不会执行的反向闭包**。单 token 前向时计算量极小，
建图开销甚至超过矩阵乘本身。

**改动**：加入 thread-local 开关与 `Tensor::req()`（`src/tensor.rs`），
所有「要不要建图」的判断改走 `req()`（no_grad 模式下恒为 false）；
`generate` / `beam_search` / `eval_loss` 的前向包进 `no_grad`。
语义与 PyTorch 的 `torch.no_grad()` 一致，用 RAII 守卫恢复状态。

**收益**：推理 1.9~3.9×，且**训练路径完全不受影响**（已验证）。

### 实测结果

同机（8 线程 CPU）用 `bench` 默认参数，优化前后各自多次运行取较优值：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 训练 | ~440 tok/s | ~1600 tok/s | **约 3.6×** |
| 推理（KV cache） | ~591 tok/s | ~1129 tok/s | **约 1.9×** |
| 推理（全量前向） | ~43 tok/s | ~165 tok/s | **约 3.9×** |

**正确性**：48 个单元测试全部通过；同一 seed 下 loss 与优化前完全一致；`demo` 端到端正常。
（该组数据是优化当时的同机 A/B，绝对值有 ±20% 噪声；当前可复现的对照点见 [§9 `bench`](#9-bench--性能基准) 与 [§11 GPU 章节](#11-gpu-加速可选-feature)。）

### 还能压的地方

- `Tensor::flash_attention` 已改为矩阵乘内核（见 [第 24 课](docs/24-Flash-Attention.md)），
  但**没有**分块与在线 softmax，显存仍是 O(T²)：长上下文时 P / dP 会完整占显存
- GPU 注意力常驻链路的 `col_sum` / 共享内存树形归约与 CPU 求和顺序不同，
  只做到相对误差 < 1e-3，未与 CPU 逐位对齐（其余 GPU 路径都是逐位一致）
- 整叠常驻路径（`LLM_GPU_STACK=1`）虽然比逐子层常驻快约 12%，但默认关闭：
  它的准入条件更严（要求所有层都是 LayerNorm + GELU MLP，任一层不满足就整条放弃），
  且整叠中间量同时驻留显存，显存峰值更高
- `KVCache::append` 已改为在 `Vec<f32>` 上就地 `extend`（不再每步重拼整段历史），
  但 `k()` / `v()` 每步仍会克隆一次整段缓存（打分算子需要一个拥有所有权的 `Tensor`），可改为借用视图
- Beam Search 的打分目前累加原始 logit（非 log_softmax），与「对数概率和」的严格语义有偏差

### GPU matmul 内核：从 128×128 改到 64×64

**根因**：原内核每个 workgroup 是 16×16 = 256 线程、输出 tile 128×128，每线程约 100 个寄存器。
一块 SM 只塞得下 2 个这样的工作组，于是每次 `workgroupBarrier` 和每次全局 load 的等待都**无处躲藏**。
改成 8×8 = 64 线程、tile 64×64（每线程仍是 8×8 = 64 个命名标量累加器）后，同一张 SM 能并存多得多的工作组，
用一个组的访存去盖另一个组的等待。内层循环逐字未改，只把共享内存 stride 从 32 降到 16。

**方法**：GPU 连续满载会降频，两次独立运行的结果能差 15%，所以标定必须**同进程内交替测量**
（`mm_tile_ab_probe`），而不是跑两次二进制再比。

训练真实形状上逐个体测（大 tile → 小 tile 交替，各 3 轮）：

| 形状 | 大 tile | 小 tile | 小/大 |
|------|---------|---------|-------|
| PV fwd 512×512×32 b=32 | 20.91ms / 103 GF/s | **11.78ms / 182 GF/s** | 0.56 |
| dV bwd 512×512×32 b=32 | 15.79 / 136 | **9.05 / 237** | 0.57 |
| QKᵀ fwd 512×32×512 b=32 | 20.35 / 106 | **12.16 / 177** | 0.60 |
| QKV/c_proj fwd 4096×128×128 | 7.86 / 137 | **5.60 / 192** | 0.71 |
| dW proj bwd 128×4096×128 | 6.77 / 159 | 6.78 / 158 | 1.00 |
| MLP w2 fwd 4096×512×128 | 8.89 / 242 | **7.04 / 305** | 0.79 |
| dX proj bwd 4096×128×128 | 8.40 / 128 | **6.24 / 172** | 0.74 |
| lm_head fwd 4096×128×8192 | 30.78 / 279 | **26.17 / 328** | 0.85 |

**结论**：小 tile 在**每一个**形状上都不输、多数快一截——原本「大 tile 共享内存复用率更高，
只该在 n 很小或 workgroup 数太少时才换小的」的假设被数据否掉了：连 n = 8192、完全没有 tile 浪费的形状也快 18%。

端到端（同一二进制，用 `LLM_GPU_MM_SMALL` 切换，`n_embd=128 / 4 层 / block=512 / batch=8 / dropout=0`，各跑一次）：

| 内核 | 稳态 ms/次提交（末尾 20 批） | 训练步时 tok/s | 总耗时 / 25 步 |
|------|---------------------------|--------------|---------------|
| 大 tile（`LLM_GPU_MM_SMALL=0`） | 26.44 / 29.00 | 6532 / 6814 | ~19s |
| 小 tile（默认） | 23.43 / 25.13 | 7357 / 7631 | 16~18s |

即端到端小 tile 约快 **9%**（两个配置的 loss 完全相同，都是 `8.9662`）。
注意这两次是**跨进程**测量，会被 GPU 降频影响（本机可差 15%），
所以 tile 的取舍结论以上面**同进程交替**测出的内核级对照为准，端到端数字只作旁证。

正确性用 `gpu_matmul_small_tile_matches_big_tile_bits` 保证：4 种形状 × 4 种转置组合下
两内核输出**逐位相同**（每个输出元素都按 k = 0,1,2,… 顺序累加，浮点加法不满足结合律，顺序一变就对不上）。
用随机数据而非常数也是刻意的：常数在每步都精确无舍入，测不出求和顺序写错。

**踩坑：按形状做消融实验有失效边界**。`LLM_GPU_ABLATE_MM=<形状谓词>` 能摘掉某个形状的 matmul 看步时差，
但这只对「输出不进入后续依赖链」的形状成立——消融掉 m = 128 的权重梯度 matmul 后参数永远得不到更新，
loss 直接变 NaN，整轮实验作废。

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
- **梯度缓冲不能按需分配**：做推理 `no_grad` 优化时，曾顺手把「不求导的张量就不分配 grad 缓冲」也加上，
  结果 `test_masked_softmax_matches_chain` 直接越界 panic —— 反向闭包可能写入**不需要梯度**的父节点
  （`masked_softmax` 的 mask 就不参与求导，但闭包仍会往它的 grad 里累加）。
  结论：`grad` 缓冲必须始终按 `data` 全长分配，想省这块只能改闭包的写入逻辑，不能改分配策略。
- **checkpoint 别用 JSON 存张量**：参数早就是 f32 二进制块了，但优化器动量 `m`/`v` 一开始是当 JSON 头里的
  两个字段存的——一个 f32 要 13~14 字节（二进制 4 字节），184 万参数的模型存档 58.6MB 里 87% 是这么浪费的。
  更糟的是 JSON 没有 `NaN`/`Infinity` 字面量，`serde_json` 会写成 `null`，读回直接反序列化失败——训练一发散
  存档就报废。现在只把标量和元信息留在 JSON 头（且 `best_val_loss` 用 `Option` 表达"未产生"），
  参数 / `m` / `v` 一律按 f32 小端写二进制块。
- **推理建图是纯浪费**：模型参数的 `requires_grad` 恒为 `true`，若算子只用它决定是否建图，
  推理时也会一路把整张计算图（含 backward 闭包）建出来。加一个全局 `no_grad` 开关、
  把判断改走 `Tensor::req()` 后，推理提升 1.9~3.9×。
- **SFT 的 loss 掩码要右移一位对齐**：窗口里 `y[i] = tokens[start+1+i]`，所以第 `i` 个位置的掩码
  要看**目标 token 所在的位置** `sup[start+1+i]`，而不是输入位置 `sup[start+i]`。少移这一位，
  整个批次的监督信号会整体错开一个 token——loss 照降、形状全对，极难从结果看出来。
- **SFT 模板标记只用语料里出现过的字**：`#` 在 4.7M 字的中文语料里只出现 2 次，它的 embedding
  基本没被训过，模板一带上 `### ` 就把模型推进 ASCII/数字乱码模式（实测基座输出
  `'何人？」621.-----2..3E8188`）。模板用 `用户：` / `助手：` / `（结束）` 而不是 Markdown 风格标记；
  `【】` 同理不能用（也只出现 2 次）。加特殊 token 也能解，但会给已有 checkpoint 带来
  "embedding 行数对不上"的兼容问题，字节级 BPE 本来就能编码任意 UTF-8 字符串，不必动词表。
- **SFT 的说话人角色是文件级属性**：人名标签（`陈教授：` / `面试官：`）没有固定含义，按"该文件里
  谁先说"判定；**跨文件共用一张表会把后面文件的角色弄反**，所以加载时要保留文件边界
  （`load_texts` 而不是拼成一份）。人名标签只在整份文件 ≥90% 非空行是角色行、且说话人只有两个时
  才启用——否则小说正文（`秦琼道：……`）会被当成对话喂进来。

## 后续方向

以下方向已在第 31-38 课教程文档中详细讲解，代码实现可作为进阶练习：

- **Scaling Laws**（第 31 课）：指导训练资源分配的幂律公式，Chinchilla 最优配比
- **MoE 混合专家模型**（第 32 课）：Router 门控 + 多专家稀疏激活，Mixtral / DeepSeek 架构
- **量化部署**（第 33 课）：INT4/INT8 量化、GPTQ/AWQ，让大模型跑在消费级显卡上
- **推测解码**（第 34 课）：小模型猜 + 大模型验，无损加速推理 2-4×
- **多 Token 预测**（第 35 课）：同时预测未来 K 个 token，提升表征质量
- **人类对齐**（第 36 课）：RLHF (PPO) / DPO / GRPO，让模型"有用、无害、诚实"
- **RAG 检索增强生成**（第 37 课）：向量检索 + LLM 生成，解决知识截止和幻觉问题
- **分布式训练**（第 38 课）：数据并行、ZeRO、张量/流水线并行，训练百亿级模型
- 长度外推：RoPE 配合 NTK-aware scaling、YaRN 等技巧（见第 20 课文档）
- 更大的语料与模型规模（`config/config.json` 可直接调大，CPU 训练需耐心）
