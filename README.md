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
- **工程化完整**：CLI 子命令（train / eval / generate / chat / sft / finetune / preset / demo / bench / scaling）、
  外部语料、train/val 划分、验证集评估与困惑度、checkpoint 保存/恢复、断点续训。
- **性能可量化**：内置 `bench` 基准子命令，用固定小模型在秒级内测出训练/推理吞吐（tok/s），
  优化改动前后可同机对比（详见 [性能优化与基准测试](#性能优化与基准测试)）。
- **现代 LLM 技术栈**：RoPE、RMSNorm、SwiGLU、GQA、Flash Attention、梯度累积、Beam Search、
  KV Cache 滑动窗口（长上下文可持续生成）、混合精度训练（AMP 动态损失缩放，已接入训练循环）。
- **真实可用**：加载预训练权重微调、交互式对话、分词器持久化、训练指标日志、运行日志（每次训练/推理自动存档）、预设模型配置。
- **LoRA 已接入**：`finetune` 子命令会冻结预训练主干、只训练低秩适配层（缺省挂 Q/K/V，`--lora-targets` 可扩到 O 与 MLP）
  （本项目实测可训练参数 **24,576 / 1,866,496 = 1.32%**），支持**链式续训**（`--resume-lora`，接着训旧适配层）
  与**推理合并**（`--merge-lora`，把增量就地并进主干），存档头部记录 LoRA 形态，加载后可直接对话，
  见 [§6](#6-finetune--lora-微调) 与第 29 课。
- **混合精度训练（AMP）已接入训练循环**：`train.amp` 打开后，loss 先乘上动态 `scale` 再反向，
  参数更新前检查梯度是否溢出（含 Inf/NaN 就丢弃本步、不更新参数，`scale` 自动减半）、并把梯度
  反缩放回真实尺度再做裁剪（保证裁剪阈值仍然作用在真实梯度上），见第 26 课。
- **透明度高**：训练过程中每一步的中间结果、梯度、损失都可以直接打印检查。

### 包含的功能（对应 39 课）

| 模块 | 文件 | 内容 |
|------|------|------|
| 张量运算 | `src/tensor.rs` | Tensor 结构体、广播、逐元素/标量运算、matmul、softmax、permute、gather；**`matmul_frozen`**（冻结权重：反向只求 `dx`） |
| 自动微分 | `src/autograd.rs` | backward 反向传播、拓扑排序（计算图 → 梯度流） |
| 模块接口 | `src/module.rs` | `Module` trait：参数收集的统一接口（`parameters()` / **`trainable_parameters()`**） |
| RoPE 位置编码 | `src/rope.rs` | 旋转位置编码：把相对位置揉进 Q/K 向量 |
| 神经网络层 | `src/layers.rs` | Linear、LayerNorm、**RMSNorm**、Embedding、ReLU/GELU/Tanh、**SwiGLU**、**LoRA 适配层（`LoraAdapter`，挂在 `Linear` 上）** |
| 损失与优化器 | `src/loss.rs` `src/optim.rs` | MSE、CrossEntropy、SGD、AdamW（动量 + 权重衰减）；**`step()` 跳过冻结参数（含权重衰减）** |
| 分词器 | `src/tokenizer.rs` | 字符级分词 + BPE（字节对编码），**save/load 持久化**，配置可切换；生成时做 UTF-8 约束，不会拼出乱码字符 |
| 注意力机制 | `src/attention.rs` | 多头自注意力、因果掩码、RoPE、**KV Cache（含滑动窗口丢弃）**、**GQA 分组查询注意力**、**Q/K/V 注入 LoRA** |
| GPT 模型 | `src/model.rs` | Transformer Block 堆叠、GPT 整体前向、checkpoint 参数名、**Dropout**、**`apply_lora`（冻结 + 注入）** |
| 数据加载 | `src/data.rs` | 外部文本文件、**目录批量加载**、train/val 划分、随机 batch 采样；**SFT 对话语料解析 + loss 掩码** |
| 训练与评估 | `src/train.rs` | 训练循环、梯度裁剪、warmup+cosine 学习率、验证集 loss / 困惑度、**梯度累积**、早停、**CSV 指标日志**、**SFT 掩码透传**、**按可训练子集统计梯度范数**、**`MixedPrecision` 动态损失缩放（AMP：溢出跳步 + 梯度反缩放）** |
| 采样 | `src/sample.rs` | temperature / top-k / top-p 采样 + 重复惩罚，KV cache 推理，**Beam Search**，**停止标记** |
| 缩放定律 | `src/scaling.rs` | 幂律拟合（固定 `b` 的闭式最小二乘 + 黄金分割搜 `b`）、`C ≈ 6ND` 与非嵌入参数口径、Chinchilla 20:1 与参数化闭式解两条最优配比、训练时长/电费估算、**真实跑多规模扫描并拟合实测指数** |
| MoE 稀疏专家 | `src/moe.rs` | Top-K 路由（并列按下标、确定性）、两种门控口径（Top-K 重归一化 / Switch 原概率，**含 K = 1 的梯度陷阱**）、gather→expert→weighted→scatter 稀疏前向、负载均衡辅助损失、容量因子与 Token Dropping、参数/激活量口径 |
| 量化 | `src/quant.rs` | INT8/INT4 的逐张量 / 逐通道 / 逐 token 量化与位打包、GPTQ（Hessian 逆 + 逐通道误差分摊）、AWQ（激活感知的缩放搜索）、校准集统计、逐层量化与 checkpoint 元信息、KIVI 式 KV cache 量化 |
| 推测解码 | `src/speculative.rs` | 草稿→验证循环（拒绝采样 + 残差分布修正，输出**严格无损**）、`TargetStream` 缓存不变量与 `rollback_to`、`ModelDrafter`、多 Token 预测头（`MtpHeads` / `MtpDrafter`） |
| 对齐 | `src/align.rs` | 奖励模型（标量头 + Bradley-Terry）、序列 logprob 与 loss 掩码、DPO（隐式奖励 + 参考模型）、GRPO 组内相对优势、PPO clip 目标 + KL(k3) 惩罚 |
| RAG | `src/rag.rs` | 分块（字符域切分 + 句读对齐、可配重叠）、三种向量化（TF-IDF / FNV 哈希 / 模型隐状态池化）、余弦检索与 MMR 多样化重排、按预算组装提示 |
| 分布式 | `src/distributed.rs` | 环形 allreduce（reduce-scatter + all-gather）、数据并行、ZeRO-1/2（状态分片）、张量并行 MLP 与 QKV 列切分、GPipe / 1F1B 流水线、3D 并行规划（`DistConfig`） |
| 配置 | `src/config.rs` | `config/config.json`：模型超参 + 训练参数 + **预设配置**（small/medium/large）+ **LoRA 配置** + **SFT 语料** |
| Checkpoint | `src/checkpoint.rs` | 模型参数 + 优化器状态保存/恢复（latest / best / final），`LLMCP2` 二进制格式，**头部记录 LoRA 形态（旧档兼容）** |
| 命令行 | `src/cli.rs` | clap 子命令：train / eval / generate / **chat** / **sft** / **finetune** / **preset** / demo / **bench** / **scaling** / **moe** / **quant** / **distributed** / **align** / **rag** / **speculative** |
| 随机数 | `src/rng.rs` | 自实现 xorshift64 伪随机数发生器 |
| GPU 加速 | `src/gpu.rs` | 可选（`--features gpu`）：wgpu 计算着色器加速 matmul/scale/add/relu，失败自动回退 CPU |

### 前沿技术（第 31-38 课）

| 主题 | 教程文档 | 内容 |
|------|---------|------|
| Scaling Laws | `docs/31-Scaling-Laws.md` | 幂律关系、Chinchilla 最优配比、算力估算、涌现能力（**已落地代码**：`src/scaling.rs` + `scaling` 子命令） |
| MoE 混合专家模型 | `docs/32-MoE混合专家模型.md` | 稀疏激活、Router 门控网络、负载均衡、Switch/Mixtral/DeepSeek 架构（**已落地代码**：`src/moe.rs` + `moe` 子命令） |
| 量化技术 | `docs/33-量化技术.md` | INT8/INT4 量化、GPTQ、AWQ、GGUF、PTQ vs QAT、STE（**已落地代码**：`src/quant.rs` + `quant` 子命令） |
| 推测解码 | `docs/34-推测解码.md` | 草稿模型 + 验证、拒绝采样、无损保证、Medusa/EAGLE（**已落地代码**：`src/speculative.rs` + `speculative` 子命令） |
| 多 Token 预测 | `docs/35-多token预测.md` | MTP 训练目标、DeepSeek 实现、与推测解码结合（**已落地代码**：`src/speculative.rs` 的 `MtpHeads` / `MtpDrafter`） |
| RLHF 与对齐 | `docs/36-RLHF与对齐.md` | SFT、奖励模型（Bradley-Terry）、PPO、DPO、GRPO、Constitutional AI（**已落地代码**：`src/align.rs` + `align` 子命令） |
| RAG 检索增强生成 | `docs/37-RAG检索增强生成.md` | 文档分块、向量嵌入、相似度检索、重排序、HyDE、Self-RAG（**已落地代码**：`src/rag.rs` + `rag` 子命令） |
| 分布式训练 | `docs/38-分布式训练.md` | 数据并行、ZeRO、张量并行、流水线并行、3D 并行、通信原语（**已落地代码**：`src/distributed.rs` + `distributed` 子命令） |

### 工程化完善教程（第 39 课，代码+文档）

| 主题 | 教程文档 | 内容 |
|------|---------|------|
| 工程化完善 | `docs/39-工程化完善.md` | 9 个 CLI 子命令、分词器序列化、预设配置、微调工作流、SFT 监督微调、交互式对话、Beam Search CLI、多文件数据加载、CSV 指标日志 |

## 快速开始

需要 **Rust 2024 edition** 工具链（Rust 1.85+，建议使用最新的 stable）。

```bash
# ═══════════════════════════════════════════
#  最简方式：训练 + 生成（推理不需要语料）
# ═══════════════════════════════════════════
cargo run --release -- train --config config/config.json
# 权重写到 config.train.out_dir，config/config.json 里配的是 checkpoints/zh
# 训练完成后，推理只需 checkpoint，分词器自动加载
cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "Once upon a" --max-new 100

cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "The" --max-new 200

# ═══════════════════════════════════════════
#  使用预设配置（推荐）
# ═══════════════════════════════════════════
# 生成中等模型配置（LLaMA 风格，~26M 参数）
cargo run --release -- preset --name medium --output config/config_medium.json
cargo run --release -- train --config config/config_medium.json

# ═══════════════════════════════════════════
#  交互式对话（训练后直接对话，无需语料）
# ═══════════════════════════════════════════
cargo run --release -- chat --ckpt checkpoints/zh/best.ckpt

# ═══════════════════════════════════════════
#  监督微调（把只会续写的预训练模型教会"应答"）
# ═══════════════════════════════════════════
cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt
cargo run --release -- chat --ckpt checkpoints/zh-sft/best.ckpt   # 之后用 SFT 权重对话

# ═══════════════════════════════════════════
#  LoRA 微调（冻结预训练主干，只训低秩适配层，只占 1.3% 参数）
# ═══════════════════════════════════════════
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt --lora-rank 8 --lora-alpha 8
cargo run --release -- chat --ckpt checkpoints/zh-lora/best.ckpt --temperature 0.3
#  扩挂载（q/k/v/o/mlp/all）、链式续训、推理时合并增量
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt --lora-targets all
cargo run --release -- finetune --pretrained checkpoints/zh-lora/best.ckpt --resume-lora --steps 60
cargo run --release -- chat --ckpt checkpoints/zh-lora/best.ckpt --merge-lora

# ═══════════════════════════════════════════
#  端到端演示（验证所有算法正确性）
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

程序提供 16 个子命令：`train` / `eval` / `generate` / `chat` / `sft` / `finetune` / `preset` / `demo` / `bench` /
`scaling` / `moe` / `quant` / `distributed` / `align` / `rag` / `speculative`。

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

**运行日志**：`train` / `eval` / `generate` / `chat` / `sft` / `finetune` / `scaling` / `moe` / `quant` / `distributed` / `align` / `rag` / `speculative` 十三个子命令**每次运行都会自动在 `logs/` 下写一份运行日志**，文件名是 `{操作}_{年-月-日_时-分-秒-毫秒}.log`（如 `logs/generate_2026-09-19_14-30-12-345.log`），操作名区分命令、毫秒时间戳区分同命令的多次运行，互不覆盖。每份日志包含：运行头部（操作名、开始时间（命令开始执行的时刻）、**完整命令行**、工作目录、版本 / 平台 / 线程数 / GPU）、**完整配置**（`--config` 解析后的全部字段，含被 CLI 覆盖后的最终值）、本次运行的关键参数（采样参数 / prompt / checkpoint 等）与全部过程输出（训练进度、评估点、生成文本、对话轮次），结尾附结束时间与总耗时。文件名时间戳、开始时间、总耗时同源，都取自 `main()` 入口记下的时刻，所以耗时覆盖参数解析、配置与模型加载在内的**全过程**。写入由 `src/runlog.rs` 统一负责，控制台与日志内容一致，不需要再手动重定向。

**分词器自动加载；`eval` 仍需要语料**：`eval` / `generate` / `chat` 都会从 checkpoint 目录自动加载 `tokenizer.json`（不必再指定分词器）。其中 **`generate` / `chat` 只依赖 checkpoint，不需要语料**；但 **`eval` 要算验证集 loss，仍会读配置里的 `train_file`**（或 `val_file`），所以它的 `--config` 必须指向语料还在的原配置：

```bash
# 训练
cargo run --release -- train --config config/config.json
# 推理（只需 checkpoint，分词器自动加载）
cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "Once upon a" --max-new 100
cargo run --release -- chat --ckpt checkpoints/zh/best.ckpt
# 评估（还需语料：从默认 config/config.json 的 train_file 读）
cargo run --release -- eval --ckpt checkpoints/zh/best.ckpt
```

**分词器加载优先级**：
1. `--tokenizer` 参数（命令行显式指定）
2. `config/config.json` 的 `tokenizer_file` 字段
3. `{out_dir}/tokenizer.json`（训练时自动保存的，推荐）
4. 从 `train_file` 语料训练（兜底，不推荐）

**示例**：

```bash
# ── 基础训练 ──
# 用默认 config/config.json 训练（BPE 词表 8192、4000 步、batch=8、out_dir=checkpoints/zh）
cargo run --release -- train --config config/config.json

# ── 断点续训 ──
# 从最近的 checkpoint 继续（恢复参数、优化器状态、步数）
cargo run --release -- train --config config/config.json --resume checkpoints/zh/latest.ckpt

# 从最优 checkpoint 续训（继续微调）
cargo run --release -- train --config config/config.json --resume checkpoints/zh/best.ckpt

# ── GPU 加速训练 ──
# 开启 wgpu 计算着色器（NVIDIA / Intel 核显），失败自动回退 CPU
cargo run --release --features gpu -- train --config config/config.json

# GPU 加速 + 断点续训
cargo run --release --features gpu -- train --config config/config.json --resume checkpoints/zh/latest.ckpt

# ── 不同模型规模的训练（修改 config/config.json，或用 preset 生成）──
# 小模型（秒级完成，适合快速验证）：n_embd=64, n_layer=2, block_size=32（GPTConfig 的默认值）
# 中模型（几分钟）：n_embd=256, n_layer=4, block_size=256
# 大模型（需要耐心）：n_embd=512, n_layer=8, block_size=256
# 不想手改就直接用预设：preset small(256维/4层/128上下文) / medium(512维/8层) / large(768维/12层)

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
| `--merge-lora` | flag | 关 | 推理前把 LoRA 增量就地并进主干权重（数值不变，省掉每层两次小矩阵乘）；基座无适配层时只打一行警告 |

**评估指标**：
- `val_loss` —— 验证集上的交叉熵损失
- `perplexity`（困惑度）—— `e^val_loss`，越低越好（理想值接近 1）

**示例**：

```bash
# ── 最简评估（分词器自动加载，但语料仍要从配置里读）──
cargo run --release -- eval --ckpt checkpoints/zh/best.ckpt

# ── 基础评估 ──
# 不传 --ckpt 时用 {out_dir}/latest.ckpt（config/config.json 的 out_dir 是 checkpoints/zh）
cargo run --release -- eval --config config/config.json

# ── 评估不同 checkpoint ──
# 评估最优 checkpoint
cargo run --release -- eval --config config/config.json --ckpt checkpoints/zh/best.ckpt

# 评估最终 checkpoint
cargo run --release -- eval --config config/config.json --ckpt checkpoints/zh/final.ckpt

# 评估指定路径的 checkpoint
cargo run --release -- eval --config config/config.json --ckpt /path/to/my_model.ckpt

# ── 评估不同模型配置 ──
# 用不同的 config 评估（config 决定模型架构，必须与 checkpoint 训练时一致）。
# 仓库里只有 config/config.json，其余配置要先 `preset` 生成，例如：
cargo run --release -- preset --name large --output config/config_large.json
cargo run --release -- eval --config config/config_large.json --ckpt checkpoints/latest.ckpt

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
| `--merge-lora` | flag | 关 | 推理前把 LoRA 增量就地并进主干权重（数值不变，省掉每层两次小矩阵乘） |

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
cargo run --release -- generate --ckpt checkpoints/zh/best.ckpt --prompt "Alice was" --max-new 100

# ═══════════════════════════════════════════
#  基础生成
# ═══════════════════════════════════════════

# 用默认参数生成（temperature=0.8, top-k=40, top-p=0.9）
cargo run --release -- generate --config config/config.json --prompt "Alice was" --max-new 100

# 指定 checkpoint 生成
cargo run --release -- generate --config config/config.json --ckpt checkpoints/zh/best.ckpt --prompt "Once upon a" --max-new 200

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
#   本项目：  固定种子 → 每次输出相同 → 可精确对比不同参数/模型的效果（实验可复现的前提）
#

# ═══════════════════════════════════════════
#  KV Cache 控制（--no-kv-cache）
# ═══════════════════════════════════════════

# 默认开启 KV cache（推荐，推理速度快）
cargo run --release -- generate --config config/config.json --prompt "Once" --max-new 100

# 禁用 KV cache（每个 token 都全量前向，慢但省显存；总长不超 block_size 时结果与开 cache 完全一致）
cargo run --release -- generate --config config/config.json --prompt "Once" --max-new 100 --no-kv-cache

# ═══════════════════════════════════════════
#  使用不同 checkpoint 生成
# ═══════════════════════════════════════════

# 使用最新 checkpoint（默认）
cargo run --release -- generate --config config/config.json --prompt "The key"

# 使用验证 loss 最优的 checkpoint
cargo run --release -- generate --config config/config.json --ckpt checkpoints/zh/best.ckpt --prompt "The key" --max-new 100

# 使用训练结束时的 checkpoint
cargo run --release -- generate --config config/config.json --ckpt checkpoints/zh/final.ckpt --prompt "The key" --max-new 100

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
| `--merge-lora` | flag | 关 | 推理前把 LoRA 增量就地并进主干权重（数值不变，省掉每层两次小矩阵乘） |

**推理不需要语料**：训练时自动保存 `tokenizer.json` 到 checkpoint 目录，对话时自动加载。

**`--prompt-format`**：默认 `sft`，prompt 会被拼成训练时的模板形态（`用户：` / `助手：`，见 [§5](#5-sft--监督微调把续写变成应答)），
并在模型吐出 `。。` 或 `用户：` 时停下——这样它接的是"该我回答了"的位置，而且不会顺着模板继续编下一轮提问。
**未做过 SFT 的预训练权重请用 `--prompt-format raw`**：它没见过这套标记，套上模板只会更差。

**上下文预算**：`block_size` 是「system prompt + 对话历史 + 本轮生成」三者共用的窗口。分配优先级依次是：
system prompt 永远保留（它是序列开头的位置锚点），本轮生成预留 `--max-new` 个 token，剩下的额度给对话历史。
历史按真实 token 数裁剪，超出即从最老的轮次开始丢弃；窗口紧张时可调小 `--max-new` 换取更长记忆。

**示例**：

```bash
# ── 最简对话（只需 checkpoint，无需 config 和语料）──
cargo run --release -- chat --ckpt checkpoints/zh/best.ckpt

# ── 带系统提示的对话 ──
cargo run --release -- chat --ckpt checkpoints/zh/best.ckpt --system "You are a helpful assistant."

# ── 创意对话（高温采样）──
cargo run --release -- chat --ckpt checkpoints/zh/best.ckpt --temperature 1.0 --max-new 300
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

**学习率默认取预训练配置的 1/10，`min_lr` 跟着变成它的 1/10**：SFT 是在已收敛的权重上继续训，
用预训练那种步长会把预训练攒下的语言能力一起冲掉。显式传 `--lr` 时以你给的为准，
此时 `min_lr` 仍按 `max_lr × 0.1` 重算（cosine 衰减的终值跟着峰值走，不会出现 min > max）。

**评估间隔与预热步数都会按步数自动收窄**：配置里的 `eval_every`（`config/config.json` 里是 200）
和 `warmup_steps`（300）都是给预训练上万步用的。`eval_every` 直接套在几百步的 SFT 上会把整段训练
压成"只在最后评估一次"——`best.ckpt` 退化成 `final.ckpt`，早停也永远等不到第二次评估；
`warmup_steps` 太长则会让大部分步数都耗在爬坡上。所以实际取
`eval_every = min(eval_every, steps/10)`、`warmup_steps = min(warmup_steps, steps/10).max(1)`。

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
- 每段对话末尾的 `。。` 由程序自动补，**不用自己写**。

**模板标记要用预训练语料里"高频"的字，光"出现过"远远不够**（自定义模板时同样适用）。结束标记是
每段对话都要被监督的目标，模型必须有本事把它生成出来，所以判据是**出现频率**而不是"出现过"。
实测 `data/corpus_zh/`（471.9 万字）：

| 字符 | 次数 | 频率 |
|------|------|------|
| `。` | 131,594 | 2.79% |
| `：` | 64,400 | 1.36% |
| `结` | 204 | 0.0043% |
| `（` | 129 | 0.0027% |
| `）` | 130 | 0.0028% |
| `#` | 12 | 0.00025% |

`#` 只出现 12 次，它的 embedding 基本没被训过，模板一带上 `### ` 就把模型推进 ASCII 乱码模式
（实测基座模型输出 `'何人？」621.-----2..3E8188`）；`【】` 只出现 2 次，同理不能用。**结束标记
最早选的 `（结束）` 其实犯了同一个错**：四个字全在 0.002%~0.008% 量级，和 `#` 是一个数量级，
模型学到的只是"回答完吐一个 `（`"，之后全无先验，输出被 `（结…` 这类括号乱码淹没、停止标记也
形同虚设。现在改成 `。。`：`。` 是语料最高频字符（输出行训得最充分），而 `。。` 这个组合在
471.9 万字里只出现 **2 次**，不与正文撞车。所以模板是 `用户：` / `助手：` / `。。`，不是
`### 用户：` 这种 Markdown 风格标记。

**示例**：

```bash
# ── 基础 SFT（步数用 config 的，学习率用 config lr 的 1/10）──
cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt

# ── 指定语料与步数 ──
cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt --steps 300 --lr 4e-4

cargo run --release --features gpu -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt --steps 300 --lr 4e-4

cargo run --release -- sft --config config/config.json --pretrained checkpoints/zh/best.ckpt \
    --sft-file "data/corpus/zh_dialogue_*.txt,data/sft/" --steps 300 --lr 1e-4

# ── 训练完对话（--prompt-format 默认就是 sft）──
cargo run --release -- chat --ckpt checkpoints/zh-sft/best.ckpt --max-new 60
cargo run --release -- chat --ckpt checkpoints/zh-sft/final.ckpt --max-new 60
cargo run --release -- chat --ckpt checkpoints/zh-sft/final.ckpt --tokenizer checkpoints/zh-sft/tokenizer.json --max-new 60
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

### 6. `finetune` —— LoRA 微调

```bash
cargo run --release -- finetune [参数]
```

加载预训练权重后**冻结全部主干**，只训练挂在各投影上的低秩适配层（LoRA，默认 Q/K/V，可用 `--lora-targets` 扩到 O 与 MLP）。
语料与训练循环跟 `sft` 完全共用，只是参数集合不同——想对比"全参 SFT vs LoRA"，跑两条命令、看两份 CSV 即可。

**挂载位置**（`--lora-targets`）决定增量加在哪几个 `Linear` 上，写法与效果：

| 取值 | 含义 | 每层对数 |
|------|------|----------|
| `q,k,v`（缺省） | 只动注意力输入侧 | 3 |
| `q,k,v,o` | 加上输出投影 `c_proj` | 4 |
| `all` | 再加 MLP 的两个（或三个）线性层 | 6（SwiGLU 时 8） |

别名：`proj` / `c_proj` → `o`，`ffn` / `mlp` → `mlp`。大小写与空格随意，重复项会去重，未命中的词直接报错。
**结构写进 checkpoint 头部**，所以训练与推理必须一致；加载时按存档头部的记录重建，不需要再传一遍。

**冻结是三处一起做实的**（缺一处就会"以为冻结了、其实没冻"，详见第 29 课）：

| 位置 | 做法 |
|------|------|
| 前向 | 冻结权重走 `Tensor::matmul_frozen`，反向只求 `dx`、不算 `dW` |
| 优化器 | `AdamW` / `SGD` 的 `step()` 跳过 `!requires_grad()` 的参数，**连带不做权重衰减** |
| GPU 常驻快路 | LoRA 形态下 `attn_resident` / `mlp_resident` / `blocks_resident` 整体让路（它们绕过 `Linear::forward`，会把增量静默丢掉） |

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径 |
| `--pretrained <路径>` | string | 必填 | 预训练模型 checkpoint（主干来源） |
| `--lora-rank <秩>` | int | `16` | LoRA 秩 r（低秩维度，通常 4-64）；越小适配层越少 |
| `--lora-alpha <系数>` | float | `= rank` | LoRA 缩放因子 α（缺省 = rank，即增量不额外缩放） |
| `--lora-targets <位置>` | string | `q,k,v` | 适配层挂载位置，逗号分隔：`q` / `k` / `v` / `o` / `mlp` / `all`（见上表） |
| `--resume-lora` | flag | 关 | **链式续训**：基座是 LoRA 存档时接着训**存档里那套**适配层（保留已学增量），此时不能再传 rank / alpha / targets |
| `--sft-file <路径>` | string | config 的 `train.sft_file` | 对话语料，逗号分隔、可含 `*` 通配 |
| `--steps <步数>` | int | `1000` | 微调步数 |
| `--lr <学习率>` | float | `1e-4` | 微调学习率 |
| `--out-dir <目录>` | string | `{out_dir}-lora` | 输出目录，缺省加 `-lora` 后缀，绝不覆盖预训练权重 |

**示例**：

```bash
# ── 基础用法（缺省 rank=16 / alpha=16 / 挂 q,k,v / 1000 步）──
cargo run --release -- finetune --config config/config.json --pretrained checkpoints/zh/latest.ckpt

# ── 本项目实测的配方：rank=8、alpha=8、60 步、lr=5e-5 ──
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt \
    --lora-rank 8 --lora-alpha 8 --steps 60 --lr 5e-5

# ── 扩到 O 与 MLP（容量更大，参数仍只占几个百分点）──
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt --lora-targets all

# ── 用这个存档对话（目录是自包含的，不必再指回基座）──
cargo run --release -- chat --ckpt checkpoints/zh-lora/best.ckpt

# ── 链式续训：接着训存档里那套适配层，已学的增量不丢 ──
cargo run --release -- finetune --pretrained checkpoints/zh-lora/best.ckpt --resume-lora --steps 60

# ── 换语料 / 换输出目录 ──
cargo run --release -- finetune --pretrained checkpoints/zh/latest.ckpt \
    --sft-file data/sft/ --out-dir checkpoints/zh-lora2
```

**链式续训 vs 重挂一套**：把 LoRA 存档当基座时，缺省语义是**重挂一套全新适配层**（原增量丢弃，只继承主干），这时会打一行 `[warn]` 提示；
要接着训存档里那套就加 `--resume-lora`，它保留 A/B 的现有数值并重新解冻，rank / alpha / 挂载位置一律照存档，
再传 `--lora-rank` 之类会直接报错。**这两种语义互斥**，选哪个取决于你是想在新任务上重来，还是想在旧任务上接着磨。

**推理侧合并**：`eval` / `generate` / `chat` 都支持 `--merge-lora`，把 `ΔW = (α/r)·AᵀBᵀ` 就地并进主干权重
（`W ← W + ΔW`），前向于是省掉每层两次小矩阵乘。**数值不受影响**——同 seed 下合并前后逐字相同（已实测）。
注意合并会**丢弃**适配层本体，所以存档里的 A/B 不会被改写，`--merge-lora` 只作用于本次推理的内存副本；
但它是**不可逆的运行时操作**，合并后的模型对象不能再拿来续训。

**输出**（实测，[config/config.json](config/config.json) 的 1.84M 小模型 + 89 段对话语料）：

```text
LoRA 微调：rank=8 alpha=8 挂载=q,k,v steps=60 lr=0.00005｜冻结主干，只训适配层
模型参数：1866496（含冻结主干）｜可训练：24576（1.317%）｜12 对低秩矩阵（挂载 q,k,v），共 24 个适配参数张量
SFT 语料：89 段对话，打包 17898 token | 监督位置（回答段）占 56.1%
LoRA：rank=8 alpha=8 挂载=q,k,v｜可训练参数 24576 / 1866496（1.32%）｜主干冻结：反向不算 dW、优化器不更新、不吃权重衰减
...
step    54 | lr 0.000006 | loss 6.2499 | val 6.5596 (ppl 706.0) *
[done] checkpoint 已保存到 checkpoints/zh-lora/ | 总耗时 478s | 7.96s/步（共 60 步）
```

链式续训（`--resume-lora`，实测 5 步）的差别只在开头两行——结构照存档、命名换成"链式续训"：

```text
链式续训：沿用存档里的适配层（rank=8 alpha=8 挂载=q,k,v）steps=5 lr=0.0001｜已学的增量保留，只训 A/B
模型参数：1866496（含冻结主干）｜可训练：24576（1.317%）｜12 对低秩矩阵（挂载 q,k,v），共 24 个适配参数张量
```

推理侧合并（`--merge-lora`）会多打一行 `LoRA 合并：12 个适配层的增量已并入主干权重（前向不再有额外矩阵乘）`。

产物落在 `checkpoints/zh-lora/`：`{best,final,latest}.ckpt` + `tokenizer.json` + `lora.csv`
（目录自包含——**冻结的主干也写在档案里**，所以加载时不必再指定 `--pretrained` 之外的任何东西）。
实测档案只比基座大 296,186 字节，全部来自那 24,576 个适配参数的参数/动量块。

> ⚠️ 别把这个 SFT val 当成效果指标：语料只有 89 段、验证集是切出来的 9 段，数字没有判别力。
> 真正要看的是①预训练能力有没有退化（主干冻结 ⇒ 不会）②`chat` 聊两句有没有怪字。
> 另外本基座对温度敏感，对话建议加 `--temperature 0.3`。

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
| `small` | ~3.3M | GPT-2 风格（4层，256维，block=128，BPE 512） | 快速验证，CPU 几分钟 |
| `medium` | ~26M | LLaMA 风格（8层，512维，GQA，RMSNorm+SwiGLU） | 中等语料，推荐 GPU |
| `large` | ~79M | LLaMA 风格（12层，768维，GQA，RMSNorm+SwiGLU） | 较大语料，需要 GPU |

> 参数量口径：`词嵌入(vocab×d)` + `每层(注意力 Q/K/V/O + MLP + 归一化)` × 层数 + `ln_f`。
> 模型用 RoPE，**没有可学习的位置嵌入表**，所以不含 `block_size×d` 那一项。
> 词表大小由语料训练出的分词器决定（`vocab_size: 0`），实际参数会随语料略有浮动。
>
> 预设生成的配置沿用 `TrainConfig` 的默认值，`out_dir` 是 `checkpoints`（仓库自带的
> `config/config.json` 才被改成了 `checkpoints/zh`）；`train_file` 默认 `data/alice.txt`。

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

### 8. `demo` —— 端到端演示

```bash
cargo run --release -- demo
```

无参数。依次运行 3 个演示（启用 `--features gpu` 时追加第 4 个 GPU 演示）：

1. **MLP 学习 XOR**（第 7 课）：验证神经网络 + 反向传播正确，训练后正确率 4/4（100%）
2. **BPE 分词器**（第 8 课）：在示例语料上训练 BPE 词表（400 个 token），演示编码/解码往返
3. **训练小 GPT 并生成文本**（第 12-20、25 课）：669 字符英文故事上训练 600 步，每 100 步记录一次
   （loss `1.63 → 0.15`），然后用 temperature=0.8 / top-k=10 / top-p=0.9 做三次生成：
   生成 1 全量前向、生成 2 同 prompt + 同种子的 KV cache（滑动窗口让它同样生成满 80 个 token），
   再做一次**窗口内等价自检**（prompt + 生成不超 `block_size` 时两种模式必须逐 token 完全相等，
   这是"cache 不改生成分布"的验证）、生成 3 换个开头看小模型的泛化毛病
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

### 10. `scaling` —— Scaling Laws 实验（预算规划 + 实测幂律拟合）

```bash
cargo run --release -- scaling [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--config <路径>` | string | `config/config.json` | 配置文件路径（提供模型结构开关与语料路径） |
| `--sizes <规模>` | string | `2x64,4x128,6x192` | 参数量扫描的规模网格，`层数x宽度` 逗号分隔 |
| `--steps <步数>` | int | `600` | 每个规模训练多少步（固定 token 预算） |
| `--data-multiples <倍数>` | string | `1,2,4,8` | 数据量扫描的倍数列表 |
| `--batch-size <批>` | int | `8` | 扫描用批大小 |
| `--block-size <长度>` | int | `64` | 扫描用上下文长度 |
| `--lr <学习率>` | float | `3e-3` | 扫描用学习率 |
| `--seed <种子>` | int | `42` | 随机种子（所有扫描点共用，保证可比） |
| `--budget <FLOPs>` | float | `1e22` | 算力预算（用于最优配比与时长/电费估算） |
| `--gpu-tflops <TFLOPS>` | float | `312.0` | 单卡峰值（A100 FP16 = 312） |
| `--n-gpu <数量>` | int | `64` | 卡数 |
| `--mfu <利用率>` | float | `0.4` | MFU（算力利用率，常见 0.3~0.6） |
| `--out <路径>` | string | 无 | 把扫描点写成 CSV（可直接画图），不填则只打印 |

**输出分两半**：

1. **预算规划（纯解析，毫秒级）**：同一份算力下 20:1 法则与参数化损失闭式解两条路线的最优规模、
   训练时长/电费、Chinchilla 论文配比表、过训练/欠训练曲线；
2. **实测扫描（真训，分钟级）**：固定 token 预算放大模型（loss vs N）、固定模型放大数据
   （loss vs D），再用幂律拟合出**实测指数**，并逐点打印实测 loss 与拟合值的偏差。

> 实测出来的指数会明显大于论文值（$\alpha_N \approx 0.076$）——本项目规模比论文小 3~5 个数量级，
> 不在论文的幂律区间内。这里验证的是**方法**（口径统一、预算固定、可复现），不是复刻论文数字。
> 详见 [`docs/31-Scaling-Laws.md`](docs/31-Scaling-Laws.md)。

**用法建议**：

```bash
# ── 快速看一眼（规模小、步数少）──
cargo run --release -- scaling --steps 200 --sizes 2x64,4x128,6x192 --data-multiples 1,2,4

# ── 完整默认实验（三个规模各 600 步 + 数据量 1/2/4/8 倍）──
cargo run --release -- scaling

# ── 只做预算规划（规模/倍数压到最小，几十秒出结果；两组都要 ≥3 个点才能拟合）──
cargo run --release -- scaling --sizes 2x64,3x96,4x128 --steps 30 --data-multiples 1,2,4

# ── 换硬件假设（H100 ×1024、MFU 45%）并导出 CSV ──
cargo run --release -- scaling --budget 1e23 --gpu-tflops 989 --n-gpu 1024 --mfu 0.45 --out logs/scaling.csv
```

---

### 11. `moe` —— MoE 稀疏专家实验（第 32 课）

```bash
cargo run --release -- moe [参数]
```

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--experts <网格>` | string | `8` | 专家数网格（逗号分隔，逐个跑一遍对照实验） |
| `--top-k <K>` | int | `2` | 每个 token 激活的专家数 |
| `--steps <步数>` | int | `300` | 端到端对照的训练步数 |
| `--batch-size <批>` | int | `8` | 每步批大小 |
| `--block-size <长度>` | int | `64` | 上下文长度 |
| `--lr <学习率>` | float | `3e-3` | 学习率 |
| `--n-embd <维度>` | int | `64` | 隐藏维度（专家与路由器的宽度） |
| `--n-layer <层数>` | int | `2` | Transformer 层数 |
| `--aux-coef <α>` | float | `0.01` | 均衡辅助损失系数（对照组固定为 0） |
| `--capacity-factors <列表>` | string | `0,1.0,1.25,2.0` | 容量因子扫描（0 = 不限容量） |
| `--seed <种子>` | int | `42` | 随机种子（各组共用，保证初始权重一致、可比） |

**输出分四节**：

1. **参数 / 计算量口径**：`total = E·expert + router`（全驻显存）、`active = K·expert + router`（每 token 只算 K 个），
   并用 `assert_eq!` 把公式与真实建层参数对账（公式漂移会当场 panic 而不是给出错数字）；
2. **负载均衡辅助损失：隔离实验**：人为把路由器摆到塌缩点，然后**不跑主损失、只优化 `L_aux`**；
3. **端到端对照**：同种子 / 同语料 / 同步数，α = 0 与 α = `aux_coef` 各训一遍，比对 loss 与路由不均衡度；
4. **容量因子与 Token Dropping**：拿负载最不均的那份模型扫容量因子，看丢弃比例。

实测输出（`cargo run --release -- moe --steps 60`）：

```text
=== 一、参数 / 计算量口径：MoE 省的是 FLOPs，不是显存 ===
  E=8   K=2 | 单层 总 265224 / 激活 66696（3.98× / 省 74.9% FLOPs） | 模型 总 566608 / 激活 169552（3.34× / 省 70.1% FLOPs）

=== 二、负载均衡辅助损失：隔离实验（不跑主损失，只优化 L_aux） ===
  汇总：K=1：L_aux 5.932 → 1.039（p 摊平 ⇒ 1.0，不是下界），不均衡度 8.00 → 6.32，用到的专家 1/8 → 4/8
       ｜K=2：L_aux 3.114 → 1.010（p 摊平 ⇒ 1.0，不是下界），不均衡度 4.00 → 3.92，用到的专家 2/8 → 6/8

=== 三、端到端对照（同种子 / 同语料 / 同 60 步） ===
  汇总：α=0: loss 1.9459 / 不均衡度 1.48｜α=0.01: loss 1.9771 / 不均衡度 1.26

=== 四、容量因子与 Token Dropping ===
  cf = 0     → 每专家容量 不限         丢弃 0/32768 = 0.00% ｜读到 token 的专家 8/8
  cf = 1     → 每专家容量 128        丢弃 8171/32768 = 24.94% ｜读到 token 的专家 8/8
  cf = 1.25  → 每专家容量 160        丢弃 5317/32768 = 16.23% ｜读到 token 的专家 8/8
  cf = 2     → 每专家容量 256        丢弃 862/32768 = 2.63% ｜读到 token 的专家 8/8
```

**两个容易踩的坑**（详见 [`docs/32-MoE混合专家模型.md`](docs/32-MoE混合专家模型.md)）：

- **K = 1 配错门控口径 ⇒ 路由器拿不到主损失梯度**。Top-K 内部**重归一化**（Mixtral / DeepSeek 式）在
  K = 1 时权重恒等于 1，对 logits 的雅可比整体是 0。K = 1 必须配 `switch_gate`（Switch Transformer 式
  原概率）。`moe` 子命令与单测 `test_k1_renorm_gate_has_no_router_gradient` 都验证了这一点。
- **「`L_aux` 掉到 1」不是负载均衡的证书**。`L_aux = E·Σ f_i·p_i` 里 `f` 由 argmax 给出（不可导），
  `p` 才是可导的；`p` 均匀时无论 `f` 长什么样都有 `L_aux = 1`。所以第二节的隔离实验里
  `L_aux` 被压到 1.0 附近，**硬路由几乎没动**（K = 2 组 4.00 → 3.92）。1 也不是下界：
  `f` 与 `p` 支撑集不交时 `L_aux` 可以是 0。

> **端到端对照的诚实读法**：这个规模（2 层 × 64 维、几十步）下 α = 0 那一份**不会**塌缩——
> 随机初始化的路由器在几百步内大体保持对称，路由塌缩是「富者愈富」的**长期**动力学，
> 所以第二节才改用隔离实验。但塌缩的**代价**在任何规模下都一样：主损失不关心是谁在算，
> 未被选中的专家拿不到任何梯度（等于白占显存），而 loss 曲线看不出异常。

---

### 12. `quant` / `distributed` / `align` / `rag` / `speculative` —— 第 33~38 课实验

第 33~38 课各带一个子命令，把该课的算法跑成**带断言自检**的实验：输出的是实测数字与结论，
参数配得不合法（如 `--gamma` 相对 `--block-size` 过大）会当场 panic，而不是给出一份看起来正常的结果。

```bash
cargo run --release -- quant [参数]        # 第 33 课：权重量化（RTN / GPTQ / AWQ）
cargo run --release -- distributed [参数]  # 第 38 课：集合通信 + DP / ZeRO / TP / PP / 3D
cargo run --release -- align [参数]        # 第 36 课：奖励模型 / DPO / GRPO / PPO
cargo run --release -- rag [参数]          # 第 37 课：分块 / 向量化 / 检索 / 提示组装
cargo run --release -- speculative [参数]  # 第 34/35 课：推测解码 + 多 Token 预测
```

| 子命令 | 关键参数 | 输出分节 |
|--------|---------|---------|
| `quant` | `--bits int8\|int4`、`--method rtn\|gptq\|awq`、`--calib-file` / `--calib-samples` / `--calib-tokens`、`--act-order`、`--damp`、`--alpha`、`--eval`、`--out` | 逐层量化误差报告（各算法对照）→（`--eval`）量化前后验证集 loss / 困惑度 → 量化权重落盘 |
| `distributed` | `--dp` `--tp` `--pp`（三轴相乘 = 卡数）、`--micro-batches`、`--steps`、`--batch-size`、`--lr`、`--weight-decay` | 集合通信量对照 → DP / ZeRO-1/2 训练轨迹与状态分片 → TP 数值一致 → PP（GPipe / 1F1B）→ 3D 切分报告 |
| `align` | `--rm-steps`、`--steps`、`--beta`、`--clip-eps`、`--kl-coef`、`--group-size`、`--rm-lr` / `--lr` | 奖励模型排序准确率 → DPO 偏好边界 → GRPO 组内优势 → PPO 裁剪分支 + KL(k3) 惩罚 |
| `rag` | `--chunk-size` / `--overlap`、`--top-k`、`--mmr-lambda` / `--mmr-pool`、`--hash-dim`、`--context-chars`、`--query` | 分块不变量 → 三种向量化对照 → top-k 与 MMR 检索 → 提示组装（含预算截断） |
| `speculative` | `--gamma`、`--mtp-heads` / `--mtp-steps`、`--trials`、`--temperature`、`--block-size` | 无损性核对（与逐 token 解码逐位对照）→ 接受率与 γ 取舍 → 缓存不变量 → 首 token 分布等价 → MTP 头当草稿 |

> `quant` 需要一个已训练好的 checkpoint（缺省读 `out_dir/latest.ckpt`，可 `--ckpt` 指定）；
> 其余四个自带小模型与内置语料，直接跑即可。

---

### 13. `cargo test` —— 单元测试

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

默认构建运行 **200 个单元测试**（零外部依赖；全量约 14 分钟，开发中按名字过滤跑单模块通常只要几秒）；
加 `--features gpu` 再跑 10 个 GPU 一致性 / 标定测试，
合计 210 个（其中 2 个是 `#[ignore]` 的性能探针，需手动运行）。GPU 用例会真实创建 wgpu 设备并逐个形状比对
CPU 参考实现，单个用例就要几十秒，建议加 `-- --test-threads=1` 串行跑：并行跑多个 GPU 用例会互相抢设备，
曾观察到随机失败。

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
| `test_kv_cache_sliding_window_drops_oldest_rows` | 滑动窗口只丢最旧的行：`seq_len` 封顶在 `window`，绝对位置计数继续累加（RoPE 基准靠它） |
| `test_kv_cache_sliding_window_matches_full_window_forward` | 超窗后 KV cache 增量推理与全量窗口前向的 logits 数值吻合（单层模型下两者严格等价） |
| `test_kv_cache_generate_beyond_window_is_not_truncated` | 缓存写满后不再提前结束：加/不加 `--no-kv-cache` 生成同样多的 token |
| `test_amp_loss_scaling_is_numerically_transparent` | 开/关 AMP 的最终 loss 完全相同（scale 是 2 的幂，f32 下缩放精确） |
| `test_mixed_precision_grows_shrinks_and_unscales` | AMP：无溢出按间隔翻倍、溢出减半并要求跳过本步、梯度反缩放回真实尺度 |
| `test_char_tokenizer_roundtrip` / `test_bpe_roundtrip` | 分词器编码/解码往返 |
| `test_utf8_pending_accepts_only_legal_prefix` / `test_decode_drops_incomplete_tail` | 字节级 BPE 的 UTF-8 约束：只接受合法的字节前缀、解码时丢掉不完整尾部 |
| `test_linear_regression_converges` | 线性回归收敛 |
| `test_repetition_penalty_suppresses_recent_token` | 重复惩罚确实压低最近出现过的 token（正负 logit 都验证） |
| `test_utf8_mask_blocks_half_char_tokens` / `test_pending_tail_detects_incomplete_char` | 采样时的 UTF-8 约束：屏蔽"半个汉字"的 token、识别未收尾的字符 |
| `test_save_load_roundtrip_is_bit_exact` | checkpoint 保存/恢复后参数逐位一致 |
| `test_load_params_skips_optimizer_state` | 只加载参数时跳过优化器状态（`eval` / `generate` 路径） |
| `test_non_finite_values_survive_roundtrip` | NaN / ±Inf 能原样保存并读回（二进制格式的收益） |
| `test_lora_checkpoint_roundtrip` | LoRA 存档：头部记下 rank/alpha/targets、按名对齐逐位还原；**不注入适配层就加载必须报错** |
| `test_header_without_lora_field_is_accepted` | LoRA 之前存的旧档（无 `lora` 字段）仍能读入 |
| `test_lora_targets_parse` / `test_lora_config_serde_compatibility` | `--lora-targets` 解析（别名 / 大小写 / 去重 / 报错）；旧档无 `targets` 回落 `q,k,v` |
| `test_matmul_frozen` | 冻结权重的矩阵乘：前向与 `matmul` 一致、反向只回传 `dx`、`w.grad()` 全 0 |
| `test_lora_forward_matches_formula` / `test_lora_forward_3d` | 适配层前向 = 主干 + (α/r)·(x@Aᵀ)@Bᵀ；α 只以 α/r 进入；2D / 3D 输入都对 |
| `test_lora_zero_init_keeps_forward_identical` | B 初始为 0 ⇒ 挂上适配器前后前向**逐位相同** |
| `test_frozen_weight_gets_no_grad_while_lora_does` | 单独冻结主干时 `weight.grad()` 全 0；主干+适配器时只有 A/B 拿到梯度；优化器带权重衰减也拉不动冻结主干 |
| `test_apply_lora_freezes_backbone_and_trains_adapters_only` | 走完整训练步（前向→backward→AdamW）：可训练集合恰好是 12 个 `lora_a`/`lora_b`、主干逐位不变、适配层确实在动 |
| `test_lora_targets_control_where_adapters_land` | `--lora-targets` 真生效：只开 `v,mlp` 时 Q/K/O 上没有适配参数，张量数 = `(1+MLP线性层数) × 2 × n_layer` |
| `test_merge_lora_into_weight_is_in_place` | 合并改的是**原 `weight` 张量的数据**（旧句柄能读到新值），合并后适配器被摘除，前向不变 |
| `test_merge_lora_matches_two_branch_forward` | 灌非零 A/B 后，合并前（双分支）与合并后（单分支）前向偏差 < 1e-4 |
| `test_resume_lora_keeps_adapter_values_and_unfreezes` | 链式续训：A/B 数值原样保留（不重随机）、只有适配层被解冻、主干仍冻结 |
| `test_rng_deterministic` / `test_rng_range` / `test_choice_range` | 随机数生成器 |
| `parse_accepts_both_prefix_styles` / `parse_keeps_multiline_answer_and_drops_incomplete` / `parse_skips_text_without_role_markers` | SFT 对话解析：两种角色前缀、多行回答与残缺段、无标记文本跳过 |
| `parse_accepts_named_speakers_in_dialogue_corpus` / `parse_rejects_named_speakers_in_prose` / `parse_resets_speakers_per_file` | 人名说话人：对话语料放行、小说正文拒绝、角色按文件重置 |
| `build_stream_masks_question_and_marks_answer` | SFT 打包流：提问与角色标记被掩码、只有回答段是监督目标 |
| `window_mask_is_shifted_by_one` | 窗口掩码右移一位对齐 `y[i] = tokens[start+1+i]` |
| `sft_loader_batch_shapes_match` | SFT 加载器批次形状与"至少一个监督位置" |
| `test_fit_power_law_recovers_known_exponent` | 幂律拟合能把已知的 `(a, α, b)` 还原回来（固定 `b` 闭式最小二乘 + 黄金分割搜 `b`） |
| `test_fit_power_law_with_zero_irreducible_loss` | `b = 0` 的退化工况：不可约损失不参与时 `α` 不被带偏 |
| `test_param_accounting_matches_real_model` | 参数量公式 vs 真实建层（RMSNorm / SwiGLU / GQA 各组合逐项核对） |
| `test_ratio20_allocation_invariants` | 20:1 最优解满足 `C = 6ND`、`D/N = 20`、`N ∝ C^0.5` |
| `test_chinchilla_table_rows_are_consistent` | 文档那张 Chinchilla 配比表逐行满足 `C = 6ND` 与 `D/N = 20` |
| `test_parametric_optimal_matches_grid_search` | 参数化损失的闭式最优解与网格搜索的最小值点一致 |
| `test_wall_clock_matches_doc_example` | 时长/电费估算复现文档例子（7B × 1T token → 约 61 天 / 约 \$3700） |
| `test_overtrain_curve_has_interior_minimum` | 固定算力沿 IsoFLOP 曲线移动时，预测 loss 有内点最小值（不在 `k = 1`） |
| `test_params_scan_larger_models_reach_lower_loss` | 真训参数量扫描：更大的模型在同 token 预算下 loss 更低 |
| `test_tokens_scan_is_monotone` | 真训数据量扫描：token 数翻倍 loss 单调下降（过训练曲线形状正确） |
| `test_top_k_gate_picks_highest_logits` | Top-K 路由取到 logits 最大的 K 个；并列按下标升序（确定性）；`penalty`/`mask` 两个掩码互为 0/1 对偶 |
| `test_gate_weights_are_softmax_over_topk_only` | 只在 Top-K 上做 softmax ⇒ 恰好 K 个非零、和为 1；未选中位置概率与**梯度**都恰好为 0（"只在 Top-K 取权重"天然可导） |
| `test_moe_forward_matches_dense_reference` | 稀疏前向（gather→专家→加权→scatter）与逐 token 逐专家的稠密参考实现逐元素吻合 |
| `test_moe_routing_is_per_token` | 路由与门控都逐 token 进行：同一行在整批与单条前向里结果一致（容量不限时无跨 token 泄漏） |
| `test_capacity_factor_drops_overflow` | 容量因子：超出容量的分配按 token 顺序先到先得地丢弃，被丢弃的 token 该层输出为 0 |
| `test_aux_loss_minimum_at_uniform` | `L_aux = E·Σf_i p_i` 的取值规律：`p = f` 时 `≥ 1`（均匀取 1）；支撑集不交时可为 0；**p 均匀时恒为 1（与 f 无关，不是均衡证书）** |
| `test_layer_aux_loss_is_above_one_when_skewed` | 真实 MoE 层上的辅助损失：路由偏斜时明显大于 1，门控压平后回到 1 附近，α = 0 时不返回该节点 |
| `test_k1_renorm_gate_has_no_router_gradient` | K = 1 的门控陷阱：重归一化口径下 `w ≡ 1`、路由器**拿不到**主损失梯度；Switch 原概率口径下拿得到 |
| `test_only_selected_experts_receive_gradients` | 稀疏的梯度也是稀疏的：只有被选中的专家拿到梯度，路由器一定拿到 |
| `test_sparse_stats_matches_real_layer` | 参数/激活量公式与真实建层逐位一致（GELU 与 SwiGLU 两种专家）；4 专家取 2 ⇒ 激活约一半、省约一半 FLOPs |
| `test_aux_loss_gradient_balances_routing` | 只优化 `L_aux` 能把偏斜的负载推平（不均衡度下降）——这是"加不加辅助损失"对照实验的机理 |
| `test_moe_layer_trains` | 端到端：MoE 层 + 输出头在簇状合成任务上真的能学起来（专家可分工） |

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

### 14. GPU 加速（可选 feature）

默认构建**不启用 GPU**，保持依赖轻量。通过 `--features gpu` 开启 wgpu 计算着色器加速：

```bash
# ── 所有子命令都支持 --features gpu ──

# GPU 加速训练
cargo run --release --features gpu -- train --config config/config.json

# GPU 加速训练 + 断点续训
cargo run --release --features gpu -- train --config config/config.json --resume checkpoints/zh/latest.ckpt

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
GPT-2 风格的 GELU MLP（SwiGLU 由调用方让路）、形状与规模够大。

归一化用 LayerNorm 还是 RMSNorm、K/V 是不是 GQA 的头数，都**不需要让路**——两者在参数层面
就被抹平了：RMSNorm 与 LayerNorm 共用同一套归约内核，只靠一个模式位切换（RMSNorm 是
"μ≡0、无 β"的特例）；GQA 则把 K/V 的**参数**按 `repeat_kv` 的同一套头顺序展开成 `n_head` 份
（`expand_kv_head`）、反向再把梯度按组折回（`fold_kv_head_grad`），展开后与标准 MHA 同形，
常驻路径一行内核都不用改。整叠路径默认**关闭**，用 `LLM_GPU_STACK` 打开——它与逐子层路径数值
逐位一致，实测还**更快**（同配置 ABBA 两轮：0.59/0.60 s/步 vs 0.67/0.70 s/步，约快 12%）；
默认关闭只是因为它的准入条件更严（要求所有层都是 GELU MLP——各层共用同一份配置，所以一个
SwiGLU 就足以让整条路径放弃）。
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
| small（~3.3M） | 256 | **GPU** | FLOPs 阈值已调低至 5000 万，QKV/MLP 投影走 GPU |
| medium（~26M） | 512 | **GPU** | 矩阵更大，GPU 加速明显 |
| large（~79M） | 768 | **GPU** | 矩阵够大，GPU 充分利用 |
| xlarge（~300M） | 1024+ | **GPU** | 矩阵足够大，GPU 加速显著 |

> ⚠️ **只有预训练（`train`）适合开 GPU**：SFT 的 loss 带逐位置掩码，常驻输出头路径不吃逐位置权重，
> 会整段回落到逐算子路径——实测（MX150）比纯 CPU 还慢（392 tok/s vs 780 tok/s），长跑还会崩。
> `sft` 子命令检测到 gpu feature 时会打印这条提示，跑 SFT 请用不带 `--features gpu` 的构建。

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
    "dropout": 0.0,        // Dropout 概率。0 = 不丢弃，>0 时训练中随机丢弃
    // ---- MoE 稀疏专家（第 32 课；n_expert = 1 就是稠密 FFN，行为与以前逐位相同）----
    "n_expert": 1,             // 每个 MoE 层的专家数（≥ 2 时该前馈子层换成 MoE）
    "moe_top_k": 1,            // 每个 token 激活几个专家
    "moe_capacity_factor": 0.0,// 专家容量因子。0 = 不限（推理必须 0）
    "moe_aux_coef": 0.0,       // 负载均衡辅助损失系数 α。0 = 不加
    "moe_switch_gate": false   // 门控口径：false = Top-K 重归一化（Σw = 1）；true = Switch 原概率
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
| `n_expert` | int | `1` | 每个 MoE 层的专家数。**`1` = 稠密 FFN**（与加 MoE 之前逐位相同）；`≥ 2` 时该前馈子层换成 MoE |
| `moe_top_k` | int | `1` | 每个 token 激活几个专家（`1 ≤ K ≤ n_expert`） |
| `moe_capacity_factor` | float | `0.0` | 专家容量因子（`capacity = cf × n·K/E`，超出按 token 顺序先到先得丢弃）。`0` = 不限容量，**推理时必须 0** |
| `moe_aux_coef` | float | `0.0` | 负载均衡辅助损失系数 α（`L_aux = α·E·Σ f_i·p_i`）。`0` = 不加 |
| `moe_switch_gate` | bool | `false` | 门控口径：`false` = Top-K 内部重归一化（Mixtral / DeepSeek 式，`Σw = 1`）；`true` = 全部专家上的 softmax 原概率（Switch Transformer 式，`Σw < 1`）。**`moe_top_k = 1` 时必须置 `true`**，否则权重恒为 1、路由器拿不到主损失梯度 |

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
    "amp": true,              // 动态损失缩放（AMP）：loss 先乘 scale 再反向，更新前查溢出并反缩放
    "amp_init_scale_log2": 16, // 初始 scale = 2^16 = 65536
    "amp_growth_interval": 2000, // 连续这么多步无溢出就把 scale 翻倍（上限 2^24）
    "eval_every": 100,        // 每 N 步评估一次验证集（同时保存 latest checkpoint）
    "eval_iters": 20,         // 评估时采样的批数（取平均减少方差）
    "tokenizer": "bpe",       // 分词器类型："char"（字符级）或 "bpe"（字节对编码）
    "bpe_vocab": 512,         // BPE 目标词表大小（= 256 字节 + 合并数）
    "train_file": "data/alice.txt", // 训练语料文件路径（支持目录路径，自动合并 .txt 文件）
    "val_file": null,         // 验证语料文件路径。null = 自动从训练文本末尾切 10%
    "out_dir": "checkpoints", // checkpoint 输出目录（权重 + tokenizer.json）
    "accum_steps": 1,         // 梯度累积步数。有效 batch = batch_size × accum_steps
    "tokenizer_file": null,   // 分词器文件路径。null = 从语料训练并保存；Some = 从文件加载
    "lora": null,             // LoRA 配置；finetune 子命令读它（CLI 的 --lora-rank/--lora-alpha/--lora-targets 会覆盖）
    "log_file": "logs/train.csv", // 训练指标日志。null = 不记录；默认 logs/train.csv（自动建目录）
    "early_stop_patience": 0,  // 早停耐心值。0 = 不启用；N = 连续 N 次评估不改善则停止
    "sft_file": null          // SFT / LoRA 微调语料（逗号分隔，可为文件/目录/含 * 的路径）
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
| `amp` | bool | `true` | 动态损失缩放（AMP，见第 26 课）。开启后 loss 先乘 `scale` 再反向；参数更新前检查梯度是否含 Inf/NaN（含则丢弃本步、不更新参数，并把 `scale` 减半）与梯度反缩放，再执行裁剪与更新。`scale` 恒为 2 的幂，f32 下乘除都是精确的指数移位，所以数值上与关闭 AMP 完全一致——它换来的是**溢出保护**与裁剪阈值的正确性 |
| `amp_init_scale_log2` | int | `16` | AMP 初始缩放因子以 2 的幂给出（`scale = 2^16 = 65536`）。取值须 ≤ `24`（`scale` 上限就是 2^24） |
| `amp_growth_interval` | int | `2000` | AMP 缩放因子增长间隔：连续这么多步没有溢出就把 `scale` 翻倍 |
| `eval_every` | int | `100` | 每 N 步在验证集上评估 loss / 困惑度，并保存 `latest.ckpt` |
| `eval_iters` | int | `20` | 评估时采样多少批取平均（减少随机波动） |
| `tokenizer` | string | `"bpe"` | `"char"` = 字符级分词；`"bpe"` = 字节对编码 |
| `bpe_vocab` | int | `512` | BPE 词表大小。仅当 `tokenizer = "bpe"` 时生效 |
| `train_file` | string | `"data/alice.txt"` | 训练语料文件路径（纯文本或目录路径，目录会合并其中的 `.txt`）。仓库自带语料见 `data/`，`config/config.json` 里指向 `data/corpus_zh/` |
| `val_file` | string/null | `null` | 验证语料文件。`null` = 自动从训练文本末尾切约 10% |
| `out_dir` | string | `"checkpoints"` | 权重输出目录：checkpoint（latest / best / final）与 `tokenizer.json` 都写在这里。目录不存在时自动创建 |
| `accum_steps` | int | `1` | 梯度累积步数。有效 batch = `batch_size × accum_steps` |
| `tokenizer_file` | string/null | `null` | 分词器文件路径。`null` = 从语料训练并自动保存；指定路径 = 直接加载 |
| `lora` | object/null | `null` | LoRA 配置 `{ "rank": 16, "alpha": 16.0, "targets": { "q": true, "k": true, "v": true, "o": false, "mlp": false } }`。**只被 `finetune` 子命令读取**（`sft` 忽略它），`--lora-rank` / `--lora-alpha` / `--lora-targets` 优先于它（CLI 没传才看这里）；`train` 子命令不注入适配层，读到它只会打印一行提示。`targets` 字段可省（回落 `q,k,v`，兼容旧档）。详见 [§6](#6-finetune--lora-微调) 与第 29 课 |
| `log_file` | string/null | `"logs/train.csv"` | 训练指标日志文件路径。默认 `logs/train.csv`（`logs/` 目录自动创建）；`null` = 不记录；指定路径 = **每个评估点**（每 `eval_every` 步 + 最后一步）写一行 CSV，列为 `step,lr,train_loss,val_loss,ppl,tokens_per_sec`。无验证集时 `val_loss` / `ppl` 两列留空。**每次训练覆盖该文件**，不是追加 |
| `early_stop_patience` | int | `0` | 早停耐心值。`0` = 不启用；`N` = 验证 loss 连续 N 次评估不改善就提前停止（停止前仍会保存 checkpoint 与日志） |
| `sft_file` | string/null | `null` | 对话语料路径，逗号分隔，每项可以是文件、目录或含 `*` 的路径。**被 `sft` 与 `finetune` 两个子命令读取**（LoRA 微调与全参 SFT 共用同一份语料，效果才可对比）；`--sft-file` 优先于它。两处命令行都未指定且这里也是 `null` 时直接报错 |

### 完整配置示例（仓库当前 `config/config.json`）

```jsonc
{
  "model": {
    "vocab_size": 0,        // 0 = 词表大小由分词器训练结果决定
    "n_embd": 128,
    "n_head": 4,
    "n_layer": 4,
    "block_size": 512,
    "n_kv_head": 0,         // 0 = 标准 MHA
    "use_rmsnorm": false,   // GPT-2 风格（LayerNorm + GELU）
    "use_swiglu": false,
    "dropout": 0.1
  },
  "train": {
    "seed": 42,
    "batch_size": 8,
    "steps": 4000,
    "max_lr": 5e-4,
    "min_lr": 5e-5,
    "warmup_steps": 300,
    "weight_decay": 0.1,
    "grad_clip": 1.0,
    "amp": true,
    "amp_init_scale_log2": 16,
    "amp_growth_interval": 2000,
    "eval_every": 200,
    "eval_iters": 20,
    "tokenizer": "bpe",
    "bpe_vocab": 8192,
    "train_file": "data/corpus_zh/",
    "val_file": null,
    "out_dir": "checkpoints/zh",
    "accum_steps": 1,
    "tokenizer_file": "checkpoints/zh/tokenizer.json",
    "lora": null,
    "log_file": "checkpoints/zh/train.csv",
    "early_stop_patience": 10,
    "sft_file": "data/corpus/zh_dialogue_*.txt,data/sft/"
  }
}
```

这是仓库里唯一自带、且**已经跑过完整训练**的配置：中文语料 + BPE 8192 词表 + 4000 步，
产物落在 `checkpoints/zh/`。上面示例里的 `checkpoints/zh/...` 路径都对应它。

### 最小配置示例（其余字段取默认值）

```json
{
  "train": {
    "steps": 2000,
    "batch_size": 16,
    "max_lr": 6e-4,
    "min_lr": 6e-5,
    "warmup_steps": 100,
    "eval_every": 250,
    "train_file": "data/alice.txt"
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
├── checkpoints/        # 权重目录（自动创建）；各实验按 out_dir 分成子目录
│   ├── zh/             #   本仓库已训好的中文权重：latest/best/final.ckpt + tokenizer.json + train.csv
│   ├── zh-sft/         #   `sft` 的默认输出（{out_dir}-sft），不会覆盖上面的预训练权重
│   └── ...             #   其他实验目录，如 checkpoints/perf、checkpoints/probe_b2
├── logs/               # 日志目录（自动创建）：运行日志（每次 train/eval/generate/chat/sft/finetune 各一份）
│                       #      与训练指标 CSV（由 train.log_file 指定，本项目配的是 checkpoints/zh/train.csv）
├── data/               # 语料：alice.txt（公版《爱丽丝梦游仙境》）
│                       #      corpus/（中英文混合语料，含文章/代码/对话/新闻/诗歌）
│                       #      corpus_zh/（《红楼梦》《三国演义》等中文名著）
│                       #      corpus_perf/（性能测试用节选）、sft/（SFT 问答语料 zh_qa.txt）
├── src/
│   ├── main.rs         # CLI 入口：train / eval / generate / chat / sft / finetune / preset / demo / bench / scaling / moe
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
│   ├── data.rs         # 数据集 + SFT 对话解析与 loss 掩码（第 14 课）
│   ├── train.rs        # 训练循环、学习率调度、梯度累积、早停、CSV 日志、SFT 掩码透传（第 13、18、28 课）
│   ├── sample.rs       # 推理与采样（第 15、30 课）
│   ├── scaling.rs      # Scaling Laws：幂律拟合、算力/参数口径、Chinchilla 最优配比、实测扫描（第 31 课）
│   └── moe.rs          # MoE 稀疏专家：Top-K 路由、两种门控口径、稀疏前向、辅助损失、容量因子（第 32 课）
└── docs/               # 39 课教程文档（00-学习计划 + 01~39 各课）
```

### 产物目录约定（自动创建，无需手动 mkdir）

| 产物 | 默认位置 | 由谁决定 | 说明 |
|------|----------|----------|------|
| 配置文件 | `config/config.json` | `--config` / `--output` | 所有子命令的配置默认路径；`preset --output` 写同类路径 |
| 权重 | `checkpoints/` | `train.out_dir` | `latest.ckpt` / `best.ckpt` / `final.ckpt` 与 `tokenizer.json`；`sft` 另写 `{out_dir}-sft`，`finetune` 写回 `{out_dir}` |
| 训练指标日志 | `logs/train.csv` | `train.log_file` | CSV：`step,lr,train_loss,val_loss,ppl,tokens_per_sec` |
| 运行日志 | `logs/{操作}_{时间戳}.log` | 程序自动生成 | 每次 `train` / `eval` / `generate` / `chat` / `sft` / `finetune` 各写一份，文件名含操作名与毫秒级本地时间；内容 = 完整命令行 + 完整配置 + 该次运行的全部输出 |

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
4. 读完每课末尾的"拓展方向"（核心内容已全部实现，那里是进阶拓展）。

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
> | 九、前沿技术 | 31-38 | Scaling Laws、MoE、量化、推测解码、RLHF、RAG、分布式 | 31-38 ✅ |
> | 十、工程化完善 | 39 | CLI 工程、分词器持久化、预设配置、微调工作流、SFT 监督微调、交互式对话 | ✅ |

## 代码验证状态

- **200 个单元测试全部通过**（`cargo test --bin llm_from_scratch` → `200 passed; 0 failed`，耗时约 14 分钟；
  GPU 用例需 `--features gpu`，另计）（详见 [§13 `cargo test`](#13-cargo-test--单元测试)）
- **`cargo build` 与 `cargo build --features gpu` 编译通过、无 error**：剩余提示是 `dead_code` 警告，
  分两类。一类是**只有单测 / CLI 子命令里某条路径才用到的 API**（如 `quant.rs` 的 `cholesky_inverse` /
  `Calibration` / `awq_best_alpha`、`distributed.rs` 的 `World::barrier` / `broadcast` / `qkv_head_columns`、
  `rag.rs` 的 `Retriever::from_text`、`speculative.rs` 的 `MtpDrafter::heads_mut`、`rope.rs` 的 `with_window`）——
  它们都有单测覆盖，只是非测试构建里没有直接调用点。另一类是少数只在测试基准或 `gpu` feature 常驻路径上
  被调用的算子：`Tensor::mul` / `div` / `sum_last_dim` / `add_scalar` 是融合算子的
  **分步参考实现**（测试用它们串出定义式，再与融合算子比对数值），`Tensor::external` / `NormLayer::ln_params` /
  `Gelu::gelu_weights` 的调用点全在 GPU 常驻路径，`Tensor::neg` 则是逐元素算子集里的一员。
  后一类已按调用条件标了 `#[allow(dead_code)]` / `#[cfg_attr(not(feature = "gpu"), allow(dead_code))]` 并在注释里写明，
  不带 gpu feature 编译时不会混进 `never used` 噪声
- 测试覆盖：自动微分、广播、softmax、BPE 编解码、RoPE 正交性与梯度、KV cache 与全量前向一致性、
  RMSNorm/SwiGLU 融合算子与分步实现一致性、Flash Attention 与标准注意力一致性、Dropout、线性回归收敛、
  重复惩罚、checkpoint 二进制往返（含 NaN/±Inf）、**SFT 对话解析 / loss 掩码对齐 / 批次形状**、
  **LoRA（零初始化前向不变、冻结主干无梯度、可训练集合、挂载位置生效、合并前后前向等价、续训保留旧数值、存档往返）**、
  **KV cache 滑动窗口（超窗丢弃最旧行、绝对位置继续累计、单层模型下与全量窗口前向数值等价）**、
  **AMP 动态损失缩放（开/关结果逐位一致、scale 翻倍与减半、梯度反缩放）**、
  **Scaling Laws（幂律拟合还原已知指数、参数口径与真实建层逐位一致、Chinchilla 表自洽、20:1 与参数化闭式解两条路线、时长/电费复现文档例子、真训多规模扫描与数据量扫描）**、
  **MoE（Top-K 路由确定性与掩码对偶、稀疏前向与稠密参考逐元素吻合、路由逐 token 无跨 token 泄漏、容量丢弃语义、辅助损失取值规律与梯度、K = 1 门控梯度陷阱、稀疏梯度、参数口径、端到端可训）**、
  **量化（逐元素 / 逐通道 / 逐 token 量化与反量化、位打包往返、GPTQ 分块不改结果、AWQ α 搜索、校准集合并、Hessian 逆的数值校验）**、
  **推测解码（贪心路径逐位一致、拒绝采样的分布等价、缓存回滚不变量、MTP 头形状与梯度、多步训练 loss 下降）**、
  **对齐（DPO = `ln 2`、PPO 裁剪梯度为 0、GRPO 优势均值 0 / 方差 1、奖励模型排序准确率高于随机）**、
  **RAG（分块不变量、余弦相似度自比 1 / 正交 0、MMR 多样化、提示预算截断）**、
  **分布式（环形 allreduce = 数据和、DP / ZeRO 与单进程全 batch 一致、TP 前向反向与单卡一致、PP 两种调度等价、3D 切分互不重叠且覆盖完整）**
- **Demo 端到端验证通过**（`cargo run --release -- demo`）：XOR 100%、BPE 往返、GPT 训练 loss 1.63→0.15、文本生成正常
- **工程化功能已全部集成**：监督微调（`sft`，带 loss 掩码，默认输出到 `{out_dir}-sft`，不覆盖预训练权重）、
  LoRA 微调（`finetune`，冻结主干只训适配层，实测可训练参数 1.32%，可选挂载位置 / 链式续训 / 推理合并，默认输出到 `{out_dir}-lora`）、
  Beam Search（`generate --beam`）、交互式对话（`chat`，含 `--prompt-format sft/raw`）、分词器持久化（`tokenizer.json`）、
  预设配置（`preset`）、训练指标日志（CSV）、运行日志（`logs/{操作}_{时间戳}.log`）、早停（`early_stop_patience`）
- **Scaling Laws 已落地**（第 31 课）：`scaling` 子命令上半场算预算（20:1 法则 vs 参数化损失闭式解、
  Chinchilla 配比表、训练时长/电费、过训练曲线），下半场**真训**多个规模做幂律拟合（loss vs N / vs D）
  并导出 CSV；参数口径与真实建层共用公式 + 扫描时逐位断言，公式漂移会当场 panic 而不是给出错数字
- **MoE 稀疏专家已落地**（第 32 课）：`moe` 子命令四节实验（参数口径 / 负载均衡隔离实验 / 端到端对照 /
  容量因子与 Token Dropping）；`GPTConfig.n_expert = 1` 时行为与加 MoE 之前**逐位相同**；
  `moe_top_k = 1` 时必须配 `moe_switch_gate`，否则路由器拿不到主损失梯度（子命令与单测都会指出这一点）
- **量化已落地**（第 33 课）：`quant` 子命令支持 RTN / GPTQ / AWQ 三种 weight-only 量化（int8 / int4），
  校准集走 Hessian `XᵀX`（含 act-order、阻尼、分块），AWQ 的缩放指数 α 可在 0~1 网格上按代理误差逐层搜索，
  可选 `--eval` 对比量化前后验证集 loss / 困惑度，并把量化权重与元信息落盘到 checkpoint
- **推测解码与多 Token 预测已落地**（第 34/35 课）：`speculative` 子命令五节实验——贪心路径与逐 token
  解码**逐位对照**、γ 扫描下的目标前向次数对照、缓存不变量逐轮核对（`cache_fed` 恒等于序列长度 − 1）、
  首 token 经验分布对目标分布（4σ 统计判据）、MTP 头训练后当草稿且仍然无损；
  `gamma + 1 <= block_size`、被拒位置 `rollback_to` 对齐等约束在代码里直接断言
- **RLHF 与对齐已落地**（第 36 课）：`align` 子命令依次验证奖励模型排序准确率、DPO 偏好边界
  （策略与参考相同时 loss = `ln 2`，一步训练后 chosen / rejected 的 logprob 差严格变大）、
  GRPO 组内优势（均值 0、方差 1）、PPO 裁剪分支梯度为 0 与 KL(k3) 惩罚
- **RAG 已落地**（第 37 课）：`rag` 子命令验证分块不变量（除重叠外每字符恰好出现一次、中文多字节不被切开）、
  三种向量化（TF-IDF / FNV 哈希 / 模型隐状态池化）对照、top-k 与 MMR 检索、按预算组装提示
- **分布式训练已落地**（第 38 课）：`distributed` 子命令给出集合通信量对照（环形 allreduce =
  reduce-scatter + all-gather，每 rank 只与左右邻居通信）、DP / ZeRO-1/2 的训练轨迹与状态分片、
  TP / PP / 3D 的切分与激活驻留报告；单测保证 DP / ZeRO 与单进程全 batch **数值一致**、
  TP 前向反向与单卡一致

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

**正确性**：当时 69 个单元测试全部通过；同一 seed 下 loss 与优化前完全一致；`demo` 端到端正常。
（该组数据是优化当时的同机 A/B，绝对值有 ±20% 噪声；当前可复现的对照点见 [§9 `bench`](#9-bench--性能基准) 与 [§14 GPU 章节](#14-gpu-加速可选-feature)。）

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
- Beam Search（`sample::beam_search`）目前**不用 KV cache**：每一步都对每个 beam 做一次全量前向
  （`beam_size × max_new` 次前向），长文本时是主要开销。打分已按 `log_softmax` 后累加对数概率
  （此前累加原始 logit，既没归一化、又让长度惩罚方向与文档相反，已修正），但仍是 O(beam×T) 的重算

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
- **Beam Search 要累加对数概率而不是原始 logit**：`log_softmax` 会把 logit 归一化成合法分布的对数
  （各项 ≤ 0、序列越长和越小），除以 `len^α` 才有"平均每 token 的对数概率"的含义。原先直接累加原始
  logit 既没归一化（剪枝会系统性偏向"logit 整体偏大"的路径），又让长度惩罚方向与参数文档相反。
  注意 `log_softmax` 要先减去行内最大值做数值稳定化；被掩码的 `-inf` 在 `exp(-inf - max) = 0`，
  不参与配分函数也不影响结果。
- **SFT 的 loss 掩码要右移一位对齐**：窗口里 `y[i] = tokens[start+1+i]`，所以第 `i` 个位置的掩码
  要看**目标 token 所在的位置** `sup[start+1+i]`，而不是输入位置 `sup[start+i]`。少移这一位，
  整个批次的监督信号会整体错开一个 token——loss 照降、形状全对，极难从结果看出来。
- **SFT 模板标记要用语料里"高频"的字，"出现过"不够**：`#` 在 4.7M 字的中文语料里只出现 2 次，
  它的 embedding 基本没被训过，模板一带上 `### ` 就把模型推进 ASCII/数字乱码模式（实测基座输出
  `'何人？」621.-----2..3E8188`）；`【】` 同理不能用在标记里（也只出现 2 次）。**结束标记尤其
  不能踩这个坑**——它每段对话都要被监督，模型必须能把它生成出来，所以只挑"出现过"的字是不够的：
  最初的 `（结束）` 四个字频率都在 0.002%~0.008%（`（`129 次 / `结`204 次 / `束`391 次 /
  `）`130 次），和 `#` 同一个数量级，结果模型只会吐一个 `（` 然后彻底跑偏，输出全是括号乱码。
  现在用 `。。`：`。` 是最高频字符（13.2 万次），`。。` 组合在语料里只出现 2 次。
  加特殊 token 也能解，但会给已有 checkpoint 带来"embedding 行数对不上"的兼容问题，
  字节级 BPE 本来就能编码任意 UTF-8 字符串，不必动词表。
- **SFT 的说话人角色是文件级属性**：人名标签（`陈教授：` / `面试官：`）没有固定含义，按"该文件里
  谁先说"判定；**跨文件共用一张表会把后面文件的角色弄反**，所以加载时要保留文件边界
  （`load_texts` 而不是拼成一份）。人名标签只在整份文件 ≥90% 非空行是角色行、且说话人只有两个时
  才启用——否则小说正文（`秦琼道：……`）会被当成对话喂进来。

## 后续方向

第 31~38 课（Scaling Laws、MoE、量化、推测解码、多 Token 预测、RLHF、RAG、分布式训练）
**均已落地为可运行代码 + 单测 + CLI 子命令**：

- **Scaling Laws**（第 31 课）→ `src/scaling.rs` + `scaling` 子命令
- **MoE 混合专家模型**（第 32 课）→ `src/moe.rs` + `moe` 子命令
- **量化部署**（第 33 课）→ `src/quant.rs`（INT8/INT4、GPTQ、AWQ、校准、逐层量化）+ `quant` 子命令
- **推测解码**（第 34 课）→ `src/speculative.rs`（拒绝采样 + 残差修正 + KV cache 回滚）+ `speculative` 子命令
- **多 Token 预测**（第 35 课）→ `src/speculative.rs` 的 `MtpHeads` / `MtpDrafter`
- **人类对齐**（第 36 课）→ `src/align.rs`（奖励模型、DPO、GRPO、PPO）+ `align` 子命令
- **RAG 检索增强生成**（第 37 课）→ `src/rag.rs`（分块、向量化、检索、提示组装）+ `rag` 子命令
- **分布式训练**（第 38 课）→ `src/distributed.rs`（DP、ZeRO、TP、PP、3D）+ `distributed` 子命令
- **长度外推**（第 20 课）→ `src/rope.rs`（Linear PI / NTK-aware / YaRN）

仍可继续推进的方向：

- 更大的语料与模型规模（`config/config.json` 可直接调大，CPU 训练需耐心）
- 真实多机多卡通信后端：`distributed.rs` 目前是单进程模拟 world，语义与数值对齐真实集合通信，
  换成 NCCL / MPI 时算法本身无需改动，但需要另写通信层
