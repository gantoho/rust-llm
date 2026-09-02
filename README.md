# llm_from_scratch —— 用纯 Rust 从零实现大语言模型

> **深度学习算法全部纯手写**：不使用任何深度学习框架（如 tch-rs / candle / burn），
> 从零手写张量、自动微分、神经网络层、Transformer 架构。
> 仅引入少量**工具库**（serde_json 做配置/序列化、clap 做命令行、windows-sys 修控制台编码、
> 可选的 wgpu 做 GPU 计算后端），它们都不参与任何算法实现。
>
> 每个实现步骤都配套一篇中文教程文档（见 `docs/`），边写代码边学原理。

## 项目简介

本项目是一个**教学性质**的深度学习项目，目标是让你理解大语言模型（LLM）的底层原理：

- **算法零依赖**：所有张量运算、自动微分、网络层全部手写，算法部分不用任何第三方库。
- **循序渐进**：按 [docs/00-学习计划.md](docs/00-学习计划.md) 划分 6 个阶段、21 课，从张量一路写到可生成文本的小 GPT。
- **工程化完整**：CLI 子命令（train / eval / generate / demo）、外部语料、train/val 划分、
  验证集评估与困惑度、checkpoint 保存/恢复、断点续训。
- **透明度高**：训练过程中每一步的中间结果、梯度、损失都可以直接打印检查。

### 包含的功能（对应 20 课）

| 模块 | 文件 | 内容 |
|------|------|------|
| 张量运算 | `src/tensor.rs` | Tensor 结构体、广播、逐元素/标量运算、matmul、softmax、permute、gather |
| 自动微分 | `src/autograd.rs` | backward 反向传播、拓扑排序（计算图 → 梯度流） |
| RoPE 位置编码 | `src/rope.rs` | 旋转位置编码：把相对位置揉进 Q/K 向量 |
| 神经网络层 | `src/layers.rs` | Linear、LayerNorm、Embedding、ReLU/GELU/Tanh |
| 损失与优化器 | `src/loss.rs` `src/optim.rs` | MSE、CrossEntropy、SGD、AdamW（动量 + 权重衰减） |
| 分词器 | `src/tokenizer.rs` | 字符级分词 + BPE（字节对编码），配置可切换 |
| 注意力机制 | `src/attention.rs` | 多头自注意力、因果掩码、RoPE 位置编码、KV Cache |
| GPT 模型 | `src/model.rs` | Transformer Block 堆叠、GPT 整体前向、checkpoint 参数名 |
| 数据加载 | `src/data.rs` | 外部文本文件、train/val 划分、随机 batch 采样 |
| 训练与评估 | `src/train.rs` | 训练循环、梯度裁剪、warmup+cosine 学习率、验证集 loss / 困惑度 |
| 采样 | `src/sample.rs` | temperature / top-k / top-p 采样，KV cache 推理 |
| 配置 | `src/config.rs` | `config.json`：模型超参 + 训练参数（serde 序列化） |
| Checkpoint | `src/checkpoint.rs` | 模型参数 + 优化器状态保存/恢复（latest / best / final） |
| 命令行 | `src/cli.rs` | clap 子命令：train / eval / generate / demo |
| 随机数 | `src/rng.rs` | 自实现 xorshift64 伪随机数发生器 |
| GPU 加速 | `src/gpu.rs` | 可选（`--features gpu`）：wgpu 计算着色器加速 matmul/scale/add/relu，失败自动回退 CPU |

## 快速开始

需要 **Rust 2024 edition** 工具链（Rust 1.85+，建议使用最新的 stable）。

```bash
# 1. 训练模型（超参数与数据路径在 config.json 里）
cargo run --release -- train --config config.json

# 2. 在验证集上评估（loss 与困惑度）
cargo run --release -- eval --config config.json

# 3. 用训练好的模型生成文本
cargo run --release -- generate --config config.json --prompt "Alice was" --max-new 100

# 4. 断点续训（从最近的 checkpoint 继续）
cargo run --release -- train --config config.json --resume checkpoints/latest.ckpt

# 5. 教学演示（XOR + BPE + 内置语料小 GPT + GPU 演示）
cargo run --release -- demo

# 6. 单元测试（18 个：自动微分、广播、softmax、BPE、RoPE、KV cache 一致性等）
cargo test
```

### GPU 加速（可选）

默认构建**不启用 GPU**，保持依赖轻量；用 `--features gpu` 开启 wgpu 计算着色器加速：

```bash
# 开启 GPU 后运行教学演示（第 4 个演示会做 GPU 正确性与性能对比）
cargo run --release --features gpu -- demo

# 训练 / 推理同样可加 --features gpu
cargo run --release --features gpu -- train --config config.json

# 推理时可加 --features gpu
cargo run --release --features gpu -- generate --config config.json --prompt "Alice was" --max-new 100
```

原理与特性：

- **实现**：用 wgpu 计算着色器（WGSL）手写 GPU 算子，支持 NVIDIA 与 Intel 核显
  （Windows 下走 DX12 / Vulkan，无需额外驱动安装）。
- **加速范围**：先覆盖热点算子——批量矩阵乘（tiled 16×16 共享内存实现）、逐元素
  scale / add / ReLU；训练与推理中的矩阵乘自动走 GPU。
- **分流策略（利用率优化）**：每个矩阵乘按规模分流——FLOPs（`2·m·k·n·batch`）低于
  `MATMUL_MIN_FLOPS`（默认 200_000）的**微型矩阵**（如注意力的 scores / attn·v）直接走 CPU，
  因为 GPU 一次 dispatch 的固定开销（上传/调度/同步/下载）超过计算本身；只有足够大的矩阵
  （QKV 投影、MLP、512×512 基准）才真正上 GPU。GPU 侧做了 **buffer 池化 + 参数 buffer 复用**，
  不再每次调用新建 GPU 对象，固定开销大幅降低。训练结束时（`--features gpu`）会打印
  `matmul 分流统计：GPU x 次 / CPU y 次`，方便观察利用率。内置小模型（n_embd=64）矩阵普遍偏小，
  GPU 收益有限；调大 `config.json` 的 `n_embd` / `block_size` / `batch_size` 后大矩阵（QKV/MLP）
  会走上 GPU。注意：教学实现中每个矩阵乘仍**串行同步取回**（一次一算子），低端 GPU（如 MX150）
  上每步多次 GPU 调用会累积固定开销，大模型训练依然偏慢——这是"教学优先"的架构取舍。
- **自动回退**：GPU 初始化失败、形状超限（workgroup 超出 65535）或任何一次调用出错时，
  自动回退 CPU 计算，**不影响正确性**。`cargo run --features gpu -- demo` 在无可用 GPU 时会打印
  `未检测到可用 GPU，已回退 CPU`；有 GPU 时演示 4 末尾会打印
  `（训练/推理中 matmul 已自动走 GPU，失败自动回退 CPU）`。
- **提示**：首次在 Windows 上运行，GPU 驱动会写 DX 着色器缓存（如 `NVIDIA DXCache`、`D3DSCache`），
  若程序运行环境限制了这些目录的访问（例如沙箱），请放行或直接在普通终端中运行。

### 配置说明（`config.json`）

```jsonc
{
  "model": {          // 模型超参数
    "vocab_size": 0,  // 0 = 由分词器决定（训练时自动填入）
    "n_embd": 128,    // 隐藏维度
    "n_head": 4,      // 注意力头数
    "n_layer": 4,     // Transformer 层数
    "block_size": 64  // 最大上下文长度
  },
  "train": {
    "seed": 42,           // 随机种子（复现实验）
    "batch_size": 8,      // 每批序列条数
    "steps": 2000,        // 总训练步数
    "max_lr": 6e-4,       // 峰值学习率
    "min_lr": 6e-5,       // cosine 衰减的最低学习率
    "warmup_steps": 50,   // 线性预热步数
    "weight_decay": 0.01, // AdamW 权重衰减
    "grad_clip": 1.0,     // 梯度裁剪阈值
    "eval_every": 250,    // 每 N 步评估验证集并保存 latest checkpoint
    "eval_iters": 20,     // 评估时采样的批数
    "tokenizer": "bpe",   // "char" 字符级 / "bpe" 字节对编码
    "bpe_vocab": 512,     // BPE 目标词表大小（= 256 字节 + 合并数）
    "train_file": "data/alice.txt", // 训练语料
    "val_file": null,     // 验证语料；null 时自动从训练文本末尾切 10%
    "out_dir": "checkpoints"        // checkpoint 输出目录
  }
}
```

缺省字段自动取默认值（同 `GPTConfig::default` / `TrainConfig::default`）。

### 演示内容（`cargo run --release -- demo`）

`main.rs` 依次运行四个演示，验证各阶段成果：

1. **MLP 学习 XOR**（第 7 课）：验证神经网络 + 反向传播正确，训练后正确率 4/4（100%）。
2. **BPE 分词器**（第 8 课）：在示例语料上训练字节对编码词表（400 个 token），演示编码/解码往返。
3. **训练小 GPT 并生成文本**（第 12-21 课）：在 669 字符的英文小故事上训练 600 步，
   loss 从约 1.46 降到约 0.16，随后用 temperature=0.8 / top-k=10 / top-p=0.9 生成文本，
   并分别演示"无 KV cache"与"带 KV cache"两种推理方式（位置编码为第 19 课的 RoPE）。
4. **GPU 加速对比**（第 21 课，仅 `--features gpu`）：打印设备信息，验证批量矩阵乘
   CPU vs GPU 数值一致性（最大误差），实测 512×512 矩阵乘加速比，并校验逐元素算子
   scale / relu / add。无可用 GPU 时自动回退 CPU 并提示。

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
└── docs/               # 21 课教程文档（00-学习计划 + 01~21 各课）
```

## 学习路线

1. 先读每课教程文档（`docs/XX-xxx.md`）理解原理；
2. 再看对应源码，对照实现；
3. 自己动手改代码、跑实验，验证理解；
4. 完成每课末尾的"动手练习"。

> 推荐从 [docs/00-学习计划.md](docs/00-学习计划.md) 开始，按顺序阅读 01→21 课。

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

- 长度外推：RoPE 配合 NTK-aware scaling、YaRN 等技巧（见第 19 课文档）
- 更大的语料与模型规模（`config.json` 可直接调大，CPU 训练需耐心）
- 更多现代技巧：MoE、Grouped Query Attention、Flash Attention 等
