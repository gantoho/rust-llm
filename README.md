# llm_from_scratch —— 用纯 Rust 从零实现大语言模型

> 不使用任何深度学习框架（如 tch-rs / candle / burn），仅用 Rust 标准库，
> 从零手写张量、自动微分、神经网络层、Transformer 架构，最终训练出一个能生成文本的小型 GPT 模型。
>
> 每个实现步骤都配套一篇中文教程文档（见 `docs/`），边写代码边学原理。

## 项目简介

本项目是一个**教学性质**的深度学习项目，目标是让你理解大语言模型（LLM）的底层原理：

- **不依赖任何框架**：所有张量运算、自动微分、网络层全部手写，`Cargo.toml` 零依赖。
- **循序渐进**：按 [docs/00-学习计划.md](docs/00-学习计划.md) 划分 6 个阶段、20 课，从张量一路写到可生成文本的小 GPT。
- **透明度高**：训练过程中每一步的中间结果、梯度、损失都可以直接打印检查。

### 包含的功能（对应 20 课）

| 模块 | 文件 | 内容 |
|------|------|------|
| 张量 + 自动微分 | `src/tensor.rs` | 广播、softmax、permute、gather、3D matmul、反向传播、RoPE 旋转位置编码 |
| 神经网络层 | `src/layers.rs` | Linear、LayerNorm、Embedding、ReLU/GELU/Tanh |
| 损失与优化器 | `src/loss.rs` `src/optim.rs` | MSE、CrossEntropy、SGD、AdamW（动量 + 权重衰减） |
| 分词器 | `src/tokenizer.rs` | 字符级分词 + BPE（字节对编码） |
| GPT 模型 | `src/model.rs` | 多头注意力、因果掩码、Transformer Block、正弦位置编码、KV Cache |
| 训练与采样 | `src/train.rs` `src/data.rs` `src/sample.rs` | 训练循环、梯度裁剪、warmup + cosine 学习率、temperature/top-k/top-p 采样 |
| 随机数 | `src/rng.rs` | 自实现 xorshift64* 伪随机数发生器 |

## 运行方式

需要 **Rust 2024 edition** 工具链（Rust 1.85+，建议使用最新的 stable）。

```bash
# 1. 运行完整演示（推荐用 release 模式，速度快约 10 倍）
cargo run --release

# 2. 运行单元测试（16 个测试：自动微分、广播、softmax、BPE、RoPE 等）
cargo test

# 3. 调试模式运行（编译快但训练慢，适合修改代码时排查）
cargo run
```

### 演示内容（`cargo run --release` 输出）

`main.rs` 依次运行三个演示，验证各阶段成果：

1. **MLP 学习 XOR**（第 7 课）：验证神经网络 + 反向传播正确，训练后正确率 4/4（100%）。
2. **BPE 分词器**（第 8 课）：在示例语料上训练字节对编码词表（400 个 token），演示编码/解码往返。
3. **训练小 GPT 并生成文本**（第 12-20 课）：在 669 字符的英文小故事上训练 600 步，
   loss 从约 4.06 降到约 0.34，随后用 temperature=0.8 / top-k=10 / top-p=0.9 生成文本，
   并分别演示"无 KV cache"与"带 KV cache"两种推理方式。

## 目录结构

```
llm_from_scratch/
├── Cargo.toml          # 零依赖配置
├── README.md           # 本文件
├── src/
│   ├── main.rs         # 入口：三个演示
│   ├── tensor.rs       # 张量 + 自动微分（第 1-4、19 课）
│   ├── rng.rs          # 随机数（第 5 课）
│   ├── layers.rs       # 网络层（第 5、11 课）
│   ├── loss.rs         # 损失函数（第 6 课）
│   ├── optim.rs        # 优化器（第 6、17 课）
│   ├── module.rs       # 参数管理 trait（第 5 课）
│   ├── tokenizer.rs    # 分词器（第 8 课）
│   ├── model.rs        # GPT 模型（第 9-12、18 课）
│   ├── data.rs         # 数据集（第 14 课）
│   ├── train.rs        # 训练循环与学习率调度（第 13、20 课）
│   └── sample.rs       # 推理与采样（第 15 课）
└── docs/               # 20 课教程文档（00-学习计划 + 01~20 各课）
```

## 学习路线

1. 先读每课教程文档（`docs/XX-xxx.md`）理解原理；
2. 再看对应源码，对照实现；
3. 自己动手改代码、跑实验，验证理解；
4. 完成每课末尾的"动手练习"。

> 推荐从 [docs/00-学习计划.md](docs/00-学习计划.md) 开始，按顺序阅读 01→20 课。

## 实现要点与踩坑记录

- **性能陷阱**：`Tensor` 的父节点列表必须用 `Rc<Vec<Tensor>>` 存储。若直接存 `Vec<Tensor>`，
  `#[derive(Clone)]` 会递归深拷贝整棵祖先计算图，深层图上每次建节点都是 O(图深) 开销，
  曾导致单步训练耗时 26 秒；改用 Rc 共享后降至约 150ms。
- **同张量特判**：`x * x`、`x + x` 这类同一张量参与运算的情况，反向传播时梯度要走两条路径
  合并（用 `Rc::ptr_eq` 判断），否则会 `RefCell` 双重借用报错或梯度算错。
- **数值稳定**：softmax 先减每行最大值再 exp，防止指数溢出。
- **RoPE 反向**：旋转矩阵正交，梯度回传要用其转置 `R(θ)ᵀ`（相当于负角度旋转），
  符号写反不会影响梯度范数，但方向会错——务必用逐元素断言测试校验。

## 后续方向

- 将 RoPE 接入注意力模块（替换正弦位置编码，代码已就绪，见第 19 课文档）
- 更大的语料与模型规模（当前为演示用的小配置）
- 更多现代技巧：MoE、Grouped Query Attention、Flash Attention 等
