# 依赖边界与自研清单

> 这份文档回答两个问题：**哪些东西是本项目自己实现的？** 哪些是借外部 crate 的、为什么借、
> 什么时候换成自研？每次引入新 crate 或替换旧 crate，都要同步更新这里
> （`tests/dependency_policy.rs` 会检查"白名单 ↔ Cargo.toml ↔ 本文档"三者一致）。

## 一、原则

1. **算法一律自研**：张量与广播、自动微分、注意力（含 GQA/MLA/滑动窗口）、归一化、
   优化器、采样、量化（INT8/INT4/FP8/FP4）、分词器（BPE 训练与编码）、训练循环、蒸馏、
   投机解码、JSON 约束解码——全部在 `src/` 里逐行实现，没有调用任何深度学习框架。
2. **基础设施允许用现成库**：HTTP 服务、命令行解析、JSON、并行调度、图像编解码、平台 FFI
   都不属于 LLM 的"功能板块"，用成熟库是正确取舍（自己写 HTTP 只会引入协议 bug）。
   判断标准只有一条：**它是否参与"把输入变成输出"的数学计算**——参与的必须自研。
3. **每一项外部依赖都要登记**：用途、触及面、为什么不自研（或何时自研）。

## 二、自研模块（行数按当前工作区统计）

### 2.1 算法核心

| 模块 | 行数 | 测试数 | 实现内容 |
|---|---|---|---|
| `tensor.rs` | 6,769 | 51 | N 维张量、广播、matmul（分块/打包/bf16/AVX2-FMA 快路）、归约、卷积、反向算子 |
| `gpu.rs` | 8,958 | — | 手写 wgpu 计算着色器（HLSL）：matmul / attention / norm 的 GPU 版本、常驻权重缓存 |
| `model.rs` | 3,731 | 24 | Transformer 组网、参数命名、LoRA 挂载、权重导入导出、视觉塔接线 |
| `quant.rs` | 3,427 | 34 | INT8/INT4/FP8/FP4 量化与反量化、校准、Hessian 误差补偿、AWQ |
| `attention.rs` | 2,379 | 27 | MHA / GQA / MLA / 滑动窗口 / Attention Sink / KV cache |
| `tokenizer.rs` | 2,134 | 20 | BPE 训练（精确合并）、编码解码、特殊 token、HF tokenizer 导入 |
| `data.rs` | 1,726 | 15 | 语料加载、SFT 对话解析与掩码、批采样、多模态样本 |
| `sample.rs` | 2,104 | 22 | 采样（top-k/top-p/repetition/presence/frequency）、KV cache 生成、批解码、logprobs |
| `train.rs` | 1,973 | 20 | 训练循环、梯度累积、AMP/bf16、早停、评估、checkpoint 落盘 |
| `hf.rs` | 1,450 | 10 | safetensors 解析、HF config → 本仓库结构映射 |
| `optim.rs` | 781 | 12 | AdamW / Muon / SGD、梯度裁剪、学习率调度 |
| `layers.rs` | 1,591 | 13 | Linear / LayerNorm / RMSNorm / QK-Norm / SiLU / SwiGLU / GELU |
| `rope.rs` | 1,089 | 16 | RoPE（含 NTK/YaRN 缩放）、`rope_pair` 融合旋转 |
| `moe.rs` | 1,550 | 18 | MoE 路由、容量因子、均衡损失、专家并行口径 |
| `loss.rs` | 216 | 6 | 交叉熵（含 mask 与 label smoothing）、MSE |
| `mhc.rs` | 545 | 10 | mHC 流形约束超连接（Sinkhorn 归一化） |
| `align.rs` | 1,033 | 17 | RLHF / DPO 对齐 |
| `distill.rs` | 872 | 7 | 知识蒸馏（温度、软标签、KD 梯度） |
| `speculative.rs` | 1,163 | 8 | 投机解码（草稿模型 + 拒绝采样校正） |
| `rag.rs` | 1,045 | 9 | 检索增强生成（向量索引 + 召回） |
| `scaling.rs` | 1,297 | 13 | Scaling Laws 拟合与预测 |
| `distributed.rs` | 3,027 | 18 | 数据并行 / 梯度 all-reduce / 通信原语 |
| `comm.rs` | 886 | 8 | 通信后端（环状 all-reduce 等） |
| `fp8.rs` | 906 | 10 | FP8 低精度模拟（E4M3/E5M2、分块缩放） |
| `fp4.rs` | 536 | 9 | FP4 量化感知训练 |
| `csa.rs` | 776 | 13 | CSA/HCA 混合注意力 |
| `vqvae.rs` | 543 | 5 | VQ-VAE 图像编解码（手写码本与残差块） |
| `vision.rs` | 454 | 3 | 视觉塔输入预处理、PNG 编码 |
| `json_constraint.rs` | 656 | 5 | JSON 约束解码的后缀自动机 + 词表掩码（`response_format`） |
| `autograd.rs` | 314 | 4 | 线性 tape 自动微分、`no_grad` |
| `config.rs` | 727 | 7 | 配置结构、校验、预设 |
| `prompt.rs` | 494 | 9 | 对话模板（sft / chatml / raw）、历史裁剪、预算分配 |
| `checkpoint.rs` | 1,248 | 18 | checkpoint 读写（`try_*` 错误处理）、参数按名对齐 |
| `sftgen.rs` | 390 | 4 | 合成 SFT 语料生成（`sft-gen` 子命令，替代原 Python 脚本） |
| `rng.rs` | 187 | 4 | 确定性 PRNG（含权重初始化、采样） |
| `module.rs` | 33 | — | `Module` trait 与参数遍历 |
| `lib.rs` | 74 | — | crate 根：模块声明与重导出 |

算法核心合计 **57,084 行**。

### 2.2 基础设施（自研，但属于工程管道）

| 模块 | 行数 | 测试数 | 实现内容 |
|---|---|---|---|
| `serve.rs` | 4,629 | 41 | HTTP 服务：OpenAI 协议实现（chat/embeddings/completions/metrics/OpenAPI） |
| `cli.rs` | 1,041 | 2 | CLI 定义（clap derive）、23 个子命令参数 |
| `runlog.rs` | 394 | — | 运行日志（表格化字段、颜色、UTF-8 控制台） |
| `syncutil.rs` | 55 | 2 | 中毒免疫的锁获取（`Mutex::lock` 的防护层） |
| `winutil.rs` | 111 | 2 | **手写 Windows FFI**（控制台 UTF-8、本地时间），替代 `windows-sys` |

### 2.3 命令层

| 文件 | 行数 | 测试数 | 实现内容 |
|---|---|---|---|
| `core.rs` | 1,441 | — | 核心命令：train / eval / quant / generate / sft / finetune / preset / bench |
| `deploy.rs` | 477 | — | 服务命令：serve / chat |
| `experiments.rs` | 5,178 | — | 实验命令：scaling / moe / distributed / align / rag / speculative / distill / fp8 / fp4 / csa / demo / sft-gen |
| `mod.rs` | 404 | — | 跨命令共享的辅助函数（含 `common_run_fields`） |
| `main.rs` | 365 | — | 二进制入口：子命令分发 |

**总计**：`src/` 共 **71,179 行**（47 个文件），其中算法核心 57,084 行、
基础设施 6,230 行、命令层 7,865 行。

## 三、当前外部依赖（`cargo tree --depth 1`，默认构建）

| crate | 用途 | 触及面 | 结论 |
|---|---|---|---|
| `tiny_http` | `serve` 的 HTTP/1.1 服务 | `serve.rs` | **保留**（基础设施：HTTP 协议不是 LLM 算法） |
| `clap` | CLI 子命令解析与帮助 | `cli.rs`（derive） | **保留**（基础设施：参数解析与算法无关） |
| `serde` + `serde_json` | 配置、checkpoint 头、分词器文件、OpenAI 协议 JSON | 全仓库 | **保留**（基础设施：JSON 编解码不是算法） |
| `rayon` | 逐元素/逐行算子的多核分摊 | 热路径 | **保留**：自研线程池在负载不均时几乎必然更慢，属于"不该自研" |
| `image`（可选） | PNG/JPEG 编解码与缩放 | 仅 `vision.rs` | **保留且可选**：`vision-io` 特性，默认构建不含它 |
| `wgpu` + `pollster`（可选） | GPU 后端 | `gpu` 特性 | 保留（可选依赖） |
| ~~`windows-sys`~~ | 平台调用 | 2 处 | **已移除**：`src/winutil.rs` 手写 2 个符号 + 16 字节布局测试 |

依赖树随特性变化：

| 构建方式 | 直接依赖 | 数量 |
|---|---|---|
| 默认（纯文本） | clap / rayon / serde / serde_json / tiny_http | **5** |
| `--features vision-io` | 上面 5 个 + image | 6 |
| `--features gpu` | 上面 5 个 + wgpu / pollster | 7 |

**没有任何一个依赖参与数学计算**——这是本项目的依赖边界。

## 四、决策记录

| 决定 | 日期 | 理由 |
|---|---|---|
| `windows-sys` 移除，改手写 FFI（`src/winutil.rs`） | 2026-11 | 只用到 2 个符号，手写更轻；且是"自己动手"的示范 |
| `tiny_http` **保留** | 2026-11 | HTTP 是基础设施，不属于 LLM 功能板块；自研只会引入协议 bug |
| `clap` / `serde` / `serde_json` **保留** | 2026-11 | 同上：CLI 解析、JSON 编解码与算法无关 |
| `rayon` **保留** | 2026-11 | 并行调度是基础设施；自研线程池在负载不均时几乎必然更慢 |
| `image` **保留且已可选化** | 2026-11 | PNG/JPEG 编解码不属于算法；已做成 `vision-io` 特性 |
| `wgpu` / `pollster` **保留（可选特性）** | 2026-11 | GPU 后端是加速手段，默认不引入 |
| 合成语料生成从 Python 脚本改为 Rust 子命令 `sft-gen` | 2026-11 | 守住"仓库只有 Rust"；同种子产物逐字节可复现 |
| `response_format: json_object` 用自研约束解码 | 2026-11 | 不引入 JSON 语法库；前沿自动机 + 词表掩码，见 `src/json_constraint.rs` |

## 五、怎么验证「确实是自研」

**自动化约束**（`tests/dependency_policy.rs`，4 条测试）：

1. `Cargo.toml` 的依赖必须在白名单内，且白名单里的每一项都必须真的被声明（防文档漂移）；
2. `src/` 里不允许出现任何深度学习框架 / 张量库 / 模型加载器 / 分词器库的引用
   （`candle`、`tch`、`ndarray`、`burn`、`tokenizers`、`safetensors`、`ort` …）；
3. **算法模块**只允许 `std` + `rayon` + `serde*`——把数学计算外包给别的库会直接让测试失败；
4. 本文档与 `Cargo.toml` 必须同步（白名单里的 crate 都要在本文档登记）。

**人工核查**：

```bash
cargo tree --depth 1          # 默认构建只剩 5 个直接依赖，全部是基础设施
cargo test --release          # 504 单元 + 10 CLI 冒烟 + 8 策略 + 29 HTTP（+1 ignored）+ 2 smoke = 553
cargo test --release --features gpu    # GPU 路径全量（在本机为 500+ 条）
```

各算法模块的测试数（`cargo test -- --list` 统计）：

| 模块 | 测试数 | 模块 | 测试数 | 模块 | 测试数 |
|---|---|---|---|---|---|
| `tensor` | 51 | `quant` | 34 | `attention` | 27 |
| `model` | 24 | `sample` | 22 | `tokenizer` | 20 |
| `train` | 20 | `moe` | 18 | `distributed` | 18 |
| `checkpoint` | 18 | `align` | 17 | `rope` | 16 |
| `data` | 15 | `layers` | 13 | `scaling` | 13 |
| `csa` | 13 | `optim` | 12 | `hf` | 10 |
| `mhc` | 10 | `fp8` | 10 | `rag` | 9 |
| `fp4` | 9 | `prompt` | 9 | `speculative` | 8 |
| `comm` | 8 | `distill` | 7 | `config` | 7 |
| `loss` | 6 | `vqvae` | 5 | `json_constraint` | 5 |
| `autograd` | 4 | `sftgen` | 4 | `rng` | 4 |
| `vision` | 3 | `gpu` | 0 | `module` | 0 |
| `lib` | 0 | — | — | — | — |

库测试共 **516 条**（504 条默认运行 + 12 条 `#[ignore]` 性能探针；
其中 serve 的 41 条在 `serve.rs` 内部，另有 29 条走真实 socket 的集成测试）。

## 六、文档位置约定

- `docs/` **只放编号课程**（`NN-课程名.md`，`00-学习计划.md` 是总纲）；
- 工程元文档放在**仓库根目录**：`README.md`（总览与实验记录）、`DEPENDENCIES.md`（本文）。
  不设"任务清单"文件——已完成项由 README 的历史记录承载，待办项写进 README 的对应小节，
  避免多一份会漂移的文档。
- `tests/doc_layout_policy.rs` 断言 `docs/` 里不出现非编号文件——想加课程就按编号命名，
  想加工程文档就放根目录。
