# 第 27 课：GPU 加速 —— 手写 WGSL 计算着色器

> 代码位置：[src/gpu.rs](../src/gpu.rs)、[src/tensor.rs](../src/tensor.rs)（逐算子分流点）、
> [src/model.rs](../src/model.rs)（常驻链路接线）、[src/train.rs](../src/train.rs)（稳态诊断）
>
> 目标：用 wgpu 计算着色器（WGSL）把最耗时的算子搬到 GPU 上跑，
> 支持 NVIDIA 与 Intel 核显，同时保证「GPU 不可用时自动回退 CPU、数值不出错」。

---

## 1. 为什么选 wgpu

- **跨平台**：Windows 走 DX12 / Vulkan，NVIDIA 独显和 Intel 核显都能用；
- **纯 Rust**：不依赖 CUDA，也不引入深度学习框架（`wgpu`/`pollster` 是**可选**依赖，默认不编译）；
- **计算着色器**：WGSL 手写算子，和写 CPU 的 for 循环思路一致——每一行都能对着数学公式核对，改完还能拿 CPU 参考实现逐值比对。

本机标定用的是一块 **NVIDIA GeForce MX150**（3 个 SM、2GB 显存，Dx12 后端）。
它很小，所以本课所有实测数字都能一眼看出「什么值得上 GPU、什么不值得」。

---

## 2. 架构总览

```
Cargo.toml      wgpu / pollster 为可选依赖（feature = "gpu"）
src/gpu.rs      GPU 上下文 + 22 个 WGSL 计算入口 + 「逐算子」与「常驻录制」两条路径
src/tensor.rs   matmul / scale / add / relu / softmax 的分流点；
                accumulate_grad / external / external_scalar_loss —— 常驻路径回注梯度的入口
src/model.rs    TransformerBlock::forward、Transformer::forward_core 里的常驻链路接线
src/train.rs    第一步打印 dispatch 分解、结尾打印稳态与「matmul 分流：GPU x / CPU y」
```

- `--features gpu` 开启；**默认零 GPU 依赖**，构建轻量。
- 初始化用 `OnceLock<Option<GpuContext>>`：失败静默置 `None`，之后全部走 CPU。
- 适配器按 `DiscreteGpu` 优先挑选（本机枚举到 Intel UHD 620 / MX150 / Basic Render Driver，选中 MX150）。

---

## 3. 22 个 WGSL 计算入口

分成 **3 个 ShaderModule**（同一 module 内的 storage 绑定声明是全局的，见 [§9](#9-踩坑记录)）；
下面的分组与个数是写作时的快照，以 `src/gpu.rs` 里的 `const SHADER*` 实际内容为准：

### 3.1 `SHADER`（8 个）

| 入口 | workgroup | 计算 |
|------|-----------|------|
| `matmul_main` | 16×16 = 256 | `out[B,M,N] = a[B,M,K] @ b[B,K,N]`，输出 tile 128×128 |
| `matmul_small_main` | 8×8 = 64 | 同一数学，输出 tile 64×64（**当前默认**，见 [§5](#5-大小-tile从-128128-到-6464)） |
| `softmax_fwd_main` | 256 | 掩码 softmax（每行一个 workgroup） |
| `softmax_bwd_main` | 256 | `dS = P ⊙ (dP − Σ dP·P)` |
| `scale_main` | 256 | `out[i] = a[i] * s` |
| `add_main` | 256 | `out[i] = a[i] + b[i]` |
| `relu_main` | 256 | `out[i] = max(a[i], 0)` |
| `fma_main` | 256 | 纯 FMA 循环，只用来探测量卡的峰值算力（标定用） |

### 3.2 `SHADER_LM_CE`（1 个）

| 入口 | 计算 |
|------|------|
| `lm_ce_main` | 输出头融合交叉熵：`logits → log_softmax → CE → dlogits` 一次算完 |

### 3.3 `SHADER_ELEM`（13 个）

| 入口 | 用途 |
|------|------|
| `ln_fwd_main` / `ln_bwd_x_main` / `ln_bwd_gb_main` | LayerNorm 前向、对输入的梯度、对 γ/β 的梯度（RMSNorm 共用同一套内核，靠模式位切换） |
| `gelu_fwd_bias_main` / `gelu_bwd_main` | GELU（+bias）前向与反向 |
| `swiglu_fwd_main` / `swiglu_bwd_main` | SwiGLU 门控前向与反向 |
| `add_bias_main` | 加偏置（`use_bias = false` 的层不调它） |
| `bias_dropout_residual_main` | `out = c + mask·(x + b)` 三合一 |
| `dropout_bwd_main` | dropout 反向（用同一 seed **重算**掩码，前向/反向逐位一致） |
| `col_sum_main` | 按列求和（沿行归约），用于 bias 梯度与 softmax 分母 |
| `heads_split_main` / `heads_join_main` | `[B,T,H,D] ↔ [B·H,T,D]` 重排，**顺带做 RoPE 正/逆旋转** |

> **为什么单独写 `col_sum`**：WGSL 没有 `atomicAdd<f32>`（只有整数原子），
> 多 workgroup 直接往同一地址累加会写飞，所以改成「一个 workgroup 负责一列 + 共享内存树形归约」。

> **为什么把 heads_split 和 RoPE 合并**：重排本身是纯搬数据（访存型，GPU 不擅长），
> 但既然要把 Q/K 搬一遍，就顺手把旋转做掉，省一次 4.2MB 的往返。

参数统一走 16 字节对齐的 uniform：WGSL 里 uniform 数组的 stride 必须是 16 字节，所以参数不用 `array<u32, 6>`（会被摊成 96 字节），而是写成 6 个独立 u32 字段（共 24 字节）：

```wgsl
struct Params {
    p0: u32, // batch（scale/add/relu 时 = len）
    p1: u32, // m
    p2: u32, // k
    p3: u32, // n
    p4: u32, // a 转置标志（1 = 物理 a 是 [B,K,M]）
    p5: u32, // b 转置标志（1 = 物理 b 是 [B,N,K]）
}
```

（f32 标量用 `bitcast<f32>` 传位模式。不同着色器复用同一布局，个别字段暂时空着不用。）

---

## 4. 两条执行路径

这是本课最重要的一节：**同一个 GPU 后端里，粒度不同的两套跑法共存**。

### 4.1 逐算子路径（per-op）

一个算子 = 一次提交 + 一次 poll + 一次回读（`GpuContext::run`）。
matmul / scale / add / relu / 掩码 softmax 都走它，`tensor.rs` 里按 FLOPs 阈值分流。

代价是**每个算子都要付一次固定开销**。本机实测（`[gpu] dispatch 分解`，采样 50 次逐算子调用）：

```
操作数上传 0.0ms (0%) | 参数上传 0.0ms (0%) | 绑定+编码+提交 0.8ms (6%) | poll+回读 13.0ms (94%)
平均 13.9ms/次
回读细分：poll(Wait) 9.0ms (69%) | 拷回 Vec 4.0ms (31%) —— 回读带宽约 646 MB/s
```

也就是说：**九成以上的时间在等回读，而不是在算**。所以这条路只适合大形状——
小矩阵自动回退 CPU（阈值默认 `5e7` FLOPs，见 [§7](#7-环境变量)）。

### 4.2 常驻 / 批量录制路径（resident）

换一个粒度：**一个子层**（甚至整个 Block 栈）的算子全部录进**同一个 command encoder**，
中间张量全部留在显存，最后只提交一次、只回读边界张量。
反向同理——在显存里连续算完，只把边界梯度回读出来，由
[`Tensor::accumulate_grad`](../src/tensor.rs) / [`Tensor::external`](../src/tensor.rs) 注回计算图。

四条链路：

| 链路 | 入口 | 覆盖范围 | 回注的边界梯度 |
|------|------|---------|--------------|
| 输出头融合 CE | `gpu::lm_head_ce` | matmul → log_softmax → CE → dlogits | 2 项（d_hidden / d_weight） |
| 注意力子层 | `gpu::attn_layer_forward` | ln1 → QKV → RoPE → attn → c_proj → dropout → 残差 | 11 项 |
| 前馈子层 | `gpu::mlp_forward` | ln2 → linear1 → GELU/SwiGLU → linear2 → dropout → 残差 | GELU 7 项 / SwiGLU 9 项 |
| 整叠 Block | `gpu::stack_forward` | 上述两条链路 × n_layer，子层边界也留在显存 | 每层 16 项（SwiGLU 18 项） |

效果最直观的是输出头：logits 有 4096×8192 = 3355 万个数，逐算子版一步来回 **268MB（两次回读实测 355ms）**，
融合后只回读每行一个 f32 的 `row_loss`（约 16KB）。

**关键点：`attn_layer_forward` / `mlp_forward` 只替换「子层」**，调用方拿到结果后**必须继续往下走**
（注意力子层的结果要作为前馈子层的输入）。这一点曾经写错过，代价很大，见 [§9](#9-踩坑记录)。

**适用条件**（任一不满足就自动回退逐算子 → CPU）：无 KV cache、`base = 0`（**训练与评估的
no_grad 前向都走这里**）、GELU 或 SwiGLU MLP、形状与规模够大。

**评估也走常驻路径**：`Tensor::external` / `Tensor::external_scalar_loss` 原先无条件 `record`，
no_grad 下会把永不 backward 的条目登进 tape，所以 eval 只能走逐算子。给这两处的 record 加上
`grad_enabled()` 守卫（与普通算子的 `req()` 同口径）后，评估的前向与融合 CE 也进常驻链路。
同机短程 bench：**0.42 → 0.31 s/步**、总时长 21s → 15s、eval 的逐算子 matmul 分流 **580 → 0**、
`val 8.5975 ≈ 8.5977`（逐算子版）；训练步 fwd/bwd 67/173ms 无回归。

归一化用 LayerNorm 还是 RMSNorm、K/V 是不是 GQA 的头数，都**不需要让路**——两者在参数层面
就被抹平了：

- **RMSNorm**：和 LayerNorm 共用同一套归约内核，靠一个模式位切换。RMSNorm 是 LayerNorm
  在"μ≡0、无 β"下的特例，所以前向只需把第一趟归约从"求和"换成"求平方和"、第三趟的仿射项取 0，
  反向只需把 `m1` 置 0（`dγ` 的公式本来就一致，写出的 `dβ` 无人接收）。
- **GQA**：不动内核，而是在**权重**上做一次等价变换——把 K/V 的参数按
  [`repeat_kv`](../src/attention.rs) 的同一套头顺序展开成 `n_head` 份
  （[`expand_kv_head`](../src/attention.rs)），反向再把梯度按组折回
  （[`fold_kv_head_grad`](../src/attention.rs)）。展开后与标准 MHA 同形，整条常驻路径一行不用改；
  代价只是每步每层多上传 `(n_rep-1)` 份 K/V 权重（默认配置下 64KB 量级）。

整叠路径默认**关闭**，用 `LLM_GPU_STACK=1` 打开。它在数值上与逐子层路径完全一致
（同 seed 下 loss / val 逐位相同），实测也**更快**：同一配置 ABBA 两轮，
逐子层 `0.67 / 0.70 s/步`，整叠 `0.59 / 0.60 s/步`（详见 [§8](#8-实测mx150--dx12)）。
原因是**提交粒度**：逐子层路径按子层粒度走，每层的注意力、前馈各算一次提交（前向、反向都要各来一遍），
整叠把整叠的前向、反向各压成一次提交——省掉的是每次提交之后的 poll + 回读等待
（本机实测：逐子层约 `20.7ms/次 × 多次/步`，整叠一次提交 66~96ms 覆盖所有层）。

默认仍关闭，因为它的**准入条件更严**：整叠要求所有 Block 都满足常驻条件、整个模型走同一条
链路（单流 + 无 KV cache，训练与 no_grad 评估都可走），而逐子层是"哪个子层不满足就只回退那一个"。
GELU 与 SwiGLU 现在都已支持（每层分别回注 16/18 项参数梯度）——SwiGLU 配置同机对照实测
整叠 `0.44 s/步` vs 逐子层 `0.53 s/步`，约快 17%。
想复现对照实验或追求吞吐时设 `LLM_GPU_STACK=1`。

---

## 5. 大小 tile：从 128×128 到 64×64

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

大 tile 内核保留（`LLM_GPU_MM_SMALL=0` 可强制回去），因为它是这张对照表的另一半，删掉就没法复现结论了。

---

## 6. 数值正确性：哪些是逐位一致、哪些只是近似

**这一点比性能数字更重要**，因为它决定了「GPU 训练出来的模型能不能和 CPU 版本对得上」。

| 路径 | 与 CPU 的关系 | 原因 |
|------|--------------|------|
| 逐算子（matmul / softmax / scale / add / relu） | **逐位一致** | matmul 每个输出元素的 k 求和顺序与 CPU 完全相同；softmax 融合内核也是逐元素决定 |
| 大 tile vs 小 tile matmul | **逐位一致** | 只换并行粒度，不改累加顺序（`gpu_matmul_small_tile_matches_big_tile_bits` 守住） |
| MLP 常驻链路 | 逐位一致（本机实测） | 归约结构与 CPU 参考同序 |
| 注意力常驻链路 | **相对误差 < 1e-3** | 用了 col_sum / 共享内存树形归约，求和结构与 CPU 不同 |

端到端验证（同一 seed、dropout=0、block=512、batch=8、4 层、`data/alice.txt`）：

| 配置 | step 1 loss（GPU） | step 1 loss（CPU） |
|------|------------------|------------------|
| GQA 配置（**该行是"注意力常驻尚不支持 GQA"时的历史记录**，故当时该链路未生效） | 4.5354 | 4.5354（**逐位一致**） |
| MHA 配置（注意力常驻生效） | 4.5075 | 4.5169（差 0.2%） |

> 注意上表第一行的前提已经变了：GQA 现在**能**走注意力常驻路径（K/V 的头复制由调用方
> 把权重按头展开后喂进内核），所以"GQA ⇒ 本轮常驻链路不生效 ⇒ 逐位一致"不再成立——
> GQA 形态下会落进 1e-3 相对误差那一档。四种形态（LayerNorm / RMSNorm / GQA / RMSNorm+GQA）
> 现在都由 `test_gpu_resident_path_matches_loop_path` 按同一门槛守着。

也就是说：**想让 GPU 结果与 CPU 逐位对上，就把常驻链路整体让路**——把
`LLM_GPU_MATMUL_MIN_FLOPS` 设得比实际形状的 FLOPs 都大即可（见 [§7](#7-环境变量)）。
只要注意力常驻链路生效，差值就在 `1e-3` 相对误差量级，测试里的 `assert_close` 正是按这个门槛设的；
`test_gpu_resident_path_matches_loop_path` 把 LayerNorm / RMSNorm / GQA / RMSNorm+GQA
四种形态都纳入了同一条门槛的守护（同一份参数、同一批数据跑常驻与逐算子两遍，比 loss 与全部参数梯度）。

---

## 7. 环境变量

| 变量 | 作用 | 默认 |
|------|------|------|
| `LLM_GPU_MATMUL_MIN_FLOPS` | 覆盖分流阈值（FLOPs 低于它走 CPU） | `50000000` |
| `LLM_GPU_MM_SMALL` | `0` = 强制 128×128 大 tile；其他值 = 64×64 小 tile | 小 tile |
| `LLM_GPU_STACK` | 存在即启用整叠 Block 常驻路径 | 关 |
| `LLM_GPU_PROBE` | 录制第一步的全部 matmul 形状并逐形状回放，打印每形状吞吐 + 纯 FMA 峰值 | 关 |
| `LLM_GPU_ABLATE` | 逗号分隔算子类别名（`heads_split,heads_join,col_sum`），命中的只计数不提交，用步时差反推耗时 | 空 |
| `LLM_GPU_ABLATE_MM` | 逗号分隔形状谓词（`k<=32`、`n>=8192`、`m<=128`），命中的 matmul 只计数不提交 | 空 |

---

## 8. 实测（MX150 / Dx12）

配置：`n_embd=128 n_head=4 n_layer=4 block_size=512 batch_size=8`，BPE 词表 8192，`dropout=0`，
语料 `data/alice.txt`，25 步。

**先分清两个口径**，否则数字没法比：

- `[gpu] 稳态（末尾 20 批）` 里的"批"是**一次录制提交**，不是一步训练。
  逐子层路径一次提交 = 一个子层；整叠路径一次提交 = 整叠的前向（或反向）。
- `[train] step k/25` 那一行的 `st/s` 与 `tok/s` 是**整步**吞吐（含进程内的首步热身）。

同一配置下 ABBA 两轮（同一二进制、同一 seed，A→C→C→A）：

| 路径 | 总耗时 / 25 步 | s/步 | 整步 tok/s（进度行） | 稳态 ms/次提交 | 前向+反向 |
|------|---------------|------|---------------------|---------------|----------|
| 逐子层常驻 + 小 tile（默认） | 18s / 17s | **0.70 / 0.67** | 6766~7440 | 23.6~26.1（每子层） | ~490ms |
| 逐子层常驻 + 大 tile（`LLM_GPU_MM_SMALL=0`） | — | — | 6532~6814 | 26.4~29.0（每子层） | — |
| 整叠常驻（`LLM_GPU_STACK=1`） | 15s / 15s | **0.60 / 0.59** | 8113~8767 | 66.6~96.5（每整叠） | ~415ms |

四个运行的 `loss 8.9662 / val 9.0090` 完全一致，说明三条路径（含大小 tile）数值上等价。

> ⚠️ **别跨粒度比较 ms/批**：逐子层 24ms/次 vs 整叠 67ms/次看起来是整叠慢 3 倍，
> 但前者一步要提交多次、后者一步只提交 2 次。看**整步**才得到真实结论——整叠反而快约 12%（0.59 vs 0.67 s/步）。

单看 tile 的取舍，则以 [§5](#5-大小-tile从-128128-到-6464) 的**同进程交替**标定为准（端到端的跨进程对比会被 GPU 降频盖过）。

另外两点：

1. **进程启动有一次性开销**（首步的管线绑定、缓冲池分配等）。短跑（几步）时它会把总耗时淹没，
   所以判断快慢要看**稳态**那一行或进度行的 `st/s`，不能看「总耗时/步数」。
2. MX150 每次 dispatch 的固定开销约 10ms。手写 CPU 矩阵乘（朴素三重循环、无 SIMD/BLAS）
   实际吞吐只有约 3.5 GFLOPS，所以即使是 `n_embd=256` 的小模型，GPU 依然比 CPU 快
   （`bench` 子命令里 GPU 1900 tok/s vs CPU 1540 tok/s）。
   这一档的提升空间不在"把单个着色器写得更玄"，而在**减少 dispatch 次数**：把相邻算子的形状凑齐、
   用常驻显存路径一次提交算完一整段（见 §4.2 的路径设计与 §8 的实测数字）。

`bench` 子命令是稳定的对照点（固定模型/语料/种子，step 30 的 loss 恒为 `2.6754`）：

| 构建 | 训练吞吐 | 备注 |
|------|---------|------|
| `--release`（CPU） | ~1540 tok/s | 8 线程 |
| `--release --features gpu` | ~1900 tok/s | 该配置形状偏小，常驻路径大多不触发 |

---

## 9. 踩坑记录

1. **WGSL 变量遮蔽**：`let b = params.p0` 把全局 storage 数组 `b` 遮蔽成 u32，
   再写 `b[...]` 报 `Invalid access into expression`。局部变量改名即可。
2. **uniform 数组对齐**：uniform 地址空间数组 stride 必须 16 字节对齐，
   `array<u32,4>` 实际占 64 字节；改用 4 个独立 u32 字段（16 字节）最省事。
3. **绑定编号**：scale/relu 不用 binding 1，但声明仍是全局的；创建 bind group
   时必须显式指定 binding 编号（0/2/3），不能从 0 连续排。
4. **wgpu 30 API**：`PipelineLayoutDescriptor` 无 `push_constant_ranges`（用
   `immediate_size`）、`bind_group_layouts` 元素是 `Option<_>`、
   `PollType::Wait` 是带字段 struct、`get_mapped_range()` 返回 `Result`。
5. **沙箱限制**：Windows 上 GPU 驱动会写 `NVIDIA DXCache`、`D3DSCache` 等目录，
   受限环境需要放行，否则进程会被杀（程序自身会先打印完结果）。
6. **`workgroupBarrier` 之前不能 return**：dispatch 向上取整后，同一 workgroup
   内的控制流分歧会让 barrier 变成未定义行为。正确做法是「越界读补 0、写回再保护」。
7. **共享内存排布要按 `(lid.x, lid.y)`**：线程行对应输出行；按习惯写反了矩阵乘结果全错，
   表现为「loss 卡住不降」——这类错误不会 panic，只会让模型学不动。
8. **常驻路径只替换子层，不能提前 `return`**（本课最贵的一个 bug）：
   `TransformerBlock::forward` 里注意力常驻链路命中后直接 `return`，
   导致**每一层的 MLP 子层都被整段丢掉**，模型退化成「只有注意力」。
   它不报错、不 panic，只是 loss 对不上（step 1 差 0.023，10 步后差 0.11 且持续放大），
   是靠「CPU/GPU 同 seed 对拍」才抓到的。教训：**融合算子替换的是子层，不是整个 Block**。
9. **懒零填充会污染计时**：新建的 buffer 首次被写入前是「懒零填充」的，
   第一次 dispatch 会额外付一笔清零开销。标定前必须先把两侧各跑一遍预热。
10. **回读量本身会造成假象**：单步回读 1.46GB 时，91% 的时间在 `poll(Wait)` 里等，
    看起来像是「GPU 算得慢」，实际是「数据搬得多」。要判断内核本身的快慢，
    必须用 `matmul_throughput_probe`（一次提交内连跑同一 matmul，排除回读）。
11. **GPU 会降频**：连续满载后同一内核能慢 15%，两次独立运行的对比会被温度差盖过，
    所以 tile 对照必须**同进程交替**测（`mm_tile_ab_probe`）。
12. **按形状做消融实验有失效边界**：`LLM_GPU_ABLATE_MM=<形状谓词>` 能摘掉某个形状的 matmul 看步时差，
    但只对「输出不进后续依赖链」的形状成立——消融掉 m = 128 的权重梯度 matmul 后参数永远得不到更新，
    loss 直接变 NaN，整轮实验作废。
13. **跨粒度比较计时会得出相反的结论**：`[gpu] 稳态（末尾 20 批）` 的"批"是**一次录制提交**，
    逐子层路径一次提交 = 一个子层（约 24ms），整叠路径一次提交 = 一叠的前向（约 67ms）。
    直接比这两个数会得到"整叠慢 3 倍"的错误结论；换成**整步**（s/步、`st/s`）再看，
    整叠其实快约 12%。教训：**比性能之前先对齐口径**——单位不一致的对比比不做对比更危险。

---

## 10. 拓展方向

> 核心内容已全部实现，这里是进阶拓展。

1. 把 `matmul_small_main` 的 tile 从 64×64 改成 32×32 或 128×64，用 `mm_tile_ab_probe` 同进程交替标定，
   看趋势是否还成立（小 tile 赢在**占用率**，再小就可能输在访存**复用率**上）；
2. 给 `MLPEnum` 再加一条常驻链路（把 SwiGLU 的三次投影也做成融合内核），并补一个 `assert_close` 数值测试；
3. 把注意力常驻链路的归约结构改成与 CPU 同序，验证能否把它从「1e-3 近似」拉回**逐位一致**；
4. 用 `LLM_GPU_STACK=1` 与默认配置各跑一次同 seed 训练，
   验证两条路径的 loss 逐位相同，并用 `s/步` 解释"整叠为什么反而略快"（提示：数一数每步各提交了几次）；
5. 用小 tile 内核算一下这张卡的实测峰值（`LLM_GPU_PROBE` 会打印纯 FMA 峰值），
   对比 MX150 的理论 FP32 算力，看看利用率还差多少、差在哪。

---

## 11. 显存上限与申请护栏：放不下就回退，绝不 panic

> 与全课其它小节不同，这一节讲的不是「怎么更快」，而是「**怎么不崩**」：
> GPU 后端在把申请递给 wgpu 之前先自己判一次尺寸，放不下就沿既有的 `Option` 约定回退 CPU。
> 它来自一次真实崩溃（[§11.2](#112-起因一次真实崩溃)），判据由三个 `mem_guard_*` 单测守着。
>
> **放不下现在有两条出路**：大矩阵乘只要还能切出一块就**分块留在 GPU 上算**（第 12 章），
> 连最小块都放不下才真的回退 CPU。本节讲的是护栏本身（也就是「真的放不下」那一条），
> 分块是它的第二条出路——两条路的提示都只打一次，用户看输出就知道走的是哪条。

### 11.1 一次申请要过三类上限

池化 buffer 全部是**整块绑定**（`BufferBinding { size: None }`），所以除了 `create_buffer` 自己的
硬上限，还要过对应 binding 类型的上限。`GpuContext::alloc_limit(kind)` 取的就是二者的小值：

| 上限（来自 `device.limits()`） | 什么时候卡你 | 本机（Radeon RX 560 / DX12） |
|------|------|------|
| `max_buffer_size` | **任何**用途的 `create_buffer` 都躲不开 | **256 MiB —— 实际卡住那张 519.4 MiB 大表的就是它** |
| `max_storage_buffer_binding_size` | buffer 作为 **storage** 绑定（`KIND_IN` / `KIND_OUT`）时的上限 | 初始化时按 512 MiB 申请（`max(x, 512MiB)`），本机不是瓶颈 |
| `max_uniform_buffer_binding_size` | buffer 作为 **uniform** 绑定（`KIND_UNIFORM`，参数块只有 24 B）时的上限 | 不是瓶颈 |

用途到上限的映射写在一处：

- `KIND_READ`（只做拷贝目标与映射回读，**从不**作 storage / uniform 绑定）→ 只看 `max_buffer_size`；
- `KIND_UNIFORM` → `min(max_uniform_buffer_binding_size, max_buffer_size)`；
- `KIND_IN` / `KIND_OUT`（storage）→ `min(max_storage_buffer_binding_size, max_buffer_size)`。

> 那三个上限在**初始化时缓存一次**（`GpuContext` 的三个字段）。原因是护栏在每个算子入口都要查，
> 而 `device.limits()` 每次调用都要拷一份很大的 `Limits` 结构体出来——逐算子取会明显拖慢。

### 11.2 起因：一次真实崩溃

Qwen2.5-0.5B（vocab = 151936、n_embd = 896）的输出头 / 词嵌入整表是 `[vocab, n_embd]`，
也就是 `151936 × 896 × 4 = 544538624 B ≈ 519.4 MiB`。生成时算 logits 的那次 matmul 会把这整张表
当**操作数**上传（`GpuContext::matmul` 里的 `take_buf(b.len(), KIND_IN)`：物理布局 `[vocab, d]` 即
`[N, K]`，内核按 `b_t = true` 读），而它的 FLOPs（`2·rows·d·vocab`，rows = 1 时也有 2.7 亿）
远超分流阈值、必然尝试上 GPU —— 于是这次申请被原样递给 `create_buffer`，wgpu 的校验失败，
默认的 uncaptured error handler 当场 panic：

```text
cargo run --release --features gpu -- generate --ckpt checkpoints\qwen05\model.ckpt \
    --tokenizer checkpoints\qwen05\tokenizer.json --prompt "你可以做什么"

wgpu error: Validation Error
Caused by: In Device::create_buffer, label = 'pooled'
    Buffer size 544538624 is greater than the maximum buffer size (268435456)
```

`268435456` = 256 MiB，正是本机 `device.limits().max_buffer_size`；`544538624` 恰好是
64 KiB 的整数倍（8309 × 64 KiB），所以池的 size class 上取整这次没有额外放大它。
常驻显存版的输出头（`lm_head_ce`）也会整块 `upload` 同一张表，走的是同一条护栏 ——
**每个会把这表搬上显存的地方都从 `take_buf` 走**，所以护栏只需在一处把住。

### 11.3 判据：与 wgpu 的校验**同源**

```rust
fn alloc_limit(&self, kind: u8) -> u64            // 按用途取上限，并 min(max_buffer_size)（见 §11.1）
fn can_alloc(&self, len: usize, kind: u8) -> bool // fits(alloc_limit(kind), size_class(len * 4))
fn fits(limit: u64, bytes: usize) -> bool         // bytes <= limit
```

三点都是刻意的：

1. **按池里真正创建的字节数判**（`size_class` 上取整之后），而不是只看 `len * 4`。
   池对 >64 KiB 的申请按 64 KiB 对齐，只看元素字节数会漏掉最多 65535 B（近 64 KiB）—— 而边界上
   恰好是这点对齐量决定 `create_buffer` 会不会报校验错误（单测里的例子：`size_class(268435457) = 268435456 + 65536`）。
2. **用 `<=` 而不是 `<`**：wgpu 的判据是 `size > max_buffer_size` 才报错，**恰好等于上限是合法的**。
3. **判据只来自 `device.limits()`，不写死常量**：同一份代码在 1 GiB 上限的卡上照旧让这张表上 GPU
   （`mem_guard_fits_boundaries` 里有这条对照）——护栏只拦「放不下」，不放宽也不收紧其它行为。

### 11.4 回退路径与诊断提示

- `take_buf` 是 GPU 后端**唯一**的分配入口（池命中与新建都从它走）：超限直接返回 `None`，
  绝不把申请递到 wgpu——那边只会 panic，不会给调用方一个失败返回。
- 逐算子提交点 `run()` 还**单独**对回读暂存区（`KIND_READ`）判一次：即使某个调用点漏判了自己的操作数，
  也不会有超限 buffer 走到 `create_buffer`。
- `None` 一路沿既有约定回退 CPU（`gpu::matmul` → `tensor.rs` 的 CPU 实现），数值不变，只是慢。
  大矩阵乘在回退之前还会先试一次**分块**（第 12 章）：只有 `chunk_plan` 也返回 `None`
  （连最小块宽都放不下）时，才真的走到这条 CPU 回退。
- 初始化另有一道**下限检查**：连 4 KiB（池的最小 size class）都放不下的设备直接判为不可用。
  没有它，参数 uniform、占位 buffer 这类「必然放得下」的小申请也会被护栏拦掉，
  护栏就从「优雅回退」退化成「什么都干不了」。

同一条 generate 命令修复后 **exit 0**；而且输出头那一刀**已经不再撞护栏**了——它现在走第 12 章的
分块路径，打印的是分块提示（本轮实测原文见 [§12.5](#125-实测标定)）。护栏提示本身只在**真的回退**
（连分块都救不了）时出现，文本如下（其中的数字是护栏挡下那张 519.4 MiB 表的实测，提示只打一次）：

```text
[gpu] 显存护栏：申请 136134656 个 f32 = 519.4 MiB（size class 上取整后 519.4 MiB）超过设备上限 256.0 MiB → 该算子回退 CPU。（只提示一次；此刻 matmul 分流：GPU 0 / CPU 96）
```

- **每个进程只提示一次**（`MEM_GUARD_WARNED` 是个 `AtomicBool`，用 `swap(true)` 抢）：生成时同一张
  大表每个 token 都会撞上，逐次打印只会刷屏（分块提示由 `CHUNK_NOTE_WARNED` 同样只打一次）；
- 顺带打印**此刻的 matmul 分流计数**（进程内累计，含「低于 FLOPs 阈值本来就走 CPU」的调用点，
  见 `matmul_may_use_gpu`），用来一眼看清这一刻 GPU / CPU 各接了多少次 matmul；
- 打印完照常生成，输出的中文是正确的。

### 11.5 常驻录制路径的占位句柄

常驻路径（[§4.2](#42-常驻--批量录制路径resident)）的 `GpuRecorder::*` 是**录制**接口，几十个调用点
（`upload` / `matmul` / `softmax_mask` / `empty` / …）都返回 `GpuHandle` 而不是 `Option`。放不下时的做法是：

1. `oversize_handle(len)` 把**整批**标记为作废（`GpuBatch::oversize = true`），并返回一块 **4 KiB 的
   占位 buffer**（初始化时创建，尺寸由下限检查保证合法）；
2. 录制代码拿着这个句柄照常往下录——下游基于垃圾数据算出来的内容无所谓；
3. `submit_and_read` / `batch_finish` 见到 `oversize` 就**直接不提交**：encoder 连同里面那些读了垃圾
   数据的 dispatch 一起被丢弃，占位 buffer 从不写入、**永远不会被提交**；
4. 调用方拿到 `None`，回退 CPU 重算，数值不受影响。

这样就不必把几十个调用点的签名都改成 `Option`（改一处就要跟着改整条录制链）。
另一个必须**在写入前**早退的理由：往一块小 buffer 里 `write_buffer` 大块数据同样会 panic，
所以 `upload` 在 `take_buf` 失败时立刻返回占位句柄，根本走不到写入。

### 11.6 边界：这是「不崩」，不是「提速」

- 护栏**不改变能放下时的既有行为**：放得下就照旧上 GPU；
- 它兜的是「**某次申请确实放不下**」这种场景。放不下时先试**分块**（第 12 章）：只要还能切出
  ≥ 256 列的一块，这张表参与的矩阵乘就留在 GPU 上算，**不**回退 CPU；连最小块都放不下才真的回退，
  那时才付「整单在 CPU 上算」的代价；
- 所以它是**最后兜底**，不是性能方案；分块同样不是"全面加速"，只是把"回退"换成"慢一点"
  （边界见 [§12.6](#126-边界分块不是全面加速)）。
- ⚠️ **别把冷启动的墙钟当成每 token 成本**：`generate --max-new 4` 那条命令的墙钟约 **37 s**，
  但其中约 **27 s 是 GPU 初始化**（`request_device` + 管线编译）、约 **7.6 s 载入 ckpt**，
  真正生成只有约 **2 s**（本轮实测：日志里的总耗时 **36.9 s**，另用一条"加载 ckpt 前就失败"的命令
  单独量到 GPU 初始化 **27.2 s**）。早先这里写过"实测 Qwen2.5-0.5B 生成为 **~47 s/token**"，
  那是把这条墙钟除以 4 的**误读**——它量的是冷启动，不是每 token 的成本；长生成的真实成本是
  **0.41 s/token**（[§12.6](#126-边界分块不是全面加速)）。
- 三个单测把关：`mem_guard_fits_boundaries`（判据边界：`<=`、size class 上取整、真实崩溃尺寸）、
  `mem_guard_take_buf_rejects_oversize`（分配入口真的返回 `None`，且拒绝之后池与上下文完好）、
  `mem_guard_limits_match_device`（判据来自 `device.limits()`、`alloc_limit` 按用途取 min、
  0 个元素与极小张量在任何设备上都放得下）。

---

## 12. 整块放不下就分块：沿 N 切，结果逐位一致

> 第 11 章的护栏回答了「放不下怎么才能不崩」。这一章讲它的**第二条出路**：大矩阵乘整块放不下时，
> 不再整单回退 CPU，而是**沿 N（输出列）切成几块、逐块在 GPU 上算、再按列拼回**。
> 护栏本身一条都没改 —— 连最小块宽都放不下时，仍然照旧回退 CPU。

### 12.1 为什么切 N，不切 K

`out[M,N] = A[M,K]·B[K,N]` 三条轴都能切，实际只有 N 能用：

- **切 K 会改浮点累加顺序**：每个输出元素都是 `Σ_k` 的 f32 累加，浮点不满足结合律，分段算完再相加
  与整段一次算**不再逐位一致**；跨块还得多一轮归约（多一份显存、多一次 dispatch）。
- **切 N 不动 K 循环**：块边界只决定「这次算哪几列」，每个输出元素仍是**同一个内核、同一份 K 循环**、
  同一累加顺序 ⇒ 与整块一次算**逐位相同**。这条由 `gpu_matmul_chunked_matches_whole_bits` 守住：
  它用 `assert_eq!`（**不放任何容差**）比对「整块一次算」与「分块算」。
- 切 M 与切 N 同理（行与行之间本来就独立），但 M 是左操作数的行，切它只对「A 块自己就超限」有用 ——
  实现里并不在 M 上切，需要时切的是 **batch 维**（`[B,M,K]` 的最外层）：切它等于同时缩小 A 块与输出块。

### 12.2 两种 `b_t` 布局的切片代价

块宽 `w` 一定时，B 块的元素数恒为 `bc·k·w`，但**取这 w 列要付的代价取决于物理布局**：

| 情形 | 物理布局 | 这一块怎么取 | 额外拷贝 |
|------|---------|-------------|---------|
| 整块就放得下（`plan.is_whole`） | 任意 | **直接借整个 `b`** | **零** —— 同尺寸 buffer、同参数、同一次 dispatch，与加分块之前完全一致 |
| `b_t = true`、`batch = 1` | `[N,K]` | 列切片就是连续的 `w·k` 个元素，直接借位 | **零** |
| `b_t = true`、`batch > 1` | `[B,N,K]` | 每个 batch 一段连续行段 memcpy | `bc·k·w` |
| `b_t = false` | `[B,K,N]` | 列切片跨步 `N`，只能按 K 行逐段取 `w` 个元素 | `bc·k·w`（所有列块合起来 ≈ **一遍全表扫描**） |

最后一行要按 K 行一段段取，代价是「扫一遍表」；真实输出头走的其实是上面那行（`batch = 1` 的连续行段，
零拷贝借位）。而**两种布局共同的收获**是：B 块每个列块只上传一份，合起来仍只是整表一遍 ——
**不再需要先在 CPU 上把整张转置表物化出来**（那是改动前真正贵的一步，见 [§12.4](#124-配套改动转置视图免物化判据从严)）。

### 12.3 块宽：均分、最小列宽、块数上限，失败仍然回退

`chunk_plan(m, k, n, batch, limit, elem) -> Option<ChunkPlan { batch_chunk, cols }>` 是个**纯函数**
（单测 `chunk_plan_boundaries` 不需要 GPU），判据与分配入口 `take_buf` 的 `size_class` 上取整**同源**：

```rust
fn chunk_bufs_fit(bc, m, k, w, limit, elem)  // A[bc,m,k] + B[bc,k,w] + out[bc,m,w] 三块都要放得下
fn chunk_max_cols(bc, m, k, n, limit, elem)  // 这一档 batch 下能放下的最大列宽
```

- **按「池里真正创建的字节数」判**：`size_class(元素数 × 4)` 之后才和上限比。只看 `元素数 × 4` 会漏掉
  最后那几十 KiB 的上取整（> 64 KiB 按 64 KiB 对齐），于是「算得下、取不到」，分块白分一场。
- **先整批、后分档**：B 那块大表与 batch 无关，所以 `bc` 从 `batch` 往下试，只在 **A / 输出自己就超限**
  时才切 batch（真实形状 `[151936, 896]`、`batch = 1`，于是 `batch_chunk` 恒为 1）。
- **最小列宽 `MM_CHUNK_MIN_COLS = 256`**：块比这更窄时，每次 dispatch 的固定开销（编码 / 提交 / poll / 回读）
  会盖过计算本身，64 列的 tile 也盖不满。没有这条下限，`limit` 极小时会一路细分到「每块一列」
  （几万次 dispatch，比 CPU 还慢）。
- **块数上限 `MM_CHUNK_MAX = 256`**：最小列宽取 `w_lo = max(256, ⌈n/256⌉).min(n)` —— 真实形状下是
  **594 列**（`⌈151936/256⌉`），块数上限定住了最坏情况的 dispatch 次数。
- **均分而不是「满块 + 小尾巴」**：先算能放下的最大列宽 `w_max`（从 `limit/(k+m)/4` 估个起点，
  再用同源判据校正），再按 `cnt = ⌈n/w_max⌉` 反算 `base = n / cnt`、余数摊在前 `rem` 块上。
  因为 `n ≤ cnt·w_max ⇒ ⌈n/cnt⌉ ≤ w_max`，均分后每块必然仍放得下；好处是块宽只差 ≤ 1（池里只出现
  1~2 个 size class），也少一次零头 dispatch。真实形状的账（本机 256 MiB 上限）：
  `w_max = 74898`（此时 B 块的 `size_class` 恰好等于 256 MiB —— 判据是 `<=`，再多一列就顶出去），
  `cnt = ⌈151936/74898⌉ = 3` ⇒ **`[50646, 50645, 50645]`**。
- **失败即回退**：`chunk_plan` 返回 `None`（形状退化、上限为 0、连最小块宽都放不下）时，
  `GpuContext::matmul` 返回 `None` → 该算子按第 11 章的老约定回退 CPU。**护栏语义一个字没改**，
  只是多了一条出路。

### 12.4 配套改动：转置视图免物化（判据从严）

分块要真能救下输出头，还差一步：`hidden @ Wᵀ` 里的 `Wᵀ` 过去会在 `Tensor::matmul` 入口被
`contiguous()` 物化成一张 `[d, vocab]` 整表（519.4 MiB，按 strides 把整表 walk 一遍）—— 物化发生在
内核之前，分块再优雅也碰不到它。现在入口先判一次：

- [`Tensor::is_transposed_2d_view`](../src/tensor.rs)：**2 维**且 `strides == [1, shape[0]]`，
  也就是 `permute([1, 0])` 的产物。判据**从严**：连续张量唯一能满足它的是 1×1（那时转置是恒等，
  走哪条路结果都一样）；三维的 `permute([0, 2, 1])` 不在其列。
- [`gpu_would_use_matmul`](../src/tensor.rs)：`gpu::matmul_would_use_gpu` 的**不计数**版（判据与内核入口
  同源：GPU 可用 + FLOPs 阈值），只用来决定「要不要为内核省掉这次物化」；紧接着那次
  `matmul_may_use_gpu` 照旧记分流计数，不会把一次调用算成两次。

两条同时命中才把转置视图**原样**交给内核（内核本来就按 `b_t = true` 读物理 `[N,K]`）；**无 GPU /
低于 FLOPs 阈值时行为与改动前逐位一致**（照旧 `contiguous()` 并行 gather）。

> ⚠️ **踩坑：免物化之后，反向的 `b_t` 必须按前向取反**。`matmul_backward` 里 `∂a = g·bᵀ` 的内核标志
> 原先**写死 `b_t = true`**，那只在「右操作数已经被物化成 `[K,N]` 行主序」时成立。免物化后 b 的物理是
> `[N,K]`，`bᵀ` 正好是 `[K,N]` 行主序，内核必须按**非**转置读。写死 `true` 会把两种布局算串 ——
> 训练里 `hidden.matmul(&head.transpose())` 的反向梯度会错，而**前向完全正常**，只看 loss 曲线发现不了。
> 现在按前向 `b_t` 取反，`tensor::tests::test_matmul_transposed_rhs_view` 用「两种布局 × 前向 / 反向」
> 逐位对照把这条钉死。

### 12.5 实测标定

输出头那一刀到底该走 CPU 兜底还是分块 GPU：同一份权重、同一个 `Tensor::matmul` 入口，**只切换分派**
（CPU 兜底 = 把分流阈值抬到 `usize::MAX`；分块 GPU = 阈值归 0），逐 m 采样。形状就是真实模型
（`[vocab = 151936, d = 896]`，整表 519.4 MiB > 本机 256 MiB 上限）：

| m | CPU 兜底（物化转置表 + 20 线程三重循环） | 分块 GPU（3 块） | 比值 |
|---|---|---|---|
| 1 | 326.5 ms | 333.5 ms | 打平（CPU 略快 2%） |
| 8 | 366.9 ms | 300.2 ms | GPU 快 1.22× |
| 64 | 813.3 ms | 349.5 ms | **GPU 快 2.3×** |
| 512 | 953.9 ms | 690.0 ms | GPU 快 1.38× |

- **口径**：同一台机（Radeon RX 560 / DX12、20 逻辑核）、逐 m 采样；两条路的输出**最大差 0.00e0**
  （分块结果与 CPU 兜底逐位相同）。探针是 `gpu::tests::chunked_head_matmul_vs_cpu_probe`（`#[ignore]`）：
  `cargo test --release --features gpu chunked_head_matmul_vs_cpu_probe -- --ignored --nocapture`。
- **可复现性**：写这一章时复跑同一探针得到 m = 1：301.4 / 331.3 ms、m = 8：357.4 / 274.7 ms、
  m = 64：736.1 / 371.3 ms（1.98×）、m = 512：965.1 / 706.3 ms —— 绝对值随 GPU 降频与后台负载浮动
  （同一张卡连续满载可差 15%，见第 5 章），**结论不变**：m = 1 与 CPU 打平、m ≥ 8 分块占优、
  m = 64 差距最大、两条路逐位一致。
- **提示原文**（本轮实跑 `generate --max-new 4` 的输出，与「回退 CPU」那条一样只打一次）：

```text
[gpu] 分块 matmul：out[3,151936]（k=896, batch=1）放不下设备上限 256.0 MiB，沿 N 切成 3 块（最宽 50646 列）× 1 档 batch，逐块在 GPU 上算后按列拼回—— 本算子留在 GPU，没有整单回退 CPU。（只提示一次；此刻 matmul 分流：GPU 0 / CPU 96）
```

- **守着的单测**：`chunk_plan_boundaries`（纯函数边界：真实形状必须切成 3 块 `[50646, 50645, 50645]`、
  恰好放得下 ⇒ 单块且 `is_whole`、n = 501 时余数落在前块、A 自己超限 ⇒ 改切 batch、上限极窄 ⇒ 由
  「最小列宽 + 块数上限」挡下返回 `None`）、`gpu_matmul_chunked_matches_whole_bits`（分块 vs 整块
  **逐位相同**，覆盖两种 `b_t` 布局、`a_t`、batch 分档、块宽不落在 tile 整数倍上）。

### 12.6 边界：分块不是「全面加速」

- **m = 1 与 CPU 打平**：增量解码每个 token 都要把 519 MiB 权重**重传一遍 PCIe**，而 CPU 那边是
  20 线程直接吃内存带宽 —— 两条路都在搬同一张表，谁也快不了。分块的价值在 m ≥ 8 之后（预填充、评估、
  训练），m = 64 时是 2.3×。
- **长生成实测**（`--max-new 100`，上一轮同机实测）：约 **0.41 s/token**，分块前后一致 —— 瓶颈就是上面那次重传，
  不是这次矩阵乘本身。
- ⚠️ **别被 generate 的墙钟误导**：`generate --max-new 4` 的墙钟约 **37 s**，但其中约 27 s 是
  **GPU 初始化**（`request_device` + 管线编译）、约 7.6 s 载入 ckpt，真正生成只有约 2 s；
  同一命令不带 `--features gpu` 只要 **9.8 s**（走到「生成」8.14 s）。把 37 s 除以 4 得到的
  「9 s/token」是**误读**（第 11 章里早先写错的 `~47 s/token` 也是同一处误读）。
- **下一步**：**权重常驻显存**（别再每个 token 重传）+ 把多次 dispatch **合并 submit / poll**。
  范式已经有了 —— 第 [4.2](#42-常驻--批量录制路径resident) 节的常驻链路就是「一个子层只提交一次」，
  输出头在 `lm_head_ce` 上也有常驻版本；把这条思路接到大矩阵乘的权重上，才能把 m = 1 那一刀也拿下来。
