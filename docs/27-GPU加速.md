# 第 27 课：GPU 加速 —— 手写 WGSL 计算着色器

> 代码位置：[src/gpu.rs](../src/gpu.rs)（约 5800 行）、[src/tensor.rs](../src/tensor.rs)（逐算子分流点）、
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
src/gpu.rs      GPU 上下文 + 19 个 WGSL 计算入口 + 「逐算子」与「常驻录制」两条路径
src/tensor.rs   matmul / scale / add / relu / softmax 的分流点；
                accumulate_grad / external / external_scalar_loss —— 常驻路径回注梯度的入口
src/model.rs    TransformerBlock::forward、Transformer::forward_core 里的常驻链路接线
src/train.rs    第一步打印 dispatch 分解、结尾打印稳态与「matmul 分流：GPU x / CPU y」
```

- `--features gpu` 开启；**默认零 GPU 依赖**，构建轻量。
- 初始化用 `OnceLock<Option<GpuContext>>`：失败静默置 `None`，之后全部走 CPU。
- 适配器按 `DiscreteGpu` 优先挑选（本机枚举到 Intel UHD 620 / MX150 / Basic Render Driver，选中 MX150）。

---

## 3. 19 个 WGSL 计算入口

分成 **3 个 ShaderModule**（同一 module 内的 storage 绑定声明是全局的，见 [§9](#9-踩坑记录)）：

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

### 3.3 `SHADER_ELEM`（10 个）

| 入口 | 用途 |
|------|------|
| `ln_fwd_main` / `ln_bwd_x_main` / `ln_bwd_gb_main` | LayerNorm 前向、对输入的梯度、对 γ/β 的梯度 |
| `gelu_fwd_bias_main` / `gelu_bwd_main` | GELU（+bias）前向与反向 |
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
| 前馈子层 | `gpu::mlp_forward` | ln2 → linear1 → GELU → linear2 → dropout → 残差 | 7 项 |
| 整叠 Block | `gpu::stack_forward` | 上述两条链路 × n_layer，子层边界也留在显存 | 每层 16 项 |

效果最直观的是输出头：logits 有 4096×8192 = 3355 万个数，逐算子版一步来回 **268MB（两次回读实测 355ms）**，
融合后只回读每行一个 f32 的 `row_loss`（约 16KB）。

**关键点：`attn_layer_forward` / `mlp_forward` 只替换「子层」**，调用方拿到结果后**必须继续往下走**
（注意力子层的结果要作为前馈子层的输入）。这一点曾经写错过，代价很大，见 [§9](#9-踩坑记录)。

**适用条件**（任一不满足就自动回退逐算子 → CPU）：训练模式（无 KV cache、`base = 0`）、
经典风格的 GELU MLP（SwiGLU 由调用方让路）、形状与规模够大。

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

默认仍关闭，因为它的**准入条件更严**：整叠要求所有 Block 都满足常驻条件
（GELU MLP——各层共用同一份配置，所以一个 SwiGLU 就足以让整条路径放弃），
而逐子层是"哪个子层不满足就只回退那一个"。
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
| GQA 配置（测得时注意力常驻尚不支持 GQA，故该链路未生效） | 4.5354 | 4.5354（**逐位一致**） |
| MHA 配置（注意力常驻生效） | 4.5075 | 4.5169（差 0.2%） |

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
