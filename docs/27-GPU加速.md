# 第 27 课：GPU 加速训练与推理

> 目标：用 wgpu 计算着色器（WGSL）把最耗时的算子搬到 GPU 上跑，
> 支持 NVIDIA 与 Intel 核显，同时保证"GPU 不可用时自动回退 CPU"。

## 1. 为什么选 wgpu

- **跨平台**：Windows 走 DX12 / Vulkan，NVIDIA 独显和 Intel 核显都能用；
- **纯 Rust**：不依赖 CUDA，也不引入深度学习框架；
- **计算着色器**：WGSL 语言手写算子，和写 CPU 的 for 循环思路一致，适合教学。

## 2. 架构总览

```
Cargo.toml          gpu feature（可选依赖 wgpu / pollster）
src/gpu.rs          GPU 上下文 + 4 个 WGSL 计算入口 + 同步取回
src/tensor.rs       matmul_data()：GPU 优先，失败回退 CPU
src/main.rs         gpu::init() + demo_gpu()（第 4 个演示）
```

- `--features gpu` 开启；默认零 GPU 依赖，构建轻量。
- 初始化用 `OnceLock<Option<GpuContext>>`：失败静默置 None，后续自动走 CPU。

## 3. WGSL 计算着色器

4 个计算入口共用同一个 ShaderModule（绑定声明是 module 级的）：

| 入口 | 计算 | 绑定 |
|------|------|------|
| `matmul_main` | `out[B,M,N] = a[B,M,K] @ b[B,K,N]` | 0:a 1:b 2:out 3:params |
| `scale_main` | `out[i] = a[i] * s` | 0:a 2:out 3:params |
| `add_main` | `out[i] = a[i] + b[i]` | 0:a 1:b 2:out 3:params |
| `relu_main` | `out[i] = max(a[i], 0)` | 0:a 2:out 3:params |

参数统一走 16 字节 uniform：`struct Params { p0: u32, p1: u32, p2: u32, p3: u32 }`
（f32 标量用 `bitcast<f32>` 位模式传参）。

matmul 用三维 workgroup：`@workgroup_size(16,16,1)`，`global_invocation_id`
的 x/y/z 分别对应行/列/batch。**tiled 版**：每个 workgroup 负责一个 16×16 输出块，
先把 A/B 的 16×16 小块载入共享内存（`var<workgroup> sh_a/sh_b: array<f32,256>`），
再在片内做内积，把 K 维的全局内存读取从 K 次降到 K/16 次。

> 坑：`sh_a` 的排布必须按 `(lid.x, lid.y)`（线程行对应输出行），若按习惯的
> `(lid.y, lid.x)` 会行列错乱，矩阵乘结果全错（训练 loss 卡住不下降）。
> 另外不能在 barrier 之前对越界线程 `return`——dispatch 向上取整后同一 workgroup
> 内控制流分歧会让 `workgroupBarrier()` 变成未定义行为，应"越界读补 0、写回再保护"。

## 4. 踩坑记录

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

## 5. 运行方式

```bash
# 普通运行（无 GPU，纯 CPU）
cargo run --release -- demo

# 开启 GPU 加速（含第 4 个演示：正确性 + 性能对比）
cargo run --release --features gpu -- demo

# 训练 / 推理加 --features gpu 即自动走 GPU
cargo run --release --features gpu -- train --config config.json
```

实测（NVIDIA GeForce MX150 / Vulkan）：

```
512x512 矩阵乘：CPU 506.7ms vs GPU 57.0ms（快 8.9x）
批量矩阵乘 CPU vs GPU 最大误差 2.86e-6
逐元素算子（scale/relu/add）验证：通过
```

> 注意：MX150 上 512×512 从 naive 的 57.6ms 只微降到 57.0ms——该规模下每次调用的
> 固定开销（上传/调度/同步取回）已接近计算时间，共享内存的收益被抵消。大模型训练
> （如 n_embd=256、block=128、batch=16）每步有几十次 GPU 矩阵乘**串行同步**，
> 固定开销会累积，低端 GPU 上仍然偏慢。教学实现以"清晰、可回退"优先，不追求极致吞吐。

## 6. 动手练习

1. 把 tiled 块从 16×16 改成 8×8 或 32×32，观察性能与占用率变化；
2. 把 LayerNorm 也写成 WGSL 着色器，减少 CPU↔GPU 往返；
3. 思考：为什么 GPU 矩阵乘没有"快 50x"？瓶颈在哪里（显存搬运、每次调用的同步取回）？
4. 进阶：把一次 forward/backward 的多次 dispatch 合并提交、只在最后同步一次（计算图）。
