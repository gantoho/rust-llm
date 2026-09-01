# 第 21 课：GPU 加速训练与推理

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

matmul 用三维 workgroup：`@workgroup_size(8,8,1)`，`global_invocation_id`
的 x/y/z 分别对应行/列/batch，每线程算一个输出元素（naive 教学版）。

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
512x512 矩阵乘：CPU 338.4ms vs GPU 105.5ms（快 3.2x）
批量矩阵乘 CPU vs GPU 最大误差 2.86e-6
逐元素算子（scale/relu/add）验证：通过
```

## 6. 动手练习

1. 给 matmul 换成"每线程算 8 个元素"的版本，观察性能变化；
2. 把 LayerNorm 也写成 WGSL 着色器；
3. 思考：为什么 naive GPU 矩阵乘只快 3x？瓶颈在哪里（显存搬运、同步取回）？
