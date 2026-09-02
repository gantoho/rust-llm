//! GPU 计算后端（第 21 课）
//!
//! 用 wgpu 计算着色器（WGSL）加速最耗时的算子：矩阵乘、逐元素缩放/相加/ReLU。
//! - 仅在 `--features gpu` 时编译（`Cargo.toml` 中 `wgpu` 为可选依赖）
//! - 支持 NVIDIA 与 Intel 核显（Windows 下走 DX12 / Vulkan）
//! - 初始化失败或某次调用失败时，调用方自动回退 CPU，不影响训练/推理正确性
//!
//! 用法（main.rs 启动时）：
//! ```ignore
//! gpu::init();
//! if gpu::is_available() { println!("GPU: {}", gpu::name()); }
//! ```

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use pollster::block_on;

/// matmul 最小规模阈值：FLOPs = 2·m·k·n·batch 低于该值时走 CPU。
/// GPU 一次 dispatch 的固定开销（上传/调度/同步/下载）对小矩阵而言超过计算本身，
/// 训练中 scores/attn 这类微型矩阵乘直接走 CPU 反而更快；
/// 只有足够大的矩阵（如 QKV 投影、MLP、512×512 基准）才值得上 GPU。
pub const MATMUL_MIN_FLOPS: usize = 200_000;

/// softmax 最小规模阈值（元素数）：低于该值时走 CPU。
/// 一次 GPU dispatch 固定 ~10ms 开销，元素太少不划算（推理单 token 的 softmax 走 CPU）。
const SOFTMAX_MIN_ELEMS: usize = 200_000;

// 分流统计：实际走 GPU 的次数 / 回退 CPU 的次数（含未启用 GPU 或尺寸不足）
static STATS_GPU: AtomicUsize = AtomicUsize::new(0);
static STATS_CPU: AtomicUsize = AtomicUsize::new(0);

/// matmul 分流统计：(走 GPU 次数, 走 CPU 次数)，用于训练结束后向用户说明利用率。
pub fn stats() -> (usize, usize) {
    (
        STATS_GPU.load(Ordering::Relaxed),
        STATS_CPU.load(Ordering::Relaxed),
    )
}

/// buffer 池的三种用途（决定 usage 与归还键）
const KIND_IN: u8 = 0; // 输入：STORAGE | COPY_DST（write_buffer 上传）
const KIND_OUT: u8 = 1; // 输出：STORAGE | COPY_DST | COPY_SRC（再拷到 readback）
const KIND_READ: u8 = 2; // 读回：COPY_DST | MAP_READ（同步取回结果）

/// 计算着色器源码（WGSL）。
/// 统一用一个参数块 `params: Params`（6 个 u32，共 24 字节）传参：
/// - matmul：m / k / n / batch / a 转置 / b 转置
/// - scale：len / 标量的 f32 位模式
/// - add / relu：len
const SHADER: &str = r#"
// 注意：uniform 地址空间中数组 stride 必须 16 字节对齐，
// 故参数不用 array<u32,6>（会被摊到 96 字节），而是 6 个独立 u32 字段（共 24 字节）。
struct Params {
    p0: u32, // batch（scale/add/relu 时 = len）
    p1: u32, // m
    p2: u32, // k
    p3: u32, // n
    p4: u32, // a 转置标志（1 = 物理 a 是 [B,K,M]）
    p5: u32, // b 转置标志（1 = 物理 b 是 [B,N,K]）
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// 批量矩阵乘：out[B,M,N] = a[B,M,K] @ b[B,K,N]（B=1 时退化为普通 2D 矩阵乘）
// tiled 实现：16×16 workgroup，每次把 A/B 的 16×16 块载入共享内存再算内积，
// 把 K 维的全局内存访问次数从 K 次降到 K/16 次（naive 版每线程一元素、逐 K 读全局）。
// 注意：全局 storage 变量已用 a / b，局部批次计数命名为 nb，避免遮蔽数组 b。
// 转置访问（p4/p5）：反向传播的 ∂a = g @ bᵀ、∂b = aᵀ @ g 直接按转置读物理矩阵，
// 免去在 CPU 上构造 52 万~210 万元素的转置矩阵再上传的开销。
//   逻辑 A_l[i][j] = 物理 a[j][i]（物理布局 [B,K,M]）→ sh_a 读 a[acol * m + row]
//   逻辑 B_l[i][j] = 物理 b[j][i]（物理布局 [B,N,K]）→ sh_b 读 b[col * k + brow]
// 共享内存 tile（16×16 f32 = 1KB 各一块），每线程负责 1 个输出元素。
// WGSL 规定 workgroup 地址空间变量必须声明在模块作用域。
var<workgroup> sh_a: array<f32, 256>;
var<workgroup> sh_b: array<f32, 256>;

@compute @workgroup_size(16, 16, 1)
fn matmul_main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let nb = params.p0;
    let m = params.p1;
    let k = params.p2;
    let n = params.p3;
    let batch = gid.z;
    let row = gid.x;
    let col = gid.y;
    // 注意：不能在这里 `if (row >= m || col >= n) return;` 提前退出——
    // dispatch 向上取整后（如 n=35 → col 到 47），同一 workgroup 内部分线程会返回、
    // 其余线程继续执行 workgroupBarrier()，WGSL 要求 barrier 处控制流必须一致，这是未定义行为。
    // 因此所有线程走完相同的 tile 循环：越界读用安全下标取有效数据，写回时再按 (row,col) 保护。
    let safe_row = min(row, m - 1u); // 越界线程退化为最后一行（计算无用功，但不写回）
    let safe_col = min(col, n - 1u);
    let off_a = batch * m * k;
    let off_b = batch * k * n;

    // 线程布局：gid.x→输出行 row，gid.y→输出列 col，故 row 的低 4 位是 lid.x、col 的低 4 位是 lid.y。
    // tid 按 (lid.x, lid.y) 排布（注意不是 (lid.y, lid.x)！），使共享内存 tile 的"行"维正好对齐 A 的行：
    //   sh_a[lid.x][lid.y] = A[row][tb*16 + lid.y]
    //   sh_b[lid.x][lid.y] = B[tb*16 + lid.x][col]
    let tid = lid.x * 16u + lid.y;

    var acc = 0.0;
    let tiles = (k + 15u) / 16u;
    for (var tb = 0u; tb < tiles; tb = tb + 1u) {
        let acol = tb * 16u + lid.y; // 本块 A 的列下标
        let brow = tb * 16u + lid.x; // 本块 B 的行下标
        // 越界补 0（K 不是 16 的倍数时）
        if (acol < k) {
            if (params.p4 == 1u) {
                sh_a[tid] = a[off_a + acol * m + safe_row];
            } else {
                sh_a[tid] = a[off_a + safe_row * k + acol];
            }
        } else {
            sh_a[tid] = 0.0;
        }
        if (brow < k) {
            if (params.p5 == 1u) {
                sh_b[tid] = b[off_b + safe_col * k + brow];
            } else {
                sh_b[tid] = b[off_b + brow * n + safe_col];
            }
        } else {
            sh_b[tid] = 0.0;
        }
        workgroupBarrier();
        for (var i = 0u; i < 16u; i = i + 1u) {
            acc = acc + sh_a[lid.x * 16u + i] * sh_b[i * 16u + lid.y];
        }
        workgroupBarrier();
    }
    if (batch < nb && row < m && col < n) {
        out[batch * m * n + row * n + col] = acc;
    }
}

// 掩码 softmax（最后一维，右后缀掩码广播）：out[r,j] = softmax(x[r,j] + mask[mb+j])，
// 其中 mb = (r*d) % mask_numel（mask 是输入形状的精确右后缀，逐行对齐）。
// 每线程负责一行：先扫出行最大值做数值稳定，再 exp+归一化。
// 训练里 scores 是 [B*H,T,T_total]（~200 万元素），exp 逐元素调用是 CPU 上的大热点。
@compute @workgroup_size(256, 1, 1)
fn softmax_fwd_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = params.p0;
    let d = params.p1;
    let mn = params.p2;
    let r = gid.x;
    if (r >= rows) {
        return;
    }
    let base = r * d;
    let mb = base % mn;
    var maxv = a[base] + b[mb];
    for (var j = 1u; j < d; j = j + 1u) {
        maxv = max(maxv, a[base + j] + b[mb + j]);
    }
    var sum = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        let e = exp(a[base + j] + b[mb + j] - maxv);
        out[base + j] = e;
        sum = sum + e;
    }
    let inv = 1.0 / sum;
    for (var j = 0u; j < d; j = j + 1u) {
        out[base + j] = out[base + j] * inv;
    }
}

// 掩码 softmax 反向：dx[r,j] = p[r,j] * (g[r,j] - Σ_j p·g)（p 为前向概率，与掩码无关）
@compute @workgroup_size(256, 1, 1)
fn softmax_bwd_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let rows = params.p0;
    let d = params.p1;
    let r = gid.x;
    if (r >= rows) {
        return;
    }
    let base = r * d;
    var dot = 0.0;
    for (var j = 0u; j < d; j = j + 1u) {
        dot = dot + a[base + j] * b[base + j]; // g · p
    }
    for (var j = 0u; j < d; j = j + 1u) {
        out[base + j] = b[base + j] * (a[base + j] - dot);
    }
}

// 逐元素缩放：out[i] = a[i] * s
@compute @workgroup_size(256, 1, 1)
fn scale_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let i = gid.x;
    if (i >= len) {
        return;
    }
    let s = bitcast<f32>(params.p1);
    out[i] = a[i] * s;
}

// 逐元素相加：out[i] = a[i] + b[i]
@compute @workgroup_size(256, 1, 1)
fn add_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let i = gid.x;
    if (i >= len) {
        return;
    }
    out[i] = a[i] + b[i];
}

// ReLU：out[i] = max(a[i], 0)
@compute @workgroup_size(256, 1, 1)
fn relu_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let i = gid.x;
    if (i >= len) {
        return;
    }
    out[i] = max(a[i], 0.0);
}
"#;

/// GPU 上下文：持有设备、队列与编译好的计算管线
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    matmul_pipe: wgpu::ComputePipeline,
    matmul_layout: wgpu::BindGroupLayout,
    scale_pipe: wgpu::ComputePipeline,
    relu_pipe: wgpu::ComputePipeline,
    unary_layout: wgpu::BindGroupLayout,
    add_pipe: wgpu::ComputePipeline,
    add_layout: wgpu::BindGroupLayout,
    softmax_fwd_pipe: wgpu::ComputePipeline,
    softmax_bwd_pipe: wgpu::ComputePipeline,
    /// 复用的参数 uniform buffer（24 字节，每次 write_buffer 覆盖）
    params_buf: wgpu::Buffer,
    /// 存储/读回 buffer 池：键 = (字节数, 用途)，按需取还，避免每算子新建 GPU 对象
    /// （用 Mutex 保证 GpuContext 可放进静态 OnceLock）
    pool: Mutex<HashMap<(u64, u8), Vec<wgpu::Buffer>>>,
}

/// 全局 GPU 上下文（初始化一次；None 表示不可用）
static GPU: OnceLock<Option<GpuContext>> = OnceLock::new();

/// 初始化 GPU 后端（幂等）。失败时静默置为不可用，后续自动走 CPU。
pub fn init() {
    let _ = GPU.set(create());
}

/// GPU 是否可用
pub fn is_available() -> bool {
    GPU.get().is_some_and(|g| g.is_some())
}

/// 适配器名称（设备型号）
pub fn name() -> String {
    GPU.get()
        .and_then(|g| g.as_ref())
        .map(|g| g.info.name.clone())
        .unwrap_or_default()
}

/// 后端类型（如 Vulkan / Dx12）
pub fn backend() -> String {
    GPU.get()
        .and_then(|g| g.as_ref())
        .map(|g| format!("{:?}", g.info.backend))
        .unwrap_or_default()
}

// ---------------- 公共算子（失败返回 None，调用方回退 CPU） ----------------

/// GPU 批量矩阵乘（行优先）：out[B,M,N] = a[B,M,K] @ b[B,K,N]；batch=1 即普通 2D。
/// `a_t`/`b_t` 为转置访问标志：为 true 时物理 a/b 分别是 [B,K,M]、[B,N,K]，
/// 内核按转置读取（反向传播的 ∂a = g @ bᵀ、∂b = aᵀ @ g 直接复用，免构造转置矩阵）。
pub fn matmul(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    a_t: bool,
    b_t: bool,
) -> Option<Vec<f32>> {
    let r = GPU
        .get()
        .and_then(|g| g.as_ref())
        .and_then(|g| g.matmul(a, b, m, k, n, batch, a_t, b_t));
    if r.is_some() {
        STATS_GPU.fetch_add(1, Ordering::Relaxed);
    } else {
        STATS_CPU.fetch_add(1, Ordering::Relaxed);
    }
    r
}

/// GPU 逐元素缩放：out[i] = a[i] * s
pub fn scale(a: &[f32], s: f32) -> Option<Vec<f32>> {
    GPU.get().and_then(|g| g.as_ref()).and_then(|g| g.scale(a, s))
}

/// GPU 逐元素相加：out[i] = a[i] + b[i]（要求等长）
pub fn add(a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
    GPU.get().and_then(|g| g.as_ref()).and_then(|g| g.add(a, b))
}

/// GPU ReLU：out[i] = max(a[i], 0)
pub fn relu(a: &[f32]) -> Option<Vec<f32>> {
    GPU.get().and_then(|g| g.as_ref()).and_then(|g| g.relu(a))
}

/// GPU 掩码 softmax（最后一维，右后缀掩码广播）：
/// out[r,j] = softmax(x[r,j] + mask[(r*d) % mask_numel + j])。
/// 元素数不足（`SOFTMAX_MIN_ELEMS`）或失败时返回 None，调用方回退 CPU。
pub fn softmax_mask(
    x: &[f32],
    mask: &[f32],
    rows: usize,
    d: usize,
    mask_numel: usize,
) -> Option<Vec<f32>> {
    GPU.get()
        .and_then(|g| g.as_ref())
        .and_then(|g| g.softmax_mask(x, mask, rows, d, mask_numel))
}

/// GPU 掩码 softmax 反向：dx[r,j] = p[r,j] * (g[r,j] - Σ_j p·g)
pub fn softmax_mask_backward(g: &[f32], p: &[f32], rows: usize, d: usize) -> Option<Vec<f32>> {
    GPU.get()
        .and_then(|ctx| ctx.as_ref())
        .and_then(|ctx| ctx.softmax_mask_backward(g, p, rows, d))
}

// ---------------- 内部实现 ----------------

fn create() -> Option<GpuContext> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends: wgpu::Backends::PRIMARY,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });
    let adapter = block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
        apply_limit_buckets: false,
    }))
    .ok()?;
    let info = adapter.get_info();
    let (device, queue) = block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("llm_from_scratch"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::default(),
        experimental_features: wgpu::ExperimentalFeatures::default(),
        memory_hints: wgpu::MemoryHints::default(),
        trace: wgpu::Trace::Off,
    }))
    .ok()?;

    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("compute"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER)),
    });

    let make_pipeline =
        |device: &wgpu::Device, layout: &wgpu::PipelineLayout, entry: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry),
                layout: Some(layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };

    // matmul：a, b, out, params（4 个绑定）
    let matmul_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("matmul_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let matmul_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("matmul_pl"),
        bind_group_layouts: &[Some(&matmul_layout)],
        immediate_size: 0,
    });

    // 一元运算（scale / relu）：a, out, params（3 个绑定）
    let unary_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("unary_layout"),
        entries: &[storage_entry(0, true), storage_entry(2, false), uniform_entry(3)],
    });
    let unary_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("unary_pl"),
        bind_group_layouts: &[Some(&unary_layout)],
        immediate_size: 0,
    });

    // 二元运算（add）：a, b, out, params（4 个绑定）
    let add_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("add_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
        ],
    });
    let add_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("add_pl"),
        bind_group_layouts: &[Some(&add_layout)],
        immediate_size: 0,
    });

    let matmul_pipe = make_pipeline(&device, &matmul_pl, "matmul_main");
    let scale_pipe = make_pipeline(&device, &unary_pl, "scale_main");
    let relu_pipe = make_pipeline(&device, &unary_pl, "relu_main");
    let add_pipe = make_pipeline(&device, &add_pl, "add_main");
    // softmax 前向/反向共用 matmul 的 4 绑定布局（a/b/out/params）
    let softmax_fwd_pipe = make_pipeline(&device, &matmul_pl, "softmax_fwd_main");
    let softmax_bwd_pipe = make_pipeline(&device, &matmul_pl, "softmax_bwd_main");

    // 参数 uniform buffer 只建一次，全程复用（6 个 u32 = 24 字节）
    let params_buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("params"),
        size: 24,
        usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    Some(GpuContext {
        device,
        queue,
        info,
        matmul_pipe,
        matmul_layout,
        scale_pipe,
        relu_pipe,
        unary_layout,
        add_pipe,
        add_layout,
        softmax_fwd_pipe,
        softmax_bwd_pipe,
        params_buf,
        pool: Mutex::new(HashMap::new()),
    })
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

impl GpuContext {
    /// 批量矩阵乘（tiled：16×16 共享内存块，教学实现）。
    /// `a_t`/`b_t`：物理存储转置标志，见 [`matmul`]。
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        batch: usize,
        a_t: bool,
        b_t: bool,
    ) -> Option<Vec<f32>> {
        // wgpu 默认限制每维度最多 65535 个 workgroup，超出则回退 CPU
        if m == 0
            || k == 0
            || n == 0
            || batch == 0
            || (m + 15) / 16 > 65535
            || (n + 15) / 16 > 65535
            || batch > 65535
        {
            return None;
        }
        // 尺寸阈值：太小不值得上 GPU（固定调度开销 > 计算收益），回退 CPU
        if 2 * m * k * n * batch < MATMUL_MIN_FLOPS {
            return None;
        }
        let buf_a = self.take_buf(a.len(), KIND_IN);
        self.queue.write_buffer(&buf_a, 0, bytemuck_bytes(a));
        let buf_b = self.take_buf(b.len(), KIND_IN);
        self.queue.write_buffer(&buf_b, 0, bytemuck_bytes(b));
        let out_len = batch * m * n;
        let buf_out = self.take_buf(out_len, KIND_OUT);
        let r = self.run(
            &self.matmul_pipe,
            &self.matmul_layout,
            &[(&buf_a, 0), (&buf_b, 1), (&buf_out, 2)],
            [
                batch as u32,
                m as u32,
                k as u32,
                n as u32,
                a_t as u32,
                b_t as u32,
            ],
            &buf_out,
            out_len,
            ((m + 15) / 16) as u32,
            ((n + 15) / 16) as u32,
            batch as u32,
        );
        self.put_buf(buf_a);
        self.put_buf(buf_b);
        self.put_buf(buf_out);
        r
    }

    fn scale(&self, a: &[f32], s: f32) -> Option<Vec<f32>> {
        let buf_a = self.take_buf(a.len(), KIND_IN);
        self.queue.write_buffer(&buf_a, 0, bytemuck_bytes(a));
        let buf_out = self.take_buf(a.len(), KIND_OUT);
        let r = self.run(
            &self.scale_pipe,
            &self.unary_layout,
            &[(&buf_a, 0), (&buf_out, 2)],
            [a.len() as u32, s.to_bits(), 0, 0, 0, 0],
            &buf_out,
            a.len(),
            ((a.len() + 255) / 256) as u32,
            1,
            1,
        );
        self.put_buf(buf_a);
        self.put_buf(buf_out);
        r
    }

    fn relu(&self, a: &[f32]) -> Option<Vec<f32>> {
        let buf_a = self.take_buf(a.len(), KIND_IN);
        self.queue.write_buffer(&buf_a, 0, bytemuck_bytes(a));
        let buf_out = self.take_buf(a.len(), KIND_OUT);
        let r = self.run(
            &self.relu_pipe,
            &self.unary_layout,
            &[(&buf_a, 0), (&buf_out, 2)],
            [a.len() as u32, 0, 0, 0, 0, 0],
            &buf_out,
            a.len(),
            ((a.len() + 255) / 256) as u32,
            1,
            1,
        );
        self.put_buf(buf_a);
        self.put_buf(buf_out);
        r
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        if a.len() != b.len() {
            return None;
        }
        let buf_a = self.take_buf(a.len(), KIND_IN);
        self.queue.write_buffer(&buf_a, 0, bytemuck_bytes(a));
        let buf_b = self.take_buf(b.len(), KIND_IN);
        self.queue.write_buffer(&buf_b, 0, bytemuck_bytes(b));
        let buf_out = self.take_buf(a.len(), KIND_OUT);
        let r = self.run(
            &self.add_pipe,
            &self.add_layout,
            &[(&buf_a, 0), (&buf_b, 1), (&buf_out, 2)],
            [a.len() as u32, 0, 0, 0, 0, 0],
            &buf_out,
            a.len(),
            ((a.len() + 255) / 256) as u32,
            1,
            1,
        );
        self.put_buf(buf_a);
        self.put_buf(buf_b);
        self.put_buf(buf_out);
        r
    }

    /// 掩码 softmax（前向）：每线程处理一行。x [rows, d]，mask 右后缀广播。
    fn softmax_mask(
        &self,
        x: &[f32],
        mask: &[f32],
        rows: usize,
        d: usize,
        mask_numel: usize,
    ) -> Option<Vec<f32>> {
        if rows == 0 || d == 0 || mask_numel == 0 || rows * d < SOFTMAX_MIN_ELEMS {
            return None; // 太小：GPU 固定开销不划算（如推理单 token），回退 CPU
        }
        let total = rows * d;
        let buf_x = self.take_buf(total, KIND_IN);
        self.queue.write_buffer(&buf_x, 0, bytemuck_bytes(x));
        let buf_m = self.take_buf(mask_numel, KIND_IN);
        self.queue.write_buffer(&buf_m, 0, bytemuck_bytes(mask));
        let buf_out = self.take_buf(total, KIND_OUT);
        let r = self.run(
            &self.softmax_fwd_pipe,
            &self.matmul_layout,
            &[(&buf_x, 0), (&buf_m, 1), (&buf_out, 2)],
            [rows as u32, d as u32, mask_numel as u32, 0, 0, 0],
            &buf_out,
            total,
            ((rows + 255) / 256) as u32,
            1,
            1,
        );
        self.put_buf(buf_x);
        self.put_buf(buf_m);
        self.put_buf(buf_out);
        r
    }

    /// 掩码 softmax 反向：g 为输出梯度 [rows,d]，p 为前向概率 [rows,d]。
    fn softmax_mask_backward(&self, g: &[f32], p: &[f32], rows: usize, d: usize) -> Option<Vec<f32>> {
        if rows == 0 || d == 0 || rows * d < SOFTMAX_MIN_ELEMS {
            return None;
        }
        let total = rows * d;
        let buf_g = self.take_buf(total, KIND_IN);
        self.queue.write_buffer(&buf_g, 0, bytemuck_bytes(g));
        let buf_p = self.take_buf(total, KIND_IN);
        self.queue.write_buffer(&buf_p, 0, bytemuck_bytes(p));
        let buf_out = self.take_buf(total, KIND_OUT);
        let r = self.run(
            &self.softmax_bwd_pipe,
            &self.matmul_layout,
            &[(&buf_g, 0), (&buf_p, 1), (&buf_out, 2)],
            [rows as u32, d as u32, 0, 0, 0, 0],
            &buf_out,
            total,
            ((rows + 255) / 256) as u32,
            1,
            1,
        );
        self.put_buf(buf_g);
        self.put_buf(buf_p);
        self.put_buf(buf_out);
        r
    }

    /// 从池中取一个 buffer（没有就新建）。池按 (字节数, 用途) 键控，
    /// 训练中形状反复出现（QKV/MLP 都是 [B*T,D]×[D,D]），命中率很高。
    fn take_buf(&self, len: usize, kind: u8) -> wgpu::Buffer {
        let key = ((len * 4) as u64, kind);
        self.pool
            .lock()
            .unwrap()
            .get_mut(&key)
            .and_then(|v| v.pop())
            .unwrap_or_else(|| self.make_buf(len, kind))
    }

    /// 归还 buffer 回池。调用点都保证 GPU 已同步完成（poll wait 之后），可安全复用。
    fn put_buf(&self, buf: wgpu::Buffer) {
        let key = (buf.size(), kind_of(&buf));
        self.pool.lock().unwrap().entry(key).or_default().push(buf);
    }

    fn make_buf(&self, len: usize, kind: u8) -> wgpu::Buffer {
        let usage = match kind {
            KIND_READ => wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            KIND_OUT => {
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC
            }
            _ => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        };
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled"),
            size: (len * 4) as u64,
            usage,
            mapped_at_creation: false,
        })
    }

    /// 提交一次计算 dispatch 并同步取回结果。
    /// `bufs` 是 (buffer, 着色器 binding 编号) 对；参数 uniform 固定绑定在 binding 3。
    /// 复用 `params_buf` 与池化 readback buffer，不再每次创建 GPU 对象。
    #[allow(clippy::too_many_arguments)]
    fn run(
        &self,
        pipe: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bufs: &[(&wgpu::Buffer, u32)],
        params: [u32; 6],
        out_buf: &wgpu::Buffer,
        out_len: usize,
        x: u32,
        y: u32,
        z: u32,
    ) -> Option<Vec<f32>> {
        // [诊断] 临时计时分解（前几次调用）
        let diag_t = std::time::Instant::now();
        static DIAG_COUNT: AtomicUsize = AtomicUsize::new(0);
        let diag_n = DIAG_COUNT.fetch_add(1, Ordering::Relaxed);
        let diag_dump = diag_n < 6;

        self.queue
            .write_buffer(&self.params_buf, 0, bytemuck_bytes(&params));
        let diag_t_upload = diag_t.elapsed();

        // bind group（buffer 来自池、指针稳定，仍每次重建；开销远小于创建 buffer）
        let mut entries: Vec<wgpu::BindGroupEntry> = bufs
            .iter()
            .map(|(buf, binding)| wgpu::BindGroupEntry {
                binding: *binding,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
            })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &self.params_buf,
                offset: 0,
                size: None,
            }),
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg"),
            layout,
            entries: &entries,
        });

        // 录制并提交
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        let out_size = (out_len * 4) as u64;
        let readback = self.take_buf(out_len, KIND_READ);
        encoder.copy_buffer_to_buffer(out_buf, 0, &readback, 0, out_size);
        self.queue.submit([encoder.finish()]);
        let diag_t_submit = diag_t.elapsed();

        // 同步取回（教学简化：每次调用都等 GPU 完成，保证调用方拿到的数据可用）
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let diag_t_sync = diag_t.elapsed();
        let _ = self
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let view = match slice.get_mapped_range() {
            Ok(v) => v,
            Err(_) => {
                self.put_buf(readback);
                return None;
            }
        };
        let result = unsafe {
            std::slice::from_raw_parts(view.as_ptr() as *const f32, out_len).to_vec()
        };
        drop(view);
        readback.unmap();
        self.put_buf(readback);
        if diag_dump {
            let total = diag_t.elapsed();
            println!(
                "[diag] run#{} x={x} y={y} z={z} out={out_len} 总 {:.2}ms | 上传 {:.2}ms | 调度 {:.2}ms | 同步等待 {:.2}ms",
                diag_n,
                total.as_secs_f64() * 1000.0,
                diag_t_upload.as_secs_f64() * 1000.0,
                (diag_t_submit - diag_t_upload).as_secs_f64() * 1000.0,
                (total - diag_t_sync).as_secs_f64() * 1000.0,
            );
        }
        Some(result)
    }
}

/// 根据 buffer 的 usage 反查它在池中的用途键
fn kind_of(buf: &wgpu::Buffer) -> u8 {
    if buf.usage().contains(wgpu::BufferUsages::MAP_READ) {
        KIND_READ
    } else if buf.usage().contains(wgpu::BufferUsages::COPY_SRC) {
        KIND_OUT
    } else {
        KIND_IN
    }
}

/// 把任意 POD 数据当作字节切片（f32/u32 均为 4 字节小端）
fn bytemuck_bytes<T: Sized>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}

#[cfg(all(test, feature = "gpu"))]
mod tests {
    use super::*;

    /// CPU 三重循环参考实现（与 tensor.rs 的 matmul_data 相同的算法）。
    /// `a_t`/`b_t` 时按转置物理布局读：a 实际 [B,K,M]、b 实际 [B,N,K]。
    fn cpu_matmul(
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        batch: usize,
        a_t: bool,
        b_t: bool,
    ) -> Vec<f32> {
        let mut out = vec![0.0f32; batch * m * n];
        for bi in 0..batch {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for kk in 0..k {
                        let av = if a_t {
                            a[(bi * k + kk) * m + i]
                        } else {
                            a[(bi * m + i) * k + kk]
                        };
                        let bv = if b_t {
                            b[(bi * n + j) * k + kk]
                        } else {
                            b[(bi * k + kk) * n + j]
                        };
                        s += av * bv;
                    }
                    out[(bi * m + i) * n + j] = s;
                }
            }
        }
        out
    }

    /// GPU tiled matmul 与 CPU 参考实现逐元素对比（覆盖 16 倍数与非 16 倍数形状、
    /// 以及反向传播用到的转置访问组合）
    #[test]
    fn gpu_matmul_matches_cpu() {
        init();
        if !is_available() {
            return; // 无 GPU 环境跳过（不视为失败）
        }
        // (m, k, n, batch, a_t, b_t)
        let shapes: &[(usize, usize, usize, usize, bool, bool)] = &[
            (256, 64, 64, 1, false, false),   // 2D，全 16 倍数
            (256, 256, 64, 1, false, false),  // 2D 反向形状
            (32, 16, 32, 32, false, false),   // 3D 批量（注意力 scores）
            (32, 32, 16, 32, false, false),   // 3D 批量（attn·v）
            (32, 24, 40, 8, false, false),    // 非 16 倍数边界（demo_gpu 用）
            (100, 100, 100, 1, false, false), // 非 16 倍数
            (2048, 256, 256, 1, false, false), // 大矩阵（训练 QKV 投影规模）
            // 反向传播：∂a = g @ bᵀ（b 转置读）、∂b = aᵀ @ g（a 转置读）
            (32, 32, 16, 32, false, true),
            (32, 16, 32, 32, true, false),
            (2048, 256, 256, 1, false, true),
            (2048, 256, 256, 1, true, false),
            (64, 48, 36, 4, true, true), // 双重转置（防御性，正常反向不会同时转）
        ];
        for &(m, k, n, batch, a_t, b_t) in shapes {
            let a: Vec<f32> = (0..m * k * batch).map(|i| (i as f32 * 0.01).sin()).collect();
            let b: Vec<f32> = (0..k * n * batch).map(|i| (i as f32 * 0.013).cos()).collect();
            let g = matmul(&a, &b, m, k, n, batch, a_t, b_t)
                .unwrap_or_else(|| panic!("m={m} k={k} n={n} batch={batch} a_t={a_t} b_t={b_t} 应走 GPU"));
            let c = cpu_matmul(&a, &b, m, k, n, batch, a_t, b_t);
            let max_err = g
                .iter()
                .zip(&c)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            assert!(
                max_err < 1e-3,
                "m={m} k={k} n={n} batch={batch} a_t={a_t} b_t={b_t} 最大误差 {max_err}"
            );
        }
    }
}
