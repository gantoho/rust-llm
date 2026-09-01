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

use std::sync::OnceLock;
use pollster::block_on;

/// 计算着色器源码（WGSL）。
/// 统一用一个参数块 `params: Params`（4 个 u32，共 16 字节）传参：
/// - matmul：m / k / n
/// - scale：len / 标量的 f32 位模式
/// - add / relu：len
const SHADER: &str = r#"
// 注意：uniform 地址空间中数组 stride 必须 16 字节对齐，
// 故参数不用 array<u32,4>（会被摊到 64 字节），而是 4 个独立 u32 字段（共 16 字节）。
struct Params {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
}

@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

// 批量矩阵乘：out[B,M,N] = a[B,M,K] @ b[B,K,N]（B=1 时退化为普通 2D 矩阵乘）
// 注意：全局 storage 变量已用 a / b，局部批次计数命名为 nb，避免遮蔽数组 b。
@compute @workgroup_size(8, 8, 1)
fn matmul_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let nb = params.p0;
    let m = params.p1;
    let k = params.p2;
    let n = params.p3;
    let batch = gid.z;
    let row = gid.x;
    let col = gid.y;
    if (batch >= nb || row >= m || col >= n) {
        return;
    }
    let off_a = batch * m * k;
    let off_b = batch * k * n;
    var sum = 0.0;
    for (var i = 0u; i < k; i = i + 1u) {
        sum = sum + a[off_a + row * k + i] * b[off_b + i * n + col];
    }
    out[batch * m * n + row * n + col] = sum;
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

/// GPU 批量矩阵乘（行优先）：out[B,M,N] = a[B,M,K] @ b[B,K,N]；batch=1 即普通 2D
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize, batch: usize) -> Option<Vec<f32>> {
    GPU.get()
        .and_then(|g| g.as_ref())
        .and_then(|g| g.matmul(a, b, m, k, n, batch))
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
    /// 批量矩阵乘（naive 每线程一元素，教学实现）
    fn matmul(
        &self,
        a: &[f32],
        b: &[f32],
        m: usize,
        k: usize,
        n: usize,
        batch: usize,
    ) -> Option<Vec<f32>> {
        // wgpu 默认限制每维度最多 65535 个 workgroup，超出则回退 CPU
        if m == 0
            || k == 0
            || n == 0
            || batch == 0
            || (m + 7) / 8 > 65535
            || (n + 7) / 8 > 65535
            || batch > 65535
        {
            return None;
        }
        let buf_a = self.storage_buf(a.len(), a, wgpu::BufferUsages::empty());
        let buf_b = self.storage_buf(b.len(), b, wgpu::BufferUsages::empty());
        let out_len = batch * m * n;
        let buf_out = self.empty_buf(out_len, wgpu::BufferUsages::COPY_SRC);
        self.dispatch(
            &self.matmul_pipe,
            &self.matmul_layout,
            &[&buf_a, &buf_b, &buf_out],
            &[0, 1, 2],
            [batch as u32, m as u32, k as u32, n as u32],
            &buf_out,
            out_len,
            ((m + 7) / 8) as u32,
            ((n + 7) / 8) as u32,
            batch as u32,
        )
    }

    fn scale(&self, a: &[f32], s: f32) -> Option<Vec<f32>> {
        let buf_a = self.storage_buf(a.len(), a, wgpu::BufferUsages::empty());
        let buf_out = self.empty_buf(a.len(), wgpu::BufferUsages::COPY_SRC);
        self.dispatch(
            &self.scale_pipe,
            &self.unary_layout,
            &[&buf_a, &buf_out],
            &[0, 2],
            [a.len() as u32, s.to_bits(), 0, 0],
            &buf_out,
            a.len(),
            ((a.len() + 255) / 256) as u32,
            1,
            1,
        )
    }

    fn relu(&self, a: &[f32]) -> Option<Vec<f32>> {
        let buf_a = self.storage_buf(a.len(), a, wgpu::BufferUsages::empty());
        let buf_out = self.empty_buf(a.len(), wgpu::BufferUsages::COPY_SRC);
        self.dispatch(
            &self.relu_pipe,
            &self.unary_layout,
            &[&buf_a, &buf_out],
            &[0, 2],
            [a.len() as u32, 0, 0, 0],
            &buf_out,
            a.len(),
            ((a.len() + 255) / 256) as u32,
            1,
            1,
        )
    }

    fn add(&self, a: &[f32], b: &[f32]) -> Option<Vec<f32>> {
        if a.len() != b.len() {
            return None;
        }
        let buf_a = self.storage_buf(a.len(), a, wgpu::BufferUsages::empty());
        let buf_b = self.storage_buf(b.len(), b, wgpu::BufferUsages::empty());
        let buf_out = self.empty_buf(a.len(), wgpu::BufferUsages::COPY_SRC);
        self.dispatch(
            &self.add_pipe,
            &self.add_layout,
            &[&buf_a, &buf_b, &buf_out],
            &[0, 1, 2],
            [a.len() as u32, 0, 0, 0],
            &buf_out,
            a.len(),
            ((a.len() + 255) / 256) as u32,
            1,
            1,
        )
    }

    /// 创建存储 buffer 并上传数据
    fn storage_buf(&self, len: usize, data: &[f32], extra: wgpu::BufferUsages) -> wgpu::Buffer {
        let buf = self.empty_buf(len, extra);
        self.queue.write_buffer(&buf, 0, bytemuck_bytes(data));
        buf
    }

    fn empty_buf(&self, len: usize, extra: wgpu::BufferUsages) -> wgpu::Buffer {
        self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("storage"),
            size: (len * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | extra,
            mapped_at_creation: false,
        })
    }

    /// 提交一次计算 dispatch 并同步取回结果。
    /// `bufs` 与 `bindings` 等长，指定每个存储 buffer 在着色器中的 binding 编号；
    /// 参数 uniform 固定绑定在 binding 3。
    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        pipe: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bufs: &[&wgpu::Buffer],
        bindings: &[u32],
        params: [u32; 4],
        out_buf: &wgpu::Buffer,
        out_len: usize,
        x: u32,
        y: u32,
        z: u32,
    ) -> Option<Vec<f32>> {
        // 参数 uniform buffer（16 字节）
        let params_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: 16,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&params_buf, 0, bytemuck_bytes(&params));

        // bind group
        let mut entries = Vec::with_capacity(bufs.len() + 1);
        for (i, buf) in bufs.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: bindings[i],
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: buf,
                    offset: 0,
                    size: None,
                }),
            });
        }
        entries.push(wgpu::BindGroupEntry {
            binding: 3,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &params_buf,
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
            let mut pass =
                encoder.begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        let out_size = (out_len * 4) as u64;
        let readback = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: out_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        encoder.copy_buffer_to_buffer(out_buf, 0, &readback, 0, out_size);
        self.queue.submit([encoder.finish()]);

        // 同步取回（教学简化：每次调用都等 GPU 完成）
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = self
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let view = slice.get_mapped_range().ok()?;
        let bytes: &[u8] = &view;
        let result = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, out_len).to_vec()
        };
        drop(view);
        readback.unmap();
        Some(result)
    }
}

/// 把任意 POD 数据当作字节切片（f32/u32 均为 4 字节小端）
fn bytemuck_bytes<T: Sized>(v: &[T]) -> &[u8] {
    unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, std::mem::size_of_val(v)) }
}
