//! GPU 计算后端（第 27 课）
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
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use pollster::block_on;

/// matmul 最小规模阈值：FLOPs = 2·m·k·n·batch 低于该值时走 CPU。
/// GPU 一次 dispatch 的固定开销（上传/调度/同步/下载）~10ms，CPU 做 256×256 矩阵乘只需 ~0.1ms。
/// 阈值 5000 万 FLOPs 适用于 n_embd=256 的小模型，让 QKV/MLP 投影（~2.68 亿 FLOPs）走 GPU；
/// 更小的注意力头内积（head_dim=32，~1700 万 FLOPs）仍走 CPU。
/// 用 AtomicUsize 而非 const：数值测试与性能对照需要临时把阈值压到 0，强制所有形状走 GPU。
pub static MATMUL_MIN_FLOPS: AtomicUsize = AtomicUsize::new(50_000_000);

/// matmul 输出 tile 边长（与 WGSL 里的 `MM_TILE` 必须一致）：grid = ceil(m/128) × ceil(n/128) × batch
const MM_TILE: usize = 128;

/// 小 tile matmul 的输出 tile 边长（与 WGSL 里的 `MM_TILE_S` 必须一致）。
/// 见 [`use_small_tile`]：n 很小或 workgroup 数太少时改用它，避免大 tile 掩掉大半算力。
const MM_TILE_S: usize = 64;

/// 大小 tile 的强制开关（标定用）：-1 = 走 [`use_small_tile`] 的启发式，0 = 强制大 tile，
/// 1 = 强制小 tile。环境变量 `LLM_GPU_MM_SMALL=0/1` 在 `init()` 里写进来一次；
/// 性能探针做「同一进程内交替测量」时也会临时改写它 —— 两次独立运行的 GPU 温度不同，
/// 温差带来的偏差能盖过 tile 本身的差异，只有同进程交替才比得准。
static MM_SMALL_FORCE: std::sync::atomic::AtomicI8 = std::sync::atomic::AtomicI8::new(-1);

/// 大小 tile 的分流判定（见 WGSL 里 `matmul_small_main` 的说明）。
///
/// **当前结论：一律用小 tile。** 同进程交替 A/B（`mm_tile_ab_probe`）在训练真实形状上逐一体测，
/// 64×64 在**每一个**形状上都不输给 128×128，多数还快一截：
///
/// | 形状 | 大 tile | 小 tile | 小/大 |
/// |---|---|---|---|
/// | PV fwd 512×512×32 b=32 | 103 GF/s | 182 GF/s | 0.56 |
/// | dV bwd 512×512×32 b=32 | 136 | 237 | 0.57 |
/// | QK^T fwd 512×32×512 b=32 | 106 | 177 | 0.60 |
/// | QKV/c_proj fwd 4096×128×128 | 137 | 192 | 0.71 |
/// | dW proj bwd 128×4096×128 | 159 | 158 | 1.00 |
/// | MLP w2 fwd 4096×512×128 | 242 | 305 | 0.79 |
/// | dX proj bwd 4096×128×128 | 128 | 172 | 0.74 |
/// | lm_head fwd 4096×128×8192 | 279 | 328 | 0.85 |
///
/// 原先的假设是「大 tile 共享内存复用更高，只该在 n 很小或 workgroup 数太少时才换小的」，
/// 数据把这个假设否掉了：连 n = 8192、完全没有 tile 浪费的形状，小 tile 也快 18%。
/// 真正的瓶颈是**单个 workgroup 太胖**——256 线程 / 约 100 个寄存器，一个 SM 只塞得下 2 个，
/// 于是每次 barrier 与每次全局 load 都无处躲；64 线程的工作组让同一张 SM 上能并存多得多的
/// 工作组，用一个工作组的访存去盖另一个的等待。所以这里直接把小 tile 定为默认。
///
/// 大 tile 内核保留（有 `LLM_GPU_MM_SMALL=0` 可强制回去），因为它是上面这张对照表的另一半，
/// 删掉就没法再复现结论了。
fn use_small_tile() -> bool {
    MM_SMALL_FORCE.load(Ordering::Relaxed) != 0
}

/// softmax 最小规模阈值（元素数）：低于该值时走 CPU。
const SOFTMAX_MIN_ELEMS: usize = 200_000;

/// 逐元素内核的工作组大小，必须与 `SHADER_ELEM` 里的 `@workgroup_size` 一致
const ELEM_WG: usize = 256;

/// 逐元素内核的 workgroup 数。
/// 单维上限 65535 由调用方（`mlp_forward` 的尺寸判定）保证，这里用 debug 断言兜住。
fn n_workgroups(len: usize) -> u32 {
    let n = len.div_ceil(ELEM_WG);
    debug_assert!(n <= 65535, "元素数 {len} 超出单维 workgroup 上限");
    n as u32
}

// 分流统计：实际走 GPU 的次数 / 回退 CPU 的次数（含未启用 GPU 或尺寸不足）
static STATS_GPU: AtomicUsize = AtomicUsize::new(0);
static STATS_CPU: AtomicUsize = AtomicUsize::new(0);

/// GPU dispatch 诊断记录（采集前 `DIAG_MAX` 次调用，在训练首步结束后统一打印）
struct GpuDispatchDiag {
    x: u32, y: u32, z: u32,
    out_len: usize,
    total_ms: f64,
    upload_ms: f64,
    dispatch_ms: f64,
    sync_ms: f64,
    /// sync 的子阶段：map_async 调用 / poll(Wait) / get_mapped_range / 拷回 Vec
    map_call_ms: f64,
    poll_ms: f64,
    view_ms: f64,
    copy_ms: f64,
}
/// 诊断采样上限：约覆盖训练首步的全部 dispatch，够看清开销构成又不至于长期占用内存
const DIAG_MAX: usize = 600;
static GPU_DIAG_LOG: Mutex<Vec<GpuDispatchDiag>> = Mutex::new(Vec::new());

/// 常驻录制路径（[`GpuRecorder`]）的一次「上传 → 录制 → 提交 → 回读」诊断。
///
/// 与 [`GpuDispatchDiag`] 的区别很关键：逐算子路径的固定开销是**按算子**付（每个 dispatch
/// 一次 bind group + submit + poll），而常驻路径把整个子层录进一次提交，固定开销变成
/// **按子层**付。训练实际走的是后者，所以必须单独量——否则会误以为"提交开销已经摊平"。
struct RecorderDiag {
    upload_bytes: usize,
    download_bytes: usize,
    dispatches: usize,
    /// 本批各类别（[`opk`]）的 dispatch 数
    kinds: [usize; opk::N],
    /// 从 recorder() 到 queue.submit 之前：上传 + 建绑定组 + 编码
    record_ms: f64,
    /// queue.submit 本身
    submit_ms: f64,
    /// poll(Wait) 等 GPU + 映射 + 拷回 Vec
    sync_ms: f64,
}
static RECORDER_DIAG_LOG: Mutex<Vec<RecorderDiag>> = Mutex::new(Vec::new());

/// 诊断：`device.create_buffer` 的次数与累计耗时。
///
/// 常驻录制路径的中间张量全部走 `make_buf` **新建**显存对象，submit 后随句柄释放；
/// 显存池只服务逐算子路径。若"录制"那一段的 CPU 时间主要花在这里，
/// 说明瓶颈是显存对象的创建/销毁，而不是绑定与编码。
static BUF_NEW_COUNT: AtomicUsize = AtomicUsize::new(0);
static BUF_NEW_US: AtomicU64 = AtomicU64::new(0);
/// 上次打印时的 (次数, 微秒)，用于报告增量
static BUF_NEW_LAST: Mutex<(usize, u64)> = Mutex::new((0, 0));

/// 常驻录制路径的算子类别，用于诊断分桶与消融实验。
///
/// 类别在 `batch_dispatch` 里**按管线对象识别**（`std::ptr::eq`），而不是让十几个调用点
/// 各自传标签——调用点太多，漏一个就会静默记错账。
pub mod opk {
    pub const N: usize = 18;
    pub const NAMES: [&str; N] = [
        "matmul", "softmax_fwd", "softmax_bwd", "lm_ce", "ln_fwd", "ln_bwd_x", "ln_bwd_gb",
        "gelu_fwd", "gelu_bwd", "bias_drop_res", "dropout_bwd", "col_sum", "heads_split",
        "heads_join", "add", "scale", "relu", "other",
    ];
    pub const MATMUL: u8 = 0;
    pub const SOFTMAX_FWD: u8 = 1;
    pub const SOFTMAX_BWD: u8 = 2;
    pub const LM_CE: u8 = 3;
    pub const LN_FWD: u8 = 4;
    pub const LN_BWD_X: u8 = 5;
    pub const LN_BWD_GB: u8 = 6;
    pub const GELU_FWD: u8 = 7;
    pub const GELU_BWD: u8 = 8;
    pub const BDR: u8 = 9;
    pub const DROPOUT_BWD: u8 = 10;
    pub const COL_SUM: u8 = 11;
    pub const HEADS_SPLIT: u8 = 12;
    pub const HEADS_JOIN: u8 = 13;
    pub const ADD: u8 = 14;
    pub const SCALE: u8 = 15;
    pub const RELU: u8 = 16;
    pub const OTHER: u8 = 17;
}

/// 消融实验：让某一类算子只计数、不录制 dispatch，用「步时差」反推它的真实 GPU 耗时。
///
/// 比读秒更可靠的地方在于：被消融的算子仍然分配输出 buffer、仍然参与后续依赖，
/// 所以只有那一类内核从 GPU 时间线上消失，其余结构与基线完全一致。
/// 仅用于实验，跑数值测试时不要设置。
///
/// `LLM_GPU_ABLATE` 支持逗号分隔一次摘多类（如 `heads_split,heads_join,col_sum`），
/// 这是分清「GPU 时间按内核次数走，还是按内核搬运量走」的关键对照：
/// 一次摘掉 34% 的 dispatch，若步时几乎不动，说明固定开销假设不成立。
fn ablate_kinds() -> &'static [u8] {
    static V: OnceLock<Vec<u8>> = OnceLock::new();
    V.get_or_init(|| match std::env::var("LLM_GPU_ABLATE") {
        Ok(names) => names
            .split(',')
            .filter(|s| !s.is_empty())
            .filter_map(|name| match opk::NAMES.iter().position(|n| *n == name) {
                Some(i) => Some(i as u8),
                None => {
                    println!("[gpu] LLM_GPU_ABLATE={name} 不是有效类别，忽略");
                    None
                }
            })
            .collect(),
        Err(_) => Vec::new(),
    })
}

/// 形状消融（仅实验用）：`LLM_GPU_ABLATE_MM=n<=32`、`LLM_GPU_ABLATE_MM=k<=32,m<=128`，
/// 命中的 matmul 只计数、不录制，其余依赖结构不变 —— 于是「步时差」就是这类形状的**真实**代价。
///
/// 为什么不能用 `probe_report` 的回放代替：回放是**连续满载**跑同一种形状，MX150 在这种
/// 负载下会降频，实测系统性高估约 2 倍（回放 350ms/步 vs 端到端消融 180ms/步）。
/// 只有口径一致的端到端消融才回答「把这类形状改快，最多能省多少」。
///
/// 语法：逗号分隔的 `m|k|n|b` + `<=` 或 `>=` + 数字，命中任意一条即消融。
fn mm_ablated(m: usize, k: usize, n: usize, bs: usize) -> bool {
    static RULES: OnceLock<Vec<(u8, bool, usize)>> = OnceLock::new();
    let rules = RULES.get_or_init(|| {
        let Ok(spec) = std::env::var("LLM_GPU_ABLATE_MM") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for item in spec.split(',').filter(|s| !s.is_empty()) {
            let field = match item.as_bytes().first() {
                Some(b'm') => 0u8,
                Some(b'k') => 1,
                Some(b'n') => 2,
                Some(b'b') => 3,
                _ => {
                    println!("[gpu] LLM_GPU_ABLATE_MM={item} 字段名必须是 m/k/n/b，忽略");
                    continue;
                }
            };
            let (ge, rest) = if let Some(r) = item[1..].strip_prefix("<=") {
                (false, r)
            } else if let Some(r) = item[1..].strip_prefix(">=") {
                (true, r)
            } else {
                println!("[gpu] LLM_GPU_ABLATE_MM={item} 缺少 <= 或 >=，忽略");
                continue;
            };
            match rest.parse::<usize>() {
                Ok(v) => out.push((field, ge, v)),
                Err(_) => println!("[gpu] LLM_GPU_ABLATE_MM={item} 数值无法解析，忽略"),
            }
        }
        out
    });
    rules.iter().any(|&(field, ge, v)| {
        let x = match field {
            0 => m,
            1 => k,
            2 => n,
            _ => bs,
        };
        if ge {
            x >= v
        } else {
            x <= v
        }
    })
}

/// 打印常驻录制路径（训练实际走的路径）的分段计时。
///
/// 这是回答「matmul 之外那部分步时去哪了」的地方：逐算子路径的 `[gpu] dispatch 分解`
/// 量的是**按算子**付的固定开销，而训练用的是「整个子层一次提交」，两者不能互相推算。
fn flush_recorder_diag_log() {
    let log: Vec<RecorderDiag> = {
        let mut v = RECORDER_DIAG_LOG.lock().unwrap();
        std::mem::take(&mut *v)
    };
    if log.is_empty() {
        return;
    }
    let n = log.len() as f64;
    let sum = |f: fn(&RecorderDiag) -> f64| log.iter().map(f).sum::<f64>();
    let (rec_s, sub_s, syn_s) = (
        sum(|d| d.record_ms) / 1000.0,
        sum(|d| d.submit_ms) / 1000.0,
        sum(|d| d.sync_ms) / 1000.0,
    );
    let total = (rec_s + sub_s + syn_s).max(1e-9);
    let dispatches: usize = log.iter().map(|d| d.dispatches).sum();
    let up: usize = log.iter().map(|d| d.upload_bytes).sum();
    let down: usize = log.iter().map(|d| d.download_bytes).sum();
    println!(
        "[gpu] 常驻录制分解：{} 批 | 合计 {:.1}ms（录制 {:.1}ms / 提交 {:.1}ms / 同步回读 {:.1}ms）",
        log.len(),
        total * 1000.0,
        rec_s * 1000.0,
        sub_s * 1000.0,
        syn_s * 1000.0,
    );
    println!(
        "[gpu]   占比：录制 {:.0}% | 提交 {:.0}% | 同步回读 {:.0}% —— 每批平均 {} 次 dispatch，平均 {:.2}ms/批",
        rec_s / total * 100.0,
        sub_s / total * 100.0,
        syn_s / total * 100.0,
        dispatches / log.len(),
        total * 1000.0 / n,
    );
    println!(
        "[gpu]   数据量：上传 {:.1}MB | 回读 {:.1}MB | 本步 GPU 同步等待 {:.1}ms（含真实计算）",
        up as f64 / 1e6,
        down as f64 / 1e6,
        syn_s * 1000.0,
    );
    // 上面的均值把首步预热（管线首次绑定、缓冲池首次分配、write_buffer 首次填满环带）
    // 一起算进去了，会显著高估稳态开销。末尾若干批已经跑在同一温度上，
    // 单独统计这一段才是「每个子层固定开销有多少」的答案。
    let tail_n = log.len().min(20);
    let tail = &log[log.len() - tail_n..];
    let tn = tail_n as f64;
    let tail_sum = |f: fn(&RecorderDiag) -> f64| tail.iter().map(f).sum::<f64>() / tn;
    println!(
        "[gpu]   稳态（末尾 {} 批）：{:.2}ms/批 = 录制 {:.2} + 提交 {:.2} + 同步 {:.2} | 上传 {:.2}MB/批 回读 {:.2}MB/批",
        tail_n,
        tail_sum(|d| d.record_ms + d.submit_ms + d.sync_ms),
        tail_sum(|d| d.record_ms),
        tail_sum(|d| d.submit_ms),
        tail_sum(|d| d.sync_ms),
        tail_sum(|d| d.upload_bytes as f64) / 1e6,
        tail_sum(|d| d.download_bytes as f64) / 1e6,
    );
    // 每步的算子类型分布（按批数归一，见 `opk`）。GPU 时间没法在 CPU 侧直接读秒，
    // 但把"这一步究竟录了哪些内核、各多少次"摆出来，配合 `LLM_GPU_ABLATE=<类别>`
    // 逐类摘除看步时差，就能定位到具体是哪类内核在吃时间。
    let steps = (log.len() as f64 / 11.0).max(1.0);
    let mut mix: Vec<(usize, usize)> = (0..opk::N).map(|k| (k, log.iter().map(|d| d.kinds[k]).sum())).collect();
    mix.sort_by(|a, b| b.1.cmp(&a.1));
    let shown: Vec<String> = mix
        .iter()
        .filter(|(_, c)| *c > 0)
        .take(8)
        .map(|(k, c)| format!("{} {:.1}/步", opk::NAMES[*k], *c as f64 / steps))
        .collect();
    println!("[gpu]   内核类型分布（按批数归一）：{}", shown.join(" | "));
    // 「录制」那一段的 CPU 时间都花在哪：是新建显存对象，还是绑定+编码？
    let (c, us) = (
        BUF_NEW_COUNT.load(Ordering::Relaxed),
        BUF_NEW_US.load(Ordering::Relaxed),
    );
    let (lc, lus) = {
        let mut g = BUF_NEW_LAST.lock().unwrap();
        let prev = *g;
        *g = (c, us);
        prev
    };
    let (dc, dus) = (c - lc, (us - lus) as f64 / 1000.0);
    println!(
        "[gpu]   新建显存对象：{} 次 / {:.1}ms（本次 {} 批 → 每批 {:.1} 次、{:.2}ms）",
        dc,
        dus,
        log.len(),
        dc as f64 / n,
        dus / n,
    );
    let ab = ablate_kinds();
    if !ab.is_empty() {
        let names: Vec<&str> = ab.iter().map(|k| opk::NAMES[*k as usize]).collect();
        println!("[gpu]   ⚠ 消融中：{} 的 dispatch 未录制，步时不可与基线比较数值", names.join(","));
    }
    if let Ok(spec) = std::env::var("LLM_GPU_ABLATE_MM") {
        println!("[gpu]   ⚠ 形状消融中：{spec} 的 matmul 未录制，步时不可与基线比较数值");
    }
}

/// matmul 分流统计：(走 GPU 次数, 走 CPU 次数)，用于训练结束后向用户说明利用率。
pub fn stats() -> (usize, usize) {
    (
        STATS_GPU.load(Ordering::Relaxed),
        STATS_CPU.load(Ordering::Relaxed),
    )
}

/// 打印启动阶段收集的 GPU dispatch 诊断摘要（训练首步结束后调用一次）
///
/// 分解出四段开销，用来判断瓶颈到底在哪一段：
/// - **操作数上传**：`write_buffer` 把 a/b 从 CPU 推过 PCIe（`total - 其余三项`）
/// - **参数上传**：24 字节 uniform 的 `write_buffer`
/// - **绑定+编码+提交**：`create_bind_group` + `CommandEncoder` + `queue.submit`
/// - **poll+回读**：`map_async` → `device.poll(Wait)` → 映射 → 拷回 Vec
/// 后两段是"每次 dispatch 都要重付"的固定开销，也是逐算子同步架构的病根。
pub fn flush_diag_log() {
    flush_recorder_diag_log();
    let log: Vec<GpuDispatchDiag> = {
        let mut v = GPU_DIAG_LOG.lock().unwrap();
        std::mem::take(&mut *v)
    };
    if log.is_empty() {
        return;
    }
    let n = log.len() as f64;
    let avg = |f: fn(&GpuDispatchDiag) -> f64| log.iter().map(f).sum::<f64>() / n;
    let avg_total = avg(|d| d.total_ms);
    let avg_upload = avg(|d| d.upload_ms);
    let avg_dispatch = avg(|d| d.dispatch_ms);
    let avg_sync = avg(|d| d.sync_ms);
    // 其余三项没覆盖到的部分就是操作数上传（run() 之外、matmul() 里那两次 write_buffer）
    let avg_operand = (avg_total - avg_upload - avg_dispatch - avg_sync).max(0.0);
    println!(
        "[gpu] dispatch 分解：采样 {} 次 | 平均 {:.1}ms/次，合计 {:.2}s",
        log.len(),
        avg_total,
        log.iter().map(|d| d.total_ms).sum::<f64>() / 1000.0,
    );
    println!(
        "[gpu]   操作数上传 {:.1}ms ({:.0}%) | 参数上传 {:.1}ms ({:.0}%) | 绑定+编码+提交 {:.1}ms ({:.0}%) | poll+回读 {:.1}ms ({:.0}%)",
        avg_operand, avg_operand / avg_total * 100.0,
        avg_upload, avg_upload / avg_total * 100.0,
        avg_dispatch, avg_dispatch / avg_total * 100.0,
        avg_sync, avg_sync / avg_total * 100.0,
    );
    // poll+回读 再拆：map_async 调用本身 / poll(Wait) 等 GPU / get_mapped_range / 拷回 Vec。
    // 用来分清「等内核做完」与「传输慢」，两者的对策完全不同。
    let avg_map = avg(|d| d.map_call_ms);
    let avg_poll = avg(|d| d.poll_ms);
    let avg_view = avg(|d| d.view_ms);
    let avg_copy = avg(|d| d.copy_ms);
    println!(
        "[gpu]   回读细分：map_async {:.1}ms ({:.0}%) | poll(Wait) {:.1}ms ({:.0}%) | get_mapped_range {:.1}ms ({:.0}%) | 拷回 Vec {:.1}ms ({:.0}%)",
        avg_map, avg_map / avg_sync * 100.0,
        avg_poll, avg_poll / avg_sync * 100.0,
        avg_view, avg_view / avg_sync * 100.0,
        avg_copy, avg_copy / avg_sync * 100.0,
    );
    let bytes: u64 = log.iter().map(|d| d.out_len as u64 * 4).sum();
    let sync_s: f64 = log.iter().map(|d| d.sync_ms).sum::<f64>() / 1000.0;
    if sync_s > 0.0 {
        println!(
            "[gpu]   回读总量 {:.1}MB / {:.2}s → {:.0} MB/s（含固定等待，实际带宽更高）",
            bytes as f64 / 1e6,
            sync_s,
            bytes as f64 / 1e6 / sync_s,
        );
    }
    // 回读量按输出大小分组：同一形状每步反复出现，先看是哪些张量把 1 GB 级的量顶上去的
    let mut groups: std::collections::HashMap<(usize, u32, u32, u32), (usize, f64)> =
        std::collections::HashMap::new();
    for d in &log {
        let e = groups
            .entry((d.out_len, d.x, d.y, d.z))
            .or_insert((0, 0.0));
        e.0 += 1;
        e.1 += d.total_ms;
    }
    let mut gs: Vec<_> = groups.into_iter().collect();
    gs.sort_by(|a, b| (b.0 .0 as u64 * b.1 .0 as u64).cmp(&(a.0 .0 as u64 * a.1 .0 as u64)));
    println!("[gpu]   回读量按形状分组（前 8 大）：");
    for ((out_len, x, y, z), (cnt, ms)) in gs.iter().take(8) {
        println!(
            "[gpu]     {:.1}M元素 × {} 次 = {:.1}MB | 网格 {}x{}x{} | 合计 {:.0}ms",
            *out_len as f64 / 1e6,
            cnt,
            *out_len as f64 * 4.0 / 1e6 * *cnt as f64,
            x, y, z, ms,
        );
    }
}

/// buffer 池的四种用途（决定 usage 与归还键）
const KIND_IN: u8 = 0; // 输入：STORAGE | COPY_DST（write_buffer 上传）
const KIND_OUT: u8 = 1; // 输出：STORAGE | COPY_DST | COPY_SRC（再拷到 readback）
const KIND_READ: u8 = 2; // 读回：COPY_DST | MAP_READ（同步取回结果）
const KIND_UNIFORM: u8 = 3; // 参数：UNIFORM | COPY_DST（批量路径每个 dispatch 一块，避免互相覆盖）

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
//
// 寄存器分块（register blocking）GEMM：workgroup 16×16 = 256 线程，输出 tile 128×128，
// 每线程负责 8×8 = 64 个输出元素。
//
// 为什么不用「每线程 1 个元素 + 16×16 共享内存 tile」（旧版）：那样内层每做 1 次 FMA 就要
// 付 2 次共享内存 LDS，而 LDS 吞吐是 1 条/cycle、FMA 是 4 条/cycle —— 等于被 LDS 卡住 8 倍。
// 实测依据（logs\gpu_probe4.log）：旧版只有 23.6 GFLOP/s，而同一张卡跑纯寄存器 FMA 能到
// 524 GFLOP/s，差距全在共享内存往返。改成 8×8 后，每个 k 步是 16 次 LDS 对 64 次 FMA，
// 两条流水线正好配平；再把共享内存改 vec4 布局（见下）后 LDS 只剩 4 条，LDS 不再是限制。
//
// 行/列的分配：线程持有「4 连续行」的两组（4*lid.x 与 64 + 4*lid.x）和「4 连续列」的两组
// （4*lid.y 与 64 + 4*lid.y）。这样计算阶段每次 LDS.128 取到的 4 个值正好是同一 k 上
// 连续 4 行/列，直接对上 4 个累加器；同一 warp 内 16 个线程读 16 个连续的 vec4（64 个 bank 全用满，
// 无 bank conflict），另 16 个线程读同样地址（广播）。
//
// 转置访问（p4/p5）：反向传播的 ∂a = g @ bᵀ、∂b = aᵀ @ g 直接按转置读物理矩阵，
// 免去在 CPU 上构造 52 万~210 万元素的转置矩阵再上传的开销。
//   逻辑 A_l[i][j] = 物理 a[j][i]（物理布局 [B,K,M]）
//   逻辑 B_l[i][j] = 物理 b[j][i]（物理布局 [B,N,K]）
// 转置只影响装载阶段读全局内存的下标，计算阶段完全一致。
//
// 装载还做了软件流水：第 0 个 tile 先取进寄存器，循环体是
// 「存上一轮取好的 → barrier → 发下一轮全局 load → 计算 → barrier」，
// 用 512 条 FMA 盖住全局访存的几百拍延迟（实测 130.5 → 139.5 GFLOP/s）。
// 注意 batch 的输入偏移必须用 wid.z 而不是 uniform 里的 p0（p0 是 batch 的**个数**）。
// WGSL 规定 workgroup 地址空间变量必须声明在模块作用域。
const MM_TILE: u32 = 128u; // 输出 tile 边长（M / N 方向）
const MM_KK: u32 = 8u; // 每次装载的 k 高度
const MM_TN: u32 = 8u; // 每线程输出边长（8×8）
// 注：把 MM_KK 加到 16（每线程每轮搬 2 层、共享内存加倍）实测是**退步**：
// 128.9 / 156.7 / 88.4 GFLOP/s vs MM_KK=8 的 139.5 / 165.2 / 98.1。
// 原因是 64 个累加器之外再多 4 个 vec4 的预取寄存器会把 register file 挤到溢出，
// 多出来的 barrier 收益抵不过 spill 的代价。

// 共享内存按 vec4 组织：一个元素 = 同一 k 上连续 4 行（A）/ 4 列（B）。
// 计算阶段一次 LDS.128 就能取 4 个操作数，LDS 指令数降到标量版的 1/4：
// LDS 是 1 条/clk、FMA 是 4 条/clk，标量版每个 k 步 16 条 LDS 对 64 条 FMA 正好占满两条流水线，
// vec4 后 LDS 只占 8 拍（4 条 LDS.128，每条搬 512B = 4 拍）→ 不再是瓶颈。
var<workgroup> sh_a: array<vec4<f32>, 256>; // [k(8)][行组(32)]，行组 g 覆盖行 4g..4g+3
var<workgroup> sh_b: array<vec4<f32>, 256>; // [k(8)][列组(32)]，列组 g 覆盖列 4g..4g+3

/// 装载辅助：取 A tile 的 k = k0+kmid 这一层、行组 g（行 row0+4g .. +3）装成一个 vec4。
/// 越界的行夹到 m-1（多算出的结果不会写回）；k 越界则返回 0。
/// 单独拆成函数是为了做软件流水（tile 0 与循环内各调一次，不重复代码）。
///
/// `off_a` 是当前 batch 的元素偏移（= wid.z * m * k）。注意**不能**写成 `params.p0 * m * k`：
/// p0 是 batch 的**个数**（uniform 传的是 batch 总数），不是当前 workgroup 的 batch 序号，
/// 用它当偏移在 batch=1 时会直接越界读到 0，整个 tile 结果全错。
fn fetch_a(k0: u32, row0: u32, g: u32, kmid: u32, off_a: u32) -> vec4<f32> {
    let m = params.p1;
    let k = params.p2;
    let gk = k0 + kmid;
    if (gk >= k) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (params.p4 == 1u) {
        // 物理 [K,M]：同一 k 的连续 4 行地址连续，4 次 load 能合并
        return vec4<f32>(
            a[off_a + gk * m + min(row0 + 4u * g + 0u, m - 1u)],
            a[off_a + gk * m + min(row0 + 4u * g + 1u, m - 1u)],
            a[off_a + gk * m + min(row0 + 4u * g + 2u, m - 1u)],
            a[off_a + gk * m + min(row0 + 4u * g + 3u, m - 1u)],
        );
    }
    // 物理 [M,K]：行间 stride = k
    return vec4<f32>(
        a[off_a + min(row0 + 4u * g + 0u, m - 1u) * k + gk],
        a[off_a + min(row0 + 4u * g + 1u, m - 1u) * k + gk],
        a[off_a + min(row0 + 4u * g + 2u, m - 1u) * k + gk],
        a[off_a + min(row0 + 4u * g + 3u, m - 1u) * k + gk],
    );
}

/// 装载辅助：取 B tile 的 k = k0+kmid 这一层、列组 g（列 col0+4g .. +3）。语义同 [`fetch_a`]。
/// `off_b` 是当前 batch 的元素偏移（= wid.z * k * n），理由见 [`fetch_a`]。
fn fetch_b(k0: u32, col0: u32, g: u32, kmid: u32, off_b: u32) -> vec4<f32> {
    let k = params.p2;
    let n = params.p3;
    let gk = k0 + kmid;
    if (gk >= k) {
        return vec4<f32>(0.0, 0.0, 0.0, 0.0);
    }
    if (params.p5 == 1u) {
        // 物理 [N,K]：列间 stride = k
        return vec4<f32>(
            b[off_b + min(col0 + 4u * g + 0u, n - 1u) * k + gk],
            b[off_b + min(col0 + 4u * g + 1u, n - 1u) * k + gk],
            b[off_b + min(col0 + 4u * g + 2u, n - 1u) * k + gk],
            b[off_b + min(col0 + 4u * g + 3u, n - 1u) * k + gk],
        );
    }
    // 物理 [K,N]：同一 k 的连续 4 列地址连续
    return vec4<f32>(
        b[off_b + gk * n + min(col0 + 4u * g + 0u, n - 1u)],
        b[off_b + gk * n + min(col0 + 4u * g + 1u, n - 1u)],
        b[off_b + gk * n + min(col0 + 4u * g + 2u, n - 1u)],
        b[off_b + gk * n + min(col0 + 4u * g + 3u, n - 1u)],
    );
}

@compute @workgroup_size(16, 16, 1)
fn matmul_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = params.p1;
    let k = params.p2;
    let n = params.p3;
    // 注意用 workgroup_id 而不是 global_invocation_id：workgroup 只有 16 线程宽，
    // gid.x = wid.x*16 + lid.x，直接拿它算 tile 原点会把 128 的行距重复累加 16 次。
    let batch = wid.z;
    let row0 = wid.x * MM_TILE;
    let col0 = wid.y * MM_TILE;
    let tid = lid.y * 16u + lid.x;
    // 注意：所有线程必须走完全相同的控制流（含 workgroupBarrier）——越界线程不提前 return，
    // 而是夹到边界上算无用功，最后写回时再按 (row, col) 保护。
    //
    // 64 个累加器写成独立标量而不是 `array<f32,64>`：WGSL 里用循环变量下标的数组
    // 未必被展开，一旦下标是动态的就进不了寄存器、只能落到局部内存（走 L1），
    // 每次 FMA 都多一次 load + 一次 store。实测（`matmul_throughput_probe`）：
    // 同样的算法，数组版 190 GFLOP/s，命名标量版才吃得到寄存器。
    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0; var c04 = 0.0; var c05 = 0.0; var c06 = 0.0; var c07 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0; var c14 = 0.0; var c15 = 0.0; var c16 = 0.0; var c17 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0; var c24 = 0.0; var c25 = 0.0; var c26 = 0.0; var c27 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0; var c34 = 0.0; var c35 = 0.0; var c36 = 0.0; var c37 = 0.0;
    var c40 = 0.0; var c41 = 0.0; var c42 = 0.0; var c43 = 0.0; var c44 = 0.0; var c45 = 0.0; var c46 = 0.0; var c47 = 0.0;
    var c50 = 0.0; var c51 = 0.0; var c52 = 0.0; var c53 = 0.0; var c54 = 0.0; var c55 = 0.0; var c56 = 0.0; var c57 = 0.0;
    var c60 = 0.0; var c61 = 0.0; var c62 = 0.0; var c63 = 0.0; var c64 = 0.0; var c65 = 0.0; var c66 = 0.0; var c67 = 0.0;
    var c70 = 0.0; var c71 = 0.0; var c72 = 0.0; var c73 = 0.0; var c74 = 0.0; var c75 = 0.0; var c76 = 0.0; var c77 = 0.0;

    let ktiles = (k + MM_KK - 1u) / MM_KK;
    // 当前 batch 的输入偏移：p0 是 batch 总数（见 fetch_a 的说明），这里必须用 wid.z
    let off_a = batch * m * k;
    let off_b = batch * k * n;
    // 装载阶段的固定坐标：行/列组 g（覆盖 4 行/列）、k 层 kmid（0..7）
    let g = tid % 32u;
    let kmid = tid / 32u;
    // 软件流水（software pipelining）：全局 load 的延迟有几百拍，若"先 load 再 barrier 再算"
    // 就是纯串行等待。这里把第 0 个 tile 先取进寄存器，循环体内改成
    // 「存上一轮取好的 → barrier → 立刻发下一轮的 load → 计算（覆盖 load 延迟）→ barrier」。
    // 寄存器是每线程私有的，不需要额外同步；共享内存的读写顺序仍由两道 barrier 保护。
    var na = fetch_a(0u, row0, g, kmid, off_a);
    var nb = fetch_b(0u, col0, g, kmid, off_b);

    for (var kb = 0u; kb < ktiles; kb = kb + 1u) {
        sh_a[kmid * 32u + g] = na;
        sh_b[kmid * 32u + g] = nb;
        workgroupBarrier();

        // 预取下一 tile：这几条全局 load 一发出去就返回，下面的 512 条 FMA 足够盖住访存延迟
        if (kb + 1u < ktiles) {
            na = fetch_a((kb + 1u) * MM_KK, row0, g, kmid, off_a);
            nb = fetch_b((kb + 1u) * MM_KK, col0, g, kmid, off_b);
        }

        // ② 计算：每线程 8×8 外积累加（下标全为常量）
        // 线程持有的行 = 4*lid.x + {0..3}（前 4 个累加器）与 64 + 4*lid.x + {0..3}（后 4 个）；
        // 列 = 4*lid.y + {0..3} 与 64 + 4*lid.y + {0..3}。每个 k 步只需 4 条 LDS.128。
        for (var kk = 0u; kk < MM_KK; kk = kk + 1u) {
            let va0 = sh_a[kk * 32u + lid.x];
            let va1 = sh_a[kk * 32u + 16u + lid.x];
            let vb0 = sh_b[kk * 32u + lid.y];
            let vb1 = sh_b[kk * 32u + 16u + lid.y];
            let av0 = va0.x; let av1 = va0.y; let av2 = va0.z; let av3 = va0.w;
            let av4 = va1.x; let av5 = va1.y; let av6 = va1.z; let av7 = va1.w;
            let bv0 = vb0.x; let bv1 = vb0.y; let bv2 = vb0.z; let bv3 = vb0.w;
            let bv4 = vb1.x; let bv5 = vb1.y; let bv6 = vb1.z; let bv7 = vb1.w;
            c00 = c00 + av0 * bv0; c01 = c01 + av0 * bv1; c02 = c02 + av0 * bv2; c03 = c03 + av0 * bv3;
            c04 = c04 + av0 * bv4; c05 = c05 + av0 * bv5; c06 = c06 + av0 * bv6; c07 = c07 + av0 * bv7;
            c10 = c10 + av1 * bv0; c11 = c11 + av1 * bv1; c12 = c12 + av1 * bv2; c13 = c13 + av1 * bv3;
            c14 = c14 + av1 * bv4; c15 = c15 + av1 * bv5; c16 = c16 + av1 * bv6; c17 = c17 + av1 * bv7;
            c20 = c20 + av2 * bv0; c21 = c21 + av2 * bv1; c22 = c22 + av2 * bv2; c23 = c23 + av2 * bv3;
            c24 = c24 + av2 * bv4; c25 = c25 + av2 * bv5; c26 = c26 + av2 * bv6; c27 = c27 + av2 * bv7;
            c30 = c30 + av3 * bv0; c31 = c31 + av3 * bv1; c32 = c32 + av3 * bv2; c33 = c33 + av3 * bv3;
            c34 = c34 + av3 * bv4; c35 = c35 + av3 * bv5; c36 = c36 + av3 * bv6; c37 = c37 + av3 * bv7;
            c40 = c40 + av4 * bv0; c41 = c41 + av4 * bv1; c42 = c42 + av4 * bv2; c43 = c43 + av4 * bv3;
            c44 = c44 + av4 * bv4; c45 = c45 + av4 * bv5; c46 = c46 + av4 * bv6; c47 = c47 + av4 * bv7;
            c50 = c50 + av5 * bv0; c51 = c51 + av5 * bv1; c52 = c52 + av5 * bv2; c53 = c53 + av5 * bv3;
            c54 = c54 + av5 * bv4; c55 = c55 + av5 * bv5; c56 = c56 + av5 * bv6; c57 = c57 + av5 * bv7;
            c60 = c60 + av6 * bv0; c61 = c61 + av6 * bv1; c62 = c62 + av6 * bv2; c63 = c63 + av6 * bv3;
            c64 = c64 + av6 * bv4; c65 = c65 + av6 * bv5; c66 = c66 + av6 * bv6; c67 = c67 + av6 * bv7;
            c70 = c70 + av7 * bv0; c71 = c71 + av7 * bv1; c72 = c72 + av7 * bv2; c73 = c73 + av7 * bv3;
            c74 = c74 + av7 * bv4; c75 = c75 + av7 * bv5; c76 = c76 + av7 * bv6; c77 = c77 + av7 * bv7;
        }
        workgroupBarrier();
    }

    // ---- 写回：下标同样全部写死 ----
    // 行 = row0 + 4*lid.x + {0..3}（c0x）与 row0 + 64 + 4*lid.x + {0..3}（c4x）；
    // 列 = col0 + 4*lid.y + {0..3}（cx0..cx3）与 col0 + 64 + 4*lid.y + {0..3}（cx4..cx7）。
    // 行越界整行跳过；列越界逐个跳过（同一行内 8 列的判断是重复的，为省掉动态下标只能展开）
    {
        let rbase = row0 + 4u * lid.x;
        let cbase = col0 + 4u * lid.y;
        if (rbase + 0u < m) {
            let oy = (batch * m + rbase + 0u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c00; }
            if (cbase + 1u < n) { out[oy + 1u] = c01; }
            if (cbase + 2u < n) { out[oy + 2u] = c02; }
            if (cbase + 3u < n) { out[oy + 3u] = c03; }
            if (cbase + 64u < n) { out[oy + 64u] = c04; }
            if (cbase + 65u < n) { out[oy + 65u] = c05; }
            if (cbase + 66u < n) { out[oy + 66u] = c06; }
            if (cbase + 67u < n) { out[oy + 67u] = c07; }
        }
        if (rbase + 1u < m) {
            let oy = (batch * m + rbase + 1u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c10; }
            if (cbase + 1u < n) { out[oy + 1u] = c11; }
            if (cbase + 2u < n) { out[oy + 2u] = c12; }
            if (cbase + 3u < n) { out[oy + 3u] = c13; }
            if (cbase + 64u < n) { out[oy + 64u] = c14; }
            if (cbase + 65u < n) { out[oy + 65u] = c15; }
            if (cbase + 66u < n) { out[oy + 66u] = c16; }
            if (cbase + 67u < n) { out[oy + 67u] = c17; }
        }
        if (rbase + 2u < m) {
            let oy = (batch * m + rbase + 2u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c20; }
            if (cbase + 1u < n) { out[oy + 1u] = c21; }
            if (cbase + 2u < n) { out[oy + 2u] = c22; }
            if (cbase + 3u < n) { out[oy + 3u] = c23; }
            if (cbase + 64u < n) { out[oy + 64u] = c24; }
            if (cbase + 65u < n) { out[oy + 65u] = c25; }
            if (cbase + 66u < n) { out[oy + 66u] = c26; }
            if (cbase + 67u < n) { out[oy + 67u] = c27; }
        }
        if (rbase + 3u < m) {
            let oy = (batch * m + rbase + 3u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c30; }
            if (cbase + 1u < n) { out[oy + 1u] = c31; }
            if (cbase + 2u < n) { out[oy + 2u] = c32; }
            if (cbase + 3u < n) { out[oy + 3u] = c33; }
            if (cbase + 64u < n) { out[oy + 64u] = c34; }
            if (cbase + 65u < n) { out[oy + 65u] = c35; }
            if (cbase + 66u < n) { out[oy + 66u] = c36; }
            if (cbase + 67u < n) { out[oy + 67u] = c37; }
        }
        if (rbase + 64u < m) {
            let oy = (batch * m + rbase + 64u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c40; }
            if (cbase + 1u < n) { out[oy + 1u] = c41; }
            if (cbase + 2u < n) { out[oy + 2u] = c42; }
            if (cbase + 3u < n) { out[oy + 3u] = c43; }
            if (cbase + 64u < n) { out[oy + 64u] = c44; }
            if (cbase + 65u < n) { out[oy + 65u] = c45; }
            if (cbase + 66u < n) { out[oy + 66u] = c46; }
            if (cbase + 67u < n) { out[oy + 67u] = c47; }
        }
        if (rbase + 65u < m) {
            let oy = (batch * m + rbase + 65u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c50; }
            if (cbase + 1u < n) { out[oy + 1u] = c51; }
            if (cbase + 2u < n) { out[oy + 2u] = c52; }
            if (cbase + 3u < n) { out[oy + 3u] = c53; }
            if (cbase + 64u < n) { out[oy + 64u] = c54; }
            if (cbase + 65u < n) { out[oy + 65u] = c55; }
            if (cbase + 66u < n) { out[oy + 66u] = c56; }
            if (cbase + 67u < n) { out[oy + 67u] = c57; }
        }
        if (rbase + 66u < m) {
            let oy = (batch * m + rbase + 66u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c60; }
            if (cbase + 1u < n) { out[oy + 1u] = c61; }
            if (cbase + 2u < n) { out[oy + 2u] = c62; }
            if (cbase + 3u < n) { out[oy + 3u] = c63; }
            if (cbase + 64u < n) { out[oy + 64u] = c64; }
            if (cbase + 65u < n) { out[oy + 65u] = c65; }
            if (cbase + 66u < n) { out[oy + 66u] = c66; }
            if (cbase + 67u < n) { out[oy + 67u] = c67; }
        }
        if (rbase + 67u < m) {
            let oy = (batch * m + rbase + 67u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c70; }
            if (cbase + 1u < n) { out[oy + 1u] = c71; }
            if (cbase + 2u < n) { out[oy + 2u] = c72; }
            if (cbase + 3u < n) { out[oy + 3u] = c73; }
            if (cbase + 64u < n) { out[oy + 64u] = c74; }
            if (cbase + 65u < n) { out[oy + 65u] = c75; }
            if (cbase + 66u < n) { out[oy + 66u] = c76; }
            if (cbase + 67u < n) { out[oy + 67u] = c77; }
        }
    }
}

// ---- 小 tile 变体：输出 tile 64×64，workgroup 8×8 = 64 线程，每线程仍是 8×8 = 64 个累加器 ----
//
// 这是当前**默认**的 matmul 内核（见 Rust 侧 `use_small_tile`），大 tile 版只在标定时用
// `LLM_GPU_MM_SMALL=0` 拉回来做对照。
//
// 起因是 128×128 的 tile 在 attention 的形状上要掩掉大半列（PV / dV / dK 的 n = head_dim = 32，
// 每 4 条 FMA 里 3 条算的是不存在的输出），但**同进程交替 A/B 实测把这个归因推翻了**：
// 64×64 在测过的训练形状上都不慢于 128×128，连 n = 8192、几乎没有 tile 浪费的
// lm_head 也快 18%。真正的瓶颈是工作组太胖 —— 256 线程 × 约 100 个寄存器，一个 SM 只装得下
// 两个工作组，于是 barrier 与全局 load 的等待无处躲藏；换成 64 线程后同一张 SM 上能并存
// 成倍的工作组，一个工作组的访存在给另一个的计算当背景。详细对照数据见 `use_small_tile`。
//
// 内层循环与 [`matmul_main`] **逐字相同**（每 k 步 4 条 LDS.128 对 64 条 FMA、累加器下标全为
// 常量、软件流水、越界夹边），只把共享内存 stride 从 32（128 行 / 每 vec4 4 行）改成 16、
// 第二组偏移从 64 改成 32。关键点：**每个输出元素的 k 求和顺序仍是 k = 0, 1, 2, … 逐层推进**，
// 因此结果与 [`matmul_main`] 逐位相同（浮点加法不满足结合律，换个求和顺序就不再 bit-exact；
// 两者逐位一致由 `gpu_matmul_small_tile_matches_big_tile_bits` 守住）。
const MM_TILE_S: u32 = 64u; // 小 tile 边长（M / N 方向）
const MM_STRIDE_S: u32 = 16u; // 共享内存 stride = 64 行 / 每 vec4 4 行

@compute @workgroup_size(8, 8, 1)
fn matmul_small_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let m = params.p1;
    let k = params.p2;
    let n = params.p3;
    let batch = wid.z;
    let row0 = wid.x * MM_TILE_S;
    let col0 = wid.y * MM_TILE_S;
    let tid = lid.y * 8u + lid.x;

    var c00 = 0.0; var c01 = 0.0; var c02 = 0.0; var c03 = 0.0; var c04 = 0.0; var c05 = 0.0; var c06 = 0.0; var c07 = 0.0;
    var c10 = 0.0; var c11 = 0.0; var c12 = 0.0; var c13 = 0.0; var c14 = 0.0; var c15 = 0.0; var c16 = 0.0; var c17 = 0.0;
    var c20 = 0.0; var c21 = 0.0; var c22 = 0.0; var c23 = 0.0; var c24 = 0.0; var c25 = 0.0; var c26 = 0.0; var c27 = 0.0;
    var c30 = 0.0; var c31 = 0.0; var c32 = 0.0; var c33 = 0.0; var c34 = 0.0; var c35 = 0.0; var c36 = 0.0; var c37 = 0.0;
    var c40 = 0.0; var c41 = 0.0; var c42 = 0.0; var c43 = 0.0; var c44 = 0.0; var c45 = 0.0; var c46 = 0.0; var c47 = 0.0;
    var c50 = 0.0; var c51 = 0.0; var c52 = 0.0; var c53 = 0.0; var c54 = 0.0; var c55 = 0.0; var c56 = 0.0; var c57 = 0.0;
    var c60 = 0.0; var c61 = 0.0; var c62 = 0.0; var c63 = 0.0; var c64 = 0.0; var c65 = 0.0; var c66 = 0.0; var c67 = 0.0;
    var c70 = 0.0; var c71 = 0.0; var c72 = 0.0; var c73 = 0.0; var c74 = 0.0; var c75 = 0.0; var c76 = 0.0; var c77 = 0.0;

    let ktiles = (k + MM_KK - 1u) / MM_KK;
    let off_a = batch * m * k;
    let off_b = batch * k * n;
    // 装载：64 个线程要盖住 8 个 k 层 × 16 个「4 行/列组」= 128 个 vec4，于是每线程搬两个 ——
    // 同一个行列组 g 的第 km 层与第 km+4 层。两次的 g 相同（都取 `tid % 16`），
    // 因此 16 个线程覆盖 16 个行列组、4 个线程覆盖前 4 层，另一份补上后 4 层。
    let g = tid % 16u;
    let km = tid / 16u;
    var na0 = fetch_a(0u, row0, g, km, off_a);
    var na1 = fetch_a(0u, row0, g, km + 4u, off_a);
    var nb0 = fetch_b(0u, col0, g, km, off_b);
    var nb1 = fetch_b(0u, col0, g, km + 4u, off_b);

    for (var kb = 0u; kb < ktiles; kb = kb + 1u) {
        sh_a[km * MM_STRIDE_S + g] = na0;
        sh_a[(km + 4u) * MM_STRIDE_S + g] = na1;
        sh_b[km * MM_STRIDE_S + g] = nb0;
        sh_b[(km + 4u) * MM_STRIDE_S + g] = nb1;
        workgroupBarrier();

        if (kb + 1u < ktiles) {
            na0 = fetch_a((kb + 1u) * MM_KK, row0, g, km, off_a);
            na1 = fetch_a((kb + 1u) * MM_KK, row0, g, km + 4u, off_a);
            nb0 = fetch_b((kb + 1u) * MM_KK, col0, g, km, off_b);
            nb1 = fetch_b((kb + 1u) * MM_KK, col0, g, km + 4u, off_b);
        }

        // 计算：行 = 4*lid.x + {0..3} 与 32 + 4*lid.x + {0..3}（0..7 号线程覆盖 64 行）；
        // 列 = 4*lid.y + {0..3} 与 32 + 4*lid.y + {0..3}。每 k 步仍是 4 条 LDS.128。
        for (var kk = 0u; kk < MM_KK; kk = kk + 1u) {
            let va0 = sh_a[kk * MM_STRIDE_S + lid.x];
            let va1 = sh_a[kk * MM_STRIDE_S + 8u + lid.x];
            let vb0 = sh_b[kk * MM_STRIDE_S + lid.y];
            let vb1 = sh_b[kk * MM_STRIDE_S + 8u + lid.y];
            let av0 = va0.x; let av1 = va0.y; let av2 = va0.z; let av3 = va0.w;
            let av4 = va1.x; let av5 = va1.y; let av6 = va1.z; let av7 = va1.w;
            let bv0 = vb0.x; let bv1 = vb0.y; let bv2 = vb0.z; let bv3 = vb0.w;
            let bv4 = vb1.x; let bv5 = vb1.y; let bv6 = vb1.z; let bv7 = vb1.w;
            c00 = c00 + av0 * bv0; c01 = c01 + av0 * bv1; c02 = c02 + av0 * bv2; c03 = c03 + av0 * bv3;
            c04 = c04 + av0 * bv4; c05 = c05 + av0 * bv5; c06 = c06 + av0 * bv6; c07 = c07 + av0 * bv7;
            c10 = c10 + av1 * bv0; c11 = c11 + av1 * bv1; c12 = c12 + av1 * bv2; c13 = c13 + av1 * bv3;
            c14 = c14 + av1 * bv4; c15 = c15 + av1 * bv5; c16 = c16 + av1 * bv6; c17 = c17 + av1 * bv7;
            c20 = c20 + av2 * bv0; c21 = c21 + av2 * bv1; c22 = c22 + av2 * bv2; c23 = c23 + av2 * bv3;
            c24 = c24 + av2 * bv4; c25 = c25 + av2 * bv5; c26 = c26 + av2 * bv6; c27 = c27 + av2 * bv7;
            c30 = c30 + av3 * bv0; c31 = c31 + av3 * bv1; c32 = c32 + av3 * bv2; c33 = c33 + av3 * bv3;
            c34 = c34 + av3 * bv4; c35 = c35 + av3 * bv5; c36 = c36 + av3 * bv6; c37 = c37 + av3 * bv7;
            c40 = c40 + av4 * bv0; c41 = c41 + av4 * bv1; c42 = c42 + av4 * bv2; c43 = c43 + av4 * bv3;
            c44 = c44 + av4 * bv4; c45 = c45 + av4 * bv5; c46 = c46 + av4 * bv6; c47 = c47 + av4 * bv7;
            c50 = c50 + av5 * bv0; c51 = c51 + av5 * bv1; c52 = c52 + av5 * bv2; c53 = c53 + av5 * bv3;
            c54 = c54 + av5 * bv4; c55 = c55 + av5 * bv5; c56 = c56 + av5 * bv6; c57 = c57 + av5 * bv7;
            c60 = c60 + av6 * bv0; c61 = c61 + av6 * bv1; c62 = c62 + av6 * bv2; c63 = c63 + av6 * bv3;
            c64 = c64 + av6 * bv4; c65 = c65 + av6 * bv5; c66 = c66 + av6 * bv6; c67 = c67 + av6 * bv7;
            c70 = c70 + av7 * bv0; c71 = c71 + av7 * bv1; c72 = c72 + av7 * bv2; c73 = c73 + av7 * bv3;
            c74 = c74 + av7 * bv4; c75 = c75 + av7 * bv5; c76 = c76 + av7 * bv6; c77 = c77 + av7 * bv7;
        }
        workgroupBarrier();
    }

    // ---- 写回：与 [`matmul_main`] 同构，只是第二组偏移由 64 改成 32（tile 边长减半）----
    {
        let rbase = row0 + 4u * lid.x;
        let cbase = col0 + 4u * lid.y;
        if (rbase + 0u < m) {
            let oy = (batch * m + rbase + 0u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c00; }
            if (cbase + 1u < n) { out[oy + 1u] = c01; }
            if (cbase + 2u < n) { out[oy + 2u] = c02; }
            if (cbase + 3u < n) { out[oy + 3u] = c03; }
            if (cbase + 32u < n) { out[oy + 32u] = c04; }
            if (cbase + 33u < n) { out[oy + 33u] = c05; }
            if (cbase + 34u < n) { out[oy + 34u] = c06; }
            if (cbase + 35u < n) { out[oy + 35u] = c07; }
        }
        if (rbase + 1u < m) {
            let oy = (batch * m + rbase + 1u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c10; }
            if (cbase + 1u < n) { out[oy + 1u] = c11; }
            if (cbase + 2u < n) { out[oy + 2u] = c12; }
            if (cbase + 3u < n) { out[oy + 3u] = c13; }
            if (cbase + 32u < n) { out[oy + 32u] = c14; }
            if (cbase + 33u < n) { out[oy + 33u] = c15; }
            if (cbase + 34u < n) { out[oy + 34u] = c16; }
            if (cbase + 35u < n) { out[oy + 35u] = c17; }
        }
        if (rbase + 2u < m) {
            let oy = (batch * m + rbase + 2u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c20; }
            if (cbase + 1u < n) { out[oy + 1u] = c21; }
            if (cbase + 2u < n) { out[oy + 2u] = c22; }
            if (cbase + 3u < n) { out[oy + 3u] = c23; }
            if (cbase + 32u < n) { out[oy + 32u] = c24; }
            if (cbase + 33u < n) { out[oy + 33u] = c25; }
            if (cbase + 34u < n) { out[oy + 34u] = c26; }
            if (cbase + 35u < n) { out[oy + 35u] = c27; }
        }
        if (rbase + 3u < m) {
            let oy = (batch * m + rbase + 3u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c30; }
            if (cbase + 1u < n) { out[oy + 1u] = c31; }
            if (cbase + 2u < n) { out[oy + 2u] = c32; }
            if (cbase + 3u < n) { out[oy + 3u] = c33; }
            if (cbase + 32u < n) { out[oy + 32u] = c34; }
            if (cbase + 33u < n) { out[oy + 33u] = c35; }
            if (cbase + 34u < n) { out[oy + 34u] = c36; }
            if (cbase + 35u < n) { out[oy + 35u] = c37; }
        }
        if (rbase + 32u < m) {
            let oy = (batch * m + rbase + 32u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c40; }
            if (cbase + 1u < n) { out[oy + 1u] = c41; }
            if (cbase + 2u < n) { out[oy + 2u] = c42; }
            if (cbase + 3u < n) { out[oy + 3u] = c43; }
            if (cbase + 32u < n) { out[oy + 32u] = c44; }
            if (cbase + 33u < n) { out[oy + 33u] = c45; }
            if (cbase + 34u < n) { out[oy + 34u] = c46; }
            if (cbase + 35u < n) { out[oy + 35u] = c47; }
        }
        if (rbase + 33u < m) {
            let oy = (batch * m + rbase + 33u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c50; }
            if (cbase + 1u < n) { out[oy + 1u] = c51; }
            if (cbase + 2u < n) { out[oy + 2u] = c52; }
            if (cbase + 3u < n) { out[oy + 3u] = c53; }
            if (cbase + 32u < n) { out[oy + 32u] = c54; }
            if (cbase + 33u < n) { out[oy + 33u] = c55; }
            if (cbase + 34u < n) { out[oy + 34u] = c56; }
            if (cbase + 35u < n) { out[oy + 35u] = c57; }
        }
        if (rbase + 34u < m) {
            let oy = (batch * m + rbase + 34u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c60; }
            if (cbase + 1u < n) { out[oy + 1u] = c61; }
            if (cbase + 2u < n) { out[oy + 2u] = c62; }
            if (cbase + 3u < n) { out[oy + 3u] = c63; }
            if (cbase + 32u < n) { out[oy + 32u] = c64; }
            if (cbase + 33u < n) { out[oy + 33u] = c65; }
            if (cbase + 34u < n) { out[oy + 34u] = c66; }
            if (cbase + 35u < n) { out[oy + 35u] = c67; }
        }
        if (rbase + 35u < m) {
            let oy = (batch * m + rbase + 35u) * n + cbase;
            if (cbase + 0u < n) { out[oy + 0u] = c70; }
            if (cbase + 1u < n) { out[oy + 1u] = c71; }
            if (cbase + 2u < n) { out[oy + 2u] = c72; }
            if (cbase + 3u < n) { out[oy + 3u] = c73; }
            if (cbase + 32u < n) { out[oy + 32u] = c74; }
            if (cbase + 33u < n) { out[oy + 33u] = c75; }
            if (cbase + 34u < n) { out[oy + 34u] = c76; }
            if (cbase + 35u < n) { out[oy + 35u] = c77; }
        }
    }
}

// 掩码 softmax（最后一维，右后缀掩码广播）：out[r,j] = softmax(x[r,j] + mask[mb+j])，
// 其中 mb = (r*d) % mask_numel（mask 是输入形状的精确右后缀，逐行对齐）。
//
// 一个 workgroup 负责一行：256 线程按 stride 256 扫该行，同一 warp 的 32 个线程读连续地址，
// 合并成整条 cache line。旧版是「一个线程负责一行」，同一行的相邻元素由同一线程串行访问，
// 每次 load 独占一条 128B 事务 —— 实测 8.4M 元素要 173ms（≈200 MB/s），占整步 1/3 时间。
// 行内最大值/和用共享内存做树形归约（log2(256) = 8 步）。
var<workgroup> softmax_red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn softmax_fwd_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let d = params.p1;
    let mn = params.p2;
    let r = wid.x;
    // 整块一起退出（条件只取决于 workgroup_id/uniform，块内保持一致）：
    // 后面的 workgroupBarrier 必须由全部 256 个线程执行。
    if (r >= rows) {
        return;
    }
    let base = r * d;
    let mb = base % mn;
    let t = lid.x;
    // 第一趟：行最大值（数值稳定用）
    var mx = -3.402823e38;
    for (var j = t; j < d; j = j + 256u) {
        mx = max(mx, a[base + j] + b[mb + j]);
    }
    softmax_red[t] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            softmax_red[t] = max(softmax_red[t], softmax_red[t + s]);
        }
        workgroupBarrier();
    }
    let maxv = softmax_red[0];
    // 归约结果必须先被**每个**线程读走，才能把同一块共享数组改作分母用：
    // 少了这道 barrier，先跑完 exp 的 warp 会把 softmax_red[t] 覆写成部分和，
    // 而落后的 warp 此时才来读 softmax_red[0]，拿到的不是最大值而是别人的部分和 ——
    // 这一行整行的 exp 全错（实测前向输出误差达 12%~25%，且随调度漂移而变）。
    workgroupBarrier();
    // 第二趟：exp 并原地写回，同时累加分母
    var sum = 0.0;
    for (var j = t; j < d; j = j + 256u) {
        let e = exp(a[base + j] + b[mb + j] - maxv);
        out[base + j] = e;
        sum = sum + e;
    }
    softmax_red[t] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            softmax_red[t] = softmax_red[t] + softmax_red[t + s];
        }
        workgroupBarrier();
    }
    let inv = 1.0 / softmax_red[0];
    // 第三趟：归一化
    for (var j = t; j < d; j = j + 256u) {
        out[base + j] = out[base + j] * inv;
    }
}

// 掩码 softmax 反向：dx[r,j] = p[r,j] * (g[r,j] - Σ_j p·g)（p 为前向概率，与掩码无关）
@compute @workgroup_size(256, 1, 1)
fn softmax_bwd_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let d = params.p1;
    let r = wid.x;
    if (r >= rows) {
        return;
    }
    let base = r * d;
    let t = lid.x;
    var dot = 0.0;
    for (var j = t; j < d; j = j + 256u) {
        dot = dot + a[base + j] * b[base + j]; // g · p
    }
    softmax_red[t] = dot;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            softmax_red[t] = softmax_red[t] + softmax_red[t + s];
        }
        workgroupBarrier();
    }
    let dv = softmax_red[0];
    for (var j = t; j < d; j = j + 256u) {
        out[base + j] = b[base + j] * (a[base + j] - dv);
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

// 峰值探针：纯寄存器 FMA，测这台机器上 GPU 的浮点吞吐上限。
// 用途（历史）：当时 matmul 只跑到 ~36 GFLOP/s（MX150 峰值的 3%），需要先分清是「着色器写得差」
// 还是「这块 15W 入门卡的硬件上限本来就这么低」—— 这决定后面值不值得重写内核。
// 结论：值得。探针实测 524 GFLOP/s（标称峰值的 44%），说明瓶颈在共享内存 LDS 往返而非硬件；
// 按这个结论改成 8×8 分块后，各形状已到 200~230 GFLOP/s（见下方形状回放表）。
// 8 条互不依赖的累加链（ILP=8）避免被 FMA 延迟绑死；循环外的条件写回让编译器无法删掉整个循环。
@compute @workgroup_size(256, 1, 1)
fn fma_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let iters = params.p0;
    let a = f32(gid.x % 13u) * 1e-3 + 0.5;
    var a0 = 0.0; var a1 = 0.0; var a2 = 0.0; var a3 = 0.0;
    var a4 = 0.0; var a5 = 0.0; var a6 = 0.0; var a7 = 0.0;
    for (var i = 0u; i < iters; i = i + 1u) {
        a0 = a0 * a + a; a1 = a1 * a + a; a2 = a2 * a + a; a3 = a3 * a + a;
        a4 = a4 * a + a; a5 = a5 * a + a; a6 = a6 * a + a; a7 = a7 * a + a;
    }
    let s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (s > 1e30) { out[gid.x] = s; }
}
"#;

/// 输出头交叉熵（前向 + 反向融合）的独立着色器模块。
///
/// 单独一个模块、不用 `SHADER`：它多两个绑定（targets 与每行 loss），
/// 而 wgpu 要求「着色器声明的 binding 必须在 pipeline layout 里存在」，
/// 若塞进 `SHADER` 就会让 matmul/softmax 等既有管线也非要提供这两个绑定不可。
const SHADER_LM_CE: &str = r#"
struct Params {
    p0: u32, // rows（一行 = 一个 token 位置的词表分布）
    p1: u32, // vocab
    p2: u32, p3: u32, p4: u32, p5: u32,
}

@group(0) @binding(0) var<storage, read> logits: array<f32>;
@group(0) @binding(1) var<storage, read> target_ids: array<u32>;
@group(0) @binding(2) var<storage, read_write> dlogits: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> row_loss: array<f32>;

var<workgroup> lm_red: array<f32, 256>;

// 一个 workgroup 负责一行，256 线程按 stride 扫该行，两趟归约（最大值、指数和）。
//
// 为什么融合：logits 是 [rows, vocab]，本配置下 4096×8192 = 33.6M 元素。
// 逐算子版要把 logits 回读给 CPU 做 log_softmax，再回读同样大的 dlogits 传回 GPU ——
// 一步来回 268 MB（实测 445 ms），而这两个张量都只是中间结果。
// 这里让它们全程留在显存，只回读 row_loss（每行一个 f32，16 KB）。
//
// dlogits 不乘上游梯度：loss 是标量、梯度恒为 1，softmax - onehot 就是最终梯度；
// 对 rows 求平均的 1/rows 在这里直接乘掉，调用方拿到的即「已除 rows」的梯度。
@compute @workgroup_size(256, 1, 1)
fn lm_ce_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let d = params.p1;
    let r = wid.x;
    // 整块一起退出（条件只取决于 workgroup_id），后面的 workgroupBarrier 才安全
    if (r >= rows) {
        return;
    }
    let base = r * d;
    let t = lid.x;
    let tgt = target_ids[r];
    // 第一趟：行最大值（数值稳定）
    var mx = -3.402823e38;
    for (var j = t; j < d; j = j + 256u) {
        mx = max(mx, logits[base + j]);
    }
    lm_red[t] = mx;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            lm_red[t] = max(lm_red[t], lm_red[t + s]);
        }
        workgroupBarrier();
    }
    let maxv = lm_red[0];
    // 所有线程读走 maxv 之后才能复用同一块共享数组（少了这道 barrier 就是数据竞争）
    workgroupBarrier();
    // 第二趟：指数和
    var sum = 0.0;
    for (var j = t; j < d; j = j + 256u) {
        sum = sum + exp(logits[base + j] - maxv);
    }
    lm_red[t] = sum;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s = s >> 1u) {
        if (t < s) {
            lm_red[t] = lm_red[t] + lm_red[t + s];
        }
        workgroupBarrier();
    }
    let lse = log(lm_red[0]) + maxv; // logsumexp
    if (t == 0u) {
        row_loss[r] = lse - logits[base + tgt];
    }
    let inv_rows = 1.0 / f32(rows);
    for (var j = t; j < d; j = j + 256u) {
        var g = exp(logits[base + j] - lse);
        if (j == tgt) {
            g = g - 1.0; // -onehot：正确类别那一项减 1
        }
        dlogits[base + j] = g * inv_rows;
    }
}
"#;

/// 逐元素 / 归一化算子的独立着色器模块（MLP 子层常驻显存路径用）。
///
/// 与 `SHADER_LM_CE` 同理单独成模块：这些内核用到的绑定数各不相同（2~5 个），
/// 而 wgpu 要求「着色器静态使用的 binding 必须在管线布局里存在」，
/// 塞进 `SHADER` 会逼着 matmul / softmax 等既有管线也跟着提供这些多余的绑定。
///
/// 两条约定：
/// 1. 参数 uniform 一律放 binding 3 —— `GpuContext::batch_dispatch` 固定往 3 写参数；
/// 2. 模块级声明的读写权限取所有内核的并集，各内核只用到其中几个 ——
///    naga 按**入口点实际用到的绑定**生成管线接口，所以不同内核仍能配不同布局。
///
/// dropout 的掩码不占显存：它是 (种子, 下标) 的确定性函数，
/// 反向传同一个种子**重算**一遍即可，前向既不用存也不用回读。
const SHADER_ELEM: &str = r#"
struct Params {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
}

@group(0) @binding(0) var<storage, read> ea: array<f32>;
@group(0) @binding(1) var<storage, read> eb: array<f32>;
@group(0) @binding(2) var<storage, read> ec: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;
@group(0) @binding(4) var<storage, read_write> eo: array<f32>;
@group(0) @binding(5) var<storage, read_write> ep: array<f32>;

var<workgroup> elem_red: array<f32, 256>;
var<workgroup> elem_red2: array<f32, 256>;

/// splitmix32 终混：把 32 位种子打散成 [0,1) 均匀值。
/// 同一 seed 得到相同结果，同时让相邻下标的输出彼此独立 —— dropout 掩码的质量取决于此。
fn hash01(seed: u32) -> f32 {
    var z = seed;
    z = (z ^ (z >> 16u)) * 0x7feb352du;
    z = (z ^ (z >> 15u)) * 0x846ca68bu;
    z = z ^ (z >> 16u);
    return f32(z >> 8u) * (1.0 / 16777216.0); // 取高 24 位
}

/// dropout 的保留/丢弃因子：训练中按 p 丢弃并乘 1/(1-p)，其余情况恒等（返回 1）。
/// 反向重算掩码时与前向传同一个 `params.p3`，两边逐位一致。
fn drop_scale(i: u32) -> f32 {
    if (params.p4 == 0u) { return 1.0; }
    let p = bitcast<f32>(params.p2);
    if (p <= 0.0) { return 1.0; }
    let keep = 1.0 - p;
    if (hash01(params.p3 ^ (i * 0x9e3779b9u)) < keep) { return 1.0 / keep; }
    return 0.0;
}

/// 归一化前向（LayerNorm / RMSNorm 共用一套内核，靠模式位切换）：
///
/// ```text
/// LayerNorm（p3=0）：eo[r,j] = (x[r,j]−μ_r)·√(σ²_r+ε)⁻¹·γ_j + β_j
/// RMSNorm  （p3=1）：eo[r,j] = x[r,j]·√(E[x²]_r+ε)⁻¹·γ_j
/// ```
///
/// RMSNorm 是 LayerNorm 的一个**特例**：不减均值（μ≡0）、方差取「平方的均值」
/// 而不是「离差平方的均值」、没有平移项 β。所以同一套三趟归约结构只要改两处
/// ——第一趟累加 x² 而非 x，第二趟的中心化量取 0——第二趟的离差平方和就自然退化成
/// `Σx²/d`，用不着另写一条内核，也不会多一次显存往返。
///
/// 并把每行的 (μ, 1/σ) 写进 ep（反向直接复用，省一次重算；RMS 模式下 μ 写 0）。
/// p0=rows, p1=d, p2=ε(bitcast), p3=模式(0=LayerNorm, 1=RMSNorm)；
/// 绑定：0=x, 1=γ, 2=β, 4=输出, 5=每行统计量
@compute @workgroup_size(256, 1, 1)
fn ln_fwd_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let d = params.p1;
    let eps = bitcast<f32>(params.p2);
    let is_rms = params.p3 == 1u;
    let r = wid.x;
    // 整块一起退出（条件只取决于 workgroup_id），后面的 workgroupBarrier 才安全
    if (r >= rows) { return; }
    let base = r * d;
    let t = lid.x;
    // 第一趟：LayerNorm 求和求均值；RMSNorm 直接求平方和（不做中心化）
    var s = 0.0;
    if (is_rms) {
        for (var j = t; j < d; j = j + 256u) {
            let xv = ea[base + j];
            s = s + xv * xv;
        }
    } else {
        for (var j = t; j < d; j = j + 256u) { s = s + ea[base + j]; }
    }
    elem_red[t] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (t < k) { elem_red[t] = elem_red[t] + elem_red[t + k]; }
        workgroupBarrier();
    }
    // RMSNorm 不减均值：中心化量恒为 0，第二趟的离差平方和就退化成 E[x²]
    let mean = select(elem_red[0] / f32(d), 0.0, is_rms);
    // 每个线程读走均值之后才能复用同一块共享数组（少了这道 barrier 就是数据竞争）
    workgroupBarrier();
    // 第二趟：方差（RMSNorm 下即平方的均值）
    var v = 0.0;
    for (var j = t; j < d; j = j + 256u) {
        let c = ea[base + j] - mean;
        v = v + c * c;
    }
    elem_red[t] = v;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (t < k) { elem_red[t] = elem_red[t] + elem_red[t + k]; }
        workgroupBarrier();
    }
    let istd = 1.0 / sqrt(elem_red[0] / f32(d) + eps);
    if (t == 0u) {
        ep[r * 2u] = mean;
        ep[r * 2u + 1u] = istd;
    }
    // 第三趟：归一化 + 仿射（RMSNorm 没有 β，该项取 0）
    for (var j = t; j < d; j = j + 256u) {
        eo[base + j] = (ea[base + j] - mean) * istd * eb[j] + select(ec[j], 0.0, is_rms);
    }
}

/// 归一化反向（对输入），同样 LayerNorm / RMSNorm 共用：
///   eo[r,j] = istd·(dy_j·γ_j − m1 − m2·x̂_j)
///   m1 = Σ_j dy_j·γ_j / d，m2 = Σ_j dy_j·γ_j·x̂_j / d，x̂ = (x−μ)·istd
///
/// RMSNorm 把 `m1` 置 0：它的梯度公式里没有「减去 dyγ 的均值」这一项
/// （对应前向不减均值），而 `m2` 这一项在 μ=0 时正好退化成 RMSNorm 需要的
/// `E[dyγ·x̂]·x̂`。于是同一条内核、只差一个 `select`。
///
/// p0=rows, p1=d, p2=模式(0=LayerNorm, 1=RMSNorm)；绑定：0=上游梯度 dy, 1=x, 2=每行统计量, 4=输出 dx, 5=γ
@compute @workgroup_size(256, 1, 1)
fn ln_bwd_x_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let d = params.p1;
    let is_rms = params.p2 == 1u;
    let r = wid.x;
    if (r >= rows) { return; }
    let base = r * d;
    let t = lid.x;
    let mean = ec[r * 2u];
    let istd = ec[r * 2u + 1u];
    // m1、m2 两个归约用两块共享数组同时做，省一半 barrier
    var a1 = 0.0;
    var a2 = 0.0;
    for (var j = t; j < d; j = j + 256u) {
        let xn = (eb[base + j] - mean) * istd;
        let dyg = ea[base + j] * ep[j];
        a1 = a1 + dyg;
        a2 = a2 + dyg * xn;
    }
    elem_red[t] = a1;
    elem_red2[t] = a2;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (t < k) {
            elem_red[t] = elem_red[t] + elem_red[t + k];
            elem_red2[t] = elem_red2[t] + elem_red2[t + k];
        }
        workgroupBarrier();
    }
    let inv_d = 1.0 / f32(d);
    let m1 = select(elem_red[0] * inv_d, 0.0, is_rms);
    let m2 = elem_red2[0] * inv_d;
    for (var j = t; j < d; j = j + 256u) {
        let xn = (eb[base + j] - mean) * istd;
        eo[base + j] = istd * (ea[base + j] * ep[j] - m1 - m2 * xn);
    }
}

/// 归一化反向（对 γ/β）：一个 workgroup 负责**一列**，跨行累加：
///   dγ_j = Σ_r dy[r,j]·x̂[r,j]，  dβ_j = Σ_r dy[r,j]
/// p0=rows, p1=d；绑定：0=上游梯度 dy, 1=x, 2=每行统计量, 4=dγ, 5=dβ
///
/// RMSNorm 直接复用：它的 `dγ` 就是 `Σ_r dy·x̂`（μ=0 时 x̂ 即 x·istd，同上），
/// 而 RMSNorm 没有 β，写出的 `dβ` 无人接收（调用方给一块临时缓冲即可）。
///
/// 逐列而不是逐行：dγ/dβ 天然是「列方向」的归约，让一个 workgroup 独占一列即可
/// 直接得到最终值，不必引入浮点原子加（WGSL 没有 atomicAdd<f32>）。
@compute @workgroup_size(256, 1, 1)
fn ln_bwd_gb_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let d = params.p1;
    let j = wid.x;
    if (j >= d) { return; }
    let t = lid.x;
    var a1 = 0.0;
    var a2 = 0.0;
    for (var r = t; r < rows; r = r + 256u) {
        let base = r * d;
        let mean = ec[r * 2u];
        let istd = ec[r * 2u + 1u];
        let dy = ea[base + j];
        a1 = a1 + dy * (eb[base + j] - mean) * istd;
        a2 = a2 + dy;
    }
    elem_red[t] = a1;
    elem_red2[t] = a2;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (t < k) {
            elem_red[t] = elem_red[t] + elem_red[t + k];
            elem_red2[t] = elem_red2[t] + elem_red2[t + k];
        }
        workgroupBarrier();
    }
    if (t == 0u) {
        eo[j] = elem_red[0];
        ep[j] = elem_red2[0];
    }
}

/// GELU（tanh 近似）融合偏置：eo = gelu(x + b)，ep = x + b（GELU 前的输入，反向要用；
/// 存下来比在反向重算一次线性投影便宜得多）。
/// p0=元素数, p1=偏置长度；绑定：0=线性层输出, 1=偏置, 4=输出, 5=x+b
@compute @workgroup_size(256, 1, 1)
fn gelu_fwd_bias_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let cols = params.p1;
    let i = gid.x;
    if (i >= len) { return; }
    let x = ea[i] + eb[i % cols];
    let t = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    eo[i] = 0.5 * x * (1.0 + t);
    ep[i] = x;
}

/// GELU 反向：eo = dg · d/dx[0.5·x·(1+tanh(a))]，a = √(2/π)(x + 0.044715x³)
/// p0=元素数；绑定：0=上游梯度, 1=前向存下的 x, 4=输出
@compute @workgroup_size(256, 1, 1)
fn gelu_bwd_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let i = gid.x;
    if (i >= len) { return; }
    let x = eb[i];
    let t = tanh(0.7978845608 * (x + 0.044715 * x * x * x));
    let da_dx = 0.7978845608 * (1.0 + 3.0 * 0.044715 * x * x);
    let dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * da_dx;
    eo[i] = ea[i] * dy_dx;
}

/// 线性层偏置 + dropout + 残差，三个逐元素算子融成一个：
///   eo[i] = ec[i] + drop_scale(i)·(ea[i] + b₂[i % p1])   （ec 是残差输入 x）
/// p0=元素数, p1=偏置长度, p2=p(bitcast), p3=dropout 种子, p4=是否训练
/// 绑定：0=linear2 输出, 1=b₂, 2=残差输入, 4=输出
@compute @workgroup_size(256, 1, 1)
fn bias_dropout_residual_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let cols = params.p1;
    let i = gid.x;
    if (i >= len) { return; }
    let y = ea[i] + eb[i % cols];
    eo[i] = ec[i] + drop_scale(i) * y;
}

/// dropout 反向：eo = dout · mask（掩码由种子重算，与前向逐位一致）
/// p0=元素数, p2=p(bitcast), p3=种子, p4=是否训练；绑定：0=上游梯度, 4=输出
@compute @workgroup_size(256, 1, 1)
fn dropout_bwd_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let len = params.p0;
    let i = gid.x;
    if (i >= len) { return; }
    eo[i] = ea[i] * drop_scale(i);
}

/// 按列求和：eo[j] = Σ_r a[r,j]（偏置的梯度就是这一项）。一个 workgroup 负责一列。
/// p0=rows, p1=cols；绑定：0=输入, 4=输出
@compute @workgroup_size(256, 1, 1)
fn col_sum_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let rows = params.p0;
    let cols = params.p1;
    let j = wid.x;
    if (j >= cols) { return; }
    let t = lid.x;
    var s = 0.0;
    for (var r = t; r < rows; r = r + 256u) { s = s + ea[r * cols + j]; }
    elem_red[t] = s;
    workgroupBarrier();
    for (var k = 128u; k > 0u; k = k >> 1u) {
        if (t < k) { elem_red[t] = elem_red[t] + elem_red[t + k]; }
        workgroupBarrier();
    }
    if (t == 0u) { eo[j] = elem_red[0]; }
}

/// 把 `[B,T,H,hd]` 的 Q/K/V 投影结果搬成按头分组的 `[B*H,T,hd]`，顺带叠加偏置。
/// 模式 1 同时做 RoPE 旋转（Q/K 用），模式 2 只搬运（反向第一步用，不加偏置不旋转）。
///
/// 每个线程处理**一对**相邻元素 (2p, 2p+1)：RoPE 天然按对操作，
/// 不旋转时也只是搬两个相邻元素，两种模式共用同一套下标分解。
/// 位置取自序列内下标 r —— 本路径只用于训练（`base = 0`、无 KV cache）。
///
/// p0=B, p1=T, p2=H, p3=hd, p4=模式(0=加偏置, 1=加偏置+RoPE, 2=纯搬运), p5=输出缩放(bitcast)
/// 绑定：0=输入 [B,T,H,hd], 1=偏置 [H*hd], 4=输出 [B*H,T,hd]
@compute @workgroup_size(256, 1, 1)
fn heads_split_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = params.p0;
    let t = params.p1;
    let h = params.p2;
    let hd = params.p3;
    let mode = params.p4;
    let scale = bitcast<f32>(params.p5);
    let half = hd / 2u;
    let i = gid.x;
    if (i >= b * h * t * half) { return; }
    let p = i % half;
    let rest = i / half;
    let r = rest % t;
    let grp = rest / t;          // b*H + hh
    let hh = grp % h;
    let bb = grp / h;
    let src = ((bb * t + r) * h + hh) * hd + 2u * p;
    var v0 = ea[src];
    var v1 = ea[src + 1u];
    // 偏置必须在旋转**之前**加：模型里是 `c_q.forward(x)`（含偏置）再做 RoPE，
    // 即 R(z + b) 而非 R(z) + b —— 旋转是线性的，但 R(b) ≠ b。
    if (mode != 2u) {
        let bo = hh * hd + 2u * p;
        v0 = v0 + eb[bo];
        v1 = v1 + eb[bo + 1u];
    }
    if (mode == 1u) {
        let theta = f32(r) * pow(10000.0, -2.0 * f32(p) / f32(hd));
        let cs = cos(theta);
        let sn = sin(theta);
        let n0 = v0 * cs - v1 * sn;
        let n1 = v0 * sn + v1 * cs;
        v0 = n0;
        v1 = n1;
    }
    let dst = (grp * t + r) * hd + 2u * p;
    eo[dst] = v0 * scale;
    eo[dst + 1u] = v1 * scale;
}

/// [`heads_split_main`] 的逆向：把按头分组的 `[B*H,T,hd]` 搬回 `[B,T,H,hd]`。
/// 模式 1 用旋转矩阵的转置（负角度）回传 RoPE 的梯度，模式 0 只搬运。
///
/// p0=B, p1=T, p2=H, p3=hd, p4=模式(0=搬运, 1=逆旋转), p5=输出缩放(bitcast)
/// 绑定：0=输入 [B*H,T,hd], 4=输出 [B,T,H,hd]
@compute @workgroup_size(256, 1, 1)
fn heads_join_main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let b = params.p0;
    let t = params.p1;
    let h = params.p2;
    let hd = params.p3;
    let mode = params.p4;
    let scale = bitcast<f32>(params.p5);
    let half = hd / 2u;
    let i = gid.x;
    if (i >= b * t * h * half) { return; }
    let p = i % half;
    let rest = i / half;
    let hh = rest % h;
    let rest2 = rest / h;
    let r = rest2 % t;
    let bb = rest2 / t;
    let src = ((bb * h + hh) * t + r) * hd + 2u * p;
    var g0 = ea[src];
    var g1 = ea[src + 1u];
    if (mode == 1u) {
        let theta = f32(r) * pow(10000.0, -2.0 * f32(p) / f32(hd));
        let cs = cos(theta);
        let sn = sin(theta);
        let n0 = g0 * cs + g1 * sn;
        let n1 = -g0 * sn + g1 * cs;
        g0 = n0;
        g1 = n1;
    }
    let dst = ((bb * t + r) * h + hh) * hd + 2u * p;
    eo[dst] = g0 * scale;
    eo[dst + 1u] = g1 * scale;
}
"#;

/// GPU 上下文：持有设备、队列与编译好的计算管线
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    info: wgpu::AdapterInfo,
    matmul_pipe: wgpu::ComputePipeline,
    /// 小 tile（64×64）matmul 变体，见 [`use_small_tile`]；与 `matmul_pipe` 共用 `matmul_layout`
    matmul_small_pipe: wgpu::ComputePipeline,
    matmul_layout: wgpu::BindGroupLayout,
    scale_pipe: wgpu::ComputePipeline,
    relu_pipe: wgpu::ComputePipeline,
    unary_layout: wgpu::BindGroupLayout,
    add_pipe: wgpu::ComputePipeline,
    add_layout: wgpu::BindGroupLayout,
    softmax_fwd_pipe: wgpu::ComputePipeline,
    softmax_bwd_pipe: wgpu::ComputePipeline,
    /// 输出头交叉熵（前向 + 反向融合，独立模块与布局，见 `SHADER_LM_CE`）
    lm_ce_pipe: wgpu::ComputePipeline,
    lm_ce_layout: wgpu::BindGroupLayout,
    // ---- 逐元素 / 归一化内核（独立模块 `SHADER_ELEM`，MLP 子层常驻显存路径用）----
    ln_fwd_pipe: wgpu::ComputePipeline,
    ln_bwd_x_pipe: wgpu::ComputePipeline,
    ln_bwd_gb_pipe: wgpu::ComputePipeline,
    gelu_fwd_pipe: wgpu::ComputePipeline,
    gelu_bwd_pipe: wgpu::ComputePipeline,
    /// 偏置 + dropout + 残差（三个逐元素算子融合）
    bdr_pipe: wgpu::ComputePipeline,
    dropout_bwd_pipe: wgpu::ComputePipeline,
    col_sum_pipe: wgpu::ComputePipeline,
    /// 按头重排 + 偏置 + RoPE（前向 Q/K/V）
    heads_split_pipe: wgpu::ComputePipeline,
    /// 按头重排的逆向 + RoPE 反向（反向 dQ/dK/dV）
    heads_join_pipe: wgpu::ComputePipeline,
    /// 布局 A：{0 ro, 1 ro, 2 ro, 4 rw}——ln 前向、偏置+dropout+残差
    elem_a_layout: wgpu::BindGroupLayout,
    /// 布局 B：{0 ro, 1 ro, 2 ro, 4 rw, 5 rw}——ln 前向、ln 反向两趟
    elem_b_layout: wgpu::BindGroupLayout,
    /// 布局 C：{0 ro, 1 ro, 4 rw, 5 rw}——gelu 前向（融合偏置）
    elem_c_layout: wgpu::BindGroupLayout,
    /// 布局 D：{0 ro, 1 ro, 4 rw}——gelu 反向
    elem_d_layout: wgpu::BindGroupLayout,
    /// 布局 E：{0 ro, 4 rw}——dropout 反向、按列求和
    elem_e_layout: wgpu::BindGroupLayout,
    /// 峰值探针管线（见 `probe_fma`）
    fma_pipe: wgpu::ComputePipeline,
    /// 复用的参数 uniform buffer（24 字节，每次 write_buffer 覆盖）
    params_buf: wgpu::Buffer,
    /// 存储/读回 buffer 池：键 = (字节数, 用途)，按需取还，避免每算子新建 GPU 对象
    /// （用 Mutex 保证 GpuContext 可放进静态 OnceLock）
    pool: Mutex<HashMap<(u64, u8), Vec<wgpu::Buffer>>>,
}

/// 全局 GPU 上下文（初始化一次；None 表示不可用）
static GPU: OnceLock<Option<GpuContext>> = OnceLock::new();

/// 初始化 GPU 后端（幂等）。失败时静默置为不可用，后续自动走 CPU。
///
/// 在独立线程（8 MB 栈）中初始化 wgpu，避免 Intel/Vulkan 驱动的栈缓冲区溢出。
pub fn init() {
    // 允许用环境变量覆盖 matmul 分流阈值：盈亏平衡点取决于本机的 CPU 算力与 PCIe 带宽，
    // 无法用编译期常量适配所有机器，标定时在外部设置即可，不必改代码重编。
    if let Ok(v) = std::env::var("LLM_GPU_MATMUL_MIN_FLOPS") {
        if let Ok(n) = v.parse::<usize>() {
            MATMUL_MIN_FLOPS.store(n, Ordering::Relaxed);
            println!("[gpu] matmul 分流阈值被环境变量覆盖为 {n} FLOPs");
        }
    }
    // 强制 matmul 用大/小 tile（标定对照用，见 MM_SMALL_FORCE）
    if let Ok(v) = std::env::var("LLM_GPU_MM_SMALL") {
        let f = if v == "0" { 0 } else { 1 };
        MM_SMALL_FORCE.store(f, Ordering::Relaxed);
        println!("[gpu] matmul tile 被环境变量强制为{}", if f == 0 { "大 (128×128)" } else { "小 (64×64)" });
    }
    let ctx = std::thread::Builder::new()
        .name("gpu-init".into())
        .stack_size(8 * 1024 * 1024) // 8 MB 栈，避免驱动栈溢出
        .spawn(create)
        .ok()
        .and_then(|h| h.join().ok())
        .flatten();
    let _ = GPU.set(ctx);
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
    // 判定实验用：录制模式下只记形状、返回 None（CPU 兜底保证训练数值不受影响）
    if PROBE_CAPTURE.load(Ordering::Relaxed) {
        PROBE_SHAPES
            .lock()
            .unwrap()
            .push((m, k, n, batch, a_t, b_t));
        return None;
    }
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
    // 优先 DX12（Windows 原生更稳定），Vulkan 回退；Intel 核显 Vulkan 驱动有栈溢出 bug
    let backends = if cfg!(windows) {
        wgpu::Backends::DX12
    } else {
        wgpu::Backends::PRIMARY
    };
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
        backends,
        flags: wgpu::InstanceFlags::default(),
        memory_budget_thresholds: wgpu::MemoryBudgetThresholds::default(),
        backend_options: wgpu::BackendOptions::default(),
        display: None,
    });

    // 枚举所有适配器，优先选择独显（DiscreteGpu），避免选到核显
    let mut adapters: Vec<wgpu::Adapter> = block_on(instance.enumerate_adapters(backends));
    if adapters.is_empty() {
        eprintln!("[gpu] 未找到任何 GPU 适配器");
        return None;
    }
    // 打印所有可用适配器供调试
    for (i, a) in adapters.iter().enumerate() {
        let info = a.get_info();
        let device_type = match info.device_type {
            wgpu::DeviceType::DiscreteGpu => "DiscreteGpu",
            wgpu::DeviceType::IntegratedGpu => "IntegratedGpu",
            wgpu::DeviceType::Cpu => "Cpu",
            _ => "Other",
        };
        println!("[gpu] 适配器 {}: {} ({}, {:?})", i, info.name, device_type, info.backend);
    }
    // 排序：独显优先，其次集成显卡
    adapters.sort_by(|a, b| {
        let da = a.get_info().device_type;
        let db = b.get_info().device_type;
        let pa = match da {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        };
        let pb = match db {
            wgpu::DeviceType::DiscreteGpu => 0,
            wgpu::DeviceType::IntegratedGpu => 1,
            _ => 2,
        };
        pa.cmp(&pb)
    });
    let adapter = adapters.into_iter().next()?;
    let info = adapter.get_info();
    let device_type = match info.device_type {
        wgpu::DeviceType::DiscreteGpu => "DiscreteGpu",
        wgpu::DeviceType::IntegratedGpu => "IntegratedGpu",
        wgpu::DeviceType::Cpu => "Cpu",
        _ => "Other",
    };
    println!("[gpu] 已选择适配器: {} ({})", info.name, device_type);
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
    // 小 tile 变体：绑定与布局完全一致，只是 workgroup 尺寸与 tile 边长不同
    let matmul_small_pipe = make_pipeline(&device, &matmul_pl, "matmul_small_main");
    let scale_pipe = make_pipeline(&device, &unary_pl, "scale_main");
    let relu_pipe = make_pipeline(&device, &unary_pl, "relu_main");
    let add_pipe = make_pipeline(&device, &add_pl, "add_main");
    // softmax 前向/反向共用 matmul 的 4 绑定布局（a/b/out/params）
    let softmax_fwd_pipe = make_pipeline(&device, &matmul_pl, "softmax_fwd_main");
    let softmax_bwd_pipe = make_pipeline(&device, &matmul_pl, "softmax_bwd_main");
    // 峰值探针复用一元布局（只用 out 与 params，输入绑定占位即可）
    let fma_pipe = make_pipeline(&device, &unary_pl, "fma_main");

    // 输出头交叉熵：独立模块（比通用布局多 target_ids / row_loss 两个绑定）
    let lm_ce_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("lm_ce"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_LM_CE)),
    });
    let lm_ce_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("lm_ce_layout"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            uniform_entry(3),
            storage_entry(4, false),
        ],
    });
    let lm_ce_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("lm_ce_pl"),
        bind_group_layouts: &[Some(&lm_ce_layout)],
        immediate_size: 0,
    });
    let lm_ce_pipe = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("lm_ce_main"),
        layout: Some(&lm_ce_pl),
        module: &lm_ce_module,
        entry_point: Some("lm_ce_main"),
        compilation_options: Default::default(),
        cache: None,
    });

    // 逐元素 / 归一化内核：独立模块（各内核用到的绑定数不同，见 `SHADER_ELEM` 的说明）
    let elem_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("elem"),
        source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(SHADER_ELEM)),
    });
    let make_elem = |device: &wgpu::Device, layout: &wgpu::PipelineLayout, entry: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry),
            layout: Some(layout),
            module: &elem_module,
            entry_point: Some(entry),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    // 每个内核用到的绑定集合不同（读写权限也不同），逐一定制布局；
    // 参数 uniform 固定 binding 3，由 `batch_dispatch` 自动补上。
    let elem_layouts: Vec<wgpu::BindGroupLayout> = [
        ("elem_a_layout", vec![(0u32, true), (1, true), (2, true), (4, false)]),
        ("elem_b_layout", vec![(0, true), (1, true), (2, true), (4, false), (5, false)]),
        ("elem_c_layout", vec![(0, true), (1, true), (4, false), (5, false)]),
        ("elem_d_layout", vec![(0, true), (1, true), (4, false)]),
        ("elem_e_layout", vec![(0, true), (4, false)]),
    ]
    .into_iter()
    .map(|(label, spec)| {
        let mut entries: Vec<wgpu::BindGroupLayoutEntry> =
            spec.iter().map(|&(b, ro)| storage_entry(b, ro)).collect();
        entries.push(uniform_entry(3));
        device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(label),
            entries: &entries,
        })
    })
    .collect();
    let elem_pl = |device: &wgpu::Device, layout: &wgpu::BindGroupLayout| {
        device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("elem_pl"),
            bind_group_layouts: &[Some(layout)],
            immediate_size: 0,
        })
    };
    let (pl_a, pl_b, pl_c, pl_d, pl_e) = (
        elem_pl(&device, &elem_layouts[0]),
        elem_pl(&device, &elem_layouts[1]),
        elem_pl(&device, &elem_layouts[2]),
        elem_pl(&device, &elem_layouts[3]),
        elem_pl(&device, &elem_layouts[4]),
    );
    let ln_fwd_pipe = make_elem(&device, &pl_b, "ln_fwd_main");
    let ln_bwd_x_pipe = make_elem(&device, &pl_b, "ln_bwd_x_main");
    let ln_bwd_gb_pipe = make_elem(&device, &pl_b, "ln_bwd_gb_main");
    let gelu_fwd_pipe = make_elem(&device, &pl_c, "gelu_fwd_bias_main");
    let gelu_bwd_pipe = make_elem(&device, &pl_d, "gelu_bwd_main");
    let bdr_pipe = make_elem(&device, &pl_a, "bias_dropout_residual_main");
    let dropout_bwd_pipe = make_elem(&device, &pl_e, "dropout_bwd_main");
    let col_sum_pipe = make_elem(&device, &pl_e, "col_sum_main");
    let heads_split_pipe = make_elem(&device, &pl_d, "heads_split_main");
    let heads_join_pipe = make_elem(&device, &pl_e, "heads_join_main");

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
        matmul_small_pipe,
        matmul_layout,
        scale_pipe,
        relu_pipe,
        unary_layout,
        add_pipe,
        add_layout,
        softmax_fwd_pipe,
        softmax_bwd_pipe,
        lm_ce_pipe,
        lm_ce_layout,
        ln_fwd_pipe,
        ln_bwd_x_pipe,
        ln_bwd_gb_pipe,
        gelu_fwd_pipe,
        gelu_bwd_pipe,
        bdr_pipe,
        dropout_bwd_pipe,
        col_sum_pipe,
        heads_split_pipe,
        heads_join_pipe,
        elem_a_layout: elem_layouts[0].clone(),
        elem_b_layout: elem_layouts[1].clone(),
        elem_c_layout: elem_layouts[2].clone(),
        elem_d_layout: elem_layouts[3].clone(),
        elem_e_layout: elem_layouts[4].clone(),
        fma_pipe,
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
    /// 批量矩阵乘（tiled：16×16 共享内存块，块内用共享内存复用两个输入矩阵的行/列）。
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
        // wgpu 默认限制每维度最多 65535 个 workgroup，超出则回退 CPU。
        // 按**小** tile 判：小 tile 的 workgroup 数只会更多，两边都兜得住。
        if m == 0
            || k == 0
            || n == 0
            || batch == 0
            || m.div_ceil(MM_TILE_S) > 65535
            || n.div_ceil(MM_TILE_S) > 65535
            || batch > 65535
        {
            return None;
        }
        // 尺寸阈值：太小不值得上 GPU（固定调度开销 > 计算收益），回退 CPU
        if 2 * m * k * n * batch < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) {
            return None;
        }
        let buf_a = self.take_buf(a.len(), KIND_IN);
        self.queue.write_buffer(&buf_a, 0, bytemuck_bytes(a));
        let buf_b = self.take_buf(b.len(), KIND_IN);
        self.queue.write_buffer(&buf_b, 0, bytemuck_bytes(b));
        let out_len = batch * m * n;
        let buf_out = self.take_buf(out_len, KIND_OUT);
        let (pipe, tile) = if use_small_tile() {
            (&self.matmul_small_pipe, MM_TILE_S)
        } else {
            (&self.matmul_pipe, MM_TILE)
        };
        let r = self.run(
            pipe,
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
            m.div_ceil(tile) as u32,
            n.div_ceil(tile) as u32,
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

    /// 掩码 softmax（前向）：一个 workgroup 处理一行（合并访存 + 共享内存归约）。
    fn softmax_mask(
        &self,
        x: &[f32],
        mask: &[f32],
        rows: usize,
        d: usize,
        mask_numel: usize,
    ) -> Option<Vec<f32>> {
        if rows == 0
            || d == 0
            || mask_numel == 0
            || rows * d < SOFTMAX_MIN_ELEMS
            || rows > 65535 // 每行一个 workgroup，受单维 65535 上限约束
        {
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
            rows as u32, // 一个 workgroup 一行
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
        if rows == 0 || d == 0 || rows * d < SOFTMAX_MIN_ELEMS || rows > 65535 {
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
            rows as u32, // 一个 workgroup 一行
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
            KIND_UNIFORM => wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            _ => wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        };
        let t = std::time::Instant::now();
        let buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("pooled"),
            size: (len * 4) as u64,
            usage,
            mapped_at_creation: false,
        });
        BUF_NEW_COUNT.fetch_add(1, Ordering::Relaxed);
        BUF_NEW_US.fetch_add(t.elapsed().as_micros() as u64, Ordering::Relaxed);
        buf
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
        // [诊断] 逐段计时，采集前 DIAG_MAX 次（够覆盖训练首步的全部 dispatch）
        let diag_t = std::time::Instant::now();
        static DIAG_COUNT: AtomicUsize = AtomicUsize::new(0);
        let diag_n = DIAG_COUNT.fetch_add(1, Ordering::Relaxed);
        let diag_dump = diag_n < DIAG_MAX;

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

        // 同步取回：这一步的算子需要立刻拿到结果（反向/下一层要用），所以阻塞等 GPU 完成；
        // 纯前向的整段链路走 `forward_*_resident` 那条常驻快路，不为每个算子付一次同步代价
        let slice = readback.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let diag_t_map = diag_t.elapsed();
        let _ = self
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let diag_t_poll = diag_t.elapsed();
        let view = match slice.get_mapped_range() {
            Ok(v) => v,
            Err(_) => {
                self.put_buf(readback);
                return None;
            }
        };
        let diag_t_view = diag_t.elapsed();
        let result = unsafe {
            std::slice::from_raw_parts(view.as_ptr() as *const f32, out_len).to_vec()
        };
        let diag_t_copy = diag_t.elapsed();
        drop(view);
        readback.unmap();
        self.put_buf(readback);
        if diag_dump {
            let total = diag_t.elapsed();
            // 这里的"同步"只覆盖 map+poll+拷回三段，不含随后的 unmap/归还 buffer
            let sync = diag_t_copy - diag_t_submit;
            GPU_DIAG_LOG.lock().unwrap().push(GpuDispatchDiag {
                x, y, z,
                out_len,
                total_ms: total.as_secs_f64() * 1000.0,
                upload_ms: diag_t_upload.as_secs_f64() * 1000.0,
                dispatch_ms: (diag_t_submit - diag_t_upload).as_secs_f64() * 1000.0,
                sync_ms: sync.as_secs_f64() * 1000.0,
                map_call_ms: (diag_t_map - diag_t_submit).as_secs_f64() * 1000.0,
                poll_ms: (diag_t_poll - diag_t_map).as_secs_f64() * 1000.0,
                view_ms: (diag_t_view - diag_t_poll).as_secs_f64() * 1000.0,
                copy_ms: (diag_t_copy - diag_t_view).as_secs_f64() * 1000.0,
            });
        }
        Some(result)
    }
}

/// 根据 buffer 的 usage 反查它在池中的用途键
fn kind_of(buf: &wgpu::Buffer) -> u8 {
    if buf.usage().contains(wgpu::BufferUsages::UNIFORM) {
        KIND_UNIFORM
    } else if buf.usage().contains(wgpu::BufferUsages::MAP_READ) {
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

// ==================== 批量执行器：录制式提交 ====================
//
// 为什么需要（实测依据见 logs\gpu_diag.log，单步 103 次 dispatch）：
//   参数上传 0.1ms(0%) | 绑定+编码+提交 5.3ms(10%) | poll+回读 48.3ms(90%) → 平均 53.7ms/次
// 5.5s/步里几乎全是"每算子重付一次"的固定开销，真正做浮点的时间只有百毫秒级。
//
// 批量执行器把整步的 dispatch 录进**同一个** CommandEncoder，末尾一次 submit + 一次 poll，
// 把这笔固定开销从 103 次摊到 1 次。算子结果留在显存不回读 —— 回读才是真正的同步点。
//
// 三个必须注意的点：
// 1. **参数不能共用一块 uniform**：所有 dispatch 若都读同一个 params_buf，后写的会覆盖先写的。
//    这里每个 dispatch 从池里取一块独立的 24 字节 uniform buffer。
// 2. **保活**：bind group 与临时 buffer 必须活到 submit 之后，统一挂在 `keep_*` 上。
// 3. **不下载**：结果留在显存，调用方需要时再显式回读。

/// 批量录制器：一步内的所有 dispatch 共用同一个 encoder
pub struct GpuBatch {
    enc: wgpu::CommandEncoder,
    /// 保活到 submit 之后（wgpu 在提交时才真正读取这些资源）
    keep_groups: Vec<wgpu::BindGroup>,
    keep_bufs: Vec<wgpu::Buffer>,
    /// 已录制的 dispatch 数
    pub dispatches: usize,
    /// 按 [`opk`] 类别计数的 dispatch 数（诊断用）
    kinds: [usize; opk::N],
}

impl GpuContext {
    /// 按管线对象识别算子类别，供 [`opk`] 分桶与 `LLM_GPU_ABLATE` 消融使用。
    ///
    /// 用指针同一性而不是给十几处调用点加标签：管线都在 `GpuContext` 里长期存活，
    /// 指针唯一且稳定，少一个调用点漏标就少一处静默记错账的机会。
    fn classify(&self, pipe: &wgpu::ComputePipeline) -> u8 {
        use opk::*;
        let eq = |p: &wgpu::ComputePipeline| std::ptr::eq(pipe, p);
        if eq(&self.matmul_pipe) || eq(&self.matmul_small_pipe) {
            MATMUL
        } else if eq(&self.softmax_fwd_pipe) {
            SOFTMAX_FWD
        } else if eq(&self.softmax_bwd_pipe) {
            SOFTMAX_BWD
        } else if eq(&self.lm_ce_pipe) {
            LM_CE
        } else if eq(&self.ln_fwd_pipe) {
            LN_FWD
        } else if eq(&self.ln_bwd_x_pipe) {
            LN_BWD_X
        } else if eq(&self.ln_bwd_gb_pipe) {
            LN_BWD_GB
        } else if eq(&self.gelu_fwd_pipe) {
            GELU_FWD
        } else if eq(&self.gelu_bwd_pipe) {
            GELU_BWD
        } else if eq(&self.bdr_pipe) {
            BDR
        } else if eq(&self.dropout_bwd_pipe) {
            DROPOUT_BWD
        } else if eq(&self.col_sum_pipe) {
            COL_SUM
        } else if eq(&self.heads_split_pipe) {
            HEADS_SPLIT
        } else if eq(&self.heads_join_pipe) {
            HEADS_JOIN
        } else if eq(&self.add_pipe) {
            ADD
        } else if eq(&self.scale_pipe) {
            SCALE
        } else if eq(&self.relu_pipe) {
            RELU
        } else {
            OTHER
        }
    }

    /// 开一个批量录制器
    fn batch_begin(&self) -> GpuBatch {
        GpuBatch {
            enc: self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("batch") }),
            keep_groups: Vec::new(),
            keep_bufs: Vec::new(),
            dispatches: 0,
            kinds: [0; opk::N],
        }
    }

    /// 批量内录制一次通用 dispatch：只往 encoder 写命令，不提交、不回读。
    /// `bufs` 是 (buffer, 着色器 binding) 对，参数 uniform 固定绑定 3。
    ///
    /// **每个 dispatch 都要从池里另取一块 24 字节 uniform**：所有 dispatch 若共用同一块，
    /// `write_buffer` 的写入全部在提交开头生效，后写的会把先写的覆盖掉，让先录的 dispatch 读到错参数。
    #[allow(clippy::too_many_arguments)]
    fn batch_dispatch(
        &self,
        batch: &mut GpuBatch,
        pipe: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        bufs: &[(&wgpu::Buffer, u32)],
        params: [u32; 6],
        x: u32,
        y: u32,
        z: u32,
    ) {
        let kind = self.classify(pipe);
        // 消融实验（见 ablate_kinds）：只计数、不录制，输出 buffer 由调用方照常分配，
        // 于是从 GPU 时间线上只去掉了这一类内核，其余依赖结构与基线完全一致。
        if ablate_kinds().contains(&kind) {
            batch.kinds[kind as usize] += 1;
            return;
        }
        let pbuf = self.take_buf(6, KIND_UNIFORM); // 6 个 u32 = 24 字节
        self.queue.write_buffer(&pbuf, 0, bytemuck_bytes(&params));
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
                buffer: &pbuf,
                offset: 0,
                size: None,
            }),
        });
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("batch_bg"),
            layout,
            entries: &entries,
        });
        {
            let mut pass = batch
                .enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor::default());
            pass.set_pipeline(pipe);
            pass.set_bind_group(0, &bg, &[]);
            pass.dispatch_workgroups(x, y, z);
        }
        batch.keep_groups.push(bg);
        batch.keep_bufs.push(pbuf);
        batch.dispatches += 1;
        batch.kinds[kind as usize] += 1;
    }

    /// 批量内录制一次 matmul
    #[allow(clippy::too_many_arguments)]
    fn batch_matmul(
        &self,
        batch: &mut GpuBatch,
        a: &wgpu::Buffer,
        b: &wgpu::Buffer,
        out: &wgpu::Buffer,
        m: usize,
        k: usize,
        n: usize,
        bs: usize,
        a_t: bool,
        b_t: bool,
    ) {
        if mm_ablated(m, k, n, bs) {
            batch.kinds[opk::MATMUL as usize] += 1;
            return;
        }
        // 大小 tile 分流（见 use_small_tile）：两者逐位同结果，只是并行粒度不同
        let (pipe, tile) = if use_small_tile() {
            (&self.matmul_small_pipe, MM_TILE_S)
        } else {
            (&self.matmul_pipe, MM_TILE)
        };
        self.batch_dispatch(
            batch,
            pipe,
            &self.matmul_layout,
            &[(a, 0), (b, 1), (out, 2)],
            [bs as u32, m as u32, k as u32, n as u32, a_t as u32, b_t as u32],
            m.div_ceil(tile) as u32,
            n.div_ceil(tile) as u32,
            bs as u32,
        );
    }

    /// 提交整批并等待完成 —— **整步唯一的同步点**
    fn batch_finish(&self, batch: GpuBatch) {
        let GpuBatch { enc, keep_groups, mut keep_bufs, .. } = batch;
        self.queue.submit([enc.finish()]);
        let _ = self
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        // GPU 已跑完，bind group 可以丢；buffer 归还池供后续回读/复用
        drop(keep_groups);
        for buf in keep_bufs.drain(..) {
            let key = (buf.size(), kind_of(&buf));
            self.pool.lock().unwrap().entry(key).or_default().push(buf);
        }
    }
}

// ==================== 常驻显存录制器：跨算子持有显存句柄 ====================
//
// 逐算子同步版的实测病根（见 `flush_diag_log` 的输出）：单步约 103 次 dispatch，
// 91% 的时间花在 poll+回读上；其中注意力三联算子（S = Q'·Kᵀ → P = softmax(S+mask) → O = P·V）
// 一项就占了回读量的 57%。而 S（33.6MB）与 P（33.6MB）**只是中间结果**：
// 回读给 CPU，下一个算子再原样传回 GPU，一来一回 134MB/层/次，纯属白跑 PCIe。
//
// 这里让中间结果常驻显存：一次提交内把 S→P→O 全部算完，只回读 O（4.2MB）；
// P 连同 Q'/K/V 的显存句柄留到反向直接用。反向同理，五个算子一次提交，只回读 dQ/dK/dV。
//
// 与 `GpuBatch` 的关系：GpuBatch 是"录制式提交"的雏形（结果留显存，但拿不到句柄），
// `GpuRecorder` 是它的完整形态 —— 每个算子返回一个 `GpuHandle`，可以继续喂给下一个算子。
//
// 三条不变式（违反任何一条都会**静默**算错）：
// 1. 每个 dispatch 用独立的参数 uniform（由 `batch_dispatch` 保证）；
// 2. 显存句柄不能在 `submit` 之前析构（encoder 里还引用着它的 buffer）——
//    中间结果用 `GpuRecorder::keep` 交给录制器保活，输出句柄由调用方持有到 submit 之后；
// 3. 同一块 buffer 在一次提交内不能被写两次（`write_buffer` 的写入全部在提交开头生效）。

/// 显存常驻张量句柄：一段 GPU buffer + 元素个数。
/// 由 [`GpuRecorder`] 产出；只有 [`GpuHandle::read`] 会让 CPU 看到它的内容。
pub struct GpuHandle {
    buf: wgpu::Buffer,
    len: usize,
}

impl GpuHandle {
    /// 回读为 CPU 向量 —— **这是一个同步点**，会等 GPU 把当前队列跑完。
    pub fn read(&self) -> Option<Vec<f32>> {
        let ctx = GPU.get().and_then(|g| g.as_ref())?;
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("read") });
        let rb = Readback::record(ctx, &mut enc, &[self])?;
        ctx.queue.submit([enc.finish()]);
        rb.collect()?.into_iter().next()
    }
}

/// 一次回读的暂存：提交前把拷贝命令录进 encoder，poll 之后再取数据。
/// 多个张量合成**一次**提交、一次 poll —— 分开回读要为每个张量各付一次固定开销（实测 ~10ms/次）。
struct Readback {
    staging: wgpu::Buffer,
    /// (元素偏移, 元素个数)
    spans: Vec<(usize, usize)>,
    total: usize,
}

impl Readback {
    /// 把若干句柄拷进一块暂存 buffer（命令录进 `enc`，尚未提交）
    fn record(
        ctx: &GpuContext,
        enc: &mut wgpu::CommandEncoder,
        handles: &[&GpuHandle],
    ) -> Option<Readback> {
        let mut spans = Vec::with_capacity(handles.len());
        let mut total = 0usize;
        for h in handles {
            spans.push((total, h.len));
            total += h.len;
        }
        let staging = ctx.make_buf(total.max(1), KIND_READ);
        let mut off = 0u64;
        for h in handles {
            let bytes = (h.len * 4) as u64;
            if bytes > 0 {
                enc.copy_buffer_to_buffer(&h.buf, 0, &staging, off, bytes);
                off += bytes;
            }
        }
        Some(Readback { staging, spans, total })
    }

    /// 等待并取出数据（提交之后调用）
    fn collect(self) -> Option<Vec<Vec<f32>>> {
        let Readback { staging, spans, total } = self;
        let ctx = GPU.get().and_then(|g| g.as_ref())?;
        let slice = staging.slice(..);
        slice.map_async(wgpu::MapMode::Read, |_| {});
        let _ = ctx
            .device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None });
        let view = slice.get_mapped_range().ok()?;
        let all = unsafe { std::slice::from_raw_parts(view.as_ptr() as *const f32, total) };
        let out = spans.iter().map(|&(o, n)| all[o..o + n].to_vec()).collect();
        drop(view);
        staging.unmap();
        Some(out)
    }
}

/// 常驻显存录制器：一串算子录进同一个 encoder，中间结果全程留在显存不回读。
pub struct GpuRecorder {
    ctx: &'static GpuContext,
    batch: GpuBatch,
    /// 本批的中间句柄：活到 submit 之后自动释放（encoder 里还引用着它们的 buffer）
    keep: Vec<GpuHandle>,
    /// 诊断用：录制起点（见 [`RecorderDiag`] 的 `record_ms`）
    t0: std::time::Instant,
    /// 诊断用：本批 `write_buffer` 推过 PCIe 的字节数
    upload_bytes: usize,
}

/// 开一个常驻显存录制器；GPU 不可用时返回 None（调用方回退逐算子路径）。
/// 判定实验（`LLM_GPU_PROBE`）录制形状时也返回 None，保证录到的仍是逐算子路径的真实形状。
fn recorder() -> Option<GpuRecorder> {
    if PROBE_CAPTURE.load(Ordering::Relaxed) {
        return None;
    }
    let ctx = GPU.get().and_then(|g| g.as_ref())?;
    Some(GpuRecorder {
        ctx,
        batch: ctx.batch_begin(),
        keep: Vec::new(),
        t0: std::time::Instant::now(),
        upload_bytes: 0,
    })
}

impl GpuRecorder {
    /// 上传 CPU 数据到显存，得到常驻句柄
    pub fn upload(&mut self, data: &[f32]) -> GpuHandle {
        // 用 KIND_OUT（含 COPY_SRC）：上传上去的也可能被回读（如反向里的 Q'/K/V）
        let buf = self.ctx.make_buf(data.len().max(1), KIND_OUT);
        if !data.is_empty() {
            self.ctx.queue.write_buffer(&buf, 0, bytemuck_bytes(data));
            self.upload_bytes += data.len() * 4;
        }
        GpuHandle { buf, len: data.len() }
    }

    /// 把中间结果交给录制器保活，避免在 submit 之前析构
    pub fn keep(&mut self, h: GpuHandle) {
        self.keep.push(h);
    }

    /// 录制一次矩阵乘：out[bs,m,n] = a[bs,m,k] · b[bs,k,n]（`a_t`/`b_t` 为物理转置标志）
    #[allow(clippy::too_many_arguments)]
    pub fn matmul(
        &mut self,
        a: &GpuHandle,
        a_t: bool,
        b: &GpuHandle,
        b_t: bool,
        m: usize,
        k: usize,
        n: usize,
        bs: usize,
    ) -> GpuHandle {
        let len = bs * m * n;
        let out = GpuHandle { buf: self.ctx.make_buf(len.max(1), KIND_OUT), len };
        self.ctx
            .batch_matmul(&mut self.batch, &a.buf, &b.buf, &out.buf, m, k, n, bs, a_t, b_t);
        out
    }

    /// 录制一次掩码 softmax（前向）：out[r,j] = softmax(x[r,j] + mask[(r*d) % mask_numel + j])
    pub fn softmax_mask(
        &mut self,
        x: &GpuHandle,
        mask: &GpuHandle,
        rows: usize,
        d: usize,
        mask_numel: usize,
    ) -> GpuHandle {
        let len = rows * d;
        let out = GpuHandle { buf: self.ctx.make_buf(len.max(1), KIND_OUT), len };
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.softmax_fwd_pipe,
            &self.ctx.matmul_layout,
            &[(&x.buf, 0), (&mask.buf, 1), (&out.buf, 2)],
            [rows as u32, d as u32, mask_numel as u32, 0, 0, 0],
            rows as u32, // 一个 workgroup 一行
            1,
            1,
        );
        out
    }

    /// 录制一次掩码 softmax 反向：out[r,j] = p[r,j] · (g[r,j] - Σ_j g[r,j]·p[r,j])
    pub fn softmax_mask_backward(
        &mut self,
        g: &GpuHandle,
        p: &GpuHandle,
        rows: usize,
        d: usize,
    ) -> GpuHandle {
        let len = rows * d;
        let out = GpuHandle { buf: self.ctx.make_buf(len.max(1), KIND_OUT), len };
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.softmax_bwd_pipe,
            &self.ctx.matmul_layout,
            &[(&g.buf, 0), (&p.buf, 1), (&out.buf, 2)],
            [rows as u32, d as u32, 0, 0, 0, 0],
            rows as u32, // 一个 workgroup 一行
            1,
            1,
        );
        out
    }

    /// 录制一次输出头交叉熵（前向 + 反向融合）：把 `logits` 与 `targets` 一起喂进去，
    /// 得到每行的 CE（写进 `row_loss`）和对 logits 的梯度（写进 `out`）。
    /// 调用方自己分配 `out`（[rows, vocab]）与 `row_loss`（[rows]）。
    fn lm_ce(
        &mut self,
        logits: &GpuHandle,
        targets: &GpuHandle,
        out: &GpuHandle,
        row_loss: &GpuHandle,
        rows: usize,
        vocab: usize,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.lm_ce_pipe,
            &self.ctx.lm_ce_layout,
            &[
                (&logits.buf, 0),
                (&targets.buf, 1),
                (&out.buf, 2),
                (&row_loss.buf, 4),
            ],
            [rows as u32, vocab as u32, 0, 0, 0, 0],
            rows as u32, // 一个 workgroup 一行
            1,
            1,
        );
    }

    /// 分配一块内容未初始化的显存（内核会全部覆写它）
    fn empty(&mut self, len: usize, kind: u8) -> GpuHandle {
        GpuHandle { buf: self.ctx.make_buf(len.max(1), kind), len }
    }

    /// 录制归一化前向（LayerNorm / RMSNorm 由 `is_rms` 切换，见 `ln_fwd_main`）；
    /// 同时把每行的 (mean, 1/σ) 写进 `stats`，反向直接复用（重算一遍要再读一次 x，不划算）。
    /// RMSNorm 下 `beta` 槽位仍需一块合法显存，但内核不会用它的值（丢进 `select`）。
    #[allow(clippy::too_many_arguments)]
    fn ln_fwd(
        &mut self,
        x: &GpuHandle,
        gamma: &GpuHandle,
        beta: &GpuHandle,
        out: &GpuHandle,
        stats: &GpuHandle,
        rows: usize,
        d: usize,
        eps: f32,
        is_rms: bool,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.ln_fwd_pipe,
            &self.ctx.elem_b_layout,
            &[
                (&x.buf, 0),
                (&gamma.buf, 1),
                (&beta.buf, 2),
                (&out.buf, 4),
                (&stats.buf, 5),
            ],
            [rows as u32, d as u32, eps.to_bits(), is_rms as u32, 0, 0],
            rows as u32, // 一个 workgroup 一行
            1,
            1,
        );
    }

    /// 录制归一化反向（对输入）：`dy` 是上游梯度，`gamma` 放 binding 5
    fn ln_bwd_x(
        &mut self,
        dy: &GpuHandle,
        x: &GpuHandle,
        stats: &GpuHandle,
        dx: &GpuHandle,
        gamma: &GpuHandle,
        rows: usize,
        d: usize,
        is_rms: bool,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.ln_bwd_x_pipe,
            &self.ctx.elem_b_layout,
            &[
                (&dy.buf, 0),
                (&x.buf, 1),
                (&stats.buf, 2),
                (&dx.buf, 4),
                (&gamma.buf, 5),
            ],
            [rows as u32, d as u32, is_rms as u32, 0, 0, 0],
            rows as u32,
            1,
            1,
        );
    }

    /// 录制 LayerNorm 反向（对 γ/β）：一个 workgroup 一列
    fn ln_bwd_gb(
        &mut self,
        dy: &GpuHandle,
        x: &GpuHandle,
        stats: &GpuHandle,
        dgamma: &GpuHandle,
        dbeta: &GpuHandle,
        rows: usize,
        d: usize,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.ln_bwd_gb_pipe,
            &self.ctx.elem_b_layout,
            &[
                (&dy.buf, 0),
                (&x.buf, 1),
                (&stats.buf, 2),
                (&dgamma.buf, 4),
                (&dbeta.buf, 5),
            ],
            [rows as u32, d as u32, 0, 0, 0, 0],
            d as u32,
            1,
            1,
        );
    }

    /// 录制 GELU（tanh 近似）前向 + 偏置融合：`out = gelu(a + bias)`，`z` 存下 GELU 前的输入
    fn gelu_fwd_bias(
        &mut self,
        a: &GpuHandle,
        bias: &GpuHandle,
        out: &GpuHandle,
        z: &GpuHandle,
        len: usize,
        cols: usize,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.gelu_fwd_pipe,
            &self.ctx.elem_c_layout,
            &[(&a.buf, 0), (&bias.buf, 1), (&out.buf, 4), (&z.buf, 5)],
            [len as u32, cols as u32, 0, 0, 0, 0],
            n_workgroups(len),
            1,
            1,
        );
    }

    /// 录制 GELU 反向：`dx = dy · gelu'(z)`
    fn gelu_bwd(&mut self, dy: &GpuHandle, z: &GpuHandle, dx: &GpuHandle, len: usize) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.gelu_bwd_pipe,
            &self.ctx.elem_d_layout,
            &[(&dy.buf, 0), (&z.buf, 1), (&dx.buf, 4)],
            [len as u32, 0, 0, 0, 0, 0],
            n_workgroups(len),
            1,
            1,
        );
    }

    /// 录制「偏置 + dropout + 残差」：`out = x + mask·(y + b₂)`
    #[allow(clippy::too_many_arguments)]
    fn bias_dropout_residual(
        &mut self,
        y: &GpuHandle,
        b2: &GpuHandle,
        x: &GpuHandle,
        out: &GpuHandle,
        len: usize,
        cols: usize,
        dropout: f32,
        seed: u32,
        training: bool,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.bdr_pipe,
            &self.ctx.elem_a_layout,
            &[(&y.buf, 0), (&b2.buf, 1), (&x.buf, 2), (&out.buf, 4)],
            [
                len as u32,
                cols as u32,
                dropout.to_bits(),
                seed,
                training as u32,
                0,
            ],
            n_workgroups(len),
            1,
            1,
        );
    }

    /// 录制 dropout 反向：`out = g · mask`（掩码由同一 `seed` 重算，与前向逐位一致）
    #[allow(clippy::too_many_arguments)]
    fn dropout_bwd(
        &mut self,
        g: &GpuHandle,
        out: &GpuHandle,
        len: usize,
        dropout: f32,
        seed: u32,
        training: bool,
    ) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.dropout_bwd_pipe,
            &self.ctx.elem_e_layout,
            &[(&g.buf, 0), (&out.buf, 4)],
            [
                len as u32,
                0,
                dropout.to_bits(),
                seed,
                training as u32,
                0,
            ],
            n_workgroups(len),
            1,
            1,
        );
    }

    /// 录制按列求和：`out[j] = Σ_r a[r,j]`（线性层偏置的梯度）
    fn col_sum(&mut self, a: &GpuHandle, out: &GpuHandle, rows: usize, cols: usize) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.col_sum_pipe,
            &self.ctx.elem_e_layout,
            &[(&a.buf, 0), (&out.buf, 4)],
            [rows as u32, cols as u32, 0, 0, 0, 0],
            cols as u32, // 一个 workgroup 一列
            1,
            1,
        );
    }

    /// 录制逐元素相加：`out = a + b`
    fn add(&mut self, a: &GpuHandle, b: &GpuHandle, out: &GpuHandle, len: usize) {
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.add_pipe,
            &self.ctx.add_layout,
            &[(&a.buf, 0), (&b.buf, 1), (&out.buf, 2)],
            [len as u32, 0, 0, 0, 0, 0],
            n_workgroups(len),
            1,
            1,
        );
    }

    /// 录制「按头重排 (+ 偏置 + RoPE)」：`[B,T,H,hd]` → `[B*H,T,hd]`，每个线程处理一对元素
    #[allow(clippy::too_many_arguments)]
    fn heads_split(
        &mut self,
        a: &GpuHandle,
        bias: &GpuHandle,
        out: &GpuHandle,
        b: usize,
        t: usize,
        h: usize,
        hd: usize,
        mode: u32,
        scale: f32,
    ) {
        let pairs = b * h * t * (hd / 2);
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.heads_split_pipe,
            &self.ctx.elem_d_layout,
            &[(&a.buf, 0), (&bias.buf, 1), (&out.buf, 4)],
            [b as u32, t as u32, h as u32, hd as u32, mode, scale.to_bits()],
            n_workgroups(pairs),
            1,
            1,
        );
    }

    /// 录制「按头重排的逆向 (+ RoPE 反向)」：`[B*H,T,hd]` → `[B,T,H,hd]`
    #[allow(clippy::too_many_arguments)]
    fn heads_join(
        &mut self,
        a: &GpuHandle,
        out: &GpuHandle,
        b: usize,
        t: usize,
        h: usize,
        hd: usize,
        mode: u32,
        scale: f32,
    ) {
        let pairs = b * t * h * (hd / 2);
        self.ctx.batch_dispatch(
            &mut self.batch,
            &self.ctx.heads_join_pipe,
            &self.ctx.elem_e_layout,
            &[(&a.buf, 0), (&out.buf, 4)],
            [b as u32, t as u32, h as u32, hd as u32, mode, scale.to_bits()],
            n_workgroups(pairs),
            1,
            1,
        );
    }

    /// 上传 u32 数据（如 targets 下标）
    fn upload_u32(&mut self, data: &[u32]) -> GpuHandle {
        let buf = self.ctx.make_buf(data.len().max(1), KIND_OUT);
        if !data.is_empty() {
            self.ctx.queue.write_buffer(&buf, 0, bytemuck_bytes(data));
            self.upload_bytes += data.len() * 4;
        }
        GpuHandle { buf, len: data.len() }
    }

    /// 提交整批，把 `outs` 回读给 CPU（回读拷贝也录进同一个 encoder → 一次 submit、一次 poll）。
    /// 其余张量留在显存等下一批。
    pub fn submit_and_read(self, outs: &[&GpuHandle]) -> Option<Vec<Vec<f32>>> {
        let GpuRecorder { ctx, batch, keep, t0, upload_bytes } = self;
        let GpuBatch { enc, keep_groups, mut keep_bufs, dispatches, kinds } = batch;
        let mut enc = enc;
        let rb = Readback::record(ctx, &mut enc, outs)?;
        // 诊断：本批回读字节数（与逐算子路径同一口径，便于对照）
        let download_bytes: usize = outs.iter().map(|h| h.len * 4).sum();
        let record_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        ctx.queue.submit([enc.finish()]);
        let submit_ms = t.elapsed().as_secs_f64() * 1000.0;
        let t = std::time::Instant::now();
        let result = rb.collect();
        let sync_ms = t.elapsed().as_secs_f64() * 1000.0;
        {
            let mut v = RECORDER_DIAG_LOG.lock().unwrap();
            if v.len() < DIAG_MAX {
                v.push(RecorderDiag {
                    upload_bytes,
                    download_bytes,
                    dispatches,
                    kinds,
                    record_ms,
                    submit_ms,
                    sync_ms,
                });
            }
        }
        // GPU 已跑完：bind group 可以丢，池化 buffer 归还，中间句柄在此释放
        drop(keep_groups);
        for buf in keep_bufs.drain(..) {
            let key = (buf.size(), kind_of(&buf));
            ctx.pool.lock().unwrap().entry(key).or_default().push(buf);
        }
        drop(keep);
        result
    }
}

// ==================== 注意力链常驻显存（S→P→O 一次提交） ====================

/// 把 S = Q'·Kᵀ → P = softmax(S + mask) → O = P·V 录进 `rec`（不提交），返回 (P, O)。
///
/// 抽成独立函数是为了让「注意力子层整段常驻」（ln1 + QKV + RoPE + attn + c_proj + 残差）
/// 能把这些算子录进**同一个** encoder，而不是各自提交一次。
#[allow(clippy::too_many_arguments)]
fn record_attn_fwd(
    rec: &mut GpuRecorder,
    q: &GpuHandle,
    k: &GpuHandle,
    v: &GpuHandle,
    mask: &GpuHandle,
    bh: usize,
    t: usize,
    t_total: usize,
    head_dim: usize,
) -> (GpuHandle, GpuHandle) {
    // S = Q'·Kᵀ（physical K 是 [B,N,K] → b_t）
    let s = rec.matmul(q, false, k, true, t, head_dim, t_total, bh);
    // P = softmax(S + mask)：中间结果只留在显存
    let p = rec.softmax_mask(&s, &mask, bh * t, t_total, mask.len);
    rec.keep(s);
    // O = P·V
    let o = rec.matmul(&p, false, v, false, t, t_total, head_dim, bh);
    (p, o)
}

/// 把注意力反向（dV → dP → dS → dQ/dK）录进 `rec`（不提交），返回 (dQ, dK, dV)。
/// 返回的 dQ 未乘回 scale（前向把缩放挪到了 Q 上，调用方负责）。
fn record_attn_bwd(
    rec: &mut GpuRecorder,
    p: &GpuHandle,
    q: &GpuHandle,
    k: &GpuHandle,
    v: &GpuHandle,
    g: &GpuHandle,
    bh: usize,
    t: usize,
    t_total: usize,
    head_dim: usize,
) -> (GpuHandle, GpuHandle, GpuHandle) {
    let tt = t_total;
    // dV = Pᵀ·dO（physical P 是 [B,M,K] → a_t）
    let dv = rec.matmul(p, true, g, false, tt, t, head_dim, bh);
    // dP = dO·Vᵀ（physical V 是 [B,N,K] → b_t）
    let dp = rec.matmul(g, false, v, true, t, head_dim, tt, bh);
    // dS = P ⊙ (dP - Σ_j dP·P)：与逐算子版手写的三重循环同一个公式
    let ds = rec.softmax_mask_backward(&dp, p, bh * t, tt);
    rec.keep(dp);
    // dQ = dS·K（physical dS 是 [B,M,K] → a_t）；dK = dSᵀ·Q'
    let dq = rec.matmul(&ds, false, k, false, t, tt, head_dim, bh);
    let dk = rec.matmul(&ds, true, q, false, tt, t, head_dim, bh);
    rec.keep(ds);
    (dq, dk, dv)
}

/// 一次注意力前向的「常驻显存」结果。
///
/// 前向把 S = Q'·Kᵀ → P = softmax(S + mask) → O = P·V 三个算子录进**一次提交**：
/// 中间结果 S（33.6MB）、P（33.6MB）不回读，只把 O（4.2MB）交给 CPU。
/// P 与 Q'/K/V 的显存句柄留在本结构里供反向直接复用 ——
/// 逐算子版要把 P 回读 33.6MB、下一步再原样传回 33.6MB，一来一回 67MB/层/次纯属白跑。
pub struct AttnResident {
    /// 前向输出 O：[bh, t, head_dim]，已回读到 CPU
    pub out: Vec<f32>,
    /// 注意力概率 P：[bh*t, t_total]（softmax 输出）
    p: GpuHandle,
    /// Q'（已乘 1/√head_dim）：[bh, t, head_dim]
    q: GpuHandle,
    /// K：[bh, t_total, head_dim]
    k: GpuHandle,
    /// V：[bh, t_total, head_dim]
    v: GpuHandle,
    bh: usize,
    t: usize,
    t_total: usize,
    head_dim: usize,
}

impl AttnResident {
    /// 常驻显存版的反向：五个算子录进一次提交，只回读 dQ/dK/dV（各 4.2MB）。
    /// 返回的 dQ 未乘回 scale（调用方负责，与前向把缩放挪到 Q 上对应）。
    pub fn backward(&self, dout: &[f32]) -> Option<(Vec<f32>, Vec<f32>, Vec<f32>)> {
        if dout.len() != self.out.len() {
            return None;
        }
        let (bh, t, tt, hd) = (self.bh, self.t, self.t_total, self.head_dim);
        let mut rec = recorder()?;
        let g = rec.upload(dout);
        let (dq, dk, dv) = record_attn_bwd(
            &mut rec, &self.p, &self.q, &self.k, &self.v, &g, bh, t, tt, hd,
        );
        rec.keep(g);
        let outs = rec.submit_and_read(&[&dq, &dk, &dv])?;
        let mut it = outs.into_iter();
        Some((it.next()?, it.next()?, it.next()?))
    }
}

/// 常驻显存注意力路径的适用判定：尺寸与网格约束都满足才走这条路，
/// 否则（推理单 token、KV cache 小块等）回退逐算子路径。
///
/// 单独抽成公开函数是为了让测试能断言「这个形状确实会命中常驻路径」——
/// 否则测试会静默退化成旧的逐算子路径，看着过了其实没测到新代码。
pub fn attn_resident_ok(
    bh: usize,
    t: usize,
    t_total: usize,
    head_dim: usize,
    mask_numel: usize,
) -> bool {
    if bh == 0 || t == 0 || t_total == 0 || head_dim == 0 || mask_numel < t_total {
        return false;
    }
    let Some(rows) = bh.checked_mul(t) else {
        return false;
    };
    // 太小不值得上 GPU（与逐算子路径的阈值一致：softmax 元素数 + matmul FLOPs）
    if rows * t_total < SOFTMAX_MIN_ELEMS {
        return false;
    }
    let flops = 2u128 * t as u128 * head_dim as u128 * t_total as u128 * bh as u128;
    if flops < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) as u128 {
        return false;
    }
    // workgroup 数受单维 65535 上限约束（softmax 每行一个 workgroup，matmul 每 tile 一个）。
    if rows > 65535 || bh > 65535 {
        return false;
    }
    // matmul 的 tile 数按**小** tile 判：小 tile 的 workgroup 数只会更多，两边都兜得住。
    let ntiles = |x: usize| x.div_ceil(MM_TILE_S) > 65535;
    !(ntiles(t) || ntiles(t_total) || ntiles(head_dim))
}

/// 常驻显存版的注意力前向。尺寸不合适或 GPU 不可用时返回 None，调用方回退逐算子路径。
///
/// 前向的三个算子录进一次提交，中间 S、P 不回读；P 与 Q'/K/V 的句柄随结果返回。
#[allow(clippy::too_many_arguments)]
pub fn attn_forward(
    q_scaled: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    bh: usize,
    t: usize,
    t_total: usize,
    head_dim: usize,
) -> Option<AttnResident> {
    if !attn_resident_ok(bh, t, t_total, head_dim, mask.len()) {
        return None;
    }

    let mut rec = recorder()?;
    let hq = rec.upload(q_scaled);
    let hk = rec.upload(k);
    let hv = rec.upload(v);
    let hm = rec.upload(mask);
    let (p, o) = record_attn_fwd(&mut rec, &hq, &hk, &hv, &hm, bh, t, t_total, head_dim);
    let out = rec.submit_and_read(&[&o])?.into_iter().next()?;
    Some(AttnResident {
        out,
        p,
        q: hq,
        k: hk,
        v: hv,
        bh,
        t,
        t_total,
        head_dim,
    })
}

// ==================== 注意力子层整段常驻显存 ====================
//
// 覆盖 `TransformerBlock` 的注意力子层：
//   x → LayerNorm/RMSNorm → QKV 投影 → 按头重排 + RoPE → S/P/O → 合并头 → c_proj → dropout → 残差
//
// 这一段原本是「每个算子各自提交 + 各自回读」的重灾区：4 层每步要做约 48 次提交，
// 每次固定开销数毫秒，中间量（Q/K/V、旋转后的 Q/K、注意力输出）全都要回读再传回。
// 这里整段录进**两次提交**（前向一次、反向一次），中间量全程留显存，
// CPU 只付「上传 x / 回读子层输出」与 11 项边界梯度（合计约 2.3MB）。
//
// 限制（不满足时调用方回退逐算子路径，数值行为不变）：
// - 只支持训练（`base = 0`、无 KV cache）：RoPE 的位置直接取序列内下标

/// 逐元素内核里 `heads_split` 的模式：只加偏置
const HEADS_MODE_BIAS: u32 = 0;
/// 加偏置 + RoPE（Q/K 前向）
const HEADS_MODE_ROPE: u32 = 1;
/// 纯搬运（不读偏置、不旋转；反向第一步 dmerged → dO 用）
const HEADS_MODE_PLAIN: u32 = 2;

/// 一次注意力子层前向的常驻显存结果（含反向所需的全部显存句柄）。
pub struct AttnLayerResident {
    /// 子层输出 `x + dropout(c_proj(merge(attn(rope(qkv(ln(x)))))))`，已回读到 CPU
    pub out: Vec<f32>,
    /// 子层输入 [rows, d]（残差直通 + LayerNorm 反向用）
    x: GpuHandle,
    /// LayerNorm 输出 [rows, d]（dWq/dWk/dWv 用）
    xn: GpuHandle,
    /// 归一化每行的 (mean, 1/σ)，[rows*2]（RMSNorm 时 mean 槽位是 0）
    stats: GpuHandle,
    /// 归一化的 γ
    gamma: GpuHandle,
    /// Q'（已旋转、已乘 1/√head_dim）：[b*n_head, t, head_dim]
    q: GpuHandle,
    /// K（已旋转）：[b*n_head, t, head_dim]
    k: GpuHandle,
    /// V：[b*n_head, t, head_dim]
    v: GpuHandle,
    /// 注意力概率 P：[b*n_head*t, t]
    p: GpuHandle,
    /// 合并头之后的 c_proj 输入 [rows, d]
    merged: GpuHandle,
    /// 输出投影权重 [d, d]
    wproj: GpuHandle,
    b: usize,
    t: usize,
    d: usize,
    n_head: usize,
    head_dim: usize,
    /// dropout 掩码的种子：前向与反向必须用同一个，掩码由它重算
    seed: u32,
    dropout: f32,
    training: bool,
    /// 归一化是否为 RMSNorm（反向走同一套内核、只切模式位）
    is_rms: bool,
}

/// 注意力子层反向的边界梯度（已回读到 CPU，由调用方注回计算图）
pub struct AttnLayerGrads {
    /// 对子层输入 x 的梯度（含残差直通的那一份）
    pub dx: Vec<f32>,
    pub dwq: Vec<f32>,
    pub dbq: Vec<f32>,
    pub dwk: Vec<f32>,
    pub dbk: Vec<f32>,
    pub dwv: Vec<f32>,
    pub dbv: Vec<f32>,
    pub dwproj: Vec<f32>,
    pub dbproj: Vec<f32>,
    pub dgamma: Vec<f32>,
    /// RMSNorm 下没有 β，这一项是 `Σ_r dy` 的残值，**无意义，调用方忽略**
    pub dbeta: Vec<f32>,
}

impl AttnLayerResident {
    /// 反向：把 11 项边界梯度录进一次提交后回读
    pub fn backward(&self, dout: &[f32]) -> Option<AttnLayerGrads> {
        if dout.len() != self.rows() * self.d {
            return None;
        }
        let (b, t, d, h) = (self.b, self.t, self.d, self.n_head);
        let (hd, rows) = (self.head_dim, self.rows());
        let bn = b * h;
        let scale = 1.0 / (hd as f32).sqrt();

        let mut rec = recorder()?;
        let g = rec.upload(dout);
        // 1) dropout 反向：dpre = dout ⊙ mask（掩码由同一 seed 重算）
        let dpre = rec.empty(rows * d, KIND_OUT);
        rec.dropout_bwd(&g, &dpre, rows * d, self.dropout, self.seed, self.training);
        // 2) c_proj 反向：dmerged = dpre·Wprojᵀ，dWproj = mergedᵀ·dpre，dbproj = Σ_r dpre
        let dmerged = rec.matmul(&dpre, false, &self.wproj, true, rows, d, d, 1);
        let dwproj = rec.matmul(&self.merged, true, &dpre, false, d, rows, d, 1);
        let dbproj = rec.empty(d, KIND_OUT);
        rec.col_sum(&dpre, &dbproj, rows, d);
        // 3) 拆回按头布局：dO = dmerged（纯搬运，不读偏置）
        //    布局里 binding 1 必须给一块合法 buffer，mode 2 不会读它 → 拿 x 占位
        let do_ = rec.empty(bn * t * hd, KIND_OUT);
        rec.heads_split(
            &dmerged, &self.x, &do_, b, t, h, hd, HEADS_MODE_PLAIN, 1.0,
        );
        // 4) 注意力反向：dV → dP → dS → dQ/dK
        let (dq, dk, dv) =
            record_attn_bwd(&mut rec, &self.p, &self.q, &self.k, &self.v, &do_, bn, t, t, hd);
        // 5) 合并回 [rows, d]：RoPE 用转置旋转回传，Q 的 1/√head_dim 也在这里乘回
        let dq_pre = rec.empty(rows * d, KIND_OUT);
        rec.heads_join(&dq, &dq_pre, b, t, h, hd, 1, scale);
        let dk_pre = rec.empty(rows * d, KIND_OUT);
        rec.heads_join(&dk, &dk_pre, b, t, h, hd, 1, 1.0);
        let dv_pre = rec.empty(rows * d, KIND_OUT);
        rec.heads_join(&dv, &dv_pre, b, t, h, hd, 0, 1.0);
        // 6) QKV 投影反向：偏置梯度 = 列求和；权重梯度 = xnᵀ·d_pre
        let dbq = rec.empty(d, KIND_OUT);
        rec.col_sum(&dq_pre, &dbq, rows, d);
        let dbk = rec.empty(d, KIND_OUT);
        rec.col_sum(&dk_pre, &dbk, rows, d);
        let dbv = rec.empty(d, KIND_OUT);
        rec.col_sum(&dv_pre, &dbv, rows, d);
        let dwq = rec.matmul(&self.xn, true, &dq_pre, false, d, rows, d, 1);
        let dwk = rec.matmul(&self.xn, true, &dk_pre, false, d, rows, d, 1);
        let dwv = rec.matmul(&self.xn, true, &dv_pre, false, d, rows, d, 1);
        // 7) 三路梯度汇合成对 LayerNorm 输出的梯度
        let sum2 = rec.empty(rows * d, KIND_OUT);
        rec.add(&dq_pre, &dk_pre, &sum2, rows * d);
        let dln = rec.empty(rows * d, KIND_OUT);
        rec.add(&sum2, &dv_pre, &dln, rows * d);
        // 8) 归一化反向（LayerNorm / RMSNorm 同一套内核）
        let dxln = rec.empty(rows * d, KIND_OUT);
        rec.ln_bwd_x(&dln, &self.x, &self.stats, &dxln, &self.gamma, rows, d, self.is_rms);
        let dgamma = rec.empty(d, KIND_OUT);
        let dbeta = rec.empty(d, KIND_OUT);
        rec.ln_bwd_gb(&dln, &self.x, &self.stats, &dgamma, &dbeta, rows, d);

        let outs = rec.submit_and_read(&[
            &dxln, &dwq, &dbq, &dwk, &dbk, &dwv, &dbv, &dwproj, &dbproj, &dgamma, &dbeta,
        ])?;
        let mut it = outs.into_iter();
        let dxln = it.next()?;
        // 残差直通：∂out/∂x 的那一份就是 dout 本身
        let dx: Vec<f32> = dxln.iter().zip(dout).map(|(a, b)| a + b).collect();
        Some(AttnLayerGrads {
            dx,
            dwq: it.next()?,
            dbq: it.next()?,
            dwk: it.next()?,
            dbk: it.next()?,
            dwv: it.next()?,
            dbv: it.next()?,
            dwproj: it.next()?,
            dbproj: it.next()?,
            dgamma: it.next()?,
            dbeta: it.next()?,
        })
    }

    fn rows(&self) -> usize {
        self.b * self.t
    }
}

/// 常驻显存版的注意力子层前向（见本节的模块说明）。
///
/// 入参都是 CPU 上的张量数据：`x` `[b*t, d]`、`gamma`/`beta` `[d]`、
/// `wq`/`wk`/`wv`/`wproj` `[d, d]`、`bq`/`bk`/`bv`/`bproj` `[d]`、`mask` `[t, t]`。
///
/// `is_rms` 切换归一化模式（RMSNorm 时 `beta` 的内容被内核丢弃，只需长度合法）。
/// GQA 不需要在这里区分：K/V 投影的头复制由调用方**把权重按头展开**后喂进来
/// （见 `model.rs` 的 `expand_kv`），展开后本函数的入口形态与标准 MHA 完全一致，
/// 于是「按头重排」「注意力」「合并头」三段的布局都不用改。
pub struct AttnLayerArgs<'a> {
    pub x: &'a [f32],
    pub gamma: &'a [f32],
    pub beta: &'a [f32],
    /// Q 投影权重 [d, d]
    pub wq: &'a [f32],
    pub bq: &'a [f32],
    pub wk: &'a [f32],
    pub bk: &'a [f32],
    pub wv: &'a [f32],
    pub bv: &'a [f32],
    /// 输出投影权重 [d, d]
    pub wproj: &'a [f32],
    pub bproj: &'a [f32],
    pub mask: &'a [f32],
    pub b: usize,
    pub t: usize,
    pub d: usize,
    pub n_head: usize,
    pub eps: f32,
    /// 归一化是否为 RMSNorm（true 时 `beta` 槽位被丢弃）
    pub is_rms: bool,
    pub dropout: f32,
    pub training: bool,
}

pub fn attn_layer_forward(a: &AttnLayerArgs) -> Option<AttnLayerResident> {
    let AttnLayerArgs { b, t, d, n_head, .. } = *a;
    if b == 0 || t == 0 || d == 0 || n_head == 0 || d % n_head != 0 {
        return None;
    }
    let head_dim = d / n_head;
    if head_dim == 0 || head_dim % 2 != 0 {
        return None;
    }
    let rows = b * t;
    let bn = b * n_head;
    if a.x.len() != rows * d
        || a.gamma.len() != d
        || a.beta.len() != d
        || a.wq.len() != d * d
        || a.bq.len() != d
        || a.wk.len() != d * d
        || a.bk.len() != d
        || a.wv.len() != d * d
        || a.bv.len() != d
        || a.wproj.len() != d * d
        || a.bproj.len() != d
        || a.mask.len() < t * t
    {
        return None;
    }
    if !attn_resident_ok(bn, t, t, head_dim, a.mask.len()) {
        return None;
    }
    // 单维 workgroup 上限：LN 按行、按列求和按列，重排/逐元素按元素对数
    if rows > 65535 || d > 65535 || rows * d > 65535 * ELEM_WG {
        return None;
    }
    // 太小不值得上 GPU（QKV 三个投影 + 输出投影合计 8·rows·d² FLOPs）
    let flops = 8u128 * rows as u128 * d as u128 * d as u128;
    if flops < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) as u128 {
        return None;
    }

    let mut rec = recorder()?;
    let hx = rec.upload(a.x);
    let hg = rec.upload(a.gamma);
    let hb = rec.upload(a.beta);
    let hwq = rec.upload(a.wq);
    let hbq = rec.upload(a.bq);
    let hwk = rec.upload(a.wk);
    let hbk = rec.upload(a.bk);
    let hwv = rec.upload(a.wv);
    let hbv = rec.upload(a.bv);
    let hwp = rec.upload(a.wproj);
    let hbp = rec.upload(a.bproj);
    let hm = rec.upload(a.mask);
    let scale = 1.0 / (head_dim as f32).sqrt();
    // LayerNorm（顺带把每行统计量存进显存给反向复用）
    let xn = rec.empty(rows * d, KIND_OUT);
    let stats = rec.empty(rows * 2, KIND_OUT);
    rec.ln_fwd(&hx, &hg, &hb, &xn, &stats, rows, d, a.eps, a.is_rms);
    // QKV 三个投影
    let q_pre = rec.matmul(&xn, false, &hwq, false, rows, d, d, 1);
    let k_pre = rec.matmul(&xn, false, &hwk, false, rows, d, d, 1);
    let v_pre = rec.matmul(&xn, false, &hwv, false, rows, d, d, 1);
    // 按头重排 + 偏置（Q/K 顺带 RoPE；缩放放在 Q 上，softmax 内核就不必带 scale）
    let q = rec.empty(bn * t * head_dim, KIND_OUT);
    rec.heads_split(&q_pre, &hbq, &q, b, t, n_head, head_dim, HEADS_MODE_ROPE, scale);
    let k = rec.empty(bn * t * head_dim, KIND_OUT);
    rec.heads_split(&k_pre, &hbk, &k, b, t, n_head, head_dim, HEADS_MODE_ROPE, 1.0);
    let v = rec.empty(bn * t * head_dim, KIND_OUT);
    rec.heads_split(&v_pre, &hbv, &v, b, t, n_head, head_dim, HEADS_MODE_BIAS, 1.0);
    // S = Q'·Kᵀ → P = softmax(S + mask) → O = P·V
    let (p, o) = record_attn_fwd(&mut rec, &q, &k, &v, &hm, bn, t, t, head_dim);
    // 合并头 → 输出投影 → dropout → 残差
    let merged = rec.empty(rows * d, KIND_OUT);
    rec.heads_join(&o, &merged, b, t, n_head, head_dim, 0, 1.0);
    let y = rec.matmul(&merged, false, &hwp, false, rows, d, d, 1);
    let out = rec.empty(rows * d, KIND_OUT);
    let seed = DROPOUT_SEED.fetch_add(1, Ordering::Relaxed) as u32 ^ 0x9e37_79b9;
    rec.bias_dropout_residual(&y, &hbp, &hx, &out, rows * d, d, a.dropout, seed, a.training);
    let out_host = rec.submit_and_read(&[&out])?.into_iter().next()?;
    Some(AttnLayerResident {
        out: out_host,
        x: hx,
        xn,
        stats,
        gamma: hg,
        q,
        k,
        v,
        p,
        merged,
        wproj: hwp,
        b,
        t,
        d,
        n_head,
        head_dim,
        seed,
        dropout: a.dropout,
        training: a.training,
        is_rms: a.is_rms,
    })
}

// ==================== 输出头交叉熵常驻显存 ====================

/// 一次「输出头 + 交叉熵」的常驻显存结果。
///
/// 前向把 `logits = hidden @ Wᵀ` 与 `softmax + 交叉熵` 录进一次提交：
/// logits 与 dlogits（本配置下各 33.6M 元素）**都不回读**，只取回每行的 CE（16 KB）；
/// 反向再一次性算出 d_hidden 与 d_W（合计 6 MB）。
///
/// 逐算子版这里要付的代价：回读 logits 134 MB → CPU 上 log_softmax（33.6M 元素）
/// → 回读 dlogits 134 MB → 再传回 GPU，一步来回 268 MB（实测 445 ms）。
pub struct LmHeadResident {
    /// 交叉熵（已对 rows 求平均）
    pub loss: f32,
    /// dlogits = (softmax(logits) - onehot) / rows，常驻显存 [rows, vocab]
    dlogits: GpuHandle,
    /// 前向的输入 hidden [rows, d]（反向算 d_W 用）
    x: GpuHandle,
    /// 输出头权重 [vocab, d]
    w: GpuHandle,
    rows: usize,
    d: usize,
    vocab: usize,
}

impl LmHeadResident {
    /// 反向：返回 (d_hidden [rows, d], d_weight [vocab, d])。
    ///
    /// 不乘上游梯度——交叉熵的输出是标量 loss，其上游梯度恒为 1，
    /// `dlogits` 已经就是最终梯度（含对 rows 求平均的 1/rows）。
    pub fn backward(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        let (rows, d, vocab) = (self.rows, self.d, self.vocab);
        let mut rec = recorder()?;
        // d_hidden = dlogits @ W：两侧都是常规布局 [rows, vocab] × [vocab, d]
        let dx = rec.matmul(&self.dlogits, false, &self.w, false, rows, vocab, d, 1);
        // d_W = dlogitsᵀ @ hidden：dlogits 物理是 [rows, vocab]，当 [K, M] 转置读
        let dw = rec.matmul(&self.dlogits, true, &self.x, false, vocab, rows, d, 1);
        let outs = rec.submit_and_read(&[&dx, &dw])?;
        let mut it = outs.into_iter();
        Some((it.next()?, it.next()?))
    }
}

/// 常驻显存版的「输出头 + 交叉熵」。
///
/// - `x`：`[rows, d]` 的最终 hidden（已过 ln_f）
/// - `w`：`[vocab, d]` 的输出头权重
/// - `targets`：`[rows]` 每行的正确类别
///
/// 尺寸不合适或 GPU 不可用时返回 None，调用方回退逐算子路径。
pub fn lm_head_ce(
    x: &[f32],
    w: &[f32],
    targets: &[usize],
    rows: usize,
    d: usize,
    vocab: usize,
) -> Option<LmHeadResident> {
    if rows == 0 || d == 0 || vocab == 0 || targets.len() != rows {
        return None;
    }
    if x.len() != rows * d || w.len() != vocab * d {
        return None;
    }
    // softmax 每行一个 workgroup，受单维 65535 上限约束
    if rows > 65535 {
        return None;
    }
    // 太小不值得上 GPU（与逐算子路径同一口径）
    let flops = 2u128 * rows as u128 * d as u128 * vocab as u128;
    if flops < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) as u128 {
        return None;
    }

    let mut rec = recorder()?;
    let hx = rec.upload(x);
    let hw = rec.upload(w);
    let tids: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
    let ht = rec.upload_u32(&tids);
    // logits = x @ Wᵀ（物理 W 是 [vocab, d] = [N, K] → b_t）
    let logits = rec.matmul(&hx, false, &hw, true, rows, d, vocab, 1);
    let dlogits = rec.empty(rows * vocab, KIND_OUT);
    let row_loss = rec.empty(rows, KIND_OUT);
    rec.lm_ce(&logits, &ht, &dlogits, &row_loss, rows, vocab);
    rec.keep(logits);
    rec.keep(ht);
    let loss_rows = rec.submit_and_read(&[&row_loss])?.into_iter().next()?;
    let loss = loss_rows.iter().sum::<f32>() / rows as f32;
    Some(LmHeadResident { loss, dlogits, x: hx, w: hw, rows, d, vocab })
}

// ==================== MLP 子层常驻显存（前向 / 反向各一次提交） ====================
//
// 子层结构（经典风格）：x → LayerNorm/RMSNorm → Linear₁ → GELU → Linear₂ → dropout → 残差 +
//
// 逐算子版这一段要付的代价：
// - 前向：LN 在 CPU（逐元素 + rayon）、两次线性投影各自一次「提交 + 轮询」往返、
//   GELU 在 CPU（[rows, hid] 个 tanh）、dropout 在 CPU、残差加法在 CPU；
// - 反向：四次矩阵乘反向又是一轮往返，再加 LN / GELU / dropout 的 CPU 反向。
// 每层每步约 6 次「提交 + 轮询」，而每次往返的固定开销在这类入门独显上高达数毫秒。
//
// 这里把整段录进**两次提交**（前向一次、反向一次）：中间量（LN 输出、GELU 前后、
// 线性层中间结果）全部留在显存，CPU 只付「上传 x / 回读输出」与回读边界梯度。
// dropout 掩码连存都不用存 —— 它是 (种子, 下标) 的确定性函数，反向重算即可。

/// dropout 种子计数器：每次常驻前向取一个新种子，保证同一步内不同子层、
/// 不同步之间掩码都不重复（掩码本身不落显存，反向靠这个种子重算）。
static DROPOUT_SEED: AtomicUsize = AtomicUsize::new(0x1234_5678);

/// 一次 MLP 子层常驻显存前向的结果（含反向所需的全部显存句柄）。
pub struct MlpResident {
    /// 子层输出 `x + dropout(linear2(gelu(linear1(ln(x)))))`，已回读到 CPU
    pub out: Vec<f32>,
    /// 子层输入 [rows, d]
    x: GpuHandle,
    /// 归一化输出 [rows, d]
    xn: GpuHandle,
    /// Linear₁ 输出 + b₁（GELU 的输入）[rows, hid]
    z: GpuHandle,
    /// GELU 输出 [rows, hid]
    act: GpuHandle,
    /// 归一化每行的 (mean, 1/σ)，[rows*2]（RMSNorm 时 mean 槽位是 0）
    stats: GpuHandle,
    w1: GpuHandle,
    w2: GpuHandle,
    /// γ 在反向还要用（dγ 的输出以及 dx 的权重）；β 不复用，故不在此保存
    gamma: GpuHandle,
    rows: usize,
    d: usize,
    hid: usize,
    /// dropout 掩码的种子：前向与反向必须用同一个，掩码由它重算
    seed: u32,
    dropout: f32,
    training: bool,
    /// 归一化是否为 RMSNorm（反向走同一套内核、只切模式位）
    is_rms: bool,
}

/// MLP 子层反向的边界梯度（已回读到 CPU，由调用方乘上缩放后注回计算图）
pub struct MlpGrads {
    /// 对子层输入 x 的梯度（含残差直通的那一份）
    pub dx: Vec<f32>,
    pub dw1: Vec<f32>,
    pub db1: Vec<f32>,
    pub dw2: Vec<f32>,
    pub db2: Vec<f32>,
    pub dgamma: Vec<f32>,
    /// RMSNorm 下没有 β，这一项是 `Σ_r dy` 的残值，**无意义，调用方忽略**
    pub dbeta: Vec<f32>,
}

impl MlpResident {
    /// 反向：10 个算子录进一次提交，只回读边界梯度
    /// （dx 2MB + dW₁/dW₂ 各 256KB + dγ/dβ/db₁/db₂ 共 4KB）。
    pub fn backward(&self, dout: &[f32]) -> Option<MlpGrads> {
        if dout.len() != self.rows * self.d {
            return None;
        }
        let (rows, d, hid) = (self.rows, self.d, self.hid);
        let mut rec = recorder()?;
        let g = rec.upload(dout);
        // 1) dropout 反向：dypre = dout ⊙ mask（掩码由同一 seed 重算，与前向逐位一致）
        let dypre = rec.empty(rows * d, KIND_OUT);
        rec.dropout_bwd(&g, &dypre, rows * d, self.dropout, self.seed, self.training);
        // 2) db₂ = Σ_r dypre[r,j]
        let db2 = rec.empty(d, KIND_OUT);
        rec.col_sum(&dypre, &db2, rows, d);
        // 3) dact = dypre·W₂ᵀ，dW₂ = actᵀ·dypre
        let dact = rec.matmul(&dypre, false, &self.w2, true, rows, d, hid, 1);
        let dw2 = rec.matmul(&self.act, true, &dypre, false, hid, rows, d, 1);
        // 4) GELU 反向：dz = dact ⊙ gelu'(z)
        let dz = rec.empty(rows * hid, KIND_OUT);
        rec.gelu_bwd(&dact, &self.z, &dz, rows * hid);
        // 5) db₁ = Σ_r dz[r,j]
        let db1 = rec.empty(hid, KIND_OUT);
        rec.col_sum(&dz, &db1, rows, hid);
        // 6) dxn = dz·W₁ᵀ，dW₁ = xnᵀ·dz
        let dxn = rec.matmul(&dz, false, &self.w1, true, rows, hid, d, 1);
        let dw1 = rec.matmul(&self.xn, true, &dz, false, d, rows, hid, 1);
        // 7) 归一化反向（LayerNorm / RMSNorm 同一套内核）：对输入的 dx 与对 γ/β 的梯度
        let dxln = rec.empty(rows * d, KIND_OUT);
        rec.ln_bwd_x(&dxn, &self.x, &self.stats, &dxln, &self.gamma, rows, d, self.is_rms);
        let dgamma = rec.empty(d, KIND_OUT);
        let dbeta = rec.empty(d, KIND_OUT);
        rec.ln_bwd_gb(&dxn, &self.x, &self.stats, &dgamma, &dbeta, rows, d);
        let outs = rec.submit_and_read(&[&dxln, &dw1, &db1, &dw2, &db2, &dgamma, &dbeta])?;
        let mut it = outs.into_iter();
        let dxln = it.next()?;
        let dw1 = it.next()?;
        let db1 = it.next()?;
        let dw2 = it.next()?;
        let db2 = it.next()?;
        let dgamma = it.next()?;
        let dbeta = it.next()?;
        // 残差直通：∂out/∂x 的那一份就是 dout 本身，与 LayerNorm 回传的那份相加
        let dx: Vec<f32> = dxln.iter().zip(dout).map(|(a, b)| a + b).collect();
        Some(MlpGrads { dx, dw1, db1, dw2, db2, dgamma, dbeta })
    }
}

/// 常驻显存版的 MLP 子层前向。
///
/// 入参都是 CPU 上的张量数据：`x` `[rows, d]`、`gamma`/`beta` `[d]`、
/// `w1` `[d, hid]`、`b1` `[hid]`、`w2` `[hid, d]`、`b2` `[d]`。
///
/// `is_rms` 切换归一化模式（RMSNorm 时 `beta` 的内容被内核丢弃，只需长度合法）。
/// 只覆盖经典风格的 GELU MLP（SwiGLU 由调用方让路）；
/// 形状/规模不合适或 GPU 不可用时返回 None，调用方回退逐算子路径，数值行为不变。
#[allow(clippy::too_many_arguments)]
pub fn mlp_forward(
    x: &[f32],
    gamma: &[f32],
    beta: &[f32],
    w1: &[f32],
    b1: &[f32],
    w2: &[f32],
    b2: &[f32],
    rows: usize,
    d: usize,
    hid: usize,
    eps: f32,
    is_rms: bool,
    dropout: f32,
    training: bool,
) -> Option<MlpResident> {
    if rows == 0 || d == 0 || hid == 0 {
        return None;
    }
    if x.len() != rows * d
        || gamma.len() != d
        || beta.len() != d
        || w1.len() != d * hid
        || b1.len() != hid
        || w2.len() != hid * d
        || b2.len() != d
    {
        return None;
    }
    // 单维 workgroup 上限：LN 按行、列求和按列、逐元素按元素数开工作组
    if rows > 65535 || d > 65535 || hid > 65535 {
        return None;
    }
    if rows * d > 65535 * ELEM_WG || rows * hid > 65535 * ELEM_WG {
        return None;
    }
    // 太小不值得上 GPU（前向两次投影合计 4·rows·d·hid FLOPs，与逐算子路径同一口径）
    let flops = 4u128 * rows as u128 * d as u128 * hid as u128;
    if flops < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) as u128 {
        return None;
    }

    let mut rec = recorder()?;
    let hx = rec.upload(x);
    let hg = rec.upload(gamma);
    let hb = rec.upload(beta);
    let hw1 = rec.upload(w1);
    let hb1 = rec.upload(b1);
    let hw2 = rec.upload(w2);
    let hb2 = rec.upload(b2);
    // 归一化（顺带把每行统计量存进显存给反向复用）
    let xn = rec.empty(rows * d, KIND_OUT);
    let stats = rec.empty(rows * 2, KIND_OUT);
    rec.ln_fwd(&hx, &hg, &hb, &xn, &stats, rows, d, eps, is_rms);
    // Linear₁ + GELU：偏置融进 GELU 内核，同时存下 GELU 前的 z 供反向用
    let y1 = rec.matmul(&xn, false, &hw1, false, rows, d, hid, 1);
    let act = rec.empty(rows * hid, KIND_OUT);
    let z = rec.empty(rows * hid, KIND_OUT);
    rec.gelu_fwd_bias(&y1, &hb1, &act, &z, rows * hid, hid);
    // Linear₂ + 偏置 + dropout + 残差
    let y2 = rec.matmul(&act, false, &hw2, false, rows, hid, d, 1);
    let out = rec.empty(rows * d, KIND_OUT);
    let seed = DROPOUT_SEED.fetch_add(1, Ordering::Relaxed) as u32 ^ 0x9e37_79b9;
    rec.bias_dropout_residual(&y2, &hb2, &hx, &out, rows * d, d, dropout, seed, training);
    let out_host = rec.submit_and_read(&[&out])?.into_iter().next()?;
    Some(MlpResident {
        out: out_host,
        x: hx,
        xn,
        z,
        act,
        stats,
        w1: hw1,
        w2: hw2,
        gamma: hg,
        rows,
        d,
        hid,
        seed,
        dropout,
        training,
        is_rms,
    })
}

// ==================== 整叠 Block 常驻显存（打通子层边界） ====================
//
// 逐子层的常驻路径（`attn_layer_forward` / `mlp_forward`）已经把**一个子层内部**
// 的中间量留在显存，但子层与子层之间仍在打「回读 2MB → 上传 2MB」的来回：
//   · 前向：每个子层的输出被回读到 CPU（`Tensor::external` 需要前向数据），
//     下一层再原样传回显存当输入；
//   · 反向：上游梯度在 CPU 上组装好，再作为 `dout` 传回显存。
// 本配置（n_layer=4、b=8、t=512、d=128）下，8 个子层边界每步要付
// 8×(2MB 前向回读 + 2MB 反向上传) 的 PCIe 往返；消融实验里「把内核全摘掉」
// 仍有 8.4ms/批 的地板，正是这笔纯搬运账。
//
// 这里把**整叠 Block** 一次录进同一个 encoder：
//   · 前向一次提交，只回读整叠的最终输出（2MB）；
//   · 反向一次提交，上游梯度只上传一次（2MB），层与层之间的 dX 直接在显存接力，
//     回读的只有「整叠输入梯度 + 各层参数梯度」。
// 于是每步的 submit 从 ~18 次降到 2 次，子层边界的 PCIe 往返基本清零。

/// 每层 Block 的参数个数。
/// [`StackLayerArgs`] 的字段顺序 = 参数顺序 = [`StackGrads::grads`] 里每层的顺序：
/// γ₁, β₁, Wq, bq, Wk, bk, Wv, bv, Wproj, bproj, γ₂, β₂, W₁, b₁, W₂, b₂
pub const STACK_PARAMS_PER_LAYER: usize = 16;

/// 一层 Block 的全部权重/偏置（都是 CPU 上的借用，不复制）
pub struct StackLayerArgs<'a> {
    /// 注意力子层 LayerNorm 的 γ/β
    pub gamma1: &'a [f32],
    pub beta1: &'a [f32],
    /// Q/K/V 三个投影 [d, d] 与偏置 [d]
    pub wq: &'a [f32],
    pub bq: &'a [f32],
    pub wk: &'a [f32],
    pub bk: &'a [f32],
    pub wv: &'a [f32],
    pub bv: &'a [f32],
    /// 输出投影 [d, d] 与偏置 [d]
    pub wproj: &'a [f32],
    pub bproj: &'a [f32],
    /// 前馈子层 LayerNorm 的 γ/β
    pub gamma2: &'a [f32],
    pub beta2: &'a [f32],
    /// 前馈两层线性：W₁ [d, hid]、b₁ [hid]、W₂ [hid, d]、b₂ [d]
    pub w1: &'a [f32],
    pub b1: &'a [f32],
    pub w2: &'a [f32],
    pub b2: &'a [f32],
}

/// 整叠 Block 常驻显存前向的入参（都是 CPU 上的数据）
pub struct StackArgs<'a> {
    /// token embedding 的输出 [b*t, d]
    pub x: &'a [f32],
    /// 因果掩码 [t, t]（整叠共用一份，只上传一次）
    pub mask: &'a [f32],
    /// 逐层权重
    pub layers: &'a [StackLayerArgs<'a>],
    pub b: usize,
    pub t: usize,
    pub d: usize,
    pub n_head: usize,
    pub eps: f32,
    /// 归一化是否为 RMSNorm（各层共用同一份模型配置，故整叠一个开关）
    pub is_rms: bool,
    pub dropout: f32,
    pub training: bool,
}

/// 一个注意力子层的常驻显存句柄（前向产出，反向复用）
struct AttnSublayer {
    /// 子层输入（归一化反向 + 残差直通用）
    x: GpuHandle,
    /// 归一化输出（dWq/dWk/dWv 用）
    xn: GpuHandle,
    /// 归一化每行的 (mean, 1/σ)（RMSNorm 时 mean 槽位是 0）
    stats: GpuHandle,
    gamma: GpuHandle,
    /// Q'（已旋转、已乘 1/√head_dim）/ K（已旋转）/ V：[b*n_head, t, head_dim]
    q: GpuHandle,
    k: GpuHandle,
    v: GpuHandle,
    /// 注意力概率 P：[b*n_head*t, t]
    p: GpuHandle,
    /// 合并头之后的输出投影输入 [rows, d]
    merged: GpuHandle,
    wproj: GpuHandle,
    /// dropout 掩码种子：前向与反向必须一致，掩码由它重算
    seed: u32,
    /// 归一化是否为 RMSNorm（反向切同一个模式位）
    is_rms: bool,
}

/// 一个前馈子层的常驻显存句柄（前向产出，反向复用）
struct MlpSublayer {
    /// 子层输入（归一化反向 + 残差直通用）
    x: GpuHandle,
    /// 归一化输出
    xn: GpuHandle,
    /// Linear₁ 输出 + b₁（GELU 的输入）
    z: GpuHandle,
    /// GELU 输出
    act: GpuHandle,
    stats: GpuHandle,
    w1: GpuHandle,
    w2: GpuHandle,
    gamma: GpuHandle,
    seed: u32,
    /// 归一化是否为 RMSNorm（反向切同一个模式位）
    is_rms: bool,
}

/// 整叠 Block 一次提交跑完的结果：输出已回读，其余句柄留给反向复用。
pub struct StackResident {
    /// 整叠 Block 的输出（`ln_f` 之前），已回读到 CPU
    pub out: Vec<f32>,
    attn: Vec<AttnSublayer>,
    mlp: Vec<MlpSublayer>,
    b: usize,
    t: usize,
    d: usize,
    n_head: usize,
    dropout: f32,
    training: bool,
}

/// 整叠 Block 反向的边界梯度：对外只有「整叠输入梯度 + 各层参数梯度」两类，
/// 层与层之间的 dX 全程留在显存。
pub struct StackGrads {
    /// 对整叠输入 x 的梯度 [b*t, d]
    pub dx: Vec<f32>,
    /// 逐层的 [`STACK_PARAMS_PER_LAYER`] 项参数梯度，按层顺序平铺
    /// （顺序与 [`StackLayerArgs`] 的字段顺序一致）
    pub grads: Vec<Vec<f32>>,
}

/// 把一个注意力子层的前向录进 `rec`（不提交），返回反向要复用的句柄与子层输出句柄。
#[allow(clippy::too_many_arguments)]
fn record_attn_sublayer_fwd(
    rec: &mut GpuRecorder,
    hx: GpuHandle,
    la: &StackLayerArgs,
    mask: &GpuHandle,
    dims: (usize, usize, usize, usize),
    eps: f32,
    is_rms: bool,
    dropout: f32,
    training: bool,
) -> (AttnSublayer, GpuHandle) {
    let (b, t, d, n_head) = dims;
    let (rows, bn, hd) = (b * t, b * n_head, d / n_head);
    let hg = rec.upload(la.gamma1);
    let hb = rec.upload(la.beta1);
    let hwq = rec.upload(la.wq);
    let hbq = rec.upload(la.bq);
    let hwk = rec.upload(la.wk);
    let hbk = rec.upload(la.bk);
    let hwv = rec.upload(la.wv);
    let hbv = rec.upload(la.bv);
    let hwp = rec.upload(la.wproj);
    let hbp = rec.upload(la.bproj);
    // 归一化（顺带把每行统计量存进显存给反向复用）
    let xn = rec.empty(rows * d, KIND_OUT);
    let stats = rec.empty(rows * 2, KIND_OUT);
    rec.ln_fwd(&hx, &hg, &hb, &xn, &stats, rows, d, eps, is_rms);
    // QKV 三个投影
    let q_pre = rec.matmul(&xn, false, &hwq, false, rows, d, d, 1);
    let k_pre = rec.matmul(&xn, false, &hwk, false, rows, d, d, 1);
    let v_pre = rec.matmul(&xn, false, &hwv, false, rows, d, d, 1);
    // 按头重排 + 偏置（Q/K 顺带 RoPE；缩放放在 Q 上，softmax 内核就不必带 scale）
    let scale = 1.0 / (hd as f32).sqrt();
    let q = rec.empty(bn * t * hd, KIND_OUT);
    rec.heads_split(&q_pre, &hbq, &q, b, t, n_head, hd, HEADS_MODE_ROPE, scale);
    let k = rec.empty(bn * t * hd, KIND_OUT);
    rec.heads_split(&k_pre, &hbk, &k, b, t, n_head, hd, HEADS_MODE_ROPE, 1.0);
    let v = rec.empty(bn * t * hd, KIND_OUT);
    rec.heads_split(&v_pre, &hbv, &v, b, t, n_head, hd, HEADS_MODE_BIAS, 1.0);
    // S = Q'·Kᵀ → P = softmax(S + mask) → O = P·V
    let (p, o) = record_attn_fwd(rec, &q, &k, &v, mask, bn, t, t, hd);
    // 合并头 → 输出投影 → dropout → 残差
    let merged = rec.empty(rows * d, KIND_OUT);
    rec.heads_join(&o, &merged, b, t, n_head, hd, 0, 1.0);
    let y = rec.matmul(&merged, false, &hwp, false, rows, d, d, 1);
    let out = rec.empty(rows * d, KIND_OUT);
    let seed = DROPOUT_SEED.fetch_add(1, Ordering::Relaxed) as u32 ^ 0x9e37_79b9;
    rec.bias_dropout_residual(&y, &hbp, &hx, &out, rows * d, d, dropout, seed, training);
    // 前向用完就不再需要的句柄：交给录制器保活到 submit（encoder 仍引用它们的显存）
    for h in [hb, hwq, hbq, hwk, hbk, hwv, hbv, hbp, q_pre, k_pre, v_pre, o, y] {
        rec.keep(h);
    }
    (
        AttnSublayer { x: hx, xn, stats, gamma: hg, q, k, v, p, merged, wproj: hwp, seed, is_rms },
        out,
    )
}

/// 把一个前馈子层的前向录进 `rec`（不提交），返回反向要复用的句柄与子层输出句柄。
#[allow(clippy::too_many_arguments)]
fn record_mlp_sublayer_fwd(
    rec: &mut GpuRecorder,
    hx: GpuHandle,
    la: &StackLayerArgs,
    b: usize,
    t: usize,
    d: usize,
    hid: usize,
    eps: f32,
    is_rms: bool,
    dropout: f32,
    training: bool,
) -> (MlpSublayer, GpuHandle) {
    let rows = b * t;
    let hg = rec.upload(la.gamma2);
    let hb = rec.upload(la.beta2);
    let hw1 = rec.upload(la.w1);
    let hb1 = rec.upload(la.b1);
    let hw2 = rec.upload(la.w2);
    let hb2 = rec.upload(la.b2);
    let xn = rec.empty(rows * d, KIND_OUT);
    let stats = rec.empty(rows * 2, KIND_OUT);
    rec.ln_fwd(&hx, &hg, &hb, &xn, &stats, rows, d, eps, is_rms);
    // Linear₁ + GELU：偏置融进 GELU 内核，同时存下 GELU 前的 z 供反向用
    let y1 = rec.matmul(&xn, false, &hw1, false, rows, d, hid, 1);
    let act = rec.empty(rows * hid, KIND_OUT);
    let z = rec.empty(rows * hid, KIND_OUT);
    rec.gelu_fwd_bias(&y1, &hb1, &act, &z, rows * hid, hid);
    // Linear₂ + 偏置 + dropout + 残差
    let y2 = rec.matmul(&act, false, &hw2, false, rows, hid, d, 1);
    let out = rec.empty(rows * d, KIND_OUT);
    let seed = DROPOUT_SEED.fetch_add(1, Ordering::Relaxed) as u32 ^ 0x9e37_79b9;
    rec.bias_dropout_residual(&y2, &hb2, &hx, &out, rows * d, d, dropout, seed, training);
    for h in [hb, hb1, hb2, y1, y2] {
        rec.keep(h);
    }
    (MlpSublayer { x: hx, xn, z, act, stats, w1: hw1, w2: hw2, gamma: hg, seed, is_rms }, out)
}

/// 整叠 Block 常驻显存前向：所有子层录进**一次提交**，只有整叠输出回读。
///
/// 不适用时返回 None，调用方回退逐 Block 路径（数值行为不变）：
/// 形状/规模不满足、各层形状不一致、GPU 不可用。
pub fn stack_forward(a: &StackArgs) -> Option<StackResident> {
    let StackArgs { x, mask, layers, b, t, d, n_head, eps, is_rms, dropout, training } = *a;
    if layers.is_empty() || b == 0 || t == 0 || d == 0 || n_head == 0 || d % n_head != 0 {
        return None;
    }
    let hd = d / n_head;
    if hd == 0 || hd % 2 != 0 {
        return None;
    }
    let rows = b * t;
    if x.len() != rows * d || mask.len() < t * t {
        return None;
    }
    // 单维 workgroup 上限：LN 按行、按列求和按列、重排/逐元素按元素对数
    if rows > 65535 || d > 65535 || rows * d > 65535 * ELEM_WG {
        return None;
    }
    // 与逐子层路径同一口径的规模阈值（QKV 三个投影 + 输出投影合计 8·rows·d² FLOPs）
    let attn_flops = 8u128 * rows as u128 * d as u128 * d as u128;
    if attn_flops < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) as u128 {
        return None;
    }
    if !attn_resident_ok(b * n_head, t, t, hd, mask.len()) {
        return None;
    }
    let mut hid = 0usize;
    for la in layers {
        if la.gamma1.len() != d
            || la.beta1.len() != d
            || la.wq.len() != d * d
            || la.bq.len() != d
            || la.wk.len() != d * d
            || la.bk.len() != d
            || la.wv.len() != d * d
            || la.bv.len() != d
            || la.wproj.len() != d * d
            || la.bproj.len() != d
            || la.gamma2.len() != d
            || la.beta2.len() != d
            || la.w2.len() != la.b1.len() * d
            || la.b2.len() != d
            || la.w1.len() != d * la.b1.len()
        {
            return None;
        }
        // 逐子层路径要求各层形状一致；不一致就整段回退（本项目模型天然一致）
        if hid != 0 && hid != la.b1.len() {
            return None;
        }
        hid = la.b1.len();
        let mlp_flops = 4u128 * rows as u128 * d as u128 * hid as u128;
        if mlp_flops < MATMUL_MIN_FLOPS.load(Ordering::Relaxed) as u128 {
            return None;
        }
        if hid > 65535 || rows * hid > 65535 * ELEM_WG {
            return None;
        }
    }

    let mut rec = recorder()?;
    let hmask = rec.upload(mask);
    let mut hx = rec.upload(x);
    let mut attn = Vec::with_capacity(layers.len());
    let mut mlp = Vec::with_capacity(layers.len());
    for la in layers {
        let (ah, a_out) = record_attn_sublayer_fwd(
            &mut rec, hx, la, &hmask, (b, t, d, n_head), eps, is_rms, dropout, training,
        );
        attn.push(ah);
        let (mh, m_out) =
            record_mlp_sublayer_fwd(&mut rec, a_out, la, b, t, d, hid, eps, is_rms, dropout, training);
        mlp.push(mh);
        hx = m_out;
    }
    // 只有整叠的最终输出回读；掩码与整叠输入句柄由本函数的局部变量保活到 submit 之后
    let out_host = rec.submit_and_read(&[&hx])?.into_iter().next()?;
    drop(hmask);
    Some(StackResident {
        out: out_host,
        attn,
        mlp,
        b,
        t,
        d,
        n_head,
        dropout,
        training,
    })
}

/// 把一个注意力子层的反向录进 `rec`（不提交）。
///
/// `dout` 是上游梯度句柄；返回（对子层输入的梯度句柄, 10 项参数梯度句柄），
/// 参数梯度顺序：dγ₁, dβ₁, dWq, dbq, dWk, dbk, dWv, dbv, dWproj, dbproj。
#[allow(clippy::too_many_arguments)]
fn record_attn_sublayer_bwd(
    rec: &mut GpuRecorder,
    h: &AttnSublayer,
    dout: &GpuHandle,
    dims: (usize, usize, usize, usize),
    dropout: f32,
    training: bool,
) -> (GpuHandle, Vec<GpuHandle>) {
    let (b, t, d, n_head) = dims;
    let (rows, bn, hd) = (b * t, b * n_head, d / n_head);
    let scale = 1.0 / (hd as f32).sqrt();
    // 1) dropout 反向：dpre = dout ⊙ mask（掩码由同一 seed 重算）
    let dpre = rec.empty(rows * d, KIND_OUT);
    rec.dropout_bwd(dout, &dpre, rows * d, dropout, h.seed, training);
    // 2) c_proj 反向：dmerged = dpre·Wprojᵀ，dWproj = mergedᵀ·dpre，dbproj = Σ_r dpre
    let dmerged = rec.matmul(&dpre, false, &h.wproj, true, rows, d, d, 1);
    let dwproj = rec.matmul(&h.merged, true, &dpre, false, d, rows, d, 1);
    let dbproj = rec.empty(d, KIND_OUT);
    rec.col_sum(&dpre, &dbproj, rows, d);
    // 3) 拆回按头布局：dO = dmerged（纯搬运，不读偏置，binding 1 拿 x 占位）
    let do_ = rec.empty(bn * t * hd, KIND_OUT);
    rec.heads_split(&dmerged, &h.x, &do_, b, t, n_head, hd, HEADS_MODE_PLAIN, 1.0);
    // 4) 注意力反向：dV → dP → dS → dQ/dK
    let (dq, dk, dv) = record_attn_bwd(rec, &h.p, &h.q, &h.k, &h.v, &do_, bn, t, t, hd);
    // 5) 合并回 [rows, d]：RoPE 用转置旋转回传，Q 的 1/√head_dim 也在这里乘回
    let dq_pre = rec.empty(rows * d, KIND_OUT);
    rec.heads_join(&dq, &dq_pre, b, t, n_head, hd, 1, scale);
    let dk_pre = rec.empty(rows * d, KIND_OUT);
    rec.heads_join(&dk, &dk_pre, b, t, n_head, hd, 1, 1.0);
    let dv_pre = rec.empty(rows * d, KIND_OUT);
    rec.heads_join(&dv, &dv_pre, b, t, n_head, hd, 0, 1.0);
    // 6) QKV 投影反向：偏置梯度 = 列求和；权重梯度 = xnᵀ·d_pre
    let dbq = rec.empty(d, KIND_OUT);
    rec.col_sum(&dq_pre, &dbq, rows, d);
    let dbk = rec.empty(d, KIND_OUT);
    rec.col_sum(&dk_pre, &dbk, rows, d);
    let dbv = rec.empty(d, KIND_OUT);
    rec.col_sum(&dv_pre, &dbv, rows, d);
    let dwq = rec.matmul(&h.xn, true, &dq_pre, false, d, rows, d, 1);
    let dwk = rec.matmul(&h.xn, true, &dk_pre, false, d, rows, d, 1);
    let dwv = rec.matmul(&h.xn, true, &dv_pre, false, d, rows, d, 1);
    // 7) 三路梯度汇合成对 LayerNorm 输出的梯度
    let sum2 = rec.empty(rows * d, KIND_OUT);
    rec.add(&dq_pre, &dk_pre, &sum2, rows * d);
    let dln = rec.empty(rows * d, KIND_OUT);
    rec.add(&sum2, &dv_pre, &dln, rows * d);
    // 8) 归一化反向（LayerNorm / RMSNorm 同一套内核，模式位从前向的句柄里带过来）
    let dxln = rec.empty(rows * d, KIND_OUT);
    rec.ln_bwd_x(&dln, &h.x, &h.stats, &dxln, &h.gamma, rows, d, h.is_rms);
    let dgamma = rec.empty(d, KIND_OUT);
    let dbeta = rec.empty(d, KIND_OUT);
    rec.ln_bwd_gb(&dln, &h.x, &h.stats, &dgamma, &dbeta, rows, d);
    // 9) 残差直通：∂out/∂x 的那一份就是 dout —— 在显存里相加，
    //    于是这一份梯度不必回读 CPU 再由上一层传回来
    let dx = rec.empty(rows * d, KIND_OUT);
    rec.add(&dxln, dout, &dx, rows * d);
    for h in [dpre, dmerged, do_, dq, dk, dv, dq_pre, dk_pre, dv_pre, sum2, dln, dxln] {
        rec.keep(h);
    }
    (
        dx,
        vec![dgamma, dbeta, dwq, dbq, dwk, dbk, dwv, dbv, dwproj, dbproj],
    )
}

/// 把一个前馈子层的反向录进 `rec`（不提交）。
///
/// `dout` 是上游梯度句柄；返回（对子层输入的梯度句柄, 6 项参数梯度句柄），
/// 参数梯度顺序：dγ₂, dβ₂, dW₁, db₁, dW₂, db₂。
#[allow(clippy::too_many_arguments)]
fn record_mlp_sublayer_bwd(
    rec: &mut GpuRecorder,
    h: &MlpSublayer,
    dout: &GpuHandle,
    b: usize,
    t: usize,
    d: usize,
    hid: usize,
    dropout: f32,
    training: bool,
) -> (GpuHandle, Vec<GpuHandle>) {
    let rows = b * t;
    // 1) dropout 反向
    let dypre = rec.empty(rows * d, KIND_OUT);
    rec.dropout_bwd(dout, &dypre, rows * d, dropout, h.seed, training);
    // 2) db₂ = Σ_r dypre[r,j]
    let db2 = rec.empty(d, KIND_OUT);
    rec.col_sum(&dypre, &db2, rows, d);
    // 3) dact = dypre·W₂ᵀ，dW₂ = actᵀ·dypre
    let dact = rec.matmul(&dypre, false, &h.w2, true, rows, d, hid, 1);
    let dw2 = rec.matmul(&h.act, true, &dypre, false, hid, rows, d, 1);
    // 4) GELU 反向；5) db₁ = Σ_r dz[r,j]；6) dxn = dz·W₁ᵀ，dW₁ = xnᵀ·dz
    let dz = rec.empty(rows * hid, KIND_OUT);
    rec.gelu_bwd(&dact, &h.z, &dz, rows * hid);
    let db1 = rec.empty(hid, KIND_OUT);
    rec.col_sum(&dz, &db1, rows, hid);
    let dxn = rec.matmul(&dz, false, &h.w1, true, rows, hid, d, 1);
    let dw1 = rec.matmul(&h.xn, true, &dz, false, d, rows, hid, 1);
    // 7) 归一化反向（LayerNorm / RMSNorm 同一套内核）
    let dxln = rec.empty(rows * d, KIND_OUT);
    rec.ln_bwd_x(&dxn, &h.x, &h.stats, &dxln, &h.gamma, rows, d, h.is_rms);
    let dgamma = rec.empty(d, KIND_OUT);
    let dbeta = rec.empty(d, KIND_OUT);
    rec.ln_bwd_gb(&dxn, &h.x, &h.stats, &dgamma, &dbeta, rows, d);
    // 8) 残差直通在显存里相加
    let dx = rec.empty(rows * d, KIND_OUT);
    rec.add(&dxln, dout, &dx, rows * d);
    for h in [dypre, dact, dz, dxn, dxln] {
        rec.keep(h);
    }
    (dx, vec![dgamma, dbeta, dw1, db1, dw2, db2])
}

impl StackResident {
    /// 整叠 Block 的反向：所有子层的反向录进**一次提交**。
    ///
    /// 上游梯度只上传一次，层与层之间的 dX 在显存里接力；
    /// 回读的只有「整叠输入梯度（2MB）+ 各层参数梯度（每层 ~0.3MB）」。
    pub fn backward(&self, dout: &[f32]) -> Option<StackGrads> {
        if dout.len() != self.b * self.t * self.d {
            return None;
        }
        // attn/mlp 成对出现；空层列表在 stack_forward 里已被拒掉
        if self.attn.len() != self.mlp.len() || self.attn.is_empty() {
            return None;
        }
        let hid = self.mlp[0].act.len / (self.b * self.t);
        if hid == 0 {
            return None;
        }
        let dims = (self.b, self.t, self.d, self.n_head);
        let mut rec = recorder()?;
        let mut dcur = rec.upload(dout);
        // 逐层倒着录；参数梯度按层号归位（反向走的是倒序）
        let mut per_layer: Vec<Option<Vec<GpuHandle>>> =
            (0..self.attn.len()).map(|_| None).collect();
        for i in (0..self.attn.len()).rev() {
            let (d_mlp_in, g_mlp) = record_mlp_sublayer_bwd(
                &mut rec, &self.mlp[i], &dcur, self.b, self.t, self.d, hid, self.dropout,
                self.training,
            );
            let (d_prev, g_attn) = record_attn_sublayer_bwd(
                &mut rec, &self.attn[i], &d_mlp_in, dims, self.dropout, self.training,
            );
            // 上一棒的两个上游梯度句柄前向已用尽：保活到 submit
            rec.keep(dcur);
            rec.keep(d_mlp_in);
            let mut per = Vec::with_capacity(STACK_PARAMS_PER_LAYER);
            per.extend(g_attn); // 10 项：γ₁, β₁, Wq, bq, Wk, bk, Wv, bv, Wproj, bproj
            per.extend(g_mlp); // 6 项：γ₂, β₂, W₁, b₁, W₂, b₂
            per_layer[i] = Some(per);
            dcur = d_prev;
        }
        // 回读顺序：[整叠输入梯度, 第 0 层 16 项, 第 1 层 16 项, ...]
        let mut flat: Vec<&GpuHandle> = Vec::with_capacity(1 + per_layer.len() * STACK_PARAMS_PER_LAYER);
        flat.push(&dcur);
        for slot in &per_layer {
            for h in slot.as_ref()? {
                flat.push(h);
            }
        }
        let outs = rec.submit_and_read(&flat)?;
        let mut it = outs.into_iter();
        let dx = it.next()?;
        Some(StackGrads { dx, grads: it.collect() })
    }
}

// ==================== 判定实验：批量提交下的真实吞吐 ====================
//
// 目的：在动手重写整个后端之前，先量出"把一步的 dispatch 合成一次提交"到底能拿到多少。
// 做法：先录一步训练真实用到的 matmul 形状（此时 GPU 不参与，CPU 兜底保证数值正确），
// 再用单次提交回放这些形状，测录制耗时、提交+等待耗时与有效算力。
// 触发方式：`LLM_GPU_PROBE=1`。

/// 形状录制开关：打开后 `matmul` 只记录形状并返回 None（CPU 兜底），不真算
static PROBE_CAPTURE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
/// 录到的形状：(m, k, n, batch, a_t, b_t)
static PROBE_SHAPES: Mutex<Vec<(usize, usize, usize, usize, bool, bool)>> = Mutex::new(Vec::new());

/// 打开/关闭形状录制
pub fn probe_capture(on: bool) {
    PROBE_CAPTURE.store(on, Ordering::Relaxed);
    if on {
        PROBE_SHAPES.lock().unwrap().clear();
    }
}

/// 当前是否处于 `LLM_GPU_PROBE` 录制模式。
///
/// 录制模式下常驻路径整体关闭（`recorder()` 返回 None），`Tensor::flash_attention`
/// 需要这个信号把前向拆回逐算子（`attn_forward_ops`），让 matmul 记录真实训练形状。
pub fn probe_active() -> bool {
    PROBE_CAPTURE.load(Ordering::Relaxed)
}

/// 回放录制到的形状，**按形状分组**打印批量提交下的真实吞吐。
///
/// 分组的理由：一步训练里同一形状会重复出现几十次（4 层 × 前向/反向），只看总吞吐
/// 会把「lm_head 那几块 134 MB 的巨型矩阵乘」和「几十个小形状」混在一起平均掉，
/// 无法判断该改哪个 tile 尺寸。这里先数清每种形状每一步出现几次，再**单独回放**该形状
/// 测出「一次多少 ms」，两者相乘才是这个形状每步真正吃掉的 GPU 时间。
pub fn probe_report() {
    PROBE_CAPTURE.store(false, Ordering::Relaxed);
    let shapes = std::mem::take(&mut *PROBE_SHAPES.lock().unwrap());
    if shapes.is_empty() {
        return;
    }
    let Some(Some(g)) = GPU.get() else {
        return;
    };

    // ---- 1. 按 (m,k,n,batch,a_t,b_t) 分组计数 ----
    type Key = (usize, usize, usize, usize, bool, bool);
    let mut groups: std::collections::HashMap<Key, usize> = std::collections::HashMap::new();
    for &key in &shapes {
        *groups.entry(key).or_insert(0) += 1;
    }

    // ---- 2. 操作数 buffer 按所有形状的最大物理尺寸各分配一块 ----
    // 内核只在各自的 (m,k,n,batch) 边界内读写，所以一块「够大」的 buffer 就能给所有形状复用。
    // **不往里写数据**：WebGPU 保证新建 buffer 零初始化，而往 134MB 的 buffer 写哑元数据
    // 会把几十毫秒的 PCIe 上传时间混进第一个被测形状（实测让 4096x128x512 从 2.3ms 假变成 68ms）。
    let (mut max_a, mut max_b) = (1usize, 1usize);
    for &(m, k, n, bs, a_t, b_t) in groups.keys() {
        // 转置时物理布局的行列互换（a_t 时物理 a 是 [B,K,M]，b_t 时物理 b 是 [B,N,K]）
        let (ar, ac) = if a_t { (k, m) } else { (m, k) };
        let (br, bc) = if b_t { (n, k) } else { (k, n) };
        max_a = max_a.max(ar * ac * bs);
        max_b = max_b.max(br * bc * bs);
    }
    // 分块提交：Windows TDR 会在单次 GPU 提交超过 ~2s 时判定 hang 并踢掉设备
    // （实测把 198 次 dispatch 塞进一次提交后，紧接着的 write_buffer 就报 "Buffer is invalid"）。
    // 这也更贴近真实训练「每步提交若干批」的形态。
    const CHUNK: usize = 24;
    // 输出 buffer 轮转：连续 dispatch 写同一块 buffer 会被驱动插入 WAW 屏障串起来，
    // 轮转几块才能让相邻 dispatch 真正重叠。小形状用 8 块，大形状（logits 那块 134MB）
    // 只留 2 块——8 块 134MB 会直接把 2GB 显存撑爆，测出来的数反而全错。
    const ROT_SMALL: usize = 8;
    const ROT_BIG: usize = 2;
    /// 单块输出超过这个字节数就算「大形状」，少留几块轮转
    const BIG_OUT_BYTES: u64 = 32 << 20;
    let buf_a = g.take_buf(max_a, KIND_IN);
    let buf_b = g.take_buf(max_b, KIND_IN);
    // 空提交一次，把所有待处理的初始化（零填充、管线首次绑定）排干净，
    // 免得它们的耗时落进后面第一个被测形状
    {
        let batch = g.batch_begin();
        g.batch_finish(batch);
    }

    // ---- 3. 逐形状单独回放 ----
    // 先排序：HashMap 的迭代顺序每个进程都不一样，而**第一个被测形状**会被
    // 「新建 134MB buffer 的零填充」一次性开销污染（实测能从 2.3ms 假变成 68ms）。
    // 固定顺序 + 每组预热之后，测量才可复现。
    let mut keys: Vec<Key> = groups.keys().copied().collect();
    keys.sort_by_key(|&(m, k, n, bs, a_t, b_t)| (a_t, b_t, m, k, n, bs));
    let (mut total_ms, mut total_flops, mut n_disp) = (0.0f64, 0.0f64, 0usize);
    let mut rows: Vec<(f64, Key, usize, f64)> = Vec::new(); // (该形状合计 ms, 形状, 次数, GFLOP/s)
    for &(m, k, n, bs, a_t, b_t) in &keys {
        let cnt = groups[&(m, k, n, bs, a_t, b_t)];
        // 输出只要装得下这个形状就够：按形状取精确尺寸，用完即还，避免多个 134MB 同时驻留
        let out_len = m * n * bs;
        let rot = if out_len as u64 * 4 > BIG_OUT_BYTES { ROT_BIG } else { ROT_SMALL };
        let outs: Vec<wgpu::Buffer> = (0..rot).map(|_| g.take_buf(out_len, KIND_OUT)).collect();
        // 预热：先用这一组自己的 buffer 真跑一次并等它完成。WebGPU 的 buffer 零初始化是
        // **首次使用时**才录进命令流的，不预热就会有一大段清零时间落进计时窗口。
        {
            let mut warm = g.batch_begin();
            g.batch_matmul(&mut warm, &buf_a, &buf_b, &outs[0], m, k, n, bs, a_t, b_t);
            g.batch_finish(warm);
        }
        let t0 = std::time::Instant::now();
        let mut done = 0usize;
        while done < cnt {
            let part = (cnt - done).min(CHUNK);
            let mut batch = g.batch_begin();
            for j in 0..part {
                let out = &outs[(n_disp + j) % rot];
                g.batch_matmul(&mut batch, &buf_a, &buf_b, out, m, k, n, bs, a_t, b_t);
            }
            n_disp += batch.dispatches;
            done += part;
            g.batch_finish(batch);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        let flops = 2.0 * m as f64 * k as f64 * n as f64 * bs as f64 * cnt as f64;
        rows.push((
            ms,
            (m, k, n, bs, a_t, b_t),
            cnt,
            flops / 1e9 / (ms / 1000.0).max(1e-9),
        ));
        total_ms += ms;
        total_flops += flops;
    }

    // 录制窗口可能跨了好几个训练步（CPU 兜底时 ~2.3s/步，首次进度打印在 5s 后），
    // 所以「每步几次」要用所有形状次数的**最大公约数**归一，而不是假定只录了一步。
    let steps = groups.values().fold(0usize, |acc, &c| gcd(acc, c)).max(1);
    rows.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    println!(
        "[gpu-probe] 形状回放：{} 类 / {} 次 dispatch（{} 步）| {:.1} GFLOP | 合计 {:.0}ms | {:.1} GFLOP/s",
        groups.len(),
        n_disp,
        steps,
        total_flops / 1e9,
        total_ms,
        total_flops / 1e9 / (total_ms / 1000.0).max(1e-9),
    );
    println!("[gpu-probe]   matmul 每步合计 {:.0}ms，按形状拆：", total_ms / steps as f64);
    for (ms, (m, k, n, bs, a_t, b_t), cnt, gf) in rows.iter().take(18) {
        println!(
            "[gpu-probe]     {:>7.1}ms/步 | {:>5.1}% | m={m:<5} k={k:<5} n={n:<5} b={bs:<3} \
             a_t={:<5} b_t={:<5} ×{:>3}/步 | {:.2}ms/次 | {:.0} GFLOP/s",
            ms / steps as f64,
            ms / total_ms * 100.0,
            a_t,
            b_t,
            cnt / steps,
            ms / *cnt as f64,
            gf
        );
    }
}

/// 求最大公约数（仅用于把回放计数归一成「每步几次」）
fn gcd(a: usize, b: usize) -> usize {
    if a == 0 {
        return b;
    }
    gcd(b % a, a)
}

/// 峰值探针：纯寄存器 FMA 的实测吞吐上限。
///
/// 目的（历史）：那时 matmul 只跑到 ~36 GFLOP/s（MX150 理论峰值 ~1100 GFLOP/s），需要分清瓶颈性质：
/// - 若这里能跑到几百 GFLOP/s → 是 matmul 着色器的问题（共享内存 LDS 往返太平凡，值得重写内核）
/// - 若这里也只有几十 GFLOP/s → 是这块 15W 入门卡本身的硬件/功耗上限，重写内核也没用
///
/// 实测结论：落到第一种，本机纯 FMA 能到 524 GFLOP/s。按这个结论重写分块后，
/// matmul 各形状已到 200~230 GFLOP/s，所以下面打印的峰值是**基线**，不是 matmul 的现状。
pub fn probe_fma() {
    let Some(Some(g)) = GPU.get() else {
        return;
    };
    const GROUPS: u32 = 1024; // 1024 个 workgroup × 256 线程 = 26.2 万线程，足够喂满 3 个 SM
    const ITERS: u32 = 8_000; // 每次迭代 8 条 FMA（=16 FLOP）
    let threads = GROUPS as f64 * 256.0;
    let flops = threads * ITERS as f64 * 8.0 * 2.0;

    let dummy = vec![0.0f32; 256];
    let buf_in = g.take_buf(dummy.len(), KIND_IN);
    g.queue.write_buffer(&buf_in, 0, bytemuck_bytes(&dummy));
    let out = g.take_buf(256, KIND_OUT);

    // 先热身一次（吃掉管线首次绑定的开销），再计时
    let _ = g.run(
        &g.fma_pipe,
        &g.unary_layout,
        &[(&buf_in, 0), (&out, 2)],
        [ITERS, 0, 0, 0, 0, 0],
        &out,
        256,
        GROUPS,
        1,
        1,
    );
    let t = std::time::Instant::now();
    if g
        .run(
            &g.fma_pipe,
            &g.unary_layout,
            &[(&buf_in, 0), (&out, 2)],
            [ITERS, 0, 0, 0, 0, 0],
            &out,
            256,
            GROUPS,
            1,
            1,
        )
        .is_none()
    {
        return;
    }
    let s = t.elapsed().as_secs_f64();
    println!(
        "[gpu-probe] 纯 FMA 峰值：{:.1} GFLOP 用时 {:.1}ms | 实测 {:.1} GFLOP/s（MX150 标称 ~1100）",
        flops / 1e9,
        s * 1000.0,
        flops / 1e9 / s
    );
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

    /// GPU 寄存器分块 matmul 与 CPU 参考实现逐元素对比（覆盖 128 tile 的整数倍、
    /// 部分越界 tile、以及反向传播用到的转置访问组合）
    #[test]
    fn gpu_matmul_matches_cpu() {
        init();
        if !is_available() {
            return; // 无 GPU 环境跳过（不视为失败）
        }
        // 阈值压到 0，强制下面每个形状都真的走 GPU 内核（否则小形状会被 CPU 兜底挡掉）
        MATMUL_MIN_FLOPS.store(0, Ordering::Relaxed);
        // (m, k, n, batch, a_t, b_t)
        let shapes: &[(usize, usize, usize, usize, bool, bool)] = &[
            (256, 64, 64, 1, false, false),   // 2D，全 128 倍数
            (256, 256, 64, 1, false, false),  // 2D 反向形状
            (32, 16, 32, 32, false, false),   // 3D 批量（注意力 scores）
            (32, 32, 16, 32, false, false),   // 3D 批量（attn·v）
            (32, 24, 40, 8, false, false),    // 非 128 倍数边界（demo_gpu 用）
            (100, 100, 100, 1, false, false), // 非 128 倍数
            (140, 24, 132, 2, false, false),  // 刚好跨过 128 tile 边界
            (2048, 256, 256, 1, false, false), // 大矩阵（训练 QKV 投影规模）
            // 反向传播：∂a = g @ bᵀ（b 转置读）、∂b = aᵀ @ g（a 转置读）
            (32, 32, 16, 32, false, true),
            (32, 16, 32, 32, true, false),
            (132, 16, 140, 1, true, true), // 跨 tile + 双重转置
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

    /// GPU 掩码 softmax 与 CPU 参考实现逐元素对比。
    ///
    /// 形状取训练里真实出现的规模，且 `d` 跨过 workgroup_size(256) 的多轮边界——
    /// `d > 256` 时每个线程扫多轮，workgroup 内各 warp 的进度会被拉开，
    /// 正好能暴露「共享归约数组在两趟之间复用却缺 barrier」这类竞态（表现为整行 exp 全错）。
    #[test]
    fn gpu_masked_softmax_matches_cpu() {
        init();
        if !is_available() {
            return;
        }
        // 元素数都大于 SOFTMAX_MIN_ELEMS，保证真的走 GPU 内核（否则会静默回退 CPU）
        for &(rows, d) in &[(4096usize, 256usize), (4096, 512), (1000, 300), (300, 700)] {
            let x: Vec<f32> = (0..rows * d)
                .map(|i| ((i % 97) as f32 * 0.017).sin() * 4.0)
                .collect();
            // 掩码按行右对齐（与内核的 base % mask_numel 对齐方式一致）：
            // 每行只放开前 (r % d) + 1 列，其余 -inf，保证每行至少一个可见列
            let mut mask = vec![f32::NEG_INFINITY; rows * d];
            for r in 0..rows {
                for j in 0..(r % d + 1) {
                    mask[r * d + j] = 0.0;
                }
            }
            let g = softmax_mask(&x, &mask, rows, d, mask.len())
                .unwrap_or_else(|| panic!("rows={rows} d={d} 应走 GPU 内核"));
            for r in 0..rows {
                let row = &x[r * d..(r + 1) * d];
                let mrow = &mask[r * d..(r + 1) * d];
                let mx = row
                    .iter()
                    .zip(mrow)
                    .fold(f32::NEG_INFINITY, |a, (&v, &m)| a.max(v + m));
                let sum: f32 = row.iter().zip(mrow).map(|(&v, &m)| (v + m - mx).exp()).sum();
                for j in 0..d {
                    let want = ((row[j] + mrow[j] - mx).exp()) / sum;
                    let err = (g[r * d + j] - want).abs();
                    assert!(
                        err < 1e-5,
                        "rows={rows} d={d} r={r} j={j} GPU={} 参考={want} 误差={err}",
                        g[r * d + j]
                    );
                }
            }
        }
    }

    /// 输出头交叉熵常驻路径（`lm_head_ce` + `LmHeadResident::backward`）的数值校验。
    ///
    /// 参考值用纯循环独立算一遍（loss / d_hidden / d_weight），不依赖任何 Tensor 算子，
    /// 因此前向与反向都能验证。形状取到 5 亿 FLOPs 以上，确保过得了 `lm_head_ce` 的尺寸守卫。
    #[test]
    fn gpu_lm_head_ce_matches_loop_reference() {
        init();
        if !is_available() {
            return;
        }
        let (rows, d, vocab) = (512usize, 128usize, 4096usize);
        let x: Vec<f32> = (0..rows * d)
            .map(|i| ((i % 89) as f32 * 0.031).sin())
            .collect();
        let w: Vec<f32> = (0..vocab * d)
            .map(|i| ((i % 61) as f32 * 0.017).cos() * 0.5)
            .collect();
        let targets: Vec<usize> = (0..rows).map(|i| (i * 7) % vocab).collect();

        let res = lm_head_ce(&x, &w, &targets, rows, d, vocab)
            .expect("该形状应走常驻显存路径");

        // ---- 纯循环参考：前向 logits → logsumexp → CE，以及 dlogits ----
        let mut logits = vec![0.0f32; rows * vocab];
        for r in 0..rows {
            for j in 0..vocab {
                let mut s = 0.0;
                for kk in 0..d {
                    s += x[r * d + kk] * w[j * d + kk]; // logits = x @ Wᵀ
                }
                logits[r * vocab + j] = s;
            }
        }
        let mut dl = vec![0.0f32; rows * vocab];
        let mut loss_ref = 0.0f32;
        for r in 0..rows {
            let row = &logits[r * vocab..(r + 1) * vocab];
            let mx = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let sum: f32 = row.iter().map(|&v| (v - mx).exp()).sum();
            let lse = sum.ln() + mx;
            loss_ref += lse - logits[r * vocab + targets[r]];
            for j in 0..vocab {
                let mut g = (logits[r * vocab + j] - lse).exp();
                if j == targets[r] {
                    g -= 1.0; // -onehot
                }
                dl[r * vocab + j] = g / rows as f32;
            }
        }
        loss_ref /= rows as f32;

        let loss_err = (res.loss - loss_ref).abs();
        assert!(
            loss_err < 1e-4,
            "loss 不一致：GPU {} 参考 {} 误差 {}",
            res.loss,
            loss_ref,
            loss_err
        );

        // ---- 参考反向：d_hidden = dlogits @ W，d_W = dlogitsᵀ @ hidden ----
        let mut dx_ref = vec![0.0f32; rows * d];
        for r in 0..rows {
            for kk in 0..d {
                let mut s = 0.0;
                for j in 0..vocab {
                    s += dl[r * vocab + j] * w[j * d + kk];
                }
                dx_ref[r * d + kk] = s;
            }
        }
        let mut dw_ref = vec![0.0f32; vocab * d];
        for j in 0..vocab {
            for kk in 0..d {
                let mut s = 0.0;
                for r in 0..rows {
                    s += dl[r * vocab + j] * x[r * d + kk];
                }
                dw_ref[j * d + kk] = s;
            }
        }

        let (dx, dw) = res.backward().expect("常驻反向应成功");
        for (name, got, want) in [("d_hidden", &dx, &dx_ref), ("d_weight", &dw, &dw_ref)] {
            let max_err = got
                .iter()
                .zip(want)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            let mag = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            assert!(
                max_err / mag.max(1e-6) < 1e-3,
                "{name} 不一致：最大绝对误差 {max_err}（参考量级 {mag}）"
            );
        }
    }

    /// 与 WGSL 的 `hash01`（splitmix32 终混）逐位一致的 CPU 参考
    fn ref_hash01(seed: u32) -> f32 {
        let mut z = seed;
        z = (z ^ (z >> 16)).wrapping_mul(0x7feb_352d);
        z = (z ^ (z >> 15)).wrapping_mul(0x846c_a68b);
        z ^= z >> 16;
        (z >> 8) as f32 * (1.0 / 16_777_216.0)
    }

    /// 与 WGSL 的 `drop_scale` 逐位一致的 dropout 因子（前向/反向共用同一掩码）
    fn ref_drop_scale(i: usize, p: f32, seed: u32, training: bool) -> f32 {
        if !training || p <= 0.0 {
            return 1.0;
        }
        let keep = 1.0 - p;
        if ref_hash01(seed ^ (i as u32).wrapping_mul(0x9e37_79b9)) < keep {
            1.0 / keep
        } else {
            0.0
        }
    }

    /// GELU（tanh 近似）与解析导数，与 [`Tensor::gelu`] 及 WGSL 实现同式
    fn ref_gelu(x: f32) -> f32 {
        0.5 * x * (1.0 + (0.797_884_56 * (x + 0.044_715 * x * x * x)).tanh())
    }

    fn ref_gelu_grad(x: f32) -> f32 {
        let t = (0.797_884_56 * (x + 0.044_715 * x * x * x)).tanh();
        let da_dx = 0.797_884_56 * (1.0 + 3.0 * 0.044_715 * x * x);
        0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * da_dx
    }

    /// 按列求和（偏置梯度）
    fn ref_col_sum(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; cols];
        for r in 0..rows {
            for j in 0..cols {
                out[j] += a[r * cols + j];
            }
        }
        out
    }

    /// 逐元素比对（相对参考量级，容差 1e-3）
    fn assert_close(name: &str, got: &[f32], want: &[f32]) {
        assert_eq!(got.len(), want.len(), "{name} 长度不一致");
        let max_err = got
            .iter()
            .zip(want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let mag = want.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = max_err / mag.max(1e-6);
        assert!(
            rel < 1e-3,
            "{name} 不一致：最大绝对误差 {max_err}（参考量级 {mag}，相对 {rel:.3e}）"
        );
        eprintln!("[cmp] {name}: rel = {rel:.3e}");
    }

    /// 常驻 MLP 子层的前向与全部 7 项边界梯度 vs 纯循环参考。
    ///
    /// 两组形状：一组小而规整；一组让 d/hid 跨过 workgroup_size(256) 的多轮归约边界
    /// （每线程扫多轮 + 共享归约数组复用，正是上一轮出过竞态的地方）。
    /// dropout 打开 —— 反向要按同一种子重算掩码，这里用逐位一致的 CPU 参考验证。
    #[test]
    fn gpu_mlp_resident_matches_loop_reference() {
        init();
        if !is_available() {
            return;
        }
        MATMUL_MIN_FLOPS.store(0, Ordering::Relaxed);
        // 每个形状跑两遍：LayerNorm 与 RMSNorm（同一套内核、只切模式位）。
        // RMSNorm 下喂进去的 `beta` 仍是非零向量，内核应把它整个丢弃。
        let cases: Vec<(usize, usize, usize, bool)> =
            [(48usize, 32usize, 64usize), (40, 300, 260)]
                .into_iter()
                .flat_map(|(r, d, h)| [(r, d, h, false), (r, d, h, true)])
                .collect();
        for &(rows, d, hid, is_rms) in &cases {
            let (dropout, eps) = (0.1f32, 1e-5f32);
            let x: Vec<f32> = (0..rows * d).map(|i| ((i % 37) as f32 * 0.021).sin()).collect();
            let gamma: Vec<f32> = (0..d).map(|j| 1.0 + 0.1 * ((j % 13) as f32 * 0.3).cos()).collect();
            let beta: Vec<f32> = (0..d).map(|j| 0.05 * ((j % 11) as f32 * 0.2).sin()).collect();
            let w1: Vec<f32> = (0..d * hid).map(|i| ((i % 53) as f32 * 0.017).sin() * 0.3).collect();
            let b1: Vec<f32> = (0..hid).map(|j| 0.02 * ((j % 7) as f32 * 0.5).cos()).collect();
            let w2: Vec<f32> = (0..d * hid).map(|i| ((i % 47) as f32 * 0.019).cos() * 0.2).collect();
            let b2: Vec<f32> = (0..d).map(|j| 0.01 * ((j % 5) as f32 * 0.7).sin()).collect();
            let dout: Vec<f32> = (0..rows * d).map(|i| ((i % 29) as f32 * 0.023).sin()).collect();

            let res = mlp_forward(
                &x, &gamma, &beta, &w1, &b1, &w2, &b2, rows, d, hid, eps, is_rms, dropout, true,
            )
            .expect("该形状应走常驻显存路径");
            let seed = res.seed;

            // ---- 前向参考：LayerNorm/RMSNorm → linear₁ → GELU → linear₂ → dropout → 残差 ----
            let mut xn = vec![0.0f32; rows * d];
            for r in 0..rows {
                let base = r * d;
                let mean = if is_rms {
                    0.0
                } else {
                    x[base..base + d].iter().sum::<f32>() / d as f32
                };
                let var = x[base..base + d]
                    .iter()
                    .map(|&v| (v - mean) * (v - mean))
                    .sum::<f32>()
                    / d as f32;
                let istd = 1.0 / (var + eps).sqrt();
                for j in 0..d {
                    let aff = if is_rms { 0.0 } else { beta[j] };
                    xn[base + j] = (x[base + j] - mean) * istd * gamma[j] + aff;
                }
            }
            let y1 = cpu_matmul(&xn, &w1, rows, d, hid, 1, false, false);
            let mut z = vec![0.0f32; rows * hid];
            let mut act = vec![0.0f32; rows * hid];
            for i in 0..rows * hid {
                z[i] = y1[i] + b1[i % hid];
                act[i] = ref_gelu(z[i]);
            }
            let y2 = cpu_matmul(&act, &w2, rows, hid, d, 1, false, false);
            let out_ref: Vec<f32> = (0..rows * d)
                .map(|i| x[i] + ref_drop_scale(i, dropout, seed, true) * (y2[i] + b2[i % d]))
                .collect();
            assert_close(&format!("out({rows},{d},{hid},rms={is_rms})"), &res.out, &out_ref);

            // ---- 反向参考 ----
            let dypre: Vec<f32> = (0..rows * d)
                .map(|i| dout[i] * ref_drop_scale(i, dropout, seed, true))
                .collect();
            let db2_ref = ref_col_sum(&dypre, rows, d);
            let dact = cpu_matmul(&dypre, &w2, rows, d, hid, 1, false, true);
            let dw2_ref = cpu_matmul(&act, &dypre, hid, rows, d, 1, true, false);
            let dz: Vec<f32> = (0..rows * hid).map(|i| dact[i] * ref_gelu_grad(z[i])).collect();
            let db1_ref = ref_col_sum(&dz, rows, hid);
            let dxn = cpu_matmul(&dz, &w1, rows, hid, d, 1, false, true);
            let dw1_ref = cpu_matmul(&xn, &dz, d, rows, hid, 1, true, false);
            // 归一化反向：dx = istd·(dy·γ − m1 − m2·x̂)，m1/m2 是 dy·γ 与 dy·γ·x̂ 的行均值
            // （RMSNorm 时 m1 ≡ 0，且没有 dβ）
            let mut dxln = vec![0.0f32; rows * d];
            let mut dgamma_ref = vec![0.0f32; d];
            let mut dbeta_ref = vec![0.0f32; d];
            for r in 0..rows {
                let base = r * d;
                let mean = if is_rms {
                    0.0
                } else {
                    x[base..base + d].iter().sum::<f32>() / d as f32
                };
                let var = x[base..base + d]
                    .iter()
                    .map(|&v| (v - mean) * (v - mean))
                    .sum::<f32>()
                    / d as f32;
                let istd = 1.0 / (var + eps).sqrt();
                let (mut m1, mut m2) = (0.0f32, 0.0f32);
                for j in 0..d {
                    let xh = (x[base + j] - mean) * istd;
                    let dyg = dxn[base + j] * gamma[j];
                    m1 += dyg;
                    m2 += dyg * xh;
                    dgamma_ref[j] += dxn[base + j] * xh;
                    if !is_rms {
                        dbeta_ref[j] += dxn[base + j];
                    }
                }
                m1 = if is_rms { 0.0 } else { m1 / d as f32 };
                m2 /= d as f32;
                for j in 0..d {
                    let xh = (x[base + j] - mean) * istd;
                    dxln[base + j] = istd * (dxn[base + j] * gamma[j] - m1 - m2 * xh);
                }
            }
            // 残差直通：∂out/∂x 里还要加上 dout 本身那一份
            let dx_ref: Vec<f32> = dxln.iter().zip(&dout).map(|(a, b)| a + b).collect();

            let g = res.backward(&dout).expect("常驻反向应成功");
            let tag = |n: &str| format!("{n}({rows},{d},{hid},rms={is_rms})");
            assert_close(&tag("dx"), &g.dx, &dx_ref);
            assert_close(&tag("dw1"), &g.dw1, &dw1_ref);
            assert_close(&tag("db1"), &g.db1, &db1_ref);
            assert_close(&tag("dw2"), &g.dw2, &dw2_ref);
            assert_close(&tag("db2"), &g.db2, &db2_ref);
            assert_close(&tag("dgamma"), &g.dgamma, &dgamma_ref);
            // RMSNorm 没有 β：内核仍会写出一份「Σ dy」的残值，但**无人接收**（这里就不比）
            if !is_rms {
                assert_close(&tag("dbeta"), &g.dbeta, &dbeta_ref);
            }
        }
    }

    /// 与 WGSL `heads_split_main` / `heads_join_main` 同式的 RoPE 角度（位置取序列内下标）
    fn ref_rope_theta(r: usize, p: usize, hd: usize) -> f32 {
        let freq = 10000f32.powf((2 * p) as f32 / hd as f32);
        // 与 rope.rs 的 `build_cos_sin_tab` 同一个式子
        // （否则参考与被测项各算一套三角函数，比出来的差值里混进了实现差异）
        let _ = freq;
        r as f32 / freq
    }

    /// 纯 CPU 的注意力子层前向，逐项对应 GPU 的算子顺序：
    /// ln → QKV 投影 → 按头重排(+偏置，Q/K 加 RoPE，Q 乘 1/√hd) → S/P/O → 合并头 → c_proj → dropout → 残差。
    /// 返回（输出、以及反向需要的全部中间量）。
    #[allow(clippy::type_complexity)]
    #[allow(clippy::too_many_arguments)]
    fn ref_attn_layer_fwd(
        x: &[f32],
        gamma: &[f32],
        beta: &[f32],
        wq: &[f32],
        bq: &[f32],
        wk: &[f32],
        bk: &[f32],
        wv: &[f32],
        bv: &[f32],
        wproj: &[f32],
        bproj: &[f32],
        mask: &[f32],
        b: usize,
        t: usize,
        d: usize,
        n_head: usize,
        eps: f32,
        is_rms: bool,
        dropout: f32,
        seed: u32,
        training: bool,
    ) -> (Vec<f32>, RefAttnCache) {
        let hd = d / n_head;
        let rows = b * t;
        let bn = b * n_head;
        let scale = 1.0 / (hd as f32).sqrt();
        let half = hd / 2;

        // 归一化：LayerNorm（p3=0）或 RMSNorm（p3=1，μ≡0、无 β）
        let mut xn = vec![0.0f32; rows * d];
        for r in 0..rows {
            let base = r * d;
            let mean = if is_rms {
                0.0
            } else {
                x[base..base + d].iter().sum::<f32>() / d as f32
            };
            let var = x[base..base + d]
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>()
                / d as f32;
            let istd = 1.0 / (var + eps).sqrt();
            for j in 0..d {
                let aff = if is_rms { 0.0 } else { beta[j] };
                xn[base + j] = (x[base + j] - mean) * istd * gamma[j] + aff;
            }
        }
        // QKV 投影（含偏置）
        let proj = |w: &[f32], bias: &[f32]| -> Vec<f32> {
            let mut o = cpu_matmul(&xn, w, rows, d, d, 1, false, false);
            for r in 0..rows {
                for j in 0..d {
                    o[r * d + j] += bias[j];
                }
            }
            o
        };
        let q_pre = proj(wq, bq);
        let k_pre = proj(wk, bk);
        let v_pre = proj(wv, bv);

        // 按头重排 [+ 偏置（已加）] [+ RoPE] [+ 缩放]，输出 [bn, t, hd]
        let split = |src: &[f32], rope: bool, sc: f32| -> Vec<f32> {
            let mut o = vec![0.0f32; bn * t * hd];
            for b_idx in 0..b {
                for r in 0..t {
                    for h in 0..n_head {
                        let grp = b_idx * n_head + h;
                        for p in 0..half {
                            let (mut v0, mut v1) = (
                                src[(b_idx * t + r) * d + h * hd + 2 * p],
                                src[(b_idx * t + r) * d + h * hd + 2 * p + 1],
                            );
                            if rope {
                                let theta = ref_rope_theta(r, p, hd);
                                let (c, s) = (theta.cos(), theta.sin());
                                let (n0, n1) = (v0 * c - v1 * s, v0 * s + v1 * c);
                                v0 = n0;
                                v1 = n1;
                            }
                            o[(grp * t + r) * hd + 2 * p] = v0 * sc;
                            o[(grp * t + r) * hd + 2 * p + 1] = v1 * sc;
                        }
                    }
                }
            }
            o
        };
        let q = split(&q_pre, true, scale);
        let k = split(&k_pre, true, 1.0);
        let v = split(&v_pre, false, 1.0);

        // S = Q'·Kᵀ、P = softmax(S + mask)、O = P·V
        let s = cpu_matmul(&q, &k, t, hd, t, bn, false, true);
        let mut p = vec![0.0f32; bn * t * t];
        for r in 0..bn * t {
            let row = r % t; // 每行用的都是同一张 [t, t] 因果掩码
            let base = r * t;
            let mx = (0..t)
                .map(|j| s[base + j] + mask[row * t + j])
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..t {
                let e = (s[base + j] + mask[row * t + j] - mx).exp();
                p[base + j] = e;
                sum += e;
            }
            for j in 0..t {
                p[base + j] /= sum;
            }
        }
        let o = cpu_matmul(&p, &v, t, t, hd, bn, false, false);

        // 合并头 → [rows, d]
        let mut merged = vec![0.0f32; rows * d];
        for b_idx in 0..b {
            for r in 0..t {
                for h in 0..n_head {
                    let grp = b_idx * n_head + h;
                    for jj in 0..hd {
                        merged[(b_idx * t + r) * d + h * hd + jj] =
                            o[(grp * t + r) * hd + jj];
                    }
                }
            }
        }
        // 输出投影 → dropout → 残差
        let mut y = cpu_matmul(&merged, wproj, rows, d, d, 1, false, false);
        for r in 0..rows {
            for j in 0..d {
                y[r * d + j] += bproj[j];
            }
        }
        let out: Vec<f32> = (0..rows * d)
            .map(|i| x[i] + ref_drop_scale(i, dropout, seed, training) * y[i])
            .collect();
        (
            out,
            RefAttnCache { xn, q, k, v, p, merged },
        )
    }

    /// `ref_attn_layer_fwd` 的中间量（反向参考要用）
    struct RefAttnCache {
        xn: Vec<f32>,
        q: Vec<f32>,
        k: Vec<f32>,
        v: Vec<f32>,
        p: Vec<f32>,
        merged: Vec<f32>,
    }

    /// 常驻「注意力子层」的单个形状用例：前向 + 11 项边界梯度 vs 纯循环参考。
    ///
    /// `is_rms = true` 时走 RMSNorm 模式：参考实现按「μ≡0、无 β」算，
    /// 而喂给内核的 `beta` 仍是**非零**向量 —— 内核应把它整个丢弃，
    /// 于是这一组用例同时也验证了 β 槽位确实被忽略。
    fn attn_layer_case(b: usize, t: usize, d: usize, n_head: usize, is_rms: bool) {
        let tag = |n: &str| format!("{n}[{b},{t},{d},{n_head},rms={is_rms}]");
        let hd = d / n_head;
        let (bn, rows, half) = (b * n_head, b * t, hd / 2);
        let (dropout, eps) = (0.1f32, 1e-5f32);

        let x: Vec<f32> = (0..rows * d).map(|i| ((i % 37) as f32 * 0.021).sin()).collect();
        let gamma: Vec<f32> = (0..d).map(|j| 1.0 + 0.1 * ((j % 13) as f32 * 0.3).cos()).collect();
        let beta: Vec<f32> = (0..d).map(|j| 0.05 * ((j % 11) as f32 * 0.2).sin()).collect();
        let mk_w = |s: f32| -> Vec<f32> {
            (0..d * d).map(|i| ((i % 53) as f32 * 0.017 * s).sin() * 0.3).collect()
        };
        let mk_b = |s: f32| -> Vec<f32> {
            (0..d).map(|j| 0.02 * ((j % 7) as f32 * 0.5 * s).cos()).collect()
        };
        let (wq, bq) = (mk_w(1.0), mk_b(1.0));
        let (wk, bk) = (mk_w(1.3), mk_b(1.7));
        let (wv, bv) = (mk_w(0.7), mk_b(2.3));
        let (wproj, bproj) = (mk_w(1.1), mk_b(0.9));
        // 因果掩码：允许看 j <= r（其余位置 -1e9）
        let mut mask = vec![0.0f32; t * t];
        for r in 0..t {
            for j in 0..t {
                if j > r {
                    mask[r * t + j] = -1e9;
                }
            }
        }
        let dout: Vec<f32> = (0..rows * d).map(|i| ((i % 29) as f32 * 0.023).sin()).collect();

        let res = attn_layer_forward(&AttnLayerArgs {
            x: &x,
            gamma: &gamma,
            beta: &beta,
            wq: &wq,
            bq: &bq,
            wk: &wk,
            bk: &bk,
            wv: &wv,
            bv: &bv,
            wproj: &wproj,
            bproj: &bproj,
            mask: &mask,
            b,
            t,
            d,
            n_head,
            eps,
            is_rms,
            dropout,
            training: true,
        })
        .expect("该形状应走常驻显存路径");
        let seed = res.seed;

        let (out_ref, c) =
            ref_attn_layer_fwd(&x, &gamma, &beta, &wq, &bq, &wk, &bk, &wv, &bv, &wproj,
                               &bproj, &mask, b, t, d, n_head, eps, is_rms, dropout, seed, true);
        assert_close(&tag("attn.out"), &res.out, &out_ref);

        // ---- 反向参考 ----
        let scale = 1.0 / (hd as f32).sqrt();
        // 1) dropout 反向 + c_proj 反向
        let dpre: Vec<f32> = (0..rows * d)
            .map(|i| dout[i] * ref_drop_scale(i, dropout, seed, true))
            .collect();
        let dbproj_ref = ref_col_sum(&dpre, rows, d);
        let dmerged = cpu_matmul(&dpre, &wproj, rows, d, d, 1, false, true);
        let dwproj_ref = cpu_matmul(&c.merged, &dpre, d, rows, d, 1, true, false);
        // 2) 拆回按头布局
        let mut d_o = vec![0.0f32; bn * t * hd];
        for b_idx in 0..b {
            for r in 0..t {
                for h in 0..n_head {
                    let grp = b_idx * n_head + h;
                    for jj in 0..hd {
                        d_o[(grp * t + r) * hd + jj] = dmerged[(b_idx * t + r) * d + h * hd + jj];
                    }
                }
            }
        }
        // 3) 注意力反向
        let dv = cpu_matmul(&c.p, &d_o, t, t, hd, bn, true, false);
        let mut ds = cpu_matmul(&d_o, &c.v, t, hd, t, bn, false, true);
        for r in 0..bn * t {
            let base = r * t;
            let dot: f32 = (0..t).map(|j| ds[base + j] * c.p[base + j]).sum();
            for j in 0..t {
                ds[base + j] = c.p[base + j] * (ds[base + j] - dot);
            }
        }
        let dq = cpu_matmul(&ds, &c.k, t, t, hd, bn, false, false);
        let dk = cpu_matmul(&ds, &c.q, t, t, hd, bn, true, false);
        // 4) 合并回 [rows, d]：逆旋转 + Q 乘回 1/√hd
        let join = |src: &[f32], rope: bool, sc: f32| -> Vec<f32> {
            let mut o = vec![0.0f32; rows * d];
            for b_idx in 0..b {
                for r in 0..t {
                    for h in 0..n_head {
                        let grp = b_idx * n_head + h;
                        for p in 0..half {
                            let (mut g0, mut g1) = (
                                src[(grp * t + r) * hd + 2 * p],
                                src[(grp * t + r) * hd + 2 * p + 1],
                            );
                            if rope {
                                let theta = ref_rope_theta(r, p, hd);
                                let (cs, sn) = (theta.cos(), theta.sin());
                                let (n0, n1) = (g0 * cs + g1 * sn, -g0 * sn + g1 * cs);
                                g0 = n0;
                                g1 = n1;
                            }
                            o[(b_idx * t + r) * d + h * hd + 2 * p] = g0 * sc;
                            o[(b_idx * t + r) * d + h * hd + 2 * p + 1] = g1 * sc;
                        }
                    }
                }
            }
            o
        };
        let dq_pre = join(&dq, true, scale);
        let dk_pre = join(&dk, true, 1.0);
        let dv_pre = join(&dv, false, 1.0);
        // 5) QKV 投影反向
        let dbq_ref = ref_col_sum(&dq_pre, rows, d);
        let dbk_ref = ref_col_sum(&dk_pre, rows, d);
        let dbv_ref = ref_col_sum(&dv_pre, rows, d);
        let dwq_ref = cpu_matmul(&c.xn, &dq_pre, d, rows, d, 1, true, false);
        let dwk_ref = cpu_matmul(&c.xn, &dk_pre, d, rows, d, 1, true, false);
        let dwv_ref = cpu_matmul(&c.xn, &dv_pre, d, rows, d, 1, true, false);
        // 6) 三路汇合 → LayerNorm 反向
        let dln: Vec<f32> = (0..rows * d)
            .map(|i| dq_pre[i] + dk_pre[i] + dv_pre[i])
            .collect();
        let mut dxln = vec![0.0f32; rows * d];
        let mut dgamma_ref = vec![0.0f32; d];
        let mut dbeta_ref = vec![0.0f32; d];
        for r in 0..rows {
            let base = r * d;
            let mean = if is_rms {
                0.0
            } else {
                x[base..base + d].iter().sum::<f32>() / d as f32
            };
            let var = x[base..base + d]
                .iter()
                .map(|&v| (v - mean) * (v - mean))
                .sum::<f32>()
                / d as f32;
            let istd = 1.0 / (var + eps).sqrt();
            let (mut m1, mut m2) = (0.0f32, 0.0f32);
            for j in 0..d {
                let xh = (x[base + j] - mean) * istd;
                let dyg = dln[base + j] * gamma[j];
                m1 += dyg;
                m2 += dyg * xh;
                dgamma_ref[j] += dln[base + j] * xh;
                if !is_rms {
                    dbeta_ref[j] += dln[base + j];
                }
            }
            m1 = if is_rms { 0.0 } else { m1 / d as f32 };
            m2 /= d as f32;
            for j in 0..d {
                let xh = (x[base + j] - mean) * istd;
                dxln[base + j] = istd * (dln[base + j] * gamma[j] - m1 - m2 * xh);
            }
        }
        let dx_ref: Vec<f32> = dxln.iter().zip(&dout).map(|(a, b)| a + b).collect();

        let g = res.backward(&dout).expect("常驻反向应成功");
        assert_close(&tag("attn.dx"), &g.dx, &dx_ref);
        assert_close(&tag("attn.dwq"), &g.dwq, &dwq_ref);
        assert_close(&tag("attn.dbq"), &g.dbq, &dbq_ref);
        assert_close(&tag("attn.dwk"), &g.dwk, &dwk_ref);
        assert_close(&tag("attn.dbk"), &g.dbk, &dbk_ref);
        assert_close(&tag("attn.dwv"), &g.dwv, &dwv_ref);
        assert_close(&tag("attn.dbv"), &g.dbv, &dbv_ref);
        assert_close(&tag("attn.dwproj"), &g.dwproj, &dwproj_ref);
        assert_close(&tag("attn.dbproj"), &g.dbproj, &dbproj_ref);
        assert_close(&tag("attn.dgamma"), &g.dgamma, &dgamma_ref);
        // RMSNorm 没有 β：内核仍会写出一份「Σ dy」的残值，但**无人接收**（这里就不比）
        if !is_rms {
            assert_close(&tag("attn.dbeta"), &g.dbeta, &dbeta_ref);
        }
    }

    /// 内核算力探针：单次提交内连续跑同一个 matmul，测出**纯内核**吞吐
    /// （不含每次 dispatch 的提交/同步开销），用来判断步时是被内核卡住还是被提交开销卡住。
    ///
    /// 实测（MX150，release），**只回读 1 个元素**（= 纯内核）：
    /// | 形状 | b_t=false | b_t=true |
    /// |---|---|---|
    /// | 4096x128x512 | 230.2 | 206.9 GFLOP/s |
    /// | 4096x128x128 | 207.2 | 202.8 |
    /// | 2048x128x8192（lm_head） | 228.4 | 216.9 |
    ///
    /// 优化累积：命名标量累加器 + 标量 LDS → vec4 共享内存（LDS.128）→ 软件流水预取。
    ///
    /// **教训（曾经的错误结论）**：早先这个探针直接回读 `outs[0]`，于是 `2048x128x8192`
    /// 只测出 71~98 GFLOP/s，被误判成「小 k/宽 n 形状的内核有问题」。实际上那个形状的
    /// 输出是 2048×8192 = 67 MB，回读它就占掉 59.4ms / 119.1ms 的一半——是**测量假象**。
    /// 排除回读后各形状都在 200~230 GFLOP/s（纯 FMA 探针 523 = 峰值的 44%），
    /// 且转置读取（b_t=true，相邻 lane 地址相隔 4k 个 float、无法合并）只损失 2~10%。
    /// 结论：步时的大头**不在 matmul 内核**，而在 logits 那 134 MB 量级的显存往返与逐元素算子。
    ///
    /// 手动运行：`cargo test --release --features gpu matmul_throughput_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "仅用于性能标定，不进常规测试"]
    fn matmul_throughput_probe() {
        init();
        if !is_available() {
            return;
        }
        // 每个形状跑两种 B 布局：`b_t = true`（物理 [N,K]，k 最内层）与 `false`（物理 [K,N]）。
        // 两者的 FLOP 完全相同，唯一差别是装载阶段读 B 的访存模式：
        //   b_t=false：`b[gk*n + col]`，同一 warp 相邻 lane 取 4 个连续列 → 合并成整条 cache line；
        //   b_t=true ：`b[col*k + gk]`，相邻 lane 相隔 4k 个 float → 每个 lane 各占一个 sector。
        // 对照两者就能把「慢」归因到访存还是 tile 尺寸。
        for (m, k, n, reps) in [
            (4096usize, 128usize, 512usize, 40usize),
            (4096, 128, 128, 40),
            (2048, 128, 8192, 3),
        ] {
            let a: Vec<f32> = (0..m * k).map(|i| (i % 17) as f32 * 0.01).collect();
            let b: Vec<f32> = (0..n * k).map(|i| (i % 23) as f32 * 0.01).collect();
            // 跑两遍：`sync_small = true` 时只回读 1 个元素（纯内核时间），
            // `false` 时回读整个 outs[0]（与历史基线同一口径，但宽 n 形状会背上 67 MB 传输）。
            // 训练走的是常驻路径、根本不回读 logits，所以**纯内核**才是训练里真正付的代价。
            for (b_t, sync_small) in [(true, false), (false, false), (true, true), (false, true)] {
                let mut rec = recorder().unwrap();
                let ha = rec.upload(&a);
                let hb = rec.upload(&b);
                let outs: Vec<GpuHandle> = (0..reps)
                    .map(|_| rec.matmul(&ha, false, &hb, b_t, m, k, n, 1))
                    .collect();
                let sync = rec.empty(1, KIND_OUT);
                let tail = if sync_small { &sync } else { &outs[0] };
                let t = std::time::Instant::now();
                let _ = rec.submit_and_read(&[tail]).unwrap();
                let s = t.elapsed().as_secs_f64();
                let flops = reps as f64 * 2.0 * m as f64 * k as f64 * n as f64;
                println!(
                    "[probe] matmul {m}x{k}x{n} x{reps} (b_t={b_t}, 只回读1元素={sync_small}): \
                     {:.1}ms -> {:.1} GFLOP/s",
                    s * 1000.0,
                    flops / 1e9 / s
                );
            }
        }

        // 训练真实形状组合：把 b=8/t=512/d=128/h=4/hid=512/vocab=8192 一步训练里
        // 每个 matmul 按真实 (a_t, b_t, batch) 各跑一遍，只回读 1 个元素（纯内核）。
        // 单看上面三个「大」形状会掩盖小 m/小 n 的形状：反向的权重梯度 m=d=128，
        // 配 128×128 的 tile 在 m 方向只有 **1 个 workgroup**，整卡 3 个 SM 只用上 1 个。
        println!("[probe] ---- 训练真实形状（纯内核，只回读 1 元素）----");
        for (name, m, k, n, bs, a_t, b_t, reps) in [
            ("QKV/c_proj fwd", 4096, 128, 128, 1, false, false, 8),
            ("QK^T       fwd", 512, 32, 512, 32, false, true, 2),
            ("PV         fwd", 512, 512, 32, 32, false, false, 2),
            ("MLP w1     fwd", 4096, 128, 512, 1, false, false, 2),
            ("MLP w2     fwd", 4096, 512, 128, 1, false, false, 2),
            ("lm_head    fwd", 4096, 128, 8192, 1, false, true, 1),
            ("dW proj    bwd", 128, 4096, 128, 1, true, false, 8),
            ("dX proj    bwd", 4096, 128, 128, 1, false, true, 8),
            ("dV         bwd", 512, 512, 32, 32, true, false, 2),
            ("dP         bwd", 512, 32, 512, 32, false, true, 2),
            ("dK         bwd", 512, 512, 32, 32, true, false, 2),
            ("dW1        bwd", 128, 4096, 512, 1, true, false, 2),
            ("dW2        bwd", 512, 4096, 128, 1, true, false, 2),
            ("lm_head dW bwd", 128, 4096, 8192, 1, true, false, 1),
            ("lm_head dX bwd", 4096, 8192, 128, 1, false, true, 1),
        ] {
            let (ar, ac) = if a_t { (k, m) } else { (m, k) };
            let (br, bc) = if b_t { (n, k) } else { (k, n) };
            let a = vec![0.5f32; bs * ar * ac];
            let b = vec![0.25f32; bs * br * bc];
            let mut rec = recorder().unwrap();
            let ha = rec.upload(&a);
            let hb = rec.upload(&b);
            let outs: Vec<GpuHandle> = (0..reps)
                .map(|_| rec.matmul(&ha, a_t, &hb, b_t, m, k, n, bs))
                .collect();
            let sync = rec.empty(1, KIND_OUT);
            let t = std::time::Instant::now();
            let _ = rec.submit_and_read(&[&sync]).unwrap();
            let s = t.elapsed().as_secs_f64();
            let flops = reps as f64 * 2.0 * m as f64 * k as f64 * n as f64 * bs as f64;
            println!(
                "[probe] {name} {m}x{k}x{n} b={bs} a_t={a_t} b_t={b_t} x{reps}: {:.1}ms -> {:.1} GFLOP/s",
                s * 1000.0,
                flops / 1e9 / s
            );
            drop(outs);
        }
    }

    /// 大 / 小 tile 的**同进程交替 A/B**，小 tile 之所以成为默认就是靠这张表定的。
    ///
    /// 为什么必须同进程交替：MX150 连续满载会降频，同一配置两次独立运行能差 15%（实测），
    /// 温差带来的偏差足以盖过 tile 本身的差异。这里对每个形状按「大 → 小」交替各测 3 轮，
    /// 两侧落在同一热状态上，比值才可信。两侧的 dispatch 数、绑定、uniform 完全一致，
    /// 唯一变量就是 tile（以及随之而来的 workgroup 数）。
    ///
    /// 手动运行：`cargo test --release --features gpu mm_tile_ab_probe -- --ignored --nocapture`
    #[test]
    #[ignore = "仅用于性能标定，不进常规测试"]
    fn mm_tile_ab_probe() {
        init();
        let Some(g) = GPU.get().and_then(|g| g.as_ref()) else {
            return;
        };
        // 大 tile 相对「理想」多算的倍数（m、n 各自向上取整到整数个 tile）
        let pad = |x: usize| (x.div_ceil(MM_TILE) * MM_TILE) as f64 / x as f64;

        // 挑有代表性的形状：启发式要改的几个 + 明确不该改的对照组。
        // 输出 buffer 轮转 2 块，连 logits（134MB）也塞得进 2GB 显存。
        for (name, m, k, n, bs, a_t, b_t, reps) in [
            ("PV         fwd", 512usize, 512usize, 32usize, 32usize, false, false, 4usize),
            ("dV         bwd", 512, 512, 32, 32, true, false, 4),
            ("QK^T       fwd", 512, 32, 512, 32, false, true, 4),
            ("QKV/c_proj fwd", 4096, 128, 128, 1, false, false, 8),
            ("dW proj    bwd", 128, 4096, 128, 1, true, false, 8),
            ("MLP w2     fwd", 4096, 512, 128, 1, false, false, 4),
            ("dX proj    bwd", 4096, 128, 128, 1, false, true, 8),
            ("lm_head    fwd", 4096, 128, 8192, 1, false, true, 1),
        ] {
            let (ar, ac) = if a_t { (k, m) } else { (m, k) };
            let (br, bc) = if b_t { (n, k) } else { (k, n) };
            let a = vec![0.5f32; bs * ar * ac];
            let b = vec![0.25f32; bs * br * bc];
            let buf_a = g.take_buf(a.len(), KIND_IN);
            g.queue.write_buffer(&buf_a, 0, bytemuck_bytes(&a));
            let buf_b = g.take_buf(b.len(), KIND_IN);
            g.queue.write_buffer(&buf_b, 0, bytemuck_bytes(&b));
            let outs: Vec<wgpu::Buffer> =
                (0..2).map(|_| g.take_buf(bs * m * n, KIND_OUT)).collect();

            // 预热两侧各一遍：把「新建 buffer 的懒零填充」与首次绑定排干，别混进计时窗口
            for force in [0i8, 1] {
                MM_SMALL_FORCE.store(force, Ordering::Relaxed);
                let mut warm = g.batch_begin();
                for o in &outs {
                    g.batch_matmul(&mut warm, &buf_a, &buf_b, o, m, k, n, bs, a_t, b_t);
                }
                g.batch_finish(warm);
            }

            const ROUNDS: usize = 3;
            let mut ms = [0.0f64; 2]; // [大 tile, 小 tile]
            for _ in 0..ROUNDS {
                for (slot, force) in [(0usize, 0i8), (1, 1)] {
                    MM_SMALL_FORCE.store(force, Ordering::Relaxed);
                    let t = std::time::Instant::now();
                    let mut bt = g.batch_begin();
                    for j in 0..reps {
                        g.batch_matmul(&mut bt, &buf_a, &buf_b, &outs[j % 2], m, k, n, bs, a_t, b_t);
                    }
                    g.batch_finish(bt);
                    ms[slot] += t.elapsed().as_secs_f64() / ROUNDS as f64;
                }
            }

            let flops = reps as f64 * 2.0 * m as f64 * k as f64 * n as f64 * bs as f64;
            println!(
                "[ab] {name:<16} {m}x{k}x{n} b={bs:<3} | 大 {:>7.2}ms {:>6.0} GF/s | \
                 小 {:>7.2}ms {:>6.0} GF/s | 小/大 = {:.2} | 大 tile 多算 {:.2}x",
                ms[0] * 1000.0,
                flops / 1e9 / ms[0],
                ms[1] * 1000.0,
                flops / 1e9 / ms[1],
                ms[1] / ms[0],
                pad(m) * pad(n),
            );
            g.put_buf(buf_a);
            g.put_buf(buf_b);
            for o in outs {
                g.put_buf(o);
            }
        }
        MM_SMALL_FORCE.store(-1, Ordering::Relaxed);
    }

    /// 大 / 小 tile 两个内核的**逐位一致**校验。
    ///
    /// 两个内核的每个输出元素都按 k = 0, 1, 2, … 的顺序累加，所以必须 bit-exact；
    /// 一旦不一致，说明小 tile 的装载/写回下标或夹边逻辑写错了。数据用随机数而不是常数 ——
    /// 常数（如全 0.5 × 0.25）每一步都精确无舍入，即使求和顺序不同也会碰巧相等，测不出问题。
    /// 形状刻意见跨 tile 边界（130 / 35 / 70 / 129 / 33），把部分 tile 与越界夹边都覆盖到。
    #[test]
    fn gpu_matmul_small_tile_matches_big_tile_bits() {
        init();
        if !is_available() {
            return;
        }
        MATMUL_MIN_FLOPS.store(0, Ordering::Relaxed);
        let Some(g) = GPU.get().and_then(|g| g.as_ref()) else {
            return;
        };
        let mut seed = 0x1234_5678u32;
        let mut next = move || {
            seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            (seed >> 8) as f32 / 8_388_608.0 - 1.0 // [-1, 1)
        };
        for (m, k, n, bs) in [
            (130usize, 35usize, 70usize, 2usize), // 三个维度都跨 tile 边界
            (256, 64, 32, 8),                     // n = 32：正是小 tile 要救的那类形状
            (64, 8, 64, 1),                       // 正好一个 64×64 tile
            (129, 8, 33, 1),                      // 刚过一个 tile
        ] {
            for (a_t, b_t) in [(false, false), (true, false), (false, true), (true, true)] {
                let (ar, ac) = if a_t { (k, m) } else { (m, k) };
                let (br, bc) = if b_t { (n, k) } else { (k, n) };
                let a: Vec<f32> = (0..bs * ar * ac).map(|_| next()).collect();
                let b: Vec<f32> = (0..bs * br * bc).map(|_| next()).collect();
                MM_SMALL_FORCE.store(0, Ordering::Relaxed);
                let rb = g.matmul(&a, &b, m, k, n, bs, a_t, b_t).expect("大 tile 路径");
                MM_SMALL_FORCE.store(1, Ordering::Relaxed);
                let rs = g.matmul(&a, &b, m, k, n, bs, a_t, b_t).expect("小 tile 路径");
                MM_SMALL_FORCE.store(-1, Ordering::Relaxed);
                assert_eq!(rb.len(), rs.len(), "输出长度不一致 m={m} k={k} n={n}");
                for (i, (x, y)) in rb.iter().zip(rs.iter()).enumerate() {
                    assert_eq!(
                        x.to_bits(),
                        y.to_bits(),
                        "m={m} k={k} n={n} b={bs} a_t={a_t} b_t={b_t} 第 {i} 个元素不一致：{x} vs {y}"
                    );
                }
            }
        }
    }

    /// 常驻「注意力子层」的前向与 11 项边界梯度 vs 纯循环参考。
    ///
    /// 第二组形状让 T 跨过 softmax 的 workgroup_size(256) 多轮归约边界
    /// （训练用的 T=512 正是这种情形），head_dim 也与训练配置一致。
    #[test]
    fn gpu_attn_layer_resident_matches_loop_reference() {
        init();
        if !is_available() {
            return;
        }
        MATMUL_MIN_FLOPS.store(0, Ordering::Relaxed);
        attn_layer_case(4, 128, 32, 4, false);
        attn_layer_case(2, 300, 128, 4, false);
        // RMSNorm 走同一套内核、只切模式位：形状换一组，避免与上面共用同一份随机数
        attn_layer_case(4, 128, 32, 4, true);
        attn_layer_case(2, 300, 128, 4, true);
    }

    /// 整叠 Block 常驻路径（打通子层边界）vs 逐子层常驻路径。
    ///
    /// 参照物就是把 `attn_layer_forward` / `mlp_forward` 串起来跑 —— 这两条路径各自
    /// 已有 vs 纯循环参考的数值测试，串联即可当作整叠路径的独立参照。
    /// 要比的是前向输出、整叠输入梯度、以及每层 16 项参数梯度，一项都不能错位
    /// （两条路径的拼接顺序不同，参数梯度的排列顺序最容易在这里出岔子）。
    ///
    /// dropout 一律关掉（`training = false`）：两条路径各自抽自己的种子，
    /// 掩码不可能逐位一致，开着就没法比对。
    ///
    /// `is_rms = true` 时归一化切到 RMSNorm 模式：两条路径共用同一套内核，
    /// 因此 16 个槽位仍可逐一比对（含 RMSNorm 下无意义的 `dβ` 残值——两边算的一样）。
    fn stack_case(n_layer: usize, b: usize, t: usize, d: usize, n_head: usize, is_rms: bool) {
        let tag = |n: &str| format!("{n}[L{n_layer},{b},{t},{d},{n_head},rms={is_rms}]");
        let (rows, eps, hid) = (b * t, 1e-5f32, 4 * d);
        // 每层 16 个参数，顺序与 `StackLayerArgs` 一致
        let layers_w: Vec<Vec<Vec<f32>>> = (0..n_layer)
            .map(|li| {
                let o = li as f32 * 0.37;
                let mk_w = |s: f32| -> Vec<f32> {
                    (0..d * d).map(|i| ((i % 53) as f32 * 0.017 * s + o).sin() * 0.3).collect()
                };
                let mk_b = |s: f32| -> Vec<f32> {
                    (0..d).map(|j| 0.02 * ((j % 7) as f32 * 0.5 * s + o).cos()).collect()
                };
                let w1: Vec<f32> =
                    (0..d * hid).map(|i| ((i % 41) as f32 * 0.019 + o).sin() * 0.25).collect();
                let b1: Vec<f32> = (0..hid).map(|j| 0.02 * ((j % 5) as f32 + o).cos()).collect();
                let w2: Vec<f32> =
                    (0..hid * d).map(|i| ((i % 47) as f32 * 0.013 + o).sin() * 0.25).collect();
                let b2: Vec<f32> = (0..d).map(|j| 0.02 * ((j % 3) as f32 + o).cos()).collect();
                let g1: Vec<f32> =
                    (0..d).map(|j| 1.0 + 0.1 * ((j % 13) as f32 * 0.3 + o).cos()).collect();
                let g2: Vec<f32> =
                    (0..d).map(|j| 1.0 + 0.1 * ((j % 11) as f32 * 0.2 + o).cos()).collect();
                vec![
                    g1, mk_b(0.5), mk_w(1.0), mk_b(1.0), mk_w(1.3), mk_b(1.7), mk_w(0.7),
                    mk_b(2.3), mk_w(1.1), mk_b(0.9), g2, mk_b(0.6), w1, b1, w2, b2,
                ]
            })
            .collect();
        // 因果掩码：允许看 j <= r
        let mut mask = vec![0.0f32; t * t];
        for r in 0..t {
            for j in 0..t {
                if j > r {
                    mask[r * t + j] = -1e9;
                }
            }
        }
        let x: Vec<f32> = (0..rows * d).map(|i| ((i % 37) as f32 * 0.021).sin()).collect();
        let dout: Vec<f32> = (0..rows * d).map(|i| ((i % 29) as f32 * 0.023).sin()).collect();

        // ---- 参考：逐子层常驻路径串起来 ----
        let mut xr = x.clone();
        let mut attn_ref = Vec::with_capacity(n_layer);
        let mut mlp_ref = Vec::with_capacity(n_layer);
        for w in &layers_w {
            let a = attn_layer_forward(&AttnLayerArgs {
                x: &xr,
                gamma: &w[0],
                beta: &w[1],
                wq: &w[2],
                bq: &w[3],
                wk: &w[4],
                bk: &w[5],
                wv: &w[6],
                bv: &w[7],
                wproj: &w[8],
                bproj: &w[9],
                mask: &mask,
                b,
                t,
                d,
                n_head,
                eps,
                is_rms,
                dropout: 0.0,
                training: false,
            })
            .expect("参考路径：注意力子层应走常驻显存");
            let xa = a.out.clone();
            let m = mlp_forward(
                &xa, &w[10], &w[11], &w[12], &w[13], &w[14], &w[15], rows, d, hid, eps, is_rms, 0.0,
                false,
            )
            .expect("参考路径：MLP 子层应走常驻显存");
            xr = m.out.clone();
            attn_ref.push(a);
            mlp_ref.push(m);
        }
        let mut dcur = dout.clone();
        let mut want: Vec<Vec<Vec<f32>>> = vec![Vec::new(); n_layer];
        for i in (0..n_layer).rev() {
            let gm = mlp_ref[i].backward(&dcur).expect("参考 MLP 反向");
            let ga = attn_ref[i].backward(&gm.dx).expect("参考注意力反向");
            want[i] = vec![
                ga.dgamma, ga.dbeta, ga.dwq, ga.dbq, ga.dwk, ga.dbk, ga.dwv, ga.dbv, ga.dwproj,
                ga.dbproj, gm.dgamma, gm.dbeta, gm.dw1, gm.db1, gm.dw2, gm.db2,
            ];
            dcur = ga.dx;
        }

        // ---- 整叠常驻路径 ----
        let la: Vec<StackLayerArgs> = (0..n_layer)
            .map(|i| {
                let w = &layers_w[i];
                StackLayerArgs {
                    gamma1: &w[0],
                    beta1: &w[1],
                    wq: &w[2],
                    bq: &w[3],
                    wk: &w[4],
                    bk: &w[5],
                    wv: &w[6],
                    bv: &w[7],
                    wproj: &w[8],
                    bproj: &w[9],
                    gamma2: &w[10],
                    beta2: &w[11],
                    w1: &w[12],
                    b1: &w[13],
                    w2: &w[14],
                    b2: &w[15],
                }
            })
            .collect();
        let res = stack_forward(&StackArgs {
            x: &x,
            mask: &mask,
            layers: &la,
            b,
            t,
            d,
            n_head,
            eps,
            is_rms,
            dropout: 0.0,
            training: false,
        })
        .expect("该形状应走整叠常驻路径");
        assert_close(&tag("stack.out"), &res.out, &xr);

        let g = res.backward(&dout).expect("整叠常驻反向应成功");
        assert_close(&tag("stack.dx"), &g.dx, &dcur);
        for i in 0..n_layer {
            for k in 0..STACK_PARAMS_PER_LAYER {
                assert_close(
                    &tag(&format!("stack.L{i}.p{k}")),
                    &g.grads[i * STACK_PARAMS_PER_LAYER + k],
                    &want[i][k],
                );
            }
        }
    }

    #[test]
    fn gpu_stack_resident_matches_sublayer_reference() {
        init();
        if !is_available() {
            return;
        }
        MATMUL_MIN_FLOPS.store(0, Ordering::Relaxed);
        // 小配置；再来一个 d=128、t=300 的形状，跨过 workgroup_size(256) 的多轮归约边界
        stack_case(2, 4, 128, 32, 4, false);
        stack_case(2, 2, 300, 128, 4, false);
        // RMSNorm 模式（整叠一个开关，各层共用）
        stack_case(2, 4, 128, 32, 4, true);
        stack_case(2, 2, 300, 128, 4, true);
    }
}
