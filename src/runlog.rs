//! 运行日志：每次训练 / 推理自动在 `logs/` 下生成一份完整运行记录
//!
//! 文件名形如 `logs/{操作}_{年-月-日_时-分-秒-毫秒}.log`，例如
//! `logs/generate_2026-09-19_14-30-12-345.log`：**操作名做前缀**区分不同操作，
//! **时间戳精确到毫秒**区分同一操作的多次运行，两次运行不会互相覆盖。
//! （`logs/` 与 `*.log` 已在 `.gitignore` 中，不会污染仓库。）
//!
//! 一份日志由四段组成：
//! 1. **运行头部**：操作名、命令开始执行的时间、完整命令行、工作目录、版本 / 平台 / 线程数 / GPU
//! 2. **完整配置**：`config/config.json` 解析后的全部字段（模型超参数 + 训练参数）
//! 3. **运行主体**：该次操作的关键参数与过程输出（训练进度、评估点、生成文本等）
//! 4. **运行尾部**：结束时间与总耗时
//!
//! 过程输出用 [`logln!`] 宏写：它同时打控制台和日志，因此日志是"控制台看到的一切 + 头部与配置"。
//! 需要"走 stderr 而不污染 stdout"的告警（如 stdout 是生成结果、可能被重定向到文件时），
//! 用 `eprintln!` + [`append`] 手动组合：既不污染 stdout，又能在日志里留痕。
//! 未调用 [`start`] 的命令（`demo` / `bench` / `preset`）不会产生日志文件，此时所有写入
//! 接口自动退化为空操作。
//!
//! 时间来源：Windows 走 `GetLocalTime`（系统本地时间）；其它平台标准库没有本地时区 API，
//! 退化为 UTC，并在日志头部标注。

use std::io::Write;
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

/// 运行日志目录（与 `config::DEFAULT_LOG_FILE` 同目录，见 `.gitignore` 的目录约定）
pub const LOG_DIR: &str = "logs";

/// 全局运行日志句柄：一个进程只执行一个子命令，单例足够
static RUN_LOG: OnceLock<Mutex<Option<RunLog>>> = OnceLock::new();

/// 命令开始执行的时刻（`main` 入口由 [`mark_start`] 记录）
///
/// 头部的「开始时间」、文件名里的时间戳、尾部的「总耗时」都以它为基准，
/// 而不是"日志文件创建的时刻"——否则会漏掉参数解析、GPU 初始化等开销。
static START: OnceLock<(Instant, DateTime)> = OnceLock::new();

struct RunLog {
    path: String,
    file: std::fs::File,
    started: Instant,
}

/// 记录命令开始执行的时刻。应在 `main` 的第一行调用（早于参数解析与 GPU 初始化）。
pub fn mark_start() {
    let _ = START.set((Instant::now(), local_now()));
}

/// 取命令开始执行的时刻；若 [`mark_start`] 未被调用则退化为当前时刻
fn start_point() -> (Instant, DateTime) {
    *START.get_or_init(|| (Instant::now(), local_now()))
}

/// 在内存中暂存、最后一次性取出，避免持锁做 I/O 之外的琐事
fn with_log(f: impl FnOnce(&mut RunLog)) {
    if let Some(cell) = RUN_LOG.get() {
        if let Ok(mut guard) = cell.lock() {
            if let Some(log) = guard.as_mut() {
                f(log);
            }
        }
    }
}

/// 创建本次运行的日志文件并写入头部，返回日志路径。
///
/// `op` 为操作名（`train` / `finetune` / `eval` / `generate` / `chat`），用作文件名前缀。
/// 文件名时间戳与头部「开始时间」都取命令开始执行的时刻（见 [`mark_start`]）。
/// 重复调用只保留第一次创建的日志文件。
pub fn start(op: &str) -> String {
    let (started, begin) = start_point();
    let path = format!("{LOG_DIR}/{op}_{}.log", fmt_timestamp(begin));
    crate::config::ensure_parent_dir(&path);
    let file = std::fs::File::create(&path)
        .unwrap_or_else(|e| panic!("无法创建运行日志 {path}: {e}"));
    let log = RunLog { path: path.clone(), file, started };
    let cell = RUN_LOG.get_or_init(|| Mutex::new(None));
    if let Ok(mut guard) = cell.lock() {
        *guard = Some(log);
    }
    write_header(op, begin);
    path
}

/// 追加一行（自动补换行）；未启用日志时为空操作
pub fn append(line: &str) {
    with_log(|log| {
        let _ = writeln!(log.file, "{line}");
    });
}

/// 写一个分节标题（前置空行，方便人工翻阅）
pub fn section(title: &str) {
    with_log(|log| {
        let _ = writeln!(log.file, "\n--- {title} ---");
    });
}

/// 写一节对齐的「键 : 值」列表，用于记录该次操作的关键参数
///
/// 对齐按**显示宽度**算（中日韩字符在等宽字体里占两列），否则中文键名会参差不齐。
pub fn fields(title: &str, items: &[(&str, String)]) {
    let width = items.iter().map(|(k, _)| display_width(k)).max().unwrap_or(0);
    with_log(|log| {
        let _ = writeln!(log.file, "\n--- {title} ---");
        for (k, v) in items {
            let pad = " ".repeat(width.saturating_sub(display_width(k)));
            let _ = writeln!(log.file, "{k}{pad} : {v}");
        }
    });
}

/// 字符串在等宽终端里的显示宽度：东亚宽字符算 2 列，其余算 1 列
fn display_width(s: &str) -> usize {
    s.chars()
        .map(|c| {
            let u = c as u32;
            let wide = matches!(u,
                0x1100..=0x115F | 0x2E80..=0xA4CF | 0xAC00..=0xD7A3 | 0xF900..=0xFAFF
                | 0xFE30..=0xFE6F | 0xFF00..=0xFF60 | 0xFFE0..=0xFFE6
                | 0x20000..=0x3FFFD);
            if wide { 2 } else { 1 }
        })
        .sum()
}

/// 写一节 JSON（完整配置等结构化信息，便于直接 diff 两次实验）
pub fn json<T: serde::Serialize>(title: &str, value: &T) {
    let text = serde_json::to_string_pretty(value)
        .unwrap_or_else(|e| format!("<序列化失败: {e}>"));
    section(title);
    append(&text);
}

/// 写入尾部（结束时间与总耗时）并关闭日志文件
///
/// 总耗时 = 命令开始执行的时刻 → 此刻（见 [`mark_start`]）。
pub fn finish() {
    with_log(|log| {
        let _ = writeln!(log.file, "\n--- 运行结束 ---");
        let _ = writeln!(log.file, "结束时间   : {}", fmt_time(local_now()));
        let _ = writeln!(log.file, "总耗时     : {:.3}s", log.started.elapsed().as_secs_f64());
        let _ = writeln!(log.file, "日志文件   : {}", log.path);
        let _ = log.file.flush();
    });
    if let Some(cell) = RUN_LOG.get() {
        if let Ok(mut guard) = cell.lock() {
            *guard = None;
        }
    }
}

// ==================== 运行头部 ====================

fn write_header(op: &str, begin: DateTime) {
    let (full, cargo) = command_line();
    let cwd = std::env::current_dir()
        .map(|p| p.display().to_string())
        .unwrap_or_else(|_| "<未知>".to_string());
    let threads = std::thread::available_parallelism()
        .map(|n| n.get().to_string())
        .unwrap_or_else(|_| "<未知>".to_string());
    with_log(|log| {
        let _ = writeln!(log.file, "==================== 运行日志 ====================");
        let _ = writeln!(log.file, "操作       : {op}");
        let _ = writeln!(log.file, "开始时间   : {}（命令开始执行）", fmt_time(begin));
        let _ = writeln!(log.file, "完整命令   : {full}");
        let _ = writeln!(log.file, "cargo 命令 : {cargo}");
        let _ = writeln!(log.file, "工作目录   : {cwd}");
        let _ = writeln!(
            log.file,
            "程序版本   : {}（{}，{} {}）",
            env!("CARGO_PKG_VERSION"),
            if cfg!(debug_assertions) { "debug" } else { "release" },
            std::env::consts::OS,
            std::env::consts::ARCH,
        );
        let _ = writeln!(
            log.file,
            "CPU / 线程 : {} 逻辑核，rayon {} 线程",
            threads,
            rayon::current_num_threads()
        );
        let _ = writeln!(log.file, "GPU        : {}", gpu_desc());
        let _ = writeln!(log.file, "==================================================");
    });
}

/// 完整命令行：可执行文件 + 全部参数（含空格的参数加引号）
fn command_line() -> (String, String) {
    let mut args = std::env::args();
    let exe = args.next().unwrap_or_else(|| env!("CARGO_PKG_NAME").to_string());
    let rest: Vec<String> = args.collect();
    let quoted = |xs: &[String]| {
        xs.iter()
            .map(|a| if a.is_empty() || a.contains(' ') { format!("\"{a}\"") } else { a.clone() })
            .collect::<Vec<_>>()
            .join(" ")
    };
    let full = format!("{} {}", quote(&exe), quoted(&rest)).trim_end().to_string();
    // 用户平时敲的就是 cargo 命令，单独记一条，便于直接复制复现
    let cargo = format!("cargo run --release -- {}", quoted(&rest)).trim_end().to_string();
    (full, cargo)
}

fn quote(arg: &str) -> String {
    if arg.is_empty() || arg.contains(' ') {
        format!("\"{arg}\"")
    } else {
        arg.to_string()
    }
}

#[cfg(feature = "gpu")]
fn gpu_desc() -> String {
    if crate::gpu::is_available() {
        format!("{}（{}）", crate::gpu::name(), crate::gpu::backend())
    } else {
        "未检测到可用 GPU（本次走 CPU）".to_string()
    }
}

#[cfg(not(feature = "gpu"))]
fn gpu_desc() -> String {
    "未启用 gpu feature（本次走 CPU）".to_string()
}

// ==================== 本地时间 ====================

/// 粗粒度日期时间：年 月 日 时 分 秒 毫秒
type DateTime = (u16, u16, u16, u16, u16, u16, u16);

/// 日志文件名用的时间戳：`2026-09-19_14-30-12-345`
/// （Windows 文件名不允许 `:`，所以时分秒之间用 `-`）
fn fmt_timestamp((y, mo, d, h, mi, s, ms): DateTime) -> String {
    format!("{y:04}-{mo:02}-{d:02}_{h:02}-{mi:02}-{s:02}-{ms:03}")
}

/// 日志正文用的可读时间：`2026-09-19 14:30:12.345`
fn fmt_time((y, mo, d, h, mi, s, ms): DateTime) -> String {
    format!("{y:04}-{mo:02}-{d:02} {h:02}:{mi:02}:{s:02}.{ms:03}")
}

/// 取当前时间。Windows 直接问系统要本地时间；其它平台标准库没有时区 API，退化为 UTC。
#[cfg(windows)]
fn local_now() -> DateTime {
    use windows_sys::Win32::Foundation::SYSTEMTIME;
    use windows_sys::Win32::System::SystemInformation::GetLocalTime;
    let mut st: SYSTEMTIME = unsafe { std::mem::zeroed() };
    unsafe { GetLocalTime(&mut st) };
    (
        st.wYear,
        st.wMonth,
        st.wDay,
        st.wHour,
        st.wMinute,
        st.wSecond,
        st.wMilliseconds,
    )
}

#[cfg(not(windows))]
fn local_now() -> DateTime {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| (d.as_secs() as i64, d.subsec_millis() as u16))
        .unwrap_or((0, 0));
    let (y, mo, d) = civil_from_days(secs.0.div_euclid(86_400));
    let tod = secs.0.rem_euclid(86_400);
    (
        y as u16,
        mo as u16,
        d as u16,
        (tod / 3600) as u16,
        ((tod % 3600) / 60) as u16,
        (tod % 60) as u16,
        secs.1,
    )
}

/// 把"距 1970-01-01 的天数"换成公历年月日（Howard Hinnant 的 civil_from_days 算法）
#[cfg(not(windows))]
fn civil_from_days(days: i64) -> (i64, u32, u32) {
    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

/// 同时输出到控制台与运行日志（日志未启用时等价于 `println!`）
#[macro_export]
macro_rules! logln {
    () => {{
        println!();
        $crate::runlog::append("");
    }};
    ($($arg:tt)*) => {{
        let line = format!($($arg)*);
        println!("{line}");
        $crate::runlog::append(&line);
    }};
}
