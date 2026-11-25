//! Windows 平台调用的**手写 FFI**：不引入 `windows-sys`，只声明真正用到的两个导出符号。
//!
//! 为什么值得手写：整个项目只用到「控制台切 UTF-8」和「取本地时间」两个平台调用，
//! 为它们拉一个自动生成的全量绑定 crate（几十万行）不符合"各功能板块自研"的取向。
//! 这里每个符号都带来源注释，签名与 `Win32` 头文件一一对应，读起来比查 crate 文档更直接。

#![cfg(windows)]

use std::ffi::c_void;

// `#[link(name = "kernel32")]` 等价于在链接行加 `kernel32.lib`：Windows 的默认链接库，
// 显式写出来是为了让"这个符号从哪来"一目了然。
#[link(name = "kernel32")]
// SAFETY: 这些签名与 Win32 头文件一一对应（见上方注释）；此处只声明、不调用，
// 符号由 kernel32 在链接期提供。
unsafe extern "system" {
    /// `BOOL SetConsoleOutputCP(UINT wCodePageID)`：设置**输出**代码页。
    /// 返回非零表示成功。我们只关心效果，不检查返回值（失败时只是乱码，不影响正确性）。
    fn SetConsoleOutputCP(code_page: u32) -> i32;

    /// `BOOL SetConsoleCP(UINT wCodePageID)`：设置**输入**代码页（读取用户键入时用）。
    fn SetConsoleCP(code_page: u32) -> i32;

    /// `void GetLocalTime(LPSYSTEMTIME lpSystemTime)`：取**本地**时间（已含时区与夏令时）。
    /// 标准库只有 UTC，这正是必须问系统的原因。
    fn GetLocalTime(out: *mut SystemTime);

    /// `HANDLE GetStdHandle(DWORD nStdHandle)`：判断标准输出是否被重定向。
    fn GetStdHandle(std_handle: u32) -> *mut c_void;

    /// `DWORD GetConsoleMode(HANDLE hConsoleHandle, LPDWORD lpMode)`
    fn GetConsoleMode(handle: *mut c_void, mode: *mut u32) -> i32;
}

/// `SYSTEMTIME` 的精确布局（`Win32_Foundation`）：8 个 `WORD`，两个字节一个，无填充。
#[repr(C)]
#[derive(Default, Clone, Copy)]
struct SystemTime {
    year: u16,
    month: u16,
    day_of_week: u16,
    day: u16,
    hour: u16,
    minute: u16,
    second: u16,
    milliseconds: u16,
}

/// `STD_OUTPUT_HANDLE`
const STD_OUTPUT_HANDLE: u32 = u32::MAX - 10; // 0xFFFF_FFF5，即 -11 的补码

/// 65001 = `CP_UTF8`
const CP_UTF8: u32 = 65001;

/// 把控制台输入/输出代码页切到 UTF-8。
///
/// 只在**确实连着控制台**时才切：输出被重定向到文件/管道时（CI、`> log.txt`），
/// 没有控制台可切，调用会失败且毫无意义；这里先探一次，避免多余的失败调用。
pub fn init_console_utf8() {
    // SAFETY: 只把有效的标准输出句柄与栈上 `u32` 指针交给系统调用；`GetConsoleMode`
    // 失败时立即返回，不会读取未初始化数据。
    unsafe {
        let handle = GetStdHandle(STD_OUTPUT_HANDLE);
        let mut mode: u32 = 0;
        if handle.is_null() || GetConsoleMode(handle, &mut mode) == 0 {
            return; // 没有控制台（被重定向），保持原样
        }
        SetConsoleOutputCP(CP_UTF8);
        SetConsoleCP(CP_UTF8);
    }
}

/// 取本地时间：`(年, 月, 日, 时, 分, 秒, 毫秒)`
pub fn local_time() -> (u16, u16, u16, u16, u16, u16, u16) {
    let mut st = SystemTime::default();
    // SAFETY: 传入 `&mut st`（栈上已按 `Default` 初始化为 0），`GetLocalTime` 只会写入
    // 这个 16 字节的 `SYSTEMTIME` 结构体。
    unsafe { GetLocalTime(&mut st) };
    (
        st.year,
        st.month,
        st.day,
        st.hour,
        st.minute,
        st.second,
        st.milliseconds,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `SYSTEMTIME` 必须正好 16 字节：布局错了会读到垃圾字段（年份变成 40000 之类）。
    #[test]
    fn system_time_layout_is_16_bytes() {
        assert_eq!(std::mem::size_of::<SystemTime>(), 16);
        assert_eq!(std::mem::align_of::<SystemTime>(), 2);
    }

    /// 取到的时间必须落在合理区间（2000..2100 年、月份 1..=12、时 0..=23）。
    /// 这条测试同时验证 FFI 调用没有把结构体写坏。
    #[test]
    fn local_time_returns_plausible_values() {
        let (y, mo, d, h, mi, s, ms) = local_time();
        assert!((2000..2100).contains(&y), "年份异常：{y}");
        assert!((1..=12).contains(&mo), "月份异常：{mo}");
        assert!((1..=31).contains(&d), "日异常：{d}");
        assert!(h <= 23 && mi <= 59 && s <= 60 && ms <= 999);
    }
}
