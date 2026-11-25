//! `unsafe` 策略：非测试代码里每处 `unsafe` 都必须在**前 4 行内**有 `SAFETY:` 说明。
//!
//! 为什么值得一条测试：`unsafe` 的安全性论证只能靠注释承载，而注释最容易在重构中丢失。
//! 把它变成断言后，新增 `unsafe` 忘写理由会直接失败；已有 22 处已全部补齐。
//!
//! 扫描规则：
//! - 只看 `#[cfg(test)]` 之前的代码（测试里的 `unsafe` 由测试自身负责）；
//! - 跳过整行注释（避免把文档里提到的 "`unsafe fn`" 当成代码）；
//! - 匹配 `unsafe {`、`unsafe fn`、`unsafe impl`、`unsafe extern`。

use std::fs;
use std::path::{Path, PathBuf};

fn workspace_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn rs_files(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            rs_files(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// 该行是否是 `unsafe` 的**代码**出现（排除整行注释）
fn is_unsafe_code(line: &str) -> bool {
    let t = line.trim_start();
    if t.starts_with("//") {
        return false;
    }
    let bytes = line.as_bytes();
    let mut from = 0;
    while let Some(pos) = line[from..].find("unsafe") {
        let start = from + pos;
        // 要求前面是词边界
        let prev_ok =
            start == 0 || !bytes[start - 1].is_ascii_alphanumeric() && bytes[start - 1] != b'_';
        if prev_ok {
            let rest = line[start + "unsafe".len()..].trim_start();
            if rest.starts_with('{')
                || rest.starts_with("fn ")
                || rest.starts_with("impl")
                || rest.starts_with("extern")
            {
                return true;
            }
        }
        from = start + "unsafe".len();
    }
    false
}

#[test]
fn every_unsafe_has_safety_comment() {
    let mut files = Vec::new();
    rs_files(&workspace_root().join("src"), &mut files);
    assert!(!files.is_empty(), "没扫到源码");

    let mut offenders = Vec::new();
    let mut total = 0usize;
    for path in &files {
        let text = fs::read_to_string(path).expect("读源码");
        let lines: Vec<&str> = text.lines().collect();
        // 只看非测试区（遇到第一个 #[cfg(test)] 就停）
        let cut = lines
            .iter()
            .position(|l| l.trim() == "#[cfg(test)]")
            .unwrap_or(lines.len());
        for i in 0..cut {
            if !is_unsafe_code(lines[i]) {
                continue;
            }
            total += 1;
            let window = &lines[i.saturating_sub(4)..=i];
            if !window
                .iter()
                .any(|l| l.contains("SAFETY") || l.contains("安全"))
            {
                offenders.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }

    assert!(total > 0, "一处 unsafe 都没扫到，扫描逻辑可能失效");
    assert!(
        offenders.is_empty(),
        "以下 unsafe 缺 SAFETY 注释（请补一段说明：为什么这样用是安全的）：\n  {}",
        offenders.join("\n  ")
    );
}
