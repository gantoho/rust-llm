//! `allow(dead_code)` 策略：**每一处都必须写 `reason`**。
//!
//! 为什么：`allow(dead_code)` 会永久压掉"这段代码没人用"的信号。允许它存在的前提是
//! 有人解释过"为什么保留"——是 GPU 分支专用、测试参考实现，还是对外 API。
//! 没有理由的豁免等于让死代码静默堆积。
//!
//! 规则：源码里任何 `allow(dead_code` 的属性行都必须同时包含 `reason = "…"`。

use std::fs;
use std::path::{Path, PathBuf};

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

#[test]
fn every_dead_code_allow_has_reason() {
    let mut files = Vec::new();
    rs_files(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    assert!(!files.is_empty(), "没扫到源码");

    let mut total = 0usize;
    let mut offenders = Vec::new();
    for path in &files {
        let text = fs::read_to_string(path).expect("读源码");
        for (i, line) in text.lines().enumerate() {
            let t = line.trim_start();
            if t.starts_with("//") || !t.contains("allow(dead_code") {
                continue;
            }
            total += 1;
            if !t.contains("reason = ") {
                offenders.push(format!("{}:{}  {}", path.display(), i + 1, t));
            }
        }
    }

    assert!(
        total > 0,
        "一处 allow(dead_code) 都没扫到，扫描逻辑可能失效"
    );
    assert!(
        offenders.is_empty(),
        "以下 allow(dead_code) 缺 reason（请说明为什么保留这段代码）：\n  {}",
        offenders.join("\n  ")
    );
    // 数量本身不是目标，但把它印出来便于观察豁免是否在膨胀
    eprintln!("allow(dead_code) 共 {total} 处，全部带 reason");
}
