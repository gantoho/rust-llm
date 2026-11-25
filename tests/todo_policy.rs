//! 代码卫生策略：源码里不允许留下 `TODO` / `FIXME` / `todo!()` / `unimplemented!()`。
//!
//! 这条测试本身就是"盘点"：当前为 **0 处**（此前统计的 4 处是把 `rec.xxx()` / `crate::xxx`
//! 里的 `xxx` 当成 `XXX` 的大小写误报）。留着它是为了防止"先写个 TODO 以后再说"慢慢堆积——
//! 要么当场做完，要么写成带 issue 的说明并从这里豁免（豁免需在下面列出）。

use std::fs;
use std::path::{Path, PathBuf};

/// 允许出现的豁免（文件名:行 或 说明）；当前为空。
const ALLOWED: &[&str] = &[];

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
fn no_todo_or_unimplemented_in_src() {
    let mut files = Vec::new();
    rs_files(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("src"),
        &mut files,
    );
    assert!(!files.is_empty(), "没扫到源码");

    let mut hits = Vec::new();
    for path in &files {
        let text = fs::read_to_string(path).expect("读源码");
        for (i, line) in text.lines().enumerate() {
            // 大小写敏感：`xxx` 作为占位符/示例出现是合法的，不算 TODO
            for pat in ["TODO", "FIXME", "todo!(", "unimplemented!("] {
                if line.contains(pat) {
                    let key = format!("{}:{}", path.display(), i + 1);
                    if !ALLOWED.iter().any(|a| key.contains(a)) {
                        hits.push(format!("{key}  含有 `{pat}`：{}", line.trim()));
                    }
                }
            }
        }
    }

    assert!(
        hits.is_empty(),
        "源码里不允许留 TODO / unimplemented!（要么当场做完，要么加进 ALLOWED 并写清原因）：\n  {}",
        hits.join("\n  ")
    );
}
