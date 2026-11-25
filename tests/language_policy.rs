//! 语言策略：仓库里除 `web/`（浏览器前端）之外，不允许出现 Python / Shell / PowerShell 脚本。
//!
//! 为什么值得一条测试：README 主张"纯 Rust 从零实现"，但这类脚本最容易以"临时用一下"的名义
//! 混进来（本仓库就曾有一个 `tools/gen_sft_corpus.py` 和 `train.ps1`）。数据生成、构建辅助
//! 都应该做成 **Rust 子命令**（如 `sft-gen`），而不是外部脚本。
//!
//! 允许的例外，以及理由：
//! - `web/`：浏览器界面的 TypeScript 源码与它的 node_modules（前端无法用 Rust 写）；
//! - `target/`、`.git/`：构建产物与版本库元数据，不属于源码。

use std::fs;
use std::path::{Path, PathBuf};

/// 这些目录下的内容不参与检查（构建产物 / 版本库 / 前端）
const SKIP_DIRS: &[&str] = &[
    "target",
    ".git",
    "web",
    "node_modules",
    "logs",
    "checkpoints",
];

/// 禁止的脚本扩展名
const FORBIDDEN_EXT: &[&str] = &["py", "ps1", "sh", "bat", "cmd"];

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    let Ok(entries) = fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
        if path.is_dir() {
            if SKIP_DIRS.contains(&name) {
                continue;
            }
            walk(&path, out);
        } else {
            out.push(path);
        }
    }
}

#[test]
fn no_foreign_scripts_outside_web() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    walk(&root, &mut files);
    assert!(!files.is_empty(), "没扫到任何文件，路径可能不对");

    let mut offenders = Vec::new();
    for path in &files {
        let Some(ext) = path.extension().and_then(|e| e.to_str()) else {
            continue;
        };
        if FORBIDDEN_EXT.contains(&ext) {
            let rel = path.strip_prefix(&root).unwrap_or(path);
            offenders.push(format!("{}", rel.display()));
        }
    }

    assert!(
        offenders.is_empty(),
        "仓库里不允许出现外部脚本（请做成 Rust 子命令；前端源码请放在 web/ 下）：\n  {}",
        offenders.join("\n  ")
    );
}
