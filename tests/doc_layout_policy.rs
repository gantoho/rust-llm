//! 文档放置策略：`docs/` 只放**编号课程**，工程元文档放仓库根目录。
//!
//! 为什么值得一条测试：`docs/` 原本混进过 `DEPENDENCIES.md`、`TASKS.md` 这类工程元文档，
//! 夹在 `00-学习计划.md` … `50-MLA解耦RoPE与权重吸收.md` 的课程序列里，读者按编号翻目录时
//! 会撞到"不是课"的文件。约定写成断言后：想加课程就按编号命名，想加工程文档就放根目录。
//!
//! 根目录只保留两份：`README.md`（总览 + 实验记录）与 `DEPENDENCIES.md`（依赖边界）。
//! 曾经的 `TASKS.md` 已删除——它 100% 是已完成项（等价于变更日志），
//! 与 README 的历史记录重复，且会不可避免地漂移。

use std::fs;
use std::path::PathBuf;

/// 允许出现在仓库根目录的工程元文档
const ROOT_DOCS: &[&str] = &["README.md", "DEPENDENCIES.md"];

#[test]
fn docs_dir_holds_only_numbered_lessons() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let docs = root.join("docs");
    let entries = fs::read_dir(&docs).expect("读 docs/");
    let mut bad = Vec::new();
    let mut count = 0usize;
    for e in entries.flatten() {
        let name = e.file_name().to_string_lossy().into_owned();
        if e.path().is_dir() {
            continue;
        }
        count += 1;
        // `NN-课程名.md`（两位数字前缀 + 连字符）
        let ok = name.len() > 4
            && name.as_bytes()[0].is_ascii_digit()
            && name.as_bytes()[1].is_ascii_digit()
            && name.as_bytes()[2] == b'-'
            && name.ends_with(".md");
        if !ok {
            bad.push(name);
        }
    }
    assert!(count > 0, "docs/ 是空的？");
    assert!(
        bad.is_empty(),
        "docs/ 只允许编号课程（NN-课程名.md）；工程元文档请放仓库根目录。违规文件：\n  {}",
        bad.join("\n  ")
    );
}

#[test]
fn engineering_docs_live_at_root() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for name in ROOT_DOCS {
        assert!(root.join(name).is_file(), "根目录缺少工程元文档 {name}");
        assert!(
            !root.join("docs").join(name).exists(),
            "{name} 不应该在 docs/ 里——docs/ 是编号课程目录"
        );
    }
}
