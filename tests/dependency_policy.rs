//! 依赖策略测试：把 `DEPENDENCIES.md` 里的约定变成**可执行断言**。
//!
//! 为什么要这条测试：README 声称"算法全部自研"，但这种声明最容易随时间腐化——
//! 某天有人为了图快 `use candle_core`，CI 不会有任何反应。这里把它变成硬约束：
//!
//! 1. `Cargo.toml` 的 `[dependencies]` 只允许出现**基础设施**类 crate（白名单）；
//! 2. 源码里不允许出现任何深度学习框架 / 张量库 / 预训练模型库的引用；
//! 3. 算法模块（张量、注意力、训练、采样、分词器…）只允许 `std` + 并行调度 + 序列化，
//!    不允许把数学计算外包给别的库。

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

/// 允许出现在 `[dependencies]` 里的 crate（全部属于基础设施，不参与数学计算）。
///
/// 新增条目必须同时更新 `DEPENDENCIES.md` 的决策记录——这条测试就是它的执行体。
const ALLOWED_DEPS: &[&str] = &["serde", "serde_json", "clap", "rayon", "tiny_http", "image"];

/// 可选依赖：只有显式 `--features gpu` 时才引入。
const ALLOWED_OPTIONAL_DEPS: &[&str] = &["wgpu", "pollster", "image"];

/// 明确禁止的 crate 前缀：深度学习框架、张量库、预训练模型加载器、分词器库。
///
/// 出现任何一个都意味着"自研"的主张不再成立（哪怕只是间接依赖）。
const FORBIDDEN_PREFIXES: &[&str] = &[
    "candle",
    "tch",
    "torch",
    "ndarray",
    "burn",
    "dfdx",
    "luminal",
    "gpt2",
    "tokenizers",
    "safetensors",
    "hf_hub",
    "transformers",
    "onnxruntime",
    "ort",
    "tract",
    "wgpu_matrix", // 假想中的"别人写好的矩阵乘"
];

/// 算法模块：这些文件里的计算必须逐行自研，只允许 `std` / `rayon` / 序列化。
const ALGORITHM_MODULES: &[&str] = &[
    "tensor.rs",
    "autograd.rs",
    "attention.rs",
    "model.rs",
    "quant.rs",
    "optim.rs",
    "train.rs",
    "sample.rs",
    "tokenizer.rs",
    "data.rs",
    "hf.rs",
    "vqvae.rs",
    "loss.rs",
    "rope.rs",
    "fp8.rs",
    "distill.rs",
    "speculative.rs",
];

/// 算法模块里允许出现的**非 std** 顶层 crate：并行调度与序列化（都不是数学计算）。
const ALGORITHM_ALLOWED_NON_STD: &[&str] = &["rayon", "serde", "serde_json"];

fn workspace_root() -> PathBuf {
    // 集成测试的当前目录是 crate 根
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// 极简 TOML 读取：只解析 `[dependencies]` / `[features]` 段里的 `name = ...` 与
/// `gpu = [...]` 列表，避免为此引入 toml crate（那本身就违反本测试的初衷）。
fn parse_deps(text: &str) -> (BTreeSet<String>, BTreeSet<String>) {
    let mut deps = BTreeSet::new();
    let mut optional = BTreeSet::new();
    let mut section = String::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with('[') {
            section = line.trim_matches(['[', ']']).to_string();
            continue;
        }
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        // `dep:xxx` 形式的可选依赖（写在 [features] 里）
        if section == "features" {
            for part in line
                .split('[')
                .nth(1)
                .unwrap_or("")
                .split(']')
                .next()
                .unwrap_or("")
                .split(',')
            {
                let p = part.trim().trim_matches('"');
                if let Some(name) = p.strip_prefix("dep:") {
                    optional.insert(name.to_string());
                }
            }
            continue;
        }
        if section.ends_with("dependencies")
            && let Some((name, _)) = line.split_once('=')
        {
            deps.insert(name.trim().to_string());
        }
    }
    (deps, optional)
}

/// 从一行源码里抽 `use xxx::` 的顶层 crate 名（跳过 `std`/`core`/`alloc`/`crate`/`self`/`super`）。
fn top_level_crate(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("use ")?;
    let first = rest.split([':', ';', ' ']).next()?.trim();
    if first.is_empty() || matches!(first, "std" | "core" | "alloc" | "crate" | "self" | "super") {
        return None;
    }
    // 只保留标识符开头（排除 `{a, b}` 这类分组导入）
    if !first
        .chars()
        .next()
        .is_some_and(|c| c.is_ascii_alphabetic() || c == '_')
    {
        return None;
    }
    Some(first.to_string())
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

#[test]
fn cargo_toml_only_has_infrastructure_deps() {
    let text = fs::read_to_string(workspace_root().join("Cargo.toml")).expect("读 Cargo.toml");
    let (deps, optional) = parse_deps(&text);
    let allowed: BTreeSet<&str> = ALLOWED_DEPS.iter().copied().collect();
    let allowed_opt: BTreeSet<&str> = ALLOWED_OPTIONAL_DEPS.iter().copied().collect();

    for dep in &deps {
        assert!(
            allowed.contains(dep.as_str()) || allowed_opt.contains(dep.as_str()),
            "新增依赖 `{dep}` 不在白名单里。请先更新 DEPENDENCIES.md 的决策记录，\
             并确认它属于基础设施（不参与数学计算）；若是算法库，必须自研。"
        );
    }
    for dep in &optional {
        assert!(
            allowed_opt.contains(dep.as_str()),
            "可选依赖 `{dep}` 不在白名单里（允许：{ALLOWED_OPTIONAL_DEPS:?}）"
        );
    }
    // 白名单里的 crate 都必须真的被声明（防止文档与 Cargo.toml 漂移）
    for dep in ALLOWED_DEPS {
        assert!(
            deps.contains(*dep),
            "白名单声明了 `{dep}` 但 Cargo.toml 里没有——请同步 DEPENDENCIES.md"
        );
    }
}

#[test]
fn no_forbidden_frameworks_anywhere_in_src() {
    let mut files = Vec::new();
    rs_files(&workspace_root().join("src"), &mut files);
    assert!(!files.is_empty(), "没扫到任何 .rs 文件，路径可能不对");

    for path in &files {
        let text = fs::read_to_string(path).expect("读源码");
        for line in text.lines() {
            let Some(cref) = top_level_crate(line) else {
                continue;
            };
            for bad in FORBIDDEN_PREFIXES {
                assert!(
                    !cref.starts_with(bad),
                    "{} 引用了被禁止的 crate `{cref}`——自研原则不允许把计算外包给 `{bad}*`",
                    path.display()
                );
            }
        }
    }
}

#[test]
fn algorithm_modules_only_use_std_and_infra() {
    let root = workspace_root().join("src");
    for module in ALGORITHM_MODULES {
        let path = root.join(module);
        if !path.exists() {
            continue; // 模块可能已重命名/删除，测试不该因此失败
        }
        let text = fs::read_to_string(&path).expect("读算法模块");
        for line in text.lines() {
            let Some(cref) = top_level_crate(line) else {
                continue;
            };
            assert!(
                ALGORITHM_ALLOWED_NON_STD.contains(&cref.as_str()),
                "算法模块 {module} 引用了 `{cref}`：这里的计算必须自研，\
                 只允许 std / {ALGORITHM_ALLOWED_NON_STD:?}"
            );
        }
    }
}

/// 文档与 `Cargo.toml` 必须同步：白名单里的每个 crate 都要在 `DEPENDENCIES.md` 里
/// 有登记（用途 + 结论）。这条断言防止"加了依赖但没写文档"。
#[test]
fn dependency_doc_lists_every_allowed_crate() {
    let doc_path = workspace_root().join("DEPENDENCIES.md");
    let doc = fs::read_to_string(&doc_path).expect("读 DEPENDENCIES.md");
    for dep in ALLOWED_DEPS.iter().chain(ALLOWED_OPTIONAL_DEPS) {
        assert!(
            doc.contains(&format!("`{dep}`")),
            "依赖 `{dep}` 没有在 DEPENDENCIES.md 里登记（请补一行：用途 + 结论）"
        );
    }
}
