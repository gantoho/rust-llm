//! 二进制冒烟测试：验证 CLI 分发链路端到端可用。
//!
//! 本 crate 只有 bin target（无 library target，集成测试无法 `use` 内部模块），
//! 所以走 [`CARGO_BIN_EXE_*`] 启动真实可编译产物，只跑轻量子命令
//! （help / preset），不加载模型、不训练，保持毫秒级。

use std::process::Command;

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_llm_from_scratch"))
}

/// `--help` 正常退出且列出了主要子命令（clap 解析 + 分发枚举没退化）。
#[test]
fn help_lists_subcommands() {
    let out = bin().arg("--help").output().expect("启动二进制失败");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));
    let stdout = String::from_utf8_lossy(&out.stdout);
    for cmd in ["train", "eval", "generate", "serve", "preset"] {
        assert!(stdout.contains(cmd), "help 输出缺少子命令 {cmd}:\n{stdout}");
    }
}

/// `preset` 端到端：解析参数 → 生成配置 → 写出文件。
#[test]
fn preset_writes_config_file() {
    let path = std::env::temp_dir().join(format!("llm_smoke_preset_{}.json", std::process::id()));
    let out = bin()
        .args(["preset", "--name", "small", "--output"])
        .arg(&path)
        .output()
        .expect("启动二进制失败");
    assert!(out.status.success(), "stderr: {}", String::from_utf8_lossy(&out.stderr));

    let text = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("输出文件不可读 {path:?}: {e}"));
    // 配置字段确实落盘（不是空文件/截断）
    assert!(text.contains("n_embd"), "配置内容异常:\n{}", &text[..text.len().min(400)]);
    let parsed: serde_json::Value = serde_json::from_str(&text).expect("输出不是合法 JSON");
    assert!(parsed.get("model").is_some() && parsed.get("train").is_some());

    let _ = std::fs::remove_file(&path);
}
