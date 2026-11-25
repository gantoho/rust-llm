//! 集成测试共用的辅助：临时目录、tiny 训练配置、语料写入。
//!
//! 只依赖 `std` + `serde_json`（项目已有依赖），不引入新的测试框架。

use std::path::{Path, PathBuf};

/// 建一个干净（先删后建）的临时目录。
pub fn fresh_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("llm_it_{tag}"));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("ckpt")).expect("创建临时目录");
    dir
}

/// 写一份足够小、能在 1 秒内训练完的训练配置（char 分词器 / 32 维 / 1 层）。
pub fn tiny_config(dir: &Path, corpus: &Path, steps: usize) -> serde_json::Value {
    serde_json::json!({
        "model": {
            "vocab_size": 0, "n_embd": 32, "n_head": 2, "n_layer": 1, "block_size": 8,
            "use_swiglu": false, "tie_embeddings": true, "dropout": 0.0
        },
        "train": {
            "steps": steps, "batch_size": 1, "max_lr": 0.001, "min_lr": 0.0001,
            "warmup_steps": 1, "eval_every": 1, "eval_iters": 1, "early_stop_patience": 0,
            "tokenizer": "char", "tokenizer_file": null,
            "train_file": corpus.to_string_lossy(), "val_file": null,
            "out_dir": dir.join("ckpt").to_string_lossy(),
            "log_file": dir.join("ckpt/train.csv").to_string_lossy(),
            "sft_file": null, "amp": false, "bf16": false
        }
    })
}

/// 把 JSON 写到 `<dir>/cfg.json`，返回路径。
pub fn write_config(dir: &Path, cfg: &serde_json::Value) -> PathBuf {
    let path = dir.join("cfg.json");
    std::fs::write(&path, serde_json::to_string_pretty(cfg).unwrap()).expect("写配置文件");
    path
}

/// 写一段重复的中文语料（200 遍足够切出多个 block）。
pub fn write_corpus(dir: &Path) -> PathBuf {
    let path = dir.join("corpus.txt");
    // 语料要覆盖 SFT 模板用到的字符与换行：char 分词器的词表 = 语料里出现过的字符，
    // 少一个 `\n` 都会让 SFT 阶段 encode 直接 panic。
    std::fs::write(
        &path,
        "你好世界。\n用户：你好\n助手：你好\n用户：再见\n助手：再见\n".repeat(60),
    )
    .expect("写语料");
    path
}
