//! 训练配置（`config.json`）
//!
//! 用 serde 序列化，`cargo run -- train --config config.json` 加载。
//! 缺省字段自动取 [`Config::default`]，模型超参数在 `model` 里，训练流程参数在 `train` 里。

use crate::model::GPTConfig;
use serde::{Deserialize, Serialize};

/// 训练流程参数
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainConfig {
    pub seed: u64,                // 随机种子（复现实验）
    pub batch_size: usize,        // 每批序列条数
    pub steps: usize,             // 总训练步数
    pub max_lr: f32,              // 峰值学习率
    pub min_lr: f32,              // 最低学习率（cosine 衰减到它）
    pub warmup_steps: usize,      // 线性预热步数
    pub weight_decay: f32,        // AdamW 权重衰减
    pub grad_clip: f32,           // 梯度裁剪阈值
    pub eval_every: usize,        // 每 N 步评估一次验证集并保存 latest checkpoint
    pub eval_iters: usize,        // 评估时采样的批数
    pub tokenizer: String,        // "char" 字符级 / "bpe" BPE
    pub bpe_vocab: usize,         // BPE 目标词表大小（= 256 字节 + 合并数）
    pub train_file: String,       // 训练语料文件
    pub val_file: Option<String>, // 验证语料文件；None 时自动从训练文本末尾切 10%
    pub out_dir: String,          // checkpoint 输出目录
}

impl Default for TrainConfig {
    fn default() -> Self {
        TrainConfig {
            seed: 42,
            batch_size: 8,
            steps: 1000,
            max_lr: 3e-3,
            min_lr: 3e-4,
            warmup_steps: 20,
            weight_decay: 0.01,
            grad_clip: 1.0,
            eval_every: 100,
            eval_iters: 20,
            tokenizer: "bpe".to_string(),
            bpe_vocab: 512,
            train_file: "data/sample.txt".to_string(),
            val_file: None,
            out_dir: "checkpoints".to_string(),
        }
    }
}

/// 完整配置：模型超参数 + 训练参数
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: GPTConfig,
    pub train: TrainConfig,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: GPTConfig::default(),
            train: TrainConfig::default(),
        }
    }
}

impl Config {
    /// 从 JSON 文件加载配置
    pub fn load(path: &str) -> Config {
        let text = std::fs::read_to_string(path)
            .unwrap_or_else(|e| panic!("无法读取配置文件 {path}: {e}"));
        serde_json::from_str(&text).unwrap_or_else(|e| panic!("解析配置文件 {path} 失败: {e}"))
    }
}
