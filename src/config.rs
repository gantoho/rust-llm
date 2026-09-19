//! 训练配置（`config/config.json`）
//!
//! 用 serde 序列化，`cargo run -- train --config config/config.json` 加载。
//! 缺省字段自动取 [`Config::default`]，模型超参数在 `model` 里，训练流程参数在 `train` 里。
//!
//! 目录约定：配置文件放 `config/`、权重放 `checkpoints/`、日志放 `logs/`（见下方常量）。
//! 所有产物路径在写盘前都会经过 [`ensure_parent_dir`] 自动建目录，产物不会散落到仓库根目录。

use crate::model::GPTConfig;
use serde::{Deserialize, Serialize};

/// 默认配置文件路径（放在 `config/` 目录，保持仓库根目录整洁）
pub const DEFAULT_CONFIG_PATH: &str = "config/config.json";
/// 默认权重输出目录（checkpoint 与 `tokenizer.json` 都写在这里）
pub const DEFAULT_OUT_DIR: &str = "checkpoints";
/// 默认训练指标日志文件（CSV，训练时自动创建 `logs/` 目录）
pub const DEFAULT_LOG_FILE: &str = "logs/train.csv";

/// 确保目录本身存在（checkpoint 目录、日志目录等），不存在则递归创建。
pub fn ensure_dir(dir: &str) {
    std::fs::create_dir_all(dir).unwrap_or_else(|e| panic!("创建目录 {dir} 失败: {e}"));
}

/// 写文件前确保其父目录存在：`logs/train.csv` → 自动创建 `logs/`。
///
/// 配置 / 权重 / 日志三类产物在落盘前都调用这里，所以把 `out_dir`、`log_file`
/// 改成任意嵌套路径（如 `outputs/run1/logs/train.csv`）也能自动建目录。
/// `path` 不含目录部分（如 `train.csv`）时不做任何事。
pub fn ensure_parent_dir(path: &str) {
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            ensure_dir(&parent.to_string_lossy());
        }
    }
}

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
    pub out_dir: String,          // checkpoint 输出目录（默认 checkpoints/）
    /// 梯度累积步数：每 accum_steps 步小 batch 才做一次 optimizer.step()。
    /// 有效 batch_size = batch_size * accum_steps。1 = 不累积（默认）。
    pub accum_steps: usize,
    /// 分词器文件路径：Some 时从文件加载（跳过训练），None 时从语料训练并保存。
    /// 训练完成后自动保存到 `{out_dir}/tokenizer.json`。
    pub tokenizer_file: Option<String>,
    /// LoRA 微调配置：Some(rank, alpha) 时冻结主模型，只训练 LoRA 层。
    /// rank 通常 4-64，alpha 通常 = rank。
    pub lora: Option<LoRAConfig>,
    /// 训练指标日志文件路径：**每个评估点**（每 eval_every 步 + 最后一步）记录一行 step/lr/loss/ppl 到 CSV。
    /// 默认 `logs/train.csv`（日志目录自动创建）；显式设为 `null` 时不记录。
    /// 注意每次训练会**覆盖**该文件（不是追加），要留档请一个实验用一个路径。
    pub log_file: Option<String>,
    /// 早停耐心值：验证 loss 连续 N 次评估不改善就提前终止训练。
    /// 0 = 不启用早停（默认）。
    pub early_stop_patience: usize,
    /// SFT（监督微调）语料路径：以 `,` 分隔，每项可以是文件、目录或以 `*` 通配的路径。
    ///
    /// 只对 `sft` 子命令有意义。语料按对话解析：行首带角色标记的算一轮问答，
    /// 不带任何角色标记的文本会被整体跳过（所以指向混杂目录也不会把小说正文喂进来）。
    pub sft_file: Option<String>,
}

/// LoRA 微调配置
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LoRAConfig {
    pub rank: usize,
    pub alpha: f32,
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
            grad_clip: 1000000.0,
            eval_every: 100,
            eval_iters: 20,
            tokenizer: "bpe".to_string(),
            bpe_vocab: 512,
            train_file: "data/sample.txt".to_string(),
            val_file: None,
            out_dir: DEFAULT_OUT_DIR.to_string(),
            accum_steps: 1,
            tokenizer_file: None,
            lora: None,
            log_file: Some(DEFAULT_LOG_FILE.to_string()),
            early_stop_patience: 0,
            sft_file: None,
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
        let cfg: Config = serde_json::from_str(&text)
            .unwrap_or_else(|e| panic!("解析配置文件 {path} 失败: {e}"));
        cfg.validate();
        cfg
    }

    /// 校验训练参数，防止除零/下溢等运行时 panic。
    /// 配置来自用户手写的 JSON，必须在这里拦截非法值。
    pub fn validate(&self) {
        let t = &self.train;
        let m = &self.model;
        // 训练参数
        assert!(t.steps >= 1, "train.steps 必须 >= 1");
        assert!(t.batch_size >= 1, "train.batch_size 必须 >= 1");
        assert!(t.eval_every >= 1, "train.eval_every 必须 >= 1（用于取模求余）");
        assert!(t.eval_iters >= 1, "train.eval_iters 必须 >= 1（用于求平均）");
        assert!(
            t.warmup_steps <= t.steps,
            "train.warmup_steps（{}）不能大于 train.steps（{}）",
            t.warmup_steps, t.steps
        );
        assert!(t.max_lr > 0.0, "train.max_lr 必须 > 0");
        assert!(t.min_lr >= 0.0, "train.min_lr 不能为负");
        assert!(t.min_lr <= t.max_lr, "train.min_lr（{}）不能大于 train.max_lr（{}）", t.min_lr, t.max_lr);
        assert!(t.weight_decay >= 0.0, "train.weight_decay 不能为负");
        assert!(t.grad_clip > 0.0, "train.grad_clip 必须 > 0");
        assert!(t.accum_steps >= 1, "train.accum_steps 必须 >= 1（否则除零）");
        assert!(t.bpe_vocab >= 256, "train.bpe_vocab 必须 >= 256（字节级基础词表）");
        // 模型参数
        assert!(m.n_embd >= 1, "model.n_embd 必须 >= 1");
        assert!(m.n_head >= 1, "model.n_head 必须 >= 1");
        assert!(m.n_layer >= 1, "model.n_layer 必须 >= 1");
        assert!(m.block_size >= 1, "model.block_size 必须 >= 1");
        assert!(m.n_embd % m.n_head == 0, "model.n_embd（{}）必须能被 model.n_head（{}）整除", m.n_embd, m.n_head);
        assert!(m.dropout >= 0.0 && m.dropout < 1.0, "model.dropout 必须在 [0, 1) 之间");
        if m.n_kv_head > 0 {
            assert!(m.n_kv_head <= m.n_head, "model.n_kv_head（{}）不能大于 model.n_head（{}）", m.n_kv_head, m.n_head);
            assert!(m.n_head % m.n_kv_head == 0, "model.n_head（{}）必须能被 model.n_kv_head（{}）整除", m.n_head, m.n_kv_head);
        }
        // LoRA
        if let Some(ref lora) = t.lora {
            assert!(lora.rank >= 1, "lora.rank 必须 >= 1");
            assert!(lora.alpha > 0.0, "lora.alpha 必须 > 0");
        }
    }

    /// 小模型预设（适合学习/演示，快速验证）
    ///
    /// - 4 层 Transformer，隐藏维度 256，8 头注意力
    /// - 上下文长度 128，BPE 词表 512
    /// - 约 ~2M 参数，CPU 上几分钟即可完成训练
    pub fn preset_small() -> Config {
        Config {
            model: GPTConfig {
                vocab_size: 0,
                n_embd: 256,
                n_head: 8,
                n_layer: 4,
                block_size: 128,
                ..GPTConfig::default()
            },
            train: TrainConfig {
                steps: 2000,
                batch_size: 16,
                max_lr: 6e-4,
                min_lr: 6e-5,
                warmup_steps: 100,
                eval_every: 200,
                bpe_vocab: 512,
                tokenizer: "bpe".to_string(),
                train_file: "data/alice.txt".to_string(),
                ..TrainConfig::default()
            },
        }
    }

    /// 中等模型预设（适合中等语料，性能与质量平衡）
    ///
    /// - 8 层 Transformer，隐藏维度 512，8 头注意力
    /// - 上下文长度 256，BPE 词表 2048
    /// - 支持 GQA（4 KV heads）、RMSNorm、SwiGLU（LLaMA 风格）
    /// - 约 ~15M 参数，GPU 推荐
    pub fn preset_medium() -> Config {
        Config {
            model: GPTConfig {
                vocab_size: 0,
                n_embd: 512,
                n_head: 8,
                n_layer: 8,
                block_size: 256,
                n_kv_head: 4,
                use_rmsnorm: true,
                use_swiglu: true,
                dropout: 0.1,
            },
            train: TrainConfig {
                steps: 10000,
                batch_size: 32,
                max_lr: 3e-4,
                min_lr: 3e-5,
                warmup_steps: 500,
                weight_decay: 0.1,
                eval_every: 500,
                eval_iters: 50,
                bpe_vocab: 2048,
                tokenizer: "bpe".to_string(),
                accum_steps: 2,
                ..TrainConfig::default()
            },
        }
    }

    /// 大模型预设（适合较大语料，高质量生成）
    ///
    /// - 12 层 Transformer，隐藏维度 768，12 头注意力
    /// - 上下文长度 512，BPE 词表 4096
    /// - 支持 GQA（4 KV heads）、RMSNorm、SwiGLU、Dropout
    /// - 约 ~85M 参数，需要 GPU
    pub fn preset_large() -> Config {
        Config {
            model: GPTConfig {
                vocab_size: 0,
                n_embd: 768,
                n_head: 12,
                n_layer: 12,
                block_size: 512,
                n_kv_head: 4,
                use_rmsnorm: true,
                use_swiglu: true,
                dropout: 0.1,
            },
            train: TrainConfig {
                steps: 50000,
                batch_size: 32,
                max_lr: 3e-4,
                min_lr: 3e-5,
                warmup_steps: 2000,
                weight_decay: 0.1,
                grad_clip: 1000000.0,
                eval_every: 1000,
                eval_iters: 100,
                bpe_vocab: 4096,
                tokenizer: "bpe".to_string(),
                accum_steps: 4,
                ..TrainConfig::default()
            },
        }
    }

    /// 按名称获取预设配置
    pub fn from_preset(name: &str) -> Config {
        match name {
            "small" => Config::preset_small(),
            "medium" => Config::preset_medium(),
            "large" => Config::preset_large(),
            other => panic!(
                "未知预设 '{}'（可选：small / medium / large）",
                other
            ),
        }
    }

    /// 保存配置到 JSON 文件
    pub fn save(&self, path: &str) {
        let json = serde_json::to_string_pretty(self)
            .expect("序列化配置失败");
        ensure_parent_dir(path);
        std::fs::write(path, json)
            .unwrap_or_else(|e| panic!("无法写入配置文件 {path}: {e}"));
    }
}
