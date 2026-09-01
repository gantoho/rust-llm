//! 命令行入口（clap）
//!
//! ```text
//! cargo run -- train    --config config.json [--resume checkpoints/latest.ckpt]
//! cargo run -- eval     --config config.json [--ckpt checkpoints/latest.ckpt]
//! cargo run -- generate --config config.json [--ckpt ...] [--prompt "Once"] [--max-new 100] ...
//! cargo run -- demo     # 教学演示（XOR + BPE + 内置语料小 GPT）
//! ```

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "llm_from_scratch",
    about = "从零实现的 GPT 语言模型（算法纯手写）"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// 训练模型（超参数与数据路径见 config.json）
    Train {
        /// 配置文件路径
        #[arg(long, default_value = "config.json")]
        config: String,
        /// 从已有 checkpoint 继续训练
        #[arg(long)]
        resume: Option<String>,
    },
    /// 在验证集上评估模型：loss 与困惑度（perplexity）
    Eval {
        /// 配置文件路径
        #[arg(long, default_value = "config.json")]
        config: String,
        /// checkpoint 文件（缺省用 out_dir/latest.ckpt）
        #[arg(long)]
        ckpt: Option<String>,
    },
    /// 用训练好的模型生成文本
    Generate {
        /// 配置文件路径
        #[arg(long, default_value = "config.json")]
        config: String,
        /// checkpoint 文件（缺省用 out_dir/latest.ckpt）
        #[arg(long)]
        ckpt: Option<String>,
        /// 初始提示词
        #[arg(long, default_value = "")]
        prompt: String,
        /// 生成的最大新 token 数
        #[arg(long, default_value_t = 100)]
        max_new: usize,
        /// 采样温度（>1 更随机，<1 更确定）
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        /// top-k 采样：只从概率最高的 k 个里选
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// top-p 采样：累计概率到 p 的最小集合
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// 禁用 KV cache（每个新 token 都全量前向）
        #[arg(long)]
        no_kv_cache: bool,
    },
    /// 教学演示：XOR + BPE + 内置语料小 GPT
    Demo,
}

impl Cli {
    pub fn parse_args() -> Self {
        Cli::parse()
    }
}
