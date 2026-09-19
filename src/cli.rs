//! 命令行入口（clap）
//!
//! ```text
//! cargo run -- train    --config config/config.json [--resume checkpoints/latest.ckpt]
//! cargo run -- eval     --config config/config.json [--ckpt checkpoints/latest.ckpt]
//! cargo run -- generate --config config/config.json [--ckpt ...] [--prompt "Once"] [--max-new 100] ...
//! cargo run -- chat     --config config/config.json [--ckpt ...] [--system "..."]
//! cargo run -- sft      --config config/config.json --pretrained ckpt [--sft-file "..."]
//! cargo run -- finetune --config config/config.json --pretrained ckpt [--steps 1000] [--lr 1e-4]
//! cargo run -- preset   [--name small] [--output config/config.json]
//! cargo run -- demo     # 教学演示（XOR + BPE + 内置语料小 GPT）
//! cargo run -- bench    # 性能基准（固定小模型测训练 / 推理吞吐）
//! ```
//!
//! 目录约定：配置在 `config/`、权重在 `checkpoints/`、日志在 `logs/`（见 [`crate::config`] 的常量）。
//!
//! 其中 `train` / `sft` / `finetune` / `eval` / `generate` / `chat` 每次运行都会自动在 `logs/` 下
//! 写一份 `{操作}_{时间戳}.log` 运行日志（完整命令行 + 完整配置 + 过程输出），见 [`crate::runlog`]。

use clap::{Parser, Subcommand};

#[derive(Parser)]
#[command(
    name = "llm_from_scratch",
    about = "从零实现的 GPT 语言模型（算法纯手写，零深度学习框架依赖）",
    long_about = "一个完整的 GPT 语言模型训练与推理框架，全部算法纯 Rust 手写实现。\n\
                   支持 GPT-2 和 LLaMA 风格架构（RoPE、RMSNorm、SwiGLU、GQA）、\n\
                   KV Cache 加速推理、监督微调（SFT）、Beam Search 生成、GPU 加速等。"
)]
pub struct Cli {
    #[command(subcommand)]
    pub cmd: Cmd,
}

#[derive(Subcommand)]
pub enum Cmd {
    /// 训练模型（超参数与数据路径见 config/config.json）
    Train {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// 从已有 checkpoint 继续训练
        #[arg(long)]
        resume: Option<String>,
    },
    /// 在验证集上评估模型：loss 与困惑度（perplexity）
    Eval {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// checkpoint 文件（缺省用 out_dir/latest.ckpt）
        #[arg(long)]
        ckpt: Option<String>,
        /// 分词器文件路径（缺省自动从 out_dir/tokenizer.json 加载）
        #[arg(long)]
        tokenizer: Option<String>,
    },
    /// 用训练好的模型生成文本
    Generate {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// checkpoint 文件（缺省用 out_dir/latest.ckpt）
        #[arg(long)]
        ckpt: Option<String>,
        /// 分词器文件路径（缺省自动从 out_dir/tokenizer.json 加载）
        #[arg(long)]
        tokenizer: Option<String>,
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
        /// 重复惩罚系数：>1 压低最近出现过的 token（1.0 = 关闭）
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,
        /// 重复惩罚的回看窗口：只看最近 N 个 token（0 = 关闭）
        #[arg(long, default_value_t = 64)]
        repetition_window: usize,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// 禁用 KV cache（每个新 token 都全量前向）
        #[arg(long)]
        no_kv_cache: bool,
        /// 使用 Beam Search 生成（指定束宽，通常 4-10）
        #[arg(long)]
        beam: Option<usize>,
        /// Beam Search 长度惩罚指数（0=不惩罚，>0 偏好长序列）
        #[arg(long, default_value_t = 0.6)]
        length_penalty: f32,
    },
    /// 交互式对话模式：持续输入提示词，模型逐个生成回复
    Chat {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// checkpoint 文件（缺省用 out_dir/latest.ckpt）
        #[arg(long)]
        ckpt: Option<String>,
        /// 分词器文件路径（缺省自动从 out_dir/tokenizer.json 加载）
        #[arg(long)]
        tokenizer: Option<String>,
        /// 系统提示词（可选，会在每次输入前附加）
        #[arg(long, default_value = "")]
        system: String,
        /// 采样温度
        #[arg(long, default_value_t = 0.8)]
        temperature: f32,
        /// top-k 采样
        #[arg(long, default_value_t = 40)]
        top_k: usize,
        /// top-p 采样
        #[arg(long, default_value_t = 0.9)]
        top_p: f32,
        /// 重复惩罚系数：>1 压低最近出现过的 token（1.0 = 关闭）
        #[arg(long, default_value_t = 1.1)]
        repetition_penalty: f32,
        /// 重复惩罚的回看窗口：只看最近 N 个 token（0 = 关闭）
        #[arg(long, default_value_t = 64)]
        repetition_window: usize,
        /// 每次生成的最大 token 数
        #[arg(long, default_value_t = 200)]
        max_new: usize,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// prompt 模板：`sft` = 与 `sft` 子命令训练时一致的对话模板（模型才会"回答"）；
        /// `raw` = 直接把历史拼给模型（只有预训练权重、未做过 SFT 时用）
        #[arg(long, value_parser = ["sft", "raw"], default_value = "sft")]
        prompt_format: String,
    },
    /// 监督微调（SFT）：用「提问→回答」语料把只会续写的预训练模型教会应答
    Sft {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// 预训练 checkpoint（SFT 必须从预训练权重出发）
        #[arg(long)]
        pretrained: String,
        /// SFT 语料路径（逗号分隔，可含 `*` 通配）；缺省用 config 里的 train.sft_file
        #[arg(long)]
        sft_file: Option<String>,
        /// 微调步数（缺省用 config 里的 train.steps）
        #[arg(long)]
        steps: Option<usize>,
        /// 峰值学习率（缺省用 config 里 train.max_lr 的 1/10；SFT 要比预训练小一个量级才不冲掉已有能力）
        #[arg(long)]
        lr: Option<f32>,
        /// 输出目录（缺省 `{config 的 out_dir}-sft`，避免覆盖预训练权重）
        #[arg(long)]
        out_dir: Option<String>,
    },
    /// 加载预训练权重微调（当前是**全参微调**：LoRA 参数只做校验与预估，尚未接入训练循环）
    Finetune {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// 预训练模型 checkpoint
        #[arg(long)]
        pretrained: String,
        /// LoRA 秩（通常 4-64）。⚠️ 尚未接入训练循环，只影响日志里的参数占比预估
        #[arg(long, default_value_t = 16)]
        lora_rank: usize,
        /// LoRA 缩放因子 α（通常 = rank）。⚠️ 同上，不参与训练
        #[arg(long, default_value_t = 16.0)]
        lora_alpha: f32,
        /// 微调步数
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        /// 微调学习率
        #[arg(long, default_value_t = 1e-4)]
        lr: f32,
    },
    /// 生成预设配置文件（small / medium / large）
    Preset {
        /// 预设名称
        #[arg(long, default_value = "small")]
        name: String,
        /// 输出配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        output: String,
    },
    /// 教学演示：XOR + BPE + 内置语料小 GPT
    Demo,
    /// 性能基准：固定小模型跑少量训练步与短生成，输出吞吐（tok/s）供优化前后对比
    Bench {
        /// 训练步数
        #[arg(long, default_value_t = 10)]
        steps: usize,
        /// 生成 token 数
        #[arg(long, default_value_t = 64)]
        gen_tokens: usize,
    },
}

impl Cli {
    pub fn parse_args() -> Self {
        Cli::parse()
    }
}
