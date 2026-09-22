//! 命令行入口（clap）
//!
//! ```text
//! cargo run -- train    --config config/config.json [--resume checkpoints/latest.ckpt]
//! cargo run -- eval     --config config/config.json [--ckpt checkpoints/latest.ckpt]
//! cargo run -- generate --config config/config.json [--ckpt ...] [--prompt "Once"] [--max-new 100] ...
//! cargo run -- chat     --config config/config.json [--ckpt ...] [--system "..."] [--merge-lora]
//! cargo run -- sft      --config config/config.json --pretrained ckpt [--sft-file "..."]
//! cargo run -- finetune --config config/config.json --pretrained ckpt [--lora-rank 16] [--lora-targets q,k,v] [--steps 1000] [--lr 1e-4]
//! cargo run -- finetune --config config/config.json --pretrained ckpt-lora --resume-lora   # 链式续训旧适配层
//! cargo run -- preset   [--name small] [--output config/config.json]
//! cargo run -- demo     # 端到端演示（XOR + BPE + 内置语料小 GPT）
//! cargo run -- bench    # 性能基准（固定小模型测训练 / 推理吞吐）
//! cargo run -- quant    --config config/config.json [--ckpt ...] [--bits int8|int4]
//!                       [--method rtn|gptq|awq] [--act-order true|false] [--damp 0.01] [--block 128]
//!                       [--calib-file data/calib.txt] [--calib-samples 64] [--alpha 0.5]
//!                       [--out checkpoints/quant.ckpt] [--eval]
//! cargo run -- distributed [--dp 4] [--tp 2] [--pp 2] [--micro-batches 4] [--steps 40]
//!                       [--batch-size 2] [--block-size 32] [--lr 0.05] [--weight-decay 0.01]
//!                       [--n-embd 32] [--n-layer 2] [--seed 42]
//! cargo run -- align    [--rm-steps 150] [--rm-lr 3e-3] [--steps 60] [--lr 1e-3]
//!                       [--beta 0.1] [--clip-eps 0.2] [--kl-coef 0.05] [--group-size 4]
//!                       [--batch-size 2] [--block-size 32] [--n-embd 16] [--n-layer 2] [--seed 42]
//! cargo run -- rag      [--chunk-size 120] [--overlap 30] [--top-k 4] [--mmr-lambda 0.5]
//!                       [--mmr-pool 12] [--hash-dim 512] [--context-chars 300] [--query "..."]
//! cargo run -- speculative [--max-new 48] [--gamma 4] [--mtp-heads 4] [--mtp-steps 80]
//!                       [--trials 2000] [--temperature 0.8] [--block-size 64]
//!                       [--n-embd 32] [--n-layer 2] [--seed 42] [--prompt "..."]
//! ```
//!
//! 目录约定：配置在 `config/`、权重在 `checkpoints/`、日志在 `logs/`（见 [`crate::config`] 的常量）。
//!
//! 其中 `train` / `sft` / `finetune` / `eval` / `generate` / `chat` / `scaling` / `moe` / `quant` /
//! `distributed` / `align` / `rag` / `speculative` 每次运行都会自动在
//! `logs/` 下写一份 `{操作}_{时间戳}.log` 运行日志（完整命令行 + 完整配置 + 过程输出），
//! 见 [`crate::runlog`]。

use clap::{Args, Parser, Subcommand};

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

/// 推理期覆盖 RoPE 频率参数与上下文长度（第 20 课长度外推）。
///
/// 全部留空 = 完全沿用 checkpoint 头部记录的设置（默认行为）。只有当你想拿一个在
/// 较短的 `block_size` 上训出来的权重去跑更长的上下文时，才需要显式给出
/// `--rope-scaling` 与 `--max-ctx`。
///
/// 这四项都**不改变任何参数的形状**，所以可以安全地套在已加载的权重上：
/// 权重是按"相对位置"训出来的，换一张频率表只是把这套相对位置关系引到更长的区间。
#[derive(Args, Debug, Clone)]
pub struct RopeArgs {
    /// 长度外推方式：留空 = 沿用 checkpoint 的设置（`config.json` 里的 `model.rope_scaling`）
    #[arg(long, value_parser = ["linear", "ntk", "yarn"])]
    pub rope_scaling: Option<String>,
    /// 外推倍数（仅在给了 `--rope-scaling` 时生效）：想跑 4 倍上下文就给 4
    #[arg(long, default_value_t = 4.0)]
    pub rope_factor: f32,
    /// RoPE 频率底数：留空 = 沿用 checkpoint（默认 10 000）
    #[arg(long)]
    pub rope_base: Option<f32>,
    /// 推理上下文长度上限：留空 = 沿用 checkpoint。可以大于训练时的 `block_size`
    /// （配合 `--rope-scaling`），这是"免训练扩上下文"的标准用法
    #[arg(long)]
    pub max_ctx: Option<usize>,
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
        /// 推理前把 LoRA 增量合并进主干权重（省掉每层两次小矩阵乘，数值不受影响）
        #[arg(long)]
        merge_lora: bool,
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
        /// 推理前把 LoRA 增量合并进主干权重（省掉每层两次小矩阵乘，数值不受影响）
        #[arg(long)]
        merge_lora: bool,
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
        /// KV cache 的量化位宽：none = f32；int8 / int4 = KIVI 式压缩
        /// （K 逐通道、V 逐 token），缓存显存降到 1/4 或 1/8
        #[arg(long, value_parser = ["none", "int8", "int4"], default_value = "none")]
        kv_bits: String,
        /// Attention Sink：缓存超窗丢弃时**永久保留最前面的 N 个位置**（StreamingLLM）。
        /// 流式长文本生成必须开（丢掉序列开头会让质量断崖下跌），0 = 关闭
        #[arg(long, default_value_t = 0)]
        kv_sink: usize,
        /// 使用 Beam Search 生成（指定束宽，通常 4-10）
        #[arg(long)]
        beam: Option<usize>,
        /// Beam Search 长度惩罚指数（0=不惩罚，>0 偏好长序列）
        #[arg(long, default_value_t = 0.6)]
        length_penalty: f32,
        #[command(flatten)]
        rope: RopeArgs,
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
        /// 推理前把 LoRA 增量合并进主干权重（省掉每层两次小矩阵乘，数值不受影响）
        #[arg(long)]
        merge_lora: bool,
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
        /// KV cache 的量化位宽：none = f32；int8 / int4 = KIVI 式压缩
        #[arg(long, value_parser = ["none", "int8", "int4"], default_value = "none")]
        kv_bits: String,
        /// Attention Sink：缓存超窗丢弃时永久保留最前面的 N 个位置（StreamingLLM）
        #[arg(long, default_value_t = 0)]
        kv_sink: usize,
        /// 随机种子
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// prompt 模板：`sft` = 与 `sft` 子命令训练时一致的对话模板（模型才会"回答"）；
        /// `raw` = 直接把历史拼给模型（只有预训练权重、未做过 SFT 时用）
        #[arg(long, value_parser = ["sft", "raw"], default_value = "sft")]
        prompt_format: String,
        #[command(flatten)]
        rope: RopeArgs,
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
    /// LoRA 微调：冻结预训练主干，只训练各层上的低秩适配层（缺省挂 Q/K/V）
    Finetune {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// 预训练模型 checkpoint（主干来源）
        #[arg(long)]
        pretrained: String,
        /// LoRA 秩 r（缺省 16，通常 4-64）：越小适配层参数越少、越不容易过拟合
        #[arg(long)]
        lora_rank: Option<usize>,
        /// LoRA 缩放因子 α（缺省 = rank，即增量不额外缩放）
        #[arg(long)]
        lora_alpha: Option<f32>,
        /// 适配层挂载位置（缺省 `q,k,v`）：逗号分隔，可选 q / k / v / o（输出投影）/ mlp / all
        #[arg(long)]
        lora_targets: Option<String>,
        /// 链式续训：基座是 LoRA 存档时，接着训**存档里那套**适配层（保留已学的增量），
        /// 而不是重挂一套全新的。此时 rank / alpha / 挂载位置全部由存档决定，不能再传
        #[arg(long)]
        resume_lora: bool,
        /// SFT 语料路径（逗号分隔，可含 `*` 通配）；缺省用 config 里的 train.sft_file
        #[arg(long)]
        sft_file: Option<String>,
        /// 微调步数
        #[arg(long, default_value_t = 1000)]
        steps: usize,
        /// 微调学习率
        #[arg(long, default_value_t = 1e-4)]
        lr: f32,
        /// 输出目录（缺省 `{config 的 out_dir}-lora`，避免覆盖预训练权重）
        #[arg(long)]
        out_dir: Option<String>,
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
    /// 端到端演示：XOR + BPE + 内置语料小 GPT
    Demo,
    /// Scaling Laws 实验：多规模实测扫描 + 幂律拟合 + Chinchilla 最优配比与算力/时长估算
    Scaling {
        /// 配置文件路径（提供语料、分词器与模型基底配置）
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// 规模网格：逗号分隔的 `层数x隐藏维度`，如 `2x64,4x128,8x256`
        #[arg(long, default_value = "2x64,4x128,6x192")]
        sizes: String,
        /// 每个规模训练多少步（固定 token 预算；数据量扫描以它为基础倍数放大）
        #[arg(long, default_value_t = 600)]
        steps: usize,
        /// 数据量扫描的倍数（对应文档练习 4 的"过训练分析"）
        #[arg(long, default_value = "1,2,4,8")]
        data_multiples: String,
        /// 每步批大小
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// 上下文长度
        #[arg(long, default_value_t = 64)]
        block_size: usize,
        /// 学习率
        #[arg(long, default_value_t = 3e-3)]
        lr: f32,
        /// 随机种子（整场扫描共用，保证可比）
        #[arg(long, default_value_t = 42)]
        seed: u64,
        /// 目标算力预算（FLOPs），用于最优配比与工时估算
        #[arg(long, default_value_t = 1e22)]
        budget: f64,
        /// 单卡理论峰值算力（TFLOPS，按训练精度）
        #[arg(long, default_value_t = 312.0)]
        gpu_tflops: f64,
        /// 卡数
        #[arg(long, default_value_t = 64)]
        n_gpu: usize,
        /// MFU（模型算力利用率，常见 0.3 ~ 0.6）
        #[arg(long, default_value_t = 0.4)]
        mfu: f64,
        /// 扫描结果 CSV 输出路径（缺省不落盘）
        #[arg(long)]
        out: Option<String>,
    },
    /// 性能基准：固定小模型跑少量训练步与短生成，输出吞吐（tok/s）供优化前后对比
    Bench {
        /// 训练步数
        #[arg(long, default_value_t = 10)]
        steps: usize,
        /// 生成 token 数
        #[arg(long, default_value_t = 64)]
        gen_tokens: usize,
    },
    /// MoE 稀疏专家实验（第 32 课）：参数/计算量口径 + 负载均衡对照 + 容量因子丢弃诊断
    Moe {
        /// 专家数网格：逗号分隔，逐个跑一遍对照实验
        #[arg(long, default_value = "8")]
        experts: String,
        /// 每个 token 激活的专家数（Top-K）
        #[arg(long, default_value_t = 2)]
        top_k: usize,
        /// 每个规模训练多少步
        #[arg(long, default_value_t = 300)]
        steps: usize,
        /// 每步批大小
        #[arg(long, default_value_t = 8)]
        batch_size: usize,
        /// 上下文长度
        #[arg(long, default_value_t = 64)]
        block_size: usize,
        /// 学习率
        #[arg(long, default_value_t = 3e-3)]
        lr: f32,
        /// 隐藏维度（专家与路由器的宽度）
        #[arg(long, default_value_t = 64)]
        n_embd: usize,
        /// Transformer 层数
        #[arg(long, default_value_t = 2)]
        n_layer: usize,
        /// 均衡辅助损失系数 α（对照组用 0）
        #[arg(long, default_value_t = 0.01)]
        aux_coef: f32,
        /// 容量因子扫描（逗号分隔；0 = 不限容量）
        #[arg(long, default_value = "0,1.0,1.25,2.0")]
        capacity_factors: String,
        /// 随机种子（各组共用，保证初始权重一致、可比）
        #[arg(long, default_value_t = 42)]
        seed: u64,
    },
    /// 权重量化（第 33 课）：RTN / GPTQ / AWQ 三种 weight-only 量化，出报告并可落盘
    Quant {
        /// 配置文件路径
        #[arg(long, default_value = crate::config::DEFAULT_CONFIG_PATH)]
        config: String,
        /// checkpoint 文件（缺省用 out_dir/latest.ckpt）
        #[arg(long)]
        ckpt: Option<String>,
        /// 分词器文件路径（缺省自动从 out_dir/tokenizer.json 加载）
        #[arg(long)]
        tokenizer: Option<String>,
        /// 量化位宽：int8 = 每权重 1 字节，int4 = 每权重半字节
        #[arg(long, default_value = "int8", value_parser = ["int8", "int4"])]
        bits: String,
        /// 量化算法：rtn = 直接取整（不需要校准集），gptq = Hessian 误差补偿，
        /// awq = 激活感知缩放
        #[arg(long, default_value = "rtn", value_parser = ["rtn", "gptq", "awq"])]
        method: String,
        /// 校准文本文件/目录/通配符（缺省用 `data/corpus/` 下的内置语料）。
        /// `rtn` 不需要它，给了也会被忽略（RTN 只用权重本身）
        #[arg(long)]
        calib_file: Option<String>,
        /// 校准**样本窗口数**：token 预算 = 窗口数 × 上下文长度（`block_size`）。
        /// 窗口越多 `H = XᵀX` 估得越准（`H⁻¹` 会放大采样噪声），代价是多跑几遍前向
        #[arg(long, default_value_t = 64)]
        calib_samples: usize,
        /// 校准预算的**硬上限**（token 数）：0 = 不额外设限，完全由 `--calib-samples` 决定。
        /// 想快速试一遍算法就把它压小（如 512），正式量化再放开
        #[arg(long, default_value_t = 0)]
        calib_tokens: usize,
        /// GPTQ：是否按激活重要性重排序输入通道（act-order）。
        /// 开 = 高重要性通道先量化、把误差甩给后面的通道，同一位宽下加权误差更低
        #[arg(long, default_value = "true", value_parser = ["true", "false"])]
        act_order: String,
        /// GPTQ：Hessian 阻尼系数 `H += damp·(trace(H)/n)·I`，把最小特征值抬离 0。
        /// 0 = 不加阻尼（病态 H 下 Cholesky 失败，该层退回 RTN）
        #[arg(long, default_value_t = crate::quant::GPTQ_DAMP)]
        damp: f32,
        /// GPTQ：分块大小（按输入通道计），0 = 不分块。
        /// 只影响补偿量的批处理粒度，不改变数学结果（见 `gptq_block_size_does_not_change_result`）
        #[arg(long, default_value_t = crate::quant::GPTQ_BLOCK)]
        block: usize,
        /// AWQ 的缩放指数 α（`s = mean|x|^α`）：0 = 关闭缩放（等价 RTN）；
        /// 不给 = 逐层在 0~1 的网格上按代理误差搜索最优 α（每层可能选到不同的值）
        #[arg(long)]
        alpha: Option<f32>,
        /// 量化后 checkpoint 的输出路径（缺省 out_dir/quant.ckpt）
        #[arg(long)]
        out: Option<String>,
        /// 额外打印量化前后的验证集 loss / 困惑度对比
        #[arg(long)]
        eval: bool,
    },
    /// 分布式训练实验（第 38 课）：集合通信通信量对照 + DP/ZeRO 训练轨迹 + TP/PP/3D 切分报告
    Distributed {
        #[command(flatten)]
        dist: DistArgs,
    },
    /// 对齐实验（第 36 课）：奖励模型排序准确率 + DPO 偏好边界 + GRPO 组内优势 + PPO 裁剪与 KL
    Align {
        #[command(flatten)]
        align: AlignArgs,
    },
    /// 检索增强生成实验（第 37 课）：分块不变量 + 三种向量化对照 + top-k/MMR 检索 + 提示组装
    Rag {
        #[command(flatten)]
        rag: RagArgs,
    },
    /// 推测解码与多 Token 预测实验（第 34/35 课）：无损性核对 + 接受率/加速比 + 缓存不变量
    /// + 首 token 分布等价 + MTP 头当草稿
    Speculative {
        #[command(flatten)]
        spec: SpecArgs,
    },
}

/// 检索增强生成（RAG）实验的参数（第 37 课）。
///
/// 从"怎么切块"到"怎么把检索结果塞进提示"的全流程参数都在这儿：切块的两项决定块的
/// 粒度与边界，`top_k` / MMR 的三项决定挑哪几条，`context_chars` 决定提示里给资料留多少位置。
#[derive(Args, Debug, Clone)]
pub struct RagArgs {
    /// 每块的字符数上限
    #[arg(long, default_value_t = 120)]
    pub chunk_size: usize,
    /// 相邻块重叠的字符数（避免关键句正好落在边界上被劈成两半）
    #[arg(long, default_value_t = 30)]
    pub overlap: usize,
    /// 检索返回的条数 k
    #[arg(long, default_value_t = 4)]
    pub top_k: usize,
    /// MMR 的 λ：1 = 退化成普通 top-k，越小越看重多样性
    #[arg(long, default_value_t = 0.5)]
    pub mmr_lambda: f32,
    /// MMR 的重排池大小（先按相关度取这么多候选再重排）
    #[arg(long, default_value_t = 12)]
    pub mmr_pool: usize,
    /// 特征哈希向量化的维度
    #[arg(long, default_value_t = 512)]
    pub hash_dim: usize,
    /// 提示里参考资料部分允许占用的字符预算
    #[arg(long, default_value_t = 300)]
    pub context_chars: usize,
    /// 检索用的查询语句
    #[arg(long, default_value = "the key opens the door to the hidden garden")]
    pub query: String,
    /// 稠密向量那一节的小模型宽度
    #[arg(long, default_value_t = 32)]
    pub n_embd: usize,
    /// 稠密向量那一节的小模型层数
    #[arg(long, default_value_t = 2)]
    pub n_layer: usize,
    /// 随机种子（建小模型用）
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// 对齐实验的参数（第 36 课）。
///
/// 四件套共用一个小 GPT 主干：奖励模型与 DPO 各自的训练步数分开给，方便单独观察
/// 「先训好裁判」与「再拿裁判的信号改策略」两个阶段；`beta` / `clip_eps` / `kl_coef`
/// 是三种算法各自的那个"松紧旋钮"。
#[derive(Args, Debug, Clone)]
pub struct AlignArgs {
    /// 奖励模型的训练步数（成对 Bradley-Terry 损失，一步一对样本）
    #[arg(long, default_value_t = 150)]
    pub rm_steps: usize,
    /// DPO 的训练步数
    #[arg(long, default_value_t = 60)]
    pub steps: usize,
    /// DPO 的 β：允许策略偏离参考模型多远（越小越保守）
    #[arg(long, default_value_t = 0.1)]
    pub beta: f32,
    /// PPO / GRPO 的裁剪范围 ε：重要性比超出 1±ε 的部分不再给梯度
    #[arg(long, default_value_t = 0.2)]
    pub clip_eps: f32,
    /// PPO 的 KL 惩罚系数：把策略拴在参考模型附近的绳子的松紧
    #[arg(long, default_value_t = 0.05)]
    pub kl_coef: f32,
    /// GRPO 每道 prompt 采几条回答（组内相对优势需要一组样本）
    #[arg(long, default_value_t = 4)]
    pub group_size: usize,
    /// 奖励模型的学习率（成对损失很陡，比策略用的大一档，几十步就能把边界拉开）
    #[arg(long, default_value_t = 3e-3)]
    pub rm_lr: f32,
    /// 策略（DPO）的学习率
    #[arg(long, default_value_t = 1e-3)]
    pub lr: f32,
    /// 批大小
    #[arg(long, default_value_t = 2)]
    pub batch_size: usize,
    /// 上下文长度
    #[arg(long, default_value_t = 32)]
    pub block_size: usize,
    /// 隐藏维度
    #[arg(long, default_value_t = 16)]
    pub n_embd: usize,
    /// Transformer 层数
    #[arg(long, default_value_t = 2)]
    pub n_layer: usize,
    /// 随机种子（参考模型 / 策略 / 奖励模型共用，保证可复现）
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// 分布式实验的参数（第 38 课）。
///
/// 三个轴各一个度：`dp` 决定数据并行组与 ZeRO 的分片数，`tp` 决定层内权重切几份，
/// `pp` 决定层链切几段；三者相乘就是这次实验的**卡数**。
#[derive(Args, Debug, Clone)]
pub struct DistArgs {
    /// 数据并行度（也是集合通信的 rank 数、ZeRO 的分片数）
    #[arg(long, default_value_t = 4)]
    pub dp: usize,
    /// 张量并行度（层内权重按列/行切几份）
    #[arg(long, default_value_t = 2)]
    pub tp: usize,
    /// 流水线并行度（把层链切成几段）
    #[arg(long, default_value_t = 2)]
    pub pp: usize,
    /// 流水线 micro-batch 数（越多气泡占比越低，但激活驻留越多）
    #[arg(long, default_value_t = 4)]
    pub micro_batches: usize,
    /// DP / ZeRO 对照实验的训练步数
    #[arg(long, default_value_t = 40)]
    pub steps: usize,
    /// **每 rank** 的批大小（全局 batch = 它 × 数据并行度）
    #[arg(long, default_value_t = 2)]
    pub batch_size: usize,
    /// 上下文长度
    #[arg(long, default_value_t = 32)]
    pub block_size: usize,
    /// 学习率
    #[arg(long, default_value_t = 0.05)]
    pub lr: f32,
    /// 权重衰减（必须非 0：否则抓不到 ZeRO 分片忘记灌真实 θ 时衰减项静默失效）
    #[arg(long, default_value_t = 0.01)]
    pub weight_decay: f32,
    /// 隐藏维度
    #[arg(long, default_value_t = 32)]
    pub n_embd: usize,
    /// Transformer 层数
    #[arg(long, default_value_t = 2)]
    pub n_layer: usize,
    /// 随机种子（各组共用，保证初始权重一致、可比）
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
}

/// 推测解码与多 Token 预测实验的参数（第 34/35 课）。
///
/// 核心旋钮只有两个：`gamma`（草稿每轮猜几个）与 `mtp_heads`（MTP 头数，也就是
/// 一次前向最多给出几个候选）。`trials` 是首 token 分布检验的独立试验次数——
/// 分布等价的判据要靠经验频率逼近真实分布，次数太少（几百）会抖出假阳性。
#[derive(Args, Debug, Clone)]
pub struct SpecArgs {
    /// 每轮生成的 token 上限（含白拿的那个）
    #[arg(long, default_value_t = 48)]
    pub max_new: usize,
    /// 草稿每轮提出的候选数 γ
    #[arg(long, default_value_t = 4)]
    pub gamma: usize,
    /// MTP 预测头数 K（往后预测 K 个 token）
    #[arg(long, default_value_t = 4)]
    pub mtp_heads: usize,
    /// MTP 头的训练步数
    #[arg(long, default_value_t = 80)]
    pub mtp_steps: usize,
    /// MTP 头训练用的批大小
    #[arg(long, default_value_t = 2)]
    pub batch_size: usize,
    /// 首 token 分布检验的独立试验次数
    #[arg(long, default_value_t = 2000)]
    pub trials: usize,
    /// 采样温度（贪心对照那几节固定用 top-k=1，不受它影响）
    #[arg(long, default_value_t = 0.8)]
    pub temperature: f32,
    /// 上下文长度
    #[arg(long, default_value_t = 64)]
    pub block_size: usize,
    /// 隐藏维度
    #[arg(long, default_value_t = 32)]
    pub n_embd: usize,
    /// Transformer 层数
    #[arg(long, default_value_t = 2)]
    pub n_layer: usize,
    /// 随机种子（目标模型与草稿模型分别在 `seed` / `seed + 1` 上初始化）
    #[arg(long, default_value_t = 42)]
    pub seed: u64,
    /// 生成用的提示词
    #[arg(long, default_value = "the key opens")]
    pub prompt: String,
}

impl Cli {
    pub fn parse_args() -> Self {
        Cli::parse()
    }
}
