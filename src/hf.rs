//! **HuggingFace 权重导入**：`config.json` + `*.safetensors` → 本仓库的原生权重。
//!
//! ## 为什么是"导入器"而不是"运行时加载 HF 格式"
//!
//! HF 权重与本仓库的差异不只是文件格式，还有**语义**：
//!
//! | 维度 | 本仓库 | HF（LLaMA / Qwen2 系） |
//! |------|--------|------------------------|
//! | Linear 权重布局 | `[in, out]`（`y = x·W`） | `[out, in]`（PyTorch 口径） |
//! | RoPE 配对 | 相邻配对 `(2i, 2i+1)` | `rotate_half`（`i` 与 `i+h/2`） |
//! | 参数名 | `blocks.0.attn.c_q.weight` | `model.layers.0.self_attn.q_proj.weight` |
//! | 存档格式 | `LLMCP2`（自定义二进制） | safetensors |
//!
//! 前两项**必须逐位对齐**，否则输出是静默错误的垃圾（不会报错）。所以这里做的是
//! **一次性转换**：转换时把 `rope_style = half`、`qkv_bias`、`tie_embeddings` 等语义
//! 一起写进存档头，之后训练 / 推理 / 量化 / LoRA **全部走既有原生路径**，不再有
//! "HF 分支"这种东西。现有自研权重完全不受影响（新字段都带默认值，
//! `#[serde(default)]` 保证旧存档读进来语义不变）。
//!
//! ## 支持面（明确划界，不满足就报错，绝不静默降级）
//!
//! - 结构：decoder-only + RMSNorm + SwiGLU(silu) + GQA/MHA + `rotate_half`，
//!   即 LLaMA / LLaMA-2 / LLaMA-3（不含 3.1+ 的 llama3 型 rope scaling）/ TinyLlama /
//!   Qwen2 / Qwen2.5 / Qwen3（带 QK-Norm）；
//! - dtype：F32 / F16 / BF16（其它一律报错）；
//! - 分片：支持 `model-00001-of-0000N.safetensors` 多分片；
//! - **不支持**（检测到即报错）：MoE 权重（Mixtral/Qwen-MoE）、MLA（DeepSeek）、
//!   滑动窗口注意力（Mistral）、部分旋转（`partial_rotary_factor ≠ 1`）、
//!   非 silu 激活、非 RMSNorm、`rope_scaling`（LLaMA-3.1 的 llama3 型等）。
//!
//! ## 内存提醒
//!
//! 导入路径全程 f32：参数量 P 的模型峰值约需 `2P × 4B`（骨架 + HF 侧解出的张量）。
//! TinyLlama-1.1B 约 9GB 可行；8B 级需要 60GB+，请改用更小的模型或先在 HF 侧转成更小精度
//! （本仓库内核只吃 f32，bf16 只是存储层，这条属于后续可做项）。

use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};

use serde::Deserialize;

use crate::model::{Transformer, TransformerConfig};
use crate::rope::{RopeScaling, RopeStyle, ROPE_BASE};

// ==================== 半精度解码 ====================

/// IEEE 754 半精度（f16）→ f32。
///
/// 手写而不是引依赖：本仓库的依赖表刻意保持精简（没有任何数值/格式库），
/// 而这段只有十几行、且能被穷举测试守住（见 `test_f16_roundtrip_all_bit_patterns`）。
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) & 1) as u32;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign << 31 // ±0
        } else {
            // 次正规：`value = man × 2^-24`（隐含指数 -14、尾数 10 位）。
            // 规格化成 f32：取最高位下标 k（0..=9）⇒ 值 = (1.frac) × 2^(k-24)，
            // 于是 f32 指数域 = 127 + k - 24，尾数 = `man` 去掉隐含位后再补足 23 位。
            let k = 31 - man.leading_zeros();
            let exp32 = (127 - 24 + k as i32) as u32;
            let frac = (man << (23 - k)) & 0x7f_ffff;
            (sign << 31) | (exp32 << 23) | frac
        }
    } else if exp == 0x1f {
        // Inf / NaN：尾数原样搬到 f32 的高位（NaN 载荷保留，便于排查）
        (sign << 31) | 0x7f80_0000 | (man << 13)
    } else {
        (sign << 31) | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// bf16 → f32（bf16 就是 f32 的高 16 位）
pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

/// safetensors 里出现的数据类型。只支持这三种：其余（F8/I8/U8/F64…）在导入时明确报错。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType {
    F32,
    F16,
    Bf16,
}

impl DType {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "F32" => Ok(DType::F32),
            "F16" => Ok(DType::F16),
            "BF16" => Ok(DType::Bf16),
            other => Err(format!(
                "暂不支持的 safetensors dtype `{other}`（只支持 F32 / F16 / BF16；\
                 量化权重请先在 HF 侧反量化回 f16/f32）"
            )),
        }
    }

    fn bytes_per_elem(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 | DType::Bf16 => 2,
        }
    }
}

// ==================== safetensors 读取 ====================

/// 一个张量在文件里的位置与形状
#[derive(Clone, Debug)]
struct Entry {
    dtype: DType,
    shape: Vec<usize>,
    start: usize,
    end: usize,
}

/// safetensors 的**按需读取**器：只把头部 JSON 读进内存，张量在 `read_f32` 时按偏移 seek 读。
///
/// 文件格式（[官方规范](https://github.com/huggingface/safetensors)）：
/// ```text
/// [u64 小端：头部 JSON 字节数 N][N 字节 JSON 头][张量数据段]
/// JSON 头：{ "名字": {"dtype": "F16", "shape": [..], "data_offsets": [start, end]}, ... }
/// ```
/// `data_offsets` 相对**数据段起点**（即 `8 + N`），不是文件起点。
pub struct SafeTensors {
    path: String,
    file: File,
    data_start: u64,
    entries: HashMap<String, Entry>,
}

impl SafeTensors {
    pub fn open(path: &str) -> Result<Self, String> {
        let mut file = File::open(path).map_err(|e| format!("打不开 {path}: {e}"))?;
        let mut len8 = [0u8; 8];
        file.read_exact(&mut len8)
            .map_err(|e| format!("{path} 头部损坏（读不出 8 字节头长）: {e}"))?;
        let n = u64::from_le_bytes(len8) as usize;
        // 头部异常大说明这不是 safetensors（或文件被截断），提前拦掉，避免分配几个 GB
        if n == 0 || n > 1 << 30 {
            return Err(format!("{path} 不是 safetensors：头部长度 {n} 不合理"));
        }
        let mut head = vec![0u8; n];
        file.read_exact(&mut head)
            .map_err(|e| format!("{path} 头部损坏（声明的 {n} 字节读不满）: {e}"))?;
        let json: serde_json::Value = serde_json::from_slice(&head)
            .map_err(|e| format!("{path} 头部不是合法 JSON: {e}"))?;
        let obj = json
            .as_object()
            .ok_or_else(|| format!("{path} 头部应为 JSON 对象"))?;
        let mut entries = HashMap::new();
        for (name, v) in obj {
            // `__metadata__` 是规范允许的字符串表，跳过
            if name == "__metadata__" {
                continue;
            }
            let dtype = DType::parse(
                v.get("dtype")
                    .and_then(|d| d.as_str())
                    .ok_or_else(|| format!("{path} 的 {name} 缺 dtype"))?,
            )?;
            let shape: Vec<usize> = v
                .get("shape")
                .and_then(|s| s.as_array())
                .ok_or_else(|| format!("{path} 的 {name} 缺 shape"))?
                .iter()
                .map(|x| x.as_u64().unwrap_or(0) as usize)
                .collect();
            let offs = v
                .get("data_offsets")
                .and_then(|s| s.as_array())
                .ok_or_else(|| format!("{path} 的 {name} 缺 data_offsets"))?;
            let start = offs.first().and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let end = offs.get(1).and_then(|x| x.as_u64()).unwrap_or(0) as usize;
            let want = shape.iter().product::<usize>() * dtype.bytes_per_elem();
            if end <= start || end - start != want {
                return Err(format!(
                    "{path} 的 {name}：data_offsets {start}..{end} 与 shape {shape:?} × {dtype:?} 对不上（应为 {want} 字节）"
                ));
            }
            entries.insert(name.clone(), Entry { dtype, shape, start, end });
        }
        Ok(SafeTensors { path: path.to_string(), file, data_start: (8 + n) as u64, entries })
    }

    pub fn has(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    pub fn names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.entries.keys().cloned().collect();
        v.sort();
        v
    }

    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.entries.get(name).map(|e| e.shape.as_slice())
    }

    /// 读出并解码成 f32（`numel` 与 `shape` 的乘积一致由 `open` 保证）
    pub fn read_f32(&mut self, name: &str) -> Result<Vec<f32>, String> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| format!("{} 里没有张量 {name}", self.path))?
            .clone();
        let bytes = e.end - e.start;
        let mut buf = vec![0u8; bytes];
        self.file
            .seek(SeekFrom::Start(self.data_start + e.start as u64))
            .and_then(|_| self.file.read_exact(&mut buf))
            .map_err(|err| format!("读 {} 的 {name} 失败: {err}", self.path))?;
        Ok(match e.dtype {
            DType::F32 => buf
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
            DType::F16 => buf
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
            DType::Bf16 => buf
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect(),
        })
    }
}

// ==================== HF config.json ====================

/// HF `config.json` 里我们关心的字段（其余一律忽略；用不到的字段靠 `Option` 兜住）
#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct HfConfig {
    pub model_type: Option<String>,
    pub architectures: Vec<String>,
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub max_position_embeddings: Option<usize>,
    pub vocab_size: Option<usize>,
    pub rms_norm_eps: Option<f32>,
    pub layer_norm_eps: Option<f32>,
    pub rope_theta: Option<f32>,
    pub head_dim: Option<usize>,
    pub tie_word_embeddings: Option<bool>,
    pub attention_bias: Option<bool>,
    pub mlp_bias: Option<bool>,
    pub hidden_act: Option<String>,
    // ---- 一律不支持的结构（检测到就报错，绝不静默降级）----
    pub num_local_experts: Option<usize>,
    pub num_experts: Option<usize>,
    pub n_routed_experts: Option<usize>,
    pub kv_lora_rank: Option<usize>,
    pub q_lora_rank: Option<usize>,
    pub sliding_window: Option<usize>,
    /// Qwen2/Qwen2.5 系列会带 `"sliding_window": 32768` **但同时**写
    /// `"use_sliding_window": false`（那只是为了兼容旧代码，实际不启用 SWA）。
    /// 判据因此是"有窗口 **且** 没被显式关掉"，否则会把 Qwen2.5 误杀。
    pub use_sliding_window: Option<bool>,
    pub partial_rotary_factor: Option<f32>,
    pub rope_scaling: Option<serde_json::Value>,
}

impl Default for HfConfig {
    fn default() -> Self {
        HfConfig {
            model_type: None,
            architectures: Vec::new(),
            hidden_size: 0,
            num_attention_heads: 0,
            num_key_value_heads: None,
            num_hidden_layers: 0,
            intermediate_size: 0,
            max_position_embeddings: None,
            vocab_size: None,
            rms_norm_eps: None,
            layer_norm_eps: None,
            rope_theta: None,
            head_dim: None,
            tie_word_embeddings: None,
            attention_bias: None,
            mlp_bias: None,
            hidden_act: None,
            num_local_experts: None,
            num_experts: None,
            n_routed_experts: None,
            kv_lora_rank: None,
            q_lora_rank: None,
            sliding_window: None,
            use_sliding_window: None,
            partial_rotary_factor: None,
            rope_scaling: None,
        }
    }
}

impl HfConfig {
    /// 结构层面的硬性检查：命中任何一条都直接报错（这些都不是"能凑合跑"的差异）
    pub fn check_supported(&self) -> Result<(), String> {
        for (k, v) in [
            ("num_local_experts", self.num_local_experts),
            ("num_experts", self.num_experts),
            ("n_routed_experts", self.n_routed_experts),
        ] {
            if v.unwrap_or(0) > 1 {
                return Err(format!(
                    "这是 MoE 模型（{k} = {}）：本仓库的专家结构与 HF 的 MoE 权重布局不通用，暂不支持导入",
                    v.unwrap()
                ));
            }
        }
        if self.kv_lora_rank.unwrap_or(0) > 0 || self.q_lora_rank.unwrap_or(0) > 0 {
            return Err(
                "这是 MLA（低秩 KV 压缩）模型：权重布局与本仓库的 MLA 实现不通用，暂不支持导入"
                    .to_string(),
            );
        }
        if self.sliding_window.unwrap_or(0) > 0 && self.use_sliding_window != Some(false) {
            return Err(format!(
                "这是滑动窗口注意力模型（sliding_window = {}）：本仓库的注意力没有实现 SWA，\
                 强行导入会得到与 HF 不一致的输出。（Qwen2.5 那种带窗口但 `use_sliding_window: false` 的不算）",
                self.sliding_window.unwrap()
            ));
        }
        if let Some(p) = self.partial_rotary_factor {
            if (p - 1.0).abs() > 1e-6 {
                return Err(format!(
                    "partial_rotary_factor = {p} ≠ 1：只对一部分维度做 RoPE，本仓库的普通路径不支持（MLA 解耦 RoPE 口径不同）"
                ));
            }
        }
        if let Some(rs) = &self.rope_scaling {
            if !rs.is_null() {
                return Err(format!(
                    "config.json 带 rope_scaling（{rs}）：本仓库只实现了 linear / ntk / yarn 三种，\
                     HF 的 llama3 / longrope / dynamic 等口径不同，需要先手工换算（见 docs/20「实现更正：频率表是逐头重复的」一节）"
                ));
            }
        }
        match self.hidden_act.as_deref() {
            None | Some("silu") | Some("swish") => {}
            Some(other) => {
                return Err(format!(
                    "hidden_act = `{other}`：本仓库的 FFN 只实现了 SwiGLU(silu) 与 GELU 两种，\
                     且 GELU 路径的参数名与 HF 的 gate/up/down 布局不同"
                ));
            }
        }
        if self.hidden_size == 0 || self.num_attention_heads == 0 || self.num_hidden_layers == 0 {
            return Err(
                "config.json 缺少必需字段（hidden_size / num_attention_heads / num_hidden_layers）"
                    .to_string(),
            );
        }
        Ok(())
    }

    /// 映射成本仓库的模型配置。
    ///
    /// 四个 bias/绑定开关**以张量是否存在为准**（地面真相），config 里的同名布尔只用来交叉核对：
    /// HF 各家在这些字段上的默认值不一致，猜错的后果是"少读一个 bias / 少一份 lm_head"。
    #[allow(clippy::too_many_arguments)]
    pub fn to_config(
        &self,
        has_lm_head: bool,
        qkv_bias: bool,
        o_bias: bool,
        mlp_bias: bool,
        qk_norm: bool,
    ) -> Result<TransformerConfig, String> {
        // `tie_word_embeddings` 与张量存在性不一致时**不报错**：以张量为准。
        // 很多 tied 模型（Qwen2.5-0.5B 等）仍会把 `lm_head.weight` 一起存下来，
        // 那份权重与词嵌入等价，按"独立输出头"读进来在数值上完全一样；
        // 反过来 config 说 true 而文件里确实没有 lm_head，则绑定词嵌入也对。
        let n_head = self.num_attention_heads;
        let n_kv = self.num_key_value_heads.unwrap_or(n_head);
        if n_kv == 0 || n_head % n_kv != 0 {
            return Err(format!("n_head = {n_head} 必须能被 num_key_value_heads = {n_kv} 整除"));
        }
        let vocab = self
            .vocab_size
            .ok_or_else(|| "config.json 缺 vocab_size（无法确定词表大小）".to_string())?;
        let head_dim = self.head_dim.unwrap_or(0);
        if head_dim > 0 && head_dim * n_head != self.hidden_size {
            return Err(format!(
                "head_dim = {head_dim} × n_head = {n_head} ≠ hidden_size = {}：本仓库的残差流按 hidden_size 走，\
                 这种配置需要改写注意力输出拼接，暂不支持",
                self.hidden_size
            ));
        }
        let rope_base = self.rope_theta.unwrap_or(ROPE_BASE);
        let cfg = TransformerConfig {
            vocab_size: vocab,
            n_embd: self.hidden_size,
            n_head,
            n_kv_head: n_kv,
            n_layer: self.num_hidden_layers,
            block_size: self.max_position_embeddings.unwrap_or(2048),
            head_dim,
            ffn_hidden: self.intermediate_size,
            qkv_bias,
            o_bias,
            mlp_bias,
            rms_eps: self.rms_norm_eps.or(self.layer_norm_eps).unwrap_or(1e-5),
            tie_embeddings: !has_lm_head,
            qk_norm,
            rope_base,
            rope_scaling: RopeScaling::None,
            rope_style: RopeStyle::Half,
            use_rmsnorm: true,
            use_swiglu: true,
            dropout: 0.0,
            ..Default::default()
        };
        Ok(cfg)
    }
}

// ==================== 名字映射 ====================

/// 一条映射：目标名 + 是否需要转置（HF 的 Linear 是 `[out, in]`，本仓库是 `[in, out]`）
#[derive(Debug, PartialEq, Eq)]
pub struct Mapped {
    pub ours: String,
    pub transpose: bool,
}

/// HF 权重名 → 本仓库参数名。返回 `Ok(None)` 表示"按设计忽略"。
///
/// **未识别的名字一律报错**：静默跳过会让"权重没装上"变成"模型能跑但输出乱"，
/// 那是最难定位的一类问题。
pub fn map_hf_name(name: &str) -> Result<Option<Mapped>, String> {
    let m = |ours: String, transpose: bool| Ok(Some(Mapped { ours, transpose }));
    // 顶层
    match name {
        "model.embed_tokens.weight" => return m("tok_emb.table".to_string(), false),
        "lm_head.weight" => return m("head.weight".to_string(), false),
        "model.norm.weight" => return m("ln_f.gamma".to_string(), false),
        "model.norm.bias" => return m("ln_f.beta".to_string(), false),
        // 旋转角度的频率表是算出来的，HF 里带一份也不该读
        n if n.ends_with(".rotary_emb.inv_freq") || n.ends_with(".rotary_emb.cos_cached")
            || n.ends_with(".rotary_emb.sin_cached") =>
        {
            return Ok(None)
        }
        _ => {}
    }
    // 层内
    let rest = name
        .strip_prefix("model.layers.")
        .ok_or_else(|| format!("无法识别的 HF 权重名：{name}"))?;
    let (idx, tail) = rest
        .split_once('.')
        .ok_or_else(|| format!("无法解析层号：{name}"))?;
    let i: usize = idx
        .parse()
        .map_err(|_| format!("层号不是数字：{name}"))?;
    let p = format!("blocks.{i}");
    let (ours, transpose) = match tail {
        "input_layernorm.weight" => (format!("{p}.ln1.gamma"), false),
        "input_layernorm.bias" => (format!("{p}.ln1.beta"), false),
        "post_attention_layernorm.weight" => (format!("{p}.ln2.gamma"), false),
        "post_attention_layernorm.bias" => (format!("{p}.ln2.beta"), false),
        "self_attn.q_proj.weight" => (format!("{p}.attn.c_q.weight"), true),
        "self_attn.k_proj.weight" => (format!("{p}.attn.c_k.weight"), true),
        "self_attn.v_proj.weight" => (format!("{p}.attn.c_v.weight"), true),
        "self_attn.o_proj.weight" => (format!("{p}.attn.c_proj.weight"), true),
        "self_attn.q_proj.bias" => (format!("{p}.attn.c_q.bias"), false),
        "self_attn.k_proj.bias" => (format!("{p}.attn.c_k.bias"), false),
        "self_attn.v_proj.bias" => (format!("{p}.attn.c_v.bias"), false),
        "self_attn.o_proj.bias" => (format!("{p}.attn.c_proj.bias"), false),
        "self_attn.q_norm.weight" => (format!("{p}.attn.q_norm.gamma"), false),
        "self_attn.k_norm.weight" => (format!("{p}.attn.k_norm.gamma"), false),
        "mlp.gate_proj.weight" => (format!("{p}.mlp.w_gate.weight"), true),
        "mlp.up_proj.weight" => (format!("{p}.mlp.w_up.weight"), true),
        "mlp.down_proj.weight" => (format!("{p}.mlp.w_down.weight"), true),
        "mlp.gate_proj.bias" => (format!("{p}.mlp.w_gate.bias"), false),
        "mlp.up_proj.bias" => (format!("{p}.mlp.w_up.bias"), false),
        "mlp.down_proj.bias" => (format!("{p}.mlp.w_down.bias"), false),
        // 明确的"不支持"：给出可执行的提示，而不是一句"未知名字"
        "mlp.gate.weight" => {
            return Err(format!(
                "{p}.mlp.gate.weight 是 MoE 路由器（第 {i} 层）：本仓库的 MoE 与 HF 的专家布局不通用，暂不支持"
            ))
        }
        t if t.starts_with("mlp.experts.") => {
            return Err(format!("{p}.{t} 是 MoE 专家权重：暂不支持导入 MoE 模型"))
        }
        t if t.starts_with("self_attn.q_a") || t.starts_with("self_attn.kv_a") => {
            return Err(format!("{p}.{t} 是 MLA 的低秩投影：暂不支持导入 MLA 模型"))
        }
        t if t.starts_with("pre_feedforward")
            || t.starts_with("post_feedforward")
            || t.contains("router")
            || t.contains("sliding") =>
        {
            return Err(format!("{p}.{t} 属于本仓库未实现的层结构（Gemma 式双 norm / 路由器 / SWA）"))
        }
        other => return Err(format!("无法识别的 HF 权重名：{name}（层内片段 `{other}`）")),
    };
    m(ours, transpose)
}

/// 2D 转置：`[rows, cols]` → `[cols, rows]`
fn transpose2(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

// ==================== 一次完整的导入 ====================

/// 导入结果：映射好的模型配置 + 已按**本仓库名字与布局**对齐的张量
pub struct HfModel {
    pub cfg: TransformerConfig,
    /// `(我们的参数名, 形状, 数据)`，形状已是本仓库口径（Linear 为 `[in, out]`）
    pub tensors: Vec<(String, Vec<usize>, Vec<f32>)>,
    /// 导入过程中的回显信息（检测到了什么、忽略了多少键）
    pub notes: Vec<String>,
}

/// 从 HF 模型目录导入（`config.json` + 一个或多个 `*.safetensors`）
pub fn load_hf_dir(dir: &str) -> Result<HfModel, String> {
    let cfg_path = std::path::Path::new(dir).join("config.json");
    let text = std::fs::read_to_string(&cfg_path)
        .map_err(|e| format!("读不到 {}: {e}（--model-dir 应指向 HF 模型目录）", cfg_path.display()))?;
    let hf: HfConfig = serde_json::from_str(&text)
        .map_err(|e| format!("解析 {} 失败: {e}", cfg_path.display()))?;
    hf.check_supported()?;

    // 收集 safetensors 分片（HF 大于 ~5GB 会切成 model-00001-of-0000N.safetensors）
    let mut files: Vec<String> = Vec::new();
    let rd = std::fs::read_dir(dir).map_err(|e| format!("列不出目录 {dir}: {e}"))?;
    for e in rd {
        let p = e.map_err(|e| format!("读目录项失败: {e}"))?.path();
        if p.extension().map(|x| x == "safetensors").unwrap_or(false) {
            files.push(p.to_string_lossy().to_string());
        }
    }
    files.sort();
    if files.is_empty() {
        // 给最常见的两种"名字不对"一个明确提示
        let has_bin = std::fs::read_dir(dir)
            .map(|rd| {
                rd.flatten()
                    .any(|e| e.file_name().to_string_lossy().ends_with(".bin"))
            })
            .unwrap_or(false);
        return Err(if has_bin {
            format!("{dir} 里只有 pytorch_model.bin：请先在 HF 侧另存为 safetensors（`save_pretrained(..., safe_serialization=True)`）")
        } else {
            format!("{dir} 里没有 *.safetensors")
        });
    }
    let mut shards: Vec<SafeTensors> = Vec::new();
    for f in &files {
        shards.push(SafeTensors::open(f)?);
    }

    // 结构探测：**以张量是否存在为准**（见 `HfConfig::to_config` 的说明）
    let find = |name: &str| -> Option<usize> { shards.iter().position(|s| s.has(name)) };
    let has_lm_head = find("lm_head.weight").is_some();
    let qkv_bias = find("model.layers.0.self_attn.q_proj.bias").is_some();
    let o_bias = find("model.layers.0.self_attn.o_proj.bias").is_some();
    let mlp_bias = find("model.layers.0.mlp.gate_proj.bias").is_some();
    let qk_norm = find("model.layers.0.self_attn.q_norm.weight").is_some();
    let cfg = hf.to_config(has_lm_head, qkv_bias, o_bias, mlp_bias, qk_norm)?;

    // 逐张量映射（跨分片）
    let mut tensors: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
    let mut ignored = 0usize;
    let mut all_names: Vec<(usize, String)> = Vec::new();
    for (si, s) in shards.iter().enumerate() {
        for n in s.names() {
            all_names.push((si, n));
        }
    }
    for (si, name) in all_names {
        match map_hf_name(&name)? {
            None => {
                ignored += 1;
                continue;
            }
            Some(m) => {
                let shape = shards[si]
                    .shape(&name)
                    .ok_or_else(|| format!("内部错误：{name} 形状丢失"))?
                    .to_vec();
                let v = shards[si].read_f32(&name)?;
                let (shape, data) = if m.transpose {
                    if shape.len() != 2 {
                        return Err(format!("{name} 应给 2D 权重，实际形状 {shape:?}"));
                    }
                    let t = transpose2(&v, shape[0], shape[1]);
                    (vec![shape[1], shape[0]], t)
                } else {
                    (shape, v)
                };
                tensors.push((m.ours, shape, data));
            }
        }
    }
    let mut notes = vec![
        format!(
            "结构：{} 层 / d={} / {} 头（KV {} 头）/ ffn={}",
            cfg.n_layer, cfg.n_embd, cfg.n_head, cfg.n_kv_head, cfg.ffn_hidden
        ),
        format!(
            "口径：rope_style={} base={} | qkv_bias={qkv_bias} o_bias={o_bias} mlp_bias={mlp_bias} | 输出头{}",
            cfg.rope_style.describe(),
            cfg.rope_base,
            if cfg.tie_embeddings { "绑定词嵌入" } else { "独立（head.weight）" }
        ),
    ];
    if ignored > 0 {
        notes.push(format!("按设计忽略 {ignored} 个键（旋转频率表缓存等）"));
    }
    Ok(HfModel { cfg, tensors, notes })
}

/// 把导入的张量**严格**灌进模型：每个模型参数都必须有对应数据、形状必须一致，
/// 且导入侧不能有剩余张量。任何一条不满足都报错并列出明细。
///
/// 之所以这么严：`set_data` 是按共享缓冲写入的，少灌一个张量不会 panic，
/// 只会得到一个"一半是随机初始化"的模型——那比崩掉难查得多。
pub fn assign_named_parameters(
    model: &Transformer,
    tensors: &[(String, Vec<usize>, Vec<f32>)],
) -> Result<usize, String> {
    let want = model.named_parameters();
    let mut got: HashMap<&str, (&[usize], &[f32])> = HashMap::new();
    for (n, s, d) in tensors {
        if got.insert(n.as_str(), (s.as_slice(), d.as_slice())).is_some() {
            return Err(format!("导入数据里出现重复参数名 {n}"));
        }
    }
    let mut missing = Vec::new();
    let mut bad_shape = Vec::new();
    let mut used = 0usize;
    for (name, t) in &want {
        match got.get(name.as_str()) {
            None => missing.push(name.clone()),
            Some((s, d)) => {
                if *s != t.shape() {
                    bad_shape.push(format!("{name}：导入 {:?} vs 模型 {:?}", s, t.shape()));
                } else if d.len() != t.numel() {
                    bad_shape.push(format!("{name}：数据 {} 个元素 vs 模型 {} 个", d.len(), t.numel()));
                } else {
                    t.set_data(d.to_vec());
                    used += 1;
                }
            }
        }
    }
    let extra: Vec<&str> = got
        .keys()
        .filter(|n| !want.iter().any(|(w, _)| w == *n))
        .copied()
        .collect();
    if !missing.is_empty() || !bad_shape.is_empty() || !extra.is_empty() {
        let mut msg = String::from("导入对账失败：\n");
        if !missing.is_empty() {
            msg.push_str(&format!("  模型需要但导入数据里没有（{} 个）：{:?}\n", missing.len(), &missing[..missing.len().min(8)]));
        }
        if !bad_shape.is_empty() {
            msg.push_str(&format!("  形状不一致（{} 个）：{:?}\n", bad_shape.len(), &bad_shape[..bad_shape.len().min(8)]));
        }
        if !extra.is_empty() {
            msg.push_str(&format!("  导入数据里多出来（{} 个）：{:?}\n", extra.len(), &extra[..extra.len().min(8)]));
        }
        return Err(msg);
    }
    Ok(used)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    // ---------- 半精度解码 ----------

    #[test]
    fn test_f16_known_values() {
        // 1.0 = 0x3C00、-2.0 = 0xC000、0.5 = 0x3800、0.0 = 0x0000
        assert_eq!(f16_to_f32(0x3c00), 1.0);
        assert_eq!(f16_to_f32(0xc000), -2.0);
        assert_eq!(f16_to_f32(0x3800), 0.5);
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert!(f16_to_f32(0x8000).is_sign_negative() && f16_to_f32(0x8000) == 0.0);
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // 最小次正规 = 2^-24
        assert!((f16_to_f32(0x0001) - 2f32.powi(-24)).abs() < 1e-12);
        // 最大正规 65504
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }

    /// f16 → f32 必须与"按位构造"逐一相符（穷举 65536 个码，抓边界与次正规）
    #[test]
    fn test_f16_roundtrip_all_bit_patterns() {
        for h in 0..=u16::MAX {
            let got = f16_to_f32(h);
            // 参考：用 f32 的位运算重新推导一遍（与实现同构但独立写出，防止抄错）
            let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let exp = (h >> 10) & 0x1f;
            let man = h & 0x3ff;
            let want = match (exp, man) {
                (0, 0) => sign * 0.0,
                (0, m) => sign * (m as f32) * 2f32.powi(-24),
                (0x1f, 0) => sign * f32::INFINITY,
                (0x1f, _) => f32::NAN,
                (e, m) => sign * (1.0 + m as f32 / 1024.0) * 2f32.powi(e as i32 - 15),
            };
            if want.is_nan() {
                assert!(got.is_nan(), "码 {h:#06x} 应为 NaN，实得 {got}");
            } else {
                assert_eq!(got.to_bits(), want.to_bits(), "码 {h:#06x}: {got} vs {want}");
            }
        }
    }

    // ---------- safetensors ----------

    /// 测试用 safetensors 写入（只写 F32/F16，够用即可）
    fn write_st(path: &str, tensors: &[(&str, Vec<usize>, DType, Vec<u8>)]) {
        let mut head = serde_json::Map::new();
        let mut off = 0usize;
        let mut body = Vec::new();
        for (name, shape, dt, bytes) in tensors {
            head.insert(
                name.to_string(),
                serde_json::json!({
                    "dtype": match dt { DType::F32 => "F32", DType::F16 => "F16", DType::Bf16 => "BF16" },
                    "shape": shape,
                    "data_offsets": [off, off + bytes.len()],
                }),
            );
            off += bytes.len();
            body.extend_from_slice(bytes);
        }
        let hj = serde_json::to_vec(&serde_json::Value::Object(head)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hj.len() as u64).to_le_bytes());
        out.extend_from_slice(&hj);
        out.extend_from_slice(&body);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn test_safetensors_reads_and_decodes() {
        let dir = std::env::temp_dir().join("rust_llm_hf_st_test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("a.safetensors");
        let ps = p.to_string_lossy().to_string();
        let f32s: Vec<f32> = vec![1.0, -2.5, 3.25, 0.0];
        let f32b: Vec<u8> = f32s.iter().flat_map(|v| v.to_le_bytes()).collect();
        let f16b: Vec<u8> = [0x3c00u16, 0xc000, 0x3800].iter().flat_map(|v| v.to_le_bytes()).collect();
        let bf16b: Vec<u8> = [0x3f80u16, 0xc000, 0x0000].iter().flat_map(|v| v.to_le_bytes()).collect();
        write_st(
            &ps,
            &[
                ("w", vec![2, 2], DType::F32, f32b),
                ("h", vec![3], DType::F16, f16b),
                ("b", vec![3], DType::Bf16, bf16b),
            ],
        );
        let mut st = SafeTensors::open(&ps).unwrap();
        assert_eq!(st.names(), vec!["b".to_string(), "h".to_string(), "w".to_string()]);
        assert_eq!(st.read_f32("w").unwrap(), f32s);
        assert_eq!(st.read_f32("h").unwrap(), vec![1.0, -2.0, 0.5]);
        assert_eq!(st.read_f32("b").unwrap(), vec![1.0, -2.0, 0.0]);
        assert_eq!(st.shape("w").unwrap(), &[2, 2]);
        // 未知名字必须报错（不能静默给空）
        assert!(st.read_f32("nope").is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_safetensors_rejects_bad_header() {
        let dir = std::env::temp_dir().join("rust_llm_hf_st_bad");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("bad.safetensors").to_string_lossy().to_string();
        // 头部声明的长度与实际不符
        std::fs::write(&p, 1000u64.to_le_bytes()).unwrap();
        assert!(SafeTensors::open(&p).is_err());
        // 形状与字节数不符
        let head = serde_json::json!({"w": {"dtype": "F32", "shape": [3], "data_offsets": [0, 8]}});
        let hj = serde_json::to_vec(&head).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(hj.len() as u64).to_le_bytes());
        out.extend_from_slice(&hj);
        out.extend_from_slice(&[0u8; 8]);
        std::fs::write(&p, out).unwrap();
        let err = match SafeTensors::open(&p) {
            Ok(_) => panic!("形状与字节数不符时必须报错"),
            Err(e) => e,
        };
        assert!(err.contains("对不上"), "错误信息应指出形状/字节数不符：{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    // ---------- 名字映射 ----------

    #[test]
    fn test_name_mapping_table() {
        let cases = [
            ("model.embed_tokens.weight", "tok_emb.table", false),
            ("lm_head.weight", "head.weight", false),
            ("model.norm.weight", "ln_f.gamma", false),
            ("model.layers.0.input_layernorm.weight", "blocks.0.ln1.gamma", false),
            ("model.layers.3.post_attention_layernorm.weight", "blocks.3.ln2.gamma", false),
            ("model.layers.1.self_attn.q_proj.weight", "blocks.1.attn.c_q.weight", true),
            ("model.layers.1.self_attn.k_proj.weight", "blocks.1.attn.c_k.weight", true),
            ("model.layers.1.self_attn.v_proj.weight", "blocks.1.attn.c_v.weight", true),
            ("model.layers.1.self_attn.o_proj.weight", "blocks.1.attn.c_proj.weight", true),
            ("model.layers.1.self_attn.q_proj.bias", "blocks.1.attn.c_q.bias", false),
            ("model.layers.1.self_attn.q_norm.weight", "blocks.1.attn.q_norm.gamma", false),
            ("model.layers.2.mlp.gate_proj.weight", "blocks.2.mlp.w_gate.weight", true),
            ("model.layers.2.mlp.up_proj.weight", "blocks.2.mlp.w_up.weight", true),
            ("model.layers.2.mlp.down_proj.weight", "blocks.2.mlp.w_down.weight", true),
            ("model.layers.2.mlp.down_proj.bias", "blocks.2.mlp.w_down.bias", false),
        ];
        for (hf, ours, tr) in cases {
            let got = map_hf_name(hf).unwrap().unwrap();
            assert_eq!(got.ours, ours, "{hf}");
            assert_eq!(got.transpose, tr, "{hf} 的转置标志");
        }
        // 按设计忽略
        assert!(map_hf_name("model.layers.0.self_attn.rotary_emb.inv_freq").unwrap().is_none());
        // 不支持 / 未识别：必须报错且信息可读
        for bad in [
            "model.layers.0.mlp.gate.weight",
            "model.layers.0.mlp.experts.0.w1.weight",
            "model.layers.0.self_attn.q_a_proj.weight",
            "model.layers.0.pre_feedforward_layernorm.weight",
            "some.random.key",
        ] {
            let e = map_hf_name(bad).unwrap_err();
            assert!(!e.is_empty(), "{bad} 应给出非空错误信息");
        }
    }

    // ---------- 端到端：合成一个 tiny LLaMA 目录 ----------

    /// 造一个"HF 口味"的 tiny LLaMA 目录：1 层 / d=8 / 2 头 / head_dim=4 / ffn=16 / vocab=12，
    /// 权重值都是**精心挑选的非对称数**，用来验证转置确实生效（对称矩阵会掩盖转置 bug）。
    fn write_tiny_hf(dir: &std::path::Path, with_lm_head: bool, dtype: DType) {
        std::fs::create_dir_all(dir).unwrap();
        let cfg = serde_json::json!({
            "model_type": "llama",
            "architectures": ["LlamaForCausalLM"],
            "hidden_size": 8,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "num_hidden_layers": 1,
            "intermediate_size": 16,
            "max_position_embeddings": 32,
            "vocab_size": 12,
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "hidden_act": "silu",
            "attention_bias": false,
            "mlp_bias": false,
            "tie_word_embeddings": !with_lm_head,
        });
        std::fs::write(dir.join("config.json"), serde_json::to_string_pretty(&cfg).unwrap()).unwrap();

        // 用确定性公式造权重：值 = f(名字, 下标)，非对称且各不相同
        let mut mk = |name: &str, rows: usize, cols: usize| -> (Vec<usize>, Vec<u8>) {
            let n = rows * cols;
            let vals: Vec<f32> = (0..n)
                .map(|k| {
                    let seed = name.bytes().map(|b| b as u32).sum::<u32>();
                    (((seed + k as u32 * 7) % 97) as f32) / 97.0 - 0.5
                })
                .collect();
            let bytes: Vec<u8> = match dtype {
                DType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F16 => vals.iter().map(|v| f32_to_f16_for_test(*v)).flat_map(|h| h.to_le_bytes()).collect(),
                DType::Bf16 => vals
                    .iter()
                    .map(|v| ((v.to_bits() >> 16) as u16))
                    .flat_map(|h| h.to_le_bytes())
                    .collect(),
            };
            (vec![rows, cols], bytes)
        };
        // 归一化权重要"够非平凡"但不能把数值搞爆：用小的正数
        let mut mk_norm = |name: &str, n: usize| -> (Vec<usize>, Vec<u8>) {
            let vals: Vec<f32> = (0..n).map(|k| 0.5 + (k % 5) as f32 * 0.1).collect();
            let bytes: Vec<u8> = match dtype {
                DType::F32 => vals.iter().flat_map(|v| v.to_le_bytes()).collect(),
                DType::F16 => vals.iter().map(|v| f32_to_f16_for_test(*v)).flat_map(|h| h.to_le_bytes()).collect(),
                DType::Bf16 => vals
                    .iter()
                    .map(|v| ((v.to_bits() >> 16) as u16))
                    .flat_map(|h| h.to_le_bytes())
                    .collect(),
            };
            let _ = name;
            (vec![n], bytes)
        };

        let mut ts: Vec<(&str, Vec<usize>, DType, Vec<u8>)> = Vec::new();
        let mut push = |ts: &mut Vec<(&str, Vec<usize>, DType, Vec<u8>)>, name: &'static str, s: (Vec<usize>, Vec<u8>)| {
            ts.push((name, s.0, dtype, s.1));
        };
        push(&mut ts, "model.embed_tokens.weight", mk("embed", 12, 8));
        if with_lm_head {
            push(&mut ts, "lm_head.weight", mk("lmhead", 12, 8));
        }
        push(&mut ts, "model.norm.weight", mk_norm("norm", 8));
        push(&mut ts, "model.layers.0.input_layernorm.weight", mk_norm("ln1", 8));
        push(&mut ts, "model.layers.0.post_attention_layernorm.weight", mk_norm("ln2", 8));
        // HF 的 Linear 是 [out, in]
        push(&mut ts, "model.layers.0.self_attn.q_proj.weight", mk("q", 8, 8));
        push(&mut ts, "model.layers.0.self_attn.k_proj.weight", mk("k", 4, 8));
        push(&mut ts, "model.layers.0.self_attn.v_proj.weight", mk("v", 4, 8));
        push(&mut ts, "model.layers.0.self_attn.o_proj.weight", mk("o", 8, 8));
        push(&mut ts, "model.layers.0.mlp.gate_proj.weight", mk("gate", 16, 8));
        push(&mut ts, "model.layers.0.mlp.up_proj.weight", mk("up", 16, 8));
        push(&mut ts, "model.layers.0.mlp.down_proj.weight", mk("down", 8, 16));
        let p = dir.join("model.safetensors").to_string_lossy().to_string();
        write_st(&p, &ts);
    }

    /// 测试用 f32 → f16（就近舍入到最近偶数，够小规模用例使用）
    fn f32_to_f16_for_test(v: f32) -> u16 {
        let bits = v.to_bits();
        let sign = ((bits >> 16) & 0x8000) as u16;
        let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
        let man = bits & 0x7f_ffff;
        if exp <= 0 {
            return sign; // 太小直接归零（测试数据不会走到这里）
        }
        if exp >= 0x1f {
            return sign | 0x7c00;
        }
        // 截断到 10 位尾数（不做舍入，测试里够用；对齐用同一函数即可）
        sign | ((exp as u16) << 10) | ((man >> 13) as u16)
    }

    #[test]
    fn test_load_hf_dir_maps_and_transposes() {
        let dir = std::env::temp_dir().join("rust_llm_hf_tiny");
        let _ = std::fs::remove_dir_all(&dir);
        write_tiny_hf(&dir, true, DType::F32);
        let m = load_hf_dir(&dir.to_string_lossy()).unwrap();
        // 配置映射
        assert_eq!(m.cfg.n_embd, 8);
        assert_eq!(m.cfg.n_head, 2);
        assert_eq!(m.cfg.n_kv_head, 1);
        assert_eq!(m.cfg.n_layer, 1);
        assert_eq!(m.cfg.ffn_hidden, 16);
        assert_eq!(m.cfg.vocab_size, 12);
        assert_eq!(m.cfg.block_size, 32);
        assert_eq!(m.cfg.rope_style, RopeStyle::Half, "HF 权重必须按 rotate_half 口径解释");
        assert_eq!(m.cfg.rms_eps, 1e-5);
        assert!(!m.cfg.qkv_bias && !m.cfg.o_bias && !m.cfg.mlp_bias, "LLaMA 无 bias");
        assert!(!m.cfg.tie_embeddings, "有 lm_head 就应是不绑定");

        // 名字与形状（Linear 必须是 [in, out]）
        let find = |n: &str| m.tensors.iter().find(|(x, _, _)| x == n).map(|(_, s, d)| (s.clone(), d.clone()));
        assert_eq!(find("tok_emb.table").unwrap().0, vec![12, 8]);
        assert_eq!(find("head.weight").unwrap().0, vec![12, 8]);
        assert_eq!(find("blocks.0.attn.c_q.weight").unwrap().0, vec![8, 8], "q_proj 转置后应仍是 8×8");
        assert_eq!(find("blocks.0.attn.c_k.weight").unwrap().0, vec![8, 4], "k_proj [4,8] → [8,4]");
        assert_eq!(find("blocks.0.mlp.w_gate.weight").unwrap().0, vec![8, 16]);
        assert_eq!(find("blocks.0.mlp.w_down.weight").unwrap().0, vec![16, 8]);
        assert_eq!(find("ln_f.gamma").unwrap().0, vec![8]);

        // **转置必须真的发生过**：用非对称矩阵核对
        let mut st = SafeTensors::open(&dir.join("model.safetensors").to_string_lossy()).unwrap();
        let hf_k = st.read_f32("model.layers.0.self_attn.k_proj.weight").unwrap(); // [4, 8]
        let (_, ours_k) = find("blocks.0.attn.c_k.weight").unwrap(); // [8, 4]
        for r in 0..4 {
            for c in 0..8 {
                assert_eq!(ours_k[c * 4 + r], hf_k[r * 8 + c], "转置错位 (r={r}, c={c})");
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_hf_dir_supports_f16_and_tied_head() {
        let dir = std::env::temp_dir().join("rust_llm_hf_tiny_f16");
        let _ = std::fs::remove_dir_all(&dir);
        write_tiny_hf(&dir, false, DType::F16);
        let m = load_hf_dir(&dir.to_string_lossy()).unwrap();
        assert!(m.cfg.tie_embeddings, "没有 lm_head.weight ⇒ 绑定词嵌入");
        assert!(m.tensors.iter().all(|(n, _, _)| n != "head.weight"));
        // f16 解出来的值必须有限（且与 f32 版同量级）
        for (n, _, d) in &m.tensors {
            assert!(d.iter().all(|v| v.is_finite()), "{n} 里出现非有限值");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// 灌数据必须**严格**：少一个参数 / 形状不符 / 多一个张量都要报错
    #[test]
    fn test_assign_is_strict() {
        let mut cfg = TransformerConfig::tiny(12);
        cfg.n_layer = 1;
        cfg.n_embd = 8;
        cfg.n_head = 2;
        cfg.n_kv_head = 1;
        cfg.ffn_hidden = 16;
        let model = Transformer::new(cfg, &mut Rng::new(3));
        // 正确的一整套
        let good: Vec<(String, Vec<usize>, Vec<f32>)> = model
            .named_parameters()
            .into_iter()
            .map(|(n, t)| (n, t.shape().to_vec(), vec![0.1; t.numel()]))
            .collect();
        assert_eq!(assign_named_parameters(&model, &good).unwrap(), good.len());
        // 少一个
        let mut less = good.clone();
        less.pop();
        let e = assign_named_parameters(&model, &less).unwrap_err();
        assert!(e.contains("模型需要但导入数据里没有"), "{e}");
        // 形状不对
        let mut bad = good.clone();
        bad[0].1 = vec![1, 1];
        bad[0].2 = vec![0.0];
        let e = assign_named_parameters(&model, &bad).unwrap_err();
        assert!(e.contains("形状不一致"), "{e}");
        // 多一个
        let mut more = good.clone();
        more.push(("nope".to_string(), vec![1], vec![0.0]));
        let e = assign_named_parameters(&model, &more).unwrap_err();
        assert!(e.contains("多出来"), "{e}");
    }

    /// 不支持的模型必须在 `config.json` 层面就被拦下，且给出可执行的提示
    #[test]
    fn test_unsupported_models_are_rejected() {
        let base = |extra: serde_json::Value| -> HfConfig {
            let mut v = serde_json::json!({
                "hidden_size": 8, "num_attention_heads": 2, "num_hidden_layers": 1,
                "intermediate_size": 16, "vocab_size": 12, "hidden_act": "silu",
            });
            for (k, val) in extra.as_object().unwrap() {
                v[k] = val.clone();
            }
            serde_json::from_value(v).unwrap()
        };
        assert!(base(serde_json::json!({})).check_supported().is_ok());
        // Qwen2.5 那种"带窗口但显式关掉"的必须放行（否则会把最常用的目标模型误杀）
        assert!(
            base(serde_json::json!({"sliding_window": 32768, "use_sliding_window": false}))
                .check_supported()
                .is_ok(),
            "`use_sliding_window: false` 时不应按 SWA 拒绝"
        );
        for (extra, want) in [
            (serde_json::json!({"num_local_experts": 8}), "MoE"),
            (serde_json::json!({"kv_lora_rank": 512}), "MLA"),
            (serde_json::json!({"sliding_window": 4096}), "滑动窗口"),
            (serde_json::json!({"partial_rotary_factor": 0.25}), "partial_rotary_factor"),
            (serde_json::json!({"rope_scaling": {"type": "llama3", "factor": 8.0}}), "rope_scaling"),
            (serde_json::json!({"hidden_act": "gelu"}), "hidden_act"),
        ] {
            let e = base(extra).check_supported().unwrap_err();
            assert!(e.contains(want), "错误信息应提到 `{want}`，实得：{e}");
        }
    }

    // ---------- 端到端对拍：导入后的 forward 必须与"按 HF 定义独立手写"的参考一致 ----------

    /// 参考前向用到的张量表
    struct Ref<'a> {
        t: HashMap<&'a str, (&'a [usize], &'a [f32])>,
        cfg: TransformerConfig,
    }

    impl<'a> Ref<'a> {
        fn get(&self, name: &str) -> &'a [f32] {
            self.t.get(name).unwrap_or_else(|| panic!("参考前向缺张量 {name}")).1
        }
        /// `y = x·W + b`（W 是本仓库口径 `[in, out]`）
        fn lin(&self, x: &[f32], name: &str, out: usize) -> Vec<f32> {
            let w = self.get(name);
            let inp = x.len();
            let mut y = vec![0.0f32; out];
            for i in 0..inp {
                let xi = x[i];
                if xi == 0.0 {
                    continue;
                }
                for o in 0..out {
                    y[o] += xi * w[i * out + o];
                }
            }
            y
        }
        fn rmsnorm(&self, x: &[f32], name: &str) -> Vec<f32> {
            let g = self.get(name);
            let d = x.len();
            let ms: f32 = x.iter().map(|v| v * v).sum::<f32>() / d as f32;
            let inv = 1.0 / (ms + self.cfg.rms_eps).sqrt();
            (0..d).map(|i| x[i] * inv * g[i]).collect()
        }
        /// rotate_half 口径的 RoPE（HF 定义：前半 `x·c - y·s`、后半 `x·s + y·c`）
        fn rope(&self, x: &[f32], head_dim: usize, pos: usize, base: f32) -> Vec<f32> {
            let half = head_dim / 2;
            let mut out = vec![0.0f32; x.len()];
            for h in 0..x.len() / head_dim {
                for j in 0..half {
                    let th = pos as f32 / base.powf((2 * j) as f32 / head_dim as f32);
                    let (c, s) = (th.cos(), th.sin());
                    let a = x[h * head_dim + j];
                    let b = x[h * head_dim + j + half];
                    out[h * head_dim + j] = a * c - b * s;
                    out[h * head_dim + j + half] = a * s + b * c;
                }
            }
            out
        }
        /// 整模型前向（单层 tiny LLaMA），返回 `[t][vocab]` 的 logits
        fn forward(&self, ids: &[usize]) -> Vec<Vec<f32>> {
            let cfg = &self.cfg;
            let (d, hd) = (cfg.n_embd, cfg.head_dim());
            let n_rep = cfg.n_head / cfg.n_kv_head;
            let emb = self.get("tok_emb.table");
            let t = ids.len();
            let mut logits = Vec::with_capacity(t);
            let mut hidden: Vec<Vec<f32>> = ids
                .iter()
                .map(|&i| emb[i * d..(i + 1) * d].to_vec())
                .collect();
            // 单层：ln1 → attn → 残差 → ln2 → mlp → 残差
            let mut x_attn = Vec::with_capacity(t);
            let mut qs = Vec::with_capacity(t);
            let mut ks = Vec::with_capacity(t);
            let mut vs = Vec::with_capacity(t);
            for pos in 0..t {
                let n = self.rmsnorm(&hidden[pos], "blocks.0.ln1.gamma");
                let q = self.rope(&self.lin(&n, "blocks.0.attn.c_q.weight", cfg.n_head * hd), hd, pos, cfg.rope_base);
                let k = self.rope(&self.lin(&n, "blocks.0.attn.c_k.weight", cfg.n_kv_head * hd), hd, pos, cfg.rope_base);
                let v = self.lin(&n, "blocks.0.attn.c_v.weight", cfg.n_kv_head * hd);
                qs.push(q);
                ks.push(k);
                vs.push(v);
            }
            let scale = 1.0 / (hd as f32).sqrt();
            for i in 0..t {
                // 因果：只看 j <= i；GQA：q 头 h 用 kv 头 h / n_rep
                let mut out = vec![0.0f32; cfg.n_head * hd];
                for h in 0..cfg.n_head {
                    let kvh = h / n_rep;
                    let mut sc = vec![0.0f32; i + 1];
                    for j in 0..=i {
                        let q = &qs[i][h * hd..(h + 1) * hd];
                        let k = &ks[j][kvh * hd..(kvh + 1) * hd];
                        sc[j] = q.iter().zip(k).map(|(a, b)| a * b).sum::<f32>() * scale;
                    }
                    let mx = sc.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let ex: Vec<f32> = sc.iter().map(|s| (s - mx).exp()).collect();
                    let sum: f32 = ex.iter().sum();
                    for j in 0..=i {
                        let p = ex[j] / sum;
                        let v = &vs[j][kvh * hd..(kvh + 1) * hd];
                        for kk in 0..hd {
                            out[h * hd + kk] += p * v[kk];
                        }
                    }
                }
                let o = self.lin(&out, "blocks.0.attn.c_proj.weight", d);
                x_attn.push(o.iter().zip(&hidden[i]).map(|(a, b)| a + b).collect::<Vec<f32>>());
            }
            for (i, xa) in x_attn.iter().enumerate() {
                let n = self.rmsnorm(xa, "blocks.0.ln2.gamma");
                let gate = self.lin(&n, "blocks.0.mlp.w_gate.weight", cfg.ffn_hidden);
                let up = self.lin(&n, "blocks.0.mlp.w_up.weight", cfg.ffn_hidden);
                let h: Vec<f32> = gate
                    .iter()
                    .zip(&up)
                    .map(|(g, u)| (g / (1.0 + (-g).exp())) * u)
                    .collect();
                let down = self.lin(&h, "blocks.0.mlp.w_down.weight", d);
                hidden[i] = down.iter().zip(xa).map(|(a, b)| a + b).collect();
            }
            for pos in 0..t {
                let n = self.rmsnorm(&hidden[pos], "ln_f.gamma");
                // 输出头：logits[v] = Σ_d h[d] · head[v, d]
                let head = self.get("head.weight");
                let vocab = cfg.vocab_size;
                logits.push(
                    (0..vocab)
                        .map(|v| (0..d).map(|dd| n[dd] * head[v * d + dd]).sum::<f32>())
                        .collect(),
                );
            }
            logits
        }
    }

    /// **端到端对拍**：合成 tiny LLaMA → 导入 → 灌进模型 → 前向，
    /// 与"按 HF 定义独立手写"的参考前向逐元素比对（容差 1e-4）。
    ///
    /// 参考实现刻意**不复用** rope / attention / layers 模块（自己算 rotate_half、
    /// 自己算 GQA 头映射与因果掩码、自己算 RMSNorm/SwiGLU），所以它能同时验证：
    /// RoPE 配对方式、权重转置、无 bias 开关、eps、GQA 映射、输出头布局这一整串导入语义。
    #[test]
    fn test_imported_tiny_llama_matches_independent_reference() {
        let dir = std::env::temp_dir().join("rust_llm_hf_e2e");
        let _ = std::fs::remove_dir_all(&dir);
        write_tiny_hf(&dir, true, DType::F32);
        let m = load_hf_dir(&dir.to_string_lossy()).unwrap();
        let model = Transformer::new(m.cfg.clone(), &mut Rng::new(0));
        let n = assign_named_parameters(&model, &m.tensors).unwrap();
        assert!(n > 0);

        let ids = [1usize, 5, 7, 3];
        let got = model.forward(&ids, 1, ids.len(), None, false).data().to_vec();

        let mut t = HashMap::new();
        for (name, shape, data) in &m.tensors {
            t.insert(name.as_str(), (shape.as_slice(), data.as_slice()));
        }
        let rf = Ref { t, cfg: m.cfg.clone() };
        let want = rf.forward(&ids);
        let vocab = m.cfg.vocab_size;
        for (i, row) in want.iter().enumerate() {
            for (v, w) in row.iter().enumerate() {
                let g = got[i * vocab + v];
                assert!(
                    (g - w).abs() <= 1e-4 + 1e-4 * w.abs(),
                    "位置 {i} 的词表 {v}：模型 {g} vs 参考 {w}（整串导入语义有一处不对）"
                );
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }
}
