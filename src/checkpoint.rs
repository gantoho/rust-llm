//! Checkpoint 保存 / 恢复
//!
//! 文件格式（自描述二进制，`serde_json` 只序列化头信息，参数数据按原生 f32 小端写入）：
//!
//! ```text
//! 魔数 "LLMCP1\n"（7 字节）
//! u32 小端：JSON 头长度
//! JSON 头：step、best_val_loss、模型配置、优化器状态（m/v）、参数元信息（名字+形状）
//! 之后按参数顺序拼接每个参数的 f32 小端数据
//! ```

use std::fs::File;
use std::io::{Read, Write};

use crate::model::{GPT, GPTConfig};
use crate::optim::AdamW;
use serde::{Deserialize, Serialize};

const MAGIC: &[u8; 7] = b"LLMCP1\n";

/// checkpoint 头（JSON 部分）
#[derive(Serialize, Deserialize)]
struct CkptHeader {
    step: usize,
    best_val_loss: f32,
    model: GPTConfig,
    opt_t: usize,
    opt_m: Vec<Vec<f32>>,
    opt_v: Vec<Vec<f32>>,
    params: Vec<ParamMeta>,
}

/// 单个参数的元信息
#[derive(Serialize, Deserialize)]
struct ParamMeta {
    name: String,
    shape: Vec<usize>,
}

/// 加载 checkpoint 后得到的元信息（训练 / 评估 / 生成共用）
#[derive(Clone)]
pub struct Checkpoint {
    pub step: usize,
    pub best_val_loss: f32,
    pub model: GPTConfig,
}

/// 保存 checkpoint：模型参数 + 优化器状态 + 元信息
pub fn save(path: &str, model: &GPT, opt: &AdamW, step: usize, best_val_loss: f32) {
    let named = model.named_parameters();
    let (opt_t, opt_m, opt_v) = opt.state();
    let params = named
        .iter()
        .map(|(name, t)| ParamMeta {
            name: name.clone(),
            shape: t.shape().to_vec(),
        })
        .collect();
    let header = CkptHeader {
        step,
        best_val_loss,
        model: model.cfg.clone(),
        opt_t,
        opt_m,
        opt_v,
        params,
    };
    let json = serde_json::to_vec(&header).expect("序列化 checkpoint 头失败");

    let mut f = File::create(path).unwrap_or_else(|e| panic!("无法创建 checkpoint {path}: {e}"));
    let write = |r: std::io::Result<()>| {
        r.unwrap_or_else(|e| panic!("写入 checkpoint {path} 失败: {e}"))
    };
    write(f.write_all(MAGIC));
    write(f.write_all(&(json.len() as u32).to_le_bytes()));
    write(f.write_all(&json));
    for (_, t) in &named {
        for v in t.data() {
            write(f.write_all(&v.to_le_bytes()));
        }
    }
}

/// 只读取 checkpoint 头（模型配置、步数、best loss），不加载参数。
/// 用于 eval / generate 先按 checkpoint 里的配置构造模型。
/// 只读头部 + JSON，不读参数数据，大模型下也能秒开。
pub fn load_header(path: &str) -> Checkpoint {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("无法打开 checkpoint {path}: {e}"));
    let (h, _) = read_head(&mut f, path);
    Checkpoint {
        step: h.step,
        best_val_loss: h.best_val_loss,
        model: h.model,
    }
}

/// 只加载参数，不涉及优化器（eval / generate 用）。
/// `model` 必须先按 checkpoint 里的配置构造好，参数按名字逐个恢复。
pub fn load_params(path: &str, model: &GPT) -> Checkpoint {
    let (ckpt, metas, bytes, _opt) = read_file(path);
    restore_params(model, &metas, &bytes);
    ckpt
}

/// 加载参数并恢复优化器状态（resume 用）
pub fn load_with_opt(path: &str, model: &GPT, opt: &mut AdamW) -> Checkpoint {
    let (ckpt, metas, bytes, opt_state) = read_file(path);
    restore_params(model, &metas, &bytes);
    let (t, m, v) = opt_state.expect("checkpoint 缺少优化器状态");
    opt.restore_state(t, m, v);
    ckpt
}

/// 从文件流读取并解析 checkpoint 头部（魔数 + JSON 头），返回 (header, json 长度)。
/// 用 `read_exact` 而非切片索引，文件过短/损坏时给出可读错误而非越界 panic。
fn read_head(f: &mut File, path: &str) -> (CkptHeader, usize) {
    let mut magic = [0u8; 7];
    f.read_exact(&mut magic)
        .unwrap_or_else(|e| panic!("读取 {path} 头部失败（文件过短或损坏）: {e}"));
    assert_eq!(
        &magic,
        MAGIC,
        "checkpoint 魔数不匹配：{path} 不是本项目的 checkpoint"
    );
    let mut len_buf = [0u8; 4];
    f.read_exact(&mut len_buf)
        .unwrap_or_else(|e| panic!("读取 {path} 头长度失败（文件过短或损坏）: {e}"));
    let json_len = u32::from_le_bytes(len_buf) as usize;
    let mut json = vec![0u8; json_len];
    f.read_exact(&mut json)
        .unwrap_or_else(|e| panic!("读取 {path} JSON 头失败（文件过短或损坏）: {e}"));
    let header: CkptHeader = serde_json::from_slice(&json)
        .unwrap_or_else(|e| panic!("解析 checkpoint 头失败: {e}"));
    (header, json_len)
}

/// 读取并解析整个 checkpoint 文件（头部 + 参数数据）
fn read_file(
    path: &str,
) -> (
    Checkpoint,
    Vec<ParamMeta>,
    Vec<u8>,
    Option<(usize, Vec<Vec<f32>>, Vec<Vec<f32>>)>,
) {
    let mut f = File::open(path).unwrap_or_else(|e| panic!("无法打开 checkpoint {path}: {e}"));
    let (header, _json_len) = read_head(&mut f, path);

    let ckpt = Checkpoint {
        step: header.step,
        best_val_loss: header.best_val_loss,
        model: header.model,
    };
    let opt = Some((header.opt_t, header.opt_m, header.opt_v));

    // 读取并校验参数数据总长度
    let mut data = Vec::new();
    f.read_to_end(&mut data)
        .unwrap_or_else(|e| panic!("读取 {path} 参数数据失败: {e}"));
    let total: usize = header
        .params
        .iter()
        .map(|m| m.shape.iter().product::<usize>())
        .sum();
    assert_eq!(
        data.len(),
        total * 4,
        "checkpoint 参数数据长度不匹配（文件可能损坏）"
    );
    (ckpt, header.params, data, opt)
}

/// 按名字、形状把参数数据写回模型
fn restore_params(model: &GPT, metas: &[ParamMeta], bytes: &[u8]) {
    let named = model.named_parameters();
    assert_eq!(
        named.len(),
        metas.len(),
        "checkpoint 参数数量（{}）与模型（{}）不匹配，请检查配置是否一致",
        metas.len(),
        named.len()
    );
    let mut pos = 0usize;
    for ((name, t), meta) in named.iter().zip(metas) {
        assert_eq!(
            name, &meta.name,
            "参数名不匹配：checkpoint='{}' vs 模型='{}'",
            meta.name, name
        );
        assert_eq!(
            t.shape(),
            &meta.shape[..],
            "参数 {} 形状不匹配：checkpoint={:?} vs 模型={:?}",
            name,
            meta.shape,
            t.shape()
        );
        let numel = meta.shape.iter().product::<usize>();
        let mut data = vec![0.0f32; numel];
        for j in 0..numel {
            let start = pos + j * 4;
            data[j] = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        }
        pos += numel * 4;
        t.set_data(data);
    }
    assert_eq!(pos, bytes.len(), "checkpoint 参数数据读取不完整");
}
