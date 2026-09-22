//! RoPE 旋转位置编码（第 20 课）
//!
//! RoPE（Rotary Position Embedding）是现代 LLM 的标准位置编码方式：
//! 把"位置信息"通过旋转变换揉进 Q/K 向量，让注意力天然感知相对位置。
//!
//! 核心思想：
//! - 把向量的每两个相邻元素看作一个二维平面上的点
//! - 按位置乘以旋转矩阵 R(θ)：位置越远，旋转角度越大
//! - 两个向量的点积只与"位置差"有关 → 天然编码相对位置
//! - 旋转是正交变换 → 不改变向量范数，数值稳定
//!
//! 用法：在注意力层内部，对 Q/K 做 `rotary_pair(positions)` 一次旋转两者，V 不转。
//!
//! # 长度外推
//!
//! 原版 RoPE 的每个对偶下标 `i` 对应一个固定波长 `λ_i = 2π·base^(2i/d)`。
//! 训练时模型只见过 `block_size` 以内的位置差，推理时一旦超窗，超出部分的
//! `θ = pos/λ` 就落进了训练时从未出现的取值区间——注意力分数失准、生成质量断崖下跌。
//! 三种频率缩放（[`RopeScaling`]）用不同的方式把这张频率表拉长，让模型能外推到
//! 更长的上下文：**它们都只改频率表，不改旋转公式本身**，因此相对位置这个核心性质
//! （注意力只取决于位置差）在缩放后依然成立。

use std::rc::Rc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use crate::tensor::Tensor;

/// RoPE 的频率底数：原始 RoPE / GPT-NeoX 用 10 000。
/// LLaMA-3 改成 500 000——把长波长维度整体拉长，本身就是一种静态的频率缩放。
pub const ROPE_BASE: f32 = 10000.0;

/// RoPE 的频率缩放方式（长度外推）。
///
/// 三种方案是递进关系，工程取舍各不相同：
///
/// - [`RopeScaling::Linear`]（位置插值 PI）：把位置整体压成 `pos / factor`，
///   等价于把所有波长一起乘 `factor`。改动最小，但**高频维度的分辨力也被一起压掉**，
///   短距离的词序信息受损，通常需要少量继续训练才能恢复。
/// - [`RopeScaling::Ntk`]（NTK-aware）：不改位置，改成把底数放大成
///   `base' = base · factor^(d/(d-2))`。代入 `θ_i = pos / base'^(2i/d)` 可见，
///   第 `i` 个频率被缩小了 `factor^(2i/(d-2))` 倍——**低频（长波长）维度几乎按
///   `factor` 整体拉长，高频维度（`i` 小）几乎不动**，短距离分辨力基本保住。
///   改一个常数就能用，是社区最常用的"零训练外推"。
/// - [`RopeScaling::Yarn`]：在 NTK 的基础上**按波长分段**（NTK-by-parts）。波长本来就
///   比训练窗口长的维度完全插值，波长很短的维度完全不缩放，中间平滑过渡；
///   同时用 `mscale` 补偿"插值让注意力分布变平"的副作用。外推质量最好，
///   代价是要调 `beta_fast` / `beta_slow` 两个边界。
///
/// `factor` 是外推倍数（想跑 4 倍上下文就给 4.0）；`None` 表示不外推，
/// 此时频率表与 GPU 内核里写死的式子逐位一致（见 [`RopeSpec::is_plain`]）。
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum RopeScaling {
    /// 不缩放（默认）
    None,
    /// 位置插值（PI）：`θ_i = (pos / factor) / base^(2i/d)`
    Linear { factor: f32 },
    /// NTK-aware：`base' = base · factor^(d/(d-2))`
    Ntk { factor: f32 },
    /// YaRN：NTK-by-parts + 注意力温度补偿
    Yarn {
        /// 外推倍数
        factor: f32,
        /// 高频边界：相对圈数大于它的维度完全不缩放（默认 32）
        beta_fast: f32,
        /// 低频边界：相对圈数小于它的维度完全插值（默认 1）
        beta_slow: f32,
        /// 温度补偿系数，1.0 = 标准强度（默认 1.0）
        mscale: f32,
    },
}

impl Default for RopeScaling {
    fn default() -> Self {
        RopeScaling::None
    }
}

impl RopeScaling {
    /// 一句话描述（命令行回显与运行日志用）
    pub fn describe(self) -> String {
        match self {
            RopeScaling::None => "无（训练窗口内）".to_string(),
            RopeScaling::Linear { factor } => format!("Linear 位置插值 ×{factor}"),
            RopeScaling::Ntk { factor } => format!("NTK-aware ×{factor}"),
            RopeScaling::Yarn { factor, beta_fast, beta_slow, mscale } => {
                format!("YaRN ×{factor}（beta_fast={beta_fast} beta_slow={beta_slow} mscale={mscale}）")
            }
        }
    }
}

/// RoPE 的全部频率参数：底数 + 缩放方式 + 训练窗口。
///
/// 打包成一个结构体传进注意力层，而不是给 `rotary_pair` 再加三个标量参数：
/// 三者是**同进退**的一组，且 YaRN 的分段边界必须知道训练窗口才知道该从哪个下标起插值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeSpec {
    /// 频率底数（见 [`ROPE_BASE`]）
    pub base: f32,
    /// 长度外推方式
    pub scaling: RopeScaling,
    /// 训练时的上下文窗口。YaRN 按"该维度相对**训练窗口**转了多少圈"来分段，
    /// NTK 的 `factor` 也是相对它而言；所以这里必须填训练用的 `block_size`，
    /// 而不是本次推理想跑到多长。
    pub train_ctx: usize,
}

impl Default for RopeSpec {
    fn default() -> Self {
        RopeSpec {
            base: ROPE_BASE,
            scaling: RopeScaling::None,
            train_ctx: 0,
        }
    }
}

impl RopeSpec {
    pub fn new(base: f32, scaling: RopeScaling, train_ctx: usize) -> Self {
        RopeSpec { base, scaling, train_ctx }
    }

    /// 频率表是否与 GPU 内核里写死的式子一致（`base = 10000` 且不缩放）。
    /// 不一致时旋转结果必然与常驻显存路径不同，那条快路必须让路。
    ///
    /// 唯一的调用方是 GPU 常驻注意力路径（`model.rs` 的 `attn_resident`），
    /// 不编 `gpu` feature 时它就是一段没人调用的守卫。
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn is_plain(&self) -> bool {
        self.scaling == RopeScaling::None && self.base == ROPE_BASE
    }
}

/// 预计算每个 (位置, 对偶下标) 的 cos/sin 表，长度 rows × (D/2)。
/// 同一批 positions 的三角只算一次：前向、反向、Q/K 复用。
///
/// 关键优化：频率表只与对偶下标有关，先算一遍 D/2 个逆波长，
/// 再对每个位置做 `theta = pos · inv_freq[i]` 求 cos/sin——
/// 原来在行内循环里重复计算 powf，rows=2048 时要算 26 万次 powf（约 30ms）。
fn build_cos_sin_tab(positions: &[usize], d: usize, spec: &RopeSpec) -> (Vec<f32>, Vec<f32>) {
    let rows = positions.len();
    let half = d / 2;
    let freq = inv_freqs(d, spec);
    // YaRN 的温度补偿：cos/sin 同步放大 → 旋转后的 Q/K 模长变大 → 注意力更"尖"
    let mscale = match spec.scaling {
        RopeScaling::Yarn { factor, mscale, .. } => yarn_mscale(factor, mscale),
        _ => 1.0,
    };
    let mut c_tab = vec![0.0f32; rows * half];
    let mut s_tab = vec![0.0f32; rows * half];
    for r in 0..rows {
        let pos = positions[r] as f32;
        for i in 0..half {
            let theta = pos * freq[i];
            c_tab[r * half + i] = theta.cos() * mscale;
            s_tab[r * half + i] = theta.sin() * mscale;
        }
    }
    (c_tab, s_tab)
}

/// 逆波长表：`inv_freq[i] = 1 / λ_i`，`λ_i = base^(2i/d)`，于是 `θ_i = pos · inv_freq[i]`。
///
/// **长度外推的全部改动都集中在这里这一张表上**——旋转公式、反向公式、KV cache
/// 存"已旋转的 K"这些结构性设计一概不动。
fn inv_freqs(d: usize, spec: &RopeSpec) -> Vec<f32> {
    let half = d / 2;
    let base = spec.base;
    let raw: Vec<f32> = (0..half)
        .map(|i| 1.0 / base.powf((2 * i) as f32 / d as f32))
        .collect();
    match spec.scaling {
        RopeScaling::None => raw,
        RopeScaling::Linear { factor } => {
            assert!(factor > 0.0, "RoPE Linear 的 factor 必须 > 0，当前 {factor}");
            raw.iter().map(|f| f / factor).collect()
        }
        RopeScaling::Ntk { factor } => {
            assert!(factor > 0.0, "RoPE NTK 的 factor 必须 > 0，当前 {factor}");
            // base' = base · factor^(d/(d-2))，代入 θ_i 即得第 i 个频率缩小 factor^(2i/(d-2)) 倍：
            // i = 0 完全不动，i = d/2-1 缩小整整 factor 倍，中间按指数过渡。
            // d = 2 时指数发散（维度太窄，NTK 本就无意义），退化成不缩放。
            let exp = if d > 2 { d as f32 / (d as f32 - 2.0) } else { 0.0 };
            let b = base * factor.powf(exp);
            (0..half).map(|i| 1.0 / b.powf((2 * i) as f32 / d as f32)).collect()
        }
        RopeScaling::Yarn { factor, beta_fast, beta_slow, .. } => {
            assert!(factor > 0.0, "RoPE YaRN 的 factor 必须 > 0，当前 {factor}");
            let (lo, hi) =
                yarn_correction_range(d, base, spec.train_ctx.max(1), beta_fast, beta_slow);
            (0..half)
                .map(|i| {
                    // ramp = 0（下标小 → 高频 → 波长短于窗口）保持原频率；
                    // ramp = 1（下标大 → 低频 → 波长比窗口还长）完全插值；中间线性过渡
                    let ramp = ((i as f32 - lo) / (hi - lo).max(1e-3)).clamp(0.0, 1.0);
                    let interp = raw[i] / factor;
                    interp * ramp + raw[i] * (1.0 - ramp)
                })
                .collect()
        }
    }
}

/// YaRN 的分段边界：返回「完全不缩放」与「完全插值」的下标分界，范围 `0..d/2`。
///
/// 判据是**波长**：`λ_i = 2π/base^(-2i/d) = 2π·base^(2i/d)`。定义该维度相对训练窗口
/// 转过的圈数 `r_i = train_ctx / λ_i`：
/// - `r_i > beta_fast`（波长远短于窗口 → 高频）：完全不动，短距离分辨力必须保住；
/// - `r_i < beta_slow`（波长比窗口还长 → 低频）：完全插值，让模型没见过的大位置差
///   落到见过的角度区间里；
/// - 中间线性过渡。
///
/// 反解 `r_i = β` 得到下标 `i = d·ln(train_ctx / (2πβ)) / (2·ln base)`。
fn yarn_correction_range(
    d: usize,
    base: f32,
    train_ctx: usize,
    beta_fast: f32,
    beta_slow: f32,
) -> (f32, f32) {
    let idx = |beta: f32| {
        let r = train_ctx as f32 / (beta * 2.0 * std::f32::consts::PI);
        d as f32 * r.ln() / (2.0 * base.ln())
    };
    let lo = idx(beta_fast).floor().max(0.0);
    let hi = idx(beta_slow).ceil().min(d as f32 - 1.0);
    (lo, hi)
}

/// YaRN 的注意力温度补偿。
///
/// 位置插值把大位置差映射到小角度区间，注意力 logits 的分布随之变"平"——模型对
/// "该看哪个位置"变得含糊。把 cos/sin 整体乘一个大于 1 的系数，等于放大旋转后
/// Q/K 的模长，也就放大了 logits（等价于降低 softmax 温度），把分布重新压尖。
fn yarn_mscale(factor: f32, mscale: f32) -> f32 {
    if factor <= 1.0 {
        1.0
    } else {
        0.1 * mscale * factor.ln() + 1.0
    }
}

/// 用现成的 cos/sin 表旋转一个张量（[rows, D]）。
/// 反向用旋转矩阵的转置 R(θ)ᵀ 回传梯度，闭包直接查表。
fn rotate_with_tab(x: &Tensor, c_tab: &[f32], s_tab: &[f32]) -> Tensor {
    let (rows, d) = (x.shape[0], x.shape[1]);
    let sd = x.data.borrow();
    let sd_ref: &[f32] = &sd;
    let mut out_data = vec![0.0f32; rows * d];
    let half = d / 2;
    // 并行：每行旋转独立，行间无依赖
    out_data
        .par_chunks_mut(d)
        .enumerate()
        .for_each(|(r, out_row)| {
            let base = r * d;
            let ct_base = r * half;
            for i in 0..half {
                let (c, s) = (c_tab[ct_base + i], s_tab[ct_base + i]);
                let (a, b) = (sd_ref[base + 2 * i], sd_ref[base + 2 * i + 1]);
                out_row[2 * i] = a * c - b * s;
                out_row[2 * i + 1] = a * s + b * c;
            }
        });
    drop(sd);

    let mut result = Tensor::new(out_data, x.shape.clone(), x.req());
    if x.req() {
        let rg = result.grad.clone();
        let sg = x.grad.clone();
        let ct = c_tab.to_vec();
        let st = s_tab.to_vec();
        result.parents = Rc::new(vec![x.clone()]);
        result.backward = Some(Rc::new(move || {
            // 先把梯度拷出 RefCell，再并行写回（Ref<Vec<f32>> 不是 Sync）
            let g_local: Vec<f32> = rg.borrow().to_vec();
            let mut sgm = sg.borrow_mut();
            // 并行：每行独立计算梯度，行间无依赖（与前向一致）
            sgm.par_chunks_mut(d)
                .enumerate()
                .for_each(|(r, sgm_row)| {
                    let g_base = r * d;
                    let ct_base = r * (d / 2);
                    for i in 0..d / 2 {
                        let (c, s) = (ct[ct_base + i], st[ct_base + i]);
                        let (ga, gb) = (g_local[g_base + 2 * i], g_local[g_base + 2 * i + 1]);
                        // 反向 = 前向旋转矩阵的转置 R(θ)ᵀ：grad = (ga·c + gb·s, -ga·s + gb·c)
                        sgm_row[2 * i] += ga * c + gb * s;
                        sgm_row[2 * i + 1] += -ga * s + gb * c;
                    }
                });
        }));
    }
    result
}

impl Tensor {
    /// RoPE 旋转位置编码：把位置信息揉进向量的每一对相邻元素。
    ///
    /// 输入：`[rows, D]`（D 必须为偶数），`positions` 长度 = rows
    ///
    /// 前向（每对 a, b → a', b'）：
    /// ```text
    /// θ = pos / 10000^(2i/D)
    /// a' = a·cos(θ) - b·sin(θ)
    /// b' = a·sin(θ) + b·cos(θ)
    /// ```
    ///
    /// 反向：旋转矩阵正交，梯度用其转置（负角度）回传：
    /// ```text
    /// grad_a = ga·cos(θ) + gb·sin(θ)
    /// grad_b = -ga·sin(θ) + gb·cos(θ)
    /// ```
    /// 频率表由 `spec` 决定：`scaling = None` 时就是原版 RoPE，其余见 [`RopeScaling`]。
    /// 仅供测试使用；生产代码（attention）用 `rotary_pair` 一次旋转 Q/K。
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn rotary(&self, positions: &[usize], spec: &RopeSpec) -> Tensor {
        assert_eq!(self.rank(), 2, "rotary 输入应为 [rows, D]");
        assert_eq!(self.shape[0], positions.len(), "positions 数量必须等于行数");
        let d = self.shape[1];
        assert_eq!(d % 2, 0, "最后一维必须为偶数才能两两配对旋转");
        let (c_tab, s_tab) = build_cos_sin_tab(positions, d, spec);
        rotate_with_tab(self, &c_tab, &s_tab)
    }

    /// 一次建表同时旋转 Q 和 K（两者 positions 相同，三角函数只算一遍）。
    /// 返回 `(rotated_q, rotated_k)`。
    pub fn rotary_pair(
        &self,
        other: &Tensor,
        positions: &[usize],
        spec: &RopeSpec,
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(self.shape[0], other.shape[0], "Q/K 行数必须一致");
        assert_eq!(self.shape[1], other.shape[1], "Q/K 的 head_dim 必须一致");
        let d = self.shape[1];
        let (c_tab, s_tab) = build_cos_sin_tab(positions, d, spec);
        (
            rotate_with_tab(self, &c_tab, &s_tab),
            rotate_with_tab(other, &c_tab, &s_tab),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 原版 RoPE 的参数：底数 10 000、不缩放、训练窗口 512
    fn plain() -> RopeSpec {
        RopeSpec::new(ROPE_BASE, RopeScaling::None, 512)
    }

    #[test]
    fn test_rotary() {
        // 1. 旋转是正交变换：范数不变
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
        let r = x.rotary(&[3], &plain());
        let orig_norm: f32 = x.data().iter().map(|v| v * v).sum();
        let rot_norm: f32 = r.data().iter().map(|v| v * v).sum();
        assert!(
            (orig_norm - rot_norm).abs() < 1e-3,
            "范数应守恒：{} vs {}",
            orig_norm,
            rot_norm
        );

        // 2. pos=0 时所有角度为 0，等于恒等变换
        let x2 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);
        let r2 = x2.rotary(&[0], &plain());
        assert!((r2.data()[0] - 1.0).abs() < 1e-5);
        assert!((r2.data()[3] - 4.0).abs() < 1e-5);

        // 3. 梯度：sum 的梯度是单位向量，经正交矩阵回传后范数不变（= 元素数）
        let x3 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
        let loss = x3.rotary(&[2], &plain()).sum();
        loss.backward();
        let g: Vec<f32> = x3.grad();
        assert!(
            (g.iter().map(|v| v * v).sum::<f32>() - 6.0).abs() < 1e-3,
            "梯度范数应为 6"
        );
    }

    #[test]
    fn test_rotary_grad_exact() {
        // pos=1、i=0 时 θ=1 rad，逐元素验证梯度 = R(θ)ᵀ·g（g 为全 1）
        // 前向 o_a = a·c - b·s, o_b = a·s + b·c
        // 反向 grad_a = g_a·c + g_b·s, grad_b = -g_a·s + g_b·c
        let x = Tensor::param(vec![1.0, 2.0], vec![1, 2]);
        let loss = x.rotary(&[1], &plain()).sum();
        loss.backward();
        let (c, s) = (1f32.cos(), 1f32.sin());
        let (ga, gb) = (c + s, -s + c);
        assert!((x.grad()[0] - ga).abs() < 1e-5, "grad[0] = {}", x.grad()[0]);
        assert!((x.grad()[1] - gb).abs() < 1e-5, "grad[1] = {}", x.grad()[1]);
    }

    /// 不缩放时必须与手写的原版公式逐位一致（不能因为引入 spec 就跑偏）
    #[test]
    fn test_plain_spec_reproduces_original_formula() {
        let d = 6;
        let (c_tab, s_tab) = build_cos_sin_tab(&[7], d, &plain());
        for i in 0..d / 2 {
            let theta = 7.0 / 10000f32.powf((2 * i) as f32 / d as f32);
            assert!((c_tab[i] - theta.cos()).abs() < 1e-6, "cos[{i}]");
            assert!((s_tab[i] - theta.sin()).abs() < 1e-6, "sin[{i}]");
        }
        assert!(plain().is_plain(), "默认参数应被判定为「与 GPU 内核一致」");
        assert!(!RopeSpec::new(500000.0, RopeScaling::None, 512).is_plain());
        assert!(!RopeSpec::new(ROPE_BASE, RopeScaling::Ntk { factor: 4.0 }, 512).is_plain());
    }

    /// 位置插值：`pos=2P` 在 `factor=2` 下的角度 == `pos=P` 不缩放时的角度。
    /// 这正是"把没见过的长位置压回见过的区间"的字面含义。
    #[test]
    fn test_linear_scaling_halves_effective_positions() {
        let p = RopeSpec::new(ROPE_BASE, RopeScaling::Linear { factor: 2.0 }, 512);
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![1, 8]);
        let scaled = x.rotary(&[10], &p);
        let unscaled = x.rotary(&[5], &plain());
        for (a, b) in scaled.data().iter().zip(unscaled.data().iter()) {
            assert!((a - b).abs() < 1e-6, "Linear×2 的 pos=10 应等于未缩放的 pos=5");
        }
    }

    /// NTK-aware：每个频率都被拉长（`inv_freq` 变小 ⇒ 同一位置转过的角度更小），
    /// 但**高频维度（i 小）几乎不动、低频维度（i 大）按 factor 整体拉长**——
    /// 这就是它比位置插值更保短距离分辨力的原因。
    #[test]
    fn test_ntk_keeps_high_freq_and_stretches_low_freq() {
        let d = 32;
        let factor = 8.0f32;
        let raw = inv_freqs(d, &plain());
        let ntk = inv_freqs(d, &RopeSpec::new(ROPE_BASE, RopeScaling::Ntk { factor }, 512));
        // 解析解：inv_freq_i 被缩小 factor^(2i/(d-2)) 倍
        for i in 0..d / 2 {
            let expect = raw[i] * factor.powf(-2.0 * i as f32 / (d as f32 - 2.0));
            assert!(
                (ntk[i] - expect).abs() <= expect.abs() * 1e-4 + 1e-12,
                "i={i}: {} vs 解析解 {expect}",
                ntk[i]
            );
        }
        assert!((ntk[0] - raw[0]).abs() < 1e-9, "i=0 是最高频，必须完全不动");
        assert!(
            (ntk[d / 2 - 1] - raw[d / 2 - 1] / factor).abs() < raw[d / 2 - 1] * 1e-4,
            "最低频那一路应恰好被拉长 factor 倍"
        );
    }

    /// YaRN：短波长维度原样保留，长波长维度整体插值，中间单调过渡且始终落在
    /// `[raw/factor, raw]` 区间内（不会出现比"不缩放"还大或比"全插值"还小的值）。
    #[test]
    fn test_yarn_by_parts_preserves_short_wavelength_dims() {
        let d = 64;
        let factor = 8.0f32;
        let train_ctx = 512;
        let (beta_fast, beta_slow) = (32.0, 1.0);
        let raw = inv_freqs(d, &plain());
        let yarn = inv_freqs(
            d,
            &RopeSpec::new(
                ROPE_BASE,
                RopeScaling::Yarn { factor, beta_fast, beta_slow, mscale: 1.0 },
                train_ctx,
            ),
        );
        let (lo, hi) = yarn_correction_range(d, ROPE_BASE, train_ctx, beta_fast, beta_slow);
        assert!(lo < hi, "分段区间必须非空：lo={lo} hi={hi}");

        for i in 0..d / 2 {
            let f = i as f32;
            if f < lo {
                assert!((yarn[i] - raw[i]).abs() < 1e-9, "i={i} 在 lo={lo} 之前应完全不缩放");
            } else if f >= hi {
                assert!(
                    (yarn[i] - raw[i] / factor).abs() < raw[i] * 1e-4,
                    "i={i} 在 hi={hi} 之后应完全插值"
                );
            } else {
                assert!(
                    yarn[i] <= raw[i] + 1e-9 && yarn[i] >= raw[i] / factor - 1e-9,
                    "i={i} 的过渡值必须夹在 [raw/factor, raw] 之间"
                );
            }
        }
        // 过渡段必须随下标单调下降（下标越大 = 波长越长 = 压得越狠）
        for i in (lo as usize + 1)..(hi as usize).min(d / 2) {
            assert!(yarn[i] <= yarn[i - 1] + 1e-9, "过渡段应在 i={i} 处单调不增");
        }
    }

    /// YaRN 的温度补偿：`factor > 1` 时把 cos/sin 放大（注意力更尖），
    /// `factor <= 1` 时退化为 1（不做任何补偿）。
    #[test]
    fn test_yarn_mscale_only_compensates_when_extrapolating() {
        assert_eq!(yarn_mscale(1.0, 1.0), 1.0);
        assert_eq!(yarn_mscale(0.5, 1.0), 1.0);
        let m = yarn_mscale(4.0, 1.0);
        assert!(m > 1.0, "factor>1 时补偿系数应大于 1，实际 {m}");
        // mscale = 0 时关闭补偿
        assert_eq!(yarn_mscale(4.0, 0.0), 1.0);
        // 落到表上：YaRN 的 cos/sin 整体被放大 mscale 倍
        let spec = RopeSpec::new(
            ROPE_BASE,
            RopeScaling::Yarn { factor: 4.0, beta_fast: 32.0, beta_slow: 1.0, mscale: 1.0 },
            512,
        );
        let (c_yarn, s_yarn) = build_cos_sin_tab(&[3], 8, &spec);
        let (c_plain, s_plain) = build_cos_sin_tab(&[3], 8, &plain());
        let ratio = c_yarn[0] / c_plain[0];
        assert!((ratio - m).abs() < 1e-5, "表里的 cos 应整体放大 {m} 倍，实际 {ratio}");
        let _ = s_yarn[0] + s_plain[0];
    }

    /// RoPE 的核心性质：**注意力只取决于位置差**。任何一种频率缩放都只是换了一张
    /// 频率表，旋转仍是正交变换，所以"同间距的点积相同"必须对四种模式都成立。
    #[test]
    fn test_relative_position_invariance_holds_for_every_scaling() {
        let d = 8;
        let specs = [
            RopeSpec::new(ROPE_BASE, RopeScaling::None, 64),
            RopeSpec::new(ROPE_BASE, RopeScaling::Linear { factor: 4.0 }, 64),
            RopeSpec::new(ROPE_BASE, RopeScaling::Ntk { factor: 4.0 }, 64),
            RopeSpec::new(
                ROPE_BASE,
                RopeScaling::Yarn { factor: 4.0, beta_fast: 32.0, beta_slow: 1.0, mscale: 1.0 },
                64,
            ),
        ];
        let q = Tensor::param(vec![0.5, -1.0, 2.0, 0.25, -0.75, 1.5, 0.3, -0.2], vec![1, d]);
        let k = Tensor::param(vec![1.25, 0.5, -1.5, 0.75, 0.1, -0.6, 2.0, 0.9], vec![1, d]);
        // 表按 positions 逐行建：`rotary_pair(&[m, m+gap], ..)` 把 q 转到 m、k 转到 m+gap
        let dot = |m: usize, gap: usize, spec: &RopeSpec| -> f32 {
            let (rq, rk) = q.rotary_pair(&k, &[m, m + gap], spec);
            let (qd, kd) = (rq.data(), rk.data());
            qd.iter().zip(kd.iter()).map(|(a, b)| a * b).sum::<f32>()
        };
        for spec in &specs {
            let base = dot(0, 5, spec);
            for m in [7usize, 40, 500] {
                let got = dot(m, 5, spec);
                assert!(
                    (got - base).abs() <= base.abs() * 1e-3 + 1e-5,
                    "{}：间距 5 的点积应与起点无关，m={m} 时 {got} vs {base}",
                    spec.scaling.describe()
                );
            }
        }
    }

    /// 位置 0 是最特殊的一点：所有角度都是 0（cos=1、sin=0），因此非 YaRN 模式下
    /// 旋转必须是**严格恒等**。这条约束保证了"序列第一个 token 的位置信息不被缩放扰动"，
    /// 也是 §相对位置不变性 在 m=0 处的特例。
    #[test]
    fn test_scaling_keeps_position_zero_as_identity() {
        let x = Tensor::param(vec![1.0, -2.0, 3.0, -4.0], vec![1, 4]);
        for spec in [
            RopeSpec::new(ROPE_BASE, RopeScaling::None, 64),
            RopeSpec::new(ROPE_BASE, RopeScaling::Linear { factor: 4.0 }, 64),
            RopeSpec::new(ROPE_BASE, RopeScaling::Ntk { factor: 4.0 }, 64),
        ] {
            let r = x.rotary(&[0], &spec);
            for (a, b) in r.data().iter().zip(x.data().iter()) {
                assert!((a - b).abs() < 1e-6, "pos=0 应为恒等变换（{}）", spec.scaling.describe());
            }
        }
    }
}
