//! RoPE 旋转位置编码（第 19 课）
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

use std::rc::Rc;

use rayon::prelude::*;
use crate::tensor::Tensor;

/// 预计算每个 (位置, 对偶下标) 的 cos/sin 表，长度 rows × (D/2)。
/// 同一批 positions 的三角只算一次：前向、反向、Q/K 复用。
///
/// 关键优化：`10000^(2i/D)` 只与 i 有关，先算一遍 128 个频率，
/// 再对每个位置做 `theta = pos / freq[i]` 求 cos/sin——
/// 原来在行内循环里重复计算 powf，rows=2048 时要算 26 万次 powf（约 30ms）。
fn build_cos_sin_tab(positions: &[usize], d: usize) -> (Vec<f32>, Vec<f32>) {
    let rows = positions.len();
    let half = d / 2;
    let mut c_tab = vec![0.0f32; rows * half];
    let mut s_tab = vec![0.0f32; rows * half];
    let mut freq = vec![0.0f32; half];
    for i in 0..half {
        freq[i] = 10000f32.powf((2 * i) as f32 / d as f32);
    }
    for r in 0..rows {
        let pos = positions[r] as f32;
        for i in 0..half {
            let theta = pos / freq[i];
            c_tab[r * half + i] = theta.cos();
            s_tab[r * half + i] = theta.sin();
        }
    }
    (c_tab, s_tab)
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

    let mut result = Tensor::new(out_data, x.shape.clone(), x.requires_grad);
    if x.requires_grad {
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
    /// 仅供测试使用；生产代码（attention）用 `rotary_pair` 一次旋转 Q/K。
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn rotary(&self, positions: &[usize]) -> Tensor {
        assert_eq!(self.rank(), 2, "rotary 输入应为 [rows, D]");
        assert_eq!(self.shape[0], positions.len(), "positions 数量必须等于行数");
        let d = self.shape[1];
        assert_eq!(d % 2, 0, "最后一维必须为偶数才能两两配对旋转");
        let (c_tab, s_tab) = build_cos_sin_tab(positions, d);
        rotate_with_tab(self, &c_tab, &s_tab)
    }

    /// 一次建表同时旋转 Q 和 K（两者 positions 相同，三角函数只算一遍）。
    /// 返回 `(rotated_q, rotated_k)`。
    pub fn rotary_pair(
        &self,
        other: &Tensor,
        positions: &[usize],
    ) -> (Tensor, Tensor) {
        debug_assert_eq!(self.shape[0], other.shape[0], "Q/K 行数必须一致");
        let d = self.shape[1];
        let (c_tab, s_tab) = build_cos_sin_tab(positions, d);
        (
            rotate_with_tab(self, &c_tab, &s_tab),
            rotate_with_tab(other, &c_tab, &s_tab),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rotary() {
        // 1. 旋转是正交变换：范数不变
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
        let r = x.rotary(&[3]);
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
        let r2 = x2.rotary(&[0]);
        assert!((r2.data()[0] - 1.0).abs() < 1e-5);
        assert!((r2.data()[3] - 4.0).abs() < 1e-5);

        // 3. 梯度：sum 的梯度是单位向量，经正交矩阵回传后范数不变（= 元素数）
        let x3 = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![1, 6]);
        let loss = x3.rotary(&[2]).sum();
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
        let loss = x.rotary(&[1]).sum();
        loss.backward();
        let (c, s) = (1f32.cos(), 1f32.sin());
        let (ga, gb) = (c + s, -s + c);
        assert!((x.grad()[0] - ga).abs() < 1e-5, "grad[0] = {}", x.grad()[0]);
        assert!((x.grad()[1] - gb).abs() < 1e-5, "grad[1] = {}", x.grad()[1]);
    }
}
