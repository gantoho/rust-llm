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
//! 用法：在注意力层内部，对 Q/K 做 `rotary(positions)`，V 不转。

use std::rc::Rc;

use crate::tensor::Tensor;

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
    pub fn rotary(&self, positions: &[usize]) -> Tensor {
        assert_eq!(self.rank(), 2, "rotary 输入应为 [rows, D]");
        assert_eq!(self.shape[0], positions.len(), "positions 数量必须等于行数");
        let (rows, d) = (self.shape[0], self.shape[1]);
        assert_eq!(d % 2, 0, "最后一维必须为偶数才能两两配对旋转");

        let sd = self.data.borrow();
        let mut out_data = vec![0.0f32; rows * d];
        for r in 0..rows {
            let pos = positions[r] as f32;
            for i in 0..d / 2 {
                let theta = pos / 10000f32.powf((2 * i) as f32 / d as f32);
                let (c, s) = (theta.cos(), theta.sin());
                let (a, b) = (sd[r * d + 2 * i], sd[r * d + 2 * i + 1]);
                out_data[r * d + 2 * i] = a * c - b * s;
                out_data[r * d + 2 * i + 1] = a * s + b * c;
            }
        }
        drop(sd);
        let positions_vec = positions.to_vec();

        let mut result = Tensor::new(out_data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for r in 0..rows {
                    let pos = positions_vec[r] as f32;
                    for i in 0..d / 2 {
                        let theta = pos / 10000f32.powf((2 * i) as f32 / d as f32);
                        let (c, s) = (theta.cos(), theta.sin());
                        let (ga, gb) = (g[r * d + 2 * i], g[r * d + 2 * i + 1]);
                        // 反向 = 前向旋转矩阵的转置 R(θ)ᵀ：grad = (ga·c + gb·s, -ga·s + gb·c)
                        sgm[r * d + 2 * i] += ga * c + gb * s;
                        sgm[r * d + 2 * i + 1] += -ga * s + gb * c;
                    }
                }
            }));
        }
        result
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
