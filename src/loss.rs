//! 损失函数（第 6 课）
//!
//! 损失衡量"模型预测得有多差"，训练就是最小化它。
//! - MSE：回归任务（预测连续数值）
//! - CrossEntropy：分类任务（预测属于哪个类别，LLM 用它）

use crate::autograd::record;
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::sync::Arc;

/// 均方误差：loss = mean((pred - target)²)
#[allow(dead_code)] // 回归任务损失函数 API（测试 test_linear_regression_converges 已验证）
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target)
        .pow(2.0)
        .sum()
        .mul_scalar(1.0 / pred.numel() as f32)
}

/// 交叉熵损失：适用于"预测类别"的任务。全部位置都参与（等价于 `mask = None`）。
///
/// 输入：
/// - logits: [B, D] 未归一化的分数
/// - targets: [B] 每个样本的真实类别下标
///
/// 公式：loss = -mean( log_softmax(logits)[i, targets[i]] )
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    cross_entropy_loss_masked(logits, targets, None)
}

/// 带掩码的交叉熵：`mask[i] == false` 的位置既不参与 loss、也不回传梯度。
///
/// 用途是监督微调（SFT）：一段对话里「提问」和「角色标记」不该算进 loss，
/// 只有「回答」部分的 token 才是模型要学的目标。没有掩码的话，模型会连提问的方式
/// 一起拟合，且 loss 被大量无意义的"预测用户下一句"稀释。
///
/// 两个容易踩的点：
/// - **分母是有效位置数，不是 `B`**。若仍除以 `B`，屏蔽比例一变 loss 量级就跟着变，
///   等价于暗中改了学习率；按有效位置数归一化才让不同掩码比例的批次可比。
/// - **`None` 与"全 true"必须等价**，否则预训练与微调的 loss 不可比。
///
/// 融合实现：单次行并行 log-sum-exp 直接得到各行 NLL，不分配 [B, D] 的 log_probs
/// 中间张量（原实现需三遍扫描：log_softmax + gather + 求和）；反向按行重算 softmax，
/// 等价于 log_softmax+gather 反向的链式展开（∂loss/∂logit = mask/valid·(softmax−onehot)）。
/// 行内运算顺序与 `log_softmax_last_dim` 一致，loss 数值逐位不变。
pub fn cross_entropy_loss_masked(
    logits: &Tensor,
    targets: &[usize],
    mask: Option<&[bool]>,
) -> Tensor {
    assert_eq!(logits.rank(), 2, "交叉熵的 logits 应为 [B, D]");
    let (b, d) = (logits.shape()[0], logits.shape()[1]);
    assert_eq!(targets.len(), b, "targets 数量应与 logits 行数一致");
    if let Some(m) = mask {
        assert_eq!(m.len(), b, "mask 长度应与 logits 行数一致");
    }
    for &t in targets {
        assert!(t < d, "目标类别越界：{} >= {}", t, d);
    }

    let is_sup = |i: usize| mask.map_or(true, |m| m[i]);
    // 有效位置数；全被屏蔽时用 1 兜底避免除零（此时 loss 恒为 0，不需要真的回传梯度）
    let valid = (0..b).filter(|&i| is_sup(i)).count().max(1);

    let targets_rc = Arc::new(targets.to_vec());
    let mask_rc = mask.map(|m| Arc::new(m.to_vec()));

    // 行并行前向：运算顺序镜像 log_softmax_last_dim（max → Σexp → ln → 取目标位）；
    // 屏蔽位给 0：既不贡献 loss，也让梯度那条路径彻底断开
    let lb = logits.decode();
    let lr: &[f32] = &lb;
    let t_ref: &[usize] = &targets_rc;
    let row_nll: Vec<f32> = lr
        .par_chunks(d)
        .enumerate()
        .map(|(i, row)| {
            if !is_sup(i) {
                return 0.0;
            }
            let mut maxv = f32::NEG_INFINITY;
            for j in 0..d {
                maxv = maxv.max(row[j]);
            }
            let mut sum_exp = 0.0f32;
            for j in 0..d {
                sum_exp += (row[j] - maxv).exp();
            }
            // sum_exp ≥ 1（至少一项 e^0），ln 天然安全，无需防零常数
            -(row[t_ref[i]] - maxv - sum_exp.ln())
        })
        .collect();
    drop(lb);

    // 标量 loss = Σ(每行 -log_prob) / 有效位置数（保序收集 + 顺序求和，与原实现逐位一致）
    let mean_loss: f32 = row_nll.iter().sum::<f32>() / valid as f32;
    let result = Tensor::new(vec![mean_loss], vec![], logits.req());
    if logits.req() {
        let rg = result.grad.clone();
        let sg = logits.grad.clone();
        let ld = logits.data.clone();
        record(&result, vec![logits.clone()], Arc::new(move || {
            let g = rg.borrow()[0];
            // 反向重算行 softmax（不存中间张量）；不加 EPS，与 log_softmax 数值一致
            let lb2 = ld.decode();
            let lr2: &[f32] = &lb2;
            let mut sgm = sg.borrow_mut();
            let t: &[usize] = &targets_rc;
            // 先降级为共享切片（&[bool] 是 Sync）再进并行闭包
            let m_ref: Option<&[bool]> = mask_rc.as_ref().map(|m| m.as_slice());
            let scale = g / valid as f32;
            sgm.par_chunks_mut(d).enumerate().for_each(|(i, row)| {
                let sup = m_ref.map_or(true, |m| m[i]);
                if !sup {
                    return; // 屏蔽位不回传梯度
                }
                let base = i * d;
                let mut maxv = f32::NEG_INFINITY;
                for j in 0..d {
                    maxv = maxv.max(lr2[base + j]);
                }
                let mut sum_exp = 0.0f32;
                for j in 0..d {
                    sum_exp += (lr2[base + j] - maxv).exp();
                }
                let inv = 1.0 / sum_exp;
                for j in 0..d {
                    row[j] += scale * ((lr2[base + j] - maxv).exp() * inv);
                }
                row[t[i]] -= scale;
            });
        }));
    }
    result
}
