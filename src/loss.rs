//! 损失函数（第 6 课）
//!
//! 损失衡量"模型预测得有多差"，训练就是最小化它。
//! - MSE：回归任务（预测连续数值）
//! - CrossEntropy：分类任务（预测属于哪个类别，LLM 用它）

use crate::tensor::Tensor;

/// 均方误差：loss = mean((pred - target)²)
#[allow(dead_code)] // 回归任务损失函数 API（测试 test_linear_regression_converges 已验证）
pub fn mse_loss(pred: &Tensor, target: &Tensor) -> Tensor {
    pred.sub(target)
        .pow(2.0)
        .sum()
        .mul_scalar(1.0 / pred.numel() as f32)
}

/// 交叉熵损失：适用于"预测类别"的任务。
///
/// 输入：
/// - logits: [B, D] 未归一化的分数
/// - targets: [B] 每个样本的真实类别下标
///
/// 公式：loss = -mean( log_softmax(logits)[i, targets[i]] )
///
/// 实现使用 log-sum-exp 技巧（`log_softmax_last_dim`），
/// 避免先算 softmax（可能下溢到 0）再取 log（log(0) = -inf）的问题。
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    assert_eq!(logits.rank(), 2, "交叉熵的 logits 应为 [B, D]");
    let (b, d) = (logits.shape()[0], logits.shape()[1]);

    // one-hot 编码：正确类别位置为 1，其余为 0
    let mut onehot = vec![0.0f32; b * d];
    for (i, &t) in targets.iter().enumerate() {
        assert!(t < d, "目标类别越界：{} >= {}", t, d);
        onehot[i * d + t] = 1.0;
    }
    let oh = Tensor::from_vec(onehot, vec![b, d]);

    // log_softmax（数值稳定）→ 用 one-hot 取出正确类别的 log 概率 → 取负求均值
    let log_probs = logits.log_softmax_last_dim();
    log_probs
        .mul(&oh)
        .sum_last_dim()
        .neg()
        .sum()
        .mul_scalar(1.0 / b as f32)
}
