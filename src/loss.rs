//! 损失函数（第 6 课）
//!
//! 损失衡量"模型预测得有多差"，训练就是最小化它。
//! - MSE：回归任务（预测连续数值）
//! - CrossEntropy：分类任务（预测属于哪个类别，LLM 用它）

use crate::layers::softmax;
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
/// 公式：loss = -mean( log softmax(logits)[i, targets[i]] )
///
/// 原理（第 6 课详解）：
/// 我们希望模型给正确类别分配接近 1 的概率。
/// softmax 把分数变成概率，log 后取负——正确类概率越高，loss 越低。
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

    // log_softmax(logits)，再用 one-hot 取出每个样本正确类别的 log 概率
    let log_probs = softmax(logits).log();
    // 每个样本的 loss = -sum(onehot * log_probs)（只有正确类别位置非 0）
    // 再对所有样本取平均
    log_probs
        .mul(&oh)
        .sum_last_dim()
        .neg()
        .sum()
        .mul_scalar(1.0 / b as f32)
}
