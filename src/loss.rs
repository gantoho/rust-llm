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
/// 使用 gather 索引直接取正确类别的 log_prob，避免分配 [B, vocab_size] 的 one-hot 矩阵。
pub fn cross_entropy_loss(logits: &Tensor, targets: &[usize]) -> Tensor {
    assert_eq!(logits.rank(), 2, "交叉熵的 logits 应为 [B, D]");
    let (b, d) = (logits.shape()[0], logits.shape()[1]);
    for &t in targets {
        assert!(t < d, "目标类别越界：{} >= {}", t, d);
    }

    let log_probs = logits.log_softmax_last_dim();

    // gather 操作：直接取 log_probs[i, targets[i]]，省掉 one-hot 分配和乘法
    let lp = log_probs.data.borrow();
    let mut gathered = vec![0.0f32; b];
    for (i, &t) in targets.iter().enumerate() {
        gathered[i] = lp[i * d + t];
    }
    drop(lp);

    // 构建标量 loss = -mean(gathered)
    let mean_loss: f32 = -gathered.iter().sum::<f32>() / b as f32;
    let mut result = Tensor::new(vec![mean_loss], vec![], log_probs.requires_grad);
    if log_probs.requires_grad {
        let rg = result.grad.clone();
        let sg = log_probs.grad.clone();
        let targets_rc = std::rc::Rc::new(targets.to_vec());
        let b2 = b;
        let d2 = d;
        result.parents = std::rc::Rc::new(vec![log_probs]);
        result.backward = Some(std::rc::Rc::new(move || {
            let g = rg.borrow()[0];
            let mut sgm = sg.borrow_mut();
            let t = targets_rc.clone();
            // 反向：d_loss/d_log_probs[i, targets[i]] = -1/B，其余为 0
            let scale = -g / b2 as f32;
            for (i, &tgt) in t.iter().enumerate() {
                sgm[i * d2 + tgt] += scale;
            }
        }));
    }
    result
}
