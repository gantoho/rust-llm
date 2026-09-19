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
/// 使用 gather 索引直接取正确类别的 log_prob，避免分配 [B, vocab_size] 的 one-hot 矩阵。
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

    let log_probs = logits.log_softmax_last_dim();

    // gather 操作：直接取 log_probs[i, targets[i]]，省掉 one-hot 分配和乘法
    let lp = log_probs.data.borrow();
    let mut gathered = vec![0.0f32; b];
    for (i, &t) in targets.iter().enumerate() {
        // 屏蔽位不取 log_prob：给 0 既不贡献 loss，也让梯度那条路径彻底断开
        gathered[i] = if is_sup(i) { lp[i * d + t] } else { 0.0 };
    }
    drop(lp);

    // 构建标量 loss = -mean(gathered)，分母为有效位置数
    let mean_loss: f32 = -gathered.iter().sum::<f32>() / valid as f32;
    let mut result = Tensor::new(vec![mean_loss], vec![], log_probs.req());
    if log_probs.req() {
        let rg = result.grad.clone();
        let sg = log_probs.grad.clone();
        let targets_rc = std::rc::Rc::new(targets.to_vec());
        let mask_rc = mask.map(|m| std::rc::Rc::new(m.to_vec()));
        let d2 = d;
        result.parents = std::rc::Rc::new(vec![log_probs]);
        result.backward = Some(std::rc::Rc::new(move || {
            let g = rg.borrow()[0];
            let mut sgm = sg.borrow_mut();
            let t = targets_rc.clone();
            // 反向：d_loss/d_log_probs[i, targets[i]] = -1/valid，其余（含屏蔽位）为 0
            let scale = -g / valid as f32;
            for (i, &tgt) in t.iter().enumerate() {
                let sup = mask_rc.as_ref().map_or(true, |m| m[i]);
                if sup {
                    sgm[i * d2 + tgt] += scale;
                }
            }
        }));
    }
    result
}
