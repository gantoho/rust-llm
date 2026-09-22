//! 模块抽象（第 5 课）
//!
//! 深度学习里一切"可训练的结构"都是模块（Linear、LayerNorm、Transformer Block...）。
//! `Module` trait 提供统一的参数收集接口，配合 `zero_grad_all` 辅助函数可以：
//! - 收集所有参数（供优化器更新）
//! - 清零所有梯度

use crate::tensor::Tensor;

/// 模块接口：任何可训练结构都实现它
pub trait Module {
    /// 返回模块的所有参数（含嵌套子模块，含被冻结的）
    fn parameters(&self) -> Vec<Tensor>;

    /// 返回**可训练**参数子集（默认 = 全部参数）。
    ///
    /// 判定依据是 [`Tensor::requires_grad`]：LoRA 微调会把主干参数置为 false，
    /// 于是这里只剩适配层的 `lora_a` / `lora_b`。用于统计「本次训练到底动了多少参数」，
    /// 以及让调用方看清冻结是否真的生效。
    fn trainable_parameters(&self) -> Vec<Tensor> {
        self.parameters()
            .into_iter()
            .filter(|p| p.requires_grad())
            .collect()
    }
}

/// 便捷方法：清零所有参数的梯度
pub fn zero_grad_all(module: &dyn Module) {
    for p in module.parameters() {
        p.zero_grad();
    }
}
