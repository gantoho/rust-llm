//! 模块抽象（第 5 课）
//!
//! 深度学习里一切"可训练的结构"都是模块（Linear、LayerNorm、Transformer Block...）。
//! `Module` trait 提供统一的参数收集接口，配合 `zero_grad_all` 辅助函数可以：
//! - 收集所有参数（供优化器更新）
//! - 清零所有梯度

use crate::tensor::Tensor;

/// 模块接口：任何可训练结构都实现它
pub trait Module {
    /// 返回模块的所有参数（含嵌套子模块）
    fn parameters(&self) -> Vec<Tensor>;
}

/// 便捷方法：清零所有参数的梯度
pub fn zero_grad_all(module: &dyn Module) {
    for p in module.parameters() {
        p.zero_grad();
    }
}
