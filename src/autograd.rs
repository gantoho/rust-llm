//! 自动微分（第 2 课）
//!
//! 自动微分是深度学习的"魔法"：你只需写前向计算，梯度自动算好。
//!
//! 核心思想：
//! 1. 每次运算都记录"谁参与了"（parents）和"怎么回传梯度"（backward 闭包）
//! 2. 最终调用 `loss.backward()` 时，按拓扑序从后往前执行所有闭包
//! 3. 链式法则保证每一步的梯度正确传递

use std::collections::HashSet;
use std::rc::Rc;

use crate::tensor::Tensor;

impl Tensor {
    /// 反向传播：从标量 loss 出发，沿计算图逆序传播梯度。
    ///
    /// 原理（第 2 课详解）：
    /// 1. 拓扑排序（DFS）：把计算图排成"先算的先处理"的顺序
    /// 2. 逆序遍历：从离 loss 最远的节点开始，逐个执行 backward 闭包
    /// 3. 每个闭包把"输出的梯度"按链式法则加到"输入的梯度"上
    ///
    /// 拓扑排序用 DFS 实现：递归访问所有父节点，回溯时把自己加入列表。
    /// 用 `Rc::as_ptr(&t.grad)` 去重：`grad` 是每个计算节点独有的 Rc，
    /// 同一张量不会被处理两次。注意不能用 `data` 指针判重——
    /// `reshape` 等视图运算与输入共享 `data`，会误把父节点跳过导致梯度截断。
    pub fn backward(&self) {
        assert_eq!(
            self.rank(),
            0,
            "backward() 只支持标量（0 维）输出，当前形状 {:?}",
            self.shape
        );
        {
            let mut g = self.grad.borrow_mut();
            g[0] = 1.0;
        }

        fn dfs(t: &Tensor, order: &mut Vec<Tensor>, visited: &mut HashSet<usize>) {
            let key = Rc::as_ptr(&t.grad) as usize;
            if !visited.insert(key) {
                return;
            }
            for p in t.parents.iter() {
                dfs(p, order, visited);
            }
            order.push(t.clone());
        }

        let mut order = Vec::new();
        let mut visited = HashSet::new();
        dfs(self, &mut order, &mut visited);

        for t in order.iter().rev() {
            if let Some(b) = &t.backward {
                b();
            }
        }
    }
}
