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

        // 迭代式 DFS 拓扑排序（避免递归栈溢出，深层计算图可能有数千节点）。
        // 三色标记法：白色=未访问、灰色=在栈中（正在展开子节点）、黑色=已完成。
        // 用栈模拟递归：每个元素 (node, child_index) 表示"该节点的第 child_index 个子节点待访问"。
        let mut order: Vec<Tensor> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        // 栈元素：(节点, 下一个待展开的子节点索引)
        let mut stack: Vec<(Tensor, usize)> = Vec::new();

        let key = Rc::as_ptr(&self.grad) as usize;
        if visited.insert(key) {
            stack.push((self.clone(), 0));
        }

        while let Some((node, idx)) = stack.last_mut() {
            if *idx < node.parents.len() {
                let child = node.parents[*idx].clone();
                *idx += 1;
                let child_key = Rc::as_ptr(&child.grad) as usize;
                if visited.insert(child_key) {
                    stack.push((child, 0));
                }
            } else {
                let (node, _) = stack.pop().unwrap();
                order.push(node);
            }
        }

        for t in order.iter().rev() {
            if let Some(b) = &t.backward {
                b();
            }
        }
    }
}
