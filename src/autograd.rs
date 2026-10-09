//! 自动微分（第 2 课）
//!
//! 自动微分是深度学习的"魔法"：你只需写前向计算，梯度自动算好。
//!
//! 核心思想（线性 tape 方案）：
//! 1. 每次"需要梯度"的算子运算都经 [`record`] 把反向闭包登记到**线性 tape**
//!    —— 一张按前向发生顺序追加的线程本地条目表，条目里存输出/父张量的身份键
//! 2. 最终调用 `loss.backward()` 时，从 tape 末尾**逆序**扫描：逆序 ==
//!    拓扑逆序（子节点总是比父节点登记得晚），凡是能沿身份键回溯到 loss
//!    的条目就执行其闭包（链式法则），执行完从 tape 上移除
//! 3. 相比旧的「每节点挂 parents + 闭包、DFS 拓扑排序」：建图期不再为
//!    每个节点维护 parents 指针链，反向期不再做图遍历——一次线性扫描搞定

use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::Arc;

use crate::tensor::{Shared, Tensor};

/// 反向函数类型：无参数、无返回值，通过闭包捕获的 `Shared` 句柄直接读写各节点的梯度。
///
/// 条目登记与执行都在**同一线程**（tape 是 thread_local，见 `TAPE`）；
/// 保留 `Send + Sync` 是因为闭包捕获的全是 `Shared` 句柄（天然满足），
/// 且部分闭包体内部还用 rayon 并行写 grad 缓冲（loss / distill 反向）。
pub type BackwardFn = Arc<dyn Fn() + Send + Sync>;

/// tape 条目：一次"需要梯度"的算子调用在线性 tape 上留下的记录。
struct TapeEntry {
    /// 本条目输出张量的**身份键**（其 `grad` 缓冲的地址）。
    /// 每个张量的梯度缓冲唯一（视图运算也分配独立 grad），
    /// 因此 grad 地址可以唯一标识计算图上的一个节点。
    out: usize,
    /// 各父张量的身份键：反向时沿这些键向上游回溯。
    /// 不需要梯度的父张量也登记——它们的键回溯不到任何条目，闭包仍会
    /// 直接写它们的梯度缓冲（elementwise 半冻结语义）。
    parents: Vec<usize>,
    /// 链式法则闭包：把已算出的"本节点梯度"分发到各父节点的梯度缓冲。
    f: BackwardFn,
    /// 本次 `backward()` 是否已执行；扫描结束后统一从 tape 移除。
    done: bool,
}

thread_local! {
    /// 线程本地线性 tape：按前向发生顺序排列的反向条目表。
    ///
    /// 为什么是 thread_local 而不是全局单例：全库的建图与 backward 都发生在
    /// **调用线程**（唯一的多线程前向是训练循环的 batch 预取线程，只读数据、
    /// 不建图；serve 的推理线程不 backward；rayon 只并行 data/grad 缓冲、
    /// 不构造 Tensor），每线程一张表即可，还顺带消除两类全局态问题：
    /// - cargo test 并行下各测试线程的 tape 互不干扰；
    /// - 推理线程登记的条目随线程退出自动回收，不再在全局表上无限累积。
    ///
    /// 守约（同线程约束）：**登记与 backward 必须同线程**——张量在 A 线程建图、
    /// 拿到 B 线程去 `backward()`，B 的 tape 上没有这些条目，可达集裁剪会
    /// 静默跳过、什么都不算。当前代码库全在调用线程完成（见上）。
    ///
    /// 同一 tape 上多个 micro-batch 的前向会交错登记，每次 [`Tensor::backward`]
    /// 只执行、只移除**能回溯到自己根**的条目，其它图的条目原样保留。
    ///
    /// 守约：**反向闭包内不得构造 Tensor**——构造算子会经 [`record`] 再次
    /// `borrow_mut()` 本表，`RefCell` 不可重入 ⇒ 同线程二次可变借位直接 panic
    /// （旧全局 Mutex 版是死锁，语义守约不变、失败方式更显式）。
    /// 已登记的闭包只读写 `Shared` 梯度/数据缓冲（全库接线点均已核查）。
    static TAPE: RefCell<Vec<TapeEntry>> = RefCell::new(Vec::new());
}

/// 把惰性分配的空 grad 缓冲补齐到全长（no_grad 构造的张量进图的兜底）。
///
/// 不变量：进入反向后，闭包对**全部 parents** 无条件写全长缓冲
/// （含不需要梯度的 mask/frozen）。空缓冲只可能来自 no_grad 下构造的
/// 张量跨模式进图（如 distill 教师 logits、align ref_logprobs），
/// 首次登记时在这里按 numel 补齐。补齐不改变节点身份
/// （键 = `Shared::as_ptr(&grad)`，Arc 地址不随内部 Vec 重分配移动）。
///
/// 守卫在函数返回前释放——调用方随后才借 TAPE（thread_local `RefCell`，
/// 不参与跨线程锁序）；顺序恒为「grad（释放）→ TAPE」，而 backward 持 TAPE
/// 时才借 grad——两步反向不重叠，不构成环；且 TAPE 只在本线程被借，
/// 跨线程死锁在结构上不可能发生。
fn ensure_grad(t: &Tensor) {
    let n = t.numel();
    let mut g = t.grad.borrow_mut();
    if g.len() != n {
        assert!(
            g.is_empty(),
            "grad 缓冲长度 {} 与元素数 {} 都不匹配（既非空也非全长）",
            g.len(),
            n
        );
        g.resize(n, 0.0);
    }
}

/// 前向算子的统一登记门：把一次反向闭包登记到线性 tape 上。
///
/// - `out`：本次算子的输出张量
/// - `parents`：参与运算的输入张量（可含不需要梯度的输入）
/// - `f`：链式法则闭包，体与旧实现完全一致，只换存放位置
///
/// 只在 `requires_grad` 守卫内调用：no_grad / 冻结路径不登记，tape 不增长。
/// 入口先 [`ensure_grad`] 补齐 out 与全部 parents 的空缓冲（跨模式进图兜底），
/// 补齐完释放 grad 锁后才借本线程的 TAPE。
pub fn record(out: &Tensor, parents: Vec<Tensor>, f: BackwardFn) {
    ensure_grad(out);
    for p in &parents {
        ensure_grad(p);
    }
    let entry = TapeEntry {
        out: Shared::as_ptr(&out.grad),
        parents: parents.iter().map(|p| Shared::as_ptr(&p.grad)).collect(),
        f,
        done: false,
    };
    TAPE.with(|t| t.borrow_mut().push(entry));
}

/// 丢弃 tape 上尚未执行的全部条目——仅用于「已确定不会再反向」的丢弃场景。
///
/// 正常训练每步的根 `backward()` 会按可达集回收本次执行的条目，tape 稳态有界，
/// 不需要调用本函数。唯一典型的调用方是 `main.rs` 的 `aux_only_balance`
/// 这类 demo：它每步 forward 整张图却只反辅助损失，主损失那半张图的条目
/// 永远不可达、无法被 backward 回收，只能整条清掉。
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // 仅 demo 路径使用
pub fn clear_tape() {
    TAPE.with(|t| t.borrow_mut().clear());
}

impl Tensor {
    /// 反向传播：从标量 loss 出发，逆序执行线性 tape 上可达的反向闭包。
    ///
    /// 原理（第 2 课详解）：
    /// 1. 置根梯度为 1（链式法则的起点 dL/dL = 1）
    /// 2. 从 tape 末尾向前逆序扫描。tape 按前向顺序追加 ⇒ 逆序 == 拓扑逆序
    ///    （子节点总是比父节点登记得晚），所以单趟扫描就是合法的反向执行序
    /// 3. 用身份键可达集裁剪：只有能沿"输出 → 父"键回溯到根的条目才执行，
    ///    闭包执行时把父键加入可达集
    /// 4. 执行过的条目从 tape 移除；回溯不到根的条目（其它交错前向的图、
    ///    故意不反的分支）原样保留，等它们自己的 backward
    ///
    /// 不能简单"执行到根条目为止并截断"：流水线调度会让多个 micro-batch 的
    /// 前向在 tape 上交错（且反向不一定是先进先出），按位置截断会误杀别的图。
    pub fn backward(&self) {
        assert_eq!(
            self.rank(),
            0,
            "backward() 只支持标量（0 维）输出，当前形状 {:?}",
            self.shape
        );
        {
            let mut g = self.grad.borrow_mut();
            if g.is_empty() {
                // 根 loss 若在 no_grad 下构造（空缓冲），补齐后再置起点
                g.resize(self.numel(), 0.0);
            }
            g[0] = 1.0;
        }

        let root = Shared::as_ptr(&self.grad);
        // 借位不逃逸出 `.with`（higher-ranked 闭包约束），整段扫描在闭包内完成；
        // 期间持有本线程 tape 的可变借位，反向闭包重入本表会 borrow panic（见守约）。
        TAPE.with(|t| {
            let mut tape = t.borrow_mut();

            let mut reachable: HashSet<usize> = HashSet::new();
            reachable.insert(root);
            for i in (0..tape.len()).rev() {
                if reachable.contains(&tape[i].out) {
                    // 克隆 Arc 后立即执行：闭包不碰 tape（见 TAPE 守约——RefCell
                    // 下重入是 borrow panic 而非死锁），且不能在持有条目可变借位时
                    // 执行任何可能重入本表的东西。
                    let f = Arc::clone(&tape[i].f);
                    f();
                    reachable.extend(tape[i].parents.iter().copied());
                    tape[i].done = true;
                }
            }
            // 只移除本次执行的条目；其它图的条目（含未回溯到的更早条目）保留。
            tape.retain(|e| !e.done);
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 可达集裁剪：故意不反的分支条目留在 tape 上、不影响本次 backward；
    /// 等它自己的根 backward 时再执行（懒回收语义）。
    #[test]
    fn reachable_prune_keeps_cold_branch() {
        clear_tape(); // 隔离：本线程可能跑过留下冷条目的前序测试
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let w1 = Tensor::param(vec![0.5; 4], vec![2, 2]);
        let w2 = Tensor::param(vec![2.0; 4], vec![2, 2]);

        let a = x.mul(&w1); // 走反向的分支
        let cold = x.mul(&w2); // 冷分支：本次不反
        let loss = a.sum();
        loss.backward();

        // 活分支梯度：dx = w1 ⊙ 1、dw1 = x ⊙ 1
        assert_eq!(x.grad(), vec![0.5; 4]);
        assert_eq!(w1.grad(), vec![1.0, 2.0, 3.0, 4.0]);
        // 冷分支条目没被执行：w2 一点梯度都没攒到
        assert_eq!(w2.grad(), vec![0.0; 4]);

        // 冷分支的根 backward：条目仍在 tape 上，此时才执行
        cold.sum().backward();
        assert_eq!(w2.grad(), vec![1.0, 2.0, 3.0, 4.0]); // dw2 = x ⊙ 1
        // x 追加冷分支贡献（mul 反向 dx = w2 ⊙ upstream）：0.5 + 2.0
        assert_eq!(x.grad(), vec![2.5; 4]);
        clear_tape(); // 清场，不让冷条目泄漏给后续测试
    }

    /// 交错两图独立 backward：先反 A 只回收 A 的条目，B 的条目原样保留，
    /// 随后反 B 结果照常（流水线 micro-batch 交错登记语义）。
    #[test]
    fn interleaved_graphs_backward_independently() {
        clear_tape();
        let p = Tensor::param(vec![1.0, -2.0, 0.5, 3.0], vec![2, 2]);
        let q = Tensor::param(vec![4.0, 0.0, -1.0, 2.0], vec![2, 2]);

        // 两张图交错登记到同一 tape
        let loss_a = p.mul_scalar(2.0).sum();
        let loss_b = q.neg().sum();
        let loss_c = p.add(&q).sum();

        loss_b.backward();
        assert_eq!(q.grad(), vec![-1.0; 4]);
        assert_eq!(p.grad(), vec![0.0; 4]); // A/C 都还没反

        loss_a.backward();
        assert_eq!(p.grad(), vec![2.0; 4]);
        assert_eq!(q.grad(), vec![-1.0; 4]); // 不受 A 反向影响

        // C 的条目历经两次 backward 仍保留：p、q 各追加一份上游为 1 的贡献
        loss_c.backward();
        assert_eq!(p.grad(), vec![3.0; 4]);
        assert_eq!(q.grad(), vec![0.0; 4]);
        clear_tape();
    }

    /// tape 作用域：clear_tape 丢弃全部待执行条目后，backward 只能置根
    /// 起点、算不出任何梯度；之后新登记的图照常工作。
    #[test]
    fn clear_tape_discards_pending_entries() {
        clear_tape();
        let x = Tensor::param(vec![1.0, 2.0], vec![2]);
        let loss = x.mul_scalar(3.0).sum();
        clear_tape(); // 丢弃：本次 backward 无可执行条目
        loss.backward();
        assert_eq!(x.grad(), vec![0.0, 0.0]);

        // 新图照常登记、照常执行：d(sum(5x))/dx = 5（逐分量常数）
        x.mul_scalar(5.0).sum().backward();
        assert_eq!(x.grad(), vec![5.0, 5.0]);
        clear_tape();
    }

    /// 中心差分数值梯度与解析梯度对拍（复合函数 f = sum(tanh(x@W) ⊙ c)）：
    /// 覆盖 matmul / tanh / mul / sum 四段链式法则。
    #[test]
    fn finite_difference_matches_analytic() {
        clear_tape();
        let x0 = vec![0.3, -0.7, 1.1, 0.2, 0.5, -0.4]; // 2×3
        let w0 = vec![0.2, -0.1, 0.6, 0.9, -0.3, 0.4]; // 3×2
        let c0 = vec![0.8, -0.5, 1.2, 0.7]; // 2×2（同形状参数，零广播歧义）

        // 解析梯度：一次前向 + backward
        let x = Tensor::param(x0.clone(), vec![2, 3]);
        let w = Tensor::param(w0.clone(), vec![3, 2]);
        let c = Tensor::param(c0.clone(), vec![2, 2]);
        x.matmul(&w).tanh().mul(&c).sum().backward();
        let (gx, gw, gc) = (x.grad(), w.grad(), c.grad());

        // 数值梯度：叶子用 from_vec（requires_grad = false ⇒ 不登记 tape），
        // 只跑前向；中心差分 (f(+ε)−f(−ε))/2ε
        let f = |x: &[f32], w: &[f32], c: &[f32]| -> f32 {
            Tensor::from_vec(x.to_vec(), vec![2, 3])
                .matmul(&Tensor::from_vec(w.to_vec(), vec![3, 2]))
                .tanh()
                .mul(&Tensor::from_vec(c.to_vec(), vec![2, 2]))
                .sum()
                .item()
        };
        let eps = 1e-3;
        // 逐分量扰动：block 0=x、1=w、2=c，加 ±ε 后重跑前向求值
        let value = |block: usize, i: usize, e: f32| -> f32 {
            let (mut x, mut w, mut c) = (x0.clone(), w0.clone(), c0.clone());
            match block {
                0 => x[i] += e,
                1 => w[i] += e,
                _ => c[i] += e,
            }
            f(&x, &w, &c)
        };
        let check = |ana: &[f32], block: usize| {
            for i in 0..ana.len() {
                let num = (value(block, i, eps) - value(block, i, -eps)) / (2.0 * eps);
                assert!(
                    (num - ana[i]).abs() < 5e-3,
                    "block {block}[{i}]: analytic {}, numeric {num}",
                    ana[i]
                );
            }
        };
        check(&gx, 0);
        check(&gw, 1);
        check(&gc, 2);
        clear_tape();
    }
}

