//! 自动微分（第 2 课）
//!
//! 自动微分是深度学习的"魔法"：你只需写前向计算，梯度自动算好。
//!
//! 核心思想（线性 tape 方案）：
//! 1. 每次"需要梯度"的算子运算都经 [`record`] 把反向闭包登记到**线性 tape**
//!    —— 一张按前向发生顺序追加的全局条目表，条目里存输出/父张量的身份键
//! 2. 最终调用 `loss.backward()` 时，从 tape 末尾**逆序**扫描：逆序 ==
//!    拓扑逆序（子节点总是比父节点登记得晚），凡是能沿身份键回溯到 loss
//!    的条目就执行其闭包（链式法则），执行完从 tape 上移除
//! 3. 相比旧的「每节点挂 parents + 闭包、DFS 拓扑排序」：建图期不再为
//!    每个节点维护 parents 指针链，反向期不再做图遍历——一次线性扫描搞定

use std::collections::HashSet;
use std::sync::{Arc, LazyLock, Mutex, PoisonError};

use crate::tensor::{Shared, Tensor};

/// 反向函数类型：无参数、无返回值，通过闭包捕获的 `Shared` 句柄直接读写各节点的梯度。
/// 要求 `Send + Sync`：条目登记与执行可能跨线程发生（rayon 并行、数据并行分桶）。
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

/// 全局线性 tape：按前向发生顺序排列的反向条目表。
///
/// 为什么可以是全局单例：当前代码库**没有并发建图、也没有并发 backward**
/// （唯一的多线程前向是训练循环的 batch 预取线程，只读数据、不建图），
/// 单把锁即可；多 micro-batch 的前向会在 tape 上交错登记，
/// 每次 `backward()` 只执行、只移除**能回溯到自己根**的条目，
/// 其它图的条目原样保留（见 [`Tensor::backward`]）。
///
/// 守约：**反向闭包内不得构造 Tensor**——构造算子会经 [`record`] 再锁本表，
/// 而 std 的 `Mutex` 不可重入 ⇒ 同线程二次加锁死锁。
/// 已登记的闭包只读写 `Shared` 梯度/数据缓冲（全库接线点均已核查）。
static TAPE: LazyLock<Mutex<Vec<TapeEntry>>> = LazyLock::new(|| Mutex::new(Vec::new()));

/// 前向算子的统一登记门：把一次反向闭包登记到线性 tape 上。
///
/// - `out`：本次算子的输出张量
/// - `parents`：参与运算的输入张量（可含不需要梯度的输入）
/// - `f`：链式法则闭包，体与旧实现完全一致，只换存放位置
///
/// 只在 `requires_grad` 守卫内调用：no_grad / 冻结路径不登记，tape 不增长。
pub fn record(out: &Tensor, parents: Vec<Tensor>, f: BackwardFn) {
    let entry = TapeEntry {
        out: Shared::as_ptr(&out.grad),
        parents: parents.iter().map(|p| Shared::as_ptr(&p.grad)).collect(),
        f,
        done: false,
    };
    TAPE.lock().unwrap_or_else(PoisonError::into_inner).push(entry);
}

/// 丢弃 tape 上尚未执行的全部条目——仅用于「已确定不会再反向」的丢弃场景。
///
/// 正常训练每步的根 `backward()` 会按可达集回收本次执行的条目，tape 稳态有界，
/// 不需要调用本函数。唯一典型的调用方是 `main.rs` 的 `aux_only_balance`
/// 这类 demo：它每步 forward 整张图却只反辅助损失，主损失那半张图的条目
/// 永远不可达、无法被 backward 回收，只能整条清掉。
#[cfg_attr(not(feature = "gpu"), allow(dead_code))] // 仅 demo 路径使用
pub fn clear_tape() {
    TAPE.lock().unwrap_or_else(PoisonError::into_inner).clear();
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
            g[0] = 1.0;
        }

        let root = Shared::as_ptr(&self.grad);
        let mut tape = TAPE.lock().unwrap_or_else(PoisonError::into_inner);

        let mut reachable: HashSet<usize> = HashSet::new();
        reachable.insert(root);
        for i in (0..tape.len()).rev() {
            if reachable.contains(&tape[i].out) {
                // 克隆 Arc 后立即执行：闭包不碰 tape（见 TAPE 守约），
                // 且不能在持有条目可变借位时执行任何可能重入的东西。
                let f = Arc::clone(&tape[i].f);
                f();
                reachable.extend(tape[i].parents.iter().copied());
                tape[i].done = true;
            }
        }
        // 只移除本次执行的条目；其它图的条目（含未回溯到的更早条目）保留。
        tape.retain(|e| !e.done);
    }
}
