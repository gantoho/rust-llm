//! 张量库（第 1-4 课）
//!
//! 功能清单：
//! - 构造：from_vec / param
//! - 访问：data / set_data / grad / zero_grad / shape / item
//! - 形状：reshape / transpose(2D) / permute(任意维)
//! - 逐元素：add / sub / mul / div（**支持广播**）/ add_scalar / mul_scalar
//! - 激活：neg / relu / tanh / gelu / log / pow / sqrt
//! - 矩阵：matmul（2D 与 3D 批量）
//! - 归约：sum / sum_last_dim / softmax_last_dim / log_softmax_last_dim
//! - 索引：gather_rows（Embedding 用）
//!
//! 自动微分（backward）见 `src/autograd.rs`，
//! 旋转位置编码（rotary）见 `src/rope.rs`。

use rayon::prelude::*;
use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

/// 防除零小常数（div / log 反向用）
const EPS: f32 = 1e-8;

// permute 的映射表缓存。
// 训练中同一形状每步反复出现（Q/K/V 拆头 [0,2,1,3]、Kᵀ [0,2,1]、合头 [0,2,1,3]），
// map 只依赖 (源形状, dims)，建一次后用 Rc 共享给前向/反向，避免每步重建几十万元素的下标表。
// 用 thread_local 而非 static：Rc 不是 Send/Sync，而本项目训练/推理是单线程的。
thread_local! {
    static PERMUTE_MAP_CACHE: RefCell<HashMap<(Vec<usize>, Vec<usize>), Rc<Vec<usize>>>> =
        RefCell::new(HashMap::new());
}

// dropout 的独立随机流计数器：每次 dropout 调用递增。
// 旧实现按「数据指针 + 首元素」派生种子，Rust 分配器会复用同一地址，
// 导致同一 dropout 位点在每步几乎拿到相同 mask —— 那不是 dropout，
// 而是「永久丢掉固定通道、放大其余通道」的固定掩码，正则化效果完全失效。
thread_local! {
    static DROPOUT_STREAM: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0x2545_F491_4F6C_DD1D) };
}

// 推理模式开关（no_grad）：置 false 时所有算子的"是否建图"判断一律为假，
// 既不挂 parents / backward 闭包，也不分配梯度缓冲。
// 推理不需要反向，建图是纯开销 —— 单 token 前向时，分配 Rc + 闭包的代价
// 甚至超过矩阵乘本身。
thread_local! {
    static GRAD_ENABLED: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

/// 当前是否处于"需要梯度"模式（默认 true）。
pub fn grad_enabled() -> bool {
    GRAD_ENABLED.with(|c| c.get())
}

/// 在禁用梯度（推理）模式下执行 `f`，语义同 PyTorch 的 `torch.no_grad()`：
/// 块内产生的张量一律不参与自动微分，也不分配梯度缓冲。
/// 通过 RAII 守卫恢复原状态，`f` 内即使 panic 也不会把开关卡在 off。
pub fn no_grad<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            GRAD_ENABLED.with(|c| c.set(self.0));
        }
    }
    let prev = GRAD_ENABLED.with(|c| c.replace(false));
    let _restore = Restore(prev);
    f()
}

/// 反向函数类型：无参数、无返回值，通过闭包捕获的 Rc 句柄直接读写各节点的梯度
type BackwardFn = Rc<dyn Fn()>;

/// 张量结构体
///
/// 内部使用 `Rc<RefCell<_>>` 共享可变数据：
/// - `Rc`    让多个张量可以"引用同一个底层数据"
/// - `RefCell` 允许在运行时借用可变
#[derive(Clone)]
pub struct Tensor {
    pub(crate) data: Rc<RefCell<Vec<f32>>>,
    pub(crate) shape: Vec<usize>,
    pub(crate) grad: Rc<RefCell<Vec<f32>>>,
    pub(crate) requires_grad: bool,
    /// 父节点列表。
    /// 注意：用 `Rc<Vec<_>>` 而不是 `Vec<Tensor>`——
    /// 若直接存 Vec，`derive(Clone)` 会递归深拷贝整棵祖先计算图，
    /// 深层图上每次建节点都是 O(图深) 的灾难。用 Rc 共享后克隆是 O(1)。
    pub(crate) parents: Rc<Vec<Tensor>>,
    pub(crate) backward: Option<BackwardFn>,
}

// ==================== 广播工具（第 3 课） ====================

/// 计算两个形状广播后的形状（numpy 广播规则，从右向左对齐）：
/// - 维度相等或其中一个为 1 即可广播
fn broadcast_shapes(a: &[usize], b: &[usize]) -> Option<Vec<usize>> {
    let n = a.len().max(b.len());
    let mut out = vec![1usize; n];
    for i in 0..n {
        let da = if i + a.len() >= n {
            a[i + a.len() - n]
        } else {
            1
        };
        let db = if i + b.len() >= n {
            b[i + b.len() - n]
        } else {
            1
        };
        if da == db {
            out[i] = da;
        } else if da == 1 {
            out[i] = db;
        } else if db == 1 {
            out[i] = da;
        } else {
            return None;
        }
    }
    Some(out)
}

/// 计算源形状 src 广播到目标形状 target 时，每个目标元素对应的源展平下标。
/// src 的维度必须 <= target 且右对齐；src 中大小为 1 的维度索引固定为 0。
fn broadcast_map(target: &[usize], src: &[usize]) -> Vec<usize> {
    let offset = target.len() - src.len();
    let total: usize = target.iter().product();
    let mut map = vec![0usize; total];
    // 热路径：训练中每次前向都要重建广播 map（mask/偏置/LayerNorm），
    // 原来每元素 `vec![0; rank]` 是一次堆分配，几百万元素 = 每次前向几百万次 malloc。
    // 张量最多 4 维，用固定栈数组即可（超过 8 维防御性断言）。
    assert!(target.len() <= 8, "广播维度过多：{}", target.len());
    let mut t_idx = [0usize; 8];
    for flat in 0..total {
        // 反解目标多维索引（行优先）
        let mut r = flat;
        for d in (0..target.len()).rev() {
            t_idx[d] = r % target[d];
            r /= target[d];
        }
        // 映射到源索引并展平
        let mut s_flat = 0usize;
        for d in 0..src.len() {
            let td = t_idx[d + offset];
            let sd = if src[d] == 1 { 0 } else { td };
            s_flat = s_flat * src[d] + sd;
        }
        map[flat] = s_flat;
    }
    map
}

/// 二进制广播时，目标元素下标 t 到"源元素展平下标"的映射方式。
/// - `Ident`：两形状相同，直接 1:1
/// - `Mod(n)`：src 是 target 的右后缀且无大小为 1 的维度，源下标 = t % n
///   （覆盖偏置 [d]→[rows,d]、mask [t,tt]→[bh,t,tt] 等训练热路径，免建 4-16MB map）
/// - `Map(m)`：通用情况，查预建表（Rc 共享给反向闭包，不克隆）
enum SrcIdx {
    Ident,
    Mod(usize),
    Map(Rc<Vec<usize>>),
}

/// 查源下标
fn src_idx(t: usize, s: &SrcIdx) -> usize {
    match s {
        SrcIdx::Ident => t,
        SrcIdx::Mod(n) => t % n,
        SrcIdx::Map(m) => m[t],
    }
}

/// SrcIdx 的跨线程轻量视图（rayon 并行闭包用）。
/// `Rc<Vec<usize>>` 不是 Sync，不能直接进并行闭包；这里借用它的切片共享只读访问。
#[derive(Clone, Copy)]
enum SrcIdxView<'a> {
    Ident,
    Mod(usize),
    Map(&'a [usize]),
}

impl<'a> From<&'a SrcIdx> for SrcIdxView<'a> {
    fn from(s: &'a SrcIdx) -> Self {
        match s {
            SrcIdx::Ident => SrcIdxView::Ident,
            SrcIdx::Mod(n) => SrcIdxView::Mod(*n),
            SrcIdx::Map(m) => SrcIdxView::Map(m.as_slice()),
        }
    }
}

impl SrcIdxView<'_> {
    fn idx(&self, t: usize) -> usize {
        match self {
            SrcIdxView::Ident => t,
            SrcIdxView::Mod(n) => t % n,
            SrcIdxView::Map(m) => m[t],
        }
    }
}

/// 矩阵乘数据计算（GPU 优先，失败自动回退 CPU）。
/// 行优先存储：out[B,M,N] = a[B,M,K] @ b[B,K,N]；batch=1 时即普通 2D 矩阵乘。
/// `a_t`/`b_t` 为转置访问标志：为 true 时物理 a/b 分别是 [B,K,M]、[B,N,K]，
/// GPU 内核直接按转置读；CPU 回退时先把转置请求物化为逻辑矩阵再算。
fn matmul_data(
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
    a_t: bool,
    b_t: bool,
) -> Vec<f32> {
    #[cfg(feature = "gpu")]
    if let Some(v) = crate::gpu::matmul(a, b, m, k, n, batch, a_t, b_t) {
        return v;
    }
    // CPU 回退：a_t 时物理 a 是 [B,K,M]，转置成逻辑 [B,M,K]；b_t 同理 [B,N,K] -> [B,K,N]
    let a_owned;
    let b_owned;
    let (a2, b2): (&[f32], &[f32]) = match (a_t, b_t) {
        (false, false) => (a, b),
        (true, false) => {
            a_owned = transpose_flat(a, k, m, batch);
            (a_owned.as_slice(), b)
        }
        (false, true) => {
            b_owned = transpose_flat(b, n, k, batch);
            (a, b_owned.as_slice())
        }
        (true, true) => {
            a_owned = transpose_flat(a, k, m, batch);
            b_owned = transpose_flat(b, n, k, batch);
            (a_owned.as_slice(), b_owned.as_slice())
        }
    };
    let mut out = vec![0.0f32; batch * m * n];
    // 按**输出行**并行：把 (batch × m) 行拉平后交给 rayon，行与行之间无依赖。
    // 旧实现按 batch 并行，而 Linear 会把 3D 输入展平成 2D（batch=1）调用，
    // 于是注意力/MLP/输出头这些最重的矩阵乘全部退化成单线程三重循环，
    // 多核完全用不上 —— 这是训练与推理的主要瓶颈。
    //
    // 循环顺序从 i-j-k 改为 i-k-j（axpy 累加）：
    // - b 的一行、out 的一行都是**连续**访问，对缓存和自动向量化友好；
    // - 原 i-j-k 里 b[kk*n+j] 每步跨 n 个元素，命中率极差。
    // 每个输出元素仍在 kk 上按同样顺序累加，浮点结果与旧实现逐位一致。
    //
    // 2026-09-16 追加 **MC×NC 双重分块**（此前是"每行流一遍 B"，纯访存瓶颈）：
    // 以输出头 m=4096,k=128,n=8192 为例，一张 4 MB 的 B 被反复流 4096 次
    // ≈ 17 GB 访存，一次前向就把内存带宽打满 —— 整机有效算力只有 4 GFLOP/s，
    // 实测瓶颈压根不是算力，是访存。分块后的数据复用：
    // - b 的一条长 nc 的切片（nc×4 字节）留在 L1，被本块 MC 行**复用 MC 次**；
    // - 输出块留在 L2/寄存器，整个 k 循环只出入内存一次。
    // B 的访存量因此降到约 1/MC，matmul 从访存瓶颈变成算力瓶颈。
    // 累加顺序仍是"每个输出元素按 kk 升序累加"，浮点结果与分块前**逐位一致**。
    const MC: usize = 64; // 输出行块：一个 rayon 任务处理 64 行
    const NC: usize = 512; // 输出列块：B 切片 512×4 = 2 KB，留在 L1 被 64 行复用

    out.par_chunks_mut(MC * n)
        .enumerate()
        .for_each(|(blk, out_blk)| {
            let rows = out_blk.len() / n;
            let first_row = blk * MC;
            // 行块可能跨 batch 边界，所以每行单独算 a/b 的基址
            let mut a_bases = [0usize; MC];
            let mut b_bases = [0usize; MC];
            for r in 0..rows {
                let grow = first_row + r;
                let bi = grow / m;
                a_bases[r] = (bi * m + grow % m) * k;
                b_bases[r] = bi * k * n;
            }
            let mut acc = vec![0.0f32; rows * NC];
            let mut nb = 0;
            while nb < n {
                let nc = NC.min(n - nb);
                acc[..rows * nc].fill(0.0);
                for kk in 0..k {
                    for r in 0..rows {
                        let a_v = a2[a_bases[r] + kk];
                        let b_off = b_bases[r] + kk * n + nb;
                        let b_row = &b2[b_off..b_off + nc];
                        let acc_row = &mut acc[r * nc..r * nc + nc];
                        for (o, &b_v) in acc_row.iter_mut().zip(b_row) {
                            *o += a_v * b_v;
                        }
                    }
                }
                for r in 0..rows {
                    let dst = r * n + nb;
                    out_blk[dst..dst + nc].copy_from_slice(&acc[r * nc..r * nc + nc]);
                }
                nb += NC;
            }
        });
    out
}

/// 展平批矩阵转置：[B, rows, cols] -> [B, cols, rows]（CPU 回退用，GPU 路径不需要）
///
/// 按 batch 并行：注意力反向要转置的是 dS / P（[BH, T, T_total]，8.4M 元素），
/// 串行版本单个转置就要几百毫秒，会成为反向的新瓶颈。
fn transpose_flat(v: &[f32], rows: usize, cols: usize, batch: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; batch * rows * cols];
    t.par_chunks_mut(rows * cols).enumerate().for_each(|(b, chunk)| {
        let src = &v[b * rows * cols..(b + 1) * rows * cols];
        for i in 0..rows {
            let srow = &src[i * cols..(i + 1) * cols];
            for (j, &s) in srow.iter().enumerate() {
                chunk[j * rows + i] = s;
            }
        }
    });
    t
}

/// 掩码 softmax 的 CPU 参考实现（GPU 不可用或数组太小时回退用）。
/// mask 必须是输入的右后缀：行 r 的掩码偏移 mb = (r*d) % mask_numel。
/// 并行：每行独立 softmax，行间无依赖。
fn masked_softmax_cpu(x: &[f32], mask: &[f32], rows: usize, d: usize, m_n: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * d];
    out.par_chunks_mut(d).enumerate().for_each(|(r, chunk)| {
        let base = r * d;
        let mb = base % m_n;
        let mut maxv = f32::NEG_INFINITY;
        for j in 0..d {
            maxv = maxv.max(x[base + j] + mask[mb + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..d {
            let e = (x[base + j] + mask[mb + j] - maxv).exp();
            chunk[j] = e;
            sum += e;
        }
        for j in 0..d {
            chunk[j] /= sum;
        }
    });
    out
}

/// flash attention 前向的中间状态：决定反向走「常驻显存」路径还是逐算子路径。
enum AttnCache {
    /// 逐算子/纯 CPU 路径产出的注意力概率 P（留在 CPU）
    Cpu(Rc<Vec<f32>>),
    /// 常驻显存路径：S 与 P 从未离开显存，反向也只回读 dQ/dK/dV
    #[cfg(feature = "gpu")]
    Gpu(Box<crate::gpu::AttnResident>),
}

/// 注意力前向的逐算子路径：S = Q'·Kᵀ → P = softmax(S + mask) → O = P·V。
/// 每个算子各自做 GPU/CPU 分流（`matmul_data` / `gpu::softmax_mask`），返回 (O, P)。
/// 常驻显存路径不可用时由 `flash_attention` 调用（也是 `LLM_GPU_PROBE` 录制形状时走的路径）。
#[allow(clippy::too_many_arguments)]
fn attn_forward_ops(
    q_scaled: &[f32],
    k: &[f32],
    v: &[f32],
    mask: &[f32],
    bh: usize,
    t: usize,
    t_total: usize,
    head_dim: usize,
) -> (Vec<f32>, Rc<Vec<f32>>) {
    let scores = matmul_data(q_scaled, k, t, head_dim, t_total, bh, false, true);
    let rows = bh * t;
    let m_n = mask.len();
    #[cfg(feature = "gpu")]
    let attn = crate::gpu::softmax_mask(&scores, mask, rows, t_total, m_n)
        .unwrap_or_else(|| masked_softmax_cpu(&scores, mask, rows, t_total, m_n));
    #[cfg(not(feature = "gpu"))]
    let attn = masked_softmax_cpu(&scores, mask, rows, t_total, m_n);
    let out = matmul_data(&attn, v, t, t_total, head_dim, bh, false, false);
    (out, Rc::new(attn))
}

/// 构造 matmul 的反向闭包（2D 与 3D 批量共用，batch=1 时 bi 循环退化）。
/// 反向公式（对 a）：∂a = g @ bᵀ；对 b）：∂b = aᵀ @ g。
/// 当 a、b 是同一个张量（如 x@x）时，梯度累加到同一块缓冲。
///
/// 梯度矩阵乘同样优先走 GPU（`matmul_data`，与 forward 一致），
/// 否则 backward 的三重循环在 CPU 上会主导训练时间（实测大模型每步 >2 分钟）。
fn matmul_backward(
    result: &mut Tensor,
    a: &Tensor,
    b: &Tensor,
    m: usize,
    k: usize,
    n: usize,
    batch: usize,
) {
    let rg = result.grad.clone();
    let sg = a.grad.clone();
    let og = b.grad.clone();
    let sd = a.data.clone();
    let od = b.data.clone();
    result.parents = Rc::new(vec![a.clone(), b.clone()]);
    result.backward = Some(Rc::new(move || {
        let g = rg.borrow();
        let sd_b = sd.borrow();
        let od_b = od.borrow();
        // ∂a = g @ bᵀ、∂b = aᵀ @ g：GPU 内核支持按转置读物理矩阵，
        // 无需在 CPU 构造 52 万~210 万元素的转置矩阵（仅 CPU 回退时才物化）
        let da = matmul_data(&g, &od_b, m, n, k, batch, false, true);
        let db = matmul_data(&sd_b, &g, k, m, n, batch, true, false);
        drop(g);
        drop(sd_b);
        drop(od_b);
        if Rc::ptr_eq(&sg, &og) {
            let mut sgm = sg.borrow_mut();
            for i in 0..batch * m * k {
                sgm[i] += da[i];
            }
            for j in 0..batch * k * n {
                sgm[j] += db[j];
            }
        } else {
            let mut sgm = sg.borrow_mut();
            let mut ogm = og.borrow_mut();
            for i in 0..batch * m * k {
                sgm[i] += da[i];
            }
            for j in 0..batch * k * n {
                ogm[j] += db[j];
            }
        }
    }));
}

// ==================== Tensor ====================

impl Tensor {
    // ---------- 构造 ----------
    pub(crate) fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        let len = data.len();
        // no_grad 模式下强制关闭求导
        let requires_grad = requires_grad && grad_enabled();
        Tensor {
            data: Rc::new(RefCell::new(data)),
            shape,
            // 注意：grad 缓冲必须始终按 data 全长分配。反向闭包可能写入
            // **不需要梯度**的父节点（如 masked_softmax 的 mask：它不参与求导，
            // 但闭包仍会往它的 grad 里累加），缓冲长度不足会直接越界 panic。
            grad: Rc::new(RefCell::new(vec![0.0; len])),
            requires_grad,
            parents: Rc::new(Vec::new()),
            backward: None,
        }
    }

    /// 该张量在当前模式下是否需要自动微分。
    ///
    /// 与直接读字段 `requires_grad` 的区别：no_grad 模式下恒为 false。
    /// 算子在决定"要不要挂 backward 闭包"时必须用它，否则推理时
    /// 仍会拿着参数张量的 `requires_grad = true` 一路建出整张计算图。
    #[inline]
    pub(crate) fn req(&self) -> bool {
        self.requires_grad && grad_enabled()
    }

    /// 用数据 + 形状构造叶子张量（不追踪梯度，例如输入数据）
    pub fn from_vec(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "数据长度 {} 与形状 {:?} 要求的元素数 {} 不一致",
            data.len(),
            shape,
            numel
        );
        Tensor::new(data, shape, false)
    }

    /// 构造参数张量（requires_grad = true）
    pub fn param(data: Vec<f32>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        assert_eq!(
            data.len(),
            numel,
            "参数数据长度 {} 与形状 {:?} 不一致",
            data.len(),
            shape
        );
        Tensor::new(data, shape, true)
    }

    // ---------- 访问器 ----------

    /// 返回数据副本（兼容旧调用点，热路径建议用 `data_ref`）
    pub fn data(&self) -> Vec<f32> {
        self.data.borrow().clone()
    }

    /// 只读借用底层数据，避免 O(N) 克隆。
    /// 适用于只读遍历（checkpoint 写入、采样取最后一行等）。
    pub fn data_ref(&self) -> std::cell::Ref<'_, Vec<f32>> {
        self.data.borrow()
    }

    /// 读取标量值（0 维张量专用，避免克隆整个 Vec）
    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() 只适用于单元素张量");
        self.data.borrow()[0]
    }

    pub fn set_data(&self, new_data: Vec<f32>) {
        let mut d = self.data.borrow_mut();
        assert_eq!(d.len(), new_data.len(), "set_data 长度不一致");
        *d = new_data;
    }

    /// 读取梯度副本（测试/调试用；训练代码用 `p.grad.borrow()` 原位访问避免拷贝）
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn grad(&self) -> Vec<f32> {
        self.grad.borrow().clone()
    }

    /// 把「外部算好的梯度」乘 `scale` 后累加进本张量。
    ///
    /// 供 GPU 常驻显存路径使用：那些路径在显存里连续算完一整段前向/反向，
    /// 只把边界上的梯度回读出来，由这里注回计算图继续传播。
    pub fn accumulate_grad(&self, g: &[f32], scale: f32) {
        let mut slot = self.grad.borrow_mut();
        assert_eq!(
            slot.len(),
            g.len(),
            "注入梯度长度不匹配：本张量 {} 个元素，注入 {} 个",
            slot.len(),
            g.len()
        );
        for (a, b) in slot.iter_mut().zip(g) {
            *a += b * scale;
        }
    }

    /// 构造一个标量 loss 张量，其反向由外部提供（`backward`），
    /// 之后由 autograd 继续沿 `parents` 传播。
    ///
    /// 用于「反向公式解析已知、不必建图」的融合算子：例如输出头的交叉熵，
    /// 其 dlogits 就是 softmax - onehot，直接在显存里算完即可，
    /// 没必要为了回传梯度而把 33.6M 元素的 logits 拉回 CPU 建一张计算图。
    pub fn external_scalar_loss(
        value: f32,
        parents: Vec<Tensor>,
        backward: impl Fn() + 'static,
    ) -> Tensor {
        Tensor {
            data: Rc::new(RefCell::new(vec![value])),
            shape: vec![],
            grad: Rc::new(RefCell::new(vec![0.0])),
            requires_grad: true,
            parents: Rc::new(parents),
            backward: Some(Rc::new(backward)),
        }
    }

    /// 构造一个「前向数据已由外部算好、反向也由外部提供」的张量。
    ///
    /// 供 GPU 常驻显存路径使用：那些路径在显存里连续算完一整段前向/反向，
    /// 只把边界梯度回读出来，再由 `backward` 里调用 [`Tensor::accumulate_grad`]
    /// 注回计算图（包括那些不在 `parents` 里的参数张量）。
    ///
    /// `backward` 是一个**工厂**而不是闭包本身：它拿到本张量自己的梯度槽，
    /// 返回的闭包在 autograd 反向时被调用 —— 那一刻槽里已经攒好了上游梯度。
    /// 之所以绕这一道，是因为闭包需要读自己的 grad，而梯度槽要等张量构造出来才存在。
    ///
    /// 调用点只在 `model.rs` 的 gpu 常驻路径里，不带 gpu feature 时是"死代码"，别删。
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn external(
        data: Vec<f32>,
        shape: Vec<usize>,
        parents: Vec<Tensor>,
        backward: impl FnOnce(Rc<RefCell<Vec<f32>>>) -> Box<dyn Fn()>,
    ) -> Tensor {
        let grad = Rc::new(RefCell::new(vec![0.0f32; data.len()]));
        let f = backward(grad.clone());
        Tensor {
            data: Rc::new(RefCell::new(data)),
            shape,
            grad,
            requires_grad: true,
            parents: Rc::new(parents),
            backward: Some(Rc::from(f)),
        }
    }

    pub fn zero_grad(&self) {
        let mut g = self.grad.borrow_mut();
        for v in g.iter_mut() {
            *v = 0.0;
        }
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    // ---------- 形状工具 ----------

    /// reshape：不改变元素顺序，梯度按 1:1 传回。
    /// 共享底层数据 Rc（不克隆 Vec），只分配新的梯度缓冲。
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            numel,
            "无法把 {:?} reshape 成 {:?}，元素总数不一致",
            self.shape,
            new_shape
        );
        let requires_grad = self.req();
        let mut result = Tensor {
            data: self.data.clone(), // Rc 共享，不克隆 Vec
            shape: new_shape,
            grad: Rc::new(RefCell::new(vec![0.0; numel])),
            requires_grad,
            parents: Rc::new(Vec::new()),
            backward: None,
        };
        if requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let g_ref: &[f32] = &g;
                let mut sgm = sg.borrow_mut();
                // reshape 只改形状、不改元素顺序，所以梯度是 1:1 的逐元素累加。
                // 按 4096 分块并行：输出头的 logits 是 [B*T, vocab] = 3355 万元素，
                // 串行版单步要 ~0.4s（占单步 ~9%），是矩阵乘之外最大的纯访存热点。
                sgm.par_chunks_mut(4096)
                    .zip(g_ref.par_chunks(4096))
                    .for_each(|(s, gg)| {
                        for (a, &b) in s.iter_mut().zip(gg) {
                            *a += b;
                        }
                    });
            }));
        }
        result
    }

    /// 任意维度重排（如 [0,2,1] 把 2、3 维交换）。
    /// 反向：梯度按逆重排传回。
    pub fn permute(&self, dims: &[usize]) -> Tensor {
        assert_eq!(dims.len(), self.rank(), "permute 必须提供所有维度");
        let mut seen = vec![false; self.rank()];
        for &d in dims {
            assert!(
                d < self.rank() && !seen[d],
                "permute 维度必须是不重复的排列"
            );
            seen[d] = true;
        }
        let new_shape: Vec<usize> = dims.iter().map(|&d| self.shape[d]).collect();
        let total = self.numel();

        // 反解 permute 的逆映射：inv[perm[i]] = i
        let mut inv = vec![0usize; self.rank()];
        for (i, &d) in dims.iter().enumerate() {
            inv[d] = i;
        }
        // 前向：out_flat -> src_flat
        // 热路径：训练中每步都要对 Q/K/V 拆头、合头做多次 permute，
        // 原实现每次重建几十万元素的下标表；训练形状固定，缓存一次、Rc 共享即可。
        // （张量最多 4 维，固定栈数组即可，超过 8 维防御性断言）
        assert!(self.rank() <= 8, "permute 维度过多：{}", self.rank());
        let map: Rc<Vec<usize>> = PERMUTE_MAP_CACHE.with(|c| {
            let mut cache = c.borrow_mut();
            cache
                .entry((self.shape.clone(), dims.to_vec()))
                .or_insert_with(|| {
                    let mut map = vec![0usize; total];
                    let mut out_idx = [0usize; 8];
                    for out_flat in 0..total {
                        let mut r = out_flat;
                        for d in (0..self.rank()).rev() {
                            out_idx[d] = r % new_shape[d];
                            r /= new_shape[d];
                        }
                        let mut src_flat = 0usize;
                        for d in 0..self.rank() {
                            let sd = out_idx[inv[d]]; // 源的第 d 维来自输出的第 inv[d] 维
                            src_flat = src_flat * self.shape[d] + sd;
                        }
                        map[out_flat] = src_flat;
                    }
                    Rc::new(map)
                })
                .clone()
        });

        let sd = self.data.borrow();
        let sd_ref: &[f32] = &sd;
        let map_ref: &[usize] = &map;
        let mut out_data = vec![0.0f32; total];
        // 并行：每块 4096 元素，permute 只是查表搬移，适合多核
        out_data
            .par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096;
                for (j, slot) in chunk.iter_mut().enumerate() {
                    *slot = sd_ref[map_ref[base + j]];
                }
            });
        drop(sd);

        let mut result = Tensor::new(out_data, new_shape, self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let map_bw = map.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for (of, &sf) in map_bw.iter().enumerate() {
                    sgm[sf] += g[of];
                }
            }));
        }
        result
    }

    /// 2 维转置（permute([1,0]) 的特例）
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.rank(), 2, "transpose 只支持 2 维");
        self.permute(&[1, 0])
    }

    // ---------- 逐元素运算（支持广播） ----------

    /// 判断 src（self）广播到 target 时能否用"取模快路径"：
    /// 要求 self 是 target 的右后缀，且所有维度都大于 1（没有 1 维插值）。
    /// 满足时返回 src 的元素总数 n，源下标 = t % n。
    fn suffix_mod(&self, target: &[usize]) -> Option<usize> {
        let off = target.len() - self.shape.len();
        for (i, &s) in self.shape.iter().enumerate() {
            if s == 1 || target[off + i] != s {
                return None;
            }
        }
        Some(self.numel())
    }

    /// 内部工具：判断是否同形状；不同则计算广播索引方式。
    ///
    /// 返回 (目标形状, 本张量源索引方式, 另一张量源索引方式)。
    /// 训练热路径里大量出现"偏置 [d] → [rows,d]"、"mask [t,tt] → [bh,t,tt]"这类
    /// 右后缀广播，直接用取模快路径，省掉每次前向重建 4-16MB 的广播 map。
    fn broadcast_plan(&self, other: &Tensor) -> (Vec<usize>, SrcIdx, SrcIdx) {
        if self.shape == other.shape {
            (self.shape.clone(), SrcIdx::Ident, SrcIdx::Ident)
        } else {
            let target = broadcast_shapes(&self.shape, &other.shape)
                .unwrap_or_else(|| panic!("形状无法广播：{:?} vs {:?}", self.shape, other.shape));
            let a_src = if self.shape == target {
                SrcIdx::Ident
            } else {
                match self.suffix_mod(&target) {
                    Some(n) => SrcIdx::Mod(n),
                    None => SrcIdx::Map(Rc::new(broadcast_map(&target, &self.shape))),
                }
            };
            let b_src = if other.shape == target {
                SrcIdx::Ident
            } else {
                match other.suffix_mod(&target) {
                    Some(n) => SrcIdx::Mod(n),
                    None => SrcIdx::Map(Rc::new(broadcast_map(&target, &other.shape))),
                }
            };
            (target, a_src, b_src)
        }
    }

    pub fn add(&self, other: &Tensor) -> Tensor {
        self.binary(other, |a, b| a + b, |_, _| (1.0, 1.0))
    }

    pub fn sub(&self, other: &Tensor) -> Tensor {
        self.binary(other, |a, b| a - b, |_, _| (1.0, -1.0))
    }

    /// 逐元素乘法。与 [`Tensor::div`] 一样属于"算子集完整性"的保留项：
    /// 训练路径走的是 `*` 运算符重载，这里的方法版没有调用点。
    #[allow(dead_code)]
    pub fn mul(&self, other: &Tensor) -> Tensor {
        // ∂c/∂a = b，∂c/∂b = a
        self.binary(other, |a, b| a * b, |a, b| (b, a))
    }

    /// 逐元素除法。保留项，理由同 [`Tensor::mul`]。
    #[allow(dead_code)]
    pub fn div(&self, other: &Tensor) -> Tensor {
        // 反向必须与前向 `a / b` 严格对应（∂/∂a = 1/b，∂/∂b = -a/b²）。
        // 给 b 加 `1e-8` 会让梯度与自己的前向不一致——教学代码里宁可得到 ±inf，
        // 也不要一个"看起来没事但数学上是错的"梯度。
        self.binary(other, |a, b| a / b, |a, b| (1.0 / b, -a / (b * b)))
    }

    /// 通用逐元素二元运算（含广播）+ 反向传播
    ///
    /// - `fwd(a, b) -> c`：前向计算
    /// - `back(a, b) -> (∂c/∂a, ∂c/∂b)`：返回两个输入的导数值
    ///   例：mul 的 back 返回 (b, a)；div 返回 (1/b, -a/b²)
    fn binary(
        &self,
        other: &Tensor,
        fwd: impl Fn(f32, f32) -> f32 + Sync + 'static,
        back: impl Fn(f32, f32) -> (f32, f32) + Sync + 'static,
    ) -> Tensor {
        let (target_shape, a_src, b_src) = self.broadcast_plan(other);
        let sa = self.data.borrow();
        let sb = other.data.borrow();
        let sa_ref: &[f32] = &sa;
        let sb_ref: &[f32] = &sb;
        let total: usize = target_shape.iter().product();
        let a_view: SrcIdxView<'_> = (&a_src).into();
        let b_view: SrcIdxView<'_> = (&b_src).into();
        let mut out_data = vec![0.0f32; total];
        out_data
            .par_chunks_mut(4096)
            .enumerate()
            .for_each(|(ci, chunk)| {
                let base = ci * 4096;
                for (j, slot) in chunk.iter_mut().enumerate() {
                    let t = base + j;
                    *slot = fwd(sa_ref[a_view.idx(t)], sb_ref[b_view.idx(t)]);
                }
            });
        drop(sa);
        drop(sb);

        let requires = self.req() || other.req();
        let mut result = Tensor::new(out_data, target_shape, requires);
        if requires {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let og = other.grad.clone();
            let sd = self.data.clone();
            let od = other.data.clone();
            // 广播索引方式随闭包带走（Mod 不占内存，Map 是 Rc 共享，无需克隆 4-16MB map）
            let a_src_c = a_src;
            let b_src_c = b_src;
            let same_shape = self.shape == other.shape;
            result.parents = Rc::new(vec![self.clone(), other.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.borrow();
                let od_b = od.borrow();
                let g_ref: &[f32] = &g;
                let sd_ref: &[f32] = &sd_b;
                let od_ref: &[f32] = &od_b;
                if same_shape {
                    // 形状相同 ⇒ 两侧索引都是恒等（见 broadcast_plan），可以按块并行。
                    // 残差相加、x*x 这类占反向的大头，串行时单步 ~0.5s。
                    if Rc::ptr_eq(&sg, &og) {
                        // 同一张量参与运算（如 x*x、x/x），两条路径梯度叠加
                        let mut sgm = sg.borrow_mut();
                        sgm.par_chunks_mut(4096).enumerate().for_each(|(ci, ch)| {
                            let base = ci * 4096;
                            for (j, s) in ch.iter_mut().enumerate() {
                                let t = base + j;
                                let (da, db) = back(sd_ref[t], od_ref[t]);
                                *s += g_ref[t] * (da + db);
                            }
                        });
                    } else {
                        let mut sgm = sg.borrow_mut();
                        let mut ogm = og.borrow_mut();
                        sgm.par_chunks_mut(4096)
                            .zip(ogm.par_chunks_mut(4096))
                            .enumerate()
                            .for_each(|(ci, (ca, cb))| {
                                let base = ci * 4096;
                                for (j, (sa, sb)) in ca.iter_mut().zip(cb.iter_mut()).enumerate() {
                                    let t = base + j;
                                    let (da, db) = back(sd_ref[t], od_ref[t]);
                                    *sa += g_ref[t] * da;
                                    *sb += g_ref[t] * db;
                                }
                            });
                    }
                } else {
                    let mut sgm = sg.borrow_mut();
                    let mut ogm = og.borrow_mut();
                    for t in 0..g.len() {
                        let ia = src_idx(t, &a_src_c);
                        let ib = src_idx(t, &b_src_c);
                        let (da, db) = back(sd_ref[ia], od_ref[ib]);
                        sgm[ia] += g[t] * da;
                        ogm[ib] += g[t] * db;
                    }
                }
            }));
        }
        result
    }

    // ---------- 标量运算 ----------

    #[allow(dead_code)]
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let data = self.data.borrow().iter().map(|a| a + scalar).collect();
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i];
                }
            }));
        }
        result
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        let data = self.data.borrow().iter().map(|a| a * scalar).collect();
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] * scalar;
                }
            }));
        }
        result
    }

    // ---------- 激活函数（一元运算） ----------

    /// 取负：c = -x，∂x = -g
    ///
    /// 保留项：属于"算子集完整性"（教学对照用），当前训练/推理路径没有调用点。
    #[allow(dead_code)]
    pub fn neg(&self) -> Tensor {
        let data = self.data.borrow().iter().map(|a| -a).collect();
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] -= g[i];
                }
            }));
        }
        result
    }

    /// ReLU：c = max(0, x)，∂x = g * (x>0)
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn relu(&self) -> Tensor {
        let sd = self.data.borrow();
        let data = sd.iter().map(|&a| a.max(0.0)).collect();
        let mask: Vec<f32> = sd
            .iter()
            .map(|&a| if a > 0.0 { 1.0 } else { 0.0 })
            .collect();
        drop(sd);
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] * mask[i];
                }
            }));
        }
        result
    }

    /// tanh：c = tanh(x)，∂x = g * (1 - c²)
    #[allow(dead_code)]
    pub fn tanh(&self) -> Tensor {
        let sd = self.data.borrow();
        let data: Vec<f32> = sd.iter().map(|&a| a.tanh()).collect();
        drop(sd);
        let mut result = Tensor::new(data.clone(), self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] * (1.0 - data[i] * data[i]);
                }
            }));
        }
        result
    }

    /// GELU（用 tanh 近似）：c = 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
    /// 这是 GPT 系列使用的激活函数。
    /// 反向：dGELU/dx = 0.5(1+t) + 0.5x(1-t²)·da/dx，其中 a = √(2/π)(x+0.044715x³)，t = tanh(a)
    pub fn gelu(&self) -> Tensor {
        const SQRT_2_PI: f32 = 0.797_884_560_8; // sqrt(2/π)
        const COEF: f32 = 0.044_715;
        let sd = self.data.borrow();
        let sd_ref: &[f32] = &sd;
        let len = sd_ref.len();
        let mut data = vec![0.0f32; len];
        let mut t_vals = vec![0.0f32; len];
        // 并行：gelu 是逐元素，每元素独立计算 tanh + 乘法，多核收益显著
        data.par_chunks_mut(4096)
            .zip(t_vals.par_chunks_mut(4096))
            .enumerate()
            .for_each(|(ci, (dc, tc))| {
                let base = ci * 4096;
                for (j, (do_, to_)) in dc.iter_mut().zip(tc.iter_mut()).enumerate() {
                    let x = sd_ref[base + j];
                    let a = SQRT_2_PI * (x + COEF * x * x * x);
                    let t = a.tanh();
                    *to_ = t;
                    *do_ = 0.5 * x * (1.0 + t);
                }
            });
        drop(sd);
        let mut result = Tensor::new(data.clone(), self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let sd = self.data.clone();
            let tv = t_vals;
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let x_b = sd.borrow();
                let mut sgm = sg.borrow_mut();
                // 提取为 Vec（Send），供 rayon 闭包安全使用
                let g_vec: Vec<f32> = g.iter().copied().collect();
                let x_vec: Vec<f32> = x_b.iter().copied().collect();
                let len = g_vec.len();
                let mut dx = vec![0.0f32; len];
                dx.par_chunks_mut(4096)
                    .enumerate()
                    .for_each(|(ci, chunk)| {
                        let base = ci * 4096;
                        for (j, slot) in chunk.iter_mut().enumerate() {
                            let i = base + j;
                            if i >= len { break; }
                            let x = x_vec[i];
                            let t = tv[i];
                            let da_dx = SQRT_2_PI * (1.0 + 3.0 * COEF * x * x);
                            let dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * da_dx;
                            *slot = g_vec[i] * dy_dx;
                        }
                    });
                for (i, v) in dx.iter().enumerate() {
                    sgm[i] += v;
                }
            }));
        }
        result
    }

    /// SwiGLU 激活函数（融合实现）：
    /// SwiGLU(x) = SiLU(xW₁) ⊙ (xW₃)
    ///
    /// SiLU(x) = x · sigmoid(x)，比 GELU 更平滑，LLaMA / Mistral 标配。
    /// 门控机制：(xW₃) 控制哪些信息通过，比纯 GELU 表达力更强。
    ///
    /// 注意：这个方法只实现 SiLU ⊙ gate 的逐元素融合，线性投影由调用方（SwiGLU MLP 层）完成。
    ///
    /// - x: 已过线性层的激活输入 [B*T, hidden_dim]
    /// - gate: 已过线性层的门控值 [B*T, hidden_dim]
    /// - 返回: [B*T, hidden_dim]
    ///
    /// 反向：
    /// ```text
    /// d_silu = d_out · gate         (对 SiLU 分支)
    /// d_gate = d_out · silu(x)      (对门控分支)
    /// d_x = d_silu · (sigmoid(x) + x · sigmoid(x) · (1 - sigmoid(x)))
    ///     = d_silu · sigmoid(x) · (1 + x · (1 - sigmoid(x)))
    /// ```
    pub fn swiglu(&self, gate: &Tensor) -> Tensor {
        assert_eq!(
            self.shape, gate.shape,
            "SwiGLU 的两个输入形状必须一致"
        );
        let sd = self.data.borrow();
        let gd = gate.data.borrow();
        let len = sd.len();
        let mut out_data = vec![0.0f32; len];
        let mut silu_vals = vec![0.0f32; len]; // SiLU(x) = x * sigmoid(x)
        let mut sig_vals = vec![0.0f32; len]; // sigmoid(x)
        let sd_ref: &[f32] = &sd;
        let gd_ref: &[f32] = &gd;
        // 并行：逐元素融合
        out_data
            .par_chunks_mut(4096)
            .zip(silu_vals.par_chunks_mut(4096))
            .zip(sig_vals.par_chunks_mut(4096))
            .enumerate()
            .for_each(|(ci, ((oc, sc), sg))| {
                let base = ci * 4096;
                for (j, ((o, s), sig)) in oc.iter_mut().zip(sc.iter_mut()).zip(sg.iter_mut()).enumerate() {
                    let idx = base + j;
                    if idx >= len { break; }
                    let x = sd_ref[idx];
                    let sig_v = 1.0 / (1.0 + (-x).exp());
                    let silu_v = x * sig_v;
                    *sig = sig_v;
                    *s = silu_v;
                    *o = silu_v * gd_ref[idx];
                }
            });
        drop(sd);
        drop(gd);

        let requires = self.req() || gate.req();
        let mut result = Tensor::new(out_data, self.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sx = self.grad.clone();
            let sg = gate.grad.clone();
            let xd = self.data.clone();
            let gd = gate.data.clone();
            let sv = silu_vals;
            let sig = sig_vals;
            result.parents = Rc::new(vec![self.clone(), gate.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let x_b = xd.borrow();
                let g_b = gd.borrow();
                let mut gx = sx.borrow_mut();
                let mut gg = sg.borrow_mut();
                let len = g.len();
                // gx 和 gg 写不同数组，可以安全并行
                // 但 RefCell 限制同时可变借用。用索引分块处理。
                let chunk = 4096;
                for start in (0..len).step_by(chunk) {
                    let end = (start + chunk).min(len);
                    for i in start..end {
                        let sig_v = sig[i];
                        let silu_v = sv[i];
                        gg[i] += g[i] * silu_v;
                        let dsig = sig_v * (1.0 + x_b[i] * (1.0 - sig_v));
                        gx[i] += g[i] * g_b[i] * dsig;
                    }
                }
            }));
        }
        result
    }

    /// log：c = ln(x)，∂x = g / x（cross_entropy 已改用 log_softmax_last_dim，仅测试使用）
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn log(&self) -> Tensor {
        let sd = self.data.borrow();
        let data: Vec<f32> = sd.iter().map(|&a| a.ln()).collect();
        drop(sd);
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let sd = self.data.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] / (sd_b[i] + EPS);
                }
            }));
        }
        result
    }

    /// 幂：c = x^p，∂x = g * p * x^(p-1)
    pub fn pow(&self, p: f32) -> Tensor {
        let sd = self.data.borrow();
        let data: Vec<f32> = sd.iter().map(|&a| a.powf(p)).collect();
        drop(sd);
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let sd = self.data.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] * p * sd_b[i].powf(p - 1.0);
                }
            }));
        }
        result
    }

    /// sqrt：c = sqrt(x)，∂x = g / (2c)
    #[allow(dead_code)]
    pub fn sqrt(&self) -> Tensor {
        self.pow(0.5)
    }

    // ---------- 矩阵运算 ----------

    /// 矩阵乘法，支持：
    /// - 2D：C[m,n] = A[m,k] @ B[k,n]
    /// - 3D 批量：C[B,m,n] = A[B,m,k] @ B[B,k,n]
    ///
    /// 反向公式（2D）：∂A = g @ Bᵀ，∂B = Aᵀ @ g
    pub fn matmul(&self, other: &Tensor) -> Tensor {
        assert!(
            (self.rank() == 2 && other.rank() == 2) || (self.rank() == 3 && other.rank() == 3),
            "matmul 只支持 2D 或 3D（批量），当前 {}-D x {}-D",
            self.rank(),
            other.rank()
        );
        if self.rank() == 2 {
            return self.matmul_2d(other);
        }
        // 3D 批量
        assert_eq!(self.shape[0], other.shape[0], "批量维度必须一致");
        let (b, m, k1) = (self.shape[0], self.shape[1], self.shape[2]);
        let (_, k2, n) = (other.shape[0], other.shape[1], other.shape[2]);
        assert_eq!(k1, k2, "矩阵乘法维度不匹配");

        let sd = self.data.borrow();
        let od = other.data.borrow();
        let out_data = matmul_data(&sd, &od, m, k1, n, b, false, false);
        drop(sd);
        drop(od);

        let requires = self.req() || other.req();
        let mut result = Tensor::new(out_data, vec![b, m, n], requires);
        if requires {
            matmul_backward(&mut result, self, other, m, k1, n, b);
        }
        result
    }

    fn matmul_2d(&self, other: &Tensor) -> Tensor {
        let (m, k1) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        assert_eq!(
            k1, k2,
            "矩阵乘法维度不匹配：{:?} x {:?}",
            self.shape, other.shape
        );

        let sd = self.data.borrow();
        let od = other.data.borrow();
        let out_data = matmul_data(&sd, &od, m, k1, n, 1, false, false);
        drop(sd);
        drop(od);

        let requires = self.req() || other.req();
        let mut result = Tensor::new(out_data, vec![m, n], requires);
        if requires {
            matmul_backward(&mut result, self, other, m, k1, n, 1);
        }
        result
    }

    // ---------- 归约运算 ----------

    /// 求和成标量，梯度均匀传给每个元素
    pub fn sum(&self) -> Tensor {
        let total = self.data.borrow().iter().sum();
        let mut result = Tensor::new(vec![total], vec![], self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow()[0];
                let mut sgm = sg.borrow_mut();
                for v in sgm.iter_mut() {
                    *v += g;
                }
            }));
        }
        result
    }

    /// 沿最后一维求和，**保持维度**：[..., D] -> [..., 1]
    /// 反向：梯度广播回最后一维
    ///
    /// 保留项：属于"算子集完整性"（教学对照用），当前没有调用点
    /// （需要求和的路径各自用了更专门的融合算子）。
    #[allow(dead_code)]
    pub fn sum_last_dim(&self) -> Tensor {
        assert!(self.rank() >= 1, "sum_last_dim 需要至少 1 维");
        let (pre, d) = (
            self.numel() / self.shape[self.rank() - 1],
            self.shape[self.rank() - 1],
        );
        let sd = self.data.borrow();
        let mut out_data = vec![0.0f32; pre];
        for p in 0..pre {
            let mut s = 0.0;
            for j in 0..d {
                s += sd[p * d + j];
            }
            out_data[p] = s;
        }
        drop(sd);
        let mut new_shape = self.shape.clone();
        *new_shape.last_mut().unwrap() = 1;

        let mut result = Tensor::new(out_data, new_shape, self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for p in 0..pre {
                    for j in 0..d {
                        sgm[p * d + j] += g[p];
                    }
                }
            }));
        }
        result
    }

    /// 沿最后一维做 softmax：[..., D] 每行独立归一化。
    ///
    /// 数值稳定技巧：先减去每行最大值再 exp（防止指数爆炸）。
    /// 反向公式：∂x_i = s_i * (g_i - Σ_j g_j * s_j)
    #[allow(dead_code)]
    pub fn softmax_last_dim(&self) -> Tensor {
        assert!(self.rank() >= 1, "softmax_last_dim 需要至少 1 维");
        let (rows, d) = (
            self.numel() / self.shape[self.rank() - 1],
            self.shape[self.rank() - 1],
        );
        let sd = self.data.borrow();
        let mut out_data = vec![0.0f32; rows * d];
        // 先存 softmax 结果（反向需要）
        for r in 0..rows {
            let mut maxv = f32::NEG_INFINITY;
            for j in 0..d {
                maxv = maxv.max(sd[r * d + j]);
            }
            let mut sum = 0.0;
            for j in 0..d {
                out_data[r * d + j] = (sd[r * d + j] - maxv).exp();
                sum += out_data[r * d + j];
            }
            for j in 0..d {
                out_data[r * d + j] /= sum;
            }
        }
        drop(sd);

        let mut result = Tensor::new(out_data.clone(), self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let (_rows, d) = (rows, d);
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for r in 0.._rows {
                    let mut dot = 0.0;
                    for j in 0..d {
                        dot += g[r * d + j] * out_data[r * d + j];
                    }
                    for i in 0..d {
                        sgm[r * d + i] += out_data[r * d + i] * (g[r * d + i] - dot);
                    }
                }
            }));
        }
        result
    }

    /// 层归一化（融合实现）：y = (x - μ)/√(σ²+ε) * γ + β，按最后一维归一化。
    ///
    /// 一个算子完成 `sum_last_dim → mul_scalar → sub → mul → sum_last_dim → mul_scalar
    /// → add_scalar → sqrt → div → mul → add` 11 个基础算子的工作（前向 + 反向各一遍循环），
    /// 训练热路径里每个 Transformer block 有 2 个 LayerNorm，原来拼接方式既慢又多建中间张量。
    ///
    /// 反向用经典融合公式（避免存中间量，只需存 mean / inv_std）：
    /// ```text
    /// d_norm = d_y · γ
    /// m1 = mean(d_norm)，m2 = mean(d_norm · norm)
    /// d_x = inv_std · (d_norm - m1 - m2 · norm)
    /// d_γ_j = Σ_r d_y[r,j] · norm[r,j]，d_β_j = Σ_r d_y[r,j]
    /// ```
    pub fn layernorm(&self, gamma: &Tensor, beta: &Tensor, eps: f32) -> Tensor {
        let d = *self.shape.last().unwrap();
        assert_eq!(gamma.rank(), 1, "LayerNorm 的 γ 必须是一维");
        assert_eq!(gamma.shape, beta.shape, "LayerNorm 的 γ/β 形状必须一致");
        assert_eq!(gamma.shape[0], d, "LayerNorm 的 γ/β 长度必须等于输入最后一维");
        let rows = self.numel() / d;
        let sd = self.data.borrow();
        let gv = gamma.data.borrow();
        let bv = beta.data.borrow();
        let sd_ref: &[f32] = &sd;
        let gv_ref: &[f32] = &gv;
        let bv_ref: &[f32] = &bv;
        let mut out = vec![0.0f32; rows * d];
        let mut mean = vec![0.0f32; rows];
        let mut inv_std = vec![0.0f32; rows];
        // 并行：每行独立计算均值/方差/归一化，行间无依赖
        out.par_chunks_mut(d)
            .zip(mean.par_iter_mut())
            .zip(inv_std.par_iter_mut())
            .enumerate()
            .for_each(|(r, ((out_row, m), is))| {
                let base = r * d;
                let mut acc = 0.0f32;
                for j in 0..d {
                    acc += sd_ref[base + j];
                }
                acc /= d as f32;
                let mut v = 0.0f32;
                for j in 0..d {
                    let c = sd_ref[base + j] - acc;
                    v += c * c;
                }
                v /= d as f32;
                let inv = 1.0 / (v + eps).sqrt();
                *m = acc;
                *is = inv;
                for j in 0..d {
                    out_row[j] = (sd_ref[base + j] - acc) * inv * gv_ref[j] + bv_ref[j];
                }
            });
        drop(sd);
        drop(gv);
        drop(bv);

        let requires = self.req() || gamma.req() || beta.req();
        let mut result = Tensor::new(out, self.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sx = self.grad.clone();
            let sg = gamma.grad.clone();
            let sb = beta.grad.clone();
            let xd = self.data.clone();
            let gd = gamma.data.clone();
            result.parents = Rc::new(vec![self.clone(), gamma.clone(), beta.clone()]);
            result.backward = Some(Rc::new(move || {
                let g: Vec<f32> = rg.borrow().to_vec();
                let x_b: Vec<f32> = xd.borrow().to_vec();
                let gam: Vec<f32> = gd.borrow().to_vec();
                let mut gx = sx.borrow_mut();
                let mut gg = sg.borrow_mut();
                let mut gb = sb.borrow_mut();
                // 第一遍：顺序计算 dg/db（跨行累加，无法并行）
                let mut dg = vec![0.0f32; d];
                let mut db = vec![0.0f32; d];
                for r in 0..rows {
                    let base = r * d;
                    let is = inv_std[r];
                    for j in 0..d {
                        let dy = g[base + j];
                        dg[j] += dy * (x_b[base + j] - mean[r]) * is;
                        db[j] += dy;
                    }
                }
                for j in 0..d { gg[j] += dg[j]; gb[j] += db[j]; }
                // 第二遍：并行计算 gx（行间无写冲突）
                let chunk_size = 1024;
                gx.par_chunks_mut(d * chunk_size)
                    .enumerate()
                    .for_each(|(ci, gx_chunk)| {
                        let start_row = ci * chunk_size;
                        let chunk_rows = gx_chunk.len() / d;
                        for ri in 0..chunk_rows {
                            let r = start_row + ri;
                            let base = r * d;
                            let is = inv_std[r];
                            let mut m1 = 0.0f32;
                            let mut m2 = 0.0f32;
                            for j in 0..d {
                                let dy_g = g[base + j] * gam[j];
                                m1 += dy_g;
                                m2 += dy_g * (x_b[base + j] - mean[r]) * is;
                            }
                            m1 /= d as f32;
                            m2 /= d as f32;
                            for j in 0..d {
                                let norm = (x_b[base + j] - mean[r]) * is;
                                gx_chunk[ri * d + j] += is * (g[base + j] * gam[j] - m1 - m2 * norm);
                            }
                        }
                    });
            }));
        }
        result
    }

    /// RMSNorm（Root Mean Square Layer Normalization）：
    /// y = x / √(mean(x²) + ε) * γ
    ///
    /// 比 LayerNorm 更简单高效：
    /// - 不减均值（省一次 reduction）
    /// - 没有 β 偏置（省一个参数和一次加法）
    /// - LLaMA / Mistral / Qwen 等现代 LLM 全部使用
    ///
    /// 反向公式：
    /// ```text
    /// d_y_γ = d_y · γ
    /// Σxg = Σ_j(x_j · d_y_γ_j)    // 每行一个标量
    /// d_x_i = is · (d_y_γ_i - (Σxg / d) · is² · x_i)
    /// d_γ_j = Σ_r d_y[r,j] · (x[r,j] · is_r)
    /// ```
    pub fn rmsnorm(&self, gamma: &Tensor, eps: f32) -> Tensor {
        let d = *self.shape.last().unwrap();
        assert_eq!(gamma.rank(), 1, "RMSNorm 的 γ 必须是一维");
        assert_eq!(gamma.shape[0], d, "RMSNorm 的 γ 长度必须等于输入最后一维");
        let rows = self.numel() / d;
        let sd = self.data.borrow();
        let gv = gamma.data.borrow();
        let sd_ref: &[f32] = &sd;
        let gv_ref: &[f32] = &gv;
        let mut out = vec![0.0f32; rows * d];
        let mut inv_rms = vec![0.0f32; rows]; // 1/rms，反向需要
        // 并行：每行独立计算 rms / 归一化
        out.par_chunks_mut(d)
            .zip(inv_rms.par_iter_mut())
            .enumerate()
            .for_each(|(r, (out_row, ir))| {
                let base = r * d;
                let mut ms = 0.0f32;
                for j in 0..d {
                    ms += sd_ref[base + j] * sd_ref[base + j];
                }
                ms /= d as f32;
                let inv = 1.0 / (ms + eps).sqrt();
                *ir = inv;
                for j in 0..d {
                    out_row[j] = sd_ref[base + j] * inv * gv_ref[j];
                }
            });
        drop(sd);
        drop(gv);

        let requires = self.req() || gamma.req();
        let mut result = Tensor::new(out, self.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sx = self.grad.clone();
            let sg = gamma.grad.clone();
            let xd = self.data.clone();
            let gd = gamma.data.clone();
            let ir = inv_rms;
            result.parents = Rc::new(vec![self.clone(), gamma.clone()]);
            result.backward = Some(Rc::new(move || {
                let g: Vec<f32> = rg.borrow().to_vec();
                let x_b: Vec<f32> = xd.borrow().to_vec();
                let gam: Vec<f32> = gd.borrow().to_vec();
                let mut gx = sx.borrow_mut();
                let mut gg = sg.borrow_mut();
                // 第一遍：顺序计算 dg
                let mut dg = vec![0.0f32; d];
                for r in 0..rows {
                    let base = r * d;
                    let is = ir[r];
                    for j in 0..d {
                        dg[j] += g[base + j] * x_b[base + j] * is;
                    }
                }
                for j in 0..d { gg[j] += dg[j]; }
                // 第二遍：并行计算 gx
                let chunk_size = 1024;
                gx.par_chunks_mut(d * chunk_size)
                    .enumerate()
                    .for_each(|(ci, gx_chunk)| {
                        let start_row = ci * chunk_size;
                        let chunk_rows = gx_chunk.len() / d;
                        for ri in 0..chunk_rows {
                            let r = start_row + ri;
                            let base = r * d;
                            let is = ir[r];
                            let mut sxg = 0.0f32;
                            for j in 0..d {
                                let dy_g = g[base + j] * gam[j];
                                sxg += x_b[base + j] * dy_g;
                            }
                            sxg /= d as f32;
                            let is2 = is * is;
                            for j in 0..d {
                                let dy_g = g[base + j] * gam[j];
                                gx_chunk[ri * d + j] += is * (dy_g - sxg * is2 * x_b[base + j]);
                            }
                        }
                    });
            }));
        }
        result
    }

    /// 注意力（缩放点积 + 因果掩码）：O = softmax(Q·Kᵀ/√d + M) · V
    ///
    /// **2026-09-16 重写（性能）**：原实现是"Flash Attention 分块 + 在线 softmax"的
    /// 标量三重循环版本。逐算子插桩显示它占单步 78% 的时间（flash_f 4.6s + flash_b 8.7s
    /// / 17.1s），有效算力只有 0.12~0.24 GFLOP/s，比 `matmul_data` 慢约 100 倍——
    /// 瓶颈既不是访存也不是算法，而是没走上已经分块/向量化/可走 GPU 的矩阵乘内核。
    ///
    /// 现流程（数学上与标准 attention 完全一致）：
    /// ```text
    /// Q' = Q / √d                       // 缩放挪到 Q 上，只要 524K 次乘法
    /// S  = Q'·Kᵀ                        // matmul_data [BH,T,T_total]，一次算完
    /// P  = softmax(S + M)               // 融合内核，每行一遍过
    /// O  = P·V                          // matmul_data
    /// ```
    /// 反向同样全用矩阵乘：dV = Pᵀ·dO、dP = dO·Vᵀ、dS = P⊙(dP - ΣdP·P)、
    /// dQ = dS·K·scale、dK = dSᵀ·Q'。
    ///
    /// **代价**：P 与 dP 需要 O(T²) 的显存（原实现存 P 也已是 O(T²)），
    /// 换来的是 6 次大矩阵乘走内核，单步注意力从 ~13.3s 降到亚秒级。
    ///
    /// - q: [B*H, T, head_dim]
    /// - k: [B*H, T_total, head_dim]
    /// - v: [B*H, T_total, head_dim]
    /// - mask: [T, T_total]（因果掩码，-inf 的位置屏蔽）
    /// - _block_size: 兼容旧接口保留（现在的分块由 matmul_data 内部负责）
    ///
    /// 返回 out: [B*H, T, head_dim]
    pub fn flash_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: &Tensor,
        _block_size: usize,
    ) -> Tensor {
        assert_eq!(q.rank(), 3, "flash_attention: Q 必须为 3D");
        assert_eq!(k.rank(), 3, "flash_attention: K 必须为 3D");
        assert_eq!(v.rank(), 3, "flash_attention: V 必须为 3D");
        let (bh, t, head_dim) = (q.shape[0], q.shape[1], q.shape[2]);
        let t_total = k.shape[1];
        assert_eq!(k.shape, v.shape, "K 和 V 形状必须一致");
        assert_eq!(k.shape[2], head_dim, "K/V 的 head_dim 必须与 Q 一致");
        assert_eq!(k.shape[0], bh, "K/V 的 batch*head 必须与 Q 一致");

        let scale = 1.0 / (head_dim as f32).sqrt();
        let md = mask.data.borrow();
        let mask_data: &[f32] = &md;

        // 读取输入数据
        let qd = q.data.borrow();
        let kd = k.data.borrow();
        let vd = v.data.borrow();

        // 1) Q' = Q / √d：把缩放挪到 Q 上（只 524K 次乘法），
        //    这样 S = Q'·Kᵀ 就是最终打分，softmax 内核不必带 scale 参数。
        //    数学上等价于"先算 Q·Kᵀ 再乘 scale"，与标准实现逐位一致。
        let mut q_scaled: Vec<f32> = qd.to_vec();
        q_scaled.par_iter_mut().for_each(|v| *v *= scale);

        // 2)~4) 前向：优先走「常驻显存」路径。
        //    S = Q'·Kᵀ → P = softmax(S+mask) → O = P·V 三个算子录进**一次提交**，
        //    S（33.6MB）与 P（33.6MB）全程留在显存，只把 O（4.2MB）回读给 CPU，
        //    P 的显存句柄随结果留到反向用 —— 逐算子路径要把 P 回读 33.6MB、下一步再原样传回，
        //    一来一回 67MB/层/次纯属白跑（实测单步回读 1.46GB，91% 的时间花在等回读）。
        //    失败（GPU 不可用/尺寸太小）自动回退下面的逐算子路径，数值行为不变。
        #[cfg(feature = "gpu")]
        let gpu_attn =
            crate::gpu::attn_forward(&q_scaled, &kd, &vd, mask_data, bh, t, t_total, head_dim);
        #[cfg(feature = "gpu")]
        let (out_data, cache) = match gpu_attn {
            Some(r) => {
                let out = r.out.clone();
                (out, AttnCache::Gpu(Box::new(r)))
            }
            None => {
                let (o, p) =
                    attn_forward_ops(&q_scaled, &kd, &vd, mask_data, bh, t, t_total, head_dim);
                (o, AttnCache::Cpu(p))
            }
        };
        #[cfg(not(feature = "gpu"))]
        let (out_data, cache) = {
            let (o, p) = attn_forward_ops(&q_scaled, &kd, &vd, mask_data, bh, t, t_total, head_dim);
            (o, AttnCache::Cpu(p))
        };

        drop(qd);
        drop(kd);
        drop(vd);
        drop(md);

        let requires = q.req() || k.req() || v.req();
        let mut result = Tensor::new(out_data, vec![bh, t, head_dim], requires);
        if requires {
            let rg = result.grad.clone();
            let sq = q.grad.clone();
            let sk = k.grad.clone();
            let sv = v.grad.clone();
            let k_data = k.data.clone();
            let v_data = v.data.clone();
            result.parents = Rc::new(vec![q.clone(), k.clone(), v.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                // 常驻显存路径：dV/dP/dS/dQ/dK 五个算子录进一次提交，只回读 dQ/dK/dV（各 4.2MB）
                #[cfg(feature = "gpu")]
                if let AttnCache::Gpu(r) = &cache {
                    if let Some((dq_out, dk_out, dv_out)) = r.backward(&g) {
                        {
                            let mut sgm = sq.borrow_mut();
                            for (i, v) in dq_out.iter().enumerate() {
                                sgm[i] += v * scale;
                            }
                        }
                        {
                            let mut sgm = sk.borrow_mut();
                            for (i, v) in dk_out.iter().enumerate() {
                                sgm[i] += v;
                            }
                        }
                        {
                            let mut sgm = sv.borrow_mut();
                            for (i, v) in dv_out.iter().enumerate() {
                                sgm[i] += v;
                            }
                        }
                        return;
                    }
                }
                // 逐算子（或纯 CPU）路径：P 在 CPU 上；常驻路径反向失败时把 P 回读出来兜底
                let p: Rc<Vec<f32>> = match &cache {
                    AttnCache::Cpu(p) => p.clone(),
                    #[cfg(feature = "gpu")]
                    AttnCache::Gpu(r) => {
                        Rc::new(r.read_p().expect("常驻显存反向失败，回读 P 也失败"))
                    }
                };
                // 反向：全部交给矩阵乘内核（旧的标量三重循环只有 0.24 GFLOP/s，慢 100 倍）
                // softmax 反向：dS[i,j] = P[i,j] * (dP[i,j] - Σ_k dP[i,k]·P[i,k])
                // 注意必须用 V 算 dP（out_i = Σ_j P_ij·V_j，故 ∂out_i/∂P_ij 的因子是 V_j）；
                // 用 K 会让 dP/dS/dQ/dK 全部错误（实测 dQ 偏差 2.7 倍）。
                let kd_b = k_data.borrow();
                let vd_b = v_data.borrow();
                // dV = Pᵀ·dO
                let dv_out = matmul_data(&p, &g, t_total, t, head_dim, bh, true, false);
                // dP = dO·Vᵀ
                let mut ds = matmul_data(&g, &vd_b, t, head_dim, t_total, bh, false, true);
                drop(g);
                drop(vd_b);
                // dS ← P ⊙ (dP - Σ_j dP·P)：逐行独立，按行并行（一行读两遍写一遍）
                let p_ref: &[f32] = &p;
                ds.par_chunks_mut(t_total).enumerate().for_each(|(r, drow)| {
                    let base = r * t_total;
                    let mut dot = 0.0f32;
                    for j in 0..t_total {
                        dot += drow[j] * p_ref[base + j];
                    }
                    for j in 0..t_total {
                        drow[j] = p_ref[base + j] * (drow[j] - dot);
                    }
                });
                // dQ = (dS·K)·scale（前向里 Q 先被缩成 Q' = Q·scale，故 dQ 要乘回来）；dK = dSᵀ·Q'
                let dq_out = matmul_data(&ds, &kd_b, t, t_total, head_dim, bh, false, false);
                let dk_out = matmul_data(&ds, &q_scaled, t_total, t, head_dim, bh, true, false);
                drop(kd_b);
                {
                    let mut sgm = sq.borrow_mut();
                    for (i, v) in dq_out.iter().enumerate() {
                        sgm[i] += v * scale;
                    }
                }
                {
                    let mut sgm = sk.borrow_mut();
                    for (i, v) in dk_out.iter().enumerate() {
                        sgm[i] += v;
                    }
                }
                {
                    let mut sgm = sv.borrow_mut();
                    for (i, v) in dv_out.iter().enumerate() {
                        sgm[i] += v;
                    }
                }
            }));
        }
        result
    }

    /// 融合"因果掩码相加 + softmax"：out = softmax_last_dim(x + mask)。
    ///
    /// mask 必须是 x 形状的右后缀（如 x [bh,t,tt] + mask [t,tt]），逐维相等；
    /// 每个元素对应的 mask 下标 = `flat % mask.numel()`。mask 中 -inf 的位置 softmax 后为 0。
    /// 反向与普通 softmax 相同（s=0 的位置梯度自然为 0，且 mask 是常量不需要梯度）。
    /// 一个算子替代 `add` + `softmax_last_dim` 两个算子，训练热路径里每层 block 一次。
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn masked_softmax(&self, mask: &Tensor) -> Tensor {
        let d = *self.shape.last().unwrap();
        assert!(
            mask.rank() <= self.rank(),
            "mask 维度必须 <= 输入（mask {:?} vs x {:?}）",
            mask.shape,
            self.shape
        );
        // mask 必须是输入形状的精确右后缀（逐维相等）。尺寸 1 的维只有当目标对应维也是 1
        // 时才会出现（如 t=1 的 KV cache 单步生成），此时"源下标 = flat % numel"依然成立。
        let off = self.rank() - mask.rank();
        assert_eq!(
            &self.shape[off..],
            mask.shape.as_slice(),
            "mask 必须是输入的右后缀（x {:?} vs mask {:?}）",
            self.shape,
            mask.shape
        );
        let m_n = mask.numel();
        let rows = self.numel() / d;
        let sd = self.data.borrow();
        let md = mask.data.borrow();
        // GPU 优先：训练里 scores 是 [B*H,T,T_total]（~200 万元素），GPU 计算 ~2ms，
        // CPU 计算 ~30ms。太小（推理单 token）或 GPU 不可用时自动回退 CPU。
        #[cfg(feature = "gpu")]
        let out = crate::gpu::softmax_mask(&sd, &md, rows, d, m_n)
            .unwrap_or_else(|| masked_softmax_cpu(&sd, &md, rows, d, m_n));
        #[cfg(not(feature = "gpu"))]
        let out = masked_softmax_cpu(&sd, &md, rows, d, m_n);
        drop(sd);
        drop(md);

        let requires = self.req() || mask.req();
        let out_shared = Rc::new(out);
        let mut result = Tensor::new(out_shared.as_ref().clone(), self.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let out = out_shared;
            result.parents = Rc::new(vec![self.clone(), mask.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                // 反向同样优先 GPU（p = 前向概率），失败回退 CPU
                #[cfg(feature = "gpu")]
                if let Some(dx) = crate::gpu::softmax_mask_backward(&g, &out, rows, d) {
                    for (i, v) in dx.iter().enumerate() {
                        sgm[i] += v;
                    }
                    return;
                }
                for r in 0..rows {
                    let mut dot = 0.0;
                    for j in 0..d {
                        dot += g[r * d + j] * out[r * d + j];
                    }
                    for i in 0..d {
                        sgm[r * d + i] += out[r * d + i] * (g[r * d + i] - dot);
                    }
                }
            }));
        }
        result
    }

    /// log_softmax（数值稳定版，最后一维）。
    ///
    /// 等价于 `softmax_last_dim().log()`，但用 log-sum-exp 技巧避免 `log(0) = -inf`。
    ///
    /// 公式：`log_softmax(x_i) = x_i - max - log(Σ exp(x_j - max))`
    ///
    /// 反向：`grad_input_i = grad_output_i - softmax(x_i) · Σ grad_output_j`
    /// （比 softmax+log 的链式法则更简洁，不需存储中间的 softmax 结果乘以 log 的梯度）
    #[allow(dead_code)]
    pub fn log_softmax_last_dim(&self) -> Tensor {
        assert!(self.rank() >= 1, "log_softmax 至少需要 1 维");
        let (rows, d) = (
            self.numel() / self.shape[self.rank() - 1],
            self.shape[self.rank() - 1],
        );
        let sd = self.data.borrow();
        let sd_ref: &[f32] = &sd;
        let mut out_data = vec![0.0f32; rows * d];
        // 存 softmax 值（反向需要）
        let mut softmax_data = vec![0.0f32; rows * d];
        // 并行：每行独立做 log-sum-exp，行间无依赖。
        // 输出头是 [B*T, vocab]（4096×8192 = 3355 万元素），单线程时这一步
        // 占单步 ~11%（1.0s），是矩阵乘之外最大的热点。
        out_data
            .par_chunks_mut(d)
            .zip(softmax_data.par_chunks_mut(d))
            .enumerate()
            .for_each(|(r, (orow, srow))| {
                let base = r * d;
                let mut maxv = f32::NEG_INFINITY;
                for j in 0..d {
                    maxv = maxv.max(sd_ref[base + j]);
                }
                let mut sum_exp = 0.0f32;
                for j in 0..d {
                    let e = (sd_ref[base + j] - maxv).exp();
                    srow[j] = e;
                    sum_exp += e;
                }
                let log_sum = sum_exp.ln();
                for j in 0..d {
                    srow[j] /= sum_exp; // 归一化为 softmax 概率
                    orow[j] = sd_ref[base + j] - maxv - log_sum;
                }
            });
        drop(sd);

        let mut result = Tensor::new(out_data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g: Vec<f32> = rg.borrow().to_vec();
                let mut sgm = sg.borrow_mut();
                // 并行：同样按行切分，行间无依赖（每一行只写自己的 d 个元素）
                sgm.par_chunks_mut(d).enumerate().for_each(|(r, grow)| {
                    let base = r * d;
                    let mut dot = 0.0;
                    for j in 0..d {
                        dot += g[base + j];
                    }
                    for i in 0..d {
                        grow[i] += g[base + i] - softmax_data[base + i] * dot;
                    }
                });
            }));
        }
        result
    }

    // ---------- 正则化 ----------

    /// Dropout（反转实现）：训练时随机置零 + 缩放，推理时恒等。
    ///
    /// - `p`：每个元素被置零的概率（0 = 不丢弃，1 = 全丢弃）
    /// - `training`：true 时启用随机丢弃，false 时直接返回克隆
    ///
    /// 反转技巧（inverted dropout）：
    /// - 训练时 `out = mask · x / (1-p)`（mask ∈ {0, 1}），期望 E[out] = E[x]
    /// - 推理时 `out = x`（无需额外操作）
    /// - 反向：梯度同样乘以 `mask / (1-p)`
    ///
    /// mask 用 xorshift64* 从 thread_local 独立随机流生成，无需外部依赖。
    /// 每次调用推进一次计数器，因此同一张量在不同步会拿到不同 mask；
    /// 训练是单线程的，调用顺序确定，所以结果仍可复现。
    pub fn dropout(&self, p: f32, training: bool) -> Tensor {
        assert!((0.0..=1.0).contains(&p), "dropout 概率 p 必须在 [0, 1] 之间");
        if !training || p == 0.0 {
            // 推理或不丢弃：恒等（需要梯度时设 requires_grad）
            return Tensor::new(self.data.borrow().clone(), self.shape.clone(), self.requires_grad);
        }
        if p >= 1.0 {
            return Tensor::new(vec![0.0; self.numel()], self.shape.clone(), self.requires_grad);
        }
        let keep = 1.0 - p;
        let scale = 1.0 / keep;
        let sd = self.data.borrow();
        let len = sd.len();
        let mut mask = vec![0.0f32; len];
        // 独立随机流：thread_local 计数器推进 + splitmix64 打散，
        // 保证同一 dropout 位点每步得到**不同**的 mask（详见 DROPOUT_STREAM 注释）。
        let mut state = DROPOUT_STREAM.with(|s| {
            let v = s.get();
            s.set(v.wrapping_add(0x9E37_79B9_7F4A_7C15));
            v
        });
        // splitmix64 finalizer：打散计数器低位的规律性，避免相邻种子生成相关序列
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        state = (z ^ (z >> 31)) | 1; // |1 保证状态非零（xorshift 全零会退化）
        for m in mask.iter_mut() {
            // xorshift64*
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            // 取高 24 位映射到 [0,1)：f32 尾数只有 24 位，
            // 若直接 `state as f32 / u64::MAX as f32` 会在两端损失分辨率
            let u = ((state >> 40) as f32) * (1.0 / 16_777_216.0);
            *m = if u < keep { scale } else { 0.0 };
        }
        let out_data: Vec<f32> = sd.iter().zip(&mask).map(|(x, m)| x * m).collect();
        drop(sd);

        let mut result = Tensor::new(out_data, self.shape.clone(), self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] * mask[i];
                }
            }));
        }
        result
    }

    // ---------- 索引运算 ----------

    /// 按行索引取值：table [V, D]，indices [N] -> out [N, D]。
    /// 反向：梯度 scatter-add 回 table 对应行（Embedding 用）。
    pub fn gather_rows(&self, indices: &[usize]) -> Tensor {
        assert_eq!(self.rank(), 2, "gather_rows 的 table 必须为 2 维");
        let (v, d) = (self.shape[0], self.shape[1]);
        let n = indices.len();
        let sd = self.data.borrow();
        let mut out_data = vec![0.0f32; n * d];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < v, "gather 索引越界：{} >= {}", idx, v);
            for j in 0..d {
                out_data[i * d + j] = sd[idx * d + j];
            }
        }
        drop(sd);
        let idx_vec = indices.to_vec();

        let mut result = Tensor::new(out_data, vec![n, d], self.requires_grad);
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let d2 = d;
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..idx_vec.len() {
                    let row = idx_vec[i];
                    for j in 0..d2 {
                        sgm[row * d2 + j] += g[i * d2 + j];
                    }
                }
            }));
        }
        result
    }

    // ---------- 自动微分核心（实现见 autograd.rs） ----------
}

impl std::fmt::Display for Tensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Tensor shape={:?}:\n", self.shape)?;
        let data = self.data.borrow();
        if self.rank() == 0 {
            return write!(f, "[ {} ]", data[0]);
        }
        match self.rank() {
            1 => {
                write!(f, "[ ")?;
                for v in data.iter() {
                    write!(f, "{} ", v)?;
                }
                write!(f, "]")
            }
            2 => {
                let (rows, cols) = (self.shape[0], self.shape[1]);
                for i in 0..rows {
                    write!(f, "[ ")?;
                    for j in 0..cols {
                        write!(f, "{} ", data[i * cols + j])?;
                    }
                    write!(f, "]\n")?;
                }
                Ok(())
            }
            _ => {
                write!(f, "[ ")?;
                for v in data.iter() {
                    write!(f, "{} ", v)?;
                }
                write!(f, "]")
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matmul_2d() {
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let c = a.matmul(&b);
        assert_eq!(c.data(), vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn test_matmul_3d() {
        // A: [2,1,2] x B: [2,2,1]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
        let b = Tensor::from_vec(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2, 1]);
        let c = a.matmul(&b);
        assert_eq!(c.shape(), &[2, 1, 1]);
        assert_eq!(c.data(), vec![17.0, 53.0]); // [1*5+2*6=17, 3*7+4*8=53]
    }

    #[test]
    fn test_broadcast_add() {
        // [2,3] + [3] 广播
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::from_vec(vec![10.0, 20.0, 30.0], vec![3]);
        let c = a.add(&b);
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data(), vec![11.0, 22.0, 33.0, 14.0, 25.0, 36.0]);
        // 梯度验证：loss = sum(c)
        let a = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = Tensor::param(vec![10.0, 20.0, 30.0], vec![3]);
        let loss = a.add(&b).sum();
        loss.backward();
        // b 的每个元素被广播两次，梯度应为 2
        assert_eq!(b.grad(), vec![2.0, 2.0, 2.0]);
        assert_eq!(a.grad(), vec![1.0; 6]);
    }

    #[test]
    fn test_layernorm_fused_matches_chain() {
        // 融合 LayerNorm 与"基础算子链"参考实现对比：前向 + x/γ/β 三路梯度
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
        ];
        let d = 3;
        let (gamma_data, beta_data) = (vec![1.0, 0.5, 2.0], vec![0.1, -0.2, 0.3]);
        // 融合实现
        let x1 = Tensor::param(data.clone(), vec![4, 3]);
        let g1 = Tensor::param(gamma_data.clone(), vec![3]);
        let b1 = Tensor::param(beta_data.clone(), vec![3]);
        let y1 = x1.layernorm(&g1, &b1, 1e-5);
        // 基础算子链（等价参考实现）
        let x2 = Tensor::param(data.clone(), vec![4, 3]);
        let g2 = Tensor::param(gamma_data, vec![3]);
        let b2 = Tensor::param(beta_data, vec![3]);
        let mean = x2.sum_last_dim().mul_scalar(1.0 / d as f32);
        let centered = x2.sub(&mean);
        let var = centered.mul(&centered).sum_last_dim().mul_scalar(1.0 / d as f32);
        let norm = centered.div(&var.add_scalar(1e-5).sqrt());
        let y2 = norm.mul(&g2).add(&b2);
        // 前向一致
        for (i, (u, v)) in y1.data().iter().zip(&y2.data()).enumerate() {
            assert!((u - v).abs() < 1e-4, "前向 {i}: {u} vs {v}");
        }
        // 反向一致
        y1.sum().backward();
        y2.sum().backward();
        for i in 0..data.len() {
            assert!(
                (x1.grad()[i] - x2.grad()[i]).abs() < 1e-3,
                "x 梯度 {i}: {} vs {}",
                x1.grad()[i],
                x2.grad()[i]
            );
        }
        for i in 0..3 {
            assert!(
                (g1.grad()[i] - g2.grad()[i]).abs() < 1e-3,
                "γ 梯度 {i}: {} vs {}",
                g1.grad()[i],
                g2.grad()[i]
            );
            assert!(
                (b1.grad()[i] - b2.grad()[i]).abs() < 1e-3,
                "β 梯度 {i}: {} vs {}",
                b1.grad()[i],
                b2.grad()[i]
            );
        }
    }

    #[test]
    fn test_masked_softmax_matches_chain() {
        // 融合 mask+softmax 与 add+softmax 参考实现对比：前向 + 梯度
        let data = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0,
        ];
        let mask = Tensor::from_vec(
            vec![
                0.0,
                f32::NEG_INFINITY,
                0.0,
                0.0,
                0.0,
                f32::NEG_INFINITY,
            ],
            vec![2, 3],
        );
        let x1 = Tensor::param(data.clone(), vec![2, 2, 3]);
        let y1 = x1.masked_softmax(&mask);
        let x2 = Tensor::param(data.clone(), vec![2, 2, 3]);
        let y2 = x2.add(&mask).softmax_last_dim();
        for (i, (u, v)) in y1.data().iter().zip(&y2.data()).enumerate() {
            assert!((u - v).abs() < 1e-5, "前向 {i}: {u} vs {v}");
        }
        y1.sum().backward();
        y2.sum().backward();
        for i in 0..data.len() {
            assert!(
                (x1.grad()[i] - x2.grad()[i]).abs() < 1e-4,
                "梯度 {i}: {} vs {}",
                x1.grad()[i],
                x2.grad()[i]
            );
        }
    }

    #[test]
    fn test_chain_rule() {
        let x = Tensor::param(vec![2.0], vec![]);
        let y = Tensor::param(vec![3.0], vec![]);
        let w = Tensor::param(vec![1.0], vec![]);
        let z = x.mul(&y).add(&w);
        assert_eq!(z.data(), vec![7.0]);
        z.backward();
        assert!((x.grad()[0] - 3.0).abs() < 1e-6);
        assert!((y.grad()[0] - 2.0).abs() < 1e-6);
        assert!((w.grad()[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_relu_grad() {
        let x = Tensor::param(vec![1.0, -2.0, 3.0], vec![3]);
        let y = x.relu().sum();
        y.backward();
        assert_eq!(x.grad(), vec![1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_softmax() {
        // softmax([1,2,3]) = [0.0900, 0.2447, 0.6652]
        let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![1, 3]);
        let s = x.softmax_last_dim();
        let d = s.data();
        let sum: f32 = d.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
        assert!((d[0] - 0.0900).abs() < 1e-3);
        assert!((d[2] - 0.6652).abs() < 1e-3);
    }

    #[test]
    fn test_log_softmax() {
        // log_softmax 应等价于 log(softmax(x))
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let ls = x.log_softmax_last_dim();
        let s = x.softmax_last_dim();
        let log_s = s.log();
        let d1 = ls.data();
        let d2 = log_s.data();
        for (a, b) in d1.iter().zip(d2.iter()) {
            assert!((a - b).abs() < 1e-5, "log_softmax vs log(softmax): {} vs {}", a, b);
        }
        // 反向梯度也应一致
        let loss1 = ls.sum();
        loss1.backward();
        let g1 = x.grad();
        x.zero_grad();
        let loss2 = log_s.sum();
        loss2.backward();
        let g2 = x.grad();
        for (a, b) in g1.iter().zip(g2.iter()) {
            assert!((a - b).abs() < 1e-2, "梯度不一致: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_permute() {
        // [2,3] -> [3,2]
        let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let p = a.permute(&[1, 0]);
        assert_eq!(p.shape(), &[3, 2]);
        assert_eq!(p.data(), vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_gather_rows() {
        // 表 [3,2]，取第 0、2 行
        let t = Tensor::param(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let g = t.gather_rows(&[0, 2]);
        assert_eq!(g.data(), vec![1.0, 2.0, 5.0, 6.0]);
        let loss = g.sum();
        loss.backward();
        // 第 0 行梯度 1，第 2 行梯度 1，第 1 行梯度 0
        assert_eq!(t.grad(), vec![1.0, 1.0, 0.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn test_linear_regression_converges() {
        use crate::loss::mse_loss;
        let x_aug = Tensor::from_vec(
            vec![1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0],
            vec![5, 2],
        );
        let y_true = Tensor::from_vec(vec![3.0, 5.0, 7.0, 9.0, 11.0], vec![5, 1]);
        let w = Tensor::param(vec![0.0, 0.0], vec![2, 1]);
        let lr = 0.005; // mse_loss 取均值，梯度更小，lr 相应放大
        for _ in 0..5000 {
            let pred = x_aug.matmul(&w);
            let loss = mse_loss(&pred, &y_true);
            loss.backward();
            let gw = w.grad();
            w.set_data(vec![w.data()[0] - lr * gw[0], w.data()[1] - lr * gw[1]]);
            w.zero_grad();
        }
        let f = w.data();
        assert!((f[0] - 2.0).abs() < 0.1, "w = {}", f[0]);
        assert!((f[1] - 1.0).abs() < 0.1, "b = {}", f[1]);
    }

    /// 回归测试：3D 输入经 Linear（内部 reshape）后，梯度必须完整流到 weight 与输入。
    /// 曾因 backward 用 data 指针判重、而 reshape 与输入共享 data Rc，
    /// 导致父节点被 DFS 跳过、weight/输入梯度全 0（见 src/autograd.rs 注释）。
    #[test]
    fn test_reshape_grad_flows() {
        use crate::layers::Linear;
        use crate::rng::Rng;
        let mut rng = Rng::new(0);
        // 3D 输入走 Linear（内部 reshape 2D -> matmul -> reshape 3D）
        let x = Tensor::param(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            vec![2, 2, 2],
        );
        let fc = Linear::new(2, 3, &mut rng);
        let y = fc.forward(&x); // [2,2,3]
        let loss = y.mul(&y).sum(); // 标量
        loss.backward();
        let wg = fc.weight.grad();
        let xg = x.grad();
        assert!(
            wg.iter().any(|&v| v != 0.0),
            "Linear weight 梯度全 0：reshape 路径梯度被截断"
        );
        assert!(
            xg.iter().any(|&v| v != 0.0),
            "Linear 输入梯度全 0：reshape 路径梯度被截断"
        );
    }

    /// RMSNorm 前向+反向：对比逐算子实现验证正确性
    #[test]
    fn test_rmsnorm_fused_matches_chain() {
        // 手工数据
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let x = Tensor::param(x_data.clone(), vec![2, 2]);
        let gamma = Tensor::param(vec![1.0, 2.0], vec![2]);
        let eps = 1e-5f32;

        // 融合实现
        let out = x.rmsnorm(&gamma, eps);
        let out_ref = out.data();

        // 手算参考值
        // row0: [1, 2] → ms = (1+4)/2 = 2.5, rms = √2.5, is = 1/√2.5
        // out = [1*is*1, 2*is*2] = [0.6325, 2.5298]
        // row1: [3, 4] → ms = (9+16)/2 = 12.5, rms = √12.5, is = 1/√12.5
        // out = [3*is*1, 4*is*2] = [0.8485, 2.2627]
        let rms0 = 1.0f32 / (2.5f32 + eps).sqrt();
        assert!(
            (out_ref[0] - 1.0 * rms0).abs() < 1e-4,
            "out[0] = {} vs {}",
            out_ref[0],
            1.0 * rms0
        );
        assert!(
            (out_ref[1] - 2.0 * rms0 * 2.0).abs() < 1e-4,
            "out[1] = {} vs {}",
            out_ref[1],
            2.0 * rms0 * 2.0
        );

        // 反向：loss = sum(out²)，梯度应非零
        let loss = out.mul(&out).sum();
        loss.backward();
        let xg = x.grad();
        let gg = gamma.grad();
        assert!(xg.iter().any(|&v| v.abs() > 1e-6), "RMSNorm 输入梯度全 0");
        assert!(gg.iter().any(|&v| v.abs() > 1e-6), "RMSNorm γ 梯度全 0");
    }

    /// SwiGLU 前向+反向：对比逐元素实现验证正确性
    #[test]
    fn test_swiglu_matches_elementwise() {
        let x_data = vec![0.5, -1.0, 2.0, -0.5];
        let g_data = vec![1.0, 0.5, -0.5, 2.0];
        let x = Tensor::param(x_data.clone(), vec![4]);
        let gate = Tensor::param(g_data.clone(), vec![4]);

        let out = x.swiglu(&gate);
        let out_ref = out.data();

        // 手算 SiLU(0.5) = 0.5 * sigmoid(0.5) ≈ 0.3113
        // out[0] = SiLU(0.5) * 1.0 ≈ 0.3113
        let sigmoid_05 = 1.0 / (1.0 + (-0.5f32).exp());
        let silu_05 = 0.5 * sigmoid_05;
        assert!(
            (out_ref[0] - silu_05).abs() < 1e-4,
            "SwiGLU[0] = {} vs {}",
            out_ref[0],
            silu_05
        );

        // 反向
        let loss = out.mul(&out).sum();
        loss.backward();
        let xg = x.grad();
        let gg = gate.grad();
        assert!(xg.iter().any(|&v| v.abs() > 1e-6), "SwiGLU x 梯度全 0");
        assert!(gg.iter().any(|&v| v.abs() > 1e-6), "SwiGLU gate 梯度全 0");
    }

    /// Dropout：推理模式（training=false）应恒等
    #[test]
    fn test_dropout_eval_identity() {
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = x.dropout(0.5, false);
        let out_ref = out.data();
        assert_eq!(out_ref, &[1.0, 2.0, 3.0, 4.0], "推理模式 dropout 应恒等");
    }

    /// Dropout：p=0 应恒等
    #[test]
    fn test_dropout_p0_identity() {
        let x = Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let out = x.dropout(0.0, true);
        assert_eq!(out.data(), &[1.0, 2.0, 3.0, 4.0], "p=0 dropout 应恒等");
    }

    /// Dropout：连续两次训练态调用必须产生**不同** mask。
    ///
    /// 回归测试：旧实现按「数据指针 + 首元素」派生种子，同一个张量连续调用会拿到
    /// 完全相同的 mask，dropout 退化成固定掩码（永久丢掉固定通道），失去正则化作用。
    #[test]
    fn test_dropout_masks_differ_across_calls() {
        // 全 1 输入：保留的元素应正好等于 1/(1-p) = 2，丢弃的元素为 0
        let x = Tensor::param(vec![1.0; 4096], vec![4096]);
        let a = x.dropout(0.5, true).data();
        let b = x.dropout(0.5, true).data();
        assert_ne!(a, b, "连续两次 dropout 产生了相同 mask（种子未推进）");

        let frac = a.iter().filter(|&&v| v == 0.0).count() as f32 / a.len() as f32;
        assert!((frac - 0.5).abs() < 0.1, "丢弃比例 {frac:.3} 偏离 0.5 过多（mask 分布异常）");
        assert!(
            a.iter().all(|&v| v == 0.0 || (v - 2.0).abs() < 1e-6),
            "保留元素未按 1/(1-p) = 2 缩放"
        );
    }

    /// Flash Attention 前向：对比标准 attention 验证输出一致
    #[test]
    fn test_flash_attention_matches_standard() {
        use crate::rng::Rng;
        let mut rng = Rng::new(42);
        let (bh, t, d) = (1, 4, 2);

        // 构造 Q/K/V
        let q_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn()).collect();
        let k_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn()).collect();
        let v_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn()).collect();
        let q = Tensor::from_vec(q_data.clone(), vec![bh, t, d]);
        let k = Tensor::from_vec(k_data.clone(), vec![bh, t, d]);
        let v = Tensor::from_vec(v_data.clone(), vec![bh, t, d]);

        // 因果掩码
        let mut mask_data = vec![f32::NEG_INFINITY; t * t];
        for i in 0..t {
            for j in 0..=i {
                mask_data[i * t + j] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, vec![t, t]);

        // 标准 attention
        let scale = 1.0 / (d as f32).sqrt();
        let kt = k.permute(&[0, 2, 1]);
        let scores = q.mul_scalar(scale).matmul(&kt);
        let attn_std = scores.masked_softmax(&mask);
        let out_std = attn_std.matmul(&v);

        // Flash attention
        let out_flash = Tensor::flash_attention(&q, &k, &v, &mask, 2);

        let std_data = out_std.data();
        let flash_data = out_flash.data();
        for i in 0..std_data.len() {
            assert!(
                (std_data[i] - flash_data[i]).abs() < 1e-4,
                "flash[{}] = {} vs std = {}",
                i,
                flash_data[i],
                std_data[i]
            );
        }
    }

    /// Flash Attention **反向**：与标准 attention 对比梯度（多 K/V 块场景）。
    ///
    /// 回归测试。在线 softmax 中运行最大值 m 增大时，必须把**此前已存入 attn_data 的 P**
    /// 一并乘以 rescale = exp(m_old - m_new)。漏掉这一步，同一行各块就用了不同的归一化
    /// 基准；末尾统一除以 l（对应最终 max）时，前序块被整体放大 exp(m_final - m_old) 倍。
    /// 该因子随注意力变尖锐**指数增长**，反向的 dS = P·(dP - ΣdP·P) 随之指数爆炸
    /// （实测正式训练 1500 步内梯度从 0.67 涨到 5.8e6，而权重几乎不动，因为前向是对的）。
    ///
    /// 旧测试 `test_flash_attention_matches_standard` 只有 1 个 K/V 块且用
    /// `Tensor::from_vec`（不追踪梯度），因此前向输出正确、bug 却完全隐形。
    #[test]
    fn test_flash_attention_backward_matches_standard() {
        use crate::rng::Rng;
        let (bh, t, d, bs) = (2, 16, 8, 4); // t/bs = 4 个 K/V 块，确保跨块 max 更新
        let mut rng = Rng::new(7);

        // 放大 Q/K 让 score 动态范围更大，从而更容易出现「最大值落在后续块」的行，
        // 否则这些行的 P 基准容易碰巧一致，测试就抓不到 bug。
        let amp = 3.0f32;
        let q_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn() * amp).collect();
        let k_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn() * amp).collect();
        let v_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn()).collect();

        // 因果掩码
        let mut mask_data = vec![f32::NEG_INFINITY; t * t];
        for i in 0..t {
            for j in 0..=i {
                mask_data[i * t + j] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, vec![t, t]);
        let scale = 1.0 / (d as f32).sqrt();

        // ---- 标准 attention（参考实现）的梯度 ----
        let q_s = Tensor::param(q_data.clone(), vec![bh, t, d]);
        let k_s = Tensor::param(k_data.clone(), vec![bh, t, d]);
        let v_s = Tensor::param(v_data.clone(), vec![bh, t, d]);
        let kt = k_s.permute(&[0, 2, 1]);
        let scores = q_s.mul_scalar(scale).matmul(&kt);
        let attn = scores.masked_softmax(&mask);
        let out_std = attn.matmul(&v_s);
        out_std.sum().backward();

        // ---- flash attention 的梯度 ----
        let q_f = Tensor::param(q_data, vec![bh, t, d]);
        let k_f = Tensor::param(k_data, vec![bh, t, d]);
        let v_f = Tensor::param(v_data, vec![bh, t, d]);
        let out_flash = Tensor::flash_attention(&q_f, &k_f, &v_f, &mask, bs);
        out_flash.sum().backward();

        for (name, a, b) in [
            ("dQ", q_s.grad(), q_f.grad()),
            ("dK", k_s.grad(), k_f.grad()),
            ("dV", v_s.grad(), v_f.grad()),
        ] {
            let max_err = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let ref_mag = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
            assert!(
                max_err / ref_mag < 1e-3,
                "{name} 梯度不一致：最大绝对误差 {max_err}（参考量级 {ref_mag}）—— \
                 跨块 P 未随运行最大值同步缩放，反向已指数放大"
            );
        }
    }

    /// 常驻显存路径（`gpu::attn_forward` + `AttnResident::backward`）的数值校验。
    ///
    /// 为什么另起一个测试：上面两个 flash attention 测试用的是 (bh=1,t=4,d=2) 与 (2,16,8)
    /// 这种极小形状，会被 `attn_resident_ok` 的尺寸守卫挡掉、回退到逐算子路径 —— 测不到新代码。
    /// 这里的形状能过守卫（并在 GPU 可用时断言确实命中了常驻路径），
    /// 参考值用**纯循环独立算一遍**、不依赖任何 Tensor 算子，因此前向与反向都验证得到。
    #[test]
    fn test_flash_attention_resident_path_matches_loop_reference() {
        use crate::rng::Rng;
        #[cfg(feature = "gpu")]
        crate::gpu::init();

        let (bh, t, tt, hd) = (16usize, 256usize, 256usize, 32usize);
        let scale = 1.0 / (hd as f32).sqrt();
        let mut rng = Rng::new(11);

        let q: Vec<f32> = (0..bh * t * hd).map(|_| rng.randn()).collect();
        let k: Vec<f32> = (0..bh * tt * hd).map(|_| rng.randn()).collect();
        let v: Vec<f32> = (0..bh * tt * hd).map(|_| rng.randn()).collect();
        // 因果掩码 [t, tt]：j <= i 时为 0，其余 -inf
        let mut mask = vec![f32::NEG_INFINITY; t * tt];
        for i in 0..t {
            for j in 0..=i.min(tt - 1) {
                mask[i * tt + j] = 0.0;
            }
        }

        // GPU 可用时必须真的走常驻路径，否则这个测试会静默退化成旧路径而失去意义
        #[cfg(feature = "gpu")]
        if crate::gpu::is_available() {
            assert!(
                crate::gpu::attn_resident_ok(bh, t, tt, hd, mask.len()),
                "该形状未命中常驻显存路径，测试失去意义"
            );
        }

        // ---- 纯循环参考实现：前向 O，以及损失 = sum(O) 的反向 ----
        let rows = bh * t;
        let mut p_ref = vec![0.0f32; rows * tt];
        let mut o_ref = vec![0.0f32; bh * t * hd];
        let mut dq_ref = vec![0.0f32; bh * t * hd];
        let mut dk_ref = vec![0.0f32; bh * tt * hd];
        let mut dv_ref = vec![0.0f32; bh * tt * hd];
        for b in 0..bh {
            for i in 0..t {
                let r = b * t + i;
                let mb = (r * tt) % (t * tt);
                // 手写循环用 float 累加，与内核/rayon 的累加顺序不同 → 只比到 1e-3 相对量级
                let mut srow = vec![0.0f32; tt];
                let mut mx = f32::NEG_INFINITY;
                for j in 0..tt {
                    let mut acc = 0.0f32;
                    for d in 0..hd {
                        acc += q[r * hd + d] * k[(b * tt + j) * hd + d];
                    }
                    srow[j] = acc * scale + mask[mb + j];
                    mx = mx.max(srow[j]);
                }
                let mut sum = 0.0f32;
                for j in 0..tt {
                    let e = (srow[j] - mx).exp();
                    p_ref[r * tt + j] = e;
                    sum += e;
                }
                for j in 0..tt {
                    p_ref[r * tt + j] /= sum;
                }
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..tt {
                        acc += p_ref[r * tt + j] * v[(b * tt + j) * hd + d];
                    }
                    o_ref[r * hd + d] = acc;
                }
                // dO = 1 → dP[j] = Σ_d V[j,d]；dS[j] = P[j]·(dP[j] - Σ_j dP·P)
                let mut dot = 0.0f32;
                for j in 0..tt {
                    let mut dp = 0.0f32;
                    for d in 0..hd {
                        dp += v[(b * tt + j) * hd + d];
                    }
                    dot += dp * p_ref[r * tt + j];
                }
                let mut ds = vec![0.0f32; tt];
                for j in 0..tt {
                    let mut dp = 0.0f32;
                    for d in 0..hd {
                        dp += v[(b * tt + j) * hd + d];
                    }
                    ds[j] = p_ref[r * tt + j] * (dp - dot);
                }
                // dQ = (dS·K)·scale；dK = dSᵀ·Q'（Q' = Q·scale）；dV = Pᵀ·dO
                for d in 0..hd {
                    let mut acc = 0.0f32;
                    for j in 0..tt {
                        acc += ds[j] * k[(b * tt + j) * hd + d];
                    }
                    dq_ref[r * hd + d] += acc * scale;
                }
                for j in 0..tt {
                    for d in 0..hd {
                        dk_ref[(b * tt + j) * hd + d] += ds[j] * q[r * hd + d] * scale;
                        dv_ref[(b * tt + j) * hd + d] += p_ref[r * tt + j];
                    }
                }
            }
        }

        // ---- 走 Tensor（GPU 可用时即常驻显存路径）----
        let qt = Tensor::param(q, vec![bh, t, hd]);
        let kt = Tensor::param(k, vec![bh, tt, hd]);
        let vt = Tensor::param(v, vec![bh, tt, hd]);
        let mt = Tensor::from_vec(mask, vec![t, tt]);
        let out = Tensor::flash_attention(&qt, &kt, &vt, &mt, 64);
        out.sum().backward();

        for (name, a, b) in [
            ("前向 O", &o_ref, &out.data()),
            ("dQ", &dq_ref, &qt.grad()),
            ("dK", &dk_ref, &kt.grad()),
            ("dV", &dv_ref, &vt.grad()),
        ] {
            let max_err = a
                .iter()
                .zip(b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let ref_mag = a.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
            assert!(
                max_err / ref_mag < 1e-3,
                "{name} 与纯循环参考不一致：最大绝对误差 {max_err}（参考量级 {ref_mag}）"
            );
        }
    }
}
