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
    for bi in 0..batch {
        let (oa, ob, oo) = (bi * m * k, bi * k * n, bi * m * n);
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for kk in 0..k {
                    s += a2[oa + i * k + kk] * b2[ob + kk * n + j];
                }
                out[oo + i * n + j] = s;
            }
        }
    }
    out
}

/// 展平批矩阵转置：[B, rows, cols] -> [B, cols, rows]（CPU 回退用，GPU 路径不需要）
fn transpose_flat(v: &[f32], rows: usize, cols: usize, batch: usize) -> Vec<f32> {
    let mut t = vec![0.0f32; batch * rows * cols];
    for b in 0..batch {
        for i in 0..rows {
            for j in 0..cols {
                t[(b * cols + j) * rows + i] = v[(b * rows + i) * cols + j];
            }
        }
    }
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
        Tensor {
            data: Rc::new(RefCell::new(data)),
            shape,
            grad: Rc::new(RefCell::new(vec![0.0; len])),
            requires_grad,
            parents: Rc::new(Vec::new()),
            backward: None,
        }
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

    pub fn data(&self) -> Vec<f32> {
        self.data.borrow().clone()
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
        let mut result = Tensor {
            data: self.data.clone(), // Rc 共享，不克隆 Vec
            shape: new_shape,
            grad: Rc::new(RefCell::new(vec![0.0; numel])),
            requires_grad: self.requires_grad,
            parents: Rc::new(Vec::new()),
            backward: None,
        };
        if self.requires_grad {
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
        if self.requires_grad {
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

    pub fn mul(&self, other: &Tensor) -> Tensor {
        // ∂c/∂a = b，∂c/∂b = a
        self.binary(other, |a, b| a * b, |a, b| (b, a))
    }

    pub fn div(&self, other: &Tensor) -> Tensor {
        self.binary(other, |a, b| a / b, |a, b| { let s = b + 1e-8; (1.0 / s, -a / (s * s)) })
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

        let requires = self.requires_grad || other.requires_grad;
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
                if same_shape && Rc::ptr_eq(&sg, &og) {
                    // 特例：同一张量参与运算（如 x*x、x/x），两条路径梯度叠加
                    let mut sgm = sg.borrow_mut();
                    for i in 0..g.len() {
                        let (da, db) = back(sd_b[i], od_b[i]);
                        sgm[i] += g[i] * (da + db);
                    }
                } else {
                    let mut sgm = sg.borrow_mut();
                    let mut ogm = og.borrow_mut();
                    for t in 0..g.len() {
                        let ia = src_idx(t, &a_src_c);
                        let ib = src_idx(t, &b_src_c);
                        let (da, db) = back(sd_b[ia], od_b[ib]);
                        sgm[ia] += g[t] * da;
                        ogm[ib] += g[t] * db;
                    }
                }
            }));
        }
        result
    }

    // ---------- 标量运算 ----------

    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        let data = self.data.borrow().iter().map(|a| a + scalar).collect();
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
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
        if self.requires_grad {
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
    pub fn neg(&self) -> Tensor {
        let data = self.data.borrow().iter().map(|a| -a).collect();
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
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
        if self.requires_grad {
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
        if self.requires_grad {
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
        if self.requires_grad {
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

    /// log：c = ln(x)，∂x = g / x（cross_entropy 已改用 log_softmax_last_dim，仅测试使用）
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn log(&self) -> Tensor {
        let sd = self.data.borrow();
        let data: Vec<f32> = sd.iter().map(|&a| a.ln()).collect();
        drop(sd);
        let mut result = Tensor::new(data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
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
        if self.requires_grad {
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

        let requires = self.requires_grad || other.requires_grad;
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

        let requires = self.requires_grad || other.requires_grad;
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
        if self.requires_grad {
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
        if self.requires_grad {
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
        if self.requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let (rows, d) = (rows, d);
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for r in 0..rows {
                    // dot = Σ_j g_j * s_j
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

        let requires = self.requires_grad || gamma.requires_grad || beta.requires_grad;
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
                let g = rg.borrow();
                let x_b = xd.borrow();
                let gam = gd.borrow();
                let mut gx = sx.borrow_mut();
                let mut gg = sg.borrow_mut();
                let mut gb = sb.borrow_mut();
                let mut dg = vec![0.0f32; d];
                let mut db = vec![0.0f32; d];
                for r in 0..rows {
                    let base = r * d;
                    let is = inv_std[r];
                    let mut m1 = 0.0f32; // mean(d_y·γ)
                    let mut m2 = 0.0f32; // mean(d_y·γ·norm)
                    for j in 0..d {
                        let dy = g[base + j];
                        let dy_g = dy * gam[j];
                        m1 += dy_g;
                        m2 += dy_g * (x_b[base + j] - mean[r]) * is;
                        dg[j] += dy * (x_b[base + j] - mean[r]) * is;
                        db[j] += dy;
                    }
                    m1 /= d as f32;
                    m2 /= d as f32;
                    for j in 0..d {
                        let norm = (x_b[base + j] - mean[r]) * is;
                        gx[base + j] += is * (g[base + j] * gam[j] - m1 - m2 * norm);
                    }
                }
                for j in 0..d {
                    gg[j] += dg[j];
                    gb[j] += db[j];
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

        let requires = self.requires_grad || mask.requires_grad;
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
        let mut out_data = vec![0.0f32; rows * d];
        // 存 softmax 值（反向需要）
        let mut softmax_data = vec![0.0f32; rows * d];
        for r in 0..rows {
            let mut maxv = f32::NEG_INFINITY;
            for j in 0..d {
                maxv = maxv.max(sd[r * d + j]);
            }
            let mut sum_exp = 0.0f32;
            for j in 0..d {
                let e = (sd[r * d + j] - maxv).exp();
                softmax_data[r * d + j] = e;
                sum_exp += e;
            }
            let log_sum = sum_exp.ln();
            for j in 0..d {
                softmax_data[r * d + j] /= sum_exp; // 归一化为 softmax 概率
                out_data[r * d + j] = sd[r * d + j] - maxv - log_sum;
            }
        }
        drop(sd);

        let mut result = Tensor::new(out_data, self.shape.clone(), self.requires_grad);
        if self.requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for r in 0..rows {
                    let mut dot = 0.0;
                    for j in 0..d {
                        dot += g[r * d + j];
                    }
                    for i in 0..d {
                        sgm[r * d + i] += g[r * d + i] - softmax_data[r * d + i] * dot;
                    }
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
        if self.requires_grad {
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
}
