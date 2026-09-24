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
use std::ops::{Deref, DerefMut};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

use crate::autograd::record;

/// 防除零小常数（div / log 反向用）
const EPS: f32 = 1e-8;

/// 线程安全的共享可变句柄：`Arc<Mutex<T>>` 的薄封装。
///
/// 保留了 `RefCell` 时代 `borrow()` / `borrow_mut()` 的调用习惯，
/// 全库调用点几乎零改动；内部是 `Arc<Mutex<T>>`，因此张量可以安全地
/// 被 rayon 并行闭包与多线程（数据并行、多卡训练）共享 —— 这正是旧实现
/// `Rc<RefCell<_>>` 做不到的（`Rc` 非 `Send`/`Sync`，把整张计算图锁死在单线程）。
///
/// **加锁纪律**：`Mutex` 不可重入，同一线程对同一个 `Shared` 嵌套加锁会死锁。
/// 因此所有「一个算子的两个输入可能是同一张量」的位点（`x + x`、`x @ x`、
/// `swiglu(x, x)`）必须先用 [`Shared::ptr_eq`] 判同一、只借一把锁
/// （见 binary / matmul / swiglu 的前向与反向）。
/// 另外，锁持有期间 panic 不会毒化后续访问：这里统一用
/// `PoisonError::into_inner` 忽略毒化，避免一次断言失败引发连锁误报。
pub struct Shared<T>(Arc<Mutex<T>>);

impl<T> Shared<T> {
    /// 用一个值新建共享句柄（对应原 `Rc::new(RefCell::new(v))`）
    pub fn new(v: T) -> Self {
        Shared(Arc::new(Mutex::new(v)))
    }

    /// 只读借用（对应 `RefCell::borrow`）。guard 存活期内持有锁。
    pub fn borrow(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// 可变借用（对应 `RefCell::borrow_mut`）。guard 存活期内持有锁。
    pub fn borrow_mut(&self) -> MutexGuard<'_, T> {
        self.0.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// 两个句柄是否指向同一份底层数据（对应 `Rc::ptr_eq` / `Arc::ptr_eq`）。
    /// 反向闭包用它区分「同一张量参与两次 → 单锁叠加」与「两张量 → 双锁」。
    pub fn ptr_eq(a: &Self, b: &Self) -> bool {
        Arc::ptr_eq(&a.0, &b.0)
    }

    /// 底层缓冲的裸指针标识（autograd 拓扑去重用，对应 `Rc::as_ptr`）。
    /// 只取地址做键，不加锁、不解引用。
    pub fn as_ptr(s: &Self) -> usize {
        Arc::as_ptr(&s.0) as usize
    }
}

impl<T> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Shared(Arc::clone(&self.0))
    }
}

// ==================== bf16 混合精度（批次 10b） ====================

/// 张量数据的物理存储类型。
///
/// - [`DType::F32`]：32 位浮点，算子直接锁借位读写，零转换开销；
/// - [`DType::Bf16`]：bf16 位模式（真 `u16` 存储），内存真实减半。
///
/// **decode-at-boundary 约定**：bf16 只存在于存储层。算子入口通过
/// [`Tensor::decode`] 拿到 f32 视图、算子内全程 f32 计算，
/// 出口经 [`Tensor::new_like`] 统一 encode 回 bf16 落盘。
/// 梯度缓冲（`grad`）恒为 f32，不参与该约定。
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DType {
    F32,
    Bf16,
}

/// f32 → bf16 位模式（round-to-nearest-even，IEEE 754 bfloat16）。
///
/// 取高 16 位，按第 17 位（舍入位）+ 低位是否有残留做「就近取偶」进位：
/// `+ 0x7FFF` 补足舍入阈值，`+ ((bits >> 16) & 1)` 在恰好半数时向偶数靠拢。
/// NaN 单独处理：直接截取高 16 位并强制尾数最高位为 1，保证转换后仍是
/// NaN（纯 RNE 进位可能把某些 NaN 尾数进成全 0 → 变成无穷大）。
fn f32_to_bf16(x: f32) -> u16 {
    if x.is_nan() {
        // 保留符号位与全 1 指数，尾数置非零
        return ((x.to_bits() >> 16) | 0x0040) as u16;
    }
    let bits = x.to_bits();
    let rounding_bias = ((bits >> 16) & 1) + 0x7FFF;
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

/// bf16 → f32：位模式左移 16 位补满尾数，指数/符号位原样保留。
/// ±0、±∞、NaN 都按位无损映射。
fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

/// 张量数据缓冲的**变体内容**（住在 [`Shared`] 单层锁槽里）。
///
/// 用枚举而不是「统一 u16 + tag」：f32 路径的锁借位语义完全不变；
/// bf16 路径只在算子边界付 decode（入口解码）/ encode（出口编码）成本。
///
/// **变体本身是共享状态**：槽是 `Shared<Buffer>`（`Arc<Mutex<Buffer>>`），
/// 所有克隆句柄看到同一个变体——[`Tensor::to_bf16`] 原地换变体后，
/// 模型里其余句柄同步生效（若变体直接放在每个 Tensor 克隆体上会脱节）。
#[derive(Clone)]
pub enum Buffer {
    F32(Vec<f32>),
    Bf16(Vec<u16>),
}

impl Buffer {
    /// 缓冲的存储类型（只看变体，本体已在锁槽内、无需再加锁）。
    pub(crate) fn dtype(&self) -> DType {
        match self {
            Buffer::F32(_) => DType::F32,
            Buffer::Bf16(_) => DType::Bf16,
        }
    }

    /// 缓冲元素个数（bf16 下也是逻辑元素数：每元素恰一个 `u16`）。
    #[allow(dead_code)]
    pub(crate) fn len(&self) -> usize {
        match self {
            Buffer::F32(v) => v.len(),
            Buffer::Bf16(v) => v.len(),
        }
    }
}

impl Shared<Buffer> {
    /// 只读解码视图（与 [`Tensor::decode`] 相同；反向闭包持 `data` 句柄时用）。
    pub fn decode(&self) -> DataGuard<'_> {
        let slot = self.borrow();
        match &*slot {
            Buffer::F32(_) => DataGuard::F32(slot),
            Buffer::Bf16(u) => {
                let v = u.par_iter().map(|&x| bf16_to_f32(x)).collect();
                drop(slot); // 解码完立即释放槽锁
                DataGuard::Bf16(v)
            }
        }
    }

    /// 可变解码视图（与 [`Tensor::decode_mut`] 相同；闭包持 `data` 句柄的写入点用）。
    pub fn decode_mut(&self) -> DataGuardMut<'_> {
        let slot = self.borrow_mut();
        // 先判变体再构造（借出 guard 后不能再摸 slot）
        if matches!(&*slot, Buffer::Bf16(_)) {
            let cache = match &*slot {
                Buffer::Bf16(u) => u.par_iter().map(|&x| bf16_to_f32(x)).collect(),
                Buffer::F32(_) => unreachable!("持锁期内变体不会改变"),
            };
            DataGuardMut::Bf16 { guard: slot, cache }
        } else {
            DataGuardMut::F32(slot)
        }
    }
}

/// 数据的**只读解码视图**（算子入口用，对应旧的 `data.borrow()`）。
///
/// - `F32`：直接持 `MutexGuard<Buffer>`（槽内就是 `Vec<f32>`），
///   零拷贝零转换，语义与旧 `borrow()` 一致；
/// - `Bf16`：锁内并行解码为 `Vec<f32>` 后立即释放锁，持有解码副本。
///
/// 实现 `Deref<Target = [f32]>`：索引、切片、`par_iter`、
/// `let sd_ref: &[f32] = &guard;`（解引用 coercion）等既有用法原样成立。
pub enum DataGuard<'a> {
    F32(MutexGuard<'a, Buffer>),
    Bf16(Vec<f32>),
}

impl Deref for DataGuard<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            DataGuard::F32(g) => match &**g {
                Buffer::F32(v) => v.as_slice(),
                Buffer::Bf16(_) => unreachable!("DataGuard::F32 持锁期内变体不会改变"),
            },
            DataGuard::Bf16(v) => v.as_slice(),
        }
    }
}

/// 数据的**可变解码视图**（优化器就地更新等写入点用，对应旧 `data.borrow_mut()`）。
///
/// - `F32`：直接可变锁借位，零转换；
/// - `Bf16`：借位时解码进 `cache`，全程按 f32 就地改，**`Drop` 时
///   统一并行 encode 回槽内 u16 存储**——写路径无需逐点手工编码
///   （master weights 语义：参数真存 bf16，更新在 f32 算完后截断回写）。
pub enum DataGuardMut<'a> {
    F32(MutexGuard<'a, Buffer>),
    Bf16 {
        guard: MutexGuard<'a, Buffer>,
        cache: Vec<f32>,
    },
}

impl Deref for DataGuardMut<'_> {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        match self {
            DataGuardMut::F32(g) => match &**g {
                Buffer::F32(v) => v.as_slice(),
                Buffer::Bf16(_) => unreachable!("DataGuardMut::F32 持锁期内变体不会改变"),
            },
            DataGuardMut::Bf16 { cache, .. } => cache.as_slice(),
        }
    }
}

impl DerefMut for DataGuardMut<'_> {
    fn deref_mut(&mut self) -> &mut [f32] {
        match self {
            DataGuardMut::F32(g) => match &mut **g {
                Buffer::F32(v) => v.as_mut_slice(),
                Buffer::Bf16(_) => unreachable!("DataGuardMut::F32 持锁期内变体不会改变"),
            },
            DataGuardMut::Bf16 { cache, .. } => cache.as_mut_slice(),
        }
    }
}

impl Drop for DataGuardMut<'_> {
    fn drop(&mut self) {
        if let DataGuardMut::Bf16 { guard, cache } = self {
            use rayon::prelude::*;
            match &mut **guard {
                Buffer::Bf16(raw) => {
                    // 并行 encode：bf16 舍入逐元素独立，块间无写冲突
                    raw.par_iter_mut()
                        .zip(cache.par_iter())
                        .for_each(|(slot, &v)| *slot = f32_to_bf16(v));
                }
                Buffer::F32(_) => unreachable!("DataGuardMut::Bf16 持锁期内变体不会改变"),
            }
        }
    }
}

// permute 的映射表缓存。
// 训练中同一形状每步反复出现（Q/K/V 拆头 [0,2,1,3]、Kᵀ [0,2,1]、合头 [0,2,1,3]），
// map 只依赖 (源形状, dims)，建一次后用 Arc 共享给前向/反向，避免每步重建几十万元素的下标表。
// 用 thread_local 而非 static：缓存按线程隔离，各训练/推理线程互不竞争。
thread_local! {
    static PERMUTE_MAP_CACHE: RefCell<HashMap<(Vec<usize>, Vec<usize>), Arc<Vec<usize>>>> =
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
// 既不向线性 tape 登记反向条目，也不分配梯度缓冲。
// 推理不需要反向，建图是纯开销 —— 单 token 前向时，分配 Arc + 闭包的代价
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

/// 张量结构体
///
/// 内部使用 `Shared<_>`（即 `Arc<Mutex<_>>`）共享可变数据：
/// - `Arc`  让多个张量可以"引用同一个底层数据"，且跨线程安全（`Send + Sync`）
/// - `Mutex` 允许在运行时互斥借用（对应原 `RefCell` 的运行时借用检查）
#[derive(Clone)]
pub struct Tensor {
    /// 数据缓冲（`Shared<Buffer>` = `Arc<Mutex<Buffer>>` 单层锁槽）。
    /// [`Buffer::F32`] 为默认（算子直接锁借位零转换）；
    /// [`Buffer::Bf16`] 为真 u16 bf16 存储（内存减半）——bf16 只存在于
    /// 存储层，算子入口经 [`Tensor::decode`] 解码成 f32 视图计算，
    /// 出口经 [`Tensor::new_like`] encode 落盘（decode-at-boundary 约定）。
    /// 变体住在共享槽里：原地换变体（[`Tensor::to_bf16`]）对所有克隆句柄可见。
    pub(crate) data: Shared<Buffer>,
    pub(crate) shape: Vec<usize>,
    /// 各维度在**物理缓冲** `data` 中的步长（以元素计）。
    ///
    /// 绝大多数张量是「行主序连续」的：strides 可由 shape 推出（见
    /// [`row_major_strides`]），此时逻辑序 == 物理序，与没有该字段时完全一致。
    ///
    /// 唯一的非连续来源是 [`Tensor::permute`]：视图与父张量共享同一块物理
    /// 缓冲（零拷贝），只换 shape/strides。因此：
    /// - 按逻辑索引读数据要用 [`Tensor::data`]（按 strides gather）或先
    ///   [`Tensor::contiguous`] 物化；
    /// - 直接 `data.borrow()` 线性读物理缓冲的算子，只接收连续输入——
    ///   permute 的消费点都在 reshape / matmul / sum_last_dim / flash_attention
    ///   入口处物化（连续时物化是零开销的自身克隆）。
    /// - 读数据一律用 [`Tensor::decode`]（f32/bf16 统一解码视图），
    ///   不要直接 match `data` 变体。
    ///
    /// `grad` 不带 strides：梯度缓冲永远按**逻辑行主序**全长分配，
    /// 反向闭包之间传递的都是逻辑序，视图的反向只需在入口处做一次映射。
    pub(crate) strides: Vec<usize>,
    pub(crate) grad: Shared<Vec<f32>>,
    /// 是否参与训练（`false` = 冻结）。与 `data` / `grad` 一样是**共享**的
    /// （`Arc<AtomicBool>`）：同一参数被克隆出多个句柄（优化器的参数表、`named_parameters`
    /// 的返回、各层的持有）后，冻结其中任何一个都等于冻结这一个参数本身。
    ///
    /// 若用普通 `bool`，`#[derive(Clone)]` 会把它逐句柄复制一份——在模型上冻结、
    /// 优化器手里那份却仍是 true，就会出现"以为冻结了、其实照旧更新"的隐性错误。
    /// 用原子 bool 而非加锁：读取在每步、每个算子的建图判断里都会发生，Relaxed 原子最便宜。
    pub(crate) requires_grad: Arc<AtomicBool>,
    // 计算图结构不在张量上：反向闭包统一登记到 autograd 的**线性 tape**
    // （见 `crate::autograd::record`），`Tensor` 只保留数据/形状/梯度本体。
}

/// 由形状推导行主序（C 风格）步长：最后一维步长 1，向左逐维乘该维长度。
fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        strides[d] = strides[d + 1] * shape[d + 1];
    }
    strides
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
/// - `Map(m)`：通用情况，查预建表（Arc 共享给反向闭包，不克隆）
enum SrcIdx {
    Ident,
    Mod(usize),
    Map(Arc<Vec<usize>>),
}

/// SrcIdx 的跨线程轻量视图（rayon 并行闭包用）。
/// 借用内部切片共享只读访问（`Copy` 视图），免去在并行闭包里逐元素克隆下标表。
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

/// 构造后缀因果掩码 `[t, t_total]`：query i 只能看到 key `j <= i + (t_total - t)`，
/// 其余位置为 -inf。
///
/// 分块前向（`flash_forward_cpu`）已在核内按同一规则屏蔽，CPU 路径不再需要这块
/// O(T×T_total) 的分配；只有 GPU 常驻路径与 `LLM_GPU_PROBE` 录制（内核按真实
/// buffer 消费掩码）才在这里真的物化一次。
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
pub fn causal_mask_data(t: usize, t_total: usize) -> Vec<f32> {
    let visible_before = t_total - t;
    let mut mask = vec![0.0f32; t * t_total];
    for i in 0..t {
        for j in 0..t_total {
            if j > i + visible_before {
                mask[i * t_total + j] = f32::NEG_INFINITY;
            }
        }
    }
    mask
}

/// Flash Attention 的 CPU 分块前向：按输出行（`B*H × T`）并行，行内按键块扫描 +
/// 在线 softmax，中间 P 从不落地。
///
/// 可见范围不物化掩码：query i 只算 `j < k_end`（`k_end = i + visible_before + 1`），
/// 与 `causal_mask_data` 生成的掩码逐位一致，被屏蔽的键根本不进打分与累加。
///
/// 行内维护运行最大值 m 与指数和 l：新块并入时先 `m ← max(m, m_blk)`，
/// 旧累计乘 `exp(m_old - m_new)` 重标定，再加新块的 `exp·V`，最后除以 l。
/// **漏掉重标定**会让前序块按 `exp(m_final - m_old)` 整体放大——该因子随注意力
/// 变尖锐指数增长，反向随之爆炸（历史 bug，回归测试见
/// `test_flash_attention_backward_matches_standard`）。
#[allow(clippy::too_many_arguments)]
fn flash_forward_cpu(
    q_scaled: &[f32],
    k: &[f32],
    v: &[f32],
    bh: usize,
    n_rep: usize, // GQA：Q 头 → 共享 KV 头的除数（MHA 时为 1）
    t: usize,
    t_total: usize,
    head_dim: usize,
    visible_before: usize,
    block_size: usize,
) -> Vec<f32> {
    let bs = block_size.max(1);
    let mut out = vec![0.0f32; bh * t * head_dim];
    out.par_chunks_mut(head_dim).enumerate().for_each(|(r, ochunk)| {
        let i = r % t;
        let k_end = (i + visible_before + 1).min(t_total);
        // K/V 是 [kv_bh, T_total, HD] 展平：本行 Q 头 hh = r/t 映射到共享 KV 头
        // hh/n_rep（与旧 repeat_kv 的「第 b 个 KV 头复制成 n_rep 份、头序 b*n_rep+r」
        // 互逆；n_rep=1 时退化为按行所属批组起算）
        let kv_base = ((r / t) / n_rep) * t_total * head_dim;
        let qrow = &q_scaled[r * head_dim..(r + 1) * head_dim];
        let mut m = f32::NEG_INFINITY;        // 行内运行最大值
        let mut l = 0.0f32;                   // 行内指数和（最终归一化分母）
        let mut acc = vec![0.0f32; head_dim]; // 未归一化的输出累加
        let mut sbuf = vec![0.0f32; bs];      // 块内打分暂存（每行的工作集 = O(block_size)）
        let mut ks = 0;
        while ks < k_end {
            let ke = (ks + bs).min(k_end);
            let n = ke - ks;
            // 1) 块内打分 S = Q'·Kᵀ（缩放已在 Q' 里）
            for jj in 0..n {
                let krow = &k[kv_base + (ks + jj) * head_dim..kv_base + (ks + jj + 1) * head_dim];
                let mut s = 0.0f32;
                for d in 0..head_dim {
                    s += qrow[d] * krow[d];
                }
                sbuf[jj] = s;
            }
            // 2) 在线 softmax：运行最大值更新后，旧累计整体乘 exp(m_old - m_new) 重标定
            let blk_max = sbuf[..n].iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let m_new = m.max(blk_max);
            let f = (m - m_new).exp(); // m = -inf（首块）时为 0，等价于从零开始累加
            l *= f;
            for a in acc.iter_mut() {
                *a *= f;
            }
            // 3) 累加新块的 exp·V
            for jj in 0..n {
                let e = (sbuf[jj] - m_new).exp();
                l += e;
                let vrow = &v[kv_base + (ks + jj) * head_dim..kv_base + (ks + jj + 1) * head_dim];
                for d in 0..head_dim {
                    acc[d] += e * vrow[d];
                }
            }
            m = m_new;
            ks = ke;
        }
        if l > 0.0 {
            for d in 0..head_dim {
                ochunk[d] = acc[d] / l;
            }
        }
    });
    out
}

/// GQA 物化的数据版：把 K/V 的 `kv_bh` 个头按 `n_rep` 连续复制成 `kv_bh * n_rep` 个头，
/// 头序与旧 `repeat_kv` 张量算子一致（输出头 i ← 源头 i / n_rep）。
///
/// CPU 分块核在 `flash_forward_cpu` 里用核内索引直接省掉这次物化；
/// 它只为两条路径保留语义基准：GPU 内核的核外展开，以及反向 matmul 链的输入展开。
pub(crate) fn repeat_flat(src: &[f32], kv_bh: usize, n_rep: usize) -> Vec<f32> {
    if n_rep == 1 {
        return src.to_vec();
    }
    let per = src.len() / kv_bh;
    let mut out = Vec::with_capacity(kv_bh * n_rep * per);
    for h in 0..kv_bh {
        let chunk = &src[h * per..(h + 1) * per];
        for _ in 0..n_rep {
            out.extend_from_slice(chunk);
        }
    }
    out
}

/// [`repeat_flat`] 的伴随：把展开后（`kv_bh * n_rep` 组）的梯度按 n_rep 个相邻副本
/// 求和折回 `kv_bh` 组（⟨repeat_flat(x), g⟩ == ⟨x, fold(g)⟩），即旧 `repeat_kv`
/// 反向的求和语义。dK/dV 在 GQA 下算在展开布局上，写父梯度前必须先折回。
fn fold_repeat_grad(src: &[f32], kv_bh: usize, n_rep: usize) -> Vec<f32> {
    let per = src.len() / (kv_bh * n_rep);
    let mut out = vec![0.0f32; kv_bh * per];
    for b in 0..kv_bh {
        for r in 0..n_rep {
            let seg = &src[(b * n_rep + r) * per..(b * n_rep + r + 1) * per];
            for (i, v) in seg.iter().enumerate() {
                out[b * per + i] += v;
            }
        }
    }
    out
}

/// 反向重算注意力概率 P：`softmax(Q'·Kᵀ)`，行内按后缀因果屏蔽（超出 k_end 的位置为 0）。
///
/// 分块前向不在前后向之间保留 P（O(T²) 不驻留）；反向先用矩阵乘算 scores，
/// 再在这里归一化——O(T²) 只作为反向的临时工作集存在（dP/dS 本来也是 O(T²)）。
#[allow(clippy::too_many_arguments)]
fn causal_softmax_cpu(
    x: &[f32],
    rows: usize,
    t: usize,
    t_total: usize,
    visible_before: usize,
) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * t_total];
    out.par_chunks_mut(t_total).enumerate().for_each(|(r, row)| {
        let i = r % t;
        let k_end = (i + visible_before + 1).min(t_total);
        let base = r * t_total;
        let mut mx = f32::NEG_INFINITY;
        for j in 0..k_end {
            mx = mx.max(x[base + j]);
        }
        let mut sum = 0.0f32;
        for j in 0..k_end {
            let e = (x[base + j] - mx).exp();
            row[j] = e;
            sum += e;
        }
        for j in 0..k_end {
            row[j] /= sum;
        }
        // k_end..t_total 保持 0：被屏蔽的位置不参与任何反向累加
    });
    out
}

/// flash attention 前向的中间状态：决定反向走「常驻显存」路径还是本地重算。
enum AttnCache {
    /// 分块前向路径：P 不落地，反向时按同一因果规则就地重算
    Cpu,
    /// 常驻显存路径：S 与 P 从未离开显存，反向也只回读 dQ/dK/dV
    #[cfg(feature = "gpu")]
    Gpu(Box<crate::gpu::AttnResident>),
}

/// 注意力前向的逐算子路径：S = Q'·Kᵀ → P = softmax(S + mask) → O = P·V。
/// 每个算子各自做 GPU/CPU 分流（`matmul_data` / `gpu::softmax_mask`），返回 (O, P)。
///
/// 现在只服务 `LLM_GPU_PROBE` 形状录制（普通回退已换成 `flash_forward_cpu` 分块核，
/// 见 `Tensor::flash_attention`）：录制模式下常驻路径整体关闭，这里把前向拆回
/// 逐算子，让 `matmul` 把真实训练形状记进探针。
#[cfg_attr(not(feature = "gpu"), allow(dead_code))]
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
) -> (Vec<f32>, Arc<Vec<f32>>) {
    let scores = matmul_data(q_scaled, k, t, head_dim, t_total, bh, false, true);
    let rows = bh * t;
    let m_n = mask.len();
    #[cfg(feature = "gpu")]
    let attn = crate::gpu::softmax_mask(&scores, mask, rows, t_total, m_n)
        .unwrap_or_else(|| masked_softmax_cpu(&scores, mask, rows, t_total, m_n));
    #[cfg(not(feature = "gpu"))]
    let attn = masked_softmax_cpu(&scores, mask, rows, t_total, m_n);
    let out = matmul_data(&attn, v, t, t_total, head_dim, bh, false, false);
    (out, Arc::new(attn))
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
    record(
        &result,
        vec![a.clone(), b.clone()],
        Arc::new(move || {
            let g = rg.borrow();
        let sd_b = sd.decode();
        let od_b = od.decode();
        // ∂a = g @ bᵀ、∂b = aᵀ @ g：GPU 内核支持按转置读物理矩阵，
        // 无需在 CPU 构造 52 万~210 万元素的转置矩阵（仅 CPU 回退时才物化）
        let da = matmul_data(&g, &od_b, m, n, k, batch, false, true);
        let db = matmul_data(&sd_b, &g, k, m, n, batch, true, false);
        drop(g);
        drop(sd_b);
        drop(od_b);
        // a、b 是同一张量（x@x）时梯度缓冲也是同一把锁——Mutex 不可重入，
        // 必须走单锁分支把 da/db 依次叠加进同一块缓冲
        if Shared::ptr_eq(&sg, &og) {
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
        }),
    );
}

// ==================== Tensor ====================

impl Tensor {
    // ---------- 构造 ----------
    pub(crate) fn new(data: Vec<f32>, shape: Vec<usize>, requires_grad: bool) -> Self {
        Self::new_with_dtype(data, shape, requires_grad, DType::F32)
    }

    /// 按指定 dtype 落盘的输出构造门：bf16 时在末尾把 f32 计算结果
    /// 统一 encode 成 u16 位模式存储（算子内的计算始终是 f32）。
    pub(crate) fn new_with_dtype(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        dtype: DType,
    ) -> Self {
        let len = data.len();
        // no_grad 模式下强制关闭求导
        let requires_grad = requires_grad && grad_enabled();
        let strides = row_major_strides(&shape);
        let buf = match dtype {
            DType::F32 => Buffer::F32(data),
            DType::Bf16 => {
                use rayon::prelude::*;
                Buffer::Bf16(data.into_par_iter().map(f32_to_bf16).collect())
            }
        };
        Tensor {
            data: Shared::new(buf),
            shape,
            strides,
            // 注意：grad 缓冲必须始终按 data 全长分配。反向闭包可能写入
            // **不需要梯度**的父节点（如 masked_softmax 的 mask：它不参与求导，
            // 但闭包仍会往它的 grad 里累加），缓冲长度不足会直接越界 panic。
            // grad 恒为 f32，不随数据 dtype 变化（见字段注释）。
            grad: Shared::new(vec![0.0; len]),
            requires_grad: Arc::new(AtomicBool::new(requires_grad)),
        }
    }

    /// 以「源张量」的 dtype 继承构造输出——算子出口统一 encode 的接线点。
    ///
    /// 算子内全程 f32 计算，构造输出时若源是 bf16 则末尾一次性
    /// encode 落盘，保持整条计算流的存储 dtype 与源一致。
    /// `requires_grad` 仍由调用方显式传入（与 dtype 无关）。
    pub(crate) fn new_like(
        &self,
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> Self {
        Self::new_with_dtype(data, shape, requires_grad, self.dtype())
    }

    /// 该张量在当前模式下是否需要自动微分。
    ///
    /// 与直接读字段 `requires_grad` 的区别：no_grad 模式下恒为 false。
    /// 算子在决定"要不要挂 backward 闭包"时必须用它，否则推理时
    /// 仍会拿着参数张量的 `requires_grad = true` 一路建出整张计算图。
    #[inline]
    pub(crate) fn req(&self) -> bool {
        self.requires_grad.load(Ordering::Relaxed) && grad_enabled()
    }

    /// 冻结 / 解冻本参数（`requires_grad`）。
    ///
    /// 标志是共享的（见字段注释），所以对任意一个克隆句柄调用都作用于同一个参数——
    /// LoRA 里"冻结主干"就是靠它：优化器手里的句柄与模型里的句柄指向同一个开关。
    pub fn set_requires_grad(&self, v: bool) {
        self.requires_grad.store(v, Ordering::Relaxed);
    }

    /// 本参数是否参与训练（no_grad 模式不影响这个"参数属性"，与 [`Tensor::req`] 区分）
    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad.load(Ordering::Relaxed)
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

    /// 共享一份已有数据缓冲构造张量（**不拷贝数据**）。
    ///
    /// 用于「只读消费、由持有方负责变更」的场景：KV cache 的 f32 路径用它把
    /// 整段历史以 O(1) 交给注意力前向，省掉解码每步 O(T·D) 的克隆（见
    /// [`crate::attention::KVCache::k`]）。叶子张量：requires_grad = false、
    /// 不挂 backward；grad 缓冲仍按 data 全长分配——注意力的反向闭包可能往
    /// 输入张量的 grad 里累加，长度不足会越界（见 [`Tensor::new`] 的注释）。
    pub(crate) fn shared(data: Shared<Buffer>, shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        let len = data.borrow().len();
        assert_eq!(len, numel, "共享数据长度 {len} 与形状 {shape:?} 要求的元素数 {numel} 不一致");
        let strides = row_major_strides(&shape);
        Tensor {
            data, // 零拷贝：直接共享调用方的缓冲句柄
            shape,
            strides,
            grad: Shared::new(vec![0.0; len]),
            requires_grad: Arc::new(AtomicBool::new(false)),
        }
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

    /// 数据的物理存储类型（`F32` / `Bf16`，读一次锁槽看变体）。
    pub fn dtype(&self) -> DType {
        self.data.borrow().dtype()
    }

    /// 只读解码视图（算子入口统一读数接口，对应旧 `data.borrow()`）。
    ///
    /// - `F32`：直接锁借位，零拷贝零转换；
    /// - `Bf16`：锁内并行解码成 `Vec<f32>` 后即释放锁，持有解码副本。
    ///
    /// 实现 `Deref<Target = [f32]>`，索引 / 切片 / `par_iter` /
    /// `let r: &[f32] = &guard;` 等既有用法原样成立。
    /// F32 变体存活期内持有数据锁——期间不要对同一张量再发起读写。
    pub fn decode(&self) -> DataGuard<'_> {
        self.data.decode()
    }

    /// 可变解码视图（优化器等写入点用，对应旧 `data.borrow_mut()`）。
    ///
    /// - `F32`：直接可变锁借位；
    /// - `Bf16`：借位时解码进 f32 缓存，调用方按 f32 就地改，
    ///   **`Drop` 时统一并行 encode 回 u16 存储**（master weights 语义：
    ///   参数真存 bf16，更新在 f32 精度下完成后截断回写）。
    pub fn decode_mut(&self) -> DataGuardMut<'_> {
        self.data.decode_mut()
    }

    /// 把数据缓冲**原地**转为 bf16 存储（内存减半），形状/步长/梯度不变。
    ///
    /// 变体住在共享槽（`Shared<Buffer>`）里，所以转换后**所有克隆句柄**
    /// （模型各层的参数句柄、优化器参数表）同步看到 bf16 存储。
    /// 供训练入口在建模完成后统一调用（`config.bf16` 开启时）；
    /// 测试里默认不调用，保持既有数值断言在 f32 精度下成立。
    pub fn to_bf16(&self) {
        let mut slot = self.data.borrow_mut();
        if matches!(&*slot, Buffer::Bf16(_)) {
            return; // 已是 bf16
        }
        let encoded = match &*slot {
            Buffer::F32(v) => {
                use rayon::prelude::*;
                Buffer::Bf16(v.par_iter().map(|&x| f32_to_bf16(x)).collect())
            }
            Buffer::Bf16(_) => unreachable!(),
        };
        *slot = encoded;
    }

    /// 返回**逻辑序**数据副本（兼容旧调用点，热路径建议用 `decode`）。
    /// 连续张量直接克隆缓冲；permute 视图按 strides gather，保证返回的
    /// 永远是行主序逻辑序，与形状对齐。
    pub fn data(&self) -> Vec<f32> {
        if self.is_contiguous() {
            return self.decode().to_vec();
        }
        // 非连续视图：按逻辑下标 → strides 偏移 gather（元素数不大，串行即可）
        let sd = self.decode();
        let rank = self.rank();
        let mut out = vec![0.0f32; self.numel()];
        let mut idx = vec![0usize; rank];
        for (i, slot) in out.iter_mut().enumerate() {
            let _ = i; // 逻辑下标由 idx 里程表维护
            let mut off = 0usize;
            for d in 0..rank {
                off += idx[d] * self.strides[d];
            }
            *slot = sd[off];
            for d in (0..rank).rev() {
                idx[d] += 1;
                if idx[d] < self.shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    }

    /// 只读借用底层数据，避免 O(N) 克隆。
    /// 适用于只读遍历（checkpoint 写入、采样取最后一行等）。
    /// 返回 `MutexGuard`：存活期内持有该张量数据锁——持有期间**不要**再对
    /// 同一张量发起读写（`Mutex` 不可重入，会死锁）。
    ///
    /// 注意：返回的是**物理缓冲**，permute 视图下物理序 ≠ 逻辑序；
    /// 需要逻辑序的调用点（如 checkpoint 存盘）必须先 [`Tensor::contiguous`]。
    pub fn data_ref(&self) -> DataGuard<'_> {
        self.decode()
    }

    /// 读取标量值（0 维张量专用，避免克隆整个 Vec）
    pub fn item(&self) -> f32 {
        assert_eq!(self.numel(), 1, "item() 只适用于单元素张量");
        // 单元素张量的唯一元素偏移恒为 0（无论 strides）
        self.decode()[0]
    }

    pub fn set_data(&self, new_data: Vec<f32>) {
        let mut d = self.decode_mut();
        assert_eq!(d.len(), new_data.len(), "set_data 长度不一致");
        d.copy_from_slice(&new_data); // bf16 时 Drop 会把整段 encode 回 u16
    }

    /// 读取梯度副本（测试/调试用；训练代码用 `p.grad.borrow()` 原位访问避免拷贝）
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
    ///
    /// `backward` 收到的是**本张量自己攒到的上游梯度**（autograd 按逆拓扑序执行，
    /// 轮到它时槽里已经攒齐）。这一步必须由构造函数代劳，不能像 [`Tensor::external`]
    /// 那样把句柄交给调用方：融合算子的反向是「按上游梯度为 1 算好、再整体乘回去」，
    /// 若把上游梯度当成恒等于 1，一旦真的有别的算子改了它（例如 AMP 的
    /// [`Tensor::mul_scalar`] 把 loss 乘上 2^16），整条链路的梯度比例就会被悄悄丢掉。
    pub fn external_scalar_loss(
        value: f32,
        parents: Vec<Tensor>,
        backward: impl Fn(f32) + Send + Sync + 'static,
    ) -> Tensor {
        let grad = Shared::new(vec![0.0f32]);
        let slot = grad.clone();
        let f: Box<dyn Fn() + Send + Sync> = Box::new(move || {
            // 上游为 0 说明这条支路没人需要（如被丢弃的那一次梯度累积），
            // 连常驻显存反向都不必白跑一趟
            let upstream = slot.borrow()[0];
            if upstream != 0.0 {
                backward(upstream);
            }
        });
        let result = Tensor {
            data: Shared::new(Buffer::F32(vec![value])),
            shape: vec![],
            strides: Vec::new(), // 0 维张量无步长
            grad,
            requires_grad: Arc::new(AtomicBool::new(true)),
        };
        record(&result, parents, Arc::from(f));
        result
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
    /// 调用点有两处：`model.rs` 的 gpu 常驻路径，以及 [`Tensor::matmul_frozen`]
    /// （LoRA 冻结主干时的部分反向）。不带 gpu feature 时前者是"死代码"，别删。
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn external(
        data: Vec<f32>,
        shape: Vec<usize>,
        parents: Vec<Tensor>,
        backward: impl FnOnce(Shared<Vec<f32>>) -> Box<dyn Fn() + Send + Sync>,
    ) -> Tensor {
        let grad = Shared::new(vec![0.0f32; data.len()]);
        let f = backward(grad.clone());
        let strides = row_major_strides(&shape);
        let result = Tensor {
            data: Shared::new(Buffer::F32(data)),
            shape,
            strides,
            grad,
            requires_grad: Arc::new(AtomicBool::new(true)),
        };
        record(&result, parents, Arc::from(f));
        result
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

    /// 张量在物理缓冲中是否行主序连续（可按 numel 线性读取）。
    /// 0 维张量恒为连续；唯一非连续来源是 [`Tensor::permute`] 视图。
    pub fn is_contiguous(&self) -> bool {
        self.strides == row_major_strides(&self.shape)
    }

    /// 物化为连续张量：已连续时返回自身克隆（零开销），
    /// 非连续（permute 视图）时按 strides gather 到新缓冲。
    /// 梯度按逻辑序 1:1 回传（视图与父张量的 grad 都是行主序全长）。
    pub fn contiguous(&self) -> Tensor {
        if self.is_contiguous() {
            return self.clone();
        }
        let total = self.numel();
        let rank = self.rank();
        assert!(rank <= 8, "contiguous 维度过多：{}", rank);
        let shape = self.shape.clone();
        let strides = self.strides.clone();
        let out_data = {
            let sd = self.decode();
            let sd_ref: &[f32] = &sd;
            let mut out_data = vec![0.0f32; total];
            // 并行按 4096 分块：每个块只分解一次起点下标，
            // 块内用「里程表进位」递增多维下标，避免逐元素做除法。
            out_data.par_chunks_mut(4096).enumerate().for_each(|(ci, chunk)| {
                let mut r = ci * 4096;
                let mut idx = [0usize; 8];
                for d in (0..rank).rev() {
                    idx[d] = r % shape[d];
                    r /= shape[d];
                }
                for slot in chunk.iter_mut() {
                    let mut off = 0usize;
                    for d in 0..rank {
                        off += idx[d] * strides[d];
                    }
                    *slot = sd_ref[off];
                    // 里程表 +1（逻辑行主序）
                    for d in (0..rank).rev() {
                        idx[d] += 1;
                        if idx[d] < shape[d] {
                            break;
                        }
                        idx[d] = 0;
                    }
                }
            });
            drop(sd);
            out_data
        };
        let result = self.new_like(out_data, shape, self.requires_grad.load(Ordering::Relaxed));
        if self.req() {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            record(&result, vec![self.clone()], Arc::new(move || {
                let g = rg.borrow();
                let g_ref: &[f32] = &g;
                let mut sgm = sg.borrow_mut();
                // 物化只改内存布局、不改逻辑元素顺序，梯度 1:1 逐元素累加。
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

    /// reshape：不改变元素顺序，梯度按 1:1 传回。
    /// 连续输入共享底层数据 Arc（不克隆 Vec），只分配新的梯度缓冲；
    /// 非连续输入（permute 视图）先物化再共享。
    pub fn reshape(&self, new_shape: Vec<usize>) -> Tensor {
        let numel: usize = new_shape.iter().product();
        assert_eq!(
            self.numel(),
            numel,
            "无法把 {:?} reshape 成 {:?}，元素总数不一致",
            self.shape,
            new_shape
        );
        // 非连续视图先物化：reshape 后共享的必须是行主序缓冲，
        // 否则新形状下的线性读取会拿到错位的数据。
        let src = self.contiguous();
        let requires_grad = src.req();
        let strides = row_major_strides(&new_shape);
        let result = Tensor {
            data: src.data.clone(), // Arc 共享，不克隆 Vec
            shape: new_shape,
            strides,
            grad: Shared::new(vec![0.0; numel]),
            requires_grad: Arc::new(AtomicBool::new(requires_grad)),
        };
        if requires_grad {
            let rg = result.grad.clone();
            let sg = src.grad.clone();
            record(&result, vec![src], Arc::new(move || {
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
    ///
    /// **零拷贝视图**：与源张量共享物理缓冲，只重算 strides
    /// （`new_strides[i] = self.strides[dims[i]]`）并新建梯度缓冲。
    /// 线性读算子（matmul/sum_last_dim/flash_attention/reshape 等）在入口
    /// 检测非连续并自动 [`Tensor::contiguous`] 物化。
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
        let requires_grad = self.requires_grad.load(Ordering::Relaxed);

        // 零拷贝：共享物理缓冲，按排列取源 strides 得到新视图布局
        let new_strides: Vec<usize> = dims.iter().map(|&d| self.strides[d]).collect();
        let result = Tensor {
            data: self.data.clone(), // Arc 共享，不克隆 Vec
            shape: new_shape.clone(),
            strides: new_strides,
            grad: Shared::new(vec![0.0; total]),
            requires_grad: Arc::new(AtomicBool::new(requires_grad)),
        };
        if self.req() {
            // 反解 permute 的逆映射：inv[perm[i]] = i
            let mut inv = vec![0usize; self.rank()];
            for (i, &d) in dims.iter().enumerate() {
                inv[d] = i;
            }
            // 反向闭包持有的 map 是 out_flat -> src_flat（**逻辑行主序**下的
            // 置换，与物理 strides 无关）：视图的 grad 与父张量的 grad 都是
            // 逻辑行主序全长缓冲，所以原地查表回填依然成立。
            // 热路径：训练中每步都要对 Q/K/V 拆头、合头做多次 permute，
            // 缓存置换表一次、Arc 共享即可。
            // （张量最多 4 维，固定栈数组即可，超过 8 维防御性断言）
            assert!(self.rank() <= 8, "permute 维度过多：{}", self.rank());
            let map: Arc<Vec<usize>> = PERMUTE_MAP_CACHE.with(|c| {
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
                        Arc::new(map)
                    })
                    .clone()
            });
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            record(&result, vec![self.clone()], Arc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                let g_ref: &[f32] = &g;
                let mb: &[usize] = &map;
                // map 是 out→src 的置换（双射）：先串行建逆映射 src→out（纯整数
                // 赋值，远比浮点循环便宜），再按梯度槽位分块并行回填——双射保证
                // 每个槽位只被一个输出索引命中，块间无写冲突。
                let mut inv = vec![0usize; mb.len()];
                for (of, &sf) in mb.iter().enumerate() {
                    inv[sf] = of;
                }
                let inv_ref: &[usize] = &inv;
                sgm.par_chunks_mut(4096).enumerate().for_each(|(ci, ch)| {
                    let base = ci * 4096;
                    for (j, s) in ch.iter_mut().enumerate() {
                        *s += g_ref[inv_ref[base + j]];
                    }
                });
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
                    None => SrcIdx::Map(Arc::new(broadcast_map(&target, &self.shape))),
                }
            };
            let b_src = if other.shape == target {
                SrcIdx::Ident
            } else {
                match other.suffix_mod(&target) {
                    Some(n) => SrcIdx::Mod(n),
                    None => SrcIdx::Map(Arc::new(broadcast_map(&target, &other.shape))),
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

    /// 逐元素乘法。
    ///
    /// 主训练路径上的乘法多走融合算子（`matmul` / `swiglu` / `layer_norm` 等内部一次算完），
    /// 但 MoE 辅助损失、分布式对齐等路径仍在用这个方法版。它同时是**分步参考实现**：
    /// 测试用 `mul` / `div` / `sum_last_dim` 串出"手写版"公式，再与融合算子的输出比对，
    /// 融合实现一旦写错（比如反向系数符号反了）就会被这些测试抓住。
    pub fn mul(&self, other: &Tensor) -> Tensor {
        // ∂c/∂a = b，∂c/∂b = a
        self.binary(other, |a, b| a * b, |a, b| (b, a))
    }

    /// 逐元素除法。生产路径无调用点，作为分步参考实现保留给测试
    /// （理由同 [`Tensor::mul`]：用定义式公式与融合算子比对数值）。
    #[allow(dead_code)]
    pub fn div(&self, other: &Tensor) -> Tensor {
        // 反向必须与前向 `a / b` 严格对应（∂/∂a = 1/b，∂/∂b = -a/b²）。
        // 给 b 加 `1e-8` 会让梯度与自己的前向不一致——宁可得到 ±inf，
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
        // back 会被 move 进反向闭包（类型是 Arc<dyn Fn() + Send + Sync>），故需 Send
        back: impl Fn(f32, f32) -> (f32, f32) + Sync + Send + 'static,
    ) -> Tensor {
        // 非连续（permute 视图）先物化，下文按行主序线性读（含反向闭包按缓冲读）
        let lhs = self.contiguous();
        let rhs = other.contiguous();
        let (target_shape, a_src, b_src) = lhs.broadcast_plan(&rhs);
        // 同一张量参与两次（如 x + x）时两把锁是同一个 Mutex——不可重入会死锁，
        // 判同一后只借一把锁、两个视图都指向它（shared borrow 可安全别名）
        let same_data = Shared::ptr_eq(&lhs.data, &rhs.data);
        let sa = lhs.decode();
        let sb = if same_data { None } else { Some(rhs.decode()) };
        let sa_ref: &[f32] = &sa;
        let sb_ref: &[f32] = sb.as_deref().unwrap_or(&sa);
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

        let requires = lhs.req() || rhs.req();
        // 输出继承左操作数的 dtype（bf16 流下逐元素算子保持存储精度一致）
        let result = lhs.new_like(out_data, target_shape, requires);
        if requires {
            let rg = result.grad.clone();
            let sg = lhs.grad.clone();
            let og = rhs.grad.clone();
            let sd = lhs.data.clone();
            let od = rhs.data.clone();
            // 广播索引方式随闭包带走（Mod 不占内存，Map 是 Arc 共享，无需克隆 4-16MB map）
            let a_src_c = a_src;
            let b_src_c = b_src;
            let same_shape = lhs.shape == rhs.shape;
            record(&result, vec![lhs, rhs], Arc::new(move || {
                let g = rg.borrow();
                // 同一张量参与两次（如 x*x）时 sd/od 是同一把 Mutex——判同一后只借一把锁，
                // 两个只读视图都指向它（shared borrow 可安全别名）
                let same_data = Shared::ptr_eq(&sd, &od);
                let sd_b = sd.decode();
                let od_b = if same_data { None } else { Some(od.decode()) };
                let g_ref: &[f32] = &g;
                let sd_ref: &[f32] = &sd_b;
                let od_ref: &[f32] = od_b.as_deref().unwrap_or(&sd_b);
                if same_shape {
                    // 形状相同 ⇒ 两侧索引都是恒等（见 broadcast_plan），可以按块并行。
                    // 残差相加、x*x 这类占反向的大头，串行时单步 ~0.5s。
                    if Shared::ptr_eq(&sg, &og) {
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
                    // 广播分支：Mod/Map 的目标是多对一（如 t % numel），不能按
                    // 槽位分块直写。形状==目标的一侧槽位与 t 对齐，可直接分块并行；
                    // 另一侧先在块内收集 (槽位, 值)，再按块序串行合并——合并顺序
                    // 等于 t 递增顺序，与原串行累加逐位一致。
                    let av = SrcIdxView::from(&a_src_c);
                    let bv = SrcIdxView::from(&b_src_c);
                    let a_aligned = matches!(a_src_c, SrcIdx::Ident);
                    let b_aligned = matches!(b_src_c, SrcIdx::Ident);
                    let mut sgm = sg.borrow_mut();
                    let mut ogm = og.borrow_mut();
                    match (a_aligned, b_aligned) {
                        (true, false) => {
                            // a 恒等直写；b 多对一，块内收集后合并进 ogm
                            let partials: Vec<Vec<(usize, f32)>> = sgm
                                .par_chunks_mut(4096)
                                .enumerate()
                                .map(|(ci, ch)| {
                                    let base = ci * 4096;
                                    let mut local = Vec::with_capacity(ch.len());
                                    for (j, s) in ch.iter_mut().enumerate() {
                                        let t = base + j;
                                        let ib = bv.idx(t);
                                        let (da, db) = back(sd_ref[t], od_ref[ib]);
                                        *s += g_ref[t] * da;
                                        local.push((ib, g_ref[t] * db));
                                    }
                                    local
                                })
                                .collect();
                            for part in partials {
                                for (i, v) in part {
                                    ogm[i] += v;
                                }
                            }
                        }
                        (false, true) => {
                            // b 恒等直写；a 多对一，块内收集后合并进 sgm
                            let partials: Vec<Vec<(usize, f32)>> = ogm
                                .par_chunks_mut(4096)
                                .enumerate()
                                .map(|(ci, ch)| {
                                    let base = ci * 4096;
                                    let mut local = Vec::with_capacity(ch.len());
                                    for (j, s) in ch.iter_mut().enumerate() {
                                        let t = base + j;
                                        let ia = av.idx(t);
                                        let (da, db) = back(sd_ref[ia], od_ref[t]);
                                        *s += g_ref[t] * db;
                                        local.push((ia, g_ref[t] * da));
                                    }
                                    local
                                })
                                .collect();
                            for part in partials {
                                for (i, v) in part {
                                    sgm[i] += v;
                                }
                            }
                        }
                        _ => {
                            // 两侧都多对一：并行收集两侧 (槽位, 值) 后按块序合并
                            let (pa, pb): (
                                Vec<Vec<(usize, f32)>>,
                                Vec<Vec<(usize, f32)>>,
                            ) = g_ref
                                .par_chunks(4096)
                                .enumerate()
                                .map(|(ci, gc)| {
                                    let base = ci * 4096;
                                    let mut la = Vec::with_capacity(gc.len());
                                    let mut lb = Vec::with_capacity(gc.len());
                                    for (k, &gv) in gc.iter().enumerate() {
                                        let t = base + k;
                                        let ia = av.idx(t);
                                        let ib = bv.idx(t);
                                        let (da, db) = back(sd_ref[ia], od_ref[ib]);
                                        la.push((ia, gv * da));
                                        lb.push((ib, gv * db));
                                    }
                                    (la, lb)
                                })
                                .collect();
                            for part in pa {
                                for (i, v) in part {
                                    sgm[i] += v;
                                }
                            }
                            for part in pb {
                                for (i, v) in part {
                                    ogm[i] += v;
                                }
                            }
                        }
                    }
                }
            }));
        }
        result
    }

    // ---------- 标量运算 ----------

    /// 逐元素加标量：c = x + s，∂x = g（标量是常量，不参与梯度）
    ///
    /// 与 [`Tensor::mul_scalar`] 对称。主路径上需要加常量的地方（如 eps）都并入融合算子，
    /// 非测试构建里没有调用点；测试在分步参考实现里用它补齐公式。
    #[allow(dead_code)]
    pub fn add_scalar(&self, scalar: f32) -> Tensor {
        // 非连续先物化，下文按行主序线性读
        let x = self.contiguous();
        let data = x.decode().iter().map(|a| a + scalar).collect();
        let result = x.new_like(data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                let g_ref: &[f32] = &g;
                // 加标量的反向是恒等映射：整段梯度逐位搬运，按元素并行
                sgm.par_iter_mut()
                    .zip(g_ref.par_iter())
                    .for_each(|(s, &gv)| *s += gv);
            }));
        }
        result
    }

    pub fn mul_scalar(&self, scalar: f32) -> Tensor {
        // 非连续先物化，下文按行主序线性读
        let x = self.contiguous();
        let data: Vec<f32> = x
            .decode()
            .par_iter()
            .map(|&a| a * scalar)
            .collect();
        let result = x.new_like(data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                let g_ref: &[f32] = &g;
                sgm.par_iter_mut()
                    .zip(g_ref.par_iter())
                    .for_each(|(s, &gv)| *s += gv * scalar);
            }));
        }
        result
    }

    // ---------- 激活函数（一元运算） ----------

    /// 取负：c = -x，∂x = -g
    ///
    /// 逐元素算子集的一员（与 [`Tensor::sub`] 一起构成"求负"语义）。当前训练/推理路径
    /// 没有调用点，因为需要取负的地方都是与别的运算合并成一步算的；保留它是为了让算子集
    /// 在语义上闭合（有 `sub` 就该有 `neg`），代价只是一个 `allow(dead_code)`。
    #[allow(dead_code)]
    pub fn neg(&self) -> Tensor {
        // 非连续先物化，下文按行主序线性读
        let x = self.contiguous();
        let data = x.decode().iter().map(|a| -a).collect();
        let result = x.new_like(data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            record(&result, vec![x], Arc::new(move || {
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
        // 非连续先物化，下文按行主序线性读
        let x = self.contiguous();
        let sd = x.decode();
        let data = sd.iter().map(|&a| a.max(0.0)).collect();
        let mask: Vec<f32> = sd
            .iter()
            .map(|&a| if a > 0.0 { 1.0 } else { 0.0 })
            .collect();
        drop(sd);
        let result = x.new_like(data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            record(&result, vec![x], Arc::new(move || {
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
        // 非连续先物化，下文按行主序线性读
        let x = self.contiguous();
        let sd = x.decode();
        let data: Vec<f32> = sd.iter().map(|&a| a.tanh()).collect();
        drop(sd);
        let result = x.new_like(data.clone(), x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            record(&result, vec![x], Arc::new(move || {
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
    /// 这是现代 LLM使用的激活函数。
    /// 反向：dGELU/dx = 0.5(1+t) + 0.5x(1-t²)·da/dx，其中 a = √(2/π)(x+0.044715x³)，t = tanh(a)
    pub fn gelu(&self) -> Tensor {
        const SQRT_2_PI: f32 = 0.797_884_560_8; // sqrt(2/π)
        const COEF: f32 = 0.044_715;
        // 非连续先物化，前后向均按行主序线性读（反向闭包按缓冲读 sd）
        let x = self.contiguous();
        let sd = x.decode();
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
        let result = x.new_like(data.clone(), x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let sd = x.data.clone();
            let tv = t_vals;
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let x_b = sd.decode();
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
        // 非连续先物化，前后向均按行主序线性读（反向闭包按缓冲读 xd/gd）
        let x = self.contiguous();
        let g_in = gate.contiguous();
        // swiglu(x, x)：判同一后只借一把锁（Mutex 不可重入），两个视图都指向它
        let same_data = Shared::ptr_eq(&x.data, &g_in.data);
        let sd = x.decode();
        let gd = if same_data { None } else { Some(g_in.decode()) };
        let gd_ref_owned: &[f32] = gd.as_deref().unwrap_or(&sd);
        let len = sd.len();
        let mut out_data = vec![0.0f32; len];
        let mut silu_vals = vec![0.0f32; len]; // SiLU(x) = x * sigmoid(x)
        let mut sig_vals = vec![0.0f32; len]; // sigmoid(x)
        let sd_ref: &[f32] = &sd;
        let gd_ref: &[f32] = gd_ref_owned;
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

        let requires = x.req() || g_in.req();
        let result = x.new_like(out_data, x.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sx = x.grad.clone();
            let sg = g_in.grad.clone();
            let xd = x.data.clone();
            let gd = g_in.data.clone();
            let sv = silu_vals;
            let sig = sig_vals;
            record(&result, vec![x, g_in], Arc::new(move || {
                // 梯度缓冲判同一：swiglu(x, x) 时 self 与 gate 共用同一把锁，
                // Mutex 不可重入——同缓冲必须单锁、两份贡献依次叠加进同一块内存
                let same_grad = Shared::ptr_eq(&sx, &sg);
                let g = rg.borrow();
                let x_b_guard = xd.decode();
                let g_b_guard = if Shared::ptr_eq(&xd, &gd) {
                    None
                } else {
                    Some(gd.decode())
                };
                let x_b: &[f32] = &x_b_guard;
                let g_b: &[f32] = g_b_guard.as_deref().unwrap_or(&x_b_guard);
                let len = g.len();
                let chunk = 4096;
                if same_grad {
                    // 同一缓冲：∂gate = g·silu(x)，∂x = g·gate·dsigmoid，
                    // 两笔都累加进同一块内存，合成一次 +=
                    let mut gx = sx.borrow_mut();
                    for start in (0..len).step_by(chunk) {
                        let end = (start + chunk).min(len);
                        for i in start..end {
                            let sig_v = sig[i];
                            let silu_v = sv[i];
                            let dsig = sig_v * (1.0 + x_b[i] * (1.0 - sig_v));
                            gx[i] += g[i] * silu_v + g[i] * g_b[i] * dsig;
                        }
                    }
                } else {
                    let mut gx = sx.borrow_mut();
                    let mut gg = sg.borrow_mut();
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
                }
            }));
        }
        result
    }

    /// log：c = ln(x)，∂x = g / x（cross_entropy 已改用 log_softmax_last_dim，仅测试使用）
    ///
    /// 前向以 [`EPS`] 为下界再取 ln：与反向的 `g / max(x, EPS)` 保持一致。
    /// 否则 x=0 时前向是 -inf、反向是 g/EPS，两个不一致的奇点相遇容易滚出 NaN。
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn log(&self) -> Tensor {
        // 非连续先物化，前后向均按行主序线性读（反向闭包按缓冲读 sd）
        let x = self.contiguous();
        let sd = x.decode();
        let data: Vec<f32> = sd.iter().map(|&a| a.max(EPS).ln()).collect();
        drop(sd);
        let result = x.new_like(data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let sd = x.data.clone();
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.decode();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] / sd_b[i].max(EPS);
                }
            }));
        }
        result
    }

    /// 幂：c = x^p，∂x = g * p * x^(p-1)
    pub fn pow(&self, p: f32) -> Tensor {
        // 非连续先物化，前后向均按行主序线性读（反向闭包按缓冲读 sd）
        let x = self.contiguous();
        let sd = x.decode();
        let data: Vec<f32> = sd.iter().map(|&a| a.powf(p)).collect();
        drop(sd);
        let result = x.new_like(data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let sd = x.data.clone();
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.decode();
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
        // 入口物化：permute 视图（如权重 transpose）先 gather 成连续缓冲，
        // matmul_data 按行主序线性读；连续时 contiguous() 只是克隆句柄、零开销。
        let a = self.contiguous();
        let b = other.contiguous();
        if a.rank() == 2 {
            return a.matmul_2d(&b);
        }
        // 3D 批量
        assert_eq!(a.shape[0], b.shape[0], "批量维度必须一致");
        let (m, k1) = (a.shape[1], a.shape[2]);
        let (k2, n) = (b.shape[1], b.shape[2]);
        assert_eq!(k1, k2, "矩阵乘法维度不匹配");

        // x@x：判同一后只借一把锁（Mutex 不可重入），两个视图都指向它
        let same_data = Shared::ptr_eq(&a.data, &b.data);
        let sd = a.decode();
        let od = if same_data { None } else { Some(b.decode()) };
        let od_ref: &[f32] = od.as_deref().unwrap_or(&sd);
        let out_data = matmul_data(&sd, od_ref, m, k1, n, a.shape[0], false, false);
        drop(sd);
        drop(od);

        let requires = a.req() || b.req();
        let mut result = a.new_like(out_data, vec![a.shape[0], m, n], requires);
        if requires {
            matmul_backward(&mut result, &a, &b, m, k1, n, a.shape[0]);
        }
        result
    }

    fn matmul_2d(&self, other: &Tensor) -> Tensor {
        // 入口物化（matmul 已物化过，这里承接直接调用者）
        let a = self.contiguous();
        let b = other.contiguous();
        let (m, k1) = (a.shape[0], a.shape[1]);
        let (k2, n) = (b.shape[0], b.shape[1]);
        assert_eq!(
            k1, k2,
            "矩阵乘法维度不匹配：{:?} x {:?}",
            a.shape, b.shape
        );

        // x@x：判同一后只借一把锁（Mutex 不可重入），两个视图都指向它
        let same_data = Shared::ptr_eq(&a.data, &b.data);
        let sd = a.decode();
        let od = if same_data { None } else { Some(b.decode()) };
        let od_ref: &[f32] = od.as_deref().unwrap_or(&sd);
        let out_data = matmul_data(&sd, od_ref, m, k1, n, 1, false, false);
        drop(sd);
        drop(od);

        let requires = a.req() || b.req();
        let mut result = a.new_like(out_data, vec![m, n], requires);
        if requires {
            matmul_backward(&mut result, &a, &b, m, k1, n, 1);
        }
        result
    }

    /// `y = x @ W`，其中 `W` 是**冻结**权重：反向只求 `dx = g @ Wᵀ`，**不求 `dW`**。
    ///
    /// 与 [`Tensor::matmul`] 的区别只在反向。通用 `matmul` 会把 `dW = xᵀ @ g` 一起算出来
    /// 并写进 `W.grad`；对冻结权重（LoRA 要冻结的预训练主干）这笔计算纯属浪费——
    /// 它和 `dx` 的那次矩阵乘同量级，占了反向矩阵乘计算量的一半。挡着不算，
    /// "冻结"才不只体现在"优化器不更新它"上，而是真的省下计算。
    ///
    /// 只支持 2D × 2D：调用方（[`crate::layers::Linear`]）会先把 3D 输入展平成 2D。
    /// `W` 不参与求导这件事由调用方保证，这里不看 `W.requires_grad`。
    pub fn matmul_frozen(&self, w: &Tensor) -> Tensor {
        // 入口物化（permute 视图先 gather 成连续缓冲）
        let a = self.contiguous();
        let w = w.contiguous();
        assert_eq!(a.rank(), 2, "matmul_frozen 的左操作数必须为 2D");
        assert_eq!(w.rank(), 2, "matmul_frozen 的右操作数必须为 2D");
        let (m, k1) = (a.shape[0], a.shape[1]);
        let (k2, n) = (w.shape[0], w.shape[1]);
        assert_eq!(
            k1, k2,
            "矩阵乘法维度不匹配：{:?} x {:?}",
            a.shape, w.shape
        );

        // x@W：W 是独立权重参数，正常不会与 x 同一张量；仍判同一防 Mutex 死锁
        let same_data = Shared::ptr_eq(&a.data, &w.data);
        let sd = a.decode();
        let wd = if same_data { None } else { Some(w.decode()) };
        let wd_ref: &[f32] = wd.as_deref().unwrap_or(&sd);
        let out_data = matmul_data(&sd, wd_ref, m, k1, n, 1, false, false);
        drop(sd);
        drop(wd);

        // 左操作数不需要梯度（推理，或输入本身是常量）→ 反向整段省掉
        if !a.req() {
            // 出口继承左操作数 dtype：bf16 流下同样走 encode 落盘
            return a.new_like(out_data, vec![m, n], false);
        }
        let x_bwd = a.clone();
        let w_bwd = w.clone();
        let out_dtype = a.dtype();
        let result = Tensor::external(out_data, vec![m, n], vec![a], move |grad| {
            Box::new(move || {
                let g = grad.borrow();
                let wd = w_bwd.decode();
                // dx = g @ Wᵀ：g 是 [m, n]，W 是 [k, n]（按 n 转置读）→ [m, k]
                let dx = matmul_data(&g, &wd, m, n, k1, 1, false, true);
                x_bwd.accumulate_grad(&dx, 1.0);
            })
        });
        // external 固定按 f32 落盘，源是 bf16 时补一次原地 encode 对齐 dtype
        if matches!(out_dtype, DType::Bf16) {
            result.to_bf16();
        }
        result
    }

    // ---------- 归约运算 ----------

    /// 求和成标量，梯度均匀传给每个元素
    pub fn sum(&self) -> Tensor {
        // 入口物化：非连续视图先 gather（连续时零开销），下文按缓冲线性求和
        let a = self.contiguous();
        // 并行归约：各线程算部分和再合并（大张量求和不再占满一个核）
        let total: f32 = a.decode().par_iter().sum();
        let result = a.new_like(vec![total], vec![], a.requires_grad.load(Ordering::Relaxed));
        if a.req() {
            let rg = result.grad.clone();
            let sg = a.grad.clone();
            record(&result, vec![a], Arc::new(move || {
                let g = rg.borrow()[0];
                let mut sgm = sg.borrow_mut();
                // 并行：标量梯度均匀摊给每个元素，各写各的槽、无冲突
                sgm.par_iter_mut().for_each(|v| *v += g);
            }));
        }
        result
    }

    /// 沿最后一维求和，**保持维度**：[..., D] -> [..., 1]
    /// 反向：梯度广播回最后一维
    ///
    /// 主路径上需要求和的地方多用更专门的融合算子，但 MoE 辅助损失等路径仍直接调用它。
    /// 它也是**分步参考实现**：测试用它把归一化、注意力打分等公式按定义一步步写出来，
    /// 再与融合算子比对数值（见 [`Tensor::mul`] 的说明）。
    pub fn sum_last_dim(&self) -> Tensor {
        assert!(self.rank() >= 1, "sum_last_dim 需要至少 1 维");
        // 非连续（permute 视图）先物化，下文按行主序线性读
        let a = self.contiguous();
        let (pre, d) = (
            a.numel() / a.shape[a.rank() - 1],
            a.shape[a.rank() - 1],
        );
        let sd = a.decode();
        let sd_ref: &[f32] = &sd;
        let mut out_data = vec![0.0f32; pre];
        // 并行：每行（长度 d）独立求和，行间无依赖；行内求和顺序不变
        out_data
            .par_iter_mut()
            .enumerate()
            .for_each(|(p, slot)| {
                *slot = sd_ref[p * d..p * d + d].iter().sum();
            });
        drop(sd);
        let mut new_shape = a.shape.clone();
        *new_shape.last_mut().unwrap() = 1;

        let result = a.new_like(out_data, new_shape, a.requires_grad.load(Ordering::Relaxed));
        if a.req() {
            let rg = result.grad.clone();
            let sg = a.grad.clone();
            record(&result, vec![a], Arc::new(move || {
                let g = rg.borrow();
                let g_ref: &[f32] = &g;
                let mut sgm = sg.borrow_mut();
                // 并行：按行切分，每行只写自己那 d 个元素，行间无依赖
                sgm.par_chunks_mut(d)
                    .zip(g_ref.par_iter())
                    .for_each(|(row, &gv)| {
                        for v in row.iter_mut() {
                            *v += gv;
                        }
                    });
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
        // 非连续先物化，下文按行主序线性读
        let a = self.contiguous();
        let (rows, d) = (
            a.numel() / a.shape[a.rank() - 1],
            a.shape[a.rank() - 1],
        );
        let sd = a.decode();
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

        let result = a.new_like(out_data.clone(), a.shape.clone(), a.requires_grad.load(Ordering::Relaxed));
        if a.req() {
            let rg = result.grad.clone();
            let sg = a.grad.clone();
            let (_rows, d) = (rows, d);
            record(&result, vec![a], Arc::new(move || {
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

    /// 沿最后一维做 logsumexp：`c = log(Σ_j e^{x_j})`，输出形状的末维改成 1。
    ///
    /// 数值稳定：先减行内最大值再 exp，最后把 max 加回来——logits 多大都不会溢出。
    /// 反向公式就是 softmax：`∂x_j = g · softmax(x)_j`（减 max 是常数平移，
    /// 不改变 `e^{x_j}/Σe^{x_i}` 这个比值）。
    ///
    /// 用途是 MoE 的 router z-loss（第 32 课）：`L_z = mean(logsumexp(logits)²)`，
    /// 惩罚过大的门控 logits，防止路由器 softmax 过尖后饱和（PaLM 等都加了它）。
    /// 不要用 `log(exp(x).sum())` 裸拼——logits 稍大 exp 就直接溢出成 inf。
    pub fn logsumexp_last_dim(&self) -> Tensor {
        assert!(self.rank() >= 1, "logsumexp_last_dim 需要至少 1 维");
        // 非连续先物化，下文按行主序线性读
        let a = self.contiguous();
        let (pre, d) = (
            a.numel() / a.shape[a.rank() - 1],
            a.shape[a.rank() - 1],
        );
        let sd = a.decode();
        let mut out_data = vec![0.0f32; pre];
        let mut sm = vec![0.0f32; pre * d]; // 归一化前是 exp，归一化后是 softmax（反向要用）
        for p in 0..pre {
            let row = &sd[p * d..(p + 1) * d];
            let mut maxv = f32::NEG_INFINITY;
            for &v in row {
                maxv = maxv.max(v);
            }
            let mut sum = 0.0;
            for j in 0..d {
                let e = (row[j] - maxv).exp();
                sm[p * d + j] = e;
                sum += e;
            }
            out_data[p] = maxv + sum.ln(); // sum ≥ 1（最大值那项恰好是 e⁰），ln 安全
            for j in 0..d {
                sm[p * d + j] /= sum;
            }
        }
        drop(sd);
        let mut new_shape = a.shape.clone();
        *new_shape.last_mut().unwrap() = 1;

        let result = a.new_like(out_data, new_shape, a.requires_grad.load(Ordering::Relaxed));
        if a.req() {
            let rg = result.grad.clone();
            let sg = a.grad.clone();
            record(&result, vec![a], Arc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for p in 0..pre {
                    for j in 0..d {
                        sgm[p * d + j] += g[p] * sm[p * d + j];
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
        // 非连续先物化，前后向均按行主序线性读（反向闭包还按缓冲读 xd）
        let x = self.contiguous();
        let d = *x.shape.last().unwrap();
        assert_eq!(gamma.rank(), 1, "LayerNorm 的 γ 必须是一维");
        assert_eq!(gamma.shape, beta.shape, "LayerNorm 的 γ/β 形状必须一致");
        assert_eq!(gamma.shape[0], d, "LayerNorm 的 γ/β 长度必须等于输入最后一维");
        let rows = x.numel() / d;
        let sd = x.decode();
        let gv = gamma.decode();
        let bv = beta.decode();
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

        let requires = x.req() || gamma.req() || beta.req();
        let result = x.new_like(out, x.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sx = x.grad.clone();
            let sg = gamma.grad.clone();
            let sb = beta.grad.clone();
            let xd = x.data.clone();
            let gd = gamma.data.clone();
            record(&result, vec![x, gamma.clone(), beta.clone()], Arc::new(move || {
                // 不做 to_vec 拷贝：decode 一次转只读切片，两遍计算共享同一份数据
                // （切片是 Sync，可安全带进下面的并行闭包）
                let g_b = rg.borrow();
                let x_bd = xd.decode();
                let gam_b = gd.decode();
                let g: &[f32] = &g_b;
                let x_b: &[f32] = &x_bd;
                let gam: &[f32] = &gam_b;
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
        // 非连续先物化，前后向均按行主序线性读（反向闭包还按缓冲读 xd）
        let x = self.contiguous();
        let d = *x.shape.last().unwrap();
        assert_eq!(gamma.rank(), 1, "RMSNorm 的 γ 必须是一维");
        assert_eq!(gamma.shape[0], d, "RMSNorm 的 γ 长度必须等于输入最后一维");
        let rows = x.numel() / d;
        let sd = x.decode();
        let gv = gamma.decode();
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

        let requires = x.req() || gamma.req();
        let result = x.new_like(out, x.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sx = x.grad.clone();
            let sg = gamma.grad.clone();
            let xd = x.data.clone();
            let gd = gamma.data.clone();
            let ir = inv_rms;
            record(&result, vec![x, gamma.clone()], Arc::new(move || {
                // 不做 to_vec 拷贝：decode 一次转只读切片，供两遍计算共享（见 layer_norm 反向的注释）
                let g_b = rg.borrow();
                let x_bd = xd.decode();
                let gam_b = gd.decode();
                let g: &[f32] = &g_b;
                let x_b: &[f32] = &x_bd;
                let gam: &[f32] = &gam_b;
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

    /// 注意力（缩放点积 + 因果屏蔽）：O = softmax(Q·Kᵀ/√d) · V
    ///
    /// **CPU 路径（`flash_forward_cpu`）：真正的分块 + 在线 softmax**。
    /// 按输出行（`B*H × T`）rayon 并行，行内按 `block_size` 个 K/V 为一块串行扫描：
    /// - 打分只对可见键计算（query i 只看 `j <= i + visible_before`），**不物化**
    ///   `[T, T_total]` 掩码——原先每次前向都要 O(T×T_total) 的分配与填充；
    /// - 行内在线 softmax：维护运行最大值 m 与指数和 l，新块并入时旧累计乘
    ///   `exp(m_old - m_new)` 重标定，最终除以 l。中间 P **从不落地**，
    ///   每行工作集只有 O(block_size + head_dim)，而不是 O(T_total)；
    /// - `block_size` 是真实的分块参数（钳到 ≥ 1），控制每次读入的 K/V 块宽。
    ///
    /// **反向不保留 P**：前后向之间 CPU 侧不留 O(T²)；反向时矩阵乘重算 scores、
    /// `causal_softmax_cpu` 归一化出 P，再走既有的
    /// dV = Pᵀ·dO、dP = dO·Vᵀ、dS = P⊙(dP - ΣdP·P)、dQ = dS·K·scale、dK = dSᵀ·Q'。
    ///
    /// **GPU 常驻路径不变**（`gpu::attn_forward`）：S/P 留在显存、反向一次提交；
    /// 它与 `LLM_GPU_PROBE` 的逐算子录制按真实掩码 buffer 消费——只有走这两条路时
    /// 才物化掩码（[`causal_mask_data`]）。GPU 掩码与核内屏蔽同为后缀因果，逐位一致。
    ///
    /// - q: [B*H, T, head_dim]
    /// - k/v: [B*H, T_total, head_dim]（T_total >= T）
    /// - mask: 可选的 `[T, T_total]` 后缀因果掩码（0 / -inf），只被 GPU/probe 路径消费；
    ///   传 `None` 时 GPU 需要的话在本函数内物化一次，CPU 分块核始终在核内屏蔽
    /// - block_size: CPU 分块的 K/V 块宽（钳到 ≥ 1）
    ///
    /// 返回 out: [B*H, T, head_dim]
    pub fn flash_attention(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        mask: Option<&Tensor>,
        block_size: usize,
    ) -> Tensor {
        // 非连续（视图）先物化：前后向都按行主序线性读，反向闭包还按缓冲读 K/V。
        // parents/grad 绑定随遮蔽自动指向物化张量，梯度经 contiguous 反向 1:1 回流视图
        let q = q.contiguous();
        let k = k.contiguous();
        let v = v.contiguous();
        let mask = mask.map(|m| m.contiguous());
        assert_eq!(q.rank(), 3, "flash_attention: Q 必须为 3D");
        assert_eq!(k.rank(), 3, "flash_attention: K 必须为 3D");
        assert_eq!(v.rank(), 3, "flash_attention: V 必须为 3D");
        let (bh, t, head_dim) = (q.shape[0], q.shape[1], q.shape[2]);
        let t_total = k.shape[1];
        assert_eq!(k.shape, v.shape, "K 和 V 形状必须一致");
        assert_eq!(k.shape[2], head_dim, "K/V 的 head_dim 必须与 Q 一致");
        // GQA：允许 K/V 扁平组数 kv_bh 小于 Q 的 bh（kv_bh = bh / n_rep）。
        // CPU 分块核按 Q 头 hh → KV 头 hh/n_rep 核内索引，省掉 repeat_kv 物化；
        // GPU 内核按 Q 的 bh 布局读 K/V，进入前在核外物化展开（见下）。
        let kv_bh = k.shape[0];
        assert!(
            kv_bh > 0 && bh % kv_bh == 0,
            "flash_attention: K/V 的 batch*head（{kv_bh}）必须整除 Q 的（{bh}）"
        );
        let n_rep = bh / kv_bh;
        assert!(
            t <= t_total,
            "flash_attention: K/V 长度（{t_total}）必须 >= Q 长度（{t}）"
        );
        // mask 为物化后的 Option<Tensor>，用引用匹配避免 move（后文 GPU 分支还要读）
        if let Some(m) = &mask {
            assert_eq!(
                m.shape.as_slice(),
                &[t, t_total],
                "flash_attention: mask 必须是 [T, T_total]"
            );
        }

        let scale = 1.0 / (head_dim as f32).sqrt();
        // 后缀因果的可见范围：与 causal_mask_data 物化的掩码逐位一致（分块核内屏蔽用）
        let visible_before = t_total - t;

        // 读取输入数据
        let qd = q.decode();
        let kd = k.decode();
        let vd = v.decode();

        // 1) Q' = Q / √d：把缩放挪到 Q 上，块内打分不必带 scale（数学上与先乘等价）
        let mut q_scaled: Vec<f32> = qd.to_vec();
        q_scaled.par_iter_mut().for_each(|v| *v *= scale);

        // 2) 前向分流：GPU 常驻 →（probe 时）逐算子录制 → CPU 分块在线 softmax
        #[cfg(feature = "gpu")]
        let (out_data, cache) = {
            // GQA n_rep>1：GPU 内核按 Q 的 bh 扁平布局读 K/V，核外先物化展开
            //（CPU 分块核不必物化，直接核内索引共享 KV 头）
            let k_exp: Option<Vec<f32>> = (n_rep > 1).then(|| repeat_flat(&kd, kv_bh, n_rep));
            let v_exp: Option<Vec<f32>> = (n_rep > 1).then(|| repeat_flat(&vd, kv_bh, n_rep));
            let kd_g: &[f32] = match &k_exp {
                Some(v) => v,
                None => &kd,
            };
            let vd_g: &[f32] = match &v_exp {
                Some(v) => v,
                None => &vd,
            };
            // GPU 常驻与 probe 录制要真实掩码 buffer；CPU 分块核核内屏蔽，不必物化
            let mask_owned: Option<Vec<f32>> = (mask.is_none() && crate::gpu::is_available())
                .then(|| causal_mask_data(t, t_total));
            // mask 现为 Option<Tensor>（物化后的常量），as_ref 取引用借解码视图的生命周期
            let mask_guard = mask.as_ref().map(|m| m.decode());
            let mask_data: Option<&[f32]> = match (&mask_guard, &mask_owned) {
                (Some(g), _) => Some(&g[..]),
                (None, Some(v)) => Some(&v[..]),
                (None, None) => None,
            };
            // S = Q'·Kᵀ → P = softmax(S+mask) → O = P·V 录进一次提交，
            // S 与 P 全程留在显存，只把 O 回读；P 的句柄留到反向（见 `AttnCache::Gpu`）
            let gpu_attn = match mask_data {
                Some(md) => {
                    crate::gpu::attn_forward(&q_scaled, kd_g, vd_g, md, bh, t, t_total, head_dim)
                }
                None => None,
            };
            match gpu_attn {
                Some(r) => {
                    let out = r.out.clone();
                    (out, AttnCache::Gpu(Box::new(r)))
                }
                None => match mask_data {
                    // LLM_GPU_PROBE 形状录制：拆回逐算子，让 matmul 记录真实训练形状
                    Some(md) if crate::gpu::probe_active() => {
                        let (o, _p) =
                            attn_forward_ops(&q_scaled, kd_g, vd_g, md, bh, t, t_total, head_dim);
                        (o, AttnCache::Cpu)
                    }
                    _ => (
                        flash_forward_cpu(
                            &q_scaled,
                            &kd,
                            &vd,
                            bh,
                            n_rep,
                            t,
                            t_total,
                            head_dim,
                            visible_before,
                            block_size,
                        ),
                        AttnCache::Cpu,
                    ),
                },
            }
        };
        // 纯 CPU 构建下没有 Gpu 缓存变体，cache 不会被闭包读取
        #[cfg(not(feature = "gpu"))]
        let (out_data, _cache) = (
            flash_forward_cpu(
                &q_scaled,
                &kd,
                &vd,
                bh,
                n_rep,
                t,
                t_total,
                head_dim,
                visible_before,
                block_size,
            ),
            AttnCache::Cpu,
        );

        drop(qd);
        drop(kd);
        drop(vd);

        let requires = q.req() || k.req() || v.req();
        let result = q.new_like(out_data, vec![bh, t, head_dim], requires);
        if requires {
            let rg = result.grad.clone();
            let sq = q.grad.clone();
            let sk = k.grad.clone();
            let sv = v.grad.clone();
            let k_data = k.data.clone();
            let v_data = v.data.clone();
            record(&result, vec![q.clone(), k.clone(), v.clone()], Arc::new(move || {
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
                        // GQA：GPU 收到的是核外展开后的 dK/dV（bh 组），按 n_rep
                        // 相邻副本求和折回 kv_bh 组再写父梯度；n_rep==1 直接 move 零拷贝
                        let dk_final =
                            if n_rep > 1 { fold_repeat_grad(&dk_out, kv_bh, n_rep) } else { dk_out };
                        let dv_final =
                            if n_rep > 1 { fold_repeat_grad(&dv_out, kv_bh, n_rep) } else { dv_out };
                        {
                            let mut sgm = sk.borrow_mut();
                            for (i, v) in dk_final.iter().enumerate() {
                                sgm[i] += v;
                            }
                        }
                        {
                            let mut sgm = sv.borrow_mut();
                            for (i, v) in dv_final.iter().enumerate() {
                                sgm[i] += v;
                            }
                        }
                        return;
                    }
                }
                // CPU 路径（也兜底 GPU 常驻反向失败）：前后向之间不保留 P，
                // 反向用矩阵乘重算 scores = Q'·Kᵀ，再由 causal_softmax_cpu 归一化出 P。
                // dP/dS 本来就是 O(T²)，重算 scores 不改变反向工作集的量级。
                // 反向：全部交给矩阵乘内核（旧的标量三重循环只有 0.24 GFLOP/s，慢 100 倍）
                // softmax 反向：dS[i,j] = P[i,j] * (dP[i,j] - Σ_k dP[i,k]·P[i,k])
                // 注意必须用 V 算 dP（out_i = Σ_j P_ij·V_j，故 ∂out_i/∂P_ij 的因子是 V_j）；
                // 用 K 会让 dP/dS/dQ/dK 全部错误（实测 dQ 偏差 2.7 倍）。
                let kd_b = k_data.decode();
                let vd_b = v_data.decode();
                // GQA：核内索引只在前向省物化；反向 matmul 链要求 Q/K 扁平 bh 相等，
                // 这里把 K/V 物化展开到 bh 组参与计算，dK/dV 算完再折回 kv_bh 组
                let k_exp: Option<Vec<f32>> =
                    (n_rep > 1).then(|| repeat_flat(&kd_b, kv_bh, n_rep));
                let v_exp: Option<Vec<f32>> =
                    (n_rep > 1).then(|| repeat_flat(&vd_b, kv_bh, n_rep));
                let kd_use: &[f32] = match &k_exp {
                    Some(v) => v,
                    None => &kd_b,
                };
                let vd_use: &[f32] = match &v_exp {
                    Some(v) => v,
                    None => &vd_b,
                };
                // 重算 scores 并做后缀因果 softmax，得到与前向逐位一致的 P（含掩码位置为 0）
                // （参数顺序与 attn_forward_ops 一致：m=t, k=head_dim, n=t_total）
                let scores =
                    matmul_data(&q_scaled, kd_use, t, head_dim, t_total, bh, false, true);
                let p = causal_softmax_cpu(&scores, bh * t, t, t_total, visible_before);
                // dV = Pᵀ·dO
                let dv_out = matmul_data(&p, &g, t_total, t, head_dim, bh, true, false);
                // dP = dO·Vᵀ
                let mut ds = matmul_data(&g, vd_use, t, head_dim, t_total, bh, false, true);
                drop(g);
                drop(vd_b);
                drop(v_exp);
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
                let dq_out = matmul_data(&ds, kd_use, t, t_total, head_dim, bh, false, false);
                let dk_out = matmul_data(&ds, &q_scaled, t_total, t, head_dim, bh, true, false);
                drop(kd_b);
                drop(k_exp);
                {
                    let mut sgm = sq.borrow_mut();
                    for (i, v) in dq_out.iter().enumerate() {
                        sgm[i] += v * scale;
                    }
                }
                // GQA：dK/dV 是展开后（bh 组）的结果，按 n_rep 相邻副本求和折回
                // kv_bh 组再写父梯度（n_rep==1 直接 move 零拷贝）
                let dk_final =
                    if n_rep > 1 { fold_repeat_grad(&dk_out, kv_bh, n_rep) } else { dk_out };
                let dv_final =
                    if n_rep > 1 { fold_repeat_grad(&dv_out, kv_bh, n_rep) } else { dv_out };
                {
                    let mut sgm = sk.borrow_mut();
                    for (i, v) in dk_final.iter().enumerate() {
                        sgm[i] += v;
                    }
                }
                {
                    let mut sgm = sv.borrow_mut();
                    for (i, v) in dv_final.iter().enumerate() {
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
        // 非连续先物化，前后向均按行主序线性读
        let x = self.contiguous();
        let mask = mask.contiguous();
        let d = *x.shape.last().unwrap();
        assert!(
            mask.rank() <= x.rank(),
            "mask 维度必须 <= 输入（mask {:?} vs x {:?}）",
            mask.shape,
            x.shape
        );
        // mask 必须是输入形状的精确右后缀（逐维相等）。尺寸 1 的维只有当目标对应维也是 1
        // 时才会出现（如 t=1 的 KV cache 单步生成），此时"源下标 = flat % numel"依然成立。
        let off = x.rank() - mask.rank();
        assert_eq!(
            &x.shape[off..],
            mask.shape.as_slice(),
            "mask 必须是输入的右后缀（x {:?} vs mask {:?}）",
            x.shape,
            mask.shape
        );
        let m_n = mask.numel();
        let rows = x.numel() / d;
        let sd = x.decode();
        let md = mask.decode();
        // GPU 优先：训练里 scores 是 [B*H,T,T_total]（~200 万元素），GPU 计算 ~2ms，
        // CPU 计算 ~30ms。太小（推理单 token）或 GPU 不可用时自动回退 CPU。
        #[cfg(feature = "gpu")]
        let out = crate::gpu::softmax_mask(&sd, &md, rows, d, m_n)
            .unwrap_or_else(|| masked_softmax_cpu(&sd, &md, rows, d, m_n));
        #[cfg(not(feature = "gpu"))]
        let out = masked_softmax_cpu(&sd, &md, rows, d, m_n);
        drop(sd);
        drop(md);

        let requires = x.req() || mask.req();
        let out_shared = Arc::new(out);
        let result = x.new_like(out_shared.as_ref().clone(), x.shape.clone(), requires);
        if requires {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let out = out_shared;
            record(&result, vec![x, mask], Arc::new(move || {
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
    pub fn log_softmax_last_dim(&self) -> Tensor {
        assert!(self.rank() >= 1, "log_softmax 至少需要 1 维");
        // 非连续先物化，下文按行主序线性读
        let a = self.contiguous();
        let (rows, d) = (
            a.numel() / a.shape[a.rank() - 1],
            a.shape[a.rank() - 1],
        );
        let sd = a.decode();
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

        let result = a.new_like(out_data, a.shape.clone(), a.requires_grad.load(Ordering::Relaxed));
        if a.req() {
            let rg = result.grad.clone();
            let sg = a.grad.clone();
            record(&result, vec![a], Arc::new(move || {
                // 不做 to_vec 拷贝：borrow 后转只读切片，再 borrow_mut 梯度槽
                // （rg 与 sg 是两把不同的锁，顺序无关）
                let g_b = rg.borrow();
                let g: &[f32] = &g_b;
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
        // 非连续先物化，下文按行主序线性读
        let x = self.contiguous();
        if !training || p == 0.0 {
            // 推理或不丢弃：恒等（需要梯度时设 requires_grad）。
            // 先拷贝再构造：decode 持锁不能与 new_like 内的 dtype() 加锁同处一条语句
            let id = x.decode().to_vec();
            return x.new_like(id, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        }
        if p >= 1.0 {
            return x.new_like(vec![0.0; x.numel()], x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        }
        let keep = 1.0 - p;
        let scale = 1.0 / keep;
        let sd = x.decode();
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

        let result = x.new_like(out_data, x.shape.clone(), x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            record(&result, vec![x], Arc::new(move || {
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
        // 非连续先物化，前后向均按行主序线性读
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "gather_rows 的 table 必须为 2 维");
        let (v, d) = (x.shape[0], x.shape[1]);
        let n = indices.len();
        let sd = x.decode();
        let mut out_data = vec![0.0f32; n * d];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < v, "gather 索引越界：{} >= {}", idx, v);
            for j in 0..d {
                out_data[i * d + j] = sd[idx * d + j];
            }
        }
        drop(sd);
        let idx_vec = indices.to_vec();

        let result = x.new_like(out_data, vec![n, d], x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let d2 = d;
            record(&result, vec![x], Arc::new(move || {
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

    /// 按行散射相加（`gather_rows` 的转置）：rows [N, D] + indices [N] -> out [M, D]。
    ///
    /// `out[indices[i]] += rows[i]`，其余行保持 0；同一行被多次命中时数值（与梯度）累加。
    /// 反向恰好是 [`Tensor::gather_rows`] 的前向：`d_rows[i] = g[indices[i]]`。
    ///
    /// 用途是 MoE（第 32 课）：专家只处理"分到自己这里"的若干 token（先 `gather_rows`
    /// 把这些行取出来），算完再散射回原来的位置。真实的稀疏 MoE 就是
    /// gather → expert → scatter 三步，而不是把每个专家都跑满全部 token 再用掩码筛掉——
    /// 后者算出来的数值一样，却把稀疏激活省下的计算量全还回去了。
    pub fn scatter_add_rows(&self, indices: &[usize], n_out: usize) -> Tensor {
        // 非连续先物化，前后向均按行主序线性读
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "scatter_add_rows 的 rows 必须为 2 维");
        assert_eq!(
            indices.len(),
            x.shape[0],
            "indices 数量必须等于 rows 的行数"
        );
        let d = x.shape[1];
        let sd = x.decode();
        let mut out_data = vec![0.0f32; n_out * d];
        for (i, &idx) in indices.iter().enumerate() {
            assert!(idx < n_out, "scatter 索引越界：{} >= {}", idx, n_out);
            for j in 0..d {
                out_data[idx * d + j] += sd[i * d + j];
            }
        }
        drop(sd);
        let idx_vec = indices.to_vec();

        let result = x.new_like(out_data, vec![n_out, d], x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let d2 = d;
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for (i, &row) in idx_vec.iter().enumerate() {
                    for j in 0..d2 {
                        sgm[i * d2 + j] += g[row * d2 + j];
                    }
                }
            }));
        }
        result
    }

    /// 取 2 维张量的第 `col` 列：`[R, C] -> [R, 1]`，`out[i] = self[i, col]`。
    ///
    /// 反向把 `[R, 1]` 的梯度写回 grad 的第 col 列——每行只有一个落点，`+=` 即可。
    ///
    /// 用途是 MoE 取门控权重的第 e 列：原实现是"右乘 one-hot 列向量"的
    /// `matmul([n,E], [E,1])`，为了把梯度送回路由器把 n×E 次乘加全跑了一遍；
    /// 现在 `gather_rows(sel).select_col(e)` 两步只碰真正需要的 n_e 个元素。
    pub fn select_col(&self, col: usize) -> Tensor {
        // 非连续先物化，前后向均按行主序线性读
        let x = self.contiguous();
        assert_eq!(x.rank(), 2, "select_col 的输入必须为 2 维");
        let (rows, cols) = (x.shape[0], x.shape[1]);
        assert!(col < cols, "列索引越界：{} >= {}", col, cols);
        let sd = x.decode();
        let mut out_data = vec![0.0f32; rows];
        for i in 0..rows {
            out_data[i] = sd[i * cols + col];
        }
        drop(sd);

        let result = x.new_like(out_data, vec![rows, 1], x.requires_grad.load(Ordering::Relaxed));
        if x.req() {
            let rg = result.grad.clone();
            let sg = x.grad.clone();
            let c2 = cols;
            record(&result, vec![x], Arc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i * c2 + col] += g[i];
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
        // data() 按逻辑序返回（非连续视图会 gather），打印与形状一致
        let data = self.data();
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
    fn test_matmul_frozen() {
        // x: [2,3]，W: [3,2]
        let x_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let w_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        // ① 前向与通用 matmul 完全一致
        let x = Tensor::param(x_data.clone(), vec![2, 3]);
        let w = Tensor::param(w_data.clone(), vec![3, 2]);
        let y = x.matmul_frozen(&w);
        let y_ref = Tensor::from_vec(x_data, vec![2, 3])
            .matmul(&Tensor::from_vec(w_data, vec![3, 2]));
        assert_eq!(y.shape(), &[2, 2]);
        assert_eq!(y.data(), y_ref.data());

        // ② 反向：loss = sum(y) ⇒ 上游梯度全 1，dx = 1 @ Wᵀ 的每一行都是 W 的列和
        y.sum().backward();
        assert_eq!(x.grad(), vec![3.0, 7.0, 11.0, 3.0, 7.0, 11.0]);

        // ③ W 是 param（requires_grad = true）却一点梯度都没攒到：
        //    "冻结"不只体现在优化器不更新它，而是反向压根没算 dW
        assert_eq!(w.grad(), vec![0.0; 6]);
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

    /// 融合 loss 节点（[`Tensor::external_scalar_loss`]）必须消费自己的上游梯度。
    ///
    /// 这类节点的反向是「按上游梯度 = 1 算好、再整体乘回去」，而 autograd 按逆拓扑序
    /// 执行、轮到它时才把上游梯度攒进它的 grad 槽。若把上游当成恒等于 1，
    /// AMP 的 `scale_loss`（`mul_scalar(2^16)`）就会被静默丢掉：梯度被 scale 除回去后
    /// 整体偏小 2^16 倍，梯度裁剪随之失效、训练轨迹偏离。这里用 2 的幂做标量，
    /// 断言结果**逐位**相等（乘以 2^k 在 f32 下是精确的指数移位）。
    #[test]
    fn test_external_scalar_loss_consumes_upstream_gradient() {
        let make = || {
            let x = Tensor::param(vec![1.0f32, 2.0, 3.0], vec![3]);
            let x_bwd = x.clone();
            // 模拟融合核：dx 按「上游梯度 = 1」算好，比例留给节点自己补
            let loss = Tensor::external_scalar_loss(3.0, vec![x.clone()], move |upstream| {
                x_bwd.accumulate_grad(&[1.0, 1.0, 1.0], upstream);
            });
            (x, loss)
        };

        let (x, loss) = make();
        loss.backward();
        assert_eq!(x.grad(), vec![1.0, 1.0, 1.0], "上游为 1 时应原样传下去");

        let (x, loss) = make();
        loss.mul_scalar(2.0f32.powi(16)).backward();
        let want = 2.0f32.powi(16);
        assert_eq!(x.grad(), vec![want, want, want], "上游被缩放后必须完整传递");
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
        let out_flash = Tensor::flash_attention(&q, &k, &v, Some(&mask), 2);

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
        let out_flash = Tensor::flash_attention(&q_f, &k_f, &v_f, Some(&mask), bs);
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

    /// GQA：K/V 少头（kv_bh = bh / n_rep）直接进 flash，必须与「显式 repeat 展开成
    /// bh 组后再进 flash」逐位一致 —— 前向输出相同，反向 dQ 相同、dK/dV 折回后相同。
    ///
    /// 这是核内索引方案（CPU 分块核按 Q 头 `hh / n_rep` 取共享 KV 头，省掉
    /// repeat_kv 物化）与旧「先物化展开再算」的等价性回归：头序必须互逆
    /// （输出头 i ← 源头 i / n_rep），dK/dV 需按 n_rep 相邻副本求和折回 kv_bh 组。
    #[test]
    fn test_flash_attention_gqa_matches_expanded() {
        use crate::rng::Rng;
        let (bh, n_rep, t, d, bs) = (4usize, 2usize, 16usize, 8usize, 4usize);
        let kv_bh = bh / n_rep;
        let mut rng = Rng::new(13);
        let amp = 3.0f32;
        let q_data: Vec<f32> = (0..bh * t * d).map(|_| rng.randn() * amp).collect();
        let k_data: Vec<f32> = (0..kv_bh * t * d).map(|_| rng.randn() * amp).collect();
        let v_data: Vec<f32> = (0..kv_bh * t * d).map(|_| rng.randn()).collect();

        // 因果掩码
        let mut mask_data = vec![f32::NEG_INFINITY; t * t];
        for i in 0..t {
            for j in 0..=i {
                mask_data[i * t + j] = 0.0;
            }
        }
        let mask = Tensor::from_vec(mask_data, vec![t, t]);

        // ---- 参考：显式把 K/V 展开成 bh 组（走 n_rep=1 的普通 flash 路径）----
        let kx = repeat_flat(&k_data, kv_bh, n_rep);
        let vx = repeat_flat(&v_data, kv_bh, n_rep);
        let q_s = Tensor::param(q_data.clone(), vec![bh, t, d]);
        let k_s = Tensor::param(kx.clone(), vec![bh, t, d]);
        let v_s = Tensor::param(vx, vec![bh, t, d]);
        let out_std = Tensor::flash_attention(&q_s, &k_s, &v_s, Some(&mask), bs);
        let out_std_data = out_std.data();
        out_std.sum().backward();

        // ---- GQA：K/V 只有 kv_bh 组，靠核内索引共享 ----
        let q_f = Tensor::param(q_data, vec![bh, t, d]);
        let k_f = Tensor::param(k_data, vec![kv_bh, t, d]);
        let v_f = Tensor::param(v_data, vec![kv_bh, t, d]);
        let out_gqa = Tensor::flash_attention(&q_f, &k_f, &v_f, Some(&mask), bs);
        assert_eq!(out_gqa.shape(), &[bh, t, d], "GQA 前向形状应为 [bh, t, d]");
        out_gqa.sum().backward();

        // 前向逐位一致
        for (i, (a, b)) in out_std_data.iter().zip(out_gqa.data().as_slice()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5 * a.abs().max(1.0),
                "前向第 {i} 个元素不一致：{a} vs {b}"
            );
        }

        // dQ：两边都是 bh 组，直接比
        let (dq_s, dq_f) = (q_s.grad(), q_f.grad());
        assert_rel_close("dQ", &dq_s, &dq_f);

        // dK/dV：参考梯度是展开形状（bh 组），折回 kv_bh 组后再与 GQA 梯度比
        let dk_fold = fold_repeat_grad(&k_s.grad(), kv_bh, n_rep);
        assert_rel_close("dK(折回)", &dk_fold, &k_f.grad());
        let dv_fold = fold_repeat_grad(&v_s.grad(), kv_bh, n_rep);
        assert_rel_close("dV(折回)", &dv_fold, &v_f.grad());
    }

    /// 相对误差断言：最大绝对误差 / 参考量级 < 1e-3
    fn assert_rel_close(name: &str, want: &[f32], got: &[f32]) {
        assert_eq!(want.len(), got.len(), "{name} 长度不一致");
        let max_err = want
            .iter()
            .zip(got)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        let ref_mag = want.iter().map(|x| x.abs()).fold(0.0f32, f32::max).max(1e-6);
        assert!(
            max_err / ref_mag < 1e-3,
            "{name} 梯度不一致：最大绝对误差 {max_err}（参考量级 {ref_mag}）"
        );
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
        let out = Tensor::flash_attention(&qt, &kt, &vt, Some(&mt), 64);
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

    // ==================== bf16 混合精度（批次 10b） ====================

    /// ① RNE 编解码：往返精度、精确值无损、半数就近取偶、inf/NaN 特判。
    #[test]
    fn test_bf16_codec_rne_roundtrip() {
        // 普通值：round-to-nearest-even 相对误差 ≤ 半个 bf16 ulp（2^-8，留 1% 余量）
        for &x in &[1.0f32, -1.0, 0.5, 3.14159, -0.001234, 65504.0, 1e-30, -7.75] {
            let r = bf16_to_f32(f32_to_bf16(x));
            let rel = ((r - x) / x).abs();
            assert!(rel <= 2.0f32.powi(-8) * 1.01, "{x} -> {r} 相对误差 {rel}");
        }
        // bf16 能精确表示的值往返无损（含 -0 的符号位）
        for &x in &[0.0f32, 1.0, -2.5, 1024.0, 0.75, 19.0] {
            assert_eq!(bf16_to_f32(f32_to_bf16(x)), x);
        }
        assert_eq!(
            bf16_to_f32(f32_to_bf16(-0.0)).to_bits(),
            (-0.0f32).to_bits(),
            "-0 符号位必须保留"
        );
        // 就近取偶：恰好半数时向尾数最低位为 0 的一侧靠拢
        // 0x3F808000 恰是 0x3F80（1.0）与 0x3F81 的正中 → 取偶数 0x3F80
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F80_8000)), 0x3F80);
        // 0x3F818000 恰是 0x3F81 与 0x3F82 的正中 → 取偶数 0x3F82
        assert_eq!(f32_to_bf16(f32::from_bits(0x3F81_8000)), 0x3F82);
        // ±∞ 按位无损；NaN 特判保证转换后仍是 NaN（纯 RNE 可能把 NaN 尾数进成全 0 → ∞）
        assert_eq!(
            bf16_to_f32(f32_to_bf16(f32::INFINITY)).to_bits(),
            f32::INFINITY.to_bits()
        );
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    /// ② `to_bf16()`：真 u16 存储（内存减半依据）、克隆句柄同步、decode 容差、幂等。
    #[test]
    fn test_to_bf16_real_u16_storage() {
        let vals: Vec<f32> = (0..64).map(|i| i as f32 * 0.37).collect();
        let t = Tensor::param(vals.clone(), vec![8, 8]);
        assert_eq!(t.dtype(), DType::F32);
        t.to_bf16();
        assert_eq!(t.dtype(), DType::Bf16);
        // 物理存储确实是 u16 变体——"内存真实减半"的直接依据
        assert!(matches!(&*t.data.borrow(), Buffer::Bf16(u) if u.len() == 64));
        // 单层锁槽：克隆句柄同步看到新变体
        assert_eq!(t.clone().dtype(), DType::Bf16);
        // decode 数值在 bf16 相对误差（2^-8）内
        for (orig, dec) in vals.iter().zip(t.data().iter()) {
            let rel = ((dec - orig) / orig.abs().max(1e-6)).abs();
            assert!(rel <= 2.0f32.powi(-8) * 1.01, "{orig} -> {dec} 相对误差 {rel}");
        }
        // 重复转换幂等
        t.to_bf16();
        assert_eq!(t.dtype(), DType::Bf16);
    }

    /// ③ 算子出口 `new_like` 继承 dtype：bf16 进、bf16 出，数值对齐 f32 参考。
    #[test]
    fn test_bf16_ops_preserve_dtype() {
        let a_d = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_d = vec![5.0f32, 6.0, 7.0, 8.0];
        // f32 参考
        let c_ref = Tensor::from_vec(a_d.clone(), vec![2, 2])
            .matmul(&Tensor::from_vec(b_d.clone(), vec![2, 2]));
        let s_ref = Tensor::from_vec(a_d.clone(), vec![2, 2])
            .add(&Tensor::from_vec(b_d.clone(), vec![2, 2]));

        let a = Tensor::param(a_d, vec![2, 2]);
        let b = Tensor::param(b_d, vec![2, 2]);
        a.to_bf16();
        b.to_bf16();
        // 出口 dtype 继承左操作数 / 主输入
        let c = a.matmul(&b);
        let s = a.add(&b);
        assert_eq!(c.dtype(), DType::Bf16);
        assert_eq!(s.dtype(), DType::Bf16);
        // 1..8 与其乘积在 bf16 内均精确表示 → 与 f32 参考完全相等
        assert_eq!(c.data(), c_ref.data());
        assert_eq!(s.data(), s_ref.data());

        // 非精确值：输入编码、累加、出口编码各一档，保守界 3×2^-8 ≈ 2^-6
        let p_d = vec![0.1f32, 0.2, 0.3, 0.4];
        let q_d = vec![1.1f32, 2.2, 3.3, 4.4];
        let pref = Tensor::from_vec(p_d.clone(), vec![2, 2])
            .matmul(&Tensor::from_vec(q_d.clone(), vec![2, 2]))
            .data();
        let p = Tensor::param(p_d, vec![2, 2]);
        let q = Tensor::param(q_d, vec![2, 2]);
        p.to_bf16();
        q.to_bf16();
        let got = p.matmul(&q).data();
        for (r, g) in pref.iter().zip(&got) {
            let rel = ((g - r) / r.abs().max(1e-6)).abs();
            assert!(rel <= 2.0f32.powi(-6), "参考 {r} vs bf16 {g} 相对误差 {rel}");
        }
    }

    /// ④ `decode_mut` 就地更新语义（master weights）：Drop 时 encode 回写 u16 存储。
    #[test]
    fn test_bf16_decode_mut_writes_back() {
        let t = Tensor::param(vec![1.0f32, 2.0], vec![2]);
        t.to_bf16();
        {
            let mut g = t.decode_mut();
            g[0] = 3.5; // 3.5 / -0.25 在 bf16 内精确
            g[1] = -0.25;
        } // Drop → 并行 encode 回槽内 u16
        assert_eq!(t.dtype(), DType::Bf16);
        assert_eq!(t.data(), vec![3.5, -0.25]);
        // F32 张量的 decode_mut 仍是直接锁借位，语义不变
        let f = Tensor::param(vec![1.0f32], vec![1]);
        {
            let mut g = f.decode_mut();
            g[0] = 9.0;
        }
        assert_eq!(f.data(), vec![9.0]);
    }

    /// ⑤ 反向传播：参数存 bf16 时梯度缓冲恒 f32，数值与解析值一致。
    #[test]
    fn test_bf16_backward_grad_stays_f32() {
        // loss = sum(x²) ⇒ dloss/dx = 2x；选值均在 bf16 内精确表示
        let x = Tensor::param(vec![1.5f32, -2.0, 0.75], vec![3]);
        x.to_bf16();
        assert_eq!(x.dtype(), DType::Bf16);
        let y = x.mul(&x);
        assert_eq!(y.dtype(), DType::Bf16); // 出口同样 encode 落盘
        y.sum().backward();
        // grad 是 f32 缓冲（grad() 返回 Vec<f32>），值 = 2x 完全精确
        assert_eq!(x.grad(), vec![3.0, -4.0, 1.5]);
        // 梯度缓冲不参与 bf16 约定：数据转 bf16 不影响已攒下的梯度
        let z = x.mul(&x);
        assert_eq!(z.dtype(), DType::Bf16);
    }
}
