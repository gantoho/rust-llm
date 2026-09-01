//! 张量库（第 1-4 课）
//!
//! 功能清单：
//! - 构造：from_vec / param / zeros / ones
//! - 访问：data / set_data / grad / zero_grad / shape / get / set
//! - 形状：reshape / transpose(2D) / permute(任意维) / flat_index
//! - 逐元素：add / sub / mul / div（**支持广播**）/ add_scalar / mul_scalar
//! - 激活：neg / relu / tanh / gelu / exp / log / pow
//! - 矩阵：matmul（2D 与 3D 批量）
//! - 归约：sum / sum_last_dim / softmax_last_dim
//! - 索引：gather_rows（Embedding 用）
//!
//! 自动微分（backward）见 `src/autograd.rs`，
//! 旋转位置编码（rotary）见 `src/rope.rs`。

use std::cell::RefCell;
use std::rc::Rc;

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
    for flat in 0..total {
        // 反解目标多维索引（行优先）
        let mut r = flat;
        let mut t_idx = vec![0usize; target.len()];
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

/// 把一个"小形状"的向量广播扩展成"大形状"的向量（纯数据工具，KV cache 用）
#[allow(dead_code)] // 广播工具 API（二元运算内部已内联处理）
pub fn broadcast_data(src: &[f32], src_shape: &[usize], target_shape: &[usize]) -> Vec<f32> {
    let map = broadcast_map(target_shape, src_shape);
    map.iter().map(|&i| src[i]).collect()
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
        assert_eq!(data.len(), numel, "参数数据长度与形状不一致");
        Tensor::new(data, shape, true)
    }

    /// 全 0 张量
    #[allow(dead_code)]
    pub fn zeros(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor::new(vec![0.0; numel], shape, false)
    }

    /// 全 1 张量
    #[allow(dead_code)]
    pub fn ones(shape: Vec<usize>) -> Self {
        let numel: usize = shape.iter().product();
        Tensor::new(vec![1.0; numel], shape, false)
    }

    /// 标量张量
    #[allow(dead_code)]
    pub fn scalar(v: f32) -> Self {
        Tensor::new(vec![v], vec![], false)
    }

    /// 用指定值填充
    #[allow(dead_code)]
    pub fn fill(shape: Vec<usize>, value: f32) -> Self {
        let numel: usize = shape.iter().product();
        Tensor::new(vec![value; numel], shape, false)
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

    pub fn grad(&self) -> Vec<f32> {
        self.grad.borrow().clone()
    }

    /// 覆盖梯度（梯度裁剪用）
    #[allow(dead_code)] // clip_grad_norm 已改用 borrow_mut 原位操作
    pub fn grad_set(&self, new_grad: Vec<f32>) {
        let mut g = self.grad.borrow_mut();
        assert_eq!(g.len(), new_grad.len(), "grad_set 长度不一致");
        *g = new_grad;
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

    #[allow(dead_code)]
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    #[allow(dead_code)]
    pub fn dim(&self, index: usize) -> usize {
        self.shape[index]
    }

    /// 读取元素（只读，破坏计算图语义，仅供调试）
    #[allow(dead_code)]
    pub fn get(&self, index: &[usize]) -> f32 {
        self.data.borrow()[self.flat_index(index)]
    }

    #[allow(dead_code)]
    pub fn set(&mut self, index: &[usize], value: f32) {
        let flat = self.flat_index(index);
        self.data.borrow_mut()[flat] = value;
    }

    // ---------- 形状工具 ----------

    #[allow(dead_code)]
    fn flat_index(&self, index: &[usize]) -> usize {
        assert_eq!(index.len(), self.rank(), "索引维度与张量维度不一致");
        let mut flat = 0;
        for (i, &idx) in index.iter().enumerate() {
            assert!(
                idx < self.shape[i],
                "索引越界：{:?} 超出形状 {:?}",
                index,
                self.shape
            );
            flat = flat * self.shape[i] + idx;
        }
        flat
    }

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
            parents: Rc::new(vec![self.clone()]),
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
        let mut map = vec![0usize; total];
        for out_flat in 0..total {
            let mut r = out_flat;
            let mut out_idx = vec![0usize; self.rank()];
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

        let sd = self.data.borrow();
        let mut out_data = vec![0.0f32; total];
        for (of, &sf) in map.iter().enumerate() {
            out_data[of] = sd[sf];
        }
        drop(sd);

        let mut result = Tensor::new(out_data, new_shape, self.requires_grad);
        if self.requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for (of, &sf) in map.iter().enumerate() {
                    sgm[sf] += g[of];
                }
            }));
        }
        result
    }

    /// 2 维转置（permute([1,0]) 的特例）
    #[allow(dead_code)]
    pub fn transpose(&self) -> Tensor {
        assert_eq!(self.rank(), 2, "transpose 只支持 2 维");
        self.permute(&[1, 0])
    }

    // ---------- 逐元素运算（支持广播） ----------

    /// 内部工具：判断是否同形状；不同则计算广播 map
    fn broadcast_plan(
        &self,
        other: &Tensor,
    ) -> (Vec<usize>, Option<Vec<usize>>, Option<Vec<usize>>) {
        if self.shape == other.shape {
            (self.shape.clone(), None, None)
        } else {
            let target = broadcast_shapes(&self.shape, &other.shape)
                .unwrap_or_else(|| panic!("形状无法广播：{:?} vs {:?}", self.shape, other.shape));
            let map_a = if self.shape == target {
                None
            } else {
                Some(broadcast_map(&target, &self.shape))
            };
            let map_b = if other.shape == target {
                None
            } else {
                Some(broadcast_map(&target, &other.shape))
            };
            (target, map_a, map_b)
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
        fwd: impl Fn(f32, f32) -> f32 + 'static,
        back: impl Fn(f32, f32) -> (f32, f32) + 'static,
    ) -> Tensor {
        let (target_shape, map_a, map_b) = self.broadcast_plan(other);
        let sa = self.data.borrow();
        let sb = other.data.borrow();
        let total: usize = target_shape.iter().product();
        let mut out_data = vec![0.0f32; total];
        for t in 0..total {
            let ia = match &map_a {
                Some(m) => m[t],
                None => t,
            };
            let ib = match &map_b {
                Some(m) => m[t],
                None => t,
            };
            out_data[t] = fwd(sa[ia], sb[ib]);
        }
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
            let map_a_c = map_a.clone();
            let map_b_c = map_b.clone();
            let same_shape = self.shape == other.shape;
            result.parents = Rc::new(vec![self.clone(), other.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.borrow();
                let od_b = od.borrow();
                let (ga, gb) = if same_shape && Rc::ptr_eq(&sg, &og) {
                    // 特例：同一张量参与运算（如 x*x、x/x），两条路径梯度叠加
                    let mut sgm = sg.borrow_mut();
                    for i in 0..g.len() {
                        let (da, db) = back(sd_b[i], od_b[i]);
                        sgm[i] += g[i] * (da + db);
                    }
                    (true, true)
                } else {
                    let mut sgm = sg.borrow_mut();
                    let mut ogm = og.borrow_mut();
                    for t in 0..g.len() {
                        let ia = match &map_a_c {
                            Some(m) => m[t],
                            None => t,
                        };
                        let ib = match &map_b_c {
                            Some(m) => m[t],
                            None => t,
                        };
                        let (da, db) = back(sd_b[ia], od_b[ib]);
                        sgm[ia] += g[t] * da;
                        ogm[ib] += g[t] * db;
                    }
                    (false, false)
                };
                let _ = (ga, gb);
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

    /// 内部：构造一元运算节点
    fn unary(&self, data: Vec<f32>, fwd_shape: Vec<usize>) -> Tensor {
        Tensor::new(data, fwd_shape, self.requires_grad)
    }

    /// 取负：c = -x，∂x = -g
    pub fn neg(&self) -> Tensor {
        let data = self.data.borrow().iter().map(|a| -a).collect();
        let mut result = self.unary(data, self.shape.clone());
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
    pub fn relu(&self) -> Tensor {
        let sd = self.data.borrow();
        let data = sd.iter().map(|&a| a.max(0.0)).collect();
        let mask: Vec<f32> = sd
            .iter()
            .map(|&a| if a > 0.0 { 1.0 } else { 0.0 })
            .collect();
        drop(sd);
        let mut result = self.unary(data, self.shape.clone());
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
        let mut result = self.unary(data.clone(), self.shape.clone());
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
        let mut data = vec![0.0f32; sd.len()];
        let mut t_vals = vec![0.0f32; sd.len()]; // 缓存 tanh(a)，反向要用
        for (i, &x) in sd.iter().enumerate() {
            let a = SQRT_2_PI * (x + COEF * x * x * x);
            let t = a.tanh();
            t_vals[i] = t;
            data[i] = 0.5 * x * (1.0 + t);
        }
        drop(sd);
        let mut result = self.unary(data.clone(), self.shape.clone());
        if self.requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let sd = self.data.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let x_b = sd.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    let x = x_b[i];
                    let t = t_vals[i];
                    let da_dx = SQRT_2_PI * (1.0 + 3.0 * COEF * x * x);
                    let dy_dx = 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * da_dx;
                    sgm[i] += g[i] * dy_dx;
                }
            }));
        }
        result
    }

    /// exp：c = e^x，∂x = g * c
    #[allow(dead_code)]
    pub fn exp(&self) -> Tensor {
        let sd = self.data.borrow();
        let data: Vec<f32> = sd.iter().map(|&a| a.exp()).collect();
        drop(sd);
        let mut result = self.unary(data.clone(), self.shape.clone());
        if self.requires_grad {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            result.parents = Rc::new(vec![self.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let mut sgm = sg.borrow_mut();
                for i in 0..g.len() {
                    sgm[i] += g[i] * data[i];
                }
            }));
        }
        result
    }

    /// log：c = ln(x)，∂x = g / x
    #[allow(dead_code)] // cross_entropy 已改用 log_softmax_last_dim
    pub fn log(&self) -> Tensor {
        let sd = self.data.borrow();
        let data: Vec<f32> = sd.iter().map(|&a| a.ln()).collect();
        drop(sd);
        let mut result = self.unary(data, self.shape.clone());
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
                    sgm[i] += g[i] / (sd_b[i] + 1e-8);
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
        let mut result = self.unary(data, self.shape.clone());
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
        let mut out_data = vec![0.0f32; b * m * n];
        for bi in 0..b {
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0;
                    for k in 0..k1 {
                        s += sd[(bi * m + i) * k1 + k] * od[(bi * k1 + k) * n + j];
                    }
                    out_data[(bi * m + i) * n + j] = s;
                }
            }
        }
        drop(sd);
        drop(od);

        let requires = self.requires_grad || other.requires_grad;
        let mut result = Tensor::new(out_data, vec![b, m, n], requires);
        if requires {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let og = other.grad.clone();
            let sd = self.data.clone();
            let od = other.data.clone();
            result.parents = Rc::new(vec![self.clone(), other.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.borrow();
                let od_b = od.borrow();
                if Rc::ptr_eq(&sg, &og) {
                    let mut sgm = sg.borrow_mut();
                    for bi in 0..b {
                        for i in 0..m {
                            for k in 0..k1 {
                                let mut s = 0.0;
                                for j in 0..n {
                                    s += g[(bi * m + i) * n + j] * od_b[(bi * k1 + k) * n + j];
                                }
                                sgm[(bi * m + i) * k1 + k] += s;
                            }
                        }
                        for k in 0..k1 {
                            for j in 0..n {
                                let mut s = 0.0;
                                for i in 0..m {
                                    s += sd_b[(bi * m + i) * k1 + k] * g[(bi * m + i) * n + j];
                                }
                                sgm[(bi * k1 + k) * n + j] += s;
                            }
                        }
                    }
                } else {
                    let mut sgm = sg.borrow_mut();
                    let mut ogm = og.borrow_mut();
                    for bi in 0..b {
                        for i in 0..m {
                            for k in 0..k1 {
                                let mut s = 0.0;
                                for j in 0..n {
                                    s += g[(bi * m + i) * n + j] * od_b[(bi * k1 + k) * n + j];
                                }
                                sgm[(bi * m + i) * k1 + k] += s;
                            }
                        }
                        for k in 0..k1 {
                            for j in 0..n {
                                let mut s = 0.0;
                                for i in 0..m {
                                    s += sd_b[(bi * m + i) * k1 + k] * g[(bi * m + i) * n + j];
                                }
                                ogm[(bi * k1 + k) * n + j] += s;
                            }
                        }
                    }
                }
            }));
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
        let mut out_data = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..k1 {
                    s += sd[i * k1 + k] * od[k * n + j];
                }
                out_data[i * n + j] = s;
            }
        }
        drop(sd);
        drop(od);

        let requires = self.requires_grad || other.requires_grad;
        let mut result = Tensor::new(out_data, vec![m, n], requires);
        if requires {
            let rg = result.grad.clone();
            let sg = self.grad.clone();
            let og = other.grad.clone();
            let sd = self.data.clone();
            let od = other.data.clone();
            result.parents = Rc::new(vec![self.clone(), other.clone()]);
            result.backward = Some(Rc::new(move || {
                let g = rg.borrow();
                let sd_b = sd.borrow();
                let od_b = od.borrow();
                if Rc::ptr_eq(&sg, &og) {
                    let mut sgm = sg.borrow_mut();
                    for i in 0..m {
                        for k in 0..k1 {
                            let mut s = 0.0;
                            for j in 0..n {
                                s += g[i * n + j] * od_b[k * n + j];
                            }
                            sgm[i * k1 + k] += s;
                        }
                    }
                    for k in 0..k1 {
                        for j in 0..n {
                            let mut s = 0.0;
                            for i in 0..m {
                                s += sd_b[i * k1 + k] * g[i * n + j];
                            }
                            sgm[k * n + j] += s;
                        }
                    }
                } else {
                    let mut sgm = sg.borrow_mut();
                    let mut ogm = og.borrow_mut();
                    for i in 0..m {
                        for k in 0..k1 {
                            let mut s = 0.0;
                            for j in 0..n {
                                s += g[i * n + j] * od_b[k * n + j];
                            }
                            sgm[i * k1 + k] += s;
                        }
                    }
                    for k in 0..k1 {
                        for j in 0..n {
                            let mut s = 0.0;
                            for i in 0..m {
                                s += sd_b[i * k1 + k] * g[i * n + j];
                            }
                            ogm[k * n + j] += s;
                        }
                    }
                }
            }));
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
            let (pre, d) = (pre, d);
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
            let softmax_data = softmax_data; // move into closure
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
}
