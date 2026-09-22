//! 量化基础（第 33 课）
//!
//! 量化的本质是"用更少的比特表示同一个数"：把连续的 f32 映射到有限个整数码上，
//! 再额外存一个（或一组）缩放因子用于还原。本模块提供**对称量化**的公共底座，
//! 供两处使用：
//!
//! 1. **权重 / 激活量化**（第 33 课后面的 GPTQ、AWQ、量化推理）——把 `Linear` 的
//!    权重从 f32 压到 int8 / int4，显存占用降到 1/4 或 1/8；
//! 2. **KV cache 量化**（KIVI，见 [`crate::attention::KVCache`]）——长上下文推理时
//!    KV cache 才是显存大头，把它压到 int8 / int4 才能继续加长上下文。
//!
//! ## 对称量化的公式
//!
//! 给定一组数 `x`，取该组的最大绝对值 `s = max|x|`，则
//!
//! ```text
//! 量化：q  = round(x / s * qmax)      qmax = 2^(bits-1) - 1
//! 还原：x' = q / qmax * s
//! ```
//!
//! 量化误差的上界是半个步长 `s / (2·qmax)`：int8 时约为 `|x|max / 254`，
//! int4 时约为 `|x|max / 14`——**比特数每少 1，误差翻倍**，这就是 int4 必须配合
//! 分组（每组一个 scale）的原因，也是 GPTQ / AWQ 这些"聪明的量化"存在的意义：
//! 同样的位宽下把误差压得更低。
//!
//! ## 为什么按"组"而不是整张矩阵一个 scale
//!
//! 一张 `[4096, 4096]` 的权重里，各行的动态范围可能差几个数量级。整张矩阵共用一个
//! `s` 时，小量级的那部分权重会被量化到只剩一两个码值，信息基本丢光。按行（每个输出
//! 通道一个 scale）或按组（每 128 个元素一个 scale）量化，误差会显著下降。
//! [`QAxis`] 就是"沿哪个方向分组"的开关。
//!
//! ## 本模块的边界
//!
//! 这里只放**数据结构与逐元素映射**：整数码的位打包、逐通道 / 逐行量化、
//! 增删行（KV cache 需要）、误差统计。上层算法（GPTQ 的 Hessian 误差补偿、
//! AWQ 的激活感知缩放）与量化推理内核在后面的小节里，与这里共用同一套容器。

use serde::{Deserialize, Serialize};

/// 量化位宽
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum QBits {
    /// 8 位有符号整数，码值范围 `[-127, 127]`
    Int8,
    /// 4 位有符号整数，码值范围 `[-7, 7]`（两个码打包进一个字节）
    Int4,
}

impl QBits {
    /// 量化码的绝对值上限 `qmax = 2^(bits-1) - 1`
    ///
    /// int4 取 7 而不是 8：`-8` 在 4 位二进制补码里没有对应的正数，
    /// 用对称范围 `[-7, 7]` 可以让正负两侧的量化步长严格相等。
    pub fn qmax(self) -> f32 {
        match self {
            QBits::Int8 => 127.0,
            QBits::Int4 => 7.0,
        }
    }

    /// 每个码占多少位
    pub fn bits(self) -> usize {
        match self {
            QBits::Int8 => 8,
            QBits::Int4 => 4,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            QBits::Int8 => "int8",
            QBits::Int4 => "int4",
        }
    }

    /// 是否两个码共用一个字节
    fn packed(self) -> bool {
        self.bits() == 4
    }
}

/// 分组方向：一个 scale 覆盖矩阵的哪一维
///
/// 设矩阵形状 `[rows, cols]`：
/// - [`QAxis::Row`]：**每行一个 scale**（共 `rows` 个）。行 = 输出通道，所以这就是
///   常说的 per-channel 权重量化。
/// - [`QAxis::Col`]：**每列一个 scale**（共 `cols` 个）。KIVI 对 KV cache 里的
///   **K** 用这个方向：K 的每一维（通道）在一个给定的注意力头里数值尺度稳定，
///   而不同 token 之间的差异主要体现在整体幅度上，按通道分组能把误差压得最低。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QAxis {
    Row,
    Col,
}

/// 量化后的二维矩阵：整数码（按位打包）+ 分组缩放因子
///
/// 数据布局与 [`crate::tensor::Tensor`] 一致（行优先展平），`bytes` 的长度是
/// `rows * cols * bits / 8`，即 int8 时与原数据同字节数、int4 时只有一半。
#[derive(Clone, Debug)]
pub struct QMatrix {
    bytes: Vec<u8>,
    /// 分组缩放因子：`QAxis::Row` 时长度 = rows，`QAxis::Col` 时长度 = cols
    scales: Vec<f32>,
    bits: QBits,
    axis: QAxis,
    rows: usize,
    cols: usize,
}

/// 打包后的字节数：int8 时一码一字节，int4 时两码一字节（奇数个码向上取整）
fn packed_len(n: usize, bits: QBits) -> usize {
    if bits.packed() {
        n.div_ceil(2)
    } else {
        n
    }
}

/// 把整数码打包成字节流（int8 直接按位存，int4 两个码一个字节：低半字节在前）
pub fn pack_codes(codes: &[i8], bits: QBits) -> Vec<u8> {
    match bits {
        QBits::Int8 => codes.iter().map(|&c| c as u8).collect(),
        QBits::Int4 => codes
            .chunks(2)
            .map(|pair| {
                let lo = (pair[0] as u8) & 0x0F;
                let hi = if pair.len() > 1 {
                    (pair[1] as u8) & 0x0F
                } else {
                    0
                };
                lo | (hi << 4)
            })
            .collect(),
    }
}

/// 从字节流还原 `n` 个整数码（int4 时做符号扩展，`0xF` -> `-1`）
pub fn unpack_codes(bytes: &[u8], bits: QBits, n: usize) -> Vec<i8> {
    let mut out = Vec::with_capacity(n);
    match bits {
        QBits::Int8 => {
            out.extend(bytes.iter().take(n).map(|&b| b as i8));
        }
        QBits::Int4 => {
            for &b in bytes {
                if out.len() >= n {
                    break;
                }
                out.push((((b & 0x0F) as i8) << 4) >> 4);
                if out.len() >= n {
                    break;
                }
                out.push((((b >> 4) as i8) << 4) >> 4);
            }
        }
    }
    out
}

/// 一维对称量化：返回 (整数码, scale)。全零输入时 scale 取 1，避免除零
pub fn quantize_row_sym(x: &[f32], bits: QBits) -> (Vec<i8>, f32) {
    let max_abs = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let scale = if max_abs > 0.0 { max_abs } else { 1.0 };
    let qmax = bits.qmax();
    let codes = x
        .iter()
        .map(|&v| (v / scale * qmax).round().clamp(-qmax, qmax) as i8)
        .collect();
    (codes, scale)
}

/// 按组（`QAxis`）算出每个分组的最大绝对值 —— 也就是该组的 scale。
///
/// 逐行分组时每行自带一个 scale，行内全零的 scale 取 1（该行反量化后必然是 0，
/// 不影响任何数值）。逐列分组返回**原始**列最大值，允许为 0——见 [`resolve_scales`]：
/// 0 在列方向上表示"这一列还没有数据可定标"，不能直接当成 scale 用。
pub fn group_scales(x: &[f32], rows: usize, cols: usize, axis: QAxis) -> Vec<f32> {
    match axis {
        QAxis::Row => (0..rows)
            .map(|r| {
                x[r * cols..(r + 1) * cols]
                    .iter()
                    .fold(0.0f32, |m, v| m.max(v.abs()))
            })
            .map(|s| if s > 0.0 { s } else { 1.0 })
            .collect(),
        QAxis::Col => (0..cols)
            .map(|c| {
                (0..rows)
                    .map(|r| x[r * cols + c].abs())
                    .fold(0.0f32, f32::max)
            })
            .collect(),
    }
}

/// 补齐"未定"的分组 scale（逐列分组才会出现）。
///
/// KV cache 的 K 是**逐列冻结 scale** 的（列 scale 全列共享，追加重算会让已写入的
/// 历史码值失效）。如果某一列在冻结那一刻恰好全为 0，就没法定标；把 0 当成真 scale
/// 会让这一列之后所有非零数据被压成 0，而当成 1.0 又会让量级稍大的数据直接裁剪到
/// 饱和——两种硬选都是错的。
///
/// 这里用第三条路：**0 表示"未定"**，等这一列第一次出现非零数据时再用它定标。
/// 这个改动是安全的：历史码值在这种列上必然全是 0（`0 / s * qmax = 0`），
/// 反量化恒为 0，换一个 scale 不会改变任何一个已写入的数值。
pub fn resolve_scales(x: &[f32], rows: usize, cols: usize, axis: QAxis, scales: &[f32]) -> Vec<f32> {
    if axis == QAxis::Row || !scales.iter().any(|s| *s <= 0.0) {
        return scales.to_vec();
    }
    let mut out = scales.to_vec();
    for c in 0..cols {
        if out[c] <= 0.0 {
            out[c] = (0..rows)
                .map(|r| x[r * cols + c].abs())
                .fold(0.0f32, f32::max);
        }
    }
    out
}

/// 用给定的分组 scale 做对称量化（不重新统计 max，便于"沿用已有 scale"，
/// 例如 KV cache 追加新行时必须用冻结的 scale，否则历史码值会与新 scale 不一致）。
/// 传入的 scale 里若还有"未定"的 0，先按 [`resolve_scales`] 补齐。
pub fn quantize_with_scales(
    x: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    scales: &[f32],
) -> Vec<i8> {
    let resolved = resolve_scales(x, rows, cols, axis, scales);
    let qmax = bits.qmax();
    let mut codes = vec![0i8; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = match axis {
                QAxis::Row => resolved[r],
                QAxis::Col => resolved[c],
            };
            codes[r * cols + c] = (x[r * cols + c] / s * qmax).round().clamp(-qmax, qmax) as i8;
        }
    }
    codes
}

/// 反量化：整数码 + 分组 scale -> f32
pub fn dequantize_with_scales(
    codes: &[i8],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    scales: &[f32],
) -> Vec<f32> {
    let qmax = bits.qmax();
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            let s = match axis {
                QAxis::Row => scales[r],
                QAxis::Col => scales[c],
            };
            out[r * cols + c] = codes[r * cols + c] as f32 / qmax * s;
        }
    }
    out
}

impl QMatrix {
    /// 量化一个 `[rows, cols]` 的 f32 矩阵（scale 由本矩阵的数值现算）
    pub fn quantize(x: &[f32], rows: usize, cols: usize, bits: QBits, axis: QAxis) -> Self {
        assert_eq!(x.len(), rows * cols, "量化输入长度与形状不符");
        let scales = group_scales(x, rows, cols, axis);
        Self::quantize_with_scales(x, rows, cols, bits, axis, &scales)
    }

    /// 用指定的 scale 量化（KV cache 追加行时复用已冻结的 scale）。
    /// "未定"的 scale（0，见 [`resolve_scales`]）会先补齐，**存的与用的保证是同一份**。
    pub fn quantize_with_scales(
        x: &[f32],
        rows: usize,
        cols: usize,
        bits: QBits,
        axis: QAxis,
        scales: &[f32],
    ) -> Self {
        assert_eq!(x.len(), rows * cols, "量化输入长度与形状不符");
        assert_eq!(
            scales.len(),
            match axis {
                QAxis::Row => rows,
                QAxis::Col => cols,
            },
            "scale 个数与分组方向不符"
        );
        let scales = resolve_scales(x, rows, cols, axis, scales);
        let codes = quantize_with_scales(x, rows, cols, bits, axis, &scales);
        QMatrix {
            bytes: pack_codes(&codes, bits),
            scales,
            bits,
            axis,
            rows,
            cols,
        }
    }

    /// 全零矩阵（KV cache 预分配用）
    pub fn zeros(rows: usize, cols: usize, bits: QBits, axis: QAxis) -> Self {
        let n_scales = match axis {
            QAxis::Row => rows,
            QAxis::Col => cols,
        };
        QMatrix {
            bytes: vec![0u8; packed_len(rows * cols, bits)],
            scales: vec![1.0; n_scales],
            bits,
            axis,
            rows,
            cols,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    pub fn bits(&self) -> QBits {
        self.bits
    }

    pub fn axis(&self) -> QAxis {
        self.axis
    }

    /// 分组缩放因子（只读）
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// 整数码的原始字节（只读）
    pub fn bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// 实际占用的字节数：整数码 + 缩放因子（**不含**结构体本身的开销）
    pub fn byte_len(&self) -> usize {
        self.bytes.len() + self.scales.len() * 4
    }

    /// 还原成完整的 f32 矩阵
    pub fn dequantize(&self) -> Vec<f32> {
        let codes = unpack_codes(&self.bytes, self.bits, self.rows * self.cols);
        dequantize_with_scales(&codes, self.rows, self.cols, self.bits, self.axis, &self.scales)
    }

    /// 还原第 `i` 行（只解这一行的码，避免为了取一行解整张矩阵）
    pub fn row(&self, i: usize) -> Vec<f32> {
        assert!(i < self.rows, "行下标 {i} 越界（共 {} 行）", self.rows);
        let codes = unpack_codes(&self.bytes, self.bits, self.rows * self.cols);
        let qmax = self.bits.qmax();
        (0..self.cols)
            .map(|c| {
                let s = match self.axis {
                    QAxis::Row => self.scales[i],
                    QAxis::Col => self.scales[c],
                };
                codes[i * self.cols + c] as f32 / qmax * s
            })
            .collect()
    }

    /// 在末尾追加 `new_rows` 行（列数必须一致）。
    ///
    /// 新行用**当前的 scale** 量化，不重算：
    /// - [`QAxis::Row`]：新行自己一行就是一个分组，可以顺带把自己的 scale 算准；
    /// - [`QAxis::Col`]：列 scale 是全列共享的，追加几行就改 scale 会让**已经写进去的
    ///   历史码值**与新 scale 不匹配。所以列方向沿用冻结的 scale（KIVI 的做法），
    ///   超出初始范围的新值被裁剪到 `qmax`——代价有界，收益是 O(1) 的元数据开销。
    pub fn push_rows(&mut self, x: &[f32], new_rows: usize) {
        assert_eq!(x.len(), new_rows * self.cols, "追加数据长度与形状不符");
        let mut codes = unpack_codes(&self.bytes, self.bits, self.rows * self.cols);
        match self.axis {
            QAxis::Row => {
                let new_scales = group_scales(x, new_rows, self.cols, QAxis::Row);
                let new_codes = quantize_with_scales(
                    x,
                    new_rows,
                    self.cols,
                    self.bits,
                    QAxis::Row,
                    &new_scales,
                );
                self.scales.extend_from_slice(&new_scales);
                codes.extend_from_slice(&new_codes);
            }
            QAxis::Col => {
                // 冻结的列 scale 里可能还留着"未定"的 0（冻结那一刻该列全为零）：
                // 这一批它终于有非零数据了，顺势定标，并把定下来的值写回，
                // 之后的行就都按它走。历史码值在这种列上全是 0，换 scale 不影响它们。
                self.scales = resolve_scales(x, new_rows, self.cols, QAxis::Col, &self.scales);
                let new_codes =
                    quantize_with_scales(x, new_rows, self.cols, self.bits, QAxis::Col, &self.scales);
                codes.extend_from_slice(&new_codes);
            }
        }
        self.rows += new_rows;
        self.bytes = pack_codes(&codes, self.bits);
    }

    /// 丢掉最前面的 `n` 行（KV cache 滑动窗口 / Attention Sink 用）
    pub fn drop_front_rows(&mut self, n: usize) {
        assert!(n <= self.rows, "要丢的行数 {n} 超过现有行数 {}", self.rows);
        if n == 0 {
            return;
        }
        let codes = unpack_codes(&self.bytes, self.bits, self.rows * self.cols);
        let kept = codes[n * self.cols..].to_vec();
        if self.axis == QAxis::Row {
            self.scales.drain(..n);
        }
        self.rows -= n;
        self.bytes = pack_codes(&kept, self.bits);
    }

    /// 丢掉最末尾的 `n` 行（推测解码回滚用，见 [`crate::attention::KVCache::rollback`]）
    pub fn drop_back_rows(&mut self, n: usize) {
        assert!(n <= self.rows, "要丢的行数 {n} 超过现有行数 {}", self.rows);
        if n == 0 {
            return;
        }
        let codes = unpack_codes(&self.bytes, self.bits, self.rows * self.cols);
        let kept = codes[..(self.rows - n) * self.cols].to_vec();
        // 逐 token（V）的 scale 每行一个，跟着行一起裁；逐通道（K）的 scale 列共享，不动
        if self.axis == QAxis::Row {
            self.scales.truncate(self.rows - n);
        }
        self.rows -= n;
        self.bytes = pack_codes(&kept, self.bits);
    }
}

/// 量化误差报告：相对误差用来判断"这个位宽 / 这个分组方式够不够"
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantError {
    /// 最大绝对误差
    pub max_abs: f32,
    /// 均方根误差
    pub rmse: f32,
    /// 相对误差 `rmse / rms(x)`（原信号的均方根）
    pub relative: f32,
}

/// 统计量化误差
pub fn quant_error(x: &[f32], xq: &[f32]) -> QuantError {
    assert_eq!(x.len(), xq.len(), "误差统计要求等长");
    let n = x.len().max(1) as f32;
    let max_abs = x
        .iter()
        .zip(xq)
        .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
    let mse = x
        .iter()
        .zip(xq)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / n;
    let rms = (x.iter().map(|v| v * v).sum::<f32>() / n).sqrt();
    QuantError {
        max_abs,
        rmse: mse.sqrt(),
        relative: if rms > 0.0 { mse.sqrt() / rms } else { 0.0 },
    }
}

// ==================== 量化算法：RTN / GPTQ / AWQ（第 33 课） ====================
//
// 三种算法解的是**同一个优化问题**，差别只在"用多少信息去挑格点"：
//
// ```text
// 目标：min_Ŵ  E_x[ ‖x·(W - Ŵ)‖² ]  =  tr((W - Ŵ)ᵀ H (W - Ŵ))，  H = E[xᵀx]
// ```
//
// 上式左边是"用 Ŵ 替代 W 之后，每层输出上的均方误差"——它才是真正决定模型质量的量，
// 而不是权重矩阵上的逐元素误差。三个算法的区别就落在这个目标函数的处理方式上：
//
// - [`QMethod::Rtn`]：假装 H = I（每个权重同等重要），逐元素取最近的格点。
// - [`QMethod::Gptq`]：把 H 真的算出来（前向一批校准数据得到 `XᵀX`），逐列量化时
//   用 `H⁻¹` 把当前列的误差按**相关性**折算成对后续未量化列的修正量，
//   于是后面的列可以"提前补偿"前面列的误差。
// - [`QMethod::Awq`]：不动误差传播，改**坐标**——按激活幅度把输入通道缩放一下，
//   让本来就重要的通道（激活大）在量化时落到更细的格点上。见 [`awq_scales`]。
//
// 三者都只量化权重（weight-only），激活保持 f32 参与矩阵乘：这样不需要任何
// 融合整数内核，只要在加载时把权重量化、推理时反量化，就能拿到"显存/带宽减半到减 3/4"
// 的收益（decode 阶段本来就是 memory-bound），数值损失却远小于同等位宽下量化激活。

/// 权重量化算法
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QuantMethod {
    /// Round-To-Nearest：逐列独立四舍五入，等价于 [`QMatrix::quantize`]
    Rtn,
    /// GPTQ：用输入的二阶统计 `H = XᵀX` 做逐列误差补偿
    Gptq,
    /// AWQ：按激活幅度对输入通道做感知缩放后再量化
    Awq,
}

impl QuantMethod {
    /// 命令行的名字（也是 checkpoint 头里记录的字符串形式）
    pub fn name(self) -> &'static str {
        match self {
            QuantMethod::Rtn => "rtn",
            QuantMethod::Gptq => "gptq",
            QuantMethod::Awq => "awq",
        }
    }

    /// 一句话原理（日志 / 报告用）
    pub fn describe(self) -> &'static str {
        match self {
            QuantMethod::Rtn => "逐列独立四舍五入，列与列之间不共享任何信息（无需校准数据）",
            QuantMethod::Gptq => "用 XᵀX 的逆把每个输入通道的量化误差按相关性分摊给后面未量化的通道，逐通道最小化加权误差",
            QuantMethod::Awq => "按激活幅度缩放输入通道（组内几何均值归一），让重要通道落到更细的量化格点上",
        }
    }

    /// 是否需要校准数据（RTN 只用权重本身）
    pub fn needs_calib(self) -> bool {
        match self {
            QuantMethod::Rtn => false,
            QuantMethod::Gptq | QuantMethod::Awq => true,
        }
    }
}

/// GPTQ 的选项。
///
/// 三个字段各管一件事，且都不是"可调着玩"的超参——它们对应算法里三个**必须显式做选择**
/// 的工程点：
/// - `act_order`（激活重要性重排序）：按 `diag(H)` 降序处理输入通道。`diag(H) = E[x²]`
///   就是该通道的激活能量，能量大的通道先量化、于是能享受到后面所有通道的误差补偿；
///   关掉它就是论文里的"naive 顺序"。默认开。
/// - `damp`：Cholesky 之前的阻尼系数，见 [`GPTQ_DAMP`]。
/// - `block`：误差补偿的批处理粒度（处理多少个输入通道后，把累积的补偿量一次性作用到
///   剩余通道上），见 [`gptq_quantize`] 的"分块"一节。0 表示不分块。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GptqOpts {
    /// 是否按 `diag(H)` 降序重排输入通道（act-order / activation ordering）
    pub act_order: bool,
    /// `H += damp · (trace(H)/n) · I` 的阻尼系数；`0` = 不加阻尼
    pub damp: f32,
    /// 分块大小（按输入通道数计）；`0` = 不分块（一次性处理全部通道）
    pub block: usize,
}

impl Default for GptqOpts {
    fn default() -> Self {
        GptqOpts {
            act_order: true,
            damp: GPTQ_DAMP,
            block: GPTQ_BLOCK,
        }
    }
}

impl GptqOpts {
    /// 日志用的一句话摘要
    pub fn describe(&self) -> String {
        format!(
            "act-order={}，damp={}，block={}",
            if self.act_order { "开" } else { "关" },
            self.damp,
            if self.block == 0 {
                "不分块".to_string()
            } else {
                self.block.to_string()
            }
        )
    }
}

/// 统一的量化选项：一次调用把"分组方向 + 各算法的参数"都带全。
///
/// 为什么要一个统一结构而不是给每个算法一个函数：模型级入口
/// （[`crate::model::GPT::quantize_weights`]）只有一条路径，它必须能把用户选的
/// **任意**算法连同该算法的参数一起传下去。分开写会让调用点出现"gptq 参数在
/// awq 时无意义"的分支，而分支里漏掉一个字段不会报错、只会静默用默认值。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantOpts {
    /// 分组方向：[`QAxis::Col`] = 每个输出通道一个 scale（per-channel 权重量化，默认），
    /// [`QAxis::Row`] = 每个输入通道一个 scale。
    pub axis: QAxis,
    /// GPTQ 的选项（其余算法忽略）
    pub gptq: GptqOpts,
    /// AWQ 的缩放指数 α：`Some(α)` 用给定值；`None` = 在 [`awq_alpha_grid`] 上
    /// 按权重+激活的代理误差搜索最优 α（逐层独立搜索）。
    pub awq_alpha: Option<f32>,
}

impl Default for QuantOpts {
    fn default() -> Self {
        QuantOpts {
            axis: QAxis::Col,
            gptq: GptqOpts::default(),
            awq_alpha: None,
        }
    }
}

/// 校准选项。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CalibOpts {
    /// 校准集最多喂多少 token（统计量按 token 累加，越多越准）
    pub max_tokens: usize,
    /// 一次前向喂多长（会被 `block_size` 截断）
    pub max_seq: usize,
    /// **存全矩阵 Hessian 的最大输入维度**：`in_features` 超过它时只累加对角。
    ///
    /// 取舍：全矩阵是 `[in, in]` 的 f32，内存 `4·in²` 字节/层（in = 1024 时 4 MB、
    /// 4096 时 64 MB），换来的是**通道之间的相关性**——GPTQ 的误差补偿完全依赖它；
    /// 对角模式只存 `in` 个数（内存 O(in)），但 `H⁻¹` 退化成对角阵，
    /// 补偿量恒为 0，GPTQ 随之退化为 RTN。所以这个阈值是"显存"与"精度"之间
    /// 唯一的一个旋钮：能放进内存就存全矩阵。
    pub full_hessian_max_dim: usize,
}

impl Default for CalibOpts {
    fn default() -> Self {
        CalibOpts {
            max_tokens: 512,
            max_seq: 128,
            full_hessian_max_dim: CALIB_FULL_HESSIAN_MAX_DIM,
        }
    }
}

/// 校准默认存全矩阵 Hessian 的输入维度上限：1024 → 每层 4 MB。
///
/// 取这个值是按"能跑起来"倒推的：本项目最大的配置 `n_embd = 1024`、12 层，
/// 全部层都存全矩阵也只是 ~50 MB，与本项目动辄几百 MB 的权重同量级；
/// 再大（4096 起）就该上对角模式了。
pub const CALIB_FULL_HESSIAN_MAX_DIM: usize = 1024;

/// GPTQ 默认的分块大小（输入通道数）。
///
/// 官方实现用的是 128：分块只影响"补偿量的批处理粒度"（每 128 个通道把累积的补偿
/// 一次性结算给剩余通道，矩阵乘更接近 BLAS-3 的形状），**不改变数学结果**——
/// 见 [`gptq_quantize`] 的证明与 `gptq_block_size_does_not_change_result` 测试。
pub const GPTQ_BLOCK: usize = 128;

/// GPTQ 的默认阻尼系数：`H += damp · (trace(H)/n) · I`。
///
/// 校准集只有几百个 token，而 `H` 是 `[cols, cols]`——`cols` 大时 `H` 会接近奇异
/// （某些通道的激活几乎线性相关），此时 `H⁻¹` 的元素会爆掉，误差补偿的量级会大到
/// 把未量化的列推飞。按"对角均值的 1%"做阻尼相当于给每个通道加一点点独立噪声，
/// 把 `H` 的最小特征值抬离 0，代价是补偿精度略降。1% 是 GPTQ 实现里的通行取值。
pub const GPTQ_DAMP: f32 = 0.01;

/// AWQ 的默认激活缩放指数 α：`s_j = (mean|x_j|)^α`。
///
/// α = 1 太激进（把通道重要性差异放大到一次方，小激活通道会被压到几乎无信息），
/// α = 0 等于不缩放。论文扫下来 0.5 最稳，也是各实现里的默认值。
pub const AWQ_ALPHA: f32 = 0.5;

/// AWQ 的默认分组大小：每多少条输入通道共享一次"几何均值归一"。
///
/// 128 与权重量化的常见分组一致；组内归一是为了让缩放**不改变权重的整体量级**——
/// 否则把全部通道一起放大 `k` 倍，量化 scale 同步放大 `k` 倍，等于什么都没做。
pub const AWQ_GROUP_SIZE: usize = 128;

/// Cholesky 分解：对行优先 `n×n` 对称矩阵 `A` 求下三角 `L`，使 `A = L·Lᵀ`。
///
/// GPTQ 需要 `H⁻¹`，而直接求逆不做分解的话既慢又容易在接近奇异时失去精度；
/// 走 `H = L·Lᵀ` 后：正定性检查天然落在对角元素上（`d ≤ 0` 立刻返回 `None`，
/// 不会算出 `NaN` 污染整张矩阵），求逆则退化成两次三角回代（[`cholesky_inverse_upper`]）。
///
/// 返回 `None` 的情形（调用方一律退回 RTN，不 panic）：矩阵非正定（对角 `≤ 0`）、
/// 含 NaN/Inf、或长度与 `n` 不符。断言只检查长度，因为它是**编码错误**；
/// 数值上的病态则属于运行时输入问题，用返回值表达。
pub fn cholesky(a: &[f32], n: usize) -> Option<Vec<f32>> {
    assert_eq!(a.len(), n * n, "Cholesky 输入必须是 n×n 的方阵");
    let mut l = vec![0.0f32; n * n];
    for j in 0..n {
        // 对角元素：A[j,j] 减去已经算出的 L[j,0..j] 的平方和 = L[j,j]²
        let mut d = a[j * n + j];
        for k in 0..j {
            d -= l[j * n + k] * l[j * n + k];
        }
        // `!(d > 0.0)` 同时挡住 d == 0、d < 0 与 NaN 三种情形
        if !(d > 0.0) {
            return None;
        }
        let ljj = d.sqrt();
        l[j * n + j] = ljj;
        for i in (j + 1)..n {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            l[i * n + j] = s / ljj;
        }
    }
    Some(l)
}

/// 由 Cholesky 因子 `L` 求 `A⁻¹ = (L·Lᵀ)⁻¹`，返回行优先的 `n×n` **完整对称**逆矩阵。
///
/// 名字里的 `upper` 指的是内部那步回代：先解 `L·Y = I` 得 `Y = L⁻¹`（下三角），
/// 再算 `A⁻¹ = Yᵀ·Y`——`Yᵀ` 是上三角，"用上三角因子做回代"正是这个式子的形状。
/// **返回值是两个三角都填好的对称矩阵**，调用方可以直接按 `inv[j * n + k]` 取用，
/// 不必再自己补对称（误差补偿里 `H⁻¹[j,k]` 会以任意下标顺序被访问，
/// 只填一半会让读到的值取决于索引顺序，那是极难查的错）。
///
/// `L` 的对角元素为 0（奇异）时该列直接留 0：与其算出 Inf 再污染后续所有列，
/// 不如让这一列的逆"缺省为 0"，使补偿量归零、退化成不补偿。
pub fn cholesky_inverse_upper(l: &[f32], n: usize) -> Vec<f32> {
    assert_eq!(l.len(), n * n, "Cholesky 因子必须是 n×n 的方阵");
    // 1) Y = L⁻¹：按列做前代（第 j 列只需 j..n 的元素）
    let mut y = vec![0.0f32; n * n];
    for j in 0..n {
        let ljj = l[j * n + j];
        if !(ljj.abs() > 0.0) {
            continue;
        }
        y[j * n + j] = 1.0 / ljj;
        for i in (j + 1)..n {
            let mut s = 0.0f32;
            for k in j..i {
                s += l[i * n + k] * y[k * n + j];
            }
            y[i * n + j] = -s / l[i * n + i];
        }
    }
    // 2) A⁻¹ = Yᵀ·Y。Y 的下三角结构让内层求和从 k 开始（j < k 时 Y[m,j] 恒为 0）
    let mut inv = vec![0.0f32; n * n];
    for i in 0..n {
        for k in i..n {
            let mut s = 0.0f32;
            for m in k..n {
                s += y[m * n + i] * y[m * n + k];
            }
            inv[i * n + k] = s;
            inv[k * n + i] = s;
        }
    }
    inv
}

/// 一步求出**加了阻尼**的 `H⁻¹`：阻尼 → [`cholesky`] 分解 → [`cholesky_inverse_upper`] 回代，
/// 三步串成一件调用方真正想要的事。非正定（含 NaN/Inf）返回 `None`，由调用方决定怎么降级。
///
/// 为什么要把"加阻尼"并进这个函数，而不是留给调用方自己改对角线：
/// 阻尼值只有在看到 `H` 的量级之后才定得下来（GPTQ 用的是相对量 `damp · trace(H)/n`，
/// 见 [`GPTQ_DAMP`]），于是"非正定就换一个更大的 damp 重试"这个循环，在调用点变成一行；
/// 更重要的是，**阻尼必须紧挨着分解**——中间隔开一步就很容易写出"加了阻尼却把没加阻尼的矩阵
/// 拿去分解"这种不报错的错。
///
/// `damp` 是加到对角线上的**绝对**量（`H[i][i] += damp`）。这里不替调用方折算相对口径，
/// 因为"按 trace/n 折算"是 GPTQ 的策略而非求逆的一部分（[`gptq_quantize`] 就是这么做的）；
/// 只想要"给个绝对兜底值"的调用方（例如已经算好 `H⁻¹`、要拿它连量化多层权重）
/// 不该被迫接受另一套口径。`damp` 非有限时按"不加"处理：NaN 阻尼没有任何意义，
/// 交回 Cholesky 依原始 `H` 判定，总好过算出一整张 NaN 矩阵。
pub fn cholesky_inverse(h: &[f32], n: usize, damp: f32) -> Option<Vec<f32>> {
    assert_eq!(h.len(), n * n, "求逆的输入必须是 n×n 的方阵");
    let mut damped = h.to_vec();
    if damp.is_finite() && damp > 0.0 {
        for i in 0..n {
            damped[i * n + i] += damp;
        }
    }
    // 非正定在这里变成 `None`（而不是 NaN 逆矩阵）：调用方加大 damp 重来，或退回 RTN，
    // 两条路都不会 panic（见 [`gptq_quantize_with_hinv`] 的退化路径）
    let l = cholesky(&damped, n)?;
    Some(cholesky_inverse_upper(&l, n))
}

/// 逐层校准统计：Hessian（`H = XᵀX`）与激活幅度的累加和。
///
/// GPTQ 与 AWQ 吃的是**同一批**校准数据（把校准文本喂进前向，逐层截住该层的输入激活 `x`），
/// 但需要的统计不同：GPTQ 要 `H = Σ_t x_t x_tᵀ`（`[n, n]`，含**输入通道之间**的相关性，
/// 补偿系数全部来自它），AWQ 要 `mean|x_i|`（`[n]`，通道重要性的唯一依据）。
/// 把两者放进同一个累加器，是因为它们天然由同一遍采集产生：分两遍跑会把校准前向的成本翻倍，
/// 而两遍之间只要有半点不一致（采样长度改了、窗口换了），拿到的就是两份互不匹配的统计——
/// 那种偏差不报任何错，只会让 GPTQ 的补偿方向悄悄偏掉。
///
/// 存的是**累加和**而不是均值：均值要等数据采完才能算（阻尼、归一化都依赖总量），
/// 而分批采集（[`Self::observe`]）与并行采集（各线程各采一份再 [`Self::merge`]）
/// 都要求中间态是**可加的**；token 数一起记着，均值随要随算
/// （[`Self::mean_hessian`] / [`Self::act_abs_mean`]）。
#[derive(Clone, Debug, PartialEq)]
pub struct Calibration {
    /// 行优先 `[n, n]` 的 `H = Σ_t x_t x_tᵀ`（累加和，两个三角始终一致）
    hessian: Vec<f32>,
    /// 每个输入通道的 `Σ_t |x_t[i]|`（累加和，长度 n）
    act_abs_sum: Vec<f32>,
    /// 参与统计的 token 数（`observe` 累加、`merge` 相加）
    n_tokens: usize,
}

impl Calibration {
    /// 空的统计：`H = 0`、`Σ|x| = 0`、token 数 0。`in_features` 就是被量化那一维的长度
    pub fn zeros(in_features: usize) -> Self {
        Calibration {
            hessian: vec![0.0f32; in_features * in_features],
            act_abs_sum: vec![0.0f32; in_features],
            n_tokens: 0,
        }
    }

    /// 输入维度（= 校准数据的最后一维 = 权重矩阵的行数）
    pub fn in_features(&self) -> usize {
        self.act_abs_sum.len()
    }

    /// 累加一批校准激活：`x` 是 `[rows, in_features]` 行优先（该层的一次前向结果）。
    ///
    /// 累加 `H += Xᵀ·X` 时两个三角**一起写**：`H` 必须自始至终是对称的，
    /// 而补偿系数 `H⁻¹[j,k]` 会以任意下标顺序被访问（见 [`cholesky_inverse_upper`] 的说明），
    /// "只填一半、用的时候再补"会让读到的值取决于索引顺序，是极难查的一类错。
    /// 对角项单独加：`k == j` 时两个下标指向同一个元素，走下面的写法会把平方项算两遍。
    pub fn observe(&mut self, x: &[f32], rows: usize) {
        let n = self.in_features();
        assert_eq!(
            x.len(),
            rows * n,
            "校准数据的元素数必须等于 rows × in_features"
        );
        for t in 0..rows {
            let row = &x[t * n..(t + 1) * n];
            for j in 0..n {
                let vj = row[j];
                self.act_abs_sum[j] += vj.abs();
                self.hessian[j * n + j] += vj * vj;
                for k in (j + 1)..n {
                    // 一次乘法供两个对称位置使用：`v_j·v_k` 与 `v_k·v_j` 在 f32 下逐位相同，
                    // 于是两个三角的累加结果也逐位相同（对称性可以按 `==` 断言）
                    let p = vj * row[k];
                    self.hessian[j * n + k] += p;
                    self.hessian[k * n + j] += p;
                }
            }
        }
        self.n_tokens += rows;
    }

    /// 参与统计的 token 数。它同时是"校准够不够"的依据：token 太少时 `H` 估计不准，
    /// 采样噪声会被 `H⁻¹` 放大到补偿量上
    pub fn n_tokens(&self) -> usize {
        self.n_tokens
    }

    /// 累积的 `H = Σ_t x_t x_tᵀ`（**未**除以 token 数）
    pub fn hessian(&self) -> &[f32] {
        &self.hessian
    }

    /// `H` 的均值 `E[x xᵀ]`（`[n, n]` 行优先）。一个 token 都没采到时返回全 0（而不是 NaN）
    pub fn mean_hessian(&self) -> Vec<f32> {
        if self.n_tokens == 0 {
            return vec![0.0f32; self.hessian.len()];
        }
        let inv = 1.0 / self.n_tokens as f32;
        self.hessian.iter().map(|v| v * inv).collect()
    }

    /// 逐输入通道的平均激活幅度 `E[|x_i|]`（长度 n）。一个 token 都没采到时返回全 0
    pub fn act_abs_mean(&self) -> Vec<f32> {
        if self.n_tokens == 0 {
            return vec![0.0f32; self.act_abs_sum.len()];
        }
        let inv = 1.0 / self.n_tokens as f32;
        self.act_abs_sum.iter().map(|v| v * inv).collect()
    }

    /// 合并另一份统计（同一层、在**别的**数据上采出来的）。
    ///
    /// 为什么累加器必须是可合并的：校准集既可以按 batch 并行采集（每个线程一份本地
    /// `Calibration`，最后归并），也可以分几段数据分别喂（先通用语料、再领域语料）。
    /// 累加和是这两条路都成立的中立形式——均值不能直接相加（分母不同）。
    /// 维度不一致属于编码错误（说明两份统计来自不同的层或不同的配置），直接断言。
    pub fn merge(&mut self, other: &Calibration) {
        assert_eq!(
            self.in_features(),
            other.in_features(),
            "合并的两份校准统计维度不同"
        );
        for (a, b) in self.hessian.iter_mut().zip(&other.hessian) {
            *a += *b;
        }
        for (a, b) in self.act_abs_sum.iter_mut().zip(&other.act_abs_sum) {
            *a += *b;
        }
        self.n_tokens += other.n_tokens;
    }
}

/// Hessian 的两种表示。省内存与保精度之间的那个开关只有这一个。
///
/// 校准采集时按 [`CalibOpts::full_hessian_max_dim`] 决定用哪种：
/// - [`HessianKind::Full`]：`[n, n]` 的完整 `H`，含**通道之间的相关性**——
///   GPTQ 的误差补偿系数 `H⁻¹[i, k]`（i ≠ k）全部来自它；
/// - [`HessianKind::Diagonal`]：只存 `diag(H)`。此时 `H⁻¹` 也是对角阵，
///   跨通道补偿系数恒为 0，GPTQ 在数学上退化成 RTN（顺序重排也失效——
///   对角 H 下加权误差是可分的）。它存在的唯一理由是内存：O(n) 而不是 O(n²)。
///   AWQ 不受影响（它只用 `diag(H) = E[x²]` 这一路信息）。
#[derive(Clone, Debug, PartialEq)]
pub enum HessianKind {
    /// 行优先 `[n, n]` 的完整 `H = E[xᵀx]`（n = 输入维度）
    Full(Vec<f32>),
    /// 只存 `diag(H)`（长度 n）
    Diagonal(Vec<f32>),
}

impl HessianKind {
    /// `H` 的阶数（= 被量化的那一维的长度 = 输入维度）
    pub fn dim(&self) -> usize {
        match self {
            HessianKind::Full(h) => {
                let n = (h.len() as f64).sqrt();
                let n = n.round() as usize;
                assert_eq!(n * n, h.len(), "全矩阵 Hessian 的元素数必须是完全平方数");
                n
            }
            HessianKind::Diagonal(d) => d.len(),
        }
    }

    /// 对角元素（act-order 的排序依据、阻尼的基准）
    pub fn diagonal(&self) -> Vec<f32> {
        match self {
            HessianKind::Full(h) => {
                let n = self.dim();
                (0..n).map(|i| h[i * n + i]).collect()
            }
            HessianKind::Diagonal(d) => d.clone(),
        }
    }

    /// 是否存了完整的（非对角）信息
    pub fn is_full(&self) -> bool {
        matches!(self, HessianKind::Full(_))
    }

    /// 这份统计实际占用的字节数（Hessian 通常是校准阶段的内存大头）
    pub fn byte_len(&self) -> usize {
        match self {
            HessianKind::Full(h) => h.len() * 4,
            HessianKind::Diagonal(d) => d.len() * 4,
        }
    }
}

/// act-order 的处理顺序：按 `diag(H)` 降序给出输入通道下标。
///
/// 为什么要排序：Algorithm 1 是**顺序**的——第 i 个通道量化完之后，它的误差会被
/// 转嫁给所有 k > i 的通道；反过来说，先被量化的通道只能"自己承担"误差，后被量化的
/// 通道能吃到前面所有通道的补偿。`diag(H) = E[x²]` 大的通道对输出的贡献大，
/// 让它排在后面、多吃一点补偿，同样的位宽下加权误差能再降一截；`act_order = false`
/// 时保持原顺序（论文里的 naive 版本，用来做消融对照）。
///
/// 排序用 `sort_by` 而不是手写快排：通道数最多几千，且这一步在离线量化里只跑一次。
fn order_by_importance(diag: &[f32], act_order: bool) -> Vec<usize> {
    let mut order: Vec<usize> = (0..diag.len()).collect();
    if !act_order {
        return order;
    }
    // `partial_cmp` 可能返回 None（NaN）：把 NaN 当成"最不重要"排到最后，
    // 与"NaN 的激活能量无法定序"这一事实一致，且不会 panic。
    order.sort_by(|&a, &b| {
        let (x, y) = (diag[a], diag[b]);
        y.partial_cmp(&x).unwrap_or(std::cmp::Ordering::Equal)
    });
    order
}

/// 按 `perm` 行/列同步重排对称矩阵：`Hp[i][j] = H[perm[i]][perm[j]]`。
///
/// H 与权重行必须用**同一个** `perm`，否则补偿系数会配到别的通道上——
/// 这种错不会报错，只会让 GPTQ 的效果退化（甚至变差），是最难查的一类偏差。
fn permute_sym(h: &[f32], n: usize, perm: &[usize]) -> Vec<f32> {
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = h[perm[i] * n + perm[j]];
        }
    }
    out
}

/// GPTQ 量化：在 `H = E[xᵀx]` 的加权误差意义下逐个输入通道挑格点，并把当前误差补偿给后续通道。
///
/// ## 参数与形状
///
/// - `w`：行优先 `[rows, cols]` 的权重。本项目的 [`crate::layers::Linear`] 存的是
///   `[in_features, out_features]`，所以 `rows` = **输入维度**、`cols` = **输出通道**。
/// - `hessian`：**输入维度**上的 `H = E[xᵀx]`（把校准集喂进前向、逐 token 累加 `x⊗x`
///   得到，见 [`crate::model::GPT::calibrate`]）。必须落在输入维度上：被量化的单元是
///   "某一个输入通道在全部输出通道上的权重"（矩阵一整行），而 `H⁻¹` 描述的是**输入通道
///   之间**的相关性，两者同维，补偿系数才有意义。
/// - `opts`：见 [`GptqOpts`]（act-order / damp / block）。
///
/// ## 算法（GPTQ 论文 Algorithm 1）
///
/// ```text
/// Ŵ = W
/// 对每个输入通道 i（按 act-order 的顺序）：
///   1. 用当前（已被前面通道补偿过的）W[i, :] 定 scale 并量化 → ŵ_i，误差 e = W[i, :] - ŵ_i
///   2. 对 k > i：W[k, :] -= e / H⁻¹[i, i] · H⁻¹[i, k]
/// ```
///
/// 第 2 步的系数来自"在 `w_i` 已被量化固定的约束下，其余权重如何调整才让总加权误差最小"
/// 的一阶条件（OBQ/OBS 的闭式解）。直觉：若 `H⁻¹[i,k]` 很大，说明第 i、k 两个输入通道
/// 高度相关，那么第 i 行的量化误差对输出造成的偏差，可以近似由第 k 行反向抵消一部分；
/// 于是把这份误差按比例"转嫁"到还没量化的第 k 行上，第 k 行再来量化时就会吸收掉它。
/// 逐行推进，前面通道的误差被后面通道一路消化，最终 `tr((W-Ŵ)ᵀH(W-Ŵ))` 显著低于 RTN。
///
/// ## 分块（`opts.block`）：只是批处理粒度，不改结果
///
/// 官方实现按 `block` 个输入通道切成一段段处理，块内立即互相补偿，块末把"（块进入时的值
/// − 块最终值）"一次性结算给块外的通道。它等价于不分块的顺序版本，理由是：
///
/// - 块内第 i 行在"被量化那一刻"的取值，与顺序版本完全一样——进入本块时它已经被所有
///   **更早的块**补偿过了（上一块末尾的结算覆盖了它），而块内更早的行也按同一公式即时补偿了它；
/// - 于是块内每一行的误差 `e_i / H⁻¹[i,i]` 与顺序版本逐位相同，本块对块外通道 k 的总影响
///   `-Σ_{i∈block} e_i·H⁻¹[i,k]/H⁻¹[i,i]` 也就与"顺序版本把这些更新逐个做掉"完全一致。
///
/// 换句话说：分块把 `block` 次"逐行更新大矩阵"换成一次按块累积、块末一次性结算，
/// 数值上（浮点累加顺序之外）与顺序版等价。若真把补偿**限制在块内**（丢掉块末结算），
/// 就等于对每个块各自独立做 Algorithm 1，块与块之间的相关性信息被白白扔掉。
///
/// ## 工程取舍
///
/// - 复杂度 `O(rows³)`（求逆）+ `O(rows²·cols)`（补偿），全部发生在**离线量化**阶段，
///   与每次前向无关；收益是推理期每步都省下的权重量读带宽。
/// - `H` 病态（含 NaN、全零、非正定）或只给了对角 Hessian 时，`hinv` 取 `None`，
///   流程继续但**不做任何补偿**：得到的码值与 RTN 逐位相同（降级，不是失败）。
///   量化是部署前的最后一步，在这里 panic 会让整条流水线卡死。
/// - [`QAxis::Row`] 分组时行 scale 随行（输入通道）走，正是被量化的那个单元，可以现算；
///   [`QAxis::Col`] 分组时列 scale 跨行共享，必须**冻结**（取原始权重逐列最大值）：
///   若每行都重算它，前面行的补偿会连带改掉后面行的 scale，补偿量就失去了意义；
///   冻结之后每行里各元素的步长与 RTN 完全一致，补偿的收益才可归因于算法本身。
pub fn gptq_quantize(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    hessian: &HessianKind,
    opts: &GptqOpts,
) -> QMatrix {
    assert_eq!(w.len(), rows * cols, "GPTQ 输入长度与形状不符");
    assert_eq!(
        hessian.dim(),
        rows,
        "Hessian 必须落在输入维度上，形状 [rows, rows]"
    );
    if rows == 0 || cols == 0 {
        return QMatrix::quantize(w, rows, cols, bits, axis);
    }
    // 1) act-order：按 diag(H) 降序给出输入通道的处理顺序
    let perm = order_by_importance(&hessian.diagonal(), opts.act_order);
    // 2) H⁻¹（只有全矩阵模式才可能有非零的非对角项）
    let hinv = match hessian {
        HessianKind::Diagonal(_) => None,
        HessianKind::Full(h) => {
            let hp = permute_sym(h, rows, &perm);
            // 阻尼：按 trace/n 的固定比例给对角线加一个正数，把 H 的最小特征值抬离 0。
            // trace 非有限（NaN/Inf）时不给阻尼——加了也是 NaN，不如让 Cholesky 直接判定失败。
            let trace = (0..rows).map(|i| hp[i * rows + i]).sum::<f32>();
            let damp_term = if trace.is_finite() && trace > 0.0 && opts.damp.is_finite() {
                opts.damp.max(0.0) * trace / rows as f32
            } else {
                0.0
            };
            let mut damped = hp;
            for i in 0..rows {
                damped[i * rows + i] += damp_term;
            }
            // 非正定 / 奇异 / 含 NaN：不做补偿（`None`），流程照常走完并返回合法码值
            cholesky(&damped, rows).map(|l| cholesky_inverse_upper(&l, rows))
        }
    };
    let block = if opts.block == 0 { rows } else { opts.block.min(rows) };
    gptq_apply(w, rows, cols, bits, axis, &perm, hinv.as_deref(), block)
}

/// GPTQ 的主体：按 `perm` 给定的顺序逐行（输入通道）量化，并把每行的量化误差按 `H⁻¹`
/// 补偿给**尚未量化**的行。`perm` / `hinv` / `block` 由调用方决定，
/// 于是"从 Hessian 出发"（[`gptq_quantize`]）与"从现成的 `H⁻¹` 出发"
/// （[`gptq_quantize_with_hinv`]）这两条入口共用同一份补偿逻辑——
/// 误差补偿的公式只该有一处实现，否则两个入口的数值行为会悄悄分叉。
///
/// `hinv` 为 `None` 表示"不做补偿"：每一行都按它进入时的取值直接量化，
/// 结果与 [`QMatrix::quantize`] 逐位相同（降级路径，见 [`gptq_quantize_with_hinv`]）。
/// 长度断言只加在这里：两个公开入口都已经保证 `rows ≥ 1`，且 `perm` 由本模块内部产生。
fn gptq_apply(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    perm: &[usize],
    hinv: Option<&[f32]>,
    block: usize,
) -> QMatrix {
    assert_eq!(w.len(), rows * cols, "GPTQ 输入长度与形状不符");
    assert_eq!(perm.len(), rows, "置换的长度必须等于输入通道数");
    if let Some(inv) = hinv {
        assert_eq!(inv.len(), rows * rows, "H⁻¹ 必须落在输入维度上，形状 [rows, rows]");
    }
    // block == 0 会让下面的分块循环原地打转（i2 == i1，i1 永不前进）
    let block = block.max(1);

    let qmax = bits.qmax();
    let mut scales: Vec<f32> = match axis {
        QAxis::Row => vec![1.0; rows],
        // 逐列分组：列 scale 跨行共享，从**原始**权重上冻结（perm 不影响列，无需重排）
        QAxis::Col => group_scales(w, rows, cols, QAxis::Col),
    };
    // 工作副本按 perm 重排行；原始 w 保持不动（列 scale 冻结时要读原值）
    let mut work = vec![0.0f32; rows * cols];
    for (i, &p) in perm.iter().enumerate() {
        work[i * cols..(i + 1) * cols].copy_from_slice(&w[p * cols..(p + 1) * cols]);
    }

    let mut i1 = 0;
    while i1 < rows {
        let i2 = (i1 + block).min(rows);
        // 本块对"块外尚未处理的通道"的累积补偿量（块末一次性结算，见函数文档）
        let mut carry = vec![0.0f32; (rows - i2) * cols];
        for i in i1..i2 {
            let row: Vec<f32> = work[i * cols..(i + 1) * cols].to_vec();
            // 1) 量化第 i 行。逐行分组时整行一个 scale（在补偿后的值上重新定标，因为补偿
            //    改变了这一行的动态范围）；逐列分组时每个元素用自己那一列的冻结 scale。
            let (codes, row_scale) = if axis == QAxis::Row {
                let (c, s) = quantize_row_sym(&row, bits);
                scales[i] = s;
                (c, s)
            } else {
                let c: Vec<i8> = (0..cols)
                    .map(|c| (row[c] / scales[c] * qmax).round().clamp(-qmax, qmax) as i8)
                    .collect();
                (c, 0.0)
            };
            let scale_of = |c: usize| if axis == QAxis::Row { row_scale } else { scales[c] };
            // 本行已经定案：工作副本写回"量化后的值"。这样循环结束时 work 的每一行都恰好
            // 等于它被量化时的取值，最后交给 quantize_with_scales 组装的码值与上面逐个
            // 算出的码值逐位一致（表示与 RTN 完全同构，只是取值不同）。
            let deq: Vec<f32> = (0..cols).map(|c| codes[c] as f32 / qmax * scale_of(c)).collect();
            for c in 0..cols {
                work[i * cols + c] = deq[c];
            }
            // 2) 误差 e = w_i - ŵ_i，按 H⁻¹[i,i] 归一化后分摊给 k > i 的行
            let Some(hinv) = &hinv else { continue };
            let hii = hinv[i * rows + i];
            if !(hii.is_finite() && hii.abs() > 0.0) {
                continue;
            }
            let err: Vec<f32> = (0..cols).map(|c| (row[c] - deq[c]) / hii).collect();
            for k in (i + 1)..i2 {
                let factor = hinv[i * rows + k];
                if factor != 0.0 {
                    for c in 0..cols {
                        work[k * cols + c] -= err[c] * factor;
                    }
                }
            }
            for k in i2..rows {
                let factor = hinv[i * rows + k];
                if factor != 0.0 {
                    let base = (k - i2) * cols;
                    for c in 0..cols {
                        carry[base + c] -= err[c] * factor;
                    }
                }
            }
        }
        // 块末结算：把本块所有行的补偿一次性加到块外通道上
        for (idx, v) in carry.iter().enumerate() {
            work[i2 * cols + idx] += *v;
        }
        i1 = i2;
    }

    // 3) 还原输入通道顺序，组装量化表示
    let mut back = vec![0.0f32; rows * cols];
    for (i, &p) in perm.iter().enumerate() {
        back[p * cols..(p + 1) * cols].copy_from_slice(&work[i * cols..(i + 1) * cols]);
    }
    // 逐行分组（[`QAxis::Row`]）时 scale 是**按行**存的，必须跟着 perm 还原回原行号；
    // 逐列分组（[`QAxis::Col`]）时 scale 挂在输出通道上，与输入通道的置换无关，
    // 原样交回即可——**不能**也按 perm 重排（长度是 cols 而不是 rows，越界）。
    let scales = if axis == QAxis::Row {
        let mut back_s = vec![0.0f32; rows];
        for (i, &p) in perm.iter().enumerate() {
            back_s[p] = scales[i];
        }
        back_s
    } else {
        scales
    };
    QMatrix::quantize_with_scales(&back, rows, cols, bits, axis, &scales)
}

/// 用**已经算好的** `H⁻¹` 做 GPTQ：`h_inv` 是输入维度上的 `[rows, rows]` 逆矩阵（行优先）。
///
/// 与 [`gptq_quantize`] 的分工：后者从 [`HessianKind`] 出发，把"阻尼、act-order、分块"
/// 三件策略一起包办；这里把**求逆这一步交给调用方**——它可能已经拿着 `H⁻¹`
/// （例如来自 [`cholesky_inverse`]，或据 [`Calibration::mean_hessian`] 自己加了别的正则），
/// 也可能要用同一份逆矩阵连量化多层权重，那就没必要再求一次逆（求逆是 `O(rows³)` 的，
/// 而"同一层 Hessian"在跨 rank / 跨分组方向对比时会被反复用到）。
///
/// 因此本函数**不做** act-order（严格按原始输入通道顺序）、**不分块**，也**不再加阻尼**
/// ——阻尼是求逆之前的事，见 [`cholesky_inverse`]。要这些策略就用 [`gptq_quantize`]。
///
/// 方向约定（与全模块一致，写死在一处以免误配）：`w` 是 `[rows, cols]` 行优先，
/// 本项目的 [`crate::layers::Linear`] 存 `[in_features, out_features]`，于是
/// **`rows` = 输入维度 = 被量化的那一维**，`h_inv` 必须落在 `rows` 上。
/// 被量化的单元是"某一个输入通道在全部输出通道上的权重"（矩阵一整行），
/// 而 `H⁻¹` 描述的是输入通道之间的相关性，两者同维，补偿系数才有意义。
///
/// 退化路径（刻意的，不是失败）：`h_inv` 为空、或长度不等于 `rows·rows` 时不做任何补偿，
/// 结果与 [`QMatrix::quantize`] **逐位相同**。量化是部署前的最后一步，
/// "校准没采到 / 维度对不上"在这里 panic 会把整条流水线卡死；把判断写成返回值语义
/// （空 = 不补偿）而不是断言，调用方也就多了一条"先试 GPTQ，不行就 RTN"的现成退路。
pub fn gptq_quantize_with_hinv(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    h_inv: &[f32],
) -> QMatrix {
    assert_eq!(w.len(), rows * cols, "GPTQ 输入长度与形状不符");
    if rows == 0 || cols == 0 {
        return QMatrix::quantize(w, rows, cols, bits, axis);
    }
    // 长度不符即视为"没有 Hessian"：调用方给错维度的后果应当是退化，而不是 panic
    let hinv = (h_inv.len() == rows * rows).then_some(h_inv);
    // 恒等置换 + 不分块：低层入口不做策略，只把补偿算准
    let perm: Vec<usize> = (0..rows).collect();
    gptq_apply(w, rows, cols, bits, axis, &perm, hinv, rows)
}

/// AWQ 的逐输入通道缩放系数：`s_j = (mean|x_j|)^α`，再按 `group_size` 分组归一到组内**几何均值 = 1**。
///
/// `act_abs_mean` 按**输入通道**排列（长度 = 权重的行数 = 输入的最后一维）。
///
/// 为什么是几何均值而不是算术均值：`s` 会**乘进权重**（[`awq_fold_weight`]），
/// 量化 scale 与 `s` 成正比，于是误差也随 `s` 线性放大。用几何均值归一时，
/// "放大一个通道"与"缩小另一个通道"在组内是**对称**的（乘性中心），
/// 组内总的对数量级不变；用算术均值则会让组内尺度整体往上飘。
///
/// `mean|x_j|` 为 0 的通道（该通道在校准集里从未被激活）用一个极小正数兜底：
/// 它的重要性确实是 0，但 `s = 0` 会让权重整列被压成 0、并且归一化时除零。
pub fn awq_scales(act_abs_mean: &[f32], alpha: f32, group_size: usize) -> Vec<f32> {
    let n = act_abs_mean.len();
    if n == 0 {
        return Vec::new();
    }
    let g = group_size.max(1);
    // 几何均值是"对数的算术均值再取指数"：用 log 域做归一既避免连乘下溢，
    // 也让极小值（被兜底成 1e-8）不会被显式 0 拉成 0。
    const FLOOR: f32 = 1e-8;
    let log_s: Vec<f32> = act_abs_mean
        .iter()
        .map(|m| alpha * m.max(FLOOR).ln())
        .collect();
    let mut out = vec![1.0f32; n];
    for start in (0..n).step_by(g) {
        let end = (start + g).min(n);
        let mean_log = log_s[start..end].iter().sum::<f32>() / (end - start) as f32;
        for j in start..end {
            out[j] = (log_s[j] - mean_log).exp();
        }
    }
    out
}

/// 把输入通道缩放吸收进权重：`W'[i, j] = W[i, j] · s[i]`（行优先 `[rows, cols]`，
/// `rows` 是输入维度，所以第 `i` 行就是第 `i` 个输入通道）。
///
/// 与 [`awq_unfold_input`] 配对使用：`(x/s)·(W·s) = x·W`——缩放只是把"动态范围"
/// 从激活挪到权重上，数学上恒等，但量化的是权重，于是缩放的收益（重要通道占更宽的码值）
/// 就落在了量化误差上。
pub fn awq_fold_weight(w: &[f32], rows: usize, cols: usize, s: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "AWQ 权重长度与形状不符");
    assert_eq!(s.len(), rows, "AWQ 缩放向量长度必须等于输入通道数（权重的行数）");
    let mut out = w.to_vec();
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] *= s[r];
        }
    }
    out
}

/// 推理时把输入除回缩放：`x'[i, j] = x[i, j] / s[j]`（行优先 `[n, cols]`）。
///
/// 必须与量化时的 [`awq_fold_weight`] 严格配对：少做这一步，等于把输入整体乘了 `s`，
/// 输出会随通道缩放而系统性偏移——这种错不会报任何异常，只会让困惑度悄悄变差。
pub fn awq_unfold_input(x: &[f32], cols: usize, s: &[f32]) -> Vec<f32> {
    assert_eq!(s.len(), cols, "AWQ 缩放向量长度必须等于输入通道数");
    assert_eq!(x.len() % cols, 0, "输入元素数必须是通道数的整数倍");
    let mut out = x.to_vec();
    for row in out.chunks_mut(cols) {
        for (v, sc) in row.iter_mut().zip(s) {
            *v /= *sc;
        }
    }
    out
}

/// AWQ 量化：先按激活幅度折叠缩放，再在**缩放后的权重**上做 RTN，
/// 返回 `(量化矩阵, 需要随权重一起保存的输入缩放向量)`。
///
/// 返回的 `s` 必须存进 [`QuantWeight::input_scale`]：权重被缩放了 `s` 倍，
/// 推理时输入就得分摊 `1/s`，否则前向结果整体错位。
pub fn awq_quantize(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    act_abs_mean: &[f32],
    alpha: f32,
) -> (QMatrix, Vec<f32>) {
    let s = awq_scales(act_abs_mean, alpha, AWQ_GROUP_SIZE);
    let folded = awq_fold_weight(w, rows, cols, &s);
    (QMatrix::quantize(&folded, rows, cols, bits, axis), s)
}

/// 把输入通道缩放从权重里**除**回去：`W'[i, j] = W[i, j] / s[i]`（[`awq_fold_weight`] 的逆）。
///
/// 推理路径上这一步由 [`awq_unfold_input`] 落在激活一侧（数学上等价、且不额外改权重）；
/// 这里的权重版用于**评估 AWQ 的等效误差**：量化后的权重只有折算回原始坐标
/// （`(s·W)_q / s`）才能和原始 `W` 直接比较。若不折算，误差里会混进 `s` 带来的
/// 系统性缩放，α 越小（`s` 越接近 1）看起来就越"准"，搜索就会系统性偏向 α=0。
pub fn awq_unfold_weight(w: &[f32], rows: usize, cols: usize, s: &[f32]) -> Vec<f32> {
    assert_eq!(w.len(), rows * cols, "AWQ 权重长度与形状不符");
    assert_eq!(s.len(), rows, "AWQ 缩放向量长度必须等于输入通道数（权重的行数）");
    let mut out = w.to_vec();
    for r in 0..rows {
        for c in 0..cols {
            out[r * cols + c] /= s[r];
        }
    }
    out
}

/// AWQ 缩放指数 α 的搜索网格：`0, 0.05, 0.10, ..., 1.0`。
///
/// α 是 AWQ 唯一的超参（`s = mean|x|^α`），论文扫的就是 `[0, 1]`：α = 0 等于不缩放、
/// α = 1 把通道重要性差异放大到一次方。文献给的 0.5 是"多数层都还行"的折中值，
/// 但**逐层最优的 α 并不相同**——某些层的激活分布很平，α=1 反而更差。网格步长 0.05
/// 是精度与离线耗时的折中（21 个候选，每个候选要多跑一遍量化 + 反量化）。
pub const AWQ_ALPHA_GRID_STEP: f32 = 0.05;

/// 生成 AWQ 的 α 搜索网格（含两端点）
pub fn awq_alpha_grid() -> Vec<f32> {
    let steps = (1.0 / AWQ_ALPHA_GRID_STEP).round() as usize;
    (0..=steps).map(|i| i as f32 * AWQ_ALPHA_GRID_STEP).collect()
}

/// AWQ 的 α 搜索：在网格上按**代理误差**挑最优缩放指数，返回 `(量化结果, 缩放向量, α)`。
///
/// 代理误差就是"缩放 + 量化 + 折回原坐标"之后与原始权重的偏差：
///
/// ```text
/// err(α) = ‖ quantize(W · diag(s(α))) / diag(s(α)) − W ‖
/// ```
///
/// 为什么这个代理是合理的：AWQ 的前提是"权重误差按输出通道独立作用"，于是
/// **权重上的等效误差**是激活侧误差的可乘性上界（`‖Δy‖ ≤ ‖x/s‖·‖diag(s)·ΔW‖` 的
/// 逐通道版本）。直接拿它当目标函数，就不需要为每个 α 都真的跑一遍校准集前向——
/// 搜索成本从"21 次全模型推理"降到"21 次单层量化"，这是 AWQ 论文里
/// "不依赖反向传播/不依赖逐层重构"的同一路思路。
///
/// `act_abs_mean` 长度必须等于 `rows`（输入通道数）；`grid` 为空时用 [`awq_alpha_grid`]。
pub fn awq_quantize_search(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    act_abs_mean: &[f32],
    grid: &[f32],
) -> (QMatrix, Vec<f32>, f32) {
    let grid = if grid.is_empty() {
        awq_alpha_grid()
    } else {
        grid.to_vec()
    };
    let mut best: Option<(QMatrix, Vec<f32>, f32, f32)> = None;
    for &alpha in &grid {
        let (q, s) = awq_quantize(w, rows, cols, bits, axis, act_abs_mean, alpha);
        // 折回原坐标后再比：这样比较的才是"最终会作用在同一个输入上的两个权重"
        let equiv = awq_unfold_weight(&q.dequantize(), rows, cols, &s);
        let err = quant_error(w, &equiv).relative;
        // 并列时保留先出现的（α 更小 = 缩放更温和），`<` 而非 `<=` 保证这一点
        if best.as_ref().is_none_or(|(_, _, _, e)| err < *e) {
            best = Some((q, s, alpha, err));
        }
    }
    // 元组第 4 项（代理误差）只用于选优，不往外传：报告里的误差由调用方在
    // 折算回原坐标的等效权重上重算一次，两处口径一致。
    let (q, s, alpha, _) =
        best.expect("α 网格不能为空（网格为空时会回落到 awq_alpha_grid，恒非空）");
    (q, s, alpha)
}

/// 校准批次上的输出平方误差 `‖X·W − X·Ŵ‖²_F`（行优先：`x` 是 `[n, rows]`，
/// `w`、`wq` 是 `[rows, cols]`）。这就是"同一批输入分别过原层与量化层，输出的差别有多大"。
///
/// 为什么逐个 token 各算两次矩阵乘，而不是先算误差矩阵 `E = W − Ŵ` 再算 `X·E`：
/// 后者在浮点上会把两个大数相减的抵消误差混进来（当 `W` 与 `Ŵ` 元素量级接近、
/// 而 `x` 又很小时尤甚），而这里要的是"部署后真正看到的偏差"。
/// 这也是 [`awq_best_alpha`] 与 [`awq_quantize_search`] 在目标函数上的分水岭。
fn output_error_sq(x: &[f32], w: &[f32], wq: &[f32], n: usize, rows: usize, cols: usize) -> f32 {
    let mut total = 0.0f32;
    for t in 0..n {
        let xt = &x[t * rows..(t + 1) * rows];
        for c in 0..cols {
            let (mut y, mut yq) = (0.0f32, 0.0f32);
            for j in 0..rows {
                y += xt[j] * w[j * cols + c];
                yq += xt[j] * wq[j * cols + c];
            }
            let d = y - yq;
            total += d * d;
        }
    }
    total
}

/// AWQ 的 α 网格搜索，返回 `(选中的 α, 量化矩阵, 缩放向量)`。
///
/// 与 [`awq_quantize_search`] 的区别在**目标函数**：后者拿"权重坐标下的等效误差"
/// `‖quantize(W·diag(s))/diag(s) − W‖` 当代理（不碰校准数据的前向，一层只跑一次量化）；
/// 这里直接用真正的目标——拿校准数据算 `‖X·W − X·Ŵ‖²_F`，其中
/// `Ŵ = 反量化(W·diag(s)) / diag(s)` 是折回原坐标的等效权重（`x` 与 `W` 同在原始坐标里，
/// 所以必须折回去比，否则误差里会混进 `s` 的系统性缩放，搜索会系统性偏向 α = 0）。
///
/// 为什么值得多花这份算力：代理误差把每个输出通道的误差**同等**看待，而真正的目标里
/// 每个通道被它的输入加权。激活重尾（少数离群通道比其余大 1~2 个数量级）时两者挑出的 α
/// 可能不同，而只有后者是"部署后真正会看到的误差"。代价是每个候选 α 多两次矩阵乘
/// （`O(n·rows·cols)`），仍远低于跑一遍模型前向。
///
/// `act_abs_mean` 是逐输入通道的平均激活幅度（长度 `rows`，来自 [`Calibration::act_abs_mean`]）；
/// `x` 是 `[x_rows, rows]` 的该层输入激活（就是喂给 [`Calibration::observe`] 的那批）。
/// `alphas` 为空时用 [`awq_alpha_grid`]。并列时取先出现的候选（α 更小 = 缩放更温和）。
pub fn awq_best_alpha(
    w: &[f32],
    rows: usize,
    cols: usize,
    bits: QBits,
    axis: QAxis,
    act_abs_mean: &[f32],
    x: &[f32],
    x_rows: usize,
    alphas: &[f32],
) -> (f32, QMatrix, Vec<f32>) {
    assert_eq!(w.len(), rows * cols, "AWQ 权重长度与形状不符");
    assert_eq!(
        x.len(),
        x_rows * rows,
        "校准激活必须是 [x_rows, rows]（rows = 输入通道数）"
    );
    let alphas = if alphas.is_empty() {
        awq_alpha_grid()
    } else {
        alphas.to_vec()
    };
    let mut best: Option<(f32, QMatrix, Vec<f32>, f32)> = None;
    for &alpha in &alphas {
        let (q, s) = awq_quantize(w, rows, cols, bits, axis, act_abs_mean, alpha);
        // 折回原坐标后再算输出误差：只有同一坐标下的两个权重才配在同一批输入上比较
        let equiv = awq_unfold_weight(&q.dequantize(), rows, cols, &s);
        let err = output_error_sq(x, w, &equiv, x_rows, rows, cols);
        // `<` 而非 `<=`：并列时保留先出现的候选
        if best.as_ref().is_none_or(|(_, _, _, e)| err < *e) {
            best = Some((alpha, q, s, err));
        }
    }
    // 第 4 项（输出误差）只用于选优，不往外传：调用方若要报告误差，会在自己的口径上重算一次
    // 与 `awq_quantize_search` 同样的理由：网格为空时回落到 `awq_alpha_grid`，故恒非空
    let (alpha, q, s, _) = best.expect("α 网格不能为空（网格为空时会回落到 awq_alpha_grid，恒非空）");
    (alpha, q, s)
}

/// 一层权重的量化表示（weight-only 量化）
#[derive(Clone, Debug)]
pub struct QuantWeight {
    /// 权重的量化表示，形状与原权重一致（`[in_features, out_features]`）
    pub q: QMatrix,
    /// AWQ 的输入缩放：`Some(s)` 时前向必须先把输入逐通道除以 `s`（见 [`awq_unfold_input`]）。
    /// RTN / GPTQ 不需要它（`None`）。
    pub input_scale: Option<Vec<f32>>,
    /// 产生这份表示所用的算法。checkpoint 头部靠它记录"这个档当初是怎么量化的"，
    /// 加载端才能按同样参数重放（见 [`crate::checkpoint::requantize_after_load`]）。
    pub method: QuantMethod,
}

impl QuantWeight {
    /// RTN / GPTQ 的量化表示（无输入缩放）
    pub fn new(q: QMatrix, method: QuantMethod) -> Self {
        QuantWeight {
            q,
            input_scale: None,
            method,
        }
    }

    /// AWQ 的量化表示（带输入缩放向量）
    pub fn with_input_scale(q: QMatrix, s: Vec<f32>) -> Self {
        QuantWeight {
            q,
            input_scale: Some(s),
            method: QuantMethod::Awq,
        }
    }

    /// 量化表示实际占用的字节数（整数码 + 分组 scale + AWQ 输入缩放）
    pub fn byte_len(&self) -> usize {
        self.q.byte_len() + self.input_scale.as_ref().map_or(0, |s| s.len() * 4)
    }
}

/// 单层的校准统计：GPTQ 要 `H = E[xᵀx]`，AWQ 要 `mean|x|`
#[derive(Clone, Debug, Default)]
pub struct LayerCalib {
    /// 输入维度上的 `H = E[xᵀx]`；`None` = 这一层没采到。
    /// 是全矩阵还是对角由 [`CalibOpts::full_hessian_max_dim`] 决定（见 [`HessianKind`]）。
    pub hessian: Option<HessianKind>,
    /// 每个输入通道的 `mean|x|`（长度 = `in_features`）；`None` = 这一层没采到。
    /// 它是 AWQ 的唯一输入，也是 GPTQ act-order 的备用依据（`diag(H)` 与之同源）。
    pub act_abs_mean: Option<Vec<f32>>,
    /// 参与统计的 token 数。Hessian 与均值都是在这些 token 上累加出来的：
    /// token 太少时 `H` 估计得不准（采样噪声会被 `H⁻¹` 放大），
    /// 所以它既是日志信息，也是判断"校准够不够"的依据。
    pub n_tokens: usize,
}

/// 整模型的校准统计：key 就是 [`crate::model::GPT::named_parameters`] 的名字去掉
/// 末尾 `.weight` 之后的**参数名前缀**（如 `blocks.0.attn.c_q`）。
///
/// 用名字而不是层下标做 key，是为了让"遍历顺序"与 checkpoint / 日志里出现的名字一致：
/// 换架构（GQA、SwiGLU、MoE）时遍历顺序会变，但名字不变，
/// 统计与层的对应关系始终可核对。
#[derive(Clone, Debug, Default)]
pub struct CalibStats {
    /// 逐层的 `(参数名前缀, 统计)`
    pub per_layer: Vec<(String, LayerCalib)>,
}

impl CalibStats {
    /// 按参数名前缀取一层的统计
    pub fn get(&self, key: &str) -> Option<&LayerCalib> {
        self.per_layer
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// **单层**的量化报告：一次量化给出一份，模型级入口把它们收集成 `Vec`。
///
/// 为什么报告以"层"为单位而不是只有一个总数：量化的收益与风险都是**逐层不均匀**的
/// ——某些层（尤其是靠近输出头的层）对误差极其敏感，某些层压到 int4 也毫发无伤；
/// 分组 scale 的开销又会让很小的矩阵"越量化越大"。只给一个总数，这些问题都会被平均掉。
/// 逐层的 `f32_bytes`/`quant_bytes`/`max_abs_err`/`rel_err` 让调用方（日志、CI 门限、
/// 人工核对）能一眼看出**是哪一层**出了问题。
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct QuantReport {
    /// 参数名前缀（与 checkpoint / 日志里的名字同源）
    pub name: String,
    pub method: QuantMethod,
    pub bits: QBits,
    /// 量化前这一层权重的 f32 字节数
    pub f32_bytes: usize,
    /// 量化后这一层实际占用的字节数（整数码 + 分组 scale + AWQ 输入缩放）
    pub quant_bytes: usize,
    /// 最大绝对误差（在折算回原坐标的**等效权重**上统计，见 [`quant_error`]）
    pub max_abs_err: f32,
    /// 相对误差 `rmse / rms(W)`：跨层、跨位宽互比时用它，绝对值量级会骗人
    pub rel_err: f32,
    /// AWQ 实际使用的缩放指数 α（其余算法为 `None`；搜索模式下逐层可能不同）
    pub alpha: Option<f32>,
}

impl QuantReport {
    /// 压缩率（量化前 / 量化后）。小于 1 说明"为了量化反而多花了字节"（例如
    /// 矩阵很小、分组 scale 的开销超过省下来的码值），此时应当换分组方向或不做。
    pub fn ratio(&self) -> f64 {
        if self.quant_bytes == 0 {
            return 1.0;
        }
        self.f32_bytes as f64 / self.quant_bytes as f64
    }

    /// 单层摘要（逐层报告用的日志行）
    pub fn describe(&self) -> String {
        let alpha = match self.alpha {
            Some(a) => format!("，α={a:.2}"),
            None => String::new(),
        };
        format!(
            "{:<28} {} {:<4} {:>9} → {:>9} B（{:>5.2}×）max-abs {:.3e} rel {:.2}%{}",
            self.name,
            self.bits.name(),
            self.method.name(),
            self.f32_bytes,
            self.quant_bytes,
            self.ratio(),
            self.max_abs_err,
            self.rel_err * 100.0,
            alpha,
        )
    }
}

/// 逐层报告的汇总（`quant` 子命令打印：总量、压缩比、最差层）。
///
/// 与 [`QuantReport`] 的分工：后者回答"哪一层怎么样"，这里回答"整模型到底省了多少、
/// 最差的那层有多差、有没有层被跳过"。总字节口径是**整模型**的（含不参与量化的
/// 词嵌入表），否则压缩率会被自我感觉良好地高估。
#[derive(Clone, Debug, Serialize)]
pub struct QuantSummary {
    pub method: QuantMethod,
    pub bits: QBits,
    /// 量化前整模型的 f32 字节数（Linear 权重 + 词嵌入表）
    pub f32_bytes: usize,
    /// 量化后整模型的实际字节数
    pub quant_bytes: usize,
    /// 逐层报告（顺序与模型遍历顺序一致）
    pub layers: Vec<QuantReport>,
    /// 被跳过的层数（例如已挂 LoRA 适配器的层：量化会让适配器增量被静默丢弃）
    pub skipped: usize,
}

impl QuantSummary {
    /// 整模型压缩率
    pub fn ratio(&self) -> f64 {
        if self.quant_bytes == 0 {
            return 1.0;
        }
        self.f32_bytes as f64 / self.quant_bytes as f64
    }

    /// 相对误差最大的那一层（`None` = 一层都没量化）
    pub fn worst_layer(&self) -> Option<&QuantReport> {
        self.layers
            .iter()
            .max_by(|a, b| a.rel_err.partial_cmp(&b.rel_err).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// 中文摘要：算法、位宽、总字节与压缩率、层数、最差层、被跳过的层数
    pub fn describe(&self) -> String {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        let mut s = format!(
            "{} {} 量化（{}）：权重 {} → {}（{:.2} MB → {:.2} MB，压缩 {:.2}×，命中 {} 层",
            self.bits.name(),
            self.method.name(),
            self.method.describe(),
            self.f32_bytes,
            self.quant_bytes,
            mb(self.f32_bytes),
            mb(self.quant_bytes),
            self.ratio(),
            self.layers.len(),
        );
        if let Some(w) = self.worst_layer() {
            s.push_str(&format!(
                "，最差层 {} rel {:.2}%",
                w.name,
                w.rel_err * 100.0
            ));
        }
        if self.skipped > 0 {
            s.push_str(&format!("，跳过 {} 层", self.skipped));
        }
        s.push('）');
        s
    }
}

/// checkpoint 头里记录的量化元信息
///
/// 存的是**参数**（`bits`/`method`）与**口径**（字节数），不存量化数据本身：
/// 存档里始终写"反量化后的 f32 权重"，任何现有加载路径都能直接读；
/// 需要真省显存的场景据此重跑一次量化（见 [`crate::checkpoint::requantize_after_load`]）。
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuantMeta {
    pub bits: QBits,
    pub method: QuantMethod,
    /// 量化前的权重字节数（口径见 [`CalibStats`] 的同类说明：Linear 权重 + 词嵌入表）
    pub orig_bytes: usize,
    /// 量化后的字节数
    pub quant_bytes: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n).map(|i| (i as f32 * 0.37).sin() * (1.0 + i as f32 * 0.01)).collect()
    }

    /// 打包 / 解包必须无损（int4 的符号扩展、两码一字节都要对）
    #[test]
    fn test_pack_unpack_roundtrip() {
        for bits in [QBits::Int8, QBits::Int4] {
            let qmax = bits.qmax() as i8;
            // 奇数个码也要能处理：int4 时最后一个字节的高半字节是填充
            let codes: Vec<i8> = (0..17).map(|i| (i % 2 == 0).then(|| -qmax).unwrap_or(qmax)).collect();
            let bytes = pack_codes(&codes, bits);
            assert_eq!(bytes.len(), packed_len(codes.len(), bits));
            assert_eq!(unpack_codes(&bytes, bits, codes.len()), codes);
        }
        // int4 的字节数必须是 int8 的一半
        let codes: Vec<i8> = vec![1; 64];
        assert_eq!(pack_codes(&codes, QBits::Int4).len(), 32);
        assert_eq!(pack_codes(&codes, QBits::Int8).len(), 64);
    }

    /// 一维对称量化：误差不超过半个步长，且首尾（最大值处）能被精确表示
    #[test]
    fn test_quantize_row_sym_error_bound() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        for bits in [QBits::Int8, QBits::Int4] {
            let (codes, scale) = quantize_row_sym(&x, bits);
            let qmax = bits.qmax();
            let step = scale / qmax;
            for (i, &v) in x.iter().enumerate() {
                let back = codes[i] as f32 / qmax * scale;
                assert!(
                    (v - back).abs() <= step * 0.5 + 1e-6,
                    "{:?} 第 {i} 个元素误差超界：{v} -> {back}",
                    bits
                );
            }
            // ±max 处正好落在码值边界上，应当无误差
            assert_eq!(codes[0], -qmax as i8);
        }
    }

    /// 全零输入不应产生 NaN（scale 兜底为 1）
    #[test]
    fn test_all_zero_input_is_safe() {
        for bits in [QBits::Int8, QBits::Int4] {
            let x = vec![0.0f32; 32];
            let q = QMatrix::quantize(&x, 4, 8, bits, QAxis::Row);
            assert!(q.dequantize().iter().all(|v| *v == 0.0));
        }
    }

    /// 逐行分组的量化误差必须显著小于"整张矩阵一个 scale"
    #[test]
    fn test_per_row_beats_single_global_scale() {
        // 第 0 行的量级是其余行的 1000 倍：全局 scale 会把小行压成几个码值
        let (rows, cols) = (8, 64);
        let mut x = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let gain = if r == 0 { 1000.0 } else { 1.0 };
            for c in 0..cols {
                x[r * cols + c] = gain * ((c as f32 * 0.31).cos());
            }
        }

        let per_row = QMatrix::quantize(&x, rows, cols, QBits::Int4, QAxis::Row);

        // 手工构造"全局一个 scale"：把所有 scale 都设成全局最大值
        let global = x.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let global_scales = vec![global; rows];
        let per_all = QMatrix::quantize_with_scales(
            &x,
            rows,
            cols,
            QBits::Int4,
            QAxis::Row,
            &global_scales,
        );

        // 只看小量级的那 7 行。第 0 行的误差在两种方案里都由它自己的 scale 决定，
        // 混在一起会把差异淹没；真正体现"分组"价值的是小行有没有被大行的 scale 压死。
        let tail_rmse = |m: &QMatrix| -> f32 {
            let d = m.dequantize();
            let se: f32 = (cols..rows * cols).map(|i| (x[i] - d[i]).powi(2)).sum();
            (se / ((rows - 1) * cols) as f32).sqrt()
        };
        let e_row = tail_rmse(&per_row);
        let e_all = tail_rmse(&per_all);

        assert!(
            e_row * 5.0 < e_all,
            "逐行量化应远好于全局 scale：小行 rmse {} vs {}",
            e_row,
            e_all
        );
    }

    /// 逐列分组（KIVI 对 K 的做法）与逐行分组是两条独立路径，各自形状都要对
    #[test]
    fn test_col_axis_scales_length_and_rows() {
        let (rows, cols) = (5, 7);
        let x = ramp(rows * cols);
        let q = QMatrix::quantize(&x, rows, cols, QBits::Int8, QAxis::Col);
        assert_eq!(q.axis(), QAxis::Col);
        assert_eq!(q.scales().len(), cols, "逐列分组应有 cols 个 scale");
        assert_eq!(q.byte_len(), rows * cols + cols * 4);
        for r in 0..rows {
            assert_eq!(q.row(r), q.dequantize()[r * cols..(r + 1) * cols]);
        }
    }

    /// 追加行：逐行分组会把新行的 scale 一起算准；逐列分组沿用冻结的 scale。
    #[test]
    fn test_push_rows_uses_frozen_col_scales_but_fresh_row_scales() {
        let (rows, cols) = (4, 8);
        let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.13).sin()).collect();
        // 追加一批量级大得多的行
        let more: Vec<f32> = (0..2 * cols).map(|i| 50.0 + i as f32 * 0.01).collect();

        // 逐列：scale 冻结，新值被裁剪（误差有界但不是最优）
        let mut qc = QMatrix::quantize(&x, rows, cols, QBits::Int8, QAxis::Col);
        let scales_before = qc.scales().to_vec();
        qc.push_rows(&more, 2);
        assert_eq!(qc.rows(), 6);
        assert_eq!(qc.scales(), &scales_before[..], "逐列分组的 scale 必须冻结");
        assert_eq!(qc.dequantize().len(), 6 * cols);

        // 逐行：新行带上了自己的 scale，量级被正确表达（误差远小于逐列裁剪）
        let mut qr = QMatrix::quantize(&x, rows, cols, QBits::Int8, QAxis::Row);
        qr.push_rows(&more, 2);
        assert_eq!(qr.rows(), 6);
        assert_eq!(qr.scales().len(), 6, "逐行分组每个新行都要有自己的 scale");
        let tail_truth = more.clone();
        let tail_got = &qr.dequantize()[4 * cols..];
        let e = quant_error(&tail_truth, tail_got);
        assert!(e.relative < 0.01, "逐行分组的新行应几乎无损：{:?}", e);
    }

    /// 丢弃最前面的若干行（滑动窗口 / Attention Sink 用），剩余内容要逐位保持不变
    #[test]
    fn test_drop_front_rows_keeps_remainder_exact() {
        let (rows, cols) = (6, 5);
        let x = ramp(rows * cols);
        for axis in [QAxis::Row, QAxis::Col] {
            let q = QMatrix::quantize(&x, rows, cols, QBits::Int4, axis);
            let full = q.dequantize();
            let mut dropped = q.clone();
            dropped.drop_front_rows(2);
            assert_eq!(dropped.rows(), 4);
            assert_eq!(dropped.scales().len(), if axis == QAxis::Row { 4 } else { cols });
            assert_eq!(
                dropped.dequantize(),
                full[2 * cols..].to_vec(),
                "丢头之后剩下的行应逐位不变（{:?}）",
                axis
            );
        }
    }

    /// 逐列冻结 scale 的"未定"状态：冻结那一刻全零的列不能立刻定标，
    /// 要等这一列第一次出现非零数据时再定——否则非零数据要么被压成 0、要么被裁剪饱和。
    #[test]
    fn test_undetermined_col_scale_is_set_on_first_nonzero_batch() {
        let cols = 4;
        // 第 0 行全零，所以前两列只能停在"未定"；后两列由第 1 行定标
        let first = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, -3.0];
        let mut q = QMatrix::quantize(&first, 2, cols, QBits::Int8, QAxis::Col);
        assert_eq!((q.scales()[0], q.scales()[1]), (0.0, 0.0), "全零列应停在未定状态");
        assert_eq!((q.scales()[2], q.scales()[3]), (2.0, 3.0));
        assert_eq!(q.dequantize(), first, "未定的列反量化回 0，与原值仍然一致");

        // 前两列第一次出现非零数据：必须被正常表达
        let second = vec![4.0, -5.0, 0.0, 0.0];
        q.push_rows(&second, 1);
        assert_eq!(
            (q.scales()[0], q.scales()[1]),
            (4.0, 5.0),
            "第一批非零数据应完成定标"
        );
        for (a, b) in second.iter().zip(&q.dequantize()[2 * cols..]) {
            assert!((a - b).abs() < 0.02, "{a} -> {b}");
        }

        // 之后按冻结的 4 / 5 走：量级相当的行依然精确
        q.push_rows(&[2.0, 2.5, 0.0, 0.0], 1);
        for (a, b) in [2.0, 2.5].iter().zip(&q.dequantize()[3 * cols..]) {
            assert!((a - b).abs() < 0.03, "{a} -> {b}");
        }
    }

    /// int4 的整数码占用必须真的只有 int8 的一半（这是量化的意义所在）
    #[test]
    fn test_int4_storage_is_half_of_int8() {
        let (rows, cols) = (32, 64);
        let x = ramp(rows * cols);
        let q8 = QMatrix::quantize(&x, rows, cols, QBits::Int8, QAxis::Row);
        let q4 = QMatrix::quantize(&x, rows, cols, QBits::Int4, QAxis::Row);
        assert_eq!(q4.bytes().len() * 2, q8.bytes().len(), "int4 的码应只有 int8 的一半字节");
        // 相对原始 f32 数据：int8 是 1/4，int4 是 1/8（这里只算整数码部分）
        assert_eq!(q8.bytes().len() * 4, rows * cols * 4);
        assert_eq!(q4.bytes().len() * 8, rows * cols * 4);
        // 加上 scale 的固定开销后，int4 仍然明显更省
        assert!(q4.byte_len() < q8.byte_len());
    }

    // ---------- 第 33 课：GPTQ / AWQ 的数值测试 ----------

    /// 造一个条件数可控的正定矩阵 `A = M·Mᵀ + n·I` 与它的行优先表示
    fn random_spd(n: usize, seed: u64) -> Vec<f32> {
        let mut rng = Rng::new(seed);
        let m: Vec<f32> = (0..n * n).map(|_| rng.randn()).collect();
        let mut a = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0f32;
                for k in 0..n {
                    s += m[i * n + k] * m[j * n + k];
                }
                a[i * n + j] = s + if i == j { n as f32 } else { 0.0 };
            }
        }
        a
    }

    /// 行优先矩阵乘 `[m, k] × [k, n]`（测试里只用来核对恒等式，不追求性能）
    fn mul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = 0.0f32;
                for t in 0..k {
                    s += a[i * k + t] * b[t * n + j];
                }
                out[i * n + j] = s;
            }
        }
        out
    }

    /// GPTQ 的加权误差 `tr((W-Ŵ)ᵀ H (W-Ŵ)) = Σ_b e_bᵀ H e_b`。
    ///
    /// `err` 行优先 `[rows, cols]`（`rows` = 输入维度），`e_b` 是它的第 `b` 列
    /// ——也就是第 b 个输出通道上的误差向量（长度 = rows）；`h` 是 `[rows, rows]`。
    fn weighted_error(err: &[f32], h: &[f32], rows: usize, cols: usize) -> f32 {
        let mut total = 0.0f32;
        for b in 0..cols {
            for j in 0..rows {
                for k in 0..rows {
                    total += err[j * cols + b] * h[j * rows + k] * err[k * cols + b];
                }
            }
        }
        total
    }

    /// 造一批**通道之间强相关**的输入 `[tokens, rows]`，同时返回 `H = Σ_t x_t x_tᵀ`。
    ///
    /// 为什么不用独立同分布的高斯输入：GPTQ 的全部收益来自 `H` 的**非对角能量**。
    /// iid 输入下 `H ≈ tokens·I`（非对角只剩 O(√tokens) 的采样噪声），补偿系数
    /// `H⁻¹[i,k]`（i ≠ k）几乎为 0，GPTQ 相对 RTN 的降幅只剩百分之几——那时
    /// "GPTQ 更好"的断言实际上测的是噪声。真实 Transformer 的激活高度冗余
    /// （相邻通道强相关），这里用一阶自回归 `x_j = ρ·x_{j-1} + √(1-ρ²)·z_j`
    /// 造出 `H ≈ tokens·Toeplitz(ρ^{|i-j|})` 这种强非对角结构，
    /// 才是 GPTQ 真正被设计来处理的情形。
    fn correlated_inputs(tokens: usize, rows: usize, rho: f32, seed: u64) -> (Vec<f32>, Vec<f32>) {
        let mut rng = Rng::new(seed);
        let s = (1.0 - rho * rho).sqrt();
        let mut x = vec![0.0f32; tokens * rows];
        for t in 0..tokens {
            let mut prev = rng.randn();
            for j in 0..rows {
                prev = rho * prev + s * rng.randn();
                x[t * rows + j] = prev;
            }
        }
        let mut h = vec![0.0f32; rows * rows];
        for t in 0..tokens {
            for j in 0..rows {
                for k in 0..rows {
                    h[j * rows + k] += x[t * rows + j] * x[t * rows + k];
                }
            }
        }
        (x, h)
    }

    /// Cholesky 的两个恒等式：`L·Lᵀ = A` 与 `A·A⁻¹ = I`。
    /// 这两步是 GPTQ 的全部数学依赖，任一处偏差都会被 `H⁻¹` 放大到补偿量上。
    #[test]
    fn cholesky_inverse_is_correct() {
        let n = 6;
        let a = random_spd(n, 2024);
        let l = cholesky(&a, n).expect("正则化的正定矩阵必须能分解");

        // L 必须是下三角（上三角的元素恒为 0）
        for i in 0..n {
            for j in (i + 1)..n {
                assert_eq!(l[i * n + j], 0.0, "L 的第 ({i},{j}) 项应在对角线上方");
            }
        }
        // L·Lᵀ ≈ A
        let mut lt = vec![0.0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                lt[i * n + j] = l[j * n + i];
            }
        }
        let a_rec = mul(&l, &lt, n, n, n);
        for (x, y) in a.iter().zip(&a_rec) {
            assert!((x - y).abs() < 1e-4, "L·Lᵀ 与 A 不符：{x} vs {y}");
        }

        // A·A⁻¹ ≈ I
        let inv = cholesky_inverse_upper(&l, n);
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (inv[i * n + j] - inv[j * n + i]).abs() < 1e-6,
                    "逆矩阵必须是对称的（两个三角都要填好）"
                );
            }
        }
        let eye = mul(&a, &inv, n, n, n);
        for i in 0..n {
            for j in 0..n {
                let want = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (eye[i * n + j] - want).abs() < 1e-4,
                    "A·A⁻¹ 的第 ({i},{j}) 项应为 {want}，实得 {}",
                    eye[i * n + j]
                );
            }
        }

        // 非正定的矩阵必须被判出来（而不是算出 NaN 或负的 sqrt）
        let mut bad = a.clone();
        bad[0] = -1.0;
        bad[0 * n + 1] = 0.0;
        bad[1 * n + 0] = 0.0;
        // 直接给一个对角为负的矩阵：A = -I
        let neg_i: Vec<f32> = (0..n * n)
            .map(|i| if i / n == i % n { -1.0 } else { 0.0 })
            .collect();
        assert!(cholesky(&neg_i, n).is_none(), "负定矩阵不该分解出一个实的下三角因子");
        assert!(cholesky(&vec![0.0f32; n * n], n).is_none(), "零矩阵是奇异的");
    }

    /// GPTQ 的全部意义所在：同样的位宽、同一份权重、同一个 `H` 下，
    /// **加权误差**必须显著小于 RTN（这里给的是 20% 的硬门限）。
    ///
    /// 加权误差才是真正决定模型质量的量（它等于"用 Ŵ 替代 W 之后每层输出的均方误差"），
    /// 权重矩阵上的逐元素误差不是——GPTQ 完全可能在逐元素口径上不比 RTN 好，
    /// 却把输出误差降掉一大半。所以门限只压在加权口径上，
    /// 逐元素只检查"有限、没有把某列推出浮点范围"。
    #[test]
    fn gptq_beats_rtn_on_weighted_error() {
        let (rows, cols, tokens) = (32, 32, 256);
        let mut rng = Rng::new(7);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        let (_x, h) = correlated_inputs(tokens, rows, 0.9, 11);
        let hess = HessianKind::Full(h.clone());
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            let gptq = gptq_quantize(&w, rows, cols, bits, QAxis::Col, &hess, &GptqOpts::default());
            let err = |q: &QMatrix| {
                let d = q.dequantize();
                let e: Vec<f32> = w.iter().zip(&d).map(|(a, b)| a - b).collect();
                weighted_error(&e, &h, rows, cols)
            };
            // 开方后就是"每个输出通道上的加权 rmse"口径的相对误差，跨位宽/跨规模可比
            let rel_rtn = err(&rtn).sqrt();
            let rel_gptq = err(&gptq).sqrt();
            println!(
                "[{}] 加权相对误差 RTN = {rel_rtn:.6}，GPTQ = {rel_gptq:.6}（降幅 {:.1}%）",
                bits.name(),
                (1.0 - rel_gptq / rel_rtn) * 100.0
            );
            assert!(
                rel_gptq < rel_rtn * 0.8,
                "[{}] GPTQ 的加权误差应至少再降 20%：{rel_gptq} vs RTN {rel_rtn}",
                bits.name()
            );
            // 逐元素误差同样要有限（补偿不能把某列推出浮点范围）
            assert!(gptq.dequantize().iter().all(|v| v.is_finite()));
        }
    }

    /// 病态 Hessian（全零 / 含 NaN / 非正定）必须**退回 RTN**而不是 panic 或产出 NaN：
    /// 量化是部署前的最后一步，这里挂掉会让整条流水线卡死。
    /// 对角模式同样退化成 RTN——`H⁻¹` 是对角阵时跨通道补偿系数恒为 0。
    #[test]
    fn gptq_falls_back_to_rtn_on_singular_hessian() {
        let (rows, cols) = (8, 8);
        let w = ramp(rows * cols);
        let opts = GptqOpts::default();
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);

            // 全零 Hessian：trace = 0，加阻尼也救不回正定性
            let zero_h = HessianKind::Full(vec![0.0f32; rows * rows]);
            let q1 = gptq_quantize(&w, rows, cols, bits, QAxis::Col, &zero_h, &opts);
            assert_eq!(q1.dequantize(), rtn.dequantize(), "H 全零时必须逐位退回 RTN");

            // 对角含 NaN：Cholesky 的 `d > 0` 判断天然挡住 NaN
            let mut nan_h = vec![0.0f32; rows * rows];
            for i in 0..rows {
                nan_h[i * rows + i] = f32::NAN;
            }
            let q2 = gptq_quantize(
                &w,
                rows,
                cols,
                bits,
                QAxis::Col,
                &HessianKind::Full(nan_h),
                &opts,
            );
            assert!(q2.dequantize().iter().all(|v| v.is_finite()), "NaN 不得扩散到结果里");
            assert_eq!(q2.dequantize(), rtn.dequantize(), "H 含 NaN 时必须退回 RTN");

            // 负定的"Hessian"（对角全负）同样退回
            let neg_h: Vec<f32> = (0..rows * rows)
                .map(|i| if i / rows == i % rows { -1.0 } else { 0.0 })
                .collect();
            let q3 = gptq_quantize(
                &w,
                rows,
                cols,
                bits,
                QAxis::Col,
                &HessianKind::Full(neg_h),
                &opts,
            );
            assert_eq!(q3.dequantize(), rtn.dequantize(), "非正定 Hessian 必须退回 RTN");

            // 对角模式：`H⁻¹` 也是对角阵 ⇒ 跨通道补偿系数恒为 0 ⇒ 与 RTN 逐位相同
            let diag: Vec<f32> = (0..rows).map(|i| 1.0 + i as f32 * 0.1).collect();
            let q4 = gptq_quantize(
                &w,
                rows,
                cols,
                bits,
                QAxis::Col,
                &HessianKind::Diagonal(diag),
                &opts,
            );
            assert_eq!(
                q4.dequantize(),
                rtn.dequantize(),
                "只存对角时 GPTQ 在数学上就退化成 RTN，必须逐位一致"
            );
        }
    }

    /// 阻尼的意义：把**奇异** `H` 的 Cholesky 从"失败"救成"成功"，
    /// 从而让误差补偿路径真正跑起来（而不是静默退回 RTN）。
    #[test]
    fn gptq_damp_rescues_singular_hessian() {
        let n = 8;
        // 只有左上 2×2 有能量的奇异 H（其余行列全 0）：第 3 个主元恰好是 0，
        // 不做阻尼时 Cholesky 必然在第 3 步返回 None。
        let mut h = vec![0.0f32; n * n];
        h[0] = 2.0;
        h[1] = 1.0;
        h[n] = 1.0;
        h[n + 1] = 2.0;
        assert!(cholesky(&h, n).is_none(), "奇异 H 不该分解出实的下三角因子");

        // 按 trace/n 的比例加阻尼后，最小特征值被抬离 0，分解必定成功
        let damp = GPTQ_DAMP * (2.0 + 2.0) / n as f32;
        let mut damped = h.clone();
        for i in 0..n {
            damped[i * n + i] += damp;
        }
        assert!(cholesky(&damped, n).is_some(), "阻尼后 H 应为正定，Cholesky 必须成功");

        let (rows, cols) = (n, 16);
        let w = ramp(rows * cols);
        let hess = HessianKind::Full(h.clone());
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            // damp = 0：分解失败 ⇒ 不做补偿 ⇒ 与 RTN 逐位相同
            let no_damp = gptq_quantize(
                &w,
                rows,
                cols,
                bits,
                QAxis::Col,
                &hess,
                &GptqOpts { damp: 0.0, ..Default::default() },
            );
            assert_eq!(no_damp.dequantize(), rtn.dequantize(), "Cholesky 失败时必须退回 RTN");

            // damp > 0：分解成功 ⇒ 补偿路径真的跑起来，结果依然有限且结构合法
            let with_damp =
                gptq_quantize(&w, rows, cols, bits, QAxis::Col, &hess, &GptqOpts::default());
            assert!(with_damp.dequantize().iter().all(|v| v.is_finite()));
            assert_eq!(with_damp.axis(), QAxis::Col);
            assert_eq!((with_damp.rows(), with_damp.cols()), (rows, cols));
            assert_eq!(with_damp.scales().len(), cols);
            assert_eq!(with_damp.byte_len(), rtn.byte_len(), "阻尼不改变表示口径");
        }
    }

    /// act-order 的两个开关都必须跑通且误差有限；更强的一条：无论开不开，
    /// GPTQ 的加权误差都要优于 RTN（重排序只影响"谁先被量化"，不影响补偿本身成立）。
    #[test]
    fn gptq_act_order_runs_both_ways() {
        let (rows, cols, tokens) = (24, 20, 192);
        let mut rng = Rng::new(101);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        // 让各通道的激活能量差别很大（指数衰减），这样 act-order 才有可排的东西
        let mut x = vec![0.0f32; tokens * rows];
        for t in 0..tokens {
            for j in 0..rows {
                x[t * rows + j] = rng.randn() * 0.6f32.powi(j as i32 / 3);
            }
        }
        let mut h = vec![0.0f32; rows * rows];
        for t in 0..tokens {
            for j in 0..rows {
                for k in 0..rows {
                    h[j * rows + k] += x[t * rows + j] * x[t * rows + k];
                }
            }
        }
        let hess = HessianKind::Full(h.clone());
        let err = |q: &QMatrix| {
            let d = q.dequantize();
            let e: Vec<f32> = w.iter().zip(&d).map(|(a, b)| a - b).collect();
            weighted_error(&e, &h, rows, cols).sqrt()
        };
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            let rel_rtn = err(&rtn);
            let mut results = Vec::new();
            for act_order in [false, true] {
                let opts = GptqOpts { act_order, ..Default::default() };
                let q = gptq_quantize(&w, rows, cols, bits, QAxis::Col, &hess, &opts);
                assert!(q.dequantize().iter().all(|v| v.is_finite()), "act_order={act_order} 产出了非有限值");
                assert_eq!(q.axis(), QAxis::Col);
                assert_eq!((q.rows(), q.cols()), (rows, cols));
                assert_eq!(q.scales().len(), cols);
                assert_eq!(q.bytes().len(), packed_len(rows * cols, bits));
                let r = err(&q);
                println!(
                    "[{}] act-order={act_order} 加权相对误差 {r:.6}（RTN {rel_rtn:.6}）",
                    bits.name()
                );
                assert!(
                    r < rel_rtn,
                    "[{}] act_order={act_order} 时 GPTQ 仍应优于 RTN：{r} vs {rel_rtn}",
                    bits.name()
                );
                results.push((act_order, r));
            }
            // 重排序是"锦上添花"：它不该把结果弄坏（允许 5% 的浮点/取整抖动）
            assert!(
                results[1].1 <= results[0].1 * 1.05,
                "[{}] act-order 把误差弄差了：naive {} vs act-order {}",
                bits.name(),
                results[0].1,
                results[1].1
            );
        }
    }

    /// 分块（`opts.block`）只是**批处理粒度**：块内立即补偿、块末把累积量一次性
    /// 结算给块外通道，数学上与不分块的顺序版等价。浮点累加顺序不同，个别落在
    /// 格点边界上的元素可能相差一个码，所以这里断言"误差量级一致"而不是逐位相同。
    #[test]
    fn gptq_block_size_does_not_change_result() {
        let (rows, cols, tokens) = (16, 24, 160);
        let mut rng = Rng::new(202);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        let (_x, h) = correlated_inputs(tokens, rows, 0.85, 5);
        let hess = HessianKind::Full(h.clone());
        let err = |q: &QMatrix| {
            let d = q.dequantize();
            let e: Vec<f32> = w.iter().zip(&d).map(|(a, b)| a - b).collect();
            weighted_error(&e, &h, rows, cols).sqrt()
        };
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            let rel_rtn = err(&rtn);
            let mut base: Option<f32> = None;
            for block in [0usize, 1, 4, 16] {
                let opts = GptqOpts { block, ..Default::default() };
                let q = gptq_quantize(&w, rows, cols, bits, QAxis::Col, &hess, &opts);
                assert!(q.dequantize().iter().all(|v| v.is_finite()), "block={block} 产出了非有限值");
                assert_eq!(q.byte_len(), rtn.byte_len(), "分块不改变表示口径");
                let r = err(&q);
                println!("[{}] block={block:>3} 加权相对误差 {r:.6}", bits.name());
                // 这里的门限故意比 `gptq_beats_rtn_on_weighted_error` 松：本测试的对象是
                // **分块不变性**，而这组 `(rows, cols, ρ)` 是按"非对角能量够大"随手取的，
                // 收益本就只有一成多。严格的 0.8 倍门限由那个专门构造的用例来钉。
                assert!(
                    r < rel_rtn * 0.9,
                    "[{}] block={block} 时 GPTQ 也必须优于 RTN：{r} vs {rel_rtn}",
                    bits.name()
                );
                match base {
                    None => base = Some(r),
                    Some(b) => assert!(
                        (r - b).abs() <= b * 0.02,
                        "[{}] 分块大小不该改变结论：block={block} 得 {r}，不分块得 {b}",
                        bits.name()
                    ),
                }
            }
        }
    }

    // ---------- 第 33 课：AWQ 的新增数值测试 ----------

    /// α = 0 ⇒ `s ≡ 1` ⇒ 折叠是恒等变换 ⇒ AWQ 在数学上退化成 RTN，
    /// 码值必须**逐位**相同。这一条同时钉住了三件事：`awq_scales` 的 α=0 分支、
    /// `awq_fold_weight` 的恒等性、以及"折叠后量化"这条链路本身没有额外扰动。
    #[test]
    fn awq_alpha_zero_matches_rtn_bit_exact() {
        let (rows, cols) = (10, 12);
        let w = ramp(rows * cols);
        let act: Vec<f32> = (0..rows).map(|j| 0.3 + j as f32 * 1.1).collect();
        let s = awq_scales(&act, 0.0, rows);
        assert!(s.iter().all(|v| *v == 1.0), "α=0 时缩放必须恰好是 1.0，而不是 1+ε");
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            let (q, s2) = awq_quantize(&w, rows, cols, bits, QAxis::Col, &act, 0.0);
            assert_eq!(s2, s);
            assert_eq!(q.dequantize(), rtn.dequantize(), "α=0 时 AWQ 必须逐位等于 RTN");
        }
    }

    /// 缩放向量必须**严格配对**：折叠除回去要回到原权重；
    /// 全 1 的 `s` 下折叠/展开都是恒等变换（逐位，不是"数值接近"）。
    #[test]
    fn awq_fold_and_unfold_are_inverse() {
        let (rows, cols) = (9, 13);
        let w = ramp(rows * cols);
        let s: Vec<f32> = (0..rows).map(|i| 0.25 + i as f32 * 0.37).collect();

        let folded = awq_fold_weight(&w, rows, cols, &s);
        let back = awq_unfold_weight(&folded, rows, cols, &s);
        for (i, (a, b)) in w.iter().zip(&back).enumerate() {
            assert!((a - b).abs() < 1e-5, "第 {i} 项折叠后除不回来：{a} vs {b}");
        }

        // s 全 1：两步都必须是恒等变换，一个比特都不许动
        let ones = vec![1.0f32; rows];
        assert_eq!(awq_fold_weight(&w, rows, cols, &ones), w);
        assert_eq!(awq_unfold_weight(&w, rows, cols, &ones), w);

        // 展开后的权重与原始权重等价：`(x/s)·(W·s) = x·W`，且折算回原坐标的
        // 量化权重才能和 W 直接比误差（否则误差里会混进 s 的系统性缩放）
        let (q, s) = awq_quantize(&w, rows, cols, QBits::Int4, QAxis::Col, &s, 0.5);
        let equiv = awq_unfold_weight(&q.dequantize(), rows, cols, &s);
        let e_folded = quant_error(&w, &q.dequantize());
        let e_equiv = quant_error(&w, &equiv);
        println!(
            "折叠坐标下 rel {:.4}，折回原坐标后 rel {:.4}",
            e_folded.relative, e_equiv.relative
        );
        assert!(e_equiv.relative.is_finite() && e_equiv.relative > 0.0);
    }

    /// α 网格搜索：返回值必须落在网格上，且选出的等效误差不差于网格里**任何**候选
    /// （含 α = 0，也就是 RTN 的等效误差）。
    #[test]
    fn awq_alpha_search_picks_the_best_grid_point() {
        let (rows, cols) = (16, 18);
        let mut rng = Rng::new(303);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        // 通道重要性差异很大：α 才有可搜的空间
        let act: Vec<f32> = (0..rows).map(|j| 0.05 + 4.0f32.powi(j as i32 % 7 - 3)).collect();

        let grid = awq_alpha_grid();
        assert_eq!(grid.first(), Some(&0.0));
        assert_eq!(grid.last(), Some(&1.0));
        assert_eq!(grid.len(), 21);
        assert!(grid.windows(2).all(|w| (w[1] - w[0] - AWQ_ALPHA_GRID_STEP).abs() < 1e-6));

        for bits in [QBits::Int8, QBits::Int4] {
            let equiv_err = |alpha: f32| {
                let (q, s) = awq_quantize(&w, rows, cols, bits, QAxis::Col, &act, alpha);
                quant_error(&w, &awq_unfold_weight(&q.dequantize(), rows, cols, &s)).relative
            };
            let (q, s, alpha) = awq_quantize_search(&w, rows, cols, bits, QAxis::Col, &act, &grid);
            assert!(grid.contains(&alpha), "搜索给出的 α={alpha} 不在网格里");
            assert_eq!(s.len(), rows, "AWQ 缩放向量长度必须等于输入通道数");
            assert_eq!(q.scales().len(), cols);
            let best = quant_error(&w, &awq_unfold_weight(&q.dequantize(), rows, cols, &s)).relative;
            let rtn = equiv_err(0.0);
            println!("[{}] 选出 α={alpha:.2}，等效 rel {best:.5}；α=0 时 rel {rtn:.5}", bits.name());
            assert!(
                best <= rtn + 1e-9,
                "[{}] 搜索必须不差于 α=0：{best} vs {rtn}",
                bits.name()
            );
            for &a in &grid {
                let e = equiv_err(a);
                assert!(
                    best <= e + 1e-6,
                    "[{}] 网格里的 α={a} 比选中的 α={alpha} 更好（{e} < {best}），搜索漏了",
                    bits.name()
                );
            }
            // 空网格必须回落到默认网格（而不是 panic）
            let (_, _, a2) = awq_quantize_search(&w, rows, cols, bits, QAxis::Col, &act, &[]);
            assert!(awq_alpha_grid().contains(&a2));
        }
    }

    /// GPTQ 产出的 `QMatrix` 与 RTN 版**表示完全同构**：形状、分组方向、每列一个 scale、
    /// 字节口径都一致，且反量化值本身落在格点上（再量化一次不变）。
    /// 这是"GPTQ 只是换了每列的取值，不引入新的数据结构"的直接证据。
    #[test]
    fn gptq_partial_bits_roundtrip() {
        let (rows, cols) = (16, 12);
        let w = ramp(rows * cols);
        let h = random_spd(rows, 31); // 用良态正定矩阵当 Hessian（H 在输入维度上），确保走的是补偿分支
        let hess = HessianKind::Full(h);
        for bits in [QBits::Int8, QBits::Int4] {
            let q = gptq_quantize(&w, rows, cols, bits, QAxis::Col, &hess, &GptqOpts::default());
            assert_eq!(q.axis(), QAxis::Col);
            assert_eq!((q.rows(), q.cols()), (rows, cols));
            assert_eq!(q.scales().len(), cols, "逐列分组：每个输出通道一个 scale");
            assert!(q.scales().iter().all(|s| s.is_finite() && *s > 0.0));
            assert_eq!(
                q.byte_len(),
                q.bytes().len() + cols * 4,
                "字节口径 = 整数码 + 每列一个 fp32 scale"
            );
            assert_eq!(q.bytes().len(), packed_len(rows * cols, bits));

            let back = q.dequantize();
            assert_eq!(back.len(), rows * cols);
            assert!(back.iter().all(|v| v.is_finite()));
            // 反量化值本身就在格点上：拿它按同一套 scale 再量化一次应逐位稳定
            let again = QMatrix::quantize_with_scales(&back, rows, cols, bits, QAxis::Col, q.scales());
            for (a, b) in again.dequantize().iter().zip(&back) {
                assert!((a - b).abs() <= 1e-6, "格点值再量化发生了漂移：{a} vs {b}");
            }
            // 逐列 scale 必须真的把该列盖住（没有被裁剪掉的元素）
            for c in 0..cols {
                let m = (0..rows)
                    .map(|r| back[r * cols + c].abs())
                    .fold(0.0f32, f32::max);
                assert!(m <= q.scales()[c] * (1.0 + 1e-5), "第 {c} 列超出了自己的 scale");
            }
        }
    }

    /// AWQ 的缩放必须单调（激活越大 → scale 越大）且**组内几何均值 = 1**
    /// （组内归一保证缩放不改变权重的整体量级，否则等于什么都没做）。
    #[test]
    fn awq_scales_are_group_normalized() {
        let act = vec![1.0f32, 2.0, 4.0, 8.0, 0.5, 0.25, 16.0, 32.0];
        let s = awq_scales(&act, 0.5, 4);
        assert_eq!(s.len(), act.len());
        for g in 0..2 {
            let seg: Vec<f32> = s[g * 4..(g + 1) * 4].to_vec();
            let mean_log = seg.iter().map(|v| v.ln()).sum::<f32>() / 4.0;
            assert!(mean_log.abs() < 1e-4, "第 {g} 组的几何均值应为 1，log 均值 = {mean_log}");
            for i in 0..4 {
                for j in (i + 1)..4 {
                    let (a, b) = (act[g * 4 + i], act[g * 4 + j]);
                    if a > b {
                        assert!(
                            seg[i] > seg[j],
                            "激活更大的通道应得到更大的 scale：{a} -> {}, {b} -> {}",
                            seg[i],
                            seg[j]
                        );
                    }
                }
            }
        }
        // α = 1 时差异被放大：最大/最小的比值 = (8/1)^1 = 8（组内几何均值仍为 1）
        let s1 = awq_scales(&act, 1.0, 4);
        let ratio = s1[3] / s1[0];
        assert!((ratio - 8.0).abs() < 1e-3, "α=1 时比值应为 8，实得 {ratio}");
        // α = 0 退化成"不缩放"
        assert!(awq_scales(&act, 0.0, 4).iter().all(|v| (v - 1.0).abs() < 1e-6));
        // 全零激活（该层从未被激活）不得产生 NaN / Inf
        let z = awq_scales(&vec![0.0f32; 4], 0.5, 4);
        assert!(z.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    /// 折叠 / 展开必须严格配对：`(x / s) · (W · s) = x · W`。
    /// 少做任何一半都会让前向结果系统性地偏移，而且不报任何错。
    #[test]
    fn awq_weight_input_folding_is_equivalent() {
        // 权重 [in, out] = [10, 6]、一批 [n, in] = [5, 10] 的输入
        let (r#in, out, n) = (10, 6, 5);
        let mut rng = Rng::new(1234);
        let w: Vec<f32> = (0..r#in * out).map(|_| rng.randn()).collect();
        let x: Vec<f32> = (0..n * r#in).map(|_| rng.randn() * 2.0).collect();
        let act: Vec<f32> = (0..r#in).map(|j| 0.1 + j as f32 * 0.7).collect();
        // 组大小取 in：整条输入一个组，几何均值归一在全局生效
        let s = awq_scales(&act, 0.5, r#in);
        assert_eq!(s.len(), r#in);

        let folded = awq_fold_weight(&w, r#in, out, &s);
        let unfolded = awq_unfold_input(&x, r#in, &s);
        let y0 = mul(&x, &w, n, r#in, out);
        let y1 = mul(&unfolded, &folded, n, r#in, out);
        for (i, (a, b)) in y0.iter().zip(&y1).enumerate() {
            assert!(
                (a - b).abs() < 1e-3,
                "折叠前后的第 {i} 个输出不一致：{a} vs {b}"
            );
        }

        // `awq_quantize` 必须把同一套 s 交出来（权重被放大了，输入这一侧必须能除回去）
        let (q, s2) = awq_quantize(&w, r#in, out, QBits::Int8, QAxis::Col, &act, 0.5);
        assert_eq!(s, s2);
        assert_eq!(q.scales().len(), out, "逐列分组：每个输出通道一个 scale");
        // 量化本身是有损的，但缩放后的权重落在原权重的量级附近（不能整体放大）
        let d = q.dequantize();
        let scale_of_w = w.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale_of_d = d.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            scale_of_d < scale_of_w * 4.0,
            "AWQ 的组内归一应把权重整体量级控制在同一档：{scale_of_d} vs {scale_of_w}"
        );
    }

    // ---------- 第 33 课：校准累加器 / 现成 H⁻¹ 的 GPTQ / 按输出误差搜索的 AWQ ----------

    /// 朴素高斯-约当消元求逆（带列主元），给 `cholesky_inverse` 当**独立**对照。
    /// 为什么要另写一份：只有一条"完全不同的路径"给出同一个矩阵，才能说明
    /// "分解 + 两次三角回代"那一路的索引与转置都没写错——拿 Cholesky 自己验证 Cholesky
    /// 是发现不了符号约定错误的。
    fn gauss_jordan_inverse(a: &[f32], n: usize) -> Vec<f32> {
        let mut m = a.to_vec();
        let mut inv = vec![0.0f32; n * n];
        for i in 0..n {
            inv[i * n + i] = 1.0;
        }
        for col in 0..n {
            // 列主元：主元太小时消元会把误差放大几个数量级
            let mut piv = col;
            for r in (col + 1)..n {
                if m[r * n + col].abs() > m[piv * n + col].abs() {
                    piv = r;
                }
            }
            if piv != col {
                for k in 0..n {
                    m.swap(col * n + k, piv * n + k);
                    inv.swap(col * n + k, piv * n + k);
                }
            }
            let d = m[col * n + col];
            for k in 0..n {
                m[col * n + k] /= d;
                inv[col * n + k] /= d;
            }
            for r in 0..n {
                if r == col {
                    continue;
                }
                let f = m[r * n + col];
                if f == 0.0 {
                    continue;
                }
                for k in 0..n {
                    m[r * n + k] -= f * m[col * n + k];
                    inv[r * n + k] -= f * inv[col * n + k];
                }
            }
        }
        inv
    }

    /// `‖A − B‖_F / ‖B‖_F`：跨矩阵比误差时用它，逐元素的绝对差会被元素的量级骗人
    fn rel_frobenius(a: &[f32], b: &[f32]) -> f32 {
        let diff: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum();
        let norm: f32 = b.iter().map(|v| v * v).sum();
        assert!(norm > 0.0, "参照矩阵的范数必须为正");
        (diff / norm).sqrt()
    }

    /// 用 `H` 的**对角**给逐元素误差加权：`Σ_j H_jj · Σ_c (W − Ŵ)²_jc`。
    /// 展开成"对角矩阵 + 现成的 [`weighted_error`]"，省得再写一遍三重循环。
    fn diag_weighted_error(err: &[f32], h: &[f32], rows: usize, cols: usize) -> f32 {
        let mut hd = vec![0.0f32; rows * rows];
        for j in 0..rows {
            hd[j * rows + j] = h[j * rows + j];
        }
        weighted_error(err, &hd, rows, cols)
    }

    /// Cholesky 求逆必须与**独立实现**（朴素消元）在 1e-3 的相对偏差内一致，
    /// 且非正定输入返回 `None`（不是 NaN、更不是 panic）。
    #[test]
    fn test_cholesky_inverse_matches_explicit_inverse() {
        let n = 7;
        let a = random_spd(n, 20240921);
        let inv = cholesky_inverse(&a, n, 0.0).expect("M·Mᵀ + n·I 必为正定");
        let want = gauss_jordan_inverse(&a, n);
        let rel = rel_frobenius(&inv, &want);
        println!("无阻尼：与朴素求逆的 Frobenius 相对偏差 {rel:.2e}");
        assert!(rel < 1e-3, "与朴素求逆不符：{rel}");
        // 逆矩阵必须两个三角都填好：`H⁻¹[j,k]` 在补偿里会以任意下标顺序被访问
        for i in 0..n {
            for j in 0..n {
                assert!(
                    (inv[i * n + j] - inv[j * n + i]).abs() < 1e-5,
                    "H⁻¹ 的第 ({i},{j}) 项不对称"
                );
            }
        }

        // 阻尼的语义就是"把 (H + damp·I) 当成 H 来求逆"：结果必须与之逐项对应
        let damp = 0.25;
        let mut shifted = a.clone();
        for i in 0..n {
            shifted[i * n + i] += damp;
        }
        let inv_d = cholesky_inverse(&a, n, damp).expect("加阻尼后必为正定");
        let rel_d = rel_frobenius(&inv_d, &gauss_jordan_inverse(&shifted, n));
        println!("阻尼 {damp}：与 (H+damp·I)⁻¹ 的相对偏差 {rel_d:.2e}");
        assert!(rel_d < 1e-3, "阻尼后与 (H+damp·I)⁻¹ 不符：{rel_d}");

        // 非正定：负定矩阵、零矩阵、含 NaN 的矩阵，一律判失败
        let neg_i: Vec<f32> = (0..n * n)
            .map(|i| if i / n == i % n { -1.0 } else { 0.0 })
            .collect();
        assert!(cholesky_inverse(&neg_i, n, 0.0).is_none(), "负定矩阵不该给出逆");
        assert!(cholesky_inverse(&vec![0.0f32; n * n], n, 0.0).is_none(), "零矩阵是奇异的");
        let mut nan_h = a.clone();
        nan_h[0] = f32::NAN;
        assert!(cholesky_inverse(&nan_h, n, 1.0).is_none(), "NaN 必须判失败而不是扩散出去");
        // 反过来：阻尼够大时同一张负定矩阵又能求了——这正是"非正定就加大 damp 重试"的意义
        let rescued = cholesky_inverse(&neg_i, n, 2.0).expect("充分阻尼后必可逆");
        assert!(rescued.iter().all(|v| v.is_finite()));
    }

    /// 用真的 [`Calibration::observe`] 采 Hessian、[`cholesky_inverse`] 求逆，再用
    /// [`gptq_quantize_with_hinv`] 量化：**Hessian 加权**误差必须严格小于 RTN。
    ///
    /// 为什么口径必须是 `tr((W − Ŵ)ᵀ H (W − Ŵ))`（= `‖X·W − X·Ŵ‖²` 的期望）而不是朴素 MSE：
    /// GPTQ 优化的目标就是它，算法会**有意**把误差从重要通道挪到不重要的通道上，
    /// 于是逐元素 MSE 口径下完全可能不比 RTN 好——这是算法定义决定的，不是实现问题。
    ///
    /// 为什么连"只取 `H` 的对角"都不够：补偿系数 `H⁻¹[j,k]` 里的**非对角项**才是补偿的方向，
    /// 而对角口径看不见它们。实测（本测试的 `println` 里有对角口径的数字）相关性强时
    /// 对角口径下 GPTQ 反而比 RTN 差一成多，而同一批次的全 `H` 口径下它好 30%~45%：
    /// 补偿沿着"通道相关方向"搬动误差，代价落在对角项上、收益落在交叉项上。
    /// 所以判据必须是与算法同一个二次型，不能是它的对角近似。
    #[test]
    fn test_gptq_beats_rtn_on_hessian_weighted_error() {
        let (rows, cols, tokens) = (24, 16, 256);
        // 多组随机种子：单组权重上"更小"可能只是运气，要看到方向性
        for seed in 0..4u64 {
            let mut rng = Rng::new(9000 + seed);
            let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
            // 通道强相关的激活：GPTQ 的全部收益来自 H 的非对角能量（iid 输入下 H ≈ tokens·I）
            let (x, _h) = correlated_inputs(tokens, rows, 0.85, 300 + seed);
            let mut cal = Calibration::zeros(rows);
            cal.observe(&x, tokens);
            assert_eq!(cal.n_tokens(), tokens);
            let h = cal.mean_hessian();
            // 阻尼按 GPTQ 的通行口径取"对角均值（= trace/n）的 1%"，H 接近奇异时也能求逆
            let trace = (0..rows).map(|i| h[i * rows + i]).sum::<f32>();
            let hinv = cholesky_inverse(&h, rows, GPTQ_DAMP * trace / rows as f32)
                .expect("阻尼后 H 必为正定");
            // 返回 (全 H 加权误差, 只取对角的加权误差)
            let errs = |q: &QMatrix| {
                let d = q.dequantize();
                let e: Vec<f32> = w.iter().zip(&d).map(|(a, b)| a - b).collect();
                (
                    weighted_error(&e, &h, rows, cols).sqrt(),
                    diag_weighted_error(&e, &h, rows, cols).sqrt(),
                )
            };
            for bits in [QBits::Int8, QBits::Int4] {
                let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
                let gptq = gptq_quantize_with_hinv(&w, rows, cols, bits, QAxis::Col, &hinv);
                let (a, b) = (errs(&gptq), errs(&rtn));
                println!(
                    "[{}] seed {seed} 全 H 加权 GPTQ {:.6} < RTN {:.6}（降 {:.1}%；对角口径 {:.6} vs {:.6}）",
                    bits.name(),
                    a.0,
                    b.0,
                    (1.0 - a.0 / b.0) * 100.0,
                    a.1,
                    b.1
                );
                assert!(
                    a.0 < b.0,
                    "[{}] seed {seed}：GPTQ 必须严格优于 RTN，实得 {:.6} vs {:.6}",
                    bits.name(),
                    a.0,
                    b.0
                );
                assert!(gptq.dequantize().iter().all(|v| v.is_finite()));
            }
        }
    }

    /// 校准数据的**输出误差**口径：同一批输入分别过原层与量化层，`‖X·W − X·Ŵ‖²` 必须
    /// GPTQ < RTN。与上一条相比这里不做任何"加权"包装——它就是部署后真正会看到的偏差；
    /// 顺带用分开两次 `observe` 采统计，验证累加器与"一次喂完"等价。
    #[test]
    fn test_gptq_reduces_output_error_on_calibration_data() {
        let (rows, cols, tokens) = (20, 12, 192);
        let mut rng = Rng::new(5150);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        let (x, h) = correlated_inputs(tokens, rows, 0.9, 61);
        let mut cal = Calibration::zeros(rows);
        // 分两批喂：累加的是和，所以结果必须与一次喂完一致（下面与手算的 H 对拍）
        cal.observe(&x[..(tokens / 2) * rows], tokens / 2);
        cal.observe(&x[(tokens / 2) * rows..], tokens - tokens / 2);
        assert_eq!(cal.n_tokens(), tokens);
        let h_mean = cal.mean_hessian();
        for (i, (a, b)) in h_mean.iter().zip(&h).enumerate() {
            let want = b / tokens as f32;
            assert!(
                (a - want).abs() <= 1e-3 * want.abs().max(1.0),
                "第 {i} 项 mean_hessian 与手算的 Σxxᵀ/tokens 不符：{a} vs {want}"
            );
        }
        let trace = (0..rows).map(|i| h_mean[i * rows + i]).sum::<f32>();
        let hinv =
            cholesky_inverse(&h_mean, rows, GPTQ_DAMP * trace / rows as f32).expect("阻尼后必为正定");
        let sq = |wq: &[f32]| output_error_sq(&x, &w, wq, tokens, rows, cols);
        for bits in [QBits::Int8, QBits::Int4] {
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            let gptq = gptq_quantize_with_hinv(&w, rows, cols, bits, QAxis::Col, &hinv);
            let (a, b) = (sq(&gptq.dequantize()), sq(&rtn.dequantize()));
            println!("[{}] 校准数据输出平方误差 GPTQ {a:.6} < RTN {b:.6}", bits.name());
            assert!(a < b, "[{}] GPTQ 必须把输出误差压得更低：{a} vs {b}", bits.name());
        }
    }

    /// AWQ 的适用场景：**逐输出通道共享 scale**（[`QAxis::Col`]）遇上**重尾激活**
    /// （少数离群通道比其余大 1~2 个数量级）。此时 α 网格搜索选出的结果在输出误差上
    /// 必须严格小于不缩放（α = 0，等价于 RTN）。
    ///
    /// 为什么 AWQ 只在"较粗粒度 scale"下才有收益：若 scale 完全逐元素独立，
    /// 把某个输入通道放大 `k` 倍会让该通道的 scale 同步放大 `k` 倍、步长也放大 `k` 倍，
    /// **相对**误差一点没变——缩放被 scale 完全吸收，等于没做。只有 scale 在通道之间共享
    /// （这里每个输出通道一条列 scale，跨全部输入通道）时，放大一条通道才会改变它占用的
    /// 码值范围，显著性通道的相对误差才真的变小。
    #[test]
    fn test_awq_best_alpha_beats_rtn_on_heavy_tailed_activations() {
        let (rows, cols, tokens) = (32, 24, 128);
        let mut rng = Rng::new(909);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        // 重尾激活：每 8 条通道里有一条比其余大两个数量级（真实 Transformer 的离群通道就这形态）
        let mut x = vec![0.0f32; tokens * rows];
        for t in 0..tokens {
            for j in 0..rows {
                let gain = if j % 8 == 0 { 100.0 } else { 1.0 };
                x[t * rows + j] = rng.randn() * gain;
            }
        }
        let mut cal = Calibration::zeros(rows);
        cal.observe(&x, tokens);
        let act = cal.act_abs_mean();
        // 显著性通道必须真的排在前面，否则这组数据压根没造出重尾
        assert!(act[0] > 10.0 * act[1], "重尾激活的构造无效：{} vs {}", act[0], act[1]);

        let sq = |wq: &[f32]| output_error_sq(&x, &w, wq, tokens, rows, cols);
        for bits in [QBits::Int4, QBits::Int8] {
            let (alpha, q, s) =
                awq_best_alpha(&w, rows, cols, bits, QAxis::Col, &act, &x, tokens, &awq_alpha_grid());
            assert!(awq_alpha_grid().contains(&alpha), "α={alpha} 不在网格里");
            assert_eq!(s.len(), rows, "缩放向量长度必须等于输入通道数");
            assert_eq!(q.scales().len(), cols, "逐列分组：每个输出通道一个 scale");
            let equiv = awq_unfold_weight(&q.dequantize(), rows, cols, &s);
            let rtn = QMatrix::quantize(&w, rows, cols, bits, QAxis::Col);
            let (a, b) = (sq(&equiv), sq(&rtn.dequantize()));
            println!(
                "[{}] 选中 α={alpha:.2}，输出平方误差 AWQ {a:.6} < RTN {b:.6}（降 {:.1}%）",
                bits.name(),
                (1.0 - a / b) * 100.0
            );
            assert!(
                a < b,
                "[{}] AWQ 必须优于不缩放的 RTN：{a} vs {b}",
                bits.name()
            );
            assert!(alpha > 0.0, "[{}] 重尾激活下不该选中 α=0（那等于不缩放）", bits.name());
        }
        // 空网格回落到默认网格（而不是 panic）
        let (a2, _, _) = awq_best_alpha(&w, rows, cols, QBits::Int4, QAxis::Col, &act, &x, tokens, &[]);
        assert!(awq_alpha_grid().contains(&a2));
    }

    /// 缩放必须**严格可逆**：`(x/s)·(W·s)` 与 `x·W` 在 f32 下相对误差 < 1e-4（与量化无关）；
    /// 并且 [`awq_quantize`] 交出的 `s` 要真的能把量化权重折回等价权重。
    #[test]
    fn test_awq_scaling_is_exactly_invertible() {
        let (rows, cols, n) = (16, 10, 12);
        let mut rng = Rng::new(6161);
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.randn()).collect();
        let x: Vec<f32> = (0..n * rows).map(|_| rng.randn() * 1.5).collect();
        // 通道幅度差两个数量级：缩放才有东西可分配
        let act: Vec<f32> = (0..rows).map(|j| if j % 4 == 0 { 50.0 } else { 0.4 }).collect();
        // 组大小取 rows：整条输入一个组，几何均值归一在全局生效
        let s = awq_scales(&act, 0.5, rows);
        assert_eq!(s.len(), rows);

        // 1) 纯浮点下的恒等式：`(x/s)·(W·s) = x·W`
        let y0 = mul(&x, &w, n, rows, cols);
        let folded = awq_fold_weight(&w, rows, cols, &s);
        let unfolded_x = awq_unfold_input(&x, rows, &s);
        let rel = rel_frobenius(&mul(&unfolded_x, &folded, n, rows, cols), &y0);
        println!("(x/s)·(W·s) 与 x·W 的相对偏差 {rel:.2e}");
        assert!(rel < 1e-4, "缩放必须在浮点下恒等：{rel}");
        // `s` 必须能精确还原等价权重（折叠后再除回来 == 原权重）
        let back = awq_unfold_weight(&folded, rows, cols, &s);
        for (i, (a, b)) in w.iter().zip(&back).enumerate() {
            assert!((a - b).abs() < 1e-5 * a.abs().max(1.0), "第 {i} 项折回来不对：{a} vs {b}");
        }

        // 2) 量化之后：`s` 仍然必须能把码值折回原坐标，等价权重才与原权重可比
        let (q, s2) = awq_quantize(&w, rows, cols, QBits::Int4, QAxis::Col, &act, 0.5);
        assert_eq!(s, s2, "awq_quantize 必须交出同一套 s");
        let equiv = awq_unfold_weight(&q.dequantize(), rows, cols, &s);
        let rel_eq = rel_frobenius(&mul(&x, &equiv, n, rows, cols), &y0);
        // 反证：漏掉 1/s（拿缩放坐标下的权重乘原坐标的输入）误差会大得多
        let rel_folded = rel_frobenius(&mul(&x, &q.dequantize(), n, rows, cols), &y0);
        println!("int4：折回原坐标 rel {rel_eq:.4}；漏掉 1/s 时 rel {rel_folded:.4}");
        assert!(
            rel_eq < rel_folded * 0.5,
            "折回原坐标必须明显更准：{rel_eq} vs 漏掉 1/s 的 {rel_folded}"
        );
    }

    /// 手工对拍小例子：`XᵀX`、`mean|x|`、`mean_hessian` 都要与纸面结果一致，
    /// 且 `merge`（并行/分批采集的合并）与"逐个 observe"逐位相同。
    #[test]
    fn test_calibration_accumulates_hessian_and_activation_mean() {
        let in_features = 3;
        // 第 1 批 2 个 token，第 2 批 1 个 token
        let x1 = [1.0f32, -2.0, 3.0, 0.5, 0.25, -1.0];
        let x2 = [-4.0f32, 1.5, 0.5];
        // 逐输入通道整理成"每列一个通道"，方便手算 H[j][k] = Σ_t x_t[j]·x_t[k]
        let chan: [[f32; 3]; 3] = [[1.0, 0.5, -4.0], [-2.0, 0.25, 1.5], [3.0, -1.0, 0.5]];

        let mut a = Calibration::zeros(in_features);
        a.observe(&x1, 2);
        a.observe(&x2, 1);
        assert_eq!(a.n_tokens(), 3);
        assert_eq!(a.in_features(), in_features);

        for j in 0..in_features {
            for k in 0..in_features {
                let want: f32 = chan[j].iter().zip(&chan[k]).map(|(p, q)| p * q).sum();
                assert!(
                    (a.hessian()[j * in_features + k] - want).abs() < 1e-5,
                    "H[{j}][{k}] 应为 {want}，实得 {}",
                    a.hessian()[j * in_features + k]
                );
                // 对称性必须是逐位成立的（两个三角一起写）
                assert_eq!(
                    a.hessian()[j * in_features + k],
                    a.hessian()[k * in_features + j]
                );
            }
        }
        // mean|x|：(|1| + |0.5| + |-4|) / 3 这一类
        let want_mean = [(1.0 + 0.5 + 4.0) / 3.0, (2.0 + 0.25 + 1.5) / 3.0, (3.0 + 1.0 + 0.5) / 3.0];
        for (j, want) in want_mean.iter().enumerate() {
            assert!(
                (a.act_abs_mean()[j] - want).abs() < 1e-5,
                "mean|x_{j}| 应为 {want}，实得 {}",
                a.act_abs_mean()[j]
            );
        }
        // mean_hessian = hessian / n_tokens（求逆与阻尼都在这之后才做）
        for (m, s) in a.mean_hessian().iter().zip(a.hessian()) {
            assert!((m - s / 3.0).abs() < 1e-5);
        }

        // merge 必须与"顺序 observe"完全一致：累加和是可加的中立形式
        let mut b = Calibration::zeros(in_features);
        b.observe(&x1, 2);
        let mut c = Calibration::zeros(in_features);
        c.observe(&x2, 1);
        b.merge(&c);
        assert_eq!(b.hessian(), a.hessian());
        assert_eq!(b.act_abs_mean(), a.act_abs_mean());
        assert_eq!(b.mean_hessian(), a.mean_hessian());
        assert_eq!(b.n_tokens(), a.n_tokens());
        // 空统计（没采到数据）不得产生 NaN：均值返回全 0
        let z = Calibration::zeros(4);
        assert!(z.mean_hessian().iter().all(|v| *v == 0.0));
        assert!(z.act_abs_mean().iter().all(|v| *v == 0.0));
        assert_eq!(z.n_tokens(), 0);
    }

    /// 没有 Hessian（`h_inv` 为空或维度不符）时必须是**刻意的降级**：不做任何补偿，
    /// 码值与 [`QMatrix::quantize`] 逐位相同（不是"数值接近"）。
    /// 这条路径对应"校准没采到 / 维度对不上"——量化是部署前最后一步，
    /// 在这里 panic 会把整条流水线卡死，所以判断写成"空 = 不补偿"的语义。
    #[test]
    fn test_gptq_falls_back_to_rtn_without_hessian() {
        let (rows, cols) = (12, 9);
        let w = ramp(rows * cols);
        for bits in [QBits::Int8, QBits::Int4] {
            for axis in [QAxis::Row, QAxis::Col] {
                let rtn = QMatrix::quantize(&w, rows, cols, bits, axis);
                let q = gptq_quantize_with_hinv(&w, rows, cols, bits, axis, &[]);
                assert_eq!(q.bytes(), rtn.bytes(), "空 h_inv 必须逐位退回 RTN");
                assert_eq!(q.scales(), rtn.scales());
                assert_eq!((q.rows(), q.cols(), q.axis()), (rows, cols, axis));
                // 维度对不上（给了 [cols, cols] 而不是 [rows, rows]）同样降级，而不是 panic
                let wrong = vec![1.0f32; cols * cols];
                let q2 = gptq_quantize_with_hinv(&w, rows, cols, bits, axis, &wrong);
                assert_eq!(q2.bytes(), rtn.bytes(), "维度不符时同样退回 RTN");
            }
        }
    }
}
