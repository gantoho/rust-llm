//! FP8 低精度训练模拟（第 43 课）
//!
//! 第 33 课的量化走的是"整数码 + 缩放因子"这条线（int8 / int4，见 [`crate::quant`]），
//! 它的目标是**推理**：把训好的权重压小、算得快。到了 H100 / Ada 这一代硬件，
//! 训练侧也出现了 8 位浮点：**FP8**。它不是把数当整数看，而是沿用浮点的
//! "符号 + 指数 + 尾数"结构，只是把总位宽砍到 8 位。
//!
//! ## 两种格式，各有各的活儿
//!
//! | 格式 | 指数位 | 尾数位 | 最大值 | 最小正规数 | 用途 |
//! |------|--------|--------|--------|-----------|------|
//! | E4M3 | 4 | 3 | 448 | 2⁻⁶ | 权重、激活（范围够用，精度优先） |
//! | E5M2 | 5 | 2 | 57344 | 2⁻¹⁴ | 梯度（动态范围大，允许更粗） |
//!
//! 同一段代码里两种格式各司其职：前向的张量数值集中、用 E4M3 保住有效位；
//! 反向的梯度跨好几个数量级，用 E5M2 换更大的动态范围。**E4M3 没有 Inf 码**
//! （指数全 1 且尾数全 1 是 NaN），溢出只能饱和到 448；E5M2 才保留 Inf。
//!
//! ## 分块缩放：让 8 位也能装下真实分布
//!
//! 整张矩阵共用一个 scale 时，只要有几个离群大的元素，其余元素全被压成 0（FP8 的
//! 指数是定长的，没有"非规格化到任意小"的余地）。所以 FP8 训练普遍用 **per-block
//! scale**（也叫 micro-scaling，MXFP8 就是每 32 个元素一个 scale）：
//!
//! ```text
//! scale = max|x_block| / fmt.max()
//! q     = encode(x / scale)      // 一个字节
//! x'    = decode(q) * scale
//! ```
//!
//! 32 个元素共享一个 f32 scale，额外开销是 4 字节 / 32 元素 = 1 bit/元素，
//! 总位宽 9 bit —— 相比 f32 仍是约 3.6 倍压缩。
//!
//! ## 这个模块**不做什么**（重要）
//!
//! 本模块是**纯数值模拟**：把张量编码成 FP8 的比特、再解码回来，看这一步舍入让数值
//! 偏了多少、让 loss 变差多少。它 **不会带来任何速度或显存收益** —— 真实的 FP8 训练
//! 收益来自 GPU 的 FP8 tensor core（一条 `mma` 指令吃 8 位操作数、吞吐是 bf16 的两倍），
//! 而本仓库的算子全部跑在 CPU 的 f32 上，编解码反而更慢。
//!
//! 所以本模块的用途只有两个：
//! 1. **算得清**：给出"若权重/激活按 FP8 存，模型会损失多少精度"的定量答案；
//! 2. **对得上**：让 `cargo run -- fp8` 的输出与论文里的误差量级（FP8 相对误差
//!    约 2⁻⁴ ~ 2⁻³）互相印证，为将来接真 kernel 留下一个可对照的基准。
//!
//! 反向传播也不在这里：真实 FP8 训练是"前向 FP8、反向高精度"（或梯度用 E5M2），
//! 本模块只覆盖前向的舍入往返。

use crate::tensor::Tensor;
use rayon::prelude::*;

/// 训练侧模拟默认使用的分块大小（每 32 个元素共享一个 f32 scale）
pub const TRAIN_BLOCK: usize = 32;
/// 训练侧模拟默认使用的格式（权重与激活用 E4M3）
pub const TRAIN_FORMAT: Fp8Format = Fp8Format::E4M3;

/// FP8 的两种格式
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fp8Format {
    /// 4 位指数 + 3 位尾数：精度高、范围小，用于**权重与激活**
    E4M3,
    /// 5 位指数 + 2 位尾数：范围大、精度低，用于**梯度**
    E5M2,
}

impl Fp8Format {
    /// 尾数位数
    pub fn mantissa_bits(self) -> u32 {
        match self {
            Fp8Format::E4M3 => 3,
            Fp8Format::E5M2 => 2,
        }
    }

    /// 指数位数
    pub fn exp_bits(self) -> u32 {
        7 - self.mantissa_bits()
    }

    /// 指数偏置：`value = (1 + m/2^mb) * 2^(ef - bias)`
    pub fn bias(self) -> i32 {
        match self {
            Fp8Format::E4M3 => 7,
            Fp8Format::E5M2 => 15,
        }
    }

    /// 可表示的最大有限值
    pub fn max(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 448.0,
            Fp8Format::E5M2 => 57344.0,
        }
    }

    /// 最小正规格化数 `2^(1 - bias)`
    pub fn min_positive_normal(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 2f32.powi(-6),
            Fp8Format::E5M2 => 2f32.powi(-14),
        }
    }

    /// 最小正非规格化数 `2^(1 - bias - mantissa_bits)`
    pub fn min_positive_subnormal(self) -> f32 {
        match self {
            Fp8Format::E4M3 => 2f32.powi(-9),
            Fp8Format::E5M2 => 2f32.powi(-16),
        }
    }

    /// 是否保留 Inf 码（只有 E5M2 有；E4M3 的"指数全 1 + 尾数全 1"是 NaN）
    pub fn has_inf(self) -> bool {
        matches!(self, Fp8Format::E5M2)
    }

    pub fn name(self) -> &'static str {
        match self {
            Fp8Format::E4M3 => "e4m3",
            Fp8Format::E5M2 => "e5m2",
        }
    }

    /// 单元素相对精度：正规数区间内 `2^-(mantissa_bits+1)` 是舍入误差的上界
    pub fn unit_roundoff(self) -> f32 {
        2f32.powi(-(self.mantissa_bits() as i32) - 1)
    }

    /// 正号下的 NaN 码
    pub fn nan_code(self) -> u8 {
        match self {
            // 指数全 1（0b1111）+ 尾数全 1
            Fp8Format::E4M3 => 0x7F,
            // 指数全 1（0b11111）+ 尾数全 1
            Fp8Format::E5M2 => 0x7F,
        }
    }

    /// 正号下的 Inf 码（E4M3 没有，返回最大有限值码）
    pub fn inf_code(self) -> u8 {
        match self {
            Fp8Format::E4M3 => self.max_finite_code(),
            Fp8Format::E5M2 => 0x7C,
        }
    }

    /// 正号下的最大有限值码
    pub fn max_finite_code(self) -> u8 {
        match self {
            Fp8Format::E4M3 => 0x7E, // 指数 1111 + 尾数 110
            Fp8Format::E5M2 => 0x7B, // 指数 11110 + 尾数 11
        }
    }

    /// 指数的无偏上限（最大有限值对应的 `2^e`）
    fn max_exp(self) -> i32 {
        match self {
            Fp8Format::E4M3 => 8,  // ef=15, bias=7
            Fp8Format::E5M2 => 15, // ef=30（ef=31 是 Inf/NaN）, bias=15
        }
    }
}

/// `2^e`（`e` 落在 f32 正规数范围内时精确）
fn exp2(e: i32) -> f32 {
    f32::from_bits(((e + 127) as u32) << 23)
}

/// 四舍五入到最近的偶数（round-to-nearest-ties-to-even），输入必须非负且有限。
///
/// 硬件做浮点舍入用的就是这个规则；手写是为了不依赖 `f32::round_ties_even`
/// 的版本门槛，也让"奇偶往哪边靠"这件事在代码里看得见。
fn round_ties_even(v: f32) -> f32 {
    let f = v.floor();
    let d = v - f;
    if d > 0.5 {
        f + 1.0
    } else if d < 0.5 {
        f
    } else if (f as i64) % 2 == 0 {
        f
    } else {
        f + 1.0
    }
}

/// 把一个 f32 编成 FP8 码（舍入到最近偶数，溢出按格式处理）
///
/// - `NaN` → NaN 码
/// - `±0` → `0x00` / `0x80`（两个零码都保留）
/// - 超出 `fmt.max()`：E5M2 给 Inf，E4M3 饱和到最大有限值（它没有 Inf 码）
pub fn encode(x: f32, fmt: Fp8Format) -> u8 {
    let sign = if x.is_sign_negative() { 0x80u8 } else { 0 };
    let ax = x.abs();
    if ax.is_nan() {
        return sign | fmt.nan_code();
    }
    if ax == 0.0 {
        return sign;
    }
    if ax > fmt.max() {
        // E5M2 报 Inf；E4M3 无 Inf，按 OCP 规范饱和到最大有限值
        return sign | if fmt.has_inf() { fmt.inf_code() } else { fmt.max_finite_code() };
    }

    let mb = fmt.mantissa_bits();
    let bias = fmt.bias();
    let bits = ax.to_bits();
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    let mant23 = bits & 0x7F_FFFF;
    // ax 有限且 > 0：f32 的指数域要么是正规（1..=254），要么是次正规（0）
    let e = if raw_exp == 0 { i32::MIN } else { raw_exp - 127 };
    let emin = 1 - bias;
    let one = 1u32 << mb;

    if e >= emin {
        // 正规数：尾数 = (significand - 1) * 2^mb，再舍入
        let sig = f32::from_bits(0x3F80_0000 | mant23);
        let mut m = round_ties_even((sig - 1.0) * one as f32) as i32;
        let mut ee = e;
        if m >= one as i32 {
            // 尾数进位到 1.0：指数 +1、尾数归零
            m = 0;
            ee += 1;
            if ee > fmt.max_exp() {
                return sign | if fmt.has_inf() { fmt.inf_code() } else { fmt.max_finite_code() };
            }
        }
        sign | (((ee + bias) as u8) << mb) | (m as u8)
    } else {
        // 次正规（含 f32 次正规）：步长 `2^(emin - mb)`。
        // 若舍入到 `2^mb`，得到的码恰好是最小正规数（ef = 1, m = 0），无需特判。
        let step = exp2(emin - mb as i32);
        let m = round_ties_even(ax / step) as u8;
        sign | m
    }
}

/// 把一个 FP8 码解回 f32（精确，无额外舍入）
pub fn decode(code: u8, fmt: Fp8Format) -> f32 {
    let neg = code & 0x80 != 0;
    let sign = if neg { -1.0f32 } else { 1.0 };
    let mb = fmt.mantissa_bits();
    let ef_mask = (1u8 << fmt.exp_bits()) - 1;
    let ef = (code >> mb) & ef_mask;
    let m = (code & ((1u8 << mb) - 1)) as u32;

    if ef == ef_mask {
        if fmt.has_inf() {
            return if m == 0 { sign * f32::INFINITY } else { f32::NAN };
        }
        if m == (1u32 << mb) - 1 {
            return f32::NAN; // E4M3 的唯一 NaN 码
        }
    }
    if ef == 0 {
        // 非规格化：value = m * 2^(emin - mb)
        return sign * (m as f32) * exp2(1 - fmt.bias() - mb as i32);
    }
    let sig = 1.0 + (m as f32) / (1u32 << mb) as f32;
    sign * sig * exp2(ef as i32 - fmt.bias())
}

/// 往返量化误差统计
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ErrStats {
    /// 最大绝对误差
    pub max_abs: f32,
    /// 平均绝对误差
    pub mean_abs: f64,
    /// 信噪比（dB）：`10·log10(Σx² / Σ(x−x')²)`，越大越准；全 0 输入返回 `inf`
    pub snr_db: f64,
}

/// 逐元素往返量化后的张量（码 + 分块 scale）
///
/// 数据布局与 [`Tensor`] 一致（行优先展平），但存储是"每元素 1 字节 + 每 block 一个 f32"。
#[derive(Clone, Debug)]
pub struct Fp8Tensor {
    /// FP8 码，长度 = 元素个数
    codes: Vec<u8>,
    /// 分块缩放因子，长度 = `ceil(n / block)`
    scales: Vec<f32>,
    fmt: Fp8Format,
    block: usize,
    shape: Vec<usize>,
}

impl Fp8Tensor {
    /// 量化一个张量：按 `block` 个元素一组取 `max|x|` 定 scale，再逐元素编码
    pub fn quantize(t: &Tensor, fmt: Fp8Format, block: usize) -> Self {
        let d = t.decode();
        let dref: &[f32] = &d;
        Self::from_slice(dref, t.shape().to_vec(), fmt, block)
    }

    /// 量化一段行优先数据（`shape` 的元素个数必须等于 `data.len()`）
    pub fn from_slice(data: &[f32], shape: Vec<usize>, fmt: Fp8Format, block: usize) -> Self {
        assert!(block >= 1, "分块大小必须 >= 1（实际 {block}）");
        let expect: usize = shape.iter().product();
        assert_eq!(expect, data.len(), "shape {shape:?} 的元素数与数据长度不符");
        let n = data.len();
        let mut codes = vec![0u8; n];
        let mut scales = vec![0f32; n.div_ceil(block)];
        codes
            .par_chunks_mut(block)
            .zip(scales.par_iter_mut())
            .enumerate()
            .for_each(|(i, (chunk, scale))| {
                let base = i * block;
                let mut amax = 0f32;
                for k in 0..chunk.len() {
                    amax = amax.max(data[base + k].abs());
                }
                if amax == 0.0 {
                    // 全零块：scale 取 0，码全 0（避免 0/0）
                    *scale = 0.0;
                    return;
                }
                let s = amax / fmt.max();
                *scale = s;
                let inv = 1.0 / s;
                // 必须先 clamp 再编码：`1/(amax/max)` 的舍入误差会让"恰好等于 amax"的那个
                // 元素算出来比 max 大一丁点，E5M2 于是把它编成 Inf —— 一个元素溢出就把整张
                // 张量污染成 inf/NaN。真实的 FP8 kernel 也是"先按 amax 定 scale 再饱和"。
                let lim = fmt.max();
                for k in 0..chunk.len() {
                    chunk[k] = encode((data[base + k] * inv).clamp(-lim, lim), fmt);
                }
            });
        Fp8Tensor {
            codes,
            scales,
            fmt,
            block,
            shape,
        }
    }

    /// 解回行优先的 f32 数据
    pub fn dequantize_flat(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.codes.len()];
        out.par_chunks_mut(self.block)
            .enumerate()
            .for_each(|(i, chunk)| {
                let s = self.scales[i];
                let base = i * self.block;
                for k in 0..chunk.len() {
                    chunk[k] = decode(self.codes[base + k], self.fmt) * s;
                }
            });
        out
    }

    /// 解回一个与原张量同形状的 f32 张量（**不参与梯度**：模拟量化的存储精度）
    pub fn dequantize(&self) -> Tensor {
        Tensor::from_vec(self.dequantize_flat(), self.shape.clone())
    }

    /// 元素个数
    pub fn elements(&self) -> usize {
        self.codes.len()
    }

    /// 分块个数
    pub fn blocks(&self) -> usize {
        self.scales.len()
    }

    /// 实际占用字节：码 1 字节/元素 + scale 4 字节/块
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }

    /// 与量化前的张量比较，统计往返误差
    pub fn errors(&self, original: &Tensor) -> ErrStats {
        assert_eq!(
            original.numel(),
            self.codes.len(),
            "原始张量的元素数应与量化结果一致"
        );
        let d = original.decode();
        let dref: &[f32] = &d;
        let rt = self.dequantize_flat();
        let mut max_abs = 0f32;
        let mut sum_abs = 0f64;
        let mut sig = 0f64;
        let mut noise = 0f64;
        for (i, &x) in dref.iter().enumerate() {
            let e = (x - rt[i]).abs();
            max_abs = max_abs.max(e);
            sum_abs += e as f64;
            sig += (x as f64) * (x as f64);
            noise += (e as f64) * (e as f64);
        }
        let n = dref.len().max(1) as f64;
        ErrStats {
            max_abs,
            mean_abs: sum_abs / n,
            snr_db: if noise == 0.0 {
                f64::INFINITY
            } else {
                10.0 * (sig / noise).log10()
            },
        }
    }
}

/// 把张量按 FP8 往返量化一次（等价于"以 FP8 精度存储后读回"）
pub fn roundtrip(t: &Tensor, fmt: Fp8Format, block: usize) -> Tensor {
    Fp8Tensor::quantize(t, fmt, block).dequantize()
}

/// 模拟一次"操作数先落 FP8 再相乘"的矩阵乘法：`a @ b`，但 `a`、`b` 先各自往返量化。
///
/// 真实 FP8 是 kernel 里直接吃 8 位操作数；这里把量化/反量化摊开来做，跑的还是 f32 的
/// `matmul`，所以**只等价于数值、不等价于性能**。
pub fn simulate_matmul(a: &Tensor, b: &Tensor, fmt: Fp8Format, block: usize) -> Tensor {
    let aq = roundtrip(a, fmt, block);
    let bq = roundtrip(b, fmt, block);
    aq.matmul(&bq)
}

/// 把参数表里的**二维权重矩阵**原地往返量化到 FP8（模拟 FP8 权重存储）。
///
/// 只处理 `rank == 2` 的参数：RMSNorm 的增益、各种 bias 是一维的，它们的数值尺度
/// 与矩阵权重差好几个数量级，真实 FP8 训练也把这类参数留在高精度（bf16/f32）。
///
/// 返回 `(被量化的张量数, 被量化的元素数, 模拟占用的字节数)`。
pub fn roundtrip_params_in_place(
    params: &[Tensor],
    fmt: Fp8Format,
    block: usize,
) -> (usize, usize, usize) {
    let mut tensors = 0usize;
    let mut elements = 0usize;
    let mut bytes = 0usize;
    for p in params {
        if p.rank() != 2 {
            continue;
        }
        let shape = p.shape().to_vec();
        let numel = p.numel();
        // 先把原始数据读出来量化，再写回——两个借用不能重叠
        let rounded = {
            let d = p.decode();
            let dref: &[f32] = &d;
            let q = Fp8Tensor::from_slice(dref, shape, fmt, block);
            bytes += q.bytes();
            q.dequantize_flat()
        };
        let src: &[f32] = &rounded;
        let mut dst = p.decode_mut();
        dst.copy_from_slice(src);
        tensors += 1;
        elements += numel;
    }
    (tensors, elements, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module; // `parameters()` 来自 Module trait

    /// 二进制表示对照：这些码是 FP8 的"标准答案"，与 OCP / NVIDIA 文档一致
    #[test]
    fn test_known_bit_patterns_encode_and_decode() {
        // E4M3：1.0 → 0x38，0.5 → 0x30，1.5 → 0x3C，448 → 0x7E，-1.0 → 0xB8
        for (x, code) in [
            (1.0f32, 0x38u8),
            (0.5, 0x30),
            (1.5, 0x3C),
            (2.0, 0x40),
            (448.0, 0x7E),
            (-1.0, 0xB8),
            (-448.0, 0xFE),
        ] {
            assert_eq!(encode(x, Fp8Format::E4M3), code, "E4M3 编码 {x} 应为 {code:#04X}");
            assert_eq!(
                decode(code, Fp8Format::E4M3),
                x,
                "E4M3 解码 {code:#04X} 应为 {x}"
            );
        }
        // E5M2：1.0 → 0x3C，1.5 → 0x3E，57344 → 0x7B，-1.0 → 0xBC
        for (x, code) in [
            (1.0f32, 0x3Cu8),
            (1.5, 0x3E),
            (2.0, 0x40),
            (57344.0, 0x7B),
            (-1.0, 0xBC),
        ] {
            assert_eq!(encode(x, Fp8Format::E5M2), code, "E5M2 编码 {x} 应为 {code:#04X}");
            assert_eq!(
                decode(code, Fp8Format::E5M2),
                x,
                "E5M2 解码 {code:#04X} 应为 {x}"
            );
        }
        // 两种格式的"1.0"码不同：E4M3 是 0x38，E5M2 是 0x3C —— 指数位宽不同
        assert_ne!(
            encode(1.0, Fp8Format::E4M3),
            encode(1.0, Fp8Format::E5M2)
        );
    }

    /// 非规格化数与两个零：FP8 的次正规码用于表示比 `min_positive_normal` 更小的数
    #[test]
    fn test_subnormals_and_signed_zero() {
        // E4M3：最小非规格化 2⁻⁹ = 0x01；2⁻⁶ 是最小正规数 = 0x08
        assert_eq!(encode(2f32.powi(-9), Fp8Format::E4M3), 0x01);
        assert_eq!(decode(0x01, Fp8Format::E4M3), 2f32.powi(-9));
        assert_eq!(encode(2f32.powi(-6), Fp8Format::E4M3), 0x08);
        // 次正规区间每格 2⁻⁹：3·2⁻⁹ 应编码成 0x03
        assert_eq!(encode(3.0 * 2f32.powi(-9), Fp8Format::E4M3), 0x03);
        // E5M2：最小非规格化 2⁻¹⁶ = 0x01
        assert_eq!(encode(2f32.powi(-16), Fp8Format::E5M2), 0x01);
        assert_eq!(encode(2f32.powi(-14), Fp8Format::E5M2), 0x04);
        // 两个零码都保留，符号可分辨
        assert_eq!(encode(0.0, Fp8Format::E4M3), 0x00);
        assert_eq!(encode(-0.0, Fp8Format::E4M3), 0x80);
        assert!(decode(0x80, Fp8Format::E4M3).is_sign_negative());
        // 比最小非规格化还小一半 → 舍入到 0（不产生"负零以外的垃圾"）
        assert_eq!(encode(2f32.powi(-12), Fp8Format::E4M3), 0x00);
    }

    /// 溢出与特殊值：E4M3 无 Inf 只有饱和，E5M2 有 Inf，两者都有 NaN
    #[test]
    fn test_overflow_inf_and_nan() {
        // E4M3：超出 448 饱和到最大有限值 0x7E
        assert_eq!(encode(1e9, Fp8Format::E4M3), 0x7E);
        assert_eq!(decode(encode(1e9, Fp8Format::E4M3), Fp8Format::E4M3), 448.0);
        assert!(!Fp8Format::E4M3.has_inf());
        // E5M2：超出 57344 给 Inf（0x7C）
        assert_eq!(encode(1e9, Fp8Format::E5M2), 0x7C);
        assert!(decode(0x7C, Fp8Format::E5M2).is_infinite());
        assert!(decode(0xFC, Fp8Format::E5M2).is_infinite());
        // NaN：编码保留符号，解码回来还是 NaN
        assert!(decode(encode(f32::NAN, Fp8Format::E4M3), Fp8Format::E4M3).is_nan());
        assert!(decode(encode(-f32::NAN, Fp8Format::E5M2), Fp8Format::E5M2).is_nan());
        // 0x7F 在 E4M3 是 NaN、在 E5M2 也是 NaN
        assert!(decode(0x7F, Fp8Format::E4M3).is_nan());
        assert!(decode(0x7F, Fp8Format::E5M2).is_nan());
        // ±Inf 的编码
        assert_eq!(encode(f32::INFINITY, Fp8Format::E5M2), 0x7C);
        assert_eq!(encode(f32::NEG_INFINITY, Fp8Format::E4M3), 0xFE);
    }

    /// 舍入规则：打成平手时向偶数靠（不是简单地"四舍五入"）
    #[test]
    fn test_round_to_nearest_ties_to_even() {
        let f = Fp8Format::E5M2; // 尾数 2 位，区间 [1,2) 内步长 0.25
        let at = |x: f32| decode(encode(x, f), f);
        // 1.125 落在 1.0 与 1.25 正中间 → 偶数尾数 0 → 1.0
        assert_eq!(at(1.125), 1.0);
        // 1.375 落在 1.25 与 1.5 正中间 → 尾数 1 是奇数、2 是偶数 → 1.5
        assert_eq!(at(1.375), 1.5);
        // 1.625 落在 1.5 与 1.75 正中间 → 尾数 2 是偶数 → 1.5
        assert_eq!(at(1.625), 1.5);
        // 1.875 落在 1.75 与 2.0 正中间 → 尾数 3 是奇数、进位到 4 → 指数 +1 → 2.0
        assert_eq!(at(1.875), 2.0);
        // 非平手时朝更近的一侧：1.1 → 1.0，1.2 → 1.25
        assert_eq!(at(1.1), 1.0);
        assert_eq!(at(1.2), 1.25);
        // 正规数区间的相对误差上界就是 unit_roundoff = 2^-3 = 0.125（E5M2）
        assert_eq!(f.unit_roundoff(), 0.125);
        assert_eq!(Fp8Format::E4M3.unit_roundoff(), 0.0625);
    }

    /// 分块缩放的意义：动态范围大的数据上，per-block 明显优于"整张共用一个 scale"
    #[test]
    fn test_per_block_beats_single_scale_on_wide_dynamic_range() {
        // 前半段量级 1.0，后半段量级 1e-6 —— 相差 6 个数量级（真实权重里不同输出通道
        // 的尺度差几个数量级很常见）。注意 FP8 是浮点，**相对**精度与 scale 无关
        // （3 位尾数恒给 ~6%），整张共用一个 scale 的真正代价是把小的那些压进非规格化
        // 区间、直至压成 0 —— 所以只统计小量级那半段才看得见差别，整张 SNR 会被大元素主导。
        let mut data = vec![0f32; 4096];
        for (i, v) in data.iter_mut().enumerate() {
            let amp = if i < 2048 { 1.0 } else { 1e-6 };
            *v = amp * (1.0 + (i % 17) as f32 * 0.05);
        }
        let t = Tensor::from_vec(data.clone(), vec![4096]);

        let single = Fp8Tensor::quantize(&t, Fp8Format::E4M3, 4096);
        let block32 = Fp8Tensor::quantize(&t, Fp8Format::E4M3, 32);

        // 小量级那半段的平均相对误差
        let small_rel = |rt: &[f32]| -> f64 {
            let acc: f64 = (2048..4096)
                .map(|i| ((data[i] - rt[i]).abs() / data[i].abs()) as f64)
                .sum();
            acc / 2048.0
        };
        let r_single = small_rel(&single.dequantize_flat());
        let r_block = small_rel(&block32.dequantize_flat());

        assert!(
            r_block < 0.10,
            "分块缩放下小量级元素的相对误差应 < 10%（实际 {:.1}%）",
            r_block * 100.0
        );
        assert!(
            r_single > 0.5,
            "整张共用一个 scale 时小量级元素应基本被压平（实际相对误差 {:.1}%）",
            r_single * 100.0
        );
        // 大元素那半段两种做法一样准（相对精度与 scale 无关），差别只在小元素上
        let e_single = single.errors(&t);
        let e_block = block32.errors(&t);
        assert!(e_block.mean_abs <= e_single.mean_abs);
        assert!(e_single.snr_db > 25.0 && e_block.snr_db > 25.0);
        // 存储开销：码 1 字节/元素 + scale 4 字节/块
        assert_eq!(single.bytes(), 4096 + 4);
        assert_eq!(block32.elements(), 4096);
        assert_eq!(block32.blocks(), 128);
        assert_eq!(block32.bytes(), 4096 + 128 * 4);
    }

    /// 细粒度格式对比：数值范围够用时，E4M3 的精度优于 E5M2
    #[test]
    fn test_e4m3_more_precise_than_e5m2_on_well_scaled_data() {
        let mut rng = crate::rng::Rng::new(7);
        let data: Vec<f32> = (0..2048).map(|_| rng.randn() * 0.5).collect();
        let t = Tensor::from_vec(data, vec![64, 32]);

        let e4 = Fp8Tensor::quantize(&t, Fp8Format::E4M3, 32).errors(&t);
        let e5 = Fp8Tensor::quantize(&t, Fp8Format::E5M2, 32).errors(&t);
        assert!(
            e4.snr_db > e5.snr_db,
            "数据范围正常时 E4M3 应更准：E4M3 {:.1} dB vs E5M2 {:.1} dB",
            e4.snr_db,
            e5.snr_db
        );
        // 但 E5M2 能表示的范围大得多（这也是梯度用它的原因）
        assert!(Fp8Format::E5M2.max() > Fp8Format::E4M3.max() * 100.0);
        assert!(Fp8Format::E5M2.min_positive_normal() < Fp8Format::E4M3.min_positive_normal());
        // 两者的相对精度差一倍：E4M3 是 2^-4，E5M2 是 2^-3
        assert_eq!(Fp8Format::E5M2.unit_roundoff(), 2.0 * Fp8Format::E4M3.unit_roundoff());
    }

    /// 模拟 matmul：误差落在 FP8 应有的量级（个位数百分比），不是"差不多等于 0"
    #[test]
    fn test_simulate_matmul_error_is_in_fp8_range() {
        let mut rng = crate::rng::Rng::new(11);
        let a = Tensor::from_vec((0..128 * 64).map(|_| rng.randn()).collect(), vec![128, 64]);
        let b = Tensor::from_vec((0..64 * 96).map(|_| rng.randn()).collect(), vec![64, 96]);

        let exact = a.matmul(&b);
        let de = exact.decode();
        let de_ref: &[f32] = &de;
        let peak = de_ref.iter().fold(0f32, |m, v| m.max(v.abs()));

        let rel_fro_of = |f: Fp8Format| -> f64 {
            let sim = simulate_matmul(&a, &b, f, 32);
            assert_eq!(sim.shape(), exact.shape());
            let ds = sim.decode();
            let ds_ref: &[f32] = &ds;
            let (mut sig, mut noise, mut max_abs) = (0f64, 0f64, 0f32);
            for i in 0..de_ref.len() {
                let e = (de_ref[i] - ds_ref[i]).abs();
                assert!(e.is_finite(), "{} 模拟结果出现非有限值 —— 多半是编码溢出成了 Inf", f.name());
                sig += (de_ref[i] as f64).powi(2);
                noise += (e as f64) * (e as f64);
                max_abs = max_abs.max(e);
            }
            // 单元素误差用"相对整个输出的峰值"衡量：输出里有接近 0 的元素，
            // 逐元素相对误差会被除零放大到没有意义的数字（这类指标陷阱值得记住）
            assert!(
                max_abs < 0.2 * peak,
                "{} 的单元素最大误差应远小于输出峰值（{max_abs:.4} vs 峰值 {peak:.4}）",
                f.name()
            );
            (noise / sig).sqrt()
        };

        // E4M3：每个操作数相对误差 ~2^-4/√3 ≈ 3.6%，两项独立叠加后输出约 5%
        let rel_e4 = rel_fro_of(Fp8Format::E4M3);
        assert!(
            rel_e4 < 0.10,
            "E4M3 模拟 matmul 的相对 Frobenius 误差应在百分之几量级（实际 {:.2}%）",
            rel_e4 * 100.0
        );
        assert!(
            rel_e4 > 1e-3,
            "误差不该小到看不出来（实际 {:.3e}）——量化可能没生效",
            rel_e4
        );
        // E5M2 尾数少一位，误差应该更大（但同样必须有限：不能有元素溢出成 Inf）
        let rel_e5 = rel_fro_of(Fp8Format::E5M2);
        assert!(
            rel_e5 > rel_e4,
            "E5M2 的误差应大于 E4M3（{:.2}% vs {:.2}%）",
            rel_e5 * 100.0,
            rel_e4 * 100.0
        );
    }

    /// 分块缩放必须饱和，不能把"刚好等于块内最大值的元素"编成 Inf
    ///
    /// `1/(amax/max)` 的舍入误差会让这个元素算出来比 max 大一丁点；若不 clamp，
    /// E5M2 会把它编成 Inf，一个元素就把整张张量污染成 inf/NaN。
    #[test]
    fn test_block_scale_saturates_instead_of_overflowing_to_inf() {
        let mut rng = crate::rng::Rng::new(23);
        for (fmt, block) in [
            (Fp8Format::E4M3, 32),
            (Fp8Format::E5M2, 32),
            (Fp8Format::E5M2, 7),
        ] {
            for _ in 0..64 {
                let data: Vec<f32> = (0..1000).map(|_| rng.randn() * 3.0).collect();
                let t = Tensor::from_vec(data, vec![1000]);
                let rt = roundtrip(&t, fmt, block);
                let d = rt.decode();
                for (i, &v) in d.iter().enumerate() {
                    assert!(
                        v.is_finite(),
                        "{}（block={}）的第 {i} 个元素往返后是 {v} —— 溢出被编成了 Inf",
                        fmt.name(),
                        block
                    );
                }
            }
        }
    }

    /// 原地量化只碰二维权重：一维的归一化增益/bias 必须原样不动
    #[test]
    fn test_roundtrip_params_in_place_only_touches_matrices() {
        let model = crate::model::Transformer::new(
            crate::model::TransformerConfig {
                n_embd: 32,
                n_head: 4,
                n_layer: 2,
                block_size: 16,
                dropout: 0.0,
                ..crate::model::TransformerConfig::tiny(64)
            },
            &mut crate::rng::Rng::new(3),
        );
        let params = model.parameters();
        let before: Vec<Vec<f32>> = params.iter().map(|p| p.decode().to_vec()).collect();
        let (tensors, elements, bytes) =
            roundtrip_params_in_place(&params, Fp8Format::E4M3, TRAIN_BLOCK);

        assert!(tensors > 0 && elements > 0);
        // 压缩比：8 位码 + 每 32 元素一个 f32 scale = 9 bit/元素 ≈ 1.125 字节
        assert!(
            bytes < elements * 2,
            "FP8 模拟占用的字节（{bytes}）应远小于 f32 的 {}",
            elements * 4
        );

        let mut changed_2d = 0usize;
        let mut max_abs_1d = 0f32;
        let mut worst_snr = f64::INFINITY;
        for (i, p) in params.iter().enumerate() {
            let after = p.decode();
            if p.rank() == 2 {
                if before[i] != after.to_vec() {
                    changed_2d += 1;
                }
                // 用 SNR 而不是"逐元素最大相对误差"：权重里有接近 0 的元素，
                // 那种元素的相对误差可以到几百 %，但绝对误差始终被 block scale 卡住
                let (mut sig, mut noise) = (0f64, 0f64);
                for (a, b) in before[i].iter().zip(after.iter()) {
                    sig += (*a as f64) * (*a as f64);
                    let e = (*a - *b) as f64;
                    noise += e * e;
                }
                let snr = 10.0 * (sig / noise.max(f64::MIN_POSITIVE)).log10();
                worst_snr = worst_snr.min(snr);
            } else {
                let d = before[i]
                    .iter()
                    .zip(after.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0f32, f32::max);
                max_abs_1d = max_abs_1d.max(d);
            }
        }
        assert_eq!(changed_2d, tensors, "被改动的二维张量数应与返回值一致");
        assert_eq!(max_abs_1d, 0.0, "一维参数（norm/bias）必须一位不改");
        // 二维权重被量化过，但相对误差必须落在 FP8 的量级内（约 3~4% → 28 dB 以上）
        assert!(
            worst_snr > 25.0,
            "二维权重的往返 SNR 应 > 25 dB（最差 {worst_snr:.1} dB，对应相对误差 {:.1}%）",
            100.0 / 10f64.powf(worst_snr / 20.0)
        );
    }

    /// loss 影响：把权重按 FP8 存储后，验证 loss 的恶化必须很小
    #[test]
    fn test_fp8_weight_roundtrip_barely_moves_loss() {
        use crate::loss::cross_entropy_loss_masked;
        use crate::optim::{AdamW, Optimizer};

        let tiny = |seed: u64| {
            crate::model::Transformer::new(
                crate::model::TransformerConfig {
                    n_embd: 32,
                    n_head: 4,
                    n_layer: 2,
                    block_size: 12,
                    dropout: 0.0,
                    ..crate::model::TransformerConfig::tiny(64)
                },
                &mut crate::rng::Rng::new(seed),
            )
        };
        let corpus: Vec<usize> = crate::data::CORPUS.chars().map(|c| c as usize % 64).collect();

        // 先正常训几步，让权重进入"有结构"的状态（全随机权重下 loss 对扰动不敏感）
        let model = tiny(5);
        let mut opt = AdamW::new(0.05, model.parameters(), 0.0);
        let mut rng = crate::rng::Rng::new(9);
        for _ in 0..30 {
            let mut x = Vec::new();
            let mut y = Vec::new();
            for _ in 0..4 {
                let s = (rng.next_u64() as usize) % (corpus.len() - 12 - 1);
                x.extend_from_slice(&corpus[s..s + 12]);
                y.extend_from_slice(&corpus[s + 1..s + 13]);
            }
            let logits = model.forward(&x, 4, 12, None, true);
            let l = cross_entropy_loss_masked(&logits, &y, None);
            opt.zero_grad();
            l.backward();
            opt.step();
        }

        // 评测集
        let mut ex = Vec::new();
        let mut ey = Vec::new();
        for k in 0..4 {
            let s = 200 + k * 40;
            ex.extend_from_slice(&corpus[s..s + 12]);
            ey.extend_from_slice(&corpus[s + 1..s + 13]);
        }
        let eval = |m: &crate::model::Transformer| -> f32 {
            crate::tensor::no_grad(|| {
                let logits = m.forward(&ex, 4, 12, None, false);
                cross_entropy_loss_masked(&logits, &ey, None).item()
            })
        };

        let before = eval(&model);
        let params = model.parameters();
        roundtrip_params_in_place(&params, TRAIN_FORMAT, TRAIN_BLOCK);
        let after = eval(&model);

        // 量化是往权重里注入扰动，loss 可升可降，但幅度必须很小 ——
        // 否则说明 scale 取错（比如把 E4M3 的 448 写成别的值，权重被压成 0 或饱和）
        let rel = (after - before).abs() as f64 / before.max(1e-6) as f64;
        assert!(
            rel < 0.05,
            "FP8 权重往返后 loss 变动应 < 5%（实际 {:.2}%：{before:.4} → {after:.4}）",
            rel * 100.0
        );
        // 顺带确认重算一次是幂等的：再量化一遍不应继续变差
        roundtrip_params_in_place(&params, TRAIN_FORMAT, TRAIN_BLOCK);
        let again = eval(&model);
        assert!(
            (again - after).abs() <= 1e-5,
            "已量化的权重再量化一次应几乎不变（{after:.6} → {again:.6}）"
        );
    }
}
