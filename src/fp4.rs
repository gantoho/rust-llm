//! FP4 量化感知训练模拟（第 48 课）
//!
//! 第 43 课的 FP8 把总位宽砍到 8 位。再往下砍一半就只有 **4 位**可用了：
//! 符号 1 位、指数 2 位、尾数 1 位 —— 这就是 **E2M1**，也是 OCP MX 规范里
//! `MXFP4` 的格式。
//!
//! ## E2M1：4 位能表示的全部 16 个数
//!
//! 指数偏置取 1，于是 `value = (1 + m/2) · 2^(ef − 1)`，正号侧只有 8 个值：
//!
//! ```text
//! ef=0（非规格化）：0     0.5
//! ef=1           ：1.0   1.5
//! ef=2           ：2.0   3.0
//! ef=3           ：4.0   6.0     ← 最大值
//! ```
//!
//! 加上符号位一共 16 个码，**每一个都是有限值**：E2M1 既没有 Inf 码也没有 NaN 码
//! （对比 E4M3 至少留了一个 NaN 码）。所以溢出与 NaN 都只能**饱和到 ±6.0**。
//!
//! 代价一眼可见：相邻两个可表示数之间的相对间距在 `[4,6]` 这一段高达 `2/4 = 50%`，
//! 而 FP8 的 E4M3 只有 `2⁻⁴ ≈ 6%`。**1 位尾数就意味着"最近邻"经常差出几十个百分点。**
//!
//! ## 分块缩放是 FP4 的必需品，不是可选项
//!
//! 正因为单个数的精度太差，FP4 必须靠 **分块 scale**（MXFP4：每 32 个元素共享一个
//! f32 scale）把小尺度的块整体抬到 `[0, 6]` 这个能用的区间里：
//!
//! ```text
//! scale = max|x_block| / 6.0
//! q     = encode_e2m1(x / scale)      // 半个字节（nibble）
//! x'    = decode_e2m1(q) * scale
//! ```
//!
//! 位宽账本：4 bit/元素 + 32/32 = 1 bit/元素 = **5 bit/元素**，对 f32 是 **6.4 倍**压缩
//! （FP8 + 每 32 一个 scale 是 9 bit/元素、3.56 倍）。省下来的显存是真金白银，
//! 代价全在 §"这个模块**不做什么**"里说清了。
//!
//! ## 这个模块**不做什么**（重要）
//!
//! 与 [`crate::fp8`] 一样，本模块是**纯数值模拟**：把张量编成 4 位码再解回来，量的是
//! "按 FP4 存，数值会差多少"。它 **不带来任何速度或显存收益** —— 真实 FP4 的收益来自
//! Blackwell 一代的 FP4 tensor core（吞吐又是 FP8 的两倍），而本仓库的算子全跑在 CPU
//! 的 f32 上，编解码只会更慢。
//!
//! 码按 **nibble 打包**（两个元素挤一个字节）存放，是为了让 [`Fp4Tensor::bytes`] 报出的
//! 字节数**如实**反映 4 bit/元素；不打包而是每元素一个 `u8` 的话，压缩比会被虚报一倍。

use crate::fp8::{ErrStats, exp2, round_ties_even};
use crate::tensor::Tensor;

/// 训练侧模拟默认使用的分块大小（每 32 个元素共享一个 f32 scale，与 MXFP4 一致）
pub const TRAIN_BLOCK: usize = 32;

/// 指数位数
pub const EXP_BITS: u32 = 2;
/// 尾数位数
pub const MANTISSA_BITS: u32 = 1;
/// 指数偏置：`value = (1 + m/2) · 2^(ef − 1)`
pub const BIAS: i32 = 1;
/// 可表示的最大有限值（`ef = 3, m = 1`）
pub const MAX: f32 = 6.0;
/// 最小正规格化数 `2^(1 − bias)`
pub const MIN_POSITIVE_NORMAL: f32 = 1.0;
/// 最小正非规格化数 `2^(1 − bias − mantissa_bits)`
pub const MIN_POSITIVE_SUBNORMAL: f32 = 0.5;
/// 单元素相对精度上界：正规数区间内 `2^−(mantissa_bits + 1)`
pub const UNIT_ROUNDOFF: f32 = 0.25;
/// 正号下的最大有限值码
pub const MAX_FINITE_CODE: u8 = 0x7;

/// 格式名（与 [`crate::fp8::Fp8Format::name`] 同一套口径，便于两处对照打印）
pub const FORMAT_NAME: &str = "e2m1";

/// E2M1 正号侧可表示的全部数值（升序）。16 个码里没有 Inf / NaN，全都在这里。
pub const REPRESENTABLE: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];

/// 把一个 f32 编成 E2M1 的 4 位码（舍入到最近偶数）。
///
/// - `±0` → `0x0` / `0x8`（两个零码都保留）
/// - `NaN` → 饱和到最大幅度（E2M1 **没有 NaN 码**，无处可放）
/// - `|x| > 6` → 饱和到 `±6.0`（同样因为没有 Inf 码）
///
/// 骨架与 [`crate::fp8::encode`] 完全一致（先判特殊值 → 算指数 → 尾数整数化再舍入 →
/// 进位则指数 +1），差别只在位宽与"没有保留码"这一点上。
pub fn encode_e2m1(x: f32) -> u8 {
    let sign = if x.is_sign_negative() { 0x8u8 } else { 0 };
    let ax = x.abs();
    if ax.is_nan() {
        // 4 位里 16 个码全是有限值，NaN 只能饱和
        return sign | MAX_FINITE_CODE;
    }
    if ax == 0.0 {
        return sign;
    }
    if ax > MAX {
        return sign | MAX_FINITE_CODE;
    }

    let mb = MANTISSA_BITS;
    // 最小正规指数：`emin = 1 - bias`，低于它的走非规格化步长
    let emin = 1 - BIAS;
    let bits = ax.to_bits();
    let mant23 = bits & 0x7F_FFFF;
    let raw_exp = ((bits >> 23) & 0xFF) as i32;
    // ax 有限且 > 0：f32 的指数域要么正规（1..=254），要么次正规（0）
    let e = if raw_exp == 0 { i32::MIN } else { raw_exp - 127 };
    let one = 1u32 << mb;

    if e >= emin {
        // 正规数：尾数 = (significand − 1) · 2^mb，四舍五入到整数
        let sig = f32::from_bits(0x3F80_0000 | mant23);
        let mut m = round_ties_even((sig - 1.0) * one as f32) as i32;
        let mut ee = e;
        if m >= one as i32 {
            // 尾数进位到 1.0：指数 +1、尾数归零
            m = 0;
            ee += 1;
            if ee > max_exp() {
                return sign | MAX_FINITE_CODE;
            }
        }
        sign | (((ee + BIAS) as u8) << mb) | (m as u8)
    } else {
        // 次正规（含 f32 次正规）：步长 `2^(emin − mb) = 0.5`。
        // 若舍入到 `2^mb`，得到的码恰好是最小正规数（ef = 1, m = 0），无需特判。
        let step = exp2(emin - mb as i32);
        let m = round_ties_even(ax / step) as u8;
        sign | m
    }
}

/// 指数的无偏上限（最大有限值 `6.0` 对应的 `2^e`，即 `ef = 3, bias = 1 → e = 2`）
fn max_exp() -> i32 {
    3 - BIAS
}

/// 把一个 E2M1 码解回 f32（精确，无额外舍入）
pub fn decode_e2m1(code: u8) -> f32 {
    let sign = if code & 0x8 != 0 { -1.0f32 } else { 1.0 };
    let ef = (code >> MANTISSA_BITS) & 0x3;
    let m = (code & 0x1) as u32;
    if ef == 0 {
        // 非规格化：value = m · 2^(emin − mb) = m · 0.5
        return sign * (m as f32) * exp2(1 - BIAS - MANTISSA_BITS as i32);
    }
    let sig = 1.0 + (m as f32) / (1u32 << MANTISSA_BITS) as f32;
    sign * sig * exp2(ef as i32 - BIAS)
}

/// 逐元素往返量化后的张量（**nibble 打包**的 4 位码 + 分块 scale）
///
/// 数据布局与 [`Tensor`] 一致（行优先展平）。元素 `i` 占字节 `i / 2` 的
/// 低半字节（`i` 为偶数）或高半字节（`i` 为奇数）。
#[derive(Clone, Debug)]
pub struct Fp4Tensor {
    /// 打包后的 4 位码，长度 = `ceil(n / 2)`
    codes: Vec<u8>,
    /// 分块缩放因子，长度 = `ceil(n / block)`
    scales: Vec<f32>,
    /// 元素个数（打包后 `codes.len() * 2` 可能多出半个填充 nibble，故显式记下）
    elements: usize,
    block: usize,
}

impl Fp4Tensor {
    /// 量化一个张量：按 `block` 个元素一组取 `max|x|` 定 scale，再逐元素编码
    pub fn quantize(t: &Tensor, block: usize) -> Self {
        let d = t.decode();
        let dref: &[f32] = &d;
        Self::from_slice(dref, t.shape().to_vec(), block)
    }

    /// 量化一段行优先数据（`shape` 的元素个数必须等于 `data.len()`）
    pub fn from_slice(data: &[f32], shape: Vec<usize>, block: usize) -> Self {
        assert!(block >= 1, "分块大小必须 >= 1（实际 {block}）");
        let expect: usize = shape.iter().product();
        assert_eq!(expect, data.len(), "shape {shape:?} 的元素数与数据长度不符");
        let n = data.len();
        let mut codes = vec![0u8; n.div_ceil(2)];
        let mut scales = vec![0f32; n.div_ceil(block)];
        for (bi, scale) in scales.iter_mut().enumerate() {
            let base = bi * block;
            let end = (base + block).min(n);
            let mut amax = 0f32;
            for &v in &data[base..end] {
                amax = amax.max(v.abs());
            }
            if amax == 0.0 {
                // 全零块：scale 取 0，码全 0（避免 0/0）
                *scale = 0.0;
                continue;
            }
            let s = amax / MAX;
            *scale = s;
            let inv = 1.0 / s;
            // 必须先 clamp 再编码：`1/(amax/max)` 的舍入误差会让"恰好等于 amax"的那个
            // 元素算出来比 MAX 大一丁点而饱和，白丢一个码。
            for k in base..end {
                let c = encode_e2m1((data[k] * inv).clamp(-MAX, MAX));
                let byte = k / 2;
                if k % 2 == 0 {
                    codes[byte] = (codes[byte] & 0xF0) | c;
                } else {
                    codes[byte] = (codes[byte] & 0x0F) | (c << 4);
                }
            }
        }
        Fp4Tensor {
            codes,
            scales,
            elements: n,
            block,
        }
    }

    /// 取第 `i` 个元素的 4 位码（低位在前）
    fn code_at(&self, i: usize) -> u8 {
        let byte = self.codes[i / 2];
        if i % 2 == 0 {
            byte & 0x0F
        } else {
            byte >> 4
        }
    }

    /// 解回行优先的 f32 数据
    pub fn dequantize_flat(&self) -> Vec<f32> {
        let mut out = vec![0f32; self.elements];
        for (bi, &s) in self.scales.iter().enumerate() {
            let base = bi * self.block;
            let end = (base + self.block).min(self.elements);
            for k in base..end {
                out[k] = decode_e2m1(self.code_at(k)) * s;
            }
        }
        out
    }

    /// 元素个数
    pub fn elements(&self) -> usize {
        self.elements
    }

    /// 分块个数
    pub fn blocks(&self) -> usize {
        self.scales.len()
    }

    /// 实际占用字节：码 `ceil(n/2)` 字节 + scale 4 字节/块
    pub fn bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }

    /// 平均位宽：`(4·n + 32·blocks) / n`。FP4 + 每 32 个元素一个 scale 恰好是 5 bit。
    pub fn bits_per_element(&self) -> f64 {
        (4.0 * self.elements as f64 + 32.0 * self.blocks() as f64) / self.elements.max(1) as f64
    }

    /// 与量化前的张量比较，统计往返误差
    pub fn errors(&self, original: &Tensor) -> ErrStats {
        assert_eq!(
            original.numel(),
            self.elements,
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

/// 把参数表里的**二维权重矩阵**原地往返量化到 FP4（模拟 FP4 权重存储）。
///
/// 只处理 `rank == 2` 的参数：RMSNorm 的增益、各种 bias 是一维的，它们的数值尺度
/// 与矩阵权重差好几个数量级，真实低精度训练也把这类参数留在高精度（bf16/f32）。
///
/// 返回 `(被量化的张量数, 被量化的元素数, 模拟占用的字节数)`。
pub fn roundtrip_params_in_place(params: &[Tensor], block: usize) -> (usize, usize, usize) {
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
            let q = Fp4Tensor::from_slice(dref, shape, block);
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
    use crate::fp8;

    /// 正号侧 8 个可表示值必须"编码 → 解码"逐位往返，且码就是 `0..=7` 本身。
    ///
    /// 这条同时钉住了格式的**全部**数值：E2M1 只有这 16 个码，没有 Inf / NaN 的余地。
    #[test]
    fn test_e2m1_representable_values_are_exact() {
        for (code, &v) in REPRESENTABLE.iter().enumerate() {
            let c = code as u8;
            assert_eq!(encode_e2m1(v), c, "E2M1 编码 {v} 应为 {c:#X}");
            assert_eq!(decode_e2m1(c), v, "E2M1 解码 {c:#X} 应为 {v}");
            // 负半轴对称
            assert_eq!(encode_e2m1(-v), c | 0x8, "E2M1 编码 {} 应为 {:#X}", -v, c | 0x8);
            assert_eq!(decode_e2m1(c | 0x8), -v, "E2M1 解码 {:#X} 应为 {}", c | 0x8, -v);
        }
    }

    /// 饱和：E2M1 既无 Inf 码也无 NaN 码，超范围与 NaN 都只能落到 ±6.0。
    #[test]
    fn test_e2m1_saturates_without_inf_or_nan() {
        for x in [6.0f32, 6.001, 7.9, 1e9, f32::INFINITY] {
            assert_eq!(encode_e2m1(x), MAX_FINITE_CODE, "{x} 应饱和到 +6.0");
        }
        for x in [-6.0f32, -7.9, -1e9, f32::NEG_INFINITY] {
            assert_eq!(encode_e2m1(x), 0x8 | MAX_FINITE_CODE, "{x} 应饱和到 -6.0");
        }
        assert_eq!(encode_e2m1(f32::NAN), MAX_FINITE_CODE, "NaN 应饱和到 +6.0");
        assert_eq!(decode_e2m1(MAX_FINITE_CODE), MAX, "最大码解码必须是 6.0");
    }

    /// 单元素精度上界：正规数区间内最近邻间距是 `2^-(mb+1) = 0.25`（相对）。
    ///
    /// 拿"非可表示、且不落在舍入中点"的数逐个核对——E2M1 的尾数只有 1 位，
    /// 误差大是**格式本身**的性质，不是实现 bug。
    #[test]
    fn test_e2m1_unit_roundoff() {
        assert_eq!(UNIT_ROUNDOFF, 0.25);
        // 0.25 落在 0 与 0.5 的正中点：ties-to-even 取偶数档（0）
        assert_eq!(encode_e2m1(0.25), 0x0, "0.25 是 0 与 0.5 的中点，应取偶数档 0");
        // 0.75 落在 0.5 与 1.0 的中点：ties-to-even 取尾数为偶的 1.0（码 0x2）
        assert_eq!(encode_e2m1(0.75), 0x2, "0.75 是 0.5 与 1.0 的中点，应取尾数为偶的 1.0");
        // 5.0 落在 4.0 与 6.0 的中点：ties-to-even 取偶数档（4.0 ← 码 6）
        assert_eq!(encode_e2m1(5.0), 0x6, "5.0 是 4 与 6 的中点，应取偶数档 4");
        // 稍偏一点就跨过中点
        assert_eq!(decode_e2m1(encode_e2m1(5.1)), 6.0);
        assert_eq!(decode_e2m1(encode_e2m1(4.9)), 4.0);
        // 最差相对误差发生在 [4,6) 段：最近邻间距 2，相对 2/4 = 50%
        let worst = (6.0 - 4.0) / 4.0;
        assert!((worst - 2.0 * UNIT_ROUNDOFF).abs() < 1e-6, "最差相对误差 50%");
    }

    /// nibble 打包必须无损且**字节数如实**：`n` 个元素占 `ceil(n/2)` 字节。
    #[test]
    fn test_nibble_packing_roundtrip_and_bytes() {
        // 8 个元素、每 4 个一组：4 字节码 + 2 块 × 4 字节 scale = 12 字节
        let data: Vec<f32> = REPRESENTABLE.to_vec();
        let q = Fp4Tensor::from_slice(&data, vec![8], 4);
        assert_eq!(q.elements(), 8);
        assert_eq!(q.blocks(), 2);
        assert_eq!(q.bytes(), 4 + 2 * 4, "8 元素 FP4 应占 ceil(8/2)=4 字节码 + 2 个 scale");
        // 每个块自己的 scale = amax/6；块内数值本身就在格点上，仅被 scale 缩放
        for (i, &x) in data.iter().enumerate() {
            let back = q.dequantize_flat()[i];
            let blk_max = if i < 4 { 1.5 } else { 6.0 };
            let expect = x / blk_max * blk_max;
            assert!((back - expect).abs() < 1e-6, "{x} 往返后应为 {expect}，实际 {back}");
        }
        // 奇数长度：最后一个字节只用了低半字节，字节数仍是 ceil(n/2)
        let odd = Fp4Tensor::from_slice(&[1.0, 1.0, 1.0], vec![3], 32);
        assert_eq!(odd.elements(), 3);
        assert_eq!(odd.bytes(), 2 + 4);
    }

    /// 位宽账本：FP4 + 每 32 元素一个 scale = 5 bit/元素（6.4 倍压缩），
    /// 比 FP8 的 9 bit/元素（3.56 倍）整整省下接近一半。
    #[test]
    fn test_bits_per_element_ledger() {
        let n = 128;
        let data: Vec<f32> = (0..n).map(|i| (i as f32 * 0.37).sin()).collect();
        let q4 = Fp4Tensor::from_slice(&data, vec![n], TRAIN_BLOCK);
        assert!((q4.bits_per_element() - 5.0).abs() < 1e-9, "FP4/32 应为 5 bit/元素");
        let t = Tensor::from_vec(data, vec![n]);
        let q8 = fp8::Fp8Tensor::quantize(&t, fp8::TRAIN_FORMAT, fp8::TRAIN_BLOCK);
        let bits8 = 8.0 * q8.elements() as f64 + 32.0 * q8.blocks() as f64;
        let bits8 = bits8 / q8.elements() as f64;
        assert!((bits8 - 9.0).abs() < 1e-9, "FP8/32 应为 9 bit/元素");
        assert!(
            q4.bytes() < q8.bytes(),
            "同样元素数下 FP4 的字节数（{}）必须少于 FP8（{}）",
            q4.bytes(),
            q8.bytes()
        );
        assert!(
            (32.0 / q4.bits_per_element() - 6.4).abs() < 1e-9,
            "对 f32 的压缩比应为 6.4 倍"
        );
    }

    /// 分块缩放确实在起作用：同一个张量上，块越小 SNR 越高（scale 更贴合局部尺度）。
    ///
    /// 用一个"行尺度差 4 个数量级"的矩阵，把"整张一个 scale"的灾难放大出来。
    #[test]
    fn test_smaller_block_improves_snr() {
        let rows = 8;
        let cols = 32;
        let mut data = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let amp = 10f32.powf(-4.0 * r as f32 / rows as f32);
            for c in 0..cols {
                data.push(amp * ((r * cols + c) as f32 * 0.31).cos());
            }
        }
        let t = Tensor::from_vec(data.clone(), vec![rows, cols]);
        let whole = Fp4Tensor::from_slice(&data, vec![rows, cols], rows * cols).errors(&t);
        let per_row = Fp4Tensor::from_slice(&data, vec![rows, cols], cols).errors(&t);
        let per_32 = Fp4Tensor::from_slice(&data, vec![rows, cols], TRAIN_BLOCK).errors(&t);
        assert!(
            per_row.snr_db > whole.snr_db,
            "每行一个 scale 应优于整张一个 scale（{per_row:?} vs {whole:?}）"
        );
        assert!(
            per_32.snr_db > whole.snr_db,
            "每 32 个一个 scale 应优于整张一个 scale（{per_32:?} vs {whole:?}）"
        );
        // FP4 的 SNR 显著低于 FP8 —— 这是"省一半位宽"必须付的价
        let q8 = fp8::Fp8Tensor::quantize(&t, fp8::TRAIN_FORMAT, fp8::TRAIN_BLOCK).errors(&t);
        assert!(
            q8.snr_db > per_32.snr_db,
            "FP8 的 SNR 必须高于 FP4（{q8:?} vs {per_32:?}）"
        );
    }

    /// 全零块不能产生 NaN：scale 取 0、码全 0，解码回来还是 0。
    #[test]
    fn test_all_zero_block_no_nan() {
        let data = vec![0f32; 64];
        let q = Fp4Tensor::from_slice(&data, vec![64], TRAIN_BLOCK);
        assert!(q.dequantize_flat().iter().all(|v| *v == 0.0), "全零块解码应全为 0");
        assert!(q.scales.iter().all(|s| *s == 0.0), "全零块的 scale 应为 0");
    }

    /// `roundtrip_params_in_place` 只碰二维参数，且会如实统计元素数与字节数。
    #[test]
    fn test_roundtrip_params_in_place_only_touches_matrices() {
        let mat = Tensor::from_vec((0..24).map(|i| i as f32 * 0.1).collect(), vec![4, 6]);
        let vec1 = Tensor::from_vec(vec![1.0, 2.0, 3.0], vec![3]);
        let before = vec1.decode().to_vec();
        let params = vec![mat, vec1];
        let (nt, ne, bytes) = roundtrip_params_in_place(&params, TRAIN_BLOCK);
        assert_eq!(nt, 1, "只应量化 1 个二维参数");
        assert_eq!(ne, 24, "元素数应为 24");
        assert_eq!(bytes, 12 + 4, "24 元素打包成 12 字节 + 1 个 scale 4 字节");
        let after = params[1].decode().to_vec();
        assert_eq!(before, after, "一维参数必须原样不动");
    }

    /// 全零张量的往返必须逐位等于 0（scale=0 的分支不能算出 NaN）。
    #[test]
    fn test_roundtrip_zero_tensor_is_zero() {
        let t = Tensor::from_vec(vec![0f32; 40], vec![40]);
        let q = Fp4Tensor::quantize(&t, TRAIN_BLOCK);
        assert!(
            q.dequantize_flat().iter().all(|v| *v == 0.0),
            "全零张量往返后仍应全为 0"
        );
    }
}
