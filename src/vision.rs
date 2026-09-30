//! 图片理解（Vision Transformer，手写）与图像 IO
//!
//! # 设计
//! - **图像 IO 用 `image` crate**（PNG/JPEG 解码、双线性缩放属于工具库，同 serde/clap）；
//!   归一化、patchify、整个 ViT 前向/反向都是纯手写，跑在本项目的 Tensor/autograd 上。
//! - ViT 结构（最朴素的原始版本）：
//!   `patchify → Linear 打平成 patch embedding → 可学习位置 embedding →
//!    N 层 Transformer Block（mask=None，**双向**注意力）→ LayerNorm → Linear 投影到主干维度`
//! - 输出的每个 patch 特征不会进序列做 token，而是在 [`crate::model::Transformer::forward_mm`]
//!   里**原位覆写**文本序列中的 `<|image|>` 占位符 —— 序列长度不变，KV cache / loss 对齐
//!   完全不受影响（详见该方法的文档）。
//!
//! # 张量约定
//! 像素统一为 **CHW、边长 `image_size`、取值 [-1, 1]** 的 `f32` 切片（不含 batch 维，
//! 批量时按样本首尾相接）；patch 按**行主序**遍历（左上 → 右 → 下）。

use crate::layers::{Linear, NormLayer};
use crate::model::{TransformerBlock, TransformerConfig, LN_EPS};
use crate::rng::Rng;
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// 视觉编码器的超参（作为 [`TransformerConfig::vision`] 挂在主干配置上，
/// 随 checkpoint 一起序列化；缺席（`None`）= 纯文本模型，行为与加本模块之前逐位一致）。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VisionConfig {
    /// 输入图像边长（正方形）
    pub image_size: usize,
    /// patch 边长；须整除 `image_size`
    pub patch_size: usize,
    /// ViT 内部隐藏维度（与主干 n_embd 无关）
    pub n_embd: usize,
    pub n_head: usize,
    pub n_layer: usize,
    /// 注意力/残差 dropout
    pub dropout: f32,
    /// 文本序列中占位符的**首 id**（即分词器 `ImageTokens::ph_first`）。
    /// 由训练/推理入口从分词器读出后写入配置 —— 模型不依赖分词器，只能这样对齐。
    /// 0 = 未设置（此时任何带图前向都会报错）。
    pub ph_first: usize,
}

impl Default for VisionConfig {
    fn default() -> Self {
        VisionConfig {
            image_size: 64,
            patch_size: 16,
            n_embd: 64,
            n_head: 4,
            n_layer: 2,
            dropout: 0.0,
            ph_first: 0,
        }
    }
}

impl VisionConfig {
    /// 一张图的 patch 数 P = (image_size / patch_size)² —— 同时也是
    /// `<|image|>` 展开的占位符个数与一次前向要覆写的位置数。
    pub fn patch_count(&self) -> usize {
        let g = self.image_size / self.patch_size;
        g * g
    }

    /// 单个 patch 展平后的元素数 3·ps²（RGB）
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// 校验配置自洽（训练入口在建模型前调用，早失败早报错）
    pub fn validate(&self) {
        assert!(
            self.image_size > 0 && self.patch_size > 0 && self.image_size % self.patch_size == 0,
            "image_size（{}）必须是 patch_size（{}）的整数倍",
            self.image_size,
            self.patch_size
        );
        assert!(self.n_embd % self.n_head == 0, "ViT 的 n_embd 必须能被 n_head 整除");
        assert!(self.ph_first > 0, "VisionConfig.ph_first 未设置：模型不知道占位符 id");
    }
}

// ==================== 图像 IO ====================

/// 读图 → 缩放到 `size × size` → CHW、[-1,1]
pub fn load_image(path: &str, size: usize) -> Vec<f32> {
    let img = image::open(path).unwrap_or_else(|e| panic!("打开图片 {path} 失败: {e}"));
    let rgb = img.to_rgb8();
    let resized = image::imageops::resize(
        &rgb,
        size as u32,
        size as u32,
        image::imageops::FilterType::Triangle,
    );
    let mut chw = Vec::with_capacity(3 * size * size);
    for c in 0..3 {
        for i in 0..size * size {
            let v = resized.as_raw()[i * 3 + c];
            chw.push(v as f32 / 127.5 - 1.0);
        }
    }
    chw
}

/// CHW、[-1,1] → RGB8 图像（[`save_image`] 与 [`encode_png`] 共用的像素搬运）
fn rgb_image(chw: &[f32], size: usize) -> image::RgbImage {
    assert_eq!(chw.len(), 3 * size * size, "像素数组长度应为 3·size²");
    let mut raw = vec![0u8; 3 * size * size];
    for c in 0..3 {
        for i in 0..size * size {
            let v = (chw[c * size * size + i] * 127.5 + 127.5).round().clamp(0.0, 255.0);
            raw[i * 3 + c] = v as u8;
        }
    }
    image::RgbImage::from_raw(size as u32, size as u32, raw).expect("RgbImage::from_raw 失败：长度已校验过")
}

/// CHW、[-1,1] → PNG（`image_size` × `image_size`）
pub fn save_image(chw: &[f32], size: usize, path: &str) {
    let img = rgb_image(chw, size);
    crate::config::ensure_parent_dir(path);
    image::DynamicImage::ImageRgb8(img)
        .save(path)
        .unwrap_or_else(|e| panic!("保存图片 {path} 失败: {e}"));
}

/// CHW、[-1,1] → PNG 字节（编码逻辑与 [`save_image`] 完全一致，只是落在内存里）。
///
/// serve 用它把生成图内嵌成 `data:image/png;base64,...` 回给客户端——
/// 不必为了几张生成图再开一个静态文件端点。
pub fn encode_png(chw: &[f32], size: usize) -> Vec<u8> {
    let img = image::DynamicImage::ImageRgb8(rgb_image(chw, size));
    let mut buf = std::io::Cursor::new(Vec::new());
    img.write_to(&mut buf, image::ImageFormat::Png)
        .expect("PNG 编码失败");
    buf.into_inner()
}

/// CHW → `[P, 3·ps²]` 展平（patch 按行主序），返回 `Vec<f32>`。
///
/// 这是**纯数据搬运**（无梯度需求：像素是常量输入），因此在 Rust 侧完成，
/// 避免为了切片在 autograd 里再引入一整套索引算子。
pub fn patchify(chw: &[f32], size: usize, patch: usize) -> Vec<f32> {
    assert_eq!(chw.len(), 3 * size * size, "像素数组长度应为 3·size²");
    let grid = size / patch;
    let mut out = Vec::with_capacity(grid * grid * 3 * patch * patch);
    for gy in 0..grid {
        for gx in 0..grid {
            for c in 0..3 {
                for y in 0..patch {
                    for x in 0..patch {
                        let sy = gy * patch + y;
                        let sx = gx * patch + x;
                        out.push(chw[c * size * size + sy * size + sx]);
                    }
                }
            }
        }
    }
    out
}

// ==================== ViT 编码器 ====================

/// 手写 ViT：把一张图变成 `P` 个主干维度的 patch 特征。
///
/// 不引入 CLS token：所有 patch 都要回填到文本序列的对应占位符上，
/// 加 CLS 反而没有去处（最基础的图片理解不需要它）。
pub struct VisionEncoder {
    pub cfg: VisionConfig,
    /// patch 打平 → 隐藏维度
    patch_embed: Linear,
    /// 可学习的位置 embedding `[P, d]`
    pos: Tensor,
    /// 双向 Transformer Block（复用主干的 [`TransformerBlock`]）
    blocks: Vec<TransformerBlock>,
    ln: NormLayer,
    /// 隐藏维度 → 主干 `n_embd`
    proj: Linear,
}

impl VisionEncoder {
    /// `main_cfg` 提供归一化/FFN 风格（RMSNorm/SwiGLU）与主干维度 `n_embd`，
    /// 让视觉塔与主干的"体质"保持一致；结构层数/头数由 `vcfg` 决定。
    pub fn new(vcfg: &VisionConfig, main_cfg: &TransformerConfig, rng: &mut Rng) -> Self {
        vcfg.validate();
        let p = vcfg.patch_count();
        // Block 用的子配置：层宽/头数/层数取视觉塔自己的，风格（norm/FFN/dropout）随主干
        let block_cfg = TransformerConfig {
            vocab_size: 0,
            n_embd: vcfg.n_embd,
            n_head: vcfg.n_head,
            n_layer: vcfg.n_layer,
            // Block 内部只有 RoPE 需要窗口；双向注意力在 base=0 下等价于"窗口 ≥ 序列长"
            block_size: p.max(2),
            use_rmsnorm: main_cfg.use_rmsnorm,
            use_swiglu: main_cfg.use_swiglu,
            dropout: vcfg.dropout,
            ..Default::default()
        };
        VisionEncoder {
            cfg: vcfg.clone(),
            patch_embed: Linear::new(vcfg.patch_dim(), vcfg.n_embd, rng),
            // 位置 embedding 用小正态初始化（经典 ViT 做法）
            pos: Tensor::param(normal(p * vcfg.n_embd, 0.02, rng), vec![p, vcfg.n_embd]),
            blocks: (0..vcfg.n_layer).map(|_| TransformerBlock::new(&block_cfg, rng)).collect(),
            ln: NormLayer::new(vcfg.n_embd, LN_EPS, main_cfg.use_rmsnorm),
            proj: Linear::new(vcfg.n_embd, main_cfg.n_embd, rng),
        }
    }

    /// 前向：`pixels` 是 `[b, 3·S·S]` 首尾相接的 CHW 像素，返回 `[b·P, n_embd]`。
    ///
    /// 行顺序即样本顺序、样本内 patch 行主序 —— 与 [`patchify`]、
    /// [`crate::model::Transformer::forward_mm`] 扫描占位符的顺序严格一致。
    pub fn forward(&self, pixels: &[f32], b: usize, training: bool) -> Tensor {
        let vcfg = &self.cfg;
        let p = vcfg.patch_count();
        assert_eq!(
            pixels.len(),
            b * 3 * vcfg.image_size * vcfg.image_size,
            "像素数量应为 b·3·S²"
        );
        // 1. patchify（Rust 侧数据搬运）→ 打平成 [b·P, 3ps²]
        let mut flat = Vec::with_capacity(b * p * vcfg.patch_dim());
        for s in 0..b {
            let off = s * 3 * vcfg.image_size * vcfg.image_size;
            let end = off + 3 * vcfg.image_size * vcfg.image_size;
            flat.extend(patchify(&pixels[off..end], vcfg.image_size, vcfg.patch_size));
        }
        // 像素是常量输入（无需对输入求导），用 from_vec；梯度从 Linear 权重那侧进入。
        // Block 期望 3-D [B,T,D]，这里直接把 b 当 batch、P 当序列长
        let x = Tensor::from_vec(flat, vec![b, p, vcfg.patch_dim()]);
        let x = self.patch_embed.forward(&x).reshape(vec![b, p, vcfg.n_embd]);

        // 2. 位置 embedding：pos 是 [P,d]，按样本重复铺成 [b·P,d] 再还原成 3-D。
        //    用 gather_rows 而不是复制成常量 —— 复制成常量的话 pos 永远拿不到梯度，
        //    位置信息就再也不会更新了。
        let rows: Vec<usize> = (0..b * p).map(|i| i % p).collect();
        let pos = self.pos.gather_rows(&rows).reshape(vec![b, p, vcfg.n_embd]);
        let x = x.add(&pos);

        // 3. 双向注意力：mask=None 且 base=0（无 KV cache），patch 之间互相可见
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x, None, None, 0, training);
        }

        // 4. 归一化 + 投影到主干维度，最后摊平回 [b·P, 主干 n_embd]
        let x = self.ln.forward(&x);
        let x = self.proj.forward(&x);
        let n_out = self.proj.weight.shape()[1];
        x.reshape(vec![b * p, n_out])
    }

    /// 带名字的参数（checkpoint 用），前缀固定 `vision.*`
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut ps = self.patch_embed.named_parameters("vision.patch_embed");
        ps.push(("vision.pos".to_string(), self.pos.clone()));
        for (i, block) in self.blocks.iter().enumerate() {
            ps.extend(block.named_parameters(&format!("vision.blocks.{i}")));
        }
        ps.extend(self.ln.named_parameters("vision.ln"));
        ps.extend(self.proj.named_parameters("vision.proj"));
        ps
    }
}

/// 标准正态 × σ，用于位置 embedding 初始化
fn normal(n: usize, sigma: f32, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| sigma * rng.randn()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// patchify 必须是确定性的行主序切分：第一块 = 左上角 3·ps² 个像素
    #[test]
    fn test_patchify_row_major_order() {
        let size = 4;
        let ps = 2;
        // CHW：每个通道填不同的常数，通道 c 的值为 c
        let chw: Vec<f32> = (0..3).flat_map(|c| vec![c as f32; size * size]).collect();
        let patches = patchify(&chw, size, ps);
        assert_eq!(patches.len(), 4 * 3 * ps * ps);
        // 每个 patch 内：先 RGB 通道，再行、列（patchify 的循环顺序）
        for (i, chunk) in patches.chunks(3 * ps * ps).enumerate() {
            for c in 0..3 {
                for v in &chunk[c * ps * ps..(c + 1) * ps * ps] {
                    assert_eq!(*v, c as f32, "patch {i} 的通道 {c} 应恒为 {c}");
                }
            }
        }
    }

    /// ViT 前向的形状：输出行数 = b·P，列数 = 主干维度
    #[test]
    fn test_vit_output_shape() {
        let mut rng = Rng::new(0);
        let vcfg = VisionConfig {
            image_size: 8,
            patch_size: 4,
            n_embd: 16,
            n_head: 2,
            n_layer: 1,
            dropout: 0.0,
            ph_first: 10,
        };
        let main = TransformerConfig::default();
        let enc = VisionEncoder::new(&vcfg, &main, &mut rng);
        let b = 2;
        let pixels = vec![0.0f32; b * 3 * 8 * 8];
        let out = enc.forward(&pixels, b, false);
        assert_eq!(out.shape(), &[b * vcfg.patch_count(), main.n_embd]);
    }
}
