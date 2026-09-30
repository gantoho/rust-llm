//! 图片生成（VQ-VAE，手写）：把图像离散化成 codebook token，供语言模型 学习/生成
//!
//! # 设计
//! - 本项目没有 conv2d，因此编码/解码与 ViT 同一路数：**patchify → Linear MLP**。
//!   整条前后向都跑在本项目的 Tensor/autograd 上，零新增依赖。
//! - **无 detach 算子**，VQ 的三类损失全部用「`Tensor::from_vec` 脱图常量」实现梯度隔离：
//!   1. **recon loss**：`‖decoder(z_q) − target‖²`，target 是常量像素 → 梯度经 straight-through
//!      进入 encoder（`z_q = z_e + const(z_q − z_e)`，前向值等于量化结果、反向 ∂z_q/∂z_e = I）；
//!   2. **codebook loss**：`‖sg[z_e] − e[idx]‖²`，`z_e` 快照成常量 → 梯度只进码本；
//!   3. **commitment loss**：`β‖z_e − sg[e[idx]]‖²`，量化目标快照成常量 → 梯度只进编码器。
//! - 最近邻量化在 Rust 侧算（`‖z‖² − 2z·e + ‖e‖²`），不进 tape —— argmax 本来就不该有梯度。
//!
//! # 张量约定
//! 与 [`crate::vision`] 完全一致：像素 CHW、边长 `image_size`、取值 [-1,1]，
//! 批量时按样本首尾相接；patch 行主序。
//!
//! # 生成通路
//! 语言模型生成 `codebook_size` 范围内的 token 序列（每个 token = 一个 patch 的码本 id），
//! [`Vqvae::decode`] 把 `P` 个 id 还原成一张图 —— 这就是最基础的图片生成。

use crate::layers::Linear;
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::{self, Tensor};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// VQ-VAE 超参（独立于 [`crate::model::TransformerConfig`] —— 它是另一个模型，
/// 权重也单独存档；此处仍实现 serde，随存档头一起序列化）。
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default)]
pub struct VqConfig {
    /// 输入图像边长（正方形）
    pub image_size: usize,
    /// patch 边长；须整除 `image_size`
    pub patch_size: usize,
    /// 编码器输出维度 = 码本向量维度 D
    pub latent_dim: usize,
    /// 码本容量 K（同时也是生成时 token 的取值范围大小）
    pub codebook_size: usize,
    /// MLP 隐层宽度
    pub hidden: usize,
    /// commitment loss 权重 β（原论文 0.25）
    pub beta: f32,
}

impl Default for VqConfig {
    fn default() -> Self {
        VqConfig {
            image_size: 64,
            patch_size: 8,
            latent_dim: 64,
            codebook_size: 512,
            hidden: 256,
            beta: 0.25,
        }
    }
}

impl VqConfig {
    /// 一张图的 patch 数 P —— 生成时语言模型要输出的 token 数
    pub fn patch_count(&self) -> usize {
        let g = self.image_size / self.patch_size;
        g * g
    }

    /// 单个 patch 展平后的元素数 3·ps²（RGB）
    pub fn patch_dim(&self) -> usize {
        3 * self.patch_size * self.patch_size
    }

    /// 校验配置自洽（训练入口在建模型前调用）
    pub fn validate(&self) {
        assert!(
            self.image_size > 0 && self.patch_size > 0 && self.image_size % self.patch_size == 0,
            "image_size（{}）必须是 patch_size（{}）的整数倍",
            self.image_size,
            self.patch_size
        );
        assert!(self.latent_dim > 0, "latent_dim 必须为正");
        assert!(self.codebook_size > 1, "codebook_size 必须大于 1");
        assert!(self.beta >= 0.0, "beta 不能为负");
    }
}

/// patchify 的逆：`[P, 3·ps²]` 展平 → CHW（行主序还原）
pub fn unpatchify(patches: &[f32], size: usize, patch: usize) -> Vec<f32> {
    let grid = size / patch;
    let pd = 3 * patch * patch;
    assert_eq!(patches.len(), grid * grid * pd, "patch 元素数应为 grid²·3·ps²");
    let mut chw = vec![0.0f32; 3 * size * size];
    for (pi, chunk) in patches.chunks(pd).enumerate() {
        let (gy, gx) = (pi / grid, pi % grid);
        let mut off = 0;
        for c in 0..3 {
            for y in 0..patch {
                for x in 0..patch {
                    let dy = gy * patch + y;
                    let dx = gx * patch + x;
                    chw[c * size * size + dy * size + dx] = chunk[off];
                    off += 1;
                }
            }
        }
    }
    chw
}

/// 最近邻量化（纯 Rust，不进 tape）：`z [n, D]` 对码本 `[K, D]` 求最近邻。
/// 返回 `(每行的码本 id, 量化后的向量展平)`。
fn quantize(z: &[f32], codebook: &[f32], d: usize, k: usize) -> (Vec<usize>, Vec<f32>) {
    assert_eq!(z.len() % d, 0);
    let n = z.len() / d;
    // ‖e_j‖² 预计算，避免每行每列重复求平方和
    let e_norm: Vec<f32> = (0..k)
        .map(|j| (0..d).map(|t| codebook[j * d + t] * codebook[j * d + t]).sum())
        .collect();
    let mut idx = Vec::with_capacity(n);
    let mut zq = vec![0.0f32; z.len()];
    for i in 0..n {
        let zi = &z[i * d..(i + 1) * d];
        let z2: f32 = zi.iter().map(|v| v * v).sum();
        let mut best = 0usize;
        let mut best_dist = f32::INFINITY;
        for j in 0..k {
            let ej = &codebook[j * d..(j + 1) * d];
            let dot: f32 = zi.iter().zip(ej).map(|(a, b)| a * b).sum();
            let dist = z2 - 2.0 * dot + e_norm[j];
            if dist < best_dist {
                best_dist = dist;
                best = j;
            }
        }
        idx.push(best);
        zq[i * d..(i + 1) * d].copy_from_slice(&codebook[best * d..(best + 1) * d]);
    }
    (idx, zq)
}

/// 手写 VQ-VAE：图像 ↔ 离散码本 token
pub struct Vqvae {
    pub cfg: VqConfig,
    /// patch 打平 → 隐层
    enc1: Linear,
    /// 隐层 → 码本空间
    enc2: Linear,
    /// 码本 `[K, D]`（2 维才能 gather_rows）
    codebook: Tensor,
    /// 码本空间 → 隐层
    dec1: Linear,
    /// 隐层 → patch 像素
    dec2: Linear,
}

impl Vqvae {
    pub fn new(cfg: VqConfig, rng: &mut Rng) -> Self {
        cfg.validate();
        Vqvae {
            enc1: Linear::new(cfg.patch_dim(), cfg.hidden, rng),
            enc2: Linear::new(cfg.hidden, cfg.latent_dim, rng),
            // 小正态初始化码本（与位置 embedding 同一手法）
            codebook: Tensor::param(
                normal(cfg.codebook_size * cfg.latent_dim, 0.02, rng),
                vec![cfg.codebook_size, cfg.latent_dim],
            ),
            dec1: Linear::new(cfg.latent_dim, cfg.hidden, rng),
            dec2: Linear::new(cfg.hidden, cfg.patch_dim(), rng),
            cfg,
        }
    }

    /// 像素 → patch 行（Rust 侧数据搬运，同 [`crate::vision::VisionEncoder::forward`]）
    fn flatten_pixels(&self, pixels: &[f32], b: usize) -> Vec<f32> {
        let s = self.cfg.image_size;
        assert_eq!(pixels.len(), b * 3 * s * s, "像素数量应为 b·3·S²");
        let mut flat = Vec::with_capacity(b * self.cfg.patch_count() * self.cfg.patch_dim());
        for sample in 0..b {
            let off = sample * 3 * s * s;
            flat.extend(crate::vision::patchify(
                &pixels[off..off + 3 * s * s],
                s,
                self.cfg.patch_size,
            ));
        }
        flat
    }

    /// 编码器前向：patch 行 `[n, 3ps²]` → 码本空间 `[n, D]`
    fn encode_latent(&self, flat: &[f32]) -> Tensor {
        let n = flat.len() / self.cfg.patch_dim();
        let x = Tensor::from_vec(flat.to_vec(), vec![n, self.cfg.patch_dim()]);
        self.enc2.forward(&self.enc1.forward(&x).relu())
    }

    /// 训练用总损失（0 维标量，可直接 `backward()`）。
    ///
    /// `pixels` 为 `[b, 3·S·S]` 首尾相接的 CHW 像素；三项损失的梯度去向见模块文档。
    pub fn forward_loss(&self, pixels: &[f32], b: usize) -> Tensor {
        let cfg = &self.cfg;
        let (p, pd, d) = (cfg.patch_count(), cfg.patch_dim(), cfg.latent_dim);
        let n = b * p;
        let flat = self.flatten_pixels(pixels, b);

        // 1. 编码 + Rust 侧最近邻量化（argmax 无梯度）
        let z_e = self.encode_latent(&flat);
        let z_e_data = z_e.data();
        let (idx, zq) = quantize(&z_e_data, &self.codebook.data(), d, cfg.codebook_size);

        // 2. straight-through：z_q = z_e + 常量差值。
        //    前向值 = z_q_data（量化结果），反向 ∂z_q/∂z_e = I —— 常量不进图，
        //    码本因此收不到 recon 的梯度（由 codebook loss 负责），编码器收得到。
        let diff: Vec<f32> = zq.iter().zip(&z_e_data).map(|(a, c)| a - c).collect();
        let z_q = z_e.add(&Tensor::from_vec(diff, vec![n, d]));

        // 3. 解码重建
        let recon = self.dec2.forward(&self.dec1.forward(&z_q).relu());
        let inv_n = 1.0 / n as f32;

        // recon loss：target 常量脱图 → 梯度经 ST 进编码器/解码器
        let recon_loss = recon
            .sub(&Tensor::from_vec(flat, vec![n, pd]))
            .pow(2.0)
            .sum()
            .mul_scalar(inv_n);

        // codebook loss：z_e 快照成常量 → 梯度只进码本（scatter 回命中行）
        let codebook_loss = self
            .codebook
            .gather_rows(&idx)
            .sub(&Tensor::from_vec(z_e_data.clone(), vec![n, d]))
            .pow(2.0)
            .sum()
            .mul_scalar(inv_n);

        // commitment loss：量化目标快照成常量 → 梯度只进编码器
        let commit_loss = z_e
            .sub(&Tensor::from_vec(zq, vec![n, d]))
            .pow(2.0)
            .sum()
            .mul_scalar(cfg.beta * inv_n);

        recon_loss.add(&codebook_loss).add(&commit_loss)
    }

    /// 图像 → 每个 patch 的码本 id（推理路径，`P` 个）
    pub fn encode(&self, pixels: &[f32], b: usize) -> Vec<usize> {
        tensor::no_grad(|| {
            let flat = self.flatten_pixels(pixels, b);
            let z_e = self.encode_latent(&flat);
            quantize(&z_e.data(), &self.codebook.data(), self.cfg.latent_dim, self.cfg.codebook_size).0
        })
    }

    /// 码本 id（`b·P` 个）→ 图像（`[b, 3·S·S]` 首尾相接的 CHW）
    ///
    /// 这是生成的最后一步：语言模型输出 `P` 个 token，这里还原成像素存 PNG。
    pub fn decode(&self, ids: &[usize], b: usize) -> Vec<f32> {
        tensor::no_grad(|| {
            let cfg = &self.cfg;
            let (p, pd, d, k) = (cfg.patch_count(), cfg.patch_dim(), cfg.latent_dim, cfg.codebook_size);
            assert_eq!(ids.len(), b * p, "id 数量应为 b·P");
            let cb = self.codebook.data();
            let mut zq = Vec::with_capacity(ids.len() * d);
            for &i in ids {
                assert!(i < k, "码本 id 越界：{i} >= {k}");
                zq.extend_from_slice(&cb[i * d..(i + 1) * d]);
            }
            let z = Tensor::from_vec(zq, vec![ids.len(), d]);
            let out = self.dec2.forward(&self.dec1.forward(&z).relu());
            let data = out.data();
            let mut res = Vec::with_capacity(b * 3 * cfg.image_size * cfg.image_size);
            for s in 0..b {
                res.extend(unpatchify(&data[s * p * pd..(s + 1) * p * pd], cfg.image_size, cfg.patch_size));
            }
            res
        })
    }

    /// 带名字的参数（存档按名对齐），前缀为模型内部结构名
    pub fn named_parameters(&self) -> Vec<(String, Tensor)> {
        let mut ps = self.enc1.named_parameters("enc1");
        ps.extend(self.enc2.named_parameters("enc2"));
        ps.push(("codebook".to_string(), self.codebook.clone()));
        ps.extend(self.dec1.named_parameters("dec1"));
        ps.extend(self.dec2.named_parameters("dec2"));
        ps
    }

    /// 独立存档：魔数 + JSON 头（配置 + 参数名序）+ f32 小端数据块。
    ///
    /// 与 [`crate::checkpoint`] 的 Transformer 存档格式分离 —— VQ-VAE 是另一个模型，
    /// 没有优化器状态对齐问题（训练时 m/v 不跨进程恢复也可从头训）。
    pub fn save(&self, path: &str) {
        let params = self.named_parameters();
        let header = VqCkptHeader {
            cfg: self.cfg.clone(),
            names: params.iter().map(|(n, _)| n.clone()).collect(),
        };
        let json = serde_json::to_vec(&header).expect("VQ-VAE 存档头序列化失败");
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&(json.len() as u32).to_le_bytes());
        buf.extend_from_slice(&json);
        for (_, t) in &params {
            for v in t.data() {
                buf.extend_from_slice(&v.to_le_bytes());
            }
        }
        crate::config::ensure_parent_dir(path);
        std::fs::write(path, &buf).unwrap_or_else(|e| panic!("写入 VQ-VAE 存档 {path} 失败: {e}"));
    }

    /// 读档（按名对齐：档内参数必须全部存在于模型中）
    pub fn load(path: &str) -> Self {
        let bytes =
            std::fs::read(path).unwrap_or_else(|e| panic!("读取 VQ-VAE 存档 {path} 失败: {e}"));
        assert!(
            bytes.len() >= MAGIC.len() + 4 && &bytes[..MAGIC.len()] == MAGIC,
            "VQ-VAE 存档魔数不对：{path}"
        );
        let mut off = MAGIC.len();
        let json_len = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
        off += 4;
        let header: VqCkptHeader = serde_json::from_slice(&bytes[off..off + json_len])
            .unwrap_or_else(|e| panic!("VQ-VAE 存档头解析失败: {e}"));
        off += json_len;

        let mut rng = Rng::new(0);
        let model = Self::new(header.cfg, &mut rng);
        let by_name: HashMap<String, Tensor> =
            model.named_parameters().into_iter().collect();
        for name in &header.names {
            let p = by_name
                .get(name)
                .unwrap_or_else(|| panic!("存档参数 {name} 在模型中不存在"));
            let n = p.numel();
            let mut vals = Vec::with_capacity(n);
            for i in 0..n {
                let s = off + i * 4;
                vals.push(f32::from_le_bytes(bytes[s..s + 4].try_into().unwrap()));
            }
            off += n * 4;
            p.set_data(vals);
        }
        model
    }
}

/// VQ-VAE 存档头（JSON）
#[derive(Serialize, Deserialize)]
struct VqCkptHeader {
    cfg: VqConfig,
    names: Vec<String>,
}

const MAGIC: &[u8; 6] = b"VQCP1\n";

impl Module for Vqvae {
    fn parameters(&self) -> Vec<Tensor> {
        self.named_parameters().into_iter().map(|(_, t)| t).collect()
    }
}

/// 标准正态 × σ（同 vision.rs 的初始化手法）
fn normal(n: usize, sigma: f32, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|_| sigma * rng.randn()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn small_cfg() -> VqConfig {
        VqConfig {
            image_size: 8,
            patch_size: 4,
            latent_dim: 8,
            codebook_size: 16,
            hidden: 16,
            beta: 0.25,
        }
    }

    /// unpatchify 必须精确还原 patchify 的输入（往返恒等）
    #[test]
    fn test_unpatchify_inverts_patchify() {
        let size = 8;
        let ps = 4;
        // 用可辨识的非均匀值（sin 打散）避免常数图掩盖顺序错误
        let chw: Vec<f32> = (0..3 * size * size)
            .map(|i| (i as f32 * 0.37).sin())
            .collect();
        let patches = crate::vision::patchify(&chw, size, ps);
        let back = unpatchify(&patches, size, ps);
        assert_eq!(back.len(), chw.len());
        for (i, (a, b)) in chw.iter().zip(&back).enumerate() {
            assert!((a - b).abs() < 1e-6, "像素 {i} 不还原：{a} vs {b}");
        }
    }

    /// encode 的 id 数量 = b·P 且全部落在 [0, K)；decode 出的像素长度 = b·3·S²
    #[test]
    fn test_encode_decode_roundtrip_shapes() {
        let mut rng = Rng::new(3);
        let cfg = small_cfg();
        let p = cfg.patch_count();
        let model = Vqvae::new(cfg.clone(), &mut rng);
        let b = 2;
        let pixels: Vec<f32> = (0..b * 3 * cfg.image_size * cfg.image_size)
            .map(|i| ((i as f32 * 0.11).sin()))
            .collect();

        let ids = model.encode(&pixels, b);
        assert_eq!(ids.len(), b * p);
        for &i in &ids {
            assert!(i < cfg.codebook_size, "id {i} 越界");
        }

        let out = model.decode(&ids, b);
        assert_eq!(out.len(), b * 3 * cfg.image_size * cfg.image_size);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    /// 总损失是 0 维标量、数值有限；backward 后编码器、解码器、码本都拿到非零梯度
    /// —— 证明三损失的梯度隔离（from_vec 常量脱图）与 straight-through 都接对了
    #[test]
    fn test_forward_loss_backward_all_get_grads() {
        let mut rng = Rng::new(7);
        let cfg = small_cfg();
        let model = Vqvae::new(cfg.clone(), &mut rng);
        let b = 2;
        let pixels: Vec<f32> = (0..b * 3 * cfg.image_size * cfg.image_size)
            .map(|i| ((i as f32 * 0.13).cos() * 0.5))
            .collect();

        let loss = model.forward_loss(&pixels, b);
        assert_eq!(loss.rank(), 0, "总损失应为 0 维标量");
        let v = loss.item();
        assert!(v.is_finite() && v >= 0.0, "损失应为非负有限值：{v}");

        loss.backward();

        let grad_of = |name: &str| -> f32 {
            let ps: HashMap<String, Tensor> = model.named_parameters().into_iter().collect();
            let g = ps[name].grad();
            g.iter().map(|x| x.abs()).sum::<f32>()
        };
        assert!(grad_of("enc1.weight") > 0.0, "编码器应有梯度（recon 经 ST + commitment）");
        assert!(grad_of("dec2.weight") > 0.0, "解码器应有梯度（recon）");
        assert!(grad_of("codebook") > 0.0, "码本应有梯度（codebook loss）");
    }

    /// codebook loss 的梯度隔离：z_e 快照成常量后，码本梯度与「解码器反传」无关。
    /// 这里用一个更直接的判据：只算 codebook 项时编码器梯度为 0。
    #[test]
    fn test_codebook_loss_isolates_encoder() {
        let mut rng = Rng::new(11);
        let cfg = small_cfg();
        let model = Vqvae::new(cfg.clone(), &mut rng);
        let b = 1;
        let pixels = vec![0.25f32; b * 3 * cfg.image_size * cfg.image_size];

        // 手工搭出 codebook 项（即 forward_loss 中的中间量）
        let flat = model.flatten_pixels(&pixels, b);
        let z_e = model.encode_latent(&flat);
        let n = z_e.shape()[0];
        let z_e_data = z_e.data();
        let (idx, _) = quantize(&z_e_data, &model.codebook.data(), cfg.latent_dim, cfg.codebook_size);
        let loss = model
            .codebook
            .gather_rows(&idx)
            .sub(&Tensor::from_vec(z_e_data, vec![n, cfg.latent_dim]))
            .pow(2.0)
            .sum();
        loss.backward();
        let enc_grad: f32 = model.enc1.weight.grad().iter().map(|x| x.abs()).sum();
        let cb_grad: f32 = model.codebook.grad().iter().map(|x| x.abs()).sum();
        assert_eq!(enc_grad, 0.0, "codebook loss 的 z_e 已脱图，编码器不应收到梯度");
        assert!(cb_grad > 0.0, "码本应收到梯度");
    }

    /// 存档往返：save → load 后所有参数逐位相等
    #[test]
    fn test_save_load_roundtrip() {
        let mut rng = Rng::new(5);
        let cfg = small_cfg();
        let model = Vqvae::new(cfg.clone(), &mut rng);
        let path = std::env::temp_dir()
            .join(format!("vqvae_roundtrip_{}.bin", std::process::id()))
            .to_string_lossy()
            .to_string();
        model.save(&path);
        let loaded = Vqvae::load(&path);
        let _ = std::fs::remove_file(&path);

        let a = model.named_parameters();
        let b = loaded.named_parameters();
        assert_eq!(a.len(), b.len());
        for ((na, ta), (nb, tb)) in a.iter().zip(&b) {
            assert_eq!(na, nb);
            assert_eq!(ta.data(), tb.data(), "参数 {na} 未对齐还原");
        }
        assert_eq!(loaded.cfg.codebook_size, cfg.codebook_size);
    }
}
