//! 知识蒸馏（第 42 课）
//!
//! 前几课的监督信号都是**硬标签**：语料里下一个位置就是那个 token，其它 token 的
//! 概率一律算 0。可真实分布并不是 one-hot —— "the cat sat on the ___" 填 `mat` 与
//! `floor` 都讲得通，只是前者更可能。硬标签把这层信息抹平了，模型只能学到
//! "哪个是对的"，学不到"错的那个有多接近"。
//!
//! 蒸馏换一个监督源：**用一个已经训好的大模型（teacher）的完整概率分布当标签**。
//! 这个分布叫"软标签"，它带着 teacher 学到的类间相似度（也就是 Hinton 说的
//! *dark knowledge*：非目标类上那些小概率才是知识）。小模型（student）去拟合这套
//! 分布，能在同样的数据上拿到比对着硬标签训练更好的效果。
//!
//! 本模块实现其中最关键的一环：
//! - [`SoftTargets`]：把 teacher 的 logits 变成软标签（温度缩放 + 可选 top-k 截断）
//! - [`kd_loss`]：softmax 之间的 KL 散度，`T²` 缩放保证梯度量级与硬标签可比
//! - [`distill_loss`]：`(1-α)·硬标签交叉熵 + α·KD`，Hinton 的经典组合
//!
//! # 三个容易踩的坑
//!
//! 1. **`T²` 缩放不能省**。温度 T 会把 logits 除以 T 再 softmax，梯度里因此多出一个
//!    `1/T`；不加 `T²` 的话高温下 KD 项的梯度比硬标签小一两个数量级，`α` 就失去了意义。
//!    本模块的 `T²` **内置在 [`kd_loss`] 里**，调用方不要再乘一次。
//! 2. **teacher 与 student 必须共用同一个分词器**。软标签是"词表上的一条概率向量"，
//!    两个模型只要有一个 id→token 的映射不同，第 j 维就不再指同一个 token，蒸馏等于
//!    在教学生错误的对应关系。这里用 `vocab` 相等做断言，但**同词表大小 ≠ 同映射**，
//!    真正的保证是两者用同一个 `tokenizer.json`。
//! 3. **teacher 要在 `no_grad` 下前向**。教师只提供标签，不该被更新，也不该占显存存图。

use crate::autograd::record;
use crate::loss::cross_entropy_loss_masked;
use crate::tensor::Tensor;
use rayon::prelude::*;
use std::sync::Arc;

/// 蒸馏的超参：温度、KD 权重、top-k 截断
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DistillConfig {
    /// 温度 `T`：`p = softmax(logits / T)`。
    ///
    /// `T = 1` 就是 teacher 自己的分布；T 越大分布越平（"软"），负类上的小概率被放大，
    /// 类间相似度的信息更明显——但太大时分布接近均匀，等于在教噪声。
    /// 实践中 2~8 比较常用（Hinton 原文用 4）。
    pub temperature: f32,
    /// KD 项的权重 `α`：`L = (1-α)·CE(硬标签) + α·T²·KL(软标签)`。
    ///
    /// `α = 0` 退化成普通训练，`α = 1` 完全不要硬标签。有真实标签时取 0.5 左右通常最稳：
    /// 硬标签保证"答案别跑偏"，软标签提供额外的类间结构。
    pub alpha: f32,
    /// 只保留 teacher 分布的 top-k 个类别，其余置 0 并重新归一化。`0` = 不截断（全词表）。
    ///
    /// 两个好处：一是省算力（反向只需遍历被保留的那些列，长尾上的概率本来就小、
    /// 贡献有限）；二是去掉长尾噪声——`T` 放大后，词表尾部几万个小概率会被一起放大，
    /// 它们在 teacher 那里也只是噪声。k 取 8~64 是常见区间。
    pub top_k: usize,
}

impl Default for DistillConfig {
    fn default() -> Self {
        // T = 4 来自 Hinton 等 2015 的原始实验；α = 0.5 是硬标签/软标签对半；
        // top_k = 0 表示默认不截断（小词表下没必要，且不截断最容易对照出问题）
        DistillConfig {
            temperature: 4.0,
            alpha: 0.5,
            top_k: 0,
        }
    }
}

impl DistillConfig {
    /// 参数自检：温度必须为正（除零/反向软化）、α 落在 [0, 1]
    pub fn validate(&self) {
        assert!(
            self.temperature > 0.0,
            "蒸馏温度必须为正（实际 {}）",
            self.temperature
        );
        assert!(
            (0.0..=1.0).contains(&self.alpha),
            "KD 权重 α 必须落在 [0, 1]（实际 {}）",
            self.alpha
        );
    }
}

/// teacher 的软目标：`[rows, vocab]` 的概率表（行和为 1）
///
/// 先把 teacher 的 logits 除以温度、softmax，再按需截断到 top-k 并重新归一化。
/// **这一步不进计算图**：teacher 前向应该在 `no_grad` 下做，软目标是常量标签。
pub struct SoftTargets {
    cfg: DistillConfig,
    /// 行优先的 `[rows, vocab]` 概率；被 top-k 截断掉的列恒为 0
    probs: Vec<f32>,
    rows: usize,
    vocab: usize,
    /// 每行保留的 top-k 下标（按概率降序）；`None` 表示未截断
    topk: Option<Vec<u32>>,
    /// 每行的 `−Σ_j p ln p`（自然对数熵）。temperature=1 且不截断时就是 teacher 的
    /// 预测熵：熵越低说明 teacher 越"有把握"，蒸馏能提供的额外信息也越少。
    neg_entropy: Vec<f32>,
}

impl SoftTargets {
    /// 从 teacher 的 logits（`[rows, vocab]`）构造软标签。
    ///
    /// 这里用 `decode()` 直接读数据、不建图：teacher 的参数不参与任何梯度。
    pub fn from_logits(logits: &Tensor, cfg: DistillConfig) -> Self {
        cfg.validate();
        assert_eq!(
            logits.rank(),
            2,
            "teacher logits 应为 [rows, vocab]（实际 {:?}）",
            logits.shape()
        );
        let (rows, vocab) = (logits.shape()[0], logits.shape()[1]);
        assert!(rows > 0 && vocab > 0, "teacher logits 不能有空维度");
        let inv_t = 1.0 / cfg.temperature;
        // top_k 超过词表时按词表大小处理（等价于不截断）
        let k = if cfg.top_k == 0 {
            0
        } else {
            cfg.top_k.min(vocab)
        };
        let truncated = k > 0 && k < vocab;

        let src = logits.decode();
        let sr: &[f32] = &src;

        // 第一步：top-k 选择（只写 `topk` 一个缓冲区，与后面的概率计算分开做，
        // 免得同一个闭包里同时可变借用两块数据）。
        let mut topk: Option<Vec<u32>> = if truncated {
            Some(vec![0u32; rows * k])
        } else {
            None
        };
        if let Some(sel) = topk.as_mut() {
            sel.par_chunks_mut(k).enumerate().for_each(|(i, slot)| {
                let base = i * vocab;
                // 下标按"logits 降序，并列按下标升序"排——确定性是复现实验的前提
                let cmp = |&a: &u32, &b: &u32| {
                    let (za, zb) = (sr[base + a as usize], sr[base + b as usize]);
                    zb.partial_cmp(&za).unwrap().then(a.cmp(&b))
                };
                let mut order: Vec<u32> = (0..vocab as u32).collect();
                // `select_nth_unstable_by(k-1)` 把"第 k 大"的元素放到位置 k-1，
                // 它前面恰好是最大的 k 个 —— O(vocab)，不必全排序
                order.select_nth_unstable_by(k - 1, cmp);
                order.truncate(k);
                order.sort_unstable_by(cmp);
                slot.copy_from_slice(&order);
            });
        }

        // 第二步：逐行 softmax + 截断重归一化 + 熵
        let mut probs = vec![0.0f32; rows * vocab];
        let mut neg_entropy = vec![0.0f32; rows];
        let sel: Option<&[u32]> = topk.as_deref();
        probs
            .par_chunks_mut(vocab)
            .zip(neg_entropy.par_iter_mut())
            .enumerate()
            .for_each(|(i, (row, ent))| {
                let base = i * vocab;
                // 温度缩放后的 max：减掉它是为了让 exp 不溢出
                let mut mx = f32::NEG_INFINITY;
                for j in 0..vocab {
                    mx = mx.max(sr[base + j] * inv_t);
                }
                // 保留哪些列：截断时只看选中的 k 个，否则整行
                let cols: Vec<usize> = match sel {
                    Some(s) => s[i * k..(i + 1) * k].iter().map(|&j| j as usize).collect(),
                    None => (0..vocab).collect(),
                };
                let se: f32 = cols
                    .iter()
                    .map(|&j| (sr[base + j] * inv_t - mx).exp())
                    .sum();
                let inv_se = 1.0 / se;
                let mut e = 0.0f32;
                for &j in &cols {
                    let p = (sr[base + j] * inv_t - mx).exp() * inv_se;
                    row[j] = p;
                    if p > 0.0 {
                        e -= p * p.ln();
                    }
                }
                *ent = e;
            });

        SoftTargets {
            cfg,
            probs,
            rows,
            vocab,
            topk,
            neg_entropy,
        }
    }

    /// logits 的行数（= 监督位置数）
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// 词表大小（学生 logits 的最后一维必须与它相等）
    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// 温度
    pub fn temperature(&self) -> f32 {
        self.cfg.temperature
    }

    /// 每行保留的类别数：截断时 = top_k，否则 = 词表大小
    pub fn k(&self) -> usize {
        self.topk
            .as_ref()
            .map(|t| t.len() / self.rows.max(1))
            .unwrap_or(self.vocab)
    }

    /// 平均预测熵（自然对数）。平摊到每行，日志里用它看"软标签有多软"
    pub fn mean_entropy(&self) -> f64 {
        self.neg_entropy.iter().map(|&e| e as f64).sum::<f64>() / self.rows as f64
    }

    /// 概率表（行优先 `[rows, vocab]`）
    pub fn probs(&self) -> &[f32] {
        &self.probs
    }

    /// 第 `i` 行 top-1 的类别下标
    pub fn argmax(&self, i: usize) -> usize {
        let row = &self.probs[i * self.vocab..(i + 1) * self.vocab];
        let mut best = 0usize;
        for (j, &p) in row.iter().enumerate() {
            if p > row[best] {
                best = j;
            }
        }
        best
    }

    /// 学生（给一组 logits）与 teacher 的 **top-1 一致率**。
    ///
    /// 这是蒸馏最直观的指标：学生未必把 teacher 的整个分布都学到位，
    /// 但"第一名"是否一致直接决定了它生成时的行为。
    pub fn top1_agreement(&self, student_logits: &Tensor) -> f64 {
        assert_eq!(
            student_logits.shape(),
            &[self.rows, self.vocab],
            "学生 logits 形状应与软标签一致"
        );
        let sd = student_logits.decode();
        let sr: &[f32] = &sd;
        let hit = (0..self.rows)
            .filter(|&i| {
                let base = i * self.vocab;
                let mut best = 0usize;
                for j in 0..self.vocab {
                    if sr[base + j] > sr[base + best] {
                        best = j;
                    }
                }
                best == self.argmax(i)
            })
            .count();
        hit as f64 / self.rows as f64
    }

    /// 学生（给一组 logits）与 teacher 的平均 KL `KL(p_t ‖ p_s)`（**不乘 T²**）。
    ///
    /// 只用于观测/日志，所以不进计算图。注意它量的是"学生差 teacher 多远"，
    /// 与训练用的 `T²` 缩放版本差一个常数因子。
    pub fn kl_to(&self, student_logits: &Tensor) -> f64 {
        assert_eq!(
            student_logits.shape(),
            &[self.rows, self.vocab],
            "学生 logits 形状应与软标签一致"
        );
        let inv_t = 1.0 / self.cfg.temperature;
        let sd = student_logits.decode();
        let sr: &[f32] = &sd;
        let total: f64 = (0..self.rows)
            .map(|i| {
                let base = i * self.vocab;
                let mut mx = f32::NEG_INFINITY;
                for j in 0..self.vocab {
                    mx = mx.max(sr[base + j] * inv_t);
                }
                let mut se = 0.0f32;
                for j in 0..self.vocab {
                    se += (sr[base + j] * inv_t - mx).exp();
                }
                let lse = mx + se.ln();
                let pt = &self.probs[base..base + self.vocab];
                let mut kl = 0.0f64;
                for j in 0..self.vocab {
                    let p = pt[j];
                    if p > 0.0 {
                        let log_ps = sr[base + j] * inv_t - lse;
                        kl += p as f64 * (p.ln() as f64 - log_ps as f64);
                    }
                }
                kl
            })
            .sum();
        total / self.rows as f64
    }
}

/// 蒸馏的软标签损失：`T² · mean_i KL(p_t ‖ p_s)`。
///
/// - `student_logits`：`[rows, vocab]`，**必须是学生的原始 logits**（未除温度）
/// - `soft`：teacher 的软标签（含温度与 top-k 设置）
/// - `mask`：`mask[i] == false` 的位置既不进损失也不回传梯度（与 SFT 的口径一致）
///
/// 反向是手写的：`∂L/∂z_s,j = (T / valid) · (p_s,j − p_t,j)`。
/// 推导：`∂KL/∂z_j = (1/T)(p_s,j − p_t,j)`，再乘 `T²` 与 `1/valid`。
/// 这个式子不需要保存 `[rows, vocab]` 的中间量，反向按需重算 softmax 即可。
pub fn kd_loss(student_logits: &Tensor, soft: &SoftTargets, mask: Option<&[bool]>) -> Tensor {
    assert_eq!(
        student_logits.rank(),
        2,
        "学生 logits 应为 [rows, vocab]（实际 {:?}）",
        student_logits.shape()
    );
    let (rows, vocab) = (student_logits.shape()[0], student_logits.shape()[1]);
    assert_eq!(
        (rows, vocab),
        (soft.rows, soft.vocab),
        "学生 logits {:?} 与软标签 [{}, {}] 形状不符——teacher 与 student 多半没共用分词器",
        student_logits.shape(),
        soft.rows,
        soft.vocab
    );
    if let Some(m) = mask {
        assert_eq!(m.len(), rows, "mask 长度应与 logits 行数一致");
    }

    let is_sup = |i: usize| mask.map_or(true, |m| m[i]);
    let valid = (0..rows).filter(|&i| is_sup(i)).count().max(1);
    // 温度先落成局部 `f32`：反向闭包要 `'static`，不能再借用 `soft`
    let temperature = soft.cfg.temperature;
    let inv_t = 1.0 / temperature;
    let t2 = temperature * temperature;

    // ---------- 前向 ----------
    let sd = student_logits.decode();
    let sr: &[f32] = &sd;
    let pt: &[f32] = &soft.probs;
    // 每行的 KL（顺序收集 + 顺序求和，保证结果可复现）
    let per_row: Vec<f32> = (0..rows)
        .map(|i| {
            if !is_sup(i) {
                return 0.0;
            }
            let base = i * vocab;
            let mut mx = f32::NEG_INFINITY;
            for j in 0..vocab {
                mx = mx.max(sr[base + j] * inv_t);
            }
            let mut se = 0.0f32;
            for j in 0..vocab {
                se += (sr[base + j] * inv_t - mx).exp();
            }
            let lse = mx + se.ln();
            // p_t 在 top-k 之外恒为 0，循环整行也不会多算（0·ln 项被跳过）
            let mut kl = 0.0f32;
            for j in 0..vocab {
                let p = pt[base + j];
                if p > 0.0 {
                    kl += p * (p.ln() - (sr[base + j] * inv_t - lse));
                }
            }
            kl
        })
        .collect();
    let mean_kl: f32 = per_row.iter().sum::<f32>() / valid as f32;
    let value = t2 * mean_kl;
    drop(sd);

    let result = Tensor::new(vec![value], vec![], student_logits.req());
    if student_logits.req() {
        let rg = result.grad.clone();
        let sg = student_logits.grad.clone();
        let ld = student_logits.data.clone();
        let pt_rc = Arc::new(soft.probs.clone());
        let mask_rc = mask.map(|m| Arc::new(m.to_vec()));
        record(
            &result,
            vec![student_logits.clone()],
            Arc::new(move || {
                let g = rg.borrow()[0];
                let sld = ld.decode();
                let slr: &[f32] = &sld;
                let ptv: &[f32] = &pt_rc;
                // 上游标量 × (T / valid)：T² 与 ∂KL/∂z 里的 1/T 相乘的结果
                let scale = g * t2 / temperature / valid as f32;
                let m_ref: Option<&[bool]> = mask_rc.as_ref().map(|m| m.as_slice());
                let mut sgm = sg.borrow_mut();
                sgm.par_chunks_mut(vocab).enumerate().for_each(|(i, row)| {
                    if !m_ref.map_or(true, |m| m[i]) {
                        return; // 屏蔽位不回传梯度
                    }
                    let base = i * vocab;
                    let mut mx = f32::NEG_INFINITY;
                    for j in 0..vocab {
                        mx = mx.max(slr[base + j] * inv_t);
                    }
                    let mut se = 0.0f32;
                    for j in 0..vocab {
                        se += (slr[base + j] * inv_t - mx).exp();
                    }
                    let inv_se = 1.0 / se;
                    for j in 0..vocab {
                        let ps = (slr[base + j] * inv_t - mx).exp() * inv_se;
                        row[j] += scale * (ps - ptv[base + j]);
                    }
                });
            }),
        );
    }
    result
}

/// Hinton 的经典组合损失：`(1-α)·CE(硬标签) + α·T²·KL(软标签)`。
///
/// `hard_targets[i]` 是第 `i` 个位置的**正确** token（与 `student_logits` 的行一一对应）。
/// 有真实标签时两项都留着：硬标签把学生锚在正确答案上（防止它去模仿 teacher 的
/// 幻觉/长尾噪声），软标签补上"错得有多接近"这部分信息。
///
/// 两项各自成一棵子树，再按 α 加权相加——相加节点由 `Tensor::add` 记录，
/// 一次 `backward()` 就会同时把梯度送到两条路径上。
pub fn distill_loss(
    student_logits: &Tensor,
    soft: &SoftTargets,
    hard_targets: &[usize],
    mask: Option<&[bool]>,
) -> Tensor {
    let alpha = soft.cfg.alpha;
    let kd = kd_loss(student_logits, soft, mask);
    if alpha <= 0.0 {
        return cross_entropy_loss_masked(student_logits, hard_targets, mask);
    }
    let ce = cross_entropy_loss_masked(student_logits, hard_targets, mask);
    if alpha >= 1.0 {
        return kd;
    }
    ce.mul_scalar(1.0 - alpha).add(&kd.mul_scalar(alpha))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::{Transformer, TransformerConfig};
    use crate::module::Module;
    use crate::optim::{AdamW, Optimizer};
    use crate::rng::Rng;
    use crate::tokenizer::Tokenizer;
    use crate::tensor::no_grad;

    /// 造一份"logits → 手算 softmax"的参考实现，用于核对软标签
    fn manual_softmax(logits: &[f32], rows: usize, vocab: usize, t: f32) -> Vec<f32> {
        let mut out = vec![0.0f32; rows * vocab];
        for i in 0..rows {
            let row = &logits[i * vocab..(i + 1) * vocab];
            let mx = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let se: f32 = row.iter().map(|&z| ((z - mx) / t).exp()).sum();
            for j in 0..vocab {
                out[i * vocab + j] = ((row[j] - mx) / t).exp() / se;
            }
        }
        out
    }

    /// 不截断时软标签就是"温度缩放后的 softmax"：行和为 1、逐位对上参考实现。
    /// 温度升高必须让熵变大（分布更平），这是"软化"最直接的判据。
    #[test]
    fn test_soft_targets_match_manual_softmax_and_soften_with_temperature() {
        let logits = vec![
            2.0f32, 1.0, 0.5, -1.0, 0.0, 3.0, //
            -2.0, 4.0, 0.0, 1.0, -0.5, 0.2,
        ];
        let (rows, vocab) = (2usize, 6usize);
        let lg = Tensor::from_vec(logits.clone(), vec![rows, vocab]);

        for t in [1.0f32, 2.0, 4.0] {
            let cfg = DistillConfig {
                temperature: t,
                alpha: 0.5,
                top_k: 0,
            };
            let soft = SoftTargets::from_logits(&lg, cfg);
            let want = manual_softmax(&logits, rows, vocab, t);
            for (i, (&a, &b)) in soft.probs().iter().zip(want.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "T={t} 第 {i} 个概率不符：{a} vs {b}"
                );
            }
            // 行和必须归一
            for i in 0..rows {
                let s: f32 = soft.probs()[i * vocab..(i + 1) * vocab].iter().sum();
                assert!((s - 1.0).abs() < 1e-5, "第 {i} 行行和应为 1（实际 {s}）");
            }
        }
        // 熵随温度单调上升（更软 = 更平均）
        let ent = |t: f32| {
            SoftTargets::from_logits(
                &lg,
                DistillConfig {
                    temperature: t,
                    alpha: 0.5,
                    top_k: 0,
                },
            )
            .mean_entropy()
        };
        let (e1, e2, e4) = (ent(1.0), ent(2.0), ent(4.0));
        assert!(e1 < e2 && e2 < e4, "熵应随温度上升：{e1:.4} < {e2:.4} < {e4:.4}");
    }

    /// top-k 截断：恰好 k 个非零、下标就是最大的 k 个、截断后仍归一。
    #[test]
    fn test_topk_truncation_is_normalized_and_keeps_largest() {
        let logits = vec![0.1f32, 5.0, -3.0, 4.0, 0.0, -1.0, 3.0, 0.2];
        let (rows, vocab) = (1usize, 8usize);
        let lg = Tensor::from_vec(logits.clone(), vec![rows, vocab]);
        let soft = SoftTargets::from_logits(
            &lg,
            DistillConfig {
                temperature: 1.0,
                alpha: 0.5,
                top_k: 3,
            },
        );

        assert_eq!(soft.k(), 3, "应只保留 3 个类别");
        let row = soft.probs();
        let nz: Vec<usize> = (0..vocab).filter(|&j| row[j] > 0.0).collect();
        assert_eq!(nz, vec![1, 3, 6], "保留的应是最大的 3 个（下标 1/3/6）");
        let s: f32 = row.iter().sum();
        assert!((s - 1.0).abs() < 1e-5, "截断后行和仍须为 1（实际 {s}）");
        // 截断后的比例与不截断时相同（只是掐掉了尾部）
        let full = SoftTargets::from_logits(
            &lg,
            DistillConfig {
                temperature: 1.0,
                alpha: 0.5,
                top_k: 0,
            },
        );
        for &j in &nz {
            let want = full.probs()[j] / nz.iter().map(|&m| full.probs()[m]).sum::<f32>();
            assert!(
                (row[j] - want).abs() < 1e-6,
                "下标 {j} 的重新归一化比例不符：{} vs {want}",
                row[j]
            );
        }
        // top_k ≥ 词表 ⇒ 等价于不截断
        let wide = SoftTargets::from_logits(
            &lg,
            DistillConfig {
                temperature: 1.0,
                alpha: 0.5,
                top_k: 99,
            },
        );
        assert_eq!(wide.k(), vocab);
        for j in 0..vocab {
            assert!((wide.probs()[j] - full.probs()[j]).abs() < 1e-6);
        }
    }

    /// 学生与 teacher 完全一致时 KL = 0（前向），梯度也必须是 0——
    /// 这是"梯度符号/系数写错"最简单的照妖镜：如果 T² 缩放或 (p_s − p_t) 写反了，
    /// 这里立刻会看到非零梯度把自己推离最优点。
    #[test]
    fn test_kd_loss_is_zero_and_gradient_vanishes_when_student_matches_teacher() {
        let logits = vec![1.5f32, -0.5, 0.3, 2.0, -1.0, 0.0];
        let (rows, vocab) = (2usize, 3usize);
        let cfg = DistillConfig {
            temperature: 3.0,
            alpha: 0.5,
            top_k: 0,
        };
        let soft = SoftTargets::from_logits(
            &Tensor::from_vec(logits.clone(), vec![rows, vocab]),
            cfg,
        );
        // `param` 而非 `from_vec`：只有 requires_grad = true 的叶子才会收到梯度
        let student = Tensor::param(logits.clone(), vec![rows, vocab]);
        let loss = kd_loss(&student, &soft, None);
        assert!(
            loss.item().abs() < 1e-6,
            "学生与 teacher 一致时 KL 应为 0（实际 {}）",
            loss.item()
        );
        loss.backward();
        let g = student.grad.borrow();
        for (i, &v) in g.iter().enumerate() {
            assert!(v.abs() < 1e-6, "第 {i} 个梯度应为 0（实际 {v}）");
        }
    }

    /// 用有限差分核对 KD 的反向：解析梯度必须与数值梯度逐位吻合。
    ///
    /// 这条测试同时钉住三件事：`T²` 缩放、`1/valid` 归一化、以及"对每个 z_j 的
    /// 梯度里要减掉 p_t,j"。任何一项写错，误差都会远超 1e-3。
    #[test]
    fn test_kd_gradient_matches_finite_difference() {
        let (rows, vocab) = (3usize, 5usize);
        let mut rng = Rng::new(11);
        let teacher: Vec<f32> = (0..rows * vocab).map(|_| rng.randn()).collect();
        let student: Vec<f32> = (0..rows * vocab).map(|_| rng.randn()).collect();
        let cfg = DistillConfig {
            temperature: 2.5,
            alpha: 0.5,
            top_k: 3,
        };
        let soft = SoftTargets::from_logits(&Tensor::from_vec(teacher, vec![rows, vocab]), cfg);

        // 屏蔽掉一行：既检验 mask 生效，也让 valid 不等于 rows（归一化口径一同被核对）
        let mask = [true, false, true];

        let value_at = |z: &[f32]| -> f32 {
            let t = Tensor::from_vec(z.to_vec(), vec![rows, vocab]);
            kd_loss(&t, &soft, Some(&mask)).item()
        };

        let z = Tensor::param(student.clone(), vec![rows, vocab]);
        let loss = kd_loss(&z, &soft, Some(&mask));
        loss.backward();
        let analytic: Vec<f32> = z.grad.borrow().to_vec();

        let eps = 1e-3f32;
        for i in 0..rows {
            if !mask[i] {
                for j in 0..vocab {
                    assert_eq!(analytic[i * vocab + j], 0.0, "屏蔽行不应有梯度");
                }
                continue;
            }
            for j in 0..vocab {
                let mut plus = student.clone();
                plus[i * vocab + j] += eps;
                let mut minus = student.clone();
                minus[i * vocab + j] -= eps;
                let num = (value_at(&plus) - value_at(&minus)) / (2.0 * eps);
                let a = analytic[i * vocab + j];
                assert!(
                    (a - num).abs() < 2e-3,
                    "梯度不符 (row {i}, col {j})：解析 {a} vs 数值 {num}"
                );
            }
        }
    }

    /// 组合损失 `(1-α)·CE + α·T²·KL` 必须等于两项的手工加权和。
    #[test]
    fn test_distill_loss_matches_manual_blend() {
        let (rows, vocab) = (2usize, 4usize);
        let teacher = vec![1.0f32, 0.0, -1.0, 0.5, 0.3, 2.0, 0.0, -2.0];
        let student = vec![0.5f32, 0.5, 0.0, 0.0, -1.0, 1.0, 0.2, 0.1];
        let hard = [0usize, 1usize];

        for alpha in [0.0f32, 0.3, 0.5, 1.0] {
            let cfg = DistillConfig {
                temperature: 2.0,
                alpha,
                top_k: 0,
            };
            let soft = SoftTargets::from_logits(
                &Tensor::from_vec(teacher.clone(), vec![rows, vocab]),
                cfg,
            );
            let st = Tensor::from_vec(student.clone(), vec![rows, vocab]);
            let got = distill_loss(&st, &soft, &hard, None).item();
            let ce = cross_entropy_loss_masked(&st, &hard, None).item();
            // kd_loss 用的是同一份软标签，直接取它的值
            let kd = kd_loss(&st, &soft, None).item();
            let want = (1.0 - alpha) * ce + alpha * kd;
            assert!(
                (got - want).abs() < 1e-5,
                "α={alpha} 时组合损失不符：{got} vs {want}"
            );
        }
    }

    /// 反向必须**同时**流到两条路径：硬标签的梯度与 KD 的梯度都落在学生 logits 上。
    ///
    /// 判据用"α = 1 时与纯 KD 相同、α = 0 时与纯 CE 相同、中间是两者的加权平均"
    /// 这三条等价关系来钉——比单个数值更抗实现细节变化。
    #[test]
    fn test_distill_gradient_is_weighted_sum_of_both_paths() {
        let (rows, vocab) = (2usize, 3usize);
        let teacher = vec![2.0f32, 0.0, -1.0, 0.1, 1.0, 0.5];
        let student = vec![0.2f32, 0.2, 0.2, -0.4, 0.5, 0.1];
        let hard = [0usize, 2usize];
        let alpha = 0.4f32;
        let cfg = DistillConfig {
            temperature: 1.5,
            alpha,
            top_k: 0,
        };
        let soft = SoftTargets::from_logits(
            &Tensor::from_vec(teacher.clone(), vec![rows, vocab]),
            cfg,
        );

        // α 只影响两项的加权，不影响软标签本身 —— 所以改 α 只需重建一个 config
        let grad_of = |alpha: f32, hard_only: bool| -> Vec<f32> {
            let st = Tensor::param(student.clone(), vec![rows, vocab]);
            let l = if hard_only {
                cross_entropy_loss_masked(&st, &hard, None)
            } else {
                let mut c = cfg;
                c.alpha = alpha;
                let s =
                    SoftTargets::from_logits(&Tensor::from_vec(teacher.clone(), vec![rows, vocab]), c);
                kd_loss(&st, &s, None)
            };
            l.backward();
            st.grad.borrow().to_vec()
        };

        let st = Tensor::param(student.clone(), vec![rows, vocab]);
        let mixed = distill_loss(&st, &soft, &hard, None);
        mixed.backward();
        let got = st.grad.borrow().to_vec();

        let g_kd = grad_of(1.0, false);
        let g_ce = grad_of(0.0, true);
        for i in 0..rows * vocab {
            let want = alpha * g_kd[i] + (1.0 - alpha) * g_ce[i];
            assert!(
                (got[i] - want).abs() < 1e-4,
                "第 {i} 个梯度不是两项的加权和：{} vs {want}",
                got[i]
            );
        }
    }

    /// 端到端：同一份数据、同一个学生初始权重，用 teacher 软标签训练的学生
    /// 应该比只用硬标签的学生**更接近 teacher**（KL 更低、top-1 一致率更高），
    /// 且在真实标签上的 CE 不更差。
    ///
    /// 这就是"蒸馏有用"的最小可验证证据：数据少到只够学生学个大概时，
    /// 软标签提供的类间结构正好补上它自己看不到的那部分。
    ///
    /// 规模刻意压小（debug 模式下这仍是本套件里最慢的一条）：teacher 48 宽 2 层、
    /// student 24 宽 1 层、上下文 12。学生与教师的容量差、以及"只看一小批数据"
    /// 这两点才是蒸馏收益的来源，把它们保住即可，绝对值不重要。
    #[test]
    fn test_distillation_beats_hard_labels_on_tiny_data() {
        let text = "the cat sat on the mat. the dog sat on the log. \
                    the fox ran in the fog. the cat ran to the dog. "
            .repeat(8);
        let tokenizer = Tokenizer::char(&text);
        let vocab = tokenizer.vocab_size();
        let ids = tokenizer.encode(&text);
        let block = 12usize;

        let build = |n_embd: usize, n_layer: usize, seed: u64| {
            Transformer::new(
                TransformerConfig {
                    n_embd,
                    n_head: 4,
                    n_layer,
                    block_size: block,
                    dropout: 0.0,
                    ..TransformerConfig::tiny(vocab)
                },
                &mut Rng::new(seed),
            )
        };
        // teacher 更宽更深；student 减半再减一层
        let teacher = build(48, 2, 1);
        let mut rng = Rng::new(7);
        // 先把 teacher 训到能给出有意义的分布（这一步是"预训练好的大模型"的替身）
        let mut topt = AdamW::new(5e-3, teacher.parameters(), 0.01);
        for _ in 0..40 {
            let (tx, ty) = sample(&ids, block, 8, &mut rng);
            // forward(ids, b, t, ...)：`t` 是**每条序列**的长度，不是扁平化后的总数
            let logits = teacher.forward(&tx, 8, block, None, true);
            let loss = cross_entropy_loss_masked(&logits, &ty, None);
            loss.backward();
            topt.step();
            topt.zero_grad();
        }

        // 软标签只在训练数据上取一次（teacher 前向不进计算图）
        let soft_cfg = DistillConfig {
            temperature: 2.0,
            alpha: 0.0,
            top_k: 8,
        };
        // 固定一批训练数据，教师/学生看的是同一批，比较才公平
        let (x, y) = sample(&ids, block, 16, &mut Rng::new(99));
        let teacher_logits = no_grad(|| teacher.forward(&x, 16, block, None, false));

        // 评测集：训练时没见过的窗口
        let (ev_x, ev_y) = sample(&ids, block, 8, &mut Rng::new(1234));
        let ev_teacher = no_grad(|| teacher.forward(&ev_x, 8, block, None, false));
        let ev_soft = SoftTargets::from_logits(&ev_teacher, soft_cfg);

        let eval = |m: &Transformer| -> (f32, f64, f64) {
            let logits = no_grad(|| m.forward(&ev_x, 8, block, None, false));
            let ce = cross_entropy_loss_masked(&logits, &ev_y, None).item();
            (ce, ev_soft.kl_to(&logits), ev_soft.top1_agreement(&logits))
        };

        // 两个学生：同种子、同批数据、同步数；唯一差别是有没有 KD 项
        let run_student = |alpha: f32| -> (f32, f64, f64) {
            let s = build(24, 1, 5);
            let mut o = AdamW::new(5e-3, s.parameters(), 0.01);
            let cfg = DistillConfig {
                temperature: 2.0,
                alpha,
                top_k: 8,
            };
            let s_soft = SoftTargets::from_logits(&teacher_logits, cfg);
            for _ in 0..70 {
                let logits = s.forward(&x, 16, block, None, true);
                let loss = distill_loss(&logits, &s_soft, &y, None);
                loss.backward();
                o.step();
                o.zero_grad();
            }
            eval(&s)
        };

        let (ce_hard, kl_hard, agree_hard) = run_student(0.0);
        let (ce_kd, kl_kd, agree_kd) = run_student(0.6);

        println!(
            "  硬标签：CE={ce_hard:.4} KL(teacher)={kl_hard:.4} top-1 一致={agree_hard:.3}\n  \
             蒸馏　：CE={ce_kd:.4} KL(teacher)={kl_kd:.4} top-1 一致={agree_kd:.3}"
        );
        assert!(
            kl_kd < kl_hard,
            "蒸馏学生的分布应更接近 teacher：KL {kl_kd:.4} vs {kl_hard:.4}"
        );
        assert!(
            agree_kd >= agree_hard,
            "蒸馏学生的 top-1 一致率不应更差：{agree_kd:.3} vs {agree_hard:.3}"
        );
        assert!(
            ce_kd < ce_hard * 1.15,
            "蒸馏学生的硬标签 CE 不应明显变差：{ce_kd:.4} vs {ce_hard:.4}"
        );
    }

    /// 从 id 序列里随机切若干 `block` 长的窗口，拼成一个 batch：
    /// 返回扁平化的 `x`（`[B·T]`）与 `y`（右移一位）。
    fn sample(ids: &[usize], block: usize, batch: usize, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        assert!(ids.len() > block + 1, "语料太短，切不出一个窗口");
        let mut x = Vec::with_capacity(batch * block);
        let mut y = Vec::with_capacity(batch * block);
        for _ in 0..batch {
            let start = (rng.next_u64() as usize) % (ids.len() - block - 1);
            x.extend_from_slice(&ids[start..start + block]);
            y.extend_from_slice(&ids[start + 1..start + 1 + block]);
        }
        (x, y)
    }
}
