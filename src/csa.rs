//! CSA + HCA 混合注意力（第 49 课，DeepSeek-V4 系列的长上下文压缩注意力）
//!
//! 目的：把**注意力在序列维上的开销**从 `O(T)` 压到 `O(m + k + T/(m·h))`。
//! 做法是把每个 query 能看到的 key 集合拆成三层，拼进**同一次 softmax**：
//!
//! ```text
//!   Level 0 局部全分辨率：最近 m 个位置，一个不漏          —— 保证"眼前看得清"
//!   Level 1 细压缩（CSA）：已完成的 m 长块做块内均值压缩，
//!                          再按 q·Kc 打分做 top-k 稀疏选择 —— 保证"远处有重点"
//!   Level 2 粗压缩（HCA）：把 Level 1 的块再压 hca_block 倍，
//!                          稠密注意力，提供廉价全局覆盖     —— 保证"再远也不瞎"
//! ```
//!
//! # 覆盖不变式（本模块最重要的正确性约束）
//!
//! 对任意 query `i`，三层可见位置的**并集必须恰好是 `[0, i]`**，既不能少
//! （丢了历史），也不能多（漏进未来 token —— 训练时就是信息泄漏）。
//!
//! 关键点：一个压缩块只有在**完全落在 `i` 之前**（`块末位置 ≤ i`）时才可见。
//! 判据是 `j ≤ floor((i+1)/m) - 1`，**不是** `块起点 ≤ i` —— 后者会把块内
//! 尚未生成的 token 平均进来。推导：
//!
//! - 细压缩可见块数 `B_i = floor((i+1)/m)`，覆盖 `[0, B_i·m - 1]`；
//! - 局部覆盖 `[i-m+1, i]`，与细压缩相接（`B_i·m > i+1-m` ⇒ 重叠或相邻）；
//! - 粗压缩块的成员细块必然也已完成，因此只带来**冗余覆盖**，不会引入新位置。
//!
//! 见 [`covered_positions`]（供单测与 `csa` 子命令第一节核对）。
//!
//! # 接线方式
//!
//! [`CsaOpts::block`] `< 2` 时本模块完全不参与，[`crate::attention::MultiHeadAttention`]
//! 走原来的 `flash_attention` 路径（**逐位不变**）。开启后只在**无 KV cache 的全量前向**
//! （训练 / 预填）生效；带 cache 的自回归解码请用 MLA / KV 量化那套（见 `docs/49`）。

use crate::model::TransformerConfig;
use crate::tensor::Tensor;

/// CSA + HCA 的开关与超参（纯数据，`derive(Default)` 即为"全关"）。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CsaOpts {
    /// 压缩块大小 `m`（也是局部窗口宽度）。`< 2` = 关闭（走原 `flash_attention`）。
    pub block: usize,
    /// 细压缩保留的块数 `k`。`0` = 不稀疏，对全部已完成块做稠密注意力。
    pub top_k: usize,
    /// 粗压缩倍率 `h`（在 Level 1 之上再压 `h` 倍）。`0` / `< 2` = 不启用 HCA。
    pub hca_block: usize,
}

impl CsaOpts {
    /// 是否启用 CSA（`block >= 2`）
    pub fn enabled(&self) -> bool {
        self.block >= 2
    }
}

impl From<&TransformerConfig> for CsaOpts {
    fn from(c: &TransformerConfig) -> Self {
        CsaOpts {
            block: c.csa_block,
            top_k: c.csa_top_k,
            hca_block: c.hca_block,
        }
    }
}

/// query `i` 可见的**细压缩块数** `B_i = floor((i+1)/m)`（块末位置 ≤ i 才算完成）。
pub fn fine_blocks(i: usize, m: usize) -> usize {
    (i + 1) / m
}

/// query `i` 可见的**粗压缩块数** `floor((i+1)/(h·m))`（`h ≥ 2` 才有意义）。
pub fn coarse_blocks(i: usize, h: usize, m: usize) -> usize {
    (i + 1) / (h * m)
}

/// 三层覆盖的原始位置**并集**（升序）。不变量：结果必须恰好等于 `0..=i`。
///
/// 这是 `csa` 子命令第一节与单测共用的核对函数——把"因果性"这件事从
/// 几百行张量代码里抽出来，单独用整数算一遍。
pub fn covered_positions(i: usize, m: usize, h: usize) -> Vec<usize> {
    let mut seen = vec![false; i + 1];
    // Level 0：最近 m 个位置
    for p in (i + 1).saturating_sub(m)..=i {
        seen[p] = true;
    }
    // Level 1：已完成块全覆盖（B_i·m - 1 ≤ i 已由 B_i 的定义保证）
    for p in 0..fine_blocks(i, m) * m {
        seen[p] = true;
    }
    // Level 2：粗块同理，只会重复覆盖已有的位置
    if h >= 2 {
        for p in 0..coarse_blocks(i, h, m) * h * m {
            seen[p] = true;
        }
    }
    (0..=i).filter(|&p| seen[p]).collect()
}

/// 从 `scores[0..n_cand]` 里取分数最高的 `kk` 个下标（同分时**小下标优先**，
/// 保证可复现）。返回长度 = `min(kk, n_cand)`。
pub fn select_topk(scores: &[f32], n_cand: usize, kk: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..n_cand).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.cmp(&b))
    });
    idx.truncate(kk);
    idx
}

// ---------- 字节 / 算力账本（解析式，供 CLI 打印） ----------

/// 全量注意力的 KV 缓存字节（f32，K + V 两份）
pub fn full_cache_bytes(t: usize, n_kv: usize, hd: usize) -> usize {
    2 * t * n_kv * hd * 4
}

/// CSA 压缩缓存的 KV 字节（f32）：最近 `m` 个原始位置 + `floor(t/m)` 个压缩块。
///
/// 这是**存储口径**的压缩率来源，与 top-k 稀疏（算力口径）互相独立。
pub fn csa_cache_bytes(t: usize, n_kv: usize, hd: usize, m: usize) -> usize {
    2 * (m + t / m) * n_kv * hd * 4
}

/// 单个 query 的注意力 FLOPs（只算打分与加权求和两项，各 `2·slots·hd`）
pub fn attn_flops_per_query(slots: usize, hd: usize) -> usize {
    4 * slots * hd
}

// ==================== 张量辅助 ====================

/// 从 `[gx, nx, hd]` 里按 `(g_out, t_out)` 指定的索引取行 → `[g_out, t_out, n, hd]`。
///
/// 源张量与输出张量的序列长度可以不同（GQA 的 `gx=gk`、压缩后的 `nx=nb`），
/// 因此这里把源的 `(gx, nx)` 与输出的 `(g_out, t_out)` 分开传。
/// `idx` 长度必须是 `g_out·t_out·n`，元素是扁平后的**源**行号（即 `gx·nx` 展开下的下标）。
fn gather_seq(
    x: &Tensor,
    gx: usize,
    nx: usize,
    hd: usize,
    g_out: usize,
    t_out: usize,
    n: usize,
    idx: &[usize],
) -> Tensor {
    assert_eq!(idx.len(), g_out * t_out * n, "gather_seq 索引数量不匹配");
    x.reshape(vec![gx * nx, hd])
        .gather_rows(idx)
        .reshape(vec![g_out, t_out, n, hd])
}

/// 把 `[G, T, hd]` 沿序列维按块做均值压缩，只用**前 `nb·m` 行** → `[G, nb, hd]`。
fn block_mean(x: &Tensor, g: usize, t: usize, hd: usize, m: usize, nb: usize) -> Tensor {
    let mut idx = Vec::with_capacity(g * nb * m);
    for gg in 0..g {
        for j in 0..nb {
            for s in 0..m {
                idx.push(gg * t + j * m + s);
            }
        }
    }
    x.reshape(vec![g * t, hd])
        .gather_rows(&idx)
        .reshape(vec![g, nb, m, hd])
        .permute(&[0, 1, 3, 2])
        .sum_last_dim()
        .reshape(vec![g, nb, hd])
        .mul_scalar(1.0 / m as f32)
}

/// GQA 展开：把 `[Gk, N, hd]` 按"每个 KV 头服务 `n_rep` 个 Q 头"复制成 `[G, N, hd]`。
fn expand_gqa(x: &Tensor, gk: usize, n: usize, hd: usize, g: usize, n_rep: usize) -> Tensor {
    let mut idx = Vec::with_capacity(g * n);
    for gg in 0..g {
        let group = gg / n_rep;
        for j in 0..n {
            idx.push(group * n + j);
        }
    }
    x.reshape(vec![gk * n, hd])
        .gather_rows(&idx)
        .reshape(vec![g, n, hd])
}

/// 沿最后一维拼接若干"前缀相同、末维不同"的张量 → 前缀 + 末维之和。
///
/// 手写一个小的融合算子（复用 [`Tensor::external`] 的自定义反向），
/// 避免为了这一次拼接去给张量库加通用 `concat`。
pub(crate) fn concat_last(parts: &[Tensor]) -> Tensor {
    assert!(!parts.is_empty(), "concat_last 至少要一个张量");
    let cont: Vec<Tensor> = parts.iter().map(|t| t.contiguous()).collect();
    let rank = cont[0].rank();
    assert!(rank >= 1, "concat_last 至少 1 维");
    let prefix: Vec<usize> = cont[0].shape()[..rank - 1].to_vec();
    let rows: usize = prefix.iter().product();
    let widths: Vec<usize> = cont
        .iter()
        .map(|t| {
            assert_eq!(t.rank(), rank);
            assert_eq!(&t.shape()[..rank - 1], prefix.as_slice(), "concat_last 前缀不一致");
            t.shape()[rank - 1]
        })
        .collect();
    let total_w: usize = widths.iter().sum();

    let mut data = vec![0.0f32; rows * total_w];
    {
        let mut col = 0;
        for (t, &w) in cont.iter().zip(&widths) {
            let d = t.decode();
            for r in 0..rows {
                data[r * total_w + col..r * total_w + col + w]
                    .copy_from_slice(&d[r * w..(r + 1) * w]);
            }
            col += w;
        }
    }
    let mut shape = prefix;
    shape.push(total_w);

    if !cont.iter().any(|t| t.req()) {
        return Tensor::from_vec(data, shape);
    }
    let grads: Vec<_> = cont.iter().map(|t| t.grad.clone()).collect();
    let widths_c = widths.clone();
    Tensor::external(data, shape, cont.clone(), move |out_grad| {
        Box::new(move || {
            let gy = out_grad.borrow();
            let gy_ref: &[f32] = &gy;
            let mut col = 0usize;
            for (i, &w) in widths_c.iter().enumerate() {
                let mut sg = grads[i].borrow_mut();
                for r in 0..rows {
                    for j in 0..w {
                        sg[r * w + j] += gy_ref[r * total_w + col + j];
                    }
                }
                col += w;
            }
        })
    })
}

// ==================== 前向 ====================

/// CSA + HCA 混合注意力前向。
///
/// - `q`: `[B·n_head, t, hd]`
/// - `k` / `v`: `[B·n_kv_head, t, hd]`（GQA 允许 `n_kv_head < n_head`；长度必须等于 `t`）
///
/// 返回 `out: [B·n_head, t, hd]`，与 [`Tensor::flash_attention`] 同形状同语义，
/// 二者可直接互换（仅在屏幕内的可见 key 集合上不同）。
pub fn attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    n_head: usize,
    n_kv_head: usize,
    o: &CsaOpts,
) -> Tensor {
    let q = q.contiguous();
    let k = k.contiguous();
    let v = v.contiguous();
    let (g, t, hd) = (q.shape()[0], q.shape()[1], q.shape()[2]);
    let m = o.block;
    assert!(m >= 2, "CSA 只在 block >= 2 时启用");
    assert!(n_head > 0 && n_kv_head > 0 && n_head % n_kv_head == 0);
    let n_rep = n_head / n_kv_head;
    let b = g / n_head;
    let gk = b * n_kv_head;
    assert_eq!(q.shape(), &[g, t, hd]);
    assert_eq!(
        k.shape(),
        &[gk, t, hd],
        "CSA 目前只支持无 KV cache 的全量前向（K 长度须等于 Q 长度 = {t}）"
    );
    assert_eq!(v.shape(), k.shape(), "K/V 形状必须一致");
    // 把 1/√d 挪到 Q 上（与 flash_attention 同口径）
    let qs = q.mul_scalar(1.0 / (hd as f32).sqrt());

    // 三层槽位布局：[局部窗口 m] ++ [细压缩 kk] ++ [粗压缩 nb2]。
    // 掩码与分数必须**逐行同序**：先算好总槽数，再把各层掩码写到该行的对应偏移处，
    // 否则按行 reshape 时掩码会与分数错位（曾导致整行被 -inf 淹没 → softmax 输出 NaN）。
    let nb = t / m; // 已完成的块总数
    let kk = if o.top_k == 0 { nb } else { o.top_k.min(nb) };
    let h = o.hca_block;
    let nb2 = if h >= 2 { nb / h } else { 0 };
    let total_s = m + kk + nb2;

    let mut s_list: Vec<Tensor> = Vec::with_capacity(3);
    let mut v_list: Vec<Tensor> = Vec::with_capacity(3);
    let mut mask_rows = vec![0.0f32; g * t * total_s];

    // ---------- Level 0：局部全分辨率窗口 ----------
    // 位置 p(i, s) = i + 1 - m + s（s = 0..m-1）；p < 0 的槽位用 -inf 屏蔽。
    let mut loc_idx = vec![0usize; g * t * m];
    for gg in 0..g {
        let group = gg / n_rep;
        for i in 0..t {
            let row = (gg * t + i) * total_s;
            for s in 0..m {
                let fl = (gg * t + i) * m + s;
                let p = i as isize + 1 - m as isize + s as isize;
                if p < 0 {
                    loc_idx[fl] = group * t; // 越界槽位钳到 0，靠掩码去掉
                    mask_rows[row + s] = f32::NEG_INFINITY;
                } else {
                    loc_idx[fl] = group * t + p as usize;
                }
            }
        }
    }
    let k_loc = gather_seq(&k, gk, t, hd, g, t, m, &loc_idx); // [g,t,m,hd]
    let v_loc = gather_seq(&v, gk, t, hd, g, t, m, &loc_idx);
    s_list.push(
        qs.reshape(vec![g, t, 1, hd])
            .mul(&k_loc)
            .sum_last_dim()
            .reshape(vec![g, t, m]),
    );
    v_list.push(v_loc);

    // ---------- Level 1：细压缩（CSA） ----------
    if kk > 0 {
        let kc = block_mean(&k, gk, t, hd, m, nb); // [gk,nb,hd]
        let vc = block_mean(&v, gk, t, hd, m, nb);
        let kc_e = expand_gqa(&kc, gk, nb, hd, g, n_rep); // [g,nb,hd]
        let vc_e = expand_gqa(&vc, gk, nb, hd, g, n_rep);

        // 稠密打分：只用于"选哪些块"，其数值不参与最终输出/反向。
        // 用 no_grad 包住，避免在 tape 上留下永远不可达的节点（否则每步都堆积、内存只涨不降）。
        let sd = crate::tensor::no_grad(|| {
            qs.reshape(vec![g, t, 1, hd])
                .mul(&kc_e.reshape(vec![g, 1, nb, hd]))
                .sum_last_dim()
                .reshape(vec![g, t, nb])
                .data()
        });

        // 逐 (g,i) 选块：只在已完成块 [0, B_i) 里选 top-kk
        let mut sel = vec![0usize; g * t * kk];
        for gg in 0..g {
            for i in 0..t {
                let bi = fine_blocks(i, m).min(nb);
                let srow = (gg * t + i) * nb;
                let mrow = (gg * t + i) * total_s + m; // 细压缩段起点
                let fl = (gg * t + i) * kk;
                if o.top_k == 0 {
                    // 稠密：槽位 r 直接对应块 r，r ≥ B_i 的槽位屏蔽
                    for r in 0..kk {
                        if r < bi {
                            sel[fl + r] = gg * nb + r;
                        } else {
                            sel[fl + r] = gg * nb;
                            mask_rows[mrow + r] = f32::NEG_INFINITY;
                        }
                    }
                } else {
                    let chosen = select_topk(&sd[srow..srow + nb], bi, kk);
                    for r in 0..kk {
                        if r < chosen.len() {
                            sel[fl + r] = gg * nb + chosen[r];
                        } else {
                            sel[fl + r] = gg * nb;
                            mask_rows[mrow + r] = f32::NEG_INFINITY;
                        }
                    }
                }
            }
        }
        let kc_sel = gather_seq(&kc_e, g, nb, hd, g, t, kk, &sel); // [g,t,kk,hd]
        let vc_sel = gather_seq(&vc_e, g, nb, hd, g, t, kk, &sel);
        s_list.push(
            qs.reshape(vec![g, t, 1, hd])
                .mul(&kc_sel)
                .sum_last_dim()
                .reshape(vec![g, t, kk]),
        );
        v_list.push(vc_sel);

        // ---------- Level 2：粗压缩（HCA），在 Level 1 之上再压 h 倍 ----------
        if nb2 > 0 {
            let kc2 = block_mean(&kc, gk, nb, hd, h, nb2); // [gk,nb2,hd]
            let vc2 = block_mean(&vc, gk, nb, hd, h, nb2);
            let kc2_e = expand_gqa(&kc2, gk, nb2, hd, g, n_rep);
            let vc2_e = expand_gqa(&vc2, gk, nb2, hd, g, n_rep);
            for gg in 0..g {
                for i in 0..t {
                    let bc = coarse_blocks(i, h, m).min(nb2);
                    let mrow = (gg * t + i) * total_s + m + kk; // 粗压缩段起点
                    for r in bc..nb2 {
                        mask_rows[mrow + r] = f32::NEG_INFINITY;
                    }
                }
            }
            s_list.push(
                qs.reshape(vec![g, t, 1, hd])
                    .mul(&kc2_e.reshape(vec![g, 1, nb2, hd]))
                    .sum_last_dim()
                    .reshape(vec![g, t, nb2]),
            );
            // 值广播到每个 query：Vc2_e [g,1,nb2,hd] × ones [1,t,1,1] → [g,t,nb2,hd]
            let ones = Tensor::from_vec(vec![1.0f32; t], vec![1, t, 1, 1]);
            v_list.push(vc2_e.reshape(vec![g, 1, nb2, hd]).mul(&ones));
        }
    }

    // ---------- 三层合并进一次 softmax ----------
    let s_all = concat_last(&s_list); // [g,t,S]
    let mask_t = Tensor::from_vec(mask_rows, vec![g, t, total_s]);
    let p = s_all.masked_softmax(&mask_t); // [g,t,S]

    // 值列表拼到 [g,t,S,hd]：先把槽位维换到最后再 concat，绕开通用 concat
    let v_perm: Vec<Tensor> = v_list
        .iter()
        .map(|x| {
            let n = x.shape()[2];
            x.permute(&[0, 1, 3, 2]).reshape(vec![g, t * hd, n])
        })
        .collect();
    let v_all = concat_last(&v_perm)
        .reshape(vec![g, t, hd, total_s])
        .permute(&[0, 1, 3, 2]); // [g,t,S,hd]

    p.reshape(vec![g, t, total_s, 1])
        .mul(&v_all)
        .permute(&[0, 1, 3, 2])
        .sum_last_dim()
        .reshape(vec![g, t, hd])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn opts(block: usize, top_k: usize, hca: usize) -> CsaOpts {
        CsaOpts { block, top_k, hca_block: hca }
    }

    // ---- 一、覆盖不变式 / 因果性 ----

    /// 三层并集必须恰好等于 [0, i]：一个不多、一个不少。
    #[test]
    fn test_coverage_invariant_exact_range() {
        for m in [2usize, 3, 4, 5, 8] {
            for h in [0usize, 2, 3] {
                for i in 0..40 {
                    let cov = covered_positions(i, m, h);
                    let want: Vec<usize> = (0..=i).collect();
                    assert_eq!(cov, want, "m={m} h={h} i={i} 覆盖不等于 [0,i]");
                }
            }
        }
    }

    /// 因果性：任何可见位置都不得超过 i（不泄漏未来）。
    #[test]
    fn test_no_future_leak_in_coverage() {
        for m in [2usize, 4, 8] {
            for h in [0usize, 2] {
                for i in 0..30 {
                    let cov = covered_positions(i, m, h);
                    assert!(cov.iter().all(|&p| p <= i), "m={m} h={h} i={i} 出现未来位置");
                }
            }
        }
    }

    /// 细压缩只在"块末位置 ≤ i"时可见（禁止用块起点当判据）。
    #[test]
    fn test_fine_block_causal_boundary() {
        let m = 4;
        // i=3：块 [0,4) 的末位 3 ≤ 3，故已完成；i=2 时不算
        assert_eq!(fine_blocks(3, m), 1);
        assert_eq!(fine_blocks(2, m), 0);
        // 每个已完成块的末位都不超过 i
        for i in 0..50 {
            let b = fine_blocks(i, m);
            if b > 0 {
                assert!(b * m - 1 <= i, "i={i} 的最后一个细块越界");
            }
        }
    }

    /// 粗压缩块数同样满足因果边界。
    #[test]
    fn test_coarse_block_causal_boundary() {
        let (m, h) = (4usize, 2);
        for i in 0..60 {
            let bc = coarse_blocks(i, h, m);
            if bc > 0 {
                assert!(bc * h * m - 1 <= i, "i={i} 的最后一个粗块越界");
            }
        }
    }

    // ---- 二、块均值压缩 ----

    /// 块均值：手工算一遍对照，并覆盖"尾部不足一块被丢弃"的情形。
    #[test]
    fn test_block_mean_matches_manual_and_drops_tail() {
        // T=6, m=2 → 3 块；T=7 时最后 1 个位置不足一块应被忽略
        let data: Vec<f32> = (0..7 * 2).map(|x| x as f32).collect();
        let x = Tensor::from_vec(data, vec![1, 7, 2]);
        let y = block_mean(&x, 1, 7, 2, 2, 3); // 只用前 6 行
        assert_eq!(y.shape(), &[1, 3, 2]);
        let got = y.data();
        // 块 0：行 0,1 → [1, 2]；块 1：行 2,3 → [5, 6]；块 2：行 4,5 → [9, 10]
        assert!((got[0] - 1.0).abs() < 1e-6 && (got[1] - 2.0).abs() < 1e-6);
        assert!((got[2] - 5.0).abs() < 1e-6 && (got[3] - 6.0).abs() < 1e-6);
        assert!((got[4] - 9.0).abs() < 1e-6 && (got[5] - 10.0).abs() < 1e-6);
        // 第 7 行（值 [12, 13]）确实没被算进任何块
        assert!(got.iter().all(|&v| v < 12.0));
    }

    // ---- 三、top-k 选择 ----

    /// top-k 取分数最高的若干个，同分时小下标优先。
    #[test]
    fn test_select_topk_picks_highest_and_ties_by_index() {
        let s = [0.1f32, 0.9, 0.9, 0.2, 0.7];
        assert_eq!(select_topk(&s, 5, 2), vec![1, 2]); // 两个 0.9 取小下标 1、2
        assert_eq!(select_topk(&s, 5, 3), vec![1, 2, 4]);
        // 候选数少于 k 时只返回候选
        assert_eq!(select_topk(&s, 2, 5), vec![1, 0]);
    }

    // ---- 四、前向数值性质 ----

    fn rand_tensor(shape: Vec<usize>, rng: &mut Rng) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|_| rng.next_f32() * 2.0 - 1.0).collect();
        Tensor::from_vec(data, shape)
    }

    /// 输出形状与 flash_attention 一致；且开启 CSA 后数值确实改变（不是空开关）。
    #[test]
    fn test_attention_shape_and_differs_from_flash() {
        let (b, h, t, hd) = (1usize, 2usize, 12usize, 4usize);
        let mut rng = Rng::new(7);
        let q = rand_tensor(vec![b * h, t, hd], &mut rng);
        let k = rand_tensor(vec![b * h, t, hd], &mut rng);
        let v = rand_tensor(vec![b * h, t, hd], &mut rng);
        let out = attention(&q, &k, &v, h, h, &opts(4, 0, 0));
        assert_eq!(out.shape(), &[b * h, t, hd]);
        let flash = Tensor::flash_attention(&q, &k, &v, None, 32);
        assert_ne!(out.data(), flash.data(), "CSA 应与全量 flash 不同");
        assert!(out.data().iter().all(|x| x.is_finite()));
    }

    /// 无未来泄漏（前向口径）：改动位置 i 之后的 token，不应改变 query i 的输出。
    #[test]
    fn test_forward_has_no_future_leak() {
        let (b, h, t, hd) = (1usize, 2usize, 16usize, 4usize);
        let mut rng = Rng::new(11);
        let q = rand_tensor(vec![b * h, t, hd], &mut rng);
        let k = rand_tensor(vec![b * h, t, hd], &mut rng);
        let v = rand_tensor(vec![b * h, t, hd], &mut rng);
        let o = opts(4, 2, 0);
        let out1 = attention(&q, &k, &v, h, h, &o).data();

        // 把位置 12..16 的 K/V 改掉（都在 query i=11 之后）
        let mut k2 = k.data();
        let mut v2 = v.data();
        for row in 0..b * h {
            for p in 12..t {
                for d in 0..hd {
                    k2[(row * t + p) * hd + d] += 5.0;
                    v2[(row * t + p) * hd + d] -= 3.0;
                }
            }
        }
        let k2 = Tensor::from_vec(k2, vec![b * h, t, hd]);
        let v2 = Tensor::from_vec(v2, vec![b * h, t, hd]);
        let out2 = attention(&q, &k2, &v2, h, h, &o).data();

        for row in 0..b * h {
            for p in 0..=11 {
                for d in 0..hd {
                    let a = out1[(row * t + p) * hd + d];
                    let c = out2[(row * t + p) * hd + d];
                    assert!(
                        (a - c).abs() < 1e-5,
                        "query {p} 受未来 token 影响：{a} vs {c}（row {row} d {d}）"
                    );
                }
            }
        }
    }

    /// GQA：n_kv_head < n_head 时，同一 KV 组的多个 Q 头输出应各不相同（各自打分），
    /// 且整个前向保持有限。
    #[test]
    fn test_gqa_heads_share_kv_but_score_independently() {
        let (b, h, hkv, t, hd) = (1usize, 4usize, 2usize, 12usize, 4usize);
        let mut rng = Rng::new(13);
        let q = rand_tensor(vec![b * h, t, hd], &mut rng);
        let k = rand_tensor(vec![b * hkv, t, hd], &mut rng);
        let v = rand_tensor(vec![b * hkv, t, hd], &mut rng);
        let out = attention(&q, &k, &v, h, hkv, &opts(4, 1, 2));
        assert_eq!(out.shape(), &[b * h, t, hd]);
        let d = out.data();
        // 同一 KV 组的两个 Q 头（组 0：头 0、1）输出不应逐位相同
        let head = |hh: usize, p: usize| -> Vec<f32> {
            (0..hd).map(|x| d[(hh * t + p) * hd + x]).collect()
        };
        assert_ne!(head(0, 11), head(1, 11), "同组 Q 头应各自打分");
        assert!(d.iter().all(|x| x.is_finite()));
    }

    /// top_k = 0（稠密）与 top_k ≥ 块数应给出相同结果：都看全部已完成块。
    #[test]
    fn test_dense_topk_equals_full_block_attention() {
        let (b, h, t, hd) = (1usize, 2usize, 16usize, 4usize);
        let mut rng = Rng::new(17);
        let q = rand_tensor(vec![b * h, t, hd], &mut rng);
        let k = rand_tensor(vec![b * h, t, hd], &mut rng);
        let v = rand_tensor(vec![b * h, t, hd], &mut rng);
        let a = attention(&q, &k, &v, h, h, &opts(4, 0, 0)).data();
        let c = attention(&q, &k, &v, h, h, &opts(4, 99, 0)).data();
        for (x, y) in a.iter().zip(&c) {
            assert!((x - y).abs() < 1e-5, "top_k=0 与 top_k=∞ 应等价");
        }
    }

    /// 梯度自检：有限差分对照 dq / dk / dv。
    /// 用稠密（top_k = 0）配置：没有"选块"这一步，输出对输入处处可微，
    /// 有限差分才能与解析梯度逐点对上。
    #[test]
    fn test_gradient_matches_finite_difference() {
        let (b, h, t, hd) = (1usize, 2usize, 8usize, 3usize);
        let mut rng = Rng::new(23);
        let q = Tensor::param(
            (0..b * h * t * hd).map(|_| rng.next_f32() * 2.0 - 1.0).collect(),
            vec![b * h, t, hd],
        );
        let k = Tensor::param(
            (0..b * h * t * hd).map(|_| rng.next_f32() * 2.0 - 1.0).collect(),
            vec![b * h, t, hd],
        );
        let v = Tensor::param(
            (0..b * h * t * hd).map(|_| rng.next_f32() * 2.0 - 1.0).collect(),
            vec![b * h, t, hd],
        );
        let o = opts(4, 0, 2);

        // 用固定权重把输出聚合成标量，再反向
        let w: Vec<f32> = (0..b * h * t * hd).map(|i| 0.5 + (i % 7) as f32 * 0.1).collect();
        let w = Tensor::from_vec(w.clone(), vec![b * h, t, hd]);
        let loss = attention(&q, &k, &v, h, h, &o).mul(&w).sum();
        q.zero_grad();
        k.zero_grad();
        v.zero_grad();
        loss.backward();
        let gq = q.grad();
        let gk = k.grad();
        let gv = v.grad();

        // 数值梯度（中心差分）
        let eps = 1e-2f32;
        let scalar = |qq: &Tensor, kk: &Tensor, vv: &Tensor| -> f32 {
            attention(qq, kk, vv, h, h, &o).mul(&w).sum().item()
        };
        let check = |base: &Tensor, g: &[f32], name: &str| {
            let d0 = base.data();
            for idx in [0usize, 5, 9, 17] {
                let mut a = d0.clone();
                a[idx] += eps;
                let mut bb = d0.clone();
                bb[idx] -= eps;
                let ta = Tensor::from_vec(a, base.shape().to_vec());
                let tb = Tensor::from_vec(bb, base.shape().to_vec());
                let (qa, ka, va) = match name {
                    "q" => (&ta, &k, &v),
                    "k" => (&q, &ta, &v),
                    _ => (&q, &k, &ta),
                };
                let (qb, kb, vb) = match name {
                    "q" => (&tb, &k, &v),
                    "k" => (&q, &tb, &v),
                    _ => (&q, &k, &tb),
                };
                let num = (scalar(qa, ka, va) - scalar(qb, kb, vb)) / (2.0 * eps);
                assert!(
                    (num - g[idx]).abs() < 2e-2,
                    "{name}[{idx}] 数值梯度 {num} vs 解析 {}",
                    g[idx]
                );
            }
        };
        check(&q, &gq, "q");
        check(&k, &gk, "k");
        check(&v, &gv, "v");
    }

    /// GQA 下 K/V 的梯度要折回 KV 头（多个 Q 头共享同一 KV 头，梯度应累加）。
    #[test]
    fn test_gqa_kv_gradient_accumulates_across_heads() {
        let (b, h, hkv, t, hd) = (1usize, 4usize, 2usize, 8usize, 3usize);
        let mut rng = Rng::new(29);
        let q = Tensor::param(
            (0..b * h * t * hd).map(|_| rng.next_f32()).collect(),
            vec![b * h, t, hd],
        );
        let k = Tensor::param(
            (0..b * hkv * t * hd).map(|_| rng.next_f32()).collect(),
            vec![b * hkv, t, hd],
        );
        let v = Tensor::param(
            (0..b * hkv * t * hd).map(|_| rng.next_f32()).collect(),
            vec![b * hkv, t, hd],
        );
        let o = opts(4, 2, 2);
        let loss = attention(&q, &k, &v, h, hkv, &o).sum();
        q.zero_grad();
        k.zero_grad();
        v.zero_grad();
        loss.backward();
        // 每个 KV 头都应拿到非零梯度（两个 Q 头共同贡献）
        let gk = k.grad();
        for kh in 0..hkv {
            let mut any = false;
            for i in kh * t * hd..(kh + 1) * t * hd {
                if gk[i].abs() > 1e-9 {
                    any = true;
                }
            }
            assert!(any, "KV 头 {kh} 没拿到梯度");
        }
    }

    /// 关闭状态（block < 2）由调用方负责走 flash；这里核对账本函数在退化输入下也自洽。
    #[test]
    fn test_ledger_is_monotone_and_sane() {
        let (n_kv, hd) = (2usize, 8usize);
        for t in [16usize, 64, 256] {
            let full = full_cache_bytes(t, n_kv, hd);
            for m in [2usize, 4, 8, 16] {
                // 只有窗口远小于序列（m ≤ t/4）时压缩才划算：账本是 m + t/m。
                // m 接近 t 时局部窗口本身就等于全长，压缩无从谈起。
                if m * 4 > t {
                    continue;
                }
                let c = csa_cache_bytes(t, n_kv, hd, m);
                assert!(c < full, "t={t} m={m} 压缩后应更小");
            }
        }
        // 退化情形（m = t）明确更大，说明"压缩"仅在 m ≪ t 时成立。
        assert!(csa_cache_bytes(16, n_kv, hd, 16) > full_cache_bytes(16, n_kv, hd));
        // 压缩率随 m 增大先降后升（m + t/m 的极值在 √t 附近）
        assert!(csa_cache_bytes(64, n_kv, hd, 8) < csa_cache_bytes(64, n_kv, hd, 2));
    }
}
