//! 混合专家（MoE，第 32 课）
//!
//! 稠密模型里每个 token 都要过全部的 FFN 参数；MoE 的思路是**稀疏激活**——
//! 放很多个专家 FFN，每个 token 只走其中 Top-K 个，按门控权重加权求和：
//!
//! ```text
//! y = Σ_{i ∈ TopK(g(x))} w_i · E_i(x),    w = softmax(TopK(g(x)))
//!   g(x) = x @ W_g + b_g                  路由器（门控线性层），[d, n_expert]
//! ```
//!
//! 本文件实现四件事（对应文档 §实现要点）：
//!
//! 1. **路由**：[`top_k_gate`] 按 logits 选出每个 token 的 Top-K 专家。选择本身是
//!    数据上的决定（argmax），不参与求导。
//! 2. **门控权重**：把未选中位置的门控 logits 置 −∞ 再走普通 softmax，于是它们的概率
//!    恰好是 0，而 softmax 对它们的梯度也恰好是 0（`∂w_j/∂logit_i = w_j(δ_ij − w_i)`，
//!    `w_j = 0` 时该式恒为 0）。所以"只在 Top-K 上取权重"这件事**可导**，
//!    路由器的梯度可以正常回传，不需要为它单写一段反向。
//!
//!    这里有一个**必须知道的坑**：`w` 的求和口径有两种做法，而它们在 K = 1 时行为完全不同
//!    （由 [`MoELayer::switch_gate`] 切换）：
//!
//!    - **重归一化**（默认，Mixtral / DeepSeek / Qwen 式）：在 Top-K 内部再归一化，
//!      `Σw = 1`，输出幅度与稠密 FFN 可比。但 K = 1 时 softmax 只有一个元素，
//!      **权重恒等于 1**，对 logits 的雅可比整个是 0——主损失**给不了路由器任何梯度**，
//!      路由器只能靠辅助损失训练。所以 K = 1 请配 switch_gate。
//!    - **原概率**（`switch_gate = true`，Switch Transformer 式）：直接用全部专家上的
//!      softmax 概率，选中的位置取原值、其余置 0，`Σw < 1`。K = 1 时
//!      `w = p_0`（非 0 非 1），路由器拿得到主损失的梯度。
//!
//! 3. **稀疏前向**：逐专家 `gather_rows` 取出分到自己的 token → 跑该专家的 FFN →
//!    加权 → `scatter_add_rows` 散射回原位。专家没分到 token 就整段跳过，
//!    计算量真的按 K/E 缩下来（不是"全跑一遍再用掩码筛掉"）。
//! 4. **负载均衡辅助损失**（Switch Transformer 式）：
//!
//!    ```text
//!    L_aux = α · E · Σ_i f_i · p_i
//!      f_i = 路由到专家 i 的 (token, 专家) 对数 / (n·K)      ← 来自 argmax，常数
//!      p_i = 全部 token 上 softmax(logits) 对专家 i 的平均概率  ← 可导
//!    ```
//!
//!    它的几个性质（本项目实测过，见 `moe` 子命令第二节）：
//!
//!    - **p 均匀 ⇒ `L_aux = 1`，与 f 无关**：`Σ f_i·p_i = (1/E)·Σ f_i = 1/E`。也就是说
//!      "L_aux 掉到 1"**不是**负载均衡的证书——硬路由（argmax）可以照样压在少数专家上。
//!    - `p = f`（路由器完全自信）时 `L_aux = α·E·Σ f_i² ≥ α`，等号只在 f 均匀时成立
//!      （柯西–施瓦茨 `Σ f_i² ≥ 1/E`）。
//!    - 梯度 `∂L/∂logit_j ∝ p_j·(f_j − Σ f_i p_i)` 的方向确实指向"把被过度使用的专家按下去"，
//!      但**大小正比于 `p_j`**：p 越平，梯度越小。于是最快的下降路径是先把 p 摊平，
//!      而不是把硬路由摊平。
//!
//!    一句话：它压的是**概率分布**，不是**分配结果**；f 由 argmax 给出、不可导，只能被
//!    间接影响。后续工作（DeepSeek-V3 的 loss-free 均衡偏置、expert-choice 路由）正是
//!    为了绕开这个软/硬错配。
//!
//!    与它互补的是 **router z-loss**（PaLM / Mixtral 都加了，`model.moe_z_loss_coef`）：
//!
//!    ```text
//!    L_z = (1/n) · Σ_i log²(Σ_j e^{logits_ij})      = mean(logsumexp(logits)²)
//!    ```
//!
//!    α 压平的是路由**概率**，β 按住的是门控 logits 的**幅度**：logits 越推越大 ⇒
//!    softmax 过尖 ⇒ 路由器饱和、梯度消失。尤其 K = 1 + 重归一化口径下主损失给不了
//!    路由器梯度（§2），z-loss 是那时少数还能直接训路由器的信号之一。
//!
//! 另外实现了**容量因子**（每个专家最多处理多少 token，Switch Transformer 的
//! Token Dropping）：超出的分配直接丢弃。这是真实 MoE 训练里 GPU 显存与
//! All-to-All 通信量的硬上限，代价是某些 token 少走了一个专家。

use crate::layers::{Linear, MLPEnum, swiglu_hidden};
use crate::model::TransformerConfig;
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::{Shared, Tensor};

// ==================== 路由 ====================

/// 一次 Top-K 路由的完整结果（纯数据，不含梯度）。
#[derive(Clone, Debug)]
pub struct RoutePlan {
    /// 每个专家**实际处理**的 token 行号（升序；已按容量截断）。
    /// 长度可能小于"路由到该专家的 token 数"——差额就是被丢弃的。
    pub sel: Vec<Vec<usize>>,
    /// [n, E] 的门控掩码加项：选中的位置 0、其余 −∞（把 softmax 限制在 Top-K 上）
    pub penalty: Vec<f32>,
    /// [n, E] 的选中掩码：选中 1、其余 0。[`RoutePlan::penalty`] 的 0/1 对偶，
    /// 供 `switch_gate` 那条口径"取原概率、其余置 0"用（见 [`MoELayer::switch_gate`]）。
    pub mask: Vec<f32>,
    /// 每个专家被路由到的分配比例（Σ = 1），即辅助损失里的 `f_i`
    pub f: Vec<f32>,
    /// 因超出专家容量被丢弃的分配数
    pub dropped: usize,
    /// 每个专家被路由到的分配数（含被容量丢弃的）。与 [`RoutePlan::sel`] 的长度一比，
    /// 就能看出容量截掉了多少——诊断字段，单测已覆盖（`test_capacity_factor_drops_overflow`）。
    #[allow(dead_code)]
    pub routed: Vec<usize>,
}

/// 每个专家的容量上限：`capacity_factor` × 平均负载（`n·K/E`）。
///
/// `capacity_factor <= 0` 表示不限容量（返回 [`usize::MAX`]）。真实系统一般取
/// 1.25（Switch Transformer）～2.0：容量是显存与通信量的上限，代价是被丢弃的
/// token 少走一个专家。**推理时建议不限**——否则同一个 token 的输出会取决于
/// 同批次里别的 token 挤没挤占容量。
pub fn expert_capacity(
    n_tokens: usize,
    n_expert: usize,
    top_k: usize,
    capacity_factor: f32,
) -> usize {
    if capacity_factor <= 0.0 {
        usize::MAX
    } else {
        let avg = (n_tokens * top_k) as f32 / n_expert as f32;
        ((capacity_factor * avg).ceil() as usize).max(1)
    }
}

/// Top-K 路由：给定 `[n, E]` 的门控 logits（展平），算出每个 token 的专家分配。
///
/// 并列时按下标升序（`total_cmp` + 下标比较）——路由必须是**确定性**的，
/// 否则同一份权重、同一批数据会给出不同的 loss，实验无法复现。
///
/// 容量按 token 顺序先到先得（Switch Transformer 的做法）：第 i 个 token 想要专家 e，
/// 但 e 已经满了，这条分配就被丢弃——**不会**退而求其次塞进它没选的专家里。
pub fn top_k_gate(
    logits: &[f32],
    n_tokens: usize,
    n_expert: usize,
    top_k: usize,
    capacity: usize,
) -> RoutePlan {
    assert_eq!(logits.len(), n_tokens * n_expert, "logits 长度应为 n × E");
    assert!(top_k >= 1 && top_k <= n_expert, "Top-K 必须落在 1..=E");
    let k = top_k;
    let mut sel: Vec<Vec<usize>> = vec![Vec::new(); n_expert];
    let mut penalty = vec![f32::NEG_INFINITY; n_tokens * n_expert];
    let mut mask = vec![0.0f32; n_tokens * n_expert];
    let mut routed = vec![0usize; n_expert];
    let mut dropped = 0usize;
    // 每行选择用的下标缓冲，循环外分配一次复用（select_nth 会把它打乱，
    // 但它始终是 0..n_expert 的一个排列，下一轮照样能直接用）
    let mut order: Vec<usize> = (0..n_expert).collect();
    for i in 0..n_tokens {
        let row = &logits[i * n_expert..(i + 1) * n_expert];
        // 只挑前 k 大：select_nth_unstable_by 期望 O(E) 切出前 k 名（无序），
        // 再对这 k 个排 O(k log k)——整行 O(E log E) 全排序是浪费（E=256、k=2 时差两个数量级）。
        // 比较器与全排序完全一致（值降序、并列下标升序），路由行为逐位不变
        //（对拍测试：`test_top_k_gate_matches_full_sort`）。
        order.select_nth_unstable_by(k - 1, |&a, &b| {
            row[b].total_cmp(&row[a]).then(a.cmp(&b))
        });
        order[..k].sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
        for &e in &order[..k] {
            penalty[i * n_expert + e] = 0.0;
            mask[i * n_expert + e] = 1.0;
            routed[e] += 1;
            if sel[e].len() < capacity {
                sel[e].push(i);
            } else {
                dropped += 1;
            }
        }
    }
    let denom = (n_tokens * k) as f32;
    RoutePlan {
        sel,
        penalty,
        mask,
        f: routed.iter().map(|&c| c as f32 / denom).collect(),
        dropped,
        routed,
    }
}

// ==================== 统计 ====================

/// 一次前向的路由统计（训练日志与诊断读它）
#[derive(Clone, Debug, Default)]
pub struct RouteStats {
    /// 每个专家**实际**处理的 token 数（已扣除容量丢弃）
    pub counts: Vec<usize>,
    /// 因容量被丢弃的分配数
    pub dropped: usize,
    pub n_tokens: usize,
    /// 本次前向的均衡损失值（未乘系数 α；见文件头 §4：`p = f` 时最小值才是 1）
    pub aux: f32,
}

impl RouteStats {
    /// 负载不均衡度 = 最大负载 / 平均负载（1.0 = 完美均衡，E = 全部压在一个专家上）
    pub fn imbalance(&self) -> f64 {
        let total: usize = self.counts.iter().sum();
        if total == 0 || self.counts.is_empty() {
            return 0.0;
        }
        let avg = total as f64 / self.counts.len() as f64;
        self.counts.iter().map(|&c| c as f64).fold(0.0, f64::max) / avg
    }

    /// 被丢弃的分配占全部路由分配的比例
    pub fn dropped_ratio(&self, top_k: usize) -> f64 {
        let total = (self.n_tokens * top_k) as f64;
        if total == 0.0 {
            return 0.0;
        }
        self.dropped as f64 / total
    }
}

/// 把各层的统计合并成一份（负载求和、丢弃求和、辅助损失取均值）。
///
/// 逐层打印在几十层的模型上会刷屏，训练日志里只需要看总量。
pub fn merge_stats(stats: &[RouteStats]) -> RouteStats {
    let mut out = RouteStats::default();
    if stats.is_empty() {
        return out;
    }
    out.n_tokens = stats.iter().map(|s| s.n_tokens).sum();
    out.dropped = stats.iter().map(|s| s.dropped).sum();
    out.aux = stats.iter().map(|s| s.aux).sum::<f32>() / stats.len() as f32;
    let e = stats[0].counts.len();
    out.counts = (0..e)
        .map(|i| stats.iter().map(|s| s.counts.get(i).copied().unwrap_or(0)).sum())
        .collect();
    out
}

// ==================== 参数 / 计算量口径 ====================

/// 单个专家（= 一个普通 FFN）的参数量。
///
/// 与 [`crate::layers`] 的真实建层共用公式：GELU 版 `8d² + 5d`（两个线性层），
/// SwiGLU 版 `3dh + 2h + d`（三个线性层，`h = swiglu_hidden(d)`）。
/// 公式与建层一旦漂移，稀疏度、显存估算就整体偏掉，而且不会有编译错误提醒——
/// 所以 `test_sparse_stats_matches_real_layer` 会拿真实层的参数量逐位核对。
pub fn expert_param_count(d: usize, use_swiglu: bool) -> usize {
    if use_swiglu {
        let h = swiglu_hidden(d);
        3 * d * h + 2 * h + d
    } else {
        8 * d * d + 5 * d
    }
}

/// 路由器（门控线性层）的参数量：`d × E` 权重 + `E` 偏置
pub fn router_param_count(d: usize, n_expert: usize) -> usize {
    d * n_expert + n_expert
}

/// MoE 的参数量与激活量口径。
///
/// 这是 MoE 最容易被误解的一点：**参数量与计算量解耦**。
/// 全部专家都要驻留显存（参数量 = E 倍），但每个 token 只激活 K 个
/// （计算量 = K/E）。所以 MoE 在**显存**上并不便宜，便宜的是 FLOPs。
#[derive(Clone, Copy, Debug)]
pub struct SparseStats {
    pub n_expert: usize,
    pub top_k: usize,
    /// 单个专家的参数量
    pub expert_params: usize,
    pub router_params: usize,
}

impl SparseStats {
    /// 全部参数（所有专家都加载）
    pub fn total_params(&self) -> usize {
        self.n_expert * self.expert_params + self.router_params
    }

    /// 每个 token 实际参与计算的参数（只有 Top-K 个专家 + 路由器）
    pub fn active_params(&self) -> usize {
        self.top_k * self.expert_params + self.router_params
    }

    /// 总参数 / 激活参数（"同样的计算量能塞进多少倍容量"）
    pub fn param_ratio(&self) -> f64 {
        self.total_params() as f64 / self.active_params() as f64
    }

    /// 与**参数量相同**的稠密 FFN 相比，每 token 省下的计算比例
    pub fn flops_saving(&self) -> f64 {
        1.0 - self.active_params() as f64 / self.total_params() as f64
    }
}

/// 按配置算出 MoE 的稀疏口径（n_expert ≤ 1 时视为稠密，比例都是 1）
pub fn sparse_stats(n_expert: usize, top_k: usize, d: usize, use_swiglu: bool) -> SparseStats {
    let e = n_expert.max(1);
    let k = top_k.clamp(1, e);
    SparseStats {
        n_expert: e,
        top_k: k,
        expert_params: expert_param_count(d, use_swiglu),
        router_params: if n_expert > 1 { router_param_count(d, e) } else { 0 },
    }
}

// ==================== MoE 层 ====================

/// 稀疏 MoE 层：路由器 + N 个专家 FFN。
///
/// 结构（见文件头注释）：`x → 路由器 → Top-K → gather → 专家 → 加权 → scatter → y`。
pub struct MoELayer {
    /// 门控（路由器）：[d, n_expert]
    pub router: Linear,
    /// N 个独立的 FFN 专家
    pub experts: Vec<MLPEnum>,
    pub n_expert: usize,
    pub top_k: usize,
    /// 容量因子（0 = 不限，见 [`expert_capacity`]）
    pub capacity_factor: f32,
    /// 均衡辅助损失系数 α（0 = 不加；损失的**平方**等更花哨的变体不在本课范围）
    pub aux_coef: f32,
    /// router z-loss 系数 β（`L_z = β·mean(logsumexp(logits)²)`，0 = 不加）。
    /// 与 α 互补：α 压平路由概率，β 按住门控 logits 的幅度（见文件头 §4）。
    pub z_loss_coef: f32,
    /// 门控权重的求和口径（见文件头 §2）：
    /// - `false`（默认，Mixtral / DeepSeek / Qwen 式）：Top-K 内部**重归一化**，`Σw = 1`。
    ///   ⚠️ K = 1 时权重恒为 1，主损失给不了路由器梯度——K = 1 请置 `true`。
    /// - `true`（Switch Transformer 式）：用**全部专家**上的 softmax 原概率，`Σw < 1`。
    pub switch_gate: bool,
    /// 最近一次前向的路由统计（`forward` 走 `&self`，所以用 `Shared` 内部 `Mutex`）
    stats: Shared<RouteStats>,
}

impl MoELayer {
    /// 按模型配置建层：专家数、Top-K、容量因子、辅助损失系数都取自 [`TransformerConfig`]。
    ///
    /// 专家的隐藏维度与稠密 MLP 完全一致（[`MLPEnum`] 同一套建层函数），
    /// 这样"把某层的稠密 FFN 换成 MoE"在参数口径上是干净可比的。
    pub fn new(cfg: &TransformerConfig, rng: &mut Rng) -> Self {
        let d = cfg.n_embd;
        assert!(
            cfg.n_expert >= 2,
            "MoE 至少要 2 个专家（model.n_expert = {}，稠密 FFN 请设 1）",
            cfg.n_expert
        );
        assert!(
            cfg.moe_top_k >= 1 && cfg.moe_top_k <= cfg.n_expert,
            "model.moe_top_k（{}）必须落在 1..=n_expert（{}）",
            cfg.moe_top_k,
            cfg.n_expert
        );
        assert!(
            cfg.moe_capacity_factor >= 0.0,
            "model.moe_capacity_factor 不能为负（0 = 不限容量）"
        );
        assert!(cfg.moe_aux_coef >= 0.0, "model.moe_aux_coef 不能为负");
        assert!(cfg.moe_z_loss_coef >= 0.0, "model.moe_z_loss_coef 不能为负");
        let experts = (0..cfg.n_expert)
            .map(|_| {
                if cfg.use_swiglu {
                    MLPEnum::new_swiglu(d, rng)
                } else {
                    MLPEnum::new_gelu(d, rng)
                }
            })
            .collect();
        MoELayer {
            router: Linear::new(d, cfg.n_expert, rng),
            experts,
            n_expert: cfg.n_expert,
            top_k: cfg.moe_top_k,
            capacity_factor: cfg.moe_capacity_factor,
            aux_coef: cfg.moe_aux_coef,
            z_loss_coef: cfg.moe_z_loss_coef,
            switch_gate: cfg.moe_switch_gate,
            stats: Shared::new(RouteStats::default()),
        }
    }

    /// 只取输出（辅助损失丢弃）。训练走 [`MoELayer::forward_with_aux`]，那条路径才带均衡损失；
    /// 这里是"只想要前向结果"的入口，单测已覆盖（`test_moe_forward_matches_dense_reference` 等）。
    #[allow(dead_code)]
    pub fn forward(&self, x: &Tensor) -> Tensor {
        self.forward_with_aux(x).0
    }

    /// 前向 + 辅助损失（均衡 α · L_aux + router z-loss β · L_z，任一非 0 就加）。
    ///
    /// 返回 `(输出, 加权后的辅助损失)`；`aux_coef` 与 `z_loss_coef` 都为 0 时第二项为
    /// `None`（不加就当它不存在，别在计算图里留一个恒等于 0 的节点）。
    ///
    /// 输入可以是 `[B, T, d]`（Transformer 里就是它）或 `[N, d]`，输出形状不变。
    pub fn forward_with_aux(&self, x: &Tensor) -> (Tensor, Option<Tensor>) {
        let d = self.router.weight.shape()[0];
        let n = x.numel() / d;
        assert!(n > 0, "MoE 前向至少要有 1 个 token");
        let x2 = if x.rank() == 2 {
            x.clone()
        } else {
            x.reshape(vec![n, d])
        };

        // 1. 门控 logits [n, E]
        let logits = self.router.forward(&x2);
        let capacity = expert_capacity(n, self.n_expert, self.top_k, self.capacity_factor);
        let plan = {
            let ld = logits.data_ref();
            top_k_gate(&ld[..], n, self.n_expert, self.top_k, capacity)
        };

        // 2. 门控权重。两条口径（见文件头 §2），差别只在 K = 1 时是否还留着梯度：
        let p_full = logits.softmax_last_dim(); // [n, E]：**全部**专家上的概率
        let w = if self.switch_gate {
            // Switch 式：取原概率，未选中位置置 0。Σw < 1，但 K = 1 时 w = p_0 ≠ 1，
            // 对 logits 的雅可比非零——主损失能把路由器一起训。
            let mask = Tensor::from_vec(plan.mask.clone(), vec![n, self.n_expert]);
            p_full.mul(&mask)
        } else {
            // Mixtral / DeepSeek 式：未选中位置 −∞ 后重归一化，Σw = 1。
            // K = 1 时 softmax 只剩一个元素 ⇒ w ≡ 1 ⇒ 对 logits 的雅可比恒为 0
            // （K = 1 请关掉这条口径，用上面的 switch_gate）。
            let penalty = Tensor::from_vec(plan.penalty.clone(), vec![n, self.n_expert]);
            logits.add(&penalty).softmax_last_dim()
        };

        // 3. 逐专家稀疏前向
        let mut out: Option<Tensor> = None;
        for (e, sel) in plan.sel.iter().enumerate() {
            if sel.is_empty() {
                continue; // 没分到 token 的专家整个跳过：稀疏激活省下的就是这段
            }
            let xe = x2.gather_rows(sel); // [n_e, d]：只取分到自己的 token
            let ye = self.experts[e].forward(&xe); // [n_e, d]
            // 取 w 的第 e 列再按 sel 取行：两步都是带梯度的索引算子，
            // 路由器照常拿得到梯度；比原来"右乘 one-hot 的整表 matmul"少算 n×E 次乘加
            let we = w.gather_rows(sel).select_col(e); // [n_e, 1]
            let contrib = we.mul(&ye); // [n_e, d]（[n_e,1] 广播到 [n_e,d]）
            let scattered = contrib.scatter_add_rows(sel, n); // 散射回原位相加
            out = Some(match out {
                None => scattered,
                Some(acc) => acc.add(&scattered),
            });
        }
        // 全部 token 的分配都被容量丢掉时这里没有专家跑过：输出是全零张量。
        // 这是"被丢弃"的正确语义（该层的贡献为 0，残差照常直通）。
        let out = out.unwrap_or_else(|| Tensor::from_vec(vec![0.0f32; n * d], vec![n, d]));

        // 4. 均衡辅助损失：L_aux = E · Σ f_i · p_i（惩罚 f 与 p 的背离，见文件头 §4）
        let p_mean = p_full.transpose().sum_last_dim().mul_scalar(1.0 / n as f32); // [E, 1]
        let f = Tensor::from_vec(plan.f.clone(), vec![self.n_expert, 1]);
        let aux_raw = f.mul(&p_mean).sum().mul_scalar(self.n_expert as f32);

        // 5. router z-loss：L_z = mean_i log²(Σ_j e^{logits_ij})（PaLM 的经典正则）。
        //    与 α 互补：α 压平路由**概率**，β 按住 logits 的**幅度**——logits 越推越大
        //    ⇒ softmax 过尖 ⇒ 路由饱和、梯度消失。走 logsumexp 算子（减行 max，数值稳定），
        //    不用 log(exp(...)) 裸拼（logits 稍大就溢出成 inf）。
        let z_raw = if self.z_loss_coef > 0.0 {
            let lse = logits.logsumexp_last_dim(); // [n, 1]
            Some(lse.pow(2.0).sum().mul_scalar(1.0 / n as f32))
        } else {
            None
        };

        // 6. 统计（供训练日志 / 诊断读取；值本身不参与计算图）
        {
            let mut st = self.stats.borrow_mut();
            st.counts = plan.sel.iter().map(|s| s.len()).collect();
            st.dropped = plan.dropped;
            st.n_tokens = n;
            st.aux = aux_raw.item();
        }

        let out = if x.rank() == 2 {
            out
        } else {
            out.reshape(x.shape().to_vec())
        };
        let mut aux: Option<Tensor> = None;
        if self.aux_coef > 0.0 {
            aux = Some(aux_raw.mul_scalar(self.aux_coef));
        }
        if let Some(z) = z_raw {
            let z = z.mul_scalar(self.z_loss_coef);
            aux = Some(match aux {
                None => z,
                Some(a) => a.add(&z),
            });
        }
        (out, aux)
    }

    /// 只做路由、不做专家前向（诊断用）：拿一份输入就能看到"谁被分到哪个专家"。
    ///
    /// 内部按 `no_grad` 算门控，纯查询，不影响任何梯度。
    /// 单测已覆盖（`test_capacity_factor_drops_overflow`）；`moe` 子命令读的是
    /// [`MoELayer::stats`]，那条路会连专家前向的耗时一起算进去，更贴近真实开销。
    #[allow(dead_code)]
    pub fn route_plan(&self, x: &Tensor) -> RoutePlan {
        let d = self.router.weight.shape()[0];
        let n = x.numel() / d;
        let x2 = if x.rank() == 2 {
            x.clone()
        } else {
            x.reshape(vec![n, d])
        };
        let logits = crate::tensor::no_grad(|| self.router.forward(&x2));
        let capacity = expert_capacity(n, self.n_expert, self.top_k, self.capacity_factor);
        let ld = logits.data_ref();
        top_k_gate(&ld[..], n, self.n_expert, self.top_k, capacity)
    }

    /// 最近一次前向的路由统计
    pub fn stats(&self) -> RouteStats {
        self.stats.borrow().clone()
    }

    /// 带名字的参数（checkpoint 用）：
    /// `{prefix}.router.weight`、`{prefix}.experts.{i}.w_gate.weight` ...
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps = self.router.named_parameters(&format!("{prefix}.router"));
        for (i, e) in self.experts.iter().enumerate() {
            ps.extend(e.named_parameters(&format!("{prefix}.experts.{i}")));
        }
        ps
    }
}

impl Module for MoELayer {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.router.parameters();
        for e in &self.experts {
            ps.extend(e.parameters());
        }
        ps
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::zero_grad_all;
    use crate::optim::{AdamW, Optimizer};

    /// 造一个可控的 MoE 层：只改路由器的偏置就能精确指定"每个 token 想去哪个专家"
    /// （偏置相同、输入不同时 logits 的差异来自随机权重，用来测并列/分散的情形）。
    ///
    /// `top_k == 1` 时自动开 `moe_switch_gate`：重归一化口径在 K = 1 时 `w ≡ 1`，
    /// 对 logits 的雅可比恒为 0，**主损失给不了路由器任何梯度**（见文件头 §2）。
    /// 要测 K = 1 的路由器梯度，就必须走 switch 口径。K ≥ 2 两条口径都留着梯度，
    /// 默认的重归一化口径才是 Mixtral 的真实做法，所以只在 K = 1 时切换。
    fn build(d: usize, n_expert: usize, top_k: usize, seed: u64) -> MoELayer {
        let cfg = TransformerConfig {
            vocab_size: 16,
            n_embd: d,
            n_expert,
            moe_top_k: top_k,
            moe_capacity_factor: 0.0,
            moe_aux_coef: 0.0,
            moe_switch_gate: top_k == 1,
            ..Default::default()
        };
        let mut rng = Rng::new(seed);
        MoELayer::new(&cfg, &mut rng)
    }

    fn rand_x(n: usize, d: usize, seed: u64) -> Tensor {
        let mut rng = Rng::new(seed);
        Tensor::from_vec(
            (0..n * d).map(|_| rng.randn()).collect(),
            vec![n, d],
        )
    }

    /// 路由：Top-K 恰好取到 logits 最大的 K 个，且并列时按下标升序（确定性）
    #[test]
    fn test_top_k_gate_picks_highest_logits() {
        let e = 4;
        let n = 3;
        // token 0：专家 2 最高；token 1：专家 0 最高；token 2：全并列（应选下标最小的）
        let logits = vec![
            0.1, 0.9, 5.0, 0.2, // token 0
            7.0, 0.3, 0.4, 0.5, // token 1
            1.0, 1.0, 1.0, 1.0, // token 2
        ];
        let plan = top_k_gate(&logits, n, e, 2, usize::MAX);
        // 逐 token 的 Top-2：token 0 → {2,1}、token 1 → {0,3}、token 2 → {0,1}（并列取小下标）
        // `sel[e]` 是"专家 e 收到的 token 行号（升序）"，不是"token 选了哪些专家"
        assert_eq!(plan.sel[0], vec![1, 2], "token 1 与 token 2 都选中了专家 0");
        assert_eq!(plan.sel[1], vec![0, 2], "token 0 与 token 2 都选中了专家 1");
        assert_eq!(plan.sel[2], vec![0], "只有 token 0 选中了专家 2");
        assert_eq!(plan.sel[3], vec![1], "只有 token 1 选中了专家 3");
        assert_eq!(plan.dropped, 0);
        assert_eq!(plan.routed, vec![2, 2, 1, 1]);
        assert!((plan.f.iter().sum::<f32>() - 1.0).abs() < 1e-6, "f 必须归一");
        // 掩码：选中的位置 0（penalty）/ 1（mask），其余 −∞ / 0，两者必须互补
        for i in 0..n {
            for j in 0..e {
                let selected = plan.sel[j].contains(&i);
                assert_eq!(plan.penalty[i * e + j] == 0.0, selected, "掩码与选择不一致");
                assert_eq!(plan.mask[i * e + j] == 1.0, selected, "选中掩码与选择不一致");
            }
        }
    }

    /// top-k 选择（select_nth）与全排序参考逐字段一致：sel / penalty / mask / routed / f。
    /// 性能改造（整行 O(E log E) → O(E) + O(k log k)）不得改变任何路由行为，
    /// 尤其"并列时按下标升序"的确定性——这里刻意量化 logits 制造大量并列。
    #[test]
    fn test_top_k_gate_matches_full_sort() {
        let (n, e, k) = (17usize, 13usize, 3usize);
        let mut rng = Rng::new(123);
        // 量化到 0.25 步长：13 个专家、8 个不同取值，必然大量并列
        let logits: Vec<f32> = (0..n * e)
            .map(|_| (rng.next_f32() * 8.0).floor() * 0.25)
            .collect();
        let got = top_k_gate(&logits, n, e, k, usize::MAX);

        // 参考实现：改造前的整行全排序
        let mut sel_ref: Vec<Vec<usize>> = vec![Vec::new(); e];
        let mut penalty_ref = vec![f32::NEG_INFINITY; n * e];
        let mut mask_ref = vec![0.0f32; n * e];
        let mut routed_ref = vec![0usize; e];
        for i in 0..n {
            let row = &logits[i * e..(i + 1) * e];
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            for &j in &order[..k] {
                penalty_ref[i * e + j] = 0.0;
                mask_ref[i * e + j] = 1.0;
                routed_ref[j] += 1;
                sel_ref[j].push(i);
            }
        }
        assert_eq!(got.sel, sel_ref, "sel 与全排序不一致");
        assert_eq!(got.routed, routed_ref, "routed 与全排序不一致");
        assert_eq!(got.penalty, penalty_ref, "penalty 与全排序不一致");
        assert_eq!(got.mask, mask_ref, "mask 与全排序不一致");
        assert_eq!(got.dropped, 0, "不限容量不该有丢弃");
        let f_ref: Vec<f32> = routed_ref
            .iter()
            .map(|&c| c as f32 / (n * k) as f32)
            .collect();
        assert_eq!(got.f, f_ref, "f 与全排序不一致");
    }

    /// router z-loss 的数值与梯度：K = 1 + 重归一化口径下主损失给不了路由器梯度
    /// （文件头 §2 的坑），z-loss 必须能把梯度补上；系数为 0 时不留任何节点。
    #[test]
    fn test_z_loss_value_and_router_gradient() {
        let (d, e, n) = (4usize, 3usize, 6usize);
        let x = rand_x(n, d, 8);
        let cfg = TransformerConfig {
            vocab_size: 16,
            n_embd: d,
            n_expert: e,
            moe_top_k: 1,
            moe_switch_gate: false, // 重归一化：主损失梯度恒为 0（正是 z-loss 的用武之地）
            moe_z_loss_coef: 1.0,
            ..Default::default()
        };
        let mut rng = Rng::new(23);
        let layer = MoELayer::new(&cfg, &mut rng);

        let (_, aux) = layer.forward_with_aux(&x);
        let aux = aux.expect("β > 0 时应返回辅助损失");

        // 数值对拍：mean(logsumexp(logits)²)，纯手算参考
        let logits = layer.router.forward(&x).data();
        let mut z_ref = 0.0f32;
        for i in 0..n {
            let row = &logits[i * e..(i + 1) * e];
            let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let s: f32 = row.iter().map(|&v| (v - m).exp()).sum();
            let lse = m + s.ln();
            z_ref += lse * lse;
        }
        z_ref /= n as f32;
        assert!(
            (aux.item() - z_ref).abs() < 1e-4,
            "z-loss 数值不对：实测 {}，手算 {z_ref}",
            aux.item()
        );

        // 梯度：β · L_z 必须送到路由器（重归一化 K = 1 的主路径给不了）
        zero_grad_all(&layer);
        aux.backward();
        let gsum: f32 = layer.router.weight.grad().iter().map(|g| g.abs()).sum();
        assert!(gsum > 1e-6, "z-loss 必须给路由器梯度，实测 {gsum}");

        // β = 0（默认）且 α = 0 ⇒ 不返回（也不在计算图里留恒零节点）
        let layer0 = MoELayer::new(
            &TransformerConfig {
                vocab_size: 16,
                n_embd: d,
                n_expert: e,
                moe_top_k: 1,
                moe_switch_gate: false,
                ..Default::default()
            },
            &mut Rng::new(23),
        );
        assert!(
            layer0.forward_with_aux(&x).1.is_none(),
            "β = 0 且 α = 0 时不该返回辅助损失"
        );
    }

    /// 门控权重：只在 Top-K 上做 softmax ⇒ 恰好 K 个非零、和为 1；
    /// 未选中位置概率为 0，且 softmax 对它们的梯度也恰好为 0（可导地"只归一化 Top-K"）
    #[test]
    fn test_gate_weights_are_softmax_over_topk_only() {
        let e = 4usize;
        let n = 2usize;
        let logits = vec![0.0, 2.0, 1.0, -1.0, 3.0, 3.0, 0.0, 0.0];
        let plan = top_k_gate(&logits, n, e, 2, usize::MAX);
        // 用 param 构造：待会儿要读它的梯度
        let logits_t = Tensor::param(logits.clone(), vec![n, e]);
        let penalty = Tensor::from_vec(plan.penalty.clone(), vec![n, e]);
        let w = logits_t.add(&penalty).softmax_last_dim();
        let wd = w.data();
        for i in 0..n {
            let row = &wd[i * e..(i + 1) * e];
            assert!(
                (row.iter().sum::<f32>() - 1.0).abs() < 1e-6,
                "第 {i} 行权重和应为 1：{row:?}"
            );
            let nonzero = row.iter().filter(|v| **v > 0.0).count();
            assert_eq!(nonzero, 2, "应恰好 2 个非零权重：{row:?}");
        }
        // token 0 选中 {1, 2}，logits [2, 1] ⇒ softmax = [e/(e+1), 1/(e+1)]
        let (a, b) = (1.0f32.exp(), 1.0f32);
        assert!((wd[1] - a / (a + b)).abs() < 1e-6);
        assert!((wd[2] - b / (a + b)).abs() < 1e-6);
        // token 1 选中 {0, 1}（并列 3.0，按下标升序）⇒ 各 0.5
        assert!((wd[4] - 0.5).abs() < 1e-6 && (wd[5] - 0.5).abs() < 1e-6);

        // 梯度：被选中的 logits 才拿得到梯度（−∞ 那一列 d(softmax)/dlogit ≡ 0）。
        // 注意不能直接用 w.sum()——softmax 各行和恒为 1，那个标量的梯度恒等于 0，
        // 要用一组各不相同的系数把它"称"出来。
        let coef = Tensor::from_vec(
            (0..n * e).map(|i| 0.5 + i as f32 * 0.3).collect(),
            vec![n, e],
        );
        let scalar = w.mul(&coef).sum();
        logits_t.zero_grad();
        scalar.backward();
        let gl = logits_t.grad();
        assert!(
            gl[3].abs() < 1e-7 && gl[6].abs() < 1e-7 && gl[7].abs() < 1e-7,
            "未选中的 logits 不该有梯度：{gl:?}"
        );
        assert!(
            gl[1].abs() > 1e-3 && gl[4].abs() > 1e-3,
            "被选中的 logits 必须拿到梯度：{gl:?}"
        );
    }

    /// 稀疏前向 vs 稠密参考实现：逐 token、逐专家地按定义算 `Σ w_k E_k(x)`，
    /// 必须与 gather→expert→weighted→scatter 的实现吻合。
    ///
    /// 这条是 MoE 的核心正确性测试——它同时覆盖了路由顺序（降序）、Top-K 内 softmax、
    /// 加权求和、以及散射回原位的行对齐。
    #[test]
    fn test_moe_forward_matches_dense_reference() {
        let (d, e, k, n) = (4usize, 3usize, 2usize, 5usize);
        let layer = build(d, e, k, 11);
        let x = rand_x(n, d, 7);

        let out = layer.forward(&x);
        assert_eq!(out.shape(), &[n, d]);

        // 参考实现：纯数据、逐 token 逐专家
        let logits = layer.router.forward(&x).data();
        let mut expect = vec![0.0f32; n * d];
        for i in 0..n {
            let row = &logits[i * e..(i + 1) * e];
            let mut order: Vec<usize> = (0..e).collect();
            order.sort_by(|&a, &b| row[b].total_cmp(&row[a]).then(a.cmp(&b)));
            let picked = &order[..k];
            let m = picked.iter().fold(f32::NEG_INFINITY, |acc, &j| acc.max(row[j]));
            let denom: f32 = picked.iter().map(|&j| (row[j] - m).exp()).sum();
            let xi = x.gather_rows(&[i]);
            for &j in picked {
                let wij = (row[j] - m).exp() / denom;
                let yj = layer.experts[j].forward(&xi).data();
                for t in 0..d {
                    expect[i * d + t] += wij * yj[t];
                }
            }
        }
        let got = out.data();
        for (idx, (a, b)) in got.iter().zip(&expect).enumerate() {
            assert!(
                (a - b).abs() < 2e-5,
                "第 {idx} 个元素不一致：稀疏实现 {a} vs 稠密参考 {b}"
            );
        }
    }

    /// 每个 token 的输出只取决于它自己：同一行在不同批次里结果必须一致
    /// （路由与门控都是逐 token 的，没有跨 token 的信息泄漏——容量不限时）
    #[test]
    fn test_moe_routing_is_per_token() {
        let (d, e, k) = (6usize, 4usize, 2usize);
        let layer = build(d, e, k, 3);
        let x = rand_x(5, d, 21);
        let full = layer.forward(&x).data();
        let single = layer.forward(&x.gather_rows(&[2])).data();
        for t in 0..d {
            assert!(
                (full[2 * d + t] - single[t]).abs() < 1e-6,
                "单条前向与整批前向的第 2 行不一致（第 {t} 维）"
            );
        }
    }

    /// 容量因子：超出的分配被丢弃，且丢弃按 token 顺序先到先得
    #[test]
    fn test_capacity_factor_drops_overflow() {
        let (d, e, k, n) = (3usize, 2usize, 1usize, 6usize);
        let mut layer = build(d, e, k, 5);
        // 偏置让所有 token 的首选都是专家 0（专家 1 永不入选）
        layer.router.weight.set_data(vec![0.0; d * e]);
        layer.router.bias.set_data(vec![5.0, 0.0]);
        let x = rand_x(n, d, 9);

        // 容量因子 0.5：平均负载 6×1/2 = 3，容量 = ceil(1.5) = 2
        layer.capacity_factor = 0.5;
        let plan = layer.route_plan(&x);
        assert_eq!(plan.sel[0], vec![0, 1], "专家 0 只收前 2 个 token");
        assert!(plan.sel[1].is_empty(), "专家 1 没人选");
        assert_eq!(plan.dropped, 4, "4 条分配被丢弃");
        assert_eq!(plan.routed, vec![n, 0], "路由决定（含丢弃）仍记 6 条");

        let out = layer.forward(&x).data();
        for i in 2..n {
            for t in 0..d {
                assert_eq!(out[i * d + t], 0.0, "被丢弃的 token 该层输出应为 0");
            }
        }
        assert!(out[..2 * d].iter().any(|v| *v != 0.0), "前 2 个 token 应正常算出");

        // 不限容量：6 个 token 全部走专家 0
        layer.capacity_factor = 0.0;
        let plan = layer.route_plan(&x);
        assert_eq!(plan.sel[0].len(), n);
        assert_eq!(plan.dropped, 0);

        // 统计口径：`route_plan` 是纯查询（内部 no_grad），不写 stats——要再跑一次
        // `forward` 才能看到"不丢 token"的负载
        let _ = layer.forward(&x);
        let st = layer.stats();
        assert_eq!(st.counts, vec![n, 0]);
        assert_eq!(st.dropped, 0);
        assert!((st.imbalance() - 2.0).abs() < 1e-9, "E=2 全压一个专家时不均衡度 = 2");
    }

    /// 均衡辅助损失 `L_aux = E · Σ f_i·p_i` 的取值规律。
    ///
    /// ⚠️ 它不是"任意 f、p 都 ≥ 1"——它惩罚的是 **f 与 p 的背离**：
    /// - `p = f`（路由器完全自信）时 `L_aux = E·Σ f_i² ≥ 1`（柯西–施瓦茨），
    ///   等号只在 f 均匀时成立；
    /// - f 与 p 的支撑集不交时 `Σ f_i·p_i = 0`，`L_aux` 可以低到 0。
    #[test]
    fn test_aux_loss_minimum_at_uniform() {
        let e = 4usize;
        // p_mean 用 [E,1] 构造（与 MoELayer 里的形状一致）
        let mk = |v: Vec<f32>| Tensor::from_vec(v, vec![e, 1]);
        let aux_of = |f: Vec<f32>, p: Vec<f32>| -> f32 {
            mk(f).mul(&mk(p)).sum().mul_scalar(e as f32).item()
        };

        // 均匀 f = 均匀 p：aux = E · Σ (1/E)(1/E) = 1（全局最小值）
        assert!(
            (aux_of(vec![0.25; 4], vec![0.25; 4]) - 1.0).abs() < 1e-6,
            "均匀路由应取最小值 1"
        );
        // 极端偏斜：f = p = one-hot ⇒ aux = E
        assert!(
            (aux_of(vec![1.0, 0.0, 0.0, 0.0], vec![1.0, 0.0, 0.0, 0.0]) - 4.0).abs() < 1e-6,
            "全部压在一个专家上应为 E"
        );

        // p = f 的任意分布：E·Σ f_i² ≥ 1（柯西–施瓦茨 Σ f_i² ≥ 1/E）
        for f in [
            vec![0.7, 0.1, 0.1, 0.1],
            vec![0.4, 0.3, 0.2, 0.1],
            vec![0.25; 4],
        ] {
            let v = aux_of(f.clone(), f.clone());
            assert!(v >= 1.0 - 1e-6, "p = f 时下界是 1，f = {f:?} 实测 {v}");
        }

        // 背离的极端：支撑集不交 ⇒ 0。这就是"L_aux 恒 ≥ 1"这种说法错在哪
        let disjoint = aux_of(vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.0, 0.5, 0.5]);
        assert!(disjoint.abs() < 1e-6, "支撑集不交时下界是 0，实测 {disjoint}");

        // ⚠️ 最关键的一条：**p 均匀时 L_aux = 1，与 f 长什么样无关**
        // （Σ f_i·p_i = (1/E)·Σ f_i = 1/E ⇒ L_aux = E·(1/E) = 1）。
        // 所以"L_aux 掉到 1"不是负载均衡的证书：f 可以完全压在一个专家上。
        // `moe` 子命令第二节的隔离实验实测到了这个退化解（L_aux → 1.0，负载原地不动）。
        let f_one_hot = aux_of(vec![1.0, 0.0, 0.0, 0.0], vec![0.25; 4]);
        assert!(
            (f_one_hot - 1.0).abs() < 1e-6,
            "p 均匀时无论 f 多偏都取到 1（退化解），实测 {f_one_hot}"
        );

        // 梯度下降最小化 L_aux 的方向：把概率质量从 f 大的专家挪到 f 小的专家，
        // 于是 L_aux 变小（∂L/∂logit_j = α·E·p_j·(f_j − Σf_i p_i)：
        // f_j 高于加权平均的专家，logit 被压低）
        let matched = aux_of(vec![0.5, 0.5, 0.0, 0.0], vec![0.5, 0.5, 0.0, 0.0]);
        let shifted = aux_of(vec![0.5, 0.5, 0.0, 0.0], vec![0.0, 0.5, 0.5, 0.0]);
        assert!(
            shifted < matched,
            "质量挪到 f 小的专家后 L_aux 应更小：{shifted} 应 < {matched}"
        );
    }

    /// 真实 MoE 层算出的辅助损失：路由偏斜时 > 1
    #[test]
    fn test_layer_aux_loss_is_above_one_when_skewed() {
        let (d, e) = (4usize, 4usize);
        let mut layer = build(d, e, 1, 13);
        layer.router.weight.set_data(vec![0.0; d * e]);
        // 所有 token 都压倒性地选专家 0（但概率不是 0，别的专家仍有非零 p）
        layer.router.bias.set_data(vec![2.0, 0.0, 0.0, 0.0]);
        let x = rand_x(8, d, 4);
        // aux_coef = 1，便于直接读回未缩放的均衡损失
        layer.aux_coef = 1.0;
        let (_, aux) = layer.forward_with_aux(&x);
        let aux = aux.expect("α > 0 时应返回辅助损失");
        assert!(
            aux.item() > 1.05,
            "负载偏斜时均衡损失应明显大于 1，实测 {}",
            aux.item()
        );
        assert_eq!(layer.stats().counts, vec![8, 0, 0, 0]);

        // 把门控压平（含随机权重带来的逐 token 差异）后应回到接近 1
        layer.router.bias.set_data(vec![0.0; e]);
        let (_, aux) = layer.forward_with_aux(&x);
        assert!(
            aux.unwrap().item() < 1.3,
            "门控接近均匀时均衡损失应接近 1"
        );

        // α = 0 ⇒ 不返回辅助损失（也不在计算图里留无用节点）
        layer.aux_coef = 0.0;
        assert!(layer.forward_with_aux(&x).1.is_none());
    }

    /// K = 1 的两条门控口径在**梯度**上的差别（文件头 §2 那个坑的回归测试）：
    /// - 重归一化（Mixtral 式）：Top-K 只有一个元素 ⇒ softmax 恒为 1 ⇒
    ///   `∂w/∂logit ≡ 0`，主损失**给不了路由器任何梯度**；
    /// - 原概率（Switch 式）：`w = p_0 ∈ (0,1)`，雅可比非零，路由器拿得到梯度。
    #[test]
    fn test_k1_renorm_gate_has_no_router_gradient() {
        let (d, e) = (4usize, 3usize);
        let x = rand_x(4, d, 6);
        let router_grad_sum = |switch_gate: bool| -> f32 {
            let cfg = TransformerConfig {
                vocab_size: 16,
                n_embd: d,
                n_expert: e,
                moe_top_k: 1,
                moe_switch_gate: switch_gate,
                ..Default::default()
            };
            let mut rng = Rng::new(17);
            let layer = MoELayer::new(&cfg, &mut rng);
            zero_grad_all(&layer);
            let y = layer.forward(&x);
            // 用一组互不相同的系数把输出"称"出来：直接 sum 会被对称性抹平，
            // 读不出路由器的梯度是否存在（权重和恒为 1，那条路本就无梯度）
            let coef = Tensor::from_vec(
                (0..4 * d).map(|i| 0.3 + i as f32 * 0.07).collect(),
                vec![4, d],
            );
            y.mul(&coef).sum().backward();
            layer.router.weight.grad().iter().map(|g| g.abs()).sum()
        };
        let renorm = router_grad_sum(false);
        let switch = router_grad_sum(true);
        assert!(
            renorm < 1e-7,
            "重归一化 + K = 1 时路由器不该从主损失拿到梯度，实测 {renorm}"
        );
        assert!(
            switch > 1e-4,
            "Switch 口径 + K = 1 时路由器必须拿到梯度，实测 {switch}"
        );
    }

    /// 稀疏性与梯度稀疏性：只有被选中的专家拿得到梯度，路由器一定拿得到
    #[test]
    fn test_only_selected_experts_receive_gradients() {
        let (d, e) = (4usize, 3usize);
        let layer = build(d, e, 1, 17);
        layer.router.weight.set_data(vec![0.0; d * e]);
        layer.router.bias.set_data(vec![3.0, 0.0, 0.0]); // 永远选专家 0
        let x = rand_x(4, d, 6);
        zero_grad_all(&layer);
        let out = layer.forward(&x);
        out.sum().backward();

        let router_grad = layer.router.weight.grad();
        assert!(
            router_grad.iter().any(|g| g.abs() > 1e-6),
            "路由器必须拿到梯度（门控权重的路径不能断）"
        );
        for (i, expert) in layer.experts.iter().enumerate() {
            let has = expert
                .parameters()
                .iter()
                .any(|p| p.grad().iter().any(|g| g.abs() > 1e-6));
            if i == 0 {
                assert!(has, "被选中的专家 0 必须拿到梯度");
            } else {
                assert!(!has, "没被选中的专家 {i} 不该拿到任何梯度");
            }
        }
    }

    /// 参数口径：公式必须与真实建层逐位一致（一处漂移，稀疏度估算就整体偏掉）
    #[test]
    fn test_sparse_stats_matches_real_layer() {
        for use_swiglu in [false, true] {
            let (d, e, k) = (8usize, 4usize, 2usize);
            let cfg = TransformerConfig {
                vocab_size: 16,
                n_embd: d,
                n_expert: e,
                moe_top_k: k,
                use_swiglu,
                ..Default::default()
            };
            let mut rng = Rng::new(2);
            let layer = MoELayer::new(&cfg, &mut rng);
            let measured: usize = layer.parameters().iter().map(|p| p.numel()).sum();
            let ss = sparse_stats(e, k, d, use_swiglu);
            assert_eq!(
                measured,
                ss.total_params(),
                "SwiGLU={use_swiglu}：实测参数 {measured} vs 公式 {}",
                ss.total_params()
            );
            // 激活量口径：Top-K 个专家 + 路由器
            assert_eq!(
                ss.active_params(),
                k * ss.expert_params + ss.router_params
            );
            // 稀疏性：4 个专家取 2 个 ⇒ 激活参数约占一半，省下约一半 FLOPs
            assert!((ss.param_ratio() - 2.0).abs() < 0.05, "比例 {}", ss.param_ratio());
            assert!(
                (ss.flops_saving() - 0.5).abs() < 0.05,
                "省下的比例 {}",
                ss.flops_saving()
            );
        }
    }

    /// 均衡损失真的能把负载推平：偏斜的初始路由，只优化辅助损失若干步，
    /// 不均衡度必须下降（这就是"加不加辅助损失"的对照实验的机理）
    #[test]
    fn test_aux_loss_gradient_balances_routing() {
        let (d, e, n) = (8usize, 4usize, 32usize);
        let mut layer = build(d, e, 1, 23);
        layer.aux_coef = 1.0;
        // 初始：偏置强烈偏向专家 0 ⇒ 全部 token 都路由到专家 0
        layer.router.bias.set_data(vec![3.0, 0.0, 0.0, 0.0]);
        let x = rand_x(n, d, 8);
        let _ = layer.forward_with_aux(&x).0;
        let imb_before = layer.stats().imbalance();
        // 路由器权重是随机的，逐 token 的 logits 有差异，偏 3.0 压不住所有 token；
        // 这里只要求"明显偏斜"（E = 4 时 1.0 是完美均衡）
        assert!(
            imb_before > 2.0,
            "初始应明显偏斜，实测不均衡度 {imb_before}"
        );

        let params = layer.parameters();
        let mut opt = AdamW::new(0.1, params, 0.0);
        for _ in 0..60 {
            zero_grad_all(&layer);
            let (_, aux) = layer.forward_with_aux(&x);
            aux.expect("α = 1 必有辅助损失").backward();
            opt.step();
        }
        let _ = layer.forward_with_aux(&x);
        let imb_after = layer.stats().imbalance();
        assert!(
            imb_after < imb_before - 1.0,
            "均衡损失未能改善负载：{imb_before} -> {imb_after}（负载 {:?}）",
            layer.stats().counts
        );
    }

    /// 端到端：MoE 层 + 输出头真的能学到东西（簇状合成任务，专家可分工）
    #[test]
    fn test_moe_layer_trains() {
        let (d, e, k, cluster, steps) = (8usize, 4usize, 1usize, 2usize, 80usize);
        let mut rng = Rng::new(31);
        // 目标：两个簇各有一个专属线性映射 y = tanh(x @ W_c)
        let centers: Vec<Vec<f32>> = (0..cluster)
            .map(|_| (0..d).map(|_| rng.randn() * 2.0).collect())
            .collect();
        let maps: Vec<Vec<f32>> = (0..cluster)
            .map(|_| (0..d * d).map(|_| rng.randn() * 0.5).collect())
            .collect();
        let head = Linear::new(d, d, &mut rng);
        let cfg = TransformerConfig {
            vocab_size: 16,
            n_embd: d,
            n_expert: e,
            moe_top_k: k,
            moe_aux_coef: 0.01,
            // 固定 GELU 专家：本测验证 MoE 路由 + 训练能收敛，不测激活函数。
            // SwiGLU 在 d=8 时 hidden 被 256 对齐抬到 256（正常是 32），
            // 80 步 toy 训练下学不动
            use_swiglu: false,
            ..Default::default()
        };
        let moe = MoELayer::new(&cfg, &mut rng);
        let mut params = moe.parameters();
        params.extend(head.parameters());
        let mut opt = AdamW::new(0.05, params, 0.0);

        let batch = 32usize;
        let mut eval_rng = Rng::new(99);
        let eval = |rng: &mut Rng| -> f32 {
            let (x, y) = sample_clusters(&centers, &maps, batch, d, rng);
            let out = crate::tensor::no_grad(|| head.forward(&moe.forward(&x)));
            crate::loss::mse_loss(&out, &y).item()
        };
        let first = eval(&mut eval_rng);
        for _ in 0..steps {
            let (x, y) = sample_clusters(&centers, &maps, batch, d, &mut rng);
            zero_grad_all(&moe);
            zero_grad_all(&head);
            let (h, aux) = moe.forward_with_aux(&x);
            let pred = head.forward(&h);
            let loss = crate::loss::mse_loss(&pred, &y);
            let loss = match aux {
                Some(a) => loss.add(&a),
                None => loss,
            };
            loss.backward();
            opt.step();
        }
        let last = eval(&mut eval_rng);
        assert!(
            last < first * 0.5,
            "MoE 没能学起来：{first:.4} -> {last:.4}"
        );
    }

    /// 采一批"簇状"样本：随机选簇 → 中心 + 噪声 → 标签是簇专属映射
    fn sample_clusters(
        centers: &[Vec<f32>],
        maps: &[Vec<f32>],
        batch: usize,
        d: usize,
        rng: &mut Rng,
    ) -> (Tensor, Tensor) {
        let mut xs = vec![0.0f32; batch * d];
        let mut ys = vec![0.0f32; batch * d];
        for i in 0..batch {
            let c = (rng.next_f32() * centers.len() as f32) as usize % centers.len();
            let x: Vec<f32> = (0..d).map(|j| centers[c][j] + rng.randn() * 0.3).collect();
            for r in 0..d {
                let mut acc = 0.0;
                for j in 0..d {
                    acc += x[j] * maps[c][r * d + j];
                }
                ys[i * d + r] = acc.tanh();
            }
            xs[i * d..(i + 1) * d].copy_from_slice(&x);
        }
        (
            Tensor::from_vec(xs, vec![batch, d]),
            Tensor::from_vec(ys, vec![batch, d]),
        )
    }
}
