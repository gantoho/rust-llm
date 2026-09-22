//! RLHF 与对齐（第 36 课）：奖励模型、DPO、GRPO、PPO
//!
//! 预训练与 SFT 让模型"会接话"，对齐让它"说人话"。两者用的监督信号有本质差别：
//! 前者是"这段文本的概率"，后者是"这条回答比那条回答更好"——**成对比较**。
//! 成对信号既不需要人写标准答案，也比绝对打分稳定（人对"哪个更好"的共识远高于
//! "打几分"），代价是需要一套把比较转成梯度的目标函数，本模块实现其中四种：
//!
//! - [`RewardModel`]：奖励模型（GPT 主干 + 标量头），Bradley-Terry 成对损失。
//!   一次训练之后它就成了一个可复用的"裁判"，PPO 与拒答采样都靠它打分。
//! - DPO：**跳过**奖励模型，直接在偏好对上优化策略。
//! - GRPO：同一 prompt 采一组回答，组内相对优势当权重。
//! - PPO：经典的"奖励 + 裁剪的策略梯度 + 参考模型 KL 约束"。

use crate::layers::Linear;
use crate::loss::cross_entropy_loss_masked;
use crate::model::GPT;
use crate::module::Module;
use crate::rng::Rng;
use crate::tensor::Tensor;

// ==================== 数值工具 ====================

/// 数值稳定的 sigmoid：`x` 很负时直接算 `1/(1+e^{-x})` 会溢出成 `inf`。
pub fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// 数值稳定的 `log(1 + e^x)`：`x` 大时 `e^x` 溢出，改写成 `x + log(1 + e^{-x})`。
pub fn softplus(x: f32) -> f32 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}

// ==================== 奖励模型 ====================

/// 奖励模型：GPT 主干 + 一个标量头，给一条完整回答打"有多好"的分数。
///
/// 分数是**无界实数**，只有相对大小有意义——训练目标（Bradley-Terry）只看两个分数
/// 的差，所以不存在"标定"问题，也没必要强制它落在某个区间。
pub struct RewardModel {
    backbone: GPT,
    head: Linear,
}

impl RewardModel {
    pub fn new(backbone: GPT, rng: &mut Rng) -> Self {
        let head = Linear::new(backbone.cfg.n_embd, 1, rng);
        RewardModel { backbone, head }
    }

    /// 序列分数：取**最后一个位置**的隐状态（因果模型的最后一位在注意力里"看过"整条
    /// 序列）再过一个标量头。
    pub fn score(&self, ids: &[usize], training: bool) -> Tensor {
        assert!(!ids.is_empty(), "奖励模型不接受空序列");
        let t = ids.len();
        let hidden = self.backbone.forward_hidden(ids, 1, t, training);
        let last = hidden.gather_rows(&[t - 1]);
        self.head.forward(&last).sum()
    }

    /// 推理用取值：不建图。
    pub fn score_value(&self, ids: &[usize]) -> f32 {
        crate::tensor::no_grad(|| self.score(ids, false).item())
    }

    /// 一对偏好样本的 Bradley-Terry 损失
    pub fn pairwise_loss(&self, chosen: &[usize], rejected: &[usize], training: bool) -> Tensor {
        let r_chosen = self.score(chosen, training);
        let r_rejected = self.score(rejected, training);
        bradley_terry_loss(&r_chosen, &r_rejected)
    }
}

impl Module for RewardModel {
    fn parameters(&self) -> Vec<Tensor> {
        let mut params = self.backbone.parameters();
        params.extend(self.head.parameters());
        params
    }
}

/// Bradley-Terry 成对损失：`L = -log σ(r_chosen − r_rejected)`。
///
/// 这是"比较"到"梯度"的桥：`σ(Δ)` 建模"chosen 更好"的概率，取负对数似然即可。
/// 等价写法 `L = softplus(−Δ)`，`Δ = 0` 时取到 `ln 2 ≈ 0.6931`——两条回答不分伯仲时
/// 损失最大，训练把它推大。
///
/// 反向走 [`Tensor::external_scalar_loss`] 手写注入：`dL/dΔ = −σ(−Δ) = σ(Δ) − 1`，
/// 只依赖一个标量，没必要为此搭一张覆盖整条序列的计算图。
pub fn bradley_terry_loss(r_chosen: &Tensor, r_rejected: &Tensor) -> Tensor {
    let diff = r_chosen.sub(r_rejected);
    let d = diff.item();
    let value = softplus(-d);
    // σ(−Δ) 在 Δ → +∞（chosen 已经遥遥领先）时趋于 0，梯度自然消失——这是对的，
    // 不是缺陷：已经分对且差距很大的样本不该继续拽着参数跑。
    let g = sigmoid(-d);
    Tensor::external_scalar_loss(value, vec![diff.clone()], move |upstream| {
        diff.accumulate_grad(&[-g], upstream);
    })
}

/// 排序准确率：`chosen` 分数**严格高于** `rejected` 的比例（并列算错）。
/// 随机瞎猜是 0.5，奖励模型训成什么样就看这个指标。
pub fn ranking_accuracy(scores: &[(f32, f32)]) -> f64 {
    assert!(!scores.is_empty(), "排序准确率需要至少一对样本");
    let hit = scores.iter().filter(|(c, r)| c > r).count();
    hit as f64 / scores.len() as f64
}

// ==================== 序列对数概率 ====================

/// 一条带掩码的序列：`ids` 是完整 token 序列，`mask[i]` 表示"预测第 i 个 token"
/// 是否计入对数概率。
///
/// `mask` 与 `ids` 必须等长；`mask[0]` 没有前文可预测，任何情况下都被忽略。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaskedSequence {
    pub ids: Vec<usize>,
    pub mask: Vec<bool>,
}

impl MaskedSequence {
    /// 整条序列都参与（自回归语言建模口径）
    pub fn full(ids: Vec<usize>) -> Self {
        let mut mask = vec![true; ids.len()];
        if let Some(first) = mask.first_mut() {
            *first = false;
        }
        MaskedSequence { ids, mask }
    }

    /// 只算回答部分：前 `prompt_len` 个 token（提问与角色标记）不计入。
    ///
    /// 对齐训练里"算 loss 的位置"必须与推理时"模型要说的话"一致——把提问也计进去，
    /// 等于在奖励"复述用户的话"，DPO/PPO 会很快学会这一招。
    pub fn answer_only(ids: Vec<usize>, prompt_len: usize) -> Self {
        assert!(
            prompt_len >= 1 && prompt_len < ids.len(),
            "prompt 长度 {prompt_len} 必须落在 1..{}",
            ids.len()
        );
        let mask = (0..ids.len()).map(|i| i >= prompt_len).collect();
        MaskedSequence { ids, mask }
    }

    pub fn len(&self) -> usize {
        self.ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// 参与对数概率的位置数
    pub fn supervised(&self) -> usize {
        self.mask.iter().filter(|&&b| b).count()
    }
}

/// 序列对数概率之和：`Σ_{i ∈ mask} log π(ids[i] | ids[<i])`。
///
/// 因果位移靠"喂 `ids[..n-1]`、目标是 `ids[1..]`"完成，与训练/评估的约定一致
/// （见 `train::eval_loss`：第 i 行 logits 预测的就是第 i+1 个 token）。
///
/// 用 [`cross_entropy_loss_masked`] 的融合实现取 log_prob，再乘回有效位置数把它的
/// "按位置平均"还原成"求和"：序列级目标必须是**和**——每一步的 log 概率都是独立
/// 贡献，取平均会让长短回答的尺度不一致，DPO 里表现为"偏爱短回答"。
pub fn sequence_logprob(model: &GPT, seq: &MaskedSequence, training: bool) -> Tensor {
    assert!(!seq.is_empty(), "空序列没有对数概率可言");
    assert!(seq.supervised() > 0, "没有任何监督位置，对数概率无从谈起");
    let t = seq.len();
    assert!(t >= 2, "至少要有两个 token 才存在可预测的目标");
    let logits = model.forward(&seq.ids[..t - 1], 1, t - 1, None, training);
    let targets = &seq.ids[1..];
    let mask: Vec<bool> = seq.mask[1..].to_vec();
    let valid = mask.iter().filter(|&&b| b).count() as f32;
    cross_entropy_loss_masked(&logits, targets, Some(&mask)).mul_scalar(-valid)
}

/// 只取值的版本（不建图），用于预计算参考模型的 logprob。
pub fn sequence_logprob_value(model: &GPT, seq: &MaskedSequence) -> f32 {
    crate::tensor::no_grad(|| sequence_logprob(model, seq, false).item())
}

/// 一批样本的参考模型对数概率**预计算**。
///
/// 参考模型全程冻结，它的 logprob 在训练过程中一个数都不会变——每步重算一遍
/// 纯属浪费（DPO 的算力有一半花在这上面）。提前算好存成 f32，训练循环里只跑策略。
pub fn precompute_reference_logprobs(reference: &GPT, samples: &[MaskedSequence]) -> Vec<f32> {
    samples.iter().map(|s| sequence_logprob_value(reference, s)).collect()
}

// ==================== DPO ====================

/// 一对偏好样本（同一 prompt 下的两条回答）
#[derive(Clone, Debug)]
pub struct PreferencePair {
    pub chosen: MaskedSequence,
    pub rejected: MaskedSequence,
}

impl PreferencePair {
    pub fn new(chosen: MaskedSequence, rejected: MaskedSequence) -> Self {
        PreferencePair { chosen, rejected }
    }
}

/// 隐式奖励：`β·(log π_θ(y) − log π_ref(y))`。
///
/// DPO 的推导结论：把"奖励"取成策略与参考模型的对数概率比，RLHF 的
/// "奖励最大化 + KL 约束"就有闭式最优解，于是**不需要单独训一个奖励模型**。
/// `β` 是"允许偏离参考模型多远"的温度：β 越小越保守（贴合参考模型），
/// 越大越激进（更听偏好数据的话）。
pub fn implicit_reward(logp_policy: &Tensor, logp_reference: f32, beta: f32) -> Tensor {
    logp_policy.add_scalar(-logp_reference).mul_scalar(beta)
}

/// DPO 成对损失：`-log σ( β·[(log π_c − log π_ref_c) − (log π_r − log π_ref_r)] )`。
///
/// 形式与 Bradley-Terry 完全一样，只是把"奖励分数"换成了隐式奖励——所以直接复用
/// [`bradley_terry_loss`]：两者都是"用 sigmoid 把差值变成概率，再取负对数似然"。
///
/// 参考模型的 logprob 是**预先算好的常数**（见 [`precompute_reference_logprobs`]），
/// 不带梯度，梯度只流向策略。
pub fn dpo_loss(
    logp_chosen: &Tensor,
    logp_rejected: &Tensor,
    ref_chosen: f32,
    ref_rejected: f32,
    beta: f32,
) -> Tensor {
    let adv_chosen = implicit_reward(logp_chosen, ref_chosen, beta);
    let adv_rejected = implicit_reward(logp_rejected, ref_rejected, beta);
    bradley_terry_loss(&adv_chosen, &adv_rejected)
}

/// 一批偏好对的 DPO 损失（对样本取平均）。
///
/// `reference_logprobs[i] = (chosen, rejected)`，与 `pairs` 一一对应。
pub fn dpo_batch_loss(
    policy: &GPT,
    pairs: &[PreferencePair],
    reference_logprobs: &[(f32, f32)],
    beta: f32,
    training: bool,
) -> Tensor {
    assert!(!pairs.is_empty(), "空批次没有损失可言");
    assert_eq!(
        pairs.len(),
        reference_logprobs.len(),
        "参考 logprob 的数量必须与偏好对一致"
    );
    let mut total: Option<Tensor> = None;
    for (pair, (ref_c, ref_r)) in pairs.iter().zip(reference_logprobs) {
        let lp_c = sequence_logprob(policy, &pair.chosen, training);
        let lp_r = sequence_logprob(policy, &pair.rejected, training);
        let loss = dpo_loss(&lp_c, &lp_r, *ref_c, *ref_r, beta);
        total = Some(match total {
            None => loss,
            Some(acc) => acc.add(&loss),
        });
    }
    total.expect("非空批次").mul_scalar(1.0 / pairs.len() as f32)
}

// ==================== GRPO 与 PPO ====================

/// 组内相对优势：`(r_i − mean(r)) / std(r)`。
///
/// GRPO 省掉价值网络的关键就在这一行：同一个 prompt 采**一组**回答，组内平均分
/// 天然就是基线——比训练一个价值网络去估计"这条 prompt 大概能拿几分"便宜得多，
/// 而且顺手把"这道题本身难不难"这个与策略无关的偏移减掉了。
///
/// 两个必须小心的点：
/// - `std` 用**总体**标准差（除以 n，不是 n−1）：组内归一化的口径。
/// - 组内分数全相同时 `std = 0`，除法会得到 `NaN`，而 `NaN` 一旦流进梯度就会把
///   整份参数污染成 `NaN`（且不会报错，只会训出一个哑巴模型）。此时组内本就
///   分不出好坏，优势该是 0，直接短路返回。
pub fn group_advantages(rewards: &[f32]) -> Vec<f32> {
    assert!(!rewards.is_empty(), "组内至少要有一个样本");
    let n = rewards.len() as f32;
    let mean = rewards.iter().sum::<f32>() / n;
    let var = rewards.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    if std < 1e-6 {
        return vec![0.0; rewards.len()];
    }
    rewards.iter().map(|r| (r - mean) / std).collect()
}

/// k3 估计的 KL：`r − log r − 1`，其中 `r = π_ref / π_θ`，`delta = log π_ref − log π_θ`。
///
/// 为什么不用最常见的 `log π_θ − log π_ref`：那个估计虽然无偏，但**方差可能极大**，
/// 而且可以为负（KL 在数学上不可能为负，负值纯属估计噪声）。k3 无偏、低方差、
/// 恒非负，代价是要多算一次 `exp`。
///
/// 两处"本该如此"的性质（单测里逐条钉住）：
/// - `π_θ == π_ref`（`delta = 0`）时恰为 0；
/// - 对 `delta` 展开是 `delta²/2 + O(delta³)`：**一阶项为 0** 意味着参考模型所在处
///   不会有"把它推走"的力（否则约束会把模型从参考点推开）；二阶系数 1/2 意味着
///   偏离越远罚得越狠。
pub fn kl_k3(delta: f32) -> f32 {
    (-delta).exp() - (-delta) - 1.0
}

/// 裁剪代理目标：返回 `(损失值, 每个位置的 d损失/d(log π))`，损失为
/// `−mean_i min(ρ_i·A_i, clip(ρ_i, 1−ε, 1+ε)·A_i)`。
///
/// - `logp_policy[i]` 是**当前策略**对第 i 条轨迹的对数概率（标量张量，建图）；
/// - `logp_old[i]` 是采样时行为策略的对数概率，**常数**；
/// - `ρ_i = exp(log π_θ − log π_old)` 是"当前策略相对采样时偏好这条轨迹多少倍"。
///
/// 裁剪的作用是"一步别走太远"：超过 `1±ε` 的部分不再给梯度，避免几个高优势样本
/// 把策略一把带偏（也顺带压住 `ρ` 爆炸）。
///
/// 用的是**序列级**重要性比，而不是 PPO 教科书里的逐 token 比：逐 token 需要
/// 每个位置各自的对数概率，而本项目的前向一次给出整段 logits，为每个 token 单独
/// 建图会让计算图规模乘上序列长度。把整条轨迹当作一个动作（序列级比）在 RL 里
/// 同样合法，代价只是方差略大。
///
/// 梯度是解析已知的，直接手写注入，不为这个"输出只有一项"的算子搭图：
/// - 未裁剪分支被选中：`d obj / d log π_i = A_i·ρ_i`（因为 `dρ/d log π = ρ`）
/// - 裁剪分支被选中：`clip` 在区间外是常数，梯度恰为 **0**
fn clipped_objective(
    logp_policy: &[Tensor],
    logp_old: &[f32],
    advantages: &[f32],
    clip_eps: f32,
) -> (f32, Vec<f32>) {
    assert!(!logp_policy.is_empty(), "空轨迹没有代理目标可言");
    assert_eq!(logp_policy.len(), logp_old.len(), "轨迹条数必须与 logp_old 一致");
    assert_eq!(logp_policy.len(), advantages.len(), "轨迹条数必须与优势一致");
    assert!(clip_eps > 0.0, "裁剪范围必须为正，实际 {clip_eps}");

    let n = logp_policy.len() as f32;
    let mut value = 0.0f32;
    let mut d_loss = Vec::with_capacity(logp_policy.len());
    for i in 0..logp_policy.len() {
        let a = advantages[i];
        let rho = (logp_policy[i].item() - logp_old[i]).exp();
        let unclipped = rho * a;
        let clipped = rho.clamp(1.0 - clip_eps, 1.0 + clip_eps) * a;
        // min 落在哪一支，梯度就从哪一支走
        let (obj, d_obj) = if unclipped <= clipped {
            (unclipped, a * rho)
        } else {
            (clipped, 0.0)
        };
        value -= obj / n;
        d_loss.push(-d_obj / n);
    }
    (value, d_loss)
}

/// 把「前向值 + 每个位置的解析梯度」组装成带反向的标量损失张量。
fn analytic_loss(nodes: &[Tensor], value: f32, d_loss: Vec<f32>) -> Tensor {
    let parents: Vec<Tensor> = nodes.to_vec();
    let nodes_owned: Vec<Tensor> = nodes.to_vec();
    Tensor::external_scalar_loss(value, parents, move |upstream| {
        for (t, &g) in nodes_owned.iter().zip(&d_loss) {
            t.accumulate_grad(&[g], upstream);
        }
    })
}

/// GRPO 损失：组内相对优势 + 裁剪代理目标。
///
/// 与 PPO 的差别不在损失形式（两者都是裁剪代理目标），而在**优势从哪来**：
/// GRPO 用同组样本的相对排名，PPO 用价值网络估计的基线。少了价值网络，显存和
/// 调参负担都小了一截，这也是 GRPO 在小规模对齐里更常见的原因。
pub fn grpo_loss(
    logp_policy: &[Tensor],
    logp_old: &[f32],
    advantages: &[f32],
    clip_eps: f32,
) -> Tensor {
    let (value, d_loss) = clipped_objective(logp_policy, logp_old, advantages, clip_eps);
    analytic_loss(logp_policy, value, d_loss)
}

/// PPO 损失：裁剪代理目标 + 参考模型 KL 惩罚。
///
/// `logp_ref[i]` 是参考模型对第 i 条轨迹的对数概率（冻结的常数，见
/// [`precompute_reference_logprobs`]）。加 KL 惩罚是因为"奖励模型给高分"与
/// "说人话"并不等价：一路最大化奖励会把策略推到奖励模型没见过的区域，
/// 那里它的打分毫无意义（reward hacking）。用 KL 把策略拴在参考模型附近，
/// `kl_coef` 就是这个拴绳的松紧。
///
/// KL 项对 `log π_θ` 的导数：`d/d(log π_θ) kl_k3(delta) = 1 − e^delta`（`delta` 的定义
/// 见 [`kl_k3`]，注意 `delta` 里含 `log π_θ` 且带负号，故是"外函数导数 × −1"）。
pub fn ppo_loss(
    logp_policy: &[Tensor],
    logp_old: &[f32],
    advantages: &[f32],
    logp_ref: &[f32],
    clip_eps: f32,
    kl_coef: f32,
) -> Tensor {
    assert_eq!(
        logp_policy.len(),
        logp_ref.len(),
        "轨迹条数必须与参考对数概率一致"
    );
    let (mut value, mut d_loss) = clipped_objective(logp_policy, logp_old, advantages, clip_eps);
    let n = logp_policy.len() as f32;
    for i in 0..logp_policy.len() {
        let delta = logp_ref[i] - logp_policy[i].item();
        value += kl_coef * kl_k3(delta) / n;
        d_loss[i] += kl_coef * (1.0 - delta.exp()) / n;
    }
    analytic_loss(logp_policy, value, d_loss)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::GPTConfig;
    use crate::optim::{AdamW, Optimizer};
    use crate::tokenizer::Tokenizer;

    fn tiny_backbone(vocab: usize, seed: u64) -> GPT {
        let cfg = GPTConfig {
            n_embd: 16,
            n_head: 2,
            n_layer: 2,
            block_size: 32,
            ..GPTConfig::tiny(vocab)
        };
        GPT::new(cfg, &mut Rng::new(seed))
    }

    /// 合成偏好对：好回答 `"ab"×m + "c"`，坏回答 `"ab"×m + "a"`。
    ///
    /// 两条**长度完全相同**，唯一差别是最后一个 token——把"长的更好"这条捷径堵死，
    /// 模型只能从内容里学。评测用训练时没出现过的长度，顺带看一眼泛化。
    fn synthetic_pairs(tok: &Tokenizer, ms: &[usize]) -> Vec<(Vec<usize>, Vec<usize>)> {
        ms.iter()
            .map(|&m| {
                let body = "ab".repeat(m);
                (
                    tok.encode(&format!("{body}c")),
                    tok.encode(&format!("{body}a")),
                )
            })
            .collect()
    }

    /// 两个分数相同时，BT 损失取到最大值 `ln 2`，且梯度指向"拉开差距"
    #[test]
    fn test_bt_loss_at_equal_scores_is_ln2() {
        let a = Tensor::param(vec![1.5], vec![1]);
        let b = Tensor::param(vec![1.5], vec![1]);
        let loss = bradley_terry_loss(&a, &b);

        assert_eq!(loss.rank(), 0, "损失必须是标量");
        assert!(
            (loss.item() - std::f32::consts::LN_2).abs() < 1e-6,
            "分数相同时损失应为 ln 2 = 0.6931，实际 {}",
            loss.item()
        );

        loss.backward();
        // dL/dΔ = σ(Δ) − 1 = −0.5（Δ = 0）；Δ = a − b，故 ∂L/∂a = −0.5、∂L/∂b = +0.5
        assert!(
            (a.grad()[0] + 0.5).abs() < 1e-5,
            "∂L/∂r_chosen 应为 −0.5，实际 {}",
            a.grad()[0]
        );
        assert!(
            (b.grad()[0] - 0.5).abs() < 1e-5,
            "∂L/∂r_rejected 应为 +0.5，实际 {}",
            b.grad()[0]
        );
    }

    /// 极端的分数差：损失渐近到 0，但数值不溢出（稳定形式的检验）
    #[test]
    fn test_bt_loss_is_numerically_stable() {
        let hi = Tensor::param(vec![500.0], vec![1]);
        let lo = Tensor::param(vec![-500.0], vec![1]);
        let loss = bradley_terry_loss(&hi, &lo);
        assert!(loss.item().is_finite(), "损失不该溢出：{}", loss.item());
        assert!(loss.item() < 1e-6, "差距极大时损失应趋于 0：{}", loss.item());

        let hi2 = Tensor::param(vec![-500.0], vec![1]);
        let lo2 = Tensor::param(vec![500.0], vec![1]);
        let loss2 = bradley_terry_loss(&hi2, &lo2);
        assert!(loss2.item().is_finite(), "损失不该溢出：{}", loss2.item());
        loss2.backward();
        assert!(
            (hi2.grad()[0] + 1.0).abs() < 1e-5,
            "完全判反时梯度应饱和到 −1，实际 {}",
            hi2.grad()[0]
        );
    }

    /// 一步训练：偏好边界（chosen 减 rejected）变大、损失下降
    #[test]
    fn test_one_step_increases_preference_margin() {
        let tok = Tokenizer::char("abc");
        let model = RewardModel::new(tiny_backbone(tok.vocab_size(), 1), &mut Rng::new(2));
        let chosen = tok.encode("ababc");
        let rejected = tok.encode("ababa");

        let margin_before = model.score_value(&chosen) - model.score_value(&rejected);
        let loss_before =
            crate::tensor::no_grad(|| model.pairwise_loss(&chosen, &rejected, false).item());

        let mut opt = AdamW::new(5e-3, model.parameters(), 0.0);
        opt.zero_grad();
        let loss = model.pairwise_loss(&chosen, &rejected, false);
        loss.backward();
        opt.step();

        let margin_after = model.score_value(&chosen) - model.score_value(&rejected);
        let loss_after =
            crate::tensor::no_grad(|| model.pairwise_loss(&chosen, &rejected, false).item());

        assert!(
            margin_after > margin_before,
            "一步之后偏好边界应变大：{margin_before:.4} → {margin_after:.4}"
        );
        assert!(
            loss_after < loss_before,
            "一步之后损失应下降：{loss_before:.4} → {loss_after:.4}"
        );
    }

    /// 训练若干步后，奖励模型在**没见过的长度**上排序准确率明显高于随机
    #[test]
    fn test_reward_model_ranking_accuracy_beats_chance() {
        let tok = Tokenizer::char("abc");
        let model = RewardModel::new(tiny_backbone(tok.vocab_size(), 3), &mut Rng::new(4));
        let train = synthetic_pairs(&tok, &[2, 3, 4]);
        let eval = synthetic_pairs(&tok, &[5, 6, 7]);

        let mut opt = AdamW::new(3e-3, model.parameters(), 0.0);
        for step in 0..150 {
            let (chosen, rejected) = &train[step % train.len()];
            opt.zero_grad();
            let loss = model.pairwise_loss(chosen, rejected, false);
            loss.backward();
            opt.step();
        }

        let scored: Vec<(f32, f32)> = eval
            .iter()
            .map(|(c, r)| (model.score_value(c), model.score_value(r)))
            .collect();
        let acc = ranking_accuracy(&scored);
        assert!(
            acc > 0.75,
            "排序准确率应明显高于随机（0.5），实际 {acc:.3}，逐对分数 {scored:?}"
        );
    }

    // ==================== DPO ====================

    /// 构造一条"只有回答计入对数概率"的序列：BOS + prompt + answer。
    ///
    /// 带上 BOS 是为了与实际推理/训练口径一致（`generate` 与 `data::encode_document`
    /// 都会补 BOS），`prompt_len` 取到 prompt 结束为止。
    fn masked_answer(tok: &Tokenizer, prompt: &str, answer: &str) -> MaskedSequence {
        let mut ids: Vec<usize> = tok.bos_id().into_iter().collect();
        ids.extend(tok.encode(prompt));
        let prompt_len = ids.len();
        ids.extend(tok.encode(answer));
        MaskedSequence::answer_only(ids, prompt_len)
    }

    /// 一批偏好对：prompt 相同，回答有优劣之分（"cab" 优于 "caa"）
    fn dpo_fixture(tok: &Tokenizer) -> Vec<PreferencePair> {
        vec![
            PreferencePair::new(
                masked_answer(tok, "ab", "cab"),
                masked_answer(tok, "ab", "caa"),
            ),
            PreferencePair::new(
                masked_answer(tok, "ba", "cbca"),
                masked_answer(tok, "ba", "cbac"),
            ),
        ]
    }

    /// 把 `pairs` 展平成 [chosen, rejected, chosen, rejected, ...] 的顺序，
    /// 供 `precompute_reference_logprobs` 按同一顺序算参考对数概率。
    fn flatten_pairs(pairs: &[PreferencePair]) -> Vec<MaskedSequence> {
        pairs
            .iter()
            .flat_map(|p| [p.chosen.clone(), p.rejected.clone()])
            .collect()
    }

    /// 把展平的参考对数概率折回成逐对的 `(chosen, rejected)`
    fn pair_up(logprobs: Vec<f32>) -> Vec<(f32, f32)> {
        logprobs.chunks(2).map(|c| (c[0], c[1])).collect()
    }

    /// 掩码语义：`answer_only` 只把 prompt 之后的位置标为监督位
    #[test]
    fn test_answer_only_masks_prompt_positions() {
        let tok = Tokenizer::char("abc");
        let seq = masked_answer(&tok, "ab", "cab");
        // BOS + "ab" 不参与，后面 3 个回答 token 参与
        assert_eq!(seq.supervised(), 3, "只有回答的 3 个 token 应参与");
        assert_eq!(
            seq.mask,
            vec![false, false, false, true, true, true],
            "前 prompt_len 个位置应为 false（含 BOS）"
        );
        assert!(!seq.mask[0], "第一个 token 没有前文可预测，永远不参与");

        // full：除首位外全部参与
        let full = MaskedSequence::full(tok.encode("abc"));
        assert_eq!(full.supervised(), full.len() - 1);
        assert!(!full.mask[0]);
    }

    /// 序列对数概率 == 手写"逐位置 log_softmax 取目标 token 再求和"
    #[test]
    fn test_sequence_logprob_matches_manual_sum() {
        let tok = Tokenizer::char("abc");
        let model = tiny_backbone(tok.vocab_size(), 12);
        let seq = masked_answer(&tok, "ab", "cab");
        let t = seq.len();

        let manual = crate::tensor::no_grad(|| {
            let logits = model.forward(&seq.ids[..t - 1], 1, t - 1, None, false);
            let lp = logits.log_softmax_last_dim();
            let d = lp.shape()[1];
            let data = lp.data_ref();
            let mut sum = 0.0f32;
            for i in 1..t {
                if seq.mask[i] {
                    sum += data[(i - 1) * d + seq.ids[i]];
                }
            }
            sum
        });

        let got = sequence_logprob_value(&model, &seq);
        assert!(
            (got - manual).abs() < 1e-4,
            "序列对数概率应与手写求和一致：{got} vs {manual}"
        );
    }

    /// 策略与参考模型完全同一个模型时，隐式奖励之差为 0 ⇒ DPO 损失 = ln 2
    #[test]
    fn test_dpo_loss_is_ln2_when_policy_equals_reference() {
        let tok = Tokenizer::char("abc");
        let model = tiny_backbone(tok.vocab_size(), 11);
        let pairs = dpo_fixture(&tok);
        let reference_logprobs = pair_up(precompute_reference_logprobs(
            &model,
            &flatten_pairs(&pairs),
        ));

        let loss = crate::tensor::no_grad(|| {
            dpo_batch_loss(&model, &pairs, &reference_logprobs, 0.1, false).item()
        });
        assert!(
            (loss - std::f32::consts::LN_2).abs() < 1e-5,
            "策略与参考模型相同时 DPO 损失应为 ln 2 = 0.6931，实际 {loss}"
        );
    }

    /// 一步 DPO：chosen 与 rejected 的对数概率边界严格变大，损失下降
    #[test]
    fn test_one_step_dpo_increases_margin_and_decreases_loss() {
        let tok = Tokenizer::char("abc");
        // 参考模型与策略模型同结构、不同初始化：Δ ≠ 0，参考项确实参与了计算
        let reference = tiny_backbone(tok.vocab_size(), 21);
        let policy = tiny_backbone(tok.vocab_size(), 22);
        let pairs = dpo_fixture(&tok);
        let reference_logprobs = pair_up(precompute_reference_logprobs(
            &reference,
            &flatten_pairs(&pairs),
        ));

        let margin = |m: &GPT| -> f32 {
            pairs
                .iter()
                .map(|p| {
                    sequence_logprob_value(m, &p.chosen)
                        - sequence_logprob_value(m, &p.rejected)
                })
                .sum()
        };
        let batch_loss = |m: &GPT| {
            crate::tensor::no_grad(|| {
                dpo_batch_loss(m, &pairs, &reference_logprobs, 0.1, false).item()
            })
        };

        let margin_before = margin(&policy);
        let loss_before = batch_loss(&policy);

        let mut opt = AdamW::new(1e-3, policy.parameters(), 0.0);
        opt.zero_grad();
        let loss = dpo_batch_loss(&policy, &pairs, &reference_logprobs, 0.1, false);
        loss.backward();
        opt.step();

        let margin_after = margin(&policy);
        let loss_after = batch_loss(&policy);

        assert!(
            margin_after > margin_before + 1e-6,
            "一步之后 chosen/rejected 对数概率边界应严格变大：{margin_before:.6} → {margin_after:.6}"
        );
        assert!(
            loss_after < loss_before,
            "一步之后 DPO 损失应下降：{loss_before:.6} → {loss_after:.6}"
        );
    }

    /// 训练循环只更新策略：参考模型全程一个参数都不动
    #[test]
    fn test_reference_model_is_frozen_during_dpo_training() {
        let tok = Tokenizer::char("abc");
        let reference = tiny_backbone(tok.vocab_size(), 41);
        let policy = tiny_backbone(tok.vocab_size(), 42);
        let pairs = dpo_fixture(&tok);
        let reference_logprobs = pair_up(precompute_reference_logprobs(
            &reference,
            &flatten_pairs(&pairs),
        ));

        let snapshot =
            |m: &GPT| -> Vec<f32> { m.parameters().iter().flat_map(|p| p.data()).collect() };
        let ref_before = snapshot(&reference);
        let policy_before = snapshot(&policy);

        // 参考模型的 logprob 是**预先算好的 f32 常数**，根本不在计算图里；
        // 优化器也只拿到策略的参数。两条路都断了，参考模型不可能被更新。
        let mut opt = AdamW::new(1e-3, policy.parameters(), 0.0);
        for _ in 0..5 {
            opt.zero_grad();
            let loss = dpo_batch_loss(&policy, &pairs, &reference_logprobs, 0.1, false);
            loss.backward();
            opt.step();
        }

        assert_eq!(ref_before, snapshot(&reference), "参考模型不应被更新");
        assert_ne!(
            policy_before,
            snapshot(&policy),
            "策略应确实被更新（否则这条测试没有验证到任何东西）"
        );
    }

    // ==================== GRPO ====================

    /// 组内相对优势：均值为 0、方差为 1
    #[test]
    fn test_group_advantages_is_standardized() {
        let adv = group_advantages(&[1.0, 2.0, 3.0, 4.0]);
        let mean = adv.iter().sum::<f32>() / adv.len() as f32;
        let var = adv.iter().map(|a| (a - mean) * (a - mean)).sum::<f32>() / adv.len() as f32;
        assert!(mean.abs() < 1e-5, "组内优势均值应为 0，实际 {mean}");
        assert!((var - 1.0).abs() < 1e-5, "组内优势方差应为 1，实际 {var}");
        // 分数越高优势越大，且严格单调
        assert!(adv[0] < adv[1] && adv[1] < adv[2] && adv[2] < adv[3], "应保持组内次序：{adv:?}");
        assert!(adv[0] < 0.0 && adv[3] > 0.0, "低于/高于组内平均的应为负/正");
    }

    /// 组内分数全相同时不能除出 NaN（std → 0 的短路分支）
    #[test]
    fn test_group_advantages_constant_rewards_are_zero_without_nan() {
        for rewards in [vec![2.5; 4], vec![0.0; 2], vec![-1.0; 8]] {
            let adv = group_advantages(&rewards);
            assert_eq!(adv.len(), rewards.len());
            assert!(
                adv.iter().all(|a| *a == 0.0),
                "分数相同则组内分不出好坏，优势应为 0：{adv:?}"
            );
            assert!(adv.iter().all(|a| a.is_finite()), "不得出现 NaN/Inf：{adv:?}");
        }
        // 单个样本的"组"同样退化为 0
        assert_eq!(group_advantages(&[1.0]), vec![0.0]);
    }

    /// 梯度方向：损失对"正优势样本"的对数概率求导为负、对"负优势样本"为正
    /// —— 沿梯度下降走，好回答更容易被生成、坏回答更难，这就是 RLHF 的全部机理。
    #[test]
    fn test_grpo_gradient_of_loss_wrt_logprobs() {
        let tok = Tokenizer::char("abc");
        let policy = tiny_backbone(tok.vocab_size(), 31);
        let good = masked_answer(&tok, "ab", "cab");
        let bad = masked_answer(&tok, "ab", "caa");

        // 行为策略 = 当前策略 ⇒ ρ 全为 1，梯度应为 −A/n
        let old = [
            sequence_logprob_value(&policy, &good),
            sequence_logprob_value(&policy, &bad),
        ];
        let lp_good = sequence_logprob(&policy, &good, false);
        let lp_bad = sequence_logprob(&policy, &bad, false);
        // 组内优势 [1, -1]：高于组内平均的为正、低于的为负
        let advantages = group_advantages(&[1.0, -1.0]);
        let loss = grpo_loss(&[lp_good.clone(), lp_bad.clone()], &old, &advantages, 0.2);
        loss.backward();

        assert!(
            (lp_good.grad()[0] + 0.5).abs() < 1e-5,
            "d损失/d log π(好回答) 应为 −A/n = −0.5，实际 {}",
            lp_good.grad()[0]
        );
        assert!(
            (lp_bad.grad()[0] - 0.5).abs() < 1e-5,
            "d损失/d log π(坏回答) 应为 −A/n = +0.5，实际 {}",
            lp_bad.grad()[0]
        );
        // 梯度确实沿计算图传到了模型参数，而不是停在注入点
        assert!(
            policy
                .parameters()
                .iter()
                .any(|p| p.grad().iter().any(|g| *g != 0.0)),
            "梯度应传到模型参数"
        );

        // ρ ≠ 1 时梯度带上 ρ 因子：log π 比采样时高了 0.1（仍在 1±ε 内，不触发裁剪）
        let shifted = Tensor::param(vec![old[0] + 0.1], vec![]);
        let loss2 = grpo_loss(&[shifted.clone()], &[old[0]], &[1.0], 0.2);
        loss2.backward();
        assert!(
            (shifted.grad()[0] + 0.1f32.exp()).abs() < 1e-4,
            "梯度应为 −A·ρ = −e^0.1，实际 {}",
            shifted.grad()[0]
        );
    }

    /// GRPO 训练使"好回答减坏回答"的对数概率边界持续变大
    ///
    /// 与上一条不同，这里不依赖单步线性近似：损失恰为 `−0.5·(log π_好 − log π_坏)`，
    /// 其梯度方向就是"把边界推大"，所以多步之后边界**必然**变大（裁剪不生效时）。
    #[test]
    fn test_grpo_training_increases_preference_margin() {
        let tok = Tokenizer::char("abc");
        let policy = tiny_backbone(tok.vocab_size(), 32);
        let good = masked_answer(&tok, "ab", "cab");
        let bad = masked_answer(&tok, "ab", "caa");
        let advantages = group_advantages(&[1.0, -1.0]);
        let margin = |m: &GPT| sequence_logprob_value(m, &good) - sequence_logprob_value(m, &bad);

        let before = margin(&policy);
        let mut opt = AdamW::new(1e-3, policy.parameters(), 0.0);
        for _ in 0..30 {
            // 在线策略：每步重新采样时策略的对数概率（即当前策略）
            let old = [
                sequence_logprob_value(&policy, &good),
                sequence_logprob_value(&policy, &bad),
            ];
            opt.zero_grad();
            let lp_good = sequence_logprob(&policy, &good, false);
            let lp_bad = sequence_logprob(&policy, &bad, false);
            let loss = grpo_loss(&[lp_good, lp_bad], &old, &advantages, 0.2);
            loss.backward();
            opt.step();
        }
        let after = margin(&policy);
        assert!(
            after > before,
            "30 步 GRPO 之后偏好边界应变大：{before:.6} → {after:.6}"
        );
    }

    // ==================== PPO ====================

    /// 裁剪分支：优势 > 0 且 ρ > 1+ε（或优势 < 0 且 ρ < 1−ε）时梯度恰为 0
    #[test]
    fn test_ppo_clip_branch_has_zero_gradient() {
        let eps = 0.2;

        // 优势 > 0 且 ρ = 2 > 1.2：目标被 clip 到 (1+ε)·A，落在区间外是常数 ⇒ 梯度 0
        let ratio_hi = Tensor::param(vec![2.0f32.ln()], vec![]);
        let loss = ppo_loss(&[ratio_hi.clone()], &[0.0], &[1.0], &[0.0], eps, 0.0);
        loss.backward();
        assert!(
            (loss.item() + 1.2).abs() < 1e-5,
            "裁剪后损失应为 −(1+ε)·A = −1.2，实际 {}",
            loss.item()
        );
        assert_eq!(ratio_hi.grad()[0], 0.0, "裁剪分支不应有梯度");

        // 优势 < 0 且 ρ = 0.5 < 1−ε：min 取到 clip 分支 ⇒ 梯度同样为 0
        let ratio_lo = Tensor::param(vec![0.5f32.ln()], vec![]);
        let loss_lo = ppo_loss(&[ratio_lo.clone()], &[0.0], &[-1.0], &[0.0], eps, 0.0);
        loss_lo.backward();
        assert!(
            (loss_lo.item() - 0.8).abs() < 1e-5,
            "裁剪后损失应为 −(1−ε)·A = 0.8，实际 {}",
            loss_lo.item()
        );
        assert_eq!(ratio_lo.grad()[0], 0.0, "裁剪分支不应有梯度");
    }

    /// 未裁剪分支：梯度按解析式 `−A·ρ/n` 注入
    #[test]
    fn test_ppo_unclipped_branch_gradient_is_analytic() {
        // ρ = 1（log π 与 log π_old 相同）：d loss/d log π = −A·1/1 = −1
        let lp = Tensor::param(vec![0.0f32], vec![]);
        let loss = ppo_loss(&[lp.clone()], &[0.0], &[1.0], &[0.0], 0.2, 0.0);
        loss.backward();
        assert!(
            (lp.grad()[0] + 1.0).abs() < 1e-5,
            "未裁剪分支梯度应为 −A·ρ = −1，实际 {}",
            lp.grad()[0]
        );

        // ρ = 1.1（仍在 1±0.2 内）：梯度应为 −A·ρ = −1.1
        let lp2 = Tensor::param(vec![1.1f32.ln()], vec![]);
        let loss2 = ppo_loss(&[lp2.clone()], &[0.0], &[1.0], &[0.0], 0.2, 0.0);
        loss2.backward();
        assert!(
            (lp2.grad()[0] + 1.1).abs() < 1e-5,
            "未裁剪分支梯度应带上 ρ 因子（−1.1），实际 {}",
            lp2.grad()[0]
        );
    }

    /// k3 的 KL：π == π_ref 时为 0，展开的一阶项为 0、二阶系数为 1/2，且恒非负
    #[test]
    fn test_kl_k3_is_zero_at_parity_and_quadratic_nearby() {
        assert_eq!(kl_k3(0.0), 0.0, "策略与参考模型一致时 KL 应为 0");

        // 为什么是"偶部/奇部"而不是直接把 kl(δ) 和 δ²/2 对拍：kl 的表达式里
        // `exp(delta) − 1` 是典型的"大数相减"，f32 在 1 附近只有约 6e-8 的绝对精度，
        // δ = 1e-3 时 kl 本身才 5e-7，有效位被吃得只剩两三位，对拍必然噪声取胜。
        // 取 δ = 0.05（kl ~ 1.25e-3，相减损失可忽略）并分离奇偶部即可稳健地定出系数。
        let h = 0.05f32;

        // 偶部 kl(δ)+kl(−δ) = δ² + O(δ⁴) ⇒ 除以 δ² 应趋于 1，即二阶系数为 1/2
        let even = (kl_k3(h) + kl_k3(-h)) / (h * h);
        assert!(
            (even - 1.0).abs() < 1e-3,
            "二阶展开系数应为 1/2（偶部/δ² → 1），实际 {even}"
        );

        // 奇部 kl(δ)−kl(−δ) = δ³/3 + O(δ⁵) ⇒ 除以 2δ 应趋于 0。
        // 若存在一阶项 c·δ，这里会收敛到常数 c 而不是 0——参考模型所在处必须没有
        // 一阶的"推走"力，否则 KL 约束会把策略从参考点推开。
        let odd = (kl_k3(h) - kl_k3(-h)) / (2.0 * h);
        assert!(odd.abs() < 1e-3, "一阶项必须为 0（奇部/2δ → 0），实际 {odd}");

        for x in [-3.0f32, -0.5, 0.5, 3.0] {
            assert!(kl_k3(x) >= 0.0, "KL 不可能为负，kl_k3({x}) = {}", kl_k3(x));
        }
    }

    /// KL 惩罚项的作用：kl_coef > 0 时，把策略往参考模型的方向拽
    #[test]
    fn test_ppo_kl_penalty_pulls_policy_toward_reference() {
        // 策略当前 log π = −1.0，参考 log π_ref = −0.5 ⇒ delta = 0.5，策略偏低
        // KL 项梯度 = 1 − e^{0.5} < 0 ⇒ 梯度下降会抬高 log π（朝参考模型靠）
        let lp = Tensor::param(vec![-1.0f32], vec![]);
        let loss = ppo_loss(&[lp.clone()], &[-1.0], &[0.0], &[-0.5], 0.2, 1.0);
        loss.backward();
        let want = 1.0 - 0.5f32.exp();
        assert!(
            (lp.grad()[0] - want).abs() < 1e-5,
            "KL 项梯度应为 1 − e^delta = {want:.6}，实际 {}",
            lp.grad()[0]
        );
        assert!(
            (loss.item() - kl_k3(0.5)).abs() < 1e-5,
            "优势为 0、ρ = 1 时损失应恰为 KL（{}），实际 {}",
            kl_k3(0.5),
            loss.item()
        );
    }
}
