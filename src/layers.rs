//! 神经网络层（第 5 课）
//!
//! 这一课把"层"抽象出来，它们是神经网络的基本积木：
//! - Linear：y = xW + b，全连接层（可挂 LoRA 适配器做低秩微调，见 [`LoraAdapter`]）
//! - LayerNorm：层归一化（第 11 课）
//! - Embedding：把 token id 变成向量（第 12 课）
//! - 激活函数：ReLU / GELU / Tanh

use crate::module::Module;
use crate::quant::{
    LayerCalib, QAxis, QBits, QMatrix, QuantMethod, QuantOpts, QuantReport, QuantWeight,
    awq_quantize, awq_quantize_search, awq_unfold_input, awq_unfold_weight, hess_quantize,
    quant_error,
};
use crate::rng::Rng;
use crate::tensor::{Shared, Tensor};

/// MLP 隐藏层放大系数（经典风格：输入维度的 4 倍）
const MLP_RATIO: usize = 4;

/// SwiGLU 的隐藏维度：`(2/3)·4d` 向上取 256 的倍数。
///
/// 抽成公开函数是为了让**参数量口径**（[`crate::scaling`] 里 `C = 6ND` 用的 N）
/// 与真实建层共用同一个公式——两处各写一遍，改一处忘另一处就会让算力估算与
/// Chinchilla 配比表整体偏掉，而且不会有任何编译错误提醒。
pub fn swiglu_hidden(d: usize) -> usize {
    ((2 * MLP_RATIO * d + 2) / 3 + 255) & !255
}

/// 线性层：y = x @ W + b
///
/// - weight: [in_features, out_features]
/// - bias:   [out_features]
///
/// 输入可以是 [B, in] 或 [B, T, in]（会自动展平处理）
///
/// `lora` 是挂在权重旁边的低秩适配器（见 [`LoraAdapter`]）：消融实验时
/// `Some` = LoRA 微调中的这一层，`None` = 普通线性层。前向变成
/// `y = x @ W + b + (α/r)·(x @ Aᵀ) @ Bᵀ`，W 被冻结、只有 A/B 在训。
///
/// `quant` 是权重的 **weight-only 量化**表示（见 [`QuantWeight`]）：`Some` 时
/// `weight` 这份 f32 数据**已经被真正释放**（换成一个 1 元素的占位张量），
/// 前向改为"把量化码反量化回 f32 再做矩阵乘"。
/// 激活始终保持 f32，所以不需要任何融合整数内核，收益全部落在**权重的存储与读取**上。
///
/// 量化层是**推理专用**的：反量化出来的权重是不参与求导的叶子张量，在求导模式下走这条路
/// 会让 `dW` 静默地恒为 0。因此 [`Linear::forward`] 在求导模式下遇到量化层会直接报错，
/// 而不是让训练白跑（详见那里的说明）。
///
/// `capture` 是**前向钩子**（校准用）：Hessian 量化要各层输入的二阶统计 `XᵀX`、AWQ 要
/// `mean|x|`，两者都只需要在校准集上**前向一次**即可采到。装上它，`forward` 就把
/// 这一层的输入存进去；平时它是 `None`，`forward` 里只多一次 `Option` 判断，
/// 没有任何其它开销。存的是 [`Tensor`] 的克隆——克隆 `Shared` 句柄（内部 `Arc`）
/// 只加一次引用计数，不会拷贝数据。
pub struct Linear {
    pub weight: Tensor,
    pub bias: Tensor,
    pub lora: Option<LoraAdapter>,
    pub quant: Option<QuantWeight>,
    pub capture: Option<Shared<Tensor>>,
}

impl Linear {
    /// 创建线性层。
    /// 权重用 Xavier 正态分布初始化（std = √(2/(in+out))，保证前向/反向方差稳定）。
    pub fn new(in_features: usize, out_features: usize, rng: &mut Rng) -> Self {
        let std = (2.0 / (in_features + out_features) as f32).sqrt();
        let w: Vec<f32> = (0..in_features * out_features)
            .map(|_| rng.randn() * std)
            .collect();
        Linear {
            weight: Tensor::param(w, vec![in_features, out_features]),
            bias: Tensor::param(vec![0.0; out_features], vec![out_features]),
            lora: None,
            quant: None,
            capture: None,
        }
    }

    /// 本层逻辑上的 `(in_features, out_features)`。
    ///
    /// 为什么需要它：量化之后 `weight` 只是一个 1 元素的**占位张量**（f32 数据已经还给
    /// 操作系统了），真实维度只存在于量化表示里（[`QMatrix::rows`] / [`QMatrix::cols`]）。
    /// 任何需要权重形状的地方（前向的 reshape、字节口径、反量化重建）都必须走这里，
    /// 直接读 `weight.shape()` 会拿到 `[1]`——那种错不会 panic，只会让输出通道数变成 1。
    pub fn dims(&self) -> (usize, usize) {
        match &self.quant {
            Some(qw) => (qw.q.rows(), qw.q.cols()),
            None => (self.weight.shape()[0], self.weight.shape()[1]),
        }
    }

    /// 给本层挂上 LoRA 适配器（只加增量，不改动也不冻结主干——冻结由调用方显式做，
    /// 见 [`crate::model::Transformer::apply_lora`]）
    pub fn attach_lora(&mut self, rank: usize, alpha: f32, rng: &mut Rng) {
        let (in_dim, out_dim) = self.dims();
        self.lora = Some(LoraAdapter::new(in_dim, out_dim, rank, alpha, rng));
    }

    /// 把适配层增量**就地**合并进主干：`W ← W + (α/r)·Aᵀ·Bᵀ`，然后丢弃适配器。
    ///
    /// 注意方向：`weight` 存的是 `[in, out]`，而 `ΔW = (α/r)·B·A` 是 `[out, in]`，
    /// 所以实际累加的是它的转置 `(α/r)·Aᵀ·Bᵀ`。
    ///
    /// 用 [`Tensor::set_data`] 就地改数据、而不是给字段换一个新张量：优化器可能已经
    /// 持有同一个 `weight` 句柄，换新张量会变成"模型用新的、优化器更新旧的"两本账。
    ///
    /// 只改数值、不动 `requires_grad`：合并是推理路径上的优化（那里本来就 no_grad），
    /// "合并后的权重是否重新可训"是另一件事，不该由这里悄悄决定。
    pub fn merge_lora(&mut self) {
        // 量化过的层必须先回到 f32 再合并：量化码是"格点上的整数 + 一个 scale"，
        // LoRA 增量是任意 f32，两者无法在量化域里精确相加（增量会被重新取整，
        // 等于把刚训出来的适配器又量化一遍）。先反量化成 f32、清空 `quant`，
        // 再走正常的 f32 加法——语义上等价于"在量化感知权重上做微调后重新展开"，
        // 这一步本身无损；此后这一层就是普通 f32 层（要再量化就重新调 quantize_weight）。
        if self.quant.is_some() {
            self.dequantize_weight();
        }
        let Some(lora) = self.lora.take() else { return };
        // 合并只在推理前做一次，不需要建图
        let delta = crate::tensor::no_grad(|| {
            lora.a
                .transpose()
                .matmul(&lora.b.transpose())
                .mul_scalar(lora.scaling())
        });
        let mut w = self.weight.data_ref().to_vec();
        assert_eq!(
            w.len(),
            delta.numel(),
            "LoRA 增量形状 {:?} 与主干权重 {:?} 不匹配",
            delta.shape(),
            self.weight.shape()
        );
        for (wi, d) in w.iter_mut().zip(delta.data_ref().iter()) {
            *wi += d;
        }
        self.weight.set_data(w);
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 支持 [B, in] 和 [B, T, in]；3D 输入在内部展平计算，输出保持 3D
        let is_3d = x.rank() == 3;
        let orig_shape = x.shape().to_vec();
        let x = match x.rank() {
            2 => x.clone(),
            3 => x.reshape(vec![x.shape()[0] * x.shape()[1], x.shape()[2]]),
            // 公开库 API：维度不合法给可读错误（带实际维度信息）
            r => panic!("Linear 输入必须为 2D 或 3D，实际是 {r}D（形状 {:?}）", x.shape()),
        };
        // 校准钩子：存的是展平后的 [tokens, in_features]——Hessian 量化的 `XᵀX` 与 AWQ 的
        // `mean|x|` 都按"行 = 一个 token"累加，用 2D 视图能省掉下游再一次展平。
        if let Some(cap) = &self.capture {
            *cap.borrow_mut() = x.clone();
        }
        // y = x @ W + b（b 是 [out]，与 [B, out] 广播相加）。
        // 权重被冻结时走 matmul_frozen：反向只算 dx、不算 dW（那段反向矩阵乘纯属浪费）。
        let base = match &self.quant {
            // 量化权重：先把整数码还原成 f32（AWQ 还要把输入逐通道除回缩放 `s`），
            // 再走普通 f32 矩阵乘。**为什么这样做仍然是划算的**：省下来的是
            // 权重的显存占用（int8 约 1/4、int4 约 1/8，外加每组一个 fp32 scale）
            // 和读权重所耗的显存带宽；而 decode 阶段每生成一个 token 都要把全部
            // 权重读一遍，本来就是 memory-bound —— 反量化是每权重一次乘加、
            // 完全被访存掩盖，代价可以忽略。
            //
            // **量化层只能用于推理**，所以这里直接拦住求导模式：反量化出来的权重是
            // 新建的叶子张量（`requires_grad = false`），AWQ 的输入缩放也不在计算图里，
            // 于是 `dW` 恒为 0——不报错、不更新，训练会"看起来在跑"却什么都没学到。
            // 与其让这种静默错误跑完几千步，不如在第一次前向就报出来：
            // 要么在 `no_grad` 下推理，要么先 `Transformer::dequantize_weights()` 烘焙回 f32 再训。
            Some(qw) => {
                let dims = self.dims();
                assert!(
                    !crate::tensor::grad_enabled(),
                    "量化层不能参与训练/反向（[in, out] = {dims:?}）：反量化路径不会产生 dW，\
                     训练会静默地毫无进展。请在 no_grad 下推理，或先 Transformer::dequantize_weights() \
                     把权重烘焙回 f32"
                );
                let w = Tensor::from_vec(self.dequant_folded(), vec![dims.0, dims.1]);
                match &qw.input_scale {
                    Some(s) => {
                        let unfolded = awq_unfold_input(&x.data_ref(), x.shape()[1], s);
                        let xu = Tensor::from_vec(unfolded, x.shape().to_vec());
                        xu.matmul_frozen(&w)
                    }
                    None => x.matmul_frozen(&w),
                }
            }
            None => {
                if self.weight.requires_grad() {
                    x.matmul(&self.weight)
                } else {
                    x.matmul_frozen(&self.weight)
                }
            }
        };
        let mut y = base.add(&self.bias);
        // LoRA 增量：Δy = (α/r)·(x @ Aᵀ) @ Bᵀ。初始 B = 0 时 Δy = 0，前向与普通线性层逐位相同
        if let Some(lora) = &self.lora {
            y = y.add(&lora.forward(&x));
        }
        if is_3d {
            // 3D 输入 [B, T, in] -> 输出 [B, T, out]（最后一维换成 out_features）
            let mut out_shape = orig_shape;
            let n = out_shape.len();
            out_shape[n - 1] = self.dims().1;
            y.reshape(out_shape)
        } else {
            y
        }
    }

    /// 量化本层权重，之后前向改走"反量化 + f32 矩阵乘"（见 [`Self::forward`]），
    /// 同时把原本的 f32 `weight` 数据**真正释放**掉（换成 1 元素占位张量）。
    ///
    /// 返回这一层的量化报告（[`QuantReport`]），`None` = 这一层被跳过
    /// （挂了 LoRA 适配器、或权重不是 2D）。名字由调用方传入：本层不知道自己在
    /// 模型里的路径名，而报告、日志、checkpoint 三处必须共用同一套名字。
    ///
    /// 已有 LoRA 适配器的层直接跳过：适配器增量是 f32 且只在 `forward` 里叠加，
    /// 主干一旦量化，这一层就成了"量化主干 + f32 增量"的混合体——既不是量化模型
    /// （显存只省了一半），也没法把增量合并回去（合并要求主干先是 f32）。
    /// 调用方的正确顺序是**先 [`Self::merge_lora`] 再量化**。
    ///
    /// `stats` 是这一层的校准统计（来自 [`crate::model::Transformer::calibrate`]）：
    /// Hessian 量化需要 `H = XᵀX`、AWQ 需要 `mean|x|`。缺失或尺寸对不上时**退回 RTN**
    /// （RTN 只需要权重本身），这样"某一层没挂上钩子"不会让整次量化失败。
    ///
    /// `opts` 带全了算法参数：分组方向、Hessian 量化的 act-order/damp/block、
    /// AWQ 的 α（`None` = 逐层在网格上搜索，见 [`awq_quantize_search`]）。
    pub fn quantize_weight(
        &mut self,
        name: &str,
        bits: QBits,
        method: QuantMethod,
        opts: &QuantOpts,
        stats: Option<&LayerCalib>,
    ) -> Option<QuantReport> {
        if self.lora.is_some() || self.weight.rank() != 2 {
            return None;
        }
        let (rows, cols) = (self.weight.shape()[0], self.weight.shape()[1]);
        if rows == 0 || cols == 0 {
            return None;
        }
        let axis: QAxis = opts.axis;
        let w = self.weight.data_ref().to_vec();
        // AWQ 实际用到的 α（其余算法恒为 None）：搜索模式下逐层可能不同，
        // 报告里必须如实写出来，否则复现时不知道该用哪个 α 重放。
        let mut used_alpha = None;
        let qw = match method {
            QuantMethod::Rtn => QuantWeight::new(
                QMatrix::quantize(&w, rows, cols, bits, axis),
                QuantMethod::Rtn,
            ),
            QuantMethod::Hess => {
                // Hessian 定义在**输入维度**上（`[rows, rows]`），行数对不上就不做补偿
                let h = stats
                    .and_then(|c| c.hessian.as_ref())
                    .filter(|h| h.dim() == rows);
                let q = match h {
                    Some(h) => hess_quantize(&w, rows, cols, bits, axis, h, &opts.hess),
                    None => QMatrix::quantize(&w, rows, cols, bits, axis),
                };
                // 算法名如实记录：**只有全矩阵 Hessian 才是真 Hessian 量化**——
                // 对角 Hessian 的 H⁻¹ 是对角阵，补偿量恒为 0，hess_quantize 内部
                // 直接走 hinv = None 的路径，码值与 RTN 逐位相同；缺统计同样退回 RTN。
                // 标成 Hess 会让"这层其实没吃到补偿"永远查不出来（见报告处的说明）。
                let real_hess = h.is_some_and(|h| h.is_full());
                QuantWeight::new(
                    q,
                    if real_hess { QuantMethod::Hess } else { QuantMethod::Rtn },
                )
            }
            QuantMethod::Awq => {
                // 激活均值按输入通道排列，长度必须等于 rows（= 输入的最后一维）
                let a = stats
                    .and_then(|c| c.act_abs_mean.as_deref())
                    .filter(|a| a.len() == rows);
                match a {
                    Some(a) => match opts.awq_alpha {
                        Some(alpha) => {
                            used_alpha = Some(alpha);
                            let (q, s) = awq_quantize(&w, rows, cols, bits, axis, a, alpha);
                            QuantWeight::with_input_scale(q, s)
                        }
                        // α 没给：逐层在网格上搜（用"折算回原坐标的等效权重误差"选优）
                        None => {
                            let (q, s, alpha) =
                                awq_quantize_search(&w, rows, cols, bits, axis, a, &[]);
                            used_alpha = Some(alpha);
                            QuantWeight::with_input_scale(q, s)
                        }
                    },
                    None => {
                        // 同上：没有激活均值就没法缩放，实际算法是 RTN
                        QuantWeight::new(
                            QMatrix::quantize(&w, rows, cols, bits, axis),
                            QuantMethod::Rtn,
                        )
                    }
                }
            }
        };

        // 误差统计落在**折算回原坐标的等效权重**上：AWQ 把 `s` 折进了权重，
        // 直接和原始 f32 比会把 `s` 的系统性缩放混进误差里（α 越小越"看着准"），
        // 逐层报告就失去了可比性。折算后比的是"最终作用在同一个输入上的两个权重"。
        let equiv = match &qw.input_scale {
            Some(s) => awq_unfold_weight(&qw.q.dequantize(), rows, cols, s),
            None => qw.q.dequantize(),
        };
        let e = quant_error(&w, &equiv);
        // 算法名取 `qw.method`（而不是请求的 `method`）：退回过 RTN 的层必须能一眼看出来，
        // 否则报告会把"这层其实没吃到补偿"粉饰成一次成功的 Hessian 量化/AWQ。
        let report = QuantReport {
            name: name.to_string(),
            method: qw.method,
            bits,
            f32_bytes: w.len() * 4,
            quant_bytes: qw.byte_len(),
            max_abs_err: e.max_abs,
            rel_err: e.relative,
            alpha: used_alpha,
        };

        self.quant = Some(qw);
        // **真正释放** f32 权重：换成一个 1 元素（4 字节）的占位张量。
        // 为什么不是"把 weight 设为 None"：`weight` 是 `Tensor` 而非 `Option`，
        // 全仓库有几十处直接读它；改成 `Option` 会把改动扩散到 attention / model /
        // checkpoint 的每一个调用点，收益只是省下 4 个字节。
        // 为什么必须换掉数据而不是留着：留着就等于"省了显存"是假的——
        // 量化最大的收益恰恰是这部分 f32 不再占用内存/带宽。
        // `requires_grad` 原样保留，这样 [`Self::dequantize_weight`] 能把它恢复回去。
        let rg = self.weight.requires_grad();
        self.weight = Tensor::from_vec(vec![0.0], vec![1]);
        self.weight.set_requires_grad(rg);
        Some(report)
    }

    /// `(原始 f32 字节数, 当前实际占用字节数)`。
    /// 未量化时两者相等——用它就能看清"量化到底省了多少"。
    /// 量化后 `orig` 仍按**逻辑维度**算（占位张量只有 4 字节，但报告里要给出
    /// "不量化会占多少"）。
    pub fn weight_bytes(&self) -> (usize, usize) {
        let (rows, cols) = self.dims();
        let orig = rows * cols * 4;
        match &self.quant {
            Some(qw) => (orig, qw.byte_len()),
            None => (orig, orig),
        }
    }

    /// 本层权重是否已被量化（GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_quant(&self) -> bool {
        self.quant.is_some()
    }

    /// 还原量化权重为 f32。AWQ 时返回的是**折叠后**的权重 `W·s`
    /// （必须与 [`Self::forward`] 里"输入除以 `s`"配对，两边都用同一份 `s` 才等价）。
    fn dequant_folded(&self) -> Vec<f32> {
        match &self.quant {
            Some(qw) => qw.q.dequantize(),
            None => self.weight.data_ref().to_vec(),
        }
    }

    /// 把量化状态**烘焙**回 f32 权重并清空 `quant`：此后这一层就是普通 f32 层。
    /// AWQ 折叠进权重的那份输入缩放也一并除掉，恢复成"未经缩放的原始权重"——
    /// 否则模型会带着一份只对旧 `input_scale` 成立的权重继续活下去。
    ///
    /// 这里必须**换一个新张量**（而不是 [`Tensor::set_data`] 就地改）：量化后的
    /// `weight` 只是 1 元素的占位张量，元素数根本对不上，`set_data` 会直接断言失败。
    /// `requires_grad` 从占位张量上读回来并原样恢复，所以"量化 → 反量化"这个来回
    /// 对模型状态是无损的（可以接着训练、存档、合并 LoRA）。
    pub fn dequantize_weight(&mut self) {
        let Some(qw) = self.quant.take() else { return };
        let (rows, cols) = (qw.q.rows(), qw.q.cols());
        let mut w = qw.q.dequantize();
        if let Some(s) = &qw.input_scale {
            for r in 0..rows {
                for c in 0..cols {
                    w[r * cols + c] /= s[r];
                }
            }
        }
        let rg = self.weight.requires_grad();
        self.weight = Tensor::param(w, vec![rows, cols]);
        self.weight.set_requires_grad(rg);
    }

    /// 带名字的参数（checkpoint 保存/恢复用）：`{prefix}.weight` / `{prefix}.bias`
    /// （挂了 LoRA 时还有 `{prefix}.lora_a` / `{prefix}.lora_b`）
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        let mut ps = vec![
            (format!("{prefix}.weight"), self.weight.clone()),
            (format!("{prefix}.bias"), self.bias.clone()),
        ];
        if let Some(lora) = &self.lora {
            ps.extend(lora.named_parameters(prefix));
        }
        ps
    }
}

impl Module for Linear {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = vec![self.weight.clone(), self.bias.clone()];
        if let Some(lora) = &self.lora {
            ps.extend(lora.parameters());
        }
        ps
    }
}

/// 层归一化（LayerNorm）：
/// 对最后一维做归一化（均值 0、方差 1），再缩放平移。
///
/// y = (x - μ) / √(σ² + ε) * γ + β
///
/// 为什么需要它？（第 11 课详解）
/// - 稳定训练：避免层输出数值范围过大导致梯度爆炸/消失
/// - 加快收敛：每层输入分布一致
pub struct LayerNorm {
    pub gamma: Tensor, // [d] 可学习缩放
    pub beta: Tensor,  // [d] 可学习平移
    pub eps: f32,
}

impl LayerNorm {
    pub fn new(d: usize, eps: f32) -> Self {
        LayerNorm {
            gamma: Tensor::param(vec![1.0; d], vec![d]),
            beta: Tensor::param(vec![0.0; d], vec![d]),
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        // 融合实现（一个算子完成 11 个基础算子的前向+反向），见 Tensor::layernorm
        x.layernorm(&self.gamma, &self.beta, self.eps)
    }

    /// 带名字的参数：`{prefix}.gamma` / `{prefix}.beta`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}.gamma"), self.gamma.clone()),
            (format!("{prefix}.beta"), self.beta.clone()),
        ]
    }
}

impl Module for LayerNorm {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone(), self.beta.clone()]
    }
}

/// RMSNorm（Root Mean Square Layer Normalization）：
/// y = x / √(mean(x²) + ε) * γ
///
/// 比 LayerNorm 更高效（不减均值、无 β），现代 LLM 标配：
/// - LLaMA / LLaMA 2 / LLaMA 3
/// - Mistral / Mixtral
/// - Qwen / Qwen2
/// - Gemma
pub struct RMSNorm {
    pub gamma: Tensor, // [d] 可学习缩放
    pub eps: f32,
}

impl RMSNorm {
    pub fn new(d: usize, eps: f32) -> Self {
        RMSNorm {
            gamma: Tensor::param(vec![1.0; d], vec![d]),
            eps,
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.rmsnorm(&self.gamma, self.eps)
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{prefix}.gamma"), self.gamma.clone())]
    }
}

impl Module for RMSNorm {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.gamma.clone()]
    }
}

/// 统一的归一化层枚举：支持 LayerNorm 和 RMSNorm 两种选择
pub enum NormLayer {
    LN(LayerNorm),
    RMS(RMSNorm),
}

impl NormLayer {
    pub fn new(d: usize, eps: f32, use_rmsnorm: bool) -> Self {
        if use_rmsnorm {
            NormLayer::RMS(RMSNorm::new(d, eps))
        } else {
            NormLayer::LN(LayerNorm::new(d, eps))
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            NormLayer::LN(ln) => ln.forward(x),
            NormLayer::RMS(rms) => rms.forward(x),
        }
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        match self {
            NormLayer::LN(ln) => ln.named_parameters(prefix),
            NormLayer::RMS(rms) => rms.named_parameters(prefix),
        }
    }

    /// GPU 常驻显存路径要的归一化参数：`(γ, β, ε, 是否 RMSNorm)`。
    ///
    /// RMSNorm 没有 β，这里返回 `None`。常驻路径的内核布局是固定的三个张量槽位
    /// （前向 `x`/`γ`/`β`，反向 `dγ`/`dβ`），所以两个 β 槽位仍必须有一块合法显存顶上，
    /// 由调用方给零张量；内核在 RMS 模式下一律把该槽位的值丢弃
    /// （前向的仿射项取 0，反向的 `m1` 取 0），写回的 `dβ` 也无人接收。
    ///
    /// 唯一调用点都在 `model.rs` 的 `#[cfg(feature = "gpu")]` 函数里，
    /// 所以不带 gpu feature 编译时它是"死代码"——这不是真死代码，别删。
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn norm_params(&self) -> (&Tensor, Option<&Tensor>, f32, bool) {
        match self {
            NormLayer::LN(ln) => (&ln.gamma, Some(&ln.beta), ln.eps, false),
            NormLayer::RMS(rms) => (&rms.gamma, None, rms.eps, true),
        }
    }
}

impl Module for NormLayer {
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            NormLayer::LN(ln) => ln.parameters(),
            NormLayer::RMS(rms) => rms.parameters(),
        }
    }
}

/// SwiGLU MLP 层（LLaMA 风格）：
///
/// ```text
/// hidden = SiLU(x @ W_gate) ⊙ (x @ W_up)
/// out = hidden @ W_down
/// ```
///
/// 与经典风格 MLP（GELU(x @ W1) @ W2）的区别：
/// - 用 SiLU(x) ⊙ gate 替代 GELU（表达力更强）
/// - 多一个 W_gate 矩阵（门控分支）
/// - hidden_dim 通常设为 (2/3) * 4d（保持参数量相近）
///
/// LLaMA 的 hidden_dim = 11008（2/3 * 4 * 4096 ≈ 10922，取 256 的倍数）
pub struct SwiGLUMLP {
    pub w_gate: Linear, // [D, hidden] 门控分支
    pub w_up: Linear,   // [D, hidden] 上投影
    pub w_down: Linear, // [hidden, D] 下投影
}

impl SwiGLUMLP {
    pub fn new(d: usize, hidden: usize, rng: &mut Rng) -> Self {
        SwiGLUMLP {
            w_gate: Linear::new(d, hidden, rng),
            w_up: Linear::new(d, hidden, rng),
            w_down: Linear::new(hidden, d, rng),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.w_gate.forward(x);
        let up = self.w_up.forward(x);
        let hidden = gate.swiglu(&up);
        self.w_down.forward(&hidden)
    }

    /// 三个投影 + 各自的参数名前缀。[`MLPEnum::named_parameters`] 由它派生，名字因此严格一致
    /// （量化要按同一套名字去取校准统计，名字对不上就会静默退回 RTN）。
    pub fn named_linears(&self, prefix: &str) -> Vec<(String, &Linear)> {
        vec![
            (format!("{prefix}.w_gate"), &self.w_gate),
            (format!("{prefix}.w_up"), &self.w_up),
            (format!("{prefix}.w_down"), &self.w_down),
        ]
    }

    pub fn named_linears_mut(&mut self, prefix: &str) -> Vec<(String, &mut Linear)> {
        vec![
            (format!("{prefix}.w_gate"), &mut self.w_gate),
            (format!("{prefix}.w_up"), &mut self.w_up),
            (format!("{prefix}.w_down"), &mut self.w_down),
        ]
    }
}

impl Module for SwiGLUMLP {
    fn parameters(&self) -> Vec<Tensor> {
        let mut ps = self.w_gate.parameters();
        ps.extend(self.w_up.parameters());
        ps.extend(self.w_down.parameters());
        ps
    }
}

/// 统一的 MLP 层枚举：支持经典风格 GELU MLP 和 LLaMA 风格 SwiGLU MLP
pub enum MLPEnum {
    GELU {
        linear1: Linear,
        linear2: Linear,
    },
    SwiGLU(SwiGLUMLP),
}

impl MLPEnum {
    pub fn new_gelu(d: usize, rng: &mut Rng) -> Self {
        MLPEnum::GELU {
            linear1: Linear::new(d, MLP_RATIO * d, rng),
            linear2: Linear::new(MLP_RATIO * d, d, rng),
        }
    }

    pub fn new_swiglu(d: usize, rng: &mut Rng) -> Self {
        // SwiGLU hidden_dim = (2/3) * 4d ≈ 2.67d，取 256 的倍数（公式见 [`swiglu_hidden`]）
        let hidden = swiglu_hidden(d);
        MLPEnum::SwiGLU(SwiGLUMLP::new(d, hidden, rng))
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        match self {
            MLPEnum::GELU { linear1, linear2 } => linear2.forward(&gelu(&linear1.forward(x))),
            MLPEnum::SwiGLU(swiglu) => swiglu.forward(x),
        }
    }

    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        self.named_linears(prefix)
            .into_iter()
            .flat_map(|(p, lin)| lin.named_parameters(&p))
            .collect()
    }

    /// 本 MLP 的全部投影 + 各自的参数名前缀。
    ///
    /// `named_parameters` 由它派生，名字因此**只有一处定义**：量化要按同一套名字
    /// 去取校准统计（[`crate::quant::CalibStats`]），名字对不上不会报错，
    /// 只会让 Hessian 量化 / AWQ 静默退回 RTN，属于最难查的那类偏差。
    pub fn named_linears(&self, prefix: &str) -> Vec<(String, &Linear)> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => vec![
                (format!("{prefix}.mlp_linear1"), linear1),
                (format!("{prefix}.mlp_linear2"), linear2),
            ],
            MLPEnum::SwiGLU(s) => s.named_linears(prefix),
        }
    }

    pub fn named_linears_mut(&mut self, prefix: &str) -> Vec<(String, &mut Linear)> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => vec![
                (format!("{prefix}.mlp_linear1"), linear1),
                (format!("{prefix}.mlp_linear2"), linear2),
            ],
            MLPEnum::SwiGLU(s) => s.named_linears_mut(prefix),
        }
    }

    /// GELU 分支的 `(W₁, b₁, W₂, b₂)`；SwiGLU 分支返回 None
    /// （GPU 常驻显存路径目前只实现了经典风格的 GELU MLP）
    ///
    /// 同 [`NormLayer::norm_params`]：调用点只在 gpu feature 下，不带时是"死代码"，别删。
    #[cfg_attr(not(feature = "gpu"), allow(dead_code))]
    pub fn gelu_weights(&self) -> Option<(&Tensor, &Tensor, &Tensor, &Tensor)> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => Some((
                &linear1.weight,
                &linear1.bias,
                &linear2.weight,
                &linear2.bias,
            )),
            MLPEnum::SwiGLU(_) => None,
        }
    }

    /// 本 MLP 的全部投影（GELU 两个 / SwiGLU 三个）
    fn linears(&self) -> Vec<&Linear> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => vec![linear1, linear2],
            MLPEnum::SwiGLU(s) => vec![&s.w_gate, &s.w_up, &s.w_down],
        }
    }

    fn linears_mut(&mut self) -> Vec<&mut Linear> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => vec![linear1, linear2],
            MLPEnum::SwiGLU(s) => vec![&mut s.w_gate, &mut s.w_up, &mut s.w_down],
        }
    }

    /// 给本 MLP 的各投影挂上适配器
    /// （是否该挂由调用方按 `targets.mlp` 决定，见 [`crate::model::Transformer::apply_lora`]）
    pub fn apply_lora(&mut self, lora: &crate::config::LoRAConfig, rng: &mut Rng) {
        for lin in self.linears_mut() {
            lin.attach_lora(lora.rank, lora.alpha, rng);
        }
    }

    /// 把各投影的适配器合并进主干（推理用，见 [`Linear::merge_lora`]）
    pub fn merge_lora(&mut self) {
        for lin in self.linears_mut() {
            lin.merge_lora();
        }
    }

    /// 本 MLP 是否挂了适配器（GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_lora(&self) -> bool {
        self.linears().iter().any(|l| l.lora.is_some())
    }

    /// 本 MLP 是否有投影被量化（GPU 常驻显存快路据此让路，见 [`crate::model`]）
    pub fn has_quant(&self) -> bool {
        self.linears().iter().any(|l| l.has_quant())
    }

    /// 把各投影的量化状态烘焙回 f32（checkpoint 里写的始终是 f32 权重）
    pub fn dequantize_weights(&mut self) {
        for lin in self.linears_mut() {
            lin.dequantize_weight();
        }
    }
}

impl Module for MLPEnum {
    fn parameters(&self) -> Vec<Tensor> {
        match self {
            MLPEnum::GELU { linear1, linear2 } => {
                let mut ps = linear1.parameters();
                ps.extend(linear2.parameters());
                ps
            }
            MLPEnum::SwiGLU(s) => s.parameters(),
        }
    }
}

/// LoRA 适配器（Low-Rank Adaptation，低秩适配）：挂在冻结权重**旁边**的一对低秩矩阵。
///
/// ```text
/// y = x @ W + b + (α/r)·(x @ Aᵀ) @ Bᵀ
///                    └──────── Δy = x @ ΔWᵀ ────────┘
/// 其中 ΔW = (α/r)·B·A ∈ R^{out×in}
/// ```
///
/// - `W` ∈ R^{in×out}：原始权重，冻结后 requires_grad = false，全程不动
///   （冻结动作在 [`crate::model::Transformer::apply_lora`] 里统一做）
/// - `a` ∈ R^{r×in}：下投影，`N(0, 1/√r)` 初始化
/// - `b` ∈ R^{out×r}：上投影，**全零**初始化
/// - `r ≪ min(in, out)`：秩（通常 4-64），`α` 是缩放因子
///
/// 可训练参数量：`r×(in+out)`，远小于 `in×out`。例：in=out=4096、r=16 时
/// 131072 个参数，是原始的 0.78%。
///
/// 为什么 B 初始化成全零：这样训练开始时 ΔW = B·A = 0，模型行为与预训练**逐位相同**，
/// 不会因为插入适配层而先抖一下。A 若也全零就永远学不动了——此时 `∂L/∂B ∝ x·Aᵀ = 0`
/// 且 `∂L/∂A ∝ ∂L/∂y·B = 0`，两条梯度同时为零；A 取随机小值就是为了让 B 先有梯度。
///
/// 该结构体**不持有**主干权重：增量与主干是两条独立支路，在 [`Linear::forward`] 里相加。
/// 推理时可以把 `ΔW` 预先进主干（`W' = W + (α/r)·B·A`，见 [`Linear::merge_lora`]），
/// 此后零额外开销；但那是**不可逆**的——合并后存档再也分不出主干与适配层，
/// 也就无法再链式续训（见 [`crate::model::Transformer::resume_lora`]）。
/// 训练与"想保留适配层"的推理都走两条支路，代价只是每层多两次小矩阵乘。
pub struct LoraAdapter {
    /// 低秩下投影 [rank, in]
    pub a: Tensor,
    /// 低秩上投影 [out, rank]
    pub b: Tensor,
    pub rank: usize,
    /// 缩放因子 α（实际乘的是 α/rank）
    pub alpha: f32,
}

impl LoraAdapter {
    /// - `in_dim`：输入维度（主干 weight 的第 0 维）
    /// - `out_dim`：输出维度（主干 weight 的第 1 维）
    /// - `rank`：低秩维度 r（越小越省参数，越大表达力越强）
    /// - `alpha`：缩放因子 α（通常 = rank）
    pub fn new(in_dim: usize, out_dim: usize, rank: usize, alpha: f32, rng: &mut Rng) -> Self {
        assert!(rank >= 1, "LoRA rank 必须 >= 1");
        let a_scale = 1.0 / (rank as f32).sqrt();
        let a = Tensor::param(
            (0..rank * in_dim).map(|_| rng.randn() * a_scale).collect(),
            vec![rank, in_dim],
        );
        let b = Tensor::param(vec![0.0; out_dim * rank], vec![out_dim, rank]);
        LoraAdapter { a, b, rank, alpha }
    }

    /// 缩放因子 α/r
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// 前向增量 Δy = (α/r)·(x @ Aᵀ) @ Bᵀ
    ///
    /// `x` 必须是**已展平**的 2D 张量 [N, in]（[`Linear::forward`] 负责展平，
    /// 保持两条支路在同一个二维视图上算，避免各自 reshape 一次）。
    pub fn forward(&self, x: &Tensor) -> Tensor {
        x.matmul(&self.a.transpose())
            .matmul(&self.b.transpose())
            .mul_scalar(self.scaling())
    }

    /// 带名字的参数：`{prefix}.lora_a` / `{prefix}.lora_b`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![
            (format!("{prefix}.lora_a"), self.a.clone()),
            (format!("{prefix}.lora_b"), self.b.clone()),
        ]
    }
}

impl Module for LoraAdapter {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.a.clone(), self.b.clone()]
    }
}

/// 嵌入层：把 token id 查表变成向量。
/// table: [vocab_size, d_model]
pub struct Embedding {
    pub table: Tensor,
}

impl Embedding {
    /// 创建嵌入表，用正态分布 N(0, 0.02) 初始化
    pub fn new(vocab_size: usize, d_model: usize, rng: &mut Rng) -> Self {
        let std = 0.02;
        let data: Vec<f32> = (0..vocab_size * d_model)
            .map(|_| rng.randn() * std)
            .collect();
        Embedding {
            table: Tensor::param(data, vec![vocab_size, d_model]),
        }
    }

    /// 前向：ids [N] -> out [N, d_model]（每一行取表里的对应向量）
    pub fn forward(&self, ids: &[usize]) -> Tensor {
        self.table.gather_rows(ids)
    }

    /// 带名字的参数：`{prefix}.table`
    pub fn named_parameters(&self, prefix: &str) -> Vec<(String, Tensor)> {
        vec![(format!("{prefix}.table"), self.table.clone())]
    }
}

impl Module for Embedding {
    fn parameters(&self) -> Vec<Tensor> {
        vec![self.table.clone()]
    }
}

// ---------- 激活函数（第 5 课） ----------

/// GELU：现代 LLM的默认激活，用 tanh 近似，比 ReLU 更平滑
pub fn gelu(x: &Tensor) -> Tensor {
    x.gelu()
}

/// Tanh：S 型，输出 (-1, 1)
pub fn tanh(x: &Tensor) -> Tensor {
    x.tanh()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::{Module, zero_grad_all};
    use crate::optim::{AdamW, Optimizer};
    use crate::quant::{AWQ_ALPHA, HessOpts, HessianKind, QuantOpts};

    /// 手工搭一个可控的 Linear：W = [[1,2],[3,4]]、b = 0（in = out = 2）
    fn hand_linear() -> Linear {
        Linear {
            weight: Tensor::param(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]),
            bias: Tensor::param(vec![0.0, 0.0], vec![2]),
            lora: None,
            quant: None,
            capture: None,
        }
    }

    /// 手工搭一个可控的适配器：r = 1、A = [[1,1]]、B = [[1],[1]]、α = 1
    /// ⇒ ΔW = (α/r)·B·A = [[1,1],[1,1]]
    fn hand_lora() -> LoraAdapter {
        LoraAdapter {
            a: Tensor::param(vec![1.0, 1.0], vec![1, 2]),
            b: Tensor::param(vec![1.0, 1.0], vec![2, 1]),
            rank: 1,
            alpha: 1.0,
        }
    }

    /// 冻结整个线性层（等价于 [`crate::model::Transformer::apply_lora`] 对全部主干参数做的事：
    /// 把共享的 requires_grad 标志置 false）
    fn freeze(lin: &Linear) {
        lin.weight.set_requires_grad(false);
        lin.bias.set_requires_grad(false);
    }

    /// 前向 = 主干 + 低秩增量，且**恰好**等于公式 (α/r)·(x·Aᵀ)·Bᵀ（含 α/r 缩放）
    #[test]
    fn test_lora_forward_matches_formula() {
        let lin = hand_linear();
        let mut lin_lora = hand_linear();
        lin_lora.lora = Some(hand_lora());

        let x = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        // 主干：x @ W = [1, 2]；增量：x @ ΔW = [1, 1] ⇒ [2, 3]
        assert_eq!(lin.forward(&x).data(), vec![1.0, 2.0]);
        assert_eq!(lin_lora.forward(&x).data(), vec![2.0, 3.0]);

        // α 只以 α/r 的形式进入：把 α 改成 2r，增量应翻倍
        let mut doubled = hand_lora();
        doubled.alpha = 2.0;
        lin_lora.lora = Some(doubled);
        assert_eq!(lin_lora.forward(&x).data(), vec![3.0, 4.0]);
    }

    /// B 全零初始化 ⇒ ΔW = 0 ⇒ 挂上适配器后前向与纯 Linear **逐位**相同。
    /// 这是"训练从预训练状态出发、不会先抖一下"的保证。
    #[test]
    fn test_lora_zero_init_keeps_forward_identical() {
        let mut rng = Rng::new(1);
        let mut lin = Linear::new(8, 5, &mut rng);
        let x = Tensor::from_vec((0..16).map(|i| (i as f32 * 0.3).sin()).collect(), vec![2, 8]);
        let before = lin.forward(&x).data();
        lin.attach_lora(4, 4.0, &mut rng);
        assert_eq!(lin.forward(&x).data(), before, "B=0 时 LoRA 增量必须为 0");
    }

    /// 3D 输入下增量支路也要正确叠加（[B,T,in] 先展平再算）
    #[test]
    fn test_lora_forward_3d() {
        let mut lin = hand_linear();
        lin.lora = Some(hand_lora());
        // [1,3,2]：三行分别是 [1,0]、[0,1]、[1,1]
        let x = Tensor::from_vec(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], vec![1, 3, 2]);
        let y = lin.forward(&x);
        assert_eq!(y.shape(), &[1, 3, 2]);
        // 主干 [1,2] [3,4] [4,6] + 增量 [1,1] [1,1] [2,2]
        assert_eq!(y.data(), vec![2.0, 3.0, 4.0, 5.0, 6.0, 8.0]);
    }

    /// 冻结的主干：反向不攒梯度、优化器也不动它；适配层则正常攒梯度。
    /// 这两件事一个由 `matmul_frozen`（不建 dW）、一个由优化器跳过保证，缺一不可。
    #[test]
    fn test_frozen_weight_gets_no_grad_while_lora_does() {
        // 先单看冻结主干这一支：dx 必须照常回传（否则上一层的适配层就断了），
        // 但 dW 完全不建。loss = sum(y) ⇒ 上游梯度全 1，dx = 1 @ Wᵀ = [3,7]
        let plain = hand_linear();
        freeze(&plain);
        let x0 = Tensor::param(vec![1.0, 0.0], vec![1, 2]);
        plain.forward(&x0).sum().backward();
        assert_eq!(x0.grad(), vec![3.0, 7.0]);
        assert_eq!(plain.weight.grad(), vec![0.0; 4], "冻结权重不该有 dW");

        // 再看主干 + 适配器一起：主干那一份依旧一分不攒，适配层正常攒
        let mut lin = hand_linear();
        lin.lora = Some(hand_lora());
        freeze(&lin);

        let x = Tensor::param(vec![1.0, 0.0], vec![1, 2]);
        lin.forward(&x).sum().backward();

        assert_eq!(lin.weight.grad(), vec![0.0; 4], "冻结权重不该有 dW");
        // bias 的梯度槽仍会被 `add` 的广播反向写入（算子只看形状，不看 requires_grad），
        // 但没人会用它：优化器跳过冻结参数、裁剪只看 trainable。真正生效的保证在下面那一步。
        // da = xᵀ·dt1 = [2, 0]（x = [1,0]，第二列乘 0 所以是 0）；db = t1ᵀ·g = [1, 1]
        assert_eq!(lin.lora.as_ref().unwrap().a.grad(), vec![2.0, 0.0]);
        assert_eq!(lin.lora.as_ref().unwrap().b.grad(), vec![1.0, 1.0]);

        // 优化器：梯度全灌 1、权重衰减给到 0.5，冻结的主干也必须一动不动
        zero_grad_all(&lin);
        let mut opt = AdamW::new(0.1, lin.parameters(), 0.5);
        for p in opt.params() {
            p.grad.borrow_mut().iter_mut().for_each(|g| *g = 1.0);
        }
        opt.step();
        assert_eq!(lin.weight.data(), vec![1.0, 2.0, 3.0, 4.0], "冻结权重被更新了");
        assert_eq!(lin.bias.data(), vec![0.0, 0.0], "冻结 bias 被更新了");
        // 适配层：初始化全零的 b 经过一步更新后不再是零
        assert!(lin.lora.as_ref().unwrap().b.data().iter().any(|v| *v != 0.0));
    }

    /// 合并：`W ← W + (α/r)·Aᵀ·Bᵀ`，且必须**就地**改 `weight` 的数据。
    ///
    /// 就地这一点值得单独测：换一个新张量也能算对数值，但优化器（或任何已持有句柄的
    /// 地方）会继续指向旧数据，于是变成"模型用新的、别处更新旧的"两本账。
    #[test]
    fn test_merge_lora_into_weight_is_in_place() {
        let mut lin = hand_linear();
        lin.lora = Some(hand_lora()); // ΔW = [[1,1],[1,1]]
        let weight_handle = lin.weight.clone(); // 旧句柄：合并后应能从它读到新数值

        let x = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let before = lin.forward(&x).data(); // 主干 [1,2] + 增量 [1,1] = [2,3]

        lin.merge_lora();

        // W = [[1,2],[3,4]]，ΔW 是 [out,in] = [[1,1],[1,1]]，加的是它的转置 ⇒ [[2,3],[4,5]]
        assert_eq!(weight_handle.data(), vec![2.0, 3.0, 4.0, 5.0]);
        assert!(lin.lora.is_none(), "合并后不能留着适配器，否则增量会被算两次");
        assert_eq!(lin.forward(&x).data(), before, "合并前后前向应一致");

        // 没挂适配器的层调它应当是纯空操作
        let mut plain = hand_linear();
        plain.merge_lora();
        assert_eq!(plain.weight.data(), vec![1.0, 2.0, 3.0, 4.0]);
    }

    /// 量化后的前向必须**恰好**等于"拿反量化权重做 f32 矩阵乘"——
    /// 这正是量化推理的全部语义：整数码只负责把权重存小，计算仍在 f32 里。
    /// 同时验证校准钩子存下的是展平后的输入。
    ///
    /// 注意所有前向都在 `no_grad` 里跑：量化层是推理专用的（见 [`Linear::forward`]）。
    #[test]
    fn quantized_forward_matches_dequantized_reference() {
        let mut rng = Rng::new(9);
        let (m, k, n) = (3, 8, 5);
        let mut lin = Linear::new(k, n, &mut rng);
        lin.bias.set_data(vec![0.1, -0.2, 0.3, 0.0, 0.5]);
        let xd: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.37).sin()).collect();
        // 用 3D 输入顺带验证：展平后算、输出仍是 3D，钩子拿到的是 [B·T, in]
        let x3 = Tensor::from_vec(xd.clone(), vec![1, m, k]);
        let cap = Shared::new(Tensor::from_vec(vec![0.0], vec![1]));
        lin.capture = Some(cap.clone());

        let before = crate::tensor::no_grad(|| lin.forward(&x3).data());
        {
            let got = cap.borrow();
            assert_eq!(got.shape(), &[m, k], "钩子应拿到展平后的 [tokens, in]");
            assert_eq!(got.data(), xd, "钩子必须原样存下这一层的输入");
        }

        let report = lin
            .quantize_weight(
                "test.weight",
                QBits::Int8,
                QuantMethod::Rtn,
                &QuantOpts::default(),
                None,
            )
            .expect("普通 Linear 必须能量化");
        assert!(lin.has_quant());
        assert_eq!(report.name, "test.weight");
        assert_eq!(report.method, QuantMethod::Rtn);
        assert_eq!(report.f32_bytes, k * n * 4);
        assert!(report.rel_err > 0.0 && report.rel_err.is_finite());
        let (orig_b, cur_b) = lin.weight_bytes();
        assert_eq!(orig_b, k * n * 4, "原始口径 = f32 元素数 × 4");
        assert!(cur_b < orig_b, "量化后必须更小：{cur_b} vs {orig_b}");

        // 参考值：x @ dequantize(W) + b（手工算，独立于 forward 的实现）
        let w = lin.dequant_folded();
        let bias = lin.bias.data();
        let mut expect = vec![0.0f32; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut s = bias[j];
                for t in 0..k {
                    s += xd[i * k + t] * w[t * n + j];
                }
                expect[i * n + j] = s;
            }
        }
        let after = crate::tensor::no_grad(|| lin.forward(&x3).data());
        for (i, (a, b)) in after.iter().zip(&expect).enumerate() {
            assert!((a - b).abs() < 1e-4, "第 {i} 个输出：{a} vs 反量化参考 {b}");
        }
        // 量化有损但只是"轻微扰动"：int8 的误差上限是 max|x| 的 1/254 量级
        let drift = after
            .iter()
            .zip(&before)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let mag = before.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(drift < mag * 0.05, "int8 量化不该造成这么大的偏差：{drift} vs {mag}");

        // AWQ 路径：权重被折叠了 s 倍，前向必须把输入除回去，否则结果整体错位
        let mut awq = Linear::new(k, n, &mut rng);
        let w0 = awq.weight.data();
        let act: Vec<f32> = (0..k).map(|j| 0.2 + j as f32 * 0.9).collect();
        let calib = LayerCalib {
            hessian: None,
            act_abs_mean: Some(act),
            n_tokens: m,
        };
        awq.quantize_weight(
            "awq.weight",
            QBits::Int8,
            QuantMethod::Awq,
            &QuantOpts {
                awq_alpha: Some(AWQ_ALPHA),
                ..Default::default()
            },
            Some(&calib),
        )
        .expect("AWQ 同样要能量化");
        let awq_out = crate::tensor::no_grad(|| awq.forward(&x3).data());
        // 与"未量化权重"的前向比：只该有量化误差，不该有系统性偏移
        let plain = Linear::new(k, n, &mut rng);
        plain.weight.set_data(w0.clone());
        plain.bias.set_data(awq.bias.data());
        let ref_out = crate::tensor::no_grad(|| plain.forward(&x3).data());
        let drift = awq_out
            .iter()
            .zip(&ref_out)
            .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        let mag = ref_out.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(
            drift < mag * 0.1,
            "AWQ 的折叠/展开必须严格配对，偏差应在量化误差量级内：{drift} vs {mag}"
        );
    }

    /// 量化后 f32 权重必须**真的被释放**：`weight` 退化成 1 元素占位张量，
    /// 逻辑维度（`dims`）则原样保留在量化表示里。只把 f32 副本留着不删，
    /// 就等于"量化省了显存"是假的——那正是这个项目最不该犯的错。
    #[test]
    fn quantize_weight_releases_the_f32_buffer() {
        let mut rng = Rng::new(77);
        let (k, n) = (32, 24);
        let mut lin = Linear::new(k, n, &mut rng);
        assert_eq!(lin.weight.numel(), k * n);
        assert_eq!(lin.dims(), (k, n));

        let report = lin
            .quantize_weight(
                "big.weight",
                QBits::Int4,
                QuantMethod::Rtn,
                &QuantOpts::default(),
                None,
            )
            .expect("普通 Linear 必须能量化");

        assert_eq!(lin.weight.numel(), 1, "f32 权重必须被换成占位张量，而不是留着副本");
        assert_eq!(lin.weight.shape(), &[1]);
        assert_eq!(lin.dims(), (k, n), "逻辑维度必须仍能从量化表示里读出来");
        assert_eq!(report.f32_bytes, k * n * 4);
        // int4 的码只占 1/8，加上每列一个 fp32 scale 仍应远小于 f32
        assert!(
            report.quant_bytes < report.f32_bytes / 2,
            "int4 的压缩率明显不足：{} vs {}",
            report.quant_bytes,
            report.f32_bytes
        );
        assert!(report.ratio() > 2.0);

        // 反量化必须把完整的 f32 权重恢复回来（形状、维度、可训标志都还原），
        // 否则"量化 → 存档 → 加载"这条链路会在中途断掉
        lin.dequantize_weight();
        assert!(!lin.has_quant());
        assert_eq!(lin.weight.shape(), &[k, n]);
        assert_eq!(lin.weight.numel(), k * n);
        assert_eq!(lin.dims(), (k, n));
        assert!(lin.weight.requires_grad(), "反量化后应恢复成可训的全参层");
        assert!(lin.weight.data().iter().all(|v| v.is_finite()));
    }

    /// 量化精度门限：int8 的输出相对误差 < 5%、int4 < 20%。
    ///
    /// 口径取"最大绝对偏差 / 最大输出幅值"——比逐元素 rmse 更严格，
    /// 因为它盯的是最坏的那个输出，而不是被平均掉的整体误差。
    #[test]
    fn quantized_output_error_stays_within_budget() {
        let mut rng = Rng::new(21);
        let (m, k, n) = (16, 64, 48);
        let lin = Linear::new(k, n, &mut rng);
        let x = Tensor::from_vec((0..m * k).map(|i| (i as f32 * 0.21).sin() * 1.5).collect(), vec![m, k]);
        let y0 = crate::tensor::no_grad(|| lin.forward(&x).data());
        let mag = y0.iter().fold(0.0f32, |m, v| m.max(v.abs()));

        for (bits, budget) in [(QBits::Int8, 0.05f32), (QBits::Int4, 0.20)] {
            let mut q = Linear::new(k, n, &mut rng);
            q.weight.set_data(lin.weight.data());
            q.bias.set_data(lin.bias.data());
            let report = q
                .quantize_weight(
                    "layer.weight",
                    bits,
                    QuantMethod::Rtn,
                    &QuantOpts::default(),
                    None,
                )
                .expect("普通 Linear 必须能量化");
            let y1 = crate::tensor::no_grad(|| q.forward(&x).data());
            let drift = y0
                .iter()
                .zip(&y1)
                .fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
            println!(
                "[{}] 输出相对误差 {:.4}%（预算 {:.0}%），权重 rel {:.4}，压缩 {:.2}×",
                bits.name(),
                drift / mag * 100.0,
                budget * 100.0,
                report.rel_err,
                report.ratio()
            );
            assert!(
                drift < mag * budget,
                "[{}] 量化误差超预算：{drift} vs 允许 {}（输出幅值 {mag}）",
                bits.name(),
                mag * budget
            );
        }
    }

    /// 校准统计必须**真的被用上**：给一个强相关的全矩阵 Hessian，Hessian 量化的码值应当
    /// 与 RTN 不同；同一个 Hessian 退化成对角模式时，码值又必须与 RTN 逐位一致
    /// （对角 `H⁻¹` ⇒ 跨通道补偿系数恒为 0）。这一正一反把"统计没被静默忽略"钉死。
    #[test]
    fn hess_layer_really_consumes_the_calibration_hessian() {
        let mut rng = Rng::new(404);
        let (k, n) = (24, 20);
        let base = Linear::new(k, n, &mut rng);
        let w = base.weight.data();
        // 强相关的 H：相邻输入通道高度冗余（真实激活就是这样）
        let rho = 0.9f32;
        let s = (1.0 - rho * rho).sqrt();
        let tokens = 200;
        let mut x: Vec<f32> = Vec::with_capacity(tokens * k);
        let mut rng2 = Rng::new(55);
        for _ in 0..tokens {
            let mut prev = rng2.randn();
            for _ in 0..k {
                prev = rho * prev + s * rng2.randn();
                x.push(prev);
            }
        }
        let mut h = vec![0.0f32; k * k];
        for t in 0..tokens {
            for i in 0..k {
                for j in 0..k {
                    h[i * k + j] += x[t * k + i] * x[t * k + j];
                }
            }
        }
        let diag: Vec<f32> = (0..k).map(|i| h[i * k + i]).collect();

        // 一个"用给定统计量化同一个权重、返回反量化结果"的小工具
        // （rng 走参数而不捕获，否则闭包的 `&mut rng` 会一直占着后续的 `Linear::new`）
        let quantized_with = |rng: &mut Rng, stats: Option<&LayerCalib>, method: QuantMethod| {
            let mut lin = Linear::new(k, n, rng);
            lin.weight.set_data(w.clone());
            let opts = if method == QuantMethod::Hess {
                QuantOpts { hess: HessOpts::default(), ..Default::default() }
            } else {
                QuantOpts::default()
            };
            lin.quantize_weight("l.weight", QBits::Int4, method, &opts, stats)
                .expect("普通 Linear 必须能量化");
            lin.dequant_folded()
        };

        // RTN 基线（同样用 int4、逐列分组）
        let rtn_code = quantized_with(&mut rng, None, QuantMethod::Rtn);

        let full = quantized_with(
            &mut rng,
            Some(&LayerCalib {
                hessian: Some(HessianKind::Full(h.clone())),
                act_abs_mean: Some(diag.clone()),
                n_tokens: tokens,
            }),
            QuantMethod::Hess,
        );
        let diagonal = quantized_with(
            &mut rng,
            Some(&LayerCalib {
                hessian: Some(HessianKind::Diagonal(diag)),
                act_abs_mean: None,
                n_tokens: tokens,
            }),
            QuantMethod::Hess,
        );

        assert_eq!(
            diagonal, rtn_code,
            "对角 Hessian 下 Hessian 量化在数学上退化成 RTN，必须逐位一致"
        );
        assert_ne!(full, rtn_code, "全矩阵 Hessian 没被用上：Hessian 量化的码值与 RTN 一字不差");

        // 该层没有校准统计时也必须能跑（退回 RTN），而不是报错
        assert_eq!(
            quantized_with(&mut rng, None, QuantMethod::Hess),
            rtn_code,
            "缺统计时应逐位退回 RTN，而不是拒绝量化"
        );
    }

    /// AWQ 的缩放向量长度必须等于**输入维度**（它逐输入通道作用），
    /// 且量化层必须把它存下来交给前向去分摊 `1/s`。
    #[test]
    fn awq_scale_length_is_in_features() {
        let mut rng = Rng::new(88);
        let (k, n) = (20, 12);
        let mut lin = Linear::new(k, n, &mut rng);
        let act: Vec<f32> = (0..k).map(|j| 0.1 + j as f32 * 0.5).collect();
        let report = lin
            .quantize_weight(
                "awq.weight",
                QBits::Int8,
                QuantMethod::Awq,
                &QuantOpts {
                    awq_alpha: Some(AWQ_ALPHA),
                    ..Default::default()
                },
                Some(&LayerCalib {
                    hessian: None,
                    act_abs_mean: Some(act),
                    n_tokens: 64,
                }),
            )
            .expect("普通 Linear 必须能量化");
        assert_eq!(report.method, QuantMethod::Awq);
        assert_eq!(report.alpha, Some(AWQ_ALPHA));
        let s = lin
            .quant
            .as_ref()
            .and_then(|qw| qw.input_scale.as_ref())
            .expect("AWQ 必须留下输入缩放");
        assert_eq!(s.len(), k, "缩放向量长度必须等于输入维度 in_features");
        assert!(s.iter().all(|v| v.is_finite() && *v > 0.0));
        // 缩放向量要计入字节口径（它是随权重一起保存的真实开销）
        let qw = lin.quant.as_ref().unwrap();
        assert_eq!(qw.byte_len(), qw.q.byte_len() + k * 4);

        // α 不给 ⇒ 走网格搜索，选出的 α 必须落在默认网格上
        let mut searched = Linear::new(k, n, &mut rng);
        let act: Vec<f32> = (0..k).map(|j| 0.1 + 4.0f32.powi(j as i32 % 5 - 2)).collect();
        let r2 = searched
            .quantize_weight(
                "awq.weight",
                QBits::Int4,
                QuantMethod::Awq,
                &QuantOpts { awq_alpha: None, ..Default::default() },
                Some(&LayerCalib {
                    hessian: None,
                    act_abs_mean: Some(act),
                    n_tokens: 64,
                }),
            )
            .expect("普通 Linear 必须能量化");
        let a = r2.alpha.expect("搜索模式必须报告选中的 α");
        assert!(
            crate::quant::awq_alpha_grid().contains(&a),
            "搜索给出的 α={a} 不在默认网格上"
        );
    }

    /// 挂着 LoRA 适配器的层必须被跳过（返回 `None`）：量化主干 + f32 增量既省不下显存，
    /// 又没法把增量合并回去。调用方的正确顺序是先 `merge_lora` 再量化。
    #[test]
    fn lora_layer_is_skipped_by_quantization() {
        let mut rng = Rng::new(3);
        let mut lin = Linear::new(8, 6, &mut rng);
        lin.attach_lora(2, 4.0, &mut rng);
        assert!(
            lin.quantize_weight(
                "lora.weight",
                QBits::Int8,
                QuantMethod::Rtn,
                &QuantOpts::default(),
                None
            )
            .is_none(),
            "挂着适配器的层必须被跳过"
        );
        assert!(!lin.has_quant());
        assert_eq!(lin.weight.numel(), 48, "被跳过的层权重必须原封不动");

        // 先合并再量化就正常了
        lin.merge_lora();
        assert!(lin
            .quantize_weight(
                "lora.weight",
                QBits::Int8,
                QuantMethod::Rtn,
                &QuantOpts::default(),
                None
            )
            .is_some());
    }

    /// 在求导模式下用量化层必须**明确报错**，而不是静默地不产生 dW 让训练白跑。
    #[test]
    #[should_panic(expected = "量化层不能参与训练")]
    fn quantized_forward_panics_in_grad_mode() {
        let mut rng = Rng::new(12);
        let mut lin = Linear::new(4, 3, &mut rng);
        lin.quantize_weight(
            "w",
            QBits::Int8,
            QuantMethod::Rtn,
            &QuantOpts::default(),
            None,
        )
        .expect("普通 Linear 必须能量化");
        let x = Tensor::param(vec![1.0; 4], vec![1, 4]);
        // 没有 no_grad 包裹 ⇒ 求导模式 ⇒ 必须 panic
        let _ = lin.forward(&x);
    }

    /// 量化层上合并 LoRA：必须**先反量化**再加增量，而不是把 f32 增量塞进量化域。
    /// 合并后这一层不再是量化层（`quant` 被清空），权重复原成 f32。
    #[test]
    fn merge_lora_on_quantized_layer_dequantizes_first() {
        let mut lin = hand_linear(); // W = [[1,2],[3,4]]
        lin.quantize_weight(
            "w",
            QBits::Int8,
            QuantMethod::Rtn,
            &QuantOpts::default(),
            None,
        )
        .expect("普通 Linear 必须能量化");
        let deq = lin.dequant_folded();
        lin.lora = Some(hand_lora()); // ΔW = [[1,1],[1,1]]（对称，转置后同形）

        lin.merge_lora();

        assert!(lin.quant.is_none(), "合并后不能再留着量化状态");
        assert!(!lin.has_quant());
        let w = lin.weight.data();
        for i in 0..4 {
            let expect = deq[i] + 1.0; // 先反量化，再加 ΔW
            assert!(
                (w[i] - expect).abs() < 1e-5,
                "第 {i} 项：{} vs 反量化后合并 {expect}",
                w[i]
            );
        }
        // 关键区分：若弄反了顺序（先合并再量化），权重里会留着量化格点的痕迹；
        // 这里 1.992… 而不是 2.0，正说明落在 f32 域里、没有二次取整
        assert!((w[0] - 2.0).abs() > 1e-3, "结果看起来像被重新量化过了：{}", w[0]);
        // x = [1, 0] ⇒ y = 权重的第一行（in = 2, out = 2）
        let x = Tensor::from_vec(vec![1.0, 0.0], vec![1, 2]);
        let y = lin.forward(&x).data();
        assert!((y[0] - w[0]).abs() < 1e-5 && (y[1] - w[1]).abs() < 1e-5);
    }
}
