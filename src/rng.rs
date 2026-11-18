//! 随机数生成器（第 5 课：权重初始化）
//!
//! 标准库没有 RNG，我们自己实现一个 xorshift64 —— 这是最简单可靠的伪随机算法之一。
//! 好处：零依赖，且可复现（用固定种子）。
//!
//! xorshift 原理：用位运算（异或 + 移位）不断"搅拌"一个 64 位状态，
//! 产生的序列统计性质接近均匀分布。速度快、实现只需几行。

/// xorshift64* 伪随机数生成器
#[derive(Clone)]
pub struct Rng {
    state: u64,
}

thread_local! {
    /// "跳过初始化计算"开关（见 [`without_init`]）。
    ///
    /// 用它而不是"少调几次 rng"：状态**照常推进**，所以同一 seed 下
    /// 构造之后的随机数流与正常构造逐位相同 —— 调用方（采样器等）的行为不受影响。
    static INIT_SKIP: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// 在"参数马上会被整体覆盖"的场景下跳过随机初值的**浮点计算部分**。
///
/// 动机（实测）：`Transformer::new` 要为每个参数调一次 `randn()`，而 `randn` 是
/// Box–Muller（每元素一次 `ln` + `sqrt` + `cos`）。Qwen2.5-0.5B 有 4.94 亿个参数，
/// 构造一次要 **6.5 s**——而 `generate` / `eval` / `serve` 紧接着就 `load_params`
/// 把每个参数整体替换掉，这 6.5 s 是纯浪费（读档本身只要 0.7 s）。
///
/// 语义约定（与 `no_grad` 同一套路，RAII 恢复，panic 也不会卡住开关）：
/// - **随机数流不变**：`next_u64` 照常推进，`randn` 照常消耗两次 `next_f32`；
/// - 只是 `randn` / `next_f32` 返回 0.0，跳过 `ln`/`sqrt`/`cos` 这些真正的开销；
/// - 因此"构造完再用同一个 rng 采样"的调用方，结果与不开开关时**逐位相同**。
///
/// **不要**在没有后续覆盖的路径上用它（例如从零训练）：那会让所有权重恒为 0。
pub fn without_init<T>(f: impl FnOnce() -> T) -> T {
    struct Restore(bool);
    impl Drop for Restore {
        fn drop(&mut self) {
            INIT_SKIP.with(|c| c.set(self.0));
        }
    }
    let prev = INIT_SKIP.with(|c| c.replace(true));
    let _restore = Restore(prev);
    f()
}

#[inline]
fn init_skipped() -> bool {
    INIT_SKIP.with(|c| c.get())
}

impl Rng {
    /// 用种子创建。种子为 0 时自动换成 1（避免状态全 0 的退化情况）。
    pub fn new(seed: u64) -> Self {
        Rng {
            state: if seed == 0 { 1 } else { seed },
        }
    }

    /// 生成下一个 [0, 2^64) 的整数
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// 生成 [0, 1) 的均匀浮点数
    pub fn next_f32(&mut self) -> f32 {
        let bits = self.next_u64() >> 40;
        // 跳过模式下状态已推进，但不必做浮点转换（见 [`without_init`]）
        if init_skipped() {
            return 0.0;
        }
        bits as f32 / (1u64 << 24) as f32
    }

    /// 生成 [lo, hi) 的均匀浮点数
    #[allow(dead_code)] // 基础随机数 API（权重初始化等场景）
    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.next_f32()
    }

    /// 标准正态分布 N(0, 1) 采样。
    /// 用 Box-Muller 变换：两个均匀数 -> 一个正态数
    ///   r = sqrt(-2 ln u1), θ = 2π u2, 则 r·cos(θ) ~ N(0,1)
    pub fn randn(&mut self) -> f32 {
        let u1 = (self.next_f32()).max(1e-12);
        let u2 = self.next_f32();
        // 跳过模式：上面两次 `next_f32` 已经把状态推进到与正常构造完全相同的位置，
        // 这里直接返回 0.0，省掉 ln/sqrt/cos 三个超越函数（见 [`without_init`]）
        if init_skipped() {
            return 0.0;
        }
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// 从一个范围内随机选一个整数下标（用于采样）
    pub fn choice(&mut self, n: usize) -> usize {
        assert!(n > 0, "choice 的范围必须大于 0");
        (self.next_u64() % n as u64) as usize
    }

    /// 取出当前内部状态。
    ///
    /// 供批次预取使用：采样跑在工作线程的 rng 克隆上，每批采完要把状态回传给
    /// 调用方的 rng（[`set_state`]），使采样序列与串行执行逐位一致、可复现。
    pub(crate) fn state(&self) -> u64 {
        self.state
    }

    /// 恢复到 [`Self::state`] 取出的状态（与之配对使用，只在预取回写时调用）。
    pub(crate) fn set_state(&mut self, state: u64) {
        // xorshift 从非零状态出发不会产生 0；这里只兜住配对误用，避免把生成器退化
        self.state = if state == 0 { 1 } else { state };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rng_deterministic() {
        let mut r1 = Rng::new(42);
        let mut r2 = Rng::new(42);
        for _ in 0..100 {
            assert_eq!(r1.next_f32(), r2.next_f32(), "同种子必须产生相同序列");
        }
    }

    #[test]
    fn test_rng_range() {
        let mut r = Rng::new(1);
        for _ in 0..1000 {
            let v = r.uniform(-1.0, 1.0);
            assert!(v >= -1.0 && v < 1.0);
            let n = r.randn();
            assert!(n.is_finite());
        }
    }

    #[test]
    fn test_choice_range() {
        let mut r = Rng::new(7);
        for _ in 0..100 {
            let c = r.choice(10);
            assert!(c < 10);
        }
    }

    /// `without_init` 只能省计算，**不能改变随机数流**。
    ///
    /// 这是它敢用在推理入口的前提：`generate` / `serve` 先用同一个 rng 构造模型，
    /// 再用它采样；只要消耗的 `next_u64` 次数一致，同一 seed 的生成结果就逐位不变。
    #[test]
    fn test_without_init_preserves_rng_stream() {
        let mut normal = Rng::new(42);
        let mut skipped = Rng::new(42);
        // 模拟参数初始化：1000 次 randn（每次消耗两次 next_f32 = 两次 next_u64）
        let v_normal: Vec<f32> = (0..1000).map(|_| normal.randn()).collect();
        let v_skipped: Vec<f32> =
            crate::rng::without_init(|| (0..1000).map(|_| skipped.randn()).collect());
        assert!(
            v_skipped.iter().all(|v| *v == 0.0),
            "跳过模式下初值应全为 0（随后会被 checkpoint 覆盖）"
        );
        assert!(v_normal.iter().any(|v| *v != 0.0), "正常模式应有非零初值");
        for i in 0..100 {
            assert_eq!(
                normal.next_u64(),
                skipped.next_u64(),
                "第 {i} 次抽取后随机数流分叉了"
            );
        }
        // 开关必须恢复：离开 without_init 后 randn 重新产出真随机值
        // （Box–Muller 恰好给出 0.0 的概率可忽略，这里用它当判据是安全的）
        let back = skipped.randn();
        assert!(back != 0.0, "离开 without_init 后应恢复真实初始化计算");
    }
}
