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
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
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
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// 从一个范围内随机选一个整数下标（用于采样）
    pub fn choice(&mut self, n: usize) -> usize {
        assert!(n > 0, "choice 的范围必须大于 0");
        (self.next_u64() % n as u64) as usize
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
}
