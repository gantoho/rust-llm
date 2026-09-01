//! 数据加载（第 14 课）
//!
//! 训练 GPT 的自监督方式：给模型一段文本，让它预测"下一个 token"。
//! 不需要人工标注——文本本身就是标签（这就是"自监督学习"）。
//!
//! 支持：
//! - 内置小语料（`CORPUS`，demo 用）或外部文本文件（正式训练用）
//! - 训练 / 验证划分：显式提供验证文件，或自动从训练文本末尾切 10%
//! - `sample_batch`（训练区随机采样）与 `eval_batch`（验证区随机采样）

use crate::rng::Rng;
use crate::tokenizer::Tokenizer;

/// 内置小语料（一个英文小故事），用于演示训练
pub const CORPUS: &str = "\
Once upon a time in a small village, there lived a curious little fox named Red. \
Every morning, Red would wake up early and explore the forest. He loved to watch \
the birds fly and the rivers flow. One day, Red found a golden key under an old \
oak tree. What could it open? Red wondered. He ran to his friend, the wise old owl. \
The owl said, the key opens the door to the hidden garden, where flowers bloom all \
year round. Red was excited! He followed the path to the garden and turned the key. \
The door creaked open, revealing a world of colors and light. From that day on, Red \
visited the garden every day, and he learned that every adventure begins with a \
single step.";

/// 数据加载器
///
/// 输入 x 是一段长度为 block_size 的 token 序列；
/// 目标 y 是 x 右移一位（x[i] 的下一个 token 是 y[i]）。
///
/// token 序列被切为两块：
/// - `tokens[..val_start]`：训练区（`sample_batch` 在这里随机采样）
/// - `tokens[val_start..]`：验证区（`eval_batch` 在这里采样，用于评估）
pub struct DataLoader {
    tokens: Vec<usize>,
    block_size: usize,
    batch_size: usize,
    val_start: usize,
}

impl DataLoader {
    /// 整个文本都作为训练数据（demo 用，无验证集）。
    /// 需要 `tokens.len() > block_size`，否则无法切出完整序列。
    pub fn new(text: &str, tokenizer: &Tokenizer, block_size: usize, batch_size: usize) -> Self {
        let tokens = tokenizer.encode(text);
        assert!(
            tokens.len() > block_size,
            "语料太短，无法切出完整序列（{} < {}）",
            tokens.len(),
            block_size
        );
        let len = tokens.len();
        DataLoader {
            tokens,
            block_size,
            batch_size,
            val_start: len,
        }
    }

    /// 从训练/验证文本构造加载器。
    /// - `val_text = Some(..)`：使用独立的验证文本；
    /// - `val_text = None`：自动从训练文本末尾切出约 10%（至少 block+1 个 token）作验证集。
    pub fn from_texts(
        train_text: &str,
        val_text: Option<&str>,
        tokenizer: &Tokenizer,
        block_size: usize,
        batch_size: usize,
    ) -> Self {
        let mut tokens = tokenizer.encode(train_text);
        assert!(
            tokens.len() > block_size,
            "训练语料太短，无法切出完整序列（{} < {}）",
            tokens.len(),
            block_size
        );
        let val_start = match val_text {
            Some(v) => {
                let split = tokens.len();
                tokens.extend(tokenizer.encode(v));
                assert!(
                    tokens.len() - split > block_size,
                    "验证文本太短，无法切出完整序列（{} token，需要 > {}）",
                    tokens.len() - split,
                    block_size
                );
                split
            }
            None => {
                // 末尾留出至少 block+1 个 token 作验证集，再按 10% 切分
                (tokens.len() as f64 * 0.9) as usize
            }
        };
        let val_start = val_start.min(tokens.len() - (block_size + 1));
        assert!(
            tokens.len() - val_start > block_size,
            "验证语料太短，无法切出完整序列"
        );
        DataLoader {
            tokens,
            block_size,
            batch_size,
            val_start,
        }
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn num_train_tokens(&self) -> usize {
        self.val_start
    }

    pub fn num_val_tokens(&self) -> usize {
        self.tokens.len() - self.val_start
    }

    pub fn has_val(&self) -> bool {
        self.val_start < self.tokens.len()
    }

    /// 采样一批训练数据（训练区随机）。
    /// 返回 (x, y)，各为 [B*T] 展平序列：随机选 B 个起点，每个起点截取 block_size+1 个 token。
    pub fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        self.sample_region(rng, 0, self.val_start, "训练")
    }

    /// 采样一批验证数据（验证区随机，调用方用固定种子的 Rng 保证可复现）。
    pub fn eval_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        assert!(self.has_val(), "没有验证数据，无法采样 eval batch");
        self.sample_region(rng, self.val_start, self.tokens.len(), "验证")
    }

    /// 在 [lo, hi) 区间内随机选起点采样
    fn sample_region(
        &self,
        rng: &mut Rng,
        lo: usize,
        hi: usize,
        tag: &str,
    ) -> (Vec<usize>, Vec<usize>) {
        assert!(
            hi > lo + self.block_size,
            "{}区数据不足，无法采样（{} token，需要 > {}）",
            tag,
            hi - lo,
            self.block_size
        );
        let max_start = hi - lo - self.block_size - 1;
        let mut x = Vec::with_capacity(self.batch_size * self.block_size);
        let mut y = Vec::with_capacity(self.batch_size * self.block_size);
        for _ in 0..self.batch_size {
            let start = lo + rng.choice(max_start);
            for j in 0..self.block_size {
                x.push(self.tokens[start + j]);
                y.push(self.tokens[start + j + 1]);
            }
        }
        (x, y)
    }
}
