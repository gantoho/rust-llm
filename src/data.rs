//! 数据加载（第 14 课）
//!
//! 训练 GPT 的自监督方式：给模型一段文本，让它预测"下一个字符"。
//! 不需要人工标注——文本本身就是标签（这就是"自监督学习"）。

use crate::rng::Rng;
use crate::tokenizer::CharTokenizer;

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

/// 数据加载器：从语料中随机切出 (输入, 目标) 对
///
/// 输入 x 是一段长度为 block_size 的 token 序列；
/// 目标 y 是 x 右移一位（x[i] 的下一个 token 是 y[i]）。
pub struct DataLoader {
    tokens: Vec<usize>,
    block_size: usize,
    batch_size: usize,
}

impl DataLoader {
    pub fn new(text: &str, tokenizer: &CharTokenizer, block_size: usize, batch_size: usize) -> Self {
        let tokens = tokenizer.encode(text);
        assert!(
            tokens.len() > block_size,
            "语料太短，无法切出完整序列（{} < {}）",
            tokens.len(),
            block_size
        );
        DataLoader {
            tokens,
            block_size,
            batch_size,
        }
    }

    pub fn num_tokens(&self) -> usize {
        self.tokens.len()
    }

    /// 采样一批训练数据。
    /// 返回 (x, y)，各为 [B*T] 展平序列。
    /// 随机选 B 个起点，每个起点截取 block_size+1 个 token：
    /// x = 前 block_size 个，y = 后 block_size 个（预测下一个）
    pub fn sample_batch(&self, rng: &mut Rng) -> (Vec<usize>, Vec<usize>) {
        let max_start = self.tokens.len() - self.block_size - 1;
        let mut x = Vec::with_capacity(self.batch_size * self.block_size);
        let mut y = Vec::with_capacity(self.batch_size * self.block_size);
        for _ in 0..self.batch_size {
            let start = rng.choice(max_start);
            for j in 0..self.block_size {
                x.push(self.tokens[start + j]);
                y.push(self.tokens[start + j + 1]);
            }
        }
        (x, y)
    }
}
