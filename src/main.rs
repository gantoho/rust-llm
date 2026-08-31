//! 从零实现大语言模型（纯 Rust，不依赖深度学习框架）
//!
//! 这是主入口，依次演示各课成果：
//! - 第 7 课：MLP 学会 XOR（验证神经网络 + 反向传播正确）
//! - 第 8 课：BPE 分词器
//! - 第 12-16 课：训练一个小 GPT 并生成文本
//! - 第 17-18 课：AdamW 优化器、KV Cache 加速
//! - 第 20 课：warmup + cosine 学习率调度
//!
//! 配套教程文档见 `docs/` 目录。

mod data;
mod layers;
mod loss;
mod model;
mod module;
mod optim;
mod rng;
mod sample;
mod tensor;
mod tokenizer;
mod train;

use data::{DataLoader, CORPUS};
use layers::{Linear, relu};
use loss::cross_entropy_loss;
use model::{GPT, GPTConfig};
use module::Module;
use optim::SGD;
use rng::Rng;
use sample::generate;
use tensor::Tensor;
use tokenizer::{BPETokenizer, CharTokenizer};

fn main() {
    demo_xor();
    demo_bpe();
    demo_gpt();
}

/// 演示 1（第 7 课）：用 MLP 学会 XOR 异或
///
/// XOR 是经典的"神经网络必须非线性"案例：
/// 单层线性模型学不会（数据线性不可分），加一层 ReLU 就能学会。
fn demo_xor() {
    println!("=== 演示 1：MLP 学习 XOR（第 7 课）===");
    let mut rng = Rng::new(42);

    // 数据集：4 个样本
    let x_data = Tensor::from_vec(
        vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
        vec![4, 2],
    );
    let y_targets = vec![0usize, 1, 1, 0]; // XOR 真值表

    // 网络：2 -> 4 (ReLU) -> 2（两个输出：0 和 1 的分数）
    let fc1 = Linear::new(2, 4, &mut rng);
    let fc2 = Linear::new(4, 2, &mut rng);
    let params: Vec<Tensor> = {
        let mut ps = fc1.parameters();
        ps.extend(fc2.parameters());
        ps
    };
    let opt = SGD::new(0.5, params);

    for step in 0..1000 {
        // 前向：relu(x @ W1 + b1) @ W2 + b2
        let h = relu(&fc1.forward(&x_data));
        let logits = fc2.forward(&h);
        let loss = cross_entropy_loss(&logits, &y_targets);

        loss.backward();
        opt.step();
        opt.zero_grad();

        if step % 200 == 0 {
            println!("  step {:>4} | loss {:.4}", step, loss.data()[0]);
        }
    }

    // 验证正确率
    let h = relu(&fc1.forward(&x_data));
    let logits = fc2.forward(&h);
    let data = logits.data();
    let mut correct = 0;
    for i in 0..4 {
        let pred = if data[i * 2] > data[i * 2 + 1] { 0 } else { 1 };
        if pred == y_targets[i] {
            correct += 1;
        }
    }
    println!("  训练后正确率：{}/4（100% 说明反向传播正确）\n", correct);
}

/// 演示 2（第 8 课）：BPE 分词器
fn demo_bpe() {
    println!("=== 演示 2：BPE 分词器（第 8 课）===");

    let tok = BPETokenizer::train(CORPUS, 400);
    println!(
        "  BPE 词表大小：{}（初始 256 字节 + {} 次合并）",
        tok.vocab_size(),
        tok.vocab_size() - 256
    );

    println!("  \"Red\" 的 token：{:?}", tok.encode("Red"));
    let full = tok.encode("the garden");
    println!("  \"the garden\" -> {} 个 token（高频子词被压缩）", full.len());
    let decoded = tok.decode(&full);
    println!("  解码验证：\"{}\"", decoded);

    let char_tok = CharTokenizer::new(CORPUS);
    let ids = char_tok.encode("fox");
    println!("  字符级词表：{}，\"fox\" -> {:?}\n", char_tok.vocab_size(), ids);
}

/// 演示 3（第 12-16、17-20 课）：训练小 GPT 并生成文本
fn demo_gpt() {
    println!("=== 演示 3：训练小 GPT 并生成文本 ===");

    let mut rng = Rng::new(1234);
    let tokenizer = CharTokenizer::new(CORPUS);
    let vocab_size = tokenizer.vocab_size();
    println!("  语料 {} 字符，字符词表 {} 个", CORPUS.len(), vocab_size);

    let model = GPT::new(GPTConfig::tiny(vocab_size), &mut rng);

    // 训练（第 13、17、20 课：训练循环 + AdamW + warmup/cosine 调度）
    let loader = DataLoader::new(CORPUS, &tokenizer, model.cfg.block_size, 8);
    train::train_gpt(&model, &tokenizer, &loader, 600, 8, 3e-3, 50, 100, &mut rng);

    // 生成（无 cache）
    println!("\n  —— 生成 1（temperature=0.8, top-k=10, top-p=0.9, 无 KV cache）——");
    let out1 = generate(&model, &tokenizer, "Once upon a", 80, 0.8, 10, 0.9, false, &mut rng);
    println!("  {}", out1);

    // 生成（带 KV cache，第 18 课）
    println!("\n  —— 生成 2（temperature=0.8, top-k=10, top-p=0.9, 带 KV cache）——");
    let out2 = generate(&model, &tokenizer, "The fox", 80, 0.8, 10, 0.9, true, &mut rng);
    println!("  {}", out2);
    println!("\n  （KV cache 只改计算方式、不改生成分布，两者应高度一致）");
}
