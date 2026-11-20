//! 从零实现大语言模型 —— 库入口（可被 `tests/` 与外部 crate 引用）
//!
//! 二进制入口在 `src/main.rs`（只保留 `main()` 与 CLI 分派）；
//! 所有功能实现都在本库的 `pub mod` 里。

// serve 里的 OpenAPI 规范是一整个巨大的 `json!{...}`，宏展开层数超过默认上限
#![recursion_limit = "512"]

// `runlog` 里的 `logln!` 宏要覆盖后面所有模块，必须最先声明并带 #[macro_use]
#[macro_use]
pub mod runlog;
pub mod align;
pub mod attention;
pub mod autograd;
pub mod checkpoint;
pub mod cli;
pub mod cmd;
pub mod comm;
pub mod config;
pub mod csa;
pub mod data;
pub mod distill;
pub mod distributed;
pub mod fp4;
pub mod fp8;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod hf;
pub mod layers;
pub mod loss;
pub mod mhc;
pub mod model;
pub mod module;
pub mod moe;
pub mod optim;
pub mod prompt;
pub mod quant;
pub mod rag;
pub mod rng;
pub mod rope;
pub mod sample;
pub mod scaling;
pub mod serve;
pub mod speculative;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod vision;
pub mod vqvae;

// 这几个原本由 bin 的根 use 引入、被 crate::xxx 引用（如 serve 里的 crate::stop_label）——
// 拆分后必须在库根重新导出，路径才仍然成立。
pub use cmd::{init_console_utf8, kv_bits_from_name, stop_label};
