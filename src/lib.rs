//! 从零实现大语言模型 —— 库入口（可被 `tests/` 与外部 crate 引用）
//!
//! 二进制入口在 `src/main.rs`（只保留 `main()` 与 CLI 分派）；
//! 所有功能实现都在本库的 `pub mod` 里。

// serve 里的 OpenAPI 规范是一整个巨大的 `json!{...}`，宏展开层数超过默认上限
#![recursion_limit = "512"]
// clippy 硬门禁下的**有意豁免**（每条都写明理由；新增豁免必须在这里登记）：
//
// - needless_range_loop：数值内核里多个数组用同一下标访问（`a[i] * b[i] + c[i]`），
//   改写成 zip/enumerate 反而看不出"同下标"这个关键事实。
// - too_many_arguments：算子签名本身带一批维度参数（matmul/gemm/attention/kv append），
//   拆结构体只会让调用点多一层解包。
// - type_complexity：autograd 的闭包类型、量化器的嵌套容器，天然复杂。
// - neg_cmp_op_on_partial_ord：`!(a < b)` 是刻意表达"含 NaN 时走另一支"。
// - doc_lazy_continuation / empty_line_after_doc_comments：文档排版风格。
// - large_enum_variant：枚举刻意内联大结构（热路径上不想多一层 Box 间接）。
#![allow(clippy::needless_range_loop)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::type_complexity)]
#![allow(clippy::neg_cmp_op_on_partial_ord)]
#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::empty_line_after_doc_comments)]
#![allow(clippy::large_enum_variant)]

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
pub mod json_constraint;
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
pub mod sftgen;
pub mod speculative;
pub mod syncutil;
pub mod tensor;
pub mod tokenizer;
pub mod train;
pub mod vision;
pub mod vqvae;
pub mod winutil;

// 这几个原本由 bin 的根 use 引入、被 crate::xxx 引用（如 serve 里的 crate::stop_label）——
// 拆分后必须在库根重新导出，路径才仍然成立。
pub use cmd::{init_console_utf8, kv_bits_from_name, stop_label};
