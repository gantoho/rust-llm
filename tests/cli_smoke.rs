//! CLI 端到端冒烟：直接跑**真实二进制**，覆盖那些只靠手工验证的子命令。
//!
//! 用 `CARGO_BIN_EXE_llm_from_scratch`（cargo 为集成测试注入的路径），不引入
//! `assert_cmd` 之类的额外依赖——与项目的依赖边界一致。

mod common;

use std::process::Command;

fn bin() -> &'static str {
    env!("CARGO_BIN_EXE_llm_from_scratch")
}

/// `--help` 必须能正常退出：clap 的 derive 结构一旦写坏，第一道冒烟就会拦下。
#[test]
fn cli_help_works() {
    let out = Command::new(bin())
        .arg("--help")
        .output()
        .expect("跑 --help");
    assert!(
        out.status.success(),
        "--help 失败：status={:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    for cmd in ["train", "serve", "sft", "bench", "chat"] {
        assert!(text.contains(cmd), "--help 里缺少子命令 {cmd}");
    }
}

/// `bench` 是秒级自检：能跑完并打印 tok/s，说明训练/推理的关键路径没被打断。
#[test]
fn cli_bench_runs() {
    let out = Command::new(bin())
        .args(["bench", "--steps", "2", "--gen-tokens", "4"])
        .output()
        .expect("跑 bench");
    assert!(
        out.status.success(),
        "bench 失败：status={:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(text.contains("tok/s"), "bench 没打印吞吐：{text}");
    assert!(text.contains("[bench] train"), "bench 没打印训练段：{text}");
}

/// `preset` 生成的配置必须能被自己再读回来（写—读闭环）。
#[test]
fn cli_preset_roundtrips() {
    let dir = std::env::temp_dir().join("llm_cli_smoke_preset");
    std::fs::create_dir_all(&dir).unwrap();
    let cfg = dir.join("cfg.json");
    let out = Command::new(bin())
        .args(["preset", "--name", "small", "--output"])
        .arg(&cfg)
        .output()
        .expect("跑 preset");
    assert!(
        out.status.success(),
        "preset 失败：status={:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(cfg.exists(), "preset 没有写出 {} ", cfg.display());

    // 用 `eval --help` 之外最轻的读取路径：`train` 会解析配置，这里只验证 JSON 可解析
    let text = std::fs::read_to_string(&cfg).unwrap();
    let v: serde_json::Value = serde_json::from_str(&text).expect("preset 写出的不是合法 JSON");
    assert!(v.get("model").is_some(), "配置里缺 model 段：{text}");
    assert!(v.get("train").is_some(), "配置里缺 train 段：{text}");
}

/// `distill` 自带数值自检（不需要 checkpoint），必须跑通并打印结论。
#[test]
fn cli_distill_runs() {
    let out = Command::new(bin())
        .args(["distill", "--steps", "3", "--teacher-steps", "3"])
        .output()
        .expect("跑 distill");
    assert!(
        out.status.success(),
        "distill 失败：status={:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(!text.trim().is_empty(), "distill 没有任何输出");
}

/// `speculative` 的分布检验要求至少 200 次试验；跑通即说明投机解码路径没断。
#[test]
fn cli_speculative_runs() {
    let out = Command::new(bin())
        .args([
            "speculative",
            "--trials",
            "200",
            "--max-new",
            "4",
            "--gamma",
            "2",
        ])
        .output()
        .expect("跑 speculative");
    assert!(
        out.status.success(),
        "speculative 失败：status={:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!String::from_utf8_lossy(&out.stdout).trim().is_empty());
}

/// `quant` 端到端：训 2 步得到 checkpoint → int4/RTN 量化并落盘。
///
/// 这条同时覆盖"char 分词器存档能被重新加载"（中文单字符校验曾经写错成按字节比较）。
#[test]
fn cli_quant_roundtrip() {
    let dir = common::fresh_dir("quant");
    let corpus = common::write_corpus(&dir);
    let cfg = common::write_config(&dir, &common::tiny_config(&dir, &corpus, 2));
    let ckpt = dir.join("ckpt/latest.ckpt");

    let out = Command::new(bin())
        .args(["train", "--config"])
        .arg(&cfg)
        .output()
        .expect("跑 train");
    assert!(
        out.status.success(),
        "train 失败：{:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(ckpt.exists(), "train 没写出 latest.ckpt");

    let q = dir.join("q.ckpt");
    let out = Command::new(bin())
        .args(["quant", "--config"])
        .arg(&cfg)
        .arg("--ckpt")
        .arg(&ckpt)
        .args(["--bits", "int4", "--method", "rtn", "--out"])
        .arg(&q)
        .output()
        .expect("跑 quant");
    assert!(
        out.status.success(),
        "quant 失败：{:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(q.exists(), "quant 没有写出 {}", q.display());
}

/// `sft` + `chat` 端到端：预训练 2 步 → SFT 2 步 → 用产物跑一轮对话。
///
/// 覆盖"训练产物能被后续子命令接上"这条最容易被重构打断的链路：
/// checkpoint 命名、分词器落盘位置、SFT 的模板停止标记。
#[test]
fn cli_sft_and_chat_end_to_end() {
    let dir = common::fresh_dir("sft");
    let corpus = common::write_corpus(&dir);
    let cfg = common::write_config(&dir, &common::tiny_config(&dir, &corpus, 2));
    let pre = dir.join("ckpt/latest.ckpt");

    let out = Command::new(bin())
        .args(["train", "--config"])
        .arg(&cfg)
        .output()
        .expect("跑 train");
    assert!(out.status.success(), "train 失败：{:?}", out.status);

    // SFT 语料：两轮对话（用户：/助手：格式）
    let sft_file = dir.join("sft.txt");
    std::fs::write(
        &sft_file,
        "用户：你好\n助手：你好\n\n用户：再见\n助手：再见\n",
    )
    .unwrap();

    let sft_out = dir.join("sft");
    let out = Command::new(bin())
        .args(["sft", "--config"])
        .arg(&cfg)
        .arg("--pretrained")
        .arg(&pre)
        .arg("--sft-file")
        .arg(&sft_file)
        .args(["--steps", "2", "--out-dir"])
        .arg(&sft_out)
        .output()
        .expect("跑 sft");
    assert!(
        out.status.success(),
        "sft 失败：{:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let best = sft_out.join("best.ckpt");
    assert!(
        best.exists(),
        "sft 没写出 best.ckpt，产物：{:?}",
        std::fs::read_dir(&sft_out).map(|d| d.count())
    );

    // 用 SFT 产物跑一轮对话（stdin 喂一句 + :quit）
    let tok = sft_out.join("tokenizer.json");
    let mut child = Command::new(bin())
        .args(["chat", "--ckpt"])
        .arg(&best)
        .arg("--tokenizer")
        .arg(&tok)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .expect("启动 chat");
    {
        use std::io::Write;
        let stdin = child.stdin.as_mut().expect("拿到 stdin");
        stdin
            .write_all("你好\n:quit\n".as_bytes())
            .expect("写 stdin");
    }
    let out = child.wait_with_output().expect("等 chat 结束");
    assert!(
        out.status.success(),
        "chat 失败：{:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    // 逐轮明细写在运行日志里（stdout 只有启动信息），从 stdout 里取日志路径再读
    let log_path = text
        .lines()
        .find_map(|l| l.trim().strip_prefix("运行日志："))
        .expect("stdout 里没有运行日志路径")
        .trim()
        .to_string();
    let log = std::fs::read_to_string(&log_path).expect("读运行日志");
    assert!(
        log.contains("第 1 轮"),
        "chat 没有完成一轮对话，日志：{log_path}"
    );
    assert!(
        log.contains("输出："),
        "chat 没有产出回答，日志：{log_path}"
    );
}

/// `eval`：训 2 步后必须能按 text 口径算出 loss/perplexity。
#[test]
fn cli_eval_after_train() {
    let dir = common::fresh_dir("eval");
    let corpus = common::write_corpus(&dir);
    let cfg = common::write_config(&dir, &common::tiny_config(&dir, &corpus, 2));
    let ckpt = dir.join("ckpt/latest.ckpt");

    let out = Command::new(bin())
        .args(["train", "--config"])
        .arg(&cfg)
        .output()
        .expect("跑 train");
    assert!(out.status.success(), "train 失败：{:?}", out.status);
    assert!(ckpt.exists(), "train 没写出 latest.ckpt");

    let out = Command::new(bin())
        .args(["eval", "--config"])
        .arg(&cfg)
        .arg("--ckpt")
        .arg(&ckpt)
        .output()
        .expect("跑 eval");
    assert!(
        out.status.success(),
        "eval 失败：{:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout).to_lowercase();
    assert!(text.contains("loss"), "eval 没打印 loss：{text}");
    assert!(
        text.contains("perplexity") || text.contains("困惑度"),
        "eval 没打印 perplexity：{text}"
    );
}

/// `generate`：训 2 步后必须能"加载 → 推理 → 生成若干 token"。
#[test]
fn cli_generate_after_train() {
    let dir = common::fresh_dir("generate");
    let corpus = common::write_corpus(&dir);
    let cfg = common::write_config(&dir, &common::tiny_config(&dir, &corpus, 2));
    let ckpt = dir.join("ckpt/latest.ckpt");

    let out = Command::new(bin())
        .args(["train", "--config"])
        .arg(&cfg)
        .output()
        .expect("跑 train");
    assert!(out.status.success(), "train 失败：{:?}", out.status);

    let out = Command::new(bin())
        .args(["generate", "--config"])
        .arg(&cfg)
        .arg("--ckpt")
        .arg(&ckpt)
        .args(["--prompt", "你好", "--max-new", "4"])
        .output()
        .expect("跑 generate");
    assert!(
        out.status.success(),
        "generate 失败：{:?}\nstderr={}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(!text.trim().is_empty(), "generate 没有任何输出");
    assert!(text.contains("你好"), "输出里应回显 prompt：{text}");
}

/// `sft-gen`：纯 Rust 的语料生成器——同种子逐字节相同，且是合法的角色交替格式。
#[test]
fn cli_sft_gen_is_deterministic_and_well_formed() {
    let dir = common::fresh_dir("sftgen");
    let (a, b) = (dir.join("a.txt"), dir.join("b.txt"));
    for out in [&a, &b] {
        let o = Command::new(bin())
            .args(["sft-gen", "--dialogues", "20", "--seed", "7", "--out"])
            .arg(out)
            .output()
            .expect("跑 sft-gen");
        assert!(
            o.status.success(),
            "sft-gen 失败：{}",
            String::from_utf8_lossy(&o.stderr)
        );
    }
    let a_bytes = std::fs::read(&a).unwrap();
    assert_eq!(a_bytes, std::fs::read(&b).unwrap(), "同种子必须逐字节相同");

    let text = String::from_utf8_lossy(&a_bytes);
    let blocks = text.split("\n\n").filter(|s| !s.trim().is_empty()).count();
    assert_eq!(blocks, 20, "应生成 20 段对话，实际 {blocks}");
    for (i, line) in text.lines().filter(|l| !l.trim().is_empty()).enumerate() {
        let ok = if i % 2 == 0 {
            line.starts_with("用户：")
        } else {
            line.starts_with("助手：")
        };
        assert!(ok, "第 {i} 行角色不对：{line}");
    }
}
