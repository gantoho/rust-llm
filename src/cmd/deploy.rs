//! deploy：服务相关命令实现（serve / chat）。

use crate::cli::RopeArgs;
use crate::config::Config;
use crate::data::{SFT_ASSISTANT, SFT_USER};
use crate::rng::Rng;
use crate::sample::{KvOpts, SampleOpts, StopReason, generate_with_reason};

use crate::prompt;
use crate::runlog;
use crate::serve;

use super::{describe_kv, describe_stop, resolve_model_spec, stop_label};

/// 把模型部署成 OpenAI 兼容的 HTTP API（`serve` 子命令）
///
/// 加载链路与 [`cmd_chat`] 完全一致（配置 → 分词器 → checkpoint → LoRA → RoPE），
/// 区别只在最后一步：不进 REPL，而是把模型与默认采样参数打包成 [`serve::ServeCfg`]，
/// 交给 `serve::run` 起 HTTP 服务——服务进程常驻，所以这里的 `runlog::finish()`
/// 由进程退出（Ctrl-C）时触发，而不是像 chat 那样每轮结束就收尾。
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_serve(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    host: &str,
    port: u16,
    api_key: Option<String>,
    cors: bool,
    no_web: bool,
    web_port: u16,
    system: &str,
    mut opts: SampleOpts,
    max_new: usize,
    kv: KvOpts,
    seed: u64,
    prompt_format: &str,
    rope_opt: RopeArgs,
) {
    runlog::announce("serve");
    // 没给 `--api-key` 就现场生成一个：API 始终有鉴权，随机 key 由启动日志回显，
    // 否则默认「谁都能调」在把端口暴露出去时等于裸奔
    let (api_key, auto_key) = match api_key {
        Some(k) => (Some(k), false),
        None => (Some(serve::random_api_key()), true),
    };
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let use_sft = prompt_format == "sft";
    if use_sft {
        opts.stop = prompt::SFT_STOP;
    }
    // 长度外推必须在下面读 `model.cfg.block_size` **之前**生效（同 cmd_chat），
    // 装配序列见 resolve_model_spec。
    let (mut model, tokenizer, ckpt, ckpt_path, rope_note) =
        resolve_model_spec(ckpt_path, tcfg, seed, tokenizer_path, merge_lora, Some(&rope_opt));
    let block_size = model.cfg.block_size;
    if max_new >= block_size {
        logln!(
            "[warn] --max-new {max_new} ≥ 上下文窗口 {block_size}，prompt 会被挤到没有空间，建议调小"
        );
    }
    // OpenAI 客户端会拿请求里的 `model` 和 `/v1/models` 的 id 对账，
    // 这里用 checkpoint 文件名当模型名，多个 checkpoint 服务并存时不会撞名。
    let model_name = std::path::Path::new(&*ckpt_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "model".to_string());
    let addr = format!("{host}:{port}");
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", format!("{ckpt_path}（step {}）", ckpt.step)),
            (
                "分词器文件",
                tokenizer_path.unwrap_or("自动（config.tokenizer_file → out_dir/tokenizer.json）").to_string(),
            ),
            ("模型结构", format!("{:?}", ckpt.model)),
            ("模型名（回给客户端）", model_name.clone()),
            ("系统提示", if system.is_empty() { "无".to_string() } else { system.to_string() }),
            ("prompt 模板", prompt_format.to_string()),
            ("每轮最大新 token", max_new.to_string()),
            ("上下文窗口", block_size.to_string()),
            (
                "视觉塔（图片输入）",
                match &model.cfg.vision {
                    Some(v) => format!(
                        "已启用：输入 {}×{}、patch {}×{}、每图 {} 个占位符（请求带 message.image_url 即可传图）",
                        v.image_size,
                        v.image_size,
                        v.patch_size,
                        v.patch_size,
                        v.patch_count()
                    ),
                    None => "未配置（config 的 model.vision 缺席，不接受图片输入）".to_string(),
                },
            ),
            ("seed", seed.to_string()),
            ("temperature", opts.temperature.to_string()),
            ("top_k", opts.top_k.to_string()),
            ("top_p", opts.top_p.to_string()),
            (
                "重复惩罚",
                format!("{}（回看窗口 {}）", opts.repetition_penalty, opts.repetition_window),
            ),
            ("KV cache", describe_kv(&kv)),
            (
                "RoPE / 上下文",
                rope_note.clone().unwrap_or_else(|| {
                    format!("沿用 checkpoint：窗口 {block_size}，base = {}", model.cfg.rope_base)
                }),
            ),
            ("监听地址", format!("http://{addr}")),
            (
                "鉴权",
                format!(
                    "开启（请求需带 Authorization: Bearer <key>）{}",
                    if auto_key {
                        "；未提供 --api-key，已生成随机 key（见控制台 [serve] 鉴权行）"
                    } else {
                        ""
                    }
                ),
            ),
            ("CORS", if cors { "开（浏览器页面可直连）" } else { "关" }.to_string()),
            (
                "前端测试页",
                if no_web {
                    "不启动（--no-web）".to_string()
                } else {
                    format!("http://localhost:{web_port}（随 API 一起启动）")
                },
            ),
        ],
    );
    if let Some(note) = &rope_note {
        logln!("[rope] {note}");
    }
    // 前端测试页：在进入常驻的 serve 循环之前拉起，失败只警告不影响 API
    // （key 一并注入，页面打开即自动填好——随机 key 用户没法预知）
    if !no_web {
        serve::spawn_web_frontend(port, web_port, api_key.as_deref());
    }
    // 图片生成能力：{out_dir}/vq.ckpt 存在就加载，模型采出完整图片段时解码回 PNG。
    // 纯文本模型没有这个文件，vq = None，serve 行为与从前完全一致
    let vq_path = format!("{}/vq.ckpt", tcfg.out_dir);
    let vq = if std::path::Path::new(&vq_path).exists() {
        logln!("VQ-VAE：加载 {vq_path}（serve 的图片生成通路已启用）");
        Some(crate::vqvae::Vqvae::load(&vq_path))
    } else {
        logln!("VQ-VAE：未找到 {vq_path}，serve 仅提供文本生成");
        None
    };
    serve::run(
        serve::ServeCfg {
            host: host.to_string(),
            port,
            api_key,
            cors,
            system: system.to_string(),
            use_sft,
            max_new,
            seed,
            kv,
            model_name,
            sample: opts,
            block_size,
            vision: model.cfg.vision.clone(),
        },
        model,
        tokenizer,
        vq,
    );
}

/// 交互式对话模式
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_chat(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    system: &str,
    opts: SampleOpts,
    max_new: usize,
    kv: KvOpts,
    seed: u64,
    prompt_format: &str,
    rope_opt: RopeArgs,
) {
    runlog::announce("chat");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    // SFT 模板：prompt 按训练时的对话模板拼，模型才会"回答"而不是"续写"。
    // 没做过 SFT 的预训练权重用这个模板只会一脸茫然（它没见过这些标记），要 `--prompt-format raw`。
    let use_sft = prompt_format == "sft";
    let mut opts = opts;
    if use_sft {
        opts.stop = prompt::SFT_STOP;
    }
    // 长度外推必须在下面读 `model.cfg.block_size` 分配上下文预算**之前**生效：
    // 否则 `block_size` 还是训练窗口，`--max-ctx` 扩出来的空间会被白白裁掉——
    // 装配序列见 resolve_model_spec。
    let (mut model, tokenizer, ckpt, ckpt_path, rope_note) =
        resolve_model_spec(ckpt_path, tcfg, seed, tokenizer_path, merge_lora, Some(&rope_opt));
    let mut rng = Rng::new(seed);
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", format!("{ckpt_path}（step {}）", ckpt.step)),
            (
                "分词器文件",
                tokenizer_path.unwrap_or("自动（config.tokenizer_file → out_dir/tokenizer.json）").to_string(),
            ),
            ("模型结构", format!("{:?}", ckpt.model)),
            ("系统提示", if system.is_empty() { "无".to_string() } else { system.to_string() }),
            ("prompt 模板", prompt_format.to_string()),
            ("每轮最大新 token", max_new.to_string()),
            ("seed", seed.to_string()),
            ("temperature", opts.temperature.to_string()),
            ("top_k", opts.top_k.to_string()),
            ("top_p", opts.top_p.to_string()),
            (
                "重复惩罚",
                format!("{}（回看窗口 {}）", opts.repetition_penalty, opts.repetition_window),
            ),
            ("KV cache", describe_kv(&kv)),
            (
                "RoPE / 上下文",
                rope_note.clone().unwrap_or_else(|| {
                    format!("沿用 checkpoint：窗口 {}，base = {}", model.cfg.block_size, model.cfg.rope_base)
                }),
            ),
        ],
    );
    if let Some(note) = &rope_note {
        logln!("[rope] {note}");
    }

    logln!("交互式对话模式（输入文本后按回车生成，输入 :quit 退出）");
    logln!(
        "参数：temperature={} top-k={} top-p={} 重复惩罚={}（窗口 {}）",
        opts.temperature, opts.top_k, opts.top_p, opts.repetition_penalty, opts.repetition_window
    );
    if !system.is_empty() {
        logln!("系统提示：{}", system);
    }
    logln!("---");

    // 上下文窗口要在「system prompt」「对话历史」「本轮生成」三者之间分配。
    //
    // 分配顺序（前面的优先保）：
    //   1. system prompt —— 永远保留。它是序列最开头的"注意力锚点"（attention sink），
    //      丢掉它不只是失忆，还会让整条序列的注意力分布失稳
    //   2. 本轮生成 —— 预留 max_new 个 token，让它整段留在窗口内。缓存是滑动窗口，
    //      生成本身不受 block_size 限制（不会提前停），但窗口只保留最近 block_size 个 token：
    //      prompt + 生成超过它时，最前面的内容会滑出模型视野（system prompt 与早期历史被"遗忘"）
    //   3. 对话历史 —— 剩下的额度都给它，不够就按 token 数从最老的开始丢
    let block_size = model.cfg.block_size;
    if max_new >= block_size {
        logln!(
            "[warn] --max-new={max_new} 不小于上下文窗口 {block_size}：本轮生成到后半程时，\
             最早的输入已滑出视野（滑动窗口只保留最近 {block_size} 个 token）"
        );
    }
    let prompt_budget = block_size.saturating_sub(max_new);

    let system = system.trim();
    let n_system = tokenizer.encode(system).len();
    let n_system_prefix = if system.is_empty() { 0 } else { n_system + 1 }; // +1 是它后面的 '\n'
    if n_system_prefix >= prompt_budget {
        logln!(
            "[warn] system prompt 已有 {n_system} 个 token，占满了 {prompt_budget} 的输入预算，本轮没有空间留给对话历史"
        );
    }

    // 裁剪不能静默：首次立即提示，之后每 HISTORY_REPORT_EVERY 次汇总一次，
    // 否则跑几十轮下来用户不知道"已经无记忆多久了"
    const HISTORY_REPORT_EVERY: usize = 10;
    let mut first_trim_turn: Option<usize> = None;
    let mut trim_count = 0usize;
    let mut dropped_tokens_total = 0usize;
    let mut warned_prompt_overflow = false;

    // 只存「对话历史」，不含 system：system 在拼接 prompt 时单独放在最前面，
    // 这样它就不会被 `trim_history` 当成最老的内容丢掉。
    let mut context_history = String::new();
    let mut turn = 0usize;

    loop {
        print!("> ");
        use std::io::Write;
        std::io::stdout().flush().ok();

        let mut input = String::new();
        if std::io::stdin().read_line(&mut input).is_err() {
            break;
        }
        let input = input.trim();
        if input == ":quit" || input == ":exit" || input.is_empty() {
            break;
        }
        turn += 1;

        // 构造 prompt：system + 历史 + 本轮。预算分配（system 永保 → 生成预留 → 历史）
        // 与 SFT/raw 两种模板都在 `prompt` 模块里，HTTP serve 走同一套代码。
        let assembled = prompt::assemble(&tokenizer, system, &context_history, input, use_sft, prompt_budget);
        let history = assembled.history;
        if let Some((n_before, n_after)) = assembled.trimmed {
            let history_budget = assembled.history_budget;
            let dropped = n_before.saturating_sub(n_after);
            trim_count += 1;
            dropped_tokens_total += dropped;
            match first_trim_turn {
                None => {
                    first_trim_turn = Some(turn);
                    logln!(
                        "[info] 对话历史超出预算（{n_before} > {history_budget} token），\
                         从本轮起丢弃最早的轮次（本次丢弃 {dropped} token，保留 {n_after} token）；\
                         想保留更多历史请调小 --max-new"
                    );
                }
                Some(t0) if trim_count % HISTORY_REPORT_EVERY == 0 => {
                    logln!(
                        "[info] 已丢弃历史 {trim_count} 次（起始于第 {t0} 轮），\
                         累计丢弃 {dropped_tokens_total} token，当前保留 {n_after} token"
                    );
                }
                Some(_) => {}
            }
        }

        let prompt = assembled.prompt;
        runlog::append(&format!("\n[第 {turn} 轮] 输入：{input}"));

        let n_prompt = tokenizer.encode(&prompt).len();
        if n_prompt > block_size && !warned_prompt_overflow {
            warned_prompt_overflow = true;
            logln!(
                "[warn] 本轮输入本身已有 {n_prompt} 个 token，超过上下文窗口 {block_size}，只有末尾内容参与生成"
            );
        }

        // 每轮都重建缓存：上一轮的缓存对不上本轮重新裁剪过的 prompt，
        // 沿用会让模型看到"已被丢弃的历史"（见 `generate` 里的 KV 复用说明）。
        let out = generate_with_reason(&model, &tokenizer, &prompt, max_new, &opts, kv, &mut rng);

        // 生成段由 `generate_with_reason` 直接给出，不必再按 `prompt.len()` 的字节偏移去切全文
        let response = out.generated.trim();
        // 收尾原因**每轮**都记：只记空输出的那几轮，非空回答"是采到 EOS 还是被停止标记截住"
        // 就查不到，只能靠猜；而两者的差别会体现在下一轮的上下文里。
        let why = stop_label(out.reason);
        if response.is_empty() {
            // 空白必须可解释：模型一个字都没说，与"程序根本没跑"在终端上是同一行空白
            logln!("[本轮无输出] {}", describe_stop(out.reason));
            runlog::append(&format!("[第 {turn} 轮] 输出：（空）{why}"));
        } else {
            logln!("{response}");
            runlog::append(&format!("[第 {turn} 轮] 输出：{response}（收尾：{why}）"));
        }

        // 更新历史：历史 = 旧历史 + 本轮问答（不含 system，它单独拼接）。
        // 不要重复拼 `input`：prompt 里已含本轮输入，那会让每轮输入在窗口里占两份。
        // 长度控制交给下一轮开头的裁剪（按真实 token 数、SFT 模式下按轮）。
        context_history = history;
        if use_sft {
            // 空回答整轮不写回：历史里一旦出现「助手：\n\n用户：」（助手什么都没说），
            // 就成了训练语料里根本没有的 in-context 示范——模型会跟着一起沉默，
            // 于是从某一轮起连续输出空白。裁剪按轮切，跳过整轮只影响"还能记住几轮"。
            //
            // 被 `--max-new` 硬截断的回答同样不写回。训练语料里助手回合都是短句，
            // 后面紧跟「用户：」或 EOS；而截断的回答是一段没有收尾的半截话
            // （实测末尾停在「…公司业」）。把这种"说了 200 个 token 还不收尾"的状态
            // 喂回去，下一轮模型第一步就采到 EOS，此后每轮全空——见 `[本轮无输出]`。
            let truncated = out.reason == StopReason::MaxNew;
            if !response.is_empty() && truncated {
                logln!("[info] 本轮回答被 --max-new={max_new} 截断（未收尾），不计入对话历史");
            }
            if !response.is_empty() && !truncated {
                if !context_history.is_empty() {
                    context_history.push('\n');
                }
                // 写成模板形态，下一轮就能被 `trim_sft_history` 按轮切开
                context_history.push_str(&format!("{SFT_USER}\n{input}\n{SFT_ASSISTANT}\n{response}"));
            }
        } else {
            if !context_history.is_empty() {
                context_history.push('\n');
            }
            context_history.push_str(input);
            if !response.is_empty() {
                context_history.push('\n');
                context_history.push_str(response);
            }
        }
    }
    runlog::append(&format!("\n对话结束：共 {} 轮", turn));
    runlog::finish();
}
