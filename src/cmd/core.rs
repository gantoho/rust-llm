//! core：核心命令实现（train / eval / quant / generate / sft / finetune / preset / bench）。

use crate::cli::RopeArgs;
use crate::comm::DistRuntime;
use crate::config::Config;
use crate::data::{BatchSource, DataLoader, SFT_ASSISTANT, SFT_END, SFT_USER, SftLoader, VlmLoader, load_documents, load_text, load_texts};
use crate::model::{Transformer, TransformerConfig};
use crate::module::Module;
use crate::quant::{CalibOpts, HessOpts, QBits, QuantMethod, QuantOpts};
use crate::rng::Rng;
use crate::sample::{KvOpts, SampleOpts, generate};
use crate::tokenizer::Tokenizer;

use crate::checkpoint;
use crate::config;
use crate::data;
#[cfg(feature = "gpu")]
use crate::gpu;
use crate::runlog;
use crate::sample;
use crate::train;

use super::{build_tokenizer, describe_kv, load_model_and_tokenizer, maybe_merge_lora, read_text, resolve_ckpt, resolve_model_spec, truncate_chars};

/// 命令行上的权重位宽名字 -> 枚举（`quant` 子命令用；这里不存在"不量化"的选项，
/// 因为不量化就不该走这个子命令）
fn q_bits_from_name(name: &str) -> QBits {
    match name {
        "int8" => QBits::Int8,
        "int4" => QBits::Int4,
        other => panic!("未知的量化位宽 '{other}'（可选：int8 / int4）"),
    }
}

/// 命令行上的量化算法名字 -> 枚举
fn q_method_from_name(name: &str) -> QuantMethod {
    match name {
        "rtn" => QuantMethod::Rtn,
        "hess" => QuantMethod::Hess,
        "awq" => QuantMethod::Awq,
        other => panic!("未知的量化算法 '{other}'（可选：rtn / hess / awq）"),
    }
}

/// 命令行上的布尔字面量 -> `bool`（只认 `true` / `false`）。
///
/// 不接受 `1` / `yes` / 空串之类的变体：量化参数直接决定产出的权重，
/// 一个拼错的写法应当当场报错，而不是被静默当成"没给"从而悄悄用上默认值。
fn bool_from_name(flag: &str, name: &str) -> bool {
    match name {
        "true" => true,
        "false" => false,
        other => panic!("{flag} 只接受 true / false，收到 '{other}'"),
    }
}

/// 训练：`train --config config/config.json [--resume ckpt]`
pub(crate) fn cmd_train(config_path: &str, resume: Option<&str>) {
    runlog::announce("train");
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("resume", resume.unwrap_or("无（从头训练）").to_string()),
        ],
    );
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;

    // ---- 多进程（DP 接线）----
    // env 里有 `WORLD_SIZE > 1` 就按 torch 约定组网（起进程的命令见
    // `docs/38-分布式训练.md` 的「启动多进程训练」一节）。
    // 单进程时是 `None`，下面所有 `dist` 分支都不进，训练路径与接分布式之前逐位一致。
    // `from_env` 会**建 TCP 全连接**，所以必须在任何集合通信之前建好：建连本身是
    // 同步的（小 rank 拨大 rank），各进程在这里互相等对方监听就绪。
    let dist = DistRuntime::from_env_if_distributed();
    // 只有 rank 0 负责落盘（分词器 / 指标 / checkpoint），见 `train_transformer`。
    let primary = dist.as_ref().map_or(true, |d| d.is_primary());
    if let Some(d) = dist.as_ref() {
        let rank = d.comm.rank();
        let c = d.cfg.coord(rank);
        // TP / PP 的接线还没做（任务清单 #6 / #7）。这里宁可起跑前拒绝，也不要
        // 「每个 rank 都拿一份完整模型」跑完——那种结果不报错，只是和切分语义不符。
        assert!(
            d.cfg.tp == 1 && d.cfg.pp == 1,
            "TP/PP 接线尚未完成（任务清单 #6/#7）：当前只支持 DIST_TP=1 且 DIST_PP=1 的纯数据并行，\
             收到 tp={} pp={}",
            d.cfg.tp,
            d.cfg.pp
        );
        logln!(
            "分布式：rank {}/{} ｜ dp {} tp {} pp {} ｜ 本机 local_rank {} ｜ \
             DP 组 {} 个 rank 做梯度同步（TP/PP 组暂未接线）",
            rank,
            d.comm.world(),
            c.dp,
            c.tp,
            c.pp,
            d.local_rank,
            d.cfg.dp_group(rank).len(),
        );
    }

    #[cfg(feature = "gpu")]
    if gpu::is_available() {
        logln!("GPU: {}（{}）", gpu::name(), gpu::backend());
    } else {
        logln!("未检测到可用 GPU，本次训练走 CPU");
    }

    let train_docs = load_documents(&tcfg.train_file);
    let val_text = tcfg.val_file.as_deref().map(read_text);
    // 分词器语料 = 所有文档以 "\n" 相连，与 load_text 的拼接口径逐字一致
    let mut tokenizer = build_tokenizer(tcfg, &train_docs.join("\n"), 0); // 训练时词表由分词器决定

    // 词表大小 0 表示"由分词器决定"
    let mut model_cfg = cfg.model.clone();

    // 图文训练：把图片 token 区间挂到分词器上，并把占位符首 id 写进视觉配置。
    // 必须在**保存分词器、计算词表大小、构建模型**之前完成——三者都依赖这个区间
    //（tokenizer.json 要带 image 字段、vocab_size 要含图片 token、视觉塔注入要用 ph_first）。
    if tcfg.vlm_file.is_some() {
        let vcfg = model_cfg
            .vision
            .as_mut()
            .expect("配置了 train.vlm_file，但 model.vision 为空——图文训练必须有视觉塔");
        let p = vcfg.patch_count();
        // K = VQ 码本大小：模型配置了 model.vq 就一并挂上码本 token（图像生成通路），
        // 否则 K=0 只做理解。挂载必须发生在保存分词器/计算词表之前。
        let k = model_cfg
            .vq
            .as_ref()
            .map(|v| v.codebook_size)
            .unwrap_or(0);
        let im = tokenizer
            .attach_image_tokens(p, k)
            .expect("分词器没有特殊 token，无法挂载图片区间");
        vcfg.ph_first = im.ph_first;
        if k > 0 {
            logln!(
                "图片 token 已挂载：P={p} 个占位符 + K={k} 码本，ph_first={}，cb_first={}（VQ 预训练见下方日志）",
                im.ph_first,
                im.cb_first
            );
        } else {
            logln!("图片 token 已挂载：P={p} 个占位符，ph_first={}（K=0，仅图片理解）", im.ph_first);
        }
    }

    // 训练完成后保存分词器（out_dir 不存在时 save 会自动创建）。
    // 多进程下只有 rank 0 落盘：N 个副本写同一个 tokenizer.json 会互相覆盖，
    // 更糟的是**边写边被读**——别的 rank 若正在加载就会读到半截 JSON。
    // 各 rank 手里的分词器仍由同一份语料、同一个种子独立构建，内容逐位相同。
    let tok_path = tcfg.tokenizer_file.clone()
        .unwrap_or_else(|| format!("{}/tokenizer.json", tcfg.out_dir));
    if primary {
        tokenizer.save(&tok_path);
    }
    logln!("分词器已保存到 {tok_path}");

    if model_cfg.vocab_size == 0 {
        model_cfg.vocab_size = tokenizer.vocab_size();
    }

    let mut rng = Rng::new(tcfg.seed);
    let mut model = Transformer::new(model_cfg.clone(), &mut rng);
    // 续训一个 LoRA 存档时同样要先重建适配层：checkpoint 头里记着 rank/alpha，
    // 少了这套参数名，load_with_opt 恢复参数时会直接断言失败。
    if let Some(path) = resume {
        let ckpt = checkpoint::load_header(path);
        if let Some(lora) = ckpt.lora.as_ref() {
            logln!(
                "续训：{path} 是 LoRA 存档（rank={} alpha={}），重建适配层后再恢复参数",
                lora.rank,
                lora.alpha
            );
            model.apply_lora(lora, &mut rng);
        }
    }
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    runlog::fields(
        "数据与模型",
        &[
            ("训练语料", tcfg.train_file.clone()),
            (
                "验证语料",
                tcfg.val_file
                    .clone()
                    .unwrap_or_else(|| "（未配置，自动从训练文本末尾切 10%）".to_string()),
            ),
            (
                "分词器",
                format!("{}（词表 {}）→ {tok_path}", tokenizer.kind(), tokenizer.vocab_size()),
            ),
            ("模型结构", format!("{model_cfg:?}")),
            ("模型参数", format!("{param_count}")),
            ("输出目录", tcfg.out_dir.clone()),
            (
                "指标 CSV",
                tcfg.log_file.clone().unwrap_or_else(|| "不记录".to_string()),
            ),
        ],
    );
    // 图文训练走 VlmLoader（JSONL 样本 + 像素张量），否则维持纯文本加载器
    let loader: Box<dyn data::BatchSource> = if let Some(vlm_path) = tcfg.vlm_file.as_deref() {
        let image_size = model_cfg
            .vision
            .as_ref()
            .expect("配置了 train.vlm_file，但 model.vision 为空")
            .image_size;
        let mut vl = VlmLoader::from_jsonl(
            vlm_path,
            &tokenizer,
            model_cfg.block_size,
            tcfg.batch_size,
            image_size,
        );
        logln!(
            "VLM 语料：{} 条样本，监督 token 占比 {:.1}%",
            vl.num_samples(),
            vl.supervised_ratio() * 100.0
        );
        // 图像生成通路：先备好 VQ-VAE（有存档直接加载，否则预训练并保存），
        // 再把每条样本的占位符段替换为码本 token，供生成批次采样。
        if let Some(vq_cfg) = model_cfg.vq.as_ref() {
            assert_eq!(
                vq_cfg.image_size, image_size,
                "model.vq.image_size 必须等于 vision.image_size（像素按 vision.image_size 加载）"
            );
            let vq_path = format!("{}/vq.ckpt", tcfg.out_dir);
            let vq = if std::path::Path::new(&vq_path).exists() {
                logln!("VQ-VAE：加载已有存档 {vq_path}（跳过预训练）");
                crate::vqvae::Vqvae::load(&vq_path)
            } else {
                assert!(
                    tcfg.vq_steps > 0,
                    "{vq_path} 不存在且 train.vq_steps = 0——没有码本就无法挂载生成 token"
                );
                logln!("VQ-VAE：开始预训练（{} 步）→ {vq_path}", tcfg.vq_steps);
                let vq = crate::vqvae::Vqvae::new(vq_cfg.clone(), &mut rng);
                train::train_vqvae(&vq, vl.images(), tcfg, &mut rng);
                vq.save(&vq_path);
                logln!("VQ-VAE：预训练完成，已保存到 {vq_path}");
                vq
            };
            vl.attach_generation(&vq, &tokenizer);
            logln!(
                "生成序列已挂载：{} 条样本的占位符段替换为 VQ 码本 token",
                vl.num_samples()
            );
        }
        Box::new(vl)
    } else {
        Box::new(DataLoader::from_documents(
            &train_docs,
            val_text.as_deref(),
            &tokenizer,
            model_cfg.block_size,
            tcfg.batch_size,
        ))
    };
    let best = train::train_transformer(
        &model,
        &tokenizer,
        loader.as_ref(),
        tcfg,
        Some(&tcfg.out_dir),
        resume,
        &mut rng,
        dist.as_ref(),
    );
    runlog::append(&format!("训练结束：best val loss = {best:.4}"));
    runlog::finish();
}

/// 评估：在验证集上计算 loss 与困惑度
pub(crate) fn cmd_eval(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    template: &str,
    sft_file: Option<&str>,
    mask_response: bool,
) {
    runlog::announce("eval");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    // 评估口径：text 与 mask_response 不兼容（纯文本没有"回答段"的概念）
    assert!(
        template == "sft" || !mask_response,
        "--mask-response 只在 --template sft 下有意义（纯文本口径没有回答段）"
    );
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", ckpt_path.to_string()),
            (
                "分词器文件",
                tokenizer_path.unwrap_or("自动（config.tokenizer_file → out_dir/tokenizer.json）").to_string(),
            ),
            ("评估批数", tcfg.eval_iters.to_string()),
            (
                "评估口径",
                match template {
                    "sft" => format!(
                        "SFT 对话语料（{}，loss 只计回答段）",
                        sft_file.or(tcfg.sft_file.as_deref()).unwrap_or("未指定")
                    ),
                    _ => "纯文本验证集（全 token）".to_string(),
                },
            ),
        ],
    );
    // 路径在 runlog 之前已解析过，这里传 `Some(&*ckpt_path)` 是同一路径的幂等解析；
    // eval 不做长度外推，`rope_opt` 传 `None`（末两项直接丢弃）。
    let (mut model, tokenizer, ckpt, _, _) =
        resolve_model_spec(Some(&*ckpt_path), tcfg, tcfg.seed, tokenizer_path, merge_lora, None);

    // 按口径选 loader：SFT 模型必须用对话口径评估——纯文本口径既没对上训练分布，
    // 也没算回答段的监督 loss，跨口径的 loss 没有可比性。
    let loader: Box<dyn BatchSource> = match template {
        "sft" => {
            let paths = sft_file
                .map(str::to_string)
                .or_else(|| tcfg.sft_file.clone())
                .unwrap_or_else(|| {
                    panic!("--template sft 需要对话语料：用 --sft-file，或在 {config_path} 里设置 train.sft_file")
                });
            let texts = load_texts(&paths);
            let mut l = SftLoader::from_texts(&texts, &tokenizer, ckpt.model.block_size, tcfg.batch_size);
            l.set_mask_response(mask_response);
            Box::new(l)
        }
        _ => {
            let train_docs = load_documents(&tcfg.train_file);
            let val_text = tcfg.val_file.as_deref().map(read_text);
            Box::new(DataLoader::from_documents(
                &train_docs,
                val_text.as_deref(),
                &tokenizer,
                ckpt.model.block_size,
                tcfg.batch_size,
            ))
        }
    };

    let mut eval_rng = Rng::new(tcfg.seed); // 固定种子，结果可复现
    let loss = train::eval_loss(&model, &*loader, tcfg.eval_iters, &mut eval_rng);
    logln!(
        "评估 step {}：val_loss {:.4} | perplexity {:.2}（{} 个 token 的验证集上采 {} 批）",
        ckpt.step,
        loss,
        loss.exp(),
        loader.num_val_tokens(),
        tcfg.eval_iters
    );
    runlog::finish();
}

/// 缺省校准语料目录：仓库自带的混合语料（中英文章 / 对话 / 代码 / 诗歌）。
///
/// 为什么不用配置里的 `train_file`：那份语料是拿去训模型的，量化校准要的是
/// "**推理时会遇到的输入**"。两者在领域上可能差很远（例如用中文小说训的模型
/// 被拿去跑英文代码），此时拿训练语料当校准集，`H` 与激活均值估的就是错的分布。
/// 所以缺省给一份覆盖面最广的语料，真要贴合部署领域就用 `--calib-file` 显式指定。
const DEFAULT_CALIB_DIR: &str = "data/corpus/";

/// 权重量化：`quant --config config/config.json --bits int8 --method hess --eval`
///
/// 完整流程：加载 f32 模型 → （非 RTN 时）在校准集上采激活统计 → 逐层量化并出报告 →
/// 可选地对比量化前后的验证 loss / 困惑度 → **把量化结果烘焙回 f32** 再落盘。
///
/// 存档里为什么仍写 f32：checkpoint 格式的语义是"按名字对齐的 f32 参数表"，
/// 让它长出第二种参数编码，会把加载、续训、合并、分词器对齐这一整条链路都变成
/// 两种可能的分支。量化真正的收益在**推理时的显存与带宽**，而推理本来就该在
/// 进程里重建量化（见 [`checkpoint::requantize_after_load`]），没必要为此污染格式。
/// 存档头里的 `quant` 字段负责记住"当初是怎么量化的"。
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_quant(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    bits_name: &str,
    method_name: &str,
    calib_file: Option<&str>,
    calib_samples: usize,
    calib_tokens: usize,
    act_order_name: &str,
    damp: f32,
    block: usize,
    alpha: Option<f32>,
    out: Option<&str>,
    do_eval: bool,
) {
    runlog::announce("quant");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    let ckpt_path = resolve_ckpt(ckpt_path, &tcfg.out_dir);
    let bits = q_bits_from_name(bits_name);
    let method = q_method_from_name(method_name);
    let act_order = bool_from_name("--act-order", act_order_name);
    // 分组方向固定为 `Col`（每个输出通道一个 scale）：这是权重量化的通行口径，
    // 见 [`QuantOpts::axis`]。算法参数一次性装进 `QuantOpts` 往下传——
    // 这样"用户选的算法"与"该算法的参数"永远是同一个对象里的两份，不会出现
    // "命令行给了 Hessian 量化的 damp，但实际跑的是 AWQ"这种没人会发现的错配。
    let opts = QuantOpts {
        hess: HessOpts {
            act_order,
            damp,
            block,
        },
        awq_alpha: alpha,
        ..QuantOpts::default()
    };
    let out_path = out
        .map(|s| s.to_string())
        .unwrap_or_else(|| format!("{}/quant.ckpt", tcfg.out_dir));
    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            ("checkpoint", ckpt_path.to_string()),
            ("位宽", bits.name().to_string()),
            ("算法", format!("{}{}", method.name(), method.describe())),
            ("Hessian 量化选项", opts.hess.describe()),
            (
                "校准集",
                match method {
                    QuantMethod::Rtn => "不适用（RTN 只用权重本身）".to_string(),
                    _ => calib_file.unwrap_or(DEFAULT_CALIB_DIR).to_string(),
                },
            ),
            (
                "校准预算",
                format!("{calib_samples} 个窗口 × 上下文长度（硬上限 {calib_tokens} token，0 = 不设限）"),
            ),
            (
                "AWQ alpha",
                match alpha {
                    Some(a) => a.to_string(),
                    None => format!(
                        "网格搜索 0~1，步长 {}（论文默认 {}）",
                        crate::quant::AWQ_ALPHA_GRID_STEP,
                        crate::quant::AWQ_ALPHA
                    ),
                },
            ),
            ("输出", out_path.clone()),
            ("量化前后对比", if do_eval { "是" } else { "否（加 --eval 开启）" }.to_string()),
        ],
    );
    let (mut model, tokenizer, ckpt) = load_model_and_tokenizer(&ckpt_path, tcfg, tcfg.seed, tokenizer_path);
    // LoRA 存档必须先合并：量化会跳过挂着适配器的层（见 [`Linear::quantize_weight`]），
    // 不合并的话报告只会打印"N 层被跳过"，等于什么都没做。
    if model.has_lora() {
        maybe_merge_lora(&mut model, true);
    }

    // 1) 量化前基线（固定种子的 Rng，保证前后两次取的是同一批验证数据）。
    //    **loader 只在这里（`--eval` 时）才建**：它要把整份训练语料编码成 token 流，
    //    语料上百万字符时这一步比量化本身还慢，而"只想看看压缩比和逐层误差"的
    //    调用根本不需要它。所以把 `Option` 一路留着，不用就不付这份代价。
    let mut eval_rng = Rng::new(tcfg.seed);
    let (loss_before, eval_loader) = if do_eval {
        let train_docs = load_documents(&tcfg.train_file);
        let val_text = tcfg.val_file.as_deref().map(read_text);
        let loader = DataLoader::from_documents(
            &train_docs,
            val_text.as_deref(),
            &tokenizer,
            ckpt.model.block_size,
            tcfg.batch_size,
        );
        let l = train::eval_loss(&model, &loader, tcfg.eval_iters, &mut eval_rng);
        logln!("量化前：val_loss {l:.4} | perplexity {:.2}", l.exp());
        (Some(l), Some(loader))
    } else {
        (None, None)
    };

    // 2) 校准（RTN 不需要，跳过能省一整遍前向）。
    //    token 预算 = 窗口数 × 上下文长度：校准是按**窗口**喂进去的，所以这是最贴近
    //    调用方直觉的口径；`--calib-tokens` 只作为硬上限（0 = 不设限）。
    let seq = ckpt.model.block_size.max(1);
    let window_budget = calib_samples.max(1).saturating_mul(seq);
    let max_tokens = if calib_tokens == 0 {
        window_budget
    } else {
        window_budget.min(calib_tokens)
    };
    let stats = if method.needs_calib() {
        let path = calib_file.unwrap_or(DEFAULT_CALIB_DIR);
        // 缺省语料被精简掉时要**当场说清怎么补救**：`load_text` 对不存在的路径
        // 只是在读文件时报"无法读取数据文件"，不会告诉你"这是缺省值、可以用
        // --calib-file 换一个"。
        assert!(
            std::path::Path::new(path).exists(),
            "校准语料 {path} 不存在。请用 --calib-file 指定一个文本文件/目录/通配符路径\
             （缺省语料在 {DEFAULT_CALIB_DIR}）"
        );
        // 只把需要的**前缀**切出来喂给分词器：缺省语料 `data/corpus/` 有近 4 MB，
        // 整个编码一遍再 truncate 掉 99% 的 token，等于把时间全花在丢弃上。
        // 字符数按 token 预算的 8 倍留余量——任何子词/字符级分词器一个 token 都不会
        // 吃掉 8 个字符，所以这段前缀一定够填满预算（宁可多编一点，也不要因为截短了
        // 而少喂校准数据：`H` 估得不准会被 `H⁻¹` 放大）。
        let text = truncate_chars(&load_text(path), max_tokens.saturating_mul(8)).to_string();
        let s = model.calibrate(
            &tokenizer,
            &[text],
            CalibOpts {
                max_tokens,
                max_seq: seq,
                ..CalibOpts::default()
            },
        );
        let n = s.per_layer.first().map(|(_, c)| c.n_tokens).unwrap_or(0);
        let full = s
            .per_layer
            .iter()
            .filter(|(_, c)| c.hessian.as_ref().is_some_and(|h| h.is_full()))
            .count();
        // 把 Hessian 的**内存代价**打出来：这是整条流水线里唯一会随 `in_features²`
        // 增长的开销，性能问题里最容易被忽略的一项。超预算的层应自动退化成对角。
        let h_bytes: usize = s
            .per_layer
            .iter()
            .filter_map(|(_, c)| c.hessian.as_ref())
            .map(|h| h.byte_len())
            .sum();
        logln!(
            "校准：{path} 上跑了 {n} 个 token（预算 {max_tokens}，窗口 {seq}），\
             命中 {} 层（{full} 层存全矩阵 Hessian，共 {:.1} MB，其余退化为对角）",
            s.per_layer.len(),
            h_bytes as f64 / (1024.0 * 1024.0)
        );
        assert!(
            s.per_layer.iter().all(|(_, c)| c.n_tokens > 0),
            "有层没采到激活统计——继续下去它会静默退回 RTN，报告也看不出来"
        );
        Some(s)
    } else {
        None
    };

    // 3) 量化 + 报告
    let reports = model.quantize_weights(bits, method, stats.as_ref(), opts);
    // 量化表示与逐层报告必须同时成立：有报告却没带上量化表示 = 权重白量化了，
    // 这种"静默无效"比直接报错更危险（下游还在按报告的压缩率做显存预算）。
    assert_eq!(
        model.is_quantized(),
        !reports.is_empty(),
        "量化表示与逐层报告不一致：报告 {} 层，is_quantized = {}",
        reports.len(),
        model.is_quantized()
    );
    let summary = model.quant_summary(reports);
    logln!("{}", summary.describe());
    // 逐层报告按**相对误差降序**打印：量化的问题几乎总是集中在少数几层上，
    // 按名字顺序打出来会让人在几十行里自己找最大值（而且很容易看漏）。
    let mut layers = summary.layers.clone();
    layers.sort_by(|a, b| {
        b.rel_err
            .partial_cmp(&a.rel_err)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    logln!("逐层误差（按相对误差降序，最多 16 行）：");
    for r in layers.iter().take(16) {
        logln!("  {}", r.describe());
    }
    if layers.len() > 16 {
        logln!("  …（其余 {} 层误差更小）", layers.len() - 16);
    }

    // 4) 量化后的验证集对比
    if let (Some(before), Some(loader)) = (loss_before, eval_loader.as_ref()) {
        let after = train::eval_loss(&model, loader, tcfg.eval_iters, &mut eval_rng);
        logln!(
            "量化后：val_loss {after:.4} | perplexity {:.2}（Δ loss {:+.4}，Δ ppl {:+.2}）",
            after.exp(),
            after - before,
            after.exp() - before.exp()
        );
    }

    // 5) 烘焙回 f32 再落盘（见本函数文档）
    model.dequantize_weights();
    // 权重 only 档：量化产物没有优化器状态可言，只写参数段（体积 1× 参数量而非 3×）。
    // 头部 `weights_only: true` 让读取端按一段校验；续训方读到它会把动量补零，
    // 拿到一个全新 AdamW 的合法起点（见 checkpoint::save_weights）。
    checkpoint::save_weights(&out_path, &model, ckpt.step, ckpt.best_val_loss);
    logln!("已写出：{out_path}（权重为反量化后的 f32，头部记录量化参数，权重 only 格式）");
    runlog::finish();
}

/// 生成：`generate --config config/config.json --ckpt ckpt --prompt "..."`
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_generate(
    config_path: &str,
    ckpt_path: Option<&str>,
    tokenizer_path: Option<&str>,
    merge_lora: bool,
    prompt: &str,
    image: Option<&str>,
    max_new: usize,
    opts: SampleOpts,
    seed: u64,
    kv: KvOpts,
    beam: Option<usize>,
    length_penalty: f32,
    rope_opt: RopeArgs,
) {
    runlog::announce("generate");
    let cfg = Config::load(config_path);
    runlog::json(&format!("完整配置（{config_path} 解析后）"), &cfg);
    let tcfg = &cfg.train;
    // 装配序列（解析路径 → 加载 → 合并 LoRA → RoPE 覆盖）见 resolve_model_spec；
    // RoPE 覆盖已在 runlog 打「模型结构」之前生效。
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
            ("prompt", format!("{prompt:?}")),
            (
                "图片理解输入",
                image.unwrap_or("无（纯文本）").to_string(),
            ),
            ("max_new", max_new.to_string()),
            ("seed", seed.to_string()),
            (
                "生成方式",
                match beam {
                    Some(b) => format!("Beam Search（beam_size={b}，length_penalty={length_penalty}）"),
                    None => "采样".to_string(),
                },
            ),
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

    // 采样分支顺带收集模型生成的图片段（码本 token），beam 分支不产出
    let mut gen_images: Vec<Vec<usize>> = Vec::new();
    let out = if let Some(beam_size) = beam {
        // Beam Search 生成（每条 beam 持有独立缓存，见 `sample::beam_search`）
        if image.is_some() {
            logln!("[image] Beam Search 不支持图片理解输入，忽略 --image（请走采样分支）");
        }
        logln!(
            "Beam Search 生成（beam_size={} length_penalty={}，{}）：",
            beam_size, length_penalty, describe_kv(&kv)
        );
        sample::beam_search(
            &model,
            &tokenizer,
            prompt,
            max_new,
            beam_size,
            length_penalty,
            kv,
        )
    } else {
        // 采样生成
        logln!(
            "生成（temperature={} top-k={} top-p={} 重复惩罚={}（窗口 {}），{}）：",
            opts.temperature,
            opts.top_k,
            opts.top_p,
            opts.repetition_penalty,
            opts.repetition_window,
            describe_kv(&kv)
        );
        // 图片理解：--image 时把像素在 prefill 首步注入视觉塔（见 Generator::with_pixels）。
        // prompt 不含 <|image|> 就在尾部补一个，给像素一个落脚的占位段
        let mut full_prompt = prompt.to_string();
        if image.is_some() {
            let vcfg = model
                .cfg
                .vision
                .as_ref()
                .expect("--image 需要模型带视觉塔（model.vision 为空）");
            if !full_prompt.contains(crate::tokenizer::IMAGE_LITERAL) {
                full_prompt.push_str(crate::tokenizer::IMAGE_LITERAL);
                logln!(
                    "[image] prompt 未含 {}，已在尾部补上（patch {}×{}）",
                    crate::tokenizer::IMAGE_LITERAL,
                    vcfg.image_size,
                    vcfg.patch_size
                );
            }
        }
        let mut g = sample::Generator::new(&model, &tokenizer, &full_prompt, max_new, &opts, kv, &mut rng);
        if let Some(path) = image {
            let vcfg = model.cfg.vision.as_ref().unwrap();
            let px = crate::vision::load_image(path, vcfg.image_size);
            logln!("[image] 已加载 {path}（{}×{}，CHW [-1,1]）", vcfg.image_size, vcfg.image_size);
            g = g.with_pixels(px);
        }
        g.run();
        gen_images = g.generated_images();
        g.into_output().text
    };
    logln!("{out}");

    // 图片生成：模型采出完整的码本 token 段时，用 VQ-VAE 解码成像素存成 PNG
    if !gen_images.is_empty() {
        let vq_path = format!("{}/vq.ckpt", tcfg.out_dir);
        let vq = crate::vqvae::Vqvae::load(&vq_path);
        for (i, ids) in gen_images.iter().enumerate() {
            let px = vq.decode(ids, 1);
            let path = format!("{}/gen_image_{i}.png", tcfg.out_dir);
            crate::vision::save_image(&px, vq.cfg.image_size, &path);
            logln!("[image] 生成图片已保存：{path}（{} 个码本 token）", ids.len());
        }
    }
    runlog::append(&format!("[done] 输出总长 {} 字符（含 prompt）", out.chars().count()));
    runlog::finish();
}

/// SFT 的步数上限与评估间隔：按 **SFT 语料自己的 epoch** 推导，不沿用预训练那套数。
///
/// 配置里的 `steps`（4000）和 `eval_every`（200）是配 471.9 万字的预训练语料定的。
/// SFT 语料常常只有一两万字，`steps_per_epoch` 只有几步，于是那两个数会同时出问题：
///
/// - `steps = 4000` 等于几千个 epoch。本例 SFT 语料 1.4 万字（一个 epoch ≈ 3 步），
///   实测 2600 步时 train 2.68 / val 7.77，而 val 早在 600 步就见底了——多出来的步数
///   全是过拟合；
/// - `eval_every` 按"步数的 1/10"收到 200 步，两次评估之间隔了约 80 个 epoch，
///   val 曲线上只有十几个点，"刚开始像样"的那一步很可能整个被跨过去（`best.ckpt`
///   就落在这么粗的网格上）。
///
/// 所以这里改按 epoch 算：**每 20 个 epoch 评估一次**，步数上限 **400 个 epoch**
/// （早停的 patience 10 次评估 ≈ 200 个 epoch，会先到，上限只是兜底）。
/// `--steps` 显式给了就完全听它的，不做任何收敛。
fn sft_schedule(
    steps_per_epoch: usize,
    cfg_steps: usize,
    cfg_eval_every: usize,
    steps_given: bool,
) -> (usize, usize) {
    /// 两次评估之间隔多少个 epoch
    const EVAL_EVERY_EPOCHS: usize = 20;
    /// 步数上限相当于多少个 epoch
    const MAX_EPOCHS: usize = 400;

    let eval_every = cfg_eval_every.min(steps_per_epoch * EVAL_EVERY_EPOCHS).max(1);
    let steps = if steps_given {
        cfg_steps
    } else {
        cfg_steps.min(steps_per_epoch * MAX_EPOCHS).max(1)
    };
    (steps, eval_every)
}

/// 监督微调（SFT）：用「提问→回答」语料教只会续写的预训练模型"回答"。
///
/// 与 `train` 的区别只有数据与 loss：样本是对话，且**只有回答段参与 loss**
///（提问与角色标记被掩码屏蔽，见 [`data::SftLoader`]）。训练循环本身完全共用。
pub(crate) fn cmd_sft(
    config_path: &str,
    pretrained_path: &str,
    sft_file: Option<&str>,
    steps: Option<usize>,
    lr: Option<f32>,
    out_dir: Option<&str>,
) {
    runlog::announce("sft");
    let mut cfg = Config::load(config_path);

    // 步数与学习率：CLI 优先，其次配置文件。
    // 学习率**不**默认沿用预训练量级：SFT 是在已收敛的权重上继续训，
    // 用预训练的步长会把预训练攒下的语言能力一起冲掉（实测几百步就退化成乱码）。
    // 未显式指定时取配置里 max_lr 的 1/10，并把实际取值记进日志。
    let steps_given = steps.is_some();
    if let Some(s) = steps {
        cfg.train.steps = s;
    }
    let (max_lr, lr_source) = match lr {
        Some(l) => (l, "--lr 指定".to_string()),
        None => (
            cfg.train.max_lr / 10.0,
            "--lr 未指定，取 config max_lr 的 1/10".to_string(),
        ),
    };
    cfg.train.max_lr = max_lr;
    cfg.train.min_lr = max_lr * 0.1;
    // 步数上限 / 预热 / 评估间隔都得看 SFT 语料**实际有多大**，而语料要等权重和分词器
    // 都加载完才能编码，所以这三项挪到下面 `SftLoader` 建好之后再算（见 `sft_schedule`）。

    let sft_paths = sft_file
        .map(str::to_string)
        .or_else(|| cfg.train.sft_file.clone())
        .unwrap_or_else(|| {
            panic!(
                "未指定 SFT 语料：用 --sft-file，或在 {config_path} 里设置 train.sft_file"
            )
        });

    // 输出目录默认加 `-sft` 后缀：绝不能直接写回 tcfg.out_dir，
    // 那会覆盖预训练攒下的 latest.ckpt / best.ckpt —— SFT 效果不好就再也回不去了。
    // 指标 CSV 同理，否则会把预训练那条 loss 曲线整条抹掉。
    let out_dir = match out_dir {
        Some(d) => d.to_string(),
        None => format!("{}-sft", cfg.train.out_dir),
    };
    cfg.train.log_file = Some(format!("{out_dir}/sft.csv"));

    // SFT 是"带 loss mask 的"训练，GPU 的常驻输出头路径不吃逐位置权重，整段会回落到
    // 逐算子路径。实测这条回落路径比纯 CPU 还慢（MX150：392 tok/s vs CPU 780 tok/s），
    // 长跑还会崩。用户多半是照着预训练的命令加 `--features gpu` 过来的，所以要说清楚。
    #[cfg(feature = "gpu")]
    logln!(
        "提示：SFT 的 loss 带逐位置掩码，GPU 常驻输出头路径不可用，会回落到逐算子路径——\
         实测比纯 CPU 更慢。建议改用不带 `--features gpu` 的构建跑 SFT。"
    );

    // 先加载权重与语料：步数 / 评估间隔要按"SFT 语料究竟有多大"推导，而语料有多大
    // 得先编码出来。这几行只借用 `&cfg.train` 读一下，用完立刻还回去，下面才能继续改它。
    let (model, tokenizer, ckpt) = {
        let t = &cfg.train;
        load_model_and_tokenizer(pretrained_path, t, t.seed, None)
    };
    // 语料可能散在多处，用逗号分隔（支持目录与 `*` 通配）。
    // 按文件加载而不是拼成一份：说话人角色是文件级属性，拼接会让后面文件的角色弄反。
    let sft_texts = load_texts(&sft_paths);
    let loader = SftLoader::from_texts(
        &sft_texts,
        &tokenizer,
        ckpt.model.block_size,
        cfg.train.batch_size,
    );

    // 一个 epoch = 把整个 SFT 语料过一遍要多少步
    let steps_per_epoch = loader
        .num_tokens()
        .div_ceil(cfg.train.batch_size * ckpt.model.block_size)
        .max(1);
    let (steps, eval_every) = sft_schedule(
        steps_per_epoch,
        cfg.train.steps,
        cfg.train.eval_every,
        steps_given,
    );
    if steps != cfg.train.steps || eval_every != cfg.train.eval_every {
        logln!(
            "[info] SFT 语料 {} token，1 epoch ≈ {steps_per_epoch} 步：\
             步数上限 {} → {steps}、评估间隔 {} → {eval_every} 步（≈{} epoch）\
             ——配置里那两个数是配预训练语料的",
            loader.num_tokens(),
            cfg.train.steps,
            cfg.train.eval_every,
            eval_every / steps_per_epoch,
        );
    }
    cfg.train.steps = steps;
    cfg.train.eval_every = eval_every;
    // 预热太长会让 SFT 大部分步数都耗在爬坡上
    cfg.train.warmup_steps = cfg.train.warmup_steps.min(cfg.train.steps / 10).max(1);

    runlog::json(&format!("完整配置（{config_path} + CLI 覆盖后）"), &cfg);
    let tcfg = &cfg.train;

    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            (
                "预训练权重",
                format!("{pretrained_path}（step {}）", ckpt.step),
            ),
            ("预训练模型结构", format!("{:?}", ckpt.model)),
            ("SFT 语料", sft_paths.clone()),
            (
                "步数 / 学习率",
                format!("{} / {}（{lr_source}）", tcfg.steps, tcfg.max_lr),
            ),
            ("输出目录", out_dir.clone()),
            ("prompt 模板", format!("{SFT_USER} … {SFT_ASSISTANT} … {SFT_END}")),
        ],
    );

    let mut rng = Rng::new(tcfg.seed);
    logln!(
        "SFT 语料：{} 段对话，打包 {} token | 监督位置（回答段）占 {:.1}%",
        loader.num_conversations(),
        loader.num_tokens(),
        100.0 * loader.supervised_ratio(),
    );
    if loader.supervised_ratio() < 0.05 {
        logln!(
            "[warn] 监督位置只有 {:.1}%，训练信号很稀——语料里提问与标记远多于回答",
            100.0 * loader.supervised_ratio()
        );
    }

    let best = train::train_transformer(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&out_dir),
        None,
        &mut rng,
        None,
    );

    // 把分词器一并放进输出目录，这个目录就是自包含的，推理时不必再指回预训练目录
    tokenizer.save(&format!("{out_dir}/tokenizer.json"));

    runlog::append(&format!("SFT 结束：best val loss = {best:.4}"));
    logln!("SFT 完成，best val loss = {best:.4}");
    // 两个 checkpoint 都给出，因为 SFT 语料往往很小：验证区与训练区同分布，val 曲线
    // 常在头几十步就见底，之后 train 还在降而 val 回升——`best.ckpt` 取的是那个
    // 「刚开始像样」的快照，`final.ckpt` 反而把回答的句式学得更足。
    // 哪个更好只有自己聊两句才知道，所以两个都列出来。
    logln!(
        "用下面的命令对话（{SFT_ASSISTANT} 之后的模板会自动拼好）；\
         建议 best.ckpt 与 final.ckpt 各聊几句再定——语料小时 final 常常更「会答」：\n  \
         cargo run --release -- chat --ckpt {out_dir}/best.ckpt --tokenizer {out_dir}/tokenizer.json\n  \
         cargo run --release -- chat --ckpt {out_dir}/final.ckpt --tokenizer {out_dir}/tokenizer.json"
    );
    runlog::finish();
}

/// LoRA 微调：加载预训练模型，冻结全部主干，只训练挂上去的低秩适配层。
///
/// 与 `sft` 共用同一套训练循环、同一份语料（`train.sft_file`，可用 `--sft-file` 覆盖），
/// 差别只在**参数集合**：
/// - `sft` 全参更新，可训练参数 = 全部
/// - `finetune` 冻结主干，可训练参数 = 各层 `lora_a` / `lora_b`（本项目实测 1.32%）
///
/// 冻结是三处一起做实的，缺一不可：前向用 `matmul_frozen` 省掉 `dW`、优化器跳过冻结参数
/// （连带不吃权重衰减）、GPU 常驻显存快路整体让路。原理见 `docs/29-LoRA低秩适配.md`。
///
/// 挂载位置由 `--lora-targets` 决定（缺省 Q/K/V）。基座本身是 LoRA 存档时有两种走法：
/// 缺省**重挂一套**（旧的增量丢弃，只继承主干），加 `--resume-lora` 则**链式续训**
/// 存档里那套（保留已学的增量），后者不能再传 rank / alpha / targets。
#[allow(clippy::too_many_arguments)]
pub(crate) fn cmd_finetune(
    config_path: &str,
    pretrained_path: &str,
    lora_rank: Option<usize>,
    lora_alpha: Option<f32>,
    lora_targets: Option<&str>,
    resume_lora: bool,
    steps: usize,
    lr: f32,
    sft_file: Option<&str>,
    out_dir: Option<&str>,
) {
    runlog::announce("finetune");
    let mut cfg = Config::load(config_path);

    // 先加载基座：该怎么挂适配层，取决于基座本身是不是 LoRA 存档
    let (mut model, tokenizer, ckpt) =
        load_model_and_tokenizer(pretrained_path, &cfg.train, cfg.train.seed, None);
    let mut rng = Rng::new(cfg.train.seed);

    // 这两个值只用于日志（运行时结构一律读模型上的实际状态，不做第二本账）
    let (rank_used, alpha_used, targets_used) = if resume_lora {
        assert!(
            model.has_lora(),
            "--resume-lora 要求基座是 LoRA 存档，但 {pretrained_path} 里没有适配层"
        );
        assert!(
            lora_rank.is_none() && lora_alpha.is_none() && lora_targets.is_none(),
            "--resume-lora 的结构由存档决定，不能再传 --lora-rank / --lora-alpha / --lora-targets"
        );
        model.resume_lora();
        let l = model.lora.clone().expect("resume_lora 之后 model.lora 必有值");
        (l.rank, l.alpha, l.targets.to_string())
    } else {
        // 基线形态来自配置文件（老代码里这一块是死字段：CLI 的默认值无条件盖过它，写进
        // config 也不生效）。现在 CLI 的 rank / alpha / targets 都是 Option，缺谁才用谁：
        // rank   = --lora-rank  → config.lora.rank  → 16
        // alpha  = --lora-alpha → config.lora.alpha（仅当配置文件显式写了 lora 块）→ rank
        // targets= --lora-targets → config.lora.targets → q,k,v
        let from_config = cfg.train.lora.clone();
        let base = from_config.clone().unwrap_or_default();
        let rank = lora_rank.unwrap_or(base.rank);
        let lora_cfg = config::LoRAConfig {
            rank,
            // 配置文件没写 lora 时，缺省 α = rank（增量不额外缩放）；写了就以它的 alpha 为准
            alpha: lora_alpha
                .or_else(|| from_config.map(|l| l.alpha))
                .unwrap_or(rank as f32),
            targets: match lora_targets {
                Some(spec) => config::LoRATargets::parse(spec).unwrap_or_else(|e| panic!("{e}")),
                None => base.targets,
            },
        };
        if model.has_lora() {
            // 拿一个 LoRA 存档当基座：默认语义是**重挂一套**，原适配层权重随之丢弃
            // ——只有主干被继承。说清楚，并把链式续训的开关指出来。
            logln!(
                "[warn] 基座存档本身已是 LoRA 形态，这里会重新挂一套适配层（原适配层权重丢弃）；\
                 要接着训存档里那套，加 --resume-lora"
            );
        }
        model.apply_lora(&lora_cfg, &mut rng);
        (lora_cfg.rank, lora_cfg.alpha, lora_cfg.targets.to_string())
    };
    // LoRA 形态写回配置，交给训练循环与 runlog（train_transformer 的实际行为仍只读模型上的形态）
    cfg.train.lora = model.lora.clone();
    cfg.train.steps = steps;
    cfg.train.max_lr = lr;
    cfg.train.min_lr = lr * 0.1;
    // 预热与评估间隔按步数收窄（与 sft 同理：配置里那是给上万步预训练的值）
    cfg.train.warmup_steps = cfg.train.warmup_steps.min(cfg.train.steps / 10).max(1);
    cfg.train.eval_every = cfg.train.eval_every.min(cfg.train.steps / 10).max(1);

    // 输出目录默认加 `-lora` 后缀：绝不写回 tcfg.out_dir，否则会覆盖预训练攒下的
    // latest.ckpt / best.ckpt。指标 CSV 同理，避免把预训练那条 loss 曲线抹掉。
    let out_dir = match out_dir {
        Some(d) => d.to_string(),
        None => format!("{}-lora", cfg.train.out_dir),
    };
    cfg.train.log_file = Some(format!("{out_dir}/lora.csv"));

    // 日志记的是**覆盖 CLI 参数之后**的最终配置（这才是本次实际使用的配置）
    runlog::json(&format!("完整配置（{config_path} + CLI 覆盖后）"), &cfg);
    let tcfg = &cfg.train;
    if resume_lora {
        logln!(
            "链式续训：沿用存档里的适配层（rank={rank_used} alpha={alpha_used} 挂载={targets_used}）\
             steps={steps} lr={lr}｜已学的增量保留，只训 A/B"
        );
    } else {
        logln!(
            "LoRA 微调：rank={rank_used} alpha={alpha_used} 挂载={targets_used} steps={steps} lr={lr}\
             ｜冻结主干，只训适配层"
        );
    }

    let total_params: usize = model.parameters().iter().map(|p| p.numel()).sum();
    let trainable_params: usize = model
        .trainable_parameters()
        .iter()
        .map(|p| p.numel())
        .sum();
    // 打印**实测**参数（不是按公式预估）：可训练集合就是 trainable_parameters，
    // 和"主干冻结是否真的生效"是同一件事，估错了这里会立刻露馅。
    logln!(
        "模型参数：{total_params}（含冻结主干）｜可训练：{trainable_params}（{:.3}%）\
         ｜{} 对低秩矩阵（挂载 {targets_used}），共 {} 个适配参数张量",
        100.0 * trainable_params as f32 / total_params.max(1) as f32,
        model.lora_parameters().len() / 2,
        model.trainable_parameters().len(),
    );
    // 语料与 `sft` 用的是同一份、同一套加载方式：LoRA 微调就是 SFT 的省算力版，
    // 只有语料对齐，"全参 SFT vs LoRA" 的对比才成立（都能直接吃 `chat --prompt-format sft`）。
    // 语料可能散在多处（逗号分隔）；按文件加载而不是拼成一份，因为说话人角色是文件级属性。
    let sft_paths = sft_file
        .map(str::to_string)
        .or_else(|| tcfg.sft_file.clone())
        .unwrap_or_else(|| {
            panic!("未指定 SFT 语料：用 --sft-file，或在 {config_path} 里设置 train.sft_file")
        });

    runlog::fields(
        "本次运行参数",
        &[
            ("配置文件", config_path.to_string()),
            (
                "预训练权重",
                format!("{pretrained_path}（step {}）", ckpt.step),
            ),
            ("预训练模型结构", format!("{:?}", ckpt.model)),
            (
                "LoRA 形态",
                format!(
                    "rank {rank_used} / alpha {alpha_used}（alpha/rank = {:.4}）｜挂载 {targets_used}｜{}",
                    alpha_used / rank_used as f32,
                    if resume_lora { "链式续训（沿用存档的适配层）" } else { "新挂一套适配层" }
                ),
            ),
            ("总参数", total_params.to_string()),
            (
                "可训练参数（适配层）",
                format!("{trainable_params}（{:.3}%）",
                    100.0 * trainable_params as f32 / total_params.max(1) as f32),
            ),
            ("SFT 语料", sft_paths.clone()),
            ("输出目录", out_dir.clone()),
        ],
    );

    let sft_texts = load_texts(&sft_paths);

    let loader = SftLoader::from_texts(&sft_texts, &tokenizer, ckpt.model.block_size, tcfg.batch_size);
    logln!(
        "SFT 语料：{} 段对话，打包 {} token | 监督位置（回答段）占 {:.1}%",
        loader.num_conversations(),
        loader.num_tokens(),
        100.0 * loader.supervised_ratio(),
    );
    let best = train::train_transformer(
        &model,
        &tokenizer,
        &loader,
        tcfg,
        Some(&out_dir),
        None,
        &mut rng,
        None,
    );

    // 分词器一并放进输出目录，这个目录就是自包含的，推理时不必再指回预训练目录
    tokenizer.save(&format!("{out_dir}/tokenizer.json"));

    let what = if resume_lora { "链式续训" } else { "LoRA 微调" };
    runlog::append(&format!("{what}结束：best val loss = {best:.4}"));
    logln!("{what}完成，best val loss = {best:.4}");
    logln!(
        "用下面的命令对话（存档里已含冻结主干，不必再指回预训练权重）：\n  \
         cargo run --release -- chat --ckpt {out_dir}/best.ckpt --tokenizer {out_dir}/tokenizer.json\n  \
         cargo run --release -- chat --ckpt {out_dir}/final.ckpt --tokenizer {out_dir}/tokenizer.json\n  \
         cargo run --release -- chat --ckpt {out_dir}/best.ckpt --tokenizer {out_dir}/tokenizer.json --merge-lora  # 增量并入主干后再推理\n\
         接着训这套适配层（链式续训，别加 rank/alpha/targets）：\n  \
         cargo run --release -- finetune --pretrained {out_dir}/best.ckpt --resume-lora"
    );
    runlog::finish();
}

/// 生成预设配置文件
pub(crate) fn cmd_preset(name: &str, output: &str) {
    let cfg = Config::from_preset(name);
    cfg.save(output);
    println!("已生成 '{}' 预设配置到 {}", name, output);
    println!("  模型：n_embd={} n_head={} n_layer={} block_size={}",
        cfg.model.n_embd, cfg.model.n_head, cfg.model.n_layer, cfg.model.block_size);
    println!("  训练：steps={} batch_size={} max_lr={}", 
        cfg.train.steps, cfg.train.batch_size, cfg.train.max_lr);
    if cfg.model.use_rmsnorm {
        println!("  架构：LLaMA 风格（RMSNorm + SwiGLU + GQA）");
    } else {
        println!("  架构：经典风格（LayerNorm + GELU + MHA）");
    }
}

// ==================== HuggingFace 权重导入（import-hf 子命令） ====================

/// 把 HF 模型目录转成本仓库的原生存档（详见 [`crate::hf`] 的模块文档）。
///
/// 流程刻意做成"**构造骨架 → 按名字严格灌数据 → 落盘**"三步：
/// - 骨架由映射出来的 `TransformerConfig` 构造，所以参数名/形状/初始化规则与原生模型**完全同源**；
/// - 灌数据由 [`crate::hf::assign_named_parameters`] 对账（缺一个、形状不符、多一个都报错）；
/// - 落盘复用 [`checkpoint::save_weights`]，于是导入产物就是一份普通权重档——
///   训练、推理、量化、LoRA 全都能直接吃，**没有"HF 专用"的运行时代码**。
pub(crate) fn cmd_import_hf(
    model_dir: &str,
    output: &str,
    tokenizer: Option<&str>,
    tokenizer_out: Option<&str>,
) {
    println!("导入 HuggingFace 权重：{model_dir}");
    let hf = crate::hf::load_hf_dir(model_dir).unwrap_or_else(|e| panic!("导入失败：{e}"));
    for note in &hf.notes {
        println!("  {note}");
    }
    // 骨架：用 0 号种子构造即可——所有参数马上会被导入数据整体覆盖
    let mut rng = Rng::new(0);
    let model = Transformer::new(hf.cfg.clone(), &mut rng);
    let n = crate::hf::assign_named_parameters(&model, &hf.tensors)
        .unwrap_or_else(|e| panic!("{e}"));
    checkpoint::save_weights(output, &model, 0, f32::INFINITY);
    let mb = std::fs::metadata(output).map(|m| m.len()).unwrap_or(0) as f64 / 1e6;
    println!("  已写出 {n} 个参数 → {output}（{mb:.1} MB，权重 only 档）");

    // 分词器（可选）：HF tokenizer.json → 本仓库格式
    if let Some(tk) = tokenizer {
        let out = tokenizer_out.map(|s| s.to_string()).unwrap_or_else(|| {
            // 缺省与存档同目录，约定名 tokenizer.json
            let dir = std::path::Path::new(output)
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            if dir.is_empty() {
                "tokenizer.json".to_string()
            } else {
                format!("{dir}/tokenizer.json")
            }
        });
        let tok = Tokenizer::load_hf(tk).unwrap_or_else(|e| panic!("分词器导入失败：{e}"));
        tok.save(&out);
        println!("  分词器 → {out}（词表 {} 个 token，HF 口径）", tok.vocab_size());
    } else {
        println!(
            "  提示：未给 --tokenizer，推理时请自备分词器；\
             HF 模型的词表口径与本仓库默认的 char/BPE-2048 不同，必须用同一个分词器"
        );
    }
}

// ==================== 性能基准（bench 子命令） ====================

/// 性能基准：用**固定、可复现、短时**的任务测训练与推理吞吐（tok/s）。
///
/// 目的：优化改动前后在同一台机器、同一套参数下对比，不必跑完整训练。
/// 模型/数据/步数全部写死，只受 `--steps`、`--gen-tokens` 影响，
/// 因此两次运行的差异只来自代码本身。
pub(crate) fn cmd_bench(steps: usize, gen_tokens: usize) {
    use std::time::Instant;

    println!("=== 性能基准（bench）===");
    println!("rayon 线程数：{}", rayon::current_num_threads());

    let mut rng = Rng::new(1234);
    let tokenizer = Tokenizer::char(data::CORPUS);
    let vocab_size = tokenizer.vocab_size();
    let gcfg = TransformerConfig {
        vocab_size,
        n_embd: 128,
        n_head: 4,
        n_layer: 2,
        block_size: 64,
        ..TransformerConfig::default()
    };
    let model = Transformer::new(gcfg.clone(), &mut rng);
    let param_count: usize = model.parameters().iter().map(|p| p.numel()).sum();
    println!(
        "模型：n_layer={} n_embd={} n_head={} block={} vocab={} | 参数 {}",
        gcfg.n_layer, gcfg.n_embd, gcfg.n_head, gcfg.block_size, vocab_size, param_count
    );

    // ---- 训练吞吐 ----
    let batch = 4;
    let loader = DataLoader::new(data::CORPUS, &tokenizer, gcfg.block_size, batch);
    let tcfg = config::TrainConfig {
        seed: 42,
        batch_size: batch,
        steps,
        max_lr: 6e-4,
        min_lr: 6e-5,
        warmup_steps: (steps / 10).max(1),
        eval_every: steps + 1, // 基准不评估，避免干扰计时
        log_file: None,        // 基准不落盘
        ..config::TrainConfig::default()
    };
    let t0 = Instant::now();
    train::train_transformer(&model, &tokenizer, &loader, &tcfg, None, None, &mut rng, None);
    let train_secs = t0.elapsed().as_secs_f64();
    let train_tokens = steps * batch * gcfg.block_size;
    println!(
        "[bench] train     : {} steps | {:.3}s | {:.4}s/step | {:.0} tok/s",
        steps,
        train_secs,
        train_secs / steps.max(1) as f64,
        train_tokens as f64 / train_secs
    );

    // ---- 推理吞吐 ----
    // 单次生成只几十毫秒，计时抖动大；先预热一次（线程池/首次分配），
    // 再重复若干次取**最短**耗时作为吞吐上限，前后对比才稳定。
    fn bench_generate(
        model: &Transformer,
        tokenizer: &Tokenizer,
        prompt: &str,
        n: usize,
        kv: KvOpts,
        reps: usize,
    ) -> f64 {
        let mut rng = Rng::new(42);
        let opts = SampleOpts::default();
        let _ = generate(model, tokenizer, prompt, n, &opts, kv, &mut rng);
        let mut best = f64::INFINITY;
        for _ in 0..reps {
            let t = Instant::now();
            let _ = generate(model, tokenizer, prompt, n, &opts, kv, &mut rng);
            best = best.min(t.elapsed().as_secs_f64());
        }
        best
    }

    let prompt = "Once upon a time";
    let prompt_len = tokenizer.encode(prompt).len();
    // 生成长度限制在 block_size 内：滑动窗口下缓存已不会提前停，这里只是控制测量规模
    let kv_new = gen_tokens.min(gcfg.block_size.saturating_sub(prompt_len + 1));
    let kv_secs = bench_generate(&model, &tokenizer, prompt, kv_new, KvOpts::on(0, None), 5);
    println!(
        "[bench] infer/kv  : {} tok | {:.4}s | {:.1} tok/s",
        kv_new,
        kv_secs,
        kv_new as f64 / kv_secs
    );

    let full_new = gen_tokens.min(24);
    let full_secs = bench_generate(&model, &tokenizer, prompt, full_new, KvOpts::off(), 3);
    println!(
        "[bench] infer/full: {} tok | {:.4}s | {:.1} tok/s",
        full_new,
        full_secs,
        full_new as f64 / full_secs
    );
    println!("（以上 tok/s 越高越好；优化前后同机对比即可看出收益）");
}
