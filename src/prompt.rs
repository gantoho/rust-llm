//! 对话 prompt 组装：system + 历史裁剪 + 角色模板。
//!
//! 终端 `chat` 与 HTTP `serve` 共用这里的逻辑，保证两条通路拼出的上下文一致：
//! system 永远拼在最前并独立于裁剪，历史按 **token** 裁剪并给本轮生成预留 `max_new`，
//! SFT 模板下本轮停在「助手：」这一行之后（训练时"轮到模型说话"的位置）。

use crate::data::{SFT_ASSISTANT, SFT_END, SFT_USER};
use crate::tokenizer::Tokenizer;

/// SFT 模板下的停止标记（兜底用）。
///
/// 当前分词器带特殊 token，训练时每段对话的收尾符号是**可训练的 EOS**
/// （见 `data::build_sft_stream` 的 `Some(id)` 分支），所以正常收尾靠 EOS 触发。
/// 这里两个文本标记是保险：
/// - `用户：` 防它顺着模板接着编下一轮提问（"回答后面跟提问"在训练语料里到处都是）；
/// - `。。` 只对**老分词器**（没有 EOS、退回文本标记 [`SFT_END`]）训出来的权重有意义，
///   当前 SFT 语料里 `。。` 从未出现过，实际不会命中。
pub const SFT_STOP: &[&str] = &[SFT_END, SFT_USER];

/// 把对话历史裁剪到不超过 `budget` 个 token：从最老的一行开始丢，保留最近的内容。
///
/// 必须真的调 `encode` 来数 token —— 按"字节数 / 字符数"估算的偏差很大：
/// 中英文、BPE 合并数都不同，同样长度的文本 token 数能差一倍以上。
pub fn trim_history(tokenizer: &Tokenizer, history: &str, budget: usize) -> String {
    if history.is_empty() || tokenizer.encode(history).len() <= budget {
        return history.to_string();
    }
    // 行粒度足够细：每轮对话都会写入多行
    let lines: Vec<&str> = history.split('\n').collect();
    for start in 1..lines.len() {
        let candidate = lines[start..].join("\n");
        if tokenizer.encode(&candidate).len() <= budget {
            return candidate;
        }
    }
    String::new()
}

/// SFT 模板下的历史裁剪：以 `SFT_USER` 为切点按**轮**丢，而不是按行。
///
/// 按行丢会把某轮拦腰截断（历史里一行只是回答中的一句），留下半截上下文。
/// 每轮都以 `SFT_USER` 开头，从它的位置切就天然对齐到轮边界。
pub fn trim_sft_history(tokenizer: &Tokenizer, history: &str, budget: usize) -> String {
    if history.is_empty() || tokenizer.encode(history).len() <= budget {
        return history.to_string();
    }
    for (pos, _) in history.match_indices(SFT_USER) {
        let candidate = &history[pos..];
        if tokenizer.encode(candidate).len() <= budget {
            return candidate.to_string();
        }
    }
    String::new()
}

/// 一轮 prompt 的组装结果。
pub struct Assembled {
    /// 拼好的完整 prompt：system + '\n' + 裁剪后历史 + '\n' + 本轮 tail
    pub prompt: String,
    /// 裁剪后的历史，直接作为下一轮的起点（不含 system）
    pub history: String,
    /// 触发裁剪时的 `(裁剪前 token 数, 裁剪后 token 数)`；未触发为 `None`
    pub trimmed: Option<(usize, usize)>,
    /// 本轮给历史的实际预算（日志里"超出预算"要用它对比）
    pub history_budget: usize,
}

/// 拼一轮 prompt，并把历史裁进预算。
///
/// - `use_sft=true`：tail 是 `用户：\n{input}\n助手：\n`，历史按轮裁（[`trim_sft_history`]）；
///   否则 tail 就是 `input` 本身，历史按行裁（[`trim_history`]）。
/// - `prompt_budget` 是输入侧的 token 预算（通常 = `block_size - max_new`）。
///   分配顺序（前面的优先保）：
///   1. system prompt —— 永远保留。它是序列最开头的"注意力锚点"（attention sink），
///      丢掉它不只是失忆，还会让整条序列的注意力分布失稳
///   2. 本轮生成 —— 调用方已在 `prompt_budget` 里扣掉 `max_new`，让整段生成留在窗口内
///   3. 对话历史 —— 剩下的额度都给它，不够就从最老的开始丢
pub fn assemble(
    tokenizer: &Tokenizer,
    system: &str,
    history: &str,
    input: &str,
    use_sft: bool,
    prompt_budget: usize,
) -> Assembled {
    // raw 模式就是输入本身；SFT 模式补上角色标记，并**停在「助手：」这一行之后**——
    // 这正是训练时"轮到模型说话"的位置，模型才会接着写回答，而不是继续续写前文。
    let tail = if use_sft {
        format!("{SFT_USER}\n{input}\n{SFT_ASSISTANT}\n")
    } else {
        input.to_string()
    };
    let n_tail = tokenizer.encode(&tail).len();

    let system = system.trim();
    // +1 是 system 后面的 '\n'
    let n_system_prefix = if system.is_empty() { 0 } else { tokenizer.encode(system).len() + 1 };
    // 历史预算 = 输入预算 - system（含其后 '\n'）- 本轮追加内容 - 历史与它之间的 '\n'
    let history_budget = prompt_budget.saturating_sub(n_system_prefix + n_tail + 1);

    let kept = if use_sft {
        trim_sft_history(tokenizer, history, history_budget)
    } else {
        trim_history(tokenizer, history, history_budget)
    };
    let trimmed = if kept.len() < history.len() {
        Some((tokenizer.encode(history).len(), tokenizer.encode(&kept).len()))
    } else {
        None
    };

    let mut prompt = String::new();
    if !system.is_empty() {
        prompt.push_str(system);
        prompt.push('\n');
    }
    if !kept.is_empty() {
        prompt.push_str(&kept);
        prompt.push('\n');
    }
    prompt.push_str(&tail);

    Assembled { prompt, history: kept, trimmed, history_budget }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 字符分词器：词表只含语料出现过的字符，所以语料要覆盖本模块测试用到的全部字符
    fn tiny_tokenizer() -> Tokenizer {
        let corpus = "hello world\nSYS\nq1\na1\nq2\nx\nline1\nline2\n\
                      你好\n世界\n用户：\n助手：\n问题一\n回答一\n问题二\n回答二\n";
        Tokenizer::char(corpus)
    }

    #[test]
    fn test_assemble_raw_prompt_shape() {
        let tok = tiny_tokenizer();
        let a = assemble(&tok, "SYS", "q1\na1", "q2", false, 1024);
        assert_eq!(a.prompt, "SYS\nq1\na1\nq2");
        // 未触发裁剪：历史原样交还
        assert_eq!(a.history, "q1\na1");
        assert!(a.trimmed.is_none());
    }

    #[test]
    fn test_assemble_sft_tail_stops_at_assistant() {
        let tok = tiny_tokenizer();
        let a = assemble(&tok, "", "", "你好", true, 1024);
        assert_eq!(a.prompt, format!("{SFT_USER}\n你好\n{SFT_ASSISTANT}\n"));
    }

    #[test]
    fn test_assemble_trims_history_within_budget() {
        let tok = tiny_tokenizer();
        // 预算只够 tail + 一点空隙：历史必须被裁掉（这里预算给 0，历史全丢）
        let a = assemble(&tok, "", "line1\nline2", "x", false, 0);
        assert_eq!(a.history, "");
        let (before, after) = a.trimmed.expect("预算为 0 必然触发裁剪");
        assert!(before > after && after == 0);
        // system 与本轮 tail 不受历史预算影响：prompt 就是裸的本轮输入
        assert_eq!(a.prompt, "x");
    }

    #[test]
    fn test_trim_sft_history_cuts_at_turn_boundary() {
        let tok = tiny_tokenizer();
        let history = format!("{SFT_USER}\n问题一\n{SFT_ASSISTANT}\n回答一\n{SFT_USER}\n问题二\n{SFT_ASSISTANT}\n回答二");
        let full = tok.encode(&history).len();
        // 预算砍半：应从「用户：」处切，保留最近一轮，且切点必是轮边界
        let kept = trim_sft_history(&tok, &history, full / 2);
        assert!(kept.starts_with(SFT_USER));
        assert!(tok.encode(&kept).len() <= full / 2);
        assert!(!kept.contains("问题一"));
        assert!(kept.contains("问题二"));
    }
}
