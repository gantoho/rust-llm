//! `response_format: {"type":"json_object"}` 的**约束解码**（自研，无第三方语法库）。
//!
//! 原理：维护一个「当前已输出字节是不是某个合法 JSON 的**前缀**」的自动机；
//! 每步采样前把「接上后会让前缀非法的 token」的 logit 掩成 `-inf`。
//! 因此输出**必然**是合法 JSON 的前缀；模型正常收尾时就是一个完整合法的 JSON。
//!
//! 为什么不用"事后校验 + 重试"：那样只能保证"最终可能是 JSON"，中间过程照样输出垃圾，
//! 且重试成本不可控。约束解码把约束放进采样，代价是每步多一次词表扫描。
//!
//! 范围：只支持 `json_object`（任意合法 JSON 值）；`json_schema` 不做——
//! 那需要把 JSON Schema 编译成语法，是另一个量级的工程。

/// 容器类型（用 `u8` 存在定长数组里，避免每个候选 token 都分配 `Vec`）
const CTX_OBJECT: u8 = 1;
const CTX_ARRAY: u8 = 2;

/// 嵌套上限：正常模型输出远达不到；超过就判非法（防栈爆）
const MAX_DEPTH: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NumState {
    /// 负号之后
    Sign,
    /// 单个 `0`（后面不能再跟数字）
    Zero,
    /// 整数部分
    Int,
    /// `.` 之后（必须至少一个数字）
    Dot,
    /// 小数部分
    Frac,
    /// `e` / `E` 之后
    Exp,
    /// 指数符号之后
    ExpSign,
    /// 指数数字
    ExpDigit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    /// 期待一个值（严格：不接受 `]`，用于 `:` 之后与数组逗号之后）
    Value,
    /// 期待一个值或 `]`（只用于 `[` 之后：允许空数组）
    ValueOrEnd,
    /// 对象里期待键或 `}`（只用于 `{` 之后）
    KeyOrEnd,
    /// 对象里期待键（严格：逗号之后不接受 `}`，否则会放过尾随逗号）
    Key,
    /// 键字符串已结束，期待 `:`
    Colon,
    /// 字符串中（`esc` = 刚见到反斜杠；`uni` = `\uXXXX` 已消费的十六进制位数）
    Str { esc: bool, uni: u8, key: bool },
    /// 数字中
    Num(NumState),
    /// 字面量（true / false / null）的进度
    Lit { word: &'static [u8], pos: usize },
    /// 值之后：期待 `,` / `}` / `]` / 空白
    After,
    /// 顶层值已完整，只允许空白
    Done,
}

/// 「已输出字节是合法 JSON 前缀」的增量自动机。
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct JsonPrefix {
    stack: [u8; MAX_DEPTH],
    depth: usize,
    mode: Mode,
    /// 已消费的字节数（用于区分"空文档"与"完整文档"）
    bytes: u64,
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

fn is_digit(b: u8) -> bool {
    b.is_ascii_digit()
}

impl Default for JsonPrefix {
    fn default() -> Self {
        Self::new()
    }
}

impl JsonPrefix {
    pub fn new() -> Self {
        JsonPrefix {
            stack: [0; MAX_DEPTH],
            depth: 0,
            mode: Mode::Value,
            bytes: 0,
        }
    }

    /// 是否已经是**完整**的 JSON（之后只允许空白）。
    pub fn is_complete(&self) -> bool {
        self.mode == Mode::Done
    }

    /// 当前前缀是否**已可**作为完整 JSON 结束。
    ///
    /// 与 [`Self::is_complete`] 的区别：数字只在看到"下一个字节"时才知道自己结束了
    /// （`1`、`1.5`、`1e3` 都是完整文档，但内部状态还停在 `Num`），
    /// 所以"能否就此收尾"要把这些终态数字算进来。生成时用它决定是否允许 EOS。
    pub fn can_finish(&self) -> bool {
        matches!(
            self.mode,
            Mode::Done
                | Mode::Num(NumState::Zero | NumState::Int | NumState::Frac | NumState::ExpDigit)
        )
    }

    /// 已消费的字节数。
    pub fn consumed(&self) -> u64 {
        self.bytes
    }

    /// 喂入一段字节；返回 `false` 表示"这会让前缀变成非法 JSON 前缀"。
    ///
    /// 失败时**不保证**状态未被部分修改——调用方要么丢弃该状态（掩码判定用副本），
    /// 要么只在返回 `true` 时继续使用。
    pub fn feed(&mut self, bytes: &[u8]) -> bool {
        for &b in bytes {
            if !self.feed_byte(b) {
                return false;
            }
            self.bytes += 1;
        }
        true
    }

    /// 顶层是否已经"什么都没输出"（用于掩码判定：空文档不允许直接收尾）
    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    fn top(&self) -> Option<u8> {
        if self.depth == 0 {
            None
        } else {
            Some(self.stack[self.depth - 1])
        }
    }

    /// 一个值（字符串/数字/字面量/容器）结束后的状态迁移。
    fn after_value(&mut self) -> bool {
        if self.depth == 0 {
            self.mode = Mode::Done;
        } else {
            self.mode = Mode::After;
        }
        true
    }

    /// 关闭容器：必须与栈顶类型一致，且弹出后要么回到 `After`，要么顶层 `Done`。
    fn close(&mut self, expected: u8) -> bool {
        if self.depth == 0 || self.stack[self.depth - 1] != expected {
            return false;
        }
        self.depth -= 1;
        self.after_value()
    }

    fn feed_byte(&mut self, b: u8) -> bool {
        match self.mode {
            Mode::Done => is_ws(b),

            Mode::Value => {
                if is_ws(b) {
                    return true;
                }
                match b {
                    b'{' => {
                        if self.depth == MAX_DEPTH {
                            return false;
                        }
                        self.stack[self.depth] = CTX_OBJECT;
                        self.depth += 1;
                        self.mode = Mode::KeyOrEnd;
                        true
                    }
                    b'[' => {
                        if self.depth == MAX_DEPTH {
                            return false;
                        }
                        self.stack[self.depth] = CTX_ARRAY;
                        self.depth += 1;
                        self.mode = Mode::ValueOrEnd;
                        true
                    }
                    b'"' => {
                        self.mode = Mode::Str {
                            esc: false,
                            uni: 0,
                            key: false,
                        };
                        true
                    }
                    b't' => {
                        self.mode = Mode::Lit {
                            word: b"true",
                            pos: 1,
                        };
                        true
                    }
                    b'f' => {
                        self.mode = Mode::Lit {
                            word: b"false",
                            pos: 1,
                        };
                        true
                    }
                    b'n' => {
                        self.mode = Mode::Lit {
                            word: b"null",
                            pos: 1,
                        };
                        true
                    }
                    b'-' => {
                        self.mode = Mode::Num(NumState::Sign);
                        true
                    }
                    b'0' => {
                        self.mode = Mode::Num(NumState::Zero);
                        true
                    }
                    b'1'..=b'9' => {
                        self.mode = Mode::Num(NumState::Int);
                        true
                    }
                    _ => false,
                }
            }

            Mode::ValueOrEnd => {
                if b == b']' {
                    return self.close(CTX_ARRAY);
                }
                // 其余与 Value 完全一致：把字节交给 Value 处理
                self.mode = Mode::Value;
                self.feed_byte(b)
            }

            Mode::Key => {
                if is_ws(b) {
                    return true;
                }
                if b == b'"' {
                    self.mode = Mode::Str {
                        esc: false,
                        uni: 0,
                        key: true,
                    };
                    true
                } else {
                    false
                }
            }

            Mode::KeyOrEnd => {
                if is_ws(b) {
                    return true;
                }
                match b {
                    b'"' => {
                        self.mode = Mode::Str {
                            esc: false,
                            uni: 0,
                            key: true,
                        };
                        true
                    }
                    b'}' => self.close(CTX_OBJECT),
                    _ => false,
                }
            }

            Mode::Colon => {
                if is_ws(b) {
                    return true;
                }
                if b == b':' {
                    self.mode = Mode::Value;
                    true
                } else {
                    false
                }
            }

            Mode::Str { esc, uni, key } => {
                if uni > 0 {
                    // \uXXXX：只允许十六进制
                    if b.is_ascii_hexdigit() {
                        let left = uni - 1;
                        self.mode = if left == 0 {
                            Mode::Str {
                                esc: false,
                                uni: 0,
                                key,
                            }
                        } else {
                            Mode::Str {
                                esc: false,
                                uni: left,
                                key,
                            }
                        };
                        true
                    } else {
                        false
                    }
                } else if esc {
                    // 转义字符：JSON 只允许这几个
                    let ok = matches!(
                        b,
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' | b'u'
                    );
                    if !ok {
                        return false;
                    }
                    self.mode = if b == b'u' {
                        Mode::Str {
                            esc: false,
                            uni: 4,
                            key,
                        }
                    } else {
                        Mode::Str {
                            esc: false,
                            uni: 0,
                            key,
                        }
                    };
                    true
                } else if b == b'\\' {
                    self.mode = Mode::Str {
                        esc: true,
                        uni: 0,
                        key,
                    };
                    true
                } else if b == b'"' {
                    if key {
                        self.mode = Mode::Colon;
                    } else {
                        return self.after_value();
                    }
                    true
                } else if b < 0x20 {
                    // 未转义的控制字符在 JSON 字符串里非法
                    false
                } else {
                    true
                }
            }

            Mode::Lit { word, pos } => {
                if b != word[pos] {
                    return false;
                }
                let next = pos + 1;
                if next == word.len() {
                    self.after_value()
                } else {
                    self.mode = Mode::Lit { word, pos: next };
                    true
                }
            }

            Mode::Num(st) => self.feed_num(st, b),

            Mode::After => {
                if is_ws(b) {
                    return true;
                }
                match b {
                    b',' => {
                        self.mode = match self.top() {
                            Some(CTX_OBJECT) => Mode::Key,
                            Some(CTX_ARRAY) => Mode::Value,
                            _ => return false,
                        };
                        true
                    }
                    b'}' => self.close(CTX_OBJECT),
                    b']' => self.close(CTX_ARRAY),
                    _ => false,
                }
            }
        }
    }

    fn feed_num(&mut self, st: NumState, b: u8) -> bool {
        // 数字结束后，当前字节要按"值之后"的规则重新处理（`,` `}` `]` 空白）
        let end_value = |s: &mut Self| s.after_value();

        match st {
            NumState::Sign => {
                if b == b'0' {
                    self.mode = Mode::Num(NumState::Zero);
                    true
                } else if b.is_ascii_digit() {
                    self.mode = Mode::Num(NumState::Int);
                    true
                } else {
                    false
                }
            }
            NumState::Zero => match b {
                b'.' => {
                    self.mode = Mode::Num(NumState::Dot);
                    true
                }
                b'e' | b'E' => {
                    self.mode = Mode::Num(NumState::Exp);
                    true
                }
                b'0'..=b'9' => false, // 前导零非法
                _ => {
                    end_value(self);
                    self.feed_byte(b)
                }
            },
            NumState::Int => match b {
                b'0'..=b'9' => true,
                b'.' => {
                    self.mode = Mode::Num(NumState::Dot);
                    true
                }
                b'e' | b'E' => {
                    self.mode = Mode::Num(NumState::Exp);
                    true
                }
                _ => {
                    end_value(self);
                    self.feed_byte(b)
                }
            },
            NumState::Dot => {
                if is_digit(b) {
                    self.mode = Mode::Num(NumState::Frac);
                    true
                } else {
                    false
                }
            }
            NumState::Frac => match b {
                b'0'..=b'9' => true,
                b'e' | b'E' => {
                    self.mode = Mode::Num(NumState::Exp);
                    true
                }
                _ => {
                    end_value(self);
                    self.feed_byte(b)
                }
            },
            NumState::Exp => {
                if b == b'+' || b == b'-' {
                    self.mode = Mode::Num(NumState::ExpSign);
                    true
                } else if is_digit(b) {
                    self.mode = Mode::Num(NumState::ExpDigit);
                    true
                } else {
                    false
                }
            }
            NumState::ExpSign => {
                if is_digit(b) {
                    self.mode = Mode::Num(NumState::ExpDigit);
                    true
                } else {
                    false
                }
            }
            NumState::ExpDigit => {
                if is_digit(b) {
                    true
                } else {
                    end_value(self);
                    self.feed_byte(b)
                }
            }
        }
    }
}

/// 这段字节是否"已经是某个合法 JSON 的前缀"（整段一起判断）。
pub fn is_valid_prefix(bytes: &[u8]) -> bool {
    let mut s = JsonPrefix::new();
    s.feed(bytes)
}

/// 这段字节是否是**完整且合法**的 JSON 文档（允许首尾空白）。
pub fn is_valid_json(bytes: &[u8]) -> bool {
    let mut s = JsonPrefix::new();
    if !s.feed(bytes) {
        return false;
    }
    s.can_finish() && !bytes.iter().all(|&b| is_ws(b))
}

/// 给定词表（每个 token 的字节）与当前前缀状态，算出**允许的 token 掩码**。
///
/// 规则：
/// - token 的每个字节都必须让自动机保持有效（逐字节喂，任一失败即不允许）；
/// - 已经完整后只允许空白 token（收尾由此处的 `EOS` 逻辑负责，见调用方）；
/// - 空文档不允许"什么都不输出"，所以 EOS 只在已完整时可用。
///
/// 复杂度 O(词表大小 × token 平均字节数)。char 分词器词表小，可直接用；
/// BPE（15 万词表）每步约几毫秒，这是"约束解码"的固有成本（可后续加前缀树优化）。
pub fn allowed_mask(vocab: &[Vec<u8>], state: &JsonPrefix) -> Vec<bool> {
    let mut out = Vec::with_capacity(vocab.len());
    let complete = state.is_complete();
    for tok in vocab {
        if complete {
            // 已经完整：只允许纯空白 token（多输出一个非空白字符就非法）
            out.push(!tok.is_empty() && tok.iter().all(|&b| is_ws(b)));
        } else {
            let mut probe = state.clone();
            out.push(!tok.is_empty() && probe.feed(tok));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn complete(s: &str) -> bool {
        is_valid_json(s.as_bytes())
    }

    #[test]
    fn test_accepts_valid_documents() {
        for doc in [
            "{}",
            "[]",
            "0",
            "-1",
            "1.5e-3",
            "\"hi\"",
            "true",
            "false",
            "null",
            "  {\"a\": 1, \"b\": [true, null, \"x\"]}  ",
            "{\"nested\": {\"deep\": [[[]]]}}",
            "[1,2,3]",
            "{\"esc\": \"a\\\"b\\\\c\\u00e9\\n\"}",
        ] {
            assert!(complete(doc), "应接受：{doc}");
        }
    }

    #[test]
    fn test_rejects_invalid_documents() {
        for doc in [
            "",
            "{",
            "{\"a\"}",
            "{\"a\":}",
            "[1,]",
            "{\"a\": 1,}",
            "01",
            "1.",
            ".5",
            "+1",
            "tru",
            "True",
            "\"unterminated",
            "\"bad\\x escape\"",
            "{\"a\": 1} extra",
            "[1 2]",
        ] {
            assert!(!complete(doc), "应拒绝：{doc}");
        }
    }

    #[test]
    fn test_prefix_semantics() {
        // 这些都还没完整，但都是合法前缀（可继续补成合法 JSON）
        for p in [
            "",
            " ",
            "{",
            "{\"a",
            "{\"a\":",
            "[1,",
            "1e",
            "tr",
            "\"ab\\u00",
        ] {
            assert!(is_valid_prefix(p.as_bytes()), "应是合法前缀：{p:?}");
        }
        // 这些已经不是任何合法 JSON 的前缀
        for p in ["{]", "[1,]", "01", "nulll", "\"\\q"] {
            assert!(!is_valid_prefix(p.as_bytes()), "不应是前缀：{p:?}");
        }
    }

    #[test]
    fn test_mask_blocks_illegal_tokens() {
        // 空状态：只有能开启一个值的 token 可用
        let vocab: Vec<Vec<u8>> = vec![
            b"{".to_vec(),
            b"[".to_vec(),
            b"\"".to_vec(),
            b"1".to_vec(),
            b"t".to_vec(),
            b"}".to_vec(), // 不能以 } 开头
            b",".to_vec(),
        ];
        let mask = allowed_mask(&vocab, &JsonPrefix::new());
        assert_eq!(mask, vec![true, true, true, true, true, false, false]);

        // 完整之后：只允许纯空白（这里的词表里没有空白 token）
        let mut st = JsonPrefix::new();
        assert!(st.feed(b"{}"));
        assert!(st.is_complete());
        let mask = allowed_mask(&vocab, &st);
        assert!(mask.iter().all(|&v| !v), "完整后不应允许任何非空白 token");

        // 有空白 token 时允许它收尾
        let vocab2: Vec<Vec<u8>> = vec![b" ".to_vec(), b"\n".to_vec(), b"x".to_vec()];
        let mask = allowed_mask(&vocab2, &st);
        assert_eq!(mask, vec![true, true, false]);
    }

    #[test]
    fn test_key_then_value_progression() {
        let mut st = JsonPrefix::new();
        assert!(st.feed(b"{\"a\":"));
        // 现在期待一个值：数字、字符串、容器、字面量都行，`}` 不行
        let vocab: Vec<Vec<u8>> = vec![
            b"1".to_vec(),
            b"\"x\"".to_vec(),
            b"true".to_vec(),
            b"{}".to_vec(),
            b"}".to_vec(),
        ];
        let mask = allowed_mask(&vocab, &st);
        assert_eq!(mask, vec![true, true, true, true, false]);

        // 喂一个完整对象后（还差外层 `}`）仍不完整，补上外层 `}` 才完整
        assert!(st.feed(b"{}"));
        assert!(!st.is_complete(), "外层对象未闭合，不应算完整");
        assert!(st.feed(b"}"));
        assert!(st.is_complete());
    }
}
