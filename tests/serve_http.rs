//! `serve` 的**真实 HTTP 层**集成测试：起一个真 socket，用裸 `TcpStream` 发请求。
//!
//! 为什么不只做单元测试：鉴权、状态端点、JSON 协议这些都在 `handle` → `route` 的路径上，
//! 单元测试绕过了 HTTP 解析与响应写出，测不出"头/状态码/连接关闭"这类问题。
//!
//! 只用 `std`：不引入 HTTP 客户端依赖（与项目的依赖边界一致）。

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::time::Duration;

use llm_from_scratch::model::{Transformer, TransformerConfig};
use llm_from_scratch::rng::Rng;
use llm_from_scratch::sample::{KvOpts, SampleOpts};
use llm_from_scratch::serve::{self, ServeCfg};
use llm_from_scratch::tokenizer::Tokenizer;

/// 让操作系统先分配一个空闲端口，再把它让给 serve（极小概率竞态：另一进程抢先绑定）。
fn free_port() -> u16 {
    let l = TcpListener::bind("127.0.0.1:0").expect("绑定临时端口");
    let port = l.local_addr().unwrap().port();
    drop(l);
    port
}

fn start_server(max_concurrent: usize) -> u16 {
    start_server_cfg(max_concurrent, 64)
}

/// 指定窗口的服务器（窗口决定 `max_tokens` 的钳制上限，见 serve 里的预算逻辑）。
fn start_server_cfg(max_concurrent: usize, block_size: usize) -> u16 {
    let port = free_port();
    let cfg = ServeCfg {
        host: "127.0.0.1".into(),
        port,
        api_key: Some("sk-test".into()),
        cors: false,
        system: String::new(),
        use_sft: true,
        chatml: false,
        max_new: 8,
        seed: 42,
        kv: KvOpts::on(0, None),
        model_name: "tiny".into(),
        sample: SampleOpts::default(),
        block_size,
        max_concurrent,
        vision: None,
    };
    // 词表里必须有 JSON 需要的字符（`response_format` 的约束解码用例要用）
    let tokenizer = Tokenizer::char("用户：你好\n助手：你好\n。。\n{\"a\":1}[], ");
    // 词表大小必须与分词器一致：否则采样出的 id 可能超出词表，decode 直接 panic
    let mut rng = Rng::new(1);
    let model = Transformer::new(TransformerConfig::tiny(tokenizer.vocab_size()), &mut rng);
    std::thread::spawn(move || {
        serve::run(cfg, model, tokenizer, None);
    });
    // 等端口起来：连上即返回，最多等 5 秒
    for _ in 0..100 {
        if TcpStream::connect_timeout(
            &(std::net::Ipv4Addr::LOCALHOST, port).into(),
            Duration::from_millis(100),
        )
        .is_ok()
        {
            break;
        }
        std::thread::sleep(Duration::from_millis(50));
    }
    port
}

/// 发一个请求，返回 `(状态码, 响应体)`。用 `Connection: close` 读到 EOF。
fn request(port: u16, raw: &str) -> (u16, String) {
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("连接服务");
    s.set_read_timeout(Some(Duration::from_secs(20))).unwrap();
    s.write_all(raw.as_bytes()).expect("写请求");
    let mut buf = Vec::new();
    let _ = s.read_to_end(&mut buf);
    let text = String::from_utf8_lossy(&buf).into_owned();
    let status = text
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .unwrap_or(0);
    (status, text)
}

fn get(port: u16, path: &str, key: Option<&str>) -> (u16, String) {
    let auth = key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    let raw = format!("GET {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n{auth}Connection: close\r\n\r\n");
    request(port, &raw)
}

fn post_json(port: u16, path: &str, key: Option<&str>, body: &str) -> (u16, String) {
    let auth = key
        .map(|k| format!("Authorization: Bearer {k}\r\n"))
        .unwrap_or_default();
    let raw = format!(
        "POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Type: application/json\r\n\
         Content-Length: {}\r\n{auth}Connection: close\r\n\r\n{body}",
        body.len()
    );
    request(port, &raw)
}

/// 需要鉴权的端点：无 key 必须 401，且错误体是 OpenAI 风格的 `{"error":{...}}`。
#[test]
fn http_requires_api_key() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":2}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", None, body);
    assert_eq!(
        status, 401,
        "缺 Authorization 应回 401，实际 {status}：{resp}"
    );
    assert!(
        resp.contains("authentication_error"),
        "错误体应带 type：{resp}"
    );
    assert!(resp.contains("\"error\""), "错误体应是 OpenAI 信封：{resp}");
}

/// `/v1/status` 是**诊断端点，有意不鉴权**（探针要能在不带 key 时读它）。
/// 这条测试把该决定钉住：如果哪天给它加了鉴权，监控会先炸。
#[test]
fn http_status_is_intentionally_public() {
    let port = start_server(2);
    let (status, body) = get(port, "/v1/status", None);
    assert_eq!(status, 200, "诊断端点不应要求 key，实际 {status}：{body}");
}

/// `/v1/status` 必须暴露并发相关的三个计数（运维靠它判断是否被打满）。
#[test]
fn http_status_reports_concurrency_fields() {
    let port = start_server(2);
    let (status, body) = get(port, "/v1/status", Some("sk-test"));
    assert_eq!(status, 200, "带上 key 应 200，实际 {status}：{body}");
    for key in [
        "\"in_flight\"",
        "\"max_concurrent\"",
        "\"rejected\"",
        "\"queued\"",
    ] {
        assert!(body.contains(key), "状态里缺 {key}：{body}");
    }
    assert!(
        body.contains("\"max_concurrent\":2"),
        "max_concurrent 应回 2：{body}"
    );
}

/// 一轮对话：200 + `finish_reason` + `choices`。
#[test]
fn http_chat_completion_returns_finish_reason() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":4}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");
    assert!(resp.contains("\"choices\""), "响应缺 choices：{resp}");
    assert!(
        resp.contains("\"finish_reason\""),
        "响应缺 finish_reason：{resp}"
    );
    assert!(resp.contains("\"usage\""), "响应缺 usage：{resp}");
}

/// 非法 JSON 必须是 400（而不是 500 或直接断连）。
#[test]
fn http_bad_json_is_400() {
    let port = start_server(2);
    let (status, body) = post_json(port, "/v1/chat/completions", Some("sk-test"), "{not json");
    assert_eq!(status, 400, "非法 JSON 应 400，实际 {status}：{body}");
    assert!(
        body.contains("invalid_request_error"),
        "错误类型应为 invalid_request_error：{body}"
    );
}

/// 未知路径 404，且不泄漏内部信息。
#[test]
fn http_unknown_path_is_404() {
    let port = start_server(2);
    let (status, body) = get(port, "/v1/nope", Some("sk-test"));
    assert_eq!(status, 404, "未知路径应 404，实际 {status}：{body}");
}

/// 流式对话必须是**真 SSE**：`text/event-stream` + chunked + `data:` 帧 + `[DONE]` 收尾。
///
/// 这里直接读裸 socket 的字节流，不解析 chunk 边界——只要帧内容完整到达即可，
/// 真正的分帧正确性由 `streaming_equals_non_streaming` 这类单元测试保证。
#[test]
fn http_chat_stream_is_sse() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":4,"stream":true}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    let lower = resp.to_lowercase();
    assert_eq!(status, 200, "流式应 200，实际 {status}：{resp}");
    assert!(
        lower.contains("text/event-stream"),
        "响应头应声明 SSE：{resp}"
    );
    assert!(
        lower.contains("transfer-encoding: chunked"),
        "SSE 应为 chunked：{resp}"
    );
    assert!(lower.contains("data:"), "缺少 SSE data 帧：{resp}");
    assert!(resp.contains("[DONE]"), "SSE 应以 [DONE] 收尾：{resp}");
}

/// 流式帧里必须带 `usage`（客户端靠它展示 tok/s，而不是自己拿墙钟估）。
#[test]
fn http_chat_stream_reports_usage() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":4,"stream":true}"#;
    let (_, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert!(resp.contains("\"usage\""), "流式帧里缺 usage：{resp}");
}

// ==================== 协议错误路径 ====================

/// 缺 `messages` → 400（而不是 500 或静默用空 prompt）。
#[test]
fn http_missing_messages_is_400() {
    let port = start_server(2);
    let (status, body) = post_json(
        port,
        "/v1/chat/completions",
        Some("sk-test"),
        r#"{"max_tokens":4}"#,
    );
    assert_eq!(status, 400, "缺 messages 应 400，实际 {status}：{body}");
    assert!(
        body.contains("缺少 messages"),
        "错误信息应指明缺字段：{body}"
    );
}

/// 空 `messages` → 400（服务端不能拿空数组去拼 prompt）。
#[test]
fn http_empty_messages_is_400() {
    let port = start_server(2);
    let (status, body) = post_json(
        port,
        "/v1/chat/completions",
        Some("sk-test"),
        r#"{"messages":[]}"#,
    );
    assert_eq!(status, 400, "空 messages 应 400，实际 {status}：{body}");
    assert!(body.contains("不能为空"), "错误信息应说明为空：{body}");
}

/// 不支持的 role → 400，且错误信息列出允许的取值（便于客户端自查）。
#[test]
fn http_unknown_role_is_400() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"tool","content":"x"}]}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 400, "未知 role 应 400，实际 {status}：{resp}");
    assert!(
        resp.contains("不支持的 role"),
        "错误信息应指明 role：{resp}"
    );
    assert!(resp.contains("system"), "错误信息应列出允许的 role：{resp}");
}

/// `max_tokens` 远超上下文窗口时必须**钳制**而不是报错：回 200 且 `length` 收尾。
#[test]
fn http_oversized_max_tokens_is_clamped_not_rejected() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":10000}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(
        status, 200,
        "超大 max_tokens 应被钳制，实际 {status}：{resp}"
    );
    assert!(
        resp.contains("\"finish_reason\":\"length\"")
            || resp.contains("\"finish_reason\":\"stop\""),
        "钳制后应以 length/stop 收尾：{resp}"
    );
}

/// 对 `/v1/chat/completions` 用 GET 应回 405（而不是 404 或把 GET 当 POST 处理）。
#[test]
fn http_wrong_method_is_405() {
    let port = start_server(2);
    let (status, body) = get(port, "/v1/chat/completions", Some("sk-test"));
    assert_eq!(status, 405, "GET 对话端点应 405，实际 {status}：{body}");
}

/// 请求体超过 16 MiB 上限 → **413**（而不是读进内存后 OOM 或 500）。
#[test]
fn http_oversized_body_is_413() {
    let port = start_server(2);
    // 造一个 16 MiB + 1 字节的合法 JSON 字符串（内容不重要，关键是不超内存又能触发上限）
    let huge = "x".repeat(16 * 1024 * 1024);
    let body = format!(r#"{{"messages":[{{"role":"user","content":"{huge}"}}]}}"#);
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), &body);
    assert_eq!(
        status,
        413,
        "超大请求体应 413，实际 {status}：{}",
        &resp[..resp.len().min(200)]
    );
    assert!(
        resp.contains("请求体超过"),
        "错误信息应说明上限：{}",
        &resp[..resp.len().min(200)]
    );
}

/// 并发上限的 HTTP 层验证：cap=1 时，第一个流式请求不出让并发位，第二个请求必须 429。
///
/// **默认 ignore**：要稳定复现"两个请求重叠"，需要一个可控的慢请求；本项目不为测试加
/// 生产后门，而 tiny 模型生成几千 token 仍可能在第二个请求到达前就结束（本地实测会偶发 200）。
/// 429 的**逻辑**已由 `serve::tests::test_permit_cap_is_hard_and_never_exceeded` 覆盖，
/// 端到端行为也已在本机用 curl 手工验证过；这条留作手工/夜间按需运行。
///
/// 做法：A 连接只读响应头、**故意不排空 body**，服务端的 SSE 写出会被 TCP 背压阻塞，
/// 于是一路占着并发位；此时 B 连接发一个短请求，必然撞上限。
/// （许可计数的确定性断言在 `serve::tests::test_permit_cap_is_hard_and_never_exceeded`。）
#[test]
#[ignore = "需要可控慢请求制造重叠；429 逻辑由 permit 单测覆盖，此处留作手工验证"]
fn http_concurrency_cap_returns_429() {
    // 窗口给大一点：`max_tokens` 会被钳到 3/4 窗口，窗口太小则首个请求瞬间结束、根本撞不上。
    let port = start_server_cfg(1, 4096);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":3000,"stream":true}"#;
    let raw = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: 127.0.0.1\r\n\
         Content-Type: application/json\r\nAuthorization: Bearer sk-test\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    let mut slow = TcpStream::connect(("127.0.0.1", port)).expect("连接 A");
    slow.set_read_timeout(Some(Duration::from_secs(5))).unwrap();
    slow.write_all(raw.as_bytes()).expect("写请求 A");
    // 只读一小段（响应头 + 若干帧），之后不再读 → 服务端写出最终会被背压卡住
    let mut buf = [0u8; 512];
    let _ = slow.read(&mut buf);
    // 给服务端一点时间进入阻塞写
    std::thread::sleep(Duration::from_millis(500));

    // 先看服务端此刻是否真有请求在跑：没有说明首个请求已经结束，本用例无法制造重叠
    let (_, st) = get(port, "/v1/status", None);
    let in_flight: usize = st
        .split("\"in_flight\":")
        .nth(1)
        .and_then(|rest| rest.split([',', '}']).next())
        .and_then(|n| n.trim().parse().ok())
        .unwrap_or(0);
    if in_flight == 0 {
        eprintln!(
            "跳过：首个流式请求已结束（in_flight=0），本次没能制造并发重叠；\
             429 的逻辑由 serve::tests 的 permit 单测覆盖"
        );
        return;
    }

    let (status, resp) = post_json(
        port,
        "/v1/chat/completions",
        Some("sk-test"),
        r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":2}"#,
    );
    assert_eq!(
        status,
        429,
        "cap=1 时第二个请求应 429，实际 {status}：{}",
        &resp[..resp.len().min(200)]
    );
    assert!(
        resp.contains("rate_limit_error"),
        "应为 rate_limit_error：{resp}"
    );
    assert!(
        resp.contains("并发已达上限"),
        "错误信息应说明并发上限：{resp}"
    );

    // 关掉 A → 许可释放，后续请求恢复 200
    drop(slow);
    let mut recovered = false;
    for _ in 0..40 {
        std::thread::sleep(Duration::from_millis(50));
        let (st, _) = post_json(
            port,
            "/v1/chat/completions",
            Some("sk-test"),
            r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":2}"#,
        );
        if st == 200 {
            recovered = true;
            break;
        }
    }
    assert!(recovered, "关掉慢连接后并发位应释放（一直没恢复 200）");
}

// ==================== keep-alive 与 embeddings ====================

/// 从流里读**一个**完整的 HTTP 响应（按 Content-Length），用于复用连接的用例。
fn read_one_response(s: &mut TcpStream) -> (u16, String) {
    let mut buf = Vec::new();
    let hdr_end = loop {
        let mut tmp = [0u8; 1024];
        let n = s.read(&mut tmp).expect("读响应头");
        assert!(n > 0, "连接被提前关闭（响应头没读完）");
        buf.extend_from_slice(&tmp[..n]);
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break p + 4;
        }
    };
    let head = String::from_utf8_lossy(&buf[..hdr_end]).into_owned();
    let status = head
        .lines()
        .next()
        .and_then(|l| l.split_whitespace().nth(1))
        .and_then(|c| c.parse::<u16>().ok())
        .unwrap_or(0);
    let len: usize = head
        .lines()
        .find_map(|l| {
            let l = l.to_ascii_lowercase();
            l.strip_prefix("content-length:")
                .map(|v| v.trim().to_string())
        })
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let mut body = buf[hdr_end..].to_vec();
    while body.len() < len {
        let mut tmp = vec![0u8; len - body.len()];
        let n = s.read(&mut tmp).expect("读响应体");
        assert!(n > 0, "连接在响应体中途关闭");
        body.extend_from_slice(&tmp[..n]);
    }
    (status, String::from_utf8_lossy(&body).into_owned())
}

/// `keep-alive`：同一条 TCP 连接上连续发两个请求，都应得到 200。
#[test]
fn http_keep_alive_reuses_connection() {
    let port = start_server(2);
    let mut s = TcpStream::connect(("127.0.0.1", port)).expect("连接");
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    for i in 0..2 {
        let req = "GET /v1/status HTTP/1.1\r\nHost: 127.0.0.1\r\nConnection: keep-alive\r\n\r\n";
        s.write_all(req.as_bytes()).expect("写请求");
        let (status, body) = read_one_response(&mut s);
        assert_eq!(status, 200, "第 {i} 个请求应 200，body={body}");
        assert!(
            body.contains("in_flight"),
            "第 {i} 个响应缺状态字段：{body}"
        );
    }
}

fn embeddings_body(input: &str) -> String {
    format!(r#"{{"input":{input}}}"#)
}

/// 字符串输入：返回一条向量，维度等于模型 `n_embd`。
#[test]
fn http_embeddings_string_input() {
    let port = start_server(2);
    let dim = llm_from_scratch::model::TransformerConfig::default().n_embd;
    let (status, resp) = post_json(
        port,
        "/v1/embeddings",
        Some("sk-test"),
        &embeddings_body(r#""你好""#),
    );
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");
    let v: serde_json::Value =
        serde_json::from_str(resp[resp.find('{').unwrap_or(0)..].trim_end()).unwrap_or_default();
    let data = v["data"].as_array().expect("响应应有 data 数组");
    assert_eq!(data.len(), 1, "单条输入应返回一条向量");
    let emb = data[0]["embedding"].as_array().expect("缺 embedding");
    assert_eq!(emb.len(), dim, "向量维度应等于 n_embd={dim}");
    assert!(v["usage"]["prompt_tokens"].as_u64().unwrap_or(0) > 0);
}

/// 批量 + token id：`["a", 5]` 应返回两条向量。
#[test]
fn http_embeddings_batch_and_ids() {
    let port = start_server(2);
    let (status, resp) = post_json(
        port,
        "/v1/embeddings",
        Some("sk-test"),
        &embeddings_body(r#"["你好", 3]"#),
    );
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");
    assert!(resp.contains("\"data\""), "缺 data：{resp}");
    // 两条结果：索引 0 / 1
    assert!(resp.contains("\"index\":0"), "缺 index 0：{resp}");
    assert!(resp.contains("\"index\":1"), "缺 index 1：{resp}");
}

/// 空数组、超批量、非法元素、缺字段：四条 400 分支。
#[test]
fn http_embeddings_rejects_bad_input() {
    let port = start_server(2);
    for (body, needle) in [
        (embeddings_body("[]"), "不能为空"),
        (embeddings_body("[true]"), "只能是字符串"),
        (r#"{"model":"latest"}"#.to_string(), "缺少 input"),
    ] {
        let (status, resp) = post_json(port, "/v1/embeddings", Some("sk-test"), &body);
        assert_eq!(status, 400, "应 400，实际 {status}：{resp}");
        assert!(resp.contains(needle), "错误信息应含「{needle}」：{resp}");
    }

    // 33 条 > MAX_EMBED_BATCH(32)
    let many = format!(
        "[{}]",
        (0..33).map(|i| i.to_string()).collect::<Vec<_>>().join(",")
    );
    let (status, resp) = post_json(
        port,
        "/v1/embeddings",
        Some("sk-test"),
        &embeddings_body(&many),
    );
    assert_eq!(status, 400, "超批量应 400，实际 {status}：{resp}");
    assert!(resp.contains("最多 32 条"), "错误信息应说明上限：{resp}");
}

// ==================== /metrics ====================

/// `/metrics` 是 Prometheus 文本格式：不鉴权、`text/plain; version=0.0.4`、
/// 且关键的 gauge/counter 都在。
#[test]
fn http_metrics_is_prometheus_text() {
    let port = start_server(2);
    let (status, body) = get(port, "/metrics", None);
    assert_eq!(
        status, 200,
        "指标端点应 200（不鉴权），实际 {status}：{body}"
    );
    let lower = body.to_lowercase();
    assert!(
        lower.contains("text/plain") && lower.contains("version=0.0.4"),
        "Content-Type 应为 Prometheus 文本：{body}"
    );
    for needle in [
        "# TYPE llm_serve_in_flight gauge",
        "# TYPE llm_serve_rejected_total counter",
        "llm_serve_max_concurrent 2",
        "# TYPE llm_serve_requests_total counter",
        "llm_serve_model_info{",
    ] {
        assert!(body.contains(needle), "指标里缺 `{needle}`：\n{body}");
    }
}

/// 跑一次对话后，chat 计数必须 +1（说明指标读到的是真实计数，不是静态文本）。
#[test]
fn http_metrics_counts_chat_requests() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":2}"#;
    let (status, _) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 200);

    let (status, metrics) = get(port, "/metrics", None);
    assert_eq!(status, 200);
    let expected = "llm_serve_requests_total{kind=\"chat\"} 1";
    assert!(
        metrics.contains(expected),
        "chat 计数应为 1：期望含 `{expected}`，实际：\n{metrics}"
    );
}

// ==================== P4-a：OpenAI 协议字段边界 ====================

/// 语义敏感的不支持字段必须 **400 说明**，而不是静默忽略（客户端才能发现"没生效"）。
#[test]
fn http_unsupported_semantic_fields_are_400() {
    let port = start_server(2);
    for body in [
        r#"{"messages":[{"role":"user","content":"q"}],"tools":[{"type":"function"}]}"#,
        r#"{"messages":[{"role":"user","content":"q"}],"logit_bias":{"5":1.0}}"#,
    ] {
        let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
        assert_eq!(status, 400, "应 400，实际 {status}：{resp}");
        assert!(resp.contains("invalid_request_error"), "错误码不对：{resp}");
        assert!(resp.contains("不支持"), "错误信息应说明不支持：{resp}");
    }
}

/// `max_completion_tokens` 与 `max_tokens` 等价（新版 OpenAI 客户端的叫法）。
#[test]
fn http_max_completion_tokens_is_honored() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_completion_tokens":2}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");
    // 生成数不得超过 2：从 usage.completion_tokens 判断
    let usage = resp.split("\"completion_tokens\":").nth(1).unwrap_or("");
    let n: u64 = usage
        .chars()
        .take_while(|c| c.is_ascii_digit())
        .collect::<String>()
        .parse()
        .unwrap_or(u64::MAX);
    assert!(
        n <= 2,
        "max_completion_tokens=2 时不应生成 {n} 个 token：{resp}"
    );

    // 两个都给 → 400
    let both =
        r#"{"messages":[{"role":"user","content":"q"}],"max_tokens":3,"max_completion_tokens":4}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), both);
    assert_eq!(status, 400, "同时给两个应 400，实际 {status}：{resp}");
    assert!(resp.contains("只能给一个"), "错误信息应说明冲突：{resp}");
}

// ==================== P4-b：logprobs 端到端 ====================

/// 打开 logprobs 后：`content` 非空、logprob ≤ 0、top_logprobs 数量受控，
/// 且把每步 token 拼起来应当还原出 message.content（同一套分词器解码）。
#[test]
fn http_logprobs_are_consistent_with_output() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":4,"logprobs":true,"top_logprobs":3}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");

    let v: serde_json::Value = serde_json::from_str(resp[resp.find('{').unwrap_or(0)..].trim_end())
        .expect("响应应是 JSON");
    let content = v["choices"][0]["logprobs"]["content"]
        .as_array()
        .expect("缺 logprobs.content");
    assert!(!content.is_empty(), "生成段非空，logprobs 不应为空");

    let mut joined = String::new();
    for item in content {
        let lp = item["logprob"].as_f64().expect("缺 logprob");
        assert!(lp <= 0.0, "logprob 应 ≤ 0，实际 {lp}");
        let top = item["top_logprobs"].as_array().expect("缺 top_logprobs");
        assert!(top.len() <= 4, "top_logprobs 最多 4 条，实际 {}", top.len());
        for t in top {
            let tlp = t["logprob"].as_f64().expect("top 项缺 logprob");
            assert!(tlp <= 0.0);
        }
        joined.push_str(item["token"].as_str().expect("缺 token"));
    }
    let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    assert_eq!(joined, text, "逐步 token 拼起来应与 message.content 一致");
}

/// 流式 + logprobs 应明确 400（而不是返回不准的数据）。
#[test]
fn http_logprobs_with_stream_is_400() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"logprobs":true,"stream":true}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 400, "应 400，实际 {status}：{resp}");
    assert!(resp.contains("流式"), "错误信息应说明流式不支持：{resp}");
}

// ==================== P4-c：/v1/completions（旧版补全） ====================

/// 基本补全：`object` 是 `text_completion`，choises[0].text 是字符串，usage 有计数。
#[test]
fn http_completions_basic() {
    let port = start_server(2);
    let body = r#"{"model":"x","prompt":"你好","max_tokens":2}"#;
    let (status, resp) = post_json(port, "/v1/completions", Some("sk-test"), body);
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");
    assert!(
        resp.contains("\"object\":\"text_completion\""),
        "object 不对：{resp}"
    );
    assert!(resp.contains("\"choices\""), "缺 choices：{resp}");
    assert!(resp.contains("\"text\""), "缺 text：{resp}");
    assert!(resp.contains("\"usage\""), "缺 usage：{resp}");
    assert!(
        resp.contains("\"finish_reason\":\"stop\"")
            || resp.contains("\"finish_reason\":\"length\""),
        "finish_reason 应是 stop/length：{resp}"
    );
}

/// 不支持的形态必须 400 并说明原因（而不是返回空结果）。
#[test]
fn http_completions_rejects_unsupported_shapes() {
    let port = start_server(2);
    for (body, needle) in [
        (r#"{"prompt":"你好","stream":true}"#, "stream"),
        (r#"{"prompt":["a","b"]}"#, "只能有 1 条"),
        (r#"{"max_tokens":2}"#, "缺少 prompt"),
        (r#"{"prompt":"你好","logprobs":1}"#, "logprobs"),
    ] {
        let (status, resp) = post_json(port, "/v1/completions", Some("sk-test"), body);
        assert_eq!(status, 400, "应 400，实际 {status}：{resp}");
        assert!(resp.contains(needle), "错误信息应含「{needle}」：{resp}");
    }
}

// ==================== P4-d：JSON 约束解码 ====================

/// `response_format: json_object`：输出必须是**合法 JSON 前缀**；
/// 正常收尾（finish_reason=stop）时必须是**完整合法 JSON**。
#[test]
fn http_json_object_output_is_valid_json() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":40,"response_format":{"type":"json_object"}}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 200, "应 200，实际 {status}：{resp}");

    let v: serde_json::Value = serde_json::from_str(resp[resp.find('{').unwrap_or(0)..].trim_end())
        .expect("响应应是 JSON");
    let text = v["choices"][0]["message"]["content"].as_str().unwrap_or("");
    let finish = v["choices"][0]["finish_reason"].as_str().unwrap_or("");
    assert!(
        llm_from_scratch::json_constraint::is_valid_prefix(text.as_bytes()),
        "输出必须是合法 JSON 前缀，实际：{text:?}"
    );
    if finish == "stop" {
        assert!(
            llm_from_scratch::json_constraint::is_valid_json(text.as_bytes()),
            "stop 时必须是完整合法 JSON，实际：{text:?}"
        );
    }
}

/// `json_schema` 不支持 → 400（明确拒绝，不返回垃圾）。
#[test]
fn http_json_schema_is_rejected() {
    let port = start_server(2);
    let body =
        r#"{"messages":[{"role":"user","content":"q"}],"response_format":{"type":"json_schema"}}"#;
    let (status, resp) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 400, "应 400，实际 {status}：{resp}");
    assert!(
        resp.contains("json_schema"),
        "错误信息应提到 json_schema：{resp}"
    );
}

/// 流式路径同样受约束：把所有 delta 拼起来也必须是合法 JSON 前缀。
#[test]
fn http_json_object_stream_is_constrained() {
    let port = start_server(2);
    let body = r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":30,"stream":true,"response_format":{"type":"json_object"}}"#;
    let (status, raw) = post_json(port, "/v1/chat/completions", Some("sk-test"), body);
    assert_eq!(status, 200, "应 200，实际 {status}：{raw}");
    // 从 SSE 帧里抽出所有 content 片段
    let mut text = String::new();
    for line in raw.lines() {
        let Some(payload) = line.strip_prefix("data: ") else {
            continue;
        };
        if payload == "[DONE]" {
            continue;
        }
        let Ok(v) = serde_json::from_str::<serde_json::Value>(payload) else {
            continue;
        };
        if let Some(d) = v["choices"][0]["delta"]["content"].as_str() {
            text.push_str(d);
        }
    }
    assert!(!text.is_empty(), "流式应产出内容");
    assert!(
        llm_from_scratch::json_constraint::is_valid_prefix(text.as_bytes()),
        "流式拼接结果必须是合法 JSON 前缀，实际：{text:?}"
    );
}
