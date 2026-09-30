//! HTTP API 服务：把模型包装成 OpenAI 兼容的接口。
//!
//! 提供的端点：
//! - `POST /v1/chat/completions` —— 对话补全，`stream=true` 时走 SSE 流式
//! - `POST /v1/embeddings` —— 文本向量化（最后一层 hidden 按行均值池化）
//! - `GET  /v1/models` —— 只有一个模型的模型列表
//! - `GET  /health` —— 存活探针
//! - `GET  /v1/status` —— 队列/计数/模型信息（诊断用，不鉴权）
//! - `GET  /openapi.json` / `/openapi.yaml` —— 本服务的 OpenAPI 规范（不鉴权）
//!
//! 并发模型：主线程 accept，每个请求 `spawn` 一个线程；模型是**单份**的，
//! 所有生成请求共用一把 `Mutex`（`with_core`），因此实际生成是串行排队的，
//! `/v1/status` 里的 `queued` 就是这条队列的长度。
//!
//! 之所以选"单模型 + 互斥"而不是多副本：单机 CPU 推理时多副本只会互相抢核心，
//! 排队反而让延迟曲线平稳；要真并发得开多进程（另起端口）或上 GPU。

use std::collections::HashMap;
use std::io::{self, Cursor, Read};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::mpsc::{Receiver, SyncSender, sync_channel};
use std::sync::{Arc, LazyLock, Mutex};
use std::thread;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde_json::{Value, json};
use tiny_http::{Header, Method, Request, Response, Server, StatusCode};

use crate::data::{SFT_ASSISTANT, SFT_USER};
use crate::model::Transformer;
use crate::prompt;
use crate::rng::Rng;
use crate::runlog;
use crate::sample::{Generator, KvOpts, SampleOpts, StopReason};
use crate::tokenizer::Tokenizer;

/// 请求体上限：纯文本 prompt 也就上下文窗口那么大，但 `message.image_url`
/// 要装下整张图的 base64（8 MB 原图 ≈ 10.7 MB 文本），所以放宽到 16 MB
const MAX_BODY: usize = 16 * 1024 * 1024;
/// 单次请求最多带多少条消息（防止有人拿一万个 message 把内存撑爆）
const MAX_MESSAGES: usize = 200;
/// 单次请求最多带几个 `stop` 字符串
const MAX_STOPS: usize = 4;
/// 单个 `stop` 字符串的最大字节数（防止拿它当内存缓冲用）
const MAX_STOP_LEN: usize = 64;
/// 停止标记切片池上限：每种**组合**泄漏一份 `&'static [&'static str]`
const MAX_STOP_SLICES: usize = 256;
/// 不同的停止标记字符串上限：每个字符串只泄漏一次，按出现过的种类封顶
const MAX_STOP_STRINGS: usize = 1024;
/// `/v1/embeddings` 单批最多多少条输入
const MAX_EMBED_BATCH: usize = 32;

// ==================== 服务状态 ====================

/// 服务的静态配置（启动时定死，请求不可改）
pub struct ServeCfg {
    pub host: String,
    pub port: u16,
    /// `Some` = 要求 `Authorization: Bearer <key>`；`None` = 不鉴权
    pub api_key: Option<String>,
    /// 是否回 CORS 头（浏览器页面直连调试用）
    pub cors: bool,
    /// 服务级 system prompt（请求里的 system 会**接在它后面**）
    pub system: String,
    /// 用 SFT 对话模板还是裸续写（同 `chat --prompt-format`）
    pub use_sft: bool,
    /// `max_tokens` 缺省时用的生成上限
    pub max_new: usize,
    /// 不带 `seed` 字段时的随机种子
    pub seed: u64,
    pub kv: KvOpts,
    /// 回给客户端的模型名（OpenAI 客户端会校验它）
    pub model_name: String,
    /// 采样默认值（请求里的同名字段可逐个覆盖）
    pub sample: SampleOpts,
    /// 上下文窗口（用于限制 `max_tokens` 与 embedding 输入长度）
    pub block_size: usize,
    /// 视觉塔超参（`model.vision`）：`Some` 时 `message.image_url` 可用；
    /// 解码图片要缩放归一化，放配置里让请求线程不占模型锁。纯文本模型为 `None`
    pub vision: Option<crate::vision::VisionConfig>,
}

/// 独占的模型核心：一次只有一个线程能碰到它
struct Core {
    model: Transformer,
    tokenizer: Tokenizer,
    /// 不带 `seed` 的请求共用它（同服务实例下连续请求的随机性来源）
    rng: Rng,
}

/// 计数器，全部是原子量：任何线程都能在不碰 `Core` 的情况下读写
#[derive(Default)]
struct Stats {
    /// 已进队但还没拿到锁的请求数
    queued: AtomicUsize,
    /// 正在跑生成/编码的请求数
    running: AtomicUsize,
    chat: AtomicUsize,
    stream: AtomicUsize,
    embeddings: AtomicUsize,
    errors: AtomicUsize,
}

/// 服务的全部运行时状态，`Arc` 共享给每个请求线程
struct State {
    cfg: ServeCfg,
    core: Mutex<Core>,
    stats: Stats,
    /// VQ-VAE（`{out_dir}/vq.ckpt`）：模型采出完整图片段时解码回像素，
    /// 内嵌成 data URI 随 `message.image_url` 回给客户端。纯文本模型为 `None`
    vq: Option<crate::vqvae::Vqvae>,
    /// `time.time()`，回给客户端当 `created` 字段
    created: u64,
    /// 进程启动时刻（算 uptime）
    started: Instant,
}

/// 当前 Unix 时间戳（秒）
fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

// ==================== 停止标记驻留（interner） ====================

/// 全局字符串/切片驻留表。
///
/// `SampleOpts.stop` 是 `&'static [&'static str]`（为了保持 `Copy`，见该字段文档），
/// 但 HTTP 请求里的 `stop` 是**运行时才知道的字符串**。两者要兼容，只能把它"钉"成静态：
/// 字符串 `Box::leak` 成 `&'static str`，切片 `Box::leak` 成 `&'static [&'static str]`。
///
/// 泄漏是有界的：字符串按**去重后的种类**封顶 [`MAX_STOP_STRINGS`]，
/// 切片按**去重后的组合**封顶 [`MAX_STOP_SLICES`]——同一组 stop 反复请求只泄漏一份。
#[derive(Default)]
struct Interner {
    strings: HashMap<String, &'static str>,
    slices: Vec<&'static [&'static str]>,
    combo: HashMap<Vec<String>, &'static [&'static str]>,
}

static INTERN: LazyLock<Mutex<Interner>> = LazyLock::new(|| Mutex::new(Interner::default()));

/// 把一组运行时 stop 字符串换成静态切片（`SampleOpts::stop` 需要的类型）。
fn intern_stops(items: &[String]) -> Result<&'static [&'static str], ApiError> {
    if items.len() > MAX_STOPS {
        return Err(bad(format!("stop 最多 {MAX_STOPS} 个字符串")));
    }
    for s in items {
        if s.is_empty() {
            return Err(bad("stop 里的字符串不能为空"));
        }
        if s.len() > MAX_STOP_LEN {
            return Err(bad(format!("单个 stop 最长 {MAX_STOP_LEN} 字节")));
        }
    }
    if items.is_empty() {
        return Ok(&[]);
    }
    let mut intern = INTERN.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(hit) = intern.combo.get(items) {
        return Ok(*hit);
    }
    if intern.slices.len() >= MAX_STOP_SLICES {
        return Err(bad(format!("本次进程见过的不同 stop 组合已达上限 {MAX_STOP_SLICES}")));
    }
    let mut ptrs = Vec::with_capacity(items.len());
    for s in items {
        if intern.strings.len() >= MAX_STOP_STRINGS && !intern.strings.contains_key(s) {
            return Err(bad(format!("本次进程见过的不同 stop 字符串已达上限 {MAX_STOP_STRINGS}")));
        }
        // get 返回 `&&'static str`（哈希表存的是引用），闭包返回 `&'static str`，
        // 两边类型对不上，只能 match 分开出（不能靠 unwrap_or_else 合流）
        let p: &'static str = match intern.strings.get(s) {
            Some(p) => p,
            None => {
                let p: &'static str = Box::leak(s.clone().into_boxed_str());
                intern.strings.insert(s.clone(), p);
                p
            }
        };
        ptrs.push(p);
    }
    let slice: &'static [&'static str] = Box::leak(ptrs.into_boxed_slice());
    intern.slices.push(slice);
    intern.combo.insert(items.to_vec(), slice);
    Ok(slice)
}

// ==================== 错误 ====================

/// OpenAI 风格的错误：`status` 决定 HTTP 码，`kind` 是 `error.type`
#[derive(Debug)]
struct ApiError {
    status: u16,
    kind: &'static str,
    message: String,
}

fn bad(message: impl Into<String>) -> ApiError {
    ApiError { status: 400, kind: "invalid_request_error", message: message.into() }
}

fn server_err(message: impl Into<String>) -> ApiError {
    ApiError { status: 500, kind: "server_error", message: message.into() }
}

impl ApiError {
    fn to_value(&self) -> Value {
        json!({
            "error": {
                "message": self.message,
                "type": self.kind,
                "code": self.status,
            }
        })
    }
}

// ==================== 拿到模型 ====================

/// 拿一把 `Core` 锁并跑 `f`，顺带维护排队/运行计数、把 panic 转成 500。
///
/// 三件事必须在这里做，否则各调用点会各漏一个：
/// - **排队计数**：进锁前 +1、拿到锁后 -1，`/v1/status` 才能看见"有多少人在等"；
/// - **panic 隔离**：字符分词器遇到词表外的字符会直接 panic（见 `tokenizer.rs`），
///   HTTP 线程上炸了会让整个进程退；转成 500 客户端才知道是输入的问题；
/// - **锁中毒**：上面那条 panic 发生时 Mutex 会被标记中毒，下一个人得能把锁打开
///   （`into_inner`），否则服务从此不可用。
fn with_core<R>(state: &Arc<State>, f: impl FnOnce(&mut Core) -> R) -> Result<R, ApiError> {
    state.stats.queued.fetch_add(1, Ordering::SeqCst);
    let mut core = state.core.lock().unwrap_or_else(|e| e.into_inner());
    state.stats.queued.fetch_sub(1, Ordering::SeqCst);
    state.stats.running.fetch_add(1, Ordering::SeqCst);
    let guard = RunGuard(&state.stats);
    let res = catch_unwind(AssertUnwindSafe(|| f(&mut core)));
    drop(guard);
    res.map_err(|_| {
        server_err("生成过程发生 panic：输入可能含分词器词表外的字符（如 emoji 或生僻字）")
    })
}

/// 退出作用域时把 `running` 减回去（panic 路径也不能漏）
struct RunGuard<'a>(&'a Stats);

impl Drop for RunGuard<'_> {
    fn drop(&mut self) {
        self.0.running.fetch_sub(1, Ordering::SeqCst);
    }
}

// ==================== HTTP 小工具 ====================

/// 取请求头（大小写不敏感）。
///
/// 不能用 `HeaderField::equiv`——它要求 `&'static str`，请求头名是运行时的值。
fn header_value(req: &Request, name: &str) -> Option<String> {
    req.headers()
        .iter()
        .find(|h| h.field.as_str().as_str().eq_ignore_ascii_case(name))
        .map(|h| h.value.as_str().to_string())
}

/// 造一个静态常量响应头（`Header::from_bytes` 对 ASCII 字面量恒成功）
fn header(name: &'static str, value: &'static str) -> Header {
    Header::from_bytes(name.as_bytes(), value.as_bytes()).expect("静态响应头恒为合法 ASCII")
}

/// 开了 `--cors` 就补上跨域头
fn with_cors<R: Read>(state: &Arc<State>, resp: Response<R>) -> Response<R> {
    if !state.cfg.cors {
        return resp;
    }
    resp.with_header(header("Access-Control-Allow-Origin", "*"))
        .with_header(header("Access-Control-Allow-Methods", "GET, POST, OPTIONS"))
        .with_header(header("Access-Control-Allow-Headers", "Content-Type, Authorization"))
        .with_header(header("Access-Control-Max-Age", "600"))
}

/// 发一个 JSON 响应，返回状态码（供请求日志记录）。
///
/// 必须自己写 `Content-Type: application/json`：`Response::from_string` 恒回 `text/plain`。
/// `data_length=Some(len)` 让 tiny_http 走 `Content-Length` 而不是 chunked。
fn reply_json(req: Request, state: &Arc<State>, status: u16, body: &Value) -> u16 {
    let bytes = serde_json::to_vec(body).unwrap_or_else(|e| {
        format!(r#"{{"error":{{"message":"响应序列化失败：{e}","type":"server_error"}}}}"#).into_bytes()
    });
    let len = bytes.len();
    let resp = Response::new(
        StatusCode(status),
        vec![header("Content-Type", "application/json; charset=utf-8")],
        Cursor::new(bytes),
        Some(len),
        None,
    );
    let _ = req.respond(with_cors(state, resp));
    status
}

/// 发一个任意文本类型的响应（`/openapi.yaml` 用），返回状态码。
///
/// 与 [`reply_json`] 的差别只在 `Content-Type`；同样给 `Content-Length`。
fn reply_text(
    req: Request,
    state: &Arc<State>,
    status: u16,
    mime: &'static str,
    body: String,
) -> u16 {
    let bytes = body.into_bytes();
    let len = bytes.len();
    let resp = Response::new(
        StatusCode(status),
        vec![header("Content-Type", mime)],
        Cursor::new(bytes),
        Some(len),
        None,
    );
    let _ = req.respond(with_cors(state, resp));
    status
}

/// 发一个错误响应；顺带把 `errors` 计数 +1（诊断"客户端在踩哪类坑"）
fn reply_err(req: Request, state: &Arc<State>, err: ApiError) -> u16 {
    state.stats.errors.fetch_add(1, Ordering::SeqCst);
    let status = err.status;
    reply_json(req, state, status, &err.to_value())
}

/// 读完整个请求体（限长 + UTF-8 校验）
fn read_body(req: &mut Request) -> Result<String, ApiError> {
    if let Some(n) = req.body_length()
        && n > MAX_BODY
    {
        return Err(too_large());
    }
    let mut buf = String::new();
    let n = req
        .as_reader()
        .take(MAX_BODY as u64 + 1)
        .read_to_string(&mut buf)
        .map_err(|e| bad(format!("读取请求体失败（非 UTF-8 或连接中断）：{e}")))?;
    if n > MAX_BODY {
        return Err(too_large());
    }
    Ok(buf)
}

fn too_large() -> ApiError {
    ApiError { status: 413, kind: "invalid_request_error", message: format!("请求体超过 {MAX_BODY} 字节") }
}

/// 生成一个随机 API key（`sk-` + 32 位十六进制，共 128 bit 熵）
///
/// 调用方没给 `--api-key` 时用它兜底——服务**始终**有鉴权，key 打印在启动日志里。
/// 熵源不引第三方 crate：[`RandomState`] 的种子由标准库从操作系统取
/// （哈希表防碰撞随机化用的就是它），再混入时间、进程号、轮次做一次哈希；
/// 两轮各出 64 bit，拼成 128 bit，两次调用撞车的概率可忽略。
///
/// [`RandomState`]: std::collections::hash_map::RandomState
pub fn random_api_key() -> String {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_nanos() as u64);
    let pid = std::process::id() as u64;
    let mut out = String::with_capacity(3 + 32);
    out.push_str("sk-");
    // 每轮新建一个 RandomState（线程局部种子 + 全局计数器，每次调用都不同），
    // 把时间/进程号/轮次喂进去搅一次；`finish` 是 64 bit → 16 位十六进制
    for round in [0u64, 1] {
        let mut h = RandomState::new().build_hasher();
        h.write_u64(nanos);
        h.write_u64(pid);
        h.write_u64(round);
        out.push_str(&format!("{:016x}", h.finish()));
    }
    out
}

/// [`b64encode`] 的逆：标准 base64 → 字节（忽略空白，校验 `=` 填充位置）。
/// 非法输入返回 `Err` 文案，由调用方转成 400。
fn b64decode(s: &str) -> Result<Vec<u8>, String> {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut rev = [255u8; 256];
    for (i, &c) in T.iter().enumerate() {
        rev[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    let mut padded = false;
    for ch in s.bytes() {
        if ch.is_ascii_whitespace() {
            continue;
        }
        if ch == b'=' {
            padded = true;
            continue;
        }
        if padded {
            return Err("base64 的 = 填充后面还有数据".to_string());
        }
        let v = rev[ch as usize];
        if v == 255 {
            return Err(format!("base64 含非法字符「{}」", ch as char));
        }
        acc = (acc << 6) | v as u32;
        nbits += 6;
        if nbits >= 8 {
            nbits -= 8;
            out.push((acc >> nbits) as u8);
        }
    }
    Ok(out)
}

/// 把请求里的 `message.image_url` 解成模型要的像素，并把问题补成训练时的理解样本格式。
///
/// - 文本：训练样本 human 是 `图:<|image|>{问题}`；用户没自带 `<|image|>` 就按同一格式补
/// - 图片：data URI 剥掉 `data:...;base64,` 前缀（没前缀当裸 base64），解码后
///   走 [`crate::vision::decode_image_bytes`]（与 `vision::load_image` 同一条
///   缩放/归一化路径），失败回 400 而不是 panic。
///
/// 必须在 `with_core` **外面**调用：解码错误要干净地回 400，而 `with_core`
/// 的闭包只能靠 panic 报错（见该函数文档）。
fn prepare_vision(
    vcfg: Option<&crate::vision::VisionConfig>,
    input: &str,
    image: Option<&str>,
) -> Result<(String, Option<Vec<f32>>), ApiError> {
    let Some(uri) = image else {
        return Ok((input.to_string(), None));
    };
    let Some(vcfg) = vcfg else {
        return Err(bad("当前模型没有视觉塔（config 的 model.vision 缺席），无法接受图片输入"));
    };
    let text = if input.contains(crate::tokenizer::IMAGE_LITERAL) {
        input.to_string()
    } else {
        format!("图:{}{input}", crate::tokenizer::IMAGE_LITERAL)
    };
    let b64 = match uri.strip_prefix("data:") {
        Some(rest) => {
            let (head, tail) = rest.split_once(',').ok_or_else(|| {
                bad("data URI 缺少 `,` 分隔符（形如 data:image/png;base64,<base64>）")
            })?;
            if !head.to_ascii_lowercase().contains("base64") {
                return Err(bad("message.image_url 目前只支持 base64 编码（`;base64,`）"));
            }
            tail
        }
        None => uri,
    };
    let bytes = b64decode(b64).map_err(|e| bad(format!("message.image_url 解码失败：{e}")))?;
    let px = crate::vision::decode_image_bytes(&bytes, vcfg.image_size).map_err(bad)?;
    Ok((text, Some(px)))
}

/// 校验 `Authorization: Bearer <key>`
fn check_auth(req: &Request, state: &Arc<State>) -> Result<(), ApiError> {
    let Some(expected) = state.cfg.api_key.as_deref() else {
        return Ok(());
    };
    let raw = header_value(req, "authorization").unwrap_or_default();
    let raw = raw.trim();
    let token = if raw.len() >= 7 && raw[..7].eq_ignore_ascii_case("bearer ") {
        raw[7..].trim()
    } else {
        ""
    };
    if token == expected {
        Ok(())
    } else {
        Err(ApiError {
            status: 401,
            kind: "authentication_error",
            message: "缺少或错误的 API key（请带 `Authorization: Bearer <key>`）".into(),
        })
    }
}

// ==================== 路由 ====================

/// 单个请求的入口：分发到具体处理器，返回 HTTP 状态码
fn route(mut req: Request, state: &Arc<State>) -> u16 {
    // 预检请求：浏览器跨域前先问一句"允许我带 Authorization 吗"
    if matches!(req.method(), Method::Options) {
        let resp = with_cors(state, Response::empty(StatusCode(204)));
        let _ = req.respond(resp);
        return 204;
    }
    // 只看路径，忽略查询串
    let path = req.url().split('?').next().unwrap_or("/").to_string();
    let is_get = matches!(req.method(), Method::Get);
    let is_post = matches!(req.method(), Method::Post);

    match path.as_str() {
        "/health" if is_get => reply_json(
            req,
            state,
            200,
            &json!({ "status": "ok", "model": state.cfg.model_name }),
        ),
        "/v1/models" if is_get => {
            // OpenAI 的 /v1/models 是要鉴权的：它会暴露服务上有哪些模型
            if let Err(e) = check_auth(&req, state) {
                return reply_err(req, state, e);
            }
            reply_json(
                req,
                state,
                200,
                &json!({
                    "object": "list",
                    "data": [{
                        "id": state.cfg.model_name,
                        "object": "model",
                        "created": state.created,
                        "owned_by": "local",
                    }],
                }),
            )
        }
        "/v1/status" if is_get => {
            let s = &state.stats;
            reply_json(
                req,
                state,
                200,
                &json!({
                    "status": "ok",
                    "uptime_seconds": state.started.elapsed().as_secs(),
                    "queue": {
                        "queued": s.queued.load(Ordering::SeqCst),
                        "running": s.running.load(Ordering::SeqCst),
                    },
                    "requests": {
                        "chat": s.chat.load(Ordering::SeqCst),
                        "stream": s.stream.load(Ordering::SeqCst),
                        "embeddings": s.embeddings.load(Ordering::SeqCst),
                        "errors": s.errors.load(Ordering::SeqCst),
                    },
                    "model": {
                        "name": state.cfg.model_name,
                        "block_size": state.cfg.block_size,
                        "max_new": state.cfg.max_new,
                        "prompt_format": if state.cfg.use_sft { "sft" } else { "raw" },
                    },
                }),
            )
        }
        // OpenAPI 规范：给 Swagger UI / Postman / Apifox 直接导入，
        // 与实际路由同源生成，改了路由这里不会过期，因此不鉴权。
        "/openapi.json" if is_get => reply_text(
            req,
            state,
            200,
            "application/json; charset=utf-8",
            // 规范是给人读、给 Postman/Apifox 导入的，缩进输出便于阅读与 git diff
            serde_json::to_string_pretty(&openapi_spec(&state.cfg))
                .unwrap_or_else(|_| "{}".into()),
        ),
        "/openapi.yaml" if is_get => reply_text(
            req,
            state,
            200,
            "text/yaml; charset=utf-8",
            json_to_yaml(&openapi_spec(&state.cfg)),
        ),
        "/v1/chat/completions" if is_post => {
            if let Err(e) = check_auth(&req, state) {
                return reply_err(req, state, e);
            }
            let body = match read_body(&mut req) {
                Ok(b) => b,
                Err(e) => return reply_err(req, state, e),
            };
            let job = match parse_chat(&body, &state.cfg) {
                Ok(j) => j,
                Err(e) => return reply_err(req, state, e),
            };
            state.stats.chat.fetch_add(1, Ordering::SeqCst);
            if job.stream {
                state.stats.stream.fetch_add(1, Ordering::SeqCst);
                return stream_reply(req, state, job);
            }
            match run_chat(state, job) {
                Ok(v) => reply_json(req, state, 200, &v),
                Err(e) => reply_err(req, state, e),
            }
        }
        "/v1/embeddings" if is_post => {
            if let Err(e) = check_auth(&req, state) {
                return reply_err(req, state, e);
            }
            let body = match read_body(&mut req) {
                Ok(b) => b,
                Err(e) => return reply_err(req, state, e),
            };
            match run_embeddings(state, &body) {
                Ok(v) => {
                    state.stats.embeddings.fetch_add(1, Ordering::SeqCst);
                    reply_json(req, state, 200, &v)
                }
                Err(e) => reply_err(req, state, e),
            }
        }
        _ if is_get || is_post => reply_err(
            req,
            state,
            ApiError { status: 404, kind: "not_found", message: format!("未知路径 {path}") },
        ),
        _ => {
            // 先取方法名再移动 `req`：参数从左到右求值，写在 reply_err 实参里会"先 move 后借用"
            let method = req.method().as_str().to_string();
            reply_err(
                req,
                state,
                ApiError {
                    status: 405,
                    kind: "invalid_request_error",
                    message: format!("不支持的方法 {method}（可用：GET / POST / OPTIONS）"),
                },
            )
        }
    }
}

/// 一个请求 = 一个线程：这里只负责"记日志 + 分发"
fn handle(req: Request, state: Arc<State>) {
    let started = Instant::now();
    let method = req.method().as_str().to_string();
    let path = req.url().split('?').next().unwrap_or("/").to_string();
    let status = route(req, &state);
    logln!(
        "[serve] {method} {path} → {status}（{:.0}ms）",
        started.elapsed().as_secs_f64() * 1000.0
    );
}

// ==================== OpenAPI 规范 ====================

/// 生成本服务的 OpenAPI 3.1 规范。
///
/// 与路由**同源**：这里写的路径、字段约束逐条对应 `route()` 与 `parse_chat()` 的实现，
/// 改动任一侧都要同步另一侧，`openapi_spec_covers_every_route` 测试会盯着关键项。
///
/// 只依赖 `&ServeCfg`（不依赖 `State`），是为了能在没有模型的单测里直接构造。
fn openapi_spec(cfg: &ServeCfg) -> Value {
    let base = format!("http://{}:{}", cfg.host, cfg.port);
    let health = json!({
        "tags": ["health"],
        "summary": "存活探针",
        "description": "不鉴权。进程活着就回 200，给网关/负载均衡当探针用。",
        "security": [],
        "responses": {
            "200": {
                "description": "服务正常",
                "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Health" } } }
            }
        }
    });
    let status = json!({
        "tags": ["health"],
        "summary": "队列与计数（诊断用）",
        "description": "不鉴权。看当前排队/运行中的请求数、各类请求累计计数与模型配置。",
        "security": [],
        "responses": {
            "200": {
                "description": "当前状态",
                "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Status" } } }
            }
        }
    });
    let models = json!({
        "tags": ["models"],
        "summary": "模型列表",
        "description": "OpenAI 风格的模型列表。本服务只加载一个 checkpoint，所以 data 恒为一条。",
        "responses": {
            "200": {
                "description": "模型列表",
                "content": { "application/json": { "schema": { "$ref": "#/components/schemas/ModelList" } } }
            },
            "401": { "$ref": "#/components/responses/Unauthorized" },
            "405": { "$ref": "#/components/responses/MethodNotAllowed" }
        }
    });
    let openapi_json = json!({
        "tags": ["health"],
        "summary": "OpenAPI 规范（JSON）",
        "description": "不鉴权。本文件自身，可直接喂给 Swagger UI / Postman / Apifox。",
        "security": [],
        "responses": {
            "200": {
                "description": "OpenAPI 3.1 文档",
                "content": { "application/json": { "schema": { "type": "object" } } }
            }
        }
    });
    let openapi_yaml = json!({
        "tags": ["health"],
        "summary": "OpenAPI 规范（YAML）",
        "description": "不鉴权。与 /openapi.json 同源，只是序列化格式不同。",
        "security": [],
        "responses": {
            "200": {
                "description": "OpenAPI 3.1 文档",
                "content": { "text/yaml": { "schema": { "type": "object" } } }
            }
        }
    });
    let chat = json!({
        "tags": ["chat"],
        "summary": "对话补全（可流式）",
        "description": concat!(
            "本服务的核心端点。\n",
            "`stream=false`（缺省）回一个完整的 chat.completion 对象；\n",
            "`stream=true` 回 SSE（`text/event-stream`），帧顺序为：",
            "role 帧 → content 增量帧… → finish_reason 帧 → usage 帧 → `data: [DONE]`。\n",
            "单份模型互斥排队，多个并发请求会依次执行；客户端断开连接会立即中止本次生成。\n",
            "未在 schema 中声明的字段一律忽略（OpenAI 客户端带的 `user`、`logit_bias` 等不会报错），",
            "声明了的字段严格校验。"
        ),
        "requestBody": {
            "required": true,
            "content": {
                "application/json": {
                    "schema": { "$ref": "#/components/schemas/ChatCompletionRequest" },
                    // 导入 Postman/Apifox 时直接拿它当请求体：让工具按 schema
                    // 自己编示例会撞 400（编出 n≠1、或 messages 以 assistant 结尾）
                    "example": {
                        "model": "latest",
                        "messages": [
                            { "role": "system", "content": "你是一个简洁的中文助手。" },
                            { "role": "user", "content": "用一句话介绍 Transformer。" }
                        ],
                        "max_tokens": 64,
                        "temperature": 0.7,
                        "stream": false
                    }
                }
            }
        },
        "responses": {
            "200": {
                "description": "补全结果（非流式为 JSON，流式为 SSE 事件流）",
                "content": {
                    "application/json": {
                        "schema": { "$ref": "#/components/schemas/ChatCompletionResponse" }
                    },
                    "text/event-stream": {
                        "schema": {
                            "type": "string",
                            "description": "每帧形如 `data: {…}\\n\\n`，最后一帧是 `data: [DONE]`；\
                                           增量帧的结构见 ChatCompletionChunk。"
                        }
                    }
                }
            },
            "400": { "$ref": "#/components/responses/BadRequest" },
            "401": { "$ref": "#/components/responses/Unauthorized" },
            "405": { "$ref": "#/components/responses/MethodNotAllowed" },
            "500": { "$ref": "#/components/responses/ServerError" }
        }
    });
    let embeddings = json!({
        "tags": ["embeddings"],
        "summary": "文本向量化",
        "description": format!(
            "取模型最后一层 hidden states 按 token 均值池化成一个向量。\n\
             `input` 可以是字符串、token id，或它们的数组（一批最多 {MAX_EMBED_BATCH} 条）；\
             超长输入只保留窗口内的尾部。"
        ),
        "requestBody": {
            "required": true,
            "content": {
                "application/json": {
                    "schema": { "$ref": "#/components/schemas/EmbeddingRequest" },
                    "example": { "model": "latest", "input": "用一句话介绍 Transformer。" }
                }
            }
        },
        "responses": {
            "200": {
                "description": "向量列表",
                "content": {
                    "application/json": { "schema": { "$ref": "#/components/schemas/EmbeddingResponse" } }
                }
            },
            "400": { "$ref": "#/components/responses/BadRequest" },
            "401": { "$ref": "#/components/responses/Unauthorized" },
            "405": { "$ref": "#/components/responses/MethodNotAllowed" },
            "500": { "$ref": "#/components/responses/ServerError" }
        }
    });

    json!({
        "openapi": "3.1.0",
        "info": {
            "title": "llm-from-scratch 本地模型 API",
            "version": env!("CARGO_PKG_VERSION"),
            "description": concat!(
                "把本项目训练/微调出的单个 checkpoint 包装成 OpenAI 兼容接口。\n",
                "特性：SSE 流式输出、背压队列、客户端断开即中止、API key 鉴权、CORS、请求日志、/v1/status 队列诊断。"
            ),
            "license": { "name": "MIT" }
        },
        "servers": [{ "url": base, "description": "启动时监听的地址（见 --host / --port）" }],
        "tags": [
            { "name": "chat", "description": "对话补全" },
            { "name": "embeddings", "description": "向量化" },
            { "name": "models", "description": "模型信息" },
            { "name": "health", "description": "探针与自描述（不鉴权）" }
        ],
        // 全局默认要鉴权；health/status/openapi 各自用 "security": [] 关掉
        "security": [{ "bearerAuth": [] }],
        "paths": {
            "/health": { "get": health },
            "/v1/status": { "get": status },
            "/v1/models": { "get": models },
            "/openapi.json": { "get": openapi_json },
            "/openapi.yaml": { "get": openapi_yaml },
            "/v1/chat/completions": { "post": chat },
            "/v1/embeddings": { "post": embeddings }
        },
        "components": {
            "securitySchemes": {
                "bearerAuth": {
                    "type": "http",
                    "scheme": "bearer",
                    "description": concat!(
                        "所有业务端点都要带 `Authorization: Bearer <key>`：key 取启动时的 `--api-key`，",
                        "没给则启动时自动生成随机 key 并打印在控制台启动日志里",
                        "（`/health`、`/v1/status`、`/openapi.*` 不鉴权）。\n",
                        "导入 Postman/Apifox 后，只需在 collection 的 Authorization → ",
                        "Bearer Token 里填一次 key，无需逐请求设置。"
                    )
                }
            },
            "responses": {
                "BadRequest": {
                    "description": "请求有误（字段缺失、类型错、超限等），`error.type` 为 invalid_request_error",
                    "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } }
                },
                "Unauthorized": {
                    "description": "缺少或错误的 Authorization 头，`error.type` 为 authentication_error",
                    "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } }
                },
                "MethodNotAllowed": {
                    "description": "该路径不支持此 HTTP 方法",
                    "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } }
                },
                "ServerError": {
                    "description": "生成/编码过程出错，`error.type` 为 server_error。\
                                    注意：SSE 已经开始下发后只能在流内发一个 error 帧，拿不到本响应。",
                    "content": { "application/json": { "schema": { "$ref": "#/components/schemas/Error" } } }
                }
            },
            "schemas": {
                "Error": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {
                        "error": {
                            "type": "object",
                            "required": ["message", "type", "code"],
                            "properties": {
                                "message": { "type": "string", "description": "给开发者看的中文错误说明" },
                                "type": {
                                    "type": "string",
                                    "enum": ["invalid_request_error", "authentication_error", "not_found", "server_error"]
                                },
                                "code": { "type": "integer", "description": "与 HTTP 状态码相同" }
                            }
                        }
                    }
                },
                "Health": {
                    "type": "object",
                    "required": ["status", "model"],
                    "properties": {
                        "status": { "type": "string", "const": "ok" },
                        "model": { "type": "string", "description": "checkpoint 文件名" }
                    }
                },
                "Status": {
                    "type": "object",
                    "required": ["status", "queue", "requests", "model"],
                    "properties": {
                        "status": { "type": "string", "const": "ok" },
                        "uptime_seconds": { "type": "integer" },
                        "queue": {
                            "type": "object",
                            "description": "queued = 已进队还没拿到模型锁；running = 正在生成/编码",
                            "properties": {
                                "queued": { "type": "integer" },
                                "running": { "type": "integer" }
                            }
                        },
                        "requests": {
                            "type": "object",
                            "properties": {
                                "chat": { "type": "integer", "description": "非流式请求数" },
                                "stream": { "type": "integer", "description": "流式请求数" },
                                "embeddings": { "type": "integer" },
                                "errors": { "type": "integer" }
                            }
                        },
                        "model": {
                            "type": "object",
                            "properties": {
                                "name": { "type": "string" },
                                "block_size": { "type": "integer" },
                                "max_new": { "type": "integer" },
                                "prompt_format": { "type": "string", "enum": ["sft", "raw"] }
                            }
                        }
                    }
                },
                "ModelList": {
                    "type": "object",
                    "required": ["object", "data"],
                    "properties": {
                        "object": { "type": "string", "const": "list" },
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["id", "object", "created", "owned_by"],
                                "properties": {
                                    "id": { "type": "string", "description": "checkpoint 文件名" },
                                    "object": { "type": "string", "const": "model" },
                                    "created": { "type": "integer" },
                                    "owned_by": { "type": "string" }
                                }
                            }
                        }
                    }
                },
                "ChatMessage": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {
                            "type": "string",
                            "enum": ["system", "developer", "user", "assistant"],
                            "description": "system/developer 汇成人设，最后一条必须是 user"
                        },
                        "content": { "type": "string" },
                        "image_url": {
                            "type": "string",
                            "description": "可选；只认最后一条 user 消息。输入图的 base64 data URI（data:image/png;base64,...）或裸 base64；服务配了视觉塔（config 的 model.vision）才接受，否则 400"
                        }
                    }
                },
                "ChatCompletionRequest": {
                    "type": "object",
                    "required": ["messages"],
                    "additionalProperties": true,
                    "description": "认识的字段严格校验，不认识的忽略",
                    "properties": {
                        "model": {
                            "type": "string",
                            "description": "可选；服务端固定用启动时加载的 checkpoint，该字段被忽略"
                        },
                        "messages": {
                            "type": "array",
                            "minItems": 1,
                            "maxItems": MAX_MESSAGES,
                            "items": { "$ref": "#/components/schemas/ChatMessage" }
                        },
                        "temperature": {
                            "type": "number", "minimum": 0, "maximum": 5,
                            "description": "缺省用服务级 --temperature"
                        },
                        "top_k": { "type": "integer", "minimum": 0, "maximum": 10_000, "description": "0 = 不限制" },
                        "top_p": { "type": "number", "exclusiveMinimum": 0, "maximum": 1 },
                        "repetition_penalty": { "type": "number", "minimum": 0.1, "maximum": 10 },
                        "repetition_window": { "type": "integer", "minimum": 0, "maximum": 100_000 },
                        "stop": {
                            "description": "缺省用服务级停止标记（SFT 模板标记）",
                            "oneOf": [
                                { "type": "string", "minLength": 1, "maxLength": MAX_STOP_LEN },
                                {
                                    "type": "array",
                                    "minItems": 1, "maxItems": MAX_STOPS,
                                    "items": { "type": "string", "minLength": 1, "maxLength": MAX_STOP_LEN }
                                }
                            ]
                        },
                        "max_tokens": {
                            "type": "integer", "minimum": 1,
                            "description": "缺省用服务级 --max-new；超过 block_size 会被钳到窗口内"
                        },
                        "seed": { "type": "integer", "minimum": 0, "description": "不给则用服务级 --seed" },
                        "stream": { "type": "boolean", "default": false },
                        "n": { "type": "integer", "const": 1, "description": "本服务一次只生成一个回答" }
                    }
                },
                "ChatCompletionResponse": {
                    "type": "object",
                    "required": ["id", "object", "created", "model", "choices", "usage"],
                    "properties": {
                        "id": { "type": "string", "examples": ["chatcmpl-1"] },
                        "object": { "type": "string", "const": "chat.completion" },
                        "created": { "type": "integer" },
                        "model": { "type": "string" },
                        "choices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": { "type": "integer", "const": 0 },
                                    "message": {
                                        "type": "object",
                                        "properties": {
                                            "role": { "type": "string", "const": "assistant" },
                                            "content": { "type": "string" },
                                            "image_url": {
                                                "type": "string",
                                                "description": "生成图片的 data:image/png;base64 URI；只有模型采出完整图片段且服务配了 vq.ckpt 时才有此字段"
                                            }
                                        }
                                    },
                                    "finish_reason": { "type": "string", "enum": ["stop", "length"] },
                                    "logprobs": { "type": "null" }
                                }
                            }
                        },
                        "usage": { "$ref": "#/components/schemas/Usage" }
                    }
                },
                "ChatCompletionChunk": {
                    "type": "object",
                    "description": "SSE 每一帧 data: 后面的 JSON（usage 帧的 choices 是空数组）",
                    "required": ["id", "object", "created", "model", "choices"],
                    "properties": {
                        "id": { "type": "string" },
                        "object": { "type": "string", "const": "chat.completion.chunk" },
                        "created": { "type": "integer" },
                        "model": { "type": "string" },
                        "choices": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "index": { "type": "integer", "const": 0 },
                                    "delta": {
                                        "type": "object",
                                        "description": "首帧只有 role+空 content，其后是增量 content；finish 前可能有一帧只带 image_url（生成图片的 data URI）",
                                        "properties": {
                                            "role": { "type": "string" },
                                            "content": { "type": "string" },
                                            "image_url": {
                                                "type": "string",
                                                "description": "生成图片的 data:image/png;base64 URI；仅当模型采出完整图片段且服务配了 vq.ckpt 时出现"
                                            }
                                        }
                                    },
                                    "finish_reason": { "type": ["string", "null"], "enum": ["stop", "length", null] }
                                }
                            }
                        },
                        "usage": { "$ref": "#/components/schemas/Usage" }
                    }
                },
                "Usage": {
                    "type": "object",
                    "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
                    "properties": {
                        "prompt_tokens": { "type": "integer" },
                        "completion_tokens": { "type": "integer" },
                        "total_tokens": { "type": "integer" },
                        "tokens_per_second": {
                            "type": ["number", "null"],
                            "description": "服务端实测的纯解码段生成速率（tok/s，不含 prefill，也不受网络/缓冲影响）；生成 token 数不足 2 时为 null"
                        }
                    }
                },
                "EmbeddingRequest": {
                    "type": "object",
                    "required": ["input"],
                    "additionalProperties": true,
                    "properties": {
                        "input": {
                            "description": "字符串、token id，或它们的数组（一批最多 32 条）",
                            "oneOf": [
                                { "type": "string" },
                                { "type": "integer", "minimum": 0 },
                                {
                                    "type": "array",
                                    "minItems": 1, "maxItems": MAX_EMBED_BATCH,
                                    "items": { "oneOf": [ { "type": "string" }, { "type": "integer", "minimum": 0 } ] }
                                }
                            ]
                        },
                        "model": { "type": "string", "description": "可选，被忽略（向量来自已加载的 checkpoint）" }
                    }
                },
                "EmbeddingResponse": {
                    "type": "object",
                    "required": ["object", "data", "model", "usage"],
                    "properties": {
                        "object": { "type": "string", "const": "list" },
                        "data": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "object": { "type": "string", "const": "embedding" },
                                    "index": { "type": "integer" },
                                    "embedding": { "type": "array", "items": { "type": "number" } }
                                }
                            }
                        },
                        "model": { "type": "string" },
                        "usage": {
                            "type": "object",
                            "properties": {
                                "prompt_tokens": { "type": "integer" },
                                "total_tokens": { "type": "integer" }
                            }
                        }
                    }
                }
            }
        }
    })
}

/// 把 `serde_json::Value` 序列化成 YAML 文档。
///
/// 不引入 yaml 库：本服务的规范文档只用到 JSON 的四种结构（对象/数组/标量/空值），
/// 按 YAML 的块序列（block sequence）+ 块映射（block mapping）写出来即可。
/// 字符串一律按最保守的规则决定裸写还是双引号，避免把 `yes`、`1.1.1` 之类
/// 写成裸标量后被 YAML 解析回 bool / 数字。
fn json_to_yaml(v: &Value) -> String {
    let mut out = String::new();
    emit_yaml(v, 0, &mut out);
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// 启动时把规范写进 `dir`（`openapi/openapi.json` + `openapi/openapi.yaml`）。
///
/// 内容与端点完全同源（都出自 [`openapi_spec`]）：仓库里的静态文件每次
/// `serve` 启动都会刷新，手删了下次启动也会自动回来。
/// 写盘失败只打一行警告，不影响服务本身。
fn dump_openapi_files(dir: &str, cfg: &ServeCfg) {
    let spec = openapi_spec(cfg);
    let files = [
        (
            format!("{dir}/openapi.json"),
            serde_json::to_string_pretty(&spec).unwrap_or_else(|_| "{}".into()),
        ),
        (format!("{dir}/openapi.yaml"), json_to_yaml(&spec)),
    ];
    if let Err(e) = std::fs::create_dir_all(dir) {
        logln!("[serve] 创建 {dir}/ 失败（不影响服务）：{e}");
        return;
    }
    for (path, body) in files {
        match std::fs::write(&path, body) {
            Ok(()) => logln!("[serve] 已写入 {path}（接口规范静态版，与端点同源）"),
            Err(e) => logln!("[serve] 写 {path} 失败（不影响服务）：{e}"),
        }
    }
}

/// 缩进 `indent` 个空格后写入一个非空对象/数组（标量由调用方内联处理）
fn emit_yaml(v: &Value, indent: usize, out: &mut String) {
    match v {
        Value::Object(map) => {
            for (k, val) in map {
                let pad = " ".repeat(indent);
                match yaml_inline(val) {
                    Some(s) => out.push_str(&format!("{pad}{}: {s}\n", yaml_key(k))),
                    None => {
                        out.push_str(&format!("{pad}{}:\n", yaml_key(k)));
                        emit_yaml(val, indent + 2, out);
                    }
                }
            }
        }
        Value::Array(items) => {
            for item in items {
                let pad = " ".repeat(indent);
                match yaml_inline(item) {
                    Some(s) => out.push_str(&format!("{pad}- {s}\n")),
                    None => {
                        // 先按 indent+2 渲染，再把首行的缩进换成 "- "：
                        // 两者宽度相同（2 个空格 = "- "），后续行缩进天然对齐。
                        // 首行 head_len 字节本来就全是空格，只需改前 2 个字节。
                        let start = out.len();
                        emit_yaml(item, indent + 2, out);
                        // 把第 indent..indent+2 这两个空格换成 "- "，
                        // 前面的缩进原样保留
                        out.replace_range(start + indent..start + indent + 2, "- ");
                    }
                }
            }
        }
        // 顶层就是标量（OpenAPI 文档不会这样，但通用工具就该能处理）
        other => out.push_str(&yaml_inline(other).unwrap_or_else(|| "null".into())),
    }
}

/// 能写成一行的值：标量、空数组、空对象
fn yaml_inline(v: &Value) -> Option<String> {
    Some(match v {
        Value::Null => "null".into(),
        Value::Bool(b) => b.to_string(),
        Value::Number(n) => n.to_string(),
        Value::String(s) => yaml_str(s),
        Value::Array(a) if a.is_empty() => "[]".into(),
        Value::Object(o) if o.is_empty() => "{}".into(),
        _ => return None,
    })
}

/// 字符串加引号：控制字符走 JSON 转义（YAML 双引号支持同样的转义）
fn yaml_str(s: &str) -> String {
    let plain_ok = !s.is_empty()
        && s == s.trim()
        && !s.contains(':')
        && !s.contains('#')
        && !s.contains('\n')
        && !s.contains('\t')
        && !s.starts_with(['-', '?', ',', '[', ']', '{', '}', '#', '&', '*', '!', '|', '>', '\'', '"', '%', '@', '`'])
        // 数字长相的一律引号：`1.1.1`、`0x1F`、`01` 在某些 YAML 版本里会被当成数/布尔
        && !s.starts_with(|c: char| c.is_ascii_digit())
        && !s.eq_ignore_ascii_case("true")
        && !s.eq_ignore_ascii_case("false")
        && !s.eq_ignore_ascii_case("null")
        && !s.eq_ignore_ascii_case("yes")
        && !s.eq_ignore_ascii_case("no")
        && !s.eq_ignore_ascii_case("on")
        && !s.eq_ignore_ascii_case("off")
        && s.parse::<f64>().is_err();
    if plain_ok { s.to_string() } else { serde_json::to_string(s).unwrap_or_else(|_| "null".into()) }
}

/// 对象键：同样按裸写/引号规则判断（键也可能被解析成数字）
fn yaml_key(k: &str) -> String {
    let plain_ok = !k.is_empty()
        && !k.contains(':')
        && !k.contains('#')
        && k.chars().next().is_some_and(|c| c.is_alphanumeric() || c == '_')
        && k.parse::<f64>().is_err();
    if plain_ok { k.to_string() } else { serde_json::to_string(k).unwrap_or_else(|_| "null".into()) }
}



/// 一次已校验过的对话请求
struct ChatJob {
    /// OpenAI 风格的补全 id：`chatcmpl-<序号>`
    id: String,
    /// 请求里的 system/developer 消息（还没和服务级 system 合并）
    system: String,
    /// 中间消息拼成的历史（不含 system、不含末条 user）
    history: String,
    /// 历史消息条数（不含 system、不含末条 user）——问答日志只记条数，不重刷全文
    history_turns: usize,
    /// 末条 user 消息
    input: String,
    /// 末条 user 消息带的输入图（`image_url` 字段，`None` = 纯文本请求）
    image: Option<String>,
    max_tokens: usize,
    sample: SampleOpts,
    /// 请求指定的种子；`None` = 用服务级 `--seed`
    seed: Option<u64>,
    stream: bool,
}

/// 单调递增的补全 id 序号（`chatcmpl-1`、`chatcmpl-2`…）
static REQ_SEQ: AtomicU64 = AtomicU64::new(1);

/// 解析并校验 `POST /v1/chat/completions` 的请求体。
///
/// 校验刻意严格：错误信息里要指出**哪个字段**错在哪，否则调用方只能看到 400 干瞪眼。
/// 未知字段一律忽略——OpenAI 的客户端（temperature 之外还会带 `user`、`logit_bias` 等）
/// 才能直连本服务。
fn parse_chat(body: &str, cfg: &ServeCfg) -> Result<ChatJob, ApiError> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| bad(format!("请求体不是合法 JSON：{e}")))?;
    let obj = v.as_object().ok_or_else(|| bad("请求体必须是 JSON 对象"))?;

    // ---- messages ----
    let msgs_val = obj.get("messages").ok_or_else(|| bad("缺少 messages 字段"))?;
    let arr = msgs_val.as_array().ok_or_else(|| bad("messages 必须是数组"))?;
    if arr.is_empty() {
        return Err(bad("messages 不能为空"));
    }
    if arr.len() > MAX_MESSAGES {
        return Err(bad(format!("messages 最多 {MAX_MESSAGES} 条，收到 {} 条", arr.len())));
    }
    let mut msgs: Vec<(String, String)> = Vec::with_capacity(arr.len());
    for m in arr {
        let m = m.as_object().ok_or_else(|| bad("messages 的元素必须是对象"))?;
        let role = m
            .get("role")
            .and_then(Value::as_str)
            .ok_or_else(|| bad("message 缺少 role 字段"))?;
        if !matches!(role, "system" | "developer" | "user" | "assistant") {
            return Err(bad(format!(
                "不支持的 role「{role}」（只支持 system / developer / user / assistant）"
            )));
        }
        let content = m
            .get("content")
            .and_then(Value::as_str)
            .ok_or_else(|| bad(format!("{role} 消息的 content 必须是字符串")))?;
        msgs.push((role.to_string(), content.to_string()));
    }
    // 末条必须是 user：prompt 模板要靠它拼"轮到模型说话"的位置（见 prompt::assemble）
    if msgs.last().map(|(r, _)| r.as_str()) != Some("user") {
        return Err(bad("messages 的最后一条必须是 role=user"));
    }

    // ---- 输入图（只认末条 user 的 image_url）----
    // null 当没带；非字符串是客户端拼包错误，直接指出字段名
    let image = match arr.last().and_then(|m| m.get("image_url")) {
        None | Some(Value::Null) => None,
        Some(Value::String(s)) => {
            if s.is_empty() {
                return Err(bad("message.image_url 不能为空字符串"));
            }
            Some(s.clone())
        }
        Some(_) => return Err(bad("message.image_url 必须是字符串（data URI 或裸 base64）")),
    };

    // 历史条数 = 非 system 消息去掉末条 user（问答日志用，只记条数不记全文）
    let history_turns =
        msgs.iter().filter(|(r, _)| !matches!(r.as_str(), "system" | "developer")).count() - 1;
    let (system, history, input) = split_messages(&msgs, cfg.use_sft);

    // ---- 采样参数（逐个覆盖服务级默认值）----
    let mut sample = cfg.sample;

    if let Some(v) = obj.get("temperature") {
        let t = v.as_f64().ok_or_else(|| bad("temperature 必须是数字"))? as f32;
        if !t.is_finite() || !(0.0..=5.0).contains(&t) {
            return Err(bad("temperature 取值范围是 [0, 5]"));
        }
        sample.temperature = t;
    }
    if let Some(v) = obj.get("top_k") {
        let k = v.as_u64().ok_or_else(|| bad("top_k 必须是非负整数"))?;
        if k > 10_000 {
            return Err(bad("top_k 最大 10000（0 = 不限制）"));
        }
        sample.top_k = k as usize;
    }
    if let Some(v) = obj.get("top_p") {
        let p = v.as_f64().ok_or_else(|| bad("top_p 必须是数字"))?;
        if !(0.0 < p && p <= 1.0) {
            return Err(bad("top_p 取值范围是 (0, 1]"));
        }
        sample.top_p = p as f32;
    }
    if let Some(v) = obj.get("repetition_penalty") {
        let r = v.as_f64().ok_or_else(|| bad("repetition_penalty 必须是数字"))?;
        if !(0.1..=10.0).contains(&r) {
            return Err(bad("repetition_penalty 取值范围是 [0.1, 10]"));
        }
        sample.repetition_penalty = r as f32;
    }
    if let Some(v) = obj.get("repetition_window") {
        let w = v.as_u64().ok_or_else(|| bad("repetition_window 必须是非负整数"))?;
        if w > 100_000 {
            return Err(bad("repetition_window 最大 100000"));
        }
        sample.repetition_window = w as usize;
    }

    // ---- stop：可能来自请求，走驻留表换成 &'static ----
    sample.stop = match obj.get("stop") {
        None | Some(Value::Null) => cfg.sample.stop,
        Some(Value::String(s)) => intern_stops(std::slice::from_ref(s))?,
        Some(Value::Array(a)) => {
            let mut items = Vec::with_capacity(a.len());
            for x in a {
                items.push(
                    x.as_str()
                        .ok_or_else(|| bad("stop 数组的元素必须是字符串"))?
                        .to_string(),
                );
            }
            intern_stops(&items)?
        }
        Some(_) => return Err(bad("stop 必须是字符串或字符串数组")),
    };

    // ---- 其余标量 ----
    let max_tokens = match obj.get("max_tokens") {
        None => cfg.max_new,
        Some(v) => {
            let n = v.as_u64().ok_or_else(|| bad("max_tokens 必须是正整数"))?;
            if n == 0 {
                return Err(bad("max_tokens 必须大于 0"));
            }
            // 钳到上下文窗口内：prompt 侧还会再减掉它算预算（见 run_chat）
            (n as usize).min(cfg.block_size.max(1))
        }
    };
    let seed = match obj.get("seed") {
        None => None,
        Some(Value::Null) => None,
        Some(v) => Some(v.as_u64().ok_or_else(|| bad("seed 必须是非负整数"))?),
    };
    let stream = match obj.get("stream") {
        None | Some(Value::Null) => false,
        Some(v) => v.as_bool().ok_or_else(|| bad("stream 必须是布尔值"))?,
    };
    if let Some(v) = obj.get("n")
        && v.as_u64() != Some(1)
    {
        return Err(bad("本服务只支持 n=1（一次生成一个回答）"));
    }

    Ok(ChatJob {
        id: format!("chatcmpl-{}", REQ_SEQ.fetch_add(1, Ordering::Relaxed)),
        system,
        history,
        history_turns,
        input,
        image,
        max_tokens,
        sample,
        seed,
        stream,
    })
}

/// 把消息列表拆成「system」「历史」「本轮输入」三段。
///
/// - `system` / `developer` 消息无论出现在哪都汇进 system（最后用换行连接）；
/// - 末条 `user` 是本轮输入，其余按 `use_sft` 套角色模板；
/// - SFT 模板下历史长这样：`用户：\nq1\n助手：\na1`（与 `prompt::assemble` 的 tail 同形，
///   拼完整条 prompt 时首尾能自然接上）。
fn split_messages(msgs: &[(String, String)], use_sft: bool) -> (String, String, String) {
    let (last_role, last_content) = msgs.last().expect("调用方已保证非空");
    debug_assert_eq!(last_role, "user");

    let mut systems: Vec<&str> = Vec::new();
    let mut history = String::new();
    // 末条 user 之前的都算历史
    for (role, content) in &msgs[..msgs.len() - 1] {
        match role.as_str() {
            "system" | "developer" => systems.push(content),
            "user" => {
                if !history.is_empty() {
                    history.push('\n');
                }
                if use_sft {
                    history.push_str(&format!("{SFT_USER}\n{content}\n{SFT_ASSISTANT}"));
                } else {
                    history.push_str(content);
                }
            }
            _ => {
                // assistant（校验阶段已挡掉其它 role）
                if !history.is_empty() {
                    history.push('\n');
                }
                history.push_str(content);
            }
        }
    }
    (systems.join("\n"), history, last_content.clone())
}

/// 合并服务级 system 与请求里的 system：**服务级在前**（它相当于"全局人设"，
/// 请求级 system 更像是本轮的临时指令，放后面优先级更高）。
fn merge_system(base: &str, extra: &str) -> String {
    let b = base.trim();
    let e = extra.trim();
    match (b.is_empty(), e.is_empty()) {
        (true, _) => e.to_string(),
        (_, true) => b.to_string(),
        _ => format!("{b}\n{e}"),
    }
}

/// 收尾原因 → OpenAI 的 `finish_reason`
fn finish_of(reason: StopReason) -> &'static str {
    match reason {
        StopReason::Eos | StopReason::StopMark(_) => "stop",
        StopReason::MaxNew => "length",
    }
}

// ==================== 问答记录 ====================

/// 一次生成的结果素材（组装问答日志用）
struct QaOutcome<'a> {
    /// 合并后的 system（真正进 prompt 的那份）
    system: &'a str,
    /// 完整回复文本（可能为空）
    output: &'a str,
    /// 收尾原因；`None` = 没跑到正常收尾（如客户端提前断开，见 `note`）
    reason: Option<StopReason>,
    /// 输入侧 token 数
    n_prompt: usize,
    /// 生成侧 token 数
    n_gen: usize,
    /// 纯解码段速率（tok/s），样本不足时为 None
    rate: Option<f64>,
    /// 附加说明（如「客户端提前断开」）
    note: Option<&'a str>,
}

/// 组装一条问答记录的标题与键值对（纯函数，便于单测）。
///
/// 只记**本轮**问答与参数：历史全文由无状态的调用方每次重发，
/// 原样记下来只会让同一段历史在日志里反复刷屏，因此历史只记条数。
fn qa_log_block(job: &ChatJob, r: &QaOutcome) -> (String, Vec<(&'static str, String)>) {
    let mode = if job.stream { "流式" } else { "非流式" };
    let title = format!("问答 {}（{mode}）", job.id);
    let history = if job.history_turns == 0 {
        "0 条（首轮问答）".to_string()
    } else {
        format!("{} 条", job.history_turns)
    };
    let output = if r.output.is_empty() { "（空）".to_string() } else { r.output.to_string() };
    let finish = match r.reason {
        Some(x) => crate::stop_label(x),
        None => "未正常收尾（见备注）".to_string(),
    };
    let seed = match job.seed {
        Some(s) => s.to_string(),
        None => "未指定（用服务级）".to_string(),
    };
    let rate = match r.rate {
        Some(v) => format!("{v:.1} tok/s"),
        None => "样本不足".to_string(),
    };
    let mut items: Vec<(&'static str, String)> = vec![
        (
            "system",
            if r.system.is_empty() { "（无）".to_string() } else { r.system.to_string() },
        ),
        ("历史", history),
        ("输入", job.input.clone()),
        ("输出", output),
        ("收尾", finish),
        (
            "采样参数",
            format!(
                "temperature={}, top_k={}, top_p={}, repetition_penalty={}, \
                 repetition_window={}, stop={:?}, seed={seed}, max_tokens={}",
                job.sample.temperature,
                job.sample.top_k,
                job.sample.top_p,
                job.sample.repetition_penalty,
                job.sample.repetition_window,
                job.sample.stop,
                job.max_tokens,
            ),
        ),
        (
            "用量",
            format!(
                "prompt {} + completion {} = {} token，{rate}",
                r.n_prompt,
                r.n_gen,
                r.n_prompt + r.n_gen
            ),
        ),
    ];
    if let Some(note) = r.note {
        items.push(("备注", note.to_string()));
    }
    (title, items)
}

/// 把一条问答记录写进运行日志（**只写文件**：控制台保持现有的简洁接口行，
/// 问答正文只落在 `logs/serve_*.log` 里）
fn log_qa(job: &ChatJob, r: &QaOutcome) {
    let (title, mut items) = qa_log_block(job, r);
    items.insert(0, ("时间", runlog::now()));
    runlog::fields(&title, &items);
}

// ==================== chat/completions：非流式 ====================

/// 把模型生成的码本段解码成 `data:image/png;base64,...`（多张图取第一张）。
///
/// 没配 VQ-VAE（`vq = None`）或一段图都没采出来时返回 `None`，
/// 响应里就不带 `image_url` 字段——纯文本模型的回包与从前逐字节一致。
fn generated_image_uri(vq: Option<&crate::vqvae::Vqvae>, imgs: &[Vec<usize>]) -> Option<String> {
    let vq = vq?;
    let ids = imgs.first()?;
    let px = vq.decode(ids, 1);
    let png = crate::vision::encode_png(&px, vq.cfg.image_size);
    logln!(
        "[serve] 生成图片：{} 个码本 token → PNG {} 字节（{}×{}，内嵌 data URI）",
        ids.len(),
        png.len(),
        vq.cfg.image_size,
        vq.cfg.image_size
    );
    Some(format!("data:image/png;base64,{}", b64encode(&png)))
}

/// 标准 base64（带 `=` 填充、无换行），只为 data URI 服务。
/// 项目零第三方编码依赖，16 行标准表比引一个 crate 划算。
fn b64encode(bytes: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for c in bytes.chunks(3) {
        let n = (c[0] as u32) << 16 | (*c.get(1).unwrap_or(&0) as u32) << 8 | *c.get(2).unwrap_or(&0) as u32;
        out.push(T[(n >> 18) as usize & 63] as char);
        out.push(T[(n >> 12) as usize & 63] as char);
        out.push(if c.len() > 1 { T[(n >> 6) as usize & 63] as char } else { '=' });
        out.push(if c.len() > 2 { T[n as usize & 63] as char } else { '=' });
    }
    out
}

fn run_chat(state: &Arc<State>, job: ChatJob) -> Result<Value, ApiError> {
    let system = merge_system(&state.cfg.system, &job.system);
    // 输入图先解码（锁外、可回 400）：图解不出来就不进生成队列。
    // 训练格式补全也在完成——`input` 是补过 `图:<|image|>` 的版本，日志仍记原始 `job.input`
    let (input, pixels) =
        prepare_vision(state.cfg.vision.as_ref(), &job.input, job.image.as_deref())?;
    let max_tokens = job.max_tokens;
    // 给生成预留 max_new，剩下的才是输入预算（见 prompt::assemble 的分配顺序）
    let prompt_budget = state.cfg.block_size.saturating_sub(max_tokens);
    let use_sft = state.cfg.use_sft;
    let kv = state.cfg.kv;
    let mut owned_rng = job.seed.map(Rng::new);

    let produced = with_core(state, |core| {
        // 拆开字段借用：`Generator` 同时要 `&Transformer` / `&Tokenizer` 和 `&mut Rng`，
        // 直接借 `core` 会因为"一个可变 + 两个共享"打架
        let Core { model, tokenizer, rng: shared_rng } = core;
        let a = prompt::assemble(tokenizer, &system, &job.history, &input, use_sft, prompt_budget);
        let rng = match owned_rng.as_mut() {
            Some(r) => r,
            None => shared_rng,
        };
        let mut g = Generator::new(model, tokenizer, &a.prompt, max_tokens, &job.sample, kv, rng);
        // 图片理解：像素已在锁外解码好，这里只是挂上（prefill 时视觉塔覆写占位符）
        if let Some(px) = pixels.as_ref() {
            g = g.with_pixels(px.clone());
        }
        let n_prompt = g.prompt_tokens();
        // 与 `g.run()` 等价的手写循环：多拿一个「prefill 结束」的计时点（见 DecodeTimer）
        let mut timer = DecodeTimer::new();
        while g.step() {
            timer.tick();
        }
        let n_gen = g.generated_tokens();
        let rate = timer.rate(n_gen);
        // 图片生成：完整的码本段先留下（VQ 解码在锁外做，别占着模型锁）
        let imgs = g.generated_images();
        let out = g.into_output();
        (out.generated, imgs, n_prompt, n_gen, out.reason, a.trimmed, rate)
    });
    // 出错也要留一条问答记录：题目在、答案没了，日志里得能查到这条请求的下文
    let (text, imgs, n_prompt, n_gen, reason, trimmed, rate) = match produced {
        Ok(v) => v,
        Err(e) => {
            let note = format!("生成出错：{}", e.message);
            log_qa(
                &job,
                &QaOutcome {
                    system: &system,
                    output: "",
                    reason: None,
                    n_prompt: 0,
                    n_gen: 0,
                    rate: None,
                    note: Some(&note),
                },
            );
            return Err(e);
        }
    };
    // 生成了图片且配了 VQ-VAE：解码成 PNG 内嵌 data URI（纯文本模型恒为 None）
    let image_url = generated_image_uri(state.vq.as_ref(), &imgs);

    if let Some((before, after)) = trimmed {
        logln!(
            "[serve] 历史超出输入预算：{before} → {after} token（预算 {prompt_budget}）"
        );
    }

    // 问答正文只写日志文件（控制台仍只有上面那行接口日志）
    log_qa(
        &job,
        &QaOutcome {
            system: &system,
            output: &text,
            reason: Some(reason),
            n_prompt,
            n_gen,
            rate,
            note: None,
        },
    );

    Ok(json!({
        "id": job.id,
        "object": "chat.completion",
        "created": state.created,
        "model": state.cfg.model_name,
        "choices": [{
            "index": 0,
            "message": match &image_url {
                // 生成了图片：text 里含 <|image|> 字面量，真图内嵌在 image_url
                Some(u) => json!({ "role": "assistant", "content": text, "image_url": u }),
                None => json!({ "role": "assistant", "content": text }),
            },
            "finish_reason": finish_of(reason),
            "logprobs": null,
        }],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": n_gen,
            "total_tokens": n_prompt + n_gen,
            // 纯解码段的生成速率（不含 prefill），样本不足时为 null
            "tokens_per_second": rate,
        },
    }))
}

// ==================== chat/completions：SSE 流式 ====================

/// 一个 SSE 帧
fn sse(text: &str) -> Vec<u8> {
    format!("data: {text}\n\n").into_bytes()
}

/// 造一个 `chat.completion.chunk`
fn delta_chunk(state: &Arc<State>, id: &str, delta: Value, finish: Option<&str>) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.cfg.model_name,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": finish.map(Value::from).unwrap_or(Value::Null),
            "logprobs": null,
        }],
    })
    .to_string()
}

/// usage 单独发一帧（choices 为空数组）——这是 OpenAI 的实际做法
fn usage_chunk(
    state: &Arc<State>,
    id: &str,
    n_prompt: usize,
    n_gen: usize,
    rate: Option<f64>,
) -> String {
    json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": state.created,
        "model": state.cfg.model_name,
        "choices": [],
        "usage": {
            "prompt_tokens": n_prompt,
            "completion_tokens": n_gen,
            "total_tokens": n_prompt + n_gen,
            // 纯解码段的生成速率（不含 prefill），样本不足时为 null
            "tokens_per_second": rate,
        },
    })
    .to_string()
}

/// 解码段计时：`Generator::step` 的第一步含 prefill（把整个 prompt 喂进模型），
/// 从第二步起才是逐 token 解码。首步结束时落表、收尾时算速率，
/// 这样 tok/s 只反映真实生成速度，不被 prefill 或网络/缓冲耗时污染
/// （前端本地墙钟测 SSE 到达间隔会被 TCP/代理缓冲骗到，见 web 端展示逻辑）。
struct DecodeTimer(Option<Instant>);

impl DecodeTimer {
    fn new() -> Self {
        Self(None)
    }

    /// 每次 `step()` 成功返回后调一次；只在第一次（prefill 刚结束）落表
    fn tick(&mut self) {
        if self.0.is_none() {
            self.0 = Some(Instant::now());
        }
    }

    /// tok/s = 解码窗口内新生成的 token 数 ÷ 窗口时长。
    /// 不足 2 个 token 时窗口里只有一个样本，无从算速率，返回 None。
    fn rate(&self, n_gen: usize) -> Option<f64> {
        let ms = self.0?.elapsed().as_secs_f64() * 1000.0;
        (n_gen >= 2 && ms > 0.0).then_some((n_gen - 1) as f64 * 1000.0 / ms)
    }
}

/// 流式回复的结果（闭包把信息带回外层，由外层发 finish/usage/DONE）
struct StreamOutcome {
    n_prompt: usize,
    n_gen: usize,
    reason: StopReason,
    /// 完整回复文本（问答日志用；中断时是已生成的部分）
    text: String,
    trimmed: Option<(usize, usize)>,
    /// 客户端断开了，别再发任何东西
    aborted: bool,
    /// 纯解码段速率（tok/s），样本不足时为 None
    rate: Option<f64>,
    /// 生成出的码本段（锁外用 VQ 解码成 data URI，在 finish 帧之前发出）
    imgs: Vec<Vec<usize>>,
}

fn stream_reply(req: Request, state: &Arc<State>, job: ChatJob) -> u16 {
    // 有界通道 = 背压：生成线程最多领先客户端 16 帧，
    // 客户端读得慢时生成会自己慢下来，而不是把内存堆爆。
    let (tx, rx) = sync_channel::<Vec<u8>>(16);
    let st = Arc::clone(state);
    thread::spawn(move || stream_generate(st, tx, job));

    let resp = Response::new(
        StatusCode(200),
        vec![
            header("Content-Type", "text/event-stream; charset=utf-8"),
            header("Cache-Control", "no-cache"),
        ],
        SsePipe { rx, buf: Vec::new(), pos: 0 },
        None, // 无长度 → HTTP/1.1 走 chunked，边生成边发
        None,
    );
    // 这里会阻塞到整条流写完（或客户端断开）
    let _ = req.respond(with_cors(state, resp));
    200
}

/// 生成线程：把 SSE 帧一帧帧塞进 channel，`tx` 一断（客户端断开）就收手
fn stream_generate(state: Arc<State>, tx: SyncSender<Vec<u8>>, job: ChatJob) {
    let id = job.id.clone();
    // 第一帧先声明角色：OpenAI 客户端靠它拿到 assistant 角色
    if tx
        .send(sse(&delta_chunk(&state, &id, json!({ "role": "assistant", "content": "" }), None)))
        .is_err()
    {
        return;
    }

    let system = merge_system(&state.cfg.system, &job.system);
    let prompt_budget = state.cfg.block_size.saturating_sub(job.max_tokens);
    let max_tokens = job.max_tokens;
    let use_sft = state.cfg.use_sft;
    let kv = state.cfg.kv;
    let mut owned_rng = job.seed.map(Rng::new);

    // 输入图解码（锁外）。响应头已是 200，出错只能走 SSE 报错：error 帧 + [DONE]
    let (input, pixels) =
        match prepare_vision(state.cfg.vision.as_ref(), &job.input, job.image.as_deref()) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(sse(&e.to_value().to_string()));
                let _ = tx.send(b"data: [DONE]\n\n".to_vec());
                let note = format!("图片输入错误：{}", e.message);
                log_qa(
                    &job,
                    &QaOutcome {
                        system: &system,
                        output: "",
                        reason: None,
                        n_prompt: 0,
                        n_gen: 0,
                        rate: None,
                        note: Some(&note),
                    },
                );
                return;
            }
        };

    let res = with_core(&state, |core| {
        let Core { model, tokenizer, rng: shared_rng } = core;
        let a = prompt::assemble(tokenizer, &system, &job.history, &input, use_sft, prompt_budget);
        let trimmed = a.trimmed;
        let rng = match owned_rng.as_mut() {
            Some(r) => r,
            None => shared_rng,
        };
        let mut g = Generator::new(model, tokenizer, &a.prompt, max_tokens, &job.sample, kv, rng);
        // 图片理解：像素已在锁外解码好，这里只是挂上（prefill 时视觉塔覆写占位符）
        if let Some(px) = pixels.as_ref() {
            g = g.with_pixels(px.clone());
        }
        let n_prompt = g.prompt_tokens();

        let mut timer = DecodeTimer::new();
        while g.step() {
            timer.tick();
            // send 失败 = 接收端（响应流）已 drop = 客户端断开：
            // 立刻停生成，别在一个没人听的请求上烧 CPU
            if let Some(delta) = g.next_text()
                && tx.send(sse(&delta_chunk(&state, &id, json!({ "content": delta }), None))).is_err()
            {
                let n_gen = g.generated_tokens();
                let text = g.into_output().generated;
                return StreamOutcome {
                    n_prompt,
                    n_gen,
                    reason: StopReason::MaxNew,
                    text,
                    trimmed,
                    aborted: true,
                    rate: timer.rate(n_gen),
                    imgs: Vec::new(),
                };
            }
        }
        // 收尾后把被 hold-back 的尾巴一次发干净（结果已定，不会再变）
        let tail = g.flush();
        let n_gen = g.generated_tokens();
        let rate = timer.rate(n_gen);
        // 图片生成：完整码本段留下，VQ 解码放到锁外做（见 finish 帧之前那段）
        let imgs = g.generated_images();
        let out = g.into_output();
        let aborted = !tail.is_empty()
            && tx.send(sse(&delta_chunk(&state, &id, json!({ "content": tail }), None))).is_err();
        StreamOutcome { n_prompt, n_gen, reason: out.reason, text: out.generated, trimmed, aborted, rate, imgs }
    });

    let outcome = match res {
        Ok(o) => o,
        Err(e) => {
            // 生成中途出错：发一个 OpenAI 风格的 error 事件，客户端才知道不是正常结束
            let _ = tx.send(sse(&e.to_value().to_string()));
            let _ = tx.send(b"data: [DONE]\n\n".to_vec());
            // 响应头早已按 200 发出，接口行看不出这次失败——问答日志必须留痕
            let note = format!("生成出错：{}", e.message);
            log_qa(
                &job,
                &QaOutcome {
                    system: &system,
                    output: "",
                    reason: None,
                    n_prompt: 0,
                    n_gen: 0,
                    rate: None,
                    note: Some(&note),
                },
            );
            return;
        }
    };
    if outcome.aborted {
        logln!("[serve] 客户端提前断开，生成已中止（{} token）", outcome.n_gen);
        // 中断也记一条：只写已生成的部分，收尾原因标为「未正常收尾」
        log_qa(
            &job,
            &QaOutcome {
                system: &system,
                output: &outcome.text,
                reason: None,
                n_prompt: outcome.n_prompt,
                n_gen: outcome.n_gen,
                rate: outcome.rate,
                note: Some("客户端提前断开，只记到已生成的部分"),
            },
        );
        return;
    }
    if let Some((before, after)) = outcome.trimmed {
        logln!("[serve] 历史超出输入预算：{before} → {after} token（预算 {prompt_budget}）");
    }
    // 图片生成：锁外解码成 data URI，在 finish 帧**之前**发一个 image_url delta 帧
    // （客户端在 `delta.image_url` 里拿到真图，与非流式 `message.image_url` 同源）
    if let Some(uri) = generated_image_uri(state.vq.as_ref(), &outcome.imgs)
        && tx.send(sse(&delta_chunk(&state, &id, json!({ "image_url": uri }), None))).is_err()
    {
        return;
    }
    let fr = finish_of(outcome.reason);
    let _ = tx.send(sse(&delta_chunk(&state, &id, json!({}), Some(fr))));
    let _ = tx.send(sse(&usage_chunk(
        &state,
        &id,
        outcome.n_prompt,
        outcome.n_gen,
        outcome.rate,
    )));
    let _ = tx.send(b"data: [DONE]\n\n".to_vec());
    logln!(
        "[serve] 流式完成 {}：{} token（{}）",
        id, outcome.n_gen, crate::stop_label(outcome.reason)
    );
    // 问答正文只写日志文件（控制台仍只有上面那行接口日志）
    log_qa(
        &job,
        &QaOutcome {
            system: &system,
            output: &outcome.text,
            reason: Some(outcome.reason),
            n_prompt: outcome.n_prompt,
            n_gen: outcome.n_gen,
            rate: outcome.rate,
            note: None,
        },
    );
}

/// 把 channel 里的 SSE 帧转成 tiny_http 能边发边读的响应体。
///
/// `Read` 的语义决定了一切：读到 `Ok(0)` 就是 EOF，所以 channel 关闭时直接返回 0，
/// 响应自然收尾（chunked 流以 `0\r\n\r\n` 结束）。
struct SsePipe {
    rx: Receiver<Vec<u8>>,
    buf: Vec<u8>,
    pos: usize,
}

impl Read for SsePipe {
    fn read(&mut self, out: &mut [u8]) -> io::Result<usize> {
        if out.is_empty() {
            return Ok(0);
        }
        loop {
            if self.pos < self.buf.len() {
                let n = (self.buf.len() - self.pos).min(out.len());
                out[..n].copy_from_slice(&self.buf[self.pos..self.pos + n]);
                self.pos += n;
                return Ok(n);
            }
            match self.rx.recv() {
                Ok(b) => {
                    self.buf = b;
                    self.pos = 0;
                }
                // 发送端全部 drop：流结束
                Err(_) => return Ok(0),
            }
        }
    }
}

// ==================== embeddings ====================

/// 一条待编码的输入
enum EmbedIn {
    Text(String),
    Ids(Vec<usize>),
}

fn run_embeddings(state: &Arc<State>, body: &str) -> Result<Value, ApiError> {
    let v: Value =
        serde_json::from_str(body).map_err(|e| bad(format!("请求体不是合法 JSON：{e}")))?;
    let obj = v.as_object().ok_or_else(|| bad("请求体必须是 JSON 对象"))?;
    let input = obj.get("input").ok_or_else(|| bad("缺少 input 字段"))?;

    let items: Vec<EmbedIn> = match input {
        Value::String(s) => vec![EmbedIn::Text(s.clone())],
        Value::Number(_) => vec![EmbedIn::Ids(vec![as_usize(input)?])],
        Value::Array(a) => {
            if a.is_empty() {
                return Err(bad("input 数组不能为空"));
            }
            if a.len() > MAX_EMBED_BATCH {
                return Err(bad(format!("input 数组最多 {MAX_EMBED_BATCH} 条")));
            }
            let mut out = Vec::with_capacity(a.len());
            for x in a {
                out.push(match x {
                    Value::String(s) => EmbedIn::Text(s.clone()),
                    Value::Number(_) => EmbedIn::Ids(vec![as_usize(x)?]),
                    _ => return Err(bad("input 数组的元素只能是字符串或 token id")),
                });
            }
            out
        }
        _ => return Err(bad("input 必须是字符串、token id，或它们的数组")),
    };

    let block_size = state.cfg.block_size;
    let (rows, prompt_tokens) = with_core(state, |core| {
        let Core { model, tokenizer, .. } = core;
        let vocab = model.cfg.vocab_size;
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(items.len());
        let mut prompt_tokens = 0usize;
        for item in &items {
            let mut ids = match item {
                EmbedIn::Text(s) => {
                    let mut v = match tokenizer.bos_id() {
                        Some(bos) => vec![bos],
                        None => Vec::new(),
                    };
                    v.extend(tokenizer.encode(s));
                    v
                }
                EmbedIn::Ids(v) => v.clone(),
            };
            // 越界的 token id 会直接下标越界 panic，这里先挡成 400
            if let Some(bad_id) = ids.iter().find(|&&x| x >= vocab) {
                return Err(bad(format!("token id {bad_id} 超出词表大小 {vocab}")));
            }
            if ids.is_empty() {
                ids.push(0); // 空输入：给一个位置，避免 0 长度前向
            }
            if ids.len() > block_size {
                // 与生成一样只看窗口内的内容（超长的头部本来也会被滑出）
                ids = ids[ids.len() - block_size..].to_vec();
            }
            prompt_tokens += ids.len();
            let hidden = crate::tensor::no_grad(|| model.forward_hidden(&ids, 1, ids.len(), false));
            rows.push(mean_pool(&hidden));
        }
        Ok((rows, prompt_tokens))
    })??;

    let data: Vec<Value> = rows
        .iter()
        .enumerate()
        .map(|(i, r)| json!({ "object": "embedding", "index": i, "embedding": r }))
        .collect();

    Ok(json!({
        "object": "list",
        "data": data,
        "model": state.cfg.model_name,
        "usage": { "prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens },
    }))
}

fn as_usize(v: &Value) -> Result<usize, ApiError> {
    v.as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| bad("token id 必须是非负整数"))
}

/// 按 token 维度对 hidden states 做**均值池化**，得到一个向量。
///
/// `forward_hidden` 回的是 `[B*T, H]`（`B=1`），所以"按行平均"就是把整条序列
/// 平均成一个长度 `H` 的向量——与常见句向量模型的 pooling 一致。
/// 非有限值（上游数值溢出）一律置 0：`null` 混进 embedding 数组会让客户端直接报错。
fn mean_pool(hidden: &crate::tensor::Tensor) -> Vec<f64> {
    let shape = hidden.shape();
    let h = *shape.last().unwrap_or(&1);
    if h == 0 {
        return Vec::new();
    }
    let data = hidden.data();
    let rows = data.len() / h;
    let mut acc = vec![0f64; h];
    for r in 0..rows {
        for (i, x) in data[r * h..(r + 1) * h].iter().enumerate() {
            acc[i] += *x as f64;
        }
    }
    let denom = rows.max(1) as f64;
    for x in &mut acc {
        *x /= denom;
        if !x.is_finite() {
            *x = 0.0;
        }
    }
    acc
}

// ==================== 前端测试页 ====================

/// 随 API 一起拉起 `web/` 前端测试页（vite dev server 子进程）。
///
/// - 定位 `web/`：优先**当前工作目录**（部署时 web/ 与二进制并排放），
///   找不到再退回编译时的仓库目录（开发时从子目录 `cargo run`）；
/// - 找不到工程或没装依赖时只警告就返回，绝不影响 API 启动；
/// - 通过 `API_PORT` 环境变量把 API 端口传给 vite 代理，页面经**同源转发**
///   调接口，因此不需要开 `--cors`；
/// - 通过 `VITE_API_KEY` 把服务的 key 交给页面（`VITE_` 前缀才会进
///   `import.meta.env`）：key 可能是启动时随机生成的，页面自动填好才免手抄；
/// - 子进程交给独立线程收割（避免僵尸进程），共享控制台收到 Ctrl-C 时
///   vite 也会一并收到，两个服务同时停止。
pub fn spawn_web_frontend(api_port: u16, web_port: u16, api_key: Option<&str>) {
    // 候选目录：cwd/web（部署形态）→ CARGO_MANIFEST_DIR/web（开发形态）
    let mut candidates = Vec::new();
    if let Ok(cwd) = std::env::current_dir() {
        candidates.push(cwd.join("web"));
    }
    candidates.push(std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("web"));

    let Some(web_dir) = candidates.into_iter().find(|p| p.join("package.json").is_file()) else {
        logln!("[serve] 未找到 web/ 前端工程，跳过测试页（用 --no-web 可去掉本提示）");
        return;
    };
    // 直接用 node 拉 vite 的 bin，避开 Windows 上 npm 是 npm.cmd 而 Command 找不到的问题
    let vite_bin = web_dir.join("node_modules").join("vite").join("bin").join("vite.js");
    if !vite_bin.is_file() {
        logln!(
            "[serve] web/ 依赖未安装，跳过测试页：cd {} && npm install",
            web_dir.display()
        );
        return;
    }

    let mut cmd = std::process::Command::new("node");
    cmd.arg(&vite_bin)
        .current_dir(&web_dir)
        // vite.config.ts 读这两个环境变量：代理目标端口 + dev server 端口
        .env("API_PORT", api_port.to_string())
        .env("WEB_PORT", web_port.to_string());
    if let Some(key) = api_key {
        // 注入鉴权 key：本地测试页与后端同机同源，注入即可用（key 只在本机生效）
        cmd.env("VITE_API_KEY", key);
    }
    let child = cmd
        .stdout(std::process::Stdio::inherit())
        .stderr(std::process::Stdio::inherit())
        .spawn();

    match child {
        Ok(mut child) => {
            logln!("[serve] 前端测试页已启动：http://localhost:{web_port}（代理 API → 127.0.0.1:{api_port}）");
            thread::spawn(move || {
                // 收割子进程；异常退出时打一行日志（如端口被占）
                if let Ok(status) = child.wait()
                    && !status.success()
                {
                    logln!("[serve] 前端测试页退出（{status}），可手动 cd web && npm run dev 重启");
                }
            });
        }
        Err(e) => {
            logln!("[serve] 前端测试页启动失败：{e}（可手动 cd web && npm run dev）");
        }
    }
}

// ==================== 服务入口 ====================

/// 启动 HTTP 服务并永不返回。
///
/// `vq` = VQ-VAE 存档（`{out_dir}/vq.ckpt`，`None` 表示纯文本模型）：
/// 采样出完整图片段时用它解码成 PNG 内嵌回包。
pub fn run(cfg: ServeCfg, model: Transformer, tokenizer: Tokenizer, vq: Option<crate::vqvae::Vqvae>) -> ! {
    let addr = format!("{}:{}", cfg.host, cfg.port);
    let seed = cfg.seed;
    let model_name = cfg.model_name.clone();
    let block_size = cfg.block_size;
    // 先抄一份 key：下面要打印它，而 cfg 随后会被搬进 Arc<State>
    let api_key = cfg.api_key.clone();
    let cors_on = cfg.cors;

    let state = Arc::new(State {
        cfg,
        core: Mutex::new(Core { model, tokenizer, rng: Rng::new(seed) }),
        stats: Stats::default(),
        vq,
        created: now_unix(),
        started: Instant::now(),
    });

    let server = Server::http(&addr).unwrap_or_else(|e| panic!("监听 {addr} 失败：{e}"));

    logln!("[serve] 已监听 http://{addr}");
    logln!("[serve] 模型 {model_name}（上下文窗口 {block_size}）");
    // 图片输入通路状态：与 main 的 runlog 行同源，控制台一眼能看出能不能传图
    match &state.cfg.vision {
        Some(v) => logln!(
            "[serve] 视觉塔已启用（输入 {}px、patch {}px）：message.image_url 可用",
            v.image_size,
            v.patch_size
        ),
        None => logln!("[serve] 视觉塔未配置：不接受图片输入"),
    }
    // 接口规范静态版：与端点同一份数据，启动即写（删了也会自动重建）
    dump_openapi_files("openapi", &state.cfg);
    logln!(
        "[serve] 鉴权{}，CORS{}",
        // key 必须回显：不给 `--api-key` 时它是启动现场生成的，只有这一行能拿到
        match &api_key {
            Some(k) => format!("开，API key = {k}（请求带 `Authorization: Bearer <key>`）"),
            None => "关（任何人都能调）".to_string(),
        },
        if cors_on { "开" } else { "关" }
    );
    logln!("[serve] 端点：");
    logln!("[serve]   POST /v1/chat/completions  （stream=true 走 SSE）");
    logln!("[serve]   POST /v1/embeddings");
    logln!("[serve]   GET  /v1/models");
    logln!("[serve]   GET  /health");
    logln!("[serve]   GET  /v1/status");
    logln!("[serve]   GET  /openapi.json  /openapi.yaml  （接口规范，可导入 Postman/Apifox）");
    logln!("[serve] Ctrl-C 停止");

    // 每个请求一个线程；真正的生成在 `with_core` 里排队
    for req in server.incoming_requests() {
        let st = Arc::clone(&state);
        thread::spawn(move || handle(req, st));
    }
    unreachable!("tiny_http::Server::incoming_requests 是无限迭代器，不会返回")
}

// ==================== 测试 ====================

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(use_sft: bool) -> ServeCfg {
        // 与 cmd_serve 同源：SFT 模板下默认带模板停止标记（请求可用 stop 逐次覆盖）
        let mut sample = SampleOpts::default();
        if use_sft {
            sample.stop = prompt::SFT_STOP;
        }
        ServeCfg {
            host: "127.0.0.1".into(),
            port: 8080,
            api_key: None,
            cors: false,
            system: String::new(),
            use_sft,
            max_new: 200,
            seed: 42,
            kv: KvOpts::off(),
            model_name: "test-model".into(),
            sample,
            block_size: 512,
            vision: None,
        }
    }

    /// 带视觉塔的配置（测 image_url 通路；64px / patch 16 → 每图 16 个占位符）
    fn vision_cfg() -> ServeCfg {
        let mut c = cfg(true);
        c.vision = Some(crate::vision::VisionConfig::default());
        c
    }

    fn msgs(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs.iter().map(|(r, c)| (r.to_string(), c.to_string())).collect()
    }

    #[test]
    fn random_api_key_shape_and_uniqueness() {
        let k = random_api_key();
        assert!(k.starts_with("sk-"), "key 必须带 sk- 前缀：{k}");
        assert_eq!(k.len(), 3 + 32, "sk- + 32 位十六进制：{k}");
        assert!(k[3..].chars().all(|c| c.is_ascii_hexdigit()), "后缀必须是十六进制：{k}");
        // 连取两次不能撞车：撞车等于两台/两次服务共用一个 key
        assert_ne!(k, random_api_key());
    }

    #[test]
    fn split_sft_history_shape() {
        let m = msgs(&[
            ("system", "你是助手"),
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
        ]);
        let (system, history, input) = split_messages(&m, true);
        assert_eq!(system, "你是助手");
        assert_eq!(history, format!("{SFT_USER}\nq1\n{SFT_ASSISTANT}\na1"));
        assert_eq!(input, "q2");
    }

    #[test]
    fn split_raw_history_shape() {
        let m = msgs(&[("user", "q1"), ("assistant", "a1"), ("user", "q2")]);
        let (system, history, input) = split_messages(&m, false);
        assert_eq!(system, "");
        assert_eq!(history, "q1\na1");
        assert_eq!(input, "q2");
    }

    #[test]
    fn split_sft_two_rounds_join_with_newline() {
        let m = msgs(&[
            ("user", "q1"),
            ("assistant", "a1"),
            ("user", "q2"),
            ("assistant", "a2"),
            ("user", "q3"),
        ]);
        let (_, history, input) = split_messages(&m, true);
        assert_eq!(
            history,
            format!("{SFT_USER}\nq1\n{SFT_ASSISTANT}\na1\n{SFT_USER}\nq2\n{SFT_ASSISTANT}\na2")
        );
        assert_eq!(input, "q3");
    }

    #[test]
    fn split_merges_multiple_system_messages() {
        let m = msgs(&[
            ("system", "s1"),
            ("developer", "s2"),
            ("user", "q"),
        ]);
        let (system, history, input) = split_messages(&m, true);
        assert_eq!(system, "s1\ns2");
        assert_eq!(history, "");
        assert_eq!(input, "q");
    }

    #[test]
    fn merge_system_service_level_comes_first() {
        assert_eq!(merge_system("全局人设", "本轮指令"), "全局人设\n本轮指令");
        assert_eq!(merge_system("", "本轮指令"), "本轮指令");
        assert_eq!(merge_system("全局人设", ""), "全局人设");
        assert_eq!(merge_system("  ", "  "), "");
    }

    #[test]
    fn parse_chat_rejects_bad_requests() {
        let c = cfg(true);
        // 非 JSON
        assert!(parse_chat("不是 JSON", &c).is_err());
        // 缺 messages
        assert!(parse_chat("{}", &c).is_err());
        // 空 messages
        assert!(parse_chat(r#"{"messages":[]}"#, &c).is_err());
        // 末条不是 user
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":"q"},{"role":"assistant","content":"a"}]}"#, &c).is_err());
        // 未知 role
        assert!(parse_chat(r#"{"messages":[{"role":"tool","content":"x"},{"role":"user","content":"q"}]}"#, &c).is_err());
        // content 不是字符串
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":[{"type":"text","text":"q"}]}]}"#, &c).is_err());
        // n != 1
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":"q"}],"n":2}"#, &c).is_err());
        // max_tokens = 0
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":"q"}],"max_tokens":0}"#, &c).is_err());
        // 越界 temperature
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":"q"}],"temperature":9}"#, &c).is_err());
        // 越界 top_p
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":"q"}],"top_p":0}"#, &c).is_err());
        // 非法 stop 元素
        assert!(parse_chat(r#"{"messages":[{"role":"user","content":"q"}],"stop":[1]}"#, &c).is_err());
    }

    /// `message.image_url` 的解析规则：只认末条 user，null 当没带，非字符串/空串 400
    #[test]
    fn parse_chat_reads_image_url() {
        let c = vision_cfg();
        let job = parse_chat(
            r#"{"messages":[{"role":"user","content":"这是什么","image_url":"data:image/png;base64,AAAA"}]}"#,
            &c,
        )
        .unwrap();
        assert_eq!(job.image.as_deref(), Some("data:image/png;base64,AAAA"));

        // 不带字段 / 显式 null → 纯文本请求
        let job = parse_chat(r#"{"messages":[{"role":"user","content":"q"}]}"#, &c).unwrap();
        assert!(job.image.is_none());
        let job =
            parse_chat(r#"{"messages":[{"role":"user","content":"q","image_url":null}]}"#, &c)
                .unwrap();
        assert!(job.image.is_none());

        // 中间消息带图不算数（只认末条 user）
        let job = parse_chat(
            r#"{"messages":[{"role":"user","content":"q1","image_url":"data:image/png;base64,AAAA"},
                            {"role":"assistant","content":"a1"},
                            {"role":"user","content":"q2"}]}"#,
            &c,
        )
        .unwrap();
        assert!(job.image.is_none());

        // 非字符串 / 空串 → 400
        assert!(parse_chat(
            r#"{"messages":[{"role":"user","content":"q","image_url":123}]}"#,
            &c
        )
        .is_err());
        assert!(parse_chat(
            r#"{"messages":[{"role":"user","content":"q","image_url":""}]}"#,
            &c
        )
        .is_err());
    }

    /// 输入图解码：补训练格式、像素归一化、各类坏输入回 400 而不是 panic
    #[test]
    fn prepare_vision_decodes_and_formats_prompt() {
        // 没带图：原样透传（视觉塔缺席也不碍事）
        let (t, px) = prepare_vision(None, "你好", None).unwrap();
        assert_eq!(t, "你好");
        assert!(px.is_none());

        // 带图但模型没视觉塔 → 400
        assert!(prepare_vision(None, "q", Some("AAAA")).is_err());

        // 真图（64px PNG，与 VisionConfig::default 对齐）→ 3×64×64、值域 [-1,1]
        let vcfg = crate::vision::VisionConfig::default();
        let px_in: Vec<f32> = (0..3 * 64 * 64).map(|i| (i % 255) as f32 / 127.5 - 1.0).collect();
        let png = crate::vision::encode_png(&px_in, 64);
        let uri = format!("data:image/png;base64,{}", b64encode(&png));

        let (t, px) = prepare_vision(Some(&vcfg), "这是什么颜色？", Some(&uri)).unwrap();
        // 问题被补成训练时的理解样本格式 `图:<|image|>{问题}`
        assert_eq!(t, format!("图:{}这是什么颜色？", crate::tokenizer::IMAGE_LITERAL));
        let px = px.unwrap();
        assert_eq!(px.len(), 3 * 64 * 64);
        assert!(px.iter().all(|v| (-1.0..=1.0).contains(v)), "像素必须在 [-1,1]");

        // 已自带占位符：不重复补前缀
        let q = format!("看图{}说颜色", crate::tokenizer::IMAGE_LITERAL);
        let (t, _) = prepare_vision(Some(&vcfg), &q, Some(&uri)).unwrap();
        assert_eq!(t, q);

        // 裸 base64（无 data: 前缀）也认
        let (t, px) = prepare_vision(Some(&vcfg), "q", Some(&b64encode(&png))).unwrap();
        assert!(t.starts_with("图:"));
        assert!(px.is_some());

        // 坏 base64 / 合法 base64 但不是图 / 非 base64 的 data URI → 全部 400
        assert!(prepare_vision(Some(&vcfg), "q", Some("data:image/png;base64,!!!")).is_err());
        assert!(prepare_vision(Some(&vcfg), "q", Some("data:image/png;base64,AAAA")).is_err());
        assert!(prepare_vision(Some(&vcfg), "q", Some("data:image/png,abc")).is_err());
        assert!(prepare_vision(Some(&vcfg), "q", Some("data:image/png;base64")).is_err());
    }

    #[test]
    fn parse_chat_applies_defaults_and_overrides() {
        let c = cfg(true);
        let job = parse_chat(r#"{"messages":[{"role":"user","content":"你好"}]}"#, &c).unwrap();
        assert_eq!(job.input, "你好");
        assert_eq!(job.max_tokens, 200);
        assert!(!job.stream);
        assert!(job.seed.is_none());
        assert_eq!(job.sample.temperature, 0.8);
        // SFT 下自动带模板停止标记
        assert_eq!(job.sample.stop, prompt::SFT_STOP);

        let job = parse_chat(
            r#"{"messages":[{"role":"user","content":"你好"}],"max_tokens":64,
                "temperature":0.1,"top_p":0.5,"stream":true,"seed":7,
                "stop":"STOP","model":"ignored"}"#,
            &c,
        )
        .unwrap();
        assert_eq!(job.max_tokens, 64);
        assert_eq!(job.sample.temperature, 0.1);
        assert_eq!(job.sample.top_p, 0.5);
        assert!(job.stream);
        assert_eq!(job.seed, Some(7));
        assert_eq!(job.sample.stop, &["STOP"]);
        assert!(job.id.starts_with("chatcmpl-"));
    }

    #[test]
    fn parse_chat_counts_history_turns() {
        let c = cfg(true);
        // 首轮：没有历史
        let job = parse_chat(r#"{"messages":[{"role":"user","content":"q"}]}"#, &c).unwrap();
        assert_eq!(job.history_turns, 0);
        // system 不计入历史
        let job = parse_chat(
            r#"{"messages":[{"role":"system","content":"s"},{"role":"user","content":"q"}]}"#,
            &c,
        )
        .unwrap();
        assert_eq!(job.history_turns, 0);
        // q1 / a1 两条历史 + 本轮输入
        let job = parse_chat(
            r#"{"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"},
                {"role":"user","content":"q2"}]}"#,
            &c,
        )
        .unwrap();
        assert_eq!(job.history_turns, 2);
        // SFT 模板下历史带模板标记（cfg(true)），但两条内容都得在里面
        assert!(job.history.contains("q1") && job.history.contains("a1"));
        // 非 SFT 模板下就是裸拼接
        let c0 = cfg(false);
        let job = parse_chat(
            r#"{"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"},
                {"role":"user","content":"q2"}]}"#,
            &c0,
        )
        .unwrap();
        assert_eq!(job.history, "q1\na1");
    }

    /// 问答记录的取值辅助：按键名取值，取不到直接 panic
    fn field(items: &[(&'static str, String)], k: &str) -> String {
        items
            .iter()
            .find(|(name, _)| *name == k)
            .unwrap_or_else(|| panic!("问答记录里缺了「{k}」字段"))
            .1
            .clone()
    }

    #[test]
    fn qa_log_block_records_round_and_params() {
        let c = cfg(true);
        let job = parse_chat(
            r#"{"messages":[{"role":"user","content":"q1"},{"role":"assistant","content":"a1"},
                {"role":"user","content":"q2"}],"stream":true,"seed":7}"#,
            &c,
        )
        .unwrap();
        let (title, items) = qa_log_block(
            &job,
            &QaOutcome {
                system: "你是助手",
                output: "答案",
                reason: Some(StopReason::Eos),
                n_prompt: 10,
                n_gen: 4,
                rate: Some(42.5),
                note: None,
            },
        );
        // 标题带 id 和模式，翻日志时一眼能对上接口行
        assert_eq!(title, format!("问答 {}（流式）", job.id));
        assert_eq!(field(&items, "system"), "你是助手");
        assert_eq!(field(&items, "历史"), "2 条");
        assert_eq!(field(&items, "输入"), "q2");
        assert_eq!(field(&items, "输出"), "答案");
        assert_eq!(field(&items, "收尾"), "采到 EOS");
        let params = field(&items, "采样参数");
        for want in ["temperature=0.8", "seed=7", "max_tokens=200", "repetition_penalty="] {
            assert!(params.contains(want), "采样参数缺 {want}：{params}");
        }
        assert_eq!(field(&items, "用量"), "prompt 10 + completion 4 = 14 token，42.5 tok/s");
        // 正常收尾不该带备注
        assert!(items.iter().all(|(k, _)| *k != "备注"));
    }

    #[test]
    fn qa_log_block_handles_empty_output_and_abort() {
        let c = cfg(false);
        let job = parse_chat(r#"{"messages":[{"role":"user","content":"q"}]}"#, &c).unwrap();
        let (title, items) = qa_log_block(
            &job,
            &QaOutcome {
                system: "",
                output: "",
                reason: None,
                n_prompt: 3,
                n_gen: 0,
                rate: None,
                note: Some("客户端提前断开，只记到已生成的部分"),
            },
        );
        assert_eq!(title, format!("问答 {}（非流式）", job.id));
        assert_eq!(field(&items, "system"), "（无）");
        assert_eq!(field(&items, "历史"), "0 条（首轮问答）");
        assert_eq!(field(&items, "输出"), "（空）");
        // 没跑到正常收尾时不许把「跑满 max-new」之类的假原因写进去
        assert_eq!(field(&items, "收尾"), "未正常收尾（见备注）");
        assert_eq!(field(&items, "备注"), "客户端提前断开，只记到已生成的部分");
        assert!(field(&items, "用量").contains("样本不足"));
        assert!(field(&items, "采样参数").contains("seed=未指定（用服务级）"));
    }

    #[test]
    fn stop_interner_dedups_identical_slices() {
        let a = vec!["结束".to_string()];
        let b = vec!["结束".to_string()];
        let pa = intern_stops(&a).unwrap();
        let pb = intern_stops(&b).unwrap();
        assert!(std::ptr::eq(pa, pb), "同一组 stop 必须复用同一份静态切片");

        let c = vec!["结束".to_string(), "STOP".to_string()];
        let pc = intern_stops(&c).unwrap();
        assert!(!std::ptr::eq(pa, pc));
        assert_eq!(pc, &["结束", "STOP"]);

        assert!(intern_stops(&[]).unwrap().is_empty());
    }

    #[test]
    fn stop_interner_rejects_oversized_input() {
        assert!(intern_stops(&[String::new()]).is_err());
        assert!(intern_stops(&[ "x".repeat(MAX_STOP_LEN + 1) ]).is_err());
        let many: Vec<String> = (0..MAX_STOPS + 1).map(|i| format!("s{i}")).collect();
        assert!(intern_stops(&many).is_err());
    }

    #[test]
    fn sse_pipe_reads_frames_then_hits_eof() {
        let (tx, rx) = sync_channel::<Vec<u8>>(4);
        let mut pipe = SsePipe { rx, buf: Vec::new(), pos: 0 };

        tx.send(b"hello".to_vec()).unwrap();
        tx.send(b"world".to_vec()).unwrap();
        drop(tx); // 发送端关闭 → 读完缓冲后返回 0

        let mut out = Vec::new();
        let mut chunk = [0u8; 3];
        loop {
            let n = pipe.read(&mut chunk).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&chunk[..n]);
        }
        assert_eq!(out, b"helloworld");
    }

    #[test]
    fn finish_reason_mapping() {
        assert_eq!(finish_of(StopReason::Eos), "stop");
        assert_eq!(finish_of(StopReason::StopMark("用户：")), "stop");
        assert_eq!(finish_of(StopReason::MaxNew), "length");
    }

    #[test]
    fn mean_pool_averages_rows() {
        // 2 行 × 3 列 → 每列取平均
        let t = crate::tensor::Tensor::from_vec(vec![1.0, 2.0, 3.0, 3.0, 6.0, 9.0], vec![2, 3]);
        assert_eq!(mean_pool(&t), vec![2.0, 4.0, 6.0]);
    }

    /// 收集文档里所有 `$ref`（形如 `#/components/schemas/X`）
    fn collect_refs(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(map) => {
                for (k, x) in map {
                    if k == "$ref" {
                        if let Some(s) = x.as_str() {
                            out.push(s.to_string());
                        }
                    } else {
                        collect_refs(x, out);
                    }
                }
            }
            Value::Array(items) => items.iter().for_each(|x| collect_refs(x, out)),
            _ => {}
        }
    }

    #[test]
    fn openapi_spec_covers_every_route() {
        let spec = openapi_spec(&cfg(true));
        assert_eq!(spec["openapi"], "3.1.0");
        let paths = spec["paths"].as_object().expect("paths 必须是对象");
        for p in [
            "/health",
            "/v1/status",
            "/v1/models",
            "/openapi.json",
            "/openapi.yaml",
            "/v1/chat/completions",
            "/v1/embeddings",
        ] {
            assert!(paths.contains_key(p), "规范里缺了 {p}");
        }
        // 全局默认要鉴权；不鉴权的端点必须显式清空，否则导入工具会强制要求带 key
        for p in ["/health", "/v1/status", "/openapi.json", "/openapi.yaml"] {
            assert_eq!(spec["paths"][p]["get"]["security"], json!([]), "{p} 应不鉴权");
        }
        // 鉴权端点不写 security → 沿用全局 bearerAuth
        assert!(spec["paths"]["/v1/chat/completions"]["post"]["security"].is_null());
        assert_eq!(spec["components"]["securitySchemes"]["bearerAuth"]["scheme"], "bearer");
    }

    #[test]
    fn openapi_spec_constraints_match_parse_chat() {
        let spec = openapi_spec(&cfg(true));
        let req = &spec["components"]["schemas"]["ChatCompletionRequest"];
        assert_eq!(req["required"][0], "messages");
        let props = &req["properties"];
        // 这些上限必须和 parse_chat 的逐字段校验一字不差，否则文档会骗人
        assert_eq!(props["messages"]["maxItems"].as_u64(), Some(MAX_MESSAGES as u64));
        assert_eq!(props["stop"]["oneOf"][1]["maxItems"].as_u64(), Some(MAX_STOPS as u64));
        assert_eq!(props["stop"]["oneOf"][1]["items"]["maxLength"].as_u64(), Some(MAX_STOP_LEN as u64));
        assert_eq!(props["temperature"]["maximum"].as_f64(), Some(5.0));
        assert_eq!(props["top_k"]["maximum"].as_u64(), Some(10_000));
        assert_eq!(props["repetition_penalty"]["minimum"].as_f64(), Some(0.1));
        assert_eq!(props["repetition_window"]["maximum"].as_u64(), Some(100_000));
        assert_eq!(props["n"]["const"], json!(1));
        let emb = &spec["components"]["schemas"]["EmbeddingRequest"]["properties"]["input"];
        assert_eq!(emb["oneOf"][2]["maxItems"].as_u64(), Some(MAX_EMBED_BATCH as u64));
    }

    #[test]
    fn openapi_spec_every_ref_resolves() {
        let spec = openapi_spec(&cfg(true));
        let mut refs = Vec::new();
        collect_refs(&spec, &mut refs);
        assert!(!refs.is_empty(), "应该至少有一个 $ref");
        for r in &refs {
            let mut cur = &spec;
            for seg in r.trim_start_matches("#/").split('/') {
                cur = cur
                    .get(seg)
                    .unwrap_or_else(|| panic!("$ref {r} 指向不存在的节点 `{seg}`"));
            }
        }
    }

    #[test]
    fn openapi_examples_pass_real_parsers() {
        // 导入 Postman/Apifox 后，工具直接把 example 当请求体发送；
        // 它必须能过真实的入参校验，否则一点 Send 就是 400（编出 n≠1、
        // 或 messages 以 assistant 结尾这类“看着合法”的坏例子）
        let spec = openapi_spec(&cfg(false));
        let chat_ex = &spec["paths"]["/v1/chat/completions"]["post"]["requestBody"]
            ["content"]["application/json"]["example"];
        assert!(!chat_ex.is_null(), "chat 缺 example，导入工具只能自己瞎编");
        let job = parse_chat(&chat_ex.to_string(), &cfg(false))
            .unwrap_or_else(|e| panic!("chat example 过不了 parse_chat：{}", e.message));
        assert_eq!(job.input, "用一句话介绍 Transformer。");

        let emb_ex = &spec["paths"]["/v1/embeddings"]["post"]["requestBody"]
            ["content"]["application/json"]["example"];
        assert!(!emb_ex.is_null(), "embeddings 缺 example");
        // input 只接受 string / number / string[]（run_embeddings 逐项解析）
        let input = &emb_ex["input"];
        assert!(
            input.is_string()
                || input.is_number()
                || (input
                    .as_array()
                    .map(|a| a.iter().all(|v| v.is_string()))
                    .unwrap_or(false)),
            "embeddings example 的 input 类型不合法：{input}"
        );
    }

    #[test]
    fn dump_openapi_files_writes_both_formats() {
        // 写进临时目录，避免污染仓库；内容必须与端点返回的字节逐字一致
        let dir = std::env::temp_dir().join(format!("llm_oss_openapi_{}", std::process::id()));
        let dir = dir.to_string_lossy().into_owned();
        dump_openapi_files(&dir, &cfg(false));

        let spec = openapi_spec(&cfg(false));
        let json_txt = std::fs::read_to_string(format!("{dir}/openapi.json")).unwrap();
        let yaml_txt = std::fs::read_to_string(format!("{dir}/openapi.yaml")).unwrap();
        assert_eq!(json_txt, serde_json::to_string_pretty(&spec).unwrap());
        assert_eq!(yaml_txt, json_to_yaml(&spec));

        // 落盘的 JSON 能解析回来，且仍是覆盖全部路由的完整规范
        let parsed: Value = serde_json::from_str(&json_txt).unwrap();
        assert_eq!(parsed["paths"].as_object().unwrap().len(), 7);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn json_to_yaml_renders_nested_structures() {
        let v = json!({
            "n": 3,
            "pi": 0.25,
            "ok": true,
            "none": null,
            "s": "带冒号: 的值",
            "list": [1, "two", {"k": "v"}, [3, 4]],
            "empty_arr": [],
            "empty_obj": {},
            "nl": "a\nb",
            "ver": "1.1.1"
        });
        // serde_json 的 Map 是 BTreeMap，键按字典序输出
        let want = concat!(
            "empty_arr: []\n",
            "empty_obj: {}\n",
            "list:\n",
            "  - 1\n",
            "  - two\n",
            "  - k: v\n",
            "  - - 3\n",
            "    - 4\n",
            "n: 3\n",
            "nl: \"a\\nb\"\n",
            "none: null\n",
            "ok: true\n",
            "pi: 0.25\n",
            "s: \"带冒号: 的值\"\n",
            "ver: \"1.1.1\"\n",
        );
        assert_eq!(json_to_yaml(&v), want);
    }

    #[test]
    fn yaml_quoting_never_leaks_ambiguous_scalars() {
        // 会被 YAML 解析成 bool / null 的写法一律上引号
        for s in ["true", "NULL", "Yes", "off"] {
            assert_eq!(yaml_str(s), format!("\"{s}\""), "{s} 必须引号");
        }
        // 看着像数字但不是数的（版本号之类）也上引号
        for s in ["1.1.1", "0x1F", "2026-09-27"] {
            assert_eq!(yaml_str(s), format!("\"{s}\""), "{s} 必须引号");
        }
        // 普通文本保持裸写，便于阅读
        assert_eq!(yaml_str("hello world"), "hello world");
        assert_eq!(yaml_str(""), "\"\"");
        // 含 # 与 : 的必须引号，否则会被当成注释/键值分隔
        assert_eq!(yaml_str("a # b"), "\"a # b\"");
        // 路径既没有 : 也没有 #，裸写即可
        assert_eq!(yaml_str("/v1/models"), "/v1/models");
    }

    /// RFC 4648 标准向量：三组经典 + 空输入 + 补位边界
    #[test]
    fn b64encode_matches_rfc4648_vectors() {
        assert_eq!(b64encode(b""), "");
        assert_eq!(b64encode(b"M"), "TQ==");
        assert_eq!(b64encode(b"Ma"), "TWE=");
        assert_eq!(b64encode(b"Man"), "TWFu");
        assert_eq!(b64encode(b"hello"), "aGVsbG8=");
        // 高位字节（UTF-8 中文）也走同一张表
        assert_eq!(b64encode("中".as_bytes()), "5Lit");
    }

    /// b64decode 是 b64encode 的逆：RFC 向量 + 任意字节往返 + MIME 换行 + 坏输入
    #[test]
    fn b64decode_inverts_b64encode() {
        assert_eq!(b64decode("").unwrap(), Vec::<u8>::new());
        assert_eq!(b64decode("TQ==").unwrap(), b"M");
        assert_eq!(b64decode("TWE=").unwrap(), b"Ma");
        assert_eq!(b64decode("TWFu").unwrap(), b"Man");
        assert_eq!(b64decode("aGVsbG8=").unwrap(), b"hello");
        // 256 种字节全量往返
        let all: Vec<u8> = (0..=255u8).collect();
        assert_eq!(b64decode(&b64encode(&all)).unwrap(), all);
        // 空白（MIME 风格换行）忽略
        assert_eq!(b64decode("TW\r\nFu").unwrap(), b"Man");
        // 非法字符、填充后再有数据
        assert!(b64decode("!!").is_err());
        assert!(b64decode("TQ==Q").is_err());
    }

    /// 生成图 data URI 端到端：tiny VQ-VAE 解码码本段 → PNG → base64；
    /// 未配 VQ-VAE 或没采出图时必须返回 None（纯文本回包不带 image_url）
    #[test]
    fn generated_image_uri_decodes_full_segment() {
        let vqcfg = crate::vqvae::VqConfig {
            image_size: 8,
            patch_size: 4,
            latent_dim: 4,
            codebook_size: 8,
            hidden: 16,
            beta: 0.25,
        };
        let vq = crate::vqvae::Vqvae::new(vqcfg, &mut Rng::new(11));
        // P = (8/4)^2 = 4，正好一个完整图片段
        let imgs = vec![vec![0usize, 1, 2, 3]];
        let uri = generated_image_uri(Some(&vq), &imgs).expect("完整码本段应产出 data URI");
        assert!(
            uri.starts_with("data:image/png;base64,"),
            "URI 前缀不对：{uri}"
        );
        // base64 部分能解回非空 PNG（PNG 魔数 0x89 'P' 'N' 'G'）
        let payload = &uri["data:image/png;base64,".len()..];
        assert!(!payload.is_empty(), "base64 载荷不应为空");

        // 没采出图 → None（不管配没配 VQ-VAE）
        assert_eq!(generated_image_uri(Some(&vq), &[]), None);
        // 没配 VQ-VAE → None（纯文本模型）
        assert_eq!(generated_image_uri(None, &imgs), None);
    }
}
