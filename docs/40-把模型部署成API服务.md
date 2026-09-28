# 第 40 课：把模型部署成 API 服务 —— 从"自己跑"到"给别人调"

> 代码位置：`src/serve.rs`（约 1300 行）、`src/cli.rs`（`serve` 子命令）、`src/main.rs`（`cmd_serve`）、`src/sample.rs`（Generator 状态机）
>
> 本课无新算法，讲的是**部署**：如何把前 39 课训练出来的模型包装成一个
> 和主流大模型厂商（OpenAI / DeepSeek / 智谱…）接口形状一致的 HTTP API，
> 让任何支持 OpenAI 协议的客户端改一行 `base_url` 就能用上你的模型。
> CLI 用法详见 [README.md](../README.md#5-serve--部署成-openai-兼容的-api-服务) 的 `serve` 章节。

---

## 1. 本课要搞懂的问题

1. 主流大模型 API 长什么样？为什么"兼容 OpenAI"是事实标准？
2. 一个常驻服务进程和一次性命令行程序，架构上差在哪？
3. 流式输出（SSE）为什么必须用**状态机**而不是"跑完再返回"？
4. HTTP 层的 chunked transfer 和 SSE 是什么关系？
5. 客户端断开连接时，服务端怎么发现自己在对着空气烧 CPU？
6. 鉴权、CORS、请求日志、队列状态这些"API 周边"各自解决什么问题？
7. 单机 CPU 推理该用什么并发模型——每请求一线程，还是排队？

---

## 2. 为什么是"兼容 OpenAI"

把模型做成 API 有两条路：

| 路线 | 做法 | 问题 |
|------|------|------|
| 自定义协议 | 自己设计 JSON 字段、自己写客户端 | 每个调用方都要写适配层，没人愿意接 |
| **兼容 OpenAI** | 按 `POST /v1/chat/completions` 的形状实现 | 生态里现成的 SDK、网关、评测工具直接可用 |

OpenAI 的 Chat Completions 协议已经事实上标准化：LangChain、LobeChat、各种
RAG 框架、甚至各家云厂商的网关，都默认会往 `{base_url}/v1/chat/completions`
发 `{"model": …, "messages": [...], "stream": true}`。本项目照这套协议实现，
于是：

- 改一行 `base_url` + `api_key`，OpenAI 官方 SDK 就能调本项目；
- SSE 帧的 `object: "chat.completion.chunk"`、`data: [DONE]` 结尾，全部照抄；
- 错误体统一是 `{"error": {"message": …, "type": …, "code": …}}`。

**代价**：OpenAI 的字段比本项目实际需要的多（`logprobs`、`tools`、`n`、
`function_call`…）。策略是**不认识的字段一律忽略**（否则官方 SDK 发来的请求
全被 400 打回），但**认识的字段严格校验并给出"哪个字段错在哪"的中文错误**
（否则调用方只能看到 400 干瞪眼）。唯一显式拒绝的是 `n != 1`——本服务一次
只生成一个回答，硬撑 `n=4` 只会返回四个重复答案，不如早说。

---

## 3. 架构：常驻进程 vs 一次性命令

`chat` 子命令的生命周期是「加载 → REPL 循环 → 退出」；`serve` 是
「加载 → **无限** accept 循环 → 直到 Ctrl-C」。改造点集中在三处：

### 3.1 模块划分

```
src/serve.rs
├── ServeCfg          启动时定死的配置（host/port/api_key/system/采样默认值…）
├── State = ServeCfg + Core + Stats + 时间戳
│   ├── Core { model, tokenizer, rng }   ← 独占，包在 Mutex 里
│   └── Stats { queued, running, chat, stream, embeddings, errors }  ← 全是原子量
├── route()           路由：只看路径，不认识就 404/405
├── parse_chat()      请求体 → ChatJob（严格校验）
├── run_chat()        非流式：with_core 里跑完整生成
├── stream_reply()    SSE：生成线程 + SsePipe + tiny_http chunked
├── run_embeddings()  文本向量化（最后一层 hidden 均值池化）
└── run()             accept 循环，每请求 spawn 一个线程
```

### 3.2 加载链路复用 `cmd_chat`

`cmd_serve`（[main.rs](../src/main.rs)）的前半段和 `cmd_chat` 一模一样：
配置 → 分词器 → checkpoint → LoRA 合并 → RoPE 覆盖。区别只在最后一步：不进
REPL，而是把模型和默认采样参数打包成 `ServeCfg` 交给 `serve::run`。这样
**chat 和 serve 永远不会跑出两套不同的加载逻辑**（RoPE 覆盖顺序、LoRA 合并
时机这类坑只用踩一次）。

模型名取 checkpoint 文件名的 stem（`checkpoints/zh-sft-v2/final.ckpt` → `final`）：
OpenAI 客户端会拿请求里的 `model` 和 `/v1/models` 的 id 对账，多个 checkpoint
服务并存时不会撞名。

### 3.3 并发模型：每请求一线程 + 单份模型排队

```rust
for req in server.incoming_requests() {
    thread::spawn(move || handle(req, st));   // 每请求一线程
}
```

HTTP 层是并发的（读请求体、鉴权、拼 JSON 都不碰模型），但**生成是串行的**：
`Core` 只有一份，所有生成请求共用一把 `Mutex`（`with_core`）。

为什么不多副本？单机 CPU 推理时，两个生成请求并行只会互相抢核心、互相污染
cache，总吞吐不变而单条延迟翻倍；排队反而让延迟曲线平稳。要真并发：另起进程
（多开几个端口）或上 GPU。`/v1/status` 里的 `queued` 就是这条队列的长度——
**这是排队可见性的价值**：调用方能看到"我在等前面 3 个人"，而不是干等一个
不返回的连接。

`with_core` 还做了两件防御性的事：

- `catch_unwind`：模型推理里万一 panic（比如词表越界），只把这个请求变成
  500，服务进程不死；
- `Mutex` 的 poison 处理：一个请求 panic 后，后续请求用 `into_inner` 继续，
  不会因为"锁被下毒"而整个服务永久瘫痪。

---

## 4. 端点一览

| 方法 | 路径 | 鉴权 | 说明 |
|------|------|------|------|
| POST | `/v1/chat/completions` | ✅ | 对话补全，`stream=true` 走 SSE |
| POST | `/v1/embeddings` | ✅ | 文本向量化（均值池化） |
| GET | `/v1/models` | ✅ | 只有一个模型的列表（OpenAI 语义：它会暴露服务上有什么） |
| GET | `/health` | ❌ | 存活探针 `{"status":"ok","model":…}` |
| GET | `/v1/status` | ❌ | 队列/计数/模型信息，诊断用 |
| GET | `/openapi.json` | ❌ | 本服务的 OpenAPI 3.1 规范（JSON），导入 Postman/Apifox/Swagger UI |
| GET | `/openapi.yaml` | ❌ | 同上，YAML 格式（两份与实际路由同源生成） |
| OPTIONS | 任意 | ❌ | CORS 预检 → 204 |

**鉴权刻意豁免 `/health`、`/v1/status` 和 `/openapi.*`**：探针和监控系统通常
不方便带 `Authorization` 头，且这些端点不泄露模型输出——一个只说"我还活着"，
一个只说计数器，规范则只描述接口形状。`/v1/models` 则必须鉴权（它暴露服务上
部署了什么）。

### 4.1 `POST /v1/chat/completions` 请求字段

| 字段 | 类型 | 缺省（服务级） | 校验 |
|------|------|------|------|
| `messages` | 数组 | 必填 | 非空、≤200 条、末条必须是 `user`、role 只认 system/developer/user/assistant |
| `model` | string | 忽略 | 不校验（单模型服务） |
| `temperature` | 数字 | `--temperature`（0.8） | `[0, 5]` |
| `top_k` | 整数 | `--top-k`（40） | ≤10000（0 = 不限制） |
| `top_p` | 数字 | `--top-p`（0.9） | `(0, 1]` |
| `repetition_penalty` | 数字 | `--repetition-penalty`（1.1） | `[0.1, 10]` |
| `repetition_window` | 整数 | `--repetition-window`（64） | ≤100000 |
| `max_tokens` | 整数 | `--max-new`（200） | >0，且钳到上下文窗口内 |
| `stop` | 字符串/数组 | SFT 模板停止标记 | ≤4 个，每个 ≤64 字节 |
| `seed` | 整数 | 服务级 `--seed` | 非负；给了就用独立 RNG，可复现 |
| `stream` | bool | false | — |
| `n` | 整数 | 1 | 只接受 1 |
| 其余 | — | — | **一律忽略**（`user`/`logit_bias`/`tools`…） |

**system 的合并规则**：服务级 `--system` 在前，请求里所有 `system`/`developer`
消息用换行接在它后面（`merge_system`）。这样运维能设一个"服务级人格"，调用方
仍能按 OpenAI 习惯在 messages 里再加一层。

**历史裁剪**：`block_size` 是「system + 历史 + 本轮生成」共用的窗口，
`prompt_budget = block_size - max_tokens`，超出的历史从最老的轮次丢起
（与 `chat` 完全同源，见 `prompt::assemble`）。裁剪发生时服务端日志会打印
`历史超出输入预算：312 → 180 token`。

### 4.2 非流式响应

```json
{
  "id": "chatcmpl-1",
  "object": "chat.completion",
  "created": 1769…,
  "model": "final",
  "choices": [{
    "index": 0,
    "message": { "role": "assistant", "content": "……" },
    "finish_reason": "stop",
    "logprobs": null
  }],
  "usage": { "prompt_tokens": 12, "completion_tokens": 33, "total_tokens": 45 }
}
```

`finish_reason` 映射：采到 EOS / 命中停止标记 → `"stop"`；跑满 `max_tokens` →
`"length"`。和 `chat` 一样，MaxNew 截断说明回答没说完——客户端看到 `length`
就该调大 `max_tokens` 或收窄输入。

---

## 5. 流式输出：Generator 状态机 + SSE

### 5.1 为什么要状态机

原来的生成循环长这样：`for` 采样到 EOS，把整段文本 return。这种**一把梭的
函数**没法流式——HTTP 层要的是一边生成一边往外写，也就是"生成器跑一步、吐一
个 token、把控制权交回 HTTP 层"。

改造做法（[sample.rs](../src/sample.rs)）：把循环拆成 `Generator` 的四个方法——

| 方法 | 作用 |
|------|------|
| `Generator::new(...)` | 装 prompt、预填首 token，不推进 |
| `step() -> bool` | 推进一个 token；返回 false = 生成结束 |
| `next_text() -> Option<&str>` | 取出**这次**产出的增量文本（可能为空） |
| `flush() -> String` | 生成结束后取回被 hold-back 的尾巴 |
| `run()` | `while step() {}` ——非流式路径的封装 |

关键细节：**安全流式（safe_len hold-back）**。分词器里存在多字节字符和
`stop` 标记的前缀——如果一个 token 单独看是半个汉字，直接发给客户端就会出现
乱码；同样，`"用户："` 只出了 `"用户"` 就发出去，等下一帧发现命中停止标记时，
客户端已经渲染了不该出现的半截。所以 `step()` 后只吐出"确定不会再变"的部分，
剩下的攒到 `flush()` 一次发干净。**流式和非流式跑的是同一个 `Generator`，
因此两者输出必然逐字一致**——这是把循环抽成状态机换来的不变量。

`generate_with_reason()` 仍保留，内部就是 `Generator::run()` 的封装，所以
`generate`/`chat` 等老命令一行没改。

### 5.2 SSE 与 HTTP chunked 的关系

```
HTTP 层：Response::new(200, headers, SsePipe, None /* 无长度 */, None)
                                    ↑              ↑
                            自定义 Read 实现    None = 不发 Content-Length，
                                               HTTP/1.1 自动用 chunked
```

- **chunked** 是传输层机制：body 分块发，每块前面写 `十六进制长度\r\n`，
  结尾发 `0\r\n\r\n`。它让"不知道总长度的流"能在 HTTP/1.1 里合法传输。
- **SSE** 是应用层协议：`Content-Type: text/event-stream`，每条消息形如
  `data: {json}\n\n`，永不关闭连接（`Cache-Control: no-cache` 防代理缓存）。
- `SsePipe` 是两者之间的桥：它实现 `Read`，从 `mpsc` channel 里取帧往外读；
  读到 `Ok(0)` 就是 EOF，tiny_http 据此写终止块、正常收尾。

### 5.3 帧序列（实测输出）

```
data: {"id":"chatcmpl-1","object":"chat.completion.chunk",…,"delta":{"role":"assistant","content":""}}
data: {"…","delta":{"content":"好的"}}
data: {"…","delta":{"content":"，"}}
…
data: {"…","delta":{},"finish_reason":"stop"}     ← 收尾帧
data: {"…","choices":[],"usage":{…}}              ← usage 单独一帧（OpenAI 实际做法）
data: [DONE]
```

第一帧先声明角色（OpenAI 客户端靠它拿到 assistant 角色）；usage 单独发一帧
且 `choices` 为空数组——`stream_options: {"include_usage": true}` 之外也发，
因为多数轻量客户端不发这个字段却仍然想要 usage。

### 5.4 背压：有界 channel

```rust
let (tx, rx) = sync_channel::<Vec<u8>>(16);
```

生成线程往 channel 塞帧，HTTP 线程从 `SsePipe` 读帧往 socket 写。容量 16 意味着
**生成最多领先客户端 16 帧**：客户端读得慢（慢速网络、逐字渲染的前端）时，
`tx.send` 会阻塞，生成自然慢下来，而不是把内存堆爆。反过来客户端读得快时，
生成速度不受影响。

---

## 6. 断开可中断：别对着空气烧 CPU

**问题**：用户点了"停止生成"或直接关掉网页，TCP 连接断了，但服务端的
`while g.step()` 还在吭哧吭哧采样——CPU 白烧，还占着模型锁让后面的人排队。

**检测链路**（三段接力）：

```
客户端断开
  → tiny_http 写 socket 失败（对端 RST）
  → req.respond() 返回 Err，SsePipe 被 drop
  → rx 关闭
  → 生成线程下一次 tx.send() 返回 Err
  → 提前 return StreamOutcome { aborted: true }
  → 日志：[serve] 客户端提前断开，生成已中止（51 token）
```

要点是**每个环节都必须真的把 drop 传导下去**：

1. `stream_reply` 里 `let _ = req.respond(...)`——返回 Err 也不能吞掉后续，
   因为 SsePipe 是 `respond` 的实参，`respond` 返回时它必然已被 drop；
2. 生成线程**每帧都检查** `tx.send` 的返回值（而不是 `let _ =` 忽略）；
3. `aborted` 通过 `StreamOutcome` 带回外层，外层据此**不再发**
   finish/usage/DONE 帧（对一个已断开的连接发这些毫无意义，还可能触发二次错误）。

实测：`curl --max-time 0.15` 中途掐断，服务端 51 token 处停止并打印中止日志；
而一个 302ms 内自己采到 EOS 的请求即使 curl 超时也走正常收尾——**两条路径
互不干扰**（生成先结束就走正常分支，连接先断才走中止分支）。

---

## 7. API 周边能力

### 7.1 鉴权（`--api-key`）

```rust
// Authorization: Bearer sk-test   → 剥掉 "bearer " 前缀（大小写不敏感）比较
if token == expected { Ok(()) } else { 401 authentication_error }
```

- 不给 `--api-key` = 不鉴权（本机自用），启动日志明确警告
  `鉴权关（任何人都能调）`——**默认不鉴权，但必须喊出来**；
- 比较前 `trim()`，容忍头值两端的空白；
- 错误体是 OpenAI 形状：`{"error":{"message":"缺少或错误的 API key…","type":"authentication_error"}}`，
  官方 SDK 能直接识别成 `AuthenticationError` 并触发它的重试/报错逻辑。

### 7.2 CORS（`--cors`）

浏览器页面（比如本地写的 prompt 调试页）直接 fetch 一个不同源的 API，会被
同源策略拦下；浏览器会先发一个 `OPTIONS` 预检，问"允许我带 `Authorization`
吗"。开 `--cors` 后：

```
OPTIONS → 204 + Access-Control-Allow-Origin: *
          + Allow-Methods: GET, POST, OPTIONS
          + Allow-Headers: Content-Type, Authorization
          + Max-Age: 600（预检结果缓存 10 分钟）
```

关掉时一个头都不加——`*` 意味着任何网页都能拿你的 key 调你的模型，**生产
环境不该开**。

### 7.3 请求日志

每个请求一行（进 `logs/` 与 runlog）：

```
[serve] GET /v1/models → 401（1ms）
[serve] POST /v1/chat/completions → 200（135ms）
[serve] 流式完成 chatcmpl-1：11 token（采到 EOS）
[serve] 客户端提前断开，生成已中止（51 token）
[serve] 历史超出输入预算：312 → 180 token（预算 312）
```

方法、路径、状态码、耗时四要素齐全；生成类事件单独一行（带 token 数和收尾
原因），因为"哪个请求慢"和"慢在生成还是慢在排队"是两个问题。

### 7.4 `/v1/status` 队列状态

```json
{
  "status": "ok",
  "uptime_seconds": 123,
  "queue": { "queued": 2, "running": 1 },
  "requests": { "chat": 10, "stream": 4, "embeddings": 2, "errors": 1 },
  "model": { "name": "final", "block_size": 512, "max_new": 64, "prompt_format": "sft" }
}
```

`queued`/`running` 来自 `with_core` 里的 `RunGuard`（RAII：进锁前 `queued++`，
拿到锁后 `queued--`/`running++`，离开作用域 `running--`——**连 panic 都不会
漏计数**，因为 Drop 一定执行）。全部计数器是原子量，读它们不需要碰模型锁，
所以状态端点在生成高峰期也瞬间返回。

### 7.5 `/v1/embeddings`

```json
POST { "input": "你好世界" }
→ { "object": "list",
    "data": [{ "object": "embedding", "index": 0, "embedding": [0.012, -0.34, …] }],
    "model": "final",
    "usage": { "prompt_tokens": 9, "total_tokens": 9 } }
```

实现：`model.forward_hidden()` 取最后一层 hidden states，按行**均值池化**成
一个向量（`no_grad` 包裹，不建计算图）。`input` 收字符串、token id 或它们的
数组（≤32 条）。维度 = `n_embd`，**不同 checkpoint 维度不同**，RAG 场景混用
前要对齐。越界 token id 先挡成 400 而不是让它 panic。

这是个**教学版 embedding**：没有专门的对比学习训练，向量质量取决于底座预训练
效果；RAG 实验（第 37 课）里的向量化对照有更细的讨论。

### 7.6 请求体与资源上限

| 限制 | 值 | 防什么 |
|------|-----|--------|
| 请求体 | 2 MB | 拿超大 body 撑爆内存 |
| messages | 200 条 | 一万个 message |
| stop | 4 个 × 64 字节 | 拿 stop 字符串当缓冲区 |
| stop 驻留表 | 256 种组合 / 1024 个字符串 | `&'static` 需求导致的无界泄漏（见下） |
| embeddings 批量 | 32 条 | 一批塞爆 |

**`stop` 为什么需要"驻留表"**：`SampleOpts.stop` 的类型是
`&'static [&'static str]`（生成器热路径上要零成本反复比对）。请求里的 stop
是运行时字符串，只能 `Box::leak` 换 `'static`——但每个请求都泄漏，服务跑一周
就是内存缓慢上涨。解法是两层 interner：字符串只在**首次出现**时泄漏一次
（1024 个封顶，超出报 400），相同组合复用同一份切片（256 份封顶）。**有界
泄漏**是这里唯一诚实的方案：Rust 的类型系统要求 `'static`，而进程内没有别
的生命周期能给它。

### 7.7 `/openapi.json` 接口规范

```bash
curl -s http://127.0.0.1:8080/openapi.json   # OpenAPI 3.1，JSON
curl -s http://127.0.0.1:8080/openapi.yaml   # 同一份，YAML
```

规范**由 `openapi_spec(&ServeCfg)` 在运行时从代码里现拼**，不维护手写文档：
路径、鉴权方式、数值上下限（`temperature ≤ 5`、`messages ≤ 200`、`stop ≤ 4 个`）
全部和 `parse_chat` 的真实校验共用同一批常量，并有单测
`openapi_spec_constraints_match_parse_chat` 盯着两者不许漂移；另有
`openapi_spec_covers_every_route` 保证 7 条路由一条不漏、
`openapi_spec_every_ref_resolves` 保证 `$ref` 没有悬空引用。

拿到规范就能直接导入 Postman / Apifox / Swagger UI，或者让 OpenAI SDK 按
schema 做请求校验（JSON 端点按 `to_string_pretty` 缩进输出，方便人读和看
git diff）。

**导入即可直接调用**（不用手改示例、不用逐请求配鉴权）：

- `chat` / `embeddings` 的 `requestBody` 在 media type 层带了**能跑通的
  `example`**——Postman 会直接拿它当请求体。若不给 `example`，导入工具会按
  schema 自己编，编出 `n≠1` 或以 assistant 结尾的 messages，一点 Send 就是
  「本服务只支持 n=1」「messages 的最后一条必须是 role=user」两个 400。
  单测 `openapi_examples_pass_real_parsers` 把 example 塞回真实 `parse_chat`
  验一遍，漂移会当场失败。
- 鉴权是**规范里写不了密钥值的**（安全），但已在 `bearerAuth.description`
  里写清做法：导入后在 collection 的 Authorization → Bearer Token 填一次
  `--api-key` 的值即可，子请求全部继承，无需逐请求设置；没给 `--api-key`
  则服务不鉴权。

同一份规范还会在 **`serve` 启动时自动写进 `openapi/` 目录**
（`dump_openapi_files`，与端点出自同一个 `openapi_spec`，逐字一致）：

- [`openapi/openapi.json`](../openapi/openapi.json) / [`openapi/openapi.yaml`](../openapi/openapi.yaml)
- 手删了下次启动也会自动重建，内容跟着代码走、永不过期
- 写盘失败只打一行警告，不影响服务本身
- 不想启服务也能手动刷新（与端点字节一致）：

```bash
curl -s -o openapi/openapi.json http://127.0.0.1:8080/openapi.json
curl -s -o openapi/openapi.yaml http://127.0.0.1:8080/openapi.yaml
```

YAML 是手写的极简序列化器（`json_to_yaml`，不引 yaml 库）：块映射 + 块序列，
字符串按最保守规则决定裸写还是双引号——`true`/`yes`、`1.1.1`、`0x1F`、
`2026-09-27` 这类会被 YAML 解析回 bool/数字的值一律上引号。

---

### 7.8 前端测试页（`web/`，随 serve 一起启动）

`serve` 默认在进入常驻循环前顺带拉起 [`web/`](../web/) 目录下的 **vite dev
server**（React + TypeScript），浏览器打开 `http://localhost:5173` 就是一个
可视化接口调试台：

| 页签 | 覆盖端点 | 能干什么 |
|------|---------|---------|
| 对话 | `POST /v1/chat/completions` | 流式开关、模型下拉、temperature/top_k/top_p/max_tokens 调参、气泡对话、meta（耗时/finish_reason/usage）、原始 JSON、等价 cURL |
| 向量化 | `POST /v1/embeddings` | 多行文本按行批量向量化，显示维度、前 8 维、usage、完整 JSON |
| 状态 | `/health`、`/v1/status`、`/v1/models` | 三卡片并行探活 + 刷新，附 openapi.yaml 下载链接 |

**为什么不需要 `--cors`**：页面请求打的是 vite dev server 自己的源，
vite 把 `/v1`、`/health`、`/openapi.*` **代理转发**到 `http://127.0.0.1:${API_PORT}`
——浏览器视角是同源，CORS 根本不参与；转发目标端口由 Rust 侧通过环境变量
`API_PORT` 传入（vite.config.ts 读取），`--web-port` 传 dev server 端口
（`WEB_PORT`，默认 5173，`strictPort` 占用即报错不静默换端口）。

**启动编排**（`serve::spawn_web_frontend`）：

- 定位 `web/`：先找工作目录（部署时 web/ 与二进制并排放），找不到再退回
  编译时的仓库目录（开发时从子目录 `cargo run`）；
- 用 `node web/node_modules/vite/bin/vite.js` 直接拉起——绕开 Windows 上
  `npm` 实际是 `npm.cmd`、`std::process::Command` 找不到的问题；
- 找不到工程或没装依赖（`node_modules/vite` 不存在）只打一行警告就返回，
  **绝不影响 API 启动**；缺依赖时提示 `cd web && npm install`；
- 子进程交给独立线程 `wait()` 收割（不留僵尸进程），与 API 共享控制台，
  Ctrl-C 时两边一起停。

**开关**：`--no-web` 只起 API；`--web-port <端口>` 换页面端口。
生产部署（systemd/nginx，见 §11）务必加 `--no-web`——dev server 只服务开发调试。

```bash
cargo run --release -- serve --ckpt checkpoints/zh-sft/best.ckpt --api-key sk-test
#   [serve] 前端测试页已启动：http://localhost:5173（代理 API → 127.0.0.1:8080）
```

---

## 8. 错误处理：OpenAI 形状 + 中文原因

```json
{ "error": { "message": "messages 的最后一条必须是 role=user",
             "type": "invalid_request_error",
             "code": null } }
```

| 状态 | type | 典型场景 |
|------|------|---------|
| 400 | `invalid_request_error` | 字段缺失/类型错/取值越界（消息指名**哪个字段**） |
| 401 | `authentication_error` | 缺 key 或 key 错 |
| 404 | `not_found` | 未知路径 |
| 405 | `invalid_request_error` | 方法不对（PUT 等） |
| 413 | `invalid_request_error` | 请求体 > 2 MB |
| 500 | `server_error` | 生成中途出错（SSE 下发一个 error 帧 + `[DONE]`） |

`type` 字段是给**机器**看的（SDK 靠它分类），`message` 是给人看的（中文、说
清怎么改）。SSE 路径下的 500 不能改状态码（200 和首帧早就发出去了），所以发
一个 error 事件帧再发 `[DONE]`——客户端拿到这两帧才知道"不是正常结束"。

---

## 9. 实测记录

`checkpoints/zh-sft-v2/final.ckpt`，`--port 8099 --api-key sk-test --cors --max-new 64`：

| 用例 | 结果 |
|------|------|
| 无 key 访问 `/v1/models` | 401 `authentication_error` |
| 带 key 访问 `/v1/models` | 200，`data[0].id = "final"` |
| 非流式 chat | 200，`finish_reason:"stop"`，`usage{12,33,45}` |
| `stream:true` | 角色帧 → 逐 token delta → finish 帧 → usage 帧 → `[DONE]` |
| embeddings（2 条） | 200，两条 128 维向量，`usage.prompt_tokens:9` |
| 错 key / 空 messages / PUT / 未知路径 | 401 / 400 / 405 / 404（均为 OpenAI 错误体） |
| `OPTIONS` 预检 | 204 + 完整 CORS 头 |
| curl 中途断开 | 服务端 51 token 处中止，打印 `客户端提前断开` |

### 踩坑：PowerShell 下 curl 发 JSON

```powershell
# ❌ 在 PowerShell 双引号字符串里写 \" 会被吞掉，发出去的 body 不合法
curl -d "{\"model\":\"final\", …}"

# ✅ 写临时 UTF-8 文件，再用 --data-binary @file 发
$json = '{"model":"final","stream":true,"messages":[{"role":"user","content":"你好"}]}'
[IO.File]::WriteAllText("$env:TEMP\req.json", $json, [Text.UTF8Encoding]::new($false))
curl.exe -s -N --data-binary "@$env:TEMP\req.json" `
  -H "Content-Type: application/json" `
  -H "Authorization: Bearer sk-test" `
  http://127.0.0.1:8099/v1/chat/completions
```

三个坑叠在一起：PowerShell 的 `\"` 转义规则、`curl` 在 Windows 上可能解析到
别的程序、`-d` 会做 URL 编辑（`--data-binary` 不会）。服务端当时的报错是
`key must be a string at line 1 column 2`——**报错指向的字段名往往不是真凶**，
先怀疑"我发出去的字节到底是什么"（Wireshark 或服务端日志打印原始 body 都行）。

---

## 10. 单元测试（19 个）

`src/serve.rs` 底部的 `#[cfg(test)] mod tests`，不启端口、纯函数级：

| 测试 | 验证内容 |
|------|---------|
| `split_sft_history_shape` / `split_raw_history_shape` | 消息拆成 system/历史/输入三段的形状 |
| `split_sft_two_rounds_join_with_newline` | 多轮历史的换行拼接 |
| `split_merges_multiple_system_messages` | 多条 system 全部汇进 system |
| `merge_system_service_level_comes_first` | 服务级 system 在前 |
| `parse_chat_rejects_bad_requests` | 各种非法请求的 400 与错误消息 |
| `parse_chat_applies_defaults_and_overrides` | 默认值套用与逐字段覆盖（含 SFT stop 缺省） |
| `stop_interner_dedups_identical_slices` | 驻留表去重 |
| `stop_interner_rejects_oversized_input` | 驻留表上限生效（不无界泄漏） |
| `sse_pipe_reads_frames_then_hits_eof` | SsePipe 帧读取 + EOF 语义 |
| `finish_reason_mapping` | EOS/StopMark → stop，MaxNew → length |
| `mean_pool_averages_rows` | 均值池化数值正确 |
| `openapi_spec_covers_every_route` | 规范含 7 条路由、版本 3.1、`bearerAuth`、不鉴权端点的 `security: []` |
| `openapi_spec_constraints_match_parse_chat` | 规范里的数值上限与 `parse_chat` 的真实校验逐项一致 |
| `openapi_spec_every_ref_resolves` | 所有 `$ref` 都能在 `components` 里找到 |
| `json_to_yaml_renders_nested_structures` | YAML 序列化：嵌套对象/数组、空容器、换行与引号 |
| `yaml_quoting_never_leaks_ambiguous_scalars` | `true`/`1.1.1`/`2026-09-27` 一律引号，普通文本裸写 |
| `dump_openapi_files_writes_both_formats` | 启动自动写出的 `openapi/{json,yaml}` 与端点字节逐字一致，JSON 可解析回 7 条路由 |
| `openapi_examples_pass_real_parsers` | requestBody 的 `example` 能过真实 `parse_chat` / input 类型校验——导入 Postman 点 Send 直接 200 |

端到端（鉴权、CORS、SSE、断开中止）靠 §9 的 curl 实测覆盖——HTTP 层的正确性
用真 socket 验证比 mock 更可信。

---

## 11. 上线清单与后续方向

**当前已具备**：流式/非流式、鉴权、CORS、embeddings、请求日志、队列状态、
OpenAPI 规范（运行时端点 + `serve` 启动自动写出 `openapi/{openapi.json,openapi.yaml}`）、
前端测试页（`web/` 随 serve 启动，§7.8，生产加 `--no-web`）、
断开可中断、OpenAI 兼容错误体、资源上限、236 个测试全绿。

**上生产前还要补的**（本课未做，刻意不半成品实现）：

1. **TLS**：`--api-key` 走明文 HTTP 等于裸奔。要么前置 nginx/caddy 终结 TLS，
   要么用 rustls 给 tiny_http 套壳。
2. **限流与超时**：单 IP QPS 上限、生成总时长上限（防止一个 `max_tokens=10^6`
   的请求永久占锁）。
3. **多副本 + 负载均衡**：真并发要么多进程多端口，要么上 GPU。
4. **持久化会话**：现在是无状态的（每次请求带全量 messages），多轮记忆靠调用
   方自己维护——这正是 OpenAI 的语义，但产品化时常要服务端会话。
5. **可观测性**：`/v1/status` 是自绘计数器，生产上通常换成 Prometheus 格式
   （`/metrics`）+ 链路追踪。

**思考题**：

1. 为什么 SSE 的 usage 要单独发一帧，而不是塞进最后一个 delta 帧？
2. `sync_channel` 容量从 16 调成 1，对"慢客户端 + 快生成"的场景各有什么影响？
3. 如果把 `Core` 的 `Mutex` 换成 `RwLock`，能让 embeddings 和生成并行吗？
   （提示：模型权重是只读的，但 KV cache 和 rng 在哪？）
4. `Box::leak` 换 `'static` 的 interner 方案，如果改成 `Arc<[&str]>` 要动哪些
   类型签名？代价是什么？
