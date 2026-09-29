import { useEffect, useRef, useState } from 'react'
import {
  ChatMessage,
  ChatParams,
  chat,
  chatStream,
  embed,
  getHealth,
  getModels,
  getStatus,
} from './api'

type Tab = 'chat' | 'embeddings' | 'status'

interface Turn {
  role: 'user' | 'assistant'
  content: string
  /** 流式生成中（末条 assistant 才为 true） */
  streaming?: boolean
  /** 本轮耗时毫秒 / finish_reason / usage / tok/s 摘要 */
  meta?: string
}

const DEFAULT_PARAMS: ChatParams = {
  temperature: 0.8,
  top_k: 40,
  top_p: 0.9,
  max_tokens: 200,
}

// ==================== 图标（内联 SVG，无外部依赖） ====================

const IconChat = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 12a8 8 0 0 1-8 8H4l2.2-2.6A8 8 0 1 1 21 12Z" />
    <path d="M8.5 10.5h7M8.5 13.5h4.5" />
  </svg>
)

const IconEmbed = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="5.5" cy="6" r="2" />
    <circle cx="5.5" cy="18" r="2" />
    <circle cx="18.5" cy="12" r="2" />
    <path d="M7.5 7l9 4M7.5 17l9-4" />
  </svg>
)

const IconStatus = () => (
  <svg viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 12h4l2.5-6 4 12L16 12h5" />
  </svg>
)

const IconSliders = () => (
  <svg viewBox="0 0 24 24" width="15" height="15" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round">
    <path d="M4 7h10M18 7h2M4 17h4M12 17h8" />
    <circle cx="16" cy="7" r="2" />
    <circle cx="10" cy="17" r="2" />
  </svg>
)

// ==================== 应用外壳 ====================

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  // API key：优先用 serve 启动前端时注入的 VITE_API_KEY（key 每次启动都可能变，
  // 只有它必然对得上当前后端）；手动 `npm run dev` 没有该变量时退回 localStorage
  const [apiKey, setApiKey] = useState(
    () => import.meta.env.VITE_API_KEY || localStorage.getItem('api_key') || '',
  )
  const [model, setModel] = useState('')
  const [models, setModels] = useState<string[]>([])
  const [serverErr, setServerErr] = useState('')

  useEffect(() => {
    localStorage.setItem('api_key', apiKey)
  }, [apiKey])

  // 探活 + 拉模型列表：挂载时跑一次，API key 改变后防抖重拉
  // （key 错/缺失时 /v1/models 会 401，补上 key 就能刷出模型下拉）
  useEffect(() => {
    const timer = setTimeout(() => {
      getHealth(apiKey)
        .then(() => getModels(apiKey))
        .then((m) => {
          setServerErr('')
          const ids = m.data.map((x) => x.id)
          setModels(ids)
          setModel((cur) => (cur && ids.includes(cur) ? cur : ids[0] || ''))
        })
        .catch((e: Error) => setServerErr(e.message))
    }, 400)
    return () => clearTimeout(timer)
  }, [apiKey])

  const navItems: [Tab, string, React.ReactNode][] = [
    ['chat', '对话补全', <IconChat key="i" />],
    ['embeddings', '向量化', <IconEmbed key="i" />],
    ['status', '服务状态', <IconStatus key="i" />],
  ]

  return (
    <div className="app">
      {/* ---- 侧边导航 ---- */}
      <aside className="sidebar">
        <div className="brand">
          <span className="logo">
            <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
              <path d="M12 3l7 4.5v9L12 21l-7-4.5v-9L12 3Z" />
              <path d="M12 8.5v7M8.5 10.5v3M15.5 10.5v3" />
            </svg>
          </span>
          <div className="brand-text">
            <strong>llm-scratch</strong>
            <small>API 测试台</small>
          </div>
        </div>

        <nav className="sidenav">
          {navItems.map(([id, label, icon]) => (
            <button key={id} className={tab === id ? 'navbtn active' : 'navbtn'} onClick={() => setTab(id)}>
              {icon}
              <span>{label}</span>
            </button>
          ))}
        </nav>

        <div className="side-foot">
          <div className={`backend ${serverErr ? 'off' : 'on'}`}>
            <i className="dot" />
            {serverErr ? '后端不可达' : '后端在线'}
          </div>
          <label className="keyrow">
            API Key
            <input
              type="password"
              placeholder="serve 启动时自动注入"
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
            />
          </label>
          <a className="openapi-link" href="/openapi.json" target="_blank" rel="noreferrer">
            OpenAPI 规范 ↗
          </a>
        </div>
      </aside>

      {/* ---- 主内容 ---- */}
      <main className="main">
        {serverErr && (
          <div className="banner err">
            后端不可达：{serverErr} —— 请先运行 <code>cargo run -- serve</code>
          </div>
        )}
        {tab === 'chat' && <ChatTab apiKey={apiKey} model={model} models={models} onModel={setModel} />}
        {tab === 'embeddings' && <EmbedTab apiKey={apiKey} model={model} />}
        {tab === 'status' && <StatusTab apiKey={apiKey} />}
      </main>
    </div>
  )
}

// ==================== 对话补全 ====================

function ChatTab({
  apiKey,
  model,
  models,
  onModel,
}: {
  apiKey: string
  model: string
  models: string[]
  onModel: (m: string) => void
}) {
  const [turns, setTurns] = useState<Turn[]>([])
  const [input, setInput] = useState('')
  const [stream, setStream] = useState(true)
  const [params, setParams] = useState<ChatParams>(DEFAULT_PARAMS)
  const [busy, setBusy] = useState(false)
  const [raw, setRaw] = useState('')
  const [drawer, setDrawer] = useState(false)
  const abortRef = useRef<AbortController | null>(null)
  const bottomRef = useRef<HTMLDivElement>(null)

  // 新内容落盘后自动滚到底
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [turns])

  async function send() {
    const question = input.trim()
    if (!question || busy) return
    setInput('')
    setBusy(true)
    setRaw('')

    // 拼历史：只保留 user/assistant 轮次，末条必为 user（服务端强校验）
    const history: ChatMessage[] = [...turns.map(({ role, content }) => ({ role, content })), { role: 'user', content: question }]
    const next: Turn[] = [...turns, { role: 'user', content: question }, { role: 'assistant', content: '', streaming: stream }]
    setTurns(next)

    const t0 = performance.now()
    // 注意：updater 必须是纯函数——StrictMode 在 dev 下会把 setState updater
    // 调两次，原地 `content +=` 会让流式回复翻倍，这里一律返回新对象
    const patch = (fn: (t: Turn[]) => Turn[]) => setTurns((cur) => fn([...cur]))
    const patchLast = (fn: (t: Turn) => Turn) =>
      patch((ts) => ts.map((t, i) => (i === ts.length - 1 ? fn(t) : t)))
    const elapsed = () => ((performance.now() - t0) / 1000).toFixed(1)

    // 流式过程中只能本地估算（服务端速率随 usage 帧在收尾时才到），收尾后优先用服务端实测值
    let tokens = 0
    let tFirst = 0
    const genRate = () =>
      tFirst ? ((tokens - 1) * 1000) / Math.max(performance.now() - tFirst, 1) : 0

    try {
      if (stream) {
        let finish = ''
        let usage: any
        await chatStream(
          apiKey,
          history,
          params,
          model,
          {
            onDelta: (d) => {
              tokens += 1
              if (!tFirst) tFirst = performance.now()
              patchLast((t) => ({
                ...t,
                content: t.content + d,
                meta: `生成中 · ${genRate().toFixed(1)} tok/s`,
              }))
            },
            onFinish: (r) => (finish = r),
            onUsage: (u) => (usage = u),
          },
          (abortRef.current = new AbortController()).signal,
        )
        // 速率优先用服务端实测值（Instant 计时，不受 TCP/代理缓冲影响），拿不到再退回本地估算
        const srvRate =
          typeof usage?.tokens_per_second === 'number' ? usage.tokens_per_second : null
        const rate = srvRate ?? (tFirst ? genRate() : null)
        patchLast((t) => ({
          ...t,
          streaming: false,
          meta: `${elapsed()}s · ${finish || 'stop'}${
            usage ? ` · ${usage.completion_tokens} tokens` : ''
          }${rate !== null ? ` · ${rate.toFixed(1)} tok/s` : ''}`,
        }))
      } else {
        const resp: any = await chat(apiKey, history, params, model)
        const text = resp.choices?.[0]?.message?.content ?? ''
        const u = resp.usage
        // 优先服务端实测速率，缺失时退回「completion_tokens ÷ 请求耗时」的本地估算
        const srv = typeof u?.tokens_per_second === 'number' ? u.tokens_per_second : null
        const rate = u
          ? ` · ${(srv ?? (u.completion_tokens * 1000) / Math.max(performance.now() - t0, 1)).toFixed(1)} tok/s`
          : ''
        patchLast((t) => ({
          ...t,
          content: text,
          streaming: false,
          meta: `${elapsed()}s · ${resp.choices?.[0]?.finish_reason ?? 'stop'}${
            u ? ` · ${u.completion_tokens} tokens` : ''
          }${rate}`,
        }))
        setRaw(JSON.stringify(resp, null, 2))
      }
    } catch (e) {
      patchLast((t) => ({
        ...t,
        streaming: false,
        meta: `失败：${(e as Error).message}`,
      }))
    } finally {
      abortRef.current = null
      setBusy(false)
    }
  }

  function stop() {
    abortRef.current?.abort()
  }

  return (
    <div className="pane">
      <header className="pane-head">
        <div className="pane-title">
          <h2>对话补全</h2>
          <code className="ep">POST /v1/chat/completions</code>
        </div>
        <div className="head-tools">
          <select className="ctl" value={model} onChange={(e) => onModel(e.target.value)}>
            {models.length ? (
              models.map((m) => (
                <option key={m} value={m}>
                  {m}
                </option>
              ))
            ) : (
              <option value={model || 'gpt-scratch'}>{model || 'gpt-scratch'}</option>
            )}
          </select>
          <label className="switch" title="SSE 逐 token 输出">
            <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
            <span className="track" />
            流式
          </label>
          <button className="ghost" onClick={() => setDrawer(true)}>
            <IconSliders /> 参数
          </button>
          <button className="ghost" onClick={() => { setTurns([]); setRaw('') }} disabled={busy}>
            清空
          </button>
        </div>
      </header>

      <div className="chat-scroll">
        <div className="chat-flow">
          {turns.length === 0 && (
            <div className="hero">
              <span className="hero-logo">
                <svg viewBox="0 0 24 24" width="26" height="26" fill="none" stroke="#fff" strokeWidth="2" strokeLinecap="round">
                  <path d="M12 3l7 4.5v9L12 21l-7-4.5v-9L12 3Z" />
                  <path d="M12 8.5v7M8.5 10.5v3M15.5 10.5v3" />
                </svg>
              </span>
              <h3>从一次对话开始测试接口</h3>
              <p>
                开启「流式」可逐 token 观察 SSE 输出与实时生成速率；右上角「参数」可调 temperature / top_k / top_p / max_tokens。
              </p>
            </div>
          )}
          {turns.map((t, i) => (
            <div key={i} className={`bubble ${t.role}${t.streaming ? ' streaming' : ''}`}>
              <span className="avatar">{t.role === 'user' ? '我' : '◆'}</span>
              <div className="body">
                <pre>{t.content}</pre>
                {t.meta && <small className="meta">{renderMeta(t.meta)}</small>}
              </div>
            </div>
          ))}
          <div ref={bottomRef} />
        </div>
      </div>

      <div className="dock">
        <div className="composer-pill">
          <textarea
            rows={2}
            placeholder="输入消息，Enter 发送，Shift+Enter 换行"
            value={input}
            disabled={busy}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault()
                void send()
              }
            }}
          />
          {busy ? (
            <button className="danger stop" onClick={stop}>
              停止
            </button>
          ) : (
            <button className="send" onClick={() => void send()} disabled={!input.trim()}>
              发送
            </button>
          )}
        </div>
      </div>

      {/* ---- 参数 / 原始响应抽屉 ---- */}
      <div className={drawer ? 'scrim show' : 'scrim'} onClick={() => setDrawer(false)} />
      <aside className={drawer ? 'drawer open' : 'drawer'}>
        <div className="drawer-head">
          <h3>请求详情</h3>
          <button className="ghost" onClick={() => setDrawer(false)}>
            ✕
          </button>
        </div>

        <section>
          <h4>生成参数</h4>
          <div className="param-grid">
            <Num label="temperature" value={params.temperature} step={0.1} onChange={(v) => setParams({ ...params, temperature: v })} />
            <Num label="top_k" value={params.top_k} step={1} onChange={(v) => setParams({ ...params, top_k: v })} />
            <Num label="top_p" value={params.top_p} step={0.05} onChange={(v) => setParams({ ...params, top_p: v })} />
            <Num label="max_tokens" value={params.max_tokens} step={10} onChange={(v) => setParams({ ...params, max_tokens: v })} />
          </div>
        </section>

        <section>
          <h4>原始响应</h4>
          {stream ? (
            <p className="hint">流式模式逐帧渲染在对话区；如需整包 JSON 请关闭流式后发送。</p>
          ) : (
            <pre className="code">{raw || '（尚未产生非流式响应）'}</pre>
          )}
        </section>

        <section>
          <h4>等价 cURL</h4>
          <pre className="code">{curlExample(apiKey, model, params, stream, turns.map(({ role, content }) => ({ role, content })))}</pre>
        </section>
      </aside>
    </div>
  )
}

/** 把 meta 里的 tok/s 片段渲染成高亮徽章 */
function renderMeta(meta: string) {
  const m = meta.match(/([\d.]+ tok\/s)/)
  if (!m) return meta
  const i = meta.indexOf(m[1])
  return (
    <>
      {meta.slice(0, i)}
      <b className="rate">{m[1]}</b>
      {meta.slice(i + m[1].length)}
    </>
  )
}

function Num({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <label className="num">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={value}
        onChange={(e) => {
          const v = Number(e.target.value)
          if (Number.isFinite(v)) onChange(v)
        }}
      />
    </label>
  )
}

/** 生成与当前会话等价的 cURL 命令，方便复制到终端验证（key 直接带当前值，复制即可跑） */
function curlExample(apiKey: string, model: string, p: ChatParams, stream: boolean, messages: ChatMessage[]): string {
  const body = JSON.stringify({ model, messages, ...p, stream }, null, 0)
  const auth = apiKey || '$API_KEY'
  return `curl http://127.0.0.1:8080/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer ${auth}" \\\n  -d '${body}'`
}

// ==================== 向量化 ====================

function EmbedTab({ apiKey, model }: { apiKey: string; model: string }) {
  const [text, setText] = useState('你好，世界\nhello world')
  const [result, setResult] = useState('')
  const [err, setErr] = useState('')
  const [busy, setBusy] = useState(false)

  async function run() {
    const inputs = text.split('\n').map((s) => s.trim()).filter(Boolean)
    if (!inputs.length || busy) return
    setBusy(true)
    setErr('')
    try {
      const r = await embed(apiKey, inputs, model || 'gpt-scratch')
      const summary = r.data
        .map((d) => {
          const head = d.embedding.slice(0, 8).map((x) => x.toFixed(4)).join(', ')
          return `#${d.index}  维度 ${d.embedding.length}  前 8 维：[${head}, …]`
        })
        .join('\n')
      setResult(`${summary}\n\nusage: ${JSON.stringify(r.usage)}\n\n完整响应：\n${JSON.stringify(r, null, 2)}`)
    } catch (e) {
      setErr((e as Error).message)
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="pane">
      <header className="pane-head">
        <div className="pane-title">
          <h2>向量化</h2>
          <code className="ep">POST /v1/embeddings</code>
        </div>
      </header>
      <div className="pane-body scroll">
        <div className="cards two">
          <div className="card">
            <h3>输入</h3>
            <p className="hint">每行一段文本，一次请求批量向量化（服务端上限 32 条）。</p>
            <textarea rows={9} value={text} onChange={(e) => setText(e.target.value)} />
            <div className="btns">
              <button onClick={() => void run()} disabled={busy || !text.trim()}>
                {busy ? '计算中…' : '生成向量'}
              </button>
            </div>
            {err && <div className="banner err">{err}</div>}
          </div>
          <div className="card">
            <h3>结果</h3>
            <pre className="code tall">{result || '（无）'}</pre>
          </div>
        </div>
      </div>
    </div>
  )
}

// ==================== 服务状态 ====================

function StatusTab({ apiKey }: { apiKey: string }) {
  const [health, setHealth] = useState('')
  const [status, setStatus] = useState('')
  const [modelsJson, setModelsJson] = useState('')
  const [err, setErr] = useState('')

  async function refresh() {
    setErr('')
    try {
      const [h, s, m] = await Promise.all([getHealth(apiKey), getStatus(apiKey), getModels(apiKey)])
      setHealth(JSON.stringify(h, null, 2))
      setStatus(JSON.stringify(s, null, 2))
      setModelsJson(JSON.stringify(m, null, 2))
    } catch (e) {
      setErr((e as Error).message)
    }
  }

  useEffect(() => {
    void refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [apiKey])

  return (
    <div className="pane">
      <header className="pane-head">
        <div className="pane-title">
          <h2>服务状态</h2>
          <code className="ep">GET /health · /v1/status · /v1/models</code>
        </div>
        <div className="head-tools">
          <button className="ghost" onClick={() => void refresh()}>刷新</button>
        </div>
      </header>
      <div className="pane-body scroll">
        <div className="cards three">
          <div className="card">
            <div className="card-head">
              <h3>探活</h3>
              <span className="method get">GET</span>
            </div>
            <code className="ep inline">/health</code>
            <pre className="code">{health || '（无）'}</pre>
          </div>
          <div className="card">
            <div className="card-head">
              <h3>运行指标</h3>
              <span className="method get">GET</span>
            </div>
            <code className="ep inline">/v1/status</code>
            <p className="hint">队列长度、请求计数、模型信息（不鉴权）。</p>
            <pre className="code">{status || '（无）'}</pre>
          </div>
          <div className="card">
            <div className="card-head">
              <h3>模型列表</h3>
              <span className="method get">GET</span>
            </div>
            <code className="ep inline">/v1/models</code>
            <pre className="code">{modelsJson || '（无）'}</pre>
            <div className="btns">
              <a className="linkbtn" href="/openapi.yaml" target="_blank" rel="noreferrer">
                openapi.yaml
              </a>
            </div>
          </div>
        </div>
        {err && <div className="banner err">{err}</div>}
      </div>
    </div>
  )
}
