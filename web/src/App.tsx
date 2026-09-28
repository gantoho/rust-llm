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
  /** 本轮耗时毫秒 / finish_reason / usage 摘要 */
  meta?: string
}

const DEFAULT_PARAMS: ChatParams = {
  temperature: 0.8,
  top_k: 40,
  top_p: 0.9,
  max_tokens: 200,
}

export default function App() {
  const [tab, setTab] = useState<Tab>('chat')
  // API key 持久化到 localStorage，刷新不丢
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('api_key') ?? '')
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

  return (
    <div className="page">
      <header className="topbar">
        <h1>llm-scratch API 测试台</h1>
        <nav>
          {(
            [
              ['chat', '对话补全'],
              ['embeddings', '向量化'],
              ['status', '服务状态'],
            ] as [Tab, string][]
          ).map(([id, label]) => (
            <button
              key={id}
              className={tab === id ? 'tab active' : 'tab'}
              onClick={() => setTab(id)}
            >
              {label}
            </button>
          ))}
        </nav>
        <div className="keybox">
          <input
            type="password"
            placeholder="API Key（服务未开鉴权可留空）"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
          />
          <a href="/openapi.json" target="_blank" rel="noreferrer" title="查看 OpenAPI 规范">
            OpenAPI
          </a>
        </div>
      </header>

      {serverErr && (
        <div className="banner err">
          后端不可达：{serverErr} —— 请先运行 <code>cargo run -- serve</code>
        </div>
      )}

      {tab === 'chat' && (
        <ChatTab apiKey={apiKey} model={model} models={models} onModel={setModel} />
      )}
      {tab === 'embeddings' && <EmbedTab apiKey={apiKey} model={model} />}
      {tab === 'status' && <StatusTab apiKey={apiKey} />}
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

    // 生成速率：服务端每个 token 发一帧增量，按「首帧→当前」窗口算 tok/s
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
        patchLast((t) => ({
          ...t,
          streaming: false,
          meta: `${elapsed()}s · ${finish || 'stop'}${
            usage ? ` · ${usage.completion_tokens} tokens` : ''
          }${tFirst ? ` · ${genRate().toFixed(1)} tok/s` : ''}`,
        }))
      } else {
        const resp: any = await chat(apiKey, history, params, model)
        const text = resp.choices?.[0]?.message?.content ?? ''
        const u = resp.usage
        const rate = u
          ? ` · ${((u.completion_tokens * 1000) / Math.max(performance.now() - t0, 1)).toFixed(1)} tok/s`
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
    <section className="grid">
      <div className="card chat">
        <div className="controls">
          <label>
            模型
            <select value={model} onChange={(e) => onModel(e.target.value)}>
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
          </label>
          <label className="switch">
            <input type="checkbox" checked={stream} onChange={(e) => setStream(e.target.checked)} />
            流式（SSE）
          </label>
          <Num label="temperature" value={params.temperature} step={0.1} onChange={(v) => setParams({ ...params, temperature: v })} />
          <Num label="top_k" value={params.top_k} step={1} onChange={(v) => setParams({ ...params, top_k: v })} />
          <Num label="top_p" value={params.top_p} step={0.05} onChange={(v) => setParams({ ...params, top_p: v })} />
          <Num label="max_tokens" value={params.max_tokens} step={10} onChange={(v) => setParams({ ...params, max_tokens: v })} />
          <button className="ghost" onClick={() => { setTurns([]); setRaw('') }} disabled={busy}>
            清空对话
          </button>
        </div>

        <div className="messages">
          {turns.length === 0 && (
            <p className="placeholder">输入消息开始测试；开启「流式」可逐 token 观察 SSE 输出。</p>
          )}
          {turns.map((t, i) => (
            <div key={i} className={`bubble ${t.role}`}>
              <span className="who">{t.role === 'user' ? '我' : '模型'}</span>
              <pre>{t.content || (t.streaming ? '▍' : '')}</pre>
              {t.meta && <small>{t.meta}</small>}
            </div>
          ))}
          <div ref={bottomRef} />
        </div>

        <div className="composer">
          <textarea
            rows={3}
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
          <div className="btns">
            {busy ? (
              <button className="danger" onClick={stop}>
                停止
              </button>
            ) : (
              <button onClick={() => void send()} disabled={!input.trim()}>
                发送
              </button>
            )}
          </div>
        </div>
      </div>

      <aside className="card side">
        <h2>原始响应</h2>
        {stream ? (
          <p className="hint">流式模式逐帧渲染在左侧；如需整包 JSON 请关闭流式后发送。</p>
        ) : (
          <pre className="code">{raw || '（尚未产生非流式响应）'}</pre>
        )}
        <h2>请求等价 cURL</h2>
        <pre className="code">{curlExample(model, params, stream, turns.map(({ role, content }) => ({ role, content })))}</pre>
      </aside>
    </section>
  )
}

function Num({ label, value, step, onChange }: { label: string; value: number; step: number; onChange: (v: number) => void }) {
  return (
    <label>
      {label}
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

/** 生成与当前会话等价的 cURL 命令，方便复制到终端验证 */
function curlExample(model: string, p: ChatParams, stream: boolean, messages: ChatMessage[]): string {
  const body = JSON.stringify({ model, messages, ...p, stream }, null, 0)
  return `curl http://127.0.0.1:8080/v1/chat/completions \\\n  -H "Content-Type: application/json" \\\n  -H "Authorization: Bearer $API_KEY" \\\n  -d '${body}'`
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
    <section className="grid">
      <div className="card">
        <h2>POST /v1/embeddings</h2>
        <p className="hint">每行一段文本，一次请求批量向量化（服务端上限 32 条）。</p>
        <textarea rows={8} value={text} onChange={(e) => setText(e.target.value)} />
        <div className="btns">
          <button onClick={() => void run()} disabled={busy || !text.trim()}>
            {busy ? '计算中…' : '生成向量'}
          </button>
        </div>
        {err && <div className="banner err">{err}</div>}
      </div>
      <aside className="card side">
        <h2>结果</h2>
        <pre className="code">{result || '（无）'}</pre>
      </aside>
    </section>
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
    <section className="grid three">
      <div className="card">
        <h2>GET /health</h2>
        <pre className="code">{health || '（无）'}</pre>
      </div>
      <div className="card">
        <h2>GET /v1/status</h2>
        <p className="hint">队列长度、请求计数、模型信息（不鉴权）。</p>
        <pre className="code">{status || '（无）'}</pre>
      </div>
      <div className="card">
        <h2>GET /v1/models</h2>
        <pre className="code">{modelsJson || '（无）'}</pre>
        <div className="btns">
          <button onClick={() => void refresh()}>刷新</button>
          <a className="linkbtn" href="/openapi.yaml" target="_blank" rel="noreferrer">
            openapi.yaml
          </a>
        </div>
        {err && <div className="banner err">{err}</div>}
      </div>
    </section>
  )
}
