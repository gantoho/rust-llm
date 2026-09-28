// API 客户端：所有请求都走同源（vite dev 代理到 Rust serve，生产时页面与 API 同域）
// 鉴权与 OpenAI 一致：`Authorization: Bearer <key>`

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant'
  content: string
}

export interface ChatParams {
  temperature: number
  top_k: number
  top_p: number
  max_tokens: number
}

/** 组装请求头：有 key 才带鉴权 */
function headers(apiKey: string, json = true): Record<string, string> {
  const h: Record<string, string> = {}
  if (json) h['Content-Type'] = 'application/json'
  if (apiKey) h['Authorization'] = `Bearer ${apiKey}`
  return h
}

/** 统一错误处理：把后端的错误 JSON 转成可读消息 */
async function fail(res: Response): Promise<never> {
  let detail = ''
  try {
    const body = await res.json()
    detail = body?.error?.message ?? JSON.stringify(body)
  } catch {
    detail = await res.text().catch(() => '')
  }
  throw new Error(`HTTP ${res.status}${detail ? `：${detail}` : ''}`)
}

export async function getHealth(apiKey: string): Promise<unknown> {
  const res = await fetch('/health', { headers: headers(apiKey, false) })
  if (!res.ok) await fail(res)
  return res.json()
}

export async function getStatus(apiKey: string): Promise<unknown> {
  const res = await fetch('/v1/status', { headers: headers(apiKey, false) })
  if (!res.ok) await fail(res)
  return res.json()
}

export async function getModels(apiKey: string): Promise<{ data: { id: string }[] }> {
  const res = await fetch('/v1/models', { headers: headers(apiKey, false) })
  if (!res.ok) await fail(res)
  return res.json()
}

export interface EmbeddingResult {
  data: { object: 'embedding'; index: number; embedding: number[] }[]
  usage: { prompt_tokens: number; total_tokens: number }
}

export async function embed(
  apiKey: string,
  input: string[],
  model: string,
): Promise<EmbeddingResult> {
  const res = await fetch('/v1/embeddings', {
    method: 'POST',
    headers: headers(apiKey),
    body: JSON.stringify({ input, model }),
  })
  if (!res.ok) await fail(res)
  return res.json()
}

/** 非流式对话补全：返回原始响应 JSON（用于「查看原始响应」） */
export async function chat(
  apiKey: string,
  messages: ChatMessage[],
  params: ChatParams,
  model: string,
): Promise<unknown> {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: headers(apiKey),
    body: JSON.stringify({ model, messages, ...params, stream: false }),
  })
  if (!res.ok) await fail(res)
  return res.json()
}

export interface StreamCallbacks {
  /** 收到角色帧（首帧） */
  onRole?: () => void
  /** 收到 content 增量 */
  onDelta: (text: string) => void
  /** 收到 finish_reason 帧 */
  onFinish?: (reason: string) => void
  /** 收到 usage 帧（choices 为空数组的最后一帧） */
  onUsage?: (usage: unknown) => void
  /** `data: [DONE]` */
  onDone?: () => void
}

/**
 * 流式对话补全：手动解析 SSE（EventSource 不能带自定义头，鉴权得用 fetch + ReadableStream）。
 * 帧序见 serve.rs：role → content 增量… → finish_reason → usage → [DONE]。
 */
export async function chatStream(
  apiKey: string,
  messages: ChatMessage[],
  params: ChatParams,
  model: string,
  cb: StreamCallbacks,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch('/v1/chat/completions', {
    method: 'POST',
    headers: headers(apiKey),
    body: JSON.stringify({ model, messages, ...params, stream: true }),
    signal,
  })
  if (!res.ok) await fail(res)
  if (!res.body) throw new Error('响应没有 body')

  const reader = res.body.getReader()
  const decoder = new TextDecoder()
  let buf = ''
  for (;;) {
    const { done, value } = await reader.read()
    if (done) break
    buf += decoder.decode(value, { stream: true })
    // SSE 帧以空行分隔；服务端固定输出 `data: {json}\n\n`
    let sep: number
    while ((sep = buf.indexOf('\n\n')) >= 0) {
      const frame = buf.slice(0, sep)
      buf = buf.slice(sep + 2)
      for (const line of frame.split('\n')) {
        if (!line.startsWith('data:')) continue
        const payload = line.slice(5).trim()
        if (payload === '[DONE]') {
          cb.onDone?.()
          return
        }
        let obj: any
        try {
          obj = JSON.parse(payload)
        } catch {
          continue
        }
        const choice = obj.choices?.[0]
        if (obj.usage) cb.onUsage?.(obj.usage)
        if (!choice) continue
        if (choice.delta?.role) cb.onRole?.()
        if (choice.delta?.content) cb.onDelta(choice.delta.content)
        if (choice.finish_reason) cb.onFinish?.(choice.finish_reason)
      }
    }
  }
}
