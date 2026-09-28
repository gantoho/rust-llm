import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 后端 API 端口：serve 启动前端时会通过环境变量注入；本地手动 dev 时可用 API_PORT=8080 npm run dev
const apiPort = process.env.API_PORT ?? '8080'
const apiTarget = `http://127.0.0.1:${apiPort}`

// 开发时浏览器只访问 vite dev server，/v1 等 API 路径代理到 Rust serve，
// 这样页面无需 --cors 也能调通，SSE 流式响应同样经代理透传。
export default defineConfig({
  plugins: [react()],
  server: {
    port: Number(process.env.WEB_PORT ?? 5173),
    strictPort: true,
    proxy: {
      '/v1': { target: apiTarget, changeOrigin: true },
      '/health': { target: apiTarget, changeOrigin: true },
      '/openapi.json': { target: apiTarget, changeOrigin: true },
      '/openapi.yaml': { target: apiTarget, changeOrigin: true },
    },
  },
})
