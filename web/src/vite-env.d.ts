/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** serve 拉起 vite 时注入的鉴权 key（`VITE_` 前缀才会出现在这里）；手动 `npm run dev` 时为空 */
  readonly VITE_API_KEY?: string
}
