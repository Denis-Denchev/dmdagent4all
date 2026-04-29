import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

declare const process: {
  env: Record<string, string | undefined>
}

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      '/v1': process.env.DMDAGENT_API_URL ?? 'http://127.0.0.1:8765',
      '/health': process.env.DMDAGENT_API_URL ?? 'http://127.0.0.1:8765',
    },
  },
})
