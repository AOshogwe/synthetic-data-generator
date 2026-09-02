import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Built as a static SPA served by Flask at /v2 (see app.py), alongside the
// existing index.html at / -- parallel-route rollout so nothing breaks
// while this is being built out to feature parity.
export default defineConfig({
  plugins: [react()],
  base: '/v2/',
  build: {
    outDir: '../static_v2',
    emptyOutDir: true,
  },
  server: {
    proxy: {
      '/api': 'http://localhost:5000',
      '/health': 'http://localhost:5000',
    },
  },
})
