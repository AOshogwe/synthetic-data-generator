import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// Built as a static SPA served by Flask at /v2 (see app.py), alongside the
// existing index.html at / -- parallel-route rollout so nothing breaks
// while this is being built out to feature parity.
//
// Tailwind is compiled into the bundle by the @tailwindcss/vite plugin
// rather than loaded from cdnjs at runtime (see index.html / index.css):
// the app's CSP (utils/security_middleware.py) only allows 'self' for
// script-src/style-src, so the old CDN <script>/<link> tags were being
// silently blocked in production -- Tailwind's utility classes never
// applied and the page rendered as unstyled HTML.
export default defineConfig({
  plugins: [react(), tailwindcss()],
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
