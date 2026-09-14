import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import seo from './vite-plugin-seo'

// Backend target for the dev proxy. Defaults to the docker-compose service
// name; set VITE_PROXY_TARGET=http://localhost:8000 to run the dev server on
// the host against a backend reachable at localhost (no CORS, same-origin).
const backend = process.env.VITE_PROXY_TARGET || 'http://backend:8000'
const renderer = process.env.VITE_RENDER_TARGET || 'http://renderer:3100'

// Docker Desktop (Windows/macOS) does not forward host file-change events into
// a container's bind mount, so the dev server inside Docker never sees an edit
// and keeps serving the old module until it is restarted (a new UI row looked
// missing on 14-sep-2026 while the file on disk already had it). Polling fixes
// that. docker-compose sets VITE_USE_POLLING=1; a dev server run on the host
// keeps chokidar's native events, where polling would only waste CPU.
const usePolling = process.env.VITE_USE_POLLING === '1'

// https://vitejs.dev/config/
export default defineConfig({
  // seo() runs on build only. It injects the crawler-visible homepage content
  // into #root and emits the static /alternatives pages, sitemap.xml and
  // llms.txt. See vite-plugin-seo.js.
  plugins: [react(), seo()],
  server: {
    // ~85 watched files (node_modules and .git are excluded by default).
    // Measured on 14-sep-2026, fresh process each, median of 20 docker-stats
    // samples (% of one core): no polling 0.39% but edits never show; 500 ms
    // 9.25% with edits served in 0.48 s; 1000 ms 4.39%, served in 0.72 s.
    // Halving the rate halved the cost, so 1000 ms keeps edits under a second
    // for half the CPU. Readings after Vite's own restart-on-config-change ran
    // 2-3x higher, so restart the container after editing this file.
    watch: usePolling ? { usePolling: true, interval: 1000, binaryInterval: 2000 } : undefined,
    allowedHosts: [
      'openshorts.app',
      'www.openshorts.app'
    ],
    proxy: {
      '/api': { target: backend, changeOrigin: true },
      '/videos': { target: backend, changeOrigin: true },
      '/thumbnails': { target: backend, changeOrigin: true },
      '/gallery': { target: backend, changeOrigin: true },
      '/video': { target: backend, changeOrigin: true },
      '/render': { target: renderer, changeOrigin: true },
    }
  }
})
