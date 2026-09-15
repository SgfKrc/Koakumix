import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // Relative asset URLs on purpose: the same dist is served by the dev server (root)
  // and by the desktop shell (mounted at /app).  Vite's default absolute "/assets/…"
  // 404s under /app and leaves the pywebview window blank.
  base: './',
  server: {
    host: '127.0.0.1',
    port: 5180,
    proxy: {
      '/healthz': 'http://127.0.0.1:8090',
      '/v1': 'http://127.0.0.1:8090',
    },
  },
  build: { outDir: 'dist', sourcemap: true },
});
