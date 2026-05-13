import path from "node:path";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    // CodeMirror's YAML grammar + state machinery ships as a single
    // ~525KB chunk. We lazy-load it only when the YAML pane is opened,
    // so a higher threshold reflects the real shipped initial JS budget.
    chunkSizeWarningLimit: 700,
  },
});
