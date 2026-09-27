import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In development the API runs on 8000 and Vite on 5173. Proxying /api keeps the
// front-end code origin-agnostic, so the same fetch calls work unchanged when
// FastAPI serves the built assets itself in the container.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: "dist",
    sourcemap: true,
  },
});
