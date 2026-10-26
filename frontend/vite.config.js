import { defineConfig, loadEnv } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig(({ mode }) => {
  // This value is read by the Vite server only. Never use a VITE_ prefix for it.
  const env = loadEnv(mode, process.cwd(), "");

  return {
    plugins: [react()],
    server: {
      host: "127.0.0.1",
      port: 5173,
      proxy: {
        "/api": {
          target: env.API_ORIGIN || "http://127.0.0.1:8000",
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api/, ""),
          configure: (proxy) => {
            proxy.on("proxyReq", (proxyRequest) => {
              if (env.CHAT_API_TOKEN) {
                proxyRequest.setHeader("X-API-Token", env.CHAT_API_TOKEN);
              }
            });
          },
        },
      },
    },
  };
});
