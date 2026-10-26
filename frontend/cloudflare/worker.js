export default {
  async fetch(request, env) {
    const url = new URL(request.url);

    if (url.pathname.startsWith("/api/")) {
      const upstream = new URL(url.pathname.replace(/^\/api/, "") + url.search, env.API_ORIGIN);
      const headers = new Headers(request.headers);
      headers.set("X-API-Token", env.CHAT_API_TOKEN);
      headers.delete("host");

      return fetch(new Request(upstream, {
        method: request.method,
        headers,
        body: request.method === "GET" || request.method === "HEAD" ? undefined : request.body,
        redirect: "manual",
      }));
    }

    return env.ASSETS.fetch(request);
  },
};
