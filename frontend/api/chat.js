export default async function handler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ detail: "Method not allowed." });
  }

  const backendUrl = process.env.BACKEND_API_URL?.replace(/\/$/, "");
  const apiToken = process.env.CHAT_API_TOKEN;

  if (!backendUrl || !apiToken) {
    console.error("Missing BACKEND_API_URL or CHAT_API_TOKEN.");
    return response.status(500).json({ detail: "Server configuration is incomplete." });
  }

  try {
    const upstream = await fetch(`${backendUrl}/chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Token": apiToken,
      },
      body: JSON.stringify(request.body),
    });

    const contentType = upstream.headers.get("content-type") || "application/json";
    const body = await upstream.text();
    response.status(upstream.status).setHeader("Content-Type", contentType);
    return response.send(body);
  } catch (error) {
    console.error("Backend request failed:", error);
    return response.status(502).json({ detail: "The local AI backend is unavailable." });
  }
}
