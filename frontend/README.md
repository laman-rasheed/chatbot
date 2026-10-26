# React frontend

This replaces the Streamlit presentation layer. `api.py` remains the FastAPI/Ollama backend.

## Local development

1. Copy `.env.example` to `.env.local` and set `CHAT_API_TOKEN` to the value in the backend `.env`.
   Do not give this variable a `VITE_` prefix.
2. Start FastAPI from the `chatbot` directory:

   ```powershell
   .\.venv\Scripts\uvicorn.exe api:app --host 127.0.0.1 --port 8000
   ```

3. In this directory, install and start the React app:

   ```powershell
   npm install
   npm run dev
   ```

Open `http://127.0.0.1:5173`. Vite sends `/api/chat` to FastAPI and attaches the token on the server, so the browser never receives it.

## Cloudflare deployment

Build the app with `npm run build`. Copy `cloudflare/wrangler.toml.example` to `wrangler.toml`, change `API_ORIGIN` to the public hostname for the FastAPI Tunnel, then run:

```powershell
npx wrangler secret put CHAT_API_TOKEN
npx wrangler deploy
```

The Worker serves the React build and proxies `/api/*` to FastAPI with the token held as a Cloudflare secret. Do not put `CHAT_API_TOKEN` in `VITE_*` variables: Vite bundles those values into browser code.

## Vercel frontend deployment

Vercel can host the React UI, but your Ollama/FastAPI server remains on your computer. Create a **named Cloudflare Tunnel** route such as `https://api.example.com` to `http://127.0.0.1:8000`; do not use a temporary `trycloudflare.com` URL for this.

Deploy this `frontend` folder as the Vercel project root. Before deploying, add these Vercel environment variables for Production and Preview:

| Name | Type | Value |
| --- | --- | --- |
| `BACKEND_API_URL` | Config | `https://api.example.com` |
| `CHAT_API_TOKEN` | Secret | The matching token from the backend `.env` |

`api/chat.js` is a Vercel Function that receives the browser's same-origin `/api/chat` request, adds the secret token, and forwards it to FastAPI. The browser never receives the token.

The public Vercel URL remains online, but messages only work while your computer, Ollama, FastAPI, and the Cloudflare Tunnel are running.
