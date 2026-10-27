
import { useState } from "react";

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

function toErrorMessage(response, body) {
  if (typeof body?.detail === "string") return body.detail;
  return `Request failed with status ${response.status}.`;
}

export default function App() {
  const [messages, setMessages] = useState([]);
  const [prompt, setPrompt] = useState("");
  const [error, setError] = useState("");
  const [isSending, setIsSending] = useState(false);

  async function sendMessage(event) {
    event.preventDefault();
    const content = prompt.trim();
    if (!content || isSending) return;

    const nextMessages = [...messages, { role: "user", content }];
    setMessages(nextMessages);
    setPrompt("");
    setError("");
    setIsSending(true);

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ messages: nextMessages }),
      });

      const body = await response.json().catch(() => null);

      if (!response.ok) throw new Error(toErrorMessage(response, body));
      if (typeof body?.reply !== "string") {
        throw new Error("The API returned an invalid response.");
      }

      setMessages((current) => [
        ...current,
        { role: "assistant", content: body.reply },
      ]);
    } catch (requestError) {
      setError(
        requestError instanceof Error
          ? requestError.message
          : "Unable to contact the agent."
      );
    } finally {
      setIsSending(false);
    }
  }

  function printConversation() {
    window.print();
  }

  return (
    <main className="app-shell">
      <header className="app-header">
        <p className="eyebrow">LOCAL AI</p>
        <h1>Ollama Deep Agent Made by Laman Rasheed</h1>
        <p className="subtitle">
          Llama 3.2 · Deep Agents · MCP · LangSmith
        </p>

        {messages.length > 0 && (
          <div className="chat-actions no-print">
            <button
              type="button"
              className="print-button"
              onClick={printConversation}
              disabled={isSending}
            >
              Print / Save as PDF
            </button>
          </div>
        )}
      </header>

      <section className="chat" aria-label="Conversation">
        {messages.length === 0 ? (
          <p className="empty-state no-print">
            Ask about LangChain, Python, or any general topic.
          </p>
        ) : (
          <>
            <div className="print-heading print-only">
              <h2>Chat Conversation</h2>
              <p>Ollama Deep Agent · Laman Rasheed</p>
              <hr />
            </div>

            {messages.map((message, index) => (
              <article
                className={`message ${message.role}`}
                key={`${message.role}-${index}`}
              >
                <strong>{message.role === "user" ? "You" : "Agent"}</strong>
                <p>{message.content}</p>
              </article>
            ))}
          </>
        )}

        {isSending && (
          <p className="thinking no-print" role="status">
            Agent is thinking…
          </p>
        )}
      </section>

      {error && (
        <p className="error no-print" role="alert">
          {error}
        </p>
      )}

      <form className="composer no-print" onSubmit={sendMessage}>
        <label htmlFor="prompt">Ask your Deep Agent</label>
        <div>
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                event.currentTarget.form?.requestSubmit();
              }
            }}
            placeholder="Ask a question…"
            disabled={isSending}
            rows="3"
          />
          <button 
          type="submit" 
          disabled={isSending || !prompt.trim()}
          >
            Send
          </button>
        </div>
      </form>
    </main>
  );
}
