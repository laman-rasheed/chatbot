import secrets
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import httpx
import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
from langsmith import traceable


# ---------------------------------------------------------
# ENVIRONMENT SETTINGS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env", override=False)

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my first app"

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://127.0.0.1:11434",
)
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")
CHAT_API_TOKEN = os.getenv("CHAT_API_TOKEN", "")

if not CHAT_API_TOKEN:
    raise RuntimeError(
        "CHAT_API_TOKEN is missing. Add it to your .env file."
    )

FRONTEND_ORIGIN = os.getenv(
    "FRONTEND_ORIGIN",
    "http://localhost:5173",
)

MCP_SERVERS = {
    "docs-langchain": {
        "transport": "http",
        "url": "https://docs.langchain.com/mcp",
    },
    "reference-langchain": {
        "transport": "http",
        "url": "https://reference.langchain.com/mcp",
    },
}


# ---------------------------------------------------------
# WIKIPEDIA TOOL
# ---------------------------------------------------------

wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for relevant facts and background."""
    try:
        result = wiki_tool.run(query)
        if not result or not str(result).strip():
            return "No relevant Wikipedia results were found."
        return str(result)
    except Exception:
        return "Wikipedia lookup failed. Please try again."


# ---------------------------------------------------------
# BUILD DEEP AGENT
# ---------------------------------------------------------

async def build_deep_agent():
    client = MultiServerMCPClient(MCP_SERVERS)
    tools = await client.get_tools()
    print("MCP tools:", [t.name for t in tools])

    print(
        "get_symbol available:",
        any(t.name == "get_symbol" for t in tools)
    )

    # Preserve the custom get_symbol validation from your app.
    symbol_tool = next(
        (item for item in tools if item.name == "get_symbol"),
        None,
    )

    if symbol_tool is not None:

        @tool("get_symbol")
        async def get_symbol(
            package: str,
            symbol: str,
            language: str = "python",
        ) -> str:
            """Look up a known API symbol in a package."""

            if not package.strip() or not symbol.strip():
                return "Both package and symbol are required."

            result = await symbol_tool.ainvoke({
                "package": package,
                "symbol": symbol,
                "language": language,
            })

            if isinstance(result, str):
                return result

            if isinstance(result, list):
                return "\n".join(
                    block["text"]
                    for block in result
                    if isinstance(block, dict)
                    and block.get("type") == "text"
                    and isinstance(block.get("text"), str)
                )

            return str(result)

        tools = [
            get_symbol if item.name == "get_symbol" else item
            for item in tools
        ]

    tools.append(search_wikipedia)

    model = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0,
    )

    return create_deep_agent(
        model=model,
        tools=tools,
        system_prompt="""
You are a helpful AI assistant powered by Llama 3.2
running locally through Ollama.

Use Wikipedia for factual questions, definitions, topic
summaries, and general background. Use official LangChain
MCP tools for LangChain documentation.

Use MCP tools for questions about LangChain, LangGraph,
Deep Agents, MCP, LangChain APIs, and Python LangChain code.

Prefer official documentation when available. Do not invent
documentation or APIs. Give clear and accurate answers.
If you are uncertain, say so.

For simple greetings and general questions that do not require tools,
answer directly. Do not delegate these questions to a subagent.
""",
    )


# ---------------------------------------------------------
# INITIALIZE AGENT WHEN API STARTS
# ---------------------------------------------------------

agent = None
agent_error = None



from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent, agent_error

    try:
        agent = await build_deep_agent()
        agent_error = None
        print("Deep Agent initialized successfully.")
    except Exception as exc:
        agent = None
        agent_error = str(exc)
        print(f"Deep Agent initialization failed: {exc}")

    yield

    print("Shutting down Ollama Deep Agent API.")


app = FastAPI(
    title="Ollama 3.2 Deep Agent API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_ORIGIN],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Token"],
)


# ---------------------------------------------------------
# CHAT REQUEST FORMAT
# ---------------------------------------------------------

class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=10000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=40)


# ---------------------------------------------------------
# HEALTH CHECK
# ---------------------------------------------------------

@app.get("/health")
async def health():
    ollama_online = False

    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(
                f"{OLLAMA_BASE_URL.rstrip('/')}/api/tags"
            )
            ollama_online = response.is_success
    except httpx.HTTPError:
        ollama_online = False

    return {
        "api": "online",
        "ollama": "online" if ollama_online else "offline",
        "model": OLLAMA_MODEL,
        "agent_ready": agent is not None,
    }


# ---------------------------------------------------------
# RUN AGENT WITH LANGSMITH TRACING
# ---------------------------------------------------------



@traceable(name="website_deep_agent", project_name="my first app")
async def run_agent(messages):
    result = await agent.ainvoke({"messages": messages})

    result_messages = result.get("messages", [])

    # Search backward for the final assistant response.
    for message in reversed(result_messages):
        if not isinstance(message, AIMessage):
            continue

        # Skip messages that are requesting tool calls.
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            continue

        content = getattr(message, "content", "")

        # Return plain text only.
        if isinstance(content, str) and content.strip():
            return content.strip()

        # Handle content represented as a list of blocks.
        if isinstance(content, list):
            text_parts = [
                block["text"]
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "text"
                and isinstance(block.get("text"), str)
            ]
            if text_parts:
                return "\n".join(text_parts).strip()

    return "I couldn't generate a final text response. Please try again."




# ---------------------------------------------------------
# CHAT ENDPOINT
# ---------------------------------------------------------

@app.post("/chat")
async def chat(
    request: ChatRequest,
    api_token: str | None = Header(default=None, alias="X-API-Token"),
):
    if not api_token or not secrets.compare_digest(
        api_token, CHAT_API_TOKEN
    ):
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API token.",
        )

    if agent is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "The agent is not ready. "
                f"{agent_error or 'Try again in a moment.'}"
            ),
        )

    messages = [
        HumanMessage(content=message.content)
        if message.role == "user"
        else AIMessage(content=message.content)
        for message in request.messages[-20:]
    ]

    try:
        reply = await run_agent(messages)
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="The agent could not complete the request.",
        ) from exc

    return {"reply": reply}


if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host=os.getenv("API_HOST", "127.0.0.1"),
        port=int(os.getenv("API_PORT", "8000")),
    )
