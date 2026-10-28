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
    
from html.parser import HTMLParser
import re
import xml.etree.ElementTree as ET
from urllib.parse import urljoin, urlparse

UNIVERSITY_BASE_URL = "https://uniteduniversity.edu.in/"
UNIVERSITY_DOMAIN = "uniteduniversity.edu.in"


class UniversityHTMLParser(HTMLParser):
    """Extract readable text and links from university webpages."""

    def __init__(self):
        super().__init__()
        self.text_parts = []
        self.links = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)

        if tag in ("script", "style", "noscript", "svg"):
            self.skip_depth += 1

        if tag == "a" and attrs.get("href"):
            self.links.append(attrs["href"])

    def handle_endtag(self, tag):
        if tag in ("script", "style", "noscript", "svg"):
            self.skip_depth = max(0, self.skip_depth - 1)

    def handle_data(self, data):
        if not self.skip_depth and data.strip():
            self.text_parts.append(data.strip())



async def fetch_university_page(client, url):
    print(f"[University] Fetching: {url}")

    response = await client.get(url, follow_redirects=True)
    print(f"[University] HTTP status: {response.status_code}")
    response.raise_for_status()

    final_url = str(response.url)
    parsed = urlparse(final_url)

    if parsed.hostname not in {
        UNIVERSITY_DOMAIN,
        "www." + UNIVERSITY_DOMAIN,
    }:
        print(f"[University] Rejected hostname: {parsed.hostname}")
        return None

    parser = UniversityHTMLParser()
    parser.feed(response.text)

    text = re.sub(r"\s+", " ", " ".join(parser.text_parts))

    print(
        f"[University] HTML size: {len(response.text)} bytes; "
        f"extracted text: {len(text)} characters; "
        f"links: {len(parser.links)}"
    )

    return {
        "url": final_url,
        "text": text[:12000],
        "links": parser.links,
    }



@tool("search_united_university")
async def search_united_university(query: str) -> str:
    print(f"[University Tool] CALLED with query: {query}")
    """Search official United University, Prayagraj webpages for
    university information such as courses, admissions, faculty,
    campus, fees, history, and contact details. Use for university
    questions before consulting Wikipedia."""

    try:
        timeout = httpx.Timeout(15.0, connect=8.0)

        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": "UniversityInfoBot/1.0"},
        ) as client:
            sitemap_urls = [
                urljoin(UNIVERSITY_BASE_URL, "sitemap.xml"),
                urljoin(UNIVERSITY_BASE_URL, "robots.txt"),
            ]


            # Start with the known official course page and homepage.
            discovered_urls = {
                urljoin(UNIVERSITY_BASE_URL, "course-list.aspx"),
                UNIVERSITY_BASE_URL,
            }

            # Discover additional official URLs from the sitemap.
            try:
                sitemap_url = urljoin(
                    UNIVERSITY_BASE_URL, "sitemap.xml"
                )
                sitemap_response = await client.get(sitemap_url)

                if sitemap_response.is_success:
                    root = ET.fromstring(sitemap_response.text)

                    for element in root.iter():
                        if element.tag.endswith("loc") and element.text:
                            candidate = element.text.strip()
                            parsed = urlparse(candidate)

                            if (
                                parsed.scheme == "https"
                                and parsed.hostname in {
                                    UNIVERSITY_DOMAIN,
                                    "www." + UNIVERSITY_DOMAIN,
                                }
                            ):
                                discovered_urls.add(candidate)

            except Exception:
                pass

            # Inspect the homepage once for official navigation links.
            try:
                homepage = await fetch_university_page(
                    client, UNIVERSITY_BASE_URL
                )

                if homepage:
                    for href in homepage["links"]:
                        candidate = urljoin(homepage["url"], href)
                        parsed = urlparse(candidate)

                        if (
                            parsed.scheme == "https"
                            and parsed.hostname in {
                                UNIVERSITY_DOMAIN,
                                "www." + UNIVERSITY_DOMAIN,
                            }
                        ):
                            discovered_urls.add(candidate)

            except Exception:
                pass

            # Prioritize URLs whose paths match the user's question.
            query_words = {
                word.lower()
                for word in re.findall(r"[a-zA-Z0-9]+", query)
                if len(word) > 2
            }

            def relevance(url):
                parsed = urlparse(url)
                url_words = set(
                    re.findall(
                        r"[a-zA-Z0-9]+",
                        (parsed.path + " " + parsed.query).lower(),
                    )
                )

                score = len(query_words & url_words)

                if "course-list.aspx" in parsed.path.lower():
                    score += 10 if any(
                        word in query_words
                        for word in {"course", "courses", "program", "programs"}
                    ) else 0

                return score

            ranked_urls = sorted(
                discovered_urls,
                key=relevance,
                reverse=True,
            )[:3]

            results = []

            results = []

            for url in ranked_urls:
                try:
                    page = await fetch_university_page(client, url)

                    if page and page["text"]:
                        results.append(
                            f"Source: {page['url']}\n"
                            f"Content: {page['text']}"
                        )

                    # Stop after finding useful content on one page.
                    if results:
                        break

                except Exception as exc:
                    print(
                        f"University page fetch failed: {url} "
                        f"({type(exc).__name__}: {exc})"
                    )
                    continue


    except Exception as exc:
        return (
            "Could not access the official university website. "
            f"Reason: {type(exc).__name__}. Do not invent facts."
        )



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
    tools.append(search_united_university)

    print("All agent tools:", [t.name for t in tools])



    model = ChatOllama(
        base_url=OLLAMA_BASE_URL,
        model=OLLAMA_MODEL,
        temperature=0,
        num_ctx=2048,
        num_predict=512,
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

For United University questions:
- Always call search_united_university first.
- Use only information returned by that tool or verified
  official university sources.
- If the tool returns an error, empty content, or cannot
  verify the answer, clearly state that verification failed.
- Never fill missing university information using memory,
  assumptions, or general knowledge.
- Never claim the official website was accessed unless
  the tool actually returned its content.
- Include the official source URL when available.

For questions about United University in Prayagraj, use
search_united_university before answering.

Prioritize information retrieved from the official university
website over Wikipedia for university-specific questions.

Use Wikipedia for other factual and general-knowledge questions.

Use official LangChain MCP tools for LangChain, LangGraph,
Deep Agents, MCP, and LangChain API questions.

Never claim a tool was used unless it was actually called.
If the official university website cannot verify an answer,
say so instead of confusing United University with another
institution.

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
