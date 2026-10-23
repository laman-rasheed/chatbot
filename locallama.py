import traceback
import requests
import asyncio
import os
from pathlib import Path
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent
from langsmith import traceable
from langchain_core.messages import HumanMessage, AIMessage

# =========================================================
# LOAD .ENV
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE, override=True)
from langsmith import traceable


@traceable(
    name="STREAMLIT_LIVE_TEST",
    project_name="my first app",
)
def streamlit_live_test():

    print("STREAMLIT TRACE STARTED")

    time.sleep(15)

    print("STREAMLIT TRACE FINISHED")

    return "Streamlit LangSmith tracing is working!"

# =========================================================
# FORCE LANGSMITH PROJECT
# =========================================================
# This makes sure the Streamlit app uses the same project
# that you want in LangSmith.

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = "my first app"


# =========================================================
# STREAMLIT CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Ollama 3.2 Deep Agent",
    page_icon="🤖",
    layout="wide",
)


# =========================================================
# TITLE
# =========================================================

st.title("🤖 Ollama 3.2 Deep Agent Made by Laman Rasheed")

st.caption(
    "Llama 3.2 + Ollama + Deep Agents + MCP + LangSmith"
)


# =========================================================
# LANGSMITH STATUS
# =========================================================

with st.sidebar:

    st.header("🔎 System Status")

    tracing_status = os.getenv(
        "LANGSMITH_TRACING"
    )

    project_name = os.getenv(
        "LANGSMITH_PROJECT"
    )

    api_key_loaded = bool(
        os.getenv("LANGSMITH_API_KEY")
    )

    st.write(
        "🟢 LangSmith Tracing:",
        tracing_status
    )

    st.write(
        "📁 Project:",
        project_name
    )

    if api_key_loaded:
        st.write(
            "🔑 API Key:",
            "✅ Loaded"
        )
    else:
        st.write(
            "🔑 API Key:",
            "❌ Missing"
        )

    st.divider()

    st.write("🤖 Model: `llama3.2`")
    st.write("🧠 Agent: `Deep Agent`")
    st.write("🔌 MCP: `Enabled`")
    st.write("📚 Wikipedia search: ✅ Enabled")
    st.caption("Wikipedia will be used as the knowledge source.")


# =========================================================
# CHAT HISTORY
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


# =========================================================
# DISPLAY OLD MESSAGES
# =========================================================

for message in st.session_state.messages:

    if isinstance(message, dict):
        role = message["role"]
        content = message["content"]

    elif isinstance(message, HumanMessage):
        role = "user"
        content = message.content

    elif isinstance(message, AIMessage):
        role = "assistant"
        content = message.content

    else:
        continue

    with st.chat_message(role):
        st.markdown(content)


# =========================================================
# MCP SERVER CONFIGURATION
# =========================================================

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


# =========================================================
# WIKIPEDIA SEARCH TOOL
# =========================================================

wiki_api = WikipediaAPIWrapper(top_k_results=3, lang="en")
wiki_tool = WikipediaQueryRun(api_wrapper=wiki_api)


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for relevant facts to answer the user's question."""

    try:
        result = wiki_tool.run(query)
        if not result or not str(result).strip():
            return "No relevant Wikipedia results were found for that query."
        return str(result)
    except Exception as exc:
        return f"Wikipedia lookup failed: {exc}"


# =========================================================
# BUILD DEEP AGENT
# =========================================================

async def build_deep_agent():

    # -----------------------------------------------------
    # MCP CLIENT
    # -----------------------------------------------------

    client = MultiServerMCPClient(
        MCP_SERVERS
    )

    # -----------------------------------------------------
    # GET MCP TOOLS
    # -----------------------------------------------------

    tools = await client.get_tools()
    print("MCP tools:", [t.name for t in tools])

    print(
        "get_symbol available:",
        any(t.name == "get_symbol" for t in tools)
    )

    symbol_tool = next(
        (mcp_tool for mcp_tool in tools if mcp_tool.name == "get_symbol"),
        None,
    )

    if symbol_tool is not None:

        @tool("get_symbol")
        async def get_symbol(
            package: str,
            symbol: str,
            language: str = "python",
        ) -> str:
            """Look up an API symbol when its package and symbol are known."""

            if not package.strip() or not symbol.strip():
                return (
                    "Both package and symbol are required. "
                    "Use search_api to identify them first."
                )

            result = await symbol_tool.ainvoke({
                "package": package,
                "symbol": symbol,
                "language": language,
            })

            # Handle unexpected MCP responses safely
            if result is None:
                return (
                    "The MCP symbol lookup returned no result. "
                    "Try searching the official documentation first."
                )

            if isinstance(result, str):
                return result

            if isinstance(result, dict):
                text_value = result.get("text")
                if isinstance(text_value, str):
                    return text_value

                return str(result)

            if isinstance(result, list):
                text_blocks = []

                for block in result:
                    if isinstance(block, dict):
                        if (
                            block.get("type") == "text"
                            and isinstance(block.get("text"), str)
                        ):
                            text_blocks.append(block["text"])

                    else:
                        block_text = getattr(block, "text", None)

                        if isinstance(block_text, str):
                            text_blocks.append(block_text)

                return "\n".join(text_blocks) or str(result)

            return str(result)

        tools = [
            get_symbol if mcp_tool.name == "get_symbol" else mcp_tool
            for mcp_tool in tools
        ]


    tools.append(search_wikipedia)

    # -----------------------------------------------------
    # OLLAMA MODEL
    # -----------------------------------------------------

    model = ChatOllama(

        model="llama3.2",

        temperature=0,
    )

    # -----------------------------------------------------
    # DEEP AGENT
    # -----------------------------------------------------

    agent = create_deep_agent(

        model=model,

        tools=tools,

        system_prompt="""

You are a helpful AI assistant powered by
Llama 3.2 running locally through Ollama.

You are a Deep Agent with access to
LangChain documentation through MCP and
Wikipedia search for general knowledge.

Use the Wikipedia search tool for factual questions,
definitions, person/topic summaries, and general background.
Use LangChain MCP tools for LangChain documentation.

Use MCP tools when the user asks about:

- LangChain
- LangGraph
- Deep Agents
- MCP
- LangChain APIs
- LangChain documentation
- Python LangChain code

Prefer official LangChain documentation
when available.

For code-writing requests, provide the code directly unless a specific
LangChain API needs verification. The get_symbol tool requires both a
non-empty package and symbol; use search_api first if either is unknown.

Give clear and accurate answers.

For complex tasks, break the task into
smaller steps and use the available tools.

Do not make up documentation or APIs.

If you don't know something, say so clearly.

""",
    )

    return agent


# =========================================================
# LANGSMITH-TRACED AGENT FUNCTION
# =========================================================

@traceable(
    name="streamlit_deep_agent",
    project_name="my first app",
)
def run_deep_agent(messages):
    """Trace the complete Deep Agent execution."""

    async def _run_agent():
        # Build the agent
        agent = await build_deep_agent()

        if agent is None:
            raise RuntimeError(
                "build_deep_agent() returned None."
            )

        # Run the agent
        result = await agent.ainvoke({
            "messages": messages
        })

        print("Agent result:", repr(result))

        if result is None:
            raise RuntimeError(
                "The Deep Agent returned None."
            )

        if not isinstance(result, dict):
            raise TypeError(
                f"Unexpected agent result: {type(result).__name__}"
            )

        response_messages = result.get("messages", [])

        if not response_messages:
            raise RuntimeError(
                "The agent returned no messages."
            )

        last_message = response_messages[-1]
        content = getattr(last_message, "content", None)

        if content is None:
            raise RuntimeError(
                f"Agent message has no content: {last_message!r}"
            )

        return content

    return asyncio.run(_run_agent())


# =========================================================
# CHAT INPUT
# =========================================================

user_input = st.chat_input(
    "Ask your Deep Agent something..."
)

if st.button("🧪 Test Live LangSmith Trace"):
    with st.spinner("Creating LangSmith trace..."):
        try:
            model = ChatOllama(model="llama3.2", temperature=0)
            result = model.invoke(
                "Test LangSmith tracing. Reply with: Trace successful!"
            )
            st.success(result.content)
        except Exception as e:
            st.error(f"Error while running the Deep Agent: {e}")
            traceback.print_exc()


# =========================================================
# PROCESS USER MESSAGE
# =========================================================

if user_input:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append(

        {
            "role": "user",
            "content": user_input,
        }

    )

    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message("user"):

        st.markdown(
            user_input
        )

    # -----------------------------------------------------
    # DISPLAY ASSISTANT
    # -----------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "🤔 Deep Agent is thinking..."
        ):

            try:

                # -----------------------------------------
                # RUN AGENT
                # -----------------------------------------
                api_url = os.getenv(
                    "BACKEND_API_URL",
                    "http://127.0.0.1:8000"
                )

                api_token = os.getenv("CHAT_API_TOKEN", "")

                if not api_token:
                    try:
                        api_token = st.secrets["CHAT_API_TOKEN"]
                    except Exception:
                        api_token = ""
                api_response = requests.post(
                    f"{api_url.rstrip('/')}/chat",
                    json={"messages": st.session_state.messages},
                    headers={"X-API-Token": api_token},
                    timeout=180,
                )

                api_response.raise_for_status()
                response = api_response.json()["reply"]
                # -----------------------------------------
                # DISPLAY RESPONSE
                # -----------------------------------------

                st.markdown(
                    response
                )

            except Exception as e:

                response = (
                    f"❌ Error while running "
                    f"the Deep Agent:\n\n{e}"
                )

                st.error(
                    response
                )

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response,
        }
    )