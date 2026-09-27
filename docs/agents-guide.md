# AI Agent with FastAPI + LangSmith Studio + LangServe — learning guide

This guide explains **how** the agent part of this repo was built, step by step, so you can rebuild it on your own and explain it to others. Each section maps to one item on the course checklist.

**Stack:** LangChain 1.x (`create_agent`), LangGraph 1.x (the runtime underneath), Groq (the LLM), SerpAPI (Google results), LangSmith (tracing and Studio), LangServe (REST endpoints on FastAPI).

---

## The big picture

```
            ┌─────────────── one agent = a LangGraph graph ───────────────┐
 question → │  model node ──(wants a tool?)──yes──► tools node ──┐        │ → answer
            │      ▲                                              │        │
            │      └──────────────── tool result ◄───────────────┘        │
            │      (no more tool calls → finish)                           │
            └──────────────────────────────────────────────────────────────┘
                 ▲ memory (checkpointer, per thread_id)   ▲ optional human approval (interrupt)

 The same graph is used three ways:
   1. agent-chat          → local terminal testing           (src/agents/chat.py)
   2. langgraph dev       → LangSmith Studio: chat + graph   (langgraph.json)
   3. LangServe           → REST API on FastAPI              (src/agents/server.py)
```

An **agent** is a loop. The LLM reads the conversation, then either answers or asks to call a tool. If it asks, the tool runs, its result is added to the conversation, and the LLM goes again. `create_agent` builds that loop as a LangGraph graph with two nodes, `model` and `tools`.

---

## 1. Building a Google Search agent (LangChain + Groq + SerpAPI)

| Piece | Role | Where |
|---|---|---|
| Groq `ChatGroq` | The LLM. Groq runs open models (Llama) very fast, and they support tool calling | [`llm.py`](../src/agents/llm.py) |
| SerpAPI | Returns Google results as JSON, so the agent doesn't scrape anything | [`tools.py`](../src/agents/tools.py) |
| LangChain `create_agent` | Wires the model and tools into the agent loop | [`graphs.py`](../src/agents/graphs.py) |

**Design decision:** the model is created inside a *factory function* (`get_chat_model()`), not at import time. `ChatGroq` fails immediately if `GROQ_API_KEY` is missing. Creating it lazily lets tests import the code with no keys and pass in a fake model instead.

## 2. Setting up a tool with a docstring and decorator

```python
@tool
def google_search(query: str, num_results: int = 5) -> str:
    """Search Google and return the top results with title, link and snippet.

    Use this for current events, recent facts, ...

    Args:
        query: The search query, written like you would type it into Google.
        num_results: How many results to return, from 1 to 10.
    """
```

`@tool` turns the function into something the LLM can call:

| From your code | Becomes | The model uses it to |
|---|---|---|
| function name | tool name `google_search` | refer to the tool |
| docstring | tool **description** | decide **when** to call it |
| type hints + `Args:` | JSON schema for the arguments | decide **what** to pass |

**Key idea to explain to your team:** the docstring is part of the prompt. A vague docstring means the model calls the tool at the wrong times. The test `test_google_search_tool_schema_comes_from_docstring_and_hints` proves this mapping.

Good tool habits used here:
- Return **short, readable text**, not raw JSON with 50 fields. It costs fewer tokens and the model understands it better.
- **Never crash.** A missing key or an API error comes back as a message the model can explain to the user.
- **Clamp inputs** (`num_results` is limited to 1–10), because models sometimes pass silly values.

The **data agent** reuses the same pattern. Its five `@tool` functions wrap the Week 1 SQL queries, so it answers from your own database instead of the web.

## 3. Creating the agent with a system prompt and memory

```python
create_agent(
    model=get_chat_model(),
    tools=[google_search],
    system_prompt=search_system_prompt(),   # role, rules, today's date
    checkpointer=InMemorySaver(),           # memory
)
```

- **System prompt:** it sets the agent's role and rules: when to search, to cite sources, and not to guess. It includes **today's date**, because otherwise the model assumes its training date and makes mistakes with words like "latest" or "this year".
- **Memory:** a *checkpointer* saves the graph's state (the message list) after every step, keyed by `thread_id`. If you send the same `thread_id`, the agent sees the earlier messages. A new `thread_id` starts a fresh conversation. `InMemorySaver` keeps this in RAM, so it's lost on restart. In production you'd use a database-backed checkpointer (Postgres or SQLite).

```python
config = {"configurable": {"thread_id": "vikas-chat-1"}}
agent.invoke({"messages": [{"role": "user", "content": "Who is the CEO of Groq?"}]}, config)
agent.invoke({"messages": [{"role": "user", "content": "How old is he?"}]}, config)  # remembers "he"
```

## 4. Testing the agent locally

Two levels:

**a) Automated tests, with no keys and no internet** ([`tests/test_agents.py`](../tests/test_agents.py))
- A `ScriptedModel` (a fake chat model) returns pre-written replies: first "call google_search", then "final answer". This tests *our* wiring (the tool loop, memory, approval, API routes) in a predictable way that costs nothing.
- `serpapi.GoogleSearch` is replaced with a fake that returns fixed results.

**b) Manual chat in the terminal, with real keys**
```bat
agent-chat search            :: Google search agent
agent-chat search --approve  :: asks you before every search
agent-chat data              :: Titanic analyst over SQL Server
```
The terminal prints each `[tool call]` and `[tool result]`, so you can see how the agent reasons. Type `/new` to start a new thread and check that memory resets.

## 5. Setting up the LangSmith API key

1. Sign in at https://smith.langchain.com, go to **Settings → API Keys**, and create a key.
2. Put it in `.env` (copied from `.env.example`):
   ```
   LANGSMITH_API_KEY=lsv2_...
   LANGSMITH_TRACING=true
   LANGSMITH_PROJECT=week2-search-agent
   ```
3. That's all. LangChain reads these variables and sends a **trace** of every run to LangSmith: each LLM call, tool call, input, output, token count and latency. No code changes are needed. `llm.py` calls `load_dotenv()`, so `.env` is picked up automatically.

`.env` is in `.gitignore`. **Never commit keys.** Commit `.env.example` with empty values instead.

## 6. Creating the `langgraph.json` config file

```json
{
  "dependencies": ["."],
  "graphs": {
    "search_agent": "./src/agents/graphs.py:search_agent",
    "search_agent_with_approval": "./src/agents/graphs.py:search_agent_with_approval",
    "data_agent": "./src/agents/graphs.py:data_agent"
  },
  "env": ".env"
}
```

| Key | Meaning |
|---|---|
| `dependencies` | What to install or put on the path. `"."` means this project (it reads `pyproject.toml`) |
| `graphs` | A name shown in Studio → `file.py:variable_or_function`. These are **factory functions** that return the graph |
| `env` | The file to load environment variables (API keys) from |

**Why the Studio factories have no checkpointer:** the LangGraph server provides its own persistence (threads and memory). If you compile your own `InMemorySaver` into the graph, it clashes with that. So `graphs.py` has builder functions that take `checkpointer=...`, used by the CLI and LangServe, and no-argument wrappers for Studio.

## 7. Deploying to LangSmith Studio with `langgraph dev`

```bat
langgraph dev
```
This starts a local LangGraph API server at `http://127.0.0.1:2024` and opens **LangSmith Studio** in your browser, connected to it (`https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024`). It's a local dev server with hot reload: edit `graphs.py` and Studio picks up the change. Use `langgraph dev --no-browser` if you don't want a browser tab.

If the browser can't connect (Safari and some strict browsers block `localhost` from an HTTPS page), use Chrome or Edge, or run `langgraph dev --tunnel`.

## 8. Chat mode testing in LangSmith Studio

Pick a graph (for example `search_agent`) from the dropdown and switch to **Chat** mode. It works like a chat app: type questions and ask follow-ups on the same thread to test memory, or start a new thread. Every run also appears as a trace in your LangSmith project.

## 9. Graph mode with node-by-node execution tracking

Switch to **Graph** mode. You see the graph drawn as `__start__ → model ⇄ tools → __end__`. When you run it, each node lights up as it executes, and you can click any step to see the **state** at that point: the messages so far, the tool-call arguments the model produced, and the tool output. This is the best way to debug questions like "why did it search for *that*?".

## 10. Adding an interrupt (human-in-the-loop) in LangSmith Studio

There are two ways, and this repo shows both:

**a) In the code:** the `search_agent_with_approval` graph
```python
HumanInTheLoopMiddleware(interrupt_on={"google_search": True})
```
Before `google_search` runs, the graph **pauses** (a LangGraph `interrupt`) and shows the pending tool call. The human replies with a decision, and the run **resumes**:
```json
{"decisions": [{"type": "approve"}]}
{"decisions": [{"type": "edit", "edited_action": {"name": "google_search", "args": {"query": "better query"}}}]}
{"decisions": [{"type": "reject", "message": "Don't search for that."}]}
```
In Graph mode this graph has an extra node, `HumanInTheLoopMiddleware.after_model`, between `model` and `tools`, and that's where the pause happens. In Studio, choose `search_agent_with_approval`, ask something that needs a search, and the run stops with the interrupt shown. Enter the decision and continue. In the terminal, `agent-chat search --approve` does the same thing with y/n prompts, and the test `test_human_in_the_loop_pauses_before_search` checks that nothing runs until you approve.

**b) In Studio only, with no code:** in Graph mode you can set a breakpoint *before* or *after* any node (for example before `tools`). This is handy for stepping through a run while debugging.

Interrupts need a checkpointer, because the graph saves its state when it pauses and continues from that saved state later.

## 11. Installing FastAPI and LangServe

Both are in [`requirements.txt`](../requirements.txt): `pip install -r requirements.txt`. LangServe also needs `sse-starlette` for streaming.

## 12. Creating the FastAPI app with `add_routes` from LangServe

```python
app = create_app()                       # the existing Week 1 FastAPI app
add_routes(app, runnable, path="/agents/search")
```
Anything that is a LangChain **Runnable** can be served this way, whether it's a chain or a compiled LangGraph agent.

We wrap each agent with `as_question_answer()` so the API is simple: it takes `{"question": "..."}` and returns the answer text. The raw agent takes and returns lists of message objects, which are awkward for API clients. `.with_types(input_type=AgentQuestion, output_type=str)` gives LangServe a Pydantic model, so the input is **validated** (422 on bad input) and **documented** in `/docs`.

Memory over HTTP: `per_req_config_modifier=ensure_thread_id` gives each request a new `thread_id` unless the client sends one:
```json
{"input": {"question": "And his age?"}, "config": {"configurable": {"thread_id": "my-chat"}}}
```

## 13. Auto-generated invoke and stream API endpoints

For each `add_routes(..., path=P)`:

| Endpoint | Use |
|---|---|
| `POST P/invoke` | Send one input, get one output |
| `POST P/batch` | Send a list of inputs, get a list of outputs |
| `POST P/stream` | Output arrives in chunks (Server-Sent Events) |
| `POST P/stream_events` | Every internal event: LLM tokens, tool start and end. Use this for a live UI |
| `GET P/input_schema`, `P/output_schema` | JSON schemas |
| `GET P/playground/` | A small web UI to try the endpoint |

```bash
curl -X POST http://127.0.0.1:8001/agents/search/invoke -H "Content-Type: application/json" -d "{\"input\": {\"question\": \"Latest LangChain release?\"}}"
```

## 14. Testing the APIs from Swagger UI

Open http://127.0.0.1:8001/docs. Every generated endpoint is listed with its schema. Click **Try it out** on `/agents/search/invoke`, enter `{"input": {"question": "..."}}`, and click **Execute**.

## 15. Adding multiple agents to the same FastAPI server

Call `add_routes` once per runnable, each with its own `path`:
- `/agents/search`: the Google search agent (tools + memory)
- `/agents/data`: the Titanic data analyst (SQL tools + memory)
- `/agents/summarize`: a plain LCEL chain (`prompt | model | StrOutputParser()`), to show that a simple chain and a full agent are served the same way

Both agents share one `InMemorySaver`. Conversations stay separate because each `thread_id` is its own conversation.

## 16. Running agent APIs alongside regular FastAPI endpoints

`agents/server.py` starts from `create_app()`, the Week 1 app, and adds the agent routes to it. One server on port 8001 serves:
- regular FastAPI routes: `/items` CRUD (SQL Server), `/passengers`, `/stats`, `/health`
- a regular route about the agents: `GET /agents` lists them
- LangServe routes: `/agents/{search,data,summarize}/...`

If `GROQ_API_KEY` is missing, the server **still starts**. The agent routes are just disabled, and `/agents` says so. Regular endpoints never depend on an LLM key.

---

## How this was built (the process, so you can repeat it)

1. **Check the real versions first.** LangChain changes quickly and many tutorials are out of date (for example, the old `AgentExecutor` has been replaced by `create_agent`). After `pip install`, the actual function signatures were read with `inspect.signature(create_agent)`, `add_routes`, and the HITL middleware source, so nothing was guessed.
2. **Test the hard-to-reach parts early.** For example: does `ChatGroq` fail without a key? It does, which is why the model is created lazily in a factory.
3. **Build from the inside out:** tool → agent → local chat → Studio config → API server. Each layer only depends on the ones before it.
4. **Keep one source of truth.** `graphs.py` builds every agent, and the CLI, Studio and LangServe all call it, so there are never three slightly different agents.
5. **Test with a fake model.** The tests check your code (tool loop, memory, interrupts, routes, validation), not the LLM's opinion, so they're fast and free and give the same result every run. Then test manually with real keys for quality.
6. **Reuse what exists.** The data agent's tools are the Week 1 SQL queries, and the API server is the Week 1 app plus new routes.

## Explaining it to your team (2-minute version)

> "We built a Google search agent using LangChain's `create_agent`. It's a loop where a Groq-hosted Llama model decides when to call a `google_search` tool (SerpAPI). The tool is a plain Python function with `@tool`; its docstring tells the model when to use it. A checkpointer gives it per-conversation memory by `thread_id`.
> The same agent runs three ways: in the terminal for quick tests, in LangSmith Studio via `langgraph dev` (chat mode, graph mode with node-by-node state, and a human-approval interrupt before each search), and as a REST API. LangServe's `add_routes` gives us `/invoke`, `/stream` and more on our existing FastAPI app, next to our normal CRUD endpoints.
> Every run is traced in LangSmith. The tests use a fake LLM, so CI needs no keys."

## Common problems

| Symptom | Fix |
|---|---|
| `GroqError: api_key ... must be set` | Put `GROQ_API_KEY` in `.env` in the project root |
| Model says it can't use tools / `tool_use_failed` | Use a Groq model that supports tool calling (set `GROQ_MODEL`) |
| `Search is unavailable: SERPAPI_API_KEY is not set` | Add the SerpAPI key to `.env` |
| `langgraph dev`: *attempted relative import with no known parent package* | Studio loads the graph file by path, so use absolute imports (`from agents.llm import ...`) in that file |
| Studio page won't connect to `127.0.0.1:2024` | Use Chrome or Edge, or run `langgraph dev --tunnel` |
| Agent "forgets" between API calls | Send the same `thread_id` in `config.configurable` |
| No traces in LangSmith | Check `LANGSMITH_TRACING=true` and the key; look in the project named by `LANGSMITH_PROJECT` |
