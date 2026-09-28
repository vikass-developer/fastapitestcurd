"""One FastAPI server for everything: the Week 1 endpoints plus agents served by LangServe.

Run:  uvicorn agents.server:app --app-dir src --port 8001
Docs: http://127.0.0.1:8001/docs

For each `add_routes(app, runnable, path=...)`, LangServe generates:
  POST {path}/invoke        run once, return the output
  POST {path}/batch         run on a list of inputs
  POST {path}/stream        stream output chunks (Server-Sent Events)
  POST {path}/stream_events stream every step: tool calls, tokens and so on
  GET  {path}/input_schema, output_schema, config_schema
  GET  {path}/playground/   a small web UI to try it
"""

from __future__ import annotations

import logging
import sys
import uuid
from typing import Any

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langserve import add_routes
from pydantic import BaseModel, Field

from app.config import Settings
from app.main import create_app

from .graphs import aask, ask, build_data_agent, build_search_agent, final_answer
from .llm import ENV_FILE, get_chat_model, has_key

log = logging.getLogger(__name__)

# LangServe prints a banner with box-drawing characters at startup. On Windows consoles that use
# a legacy code page (cp1252) this raised UnicodeEncodeError and the server exited. Print a
# replacement character instead of crashing.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(errors="replace")


class AgentQuestion(BaseModel):
    """Input for the agent endpoints."""

    question: str = Field(description="What you want to ask the agent", min_length=1)
    thread_id: str | None = Field(
        default=None,
        max_length=100,
        description="Send the thread_id from an earlier answer to continue that conversation. "
        "Leave it out to start a new one.",
    )


class AgentAnswer(BaseModel):
    answer: str
    thread_id: str = Field(description="Send this back to ask a follow-up in the same conversation")


def as_question_answer(agent: Runnable) -> Runnable:
    """Wrap a LangGraph agent so the API takes {"question", "thread_id"} and returns the answer.

    The raw agent takes and returns a list of messages, and its memory is keyed by
    config["configurable"]["thread_id"]. LangServe drops `configurable` keys the runnable does
    not declare, so a thread_id sent in "config" never reached the agent. Taking it in the input
    and setting the config here makes memory work over HTTP.
    """

    def _prepare(payload: Any, config: RunnableConfig) -> tuple[str, RunnableConfig, str]:
        data = payload if isinstance(payload, dict) else payload.model_dump()
        thread_id = data.get("thread_id") or str(uuid.uuid4())
        configurable = {**config.get("configurable", {}), "thread_id": thread_id}
        return data["question"], {**config, "configurable": configurable}, thread_id

    def run(payload: Any, config: RunnableConfig) -> dict:
        question, agent_config, thread_id = _prepare(payload, config)
        state = ask(agent, question, agent_config)
        return {"answer": final_answer(state), "thread_id": thread_id}

    async def arun(payload: Any, config: RunnableConfig) -> dict:
        question, agent_config, thread_id = _prepare(payload, config)
        state = await aask(agent, question, agent_config)
        return {"answer": final_answer(state), "thread_id": thread_id}

    return RunnableLambda(run, afunc=arun, name="agent").with_types(
        input_type=AgentQuestion, output_type=AgentAnswer
    )


class SummarizeInput(BaseModel):
    """Input for the summarize chain."""

    text: str = Field(description="The text to summarize", min_length=1)
    max_words: int = Field(default=30, ge=3, le=300, description="Upper limit for the summary")


def build_summarizer(model: BaseChatModel) -> Runnable:
    """A plain LCEL chain (prompt | model | parser) with no tools and no memory, for comparison."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the user's text in at most {max_words} words. Plain text only."),
            ("user", "{text}"),
        ]
    )
    # Without explicit types LangServe infers every prompt variable as a string, so
    # {"max_words": 10} was rejected with 422. SummarizeInput makes it a validated integer.
    to_dict = RunnableLambda(lambda p: p if isinstance(p, dict) else p.model_dump())
    chain = to_dict | prompt | model | StrOutputParser()
    return chain.with_types(input_type=SummarizeInput, output_type=str)


AGENTS = {
    "search": "Google search agent (Groq + SerpAPI) with conversation memory",
    "data": "Titanic data analyst that answers from the SQL Server analytics queries",
    "summarize": "Plain LCEL chain: summarize text in N words (no tools, no memory)",
}


def create_server(model: BaseChatModel | None = None, settings: Settings | None = None) -> FastAPI:
    # Start from the Week 1 app: items CRUD, passengers, stats and health keep working as-is.
    app = create_app(settings)
    app.title = "AI Agents + Week 1 API"
    app.openapi_tags = [
        *(app.openapi_tags or []),
        {"name": "agents", "description": "LangServe agents"},
    ]

    @app.get("/agents", tags=["agents"], summary="List the agents on this server")
    async def list_agents() -> dict[str, dict[str, str]]:
        enabled = app.state.agents_enabled
        return {
            name: {
                "description": text,
                "invoke": f"/agents/{name}/invoke" if enabled else "disabled",
            }
            for name, text in AGENTS.items()
        }

    if model is None and not has_key("GROQ_API_KEY"):
        log.warning("GROQ_API_KEY is empty: agent routes are disabled. Add it to %s.", ENV_FILE)
        app.state.agents_enabled = False
        return app

    model = model or get_chat_model()
    search_memory = InMemorySaver()
    data_memory = InMemorySaver()

    add_routes(
        app,
        as_question_answer(build_search_agent(model, checkpointer=search_memory)),
        path="/agents/search",
    )
    add_routes(
        app,
        as_question_answer(build_data_agent(model, checkpointer=data_memory)),
        path="/agents/data",
    )
    add_routes(app, build_summarizer(model), path="/agents/summarize")
    app.state.agents_enabled = True
    return app


app = create_server()
