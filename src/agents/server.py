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
import os
import uuid
from typing import Any

from fastapi import FastAPI, Request
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langserve import add_routes
from pydantic import BaseModel, Field

from app.config import Settings
from app.main import create_app

from .graphs import build_data_agent, build_search_agent
from .llm import get_chat_model

log = logging.getLogger(__name__)


class AgentQuestion(BaseModel):
    """Input for the agent endpoints."""

    question: str = Field(description="What you want to ask the agent", min_length=1)


def as_question_answer(agent: Runnable) -> Runnable:
    """Wrap a LangGraph agent so the API takes {"question": ...} and returns the answer text.

    The raw agent takes and returns a list of messages. This wrapper gives the API a simple,
    well-documented schema instead.
    """

    def to_messages(payload: Any) -> dict:
        question = payload["question"] if isinstance(payload, dict) else payload.question
        return {"messages": [HumanMessage(question)]}

    def final_answer(state: dict) -> str:
        return state["messages"][-1].content

    chain = RunnableLambda(to_messages) | agent | RunnableLambda(final_answer)
    return chain.with_types(input_type=AgentQuestion, output_type=str)


def build_summarizer(model: BaseChatModel) -> Runnable:
    """A plain LCEL chain (prompt | model | parser) with no tools and no memory, for comparison."""
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize the user's text in at most {max_words} words. Plain text only."),
            ("user", "{text}"),
        ]
    )
    return prompt | model | StrOutputParser()


def ensure_thread_id(config: dict, request: Request) -> dict:
    """Memory is keyed by thread_id. Give each request a fresh thread unless the caller sends one.

    Send {"config": {"configurable": {"thread_id": "my-chat"}}} to continue a conversation.
    """
    configurable = config.setdefault("configurable", {})
    configurable.setdefault("thread_id", str(uuid.uuid4()))
    return config


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

    if model is None and not os.getenv("GROQ_API_KEY"):
        log.warning("GROQ_API_KEY is not set: agent routes are disabled. Add it to .env.")
        app.state.agents_enabled = False
        return app

    model = model or get_chat_model()
    memory = InMemorySaver()  # shared by both agents; each thread_id is its own conversation
    agent_routes = {"per_req_config_modifier": ensure_thread_id}

    add_routes(
        app,
        as_question_answer(build_search_agent(model, checkpointer=memory)),
        path="/agents/search",
        **agent_routes,
    )
    add_routes(
        app,
        as_question_answer(build_data_agent(model, checkpointer=memory)),
        path="/agents/data",
        **agent_routes,
    )
    add_routes(app, build_summarizer(model), path="/agents/summarize")
    app.state.agents_enabled = True
    return app


app = create_server()
