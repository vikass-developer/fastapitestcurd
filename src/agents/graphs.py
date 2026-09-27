"""Agent factories.

Each function returns a compiled LangGraph graph built with `create_agent`:
model + tools + system prompt (+ optional memory and human approval).

- `langgraph.json` points LangSmith Studio at the no-argument factories at the bottom.
  Studio provides its own memory (checkpointer), so those factories do not pass one.
- `agents.chat` (local testing) and `agents.server` (LangServe) pass an InMemorySaver
  so conversations remember earlier turns per `thread_id`.
"""

from __future__ import annotations

from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.language_models import BaseChatModel
from langgraph.types import Checkpointer

# Absolute imports on purpose: `langgraph dev` loads this file by path (see langgraph.json),
# not as part of the `agents` package, so relative imports like `from .llm` would fail.
from agents.llm import get_chat_model
from agents.tools import DATA_TOOLS, SEARCH_TOOLS


def search_system_prompt() -> str:
    return f"""You are a helpful research assistant with access to Google search.
Today's date is {date.today():%d %B %Y}.

- Use the google_search tool for anything current, factual or specific that you are not sure of.
- You can search more than once, refining the query, before you answer.
- Answer concisely, then list the sources you used as markdown links.
- If the results do not answer the question, say so instead of guessing.
- Use earlier messages in the conversation to resolve follow-ups like "what about his age?"."""


DATA_SYSTEM_PROMPT = """You are a data analyst for the Titanic passenger dataset (891 passengers,
cleaned, stored in SQL Server). Answer only from the tools; never invent numbers.

- Pick the tool whose description matches the question; call several if needed.
- Quote the exact figures from the tool results, and say which table they came from.
- If a question is outside this dataset, say that you can only answer Titanic data questions."""


def build_search_agent(
    model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
    require_approval: bool = False,
):
    """Google search agent. With require_approval, every search pauses for a human decision."""
    middleware = []
    if require_approval:
        # Pauses the graph (a LangGraph interrupt) before google_search runs. The human can
        # approve, edit the query, or reject it from Studio or with Command(resume=...).
        middleware.append(HumanInTheLoopMiddleware(interrupt_on={"google_search": True}))
    return create_agent(
        model=model or get_chat_model(),
        tools=SEARCH_TOOLS,
        system_prompt=search_system_prompt(),
        middleware=middleware,
        checkpointer=checkpointer,
        name="search_agent",
    )


def build_data_agent(model: BaseChatModel | None = None, checkpointer: Checkpointer = None):
    """Titanic analyst that answers from the Week 1 SQL queries."""
    return create_agent(
        model=model or get_chat_model(),
        tools=DATA_TOOLS,
        system_prompt=DATA_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        name="data_agent",
    )


# Entry points for langgraph.json / LangSmith Studio (the platform supplies memory).


def search_agent():
    return build_search_agent()


def search_agent_with_approval():
    return build_search_agent(require_approval=True)


def data_agent():
    return build_data_agent()
