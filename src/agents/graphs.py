"""Agent factories.

Each function returns a compiled LangGraph graph built with `create_agent`:
model + tools + system prompt (+ optional memory and human approval).

- `langgraph.json` points LangSmith Studio at the no-argument factories at the bottom.
  Studio provides its own memory (checkpointer), so those factories do not pass one.
- `agents.chat` (local testing) and `agents.server` (LangServe) pass an InMemorySaver
  so conversations remember earlier turns per `thread_id`.
"""

from __future__ import annotations

import re
from datetime import date

from langchain.agents import create_agent
from langchain.agents.middleware import (
    HumanInTheLoopMiddleware,
    ModelRetryMiddleware,
    ToolCallLimitMiddleware,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.base import BaseCheckpointSaver as Checkpointer

# Absolute imports on purpose: `langgraph dev` loads this file by path (see langgraph.json),
# not as part of the `agents` package, so relative imports like `from .llm` would fail.
from agents.llm import get_chat_model
from agents.tools import DATA_TOOLS, SEARCH_TOOLS

# Without a cap the model sometimes ran 9 searches for one simple question (about 3 minutes).
MAX_SEARCHES_PER_QUESTION = 3

# gpt-oss models add markers like 【3†link】 even when told not to. Strip them from answers.
_CITATION_MARKER = re.compile(r"\s*【[^】]*】")


def clean_answer(text: str) -> str:
    return _CITATION_MARKER.sub("", text)


# gpt-oss sometimes ends with an empty message (the answer stayed in its hidden reasoning).
EMPTY_ANSWER_NUDGE = "Please write your final answer now, using what you have found."


def ask(agent, question: str, config: dict | None = None) -> dict:
    """Invoke an agent with one question; if it ends with an empty answer, nudge it once.

    The nudge continues the same thread, so it needs the agent to have a checkpointer.
    Returns the final state; an interrupted (human-in-the-loop) state is returned unchanged.
    """
    state = agent.invoke({"messages": [HumanMessage(question)]}, config)
    if "__interrupt__" not in state and not final_answer(state):
        state = agent.invoke({"messages": [HumanMessage(EMPTY_ANSWER_NUDGE)]}, config)
    return state


async def aask(agent, question: str, config: dict | None = None) -> dict:
    state = await agent.ainvoke({"messages": [HumanMessage(question)]}, config)
    if "__interrupt__" not in state and not final_answer(state):
        state = await agent.ainvoke({"messages": [HumanMessage(EMPTY_ANSWER_NUDGE)]}, config)
    return state


def final_answer(state: dict) -> str:
    return clean_answer(str(state["messages"][-1].content)).strip()


def _invented_tool_call(exc: Exception) -> bool:
    # gpt-oss was trained with built-in browser tools and sometimes calls one (e.g. "open_file")
    # that we never gave it. Groq rejects that with 400 "tool_use_failed"; a retry usually works.
    return "tool_use_failed" in str(exc)


def retry_invented_tool_calls() -> ModelRetryMiddleware:
    return ModelRetryMiddleware(
        max_retries=3,
        retry_on=_invented_tool_call,
        initial_delay=0.5,
        max_delay=2.0,
        on_failure=lambda exc: "Sorry, I could not produce a valid answer. Please ask again.",
    )


def search_system_prompt() -> str:
    return f"""You are a helpful research assistant with access to Google search.
Today's date is {date.today():%d %B %Y}.

- Your training data is older than today's date. For anything "latest", "current" or "most
  recent", trust the newest dated information in the search results over what you remember.
- Never put a year you remember into a search query for "latest"/"most recent" questions. Leave
  the year out, or use the current year ({date.today():%Y}).
- Your only tool is google_search. You cannot open, browse or read web pages or files: there is
  no open, open_file or find tool. Answer from the search snippets.
- Use the google_search tool for anything current, factual or specific that you are not sure of.
- One well-written search is usually enough. Search again only if the first results clearly
  do not answer the question, and never more than {MAX_SEARCHES_PER_QUESTION} times per question.
- Answer concisely, then list the sources you used as markdown links.
- Do not add citation markers such as 【1†source】 inside the text; use the source list only.
- If the results do not answer the question, say so instead of guessing.
- Use earlier messages in the conversation to resolve follow-ups like "what about his age?"."""


DATA_SYSTEM_PROMPT = """You are a data analyst for the Titanic passenger dataset (891 passengers,
cleaned, stored in SQL Server). Answer only from the tools; never invent numbers.

- Pick the tool whose description matches the question; call several if needed.
- Quote the exact figures from the tool results, and name the tool they came from in plain words
  (no citation markers such as 【survival_by_age_group】).
- If a question is outside this dataset, say that you can only answer Titanic data questions."""


def build_search_agent(
    model: BaseChatModel | None = None,
    checkpointer: Checkpointer = None,
    require_approval: bool = False,
):
    """Google search agent. With require_approval, every search pauses for a human decision."""
    if require_approval:
        # Pauses the graph (a LangGraph interrupt) before google_search runs. The human can
        # approve, edit the query, or reject it from Studio or with Command(resume=...).
        # The human is the limit here: ToolCallLimitMiddleware's per-run counter is not saved
        # across a pause (it is an UntrackedValue), so it resets on every resume and cannot cap
        # searches in this graph.
        middleware = [HumanInTheLoopMiddleware(interrupt_on={"google_search": True})]
    else:
        # Hard limit that backs up the prompt. Extra calls are blocked with a message telling
        # the model to answer with what it already has.
        middleware = [
            ToolCallLimitMiddleware(
                tool_name="google_search",
                run_limit=MAX_SEARCHES_PER_QUESTION,
                exit_behavior="continue",
            )
        ]
    return create_agent(
        model=model or get_chat_model(),
        tools=SEARCH_TOOLS,
        system_prompt=search_system_prompt(),
        middleware=[*middleware, retry_invented_tool_calls()],
        checkpointer=checkpointer,
        name="search_agent",
    )


def build_data_agent(model: BaseChatModel | None = None, checkpointer: Checkpointer = None):
    """Titanic analyst that answers from the Week 1 SQL queries."""
    return create_agent(
        model=model or get_chat_model(),
        tools=DATA_TOOLS,
        system_prompt=DATA_SYSTEM_PROMPT,
        middleware=[retry_invented_tool_calls()],
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
