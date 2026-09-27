"""Tools the agents can call.

The @tool decorator turns a plain function into a LangChain tool:
- the function name becomes the tool name the model sees,
- the docstring becomes the description the model reads to decide *when* to call it,
- the type hints (and the Args section) become the JSON schema for the arguments.
So the docstring is part of the prompt: write it for the model, not only for humans.
"""

from __future__ import annotations

import os
from functools import cache

from langchain_core.tools import tool

from db import Database, DbSettings, queries

# ---------------------------------------------------------------- Google search (SerpAPI)


@tool
def google_search(query: str, num_results: int = 5) -> str:
    """Search Google and return the top results with title, link and snippet.

    Use this for current events, recent facts, prices, people, or anything that may have
    changed after your training data. Do not use it for math or general knowledge you
    already know well.

    Args:
        query: The search query, written like you would type it into Google.
        num_results: How many results to return, from 1 to 10.
    """
    from serpapi import GoogleSearch

    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return "Search is unavailable: SERPAPI_API_KEY is not set."

    num_results = max(1, min(num_results, 10))
    data = GoogleSearch(
        {"engine": "google", "q": query, "num": num_results, "api_key": api_key}
    ).get_dict()
    if "error" in data:
        return f"Search failed: {data['error']}"

    lines: list[str] = []
    if box := data.get("answer_box"):
        answer = box.get("answer") or box.get("snippet") or box.get("result")
        if answer:
            lines.append(f"Answer box: {answer}")
    for result in data.get("organic_results", [])[:num_results]:
        title, link, snippet = result.get("title"), result.get("link"), result.get("snippet", "")
        lines.append(f"- {title}\n  {link}\n  {snippet}")
    return "\n".join(lines) or "No results found."


# ---------------------------------------------------------------- Titanic data (SQL Server)
# These reuse the Week 1 analytics queries, so the data agent answers from our own database.


@cache
def _db() -> Database:
    return Database(DbSettings.from_env().conn_str())


@tool
def survival_by_class_and_sex() -> list[dict]:
    """Titanic survival counts and rates for each ticket class (1-3) and sex."""
    return _db().fetch_all(queries.SURVIVAL_BY_CLASS_AND_SEX)


@tool
def survival_by_age_group() -> list[dict]:
    """Titanic survival counts and rates by age group (child, teen, adult, middle-aged, senior)."""
    return _db().fetch_all(queries.SURVIVAL_BY_AGE_GROUP)


@tool
def oldest_passengers_per_class(top_n: int = 3) -> list[dict]:
    """The oldest Titanic passengers in each ticket class, with whether they survived.

    Args:
        top_n: How many passengers to return per class, from 1 to 20.
    """
    return _db().fetch_all(queries.OLDEST_PER_CLASS, max(1, min(top_n, 20)))


@tool
def survival_by_family_size(min_passengers: int = 5) -> list[dict]:
    """Titanic survival rates by family size (passenger + siblings, spouses, parents, children).

    Args:
        min_passengers: Hide family sizes with fewer passengers than this.
    """
    return _db().fetch_all(queries.SURVIVAL_BY_FAMILY_SIZE, max(1, min_passengers))


@tool
def embarkation_summary() -> list[dict]:
    """Titanic passengers per port of embarkation, with average fare, survival rate and share."""
    return _db().fetch_all(queries.EMBARKED_SUMMARY)


SEARCH_TOOLS = [google_search]
DATA_TOOLS = [
    survival_by_class_and_sex,
    survival_by_age_group,
    oldest_passengers_per_class,
    survival_by_family_size,
    embarkation_summary,
]
