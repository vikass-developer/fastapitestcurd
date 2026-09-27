"""Chat model factory.

Keys are read from the environment. A `.env` file in the project root is loaded automatically,
so GROQ_API_KEY, SERPAPI_API_KEY and the LANGSMITH_* tracing settings can live there.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_core.language_models import BaseChatModel

load_dotenv()

DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # a Groq model that supports tool calling


def get_chat_model(temperature: float = 0) -> BaseChatModel:
    """Return the Groq chat model. Created lazily because ChatGroq needs GROQ_API_KEY."""
    from langchain_groq import ChatGroq

    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), temperature=temperature)
