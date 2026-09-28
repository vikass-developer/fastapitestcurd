"""Chat model factory and environment setup.

Keys are read from the environment. The project's `.env` file is loaded automatically,
so GROQ_API_KEY, SERPAPI_API_KEY and the LANGSMITH_* tracing settings can live there.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from langchain_core.language_models import BaseChatModel

log = logging.getLogger(__name__)

_PROJECT_ENV = Path(__file__).resolve().parents[2] / ".env"
ENV_FILE = str(_PROJECT_ENV) if _PROJECT_ENV.exists() else find_dotenv(usecwd=True)
load_dotenv(ENV_FILE)

# A Groq model that supports tool calling. Groq retires models over time; list the ones your
# key can use with:
#   python -c "from groq import Groq; print([m.id for m in Groq().models.list().data])"
DEFAULT_GROQ_MODEL = "openai/gpt-oss-120b"

KEY_URLS = {
    "GROQ_API_KEY": "https://console.groq.com/keys",
    "SERPAPI_API_KEY": "https://serpapi.com/manage-api-key",
    "LANGSMITH_API_KEY": "https://smith.langchain.com (Settings -> API Keys)",
}


class MissingKeyError(RuntimeError):
    """An API key is missing or empty."""


def has_key(name: str) -> bool:
    return bool(os.getenv(name, "").strip())


def require_key(name: str) -> None:
    if not has_key(name):
        where = ENV_FILE or "a .env file in the project folder"
        raise MissingKeyError(
            f"{name} is empty. Put it in {where} (get one at {KEY_URLS[name]}), save the file, "
            "and start again."
        )


def _disable_tracing_without_key() -> None:
    # Tracing on + no key = a 401 error on every run. Turn it off with one clear warning instead.
    flags = ("LANGSMITH_TRACING", "LANGCHAIN_TRACING_V2")
    tracing_on = any(os.getenv(flag, "").strip().lower() == "true" for flag in flags)
    if tracing_on and not (has_key("LANGSMITH_API_KEY") or has_key("LANGCHAIN_API_KEY")):
        for flag in flags:
            os.environ[flag] = "false"
        log.warning("LangSmith tracing is off: LANGSMITH_API_KEY is empty in %s.", ENV_FILE)


_disable_tracing_without_key()


def get_chat_model(temperature: float = 0) -> BaseChatModel:
    """Return the Groq chat model. Created lazily because ChatGroq needs GROQ_API_KEY."""
    require_key("GROQ_API_KEY")
    from langchain_groq import ChatGroq

    return ChatGroq(model=os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL), temperature=temperature)
