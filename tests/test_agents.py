"""Agent tests with a scripted fake chat model: no API keys, no network, deterministic."""

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents import tools
from agents.graphs import build_data_agent, build_search_agent
from agents.server import create_server
from app.config import Settings
from db import Database


class ScriptedModel(GenericFakeChatModel):
    """Replies with the given messages in order. bind_tools is a no-op so agents can use it."""

    def bind_tools(self, tools, **kwargs):
        return self


def scripted(*replies: AIMessage) -> ScriptedModel:
    return ScriptedModel(messages=iter(replies))


def search_call(query: str) -> AIMessage:
    call = {"name": "google_search", "args": {"query": query}, "id": "c1"}
    return AIMessage("", tool_calls=[call])


@pytest.fixture
def fake_serpapi(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Replace the SerpAPI client; returns the list of params it was called with."""
    calls: list[dict] = []

    class FakeGoogleSearch:
        def __init__(self, params: dict) -> None:
            calls.append(params)

        def get_dict(self) -> dict:
            return {
                "answer_box": {"answer": "Paris"},
                "organic_results": [
                    {"title": "Paris - Wikipedia", "link": "https://en.wikipedia.org/wiki/Paris",
                     "snippet": "Paris is the capital of France."},
                ],
            }

    monkeypatch.setenv("SERPAPI_API_KEY", "test-key")
    monkeypatch.setattr("serpapi.GoogleSearch", FakeGoogleSearch)
    return calls


def test_google_search_tool_schema_comes_from_docstring_and_hints() -> None:
    assert tools.google_search.name == "google_search"
    assert "current events" in tools.google_search.description
    args = tools.google_search.args
    assert args["query"]["type"] == "string"
    assert args["num_results"]["default"] == 5


def test_google_search_formats_results(fake_serpapi: list[dict]) -> None:
    out = tools.google_search.invoke({"query": "capital of France", "num_results": 50})
    assert "Answer box: Paris" in out
    assert "https://en.wikipedia.org/wiki/Paris" in out
    assert fake_serpapi[0]["num"] == 10  # clamped


def test_google_search_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)
    assert "SERPAPI_API_KEY is not set" in tools.google_search.invoke({"query": "x"})


def test_search_agent_calls_tool_then_answers(fake_serpapi: list[dict]) -> None:
    agent = build_search_agent(scripted(search_call("capital of France"), AIMessage("Paris.")))
    result = agent.invoke({"messages": [{"role": "user", "content": "Capital of France?"}]})

    kinds = [type(m).__name__ for m in result["messages"]]
    assert kinds == ["HumanMessage", "AIMessage", "ToolMessage", "AIMessage"]
    assert "Paris - Wikipedia" in result["messages"][2].content
    assert result["messages"][-1].content == "Paris."


def test_memory_keeps_the_conversation_per_thread() -> None:
    agent = build_search_agent(
        scripted(AIMessage("Hi Vikas!"), AIMessage("Your name is Vikas.")),
        checkpointer=InMemorySaver(),
    )
    config = {"configurable": {"thread_id": "t1"}}
    agent.invoke({"messages": [{"role": "user", "content": "I am Vikas"}]}, config)
    result = agent.invoke({"messages": [{"role": "user", "content": "What is my name?"}]}, config)
    assert len(result["messages"]) == 4  # both turns are in the thread
    other = agent.get_state({"configurable": {"thread_id": "t2"}}).values
    assert other.get("messages", []) == []


def test_human_in_the_loop_pauses_before_search(fake_serpapi: list[dict]) -> None:
    agent = build_search_agent(
        scripted(search_call("secret query"), AIMessage("Okay, I did not search.")),
        checkpointer=InMemorySaver(),
        require_approval=True,
    )
    config = {"configurable": {"thread_id": "hitl"}}
    paused = agent.invoke({"messages": [{"role": "user", "content": "search it"}]}, config)

    request = paused["__interrupt__"][0].value
    assert request["action_requests"][0]["name"] == "google_search"
    assert fake_serpapi == []  # nothing ran yet

    rejected = {"decisions": [{"type": "reject", "message": "Not allowed."}]}
    result = agent.invoke(Command(resume=rejected), config)
    assert fake_serpapi == []
    tool_messages = [m for m in result["messages"] if isinstance(m, ToolMessage)]
    assert "Not allowed." in tool_messages[0].content
    assert result["messages"][-1].content == "Okay, I did not search."


@pytest.fixture
def data_db(sql_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "_db", lambda: Database(sql_settings.db.conn_str()))


def test_data_agent_answers_from_sql(data_db: None) -> None:
    call = AIMessage("", tool_calls=[{"name": "survival_by_class_and_sex", "args": {}, "id": "d1"}])
    agent = build_data_agent(scripted(call, AIMessage("96.8% of first-class women survived.")))
    result = agent.invoke({"messages": [{"role": "user", "content": "Women in 1st class?"}]})
    tool_output = result["messages"][2].content  # tool results reach the model as JSON
    assert '"survival_rate_pct": 96.8' in tool_output and '"passengers": 94' in tool_output


@pytest.fixture
def server() -> Iterator[TestClient]:
    model = scripted(
        search_call("langserve"), AIMessage("LangServe serves runnables."), AIMessage("Short."),
    )
    yield TestClient(create_server(model, Settings(storage="memory")))


def test_langserve_invoke_endpoint(server: TestClient, fake_serpapi: list[dict]) -> None:
    resp = server.post("/agents/search/invoke", json={"input": {"question": "What is LangServe?"}})
    assert resp.status_code == 200
    assert resp.json()["output"] == "LangServe serves runnables."
    assert fake_serpapi[0]["q"] == "langserve"


def test_langserve_generated_routes_and_schema(server: TestClient) -> None:
    paths = server.get("/openapi.json").json()["paths"]
    for route in ("invoke", "batch", "stream", "stream_events"):
        assert f"/agents/search/{route}" in paths
        assert f"/agents/data/{route}" in paths
        assert f"/agents/summarize/{route}" in paths
    schema = server.get("/agents/search/input_schema").json()
    assert "question" in schema["properties"]


def test_langserve_rejects_bad_input(server: TestClient) -> None:
    assert server.post("/agents/search/invoke", json={"input": {"q": "x"}}).status_code == 422


def test_regular_endpoints_run_alongside_agents(server: TestClient) -> None:
    assert server.get("/health").json()["status"] == "ok"
    created = server.post("/items", json={"name": "Pen", "price": 1})
    assert created.status_code == 201
    agents = server.get("/agents").json()
    assert set(agents) == {"search", "data", "summarize"}


def test_agents_disabled_without_groq_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    client = TestClient(create_server(settings=Settings(storage="memory")))
    assert client.get("/agents").json()["search"]["invoke"] == "disabled"
    assert client.get("/items").status_code == 200
