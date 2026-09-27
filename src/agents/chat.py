"""Chat with an agent in the terminal, to test it locally before Studio or the API.

Usage:  python -m agents.chat [search|data] [--approve]

Memory: every message goes to the same thread_id, so the agent remembers the conversation.
Type /new to start a fresh thread, or /quit to exit. With --approve (search agent only),
each Google search waits for you to approve or reject it.
"""

from __future__ import annotations

import argparse
import uuid

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from .graphs import build_data_agent, build_search_agent


def _print_new_messages(messages: list, already_seen: int) -> None:
    for msg in messages[already_seen:]:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  [tool call] {call['name']}({call['args']})")
        elif isinstance(msg, ToolMessage):
            preview = str(msg.content).replace("\n", " ")
            print(f"  [tool result] {preview[:160]}{'...' if len(preview) > 160 else ''}")
        elif isinstance(msg, AIMessage) and msg.content:
            print(f"\nAgent: {msg.content}\n")


def _ask_approval(interrupts) -> dict:
    """Show the pending tool calls and collect a decision for each one."""
    decisions = []
    for action in interrupts[0].value["action_requests"]:
        answer = input(f"  Approve {action['name']}({action['args']})? [y/n] ").strip().lower()
        decisions.append({"type": "approve"} if answer.startswith("y") else {
            "type": "reject", "message": "The user rejected this search."
        })
    return {"decisions": decisions}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chat with an agent locally.")
    parser.add_argument("agent", nargs="?", choices=["search", "data"], default="search")
    parser.add_argument("--approve", action="store_true", help="Approve each search by hand")
    args = parser.parse_args(argv)

    memory = InMemorySaver()
    if args.agent == "search":
        agent = build_search_agent(checkpointer=memory, require_approval=args.approve)
    else:
        agent = build_data_agent(checkpointer=memory)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    print(f"{args.agent} agent ready. /new = new conversation, /quit = exit.\n")

    while True:
        try:
            text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if text == "/quit":
            break
        if text == "/new":
            config = {"configurable": {"thread_id": str(uuid.uuid4())}}
            print("(new conversation)\n")
            continue
        if not text:
            continue

        seen = len(agent.get_state(config).values.get("messages", []))
        result = agent.invoke({"messages": [{"role": "user", "content": text}]}, config)
        while "__interrupt__" in result:  # human-in-the-loop pause
            _print_new_messages(result["messages"], seen)
            seen = len(result["messages"])
            result = agent.invoke(Command(resume=_ask_approval(result["__interrupt__"])), config)
        _print_new_messages(result["messages"], seen)


if __name__ == "__main__":
    main()
