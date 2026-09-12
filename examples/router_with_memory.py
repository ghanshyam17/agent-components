"""Reference composition: agentic-router + memory-store.

Run after `uv sync --all-packages`:

    uv run python examples/router_with_memory.py

Demonstrates composing the two components: the router uses a memory-store
backend for sessions (in-memory here; swap build_session_store(MemorySettings(
backend="redis")) for persistence) and a long-term vector memory that the
agent could recall from. No real vLLM is required for the session part; the
agent loop itself needs models, so this example only exercises routing +
memory wiring.
"""
from __future__ import annotations

import asyncio

from agentic_router.factory import build_agent
from memory_store import (
    InMemoryVectorMemory, MemoryQuery, MemoryRecord, build_session_store,
)
from memory_store.embeddings import HashingEmbedder
from memory_store.config import MemorySettings


async def main() -> None:
    # Short-term memory: any SessionStore implementation; Redis for persistence.
    sessions = build_session_store(MemorySettings(backend="memory"))

    # Long-term memory: deterministic embedder for the demo (use OpenAIEmbedder
    # against your vLLM embeddings endpoint in real use).
    vm = InMemoryVectorMemory(HashingEmbedder(dim=128))
    await vm.add([
        MemoryRecord(content="The router scores each task and picks lower or higher model."),
        MemoryRecord(content="Sessions persist messages and an optional plan by session id."),
    ])

    # The router injects the chosen session backend.
    agent = build_agent(sessions=sessions)

    # Routing is independent of models — show a decision without hitting vLLM.
    for task in ("hi there", "debug and refactor the function"):
        decision = await agent.router.route(task)
        print(f"[route] {task!r:45} -> {decision.tier.value:6} via {decision.method}")

    # Long-term recall.
    hits = await vm.search(MemoryQuery(query="how are tasks routed to models?", top_k=1))
    print("[recall]", f"{hits[0][1]:.2f}", hits[0][0].content)


if __name__ == "__main__":
    asyncio.run(main())