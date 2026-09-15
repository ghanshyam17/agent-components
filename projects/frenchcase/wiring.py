"""Wire the chassis agent stack on top of FrenchCase's real engines.

This is the composition root: it builds an `agentic_router` Agent whose
collaborators are FrenchCase's own infrastructure, via the bridges.

    ┌─────────────────────────────────────────────────────────────┐
    │  chassis Agent (agentic_router)                             │
    │    ├── Router        ← FrenchCase tier policy (via bridge)  │
    │    ├── GatewayClient ← FrenchCase LLMRouter   (llm_bridge)  │
    │    ├── SessionStore  ← FrenchCase sessions.json (mem_bridge)│
    │    ├── tools         ← FrenchCase 5 tools  (learncase_tools)│
    │    └── RAG          ← FrenchCase ChromaDB (retriever_bridge)│
    └─────────────────────────────────────────────────────────────┘

Because every collaborator has a no-op fallback, this module imports and runs
whether or not FrenchCase is mounted — it just degrades to stubs.

Usage:
    from projects.frenchcase.wiring import build_frenchcase_stack
    stack = build_frenchcase_stack()
    print(stack["status"])
"""
from __future__ import annotations

import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# ── Chassis pieces ───────────────────────────────────────────────────────
from agentic_router.clients import GatewayClientRegistry
from agentic_router.config import Settings as RouterSettings
from agentic_router.models import ModelTier
from agentic_router.router.router import Router
from agentic_router.agent import Agent
from agentic_router.tools import build_default_registry
from memory_store.models import MemoryQuery

# ── FrenchCase bridges ───────────────────────────────────────────────────
from projects.frenchcase.bridges import (
    build_gateway,
    build_session_store,
    build_vector_memory,
)
from projects.frenchcase.bridges.llm_bridge import FRENCHCASE_AVAILABLE as LLM_AVAILABLE
from projects.frenchcase.bridges.memory_bridge import FRENCHCASE_AVAILABLE as MEM_AVAILABLE
from projects.frenchcase.bridges.retriever_bridge import FRENCHCASE_AVAILABLE as RET_AVAILABLE
from projects.frenchcase.tools.learncase_tools import build_frenchcase_tools


class FrenchCaseRAG:
    """Retrieval helper the tutor agent uses for grounding.

    Wraps the retriever bridge so the agent can call
    `await rag.retrieve("subjonctif", cefr="B2")` and get back plain dicts —
    the same shape FrenchCase's own `core.vector.retrieve()` returns, so
    prompts built upstream keep working.
    """

    def __init__(self, vector_memory=None):
        self._vm = vector_memory or build_vector_memory()

    async def retrieve(
        self,
        query: str,
        cefr: str | None = None,
        section: str | None = None,
        skill: str | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        filt: dict[str, Any] = {}
        if cefr:
            filt["cefr"] = cefr
        if section:
            filt["section"] = section
        if skill:
            filt["skill"] = skill
        hits = await self._vm.search(
            MemoryQuery(query=query, top_k=top_k, filter=filt or None)
        )
        return [
            {"text": rec.content, "metadata": rec.metadata, "score": score}
            for rec, score in hits
        ]


def build_tool_registry(fc_tools: dict | None = None):
    """Chassis tool registry extended with FrenchCase's 5 tools.

    Returns a `toolkit.base.Registry` when the chassis toolkit is importable,
    otherwise the plain dict from `build_frenchcase_tools()` so the caller can
    still inspect tool definitions.
    """
    fc = fc_tools or build_frenchcase_tools()
    try:
        from toolkit.base import Registry  # type: ignore
    except Exception:
        logger.info("chassis toolkit not importable — returning FrenchCase tool dict")
        return fc

    registry = build_default_registry()
    # Registering FrenchCase tools requires a chassis Tool class; when the
    # toolkit exposes one we use it, else we attach the executor for adapter use.
    ToolCls = None
    try:
        from toolkit.base import Tool  # type: ignore
        ToolCls = Tool
    except Exception:
        pass

    if ToolCls is None:
        return registry

    for spec in fc.get("definitions", []):
        name = spec["name"]

        def _make(n: str) -> Callable[..., Any]:
            async def _run(**kwargs: Any) -> Any:
                from projects.frenchcase.tools.learncase_tools import execute_tool
                return await execute_tool(n, kwargs)
            return _run

        try:
            registry.register(
                ToolCls(
                    name=name,
                    description=spec["description"],
                    parameters=spec["parameters"],
                    func=_make(name),
                )
            )
        except Exception as exc:
            logger.debug("could not register FrenchCase tool %s: %s", name, exc)
    return registry


def build_frenchcase_stack(
    *,
    task_tier: str = "lower",
    settings: RouterSettings | None = None,
) -> dict[str, Any]:
    """Assemble the chassis agent stack backed by FrenchCase engines.

    Returns a dict of the built pieces plus a `status` block reporting which
    FrenchCase subsystems were actually reachable (so callers never have to
    guess whether they're on real engines or stubs).
    """
    gateway = build_gateway()
    sessions = build_session_store()
    vector_memory = build_vector_memory()
    rag = FrenchCaseRAG(vector_memory)
    fc_tools = build_frenchcase_tools()
    tools = build_tool_registry(fc_tools)

    s = settings or RouterSettings()
    # GatewayClientRegistry adapts ANY gateway exposing .client(alias) — our
    # FrenchCase gateway satisfies that, so the router routes tier→alias→router.
    registry = GatewayClientRegistry(gateway, "lower", "higher")
    router = Router(s, registry.get(ModelTier.LOWER))

    # The chassis Agent requires a real toolkit Registry. When the toolkit
    # component isn't installed we stop one step short and report it, rather
    # than handing the agent a dict it can't call.
    agent = None
    if not isinstance(tools, dict):
        agent = Agent(s, registry, router, tools, sessions)
    else:
        logger.info("chassis toolkit unavailable — skipping Agent construction")

    status = {
        "llm_router": LLM_AVAILABLE and gateway.available,
        "session_store": MEM_AVAILABLE,
        "vector_memory": RET_AVAILABLE,
        "tools": fc_tools.get("count", 0),
        "tool_names": fc_tools.get("names", []),
        "gateway_aliases": ["lower", "higher"],
        "task_tier": task_tier,
        "chassis_agent_built": agent is not None,
    }

    logger.info("FrenchCase stack: %s", status)
    return {
        "gateway": gateway,
        "sessions": sessions,
        "vector_memory": vector_memory,
        "rag": rag,
        "tools": tools,
        "fc_tools": fc_tools,
        "router": router,
        "agent": agent,
        "settings": s,
        "status": status,
        "tiers": {"lower": ModelTier.LOWER, "higher": ModelTier.HIGHER},
    }


if __name__ == "__main__":
    import asyncio as _aio
    import json as _json

    async def _main() -> None:
        stack = build_frenchcase_stack()
        print(_json.dumps(stack["status"], indent=2))
        hits = await stack["rag"].retrieve("subjonctif", cefr="B2", top_k=2)
        print(f"rag hits: {len(hits)}")

    _aio.run(_main())