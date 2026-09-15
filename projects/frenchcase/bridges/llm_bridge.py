"""Bridge: FrenchCase LLMRouter → chassis model-gateway GatewayClient.

The chassis router asks the gateway for a client by alias (`lower` / `higher`)
and calls `await client.chat(messages, tools=..., temperature=..., max_tokens=...)`,
expecting `(content, tool_calls)` back.

FrenchCase's `app/core/llm_router.py` already does the hard part — 5 backends
(Azure OpenAI, vLLM, Ollama local, Ollama cloud, edge TTS), 4 routing
strategies, KV-cache management, metrics, and a fallback chain. Rather than
duplicating that inside a gateway YAML, this adapter exposes the router as a
gateway-shaped client so the chassis's agentic-router gets FrenchCase's
routing intelligence for free.

Alias → FrenchCase task mapping:
    lower  → "chat_tutor"        (simple Q&A, vocab lookup → fast model)
    higher → "grammar_explain"   (grammar/essay feedback → capable model)

Usage:
    from projects.frenchcase.bridges.llm_bridge import build_gateway
    gw = build_gateway()
    lower = gw.client("lower")
    content, tool_calls = await lower.chat([{"role": "user", "content": "..."}])
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator

logger = logging.getLogger(__name__)

# ── FrenchCase import (mounted at runtime) ───────────────────────────────
# `mount_frenchcase()` puts the repo root first (so `app.*` resolves to the
# package) and app/ last as a fallback (so bare `from config import ...` works).
from projects.frenchcase.mount import mount_frenchcase as _mount

_mount()

try:
    from app.core.llm_router import LLMRouter as _FCRouter  # type: ignore
    FRENCHCASE_AVAILABLE = True
except ImportError:
    try:
        from core.llm_router import LLMRouter as _FCRouter  # type: ignore
        FRENCHCASE_AVAILABLE = True
    except ImportError:
        _FCRouter = None  # type: ignore
        FRENCHCASE_AVAILABLE = False

# Alias → FrenchCase routing task. These task names are the real keys in
# AGENT_COMPLEXITY inside llm_router.py, and each resolves to the model the
# agent YAML's tier_strategy declares:
#   lower  → "grammar"    → MODERATE → [gpt-4o-mini, deepseek-v4-flash, ...]
#   higher → "chat_tutor" → COMPLEX  → [gpt-4o, gpt-4o-mini, ...]
ALIAS_TASK_MAP: dict[str, str] = {
    "lower": "grammar",
    "higher": "chat_tutor",
    "default": "grammar",
}


class FrenchCaseGatewayClient:
    """Gateway-shaped client over FrenchCase's LLMRouter.

    Matches the chassis `model_gateway.gateway.GatewayClient` surface:
    `.chat(...)` returns `(content, tool_calls)`, `.stream(...)` yields chunks,
    `.model` / `.tier` expose routing metadata.
    """

    def __init__(self, router: Any, alias: str):
        self._router = router
        self._alias = alias
        self._last_meta: dict[str, Any] = {}

    @property
    def model(self) -> str:
        """The model FrenchCase's policy would select for this alias' task."""
        return (
            self._last_meta.get("model")
            or self._last_meta.get("model_name")
            or f"frenchcase-{self._alias}"
        )

    @property
    def tier(self) -> str:
        return self._alias

    @property
    def last_metadata(self) -> dict[str, Any]:
        """Telemetry from the last call (backend, latency, tokens, cost, cache)."""
        return dict(self._last_meta)

    def _chat_sync(
        self,
        messages: list[dict[str, Any]],
        require_json: bool,
        task: str,
    ) -> dict[str, Any]:
        return self._router.chat(
            messages,
            task=task,
            require_json=require_json,
            stream=False,
        ) or {}

    async def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Any = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
        require_json: bool = False,
        task: str | None = None,
    ) -> tuple[str, Any]:
        """Route a chat request through FrenchCase's LLMRouter.

        Returns `(content, tool_calls)`. FrenchCase's router returns text (it
        handles tool-calling internally via the agent loop), so `tool_calls` is
        always None here — the chassis agent parses tools from its own loop.
        """
        if self._router is None:
            logger.warning("FrenchCase LLMRouter unavailable — returning placeholder")
            return "[frenchcase router unavailable]", None

        route_task = task or ALIAS_TASK_MAP.get(self._alias, "chat_tutor")
        result = await asyncio.to_thread(self._chat_sync, messages, require_json, route_task)

        self._last_meta = {
            "model": result.get("model"),
            "backend": result.get("backend"),
            "latency_ms": result.get("latency_ms"),
            "tokens": result.get("tokens"),
            "cost": result.get("cost"),
            "cache_hit": result.get("cache_hit"),
            "fallback_used": result.get("fallback_used"),
            "alias": self._alias,
            "task": route_task,
        }

        content = result.get("content") or ""
        if result.get("fallback_used"):
            logger.info(
                "FrenchCase router fell back for alias=%s task=%s backend=%s",
                self._alias, route_task, result.get("backend"),
            )
        return content, result.get("tool_calls")

    async def stream(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: Any = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Non-streaming router rendered as a single-chunk stream.

        FrenchCase's `LLMRouter.chat()` is synchronous and non-streaming (the
        app's SSE path lives in `core/llm.py::chat_stream`). To keep one
        adapter rather than two, we emit the full answer as one delta then a
        terminal event — callers that need true token streaming should call
        `core/llm.py::chat_stream` directly.
        """
        content, _ = await self.chat(messages, tools=tools, temperature=temperature, max_tokens=max_tokens)
        yield {"type": "delta", "content": content}
        yield {"type": "final", "content": content, "metadata": dict(self._last_meta)}


class FrenchCaseGateway:
    """Minimal Gateway façade: `client(alias)` → FrenchCaseGatewayClient."""

    def __init__(self, router: Any | None = None):
        if router is not None:
            self._router = router
        elif FRENCHCASE_AVAILABLE and _FCRouter is not None:
            self._router = _FCRouter()
        else:
            self._router = None
            logger.warning("FrenchCase LLMRouter unavailable — gateway bridge in no-op mode")
        self._clients: dict[str, FrenchCaseGatewayClient] = {}

    @property
    def available(self) -> bool:
        return self._router is not None

    def client(self, alias: str) -> FrenchCaseGatewayClient:
        if alias not in self._clients:
            self._clients[alias] = FrenchCaseGatewayClient(self._router, alias)
        return self._clients[alias]

    async def call_chat(self, alias: str, messages: list[dict[str, Any]], **kw: Any) -> tuple[str, Any]:
        return await self.client(alias).chat(messages, **kw)

    def metrics(self) -> dict[str, dict]:
        """Expose FrenchCase's own router metrics if it tracks them."""
        if self._router is None:
            return {}
        try:
            status = self._router.get_status() or {}
            return {"frenchcase_router": status}
        except Exception as exc:  # pragma: no cover
            logger.debug("router.get_status() failed: %s", exc)
            return {}


def build_gateway(router: Any | None = None) -> FrenchCaseGateway:
    """Factory matching the chassis `model_gateway.build_gateway()` shape."""
    return FrenchCaseGateway(router)


if __name__ == "__main__":
    import asyncio as _aio

    async def _main() -> None:
        gw = build_gateway()
        print(f"FrenchCase mounted: {FRENCHCASE_AVAILABLE}, gateway available: {gw.available}")
        client = gw.client("lower")
        print(f"alias={client.tier} model={client.model}")
        content, _ = await client.chat([{"role": "user", "content": "Bonjour, ça va ?"}])
        print(f"response: {content[:120]}")
        print(f"metadata: {client.last_metadata}")

    _aio.run(_main())