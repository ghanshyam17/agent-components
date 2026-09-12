"""The gateway: load balance + fallback + retry + rate limit across endpoints.

`EndpointRunner` wraps a single OpenAI-compatible backend (built via the `openai`
SDK) with a `Limiter` and metrics. `Gateway` owns the runners and the groups;
`Gateway.client(alias)` returns a `GatewayClient` with the same
`chat()`/`stream()` shapes the router's `ModelClient` exposes, so the router
can use the gateway as a drop-in.
"""
from __future__ import annotations

import asyncio
import itertools
import json
import time
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from openai import AsyncOpenAI

from components_core import Message, messages_to_openai
from model_gateway.metrics import EndpointMetrics
from model_gateway.models import Endpoint, GatewayConfig, Group
from model_gateway.rate_limit import Limiter


# --------------------------------------------------------------------------- #
# A minimal interface for the OpenAI-compatible chat client, so tests can
# inject a fake without constructing a real AsyncOpenAI.
# --------------------------------------------------------------------------- #
@runtime_checkable
class ChatClient(Protocol):
    @property
    def chat(self) -> Any: ...  # has .completions.create(...)


class EndpointRunner:
    """One backend: openai client + limiter + metrics."""

    def __init__(
        self,
        endpoint: Endpoint,
        client: Any | None = None,
    ):
        self.endpoint = endpoint
        self.client = client or AsyncOpenAI(
            base_url=endpoint.base_url,
            api_key=endpoint.api_key,
            timeout=endpoint.timeout,
        )
        self.limiter = Limiter(rps=endpoint.rps, max_concurrency=endpoint.max_concurrency)
        self.metrics = EndpointMetrics(name=endpoint.name)

    @property
    def enabled(self) -> bool:
        return self.endpoint.enabled

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, list]:
        from components_core import ToolCallRequest  # local to avoid import cycle on Protocol

        t0 = time.monotonic()
        await self.limiter.acquire()
        try:
            resp = await self.client.chat.completions.create(
                model=self.endpoint.model,
                messages=messages_to_openai(messages),
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception:
            self.metrics.record(time.monotonic() - t0, ok=False)
            raise
        finally:
            self.limiter.release()

        msg = resp.choices[0].message
        content = msg.content or ""
        calls = [
            ToolCallRequest(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (msg.tool_calls or [])
        ]
        self.metrics.record(time.monotonic() - t0, ok=True)
        return content, calls

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Yields the same event shapes as the router's ModelClient.stream:
            {"type": "delta", "content": str}
            {"type": "tool_calls", "tool_calls": [ToolCallRequest, ...]}
            {"type": "done", "content": full_content}
        """
        from components_core import ToolCallRequest

        await self.limiter.acquire()
        t0 = time.monotonic()
        try:
            stream = await self.client.chat.completions.create(
                model=self.endpoint.model,
                messages=messages_to_openai(messages),
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )
        except Exception:
            self.metrics.record(time.monotonic() - t0, ok=False)
            raise

        collected: list[str] = []
        tool_buffers: dict[int, dict[str, str]] = {}
        has_tool_calls = False
        try:
            async for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta
                if delta.content:
                    collected.append(delta.content)
                    yield {"type": "delta", "content": delta.content}
                if delta.tool_calls:
                    has_tool_calls = True
                    for tc in delta.tool_calls:
                        buf = tool_buffers.setdefault(tc.index, {"name": "", "args": ""})
                        if tc.function and tc.function.name:
                            buf["name"] += tc.function.name
                        if tc.function and tc.function.arguments:
                            buf["args"] += tc.function.arguments
        except Exception:
            self.metrics.record(time.monotonic() - t0, ok=False)
            raise
        finally:
            self.limiter.release()

        if has_tool_calls:
            calls = [
                ToolCallRequest(name=b["name"], arguments=json.loads(b["args"] or "{}"))
                for _, b in tool_buffers.items()
            ]
            self.metrics.record(time.monotonic() - t0, ok=True)
            yield {"type": "tool_calls", "tool_calls": calls}
        else:
            self.metrics.record(time.monotonic() - t0, ok=True)
            yield {"type": "done", "content": "".join(collected)}


# --------------------------------------------------------------------------- #
class Gateway:
    def __init__(self, config: GatewayConfig, *, client_factory: Any | None = None):
        self.config = config
        self._runners: dict[str, EndpointRunner] = {
            e.name: EndpointRunner(e, client=client_factory(e) if client_factory else None)
            for e in config.endpoints
        }
        self._groups: dict[str, Group] = {g.alias: g for g in config.groups}
        # round-robin cursor per group
        self._cursors: dict[str, itertools.cycle] = {}

    # -- lookup ------------------------------------------------------------- #
    def _group(self, alias: str) -> Group:
        g = self._groups.get(alias)
        if g is None:
            raise KeyError(f"unknown gateway group alias: {alias!r}")
        return g

    def _runners_for(self, group: Group) -> list[EndpointRunner]:
        runners = [self._runners[n] for n in group.endpoints if n in self._runners]
        enabled = [r for r in runners if r.enabled]
        return enabled or runners  # fall back to disabled list if all disabled

    def _ordered(self, group: Group) -> list[EndpointRunner]:
        runners = self._runners_for(group)
        if group.strategy == "failover":
            return runners
        if group.strategy == "weighted":
            expanded: list[EndpointRunner] = []
            for r in runners:
                expanded.extend([r] * r.endpoint.weight)
            return expanded
        # round_robin: rotate starting index per call
        cycle = self._cursors.setdefault(group.alias, itertools.cycle(range(len(runners))))
        if not runners:
            return []
        start = next(cycle) % max(1, len(runners))
        return runners[start:] + runners[:start]

    # -- public client ------------------------------------------------------ #
    def client(self, alias: str) -> "GatewayClient":
        if alias not in self._groups:
            raise KeyError(f"unknown gateway group alias: {alias!r}")
        return GatewayClient(self, alias)

    def metrics(self) -> dict[str, dict]:
        return {name: r.metrics.snapshot() for name, r in self._runners.items()}

    # -- group-level call with fallback + retry ----------------------------- #
    async def call_chat(self, alias, messages, **kw):
        group = self._group(alias)
        runners = self._ordered(group)
        if not runners:
            raise RuntimeError(f"no enabled endpoints for group {alias!r}")
        last_err: Exception | None = None
        for p in range(group.max_retries + 1):
            for runner in runners:
                try:
                    return await runner.chat(messages, **kw)
                except Exception as e:
                    last_err = e
            # whole pass failed → back off and retry the pass
            if p < group.max_retries:
                await asyncio.sleep(group.retry_backoff * (2 ** p))
        assert last_err is not None
        raise last_err

    async def call_stream(self, alias, messages, **kw) -> AsyncIterator[dict[str, Any]]:
        group = self._group(alias)
        runners = self._ordered(group)
        if not runners:
            raise RuntimeError(f"no enabled endpoints for group {alias!r}")
        last_err: Exception | None = None
        for p in range(group.max_retries + 1):
            for runner in runners:
                try:
                    agen = runner.stream(messages, **kw)
                    first = await agen.__anext__()
                except StopAsyncIteration:
                    # empty stream; treat as a normal (empty) completion
                    return
                except Exception as e:
                    last_err = e
                    continue
                # committed to this runner — yield the first event then the rest
                yield first
                async for ev in agen:
                    yield ev
                return
            if p < group.max_retries:
                await asyncio.sleep(group.retry_backoff * (2 ** p))
        if last_err is not None:
            raise last_err


class GatewayClient:
    """Router-compatible client bound to one gateway group/alias."""

    def __init__(self, gateway: Gateway, alias: str):
        self.gateway = gateway
        self.alias = alias

    @property
    def model(self) -> str:
        """Best-effort model label for display (first endpoint in the group)."""
        group = self.gateway._group(self.alias)
        runners = self.gateway._runners_for(group)
        return runners[0].endpoint.model if runners else "unknown"

    @property
    def tier(self) -> str:
        return self.alias

    async def chat(self, messages, *, tools=None, temperature=0.2, max_tokens=None):
        return await self.gateway.call_chat(
            self.alias, messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        )

    async def stream(self, messages, *, tools=None, temperature=0.2, max_tokens=None):
        async for ev in self.gateway.call_stream(
            self.alias, messages, tools=tools, temperature=temperature, max_tokens=max_tokens
        ):
            yield ev