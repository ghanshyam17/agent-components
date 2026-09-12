"""Async OpenAI-compatible clients for the two vLLM instances.

vLLM exposes an OpenAI-compatible `/v1/chat/completions` endpoint, so we use
the `openai` SDK pointed at each instance's `base_url`. Tool calling and
streaming both work over that interface.

For production with multiple replicas / cloud fallback, prefer the
`model-gateway` component: build a `Gateway` and pass it to
`GatewayClientRegistry`, which exposes the same `.get(tier)` interface and is
a drop-in replacement for `ClientRegistry`.
"""
from __future__ import annotations

import json
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from agentic_router.config import Settings
from agentic_router.models import Message, ModelTier, ToolCallRequest
from components_core import messages_to_openai


class ModelClient:
    """Wraps a single vLLM-served model behind an AsyncOpenAI client."""

    def __init__(self, tier: ModelTier, model: str, base_url: str, api_key: str):
        self.tier = tier
        self.model = model
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)

    async def chat(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> tuple[str, list[ToolCallRequest]]:
        """Non-streaming completion. Returns (content, tool_calls)."""
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages_to_openai(messages),
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        choice = resp.choices[0]
        msg = choice.message
        content = msg.content or ""
        calls = [
            ToolCallRequest(
                name=tc.function.name,
                arguments=json.loads(tc.function.arguments or "{}"),
            )
            for tc in (msg.tool_calls or [])
        ]
        return content, calls

    async def stream(
        self,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.2,
        max_tokens: int | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Streaming completion. Yields event dicts:
            {"type": "delta", "content": "..."}
            {"type": "tool_calls", "tool_calls": [ToolCallRequest, ...]}
            {"type": "done", "content": full_content}
        When tools are provided and the model picks one, no deltas are
        streamed — a single "tool_calls" event is yielded instead.
        """
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages_to_openai(messages),
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        collected: list[str] = []
        tool_buffers: dict[int, dict[str, str]] = {}
        has_tool_calls = False

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

        if has_tool_calls:
            calls: list[ToolCallRequest] = []
            for _, buf in tool_buffers.items():
                calls.append(
                    ToolCallRequest(
                        name=buf["name"],
                        arguments=json.loads(buf["args"] or "{}"),
                    )
                )
            yield {"type": "tool_calls", "tool_calls": calls}
        else:
            yield {"type": "done", "content": "".join(collected)}


class ClientRegistry:
    """Holds the two ModelClients keyed by tier."""

    def __init__(self, settings: Settings):
        s = settings
        self.clients: dict[ModelTier, ModelClient] = {
            ModelTier.LOWER: ModelClient(
                ModelTier.LOWER, s.lower_model, s.lower_vllm_base_url, s.vllm_api_key
            ),
            ModelTier.HIGHER: ModelClient(
                ModelTier.HIGHER,
                s.higher_model,
                s.higher_vllm_base_url,
                s.vllm_api_key,
            ),
        }

    def get(self, tier: ModelTier) -> ModelClient:
        return self.clients[tier]


class GatewayClientRegistry:
    """Registry backed by a `model-gateway` Gateway instead of direct vLLM.

    Exposes the same `.get(tier)` interface as `ClientRegistry`, so the router
    can use either interchangeably. Tiers map to gateway group aliases via the
    settings (`lower_alias` / `higher_alias`).
    """

    def __init__(self, gateway, lower_alias: str, higher_alias: str):
        self.gateway = gateway
        self.clients: dict[ModelTier, Any] = {
            ModelTier.LOWER: gateway.client(lower_alias),
            ModelTier.HIGHER: gateway.client(higher_alias),
        }

    def get(self, tier: ModelTier):
        return self.clients[tier]