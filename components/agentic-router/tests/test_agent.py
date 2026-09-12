"""Tests for tools and the agent loop (no real vLLM)."""
from __future__ import annotations

import asyncio

import pytest

from agentic_router.agent import Agent
from agentic_router.agent.loop import _looks_multi_step
from agentic_router.clients import ClientRegistry
from agentic_router.config import Settings
from agentic_router.models import Message, ModelTier, RouteDecision
from agentic_router.router.router import Router
from agentic_router.tools import build_default_registry
from agentic_router.tools.builtin import CalculatorTool
from memory_store import InMemorySessionStore


# ---------------- calculator tool ----------------
def test_calculator_basic():
    tool = CalculatorTool()
    res = asyncio.get_event_loop().run_until_complete(tool.run("2*(3+4)"))
    assert res.ok and res.output == "14"


def test_calculator_rejects_bad():
    tool = CalculatorTool()
    res = asyncio.get_event_loop().run_until_complete(tool.run("import os; os.system('rm')"))
    assert not res.ok


def test_registry_dispatch_unknown():
    reg = build_default_registry()
    res = asyncio.get_event_loop().run_until_complete(
        reg.dispatch("nope", {})
    )
    assert not res.ok and "unknown tool" in (res.error or "")


# ---------------- multi-step detection ----------------
def test_looks_multi_step():
    assert _looks_multi_step("first do X, then do Y, and finally Z")
    assert not _looks_multi_step("hello")


# ---------------- agent loop with stubbed clients ----------------
class _StubClient:
    """Replays a scripted sequence of (content, tool_calls) per chat call,
    and streams content chunks via stream()."""

    def __init__(self, tier, scripts):
        self.tier = tier
        self.scripts = list(scripts)
        self._i = 0

    async def chat(self, messages, **kwargs):
        idx = min(self._i, len(self.scripts) - 1)
        self._i += 1
        return self.scripts[idx]

    async def stream(self, messages, **kwargs):
        idx = min(self._i, len(self.scripts) - 1)
        self._i += 1
        content, tool_calls = self.scripts[idx]
        if tool_calls:
            yield {"type": "tool_calls", "tool_calls": tool_calls}
        else:
            # stream word-by-word
            for w in content.split():
                yield {"type": "delta", "content": w + " "}
            yield {"type": "done", "content": content}


class _StubRegistry(ClientRegistry):
    def __init__(self, lower_scripts, higher_scripts, settings):
        self.clients = {
            ModelTier.LOWER: _StubClient(ModelTier.LOWER, lower_scripts),
            ModelTier.HIGHER: _StubClient(ModelTier.HIGHER, higher_scripts),
        }

    def get(self, tier):
        return self.clients[tier]


class _StubRouter(Router):
    def __init__(self, tier):
        self._forced = tier
        self.s = Settings()

    async def route(self, task):
        m = self.s.lower_model if self._forced is ModelTier.LOWER else self.s.higher_model
        return RouteDecision(
            tier=self._forced, score=0.5, method="heuristic",
            model=m, reason="stub", signals={},
        )


def _build_agent(tier, lower_scripts, higher_scripts, tools_enabled=True):
    s = Settings(
        agent_enable_planning=False,
        agent_enable_tools=tools_enabled,
        agent_max_iterations=3,
    )
    registry = _StubRegistry(lower_scripts, higher_scripts, s)
    router = _StubRouter(tier)
    tools = build_default_registry()
    sessions = InMemorySessionStore()
    return Agent(s, registry, router, tools, sessions)


def test_agent_no_tools_streams_answer():
    agent = _build_agent(
        ModelTier.LOWER,
        lower_scripts=[("Hello world answer.", [])],
        higher_scripts=[],
        tools_enabled=False,
    )

    async def run():
        state = await agent.sessions.create()
        return await _collect(agent.stream("hi", state.session_id))

    events = asyncio.get_event_loop().run_until_complete(run())
    types = [e.type for e in events]
    assert "answer" in types
    final = [e for e in events if e.type == "final"][0]
    assert "Hello world" in final.data["content"]


def test_agent_tool_roundtrip():
    # Lower model first calls calculator, then gives the final answer.
    from agentic_router.models import ToolCallRequest
    agent = _build_agent(
        ModelTier.LOWER,
        lower_scripts=[
            ("", [ToolCallRequest(name="calculator", arguments={"expression": "6*7"})]),
            ("The answer is 42.", []),
        ],
        higher_scripts=[],
        tools_enabled=True,
    )

    async def run():
        state = await agent.sessions.create()
        return await _collect(agent.stream("compute 6*7", state.session_id))

    events = asyncio.get_event_loop().run_until_complete(run())
    types = [e.type for e in events]
    assert "tool_call" in types and "tool_result" in types
    final = [e for e in events if e.type == "final"][0]
    assert "42" in final.data["content"]


async def _collect(agen):
    out = []
    async for ev in agen:
        out.append(ev)
    return out