"""model-gateway tests — load balancing, fallback, retries, rate limiting,
metrics, and the router-compatible chat/stream shapes. Uses a fake
OpenAI-compatible client injected via `client_factory` (no real server)."""
from __future__ import annotations

import asyncio
import json

import pytest

from components_core import Message, ToolCallRequest
from model_gateway import Gateway
from model_gateway.models import Endpoint, GatewayConfig, Group


# ----------------------- fakes ----------------------- #
class _Msg:
    """Mimics openai chat completion message."""

    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _Choice:
    def __init__(self, message):
        self.message = message


class _Resp:
    def __init__(self, message):
        self.choices = [_Choice(message)]


class _Func:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _ToolCall:
    def __init__(self, name, args):
        self.index = 0
        self.function = _Func(name, json.dumps(args))


class _Delta:
    def __init__(self, content=None, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _StreamChunk:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


class _FakeCompletions:
    def __init__(self, scripts, fail_stream_before_first=False):
        self.scripts = scripts            # list of ("content", tool_calls) for chat
        self._i = 0
        self.fail_stream_before_first = fail_stream_before_first
        self.calls = 0

    async def create(self, *, model, messages, tools=None, temperature=0.2,
                     max_tokens=None, stream=False):
        self.calls += 1
        if stream:
            if self.fail_stream_before_first:
                raise RuntimeError("connection refused")
            content, tool_calls = self.scripts[min(self._i, len(self.scripts) - 1)]

            async def gen():
                if tool_calls:
                    # streaming tool call: emit name + args in one chunk
                    tc = tool_calls[0]
                    yield _StreamChunk(_Delta(tool_calls=[_ToolCall(tc.name, tc.arguments)]))
                else:
                    for w in content.split():
                        yield _StreamChunk(_Delta(content=w + " "))
            return gen()
        # non-stream
        content, tool_calls = self.scripts[min(self._i, len(self.scripts) - 1)]
        self._i += 1
        oai_tool_calls = None
        if tool_calls:
            oai_tool_calls = [
                type("TC", (), {"function": _Func(tc.name, json.dumps(tc.arguments))})()
                for tc in tool_calls
            ]
        return _Resp(_Msg(content, oai_tool_calls))


class _FakeChat:
    def __init__(self, completions):
        self.completions = completions


class _FakeClient:
    def __init__(self, scripts, fail_stream_before_first=False):
        self.chat = _FakeChat(_FakeCompletions(scripts, fail_stream_before_first))


def _factory(scripts_by_name, fail_stream=None):
    """client_factory: maps endpoint -> _FakeClient with its own scripts."""
    fail_stream = fail_stream or {}

    def factory(ep: Endpoint):
        return _FakeClient(
            scripts_by_name.get(ep.name, [("ok", [])]),
            fail_stream_before_first=fail_stream.get(ep.name, False),
        )
    return factory


# ----------------------- tests ----------------------- #
def _config(endpoints, groups):
    return GatewayConfig(
        endpoints=[Endpoint(**e) for e in endpoints],
        groups=[Group(**g) for g in groups],
    )


def test_chat_round_robin_balances():
    cfg = _config(
        endpoints=[
            {"name": "a", "base_url": "http://x", "model": "m-a"},
            {"name": "b", "base_url": "http://y", "model": "m-b"},
        ],
        groups=[{"alias": "g", "endpoints": ["a", "b"], "strategy": "round_robin"}],
    )
    factory = _factory({"a": [("from-a", [])], "b": [("from-b", [])]})
    gw = Gateway(cfg, client_factory=factory)
    client = gw.client("g")

    loop = asyncio.get_event_loop()
    msgs = [Message(role="user", content="hi")]
    r1 = loop.run_until_complete(client.chat(msgs))
    r2 = loop.run_until_complete(client.chat(msgs))
    r3 = loop.run_until_complete(client.chat(msgs))
    contents = {r1[0], r2[0], r3[0]}
    assert contents == {"from-a", "from-b"}  # both endpoints used
    m = gw.metrics()
    assert m["a"]["requests"] >= 1 and m["b"]["requests"] >= 1


def test_chat_fallback_on_error():
    # endpoint "a" always errors (script with sentinel that the fake raises on)
    cfg = _config(
        endpoints=[
            {"name": "a", "base_url": "http://x", "model": "m-a"},
            {"name": "b", "base_url": "http://y", "model": "m-b"},
        ],
        groups=[{"alias": "g", "endpoints": ["a", "b"], "strategy": "failover",
                 "max_retries": 0}],
    )

    class _ErrCompletions:
        async def create(self, **kw):
            raise RuntimeError("a is down")

    class _ErrClient:
        def __init__(self):
            self.chat = _FakeChat(_ErrCompletions())

    def factory(ep):
        if ep.name == "a":
            return _ErrClient()
        return _FakeClient([("from-b", [])])

    gw = Gateway(cfg, client_factory=factory)
    client = gw.client("g")
    loop = asyncio.get_event_loop()
    content, _ = loop.run_until_complete(client.chat([Message(role="user", content="x")]))
    assert content == "from-b"
    m = gw.metrics()
    assert m["a"]["errors"] == 1
    assert m["b"]["successes"] == 1


def test_chat_retries_exhaust_raises():
    cfg = _config(
        endpoints=[{"name": "a", "base_url": "http://x", "model": "m"}],
        groups=[{"alias": "g", "endpoints": ["a"], "max_retries": 2, "retry_backoff": 0}],
    )

    class _ErrClient:
        def __init__(self):
            class C:
                async def create(self, **kw):
                    raise RuntimeError("always fails")
            self.chat = _FakeChat(C())

    gw = Gateway(cfg, client_factory=lambda ep: _ErrClient())
    client = gw.client("g")
    loop = asyncio.get_event_loop()
    with pytest.raises(RuntimeError):
        loop.run_until_complete(client.chat([Message(role="user", content="x")]))
    # first attempt + 2 retries = 3 tries
    assert gw.metrics()["a"]["errors"] == 3


def test_chat_tool_calls_parsed():
    cfg = _config(
        endpoints=[{"name": "a", "base_url": "http://x", "model": "m"}],
        groups=[{"alias": "g", "endpoints": ["a"]}],
    )
    factory = _factory({
        "a": [("", [ToolCallRequest(name="calculator", arguments={"expression": "2+2"})])],
    })
    gw = Gateway(cfg, client_factory=factory)
    content, calls = asyncio.get_event_loop().run_until_complete(
        gw.client("g").chat([Message(role="user", content="calc")])
    )
    assert content == ""
    assert len(calls) == 1 and calls[0].name == "calculator"
    assert calls[0].arguments == {"expression": "2+2"}


def test_stream_emits_deltas_then_done():
    cfg = _config(
        endpoints=[{"name": "a", "base_url": "http://x", "model": "m"}],
        groups=[{"alias": "g", "endpoints": ["a"]}],
    )
    factory = _factory({"a": [("hello world", [])]})
    gw = Gateway(cfg, client_factory=factory)

    async def run():
        events = []
        async for ev in gw.client("g").stream([Message(role="user", content="x")]):
            events.append(ev)
        return events

    events = asyncio.get_event_loop().run_until_complete(run())
    types = [e["type"] for e in events]
    assert types.count("delta") == 2
    assert types[-1] == "done"
    assert events[-1]["content"].strip() == "hello world"


def test_stream_fallback_on_connection_error():
    cfg = _config(
        endpoints=[
            {"name": "a", "base_url": "http://x", "model": "m-a"},
            {"name": "b", "base_url": "http://y", "model": "m-b"},
        ],
        groups=[{"alias": "g", "endpoints": ["a", "b"], "strategy": "failover"}],
    )
    factory = _factory(
        {"a": [("from-a", [])], "b": [("from-b", [])]},
        fail_stream={"a": True},   # 'a' fails before first event; fall back to 'b'
    )
    gw = Gateway(cfg, client_factory=factory)

    async def run():
        events = []
        async for ev in gw.client("g").stream([Message(role="user", content="x")]):
            events.append(ev)
        return events

    events = asyncio.get_event_loop().run_until_complete(run())
    assert any(e.get("content") == "from-b" or e.get("content", "").startswith("from-b") for e in events)
    assert gw.metrics()["a"]["errors"] == 1


def test_stream_tool_calls_event():
    cfg = _config(
        endpoints=[{"name": "a", "base_url": "http://x", "model": "m"}],
        groups=[{"alias": "g", "endpoints": ["a"]}],
    )
    factory = _factory({
        "a": [("", [ToolCallRequest(name="web_fetch", arguments={"url": "http://x"})])],
    })
    gw = Gateway(cfg, client_factory=factory)

    async def run():
        return [e async for e in gw.client("g").stream([Message(role="user", content="x")])]

    events = asyncio.get_event_loop().run_until_complete(run())
    assert events[-1]["type"] == "tool_calls"
    assert events[-1]["tool_calls"][0].name == "web_fetch"


def test_rate_limit_caps_concurrency():
    # max_concurrency=1 forces serialization; two concurrent chats must not
    # overlap (we track high-water-mark in the fake).
    cfg = _config(
        endpoints=[{"name": "a", "base_url": "http://x", "model": "m",
                    "max_concurrency": 1}],
        groups=[{"alias": "g", "endpoints": ["a"]}],
    )

    state = {"in_flight": 0, "max": 0}

    class _SlowCompletions:
        async def create(self, **kw):
            state["in_flight"] += 1
            state["max"] = max(state["max"], state["in_flight"])
            await asyncio.sleep(0.05)
            state["in_flight"] -= 1
            return _Resp(_Msg("ok", None))

    class _SlowClient:
        def __init__(self):
            self.chat = _FakeChat(_SlowCompletions())

    gw = Gateway(cfg, client_factory=lambda ep: _SlowClient())
    client = gw.client("g")
    msgs = [Message(role="user", content="x")]

    async def run():
        await asyncio.gather(client.chat(msgs), client.chat(msgs), client.chat(msgs))

    asyncio.get_event_loop().run_until_complete(run())
    assert state["max"] == 1  # never more than 1 in flight


def test_unknown_alias_raises():
    gw = Gateway(_config([{"name": "a", "base_url": "http://x", "model": "m"}],
                         [{"alias": "g", "endpoints": ["a"]}]),
                 client_factory=_factory({"a": [("ok", [])]}))
    with pytest.raises(KeyError):
        gw.client("nope")


def test_config_rejects_unknown_endpoint_ref():
    with pytest.raises(ValueError):
        GatewayConfig(
            endpoints=[Endpoint(name="a", base_url="http://x", model="m")],
            groups=[Group(alias="g", endpoints=["missing"])],
        )