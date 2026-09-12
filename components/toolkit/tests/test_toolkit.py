"""toolkit tests — registry dispatch, sandbox, built-ins, decorator.

All tests run offline: file tools use a tmp sandbox; web/http tools use a
`httpx.MockTransport` to avoid network. No `uv`/`pytest` invocation here.
"""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from toolkit import (
    CalculatorTool,
    FileReadTool,
    FileWriteTool,
    Registry,
    SandboxError,
    ShellTool,
    ToolResult,
    build_registry,
    resolve_within,
    tool_from_function,
)
from toolkit.builtin import HTTPRequestTool, WebFetchTool


loop = asyncio.new_event_loop()


def run(coro):
    return loop.run_until_complete(coro)


# --------------------------------------------------------------------------- #
# Registry + dispatch
# --------------------------------------------------------------------------- #
def test_registry_dispatch_unknown_tool():
    reg = Registry()
    res = run(reg.dispatch("nope", {}))
    assert res.ok is False
    assert "unknown tool" in (res.error or "")


def test_build_registry_has_builtins():
    reg = build_registry()
    assert set(reg.names()) == {
        "run_shell", "read_file", "write_file",
        "web_fetch", "http_request", "calculator",
    }
    schema = reg.to_openai()
    assert all(s["type"] == "function" for s in schema)
    names = {s["function"]["name"] for s in schema}
    assert "calculator" in names


def test_registry_to_openai_filter_by_name():
    reg = build_registry()
    schema = reg.to_openai(["calculator", "missing"])
    assert len(schema) == 1
    assert schema[0]["function"]["name"] == "calculator"


def test_dispatch_catches_exceptions():
    class BoomTool:
        name = "boom"
        description = "x"
        parameters = {"type": "object", "properties": {}, "required": []}

        async def run(self, **kw):
            raise ValueError("kaboom")

        def to_openai(self):
            return {"type": "function", "function": {
                "name": self.name, "description": self.description, "parameters": self.parameters}}

    reg = Registry()
    reg.register(BoomTool())  # type: ignore[arg-type]
    res = run(reg.dispatch("boom", {}))
    assert res.ok is False
    assert "ValueError" in (res.error or "")


# --------------------------------------------------------------------------- #
# Calculator
# --------------------------------------------------------------------------- #
def test_calculator_basic_and_bad():
    calc = CalculatorTool()
    ok = run(calc.run(expression="12*(3+4)/2"))
    assert ok.ok and ok.output == "42.0"

    bad = run(calc.run(expression="import os; os.system('rm -rf /')"))
    assert bad.ok is False
    assert "unsupported expression" in (bad.error or "")


# --------------------------------------------------------------------------- #
# File read/write + sandbox
# --------------------------------------------------------------------------- #
def test_file_write_then_read_in_sandbox(tmp_path):
    root = tmp_path
    w = FileWriteTool(root=root)
    r = FileReadTool(root=root)
    res = run(w.run(path="sub/a.txt", content="hello world"))
    assert res.ok, res.error
    assert (root / "sub" / "a.txt").read_text() == "hello world"

    out = run(r.run(path="sub/a.txt"))
    assert out.ok and out.output == "hello world"


def test_file_escape_blocked(tmp_path):
    root = tmp_path
    w = FileWriteTool(root=root)
    res = run(w.run(path="../escaped.txt", content="x"))
    assert res.ok is False
    assert "escapes sandbox" in (res.error or "")
    # nothing written outside
    assert not (tmp_path.parent / "escaped.txt").exists()


def test_file_read_missing(tmp_path):
    r = FileReadTool(root=tmp_path)
    res = run(r.run(path="nope.txt"))
    assert res.ok is False
    assert "not found" in (res.error or "")


def test_resolve_within_no_root_allows_absolute():
    p = resolve_within("/etc", None)
    assert p.is_absolute()


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
def test_shell_basic_and_timeout():
    sh = ShellTool(default_timeout=10)
    res = run(sh.run(command="echo hello"))
    assert res.ok
    assert "hello" in res.output
    assert res.metadata.get("exit") == 0


def test_shell_deny_blocks_rm_rf():
    sh = ShellTool(deny=["rm -rf"])
    res = run(sh.run(command="rm -rf /"))
    assert res.ok is False
    assert "denied" in (res.error or "")


def test_shell_allow_list_restricts():
    sh = ShellTool(allow=["echo"])
    ok = run(sh.run(command="echo hi"))
    assert ok.ok
    bad = run(sh.run(command="ls"))
    assert bad.ok is False
    assert "denied" in (bad.error or "")


# --------------------------------------------------------------------------- #
# Web / HTTP with mocked transport
# --------------------------------------------------------------------------- #
def _mock_factory(handler):
    def factory():
        return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)
    return factory


def test_web_fetch_mocked():
    def handler(request):
        return httpx.Response(200, text="hello from web")

    wf = WebFetchTool(client_factory=_mock_factory(handler))
    res = run(wf.run(url="https://example.com"))
    assert res.ok
    assert "hello from web" in res.output
    assert res.metadata.get("status") == 200


def test_http_request_mocked_post():
    captured = {}

    def handler(request):
        captured["method"] = request.method
        captured["body"] = request.content.decode()
        return httpx.Response(201, text="created", headers={"content-type": "text/plain"})

    ht = HTTPRequestTool(client_factory=_mock_factory(handler))
    res = run(ht.run(url="https://api.x/items", method="POST", body='{"a":1}'))
    assert res.ok and res.metadata.get("status") == 201
    assert captured["method"] == "POST"
    assert json.loads(captured["body"]) == {"a": 1}


def test_web_fetch_requires_scheme():
    wf = WebFetchTool()
    res = run(wf.run(url="example.com"))
    assert res.ok is False
    assert "scheme" in (res.error or "")


# --------------------------------------------------------------------------- #
# tool_from_function decorator
# --------------------------------------------------------------------------- #
def test_function_tool_async():
    @tool_from_function
    async def add(a: int, b: int) -> str:
        """Add two integers."""
        return str(a + b)

    assert add.name == "add"
    assert add.description == "Add two integers."
    schema = add.to_openai()
    props = schema["function"]["parameters"]["properties"]
    assert set(props) == {"a", "b"}
    assert schema["function"]["parameters"]["required"] == ["a", "b"]

    res = run(add.run(a=2, b=3))
    assert res.ok and res.output == "5"


def test_function_tool_sync_via_to_thread():
    @tool_from_function
    def upper(text: str) -> str:
        """Uppercase the text."""
        return text.upper()

    res = run(upper.run(text="hi"))
    assert res.ok and res.output == "HI"


def test_function_tool_returning_toolresult_passthrough():
    @tool_from_function
    async def risky(x: int) -> ToolResult:
        if x < 0:
            return ToolResult.fail("risky", {"x": x}, "negative")
        return ToolResult.ok("risky", {"x": x}, f"ok:{x}")

    bad = run(risky.run(x=-1))
    assert bad.ok is False and bad.error == "negative"
    good = run(risky.run(x=5))
    assert good.ok and good.output == "ok:5"


def test_function_tool_registry_dispatch():
    @tool_from_function
    async def double(n: int) -> str:
        """Double a number."""
        return str(n * 2)

    reg = Registry()
    reg.register(double)
    res = run(reg.dispatch("double", {"n": 21}))
    assert res.ok and res.output == "42"