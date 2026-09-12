"""Built-in tools: shell, file read/write, web fetch, calculator, http request.

Each is hardened with the sandbox helpers and configurable limits. All
return `ToolResult` instances and never raise on ordinary failure.
"""
from __future__ import annotations

import ast
import asyncio
import json
import operator as op
import urllib.parse
from typing import Any, Callable

import httpx

from toolkit.base import Tool, _truncate
from toolkit.result import ToolResult
from toolkit.sandbox import is_command_allowed, resolve_within, SandboxError


# --------------------------------------------------------------------------- #
# Shell
# --------------------------------------------------------------------------- #
class ShellTool(Tool):
    name = "run_shell"
    description = (
        "Run a shell command and return stdout/stderr. Use for file "
        "inspection, git, listing directories, build steps. Avoid destructive "
        "commands."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {"type": "number", "description": "Timeout in seconds (default 30).", "default": 30},
            "cwd": {"type": "string", "description": "Working directory (default current)."},
        },
        "required": ["command"],
    }

    def __init__(self, *, allow: list[str] | None = None, deny: list[str] | None = None,
                 default_timeout: float = 30.0, max_output: int = 8000) -> None:
        self.allow = list(allow) if allow else []
        self.deny = list(deny) if deny else ["rm -rf /", "sudo"]
        self.default_timeout = default_timeout
        self.max_output = max_output

    async def run(self, command: str, timeout: float | None = None, cwd: str | None = None) -> ToolResult:
        args = {"command": command}
        timeout = timeout if timeout is not None else self.default_timeout
        if not is_command_allowed(command, allow=self.allow, deny=self.deny):
            return ToolResult.fail(self.name, args, "command denied by sandbox")
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
        except OSError as e:
            return ToolResult.fail(self.name, args, f"spawn failed: {e}")
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult.fail(self.name, args, f"timed out after {timeout}s")
        out = (stdout or b"").decode(errors="replace")
        err = (stderr or b"").decode(errors="replace")
        body = f"[exit={proc.returncode}]\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
        return ToolResult.ok(self.name, args, _truncate(body, self.max_output), exit=proc.returncode)


# --------------------------------------------------------------------------- #
# File read / write
# --------------------------------------------------------------------------- #
class FileReadTool(Tool):
    name = "read_file"
    description = "Read a text file from the local filesystem (sandboxed to a root)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file (relative to sandbox root)."},
            "max_bytes": {"type": "integer", "default": 200000},
        },
        "required": ["path"],
    }

    def __init__(self, root: str | None = None, max_bytes: int = 200_000) -> None:
        self.root = root
        self.max_bytes = max_bytes

    async def run(self, path: str, max_bytes: int | None = None) -> ToolResult:
        args = {"path": path}
        limit = max_bytes if max_bytes is not None else self.max_bytes
        try:
            real = resolve_within(path, self.root)
        except SandboxError as e:
            return ToolResult.fail(self.name, args, str(e))
        try:
            with open(real, "rb") as f:
                data = f.read(limit)
        except FileNotFoundError:
            return ToolResult.fail(self.name, args, "file not found")
        except OSError as e:
            return ToolResult.fail(self.name, args, str(e))
        return ToolResult.ok(self.name, args, _truncate(data.decode(errors="replace")))


class FileWriteTool(Tool):
    name = "write_file"
    description = "Write text content to a local file (overwrites; sandboxed)."
    parameters = {
        "type": "object",
        "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"],
    }

    def __init__(self, root: str | None = None) -> None:
        self.root = root

    async def run(self, path: str, content: str) -> ToolResult:
        args = {"path": path, "content_len": len(content)}
        try:
            real = resolve_within(path, self.root)
        except SandboxError as e:
            return ToolResult.fail(self.name, args, str(e))
        try:
            real.parent.mkdir(parents=True, exist_ok=True)
            with open(real, "w", encoding="utf-8") as f:
                f.write(content)
        except OSError as e:
            return ToolResult.fail(self.name, args, str(e))
        return ToolResult.ok(self.name, args, f"wrote {len(content)} chars to {real}")


# --------------------------------------------------------------------------- #
# Web fetch + generic http request
# --------------------------------------------------------------------------- #
class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return the raw response body (truncated)."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    def __init__(self, timeout: float = 20.0, max_output: int = 8000,
                 client_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self._client_factory = client_factory

    def _client(self) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(timeout=self.timeout, follow_redirects=True)

    async def run(self, url: str) -> ToolResult:
        args = {"url": url}
        if not urllib.parse.urlparse(url).scheme:
            return ToolResult.fail(self.name, args, "url must include a scheme")
        try:
            async with self._client() as client:
                resp = await client.get(url)
                body = resp.text
        except httpx.HTTPError as e:
            return ToolResult.fail(self.name, args, str(e))
        return ToolResult.ok(self.name, args, _truncate(f"[{resp.status_code}]\n{body}", self.max_output),
                             status=resp.status_code)


class HTTPRequestTool(Tool):
    name = "http_request"
    description = "Perform an HTTP request and return status, headers, and body."
    parameters = {
        "type": "object",
        "properties": {
            "url": {"type": "string"},
            "method": {"type": "string", "default": "GET",
                       "description": "HTTP method (GET/POST/PUT/DELETE/...)."},
            "headers": {"type": "object", "default": {}},
            "body": {"type": "string", "description": "Request body (raw)."},
            "timeout": {"type": "number", "default": 20},
        },
        "required": ["url"],
    }

    def __init__(self, timeout: float = 20.0, max_output: int = 8000,
                 client_factory: Callable[[], httpx.AsyncClient] | None = None) -> None:
        self.timeout = timeout
        self.max_output = max_output
        self._client_factory = client_factory

    def _client(self, timeout: float) -> httpx.AsyncClient:
        if self._client_factory is not None:
            return self._client_factory()
        return httpx.AsyncClient(timeout=timeout, follow_redirects=True)

    async def run(self, url: str, method: str = "GET", headers: dict[str, str] | None = None,
                  body: str | None = None, timeout: float | None = None) -> ToolResult:
        args = {"url": url, "method": method}
        if not urllib.parse.urlparse(url).scheme:
            return ToolResult.fail(self.name, args, "url must include a scheme")
        to = timeout if timeout is not None else self.timeout
        try:
            async with self._client(to) as client:
                resp = await client.request(method, url, headers=headers or {}, content=body)
        except httpx.HTTPError as e:
            return ToolResult.fail(self.name, args, str(e))
        out = json.dumps({
            "status": resp.status_code,
            "headers": dict(resp.headers),
            "body": resp.text,
        })
        return ToolResult.ok(self.name, args, _truncate(out, self.max_output), status=resp.status_code)


# --------------------------------------------------------------------------- #
# Safe arithmetic calculator (no eval)
# --------------------------------------------------------------------------- #
_OPS = {
    ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv,
    ast.Pow: op.pow, ast.Mod: op.mod, ast.USub: op.neg, ast.UAdd: op.pos,
    ast.FloorDiv: op.floordiv,
}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a single arithmetic expression (e.g. '12*(3+4)/2')."
    parameters = {
        "type": "object",
        "properties": {"expression": {"type": "string"}},
        "required": ["expression"],
    }

    async def run(self, expression: str) -> ToolResult:
        args = {"expression": expression}
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            value = _safe_eval(tree)
        except Exception as e:  # noqa: BLE001
            return ToolResult.fail(self.name, args, f"bad expression: {e}")
        return ToolResult.ok(self.name, args, repr(value))


def default_tools(**overrides: Any) -> list[Tool]:
    """Build the standard built-in tool set.

    Recognised overrides: `sandbox_root`, `shell_allow`, `shell_deny`,
    `shell_timeout`, `max_output`, `file_max_bytes`.
    """
    root = overrides.get("sandbox_root")
    return [
        ShellTool(
            allow=overrides.get("shell_allow"),
            deny=overrides.get("shell_deny"),
            default_timeout=overrides.get("shell_timeout", 30.0),
            max_output=overrides.get("max_output", 8000),
        ),
        FileReadTool(root=root, max_bytes=overrides.get("file_max_bytes", 200_000)),
        FileWriteTool(root=root),
        WebFetchTool(max_output=overrides.get("max_output", 8000)),
        HTTPRequestTool(max_output=overrides.get("max_output", 8000)),
        CalculatorTool(),
    ]