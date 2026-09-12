"""Built-in tools: shell, file read/write, web fetch, calculator."""
from __future__ import annotations

import asyncio
import ast
import operator as op
import urllib.parse

import httpx

from agentic_router.models import ToolResult
from agentic_router.tools.base import Tool, _ok, _truncate


class ShellTool(Tool):
    name = "run_shell"
    description = (
        "Run a shell command and return stdout/stderr. Use for file inspection, "
        "git, listing directories, build steps. Avoid destructive commands."
    )
    parameters = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "The shell command to run."},
            "timeout": {
                "type": "number",
                "description": "Timeout in seconds (default 30).",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    async def run(self, command: str, timeout: float = 30.0) -> ToolResult:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            proc.kill()
            return ToolResult(
                name=self.name, arguments={"command": command}, output="",
                ok=False, error=f"timed out after {timeout}s",
            )
        out = (stdout or b"").decode(errors="replace")
        err = (stderr or b"").decode(errors="replace")
        body = f"[exit={proc.returncode}]\n--- stdout ---\n{out}\n--- stderr ---\n{err}"
        return _ok(self.name, {"command": command}, _truncate(body))


class FileReadTool(Tool):
    name = "read_file"
    description = "Read a text file from the local filesystem."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "max_bytes": {"type": "integer", "default": 200000},
        },
        "required": ["path"],
    }

    async def run(self, path: str, max_bytes: int = 200000) -> ToolResult:
        try:
            with open(path, "rb") as f:
                data = f.read(max_bytes)
            return _ok(self.name, {"path": path}, _truncate(data.decode(errors="replace")))
        except FileNotFoundError:
            return ToolResult(
                name=self.name, arguments={"path": path}, output="",
                ok=False, error="file not found",
            )


class FileWriteTool(Tool):
    name = "write_file"
    description = "Write text content to a local file (overwrites)."
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
        },
        "required": ["path", "content"],
    }

    async def run(self, path: str, content: str) -> ToolResult:
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)
            return _ok(self.name, {"path": path}, f"wrote {len(content)} chars")
        except OSError as e:
            return ToolResult(
                name=self.name, arguments={"path": path}, output="",
                ok=False, error=str(e),
            )


class WebFetchTool(Tool):
    name = "web_fetch"
    description = "Fetch a URL and return the raw response body (truncated)."
    parameters = {
        "type": "object",
        "properties": {"url": {"type": "string"}},
        "required": ["url"],
    }

    async def run(self, url: str) -> ToolResult:
        if not urllib.parse.urlparse(url).scheme:
            return ToolResult(
                name=self.name, arguments={"url": url}, output="",
                ok=False, error="url must include a scheme",
            )
        try:
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                resp = await client.get(url)
                body = resp.text
            return _ok(
                self.name, {"url": url},
                _truncate(f"[{resp.status_code}]\n{body}"),
            )
        except httpx.HTTPError as e:
            return ToolResult(
                name=self.name, arguments={"url": url}, output="",
                ok=False, error=str(e),
            )


# --- Safe arithmetic calculator (no eval) ---
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
        try:
            tree = ast.parse(expression.strip(), mode="eval")
            value = _safe_eval(tree)
            return _ok(self.name, {"expression": expression}, repr(value))
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                name=self.name, arguments={"expression": expression}, output="",
                ok=False, error=f"bad expression: {e}",
            )


def build_default_registry() -> "Registry":  # type: ignore[name-defined]
    from agentic_router.tools.base import Registry
    reg = Registry()
    for t in (
        ShellTool(), FileReadTool(), FileWriteTool(),
        WebFetchTool(), CalculatorTool(),
    ):
        reg.register(t)
    return reg