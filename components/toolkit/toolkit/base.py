"""Tool interface, registry, and the `tool_from_function` decorator.

A `Tool` is an async callable with a JSON-schema description. The registry
exposes tools to an agent loop and renders the OpenAI `tools` array.
"""
from __future__ import annotations

import abc
import asyncio
import inspect
import json
from typing import Any, Callable, Awaitable

from toolkit.result import ToolResult, UnknownToolError


# --------------------------------------------------------------------------- #
# Tool base + registry
# --------------------------------------------------------------------------- #
class Tool(abc.ABC):
    """Abstract tool.

    Subclasses set `name`, `description`, `parameters` (a JSON schema dict) and
    implement the async `run(**kwargs)`. Use `tool_from_function` for ad-hoc
    tools instead of subclassing.
    """

    name: str
    description: str
    parameters: dict[str, Any]

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...

    def to_openai(self) -> dict[str, Any]:
        """Render as an OpenAI function-calling tool entry."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Registry:
    """An ordered, named collection of tools with dispatch helpers."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def unregister(self, name: str) -> bool:
        return self._tools.pop(name, None) is not None

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __iter__(self):
        return iter(self._tools.values())

    def __len__(self) -> int:
        return len(self._tools)

    def to_openai(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        if names is None:
            return [t.to_openai() for t in self._tools.values()]
        out: list[dict[str, Any]] = []
        for n in names:
            if t := self._tools.get(n):
                out.append(t.to_openai())
        return out

    async def dispatch(self, name: str, arguments: dict[str, Any] | None = None) -> ToolResult:
        """Execute a tool by name, catching exceptions into a failed result.

        Never raises for ordinary tool failures; only `UnknownToolError` is
        encoded as a failed `ToolResult` (so agent loops can surface it).
        """
        arguments = arguments or {}
        tool = self.get(name)
        if tool is None:
            return ToolResult.fail(name, arguments, f"unknown tool: {name!r}")
        try:
            return await tool.run(**arguments)
        except Exception as e:  # noqa: BLE001 — tools must not crash the loop
            return ToolResult.fail(name, arguments, f"{type(e).__name__}: {e}")


# --------------------------------------------------------------------------- #
# Build tools from plain functions
# --------------------------------------------------------------------------- #
_PY_TO_JSON = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _schema_from_signature(sig: inspect.Signature, description: str) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    for p in sig.parameters.values():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        ann = p.annotation
        json_type = _PY_TO_JSON.get(ann if ann is not inspect.Parameter.empty else str, "string")
        prop: dict[str, Any] = {"type": json_type}
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        else:
            prop["default"] = p.default
        props[p.name] = prop
    return {
        "type": "object",
        "properties": props,
        "required": required,
    }


class FunctionTool(Tool):
    """A tool backed by an async (or sync) callable."""

    def __init__(self, fn: Callable[..., Awaitable[Any] | Any], name: str | None = None,
                 description: str | None = None, parameters: dict[str, Any] | None = None):
        self._fn = fn
        self.name = name or fn.__name__
        self.description = description or (inspect.getdoc(fn) or "").split("\n")[0] or self.name
        self.parameters = parameters or _schema_from_signature(
            inspect.signature(fn), self.description
        )
        self._is_coro = asyncio.iscoroutinefunction(fn)

    async def run(self, **kwargs: Any) -> ToolResult:
        if self._is_coro:
            result = await self._fn(**kwargs)
        else:
            result = await asyncio.to_thread(self._fn, **kwargs)
        if isinstance(result, ToolResult):
            if result.name != self.name:
                result = result.model_copy(update={"name": self.name})
            return result
        # Coerce a raw return value into a ToolResult.ok
        output = result if isinstance(result, str) else json.dumps(result, default=str)
        return ToolResult.ok(self.name, kwargs, output)


def tool_from_function(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> FunctionTool:
    """Decorator/factory turning a callable into a `FunctionTool`.

    Works bare (`@tool_from_function`) or with args (`@tool_from_function(name="x")`).
    """
    if fn is not None and callable(fn):
        return FunctionTool(fn, name=name, description=description, parameters=parameters)

    def deco(f: Callable[..., Any]) -> FunctionTool:
        return FunctionTool(f, name=name, description=description, parameters=parameters)

    return deco


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"