"""Tool interface and registry.

Tools are simple async callables with a JSON-schema description. The registry
exposes them to the agent loop and can render the OpenAI `tools` array.
"""
from __future__ import annotations

import abc
import shlex
from typing import Any, Callable, Awaitable

from agentic_router.models import ToolResult


class Tool(abc.ABC):
    name: str
    description: str
    parameters: dict[str, Any]  # JSON schema

    @abc.abstractmethod
    async def run(self, **kwargs: Any) -> ToolResult: ...

    def to_openai(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class Registry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def to_openai(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        if names is None:
            return [t.to_openai() for t in self._tools.values()]
        out = []
        for n in names:
            if t := self._tools.get(n):
                out.append(t.to_openai())
        return out

    async def dispatch(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            return ToolResult(
                name=name, arguments=arguments, output="", ok=False,
                error=f"unknown tool: {name}",
            )
        try:
            return await tool.run(**arguments)
        except Exception as e:  # noqa: BLE001
            return ToolResult(
                name=name, arguments=arguments, output="", ok=False,
                error=f"{type(e).__name__}: {e}",
            )


def _ok(name: str, arguments: dict[str, Any], output: str) -> ToolResult:
    return ToolResult(name=name, arguments=arguments, output=output)


def _truncate(text: str, limit: int = 8000) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"