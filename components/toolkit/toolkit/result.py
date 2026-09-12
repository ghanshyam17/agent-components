"""Tool result and errors shared across the toolkit."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """Outcome of a tool execution.

    `output` is the human/LLM-facing string. `ok=False` together with `error`
    signals a failure; the loop surfaces `error` (or `output`) to the model.
    """

    name: str
    arguments: dict[str, Any]
    output: str = ""
    ok: bool = True
    error: str | None = None
    # Optional structured data a caller may inspect (not sent to the model).
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def ok(cls, name: str, arguments: dict[str, Any], output: str, **meta: Any) -> "ToolResult":
        return cls(name=name, arguments=arguments, output=output, metadata=meta)

    @classmethod
    def fail(cls, name: str, arguments: dict[str, Any], error: str, output: str = "", **meta: Any) -> "ToolResult":
        return cls(name=name, arguments=arguments, output=output, ok=False, error=error, metadata=meta)


class ToolError(Exception):
    """Raised by tool implementations for non-recoverable errors."""


class UnknownToolError(ToolError):
    def __init__(self, name: str) -> None:
        super().__init__(f"unknown tool: {name!r}")
        self.name = name