"""toolkit — a composable, hardened tool-calling framework for AI agents.

Public API:
    Tool, Registry, ToolResult, tool_from_function, FunctionTool
    built-in tools (ShellTool, FileReadTool, FileWriteTool, WebFetchTool,
    HTTPRequestTool, CalculatorTool) and `build_registry()`.
"""
from __future__ import annotations

from toolkit.base import (
    FunctionTool,
    Registry,
    Tool,
    tool_from_function,
)
from toolkit.builtin import (
    CalculatorTool,
    FileReadTool,
    FileWriteTool,
    HTTPRequestTool,
    ShellTool,
    WebFetchTool,
    default_tools,
)
from toolkit.config import ToolkitSettings, build_registry
from toolkit.result import ToolError, ToolResult, UnknownToolError
from toolkit.sandbox import SandboxError, is_command_allowed, resolve_within

__all__ = [
    "Tool",
    "FunctionTool",
    "Registry",
    "ToolResult",
    "ToolError",
    "UnknownToolError",
    "tool_from_function",
    "ShellTool",
    "FileReadTool",
    "FileWriteTool",
    "WebFetchTool",
    "HTTPRequestTool",
    "CalculatorTool",
    "default_tools",
    "ToolkitSettings",
    "build_registry",
    "SandboxError",
    "resolve_within",
    "is_command_allowed",
]