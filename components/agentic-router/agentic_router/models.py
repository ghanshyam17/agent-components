"""Router-specific data models.

Conversation types (Message, SessionState, Plan, SubTask, ToolCallRequest)
are shared across components and now live in `core`; they are re-exported here
so existing `from agentic_router.models import ...` imports keep working.

Only models specific to routing / the agent loop are defined here.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field

# Re-export shared conversation types from components_core.
from components_core import Message, Plan, SessionState, SubTask, ToolCallRequest  # noqa: F401


class ModelTier(str, Enum):
    LOWER = "lower"
    HIGHER = "higher"


class RouteDecision(BaseModel):
    """The router's verdict on which model should handle a task."""

    tier: ModelTier
    score: float = Field(ge=0.0, le=1.0, description="Heuristic complexity score")
    method: Literal["heuristic", "classifier", "override"]
    model: str
    reason: str
    signals: dict[str, float] = Field(default_factory=dict)


class ToolResult(BaseModel):
    name: str
    arguments: dict[str, Any]
    output: str
    ok: bool = True
    error: str | None = None


class AgentEvent(BaseModel):
    """One streamed event from the agent loop."""

    type: Literal[
        "route",
        "plan",
        "subtask_start",
        "delta",  # streamed token chunk
        "tool_call",
        "tool_result",
        "iteration",
        "answer",  # per-subtask answer (non-final in a multi-step plan)
        "final",   # overall combined answer
        "error",
    ]
    data: dict[str, Any] = Field(default_factory=dict)