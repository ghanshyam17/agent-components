"""Shared data models for agent-components.

These types — conversation messages, session state and plans — are used by
multiple components (the router, memory-store, future retriever/guardrails),
so they live in `core` rather than in any single component. Components import
them from here and may re-export for convenience.
"""
from __future__ import annotations

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolCallRequest(BaseModel):
    """A model's request to invoke a tool."""

    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(BaseModel):
    """A single chat message in the OpenAI shape (roles + optional tool calls)."""

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_calls: list[ToolCallRequest] | None = None
    tool_call_id: str | None = None
    name: str | None = None


class SubTask(BaseModel):
    description: str
    done: bool = False


class Plan(BaseModel):
    goal: str
    subtasks: list[SubTask] = Field(default_factory=list)


class SessionState(BaseModel):
    """A persisted conversation: messages + optional plan + metadata."""

    session_id: str
    messages: list[Message] = Field(default_factory=list)
    plan: Plan | None = None
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def touch(self) -> None:
        self.updated_at = time.time()