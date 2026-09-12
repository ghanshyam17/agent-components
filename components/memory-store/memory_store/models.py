"""Memory-store models.

Short-term memory reuses the shared conversation types from `core`. Long-term
(vector) memory introduces `MemoryRecord` — a stored item with content,
metadata and, when retrieved, a relevance score.
"""
from __future__ import annotations

import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from components_core import Message, Plan, SessionState, SubTask, ToolCallRequest

# Re-export the short-term types so callers can import everything from
# `memory_store` without depending on `components_core` directly.
__all__ = [
    "Message", "Plan", "SessionState", "SubTask", "ToolCallRequest",
    "MemoryRecord", "MemoryQuery",
]


class MemoryRecord(BaseModel):
    """An item in long-term (vector) memory."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: float = Field(default_factory=time.time)


class MemoryQuery(BaseModel):
    """A retrieval request for long-term memory."""

    query: str
    top_k: int = 4
    filter: dict[str, Any] | None = None
    min_score: float = 0.0