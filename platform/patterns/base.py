"""Base abstractions for the Agentic Design Pattern Library.

Provides the foundational classes and registries for building and composing
agent patterns in the platform.
"""
from __future__ import annotations

import abc
import time
from typing import Any, AsyncIterator, Type

from pydantic import BaseModel, Field

from components_core.models import SessionState
from agentic_router.models import AgentEvent
from toolkit.base import Registry


class PatternResult(BaseModel):
    """The result of a pattern execution."""

    final_answer: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    events_log: list[AgentEvent] = Field(default_factory=list)
    token_usage: dict[str, int] = Field(default_factory=dict)
    duration_ms: int = 0


class PatternContext(BaseModel):
    """Context passed to a pattern during execution."""

    task: str
    session_id: str
    config: dict[str, Any] = Field(default_factory=dict)
    tools: Registry | None = None
    memory: SessionState | None = None
    model_clients: dict[str, Any] = Field(default_factory=dict)
    
    # We allow arb objects like tools which might not validate cleanly under pydantic
    model_config = {"arbitrary_types_allowed": True}


class AgentPattern(abc.ABC):
    """Abstract base class for all agent patterns."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Name of the pattern."""
        ...

    @abc.abstractmethod
    async def run(self, context: PatternContext) -> PatternResult:
        """Run the pattern to completion and return the final result."""
        ...

    @abc.abstractmethod
    async def stream(self, context: PatternContext) -> AsyncIterator[AgentEvent]:
        """Stream the execution of the pattern as a sequence of events."""
        ...

    def to_foundry_config(self) -> dict[str, Any]:
        """Serialize the pattern configuration for Foundry deployment."""
        return {
            "pattern_name": self.name,
            "type": self.__class__.__name__,
        }


class PatternFactory:
    """Registry mapping pattern names to their implementing classes."""

    _registry: dict[str, Type[AgentPattern]] = {}

    @classmethod
    def register(cls, name: str, pattern_cls: Type[AgentPattern]) -> None:
        """Register a pattern class under a given name."""
        cls._registry[name] = pattern_cls

    @classmethod
    def create(cls, name: str, **kwargs: Any) -> AgentPattern:
        """Instantiate a pattern by name with optional kwargs."""
        if name not in cls._registry:
            raise ValueError(f"Unknown pattern: {name}")
        return cls._registry[name](**kwargs)

    @classmethod
    def available(cls) -> list[str]:
        """List all registered pattern names."""
        return list(cls._registry.keys())

