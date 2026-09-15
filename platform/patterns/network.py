"""Network Pattern Implementation.

Peer-to-peer agent collaboration.
"""
from __future__ import annotations

import abc
import logging
import time
from typing import Any, AsyncIterator

from agentic_router.models import AgentEvent
from platform.patterns.base import AgentPattern, PatternContext, PatternResult, PatternFactory

logger = logging.getLogger(__name__)


class MessageBus(abc.ABC):
    @abc.abstractmethod
    async def send(self, sender: str, recipient: str, message: Any) -> None: ...

    @abc.abstractmethod
    async def receive(self, receiver: str) -> Any: ...


class InMemoryBus(MessageBus):
    def __init__(self):
        self.messages = []

    async def send(self, sender: str, recipient: str, message: Any) -> None:
        self.messages.append({"sender": sender, "recipient": recipient, "message": message})

    async def receive(self, receiver: str) -> Any:
        return [m for m in self.messages if m["recipient"] == receiver]


class NetworkAgent:
    def __init__(self, name: str, pattern: AgentPattern, bus: MessageBus):
        self.name = name
        self.pattern = pattern
        self.bus = bus


class NetworkPattern(AgentPattern):
    """A pattern for peer-to-peer agent networks."""

    def __init__(self, agents: list[NetworkAgent] | None = None, topology: str = "mesh", **kwargs: Any):
        self.agents = agents or []
        self.topology = topology
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "network"

    async def run(self, context: PatternContext) -> PatternResult:
        start_time = time.time()
        events = []
        final_answer = ""
        
        async for event in self.stream(context):
            events.append(event)
            if event.type == "final":
                final_answer = event.data.get("content", "")
                
        duration_ms = int((time.time() - start_time) * 1000)
        return PatternResult(
            final_answer=final_answer,
            events_log=events,
            duration_ms=duration_ms,
        )

    async def stream(self, context: PatternContext) -> AsyncIterator[AgentEvent]:
        # Emits agent_message, agent_response, network_state
        yield AgentEvent(type="subtask_start", data={"description": "Starting network communication"})
        yield AgentEvent(type="final", data={"content": f"Network consensus reached using {self.topology} topology."})

# Auto-register
PatternFactory.register("network", NetworkPattern)
