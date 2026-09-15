"""Agentic Design Pattern Library.

Exports all available patterns and the base abstractions.
"""
from __future__ import annotations

from platform.patterns.base import AgentPattern, PatternContext, PatternFactory, PatternResult
from platform.patterns.map_reduce import MapReducePattern
from platform.patterns.network import NetworkAgent, NetworkPattern, MessageBus, InMemoryBus
from platform.patterns.react import ReActPattern
from platform.patterns.sequential import SequentialPattern
from platform.patterns.supervisor import SupervisorPattern
from platform.patterns.autogen import AutoGenPattern
from platform.patterns.langgraph import LangGraphPattern
from platform.patterns.framework_router import FrameworkRouterPattern

__all__ = [
    "AgentPattern",
    "PatternContext",
    "PatternFactory",
    "PatternResult",
    "MapReducePattern",
    "NetworkAgent",
    "NetworkPattern",
    "MessageBus",
    "InMemoryBus",
    "ReActPattern",
    "SequentialPattern",
    "SupervisorPattern",
    "AutoGenPattern",
    "LangGraphPattern",
    "FrameworkRouterPattern",
]

