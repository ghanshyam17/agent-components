"""Framework First-Router Pattern Implementation.

Routes incoming tasks to either AutoGen, LangGraph, or ReAct as first-class routes
based on task characteristics, intent signals, and policy rules.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, AsyncIterator, Dict, List, Literal, Optional

from agentic_router.models import AgentEvent
from platform.patterns.base import (
    AgentPattern,
    PatternContext,
    PatternFactory,
    PatternResult,
)
from platform.patterns.autogen import AutoGenPattern
from platform.patterns.langgraph import LangGraphPattern
from platform.patterns.react import ReActPattern

logger = logging.getLogger(__name__)

# Heuristic intent matchers for first-route selection
AUTOGEN_SIGNALS = (
    "debate", "critique", "discuss", "collaborate", "code review",
    "two agents", "group chat", "multi-agent conversation", "pair programming",
    "brainstorm", "peer review", "panel", "perspective"
)

LANGGRAPH_SIGNALS = (
    "workflow", "state machine", "pipeline", "graph", "step by step",
    "conditional", "retry on failure", "approval", "human in the loop",
    "checkpoint", "cycle", "loop until", "flowchart", "dag", "stage"
)


class FrameworkRouterPattern(AgentPattern):
    """First-route router that directs tasks to AutoGen, LangGraph, or ReAct."""

    def __init__(
        self,
        default_route: Literal["autogen", "langgraph", "react"] = "react",
        strategy: str = "intent_classifier",
        routes: Optional[List[Dict[str, str]]] = None,
        autogen_config: Optional[Dict[str, Any]] = None,
        langgraph_config: Optional[Dict[str, Any]] = None,
        react_config: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ):
        self.default_route = default_route
        self.strategy = strategy
        self.routes = routes or []
        self.kwargs = kwargs

        # Target framework pattern instances
        self.autogen_pattern = AutoGenPattern(**(autogen_config or {}))
        self.langgraph_pattern = LangGraphPattern(**(langgraph_config or {}))
        self.react_pattern = ReActPattern(**(react_config or {}))

    @property
    def name(self) -> str:
        return "framework_router"

    def classify_first_route(self, task: str) -> tuple[str, str]:
        """Classify which framework route to use first based on task intent."""
        task_lower = task.lower()

        # Check explicit custom rules first
        for rule in self.routes:
            intent = rule.get("intent", "").lower()
            if intent and intent in task_lower:
                target = rule.get("target_framework", self.default_route)
                return target, f"matched_custom_rule_{intent}"

        # 1. Check for AutoGen conversational/debate signals
        if any(signal in task_lower for signal in AUTOGEN_SIGNALS):
            return "autogen", "task_requires_multi_agent_collaboration_or_debate"

        # 2. Check for LangGraph cyclical stateful workflow signals
        if any(signal in task_lower for signal in LANGGRAPH_SIGNALS):
            return "langgraph", "task_requires_cyclical_state_graph_workflow"

        # 3. Default route (typically ReAct single-agent loop)
        return self.default_route, "default_framework_route"

    async def run(self, context: PatternContext) -> PatternResult:
        """Evaluate first-route and execute target framework pattern."""
        start_time = time.time()
        events: List[AgentEvent] = []
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
        """Stream first-route decision and subsequent framework execution."""
        selected_route, reason = self.classify_first_route(context.task)

        # Emit the first-route event
        yield AgentEvent(
            type="route",
            data={
                "first_route": selected_route,
                "strategy": self.strategy,
                "reason": reason,
                "available_routes": ["autogen", "langgraph", "react"],
            },
        )

        # Delegate execution to the routed framework pattern
        if selected_route == "autogen":
            async for event in self.autogen_pattern.stream(context):
                yield event
        elif selected_route == "langgraph":
            async for event in self.langgraph_pattern.stream(context):
                yield event
        else:
            async for event in self.react_pattern.stream(context):
                yield event

    def to_foundry_config(self) -> Dict[str, Any]:
        """Export config for Foundry routing endpoint."""
        return {
            "pattern_name": self.name,
            "default_route": self.default_route,
            "strategy": self.strategy,
            "supported_routes": ["autogen", "langgraph", "react"],
            "autogen": self.autogen_pattern.to_foundry_config(),
            "langgraph": self.langgraph_pattern.to_foundry_config(),
            "react": self.react_pattern.to_foundry_config(),
        }


# Auto-register pattern
PatternFactory.register("framework_router", FrameworkRouterPattern)
