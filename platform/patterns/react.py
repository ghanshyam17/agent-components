"""ReAct Pattern Implementation.

Wraps the existing ReAct loop from the agentic_router component into the Pattern Library interface.
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from agentic_router.factory import build_agent
from agentic_router.models import AgentEvent
from platform.patterns.base import AgentPattern, PatternContext, PatternResult, PatternFactory

logger = logging.getLogger(__name__)


class ReActPattern(AgentPattern):
    """A pattern that implements the ReAct (Reasoning and Acting) loop."""

    def __init__(self, **kwargs: Any):
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "react"

    async def run(self, context: PatternContext) -> PatternResult:
        """Run the ReAct pattern non-streaming."""
        start_time = time.time()
        
        # Build configuration for agent factory
        config = {
            "agent_max_iterations": context.config.get("max_iterations", 10),
            "agent_enable_planning": context.config.get("enable_planning", True),
            "agent_enable_tools": context.config.get("enable_tools", True),
        }
        
        # Build agent
        agent = build_agent()
        # Override config settings on the built agent instance if possible
        if hasattr(agent, "s"):
            agent.s.agent_max_iterations = config["agent_max_iterations"]
            agent.s.agent_enable_planning = config["agent_enable_planning"]
            agent.s.agent_enable_tools = config["agent_enable_tools"]
        
        # Collect events manually using run wrapper
        final_answer, route_decision, final_state = await agent.run(
            task=context.task, session_id=context.session_id
        )
        
        duration_ms = int((time.time() - start_time) * 1000)
        
        return PatternResult(
            final_answer=final_answer,
            metadata={"route_decision": route_decision.model_dump() if route_decision else None},
            events_log=[],  # To get events, use stream() directly
            duration_ms=duration_ms,
        )

    async def stream(self, context: PatternContext) -> AsyncIterator[AgentEvent]:
        """Stream the execution of the ReAct pattern."""
        config = {
            "agent_max_iterations": context.config.get("max_iterations", 10),
            "agent_enable_planning": context.config.get("enable_planning", True),
            "agent_enable_tools": context.config.get("enable_tools", True),
        }
        
        agent = build_agent()
        if hasattr(agent, "s"):
            agent.s.agent_max_iterations = config["agent_max_iterations"]
            agent.s.agent_enable_planning = config["agent_enable_planning"]
            agent.s.agent_enable_tools = config["agent_enable_tools"]
            
        async for event in agent.stream(task=context.task, session_id=context.session_id):
            yield event

    def to_foundry_config(self) -> dict[str, Any]:
        base = super().to_foundry_config()
        base.update(self.kwargs)
        return base

# Auto-register
PatternFactory.register("react", ReActPattern)
