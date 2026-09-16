"""ReAct Pattern Implementation.

Wraps the existing ReAct loop from the agentic_router component into the Pattern
Library interface.

By default the pattern builds its own agent from `agentic_router.factory`'s
ambient settings. That is usually wrong for a *deployed* graph: the caller has
already built an agent wired to a specific endpoint (e.g. the Foundry project
with per-request Entra auth), and rebuilding from defaults would silently point
at ``localhost:8001``. Pass ``agent_factory`` to reuse the caller's agent.
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator, Callable

from agentic_router.factory import build_agent
from agentic_router.models import AgentEvent
from platform.patterns.base import AgentPattern, PatternContext, PatternResult, PatternFactory

logger = logging.getLogger(__name__)


class ReActPattern(AgentPattern):
    """A pattern that implements the ReAct (Reasoning and Acting) loop.

    Parameters
    ----------
    agent_factory:
        Zero-argument callable returning a configured agent. When omitted, the
        pattern falls back to ``build_agent()`` with ambient settings.
    max_iterations / enable_planning / enable_tools:
        Per-pattern loop overrides applied to the agent's settings after build.
    """

    def __init__(
        self,
        agent_factory: Callable[[], Any] | None = None,
        max_iterations: int | None = None,
        enable_planning: bool | None = None,
        enable_tools: bool | None = None,
        **kwargs: Any,
    ):
        self.agent_factory = agent_factory
        self.max_iterations = max_iterations
        self.enable_planning = enable_planning
        self.enable_tools = enable_tools
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "react"

    def _build(self, context: PatternContext) -> Any:
        """Build the agent: prefer the injected factory, else ambient settings."""
        if self.agent_factory is not None:
            agent = self.agent_factory()
        else:
            logger.warning(
                "ReActPattern has no agent_factory; building from ambient settings "
                "(this points at the default local vLLM endpoints)"
            )
            agent = build_agent()

        # Loop overrides: context wins over constructor wins over agent default.
        cfg = context.config or {}
        max_iterations = cfg.get("max_iterations", self.max_iterations)
        enable_planning = cfg.get("enable_planning", self.enable_planning)
        enable_tools = cfg.get("enable_tools", self.enable_tools)
        s = getattr(agent, "s", None)
        if s is not None:
            if max_iterations is not None:
                s.agent_max_iterations = max_iterations
            if enable_planning is not None:
                s.agent_enable_planning = enable_planning
            if enable_tools is not None:
                s.agent_enable_tools = enable_tools
        return agent

    async def run(self, context: PatternContext) -> PatternResult:
        """Run the ReAct pattern non-streaming."""
        start_time = time.time()
        agent = self._build(context)

        final_answer, route_decision, _final_state = await agent.run(
            task=context.task, session_id=context.session_id
        )

        duration_ms = int((time.time() - start_time) * 1000)

        return PatternResult(
            final_answer=final_answer,
            metadata={
                "route_decision": route_decision.model_dump() if route_decision else None
            },
            events_log=[],  # To get events, use stream() directly
            duration_ms=duration_ms,
        )

    async def stream(self, context: PatternContext) -> AsyncIterator[AgentEvent]:
        """Stream the execution of the ReAct pattern."""
        agent = self._build(context)
        async for event in agent.stream(task=context.task, session_id=context.session_id):
            yield event

    def to_foundry_config(self) -> dict[str, Any]:
        base = super().to_foundry_config()
        base.update(self.kwargs)
        if self.max_iterations is not None:
            base["max_iterations"] = self.max_iterations
        return base


# Auto-register
PatternFactory.register("react", ReActPattern)
