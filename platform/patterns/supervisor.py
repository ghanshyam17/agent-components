"""Supervisor Pattern Implementation.

A hub agent orchestrates specialized workers to complete a complex task.
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from agentic_router.models import AgentEvent
from platform.patterns.base import AgentPattern, PatternContext, PatternResult, PatternFactory

logger = logging.getLogger(__name__)


class SupervisorPattern(AgentPattern):
    """A pattern that orchestrates specialized workers via a supervisor."""

    def __init__(self, workers: dict[str, AgentPattern] | None = None, strategy: str = "plan_and_delegate", **kwargs: Any):
        self.workers = workers or {}
        self.strategy = strategy
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "supervisor"

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
        # Emits supervisor_plan, worker_start, worker_complete, synthesis
        # Simplified mock implementation
        yield AgentEvent(type="subtask_start", data={"description": "Supervising task"})
        
        if self.strategy == "plan_and_delegate":
            # 1. Plan
            plan = {"subtasks": [{"description": context.task}]}
            yield AgentEvent(type="plan", data={"plan": plan})
            
            # 2. Delegate
            for worker_name, worker in self.workers.items():
                yield AgentEvent(type="route", data={"decision": {"tier": "supervisor", "route": worker_name}})
                
                # Create sub-context
                sub_context = PatternContext(
                    task=context.task,
                    session_id=f"{context.session_id}_{worker_name}",
                    config=context.config,
                )
                
                async for event in worker.stream(sub_context):
                    yield event
                    
            # 3. Synthesis
            yield AgentEvent(type="final", data={"content": f"Supervisor complete using {self.strategy}."})

        else:
            yield AgentEvent(type="error", data={"stage": "supervisor", "error": "Unsupported strategy"})

# Auto-register
PatternFactory.register("supervisor", SupervisorPattern)
