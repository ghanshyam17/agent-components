"""Sequential Pattern Implementation.

Sequential chain of agents, passing context forward.
"""
from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from agentic_router.models import AgentEvent
from platform.patterns.base import AgentPattern, PatternContext, PatternResult, PatternFactory

logger = logging.getLogger(__name__)


class SequentialPattern(AgentPattern):
    """A pattern that chains multiple agents sequentially."""

    def __init__(self, stages: list[AgentPattern] | None = None, **kwargs: Any):
        self.stages = stages or []
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "sequential"

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
        # Emits stage_start, stage_complete, chain_complete
        current_task = context.task
        
        for idx, stage in enumerate(self.stages):
            yield AgentEvent(type="subtask_start", data={"description": f"Stage {idx}: {stage.name}"})
            
            sub_context = PatternContext(
                task=current_task,
                session_id=f"{context.session_id}_stage_{idx}",
                config=context.config,
                tools=context.tools,
            )
            
            stage_answer = ""
            async for event in stage.stream(sub_context):
                yield event
                if event.type == "final":
                    stage_answer = event.data.get("content", "")
                    
            # Output of stage N becomes input context for stage N+1
            current_task = stage_answer
            
        yield AgentEvent(type="final", data={"content": current_task})

# Auto-register
PatternFactory.register("sequential", SequentialPattern)
