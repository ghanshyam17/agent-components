"""Map-Reduce Pattern Implementation.

Fanning out tasks to workers concurrently and reducing the results.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from agentic_router.models import AgentEvent
from platform.patterns.base import AgentPattern, PatternContext, PatternResult, PatternFactory

logger = logging.getLogger(__name__)


class MapReducePattern(AgentPattern):
    """A pattern that applies map-reduce strategy for concurrent processing."""

    def __init__(self, mapper: AgentPattern | None = None, reducer: AgentPattern | None = None, **kwargs: Any):
        self.mapper = mapper
        self.reducer = reducer
        self.max_parallel = kwargs.get("max_parallel", 5)
        self.chunk_strategy = kwargs.get("chunk_strategy", "split_by_lines")
        self.kwargs = kwargs

    @property
    def name(self) -> str:
        return "map_reduce"

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
        # Emits map_start, worker_result, reduce_start, reduce_complete
        yield AgentEvent(type="subtask_start", data={"description": "Starting Map-Reduce"})
        
        if not self.mapper or not self.reducer:
            yield AgentEvent(type="error", data={"error": "Mapper or Reducer not configured"})
            return
            
        # Simplified chunking
        chunks = [context.task] * 3 if context.task else []
        
        async def run_mapper(idx: int, chunk: str) -> str:
            sub_context = PatternContext(
                task=chunk,
                session_id=f"{context.session_id}_map_{idx}",
                config=context.config,
            )
            result = ""
            async for ev in self.mapper.stream(sub_context):
                if ev.type == "final":
                    result = ev.data.get("content", "")
            return result

        # Map phase
        tasks = [run_mapper(i, chunk) for i, chunk in enumerate(chunks)]
        map_results = await asyncio.gather(*tasks)
        
        # Reduce phase
        reduce_task = "\\n".join(map_results)
        sub_context = PatternContext(
            task=reduce_task,
            session_id=f"{context.session_id}_reduce",
            config=context.config,
        )
        
        final_answer = ""
        async for ev in self.reducer.stream(sub_context):
            yield ev
            if ev.type == "final":
                final_answer = ev.data.get("content", "")
                
        yield AgentEvent(type="final", data={"content": final_answer})

# Auto-register
PatternFactory.register("map_reduce", MapReducePattern)
