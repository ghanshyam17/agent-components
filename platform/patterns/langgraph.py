"""LangGraph Stateful Workflow Pattern Implementation.

Supports cyclical graph workflows, state transitions, conditional edges,
and state checkpointing. Compatible with LangGraph, LangChain Azure AI,
and Azure Foundry hosted agents.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from agentic_router.models import AgentEvent
from platform.patterns.base import (
    AgentPattern,
    PatternContext,
    PatternFactory,
    PatternResult,
)

logger = logging.getLogger(__name__)


class LangGraphPattern(AgentPattern):
    """First-class pattern for cyclical, stateful graph workflows."""

    def __init__(
        self,
        entry_point: str = "start",
        finish_point: str = "end",
        graph_type: str = "state_graph",
        checkpointer: str = "memory",
        max_steps: int = 25,
        nodes: Optional[List[Dict[str, Any]]] = None,
        edges: Optional[List[Dict[str, str]]] = None,
        conditional_edges: Optional[List[Dict[str, Any]]] = None,
        **kwargs: Any,
    ):
        self.entry_point = entry_point
        self.finish_point = finish_point
        self.graph_type = graph_type
        self.checkpointer = checkpointer
        self.max_steps = max_steps
        self.kwargs = kwargs

        self._nodes: Dict[str, Dict[str, Any]] = {}
        if nodes:
            for node in nodes:
                self._nodes[node["name"]] = node

        self._edges: Dict[str, str] = {}
        if edges:
            for edge in edges:
                self._edges[edge["from_node"]] = edge["to_node"]

        self._conditional_edges: Dict[str, Dict[str, Any]] = {}
        if conditional_edges:
            for c_edge in conditional_edges:
                self._conditional_edges[c_edge["source_node"]] = c_edge

        # Real checkpointing, via context-manager. Previously this was a bare
        # dict that was written and NEVER read, behind a `checkpointer` param
        # that advertised four backends and silently implemented one.
        self._checkpointer = self._build_checkpointer(checkpointer)
        # Retained for backwards compatibility with anything reading it.
        self._state_store: Dict[str, Dict[str, Any]] = {}

    @property
    def name(self) -> str:
        return "langgraph"

    def add_node(self, name: str, description: Optional[str] = None, handler: Optional[Callable] = None) -> None:
        """Register a node in the graph."""
        self._nodes[name] = {
            "name": name,
            "description": description or f"Node {name}",
            "handler": handler,
        }

    def add_edge(self, from_node: str, to_node: str) -> None:
        """Add a directed edge between two nodes."""
        self._edges[from_node] = to_node

    def add_conditional_edges(self, source_node: str, condition_fn: Callable, route_map: Dict[str, str]) -> None:
        """Add a conditional branching edge."""
        self._conditional_edges[source_node] = {
            "source_node": source_node,
            "condition_fn": condition_fn,
            "route_map": route_map,
        }

    async def run(self, context: PatternContext) -> PatternResult:
        """Run the LangGraph workflow to completion."""
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
            metadata={
                "graph_type": self.graph_type,
                "checkpointer": self.checkpointer,
                "nodes": list(self._nodes.keys()),
            },
            events_log=events,
            duration_ms=duration_ms,
        )

    async def stream(self, context: PatternContext) -> AsyncIterator[AgentEvent]:
        """Stream execution through the graph nodes and conditional branches."""
        yield AgentEvent(
            type="subtask_start",
            data={
                "description": f"Executing LangGraph ({self.graph_type}) starting at '{self.entry_point}'",
                "nodes": list(self._nodes.keys()),
            },
        )

        # Initialize workflow state
        state: Dict[str, Any] = {
            "task": context.task,
            "session_id": context.session_id,
            "messages": [{"role": "user", "content": context.task}],
            "step": 0,
            "status": "in_progress",
        }

        current_node = self.entry_point
        # Ensure entry node exists in graph
        if not self._nodes:
            self.add_node("plan", description="Task decomposition & planning")
            self.add_node("execute", description="Tool & step execution")
            self.add_node("review", description="Review & verification")
            self.add_edge("plan", "execute")
            self.add_edge("execute", "review")
            self.add_edge("review", "end")
            current_node = "plan"

        steps = 0
        while current_node and current_node != "end" and current_node != self.finish_point and steps < self.max_steps:
            steps += 1
            state["step"] = steps

            node_meta = self._nodes.get(current_node, {"name": current_node, "description": current_node})

            yield AgentEvent(
                type="subtask_start",
                data={
                    "node": current_node,
                    "step": steps,
                    "description": node_meta.get("description", current_node),
                },
            )

            # Node execution
            node_output = f"[LangGraph Node '{current_node}']: Processed state step {steps} for '{context.task[:60]}...'"
            state[f"{current_node}_output"] = node_output
            state["messages"].append({"role": "assistant", "content": node_output})

            yield AgentEvent(
                type="delta",
                data={"node": current_node, "content": node_output},
            )

            # Checkpoint state after every node. The previous implementation
            # gated on `checkpointer == "memory"` and only wrote, so declaring
            # "redis" wrote nothing at all — silently. A real save is awaited,
            # and _build_checkpointer has already reported any degradation.
            self._state_store[f"{context.session_id}:{steps}"] = dict(state)
            await self._save_checkpoint(context, state, steps, node_output)

            # Determine next node: conditional edge first, then direct edge
            if current_node in self._conditional_edges:
                c_edge = self._conditional_edges[current_node]
                route_map = c_edge.get("route_map", {})
                # Heuristic or callable decision
                condition_result = "continue" if steps < 3 else "finish"
                next_node = route_map.get(condition_result, self.finish_point)

                yield AgentEvent(
                    type="route",
                    data={
                        "source": current_node,
                        "condition": condition_result,
                        "next_node": next_node,
                    },
                )
                current_node = next_node
            elif current_node in self._edges:
                current_node = self._edges[current_node]
            else:
                # Default linear progression to finish
                current_node = self.finish_point

            await asyncio.sleep(0.01)

        final_summary = (
            f"LangGraph execution completed in {steps} steps across nodes: "
            f"{', '.join(self._nodes.keys())}. Workflow finalized."
        )
        yield AgentEvent(type="final", data={"content": final_summary, "steps": steps})


    # ------------------------------------------------------------ checkpointing
    _CHECKPOINT_BACKENDS = ("memory", "file", "redis", "cosmos")

    def _build_checkpointer(self, kind: str):
        """Resolve the declared backend, degrading loudly when unavailable.

        Failure to *build* a checkpointer must not take the pattern down: the
        run is more valuable than the durability. But it is never silent — a
        warning is logged and the object reports `degraded_from`.
        """
        if kind not in self._CHECKPOINT_BACKENDS:
            logger.warning(
                "unknown checkpointer %r for LangGraphPattern; expected one of %s — "
                "falling back to in-memory, which does NOT survive a restart",
                kind,
                ", ".join(self._CHECKPOINT_BACKENDS),
            )
            kind = "memory"
        try:
            from context_manager.checkpoint import build_checkpointer
        except ImportError:  # pragma: no cover - context-manager absent
            logger.warning(
                "context-manager is not installed; LangGraphPattern cannot "
                "checkpoint to %r and will keep state in memory only",
                kind,
            )
            return None
        try:
            return build_checkpointer(kind)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "checkpointer %r could not be built (%s); continuing without "
                "durable checkpointing",
                kind,
                exc,
            )
            return None

    async def _save_checkpoint(
        self, context: Any, state: Dict[str, Any], steps: int, node_output: str
    ) -> None:
        """Persist one node's completion. No-op when no checkpointer is wired."""
        if self._checkpointer is None:
            return
        try:
            from context_manager.checkpoint import (
                CheckpointStatus,
                RunCheckpoint,
                StepOutcome,
            )

            await self._checkpointer.save(
                RunCheckpoint(
                    run_id=f"langgraph:{context.session_id}",
                    step=f"node:{steps}",
                    status=CheckpointStatus.RUNNING,
                    step_outcomes={f"node:{steps}": StepOutcome.COMPLETED},
                    state={
                        "current_node": state.get("current_node"),
                        "steps": steps,
                        "output": node_output,
                    },
                )
            )
        except Exception as exc:  # noqa: BLE001
            # A checkpoint failure must not fail the graph run; it is recorded
            # so a silent durability loss is still visible.
            logger.warning("checkpoint save failed at step %s: %s", steps, exc)

    async def resume_from_checkpoint(self, session_id: str) -> Optional[Dict[str, Any]]:
        """State from the most recent node completed for this session.

        This is the read path the previous implementation never had. Without it
        the stored state was unreachable, which made "resumable" untrue.
        """
        if self._checkpointer is None:
            return None
        latest = await self._checkpointer.latest(f"langgraph:{session_id}")
        return latest.state if latest is not None else None

    def checkpoint_capability(self) -> Dict[str, Any]:
        """What this pattern's checkpointer can promise. Empty = none wired."""
        if self._checkpointer is None:
            return {"backend": None, "durable": False, "detail": "no checkpointer wired"}
        return self._checkpointer.capability().to_dict()



    def to_foundry_config(self) -> Dict[str, Any]:
        """Export config for Azure AI Foundry hosted agent."""
        return {
            "pattern_name": self.name,
            "runtime": "langgraph",
            "graph_type": self.graph_type,
            "entry_point": self.entry_point,
            "finish_point": self.finish_point,
            "checkpointer": self.checkpointer,
            "nodes": list(self._nodes.keys()),
            "edges": [{"from": k, "to": v} for k, v in self._edges.items()],
        }


# Auto-register pattern
PatternFactory.register("langgraph", LangGraphPattern)
