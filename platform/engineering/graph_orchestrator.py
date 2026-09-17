"""Component Graph Orchestrator.

Plugs all levels together: Data Engineering services, Azure Machine Learning
workflows, tool bridges, agentic design patterns (LangGraph, AutoGen, ReAct),
and the Azure AI Foundry SDK chassis.
"""
from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, List, Optional

from agentic_router.models import AgentEvent

try:
    from toolkit.base import Registry
except ImportError:  # toolkit component not installed — fall back to the router's registry
    from agentic_router.tools.base import Registry  # type: ignore

from platform.schema.component_graph import ComponentGraph, ComponentGraphSpec
from platform.patterns.base import AgentPattern, PatternContext, PatternFactory, PatternResult
from platform.engineering.data.tools import (
    ADFPipelineTool,
    VectorSearchTool,
    LakehouseQueryTool,
    AMLModelInferenceTool,
    DataQualityInspectionTool,
)
from platform.engineering.data.manager import DataInfraManager
from platform.engineering.aml.manager import AMLInfraManager

logger = logging.getLogger(__name__)


class ComponentGraphExecutionResult:
    """Result of an end-to-end component graph execution."""

    def __init__(
        self,
        success: bool,
        answer: str,
        events: List[AgentEvent],
        data_plane_status: Dict[str, Any],
        ml_plane_status: Dict[str, Any],
        duration_ms: int,
    ) -> None:
        self.success = success
        self.answer = answer
        self.events = events
        self.data_plane_status = data_plane_status
        self.ml_plane_status = ml_plane_status
        self.duration_ms = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "answer": self.answer,
            "duration_ms": self.duration_ms,
            "data_plane": self.data_plane_status,
            "ml_plane": self.ml_plane_status,
            "events_count": len(self.events),
        }


class ComponentGraphOrchestrator:
    """Enterprise orchestrator connecting Data, ML, Agents, and Foundry Chassis."""

    def __init__(
        self,
        graph: ComponentGraph,
        subscription_id: str = "sub-12345",
        resource_group: str = "rg-enterprise-agents",
        data_factory_name: str = "adf-enterprise-data",
        aml_workspace_name: str = "aml-enterprise-workspace",
        agent_factory: Callable[[], Any] | None = None,
        checkpointer: Any = None,
    ) -> None:
        self.graph = graph
        self.spec: ComponentGraphSpec = graph.spec
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.data_factory_name = data_factory_name
        self.aml_workspace_name = aml_workspace_name
        # Supplied by a deployment that already built an agent (e.g. the Foundry
        # hosted agent with Entra auth). Without it, the agent plane falls back
        # to ambient settings.
        self.agent_factory = agent_factory
        # Optional run checkpointing. A component-graph run touches several
        # planes in sequence, so a crash part-way loses all of it without this.
        self.checkpointer = checkpointer

        # Managers for Data & ML
        self.data_manager = DataInfraManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            factory_name=data_factory_name,
        )
        self.aml_manager = AMLInfraManager(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=aml_workspace_name,
        )

        # Tool Registry
        self.tool_registry = Registry()
        self._wire_tools()

        # Agent pattern instance
        self.agent_pattern: Optional[AgentPattern] = None
        self._wire_agent_pattern()

    def _wire_tools(self) -> None:
        """Automatically wire data plane and ML plane outputs as callable agent tools."""
        tp = self.spec.tools_plane
        dp = self.spec.data_plane
        mp = self.spec.ml_plane

        if tp.enable_data_tools or dp.auto_generate_agent_tools:
            self.tool_registry.register(ADFPipelineTool(self.data_manager.adf_client))
            self.tool_registry.register(LakehouseQueryTool())
            self.tool_registry.register(DataQualityInspectionTool())
            logger.info("Registered Data Engineering tools: ADF, Lakehouse, QualityInspection")

        if tp.enable_vector_search_tools or dp.rag_pipelines:
            self.tool_registry.register(VectorSearchTool())
            logger.info("Registered Vector RAG AI Search tool")

        if tp.enable_ml_inference_tools or mp.auto_generate_agent_tools:
            self.tool_registry.register(AMLModelInferenceTool(self.aml_manager.client))
            logger.info("Registered AML Model Inference tool")

    def _wire_agent_pattern(self) -> None:
        """Instantiate the agent pattern declared in the graph (LangGraph, AutoGen, ReAct, etc.)."""
        pattern_type = self.spec.agent_plane.pattern
        logger.info(f"Instantiating agent pattern: {pattern_type}")

        # Build pattern via PatternFactory
        kwargs: Dict[str, Any] = {}
        # Pass the injected factory through so a deployed agent is reused rather
        # than rebuilt from ambient (localhost) settings.
        if self.agent_factory is not None:
            kwargs["agent_factory"] = self.agent_factory
        if pattern_type == "autogen" and self.spec.agent_plane.autogen:
            ag = self.spec.agent_plane.autogen
            kwargs.update({
                "mode": ag.mode,
                "max_rounds": ag.max_rounds,
                "admin_name": ag.admin_name,
                "agents": [a.model_dump() for a in ag.agents],
            })
        elif pattern_type == "langgraph" and self.spec.agent_plane.langgraph:
            lg = self.spec.agent_plane.langgraph
            kwargs.update({
                "entry_point": lg.entry_point,
                "finish_point": lg.finish_point or "end",
                "nodes": [n.model_dump() for n in lg.nodes],
                "edges": [e.model_dump() for e in lg.edges],
                "conditional_edges": [c.model_dump() for c in lg.conditional_edges],
            })

        try:
            self.agent_pattern = PatternFactory.create(pattern_type, **kwargs)
        except Exception as e:
            logger.warning(f"Failed to create pattern '{pattern_type}' via factory ({e}), falling back to ReAct")
            # Keep the injected factory on the fallback path too, otherwise the
            # fallback silently loses the deployed endpoint.
            self.agent_pattern = PatternFactory.create("react", **kwargs)

    async def execute_task(self, task: str, session_id: str = "session_default") -> ComponentGraphExecutionResult:
        """Execute a task across all plugged-in layers: Data -> ML -> Tools -> Agent."""
        start_time = time.time()
        events: List[AgentEvent] = []

        data_status: Dict[str, Any] = {
            "batch_pipelines_ready": len(self.spec.data_plane.batch_pipelines),
            "medallion_pipelines_ready": len(self.spec.data_plane.medallion_pipelines),
            "rag_pipelines_ready": len(self.spec.data_plane.rag_pipelines),
            "status": "Operational",
        }

        ml_status: Dict[str, Any] = {
            "models_available": len(self.spec.ml_plane.models),
            "endpoints_active": len(self.spec.ml_plane.endpoints),
            "status": "Operational",
        }

        # Create PatternContext with plugged-in tools and configurations
        context = PatternContext(
            task=task,
            session_id=session_id,
            config={
                "data_plane": data_status,
                "ml_plane": ml_status,
                "runtime_chassis": self.spec.runtime_chassis.model_dump(),
            },
            tools=self.tool_registry,
        )

        final_answer = ""
        # Pre-write: record the run as started so a crash mid-execution is
        # distinguishable from a run that never began.
        await self._checkpoint(session_id, task, status="running", step="execute_task")

        if self.agent_pattern:
            async for event in self.agent_pattern.stream(context):
                events.append(event)
                if event.type == "final":
                    final_answer = event.data.get("content", "")

        # Mark the run complete only after the stream finished. If the process
        # died above, this line never ran and the checkpoint stays "running" —
        # which is the signal a later run uses to know it should resume.
        await self._checkpoint(
            session_id, task, status="completed", step="execute_task",
            state={"answer": final_answer, "events": len(events)},
        )

        duration_ms = int((time.time() - start_time) * 1000)
        return ComponentGraphExecutionResult(
            success=True,
            answer=final_answer,
            events=events,
            data_plane_status=data_status,
            ml_plane_status=ml_status,
            duration_ms=duration_ms,
        )

    # ------------------------------------------------------------ checkpointing
    def _run_id(self, session_id: str) -> str:
        return f"orchestrator:{session_id}"

    async def _checkpoint(
        self,
        session_id: str,
        task: str,
        *,
        status: str,
        step: str,
        state: dict[str, Any] | None = None,
    ) -> None:
        """Persist run progress. No-op when no checkpointer is configured."""
        if self.checkpointer is None:
            return
        try:
            from context_manager.checkpoint import (
                CheckpointStatus,
                RunCheckpoint,
                StepOutcome,
            )

            outcome = (
                StepOutcome.COMPLETED if status == "completed" else StepOutcome.PENDING
            )
            await self.checkpointer.save(
                RunCheckpoint(
                    run_id=self._run_id(session_id),
                    step=step,
                    status=CheckpointStatus(status),
                    step_outcomes={step: outcome},
                    state={"task": task, **(state or {})},
                )
            )
        except Exception as exc:  # noqa: BLE001
            # Never fail a run because checkpointing failed; log it so the loss
            # of durability is not invisible.
            logger.warning("orchestrator checkpoint failed (%s): %s", session_id, exc)

    async def resume_state(self, session_id: str) -> dict[str, Any] | None:
        """State from the last checkpoint for this session, or None."""
        if self.checkpointer is None:
            return None
        latest = await self.checkpointer.latest(self._run_id(session_id))
        return latest.state if latest is not None else None

    async def was_interrupted(self, session_id: str) -> bool:
        """True when a previous run started and never marked completion.

        This is the query that makes resume actionable: a ``running`` status
        with no ``completed`` write means the process died mid-execution.
        """
        if self.checkpointer is None:
            return False
        try:
            from context_manager.checkpoint import CheckpointStatus
        except ImportError:  # pragma: no cover - context-manager absent
            return False
        latest = await self.checkpointer.latest(self._run_id(session_id))
        if latest is None:
            return False
        return latest.status is not CheckpointStatus.COMPLETED

    def to_foundry_deployment_spec(self) -> Dict[str, Any]:
        """Synthesizes deployment configuration for Azure AI Foundry SDK hosted agents."""
        return {
            "name": self.graph.metadata.name,
            "runtime": self.spec.runtime_chassis.provider,
            "protocol": self.spec.runtime_chassis.protocol,
            "agent_pattern": self.spec.agent_plane.pattern,
            "data_engineering": {
                "factory": self.data_factory_name,
                "tools_exposed": [t.name for t in self.tool_registry],
            },
            "machine_learning": {
                "workspace": self.aml_workspace_name,
                "endpoints": [e.name for e in self.spec.ml_plane.endpoints],
            },
            "environment_variables": {
                "DATA_FACTORY_NAME": self.data_factory_name,
                "AML_WORKSPACE_NAME": self.aml_workspace_name,
                "FOUNDRY_PROTOCOL": self.spec.runtime_chassis.protocol,
            },
        }
