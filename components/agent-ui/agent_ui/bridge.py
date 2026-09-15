"""UIBridge: Introspects Agent specs, Component Graphs, and Tool Registries.

Transforms declarative backend definitions into standard frontend payloads for
chat configuration, starter suggestions, and interactive DAG graph visualization.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class UIBridge:
    """Bridge providing metadata and graph topology to the frontend."""

    def __init__(self, target: Any = None) -> None:
        self.target = target

    def get_config(self) -> Dict[str, Any]:
        """Returns standard metadata for the agent UI."""
        if hasattr(self.target, "graph"):
            # Target is a ComponentGraphOrchestrator
            cg = self.target.graph
            meta = cg.metadata
            spec = cg.spec
            return {
                "name": meta.name,
                "project": meta.project or meta.name,
                "description": meta.description or "Enterprise AI Component Graph",
                "pattern": spec.agent_plane.pattern,
                "model_provider": spec.agent_plane.model.provider,
                "model_name": spec.agent_plane.model.deployment_name,
                "tools": [t.name for t in getattr(self.target, "tool_registry", [])],
                "data_pipelines_count": len(spec.data_plane.batch_pipelines) + len(spec.data_plane.medallion_pipelines),
                "ml_endpoints_count": len(spec.ml_plane.endpoints),
                "suggested_prompts": [
                    "Query the latest sales metrics from the Medallion Gold Lakehouse",
                    "Run an anomaly detection check on customer transactions",
                    "Trigger an ADF data ingestion pipeline for new records",
                    "Search enterprise knowledge base for compliance guidelines",
                ],
                "starter_prompts": [
                    "Query the latest sales metrics from the Medallion Gold Lakehouse",
                    "Run an anomaly detection check on customer transactions",
                    "Trigger an ADF data ingestion pipeline for new records",
                    "Search enterprise knowledge base for compliance guidelines",
                ],
            }
        elif hasattr(self.target, "metadata"):
            # Target is an AgentSpec
            meta = self.target.metadata
            spec = self.target.spec
            prompts = [
                "Hello! What can you help me with today?",
                "Analyze current operational tasks",
                "Run reasoning loop with tools",
            ]
            return {
                "name": meta.name,
                "project": meta.project or meta.name,
                "description": meta.description or "Autonomous Agent",
                "pattern": getattr(spec, "pattern", "react"),
                "model_provider": getattr(spec.model, "provider", "openai"),
                "model_name": getattr(spec.model, "deployment_name", "gpt-4"),
                "tools": getattr(spec.tools, "builtin", []) if getattr(spec, "tools", None) else [],
                "suggested_prompts": prompts,
                "starter_prompts": prompts,
            }

        # Generic default
        default_prompts = [
            "Summarize today's agent activities",
            "Execute multi-step task with tools",
        ]
        return {
            "name": "Enterprise Agent",
            "project": "agent-components",
            "description": "Universal AI Agent Workspace",
            "pattern": "react",
            "model_provider": "azure-foundry",
            "model_name": "gpt-5-mini",
            "tools": ["web_search", "calculator"],
            "suggested_prompts": default_prompts,
            "starter_prompts": default_prompts,
        }

    def get_graph(self) -> Dict[str, Any]:
        """Returns node topology and edges for DAG visualization."""
        nodes: List[Dict[str, Any]] = []
        edges: List[Dict[str, Any]] = []

        if hasattr(self.target, "graph"):
            spec = self.target.graph.spec

            # 1. Data Plane Nodes
            nodes.append({
                "id": "data_plane",
                "label": "Data Engineering (ADF)",
                "type": "data",
                "status": "ready",
                "details": f"{len(spec.data_plane.batch_pipelines)} Batch, {len(spec.data_plane.medallion_pipelines)} Medallion",
            })
            nodes.append({
                "id": "lakehouse",
                "label": "Medallion Lakehouse",
                "type": "storage",
                "status": "active",
                "details": "Bronze -> Silver -> Gold (Delta)",
            })
            nodes.append({
                "id": "vector_rag",
                "label": "Vector RAG (AI Search)",
                "type": "search",
                "status": "active",
                "details": "Hybrid text-embedding-3",
            })
            edges.append({"from": "data_plane", "to": "lakehouse"})
            edges.append({"from": "data_plane", "to": "vector_rag"})

            # 2. ML Plane Nodes
            nodes.append({
                "id": "ml_plane",
                "label": "Azure Machine Learning",
                "type": "ml",
                "status": "ready",
                "details": f"{len(spec.ml_plane.models)} Models, {len(spec.ml_plane.endpoints)} Endpoints",
            })

            # 3. Tool Bridge Nodes
            nodes.append({
                "id": "tools_bridge",
                "label": "Agent Tools Registry",
                "type": "tools",
                "status": "active",
                "details": f"{len(getattr(self.target, 'tool_registry', []))} Tools Bound",
            })
            edges.append({"from": "lakehouse", "to": "tools_bridge"})
            edges.append({"from": "vector_rag", "to": "tools_bridge"})
            edges.append({"from": "ml_plane", "to": "tools_bridge"})

            # 4. Agent Pattern Nodes
            pattern = spec.agent_plane.pattern
            if pattern == "langgraph" and spec.agent_plane.langgraph:
                lg = spec.agent_plane.langgraph
                for node in lg.nodes:
                    nodes.append({
                        "id": f"agent_{node.name}",
                        "label": f"Step: {node.name}",
                        "type": "agent_state",
                        "status": "idle",
                        "details": node.description or "",
                    })
                for edge in lg.edges:
                    edges.append({"from": f"agent_{edge.from_node}", "to": f"agent_{edge.to_node}"})
                edges.append({"from": "tools_bridge", "to": f"agent_{lg.entry_point}"})
            elif pattern == "autogen" and spec.agent_plane.autogen:
                ag = spec.agent_plane.autogen
                nodes.append({
                    "id": "autogen_manager",
                    "label": "GroupChat Coordinator",
                    "type": "agent",
                    "status": "active",
                    "details": f"Mode: {ag.mode}",
                })
                edges.append({"from": "tools_bridge", "to": "autogen_manager"})
                for a in ag.agents:
                    nodes.append({
                        "id": f"agent_{a.name}",
                        "label": a.name,
                        "type": "agent",
                        "status": "ready",
                        "details": a.role,
                    })
                    edges.append({"from": "autogen_manager", "to": f"agent_{a.name}"})
            else:
                nodes.append({
                    "id": "agent_core",
                    "label": f"Agent Pattern: {pattern.upper()}",
                    "type": "agent",
                    "status": "active",
                    "details": f"Model: {spec.agent_plane.model.deployment_name}",
                })
                edges.append({"from": "tools_bridge", "to": "agent_core"})

            # 5. Runtime Chassis Node
            nodes.append({
                "id": "chassis",
                "label": "Foundry Runtime Chassis",
                "type": "runtime",
                "status": "active",
                "details": f"{spec.runtime_chassis.provider} ({spec.runtime_chassis.protocol})",
            })
            edges.append({"from": "agent_core" if pattern not in ("langgraph", "autogen") else "tools_bridge", "to": "chassis"})

        else:
            # Standard single-agent graph
            nodes = [
                {"id": "user", "label": "User Input", "type": "input", "status": "active", "details": "HTTP/SSE"},
                {"id": "router", "label": "Tier Router", "type": "router", "status": "active", "details": "Heuristic + LLM-Judge"},
                {"id": "tools", "label": "Tool Registry", "type": "tools", "status": "active", "details": "Sandboxed Tools"},
                {"id": "memory", "label": "Session Memory", "type": "storage", "status": "active", "details": "In-Memory / Redis"},
                {"id": "model", "label": "Inference Model", "type": "model", "status": "active", "details": "Foundry / vLLM"},
            ]
            edges = [
                {"from": "user", "to": "router"},
                {"from": "router", "to": "model"},
                {"from": "model", "to": "tools"},
                {"from": "tools", "to": "model"},
                {"from": "router", "to": "memory"},
            ]

        return {"nodes": nodes, "edges": edges}
