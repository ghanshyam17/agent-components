"""Foundry hosted agent for the agent-components monorepo (Responses protocol).

Exposes the monorepo through the Responses Protocol as a single front door:

* **tiered agentic router** — the real ``factory.build_agent()`` loop (hybrid
  router + ReAct loop) over the bundled sources, so chat behavior matches
  ``uv run agentic-router serve`` exactly.
* **multi-level component graph** — ``ComponentGraphOrchestrator``, which wires
  the Data plane (ADF/batch/medallion/RAG pipelines), the ML plane (AML jobs,
  endpoints, registries), the tool registry and the agentic pattern plane
  together (see ``platform/engineering/graph_orchestrator.py``).

Both are exposed as tools the hosting graph can call, so a single deployment can
answer "route this task" and "run this across the component graph" requests.

Model endpoints stay env-driven: point ``LOWER_*`` / ``HIGHER_*`` at any
OpenAI-compatible server (or the model-gateway). In the Foundry sandbox both
tiers point at the project's own model deployment for a zero-extra-cost tier
pair.

Environment:

``FOUNDRY_PROJECT_ENDPOINT``
    Foundry project endpoint (required).
``AZURE_AI_MODEL_DEPLOYMENT_NAME``
    Model deployment name (default ``gpt-5-mini``).
``AGCOMPS_DEPLOY_TARGET``
    ``router`` (default) to expose the router only, ``component-graph`` to also
    build the component graph at startup. Set by ``deploy_hosted_agent.py``.

Import layout: the deploy zip bundles the monorepo packages under ``_vendor/``;
from a repo checkout the live trees are used.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _bootstrap_imports() -> None:
    """Make the monorepo packages importable in the checkout *and* the zip.

    In the deployment zip everything lives under ``_vendor/``; in a checkout the
    packages are spread across ``core/src``, ``components/*/`` and ``platform/``.
    """
    here = Path(__file__).resolve().parent

    vendor = here / "_vendor"
    if (vendor / "components_core").is_dir() and str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))

    for cand in (here, *here.parents):
        core_src = cand / "core" / "src"
        comp_src = cand / "components" / "agentic-router" / "src"
        if core_src.is_dir() and comp_src.is_dir():
            for p in (str(core_src), str(comp_src)):
                if p not in sys.path:
                    sys.path.insert(0, p)
            # `platform` is a top-level package at the repo root; its parent must
            # be on the path for `import platform.schema` to resolve.
            if (cand / "platform" / "__init__.py").is_file() and str(cand) not in sys.path:
                sys.path.insert(0, str(cand))
            break


_bootstrap_imports()

import httpx  # noqa: E402
from agentic_router.config import Settings  # noqa: E402
from agentic_router.factory import build_agent  # noqa: E402

_AGENT = None
_GRAPH = None


class _EntraAuth(httpx.Auth):
    """Re-mints an Entra bearer token per request (tokens expire mid-session)."""

    def __init__(self) -> None:
        from azure.identity import DefaultAzureCredential

        self._cred = DefaultAzureCredential()

    def auth_flow(self, request):
        tk = self._cred.get_token("https://ai.azure.com/.default")
        request.headers["Authorization"] = f"Bearer {tk.token}"
        yield request


def _data_plane_base() -> str:
    """OpenAI-compatible base URL from the project endpoint (custom host!)."""
    endpoint = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    host = endpoint.split("/api/projects/")[0]
    return f"{host}/openai/v1"


def _model_name() -> str:
    return os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini")


def _agent():
    global _AGENT
    if _AGENT is None:
        from openai import AsyncOpenAI

        from agentic_router.models import ModelTier
        from agentic_router.tools import build_default_registry
        from memory_store import InMemorySessionStore

        base = _data_plane_base()
        model = _model_name()
        s = Settings(
            lower_vllm_base_url=base,
            lower_model=model,
            higher_vllm_base_url=base,
            higher_model=model,
            vllm_api_key="entra",  # replaced below by the Entra-authed client
            agent_enable_planning=False,
        )
        agent = build_agent(s)
        # Re-wire both tier clients with per-request Entra bearer auth
        # (the settings-built clients use a static placeholder key).
        for tier in (ModelTier.LOWER, ModelTier.HIGHER):
            mc = agent.registry.get(tier)
            mc.client = AsyncOpenAI(
                base_url=base,
                api_key="entra",
                http_client=httpx.AsyncClient(auth=_EntraAuth(), timeout=120.0),
            )
        _AGENT = agent
    return _AGENT


# --------------------------------------------------------------------------- #
# The tiered agentic router (existing behaviour)
# --------------------------------------------------------------------------- #
async def run_agent(
    task: Annotated[str, "Task for the tiered agentic router."],
) -> str:
    """Run the real agentic-router loop (route decision + final answer)."""

    def _run() -> str:
        a = _agent()
        final, route, _state = asyncio.run(a.run(task))
        return json.dumps(
            {
                "final": final,
                "route": route.to_dict() if route is not None and hasattr(route, "to_dict") else (route.model_dump() if route is not None else None),
            },
            default=str,
        )

    return await asyncio.to_thread(_run)


# --------------------------------------------------------------------------- #
# The multi-level component graph (Data + ML + Tools + Agents)
# --------------------------------------------------------------------------- #
def _default_graph_spec() -> Any:
    """A minimal valid ComponentGraph, used when no spec file is on disk.

    ``ComponentGraph`` requires ``metadata`` and ``spec`` (with an agent plane),
    so an empty construct would not validate. This builds the smallest graph
    that still exercises every plane the orchestrator wires.
    """
    from platform.schema.agent_spec import AgentSpecModel, Metadata, ModelConfig
    from platform.schema.component_graph import (
        ComponentGraph,
        ComponentGraphSpec,
        DataPlaneConfig,
        MLPlaneConfig,
        RuntimeChassisConfig,
        ToolsPlaneConfig,
    )

    return ComponentGraph(
        metadata=Metadata(
            name="agcomps-default",
            description="Default component graph (no spec file on disk)",
        ),
        spec=ComponentGraphSpec(
            data_plane=DataPlaneConfig(),
            ml_plane=MLPlaneConfig(),
            tools_plane=ToolsPlaneConfig(),
            agent_plane=AgentSpecModel(
                # ReAct is the dependency-free pattern, so it is the safe
                # default in a hosted sandbox where LangGraph/AutoGen may not
                # be installed.
                pattern="react",
                model=ModelConfig(
                    provider="azure-foundry",
                    deployment_name=os.environ.get(
                        "AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"
                    ),
                ),
            ),
            runtime_chassis=RuntimeChassisConfig(
                provider="azure-foundry", protocol="responses"
            ),
        ),
    )


def _load_graph() -> Any:
    """Build the ComponentGraphOrchestrator from the bundled component-graph spec.

    Uses the orchestrator's own defaults, so this works in the Foundry sandbox
    where no Azure Data Factory / AML workspace is reachable: the planes report
    their configured inventory and the agent pattern handles the task.
    """
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator
    from platform.schema.loader import load_spec

    spec_path = os.environ.get("AGCOMPS_GRAPH_SPEC")
    candidates: list[Path] = []
    if spec_path:
        candidates.append(Path(spec_path))
    here = Path(__file__).resolve().parent
    candidates.extend(
        c / "projects" / "component_graph.yaml" for c in (here, *here.parents)
    )
    candidates.append(here / "component_graph.yaml")

    for cand in candidates:
        if not cand.is_file():
            continue
        try:
            spec = load_spec(cand)
        except Exception as e:  # noqa: BLE001 - a bad spec falls through
            logger.warning("could not load graph spec %s: %s", cand, e)
            continue
        logger.info("component graph loaded from %s", cand)
        return ComponentGraphOrchestrator(spec)

    logger.warning(
        "no component_graph spec found (searched %s); using the built-in default graph",
        ", ".join(str(c) for c in candidates),
    )
    return ComponentGraphOrchestrator(_default_graph_spec())


def _graph():
    global _GRAPH
    if _GRAPH is None:
        try:
            _GRAPH = _load_graph()
        except Exception as e:  # noqa: BLE001 - never take the front door down
            logger.exception("component graph bootstrap failed: %s", e)
            return None
    return _GRAPH


async def run_graph(
    task: Annotated[str, "Task to execute across the Data + ML + Tools + Agents graph."],
    session_id: Annotated[str, "Session id for memory continuity."] = "session_default",
) -> str:
    """Execute a task across every plugged-in plane of the component graph."""
    graph = await asyncio.to_thread(_graph)
    if graph is None:
        return json.dumps(
            {"success": False, "error": "component graph unavailable; see logs"}
        )
    result = await graph.execute_task(task, session_id=session_id)
    return json.dumps(result.to_dict(), default=str)


async def graph_descriptor() -> str:
    """Describe the wired component graph: planes, tools and deployment spec.

    Async because it is registered as a tool via ``coroutine=``; LangChain
    requires a coroutine function there.
    """
    graph = await asyncio.to_thread(_graph)
    if graph is None:
        return json.dumps({"available": False})
    spec = graph.to_foundry_deployment_spec()
    tools = [t.name for t in graph.tool_registry]
    return json.dumps(
        {
            "available": True,
            "deployment": spec,
            "tools": tools,
            "pattern": graph.spec.agent_plane.pattern,
            "planes": {
                "data": graph.spec.data_plane.model_dump(exclude_defaults=True),
                "ml": graph.spec.ml_plane.model_dump(exclude_defaults=True),
            },
        },
        default=str,
    )


# --------------------------------------------------------------------------- #
# Hosting
# --------------------------------------------------------------------------- #
def _build_graph() -> Any:
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

    chat_model = AzureAIOpenAIApiChatModel(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=_model_name(),
    )

    tools = [
        StructuredTool.from_function(coroutine=run_agent, name="run_agent"),
        StructuredTool.from_function(coroutine=run_graph, name="run_graph"),
        StructuredTool.from_function(coroutine=graph_descriptor, name="graph_descriptor"),
    ]

    instructions = (
        "You are the front door of the agent-components platform. You expose two "
        "planes. For a single reasoning task use run_agent (tiered small/large "
        "model routing); it returns the routing decision and the final answer. "
        "For a task that should exercise the whole platform - data pipelines, ML "
        "workflows, registered tools and agentic patterns - use run_graph. Call "
        "graph_descriptor to report which planes, tools and patterns are wired. "
        "Present results concisely."
    )

    return create_agent(chat_model, tools=tools, system_prompt=instructions)


def main() -> None:
    graph = _build_graph()
    port = int(os.environ.get("PORT", "8088"))

    # Warm the component graph at startup when this deployment targets it, so
    # bootstrap errors surface in the agent logs rather than on first request.
    if os.environ.get("AGCOMPS_DEPLOY_TARGET") == "component-graph":
        logger.info("AGCOMPS_DEPLOY_TARGET=component-graph: warming component graph")
        _graph()

    from langchain_azure_ai.agents.hosting import ResponsesHostServer

    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
