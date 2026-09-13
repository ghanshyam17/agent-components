"""Foundry hosted agent for the agent-components monorepo (Responses protocol).

Front door to the ``agentic-router`` component: the tool calls the real
``factory.build_agent()`` loop (hybrid router + agentic loop) over the
bundled monorepo sources, so chat behavior matches ``uv run agentic-router
serve`` exactly. Model endpoints stay env-driven: point ``LOWER_*`` /
``HIGHER_*`` at any OpenAI-compatible server (or the model-gateway) - in the
Foundry sandbox set both to the project's gpt-5-mini deployment for a
zero-extra-cost tier pair.

Import layout: the deploy zip bundles the monorepo packages under
``_vendor/``; from a repo checkout the live trees are used.
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
    """Make the monorepo packages importable in the checkout and the zip."""
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
            break


_bootstrap_imports()

from agentic_router.config import Settings  # noqa: E402
from agentic_router.factory import build_agent  # noqa: E402

_AGENT = None


def _agent():
    global _AGENT
    if _AGENT is None:
        s = Settings(
            lower_vllm_base_url=os.environ.get("LOWER_BASE_URL", os.environ["FOUNDRY_PROJECT_ENDPOINT"]),
            lower_model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
            higher_vllm_base_url=os.environ.get("HIGHER_BASE_URL", os.environ["FOUNDRY_PROJECT_ENDPOINT"]),
            higher_model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
            agent_enable_planning=False,  # single-shot in the sandbox
        )
        _AGENT = build_agent(s)
    return _AGENT


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


def _build_graph() -> Any:
    from langchain.agents import create_agent
    from langchain_core.tools import StructuredTool
    from langchain_azure_ai.chat_models import AzureAIOpenAIApiChatModel

    chat_model = AzureAIOpenAIApiChatModel(
        project_endpoint=os.environ["FOUNDRY_PROJECT_ENDPOINT"],
        model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
    )

    tools = [StructuredTool.from_function(coroutine=run_agent, name="run_agent")]

    instructions = (
        "You are the front door of the agentic-router prototype (small vs large "
        "model tiering). For every user task call run_agent with the task text, "
        "then present the routing decision (which tier and why) and the final "
        "answer. Be concise."
    )

    return create_agent(chat_model, tools=tools, system_prompt=instructions)


def main() -> None:
    graph = _build_graph()
    port = int(os.environ.get("PORT", "8088"))
    from langchain_azure_ai.agents.hosting import ResponsesHostServer

    ResponsesHostServer(graph).run(port=port)


if __name__ == "__main__":
    main()
