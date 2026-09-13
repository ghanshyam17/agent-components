"""Azure Functions HTTP surface for the agentic-router component.

``GET  /api/agent`` -> capability card.
``POST /api/agent`` ``{"task": "..."}`` -> route decision + final answer from
the real ``factory.build_agent()`` loop (asyncio.run over the async Agent).

The deploy zip bundles the monorepo under ``_vendor/`` (components_core,
agentic_router, memory_store, model_gateway); from a checkout the live trees
are used. Point ``LOWER_*``/``HIGHER_*`` env vars at any OpenAI-compatible
endpoint, or leave them unset to run against the sandbox stub (no model =
the loop's error path still returns a route decision + explanation).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

import azure.functions as func


def _bootstrap_imports() -> None:
    here = Path(__file__).resolve().parent
    for base in (here, here.parent, here.parent.parent):
        vendor = base / "_vendor"
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

logger = logging.getLogger("agent-api")

app = func.FunctionApp()

_AGENT = None
_TOKEN = None  # (token_string, expires_at_epoch)


def _foundry_base_url() -> str:
    """OpenAI-compatible data plane base URL (custom-subdomain host!)."""
    explicit = os.environ.get("FOUNDRY_OPENAI_BASE_URL")
    if explicit:
        return explicit.rstrip("/")
    endpoint = os.environ.get(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://my-foundry-usecase.services.ai.azure.com/api/projects/agent-lab",
    )
    host = endpoint.split("/api/projects/")[0]
    return f"{host}/openai/v1"


def _api_key() -> str:
    """Foundry credential for the OpenAI-compatible endpoint.

    Prefers FOUNDRY_API_KEY (account key, set via `az functionapp config
    appsettings set` - never committed); falls back to an Entra bearer token
    (requires 'Cognitive Services OpenAI User' on the app's identity, which
    propagates slowly on this tenant).
    """
    key = os.environ.get("FOUNDRY_API_KEY", "").strip()
    if key:
        return key
    global _TOKEN
    import time

    now = time.time()
    if _TOKEN is None or _TOKEN[1] < now:
        from azure.identity import DefaultAzureCredential

        tk = DefaultAzureCredential().get_token("https://ai.azure.com/.default")
        _TOKEN = (tk.token, tk.expires_on - 300)
    return _TOKEN[0]


def _agent():
    global _AGENT
    if _AGENT is None:
        base = _foundry_base_url()
        s = Settings(
            lower_vllm_base_url=base,
            lower_model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
            higher_vllm_base_url=base,
            higher_model=os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
            vllm_api_key=_api_key(),  # key (app setting) or Entra bearer
            agent_enable_planning=False,
        )
        _AGENT = build_agent(s)
    return _AGENT


def _json(payload: dict, status: int = 200) -> func.HttpResponse:
    return func.HttpResponse(
        json.dumps(payload, default=str), mimetype="application/json", status_code=status
    )


@app.function_name(name="AgentApi")
@app.route(route="agent", methods=["POST", "GET"], auth_level=func.AuthLevel.FUNCTION)
def agent_api(req: func.HttpRequest) -> func.HttpResponse:
    if req.method == "GET":
        return _json(
            {
                "service": "agentic-router",
                "actions": ["run"],
                "tiers": ["lower", "higher"],
            }
        )

    try:
        body = req.get_json()
    except ValueError:
        return _json({"error": "invalid JSON body"}, 400)

    task = body.get("task") or ""
    if not task:
        return _json({"error": "task is required"}, 400)

    try:
        a = _agent()
        final, route, _state = asyncio.run(a.run(task))
        route_out = (
            route.model_dump() if route is not None and hasattr(route, "model_dump") else None
        )
        return _json({"final": final, "route": route_out})
    except Exception as exc:  # noqa: BLE001
        logger.exception("run failed")
        return _json({"error": f"{type(exc).__name__}: {exc}"}, 500)
