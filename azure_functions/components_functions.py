"""Azure Functions for agent-components — HTTP triggers over the monorepo.

  POST /api/router/decide  {task} → model-tier routing decision
  POST /api/memory/{op}    {text, session_id} → memory remember/search
  POST /api/foundry/invoke {message, component} → Foundry hosted-agent entrypoint
  GET  /api/foundry/health → Foundry wiring check
"""

import json
import logging
import os
import sys
from pathlib import Path

import azure.functions as func

_fn_dir = Path(__file__).resolve().parent
_repo = _fn_dir.parent
for p in (str(_repo), str(_repo / "components" / "agentic-router"),
          str(_repo / "components" / "memory-store"), str(_fn_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)
logger = logging.getLogger("components-functions")


def _ok(data, status=200):
    return func.HttpResponse(json.dumps(data, ensure_ascii=False, default=str),
                             status_code=status, mimetype="application/json")


def _err(msg, status=500):
    return _ok({"error": msg}, status)


def _body(req):
    try:
        return req.get_json()
    except Exception:
        return {}


@app.route(route="router/decide", methods=["POST"])
def router_decide(req: func.HttpRequest):
    from azure_functions.components_foundry import route_decision
    task = (_body(req).get("task") or "").strip()
    if not task:
        return _err("task is required", 400)
    return _ok(route_decision(task))


@app.route(route="memory/{operation}", methods=["POST"])
def memory_op(req: func.HttpRequest):
    from azure_functions.components_foundry import memory_store_op
    op = req.route_params.get("operation", "").lower()
    b = _body(req)
    return _ok(memory_store_op(op, text=b.get("text", ""), session_id=b.get("session_id", "default")))


@app.route(route="foundry/invoke", methods=["POST"])
def foundry_invoke(req: func.HttpRequest):
    from azure_functions.components_foundry import handle_foundry_request
    payload = _body(req)
    if not payload:
        return _err("JSON body required", 400)
    return _ok(handle_foundry_request(payload))


@app.route(route="foundry/health", methods=["GET"])
def foundry_health(req: func.HttpRequest):
    endpoint = os.environ.get("AZURE_FOUNDRY_PROJECT_ENDPOINT", "")
    agent = os.environ.get("AZURE_FOUNDRY_AGENT_ID", "agentic-router")
    detail = {"endpoint": endpoint or "(not set)", "agent": agent, "wired": bool(endpoint)}
    if endpoint:
        try:
            from azure.identity import DefaultAzureCredential
            from azure.ai.projects import AIProjectClient
            client = AIProjectClient(endpoint=endpoint, credential=DefaultAzureCredential())
            names = [a.name for a in getattr(client.agents.list_agents(), "data", [])]
            detail["agents"] = names
            detail["agent_present"] = agent in names
        except Exception as e:
            detail["sdk_error"] = str(e)[:200]
    return _ok({"status": "ok", "service": "agent-components-functions", **detail})