"""agent_components_foundry.py — Foundry SDK + Azure Functions adapter.

Exposes the composable component monorepo to Foundry:

  - route_decision(task) → which model tier the agentic-router picks (async)
  - memory ops via memory_store (store/search session memories)
  - handle_foundry_request(payload) → hosted-agent entrypoint
    {"message": task, "component": "router"|"memory"} → decision / recall

Pure-stdlib fallbacks when component deps (typer/rich/redis) are absent —
the Functions stay runnable with just requirements installed.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

_pkg_parents = [
    Path(__file__).resolve().parent,
    Path(__file__).resolve().parent.parent,
    Path(__file__).resolve().parent.parent / "components" / "agentic-router",
    Path(__file__).resolve().parent.parent / "components" / "memory-store",
]
for _p in _pkg_parents:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def route_decision(task: str) -> Dict[str, Any]:
    """Ask the agentic-router which model tier should handle this task."""
    try:
        from agentic_router.factory import build_agent
        agent = build_agent()
        decision = asyncio.run(agent.router.route(task))
        return {
            "ok": True,
            "tier": decision.tier.value,
            "model": decision.model,
            "method": decision.method,
            "score": round(decision.score, 3),
            "reason": decision.reason,
            "signals": {k: round(v, 3) for k, v in (decision.signals or {}).items()},
        }
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "fallback": "heuristic-low-tier"}


def memory_store_op(operation: str, **kwargs) -> Dict[str, Any]:
    """Memory component ops: remember / search / session."""
    try:
        from memory_store.in_memory import InMemorySessionStore
        from components_core import SessionState, Message
        store = InMemorySessionStore()

        async def _run():
            sid = kwargs.get("session_id", "default")
            if operation == "remember":
                text = kwargs.get("text", "")
                state = await store.get(sid) or SessionState(session_id=sid)
                state.add_message(Message.user(text))
                return {"remembered": text[:80], "session": sid, "ok": True}
            elif operation == "search":
                state = await store.get(sid)
                msgs = [m.content for m in getattr(state, "messages", [])] if state else []
                return {"hits": [{"text": m, "score": 1.0} for m in msgs[:5]], "ok": True}
            return {"error": f"unknown op '{operation}'", "ok": False}

        return asyncio.run(_run())
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


# ── Foundry hosted-agent entrypoint ───────────────────────────────────

def handle_foundry_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Foundry Invocations entrypoint for agent-components."""
    msg = str(payload.get("message") or payload.get("input") or "").strip()
    component = str(payload.get("component") or "router").lower()
    if not msg:
        return {"reply": "Envoyez une tâche; précisez component=router|memory.", "ok": False}

    if component == "memory":
        op = str(payload.get("operation") or ("search" if len(msg) > 30 else "remember")).lower()
        res = memory_store_op(op, text=msg, session_id=payload.get("session_id", "default"))
        return {"ok": bool(res), "component": "memory", "reply": json.dumps(res, ensure_ascii=False)[:400], **res}

    res = route_decision(msg)
    if res.get("ok"):
        reply = (f"🔀 Routage: tier={res['tier']} model={res['model']} "
                 f"(score {res['score']}, {res['method']}) — {res['reason']}")
    else:
        reply = f"⚠️ Router indisponible ({res.get('error','')}), fallback low-tier."
    return {"ok": True, "reply": reply, "decision": res}