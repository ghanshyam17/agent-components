"""agent_components_foundry.py — Foundry SDK + Azure Functions adapter.

Exposes the composable component monorepo to Foundry:

  - route_decision(task) -> which model tier the agentic-router picks (async)
  - memory ops via memory_store (store/search session memories)
  - handle_foundry_request(payload) -> hosted-agent entrypoint
    {"message": task, "component": "router"|"memory"} -> decision / recall

Pure-stdlib fallbacks when component deps are absent —
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
    Path(__file__).resolve().parent.parent / "core" / "src",
]
for _p in _pkg_parents:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def route_decision(task: str) -> Dict[str, Any]:
    """Ask the agentic-router which model tier should handle this task."""
    try:
        from agentic_router.router.router import Router
        from agentic_router.config import Settings
        s = Settings()
        router = Router(s, None)
        decision = asyncio.run(router.route(task))
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
                state = await store.get_or_create(sid)
                state.messages.append(Message(role="user", content=text))
                return {"remembered": text[:80], "session": sid, "ok": True}
            elif operation == "search":
                state = await store.get(sid)
                msgs = [m.content for m in getattr(state, "messages", [])] if state else []
                return {"hits": [{"text": m, "score": 1.0} for m in msgs[:5]], "ok": True}
            return {"error": f"unknown op '{operation}'", "ok": False}

        return asyncio.run(_run())
    except Exception as e:
        return {"ok": False, "error": str(e)[:200]}


def handle_foundry_request(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Foundry hosted-agent entrypoint.

    Routing:
      component="router" (default) -> agentic-router tier decision
      component="memory"           -> remember / search over session memory,
                                      with operation inferred from payload
    """
    msg = str(payload.get("message") or payload.get("input") or "").strip()
    component = str(payload.get("component") or "router").lower()

    if component == "memory":
        op = str(payload.get("operation") or ("search" if len(msg) > 30 else "remember")).lower()
        res = memory_store_op(
            op, text=msg, session_id=payload.get("session_id", "default")
        )
        return {"component": "memory", "reply": json.dumps(res, ensure_ascii=False)[:400], **res}

    dec = route_decision(msg or "hello")
    if dec.get("ok"):
        reply = (
            f"🔀 Routage: tier={dec['tier']} model={dec['model']} "
            f"(score {dec['score']}, {dec['method']}) — {dec['reason']}"
        )
    else:
        reply = f"⚠️ Router indisponible ({dec.get('error', '')}), fallback low-tier."
    return {"ok": True, "reply": reply, "decision": dec, **dec}
