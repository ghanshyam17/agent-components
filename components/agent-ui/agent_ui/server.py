"""FastAPI Server Bridge for Agent UI.

Exposes metadata, graph topology, and streaming chat endpoints over SSE
for consumption by Next.js / TypeScript frontends or the embedded UI canvas.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any, AsyncIterator, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from agent_ui.bridge import UIBridge

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = "session_default"


def create_ui_app(target: Any = None) -> FastAPI:
    """Factory creating a FastAPI server for the universal Agent UI."""
    app = FastAPI(
        title="Agent UI Server",
        description="Universal UI Bridge for Agent Components & Component Graphs",
        version="0.1.0",
    )

    # Enable CORS for local dev with Next.js
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    bridge = UIBridge(target)

    @app.get("/api/health")
    async def health() -> Dict[str, str]:
        return {"status": "ok", "service": "agent-ui"}

    @app.get("/api/config")
    async def get_config() -> Dict[str, Any]:
        """Returns agent persona, tools, and starter prompts."""
        return bridge.get_config()

    @app.get("/api/graph")
    async def get_graph() -> Dict[str, Any]:
        """Returns node topology for DAG graph visualizer."""
        return bridge.get_graph()

    @app.post("/api/chat")
    async def chat(req: ChatRequest) -> StreamingResponse:
        """Streams agent thought process, tool calls, and final answers over SSE."""
        async def event_generator() -> AsyncIterator[str]:
            # Initial event
            yield f"data: {json.dumps({'type': 'start', 'message': req.message})}\n\n"

            if hasattr(target, "execute_task"):
                # Target is a ComponentGraphOrchestrator
                res = await target.execute_task(req.message, session_id=req.session_id)
                for ev in res.events:
                    yield f"data: {json.dumps({'type': ev.type, 'data': ev.data})}\n\n"
                    await asyncio.sleep(0.02)
                yield f"data: {json.dumps({'type': 'final', 'content': res.answer})}\n\n"

            elif hasattr(target, "stream"):
                # Target is an AgentPattern or agentic-router Agent
                async for ev in target.stream(task=req.message, session_id=req.session_id):
                    ev_dict = ev.model_dump() if hasattr(ev, "model_dump") else {"type": getattr(ev, "type", "delta"), "data": getattr(ev, "data", {})}
                    yield f"data: {json.dumps(ev_dict)}\n\n"
                    await asyncio.sleep(0.01)

            else:
                # Simulated responses for demo
                yield f"data: {json.dumps({'type': 'route', 'data': {'first_route': 'langgraph', 'reason': 'stateful_workflow'}})}\n\n"
                await asyncio.sleep(0.05)
                yield f"data: {json.dumps({'type': 'subtask_start', 'data': {'description': 'Querying Gold Medallion Lakehouse table'}})}\n\n"
                await asyncio.sleep(0.08)
                yield f"data: {json.dumps({'type': 'tool_call', 'data': {'tool': 'query_lakehouse', 'arguments': {'query_or_table': 'gold_sales_kpis'}}})}\n\n"
                await asyncio.sleep(0.08)
                yield f"data: {json.dumps({'type': 'tool_result', 'data': {'result': 'Retrieved 10 curated KPI records.'}})}\n\n"
                await asyncio.sleep(0.05)
                reply = f"Based on the Medallion Gold Lakehouse metrics and data pipeline outputs, here is the synthesis for: '{req.message}'. All systems operational."
                yield f"data: {json.dumps({'type': 'delta', 'content': reply})}\n\n"
                yield f"data: {json.dumps({'type': 'final', 'content': reply})}\n\n"

            yield "data: [DONE]\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui() -> HTMLResponse:
        """Serves the embedded Fluent-styled web canvas."""
        index_path = STATIC_DIR / "index.html"
        if index_path.is_file():
            return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
        return HTMLResponse(content="<h1>Agent UI is active. Connect with Next.js or check /api/config.</h1>")

    return app


def mount_agent_ui(app: FastAPI, target: Any = None, prefix: str = "") -> None:
    """Mounts the Agent UI router onto an existing FastAPI application."""
    ui_app = create_ui_app(target)
    app.mount(prefix, ui_app)
