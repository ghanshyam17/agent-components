"""FastAPI app exposing /chat, /agent (SSE), /sessions, and /route.

The Agent is built once and shared across requests; sessions live in memory.
"""
from __future__ import annotations

import json
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from agentic_router.factory import build_agent
from agentic_router.models import AgentEvent


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.agent = build_agent()
    yield


app = FastAPI(
    title="Agentic Router",
    description="vLLM-backed agentic router: lower/higher model routing with tools, planning, streaming and sessions.",
    version="0.1.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    task: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    tier: str | None = None
    model: str | None = None


class RouteRequest(BaseModel):
    task: str


class RouteResponse(BaseModel):
    tier: str
    score: float
    method: str
    model: str
    reason: str
    signals: dict


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
async def route(req: RouteRequest) -> RouteResponse:
    agent = app.state.agent
    decision = await agent.router.route(req.task)
    return RouteResponse(
        tier=decision.tier.value, score=decision.score, method=decision.method,
        model=decision.model, reason=decision.reason, signals=decision.signals,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    agent = app.state.agent
    answer, decision, state = await agent.run(req.task, req.session_id)
    return ChatResponse(
        answer=answer, session_id=state.session_id,
        tier=decision.tier.value if decision else None,
        model=decision.model if decision else None,
    )


@app.get("/agent/stream")
async def agent_stream(task: str, session_id: str | None = None):
    """Server-Sent Events stream of AgentEvent objects.

    Each SSE `data:` line is a JSON AgentEvent. Event types include
    `plan`, `route`, `subtask_start`, `iteration`, `delta`, `tool_call`,
    `tool_result`, `answer`, `final`, `error`.
    """
    from sse_starlette.sse import EventSourceResponse

    agent = app.state.agent
    state = await agent.sessions.get_or_create(session_id)

    async def gen():
        try:
            async for ev in agent.stream(task, state.session_id):
                yield {"event": ev.type, "data": _encode(ev)}
        except Exception as e:  # noqa: BLE001
            err = AgentEvent(type="error", data={"error": str(e)})
            yield {"event": "error", "data": _encode(err)}

    return EventSourceResponse(gen())


@app.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    agent = app.state.agent
    state = await agent.sessions.get(session_id)
    if state is None:
        raise HTTPException(404, "session not found")
    return {
        "session_id": state.session_id,
        "messages": [m.model_dump() for m in state.messages],
        "plan": state.plan.model_dump() if state.plan else None,
        "created_at": state.created_at,
        "updated_at": state.updated_at,
    }


@app.get("/sessions")
async def list_sessions() -> dict:
    agent = app.state.agent
    return {"session_ids": await agent.sessions.list_ids()}


def _encode(ev: AgentEvent) -> str:
    return json.dumps(ev.model_dump())


def create_app() -> FastAPI:
    return app