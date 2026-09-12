"""Redis-backed session memory.

Uses `redis.asyncio`. Long-term vector memory over Redis is intentionally not
implemented here (use RedisVL / RediSearch or the pgvector backend in the
future `retriever` component); this module only handles short-term sessions,
stored as JSON per session id.
"""
from __future__ import annotations

import uuid
from typing import Any

from components_core import Message, Plan, SessionState
from memory_store.base import SessionStore

_PREFIX = "sess:"


class RedisSessionStore(SessionStore):
    def __init__(self, client: Any, ttl: int | None = None):
        # `client` is a redis.asyncio.Redis instance — kept untyped so this
        # module imports even when the `redis` extra isn't installed.
        self.client = client
        self.ttl = ttl

    async def _set(self, sid: str, state: SessionState) -> None:
        state.touch()
        payload = state.model_dump_json()
        if self.ttl is not None:
            await self.client.set(_PREFIX + sid, payload, ex=self.ttl)
        else:
            await self.client.set(_PREFIX + sid, payload)

    async def _get(self, sid: str) -> SessionState | None:
        raw = await self.client.get(_PREFIX + sid)
        if raw is None:
            return None
        return SessionState.model_validate_json(raw)

    async def create(self) -> SessionState:
        sid = uuid.uuid4().hex[:12]
        state = SessionState(session_id=sid)
        await self._set(sid, state)
        return state

    async def get(self, session_id: str) -> SessionState | None:
        return await self._get(session_id)

    async def get_or_create(self, session_id: str | None) -> SessionState:
        if session_id:
            state = await self._get(session_id)
            if state is not None:
                return state
        return await self.create()

    async def add_message(self, session_id: str, message: Message) -> None:
        state = await self._get(session_id)
        if state is None:
            return
        state.messages.append(message)
        await self._set(session_id, state)

    async def set_plan(self, session_id: str, plan: Plan) -> None:
        state = await self._get(session_id)
        if state is None:
            return
        state.plan = plan
        await self._set(session_id, state)

    async def delete(self, session_id: str) -> bool:
        return bool(await self.client.delete(_PREFIX + session_id))

    async def list_ids(self) -> list[str]:
        keys = await self.client.keys(_PREFIX + "*")
        return [k.decode().removeprefix(_PREFIX) for k in keys]