"""In-memory backends — zero dependencies, the default for tests and demos.

* `InMemorySessionStore` is a drop-in replacement for the router's old
  in-process session dict.
* `InMemoryVectorMemory` does brute-force cosine similarity — fine up to a
  few thousand records.
"""
from __future__ import annotations

import math
import uuid
from collections import OrderedDict

from components_core import Message, Plan, SessionState
from memory_store.base import SessionStore, VectorMemory
from memory_store.embeddings import Embedder
from memory_store.models import MemoryRecord, MemoryQuery


class InMemorySessionStore(SessionStore):
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    async def create(self) -> SessionState:
        sid = uuid.uuid4().hex[:12]
        state = SessionState(session_id=sid)
        self._sessions[sid] = state
        return state

    async def get(self, session_id: str) -> SessionState | None:
        state = self._sessions.get(session_id)
        if state:
            state.touch()
        return state

    async def get_or_create(self, session_id: str | None) -> SessionState:
        if session_id and session_id in self._sessions:
            state = self._sessions[session_id]
            state.touch()
            return state
        return await self.create()

    async def add_message(self, session_id: str, message: Message) -> None:
        if state := self._sessions.get(session_id):
            state.messages.append(message)
            state.touch()

    async def set_plan(self, session_id: str, plan: Plan) -> None:
        if state := self._sessions.get(session_id):
            state.plan = plan
            state.touch()

    async def delete(self, session_id: str) -> bool:
        return self._sessions.pop(session_id, None) is not None

    async def list_ids(self) -> list[str]:
        return list(self._sessions)


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class InMemoryVectorMemory(VectorMemory):
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._records: "OrderedDict[str, MemoryRecord]" = OrderedDict()

    async def add(self, records: list[MemoryRecord], embed: bool = True) -> list[str]:
        if embed:
            texts = [r.content for r in records]
            vecs = await self.embedder.embed(texts)
            for r, v in zip(records, vecs):
                r.embedding = v
        ids: list[str] = []
        for r in records:
            self._records[r.id] = r
            ids.append(r.id)
        return ids

    async def search(self, query: MemoryQuery) -> list[tuple[MemoryRecord, float]]:
        if not self._records:
            return []
        qvec = (await self.embedder.embed([query.query]))[0]
        scored: list[tuple[MemoryRecord, float]] = []
        for rec in self._records.values():
            if query.filter and not _matches(rec.metadata, query.filter):
                continue
            score = _cosine(qvec, rec.embedding or [])
            if score >= query.min_score:
                scored.append((rec, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[: query.top_k]

    async def count(self) -> int:
        return len(self._records)


def _matches(metadata: dict, flt: dict) -> bool:
    return all(metadata.get(k) == v for k, v in flt.items())