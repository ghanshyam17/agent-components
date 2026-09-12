"""Abstract interfaces for memory.

Two distinct concerns:

* `SessionStore` — short-term conversation memory: messages + plan keyed by
  `session_id`. Backends implement create/get/add-message/set-plan etc.
* `VectorMemory` — long-term semantic memory: store `MemoryRecord`s, recall
  by similarity. Backends implement add/search.

The router uses `SessionStore` (it injects whatever backend the app chooses).
Future components (retriever, agent loop) use `VectorMemory`.
"""
from __future__ import annotations

import abc
from typing import Any

from memory_store.models import MemoryRecord, MemoryQuery
from components_core import Message, Plan, SessionState


class SessionStore(abc.ABC):
    """Short-term conversation memory, keyed by session id."""

    @abc.abstractmethod
    async def create(self) -> SessionState: ...

    @abc.abstractmethod
    async def get(self, session_id: str) -> SessionState | None: ...

    @abc.abstractmethod
    async def get_or_create(self, session_id: str | None) -> SessionState: ...

    @abc.abstractmethod
    async def add_message(self, session_id: str, message: Message) -> None: ...

    @abc.abstractmethod
    async def set_plan(self, session_id: str, plan: Plan) -> None: ...

    @abc.abstractmethod
    async def delete(self, session_id: str) -> bool: ...

    @abc.abstractmethod
    async def list_ids(self) -> list[str]: ...


class VectorMemory(abc.ABC):
    """Long-term semantic memory over embedded records."""

    @abc.abstractmethod
    async def add(
        self, records: list[MemoryRecord], embed: bool = True
    ) -> list[str]: ...

    @abc.abstractmethod
    async def search(self, query: MemoryQuery) -> list[tuple[MemoryRecord, float]]: ...

    @abc.abstractmethod
    async def count(self) -> int: ...