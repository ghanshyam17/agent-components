"""memory-store: pluggable short-term (session) + long-term (vector) memory.

Quick start::

    from memory_store import build_session_store, build_vector_memory
    from memory_store.embeddings import HashingEmbedder

    sessions = build_session_store()                 # in-memory by default
    vm = build_vector_memory(embedder=HashingEmbedder())
"""
from __future__ import annotations

from typing import Any

from components_core import Message, Plan, SessionState, SubTask, ToolCallRequest
from memory_store.base import SessionStore, VectorMemory
from memory_store.embeddings import Embedder, HashingEmbedder, OpenAIEmbedder
from memory_store.in_memory import InMemorySessionStore, InMemoryVectorMemory
from memory_store.models import MemoryQuery, MemoryRecord
from memory_store.config import MemorySettings, get_settings

__all__ = [
    "SessionStore", "VectorMemory",
    "InMemorySessionStore", "InMemoryVectorMemory",
    "Embedder", "HashingEmbedder", "OpenAIEmbedder",
    "MemoryRecord", "MemoryQuery",
    "MemorySettings", "get_settings",
    "Message", "Plan", "SessionState", "SubTask", "ToolCallRequest",
    "build_session_store", "build_vector_memory",
]


def build_session_store(settings: MemorySettings | None = None) -> SessionStore:
    """Construct a SessionStore from settings (backend: memory | redis)."""
    s = settings or get_settings()
    if s.backend == "redis":
        try:
            import redis.asyncio  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Redis backend requires the 'redis' extra: pip install memory-store[redis]"
            ) from e
        from memory_store.redis_store import RedisSessionStore
        client = redis.asyncio.from_url(s.redis_url)
        return RedisSessionStore(client, ttl=s.redis_session_ttl)
    return InMemorySessionStore()


def build_vector_memory(
    embedder: Embedder | None = None, settings: MemorySettings | None = None
) -> VectorMemory:
    """Construct an in-memory VectorMemory with the given (or default) embedder."""
    s = settings or get_settings()
    if embedder is None:
        embedder = OpenAIEmbedder(
            model=s.embed_model, base_url=s.embed_base_url,
            api_key=s.embed_api_key, dim=s.embed_dim,
        )
    return InMemoryVectorMemory(embedder)