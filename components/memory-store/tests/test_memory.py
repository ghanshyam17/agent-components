"""memory-store tests: in-memory session + vector memory with HashingEmbedder."""
from __future__ import annotations

import asyncio

import pytest

from memory_store import (
    HashingEmbedder, InMemorySessionStore, InMemoryVectorMemory,
    MemoryQuery, MemoryRecord, build_session_store, build_vector_memory,
)
from memory_store.config import MemorySettings
from components_core import Message, Plan, SubTask


# ---------------- session store ----------------
def test_session_create_get():
    store = InMemorySessionStore()
    state = asyncio.get_event_loop().run_until_complete(store.create())
    assert state.session_id
    fetched = asyncio.get_event_loop().run_until_complete(store.get(state.session_id))
    assert fetched is not None and fetched.session_id == state.session_id


def test_session_add_message_and_plan():
    store = InMemorySessionStore()
    sid = asyncio.get_event_loop().run_until_complete(store.create()).session_id
    asyncio.get_event_loop().run_until_complete(
        store.add_message(sid, Message(role="user", content="hi"))
    )
    asyncio.get_event_loop().run_until_complete(
        store.set_plan(sid, Plan(goal="g", subtasks=[SubTask(description="a")]))
    )
    state = asyncio.get_event_loop().run_until_complete(store.get(sid))
    assert len(state.messages) == 1
    assert state.plan is not None and state.plan.subtasks[0].description == "a"


def test_session_get_or_create_and_delete():
    store = InMemorySessionStore()
    sid = asyncio.get_event_loop().run_until_complete(store.create()).session_id
    again = asyncio.get_event_loop().run_until_complete(store.get_or_create(sid))
    assert again.session_id == sid
    new = asyncio.get_event_loop().run_until_complete(store.get_or_create(None))
    assert new.session_id != sid
    assert asyncio.get_event_loop().run_until_complete(store.delete(sid)) is True
    assert asyncio.get_event_loop().run_until_complete(store.get(sid)) is None


def test_session_get_or_create_unknown_id_creates_new():
    store = InMemorySessionStore()
    res = asyncio.get_event_loop().run_until_complete(store.get_or_create("nope"))
    assert res.session_id != "nope"


# ---------------- vector memory ----------------
def test_vector_memory_recall_by_similarity():
    vm = InMemoryVectorMemory(HashingEmbedder(dim=128))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        vm.add([
            MemoryRecord(content="the python function returns a value"),
            MemoryRecord(content="rust ownership and borrowing rules"),
            MemoryRecord(content="how to bake sourdough bread"),
        ])
    )
    hits = loop.run_until_complete(
        vm.search(MemoryQuery(query="python function value", top_k=2))
    )
    assert hits, "expected at least one hit"
    top_content = hits[0][0].content
    assert "python" in top_content
    assert all(isinstance(s, float) for _, s in hits)


def test_vector_memory_filter_and_count():
    vm = InMemoryVectorMemory(HashingEmbedder(dim=64))
    loop = asyncio.get_event_loop()
    loop.run_until_complete(
        vm.add([
            MemoryRecord(content="a", metadata={"kind": "doc"}),
            MemoryRecord(content="b", metadata={"kind": "note"}),
            MemoryRecord(content="c", metadata={"kind": "doc"}),
        ])
    )
    assert loop.run_until_complete(vm.count()) == 3
    hits = loop.run_until_complete(
        vm.search(MemoryQuery(query="a", top_k=10, filter={"kind": "doc"}))
    )
    assert all(r.metadata["kind"] == "doc" for r, _ in hits)
    assert len(hits) == 2


def test_build_session_store_defaults_in_memory():
    s = build_session_store(MemorySettings(backend="memory"))
    assert isinstance(s, InMemorySessionStore)


def test_build_vector_memory_with_custom_embedder():
    e = HashingEmbedder(dim=32)
    vm = build_vector_memory(embedder=e)
    assert isinstance(vm, InMemoryVectorMemory)
    assert vm.embedder is e