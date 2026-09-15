"""Tests for the FrenchCase → chassis bridges.

These run in two modes:
  * Unmounted — FrenchCase not on sys.path. Everything must import and degrade
    to no-op without raising.
  * Mounted   — FrenchCase available (via projects.frenchcase.mount). The
    bridges must return real data from FrenchCase's engines.

NOTE ON ASYNC STYLE
-------------------
Every async test here is a native `async def` rather than a sync test wrapping
`asyncio.run(...)`. The repo sets `asyncio_mode = "auto"`, and calling
`asyncio.run()` inside a test tears down the main thread's event loop, which
breaks OTHER components' tests that use `asyncio.get_event_loop()`
(components/model-gateway does). Keeping the async style native avoids
cross-suite interference.

Run:  .venv/bin/python -m pytest projects/frenchcase/tests/ -v
"""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

# Make the repo root importable (tests may run from anywhere).
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from projects.frenchcase.mount import mount_frenchcase, frenchcase_home


# ── fixtures ─────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def _preserve_main_event_loop():
    """Keep a main-thread event loop alive for other components' tests.

    pytest-asyncio closes the loop it creates for each async test and leaves
    the main thread with no current loop. Pre-existing chassis tests (notably
    components/model-gateway) call `asyncio.get_event_loop()` synchronously and
    then fail with "There is no current event loop in thread 'MainThread'".

    Restoring a fresh loop after each of our tests keeps those suites working
    when the whole repo is collected in one session.
    """
    yield
    try:
        asyncio.get_event_loop_policy().get_event_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())


@pytest.fixture(scope="module")
def fc_available() -> bool:
    """True when the FrenchCase repo is present on this machine."""
    return mount_frenchcase()


# ── mount helper ─────────────────────────────────────────────────────────

def test_mount_reports_missing_home_gracefully(tmp_path: Path) -> None:
    """A nonexistent home must return False, not raise."""
    from projects.frenchcase import mount as m

    m._mounted = False  # reset module state
    assert m.mount_frenchcase(tmp_path / "does-not-exist") is False


def test_mount_puts_root_first_and_app_last() -> None:
    """Order matters: repo root must precede app/ or `app` shadows app/app.py."""
    ok = mount_frenchcase()
    if not ok:
        pytest.skip("FrenchCase repo not present")
    root = str(frenchcase_home())
    app_dir = str(frenchcase_home() / "app")
    assert root in sys.path and app_dir in sys.path
    assert sys.path.index(root) < sys.path.index(app_dir)


# ── memory bridge ────────────────────────────────────────────────────────

def test_memory_bridge_imports_without_frenchcase() -> None:
    """The module must import and expose the factory regardless of mounting."""
    from projects.frenchcase.bridges.memory_bridge import build_session_store

    assert callable(build_session_store)


async def test_memory_bridge_session_roundtrip(fc_available: bool) -> None:
    if not fc_available:
        pytest.skip("FrenchCase repo not present")

    from components_core import Message
    from projects.frenchcase.bridges.memory_bridge import (
        build_session_store,
        FRENCHCASE_AVAILABLE,
    )

    assert FRENCHCASE_AVAILABLE, "FrenchCase SessionStore should be importable"

    store = build_session_store()
    sid = "pytest-bridge-session"

    state = await store.get_or_create(sid)
    assert state.session_id == sid

    await store.add_message(sid, Message(role="user", content="explique le subjonctif"))
    await store.add_message(sid, Message(role="assistant", content="Le subjonctif s'emploie..."))

    fetched = await store.get(sid)
    assert fetched is not None
    assert len(fetched.messages) == 2
    assert fetched.messages[0].role == "user"
    assert fetched.messages[1].content.startswith("Le subjonctif")

    assert sid in await store.list_ids()
    assert await store.delete(sid) is True
    assert sid not in await store.list_ids()


async def test_memory_bridge_explicit_id_not_reminted(fc_available: bool) -> None:
    """FrenchCase's create() mints its own id; the bridge must honour ours."""
    if not fc_available:
        pytest.skip("FrenchCase repo not present")

    from projects.frenchcase.bridges.memory_bridge import build_session_store

    store = build_session_store()
    sid = "pytest-explicit-id"
    state = await store.get_or_create(sid)
    assert state.session_id == sid, "requested session id must be preserved"
    await store.delete(sid)


# ── retriever bridge ─────────────────────────────────────────────────────

def test_retriever_bridge_imports_without_frenchcase() -> None:
    from projects.frenchcase.bridges.retriever_bridge import build_vector_memory

    assert callable(build_vector_memory)


async def test_retriever_bridge_add_is_noop(fc_available: bool) -> None:
    """The TEF corpus is externally indexed — add() must not duplicate it."""
    if not fc_available:
        pytest.skip("FrenchCase repo not present")

    from memory_store.models import MemoryRecord
    from projects.frenchcase.bridges.retriever_bridge import build_vector_memory

    vm = build_vector_memory()
    ids = await vm.add([MemoryRecord(content="should be ignored")])
    assert ids == []


async def test_retriever_bridge_search_returns_list(fc_available: bool) -> None:
    """search() must return a list of (MemoryRecord, score) — empty is fine
    when the vector DB isn't running locally."""
    if not fc_available:
        pytest.skip("FrenchCase repo not present")

    from memory_store.models import MemoryQuery
    from projects.frenchcase.bridges.retriever_bridge import build_vector_memory

    vm = build_vector_memory()
    hits = await vm.search(MemoryQuery(query="subjonctif", top_k=3))
    assert isinstance(hits, list)
    for rec, score in hits:
        assert isinstance(score, float)
        assert isinstance(rec.content, str)


# ── llm bridge ───────────────────────────────────────────────────────────

def test_llm_bridge_imports_without_frenchcase() -> None:
    from projects.frenchcase.bridges.llm_bridge import build_gateway, ALIAS_TASK_MAP

    assert callable(build_gateway)
    assert set(ALIAS_TASK_MAP) >= {"lower", "higher"}


def test_llm_bridge_alias_task_map_targets_real_tasks(fc_available: bool) -> None:
    """Every alias must name a task that exists in FrenchCase's complexity map,
    or the router silently falls back to MODERATE."""
    if not fc_available:
        pytest.skip("FrenchCase repo not present")

    from app.core.llm_router import AGENT_COMPLEXITY
    from projects.frenchcase.bridges.llm_bridge import ALIAS_TASK_MAP

    for alias, task in ALIAS_TASK_MAP.items():
        assert task in AGENT_COMPLEXITY, f"alias {alias} → unknown task {task!r}"


def test_llm_bridge_gateway_exposes_lower_higher(fc_available: bool) -> None:
    if not fc_available:
        pytest.skip("FrenchCase repo not present")

    from projects.frenchcase.bridges.llm_bridge import build_gateway, FRENCHCASE_AVAILABLE

    assert FRENCHCASE_AVAILABLE
    gw = build_gateway()
    assert gw.available
    assert gw.client("lower").tier == "lower"
    assert gw.client("higher").tier == "higher"
    assert isinstance(gw.metrics(), dict)


# ── tools adapter ────────────────────────────────────────────────────────

def test_tools_adapter_exposes_five_tools() -> None:
    from projects.frenchcase.tools.learncase_tools import build_frenchcase_tools

    tools = build_frenchcase_tools()
    assert tools["count"] == 5
    assert set(tools["names"]) == {
        "repo_search",
        "get_conjugation",
        "grade_essay",
        "tts_pronounce",
        "load_lesson",
    }
    for d in tools["definitions"]:
        assert d["name"] and d["description"] and "parameters" in d


async def test_tools_adapter_unknown_tool_returns_error() -> None:
    from projects.frenchcase.tools.learncase_tools import execute_tool

    result = await execute_tool("no_such_tool", {})
    assert result["ok"] is False
    assert "Unknown tool" in result["error"]


# ── wiring ───────────────────────────────────────────────────────────────

def test_wiring_builds_stack_and_reports_status(fc_available: bool) -> None:
    from projects.frenchcase.wiring import build_frenchcase_stack

    stack = build_frenchcase_stack()
    status = stack["status"]

    assert status["tools"] == 5
    assert status["gateway_aliases"] == ["lower", "higher"]
    assert "rag" in stack and "sessions" in stack and "gateway" in stack

    if fc_available:
        assert status["session_store"] is True
        assert status["llm_router"] is True
        assert status["vector_memory"] is True