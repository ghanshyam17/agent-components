"""Tests for checkpointing as wired into the consumers.

The context-manager tests cover the checkpointer itself. These cover the three
wiring points — the ReAct loop, the LangGraph pattern, the orchestrator — where
the failure mode is different: not "does resume work" but "is progress recorded
at the boundary where work becomes irreversible, and is a degradation visible".
"""
from __future__ import annotations

import asyncio
import tempfile

import pytest

from context_manager.checkpoint import (
    DegradedCheckpointer,
    FileCheckpointer,
    InMemoryCheckpointer,
    StepOutcome,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


# --------------------------------------------------------------------------- #
# The bug this change fixed: a checkpointer that was write-only
# --------------------------------------------------------------------------- #
def test_langgraph_declares_checkpointing_and_can_read_it_back(tmpdir_path):
    """Regression: the old implementation stored state in a dict and never read
    it, so declaring a checkpointer produced an in-RAM log with no resume path."""
    from platform.patterns.base import PatternContext
    from platform.patterns.langgraph import LangGraphPattern

    nodes = [{"name": "start"}]
    ctx = PatternContext(task="t", session_id="s1")

    p = LangGraphPattern(nodes=nodes, edges=[], checkpointer="file", directory=tmpdir_path)
    run(_drain(p.stream(ctx)))

    # A new instance stands in for a new process.
    p2 = LangGraphPattern(nodes=nodes, edges=[], checkpointer="file", directory=tmpdir_path)
    restored = run(p2.resume_from_checkpoint("s1"))
    assert restored is not None, "state must be readable back — the old code had no read path"
    assert "steps" in restored


def test_langgraph_memory_backend_is_honest_about_durability():
    from platform.patterns.langgraph import LangGraphPattern

    cap = LangGraphPattern(nodes=[{"name": "s"}], checkpointer="memory").checkpoint_capability()
    assert cap["backend"] == "memory"
    assert cap["durable"] is False


def test_langgraph_unconfigured_backend_degrades_loudly_not_silently():
    """The old code silently did nothing when the backend was not 'memory'.
    Now the degradation is queryable rather than a vanished write."""
    from platform.patterns.langgraph import LangGraphPattern

    p = LangGraphPattern(nodes=[{"name": "s"}], checkpointer="redis")
    cap = p.checkpoint_capability()
    assert cap["durable"] is False
    assert "DEGRADED" in cap["detail"] and "redis" in cap["detail"]


def test_langgraph_unknown_backend_falls_back_with_a_warning(caplog):
    import logging

    from platform.patterns.langgraph import LangGraphPattern

    with caplog.at_level(logging.WARNING):
        p = LangGraphPattern(nodes=[{"name": "s"}], checkpointer="dynamodb")
    assert p.checkpoint_capability()["backend"] == "memory"
    assert any("unknown checkpointer" in r.message for r in caplog.records)


def test_langgraph_survives_a_failing_checkpointer(tmpdir_path):
    """A checkpoint failure must not take the graph run down."""
    from platform.patterns.base import PatternContext
    from platform.patterns.langgraph import LangGraphPattern

    p = LangGraphPattern(nodes=[{"name": "start"}], checkpointer="memory")

    class Boom:
        name = "boom"

        async def save(self, cp):
            raise RuntimeError("disk full")

        async def latest(self, run_id):
            return None

        def capability(self):
            from context_manager.checkpoint import CheckpointCapability

            return CheckpointCapability(
                backend="boom", durable=False, shared=False, atomic=False, history=False
            )

    p._checkpointer = Boom()
    ctx = PatternContext(task="t", session_id="s")
    events = run(_drain(p.stream(ctx)))          # must not raise
    assert events, "the run must still produce events"


def test_langgraph_to_foundry_config_reports_the_checkpointer():
    from platform.patterns.langgraph import LangGraphPattern

    cfg = LangGraphPattern(nodes=[{"name": "s"}], checkpointer="memory").to_foundry_config()
    assert cfg["checkpointer"] == "memory"


# --------------------------------------------------------------------------- #
# ReAct loop: tool-call checkpointing
# --------------------------------------------------------------------------- #
def _bare_agent(checkpointer=None):
    """An Agent with only the checkpointing surface wired.

    Bypasses the real constructor because these tests are about the ledger, not
    about model clients — the same reason the component uses injected engines.
    """
    from agentic_router.agent.loop import Agent

    agent = Agent.__new__(Agent)
    agent.checkpointer = checkpointer
    return agent


class _Call:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


def test_tool_keys_are_content_addressed(tmpdir_path):
    a = _bare_agent()
    call = _Call("charge_card", {"amount": 500})
    k1 = a._tool_key(0, call)
    assert a._tool_key(0, _Call("charge_card", {"amount": 500})) == k1
    assert a._tool_key(0, _Call("charge_card", {"amount": 999})) != k1, \
        "a different amount must not be mistaken for the same call"
    assert a._tool_key(0, _Call("other_tool", {"amount": 500})) != k1


def test_tool_key_is_stable_across_instances():
    """Stability across restarts is the whole point of content-addressing."""
    assert _bare_agent()._tool_key(2, _Call("t", {"a": 1})) == \
        _bare_agent()._tool_key(2, _Call("t", {"a": 1}))


def test_tool_key_survives_unserialisable_arguments():
    a = _bare_agent()

    class Weird:
        def __str__(self):
            return "weird"

    assert a._tool_key(0, _Call("t", {"x": Weird()}))


def test_subtask_keys_change_when_the_plan_changes():
    """A re-planned run must not treat the previous run's completions as done."""
    a = _bare_agent()
    assert a._subtask_key(1, "do alpha") != a._subtask_key(1, "do beta")
    assert a._subtask_key(1, "do alpha") == a._subtask_key(1, "do alpha")


def test_checkpoint_helpers_are_no_ops_without_a_checkpointer():
    """Checkpointing is opt-in and must cost nothing when absent."""
    a = _bare_agent(None)
    run(a._checkpoint_tool("s", "t", "k", _Call("x", {}), StepOutcome.PENDING))
    run(a._checkpoint_subtask("s", ["a"], "subtask:1:x", StepOutcome.PENDING))
    assert run(a._load_state("s")) == {}
    assert run(a._completed_subtasks("s", ["a"])) == set()


def test_a_pending_tool_call_is_recorded_before_dispatch(tmpdir_path):
    """The pre-write is what makes a mid-tool crash detectable."""
    a = _bare_agent(FileCheckpointer(tmpdir_path))
    call = _Call("charge_card", {"amount": 500})
    key = a._tool_key(0, call)
    run(a._checkpoint_tool("s1", "buy", key, call, StepOutcome.PENDING))

    # A fresh agent, reading only from disk.
    a2 = _bare_agent(FileCheckpointer(tmpdir_path))
    ledger = run(a2._load_state("s1"))
    assert ledger[key] is StepOutcome.PENDING

    from context_manager.checkpoint import plan_resume

    prev = run(a2.checkpointer.latest(a2._run_id("s1")))
    plan = plan_resume(prev, [key])
    assert plan.resume_at == key
    assert any("unknown" in w for w in plan.warnings)


def test_a_completed_tool_call_is_recognised_as_done(tmpdir_path):
    a = _bare_agent(FileCheckpointer(tmpdir_path))
    call = _Call("lookup", {"id": 1})
    key = a._tool_key(0, call)
    run(a._checkpoint_tool("s2", "find", key, call, StepOutcome.COMPLETED))

    a2 = _bare_agent(FileCheckpointer(tmpdir_path))
    assert run(a2._load_state("s2"))[key] is StepOutcome.COMPLETED


def test_completed_subtasks_are_reported_for_skipping(tmpdir_path):
    a = _bare_agent(FileCheckpointer(tmpdir_path))
    descs = ["first", "second"]
    key = a._subtask_key(1, descs[0])
    run(a._checkpoint_subtask("s3", descs, key, StepOutcome.COMPLETED))

    a2 = _bare_agent(FileCheckpointer(tmpdir_path))
    done = run(a2._completed_subtasks("s3", descs))
    assert key in done
    assert a2._subtask_key(2, descs[1]) not in done


def test_the_ledger_accumulates_across_tool_calls(tmpdir_path):
    a = _bare_agent(FileCheckpointer(tmpdir_path))
    for i, name in enumerate(("a", "b", "c")):
        call = _Call(name, {"i": i})
        run(a._checkpoint_tool("s4", "task", a._tool_key(0, call), call,
                               StepOutcome.COMPLETED))
    ledger = run(a._load_state("s4"))
    assert len(ledger) == 3
    assert all(v is StepOutcome.COMPLETED for v in ledger.values())


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def _orchestrator(checkpointer):
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator

    return ComponentGraphOrchestrator.__new__(ComponentGraphOrchestrator), checkpointer


def test_orchestrator_checkpoint_records_pending_then_completed(tmpdir_path):
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator

    orch = ComponentGraphOrchestrator.__new__(ComponentGraphOrchestrator)
    orch.checkpointer = FileCheckpointer(tmpdir_path)

    run(orch._checkpoint("s", "task", status="running", step="execute_task"))
    assert run(orch.resume_state("s")) is not None
    assert run(orch.was_interrupted("s")) is True, \
        "a run that started but never completed must read as interrupted"

    run(orch._checkpoint("s", "task", status="completed", step="execute_task",
                         state={"answer": "done"}))
    assert run(orch.was_interrupted("s")) is False
    assert run(orch.resume_state("s"))["answer"] == "done"


def test_orchestrator_helpers_are_no_ops_without_a_checkpointer():
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator

    orch = ComponentGraphOrchestrator.__new__(ComponentGraphOrchestrator)
    orch.checkpointer = None
    run(orch._checkpoint("s", "t", status="running", step="execute_task"))
    assert run(orch.resume_state("s")) is None
    assert run(orch.was_interrupted("s")) is False


def test_orchestrator_survives_a_failing_checkpointer():
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator

    class Boom:
        async def save(self, cp):
            raise RuntimeError("nope")

        async def latest(self, run_id):
            return None

    orch = ComponentGraphOrchestrator.__new__(ComponentGraphOrchestrator)
    orch.checkpointer = Boom()
    run(orch._checkpoint("s", "t", status="running", step="x"))   # must not raise


def test_orchestrator_run_ids_are_session_scoped():
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator

    orch = ComponentGraphOrchestrator.__new__(ComponentGraphOrchestrator)
    assert orch._run_id("abc") == "orchestrator:abc"
    assert orch._run_id("abc") != orch._run_id("def")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
async def _drain(agen):
    return [ev async for ev in agen]
