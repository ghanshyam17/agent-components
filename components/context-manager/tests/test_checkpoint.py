"""Tests for checkpointing.

The property that matters is **resume**, so tests assert what a second run does
after a first one died — not merely that state was written. A checkpointer that
stores and never loads passes a naive "did save work" test and is useless for
the crash it was installed to survive.
"""
from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

import pytest

from context_manager.checkpoint import (
    BaseCheckpointer,
    CheckpointCapability,
    CheckpointedRun,
    Checkpointer,
    CheckpointStatus,
    CosmosCheckpointer,
    DegradedCheckpointer,
    FileCheckpointer,
    InMemoryCheckpointer,
    RedisCheckpointer,
    ResumePlan,
    RunCheckpoint,
    StepOutcome,
    build_checkpointer,
    plan_resume,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def tmpdir_path():
    with tempfile.TemporaryDirectory() as d:
        yield d


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
def test_capability_reports_what_it_cannot_do():
    cap = InMemoryCheckpointer().capability()
    assert cap.durable is False
    assert cap.backend == "memory"
    assert "does not survive" in cap.detail


def test_file_and_redis_and_cosmos_all_claim_durable():
    assert FileCheckpointer(tempfile.mkdtemp()).capability().durable is True
    assert RedisCheckpointer(client=object()).capability().durable is True
    assert CosmosCheckpointer(client=object()).capability().durable is True


def test_redis_and_cosmos_are_shared_memory_is_not():
    assert RedisCheckpointer(client=object()).capability().shared is True
    assert CosmosCheckpointer(client=object()).capability().shared is True
    assert InMemoryCheckpointer().capability().shared is False


def test_run_checkpoint_resume_point_finds_an_in_flight_step():
    cp = RunCheckpoint(
        run_id="r",
        step_outcomes={"a": StepOutcome.COMPLETED, "b": StepOutcome.PENDING},
    )
    assert cp.resume_point() == "b"
    assert cp.completed_steps() == {"a"}


def test_run_checkpoint_round_trips_through_dict():
    cp = RunCheckpoint(
        run_id="r", step_outcomes={"a": StepOutcome.COMPLETED},
        state={"k": [1, 2]}, status=CheckpointStatus.RUNNING,
    )
    back = RunCheckpoint.from_dict(cp.to_dict())
    assert back.step_outcomes == cp.step_outcomes
    assert back.state == cp.state
    assert back.status is CheckpointStatus.RUNNING


def test_checkpoint_dict_is_json_safe():
    cp = RunCheckpoint(run_id="r", step_outcomes={"a": StepOutcome.UNKNOWN})
    json.dumps(cp.to_dict())


# --------------------------------------------------------------------------- #
# Protocol conformance — the bug that made every backend unusable
# --------------------------------------------------------------------------- #
def test_every_backend_satisfies_the_protocol():
    """Regression: `delete` was declared on the Protocol but never defined on
    the base class, so *no* backend was recognised as a Checkpointer."""
    for obj in (
        InMemoryCheckpointer(),
        FileCheckpointer(tempfile.mkdtemp()),
        RedisCheckpointer(client=object()),
        CosmosCheckpointer(client=object()),
    ):
        assert isinstance(obj, Checkpointer), obj.name
        assert hasattr(obj, "delete") and hasattr(obj, "list_runs")


def test_delete_is_reachable_on_the_base_class():
    cp = InMemoryCheckpointer()
    assert run(cp.delete("nope")) is False


# --------------------------------------------------------------------------- #
# Sequencing and bounded history
# --------------------------------------------------------------------------- #
def test_sequence_numbers_are_assigned_monotonically():
    cp = InMemoryCheckpointer()
    for i in range(1, 4):
        saved = run(cp.save(RunCheckpoint(run_id="r", step=f"s{i}")))
        assert saved.sequence == i


def test_history_is_bounded():
    """A checkpointer must not become an unbounded log."""
    cp = InMemoryCheckpointer(max_history=3)
    for i in range(10):
        run(cp.save(RunCheckpoint(run_id="r", step=f"s{i}")))
    hist = run(cp.history("r"))
    assert len(hist) == 3
    assert hist[-1].step == "s9", "the newest must be retained"
    assert run(cp.latest("r")).step == "s9"


def test_latest_is_the_highest_sequence():
    cp = InMemoryCheckpointer()
    run(cp.save(RunCheckpoint(run_id="r", step="a")))
    run(cp.save(RunCheckpoint(run_id="r", step="b")))
    assert run(cp.latest("r")).step == "b"


def test_latest_returns_none_for_an_unknown_run():
    assert run(InMemoryCheckpointer().latest("ghost")) is None


def test_runs_are_isolated():
    cp = InMemoryCheckpointer()
    run(cp.save(RunCheckpoint(run_id="a", step="1")))
    run(cp.save(RunCheckpoint(run_id="b", step="2")))
    assert run(cp.latest("a")).step == "1"
    assert run(cp.latest("b")).step == "2"
    assert run(cp.list_runs()) == ["a", "b"]


def test_memory_backend_deep_copies_so_callers_cannot_corrupt_state():
    cp = InMemoryCheckpointer()
    run(cp.save(RunCheckpoint(run_id="r", state={"items": [1]})))
    first = run(cp.latest("r"))
    first.state["items"].append(99)
    assert run(cp.latest("r")).state["items"] == [1]


# --------------------------------------------------------------------------- #
# File backend: durability and atomicity
# --------------------------------------------------------------------------- #
def test_file_backend_survives_a_new_instance(tmpdir_path):
    """A new object stands in for a new process — nothing in memory survives."""
    run(FileCheckpointer(tmpdir_path).save(RunCheckpoint(run_id="r", step="a")))
    assert run(FileCheckpointer(tmpdir_path).latest("r")).step == "a"


def test_file_backend_escapes_path_traversal(tmpdir_path):
    """Run ids are caller-supplied, so `../` must not decide the write location."""
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(run_id="../../etc/pwned", step="a")))
    written = os.listdir(tmpdir_path)
    assert written and all(".." not in name for name in written)
    assert not (Path(tmpdir_path).parent.parent / "etc").exists()


def test_file_backend_handles_awkward_run_ids(tmpdir_path):
    cp = FileCheckpointer(tmpdir_path)
    for run_id in ("a/b:c", "  spaced  ", "emoji-🙂", "..."):
        run(cp.save(RunCheckpoint(run_id=run_id, step="x")))
        assert run(cp.latest(run_id)) is not None


def test_an_empty_run_id_is_still_writable(tmpdir_path):
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(run_id="", step="x")))
    assert run(cp.latest("")).step == "x"


def test_file_backend_leaves_no_temp_files(tmpdir_path):
    """Atomic write must clean up after itself."""
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(run_id="r", step="a")))
    assert not [f for f in os.listdir(tmpdir_path) if f.startswith(".tmp-")]


def test_a_corrupt_checkpoint_raises_rather_than_looking_empty(tmpdir_path):
    """Silently reading a corrupt file as 'no checkpoints' would restart the
    work from scratch and replay side effects."""
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(run_id="r", step="a")))
    path = Path(tmpdir_path) / "r.json"
    path.write_text("{ this is not json")
    with pytest.raises(json.JSONDecodeError):
        run(cp.latest("r"))


def test_file_backend_delete(tmpdir_path):
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(run_id="r", step="a")))
    assert run(cp.delete("r")) is True
    assert run(cp.latest("r")) is None


def test_file_backend_list_runs(tmpdir_path):
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(run_id="one", step="a")))
    run(cp.save(RunCheckpoint(run_id="two", step="a")))
    assert run(cp.list_runs()) == ["one", "two"]


# --------------------------------------------------------------------------- #
# Redis / Cosmos via injected clients
# --------------------------------------------------------------------------- #
class _FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def delete(self, key):
        return 1 if self.store.pop(key, None) is not None else 0

    async def keys(self, pattern):
        prefix = pattern.rstrip("*")
        return [k for k in self.store if k.startswith(prefix)]


def test_redis_backend_round_trips_through_an_injected_client():
    cp = RedisCheckpointer(client=_FakeRedis(), prefix="ck")
    run(cp.save(RunCheckpoint(run_id="r", step="a", state={"n": 1})))
    got = run(RedisCheckpointer(client=cp._client, prefix="ck").latest("r"))
    assert got is not None and got.step == "a" and got.state == {"n": 1}


def test_redis_requires_a_url_or_client():
    with pytest.raises(RuntimeError, match="no url and no injected client"):
        run(RedisCheckpointer(url=None, client=None).latest("r"))


class _FakeCosmosContainer:
    def __init__(self):
        self.items: dict[str, dict] = {}

    async def read_item(self, item, partition_key):
        if item not in self.items:
            raise type("ResourceNotFound", (Exception,), {})("not found")
        return self.items[item]

    async def upsert_item(self, doc):
        self.items[doc["id"]] = doc

    async def delete_item(self, item, partition_key):
        self.items.pop(item, None)


class _FakeCosmosClient:
    def __init__(self):
        self.container = _FakeCosmosContainer()

    def get_database_client(self, db):
        return self

    def get_container_client(self, c):
        return self.container


def test_cosmos_backend_round_trips_through_an_injected_client():
    client = _FakeCosmosClient()
    cp = CosmosCheckpointer(client=client)
    run(cp.save(RunCheckpoint(run_id="r", step="a")))
    assert run(cp.latest("r")).step == "a"
    assert run(cp.delete("r")) is True
    assert run(cp.latest("r")) is None


def test_cosmos_requires_an_endpoint_or_client():
    with pytest.raises(RuntimeError, match="no endpoint and no injected client"):
        run(CosmosCheckpointer(endpoint=None, client=None).latest("r"))


# --------------------------------------------------------------------------- #
# Factory: loud, queryable degradation
# --------------------------------------------------------------------------- #
def test_factory_builds_the_requested_backends():
    assert build_checkpointer("memory").name == "memory"
    assert build_checkpointer("file", directory=tempfile.mkdtemp()).name == "file"
    assert build_checkpointer("redis", client=object()).name == "redis"
    assert build_checkpointer("cosmos", client=object()).name == "cosmos"


def test_unconfigured_redis_degrades_without_raising():
    """A missing Redis must not take down a run that could otherwise finish."""
    cp = build_checkpointer("redis")
    assert cp.name == "memory"
    assert run(cp.save(RunCheckpoint(run_id="r", step="a"))).sequence == 1


def test_degradation_is_queryable_not_just_logged(caplog):
    """The chosen behaviour is warn-and-degrade. A warning alone is missable, so
    the object itself must report the degradation."""
    cp = build_checkpointer("cosmos", endpoint=None)
    assert isinstance(cp, DegradedCheckpointer)
    assert cp.degraded_from == "cosmos"
    assert cp.capability().durable is False
    assert "DEGRADED" in cp.capability().detail
    assert "cosmos" in cp.capability().detail


def test_degradation_emits_a_warning(caplog):
    import logging

    with caplog.at_level(logging.WARNING):
        build_checkpointer("redis")
    assert any("degrading to in-memory" in r.message for r in caplog.records)


def test_degraded_checkpointer_still_delegates_every_method():
    cp = build_checkpointer("redis")
    assert run(cp.save(RunCheckpoint(run_id="r", step="a"))) is not None
    assert run(cp.latest("r")).step == "a"
    assert run(cp.history("r")) and run(cp.list_runs()) == ["r"]
    assert run(cp.delete("r")) is True


def test_unknown_backend_degrades_rather_than_raising():
    cp = build_checkpointer("dynamodb")
    assert cp.degraded_from == "dynamodb"


def test_none_backend_is_explicit_and_reports_why():
    cp = build_checkpointer("none")
    assert cp.degraded_from == "none"
    assert "disabled" in cp.reason


# --------------------------------------------------------------------------- #
# Resume planning: the core property
# --------------------------------------------------------------------------- #
def test_fresh_run_has_nothing_to_skip():
    plan = plan_resume(None, ["a", "b"])
    assert plan.resuming is False
    assert plan.is_fresh is True
    assert plan.pending == ["a", "b"]
    assert plan.skip == []


def test_completed_steps_are_skipped():
    prev = RunCheckpoint(run_id="r", step_outcomes={
        "a": StepOutcome.COMPLETED, "b": StepOutcome.COMPLETED,
    })
    plan = plan_resume(prev, ["a", "b", "c"])
    assert plan.skip == ["a", "b"]
    assert plan.pending == ["c"]


def test_a_step_missing_from_the_ledger_is_pending():
    """Absent means never started — unlike PENDING, which means started."""
    prev = RunCheckpoint(run_id="r", step_outcomes={"a": StepOutcome.COMPLETED})
    plan = plan_resume(prev, ["a", "b"])
    assert plan.pending == ["b"]
    assert plan.warnings == []


def test_a_pending_step_warns_about_unknown_side_effects():
    prev = RunCheckpoint(run_id="r", step_outcomes={"a": StepOutcome.PENDING})
    plan = plan_resume(prev, ["a"])
    assert plan.resume_at == "a"
    assert plan.pending == ["a"]
    assert any("in flight" in w and "unknown" in w for w in plan.warnings)


def test_a_failed_step_is_retried_with_a_warning():
    prev = RunCheckpoint(run_id="r", step_outcomes={"a": StepOutcome.FAILED})
    plan = plan_resume(prev, ["a"])
    assert plan.pending == ["a"]
    assert any("failed previously" in w for w in plan.warnings)


def test_a_skipped_step_is_run_on_resume():
    prev = RunCheckpoint(run_id="r", step_outcomes={"a": StepOutcome.SKIPPED})
    plan = plan_resume(prev, ["a"])
    assert plan.pending == ["a"]
    assert plan.warnings == []


def test_resume_plan_carries_state_and_sequence():
    prev = RunCheckpoint(run_id="r", sequence=7, state={"k": "v"},
                         step_outcomes={"a": StepOutcome.COMPLETED})
    plan = plan_resume(prev, ["a", "b"])
    assert plan.state == {"k": "v"}
    assert plan.from_sequence == 7


def test_resume_plan_explains_itself():
    prev = RunCheckpoint(run_id="r", sequence=3, step_outcomes={"a": StepOutcome.PENDING})
    text = plan_resume(prev, ["a"]).explain()
    assert "from checkpoint #3" in text
    assert "in flight" in text


def test_fresh_plan_explains_itself():
    assert "fresh run" in plan_resume(None, ["a"]).explain()


def test_resume_plan_is_json_safe():
    json.dumps(plan_resume(None, ["a"]).to_dict())


def test_resume_plan_handles_an_empty_step_list():
    plan = plan_resume(RunCheckpoint(run_id="r"), [])
    assert plan.pending == [] and plan.skip == []


def test_a_step_recorded_completed_but_absent_from_steps_is_ignored():
    """The step list can change between deploys; a stale ledger entry must not
    crash resume."""
    prev = RunCheckpoint(run_id="r", step_outcomes={"gone": StepOutcome.COMPLETED})
    plan = plan_resume(prev, ["a"])
    assert plan.pending == ["a"]


# --------------------------------------------------------------------------- #
# CheckpointedRun: the write discipline
# --------------------------------------------------------------------------- #
def test_execute_runs_every_step_and_checkpoints():
    cp = InMemoryCheckpointer()
    ran = []

    async def h(state):
        ran.append(1)
        return {"done": len(ran)}

    r = CheckpointedRun(cp, "r", ["a", "b", "c"])
    plan = run(r.execute({"a": h, "b": h, "c": h}))
    assert ran == [1, 1, 1]
    assert r.outcomes == {"a": StepOutcome.COMPLETED, "b": StepOutcome.COMPLETED,
                          "c": StepOutcome.COMPLETED}
    assert run(cp.latest("r")).status is CheckpointStatus.COMPLETED
    assert plan.state.get("done") == 3


def test_execute_skips_completed_steps_on_a_second_call():
    """The resume property, at the API level."""
    cp = InMemoryCheckpointer()
    ran = []

    async def h(state):
        ran.append(1)
        return None

    r1 = CheckpointedRun(cp, "r", ["a", "b"])
    run(r1.execute({"a": h, "b": h}))
    assert len(ran) == 2

    r2 = CheckpointedRun(cp, "r", ["a", "b"])
    plan = run(r2.execute({"a": h, "b": h}))
    assert len(ran) == 2, "no step may re-run after completion"
    assert plan.skip == ["a", "b"]
    assert plan.pending == []


def test_state_is_carried_across_resumes():
    cp = InMemoryCheckpointer()

    async def first(state):
        return {"value": 41}

    async def crash(state):
        raise RuntimeError("boom")

    async def finish(state):
        return {"value": state["value"] + 1}

    r1 = CheckpointedRun(cp, "r", ["calc", "middle", "finish"])
    with pytest.raises(RuntimeError):
        run(r1.execute({"calc": first, "middle": crash, "finish": finish}))

    r2 = CheckpointedRun(cp, "r", ["calc", "middle", "finish"])
    plan = run(r2.execute({"calc": first, "middle": first, "finish": finish}))
    assert plan.state["value"] == 42, "state from before the crash must survive"


def test_the_pre_write_records_pending_so_a_hard_kill_is_detectable():
    """Without the pre-write, a crash mid-step looks identical to a step that
    never started, and resume silently replays it."""
    cp = InMemoryCheckpointer()

    async def crash(state):
        raise RuntimeError("killed")

    r = CheckpointedRun(cp, "r", ["charge"])
    with pytest.raises(RuntimeError):
        run(r.execute({"charge": crash}))

    latest = run(cp.latest("r"))
    # The FAILED write happened, but the sequence must show PENDING was written
    # first — proving the pre-write exists.
    hist = run(cp.history("r"))
    assert hist[0].step_outcomes["charge"] is StepOutcome.PENDING
    assert latest.step_outcomes["charge"] is StepOutcome.FAILED


def test_a_hard_kill_between_the_two_writes_leaves_pending(tmpdir_path):
    """Simulate: pre-write lands, then the process is killed with no chance to
    write COMPLETED."""
    cp = FileCheckpointer(tmpdir_path)
    run(cp.save(RunCheckpoint(
        run_id="r", step="charge",
        step_outcomes={"charge": StepOutcome.PENDING},
        state={"amount": 500},
    )))
    plan = run(CheckpointedRun(FileCheckpointer(tmpdir_path), "r", ["charge"]).plan())
    assert plan.resume_at == "charge"
    assert any("unknown" in w for w in plan.warnings)


def test_a_missing_handler_is_recorded_as_skipped_not_failed():
    cp = InMemoryCheckpointer()
    r = CheckpointedRun(cp, "r", ["a", "b"])

    async def h(state):
        return None

    run(r.execute({"a": h}))
    assert r.outcomes["a"] is StepOutcome.COMPLETED
    assert r.outcomes["b"] is StepOutcome.SKIPPED
    # And a skipped step is NOT treated as done on resume.
    plan = run(CheckpointedRun(cp, "r", ["a", "b"]).plan())
    assert plan.pending == ["b"]


def test_a_synchronous_handler_is_accepted():
    cp = InMemoryCheckpointer()
    r = CheckpointedRun(cp, "r", ["a"])
    run(r.execute({"a": lambda state: {"sync": True}}))
    assert r.outcomes["a"] is StepOutcome.COMPLETED


def test_a_non_dict_return_is_ignored_rather_than_corrupting_state():
    cp = InMemoryCheckpointer()
    r = CheckpointedRun(cp, "r", ["a"])
    plan = run(r.execute({"a": lambda state: "just a string"}, state={"keep": 1}))
    assert plan.state == {"keep": 1}


def test_a_failing_step_marks_the_run_failed():
    cp = InMemoryCheckpointer()

    async def boom(state):
        raise ValueError("nope")

    r = CheckpointedRun(cp, "r", ["a"])
    with pytest.raises(ValueError):
        run(r.execute({"a": boom}))
    assert run(cp.latest("r")).status is CheckpointStatus.FAILED


def test_mark_completed_records_every_step():
    cp = InMemoryCheckpointer()
    r = CheckpointedRun(cp, "r", ["a", "b"])
    saved = run(r.mark_completed({"final": 1}))
    assert saved.status is CheckpointStatus.COMPLETED
    assert run(cp.latest("r")).completed_steps() == {"a", "b"}


def test_execute_exposes_the_resume_decision_to_the_caller():
    """The caller must see what was skipped, not have to infer it."""
    cp = InMemoryCheckpointer()

    async def h(s):
        return None

    run(CheckpointedRun(cp, "r", ["a"]).execute({"a": h}))
    plan = run(CheckpointedRun(cp, "r", ["a"]).execute({"a": h}))
    assert isinstance(plan, ResumePlan)
    assert plan.resuming is True
    assert plan.skip == ["a"]


def test_run_metadata_is_preserved_across_checkpoints():
    cp = InMemoryCheckpointer()
    r = CheckpointedRun(cp, "r", ["a"], metadata={"owner": "test"})
    run(r.execute({"a": lambda s: None}))
    assert run(cp.latest("r")).metadata["owner"] == "test"


# --------------------------------------------------------------------------- #
# End-to-end: survival across a genuine filesystem round trip
# --------------------------------------------------------------------------- #
def test_full_crash_and_resume_cycle_on_disk(tmpdir_path):
    """The scenario checkpointing exists for: a multi-step run dies, a new
    process picks it up, and no completed step executes twice."""
    executed: list[str] = []

    async def fetch(state):
        executed.append("fetch")
        return {"doc": "inv-42"}

    async def extract(state):
        executed.append("extract")
        return {"total": 1440.0}

    async def load(state):
        executed.append("load")
        raise RuntimeError("warehouse unavailable")

    # Process 1
    cp1 = FileCheckpointer(tmpdir_path)
    r1 = CheckpointedRun(cp1, "nightly", ["fetch", "extract", "load"])
    with pytest.raises(RuntimeError):
        run(r1.execute({"fetch": fetch, "extract": extract, "load": load}))
    assert executed == ["fetch", "extract", "load"]

    # Process 2 — a fresh object graph, reading only what is on disk
    executed.clear()

    async def load_ok(state):
        executed.append("load")
        assert state["total"] == 1440.0, "state must be restored, not recomputed"
        assert state["doc"] == "inv-42"
        return {"loaded": True}

    cp2 = FileCheckpointer(tmpdir_path)
    r2 = CheckpointedRun(cp2, "nightly", ["fetch", "extract", "load"])
    plan = run(r2.execute({"fetch": fetch, "extract": extract, "load": load_ok}))

    assert executed == ["load"], f"completed steps re-ran: {executed}"
    assert plan.skip == ["fetch", "extract"]
    assert plan.state["loaded"] is True
    assert run(cp2.latest("nightly")).status is CheckpointStatus.COMPLETED


def test_base_class_requires_the_storage_primitives():
    """A subclass that forgets a primitive must fail clearly."""
    class Incomplete(BaseCheckpointer):
        name = "incomplete"

    with pytest.raises(NotImplementedError):
        run(Incomplete().latest("r"))
