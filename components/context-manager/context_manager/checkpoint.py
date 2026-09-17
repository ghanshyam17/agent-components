"""Checkpointing: durable, resumable run state.

A checkpointer exists to answer one question after a crash:

    **"What has already happened, and is it safe to do it again?"**

That single question is why this module is more than a key-value store. Three
properties matter, and any one of them missing makes a "checkpointer" useless
for the crash it was installed to survive:

1. **Reads, not just writes.** The previous implementation in this repo stored
   state and never loaded it. A checkpointer nothing reads cannot resume
   anything — it is an in-RAM log. Every backend here is tested by *resuming*.
2. **Durability that survives the process.** `memory` cannot survive the crash
   it exists to protect against. It is a fine test double and a dishonest
   default, so :meth:`CheckpointCapability.durable` reports which is which and
   the factory says loudly when it has degraded.
3. **Honest step outcomes.** The hard part of resume is not restoring state, it
   is knowing whether a step that was *in flight* when the process died has
   already had its side effect. That is unanswerable in general, so it is
   recorded as :attr:`StepOutcome.UNKNOWN` rather than guessed — and callers can
   refuse to replay it.

## The write discipline

Two writes per step, and the second is what makes resume safe:

    save(step=1, "fetch") -> PENDING     # about to execute
    ...execute...                        # crash anywhere in here
    save(step=1, "fetch") -> COMPLETED   # finished

A crash between them leaves the last record as ``PENDING``, which on resume is
read as ``UNKNOWN``: the step started and its outcome is not knowable from here.
Without the pre-write, a crash mid-execution would look identical to a step that
never started, and resume would silently replay a step that may have moved money.
"""
from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol, runtime_checkable

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class CheckpointStatus(str, Enum):
    """Where the run as a whole stands."""

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class StepOutcome(str, Enum):
    """What is known about one step.

    The distinction between ``PENDING`` and ``SKIPPED`` is the whole point: one
    means "we started this and don't know how it ended", the other means "we
    definitely did not run this".
    """

    PENDING = "pending"      # recorded as started; completion never written
    COMPLETED = "completed"  # ran to completion; safe to skip on resume
    FAILED = "failed"        # raised; retry may re-run a partial side effect
    SKIPPED = "skipped"      # deliberately not run, or never reached; safe to run
    UNKNOWN = "unknown"      # PENDING seen at resume time — side effects unknown


class CheckpointCapability(BaseModel):
    """What a backend can actually promise.

    Reported rather than assumed, because "we use Redis" and "our checkpoints
    survive a pod restart" are different claims and only one is verifiable.
    """

    backend: str
    #: Survives the process that wrote it.
    durable: bool
    #: Visible to other replicas / processes.
    shared: bool
    #: A save is all-or-nothing (no torn reads).
    atomic: bool
    #: Retains more than the latest checkpoint.
    history: bool
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()


class RunCheckpoint(BaseModel):
    """One point-in-time record of a run in progress."""

    run_id: str
    #: Monotonic sequence within the run; total order across restarts.
    sequence: int = 0
    #: The step this checkpoint describes (None = a run-level checkpoint).
    step: str | None = None
    status: CheckpointStatus = CheckpointStatus.RUNNING
    #: The step-name -> outcome map: the resume ledger.
    step_outcomes: dict[str, StepOutcome] = Field(default_factory=dict)
    #: Whatever the run needs to continue. Must be JSON-serialisable.
    state: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def resume_point(self) -> str | None:
        """The step that was in flight, if any."""
        for name, outcome in self.step_outcomes.items():
            if outcome in (StepOutcome.PENDING, StepOutcome.UNKNOWN):
                return name
        return None

    def completed_steps(self) -> set[str]:
        return {n for n, o in self.step_outcomes.items() if o is StepOutcome.COMPLETED}

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump()
        d["status"] = self.status.value
        d["step_outcomes"] = {k: v.value for k, v in self.step_outcomes.items()}
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunCheckpoint":
        data = dict(data)
        data["step_outcomes"] = {
            k: StepOutcome(v) for k, v in (data.get("step_outcomes") or {}).items()
        }
        return cls(**data)


class ResumePlan(BaseModel):
    """What a resumed run should do, and what it must be careful about.

    Returned instead of a bare state dict so the caller cannot resume without
    seeing the warnings — the whole failure mode this prevents is a silent
    replay of a step with side effects.
    """

    run_id: str
    #: False when there was nothing to resume from.
    resuming: bool = False
    #: Steps already COMPLETED; the caller must skip these.
    skip: list[str] = Field(default_factory=list)
    #: Steps to execute, in order.
    pending: list[str] = Field(default_factory=list)
    #: The step in flight at crash time, if determinate.
    resume_at: str | None = None
    #: Restored state, or empty when starting fresh.
    state: dict[str, Any] = Field(default_factory=dict)
    #: Steps whose side effects are unknown or partial — do not blind-replay.
    warnings: list[str] = Field(default_factory=list)
    from_sequence: int | None = None

    @property
    def is_fresh(self) -> bool:
        return not self.resuming

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()

    def explain(self) -> str:
        if not self.resuming:
            return f"ResumePlan: fresh run {self.run_id} ({len(self.pending)} steps)"
        lines = [
            f"ResumePlan: {self.run_id} from checkpoint #{self.from_sequence}",
            f"  skip {len(self.skip)} completed, run {len(self.pending)} pending",
        ]
        if self.resume_at:
            lines.append(f"  resume at: {self.resume_at}")
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class Checkpointer(Protocol):
    """Anything that can durably record and return run state."""

    name: str

    async def save(self, checkpoint: RunCheckpoint) -> RunCheckpoint: ...

    async def latest(self, run_id: str) -> RunCheckpoint | None: ...

    async def history(self, run_id: str) -> list[RunCheckpoint]: ...

    async def delete(self, run_id: str) -> bool: ...

    async def list_runs(self) -> list[str]: ...

    def capability(self) -> CheckpointCapability: ...


class BaseCheckpointer:
    """Shared bookkeeping: sequence numbering and bounded history.

    Subclasses implement only storage primitives (``_read_all``/``_write_all``/
    ``_remove``), so the sequencing and trimming rules cannot drift between
    backends — the same reason the repo has a chassis at all.
    """

    name = "base"

    def __init__(self, max_history: int = 50) -> None:
        self.max_history = max(1, max_history)

    # -- storage primitives (subclass responsibility) ---------------------- #
    async def _read_all(self, run_id: str) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def _write_all(self, run_id: str, items: list[dict[str, Any]]) -> None:
        raise NotImplementedError

    async def _remove(self, run_id: str) -> bool:
        raise NotImplementedError

    # -- shared logic ------------------------------------------------------ #
    async def delete(self, run_id: str) -> bool:
        """Remove a run's checkpoints. Delegates to ``_remove``."""
        return await self._remove(run_id)

    async def save(self, checkpoint: RunCheckpoint) -> RunCheckpoint:
        """Append a checkpoint, assigning the sequence number.

        The sequence is assigned here rather than by the caller so two callers
        cannot both believe they wrote #3.
        """
        items = await self._read_all(checkpoint.run_id)
        last = max((int(i.get("sequence", 0)) for i in items), default=0)
        checkpoint.sequence = last + 1
        checkpoint.created_at = time.time()
        items.append(checkpoint.to_dict())
        # Bounded history: a checkpointer must not become the unbounded log.
        if len(items) > self.max_history:
            items = items[-self.max_history :]
        await self._write_all(checkpoint.run_id, items)
        return checkpoint

    async def latest(self, run_id: str) -> RunCheckpoint | None:
        items = await self._read_all(run_id)
        if not items:
            return None
        return RunCheckpoint.from_dict(max(items, key=lambda i: int(i.get("sequence", 0))))

    async def history(self, run_id: str) -> list[RunCheckpoint]:
        items = await self._read_all(run_id)
        return [
            RunCheckpoint.from_dict(i)
            for i in sorted(items, key=lambda i: int(i.get("sequence", 0)))
        ]

    async def list_runs(self) -> list[str]:
        """Runs this backend knows about. Optional; empty when unsupported."""
        return []

    def capability(self) -> CheckpointCapability:
        return CheckpointCapability(
            backend=self.name, durable=False, shared=False, atomic=False, history=True
        )


# --------------------------------------------------------------------------- #
# In-memory
# --------------------------------------------------------------------------- #
class InMemoryCheckpointer(BaseCheckpointer):
    """Process-local. Deterministic, zero-dependency, and **not durable**.

    Correct for tests and for a single run you do not intend to survive. It is
    the honest fallback, and :meth:`capability` says so.
    """

    name = "memory"

    def __init__(self, max_history: int = 50) -> None:
        super().__init__(max_history)
        self._runs: dict[str, list[dict[str, Any]]] = {}

    async def _read_all(self, run_id: str) -> list[dict[str, Any]]:
        # Deep copy: a caller mutating restored state must not corrupt the store.
        return json.loads(json.dumps(self._runs.get(run_id, [])))

    async def _write_all(self, run_id: str, items: list[dict[str, Any]]) -> None:
        self._runs[run_id] = items

    async def _remove(self, run_id: str) -> bool:
        return self._runs.pop(run_id, None) is not None

    async def list_runs(self) -> list[str]:
        return sorted(self._runs)

    def capability(self) -> CheckpointCapability:
        return CheckpointCapability(
            backend=self.name,
            durable=False,
            shared=False,
            atomic=True,
            history=True,
            detail="process-local; does not survive a restart — use for tests only",
        )


# --------------------------------------------------------------------------- #
# File
# --------------------------------------------------------------------------- #
def _safe_filename(run_id: str) -> str:
    """Map a run id to a filesystem-safe name.

    Run ids are caller-supplied and often contain ``/``, ``:`` or ``..``. This
    is a path-traversal guard, not cosmetics: an unescaped ``../../etc/passwd``
    would otherwise decide where the checkpoint is written.
    """
    cleaned = _UNSAFE.sub("_", run_id).strip("._") or "run"
    return cleaned[:120] + ".json"


class FileCheckpointer(BaseCheckpointer):
    """One JSON document per run on the local filesystem.

    **The honest default for durability**: it survives a process crash with zero
    infrastructure, which is the actual first requirement for resumability.
    Atomic writes (temp file + ``os.replace``) mean a crash mid-write leaves the
    previous checkpoint intact rather than a truncated file.
    """

    name = "file"

    def __init__(self, directory: str | os.PathLike[str] = ".checkpoints",
                 max_history: int = 50) -> None:
        super().__init__(max_history)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, run_id: str) -> Path:
        return self.directory / _safe_filename(run_id)

    async def _read_all(self, run_id: str) -> list[dict[str, Any]]:
        path = self._path(run_id)
        if not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            # A corrupt checkpoint must not be silently treated as "no
            # checkpoints" — that would resume from scratch and replay work.
            logger.error("checkpoint file %s unreadable: %s", path, exc)
            raise
        items = payload.get("checkpoints", [])
        return items if isinstance(items, list) else []

    async def _write_all(self, run_id: str, items: list[dict[str, Any]]) -> None:
        path = self._path(run_id)
        payload = {"schema_version": SCHEMA_VERSION, "run_id": run_id, "checkpoints": items}
        # Atomic: write a temp file in the same directory, then replace. A crash
        # mid-write therefore leaves the previous good file untouched.
        fd, tmp = tempfile.mkstemp(dir=str(self.directory), prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def _remove(self, run_id: str) -> bool:
        path = self._path(run_id)
        if path.is_file():
            path.unlink()
            return True
        return False

    async def list_runs(self) -> list[str]:
        out = []
        for p in sorted(self.directory.glob("*.json")):
            try:
                out.append(json.loads(p.read_text(encoding="utf-8")).get("run_id", p.stem))
            except (json.JSONDecodeError, OSError):
                continue
        return out

    def capability(self) -> CheckpointCapability:
        return CheckpointCapability(
            backend=self.name,
            durable=True,
            shared=False,
            atomic=True,
            history=True,
            detail=f"local directory {self.directory}; durable across restarts, "
                   "but not visible to other hosts",
        )


# --------------------------------------------------------------------------- #
# Redis
# --------------------------------------------------------------------------- #
class RedisCheckpointer(BaseCheckpointer):
    """Redis-backed, shared across replicas.

    Accepts an injected async client so the logic is testable without a server —
    the same seam ``AzureDocumentIntelligenceEngine`` uses for its fetcher.
    """

    name = "redis"

    def __init__(self, url: str | None = None, *, client: Any = None,
                 prefix: str = "ckpt", ttl_seconds: int | None = None,
                 max_history: int = 50) -> None:
        super().__init__(max_history)
        self.url = url
        self.prefix = prefix
        self.ttl_seconds = ttl_seconds
        self._client = client

    @property
    def configured(self) -> bool:
        return self._client is not None or bool(self.url)

    def _key(self, run_id: str) -> str:
        return f"{self.prefix}:{run_id}"

    async def _conn(self) -> Any:
        if self._client is not None:
            return self._client
        if not self.url:
            raise RuntimeError(
                "RedisCheckpointer has no url and no injected client; "
                "construct with url='redis://…' or client=…"
            )
        try:
            import redis.asyncio as redis  # type: ignore
        except ImportError as exc:  # pragma: no cover - env dependent
            raise RuntimeError(
                "redis is not installed; `pip install redis` or use another backend"
            ) from exc
        self._client = redis.from_url(self.url, decode_responses=True)
        return self._client

    async def _read_all(self, run_id: str) -> list[dict[str, Any]]:
        conn = await self._conn()
        raw = await conn.get(self._key(run_id))
        if not raw:
            return []
        payload = json.loads(raw)
        return payload.get("checkpoints", [])

    async def _write_all(self, run_id: str, items: list[dict[str, Any]]) -> None:
        conn = await self._conn()
        payload = json.dumps(
            {"schema_version": SCHEMA_VERSION, "run_id": run_id, "checkpoints": items}
        )
        if self.ttl_seconds:
            await conn.set(self._key(run_id), payload, ex=self.ttl_seconds)
        else:
            await conn.set(self._key(run_id), payload)

    async def _remove(self, run_id: str) -> bool:
        conn = await self._conn()
        return bool(await conn.delete(self._key(run_id)))

    async def list_runs(self) -> list[str]:
        conn = await self._conn()
        keys = await conn.keys(f"{self.prefix}:*")
        return sorted(k.split(":", 1)[1] for k in keys)

    def capability(self) -> CheckpointCapability:
        return CheckpointCapability(
            backend=self.name,
            durable=True,
            shared=True,
            atomic=True,
            history=True,
            detail=(
                f"redis at {self.url or 'injected client'}"
                + (f", ttl {self.ttl_seconds}s" if self.ttl_seconds else "")
            ),
        )


# --------------------------------------------------------------------------- #
# Cosmos DB
# --------------------------------------------------------------------------- #
class CosmosCheckpointer(BaseCheckpointer):
    """Azure Cosmos DB-backed. One document per run, partitioned by ``run_id``.

    Injected client again, for testability without an Azure account.
    """

    name = "cosmos"

    def __init__(self, endpoint: str | None = None, *, database: str = "agents",
                 container: str = "checkpoints", client: Any = None,
                 credential: Any = None, max_history: int = 50) -> None:
        super().__init__(max_history)
        self.endpoint = endpoint
        self.database = database
        self.container = container
        self._client = client
        self._credential = credential
        self._container_client: Any = None

    @property
    def configured(self) -> bool:
        return self._client is not None or bool(self.endpoint)

    async def _container(self) -> Any:
        if self._container_client is not None:
            return self._container_client
        if self._client is None:
            if not self.endpoint:
                raise RuntimeError(
                    "CosmosCheckpointer has no endpoint and no injected client"
                )
            try:
                from azure.cosmos.aio import CosmosClient  # type: ignore
            except ImportError as exc:  # pragma: no cover - env dependent
                raise RuntimeError(
                    "azure-cosmos is not installed; "
                    "`pip install azure-cosmos` or use another backend"
                ) from exc
            self._client = CosmosClient(self.endpoint, self._credential)
        self._container_client = self._client.get_database_client(
            self.database
        ).get_container_client(self.container)
        return self._container_client

    async def _read_all(self, run_id: str) -> list[dict[str, Any]]:
        container = await self._container()
        try:
            item = await container.read_item(item=run_id, partition_key=run_id)
        except Exception as exc:  # noqa: BLE001 - not-found surfaces vendor-specifically
            if "NotFound" in type(exc).__name__ or "404" in str(exc):
                return []
            raise
        return item.get("checkpoints", [])

    async def _write_all(self, run_id: str, items: list[dict[str, Any]]) -> None:
        container = await self._container()
        await container.upsert_item(
            {
                "id": run_id,
                "run_id": run_id,
                "schema_version": SCHEMA_VERSION,
                "checkpoints": items,
            }
        )

    async def _remove(self, run_id: str) -> bool:
        container = await self._container()
        try:
            await container.delete_item(item=run_id, partition_key=run_id)
            return True
        except Exception as exc:  # noqa: BLE001
            if "NotFound" in type(exc).__name__ or "404" in str(exc):
                return False
            raise

    def capability(self) -> CheckpointCapability:
        return CheckpointCapability(
            backend=self.name,
            durable=True,
            shared=True,
            atomic=True,
            history=True,
            detail=f"cosmos {self.database}/{self.container} at {self.endpoint or 'injected client'}",
        )


# --------------------------------------------------------------------------- #
# Factory — with loud degradation
# --------------------------------------------------------------------------- #
class DegradedCheckpointer:
    """Wraps a substitute so a degradation is **queryable**, not just logged.

    The repo's rule is that an absent capability must not be representable as a
    silent success (the empty ``analyzeResult`` bug). A warning in a log is easy
    to miss; ``degraded_from`` on the object itself is not, so a caller that
    needs durability can assert it:

        cp = build_checkpointer("redis", url=None)
        if not cp.capability().durable:
            raise RuntimeError("this run requires durable checkpointing")
    """

    def __init__(self, inner: Checkpointer, degraded_from: str, reason: str) -> None:
        self._inner = inner
        self.degraded_from = degraded_from
        self.reason = reason
        self.name = inner.name

    def __getattr__(self, item: str) -> Any:
        return getattr(self._inner, item)

    async def save(self, checkpoint: RunCheckpoint) -> RunCheckpoint:
        return await self._inner.save(checkpoint)

    async def latest(self, run_id: str) -> RunCheckpoint | None:
        return await self._inner.latest(run_id)

    async def history(self, run_id: str) -> list[RunCheckpoint]:
        return await self._inner.history(run_id)

    async def delete(self, run_id: str) -> bool:
        return await self._inner.delete(run_id)

    async def list_runs(self) -> list[str]:
        return await self._inner.list_runs()

    def capability(self) -> CheckpointCapability:
        cap = self._inner.capability()
        cap.detail = (
            f"⚠ DEGRADED from {self.degraded_from!r} to {cap.backend!r}: {self.reason}"
        )
        return cap

    def to_dict(self) -> dict[str, Any]:
        return {
            "degraded_from": self.degraded_from,
            "reason": self.reason,
            "capability": self.capability().to_dict(),
        }


def build_checkpointer(kind: str = "memory", **kwargs: Any) -> Checkpointer:
    """Construct a checkpointer, degrading **loudly** on an unavailable backend.

    Degrading to memory is the chosen behaviour over raising, because a missing
    Redis should not take down a run that could otherwise finish. But it is
    never silent: a warning is logged *and* the returned object reports
    ``degraded_from``, so "we asked for durable and got volatile" cannot pass
    unnoticed.
    """
    kind = (kind or "memory").lower()
    if kind == "memory":
        return InMemoryCheckpointer(**kwargs)
    if kind == "file":
        return FileCheckpointer(**kwargs)
    if kind == "redis":
        url = kwargs.get("url")
        client = kwargs.get("client")
        if not url and client is None:
            return _degrade(kind, InMemoryCheckpointer(), "no url and no injected client")
        return RedisCheckpointer(**kwargs)
    if kind == "cosmos":
        endpoint = kwargs.get("endpoint")
        client = kwargs.get("client")
        if not endpoint and client is None:
            return _degrade(kind, InMemoryCheckpointer(), "no endpoint and no injected client")
        return CosmosCheckpointer(**kwargs)
    if kind == "none":
        return _degrade("none", InMemoryCheckpointer(max_history=1), "checkpointing disabled")

    return _degrade(kind, InMemoryCheckpointer(), f"unknown backend {kind!r}")


def _degrade(kind: str, inner: Checkpointer, reason: str) -> Checkpointer:
    logger.warning(
        "checkpointer %r unavailable (%s) — degrading to in-memory, which does NOT "
        "survive a process restart; set run_id and a durable backend before relying "
        "on resume",
        kind,
        reason,
    )
    return DegradedCheckpointer(inner, degraded_from=kind, reason=reason)


# --------------------------------------------------------------------------- #
# Resume planning
# --------------------------------------------------------------------------- #
def plan_resume(
    previous: RunCheckpoint | None,
    steps: list[str],
    *,
    replay_unknown: bool = False,
) -> ResumePlan:
    """Decide what a resumed run should execute, given its last checkpoint.

    ``replay_unknown=False`` (the default) **refuses to silently re-run** a step
    that was in flight when the previous process died. Its side effects are
    genuinely unknown — it may have charged a card before the crash — so the
    step is reported in ``warnings`` and left in ``pending`` for the caller to
    decide about. Setting it requires saying so explicitly.
    """
    if previous is None:
        return ResumePlan(run_id="<new>", resuming=False, pending=list(steps))

    recorded = previous.step_outcomes
    skip: list[str] = []
    pending: list[str] = []
    warnings: list[str] = []
    resume_at: str | None = None

    for name in steps:
        outcome = recorded.get(name)
        if outcome is StepOutcome.COMPLETED:
            skip.append(name)
            continue
        if outcome in (StepOutcome.PENDING, StepOutcome.UNKNOWN):
            # In flight when the process died. We cannot know whether the side
            # effect landed.
            resume_at = name
            pending.append(name)
            warnings.append(
                f"step {name!r} was in flight when the last checkpoint was written; "
                "its side effects are unknown — verify before it runs again"
                + ("" if replay_unknown else " (it will run; set replay_unknown=True "
                                             "to acknowledge this explicitly)")
            )
            continue
        if outcome is StepOutcome.FAILED:
            pending.append(name)
            warnings.append(
                f"step {name!r} failed previously; it will be retried, and any partial "
                "side effect from that attempt is not undone"
            )
            continue
        # SKIPPED, or absent from the ledger entirely: never started.
        pending.append(name)

    return ResumePlan(
        run_id=previous.run_id,
        resuming=True,
        skip=skip,
        pending=pending,
        resume_at=resume_at,
        state=previous.state,
        warnings=warnings,
        from_sequence=previous.sequence,
    )


class CheckpointedRun:
    """A step-runner that checkpoints after every step, and can resume.

    The convenience wrapper: it owns the write discipline so a caller cannot
    forget the pre-write (which is what makes ``UNKNOWN`` detectable at all).

        run = CheckpointedRun(checkpointer, run_id, ["fetch", "extract", "load"])
        await run.execute({"fetch": fetch_fn, "extract": extract_fn, "load": load_fn})
        # ...process dies here...

        run2 = CheckpointedRun(checkpointer, run_id, ["fetch", "extract", "load"])
        print(await run2.resume())      # skips "fetch", reports what is unknown
    """

    def __init__(
        self,
        checkpointer: Checkpointer,
        run_id: str,
        steps: list[str],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.checkpointer = checkpointer
        self.run_id = run_id
        self.steps = list(steps)
        self.metadata = metadata or {}
        self.outcomes: dict[str, StepOutcome] = {}

    async def _save(self, state: dict[str, Any], status: CheckpointStatus,
                    step: str | None = None) -> RunCheckpoint:
        return await self.checkpointer.save(
            RunCheckpoint(
                run_id=self.run_id,
                step=step,
                status=status,
                step_outcomes=dict(self.outcomes),
                state=state,
                metadata=self.metadata,
            )
        )

    async def plan(self, *, replay_unknown: bool = False) -> ResumePlan:
        previous = await self.checkpointer.latest(self.run_id)
        plan = plan_resume(previous, self.steps, replay_unknown=replay_unknown)
        if previous is not None:
            plan.run_id = self.run_id
            self.outcomes = dict(previous.step_outcomes)
        return plan

    async def execute(
        self,
        handlers: dict[str, Callable[[dict[str, Any]], Any]],
        *,
        state: dict[str, Any] | None = None,
        replay_unknown: bool = False,
    ) -> ResumePlan:
        """Run the steps, checkpointing before and after each.

        Handlers receive the current state and may return a dict to merge into
        it. Returns the plan describing what was skipped and what ran, so the
        caller sees the resume decision rather than having to infer it.
        """
        plan = await self.plan(replay_unknown=replay_unknown)
        working = dict(state if state is not None else plan.state)

        for name in plan.pending:
            handler = handlers.get(name)
            if handler is None:
                self.outcomes[name] = StepOutcome.SKIPPED
                await self._save(working, CheckpointStatus.RUNNING, name)
                continue

            # Pre-write: records *intent*. Without this a crash mid-step is
            # indistinguishable from a step that never started.
            self.outcomes[name] = StepOutcome.PENDING
            await self._save(working, CheckpointStatus.RUNNING, name)

            try:
                result = handler(working)
                if hasattr(result, "__await__"):
                    result = await result
            except Exception:
                self.outcomes[name] = StepOutcome.FAILED
                await self._save(working, CheckpointStatus.FAILED, name)
                raise

            if isinstance(result, dict):
                working.update(result)
            self.outcomes[name] = StepOutcome.COMPLETED
            await self._save(working, CheckpointStatus.RUNNING, name)

        plan.state = working
        await self._save(working, CheckpointStatus.COMPLETED)
        return plan

    async def mark_completed(self, state: dict[str, Any] | None = None) -> RunCheckpoint:
        for name in self.steps:
            self.outcomes.setdefault(name, StepOutcome.COMPLETED)
        return await self._save(state or {}, CheckpointStatus.COMPLETED)
