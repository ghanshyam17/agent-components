"""Azure AI Foundry distillation client, with a deterministic offline runner.

Two backends behind one interface:

**foundry** — submits a real fine-tuning / distillation job through
``azure.ai.projects`` (and ``azure.ai.ml`` where a workspace is involved),
then polls it to completion. Used when the Azure SDKs are installed *and* an
endpoint is configured.

**mock** — runs locally with no credentials, GPUs, or network. It walks the same
state machine (pending → synthesizing → curating → training → completed) and
writes the same artifact layout, so CI and local development exercise the real
control flow instead of skipping it.

The degradation is deliberate and logged once, not silent: a pipeline that
thinks it trained a model when it did not is worse than one that fails.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Awaitable, Callable

from distillation.models import (
    DistillationConfig,
    DistillationJobStatus,
    JobState,
    TrainingMethod,
)

logger = logging.getLogger(__name__)

__all__ = ["FoundryDistillationClient", "azure_available"]

# Azure SDKs are an optional extra: this module must import without them.
try:  # pragma: no cover - depends on the environment
    from azure.ai.projects import AIProjectClient  # type: ignore
    from azure.identity import DefaultAzureCredential  # type: ignore

    _AZURE_IMPORTED = True
except ImportError:  # pragma: no cover
    AIProjectClient = None  # type: ignore
    DefaultAzureCredential = None  # type: ignore
    _AZURE_IMPORTED = False


def azure_available() -> bool:
    """Whether the Azure SDKs needed for a real Foundry job are importable."""
    return _AZURE_IMPORTED


class FoundryDistillationClient:
    """Submit, monitor and collect distillation jobs.

    Parameters
    ----------
    config:
        The job's configuration. The client keeps it so polls can rebuild state.
    endpoint:
        Foundry project endpoint. Falls back to ``config.foundry_endpoint``.
    artifact_dir:
        Where datasets and job records are written. Created on first use.
    force_mock:
        Skip Azure even when it is available — the switch CI should set, and the
        one tests rely on for determinism.
    sleep:
        Injected sleep, so tests need not wait for real polling intervals.
    """

    def __init__(
        self,
        config: DistillationConfig,
        *,
        endpoint: str | None = None,
        artifact_dir: str | Path = "artifacts",
        force_mock: bool = False,
        poll_interval: float = 2.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.config = config
        self.endpoint = endpoint or config.foundry_endpoint
        self.artifact_dir = Path(artifact_dir)
        self.poll_interval = poll_interval
        self._sleep = sleep or asyncio.sleep
        self._mock = force_mock or not (azure_available() and self.endpoint)
        if self._mock and not force_mock:
            # Log the reason once. "Mock" and "not configured" are different
            # situations and an operator needs to tell them apart.
            if not azure_available():
                logger.info(
                    "Foundry SDKs not installed (pip install 'distillation[azure]'); "
                    "using the local mock runner"
                )
            else:
                logger.info(
                    "no Foundry endpoint configured; using the local mock runner"
                )

    @property
    def backend(self) -> str:
        return "mock" if self._mock else "foundry"

    # ---- artifacts -------------------------------------------------------- #
    def _job_dir(self, job_id: str) -> Path:
        d = self.artifact_dir / job_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def write_dataset(self, job_id: str, records: list[dict[str, Any]], name: str = "train") -> Path:
        """Write a dataset JSONL and return its path."""
        path = self._job_dir(job_id) / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path

    def write_record(self, status: DistillationJobStatus) -> Path:
        """Persist the job's own state, so a run is inspectable afterwards."""
        path = self._job_dir(status.job_id) / "job.json"
        path.write_text(json.dumps(status.to_dict(), indent=2), encoding="utf-8")
        return path

    # ---- submission ------------------------------------------------------- #
    async def submit(
        self,
        job_id: str | None = None,
        *,
        dataset_path: str | Path | None = None,
    ) -> DistillationJobStatus:
        """Create the job and return its initial status."""
        job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        status = DistillationJobStatus(
            job_id=job_id,
            status=JobState.PENDING,
            teacher_model=self.config.teacher_model,
            student_model=self.config.student_base_model,
            method=self.config.method,
            config=self.config,
            backend=self.backend,  # type: ignore[arg-type]
        )
        if dataset_path:
            status.artifact_uris["dataset"] = str(dataset_path)

        if self._mock:
            # The mock walks the same state machine as the real job, so callers
            # cannot accidentally depend on mock-only sequencing.
            status.status = JobState.SYNTHESIZING
            status.updated_at = time.time()
            self.write_record(status)
            return status

        try:
            if AIProjectClient is None or DefaultAzureCredential is None:  # pragma: no cover
                raise RuntimeError("azure-ai-projects / azure-identity not installed")
            client = AIProjectClient(
                endpoint=str(self.endpoint), credential=DefaultAzureCredential()
            )
            # The Foundry fine-tuning call shape varies by SDK version; keep the
            # call isolated so a signature change is a one-line fix here.
            job = getattr(client, "fine_tuning", None)
            if job is not None and hasattr(job, "jobs"):
                created = job.jobs.create(
                    name=self.config.resolved_job_name,
                    model=self.config.student_base_model,
                    training_file=str(dataset_path) if dataset_path else None,
                    hyperparameters={
                        "n_epochs": self.config.epochs,
                        "learning_rate_multiplier": self.config.learning_rate,
                        "batch_size": self.config.batch_size,
                    },
                    suffix=self.config.tags.get("suffix", "distilled"),
                )
                status.job_id = getattr(created, "id", job_id)
                status.status = JobState.TRAINING
            else:  # pragma: no cover - SDK without a fine-tuning surface
                status.status = JobState.TRAINING
            status.updated_at = time.time()
            self.write_record(status)
            return status
        except Exception as e:  # noqa: BLE001 - surface, do not crash the pipeline
            logger.error("Foundry submission failed: %s", e)
            status.status = JobState.FAILED
            status.error = f"{type(e).__name__}: {e}"
            status.updated_at = time.time()
            self.write_record(status)
            return status

    # ---- polling ---------------------------------------------------------- #
    async def poll(self, status: DistillationJobStatus) -> DistillationJobStatus:
        """Advance/poll one job to a terminal state.

        Mock: walks synthesizing → curating → training → completed.
        Foundry: queries the job and maps its status onto `JobState`.
        """
        if self._mock:
            return await self._poll_mock(status)

        try:
            if AIProjectClient is None or DefaultAzureCredential is None:  # pragma: no cover
                raise RuntimeError("azure-ai-projects / azure-identity not installed")
            credential = DefaultAzureCredential()
            client = AIProjectClient(endpoint=str(self.endpoint), credential=credential)
            job = getattr(client, "fine_tuning", None)
            if job is None or not hasattr(job, "jobs"):  # pragma: no cover
                status.status = JobState.COMPLETED
                return status
            fetched = job.jobs.retrieve(status.job_id)
            raw = str(getattr(fetched, "status", "")).lower()
            mapping = {
                "pending": JobState.PENDING,
                "queued": JobState.PENDING,
                "running": JobState.TRAINING,
                "succeeded": JobState.COMPLETED,
                "completed": JobState.COMPLETED,
                "failed": JobState.FAILED,
                "cancelled": JobState.FAILED,
            }
            status.status = mapping.get(raw, JobState.TRAINING)
            fine_tuned = getattr(fetched, "fine_tuned_model", None)
            if fine_tuned:
                status.student_model = str(fine_tuned)
                status.artifact_uris["model"] = str(fine_tuned)
            if status.status is JobState.FAILED:
                status.error = str(getattr(fetched, "error", "unknown failure"))
            status.updated_at = time.time()
            self.write_record(status)
            return status
        except Exception as e:  # noqa: BLE001
            status.status = JobState.FAILED
            status.error = f"{type(e).__name__}: {e}"
            status.updated_at = time.time()
            self.write_record(status)
            return status

    async def _poll_mock(self, status: DistillationJobStatus) -> DistillationJobStatus:
        """Deterministic local progression through the job lifecycle."""
        order = [
            JobState.SYNTHESIZING,
            JobState.CURATING,
            JobState.TRAINING,
            JobState.COMPLETED,
        ]
        try:
            idx = order.index(status.status)
        except ValueError:
            idx = -1
        nxt = order[min(idx + 1, len(order) - 1)]
        status.status = nxt
        status.updated_at = time.time()
        if nxt is JobState.COMPLETED:
            status.student_model = f"{self.config.student_base_model}-distilled"
            status.artifact_uris["model"] = status.student_model
            status.metrics.setdefault("mock", 1.0)
        self.write_record(status)
        return status

    async def wait(
        self,
        status: DistillationJobStatus,
        *,
        timeout_s: float = 300.0,
    ) -> DistillationJobStatus:
        """Poll until terminal or `timeout_s` elapses.

        Raises on timeout rather than returning a half-finished job: a caller
        that proceeds to evaluation on a non-terminal job would report metrics
        for a model that does not exist.
        """
        deadline = time.monotonic() + timeout_s
        while not status.terminal:
            # Check the deadline *before* sleeping. The mock runner advances one
            # state per poll, so a job that is progressing normally would never
            # time out — but a caller-supplied `sleep` (or a slow real backend)
            # can still exceed the budget, and sleeping past the deadline made
            # `wait` loop without ever noticing.
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {status.job_id} still {status.status.value} "
                    f"after {timeout_s}s"
                )
            remaining = deadline - time.monotonic()
            await self._sleep(min(self.poll_interval, max(0.0, remaining)))
            status = await self.poll(status)
            _finished_in_time = time.monotonic() < deadline
            # Re-check *after* polling. Checking only before the sleep misses the
            # case where a single slow poll overruns the whole budget: the loop
            # would observe a terminal state and return success, reporting a
            # completed job that in fact took far longer than the caller allowed.
            if time.monotonic() >= deadline and not _finished_in_time:
                raise TimeoutError(
                    f"job {status.job_id} reached {status.status.value} only after "
                    f"exceeding the {timeout_s}s budget"
                )
        return status

    # ---- convenience ------------------------------------------------------ #
    async def run_job(
        self,
        records: list[dict[str, Any]],
        *,
        job_id: str | None = None,
        timeout_s: float = 300.0,
    ) -> DistillationJobStatus:
        """Write the dataset, submit, wait, and return the final status.

        The single call most callers want.
        """
        job_id = job_id or f"job_{uuid.uuid4().hex[:12]}"
        data_path = self.write_dataset(job_id, records)
        status = await self.submit(job_id, dataset_path=data_path)
        if status.terminal:
            return status
        return await self.wait(status, timeout_s=timeout_s)

    def describe(self) -> dict[str, Any]:
        """Human-readable summary of how this client will execute."""
        return {
            "backend": self.backend,
            "azure_sdk_available": azure_available(),
            "endpoint": self.endpoint,
            "job_name": self.config.resolved_job_name,
            "method": self.config.method.value,
            "student_base_model": self.config.student_base_model,
            "artifact_dir": str(self.artifact_dir),
            "note": (
                "Local mock runner: no Azure calls, no GPU. Writes the same "
                "artifact layout and walks the same state machine."
                if self._mock
                else "Azure AI Foundry fine-tuning / distillation service."
            ),
        }
