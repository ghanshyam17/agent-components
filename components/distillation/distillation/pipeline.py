"""End-to-end distillation pipeline: synthesise → curate → train → evaluate.

This is the composition root. It wires the four engines together and, crucially,
carries the **artifacts of each stage forward as data** rather than only as side
effects — so a caller can inspect the curated dataset, the curation report and
the parity matrix together, and answer "why is this student like this?".

Failure is staged and visible. If the teacher is unavailable the pipeline does
not pretend to have produced a dataset; each stage records whether it ran, and
:meth:`DistillationPipeline.run` returns a result whose ``stages`` map shows
exactly how far it got.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from distillation.curator import ContentCurator
from distillation.evaluator import ParityEvaluator
from distillation.foundry import FoundryDistillationClient
from distillation.models import (
    CurationReport,
    ContentSample,
    DatasetFormat,
    DistillationConfig,
    DistillationJobStatus,
    DPOPair,
    JobState,
    TrainingMethod,
)
from distillation.synthesizer import SyntheticContentGenerator

logger = logging.getLogger(__name__)

__all__ = ["DistillationPipeline", "DistillationResult"]


@dataclass
class DistillationResult:
    """Everything one pipeline run produced.

    ``stages`` records which stages actually executed, so a partially-completed
    run is self-describing instead of surfacing as missing keys.
    """

    config: DistillationConfig
    samples: list[ContentSample] = field(default_factory=list)
    curated: list[ContentSample] = field(default_factory=list)
    quarantine: list[dict[str, Any]] = field(default_factory=list)
    dpo_pairs: list[DPOPair] = field(default_factory=list)
    curation: CurationReport = field(default_factory=CurationReport)
    job: DistillationJobStatus | None = None
    parity: Any | None = None
    dataset_path: str | None = None
    dpo_path: str | None = None
    #: prompt -> candidate completions, populated in DPO mode so pairs can be mined.
    dpo_candidates: list[tuple[str, list[str]]] = field(default_factory=list)
    stages: dict[str, bool] = field(default_factory=dict)
    duration_ms: int = 0

    @property
    def ok(self) -> bool:
        """Whether every stage ran and the job reached completion."""
        return (
            all(self.stages.get(s, False) for s in ("synthesize", "curate", "train"))
            and self.job is not None
            and self.job.status is JobState.COMPLETED
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "stages": self.stages,
            "ok": self.ok,
            "duration_ms": self.duration_ms,
            "counts": {
                "generated": len(self.samples),
                "curated": len(self.curated),
                "quarantined": len(self.quarantine),
                "dpo_pairs": len(self.dpo_pairs),
            },
            "curation": self.curation.to_dict(),
            "job": self.job.to_dict() if self.job else None,
            "parity": self.parity.to_dict() if self.parity is not None else None,
            "dataset_path": self.dataset_path,
            "dpo_path": self.dpo_path,
        }

    def summary(self) -> str:
        """One-paragraph plain-text summary, for logs and CLI output."""
        p = self.parity
        lines = [
            f"distillation[{self.config.method.value}] "
            f"{'OK' if self.ok else 'INCOMPLETE'} in {self.duration_ms} ms",
            f"  generated {len(self.samples)} → curated {len(self.curated)} "
            f"(quarantined {len(self.quarantine)}, deduped {self.curation.deduplicated})",
        ]
        if self.dpo_pairs:
            lines.append(f"  DPO pairs: {len(self.dpo_pairs)}")
        if self.job:
            lines.append(
                f"  job {self.job.job_id} → {self.job.status.value} "
                f"({self.job.backend}) student={self.job.student_model}"
            )
        if p is not None:
            lines.append(
                f"  parity: quality {p.student_quality:.3f} vs {p.teacher_quality:.3f} "
                f"(retention {p.quality_retention:.0%}), "
                f"style {p.stylistic_similarity:.0%}, "
                f"TTFT ×{p.latency_speedup:.2f}, "
                f"cost −{p.cost_savings_pct:.1f}%"
            )
        return "\n".join(lines)


class DistillationPipeline:
    """Orchestrate the full four-stage distillation.

    Parameters
    ----------
    config:
        The job configuration.
    teacher_client:
        Teacher callable (``model_gateway`` client ``.chat``). Without it the
        synthesis stage is skipped and reported as such.
    student_client:
        Optional student callable, used only by the parity evaluator.
    curator / foundry / evaluator:
        Inject any stage's engine (tests and custom wiring).
    artifact_dir:
        Where datasets and job records are written.
    """

    def __init__(
        self,
        config: DistillationConfig,
        *,
        teacher_client: Callable[..., Awaitable[Any]] | None = None,
        student_client: Callable[..., Awaitable[Any]] | None = None,
        curator: ContentCurator | None = None,
        foundry: FoundryDistillationClient | None = None,
        evaluator: ParityEvaluator | None = None,
        registry: Any | None = None,
        artifact_dir: str | Path = "artifacts",
        force_mock: bool = True,
        dpo_candidates: int = 3,
    ) -> None:
        self.config = config
        # How many completions to sample per prompt in DPO mode. Two is the
        # minimum that yields a pair; three gives the scorer something to rank.
        self.dpo_candidates = max(2, dpo_candidates)
        self.artifact_dir = Path(artifact_dir)
        self.generator = SyntheticContentGenerator(teacher_client, registry=registry)
        self.curator = curator or ContentCurator()
        self.foundry = foundry or FoundryDistillationClient(
            config, artifact_dir=self.artifact_dir, force_mock=force_mock
        )
        # The evaluator measures latency/tokens, so it needs its own call shape.
        # The synthesis client returns `(content, tool_calls)` (the router's
        # shape) while the evaluator expects `(content, latency, tokens_in,
        # tokens_out)` — passing the raw client across made every evaluation fail
        # with "not enough values to unpack".
        self.evaluator = evaluator or ParityEvaluator(
            teacher=self._adapt(teacher_client) if teacher_client else None,
            student=self._adapt(student_client) if student_client else None,
        )

    @staticmethod
    def _adapt(client: Callable[..., Awaitable[Any]] | None) -> Any:
        """Wrap a chat-shaped client in the evaluator's measurement shape."""
        if client is None:
            return None

        async def _measured(prompt: str, **_: Any):
            from components_core import Message

            t0 = time.perf_counter()
            out = await client([Message(role="user", content=prompt)])
            latency = time.perf_counter() - t0
            content = out[0] if isinstance(out, (tuple, list)) and out else str(out or "")
            # Token counts are estimated from word counts: the chat interface
            # does not report usage. A caller needing exact counts should inject
            # its own evaluator.
            return str(content), latency, len(prompt.split()), len(str(content).split())

        return _measured

    # ---- stages ----------------------------------------------------------- #
    async def synthesize(self, result: DistillationResult) -> None:
        """Stage 1 — generate synthetic samples from the teacher.

        In DPO mode this samples **several completions per prompt**. The normal
        path produces exactly one sample per (topic, blueprint) pair, so grouping
        the result by prompt yields only singleton candidate sets and no
        preference pair can ever be mined — leaving `method='dpo'` structurally
        unable to produce a dataset.
        """
        if self.config.method is TrainingMethod.DPO:
            candidate_sets = await self.generator.generate_preference_pairs(
                self.config, candidates=self.dpo_candidates
            )
            # Keep every candidate as a sample so curation scores them all, and
            # remember the grouping so the DPO stage can pair them.
            result.dpo_candidates = candidate_sets
            samples: list[ContentSample] = []
            for prompt, completions in candidate_sets:
                for text in completions:
                    samples.append(
                        ContentSample(
                            prompt=prompt,
                            completion=text,
                            teacher_model=self.config.teacher_model,
                            format_type=self.config.format,
                            metadata={"dpo_group": prompt},
                        )
                    )
        else:
            samples = await self.generator.generate(self.config)

        result.samples = samples
        result.stages["synthesize"] = bool(samples)
        if not samples:
            logger.warning(
                "synthesis produced no samples (teacher unbound or all calls failed)"
            )

    async def curate(self, result: DistillationResult, *, build_dpo: bool | None = None) -> None:
        """Stage 2 — hygiene, quality gate, dedup, and optionally DPO pairs."""
        if not result.samples:
            result.stages["curate"] = False
            return
        curated, quarantine, report = await self.curator.curate(result.samples, self.config)
        result.curated = curated
        result.quarantine = quarantine
        result.curation = report
        result.stages["curate"] = bool(curated)

        want_dpo = self.config.method is TrainingMethod.DPO if build_dpo is None else build_dpo
        if want_dpo:
            # Prefer the candidate sets the generator actually produced for DPO.
            # Re-grouping the curated samples by prompt does not work: curation
            # deduplicates and the normal path emits one sample per prompt, so
            # the groups are singletons.
            by_prompt: dict[str, list[str]] = {}
            for prompt, completions in (result.dpo_candidates or []):
                by_prompt.setdefault(prompt, []).extend(completions)
            if not by_prompt:
                for s in curated:
                    by_prompt.setdefault(s.prompt, []).append(s.completion)
            candidates = [(p, c) for p, c in by_prompt.items() if len(c) >= 2]
            pairs, dropped = await self.curator.build_dpo_pairs(candidates, self.config)
            result.dpo_pairs = pairs
            if dropped:
                logger.info("dropped %d DPO candidate groups (too few or too close)", dropped)

    async def train(self, result: DistillationResult, *, timeout_s: float = 300.0) -> None:
        """Stage 3 — submit the curated dataset and wait for the job.

        For a DPO config the training file must contain ``(prompt, chosen,
        rejected)`` rows. If no pairs could be mined, submitting anyway would
        write raw `ContentSample` records under a DPO method — a job that
        either fails server-side or trains on the wrong schema. Refuse instead,
        and say which condition failed.
        """
        if not result.curated:
            result.stages["train"] = False
            logger.warning("train skipped: nothing survived curation")
            return

        if self.config.method is TrainingMethod.DPO and not result.dpo_pairs:
            result.stages["train"] = False
            logger.error(
                "train skipped: method='dpo' but no preference pairs were mined. "
                "Pairs need >=2 differing completions per prompt and a margin above "
                "the threshold; a deterministic teacher produces neither."
            )
            return

        # For DPO the pairs *are* the dataset.
        if self.config.method is TrainingMethod.DPO:
            records = [p_.to_dpo() for p_ in result.dpo_pairs]
        else:
            records = self.curator.to_records(result.curated, self.config)
        job_id = f"job_{int(time.time())}"
        data_path = self.foundry.write_dataset(job_id, records, "train")
        result.dataset_path = str(data_path)

        if result.dpo_pairs:
            dpo_path = self.foundry.write_dataset(
                job_id, [p.to_dpo() for p in result.dpo_pairs], "dpo"
            )
            result.dpo_path = str(dpo_path)

        status = await self.foundry.submit(job_id, dataset_path=data_path)
        if not status.terminal:
            status = await self.foundry.wait(status, timeout_s=timeout_s)
        result.job = status
        result.stages["train"] = status.status is JobState.COMPLETED

    async def evaluate(self, result: DistillationResult, prompts: Sequence[str] | None = None) -> None:
        """Stage 4 — teacher/student parity over held-out prompts."""
        if self.evaluator.teacher is None or self.evaluator.student is None:
            result.stages["evaluate"] = False
            return
        held_out = list(prompts) if prompts else [
            str(s.metadata.get("topic", "")) or s.prompt for s in result.curated[:10]
        ]
        report = await self.evaluator.evaluate(held_out, self.config)
        result.parity = report
        result.stages["evaluate"] = bool(report.examples)

    # ---- the whole thing -------------------------------------------------- #
    async def run(
        self,
        *,
        eval_prompts: Sequence[str] | None = None,
        timeout_s: float = 300.0,
    ) -> DistillationResult:
        """Run all four stages, recording which succeeded.

        Stage failures do not abort the run: a caller that got a dataset but no
        trained model is in a materially different position from one that got
        nothing, and the ``stages`` map tells them which.
        """
        start = time.time()
        result = DistillationResult(config=self.config)
        try:
            await self.synthesize(result)
        except Exception as e:  # noqa: BLE001
            logger.error("synthesis failed: %s", e)
            result.stages["synthesize"] = False
        try:
            await self.curate(result)
        except Exception as e:  # noqa: BLE001
            logger.error("curation failed: %s", e)
            result.stages["curate"] = False
        try:
            await self.train(result, timeout_s=timeout_s)
        except Exception as e:  # noqa: BLE001
            logger.error("training failed: %s", e)
            result.stages["train"] = False
        try:
            await self.evaluate(result, eval_prompts)
        except Exception as e:  # noqa: BLE001
            logger.error("evaluation failed: %s", e)
            result.stages["evaluate"] = False
        result.duration_ms = int((time.time() - start) * 1000)
        return result
