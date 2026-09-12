"""The evaluation runner.

`evaluate()` runs an async target over every example in a dataset concurrently,
scores each prediction with the supplied metrics, and returns an
`EvaluationReport`.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Awaitable, Callable

from components_core import get_logger

from eval_harness.dataset import Dataset, Example
from eval_harness.metrics import Metric
from eval_harness.report import EvaluationReport, ExampleResult

logger = get_logger("eval_harness.runner")

# Target: async (example) -> dict with at least output, latency, tool_calls.
TargetFn = Callable[[Example], Awaitable[dict[str, Any]]]


async def _run_one(
    target: TargetFn,
    example: Example,
    metrics: list[Metric],
    judge_client: Any,
    sem: asyncio.Semaphore,
) -> ExampleResult:
    """Run the target and score one example. Captures target errors so a
    single failure doesn't abort the whole run."""
    async with sem:
        t0 = time.monotonic()
        try:
            prediction = await target(example)
        except Exception as e:  # noqa: BLE001 — record and continue
            logger.warning("target error on %s: %s", example.id, e)
            return ExampleResult(
                id=example.id,
                scores={},
                latency=time.monotonic() - t0,
                output="",
                error=f"{type(e).__name__}: {e}",
            )

        # Make sure latency is present even if the target didn't supply it.
        if prediction.get("latency") is None:
            prediction["latency"] = time.monotonic() - t0

        context = {
            "latency": prediction.get("latency", 0.0),
            "judge_client": judge_client,
        }

        scores: dict[str, float] = {}
        for metric in metrics:
            try:
                scores[metric.name] = await metric.score(example, prediction, context)
            except Exception as e:  # noqa: BLE001 — record and continue
                logger.warning("metric %s error on %s: %s", metric.name, example.id, e)
                scores[metric.name] = 0.0

        return ExampleResult(
            id=example.id,
            scores=scores,
            latency=float(prediction.get("latency") or 0.0),
            output=str(prediction.get("output") or ""),
            error=None,
        )


async def evaluate(
    target: TargetFn,
    dataset: Dataset,
    metrics: list[Metric],
    *,
    judge_client: Any = None,
    concurrency: int | None = None,
) -> EvaluationReport:
    """Run ``target`` over every example in ``dataset`` and score with ``metrics``.

    Parameters
    ----------
    target:
        ``async (example) -> dict`` returning at least
        ``{"output": str, "latency": float, "tool_calls": list | None}``.
    dataset:
        The `Dataset` to evaluate over.
    metrics:
        List of `Metric` instances (sync or async).
    judge_client:
        Optional judge client available to metrics via ``context["judge_client"]``
        (used by `llm_judge` when no client is bound to the metric).
    concurrency:
        Optional cap on in-flight target calls (defaults to no cap).
    """
    sem = asyncio.Semaphore(concurrency) if concurrency else asyncio.Semaphore(len(dataset) or 1)
    coros = [
        _run_one(target, example, metrics, judge_client, sem) for example in dataset
    ]
    results = await asyncio.gather(*coros)
    return EvaluationReport(results=list(results))