"""eval-harness: dataset-driven evaluation for AI/agent systems.

Load examples, run an async target function over them, score each prediction
with metrics (including an async LLM-judge), and produce an `EvaluationReport`
with per-example scores and aggregates.

Quick start::

    from eval_harness import Dataset, Example, evaluate, exact_match, contains

    dataset = Dataset(examples=[Example(id="1", input="hi", expected="hi")])

    async def target(example):
        return {"output": example.input, "latency": 0.0, "tool_calls": None}

    report = await evaluate(target, dataset, [exact_match, contains])
    print(report.to_markdown())
"""
from __future__ import annotations

from eval_harness.dataset import Dataset, Example
from eval_harness.metrics import (
    Metric,
    contains,
    exact_match,
    latency,
    llm_judge,
    regex_match,
    tool_call_accuracy,
)
from eval_harness.report import EvaluationReport, ExampleResult
from eval_harness.runner import evaluate

__all__ = [
    "Example",
    "Dataset",
    "Metric",
    "exact_match",
    "contains",
    "regex_match",
    "tool_call_accuracy",
    "latency",
    "llm_judge",
    "evaluate",
    "EvaluationReport",
    "ExampleResult",
]