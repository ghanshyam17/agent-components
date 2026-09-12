"""Evaluation report models and rendering.

`ExampleResult` is one row; `EvaluationReport` is the full run. Aggregates are
computed from the per-example results in a `model_validator` so a report is
always self-consistent.
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator


class ExampleResult(BaseModel):
    """Per-example scores and prediction."""

    id: str
    scores: dict[str, float] = Field(default_factory=dict)
    latency: float = 0.0
    output: str = ""
    error: str | None = None


class Aggregate(BaseModel):
    """Run-level aggregates."""

    mean_scores: dict[str, float] = Field(default_factory=dict)
    overall: float = 0.0
    count: int = 0
    failure_count: int = 0


class EvaluationReport(BaseModel):
    """The result of an `evaluate()` run."""

    results: list[ExampleResult] = Field(default_factory=list)
    aggregate: Aggregate = Field(default_factory=Aggregate)

    @model_validator(mode="after")
    def _compute_aggregate(self) -> "EvaluationReport":
        if not self.results:
            self.aggregate = Aggregate()
            return self

        metric_names: list[str] = []
        seen: set[str] = set()
        for r in self.results:
            for name in r.scores:
                if name not in seen:
                    seen.add(name)
                    metric_names.append(name)

        mean_scores: dict[str, float] = {}
        for name in metric_names:
            values = [r.scores[name] for r in self.results if name in r.scores]
            mean_scores[name] = (
                sum(values) / len(values) if values else 0.0
            )

        # Overall: mean of per-example mean scores (each example contributes
        # equally regardless of how many metrics ran on it).
        per_example_means = [
            sum(r.scores.values()) / len(r.scores)
            for r in self.results
            if r.scores
        ]
        overall = sum(per_example_means) / len(per_example_means) if per_example_means else 0.0
        failure_count = sum(1 for r in self.results if r.error is not None)

        self.aggregate = Aggregate(
            mean_scores=mean_scores,
            overall=overall,
            count=len(self.results),
            failure_count=failure_count,
        )
        return self

    # -- rendering ---------------------------------------------------------- #
    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.model_dump(), indent=indent)

    def to_markdown(self) -> str:
        lines: list[str] = []
        agg = self.aggregate
        lines.append(f"### Evaluation report ({agg.count} examples, "
                     f"{agg.failure_count} failures)")
        lines.append("")
        lines.append("| metric | mean |")
        lines.append("|--------|------|")
        for name, mean in agg.mean_scores.items():
            lines.append(f"| {name} | {mean:.3f} |")
        lines.append(f"| **overall** | **{agg.overall:.3f}** |")
        lines.append("")
        lines.append("| id | output | latency | error |")
        lines.append("|----|--------|---------|-------|")
        for r in self.results:
            out = (r.output or "").replace("|", "\\|").replace("\n", " ")
            if len(out) > 60:
                out = out[:57] + "..."
            err = (r.error or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| {r.id} | {out} | {r.latency:.3f} | {err} |")
        return "\n".join(lines)