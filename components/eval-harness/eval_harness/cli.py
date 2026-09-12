"""eval-harness CLI — run an evaluation from the command line.

Demo target: ``echo`` (returns the input as output) so the CLI is fully
demonstrable offline. Wire a real target by importing `evaluate` in a script.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Annotated

import typer

from eval_harness.dataset import Dataset, Example
from eval_harness.metrics import Metric, contains, exact_match, regex_match
from eval_harness.runner import evaluate

app = typer.Typer(add_completion=False, help="Dataset-driven evaluation for AI/agent systems.")


# --------------------------------------------------------------------------- #
# Built-in targets
# --------------------------------------------------------------------------- #
async def echo_target(example: Example) -> dict:
    """Echo the input as output — a deterministic offline target."""
    return {"output": example.input, "latency": 0.0, "tool_calls": None}


async def stub_target(example: Example) -> dict:
    """A stub target that returns an empty output (everything fails)."""
    return {"output": "", "latency": 0.0, "tool_calls": None}


TARGETS = {"echo": echo_target, "stub": stub_target}


# --------------------------------------------------------------------------- #
# Metric parsing
# --------------------------------------------------------------------------- #
def _build_metrics(names: list[str]) -> list[Metric]:
    metrics: list[Metric] = []
    for raw in names:
        token = raw.strip()
        if not token:
            continue
        if token == "exact_match":
            metrics.append(exact_match)
        elif token == "contains":
            metrics.append(contains)
        elif token.startswith("regex:"):
            metrics.append(regex_match(token[len("regex:"):]))
        else:
            raise typer.BadParameter(
                f"unknown metric {token!r} (known: exact_match, contains, regex:PATTERN)"
            )
    if not metrics:
        metrics.append(exact_match)
    return metrics


def _load_dataset(path: Path) -> Dataset:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return Dataset.load_json(path)
    if suffix == ".jsonl":
        return Dataset.load_jsonl(path)
    if suffix == ".csv":
        return Dataset.load_csv(path)
    if suffix in (".yaml", ".yml"):
        return Dataset.load_yaml(path)
    raise typer.BadParameter(
        f"unsupported dataset format {suffix!r} (use .json/.jsonl/.csv/.yaml)"
    )


@app.command()
def run(
    dataset: Annotated[Path, typer.Option("--dataset", "-d", help="Path to the dataset file.")],
    target: Annotated[
        str, typer.Option("--target", "-t", help="Built-in target: echo | stub.")
    ] = "echo",
    metrics: Annotated[
        str,
        typer.Option(
            "--metrics", "-m",
            help="Comma-separated metrics: exact_match,contains,regex:PATTERN.",
        ),
    ] = "exact_match",
) -> None:
    """Load a dataset, run the target over it, and print a markdown report."""
    if target not in TARGETS:
        raise typer.BadParameter(f"unknown target {target!r} (known: {', '.join(TARGETS)})")
    ds = _load_dataset(dataset)
    metric_list = _build_metrics(metrics.split(","))
    report = evaluate(TARGETS[target], ds, metric_list)
    # evaluate is async; run it on the event loop.
    import asyncio

    rep = asyncio.run(report)
    typer.echo(rep.to_markdown())


if __name__ == "__main__":  # pragma: no cover
    app()