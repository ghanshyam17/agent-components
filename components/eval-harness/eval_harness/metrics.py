"""Evaluation metrics.

A metric is a callable ``(example, prediction, context) -> float`` returning a
score in ``[0, 1]``. It may be synchronous or async (the runner awaits async
metrics). ``prediction`` is the dict returned by the target function
(``{"output": str, "latency": float, "tool_calls": list | None}``) and
``context`` is a dict of per-run side info (``latency``, ``judge_client``, ...).

Use the `Metric` dataclass to bundle a metric with a name and an ``is_async``
flag; the bare functions below are convenient, pre-built `Metric` instances.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Protocol

from eval_harness.dataset import Example


# --------------------------------------------------------------------------- #
# Metric types
# --------------------------------------------------------------------------- #
class MetricFn(Protocol):
    def __call__(self, example: Example, prediction: dict, context: dict) -> Any: ...


@dataclass
class Metric:
    """A named metric: a callable plus a flag for async evaluation."""

    name: str
    fn: Callable[..., Any]
    is_async: bool = False

    async def score(self, example: Example, prediction: dict, context: dict) -> float:
        if self.is_async:
            result = await self.fn(example, prediction, context)
        else:
            result = self.fn(example, prediction, context)
        return float(result)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _output(prediction: dict) -> str:
    return str(prediction.get("output") or "")


def _expected_tools(example: Example) -> list[str]:
    """Extract expected tool names from an example: a list ``expected`` of
    names/dicts, a dict with ``tool_calls``, or ``metadata['expected_tools']``."""
    if isinstance(example.expected, list):
        return [_tool_name(x) for x in example.expected]
    if isinstance(example.expected, dict):
        if "tool_calls" in example.expected:
            return [_tool_name(x) for x in example.expected["tool_calls"]]
    return [str(x) for x in example.metadata.get("expected_tools", [])]


def _predicted_tools(prediction: dict) -> list[str]:
    calls = prediction.get("tool_calls") or []
    return [_tool_name(x) for x in calls]


def _tool_name(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, dict):
        return str(x.get("name") or x.get("tool") or "")
    return str(x)


# --------------------------------------------------------------------------- #
# Built-in metrics
# --------------------------------------------------------------------------- #
def exact_match_fn(example: Example, prediction: dict, context: dict) -> float:
    """1.0 if prediction output equals ``example.expected`` after strip, else 0.0."""
    if not isinstance(example.expected, str):
        return 0.0
    return 1.0 if _output(prediction).strip() == example.expected.strip() else 0.0


def contains_fn(example: Example, prediction: dict, context: dict) -> float:
    """1.0 if ``example.expected`` (string) is a substring of the output, else 0.0."""
    if not isinstance(example.expected, str):
        return 0.0
    return 1.0 if example.expected.strip() in _output(prediction) else 0.0


def make_regex_match_fn(pattern: str):
    compiled = re.compile(pattern)

    def _fn(example: Example, prediction: dict, context: dict) -> float:
        return 1.0 if compiled.search(_output(prediction)) else 0.0

    return _fn


def tool_call_accuracy_fn(example: Example, prediction: dict, context: dict) -> float:
    """Order-insensitive tool-call overlap.

    Score is ``|expected ∩ predicted| / max(|expected|, |predicted|)``: 1.0
    when the predicted tool set exactly matches the expected set, and penalized
    for both missing tools (recall) and extra tools (precision).
    """
    expected = set(_expected_tools(example))
    predicted = set(_predicted_tools(prediction))
    if not expected and not predicted:
        return 1.0
    correct = len(expected & predicted)
    return correct / max(len(expected), len(predicted))


def make_latency_fn(max_latency: float):
    def _fn(example: Example, prediction: dict, context: dict) -> float:
        latency = float(context.get("latency") or prediction.get("latency") or 0.0)
        if max_latency <= 0:
            return 0.0
        # 1.0 at latency 0, 0.0 at latency == max_latency, clamped to [0, 1].
        score = 1.0 - (latency / max_latency)
        return max(0.0, min(1.0, score))

    return _fn


DEFAULT_JUDGE_TEMPLATE = (
    "You are an evaluation judge. Decide whether the prediction satisfies the "
    "expected answer for the given input.\n\n"
    "Input:\n{input}\n\n"
    "Expected:\n{expected}\n\n"
    "Prediction:\n{prediction}\n\n"
    "Rubric: the prediction must match the expected answer in meaning and "
    "contain the key information. Ignore minor wording differences.\n"
    "Reply with a single word: PASS or FAIL."
)


def make_llm_judge_fn(judge_client: Any, prompt_template: str = DEFAULT_JUDGE_TEMPLATE):
    """Build the async scoring fn for an LLM-judge metric.

    ``judge_client`` is any object with ``async chat(messages) -> (content, _)``
    (the router's `ModelClient` / the gateway's `GatewayClient` shape). If
    ``judge_client`` is None the function falls back to ``context["judge_client"]``,
    so a judge can be supplied at the `evaluate()` call instead.
    """

    async def _fn(example: Example, prediction: dict, context: dict) -> float:
        client = judge_client if judge_client is not None else context.get("judge_client")
        if client is None:
            raise RuntimeError(
                "llm_judge requires a judge_client — pass one to llm_judge() or evaluate()"
            )
        prompt = prompt_template.format(
            input=example.input,
            expected=example.expected if example.expected is not None else "",
            prediction=_output(prediction),
        )
        messages = [{"role": "user", "content": prompt}]
        result = await client.chat(messages)
        # The chat() contract returns (content, tool_calls); the judge needs only content.
        content = result[0] if isinstance(result, tuple) else str(result)
        verdict = (str(content) or "").strip().upper()
        # Take the first token to tolerate any extra prose.
        first = verdict.split()[0] if verdict else ""
        return 1.0 if first == "PASS" else 0.0

    return _fn


# --------------------------------------------------------------------------- #
# Pre-built Metric instances (convenient for common cases)
# --------------------------------------------------------------------------- #
exact_match = Metric(name="exact_match", fn=exact_match_fn)
contains = Metric(name="contains", fn=contains_fn)
tool_call_accuracy = Metric(name="tool_call_accuracy", fn=tool_call_accuracy_fn)


def regex_match(pattern: str) -> Metric:
    """A metric that scores 1.0 if ``pattern`` matches the output, else 0.0."""
    return Metric(name=f"regex_match:{pattern}", fn=make_regex_match_fn(pattern))


def latency(max_latency: float) -> Metric:
    """A metric that normalizes prediction latency against a threshold (seconds)."""
    return Metric(name="latency", fn=make_latency_fn(max_latency))


def llm_judge(
    judge_client: Any = None,
    *,
    prompt_template: str = DEFAULT_JUDGE_TEMPLATE,
    name: str = "llm_judge",
) -> Metric:
    """An async LLM-judge metric returning 1.0 for PASS, 0.0 for FAIL."""
    return Metric(
        name=name,
        fn=make_llm_judge_fn(judge_client, prompt_template),
        is_async=True,
    )