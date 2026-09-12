"""eval-harness tests — offline: metrics, the runner with an echo-style async
target over an in-memory dataset, and an LLM-judge with a fake judge client."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from eval_harness import (
    Dataset,
    Example,
    contains,
    evaluate,
    exact_match,
    latency,
    llm_judge,
    regex_match,
    tool_call_accuracy,
)
from eval_harness.report import EvaluationReport


# ----------------------- metrics ----------------------- #
def test_exact_match_strips_whitespace():
    ex = Example(id="1", input="x", expected="hello")
    assert exact_match.fn(ex, {"output": "  hello  "}, {}) == 1.0
    assert exact_match.fn(ex, {"output": "world"}, {}) == 0.0
    # non-string expected → never matches
    assert exact_match.fn(Example(id="2", input="x", expected={"k": "v"}), {"output": "x"}, {}) == 0.0


def test_contains_metric():
    ex = Example(id="1", input="x", expected="Paris")
    assert contains.fn(ex, {"output": "The capital is Paris."}, {}) == 1.0
    assert contains.fn(ex, {"output": "London"}, {}) == 0.0
    # non-string expected → never matches
    assert contains.fn(Example(id="2", input="x", expected={"k": "v"}), {"output": "x"}, {}) == 0.0


def test_regex_match_metric():
    m = regex_match(r"\d+")
    ex = Example(id="1", input="x", expected="ignored")
    assert m.fn(ex, {"output": "answer is 42"}, {}) == 1.0
    assert m.fn(ex, {"output": "no digits here"}, {}) == 0.0


def test_tool_call_accuracy_matches_unordered():
    # expected as a list of dicts with "name"
    ex = Example(
        id="1", input="x",
        expected=[{"name": "search"}, {"name": "calc"}],
    )
    # prediction tool_calls as list of dicts, different order, one extra → penalized
    pred = {"output": "", "tool_calls": [{"name": "calc"}, {"name": "search"}, {"name": "extra"}]}
    score = tool_call_accuracy.fn(ex, pred, {})
    expected_set = {"search", "calc"}
    predicted_set = {"calc", "search", "extra"}
    assert score == len(expected_set & predicted_set) / max(len(expected_set), len(predicted_set))

    # exact set match → 1.0
    pred2 = {"output": "", "tool_calls": ["calc", "search"]}
    assert tool_call_accuracy.fn(ex, pred2, {}) == 1.0

    # missing one → 1/2
    pred3 = {"output": "", "tool_calls": ["search"]}
    assert tool_call_accuracy.fn(ex, pred3, {}) == 0.5

    # empty both → 1.0
    ex_empty = Example(id="2", input="x")
    assert tool_call_accuracy.fn(ex_empty, {"output": "", "tool_calls": None}, {}) == 1.0


def test_tool_call_accuracy_expected_from_metadata():
    ex = Example(id="1", input="x", metadata={"expected_tools": ["a", "b"]})
    assert tool_call_accuracy.fn(ex, {"output": "", "tool_calls": ["a", "b"]}, {}) == 1.0


def test_latency_metric_normalizes():
    m = latency(max_latency=2.0)
    ex = Example(id="1", input="x")
    assert m.fn(ex, {"output": "", "latency": 0.0}, {"latency": 0.0}) == 1.0
    assert m.fn(ex, {"output": "", "latency": 2.0}, {"latency": 2.0}) == 0.0
    assert m.fn(ex, {"output": "", "latency": 1.0}, {"latency": 1.0}) == 0.5
    # clamps above 1.0 / below 0.0
    assert m.fn(ex, {"output": "", "latency": 4.0}, {"latency": 4.0}) == 0.0


# ----------------------- runner ----------------------- #
def test_evaluate_echo_target_aggregates():
    dataset = Dataset(examples=[
        Example(id="1", input="hi", expected="hi"),
        Example(id="2", input="hello", expected="hello"),
        Example(id="3", input="bye", expected="hello"),  # mismatch
    ])

    async def target(example):
        return {"output": example.input, "latency": 0.01, "tool_calls": None}

    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(evaluate(target, dataset, [exact_match, contains]))

    assert isinstance(report, EvaluationReport)
    assert len(report.results) == 3
    # exact_match: 2/3 pass
    assert report.aggregate.mean_scores["exact_match"] == pytest.approx(2 / 3)
    # contains: same as exact_match here (expected == input or not)
    assert report.aggregate.mean_scores["contains"] == pytest.approx(2 / 3)
    assert report.aggregate.count == 3
    assert report.aggregate.failure_count == 0
    # per-example: id=3 has 0/0
    by_id = {r.id: r for r in report.results}
    assert by_id["1"].scores["exact_match"] == 1.0
    assert by_id["3"].scores["exact_match"] == 0.0
    # overall is mean of per-example means
    assert report.aggregate.overall == pytest.approx((1.0 + 1.0 + 0.0) / 3)


def test_evaluate_records_target_error():
    dataset = Dataset(examples=[
        Example(id="ok", input="hi", expected="hi"),
        Example(id="boom", input="x", expected="y"),
    ])

    async def target(example):
        if example.id == "boom":
            raise RuntimeError("kaboom")
        return {"output": example.input, "latency": 0.0, "tool_calls": None}

    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(evaluate(target, dataset, [exact_match]))

    by_id = {r.id: r for r in report.results}
    assert by_id["boom"].error is not None and "kaboom" in by_id["boom"].error
    assert by_id["boom"].scores == {}
    assert report.aggregate.failure_count == 1
    # ok example still scored
    assert by_id["ok"].scores["exact_match"] == 1.0


def test_evaluate_with_tool_calls():
    dataset = Dataset(examples=[
        Example(
            id="1", input="x",
            expected=[{"name": "search"}, {"name": "calc"}],
        ),
    ])

    async def target(example):
        return {
            "output": "done",
            "latency": 0.0,
            "tool_calls": [{"name": "calc"}, {"name": "search"}],
        }

    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(
        evaluate(target, dataset, [tool_call_accuracy])
    )
    assert report.results[0].scores["tool_call_accuracy"] == 1.0


# ----------------------- llm judge ----------------------- #
class _FakeJudge:
    """Fake judge client matching the (content, _) chat() contract."""

    def __init__(self, verdict: str):
        self.verdict = verdict
        self.calls = 0

    async def chat(self, messages):
        self.calls += 1
        return self.verdict, None


def test_llm_judge_pass_and_fail():
    ex = Example(id="1", input="What is 2+2?", expected="4")

    passing = _FakeJudge("PASS")
    failing = _FakeJudge("FAIL\n(because reasons)")

    m_pass = llm_judge(passing)
    m_fail = llm_judge(failing)
    assert m_pass.is_async and m_fail.is_async

    loop = asyncio.get_event_loop()
    pred = {"output": "4", "latency": 0.0, "tool_calls": None}
    assert loop.run_until_complete(m_pass.score(ex, pred, {})) == 1.0
    assert loop.run_until_complete(m_fail.score(ex, pred, {})) == 0.0
    assert passing.calls == 1 and failing.calls == 1


def test_llm_judge_via_evaluate_uses_judge_client_context():
    """When the metric is built without a client, evaluate(judge_client=...)
    supplies it via context."""
    dataset = Dataset(examples=[
        Example(id="1", input="hi", expected="hi"),
    ])

    async def target(example):
        return {"output": example.input, "latency": 0.0, "tool_calls": None}

    judge = _FakeJudge("PASS")
    metric = llm_judge()  # no client bound

    loop = asyncio.get_event_loop()
    report = loop.run_until_complete(
        evaluate(target, dataset, [metric], judge_client=judge)
    )
    assert report.results[0].scores["llm_judge"] == 1.0
    assert judge.calls == 1


# ----------------------- dataset IO ----------------------- #
def test_dataset_jsonl_roundtrip(tmp_path: Path):
    p = tmp_path / "ds.jsonl"
    p.write_text(
        "\n".join([
            json.dumps({"id": "1", "input": "hi", "expected": "hi"}),
            json.dumps({"id": "2", "input": "bye", "expected": "bye", "metadata": {"k": "v"}}),
        ]),
        encoding="utf-8",
    )
    ds = Dataset.load_jsonl(p)
    assert len(ds) == 2
    assert ds[0].input == "hi"
    assert ds[1].metadata == {"k": "v"}
    assert [e.id for e in ds] == ["1", "2"]


def test_dataset_csv_with_json_expected(tmp_path: Path):
    p = tmp_path / "ds.csv"
    p.write_text(
        "id,input,expected,metadata\n"
        '1,hi,hi,{}\n'
        '2,call tools,"[{\\"name\\":\\"search\\"}]",{"topic":"tools"}\n',
        encoding="utf-8",
    )
    ds = Dataset.load_csv(p)
    assert len(ds) == 2
    assert ds[0].expected == "hi"
    assert ds[1].expected == [{"name": "search"}]
    assert ds[1].metadata == {"topic": "tools"}


def test_dataset_save_json(tmp_path: Path):
    ds = Dataset(examples=[Example(id="1", input="hi", expected="hi")])
    p = tmp_path / "out.json"
    ds.save_json(p)
    reloaded = Dataset.load_json(p)
    assert len(reloaded) == 1
    assert reloaded[0].expected == "hi"