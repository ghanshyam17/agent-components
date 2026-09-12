"""tracing tests: span tree, errors, cost accounting, exporters."""
from __future__ import annotations

import asyncio
import json
import os
import time

import pytest

from tracing import (
    ConsoleExporter,
    CostAccountant,
    CostTable,
    JSONExporter,
    Span,
    Tracer,
    Usage,
    get_tracer,
    record_usage,
    trace_call,
    trace_tool_call,
)


# ---------------- spans ----------------
def test_nested_spans_have_correct_parent_ids():
    tracer = Tracer()
    with tracer.start_span("outer") as outer:
        with tracer.start_span("middle") as middle:
            with tracer.start_span("inner") as inner:
                assert inner.parent_id == middle.id
                assert tracer.current_span() is inner
        assert tracer.current_span() is outer
    assert tracer.current_span() is None

    spans = {s.name: s for s in tracer.spans()}
    assert spans["middle"].parent_id == spans["outer"].id
    assert spans["inner"].parent_id == spans["middle"].id
    assert spans["outer"].parent_id is None


def test_span_duration_is_positive():
    tracer = Tracer()
    with tracer.start_span("work") as span:
        time.sleep(0.005)
    assert span.duration is not None
    assert span.duration > 0
    # span appears in recorded list
    assert any(s.name == "work" for s in tracer.spans())


def test_span_end_is_idempotent():
    tracer = Tracer()
    with tracer.start_span("x") as span:
        pass
    first = span.duration
    span.end()
    assert span.duration == first


def test_error_captured_and_reraised():
    tracer = Tracer()
    with pytest.raises(ValueError):
        with tracer.start_span("risky") as span:
            raise ValueError("boom")

    assert span.status == "error"
    assert any(ev["name"] == "exception" for ev in span.events)
    exc_event = next(ev for ev in span.events if ev["name"] == "exception")
    assert exc_event["attributes"]["type"] == "ValueError"
    assert exc_event["attributes"]["message"] == "boom"


def test_reset_clears_spans():
    tracer = Tracer()
    with tracer.start_span("a"):
        pass
    assert tracer.spans()
    tracer.reset()
    assert tracer.spans() == []
    assert tracer.current_span() is None


def test_to_dict_serializes_spans():
    tracer = Tracer()
    with tracer.start_span("x", {"kind": "root"}):
        pass
    data = tracer.to_dict()
    assert "spans" in data
    assert len(data["spans"]) == 1
    assert data["spans"][0]["name"] == "x"
    assert data["spans"][0]["attributes"]["kind"] == "root"


# ---------------- async trace_call ----------------
async def test_trace_call_records_span():
    tracer = Tracer()

    async with trace_call(tracer, "model.chat", {"model": "gpt-4o"}) as span:
        assert span.name == "model.chat"
        assert tracer.current_span() is span

    assert tracer.current_span() is None
    assert tracer.spans()[0].attributes["model"] == "gpt-4o"


async def test_trace_call_captures_exception():
    tracer = Tracer()
    with pytest.raises(RuntimeError):
        async with trace_call(tracer, "failing"):
            raise RuntimeError("nope")
    span = tracer.spans()[0]
    assert span.status == "error"
    assert span.events[0]["attributes"]["type"] == "RuntimeError"


# ---------------- cost ----------------
def test_cost_table_gpt4o_expected_cost():
    table = CostTable()
    # gpt-4o: 2.5/1k prompt, 10.0/1k completion
    usage = Usage(prompt_tokens=1000, completion_tokens=500, model="gpt-4o")
    cost = table.account(usage)
    # 1.0 * 2.5 + 0.5 * 10.0 = 2.5 + 5.0 = 7.5
    assert cost == pytest.approx(7.5)


def test_cost_table_unknown_model_is_zero():
    table = CostTable()
    usage = Usage(prompt_tokens=1000, completion_tokens=500, model="mystery-model")
    assert table.account(usage) == 0.0


def test_cost_table_free_models_seeded_zero():
    table = CostTable()
    for m in ("qwen2.5-1.5b-instruct", "qwen2.5-32b-instruct"):
        usage = Usage(prompt_tokens=10_000, completion_tokens=10_000, model=m)
        assert table.account(usage) == 0.0


def test_cost_accountant_totals_by_model():
    acc = CostAccountant()
    acc.add("gpt-4o", prompt_tokens=1000, completion_tokens=500)  # 7.5
    acc.add("gpt-4o", prompt_tokens=1000, completion_tokens=500)  # 7.5
    acc.add("gpt-4o-mini", prompt_tokens=1000, completion_tokens=1000)  # 0.15 + 0.6 = 0.75

    assert acc.total_cost() == pytest.approx(7.5 + 7.5 + 0.75)
    by = acc.by_model()
    assert by["gpt-4o"]["cost"] == pytest.approx(15.0)
    assert by["gpt-4o"]["prompt_tokens"] == 2000
    assert by["gpt-4o-mini"]["completion_tokens"] == 1000


def test_record_usage_attaches_to_span_and_accountant():
    tracer = Tracer()
    tracer.cost = CostAccountant()
    with tracer.start_span("model.chat", {"model": "gpt-4o"}):
        cost = record_usage(tracer, Usage(prompt_tokens=1000, completion_tokens=500, model="gpt-4o"))
    assert cost == pytest.approx(7.5)
    span = tracer.spans()[0]
    assert span.attributes["usage.prompt_tokens"] == 1000
    assert span.attributes["usage.completion_tokens"] == 500
    assert span.attributes["usage.model"] == "gpt-4o"
    assert span.attributes["usage.cost_usd"] == pytest.approx(7.5)
    assert tracer.cost.total_cost() == pytest.approx(7.5)


def test_record_usage_without_accountant_only_attaches_to_span():
    tracer = Tracer()
    with tracer.start_span("model.chat"):
        cost = record_usage(tracer, Usage(prompt_tokens=1000, completion_tokens=500, model="gpt-4o"))
    assert cost == 0.0
    span = tracer.spans()[0]
    assert span.attributes["usage.prompt_tokens"] == 1000
    assert "usage.cost_usd" not in span.attributes


# ---------------- tool call ----------------
def test_trace_tool_call_records_attributes():
    tracer = Tracer()
    with trace_tool_call(tracer, "search", {"q": "rag"}):
        pass
    span = tracer.spans()[0]
    assert span.name == "tool.search"
    assert span.attributes["tool.name"] == "search"
    assert span.attributes["tool.arguments"] == {"q": "rag"}
    assert span.status == "ok"


def test_trace_tool_call_captures_exception():
    tracer = Tracer()
    with pytest.raises(KeyError):
        with trace_tool_call(tracer, "get", {"k": "x"}):
            raise KeyError("x")
    span = tracer.spans()[0]
    assert span.status == "error"


# ---------------- exporters ----------------
def test_json_exporter_writes_file(tmp_path):
    tracer = Tracer()
    with tracer.start_span("outer"):
        with tracer.start_span("inner"):
            pass
    path = tmp_path / "spans.json"
    JSONExporter(path).export(tracer.spans())
    data = json.loads(path.read_text(encoding="utf-8"))
    assert len(data["spans"]) == 2
    names = {s["name"] for s in data["spans"]}
    assert names == {"outer", "inner"}


def test_console_exporter_does_not_crash():
    tracer = Tracer()
    with tracer.start_span("a", {"k": "v"}):
        pass
    # Empty list and a populated list should both be fine.
    ConsoleExporter().export([])
    ConsoleExporter().export(tracer.spans())


# ---------------- config / get_tracer ----------------
def test_get_tracer_returns_process_global_tracer():
    # Don't mutate the real global; just confirm identity + shape.
    t1 = get_tracer()
    t2 = get_tracer()
    assert t1 is t2
    assert isinstance(t1, Tracer)
    assert isinstance(t1.cost, CostAccountant)


def test_tracing_settings_defaults(monkeypatch):
    # Ensure env vars don't leak in from the shell.
    for var in ("TRACE_EXPORTER", "TRACE_JSON_PATH"):
        monkeypatch.delenv(var, raising=False)
    from tracing.config import TracingSettings

    s = TracingSettings()
    assert s.exporter == "console"
    assert isinstance(s.json_path, str)