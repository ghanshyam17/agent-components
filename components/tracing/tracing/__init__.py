"""tracing: structured tracing + token/cost accounting for AI apps.

In-memory span tree by default, with an exporter interface (console / JSON /
optional OTLP) and auto-hooks for model and tool calls via async/sync context
managers.

Quick start::

    from tracing import get_tracer, trace_call, record_usage, Usage

    tracer = get_tracer()
    async with trace_call(tracer, "model.chat", {"model": "gpt-4o"}) as span:
        record_usage(tracer, Usage(prompt_tokens=120, completion_tokens=80, model="gpt-4o"))
"""
from __future__ import annotations

from tracing.config import TracingSettings, build_exporter, get_tracer
from tracing.cost import CostAccountant, CostTable, Usage
from tracing.exporters import ConsoleExporter, Exporter, JSONExporter, OTLPExporter
from tracing.instrument import record_usage, trace_call, trace_tool_call
from tracing.span import Span
from tracing.tracer import Tracer

__all__ = [
    "Span",
    "Tracer",
    "Usage",
    "CostTable",
    "CostAccountant",
    "Exporter",
    "ConsoleExporter",
    "JSONExporter",
    "OTLPExporter",
    "trace_call",
    "record_usage",
    "trace_tool_call",
    "TracingSettings",
    "build_exporter",
    "get_tracer",
]