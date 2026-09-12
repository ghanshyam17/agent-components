"""Instrumentation helpers: async/sync context managers for model and tool calls."""
from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator

from tracing.cost import CostAccountant, Usage
from tracing.span import Span
from tracing.tracer import Tracer


@contextlib.asynccontextmanager
async def trace_call(
    tracer: Tracer, name: str, attributes: dict[str, Any] | None = None
) -> AsyncIterator[Span]:
    """Async context manager that starts a span, records exceptions, and ends.

    Exceptions are recorded on the span (``set_error``) and re-raised::

        async with trace_call(tracer, "model.chat", {"model": "gpt-4o"}) as span:
            ...
    """
    span = tracer.start_span(name, attributes=attributes)
    try:
        yield span
    except BaseException as exc:
        span.set_error(exc)
        raise
    finally:
        span.__exit__(None, None, None)


def record_usage(tracer: Tracer, usage: Usage) -> float:
    """Attach ``usage`` (and its cost) to the current span, and to a
    ``CostAccountant`` if one is attached to the tracer (``tracer.cost``).

    Returns the computed cost in USD (0.0 if no cost table resolves the model).
    """
    span = tracer.current_span()
    cost = 0.0
    if isinstance(tracer.cost, CostAccountant):
        cost = tracer.cost.add(
            usage.model, usage.prompt_tokens, usage.completion_tokens
        )
        if span is not None:
            span.set_attribute("usage.cost_usd", cost)
    if span is not None:
        span.set_attribute("usage.prompt_tokens", usage.prompt_tokens)
        span.set_attribute("usage.completion_tokens", usage.completion_tokens)
        span.set_attribute("usage.model", usage.model)
    return cost


@contextlib.contextmanager
def trace_tool_call(
    tracer: Tracer, tool_name: str, arguments: dict[str, Any] | None = None
) -> Any:
    """Sync context manager wrapping a tool call in a span.

    Records the tool name and arguments as attributes; exceptions are recorded
    on the span and re-raised::

        with trace_tool_call(tracer, "search", {"q": "x"}):
            result = search(...)
    """
    attrs = {"tool.name": tool_name}
    if arguments:
        attrs["tool.arguments"] = dict(arguments)
    span = tracer.start_span(f"tool.{tool_name}", attributes=attrs)
    try:
        yield span
    except BaseException as exc:
        span.set_error(exc)
        raise
    finally:
        span.__exit__(None, None, None)