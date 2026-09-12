"""Tracer: holds a span tree and tracks the current span via contextvars."""
from __future__ import annotations

import contextvars
from typing import Any

from tracing.span import Span

# Per-context current span id. contextvars are propagated by the asyncio
# event loop, so concurrent tasks each get their own current-span chain.
_CURRENT_SPAN_ID: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "tracing_current_span_id", default=None
)


class Tracer:
    """Records completed spans and tracks the current span via ``contextvars``.

    Use ``start_span`` as a context manager so the parent is restored on exit::

        with tracer.start_span("outer") as outer:
            with tracer.start_span("inner") as inner:
                ...  # inner.parent_id == outer.id

    Optionally attach a :class:`~tracing.cost.CostAccountant` as ``tracer.cost``;
    :func:`~tracing.instrument.record_usage` will credit it in addition to the
    current span's attributes.
    """

    def __init__(self) -> None:
        self._spans: list[Span] = []
        # Open spans not yet recorded (used by current_span).
        self._open: dict[str, Span] = {}
        # Optional cost accountant attached by config/build helpers.
        self.cost: Any | None = None

    # ---- span lifecycle ---------------------------------------------------
    def start_span(self, name: str, attributes: dict[str, Any] | None = None) -> Span:
        """Start a new span, linking it to the current span as its parent.

        Sets the new span as the current one. Prefer the context-manager form
        (``with tracer.start_span(...) as span:``) so the parent is restored
        on exit; otherwise call ``span.end()`` and then ``_on_span_exit(span)``
        (or simply ``span.__exit__(None, None, None)``) to record it.
        """
        parent_id = _CURRENT_SPAN_ID.get()
        span = Span(name=name, parent_id=parent_id)
        span._tracer = self
        if attributes:
            for k, v in attributes.items():
                span.set_attribute(k, v)
        _CURRENT_SPAN_ID.set(span.id)
        self._open[span.id] = span
        return span

    def _on_span_exit(self, span: Span) -> None:
        """Record a finished span and restore its parent as current."""
        span.end()
        # Only restore if we're still the current span (don't clobber a
        # sibling that someone forgot to exit cleanly).
        if _CURRENT_SPAN_ID.get() == span.id:
            _CURRENT_SPAN_ID.set(span.parent_id)
        self._spans.append(span)
        self._open.pop(span.id, None)

    # ---- queries ----------------------------------------------------------
    def current_span(self) -> Span | None:
        """Return the currently-open span, if any (by id lookup)."""
        sid = _CURRENT_SPAN_ID.get()
        if sid is None:
            return None
        return self._open.get(sid)

    def spans(self) -> list[Span]:
        """All completed (recorded) spans, in completion order."""
        return list(self._spans)

    def reset(self) -> None:
        """Clear recorded spans, the current span, and any cost totals."""
        self._spans.clear()
        self._open.clear()
        _CURRENT_SPAN_ID.set(None)
        if self.cost is not None:
            self.cost.reset()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the recorded spans as a list under ``"spans"``."""
        return {"spans": [s.model_dump() for s in self._spans]}