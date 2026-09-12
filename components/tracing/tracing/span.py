"""Span: a single timed unit of work in a tracing tree."""
from __future__ import annotations

import time
import uuid
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, PrivateAttr

if TYPE_CHECKING:
    from tracing.tracer import Tracer


class Span(BaseModel):
    """A single span: a named, timed unit of work linked into a span tree.

    Times are recorded with ``time.monotonic()`` (not wall-clock) so durations
    are immune to system clock changes. Span ids are 16-char hex prefixes of
    ``uuid4``.

    ``Span`` is a context manager — on exit it ends the span, records it with
    the owning :class:`~tracing.tracer.Tracer`, restores the parent as current,
    and (on exception) marks itself errored and re-raises::

        with tracer.start_span("x") as span:
            ...
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    name: str
    parent_id: str | None = None
    start_time: float = Field(default_factory=time.monotonic)
    end_time: float | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[dict[str, Any]] = Field(default_factory=list)
    status: str = "ok"  # "ok" | "error"

    model_config = {"arbitrary_types_allowed": True}

    # Back-ref to the Tracer that created this span. Private — excluded from
    # ``model_dump()`` and equality — set by ``Tracer.start_span``.
    _tracer: "Tracer | None" = PrivateAttr(default=None)

    # ---- mutation helpers -------------------------------------------------
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a single attribute on this span."""
        self.attributes[key] = value

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Record a timestamped event on this span."""
        self.events.append(
            {
                "name": name,
                "timestamp": time.monotonic(),
                "attributes": dict(attributes) if attributes else {},
            }
        )

    def set_error(self, exc: BaseException) -> None:
        """Mark this span as errored and record the exception as an event."""
        self.status = "error"
        self.add_event(
            "exception",
            {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        )

    def end(self) -> None:
        """End this span if it hasn't been ended already."""
        if self.end_time is None:
            self.end_time = time.monotonic()

    # ---- queries ----------------------------------------------------------
    @property
    def duration(self) -> float | None:
        """Elapsed seconds, or ``None`` if the span is still open."""
        if self.end_time is None:
            return None
        return self.end_time - self.start_time

    # ---- context-manager protocol ----------------------------------------
    def __enter__(self) -> "Span":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        # Let exceptions propagate; we just record before re-raising.
        if exc is not None and self.status == "ok":
            self.set_error(exc)
        if self._tracer is not None:
            self._tracer._on_span_exit(self)
        return False  # do not suppress