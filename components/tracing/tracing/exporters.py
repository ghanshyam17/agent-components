"""Span exporters: console, JSON, and an OTLP stub."""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from components_core import get_logger

from tracing.span import Span

logger = get_logger("tracing.exporters")


class Exporter(ABC):
    """Abstract span exporter."""

    @abstractmethod
    def export(self, spans: list[Span]) -> None:
        """Emit a batch of completed spans."""


class ConsoleExporter(Exporter):
    """Prints a flat summary of spans to stdout/stderr via the logger."""

    def export(self, spans: list[Span]) -> None:
        if not spans:
            return
        logger.info("tracing: %d span(s)", len(spans))
        for span in spans:
            dur = f"{span.duration * 1000:.2f}ms" if span.duration is not None else "open"
            logger.info(
                "  %s%s [%s] %s",
                "  " * _depth(span, spans),
                span.name,
                span.status,
                dur,
            )


def _depth(span: Span, all_spans: list[Span]) -> int:
    """Approximate tree depth by walking parent links."""
    by_id = {s.id: s for s in all_spans}
    depth = 0
    current = span
    while current.parent_id and current.parent_id in by_id:
        depth += 1
        current = by_id[current.parent_id]
    return depth


class JSONExporter(Exporter):
    """Writes the serialized span list to a JSON file on each export call."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def export(self, spans: list[Span]) -> None:
        payload = {"spans": [s.model_dump() for s in spans]}
        self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info("tracing: wrote %d span(s) to %s", len(spans), self.path)


class OTLPExporter(Exporter):
    """OTLP exporter stub.

    Requires the ``otel`` extra (``pip install tracing[otel]``). The actual
    OTLP wire export is intentionally not implemented here — this is just the
    interface plus a lazy import guard so the package works without the extra.
    """

    def __init__(self) -> None:
        self._exporter = None
        try:
            from opentelemetry.sdk.trace.export import BatchSpanProcessor  # type: ignore
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # type: ignore
                OTLPSpanExporter,
            )
        except ImportError as e:
            raise RuntimeError(
                "OTLPExporter requires the 'otel' extra: "
                "pip install 'tracing[otel]'"
            ) from e
        # NOTE: full OTLP wiring (TracerProvider, resource, processor) is left
        # to callers; this stub only validates the extra is available.
        self._exporter = None

    def export(self, spans: list[Span]) -> None:
        # Not fully implemented — see class docstring.
        logger.info("tracing: otlp export of %d span(s) (stub, no-op)", len(spans))