"""Tracing settings — env-driven via pydantic-settings (`TRACE_` prefix)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from tracing.cost import CostAccountant
from tracing.exporters import ConsoleExporter, Exporter, JSONExporter, OTLPExporter
from tracing.tracer import Tracer


class TracingSettings(BaseSettings):
    """Env knobs for the tracing component."""

    model_config = SettingsConfigDict(
        env_prefix="TRACE_", env_file=".env", env_file_encoding="utf-8", extra="ignore",
    )

    exporter: str = Field(
        default="console",
        description="console | json | otlp | none",
    )
    json_path: str = Field(
        default="tracing-spans.json",
        description="Path written to by the json exporter.",
    )


def build_exporter(settings: TracingSettings) -> Exporter | None:
    """Construct the configured exporter (or ``None`` for ``none``)."""
    kind = settings.exporter.lower()
    if kind == "none":
        return None
    if kind == "console":
        return ConsoleExporter()
    if kind == "json":
        return JSONExporter(settings.json_path)
    if kind == "otlp":
        return OTLPExporter()  # raises if the 'otel' extra is missing
    raise ValueError(f"unknown TRACE_EXPORTER: {settings.exporter!r}")


@lru_cache
def get_tracer() -> Tracer:
    """Build and return the process-global :class:`Tracer`.

    Attaches a :class:`CostAccountant` and the configured :class:`Exporter`
    (both optional) to the returned tracer.
    """
    settings = TracingSettings()
    tracer = Tracer()
    tracer.cost = CostAccountant()
    # Exporter is attached lazily — callers invoke it explicitly via
    # ``tracer.spans()`` + ``exporter.export(...)`` (see README). We stash it on
    # the tracer so callers can reach it without re-reading settings.
    tracer.exporter = build_exporter(settings)  # type: ignore[attr-defined]
    return tracer