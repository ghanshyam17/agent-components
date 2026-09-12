"""guardrails: input/output safety for LLM apps.

Filtering, PII redaction, prompt-injection detection, length limits and
output schema validation, composable as a pipeline.

Quick start::

    from guardrails import build_pipeline
    pipeline = build_pipeline()
    result = await pipeline.check_input("hello")
    if result.blocked:
        raise SafetyError(result.reasons)
    safe_text = result.sanitized
"""
from __future__ import annotations

from guardrails.base import Guardrail, GuardrailPipeline
from guardrails.config import GuardrailSettings, build_pipeline, get_settings
from guardrails.filters import (
    InjectionDetector,
    KeywordBlocklist,
    LengthGuard,
    OutputSchemaValidator,
    PIIRedactor,
)
from guardrails.result import GuardrailResult

__all__ = [
    "Guardrail",
    "GuardrailPipeline",
    "GuardrailResult",
    "KeywordBlocklist",
    "PIIRedactor",
    "InjectionDetector",
    "LengthGuard",
    "OutputSchemaValidator",
    "GuardrailSettings",
    "get_settings",
    "build_pipeline",
]