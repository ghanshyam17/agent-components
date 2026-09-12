"""Guardrail settings — env-driven via pydantic-settings (`GUARD_` prefix)."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from guardrails.base import GuardrailPipeline
from guardrails.filters import (
    InjectionDetector,
    KeywordBlocklist,
    LengthGuard,
    PIIRedactor,
)


class GuardrailSettings(BaseSettings):
    """Env knobs for the default pipeline.

    Build a custom pipeline by constructing `GuardrailPipeline` directly; these
    settings only drive `build_pipeline()`.
    """

    model_config = SettingsConfigDict(
        env_prefix="GUARD_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    max_chars: int = Field(default=8000, description="input character budget")
    block_injection: bool = Field(
        default=True, description="block on prompt-injection cues (vs. flag only)"
    )
    redact_pii: bool = Field(default=True, description="run the PII redactor")
    # Comma-separated list of banned keywords, e.g. "secret,password".
    blocked_keywords: list[str] = Field(
        default_factory=list,
        description="comma-separated list of banned keywords",
    )


@lru_cache
def get_settings() -> GuardrailSettings:
    return GuardrailSettings()


def _parse_keywords(raw: str) -> list[str]:
    return [k.strip() for k in raw.split(",") if k.strip()]


def build_pipeline(
    settings: GuardrailSettings | None = None,
) -> GuardrailPipeline:
    """Assemble a `GuardrailPipeline` from settings.

    Input guardrails (in order): length, injection, pii (if enabled),
    keywords (if any). Output guardrails: none by default — add an
    `OutputSchemaValidator` explicitly when you have a schema to enforce.
    """
    s = settings or get_settings()

    # Allow a plain comma-separated env string to populate `blocked_keywords`
    # without a custom validator.
    keywords = s.blocked_keywords
    if len(keywords) == 1 and "," in keywords[0]:
        keywords = _parse_keywords(keywords[0])

    input_guardrails: list = [LengthGuard(max_chars=s.max_chars)]
    input_guardrails.append(InjectionDetector(block_on_match=s.block_injection))
    if s.redact_pii:
        input_guardrails.append(PIIRedactor())
    if keywords:
        input_guardrails.append(KeywordBlocklist(keywords=keywords))

    return GuardrailPipeline(input_guardrails=input_guardrails, output_guardrails=[])