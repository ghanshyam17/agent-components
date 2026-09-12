"""The result every guardrail returns.

A `GuardrailResult` carries both a verdict (`allowed`/`blocked`) and the
possibly-sanitized text, plus structured `flags` and human-readable `reasons`
that a pipeline merges across stages. Helpers `allow()` and `block()` cover
the two common cases; sanitizing guardrails (PII redaction) build a result
that is `allowed=True` but carries modified `sanitized` text.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class GuardrailResult(BaseModel):
    """Outcome of running one guardrail over a piece of text.

    Attributes:
        allowed: Whether the text may proceed to the next stage. `False` means
            the guardrail found something disqualifying (a block).
        sanitized: The text to carry forward. Sanitizing guardrails (e.g. PII
            redaction) replace matched spans here; blocking guardrails leave
            the input text in place.
        reasons: Human-readable explanations, one per issue found.
        flags: Machine-readable tags summarizing what was detected
            (e.g. `"pii"`, `"injection"`, `"blocked_keyword"`, `"too_long"`,
            `"schema"`).
        blocked: Convenience mirror of `not allowed` for call sites that read
            more naturally as "was this blocked?".
    """

    allowed: bool = True
    sanitized: str = ""
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    blocked: bool = False

    @staticmethod
    def allow(text: str) -> "GuardrailResult":
        """Pass `text` through unchanged."""
        return GuardrailResult(allowed=True, sanitized=text)

    @staticmethod
    def block(reason: str, text: str, *, flag: str | None = None) -> "GuardrailResult":
        """Block `text` for `reason`, optionally tagging it with `flag`."""
        flags = [flag] if flag else []
        return GuardrailResult(
            allowed=False,
            sanitized=text,
            reasons=[reason],
            flags=flags,
            blocked=True,
        )