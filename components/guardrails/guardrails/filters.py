"""Concrete guardrails.

Each class implements `Guardrail.check`:

* `KeywordBlocklist` — block if any banned keyword is present.
* `PIIRedactor` — replace common PII patterns with `[REDACTED:<kind>]`; always
  allowed (it sanitizes rather than blocks).
* `InjectionDetector` — heuristic regex set for prompt-injection / jailbreak
  cues; blocks on match (configurable).
* `LengthGuard` — block input above a character budget.
* `OutputSchemaValidator` — validate model output as JSON against a JSON
  Schema (output guardrail; requires the `jsonschema` extra).
"""
from __future__ import annotations

import json
import re
from typing import Any

from guardrails.base import Guardrail
from guardrails.result import GuardrailResult

# ----------------------------- defaults ----------------------------- #

# Default PII regexes. `kind` becomes the `[REDACTED:<kind>]` tag.
_DEFAULT_PII_PATTERNS: list[tuple[str, str]] = [
    ("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # North-American-ish phone numbers: (123) 456-7890, 123-456-7890, +1 123 456 7890
    ("phone", r"(?:\+?\d{1,2}[\s.-]?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b"),
    # SSN-ish: 123-45-6789
    ("ssn", r"\b\d{3}-\d{2}-\d{4}\b"),
    # Credit-card-ish: 13-16 digits, optionally space/dash separated
    ("credit_card", r"\b(?:\d[ -]*?){13,16}\b"),
]

# Heuristic prompt-injection / jailbreak cues. Case-insensitive by default.
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    ("ignore_previous", r"ignore\s+(?:all\s+)?previous\s+instructions"),
    ("disregard_above", r"disregard\s+(?:the\s+)?above"),
    ("you_are_now", r"you\s+are\s+now\s+"),
    ("system_prompt", r"system\s+prompt\s*:?"),
    ("special_token", r"<\|[^|]*\|>"),
    ("role_play", r"(?:pretend|act\s+as|role[-\s]?play)\s+(?:you\s+are\s+)?(?:a|an)\s+"),
    ("reveal_rules", r"(?:reveal|show|print|repeat)\s+(?:your\s+)?(?:system\s+)?(?:rules?|instructions?|prompt)"),
    ("override", r"override\s+(?:your\s+)?(?:safety|content)\s+(?:rules?|policy|filter)"),
]


# ----------------------------- guardrails ----------------------------- #

class KeywordBlocklist(Guardrail):
    """Block if any banned keyword appears in the text."""

    name = "keyword_blocklist"

    def __init__(
        self, keywords: list[str], case_sensitive: bool = False
    ) -> None:
        self.keywords = list(keywords)
        self.case_sensitive = case_sensitive

    async def check(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        hay = text if self.case_sensitive else text.lower()
        for kw in self.keywords:
            needle = kw if self.case_sensitive else kw.lower()
            if needle and needle in hay:
                return GuardrailResult.block(
                    f"blocked keyword present: {kw!r}",
                    text,
                    flag="blocked_keyword",
                )
        return GuardrailResult.allow(text)


class PIIRedactor(Guardrail):
    """Replace PII patterns with `[REDACTED:<kind>]`. Always allowed."""

    name = "pii_redactor"

    def __init__(self, patterns: list[str] | None = None) -> None:
        # `patterns` lets callers override the default regex set wholesale.
        if patterns is not None:
            self._compiled: list[tuple[str, re.Pattern[str]]] = [
                ("custom", re.compile(p)) for p in patterns
            ]
        else:
            self._compiled = [
                (kind, re.compile(pat)) for kind, pat in _DEFAULT_PII_PATTERNS
            ]

    async def check(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        sanitized = text
        flags: list[str] = []
        reasons: list[str] = []
        for kind, pat in self._compiled:
            new, n = pat.subn(f"[REDACTED:{kind}]", sanitized)
            if n > 0:
                sanitized = new
                flags.append("pii")
                reasons.append(f"redacted {n} {kind} match(es)")
        if not flags:
            return GuardrailResult.allow(text)
        return GuardrailResult(
            allowed=True,
            sanitized=sanitized,
            reasons=reasons,
            flags=flags,
            blocked=False,
        )


class InjectionDetector(Guardrail):
    """Heuristic prompt-injection / jailbreak detector."""

    name = "injection_detector"

    def __init__(self, block_on_match: bool = True) -> None:
        self.block_on_match = block_on_match
        self._compiled = [
            (tag, re.compile(pat, re.IGNORECASE))
            for tag, pat in _INJECTION_PATTERNS
        ]

    async def check(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        matched: list[tuple[str, str]] = []
        for tag, pat in self._compiled:
            m = pat.search(text)
            if m:
                matched.append((tag, m.group(0)))
        if not matched:
            return GuardrailResult.allow(text)
        reasons = [f"injection cue ({tag}): {snippet!r}" for tag, snippet in matched]
        flags = ["injection"] * len(matched)
        if self.block_on_match:
            return GuardrailResult(
                allowed=False,
                sanitized=text,
                reasons=reasons,
                flags=flags,
                blocked=True,
            )
        # Reporting only: allow through but surface the flags.
        return GuardrailResult(
            allowed=True,
            sanitized=text,
            reasons=reasons,
            flags=flags,
            blocked=False,
        )


class LengthGuard(Guardrail):
    """Block input above a character budget."""

    name = "length_guard"

    def __init__(self, max_chars: int = 8000) -> None:
        self.max_chars = max_chars

    async def check(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        if len(text) > self.max_chars:
            return GuardrailResult.block(
                f"input too long: {len(text)} > {self.max_chars} chars",
                text,
                flag="too_long",
            )
        return GuardrailResult.allow(text)


class OutputSchemaValidator(Guardrail):
    """Validate model output as JSON against a JSON Schema.

    Output guardrail. Requires the `jsonschema` extra; the import is lazy so
    the rest of the package works without it.
    """

    name = "output_schema_validator"

    def __init__(self, schema: dict[str, Any]) -> None:
        self.schema = schema

    async def check(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return GuardrailResult.block(
                f"output is not valid JSON: {e.msg}", text, flag="schema"
            )
        try:
            import jsonschema  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "OutputSchemaValidator requires the 'jsonschema' extra: "
                "pip install guardrails[jsonschema]"
            ) from e
        try:
            jsonschema.validate(instance=data, schema=self.schema)
        except jsonschema.ValidationError as e:  # type: ignore
            return GuardrailResult.block(
                f"output failed schema validation: {e.message}", text, flag="schema"
            )
        return GuardrailResult.allow(text)