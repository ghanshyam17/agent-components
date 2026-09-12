"""Abstract guardrail + the pipeline that composes them.

A `Guardrail` is a single async check over text. A `GuardrailPipeline` holds
an ordered list of **input** guardrails and an optional list of **output**
guardrails. Input guardrails run on user/agent-bound text; output guardrails
run on model output (e.g. enforcing a JSON schema).

Pipeline semantics:

* Guardrails run in order.
* Each guardrail receives the *running sanitized text* — i.e. the output of
  the previous guardrail's `sanitized` field — so PII redaction, length
  trimming, etc. compose. A guardrail that only inspects (e.g. keyword
  blocklist) returns `sanitized` equal to what it received.
* On a block, the pipeline short-circuits: no later guardrails in the same
  direction run. The combined result carries the blocker's reasons/flags
  plus anything accumulated so far.
* Results merge: `flags` and `reasons` accumulate; `sanitized` is whatever
  the last-executed guardrail produced; `allowed` is the conjunction of all
  executed guardrails' `allowed`.
"""
from __future__ import annotations

import abc
from typing import Any

from components_core import get_logger
from guardrails.result import GuardrailResult

logger = get_logger("guardrails")


class Guardrail(abc.ABC):
    """A single safety check over text.

    Subclasses implement `check`. Implementations should be pure functions
    of their input (no shared mutable state) so they compose safely in a
    pipeline.
    """

    name: str = "guardrail"

    @abc.abstractmethod
    async def check(
        self, text: str, *, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Inspect (and possibly sanitize) `text`, returning a result."""
        ...


def _merge(into: GuardrailResult, other: GuardrailResult) -> GuardrailResult:
    """Fold `other` into `into`, carrying sanitized text forward."""
    return GuardrailResult(
        allowed=into.allowed and other.allowed,
        sanitized=other.sanitized if other.sanitized else into.sanitized,
        reasons=[*into.reasons, *other.reasons],
        flags=[*into.flags, *other.flags],
        blocked=(not (into.allowed and other.allowed)),
    )


class GuardrailPipeline:
    """An ordered set of input guardrails plus optional output guardrails.

    Use `check_input` for text entering the model (user prompts, tool output
    being fed back, etc.) and `check_output` for text the model produced
    before it reaches a caller.
    """

    def __init__(
        self,
        input_guardrails: list[Guardrail] | None = None,
        output_guardrails: list[Guardrail] | None = None,
    ) -> None:
        self.input_guardrails: list[Guardrail] = list(input_guardrails or [])
        self.output_guardrails: list[Guardrail] = list(output_guardrails or [])

    async def check_input(
        self, text: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Run input guardrails in order, short-circuiting on block."""
        return await self._run(self.input_guardrails, text, context)

    async def check_output(
        self, text: str, context: dict[str, Any] | None = None
    ) -> GuardrailResult:
        """Run output guardrails in order, short-circuiting on block."""
        return await self._run(self.output_guardrails, text, context)

    @staticmethod
    async def _run(
        guardrails: list[Guardrail],
        text: str,
        context: dict[str, Any] | None,
    ) -> GuardrailResult:
        combined = GuardrailResult.allow(text)
        current = text
        for g in guardrails:
            result = await g.check(current, context=context)
            combined = _merge(combined, result)
            current = result.sanitized if result.sanitized else current
            if result.blocked:
                logger.info(
                    "guardrail %s blocked input: flags=%s", g.name, result.flags
                )
                break
        return combined