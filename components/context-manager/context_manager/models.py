"""Data models for the context-manager component.

The central idea: **a context window is a budget, not a buffer.** Something must
decide what gets in, and that decision must be inspectable after the fact. Every
type here exists to make that decision legible — what was included, what was
left out, and why.
"""
from __future__ import annotations

import hashlib
import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# --------------------------------------------------------------------------- #
# Roles and priority
# --------------------------------------------------------------------------- #
class ContextRole(str, Enum):
    """Who said this, in the vocabulary every LLM API understands."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"
    MEMORY = "memory"
    RETRIEVAL = "retrieval"

    def to_wire(self) -> str:
        """Map to an OpenAI-compatible role.

        ``memory`` and ``retrieval`` are things *we* inject, not things the
        conversation said. They arrive at the model as user- or system-authored
        text; pretending a retrieved document is an assistant turn would be a
        lie the model then reasons from.
        """
        if self in (ContextRole.MEMORY, ContextRole.RETRIEVAL):
            return "user"
        return self.value


class SegmentPriority(int, Enum):
    """How hard the assembler should fight to keep a segment.

    These are *relative* weights used to order eviction, not token counts. The
    numbers are spaced so intermediate values stay expressible.
    """

    CRITICAL = 100   # system prompt, safety rules, the task itself: never dropped
    HIGH = 75        # pinned facts, tool schemas the run depends on
    NORMAL = 50      # ordinary conversation history
    LOW = 25         # retrieved documents, verbose tool output
    DISPOSABLE = 0   # nice to have; first to go


# --------------------------------------------------------------------------- #
# Segments
# --------------------------------------------------------------------------- #
class ContextSegment(BaseModel):
    """One indivisible block of context, before any budget decision.

    A segment is the unit the assembler reasons about: it is kept whole, dropped
    whole, or handed to a compactor that rewrites *it*. Splitting a document
    mid-sentence to fit a budget is a compactor's job, not the assembler's.
    """

    id: str = Field(default_factory=lambda: f"seg_{uuid.uuid4().hex[:12]}")
    role: ContextRole = ContextRole.USER
    content: str
    #: Higher survives longer. See :class:`SegmentPriority`.
    priority: int = SegmentPriority.NORMAL
    #: Pinned segments bypass eviction and compaction entirely.
    pinned: bool = False
    #: Which subsystem this came from ('history', 'retrieval', 'tool:search').
    #: Used for per-source budget accounting and for explaining drops.
    source: str = "history"
    #: Set False for content that must not be rewritten (tool schemas, quotes).
    compactable: bool = True
    #: Position in the original stream. The assembler emits in sequence order so
    #: a user/assistant conversation stays chronological; system and pinned
    #: memory segments are hoisted to the front by role, not by this number.
    sequence: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("priority")
    @classmethod
    def _coerce_priority(cls, v: int) -> int:
        # Accept the enum or a raw int, and clamp into range.
        return max(0, min(100, int(v)))

    def fingerprint(self) -> str:
        """Stable identity for a segment's *content*, not its id.

        Used to detect a re-inserted duplicate: the same retrieved chunk
        arriving twice wastes budget twice.
        """
        h = hashlib.sha256(f"{self.role.value}\x00{self.content}".encode())
        return h.hexdigest()[:16]

    def is_protected(self) -> bool:
        return self.pinned or self.priority >= SegmentPriority.CRITICAL


class ContextMessage(BaseModel):
    """An emitted message: what actually goes to the model."""

    role: str
    content: str
    tokens: int = 0
    #: The segment this came from, when it was not synthesised by a compactor.
    segment_id: str | None = None
    #: True when a compactor rewrote this content.
    compacted: bool = False

    def to_openai(self) -> dict[str, str]:
        return {"role": self.role, "content": self.content}

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "tokens": self.tokens,
            "segment_id": self.segment_id,
            "compacted": self.compacted,
        }


# --------------------------------------------------------------------------- #
# Budget
# --------------------------------------------------------------------------- #
class ContextBudget(BaseModel):
    """A token allowance, split into output reserve and input allowance.

    Reserve the output *first*. A prompt that consumes the entire window leaves
    no room to answer, so the request fails at the API boundary — the single
    most common way a context-management bug reaches production.
    """

    total_tokens: int = 8192
    #: Tokens held back for the model's reply.
    reserve_output: int = 1024
    #: Optional caps per source, as a fraction of the input allowance.
    source_fractions: dict[str, float] = Field(default_factory=dict)
    #: Hard ceiling on any single segment, as a fraction of the input allowance.
    #: Stops one enormous tool result from monopolising the window.
    max_segment_fraction: float = 0.5

    @field_validator("total_tokens", "reserve_output")
    @classmethod
    def _positive(cls, v: int) -> int:
        if v < 0:
            raise ValueError("token counts must be >= 0")
        return int(v)

    @field_validator("source_fractions", "max_segment_fraction")
    @classmethod
    def _fraction(cls, v: Any) -> Any:
        vals = v.values() if isinstance(v, dict) else [v]
        for x in vals:
            if not 0.0 < float(x) <= 1.0:
                raise ValueError("fractions must be in (0, 1]")
        return v

    @property
    def input_allowance(self) -> int:
        return max(0, self.total_tokens - self.reserve_output)

    def source_cap(self, source: str) -> int | None:
        frac = self.source_fractions.get(source)
        return int(self.input_allowance * frac) if frac else None

    def max_segment_tokens(self) -> int:
        return max(1, int(self.input_allowance * self.max_segment_fraction))


# --------------------------------------------------------------------------- #
# Outcomes
# --------------------------------------------------------------------------- #
class Disposition(str, Enum):
    """What happened to a segment, recorded per segment."""

    KEPT = "kept"
    COMPACTED = "compacted"
    DROPPED = "dropped"
    DEDUPLICATED = "deduplicated"


class DropReason(str, Enum):
    OVER_BUDGET = "over_budget"
    SOURCE_CAP = "source_cap"
    DUPLICATE = "duplicate"
    SUPERSEDED = "superseded"


class SegmentOutcome(BaseModel):
    """The audit record for one segment's journey through assembly."""

    segment_id: str
    source: str
    disposition: Disposition
    tokens_before: int = 0
    tokens_after: int = 0
    reason: DropReason | None = None
    detail: str = ""

    @property
    def tokens_saved(self) -> int:
        return max(0, self.tokens_before - self.tokens_after)

    def to_dict(self) -> dict[str, Any]:
        return {
            "segment_id": self.segment_id,
            "source": self.source,
            "disposition": self.disposition.value,
            "tokens_before": self.tokens_before,
            "tokens_after": self.tokens_after,
            "tokens_saved": self.tokens_saved,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
        }


class ContextPackage(BaseModel):
    """The assembled result: messages plus the accounting behind them."""

    messages: list[ContextMessage] = Field(default_factory=list)
    outcomes: list[SegmentOutcome] = Field(default_factory=list)
    budget: ContextBudget = Field(default_factory=ContextBudget)
    #: Sum of emitted message tokens.
    tokens_used: int = 0
    #: Segments that could not be included; drives monitoring.
    dropped: int = 0
    compacted: int = 0
    #: True when protected content alone exceeded the allowance — the one
    #: situation a caller must not ignore, because the request is now unsound.
    over_budget: bool = False
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0

    @property
    def headroom(self) -> int:
        """Input tokens still unused. Spend these on better retrieval."""
        return max(0, self.budget.input_allowance - self.tokens_used)

    @property
    def utilisation(self) -> float:
        allowance = self.budget.input_allowance
        return (self.tokens_used / allowance) if allowance else 0.0

    def sources_used(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for m in self.messages:
            key = next(
                (o.source for o in self.outcomes if o.segment_id == m.segment_id),
                "synthesised",
            )
            out[key] = out.get(key, 0) + m.tokens
        return out

    def explain(self) -> str:
        """A human-readable account of what was included and dropped.

        Exists because "why did the model not know X" is answerable only if the
        assembly decision was recorded at the time.
        """
        lines = [
            f"ContextPackage: {len(self.messages)} messages, "
            f"{self.tokens_used}/{self.budget.input_allowance} input tokens "
            f"({self.utilisation:.0%} of allowance, "
            f"{self.budget.reserve_output} reserved for output)"
        ]
        for o in self.outcomes:
            if o.disposition is Disposition.KEPT:
                continue
            lines.append(
                f"  {o.disposition.value:12s} {o.source:20s} "
                f"{o.tokens_before}->{o.tokens_after} tokens"
                + (f"  [{o.reason.value}: {o.detail}]" if o.reason else "")
            )
        for w in self.warnings:
            lines.append(f"  ! {w}")
        return "\n".join(lines)

    def to_openai_messages(self) -> list[dict[str, str]]:
        return [m.to_openai() for m in self.messages]

    def to_dict(self) -> dict[str, Any]:
        return {
            "messages": [m.to_dict() for m in self.messages],
            "outcomes": [o.to_dict() for o in self.outcomes],
            "tokens_used": self.tokens_used,
            "input_allowance": self.budget.input_allowance,
            "reserve_output": self.budget.reserve_output,
            "headroom": self.headroom,
            "utilisation": round(self.utilisation, 4),
            "dropped": self.dropped,
            "compacted": self.compacted,
            "over_budget": self.over_budget,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
        }


class CompactionOutcome(BaseModel):
    """What a compactor did to one segment."""

    content: str
    tokens_before: int
    tokens_after: int
    strategy: str
    #: True when the compactor could not reach the target.
    partial: bool = False
    detail: str = ""

    @property
    def achieved(self) -> bool:
        return self.tokens_after < self.tokens_before


Method = Literal[
    "truncate", "extractive", "summarize", "hybrid", "none"
]
