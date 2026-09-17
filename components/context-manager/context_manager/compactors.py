"""Compactors: shrink a segment while keeping what matters.

Compaction is **lossy and irreversible**, so it is always the second choice
after dropping something genuinely worthless. Four strategies, each with an
honest failure mode:

``TruncateCompactor``
    Keeps the head and the tail. Deterministic and dependency-free. The head
    usually holds the request or the document's framing; the tail usually holds
    the conclusion or the error. The middle is the most redundant part of most
    text — which is why head+tail beats head-only, the common default.

``ExtractiveCompactor``
    Scores sentences by query overlap and position, keeps the top ones in
    original order. Cheap, no model. **Failure mode: it preserves statements
    without preserving their relationships**, so a passage of reasoning can
    come out as a list of assertive fragments. Good for reference material,
    poor for arguments.

``SummarizeCompactor``
    Asks a model to rewrite. Preserves reasoning. **Failure mode: it can
    invent.** A summary is a new claim about the text, so the result is marked
    compacted and its provenance is recorded rather than trusted silently.

``HybridCompactor``
    Extractive for the bulk, a model pass only if still over target. The
    default: deterministic wherever determinism is enough.
"""
from __future__ import annotations

import re
from typing import Any, Awaitable, Callable, Protocol, runtime_checkable

from context_manager.models import CompactionOutcome, ContextSegment

_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n{2,}")
_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    """a an and are as at be but by for from has have he her his i if in is it its of on or
    that the their they this to was were will with you your not no do does did""".split()
)


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


@runtime_checkable
class BaseCompactor(Protocol):
    """A compactor rewrites one segment toward a token target."""

    name: str

    async def compact(
        self, segment: ContextSegment, *, target_tokens: int, query: str = ""
    ) -> CompactionOutcome: ...


def _noop(segment: ContextSegment, target: int) -> CompactionOutcome:
    return CompactionOutcome(
        content=segment.content,
        tokens_before=len(segment.content),
        tokens_after=len(segment.content),
        strategy="none",
        partial=True,
        detail="nothing to do",
    )


class TruncateCompactor:
    """Keep the head and the tail, elide the middle.

    Deterministic and free. Always reaches the target if the text is long
    enough, because it can simply take fewer characters.
    """

    name = "truncate"

    def __init__(self, chars_per_token: float = 4.0, head_fraction: float = 0.6) -> None:
        self.chars_per_token = chars_per_token
        self.head_fraction = head_fraction

    async def compact(
        self, segment: ContextSegment, *, target_tokens: int, query: str = ""
    ) -> CompactionOutcome:
        text = segment.content
        before = self._count(text)
        budget_chars = max(16, int(target_tokens * self.chars_per_token))
        if len(text) <= budget_chars:
            return CompactionOutcome(
                content=text, tokens_before=before, tokens_after=before,
                strategy=self.name, detail="already within target",
            )

        marker = "\n[... middle elided ...]\n"
        usable = budget_chars - len(marker)
        head_len = int(usable * self.head_fraction)
        tail_len = usable - head_len
        content = text[:head_len] + marker + (text[-tail_len:] if tail_len > 0 else "")
        after = self._count(content)
        return CompactionOutcome(
            content=content, tokens_before=before, tokens_after=after,
            strategy=self.name,
            partial=after > target_tokens,
            detail=f"kept {head_len} head chars + {tail_len} tail chars",
        )

    def _count(self, text: str) -> int:
        # Character-based: this compactor thinks in characters, so pricing in
        # characters keeps its target arithmetic self-consistent.
        return max(1, len(text) // int(self.chars_per_token)) if text else 0


class ExtractiveCompactor:
    """Keep the sentences that carry the query's terms.

    Deterministic and model-free. Sentence order is preserved in the output,
    because reordering a passage changes its argument even when every sentence
    is kept.
    """

    name = "extractive"

    def __init__(self, chars_per_token: float = 4.0, first_sentence_boost: float = 1.5) -> None:
        self.chars_per_token = chars_per_token
        self.first_sentence_boost = first_sentence_boost

    async def compact(
        self, segment: ContextSegment, *, target_tokens: int, query: str = ""
    ) -> CompactionOutcome:
        text = segment.content
        before = self._count(text)
        sentences = [s.strip() for s in _SENTENCE.split(text) if s.strip()]
        if len(sentences) <= 1:
            # One long sentence: extraction cannot help, fall back to truncation
            # so the caller still gets a bounded result.
            return await TruncateCompactor(self.chars_per_token).compact(
                segment, target_tokens=target_tokens, query=query
            )

        q_terms = {t for t in _tokens(query) if t not in _STOP}
        scored: list[tuple[float, int, str]] = []
        for i, s in enumerate(sentences):
            terms = {t for t in _tokens(s) if t not in _STOP}
            overlap = len(q_terms & terms) / len(q_terms) if q_terms else 0.0
            # Position matters: openers and closers carry disproportionate
            # meaning, so they get a nudge rather than being ranked purely by
            # keyword score.
            position = 1.0 / (1 + i * 0.2)
            bonus = self.first_sentence_boost if i == 0 else 0.0
            scored.append((overlap * 2.0 + position + bonus, i, s))

        scored.sort(key=lambda t: -t[0])
        chosen: list[tuple[int, str]] = []
        used = 0
        for _score, idx, s in scored:
            cost = self._count(s)
            if used + cost > target_tokens and chosen:
                continue
            chosen.append((idx, s))
            used += cost
            if used >= target_tokens:
                break

        chosen.sort(key=lambda t: t[0])          # restore original order
        content = " ".join(s for _i, s in chosen)
        after = self._count(content)
        return CompactionOutcome(
            content=content, tokens_before=before, tokens_after=after,
            strategy=self.name, partial=after > target_tokens,
            detail=f"kept {len(chosen)}/{len(sentences)} sentences",
        )

    def _count(self, text: str) -> int:
        return max(1, len(text) // int(self.chars_per_token)) if text else 0


class SummarizeCompactor:
    """Rewrite the segment with a model.

    Accepts any ``(messages, **kw) -> (content, calls)`` client, matching the
    convention the other components use. The prompt is deliberately explicit
    that preserving the *reasoning* matters more than brevity, because a summary
    that keeps conclusions but drops the reasoning makes the model's later
    answers unfalsifiable.
    """

    name = "summarize"

    def __init__(
        self,
        llm_client: Callable[..., Awaitable[Any]] | None = None,
        chars_per_token: float = 4.0,
        max_input_chars: int = 24000,
    ) -> None:
        self.llm_client = llm_client
        self.chars_per_token = chars_per_token
        self.max_input_chars = max_input_chars

    async def compact(
        self, segment: ContextSegment, *, target_tokens: int, query: str = ""
    ) -> CompactionOutcome:
        text = segment.content
        before = self._count(text)
        if self.llm_client is None:
            # No model available: degrade to extractive rather than pretend.
            return await ExtractiveCompactor(self.chars_per_token).compact(
                segment, target_tokens=target_tokens, query=query
            )

        prompt = (
            "Compress the CONTENT below to roughly "
            f"{target_tokens} tokens ({int(target_tokens * self.chars_per_token)} "
            "characters).\nPreserve concrete facts, names, numbers and any causal "
            "reasoning.\nDrop pleasantries and repetition. Do not add information "
            "that is not present.\n"
            f"{('The reader is looking for: ' + query) if query else ''}\n\n"
            f"CONTENT:\n{text[: self.max_input_chars]}"
        )
        try:
            result = await self.llm_client(
                [{"role": "user", "content": prompt}], max_tokens=target_tokens
            )
            content = result[0] if isinstance(result, tuple) else result
        except Exception as exc:  # noqa: BLE001
            fallback = await ExtractiveCompactor(self.chars_per_token).compact(
                segment, target_tokens=target_tokens, query=query
            )
            fallback.detail = f"model failed ({type(exc).__name__}); {fallback.detail}"
            return fallback

        content = (content or "").strip()
        if not content:
            return await ExtractiveCompactor(self.chars_per_token).compact(
                segment, target_tokens=target_tokens, query=query
            )
        after = self._count(content)
        return CompactionOutcome(
            content=content, tokens_before=before, tokens_after=after,
            strategy=self.name, partial=after > target_tokens,
            detail="model-rewritten (lossy; may rephrase)",
        )

    def _count(self, text: str) -> int:
        return max(1, len(text) // int(self.chars_per_token)) if text else 0


class HybridCompactor:
    """Extractive first; a model pass only when extraction was not enough.

    The default strategy. Determinism where determinism suffices, and a model
    only where the extra fidelity is actually needed.
    """

    name = "hybrid"

    def __init__(
        self,
        llm_client: Callable[..., Awaitable[Any]] | None = None,
        chars_per_token: float = 4.0,
        min_tokens_for_model: int = 200,
    ) -> None:
        self._extractive = ExtractiveCompactor(chars_per_token)
        self._summarize = SummarizeCompactor(llm_client, chars_per_token)
        self.min_tokens_for_model = min_tokens_for_model
        self.name = "hybrid" if llm_client else "hybrid-extractive"

    async def compact(
        self, segment: ContextSegment, *, target_tokens: int, query: str = ""
    ) -> CompactionOutcome:
        first = await self._extractive.compact(
            segment, target_tokens=target_tokens, query=query
        )
        if first.tokens_after <= target_tokens:
            first.strategy = self.name
            first.detail = f"extractive pass sufficed; {first.detail}"
            return first
        # Short segments are not worth a model call, and a model asked to
        # compress 150 tokens tends to expand them instead.
        if first.tokens_after < self.min_tokens_for_model:
            first.strategy = "extractive"
            first.detail = f"below model threshold; {first.detail}"
            return first

        trimmed = segment.model_copy(update={"content": first.content})
        second = await self._summarize.compact(
            trimmed, target_tokens=target_tokens, query=query
        )
        second.strategy = self.name
        second.tokens_before = first.tokens_before
        second.detail = f"extractive then {second.detail}"
        return second


def build_compactor(
    kind: str = "hybrid",
    llm_client: Callable[..., Awaitable[Any]] | None = None,
    **kwargs: Any,
) -> BaseCompactor:
    """Factory mirroring ``build_counter``."""
    if kind == "truncate":
        return TruncateCompactor(**kwargs)
    if kind == "extractive":
        return ExtractiveCompactor(**kwargs)
    if kind == "summarize":
        return SummarizeCompactor(llm_client, **kwargs)
    if kind == "hybrid":
        return HybridCompactor(llm_client, **kwargs)
    if kind == "none":
        return _NoCompactor()
    raise ValueError(f"unknown compactor kind: {kind!r}")


class _NoCompactor:
    """Explicitly refuse to compact. Useful when loss is not acceptable."""

    name = "none"

    async def compact(
        self, segment: ContextSegment, *, target_tokens: int, query: str = ""
    ) -> CompactionOutcome:
        n = max(1, len(segment.content) // 4) if segment.content else 0
        return CompactionOutcome(
            content=segment.content, tokens_before=n, tokens_after=n,
            strategy="none", partial=True,
            detail="compaction disabled; segment will be dropped if it does not fit",
        )
