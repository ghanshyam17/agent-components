"""The assembler: decide what fits in the window, and record why.

This is the component's core. The algorithm is a short, ordered sequence of
decisions, and **the order is the design** — each step exists because doing it
later (or not at all) produces a specific failure:

1. **Price everything.** No decision is possible before the budget is known.
2. **Deduplicate by content fingerprint.** The same retrieved chunk arriving
   twice wastes budget twice, and it looks like corroboration when it is not.
3. **Separate protected from evictable.** A system prompt is not competing with
   a log dump for space.
4. **Admit protected content unconditionally.** If it alone overflows, say so
   loudly rather than silently dropping the instructions — the caller needs to
   know the request is unsound, and a prompt missing its safety rules is worse
   than a request that fails.
5. **Enforce per-source caps,** compacting before dropping.
6. **Enforce a per-segment ceiling,** so one enormous tool result cannot
   monopolise the window.
7. **Fill greedily by priority,** attempting compaction on a near-miss.
8. **Emit in sequence order** with framing hoisted to the front.
"""
from __future__ import annotations

import time

from context_manager.compactors import BaseCompactor, TruncateCompactor
from context_manager.models import (
    CompactionOutcome,
    ContextBudget,
    ContextMessage,
    ContextPackage,
    ContextRole,
    ContextSegment,
    Disposition,
    DropReason,
    SegmentOutcome,
    SegmentPriority,
)
from context_manager.tokenizer import HeuristicCounter, TokenCounter


class ContextAssembler:
    """Builds a ``ContextPackage`` from a bag of segments within a budget."""

    def __init__(
        self,
        budget: ContextBudget | None = None,
        *,
        counter: TokenCounter | None = None,
        compactor: BaseCompactor | None = None,
        dedupe: bool = True,
    ) -> None:
        self.budget = budget or ContextBudget()
        self.counter = counter or HeuristicCounter()
        self.compactor = compactor or TruncateCompactor()
        self.dedupe = dedupe

    # ------------------------------------------------------------------ api
    async def assemble(
        self,
        segments: list[ContextSegment],
        *,
        query: str = "",
        budget: ContextBudget | None = None,
    ) -> ContextPackage:
        started = time.perf_counter()
        budget = budget or self.budget
        package = ContextPackage(budget=budget)

        # 1. price
        priced: list[tuple[ContextSegment, int]] = [
            (s, self.counter.count_message(s.role.to_wire(), s.content)) for s in segments
        ]
        remaining = budget.input_allowance

        # 2. deduplicate
        priced, outcomes = self._dedupe(priced)

        # 3. split
        protected = [(s, t) for s, t in priced if s.is_protected()]
        evictable = [(s, t) for s, t in priced if not s.is_protected()]

        # 4. protected content is admitted unconditionally
        admitted: list[tuple[ContextSegment, int, bool]] = []
        compacted_content: dict[str, str] = {}
        for seg, tokens in protected:
            if tokens > remaining:
                package.warnings.append(
                    f"protected segment {seg.id} ({seg.source}) needs {tokens} tokens "
                    f"but only {remaining} remain of the "
                    f"{budget.input_allowance}-token allowance; the request is over "
                    f"budget and will likely be rejected"
                )
                package.over_budget = True
            admitted.append((seg, tokens, False))
            remaining -= tokens

        # 5. per-source caps
        evictable, outcomes, cap_content = await self._apply_source_caps(
            evictable, budget, query, outcomes
        )
        compacted_content.update(cap_content)

        # 6. per-segment ceiling
        evictable, outcomes, ceil_content = await self._apply_segment_ceiling(
            evictable, budget, query, outcomes
        )
        compacted_content.update(ceil_content)

        # 7. greedy fill, highest priority first, most recent wins ties
        evictable.sort(key=lambda st: (-st[0].priority, -st[0].sequence))
        for seg, tokens in evictable:
            if tokens <= remaining:
                admitted.append((seg, tokens, False))
                outcomes.append(
                    SegmentOutcome(
                        segment_id=seg.id,
                        source=seg.source,
                        disposition=Disposition.KEPT,
                        tokens_before=tokens,
                        tokens_after=tokens,
                    )
                )
                remaining -= tokens
                continue

            # Near miss: try to make it fit before giving up on it.
            compacted = await self._try_compact(seg, query, budget)
            if compacted is not None and compacted.tokens_after <= remaining:
                admitted.append((seg, compacted.tokens_after, True))
                outcomes.append(
                    SegmentOutcome(
                        segment_id=seg.id,
                        source=seg.source,
                        disposition=Disposition.COMPACTED,
                        tokens_before=tokens,
                        tokens_after=compacted.tokens_after,
                        detail=(
                            f"{compacted.strategy} on insert"
                            + (" (partial)" if compacted.partial else "")
                        ),
                    )
                )
                remaining -= compacted.tokens_after
                package.compacted += 1
                compacted_content[seg.id] = compacted.content
                continue

            outcomes.append(
                SegmentOutcome(
                    segment_id=seg.id,
                    source=seg.source,
                    disposition=Disposition.DROPPED,
                    tokens_before=tokens,
                    tokens_after=0,
                    reason=DropReason.OVER_BUDGET,
                    detail=f"needed {tokens}, {remaining} remaining",
                )
            )
            package.dropped += 1

        # 8. emit
        package.messages = self._emit(admitted, compacted_content)
        package.tokens_used = sum(m.tokens for m in package.messages)
        package.outcomes = self._sort_outcomes(outcomes)
        package.duration_ms = int((time.perf_counter() - started) * 1000)

        if package.tokens_used > budget.input_allowance and not package.over_budget:
            package.over_budget = True
            package.warnings.append(
                f"assembled {package.tokens_used} tokens against a "
                f"{budget.input_allowance} input allowance"
            )
        return package

    # ------------------------------------------------------------- internals
    def _dedupe(
        self, priced: list[tuple[ContextSegment, int]]
    ) -> tuple[list[tuple[ContextSegment, int]], list[SegmentOutcome]]:
        """Drop exact content duplicates, keeping the highest-priority copy."""
        if not self.dedupe:
            return priced, []
        best: dict[str, tuple[ContextSegment, int]] = {}
        dropped: dict[str, tuple[ContextSegment, int]] = {}
        for seg, tokens in priced:
            fp = seg.fingerprint()
            prev = best.get(fp)
            if prev is None:
                best[fp] = (seg, tokens)
                continue
            # Keep the more important copy; ties go to the earlier one.
            if (seg.priority, -seg.sequence) > (prev[0].priority, -prev[0].sequence):
                dropped[prev[0].id] = prev
                best[fp] = (seg, tokens)
            else:
                dropped[seg.id] = (seg, tokens)

        outcomes = [
            SegmentOutcome(
                segment_id=seg.id,
                source=seg.source,
                disposition=Disposition.DEDUPLICATED,
                tokens_before=tokens,
                tokens_after=0,
                reason=DropReason.DUPLICATE,
                detail="identical content already present",
            )
            for seg, tokens in dropped.values()
        ]
        return list(best.values()), outcomes

    async def _apply_source_caps(
        self,
        evictable: list[tuple[ContextSegment, int]],
        budget: ContextBudget,
        query: str,
        outcomes: list[SegmentOutcome],
    ) -> tuple[list[tuple[ContextSegment, int]], list[SegmentOutcome], dict[str, str]]:
        """Trim any source consuming more than its share of the allowance.

        Returns the rewritten content alongside the new sizes: recording only
        the smaller *size* while emitting the original *text* would send more
        tokens than the budget was debited for.
        """
        if not budget.source_fractions:
            return evictable, outcomes, {}
        kept: list[tuple[ContextSegment, int]] = []
        rewritten: dict[str, str] = {}
        for seg, tokens in evictable:
            cap = budget.source_cap(seg.source)
            if cap is None or tokens <= cap:
                kept.append((seg, tokens))
                continue
            compacted = await self._try_compact(seg, query, budget, target=cap)
            if compacted is not None and compacted.tokens_after <= cap:
                kept.append((seg, compacted.tokens_after))
                rewritten[seg.id] = compacted.content
                outcomes.append(
                    SegmentOutcome(
                        segment_id=seg.id,
                        source=seg.source,
                        disposition=Disposition.COMPACTED,
                        tokens_before=tokens,
                        tokens_after=compacted.tokens_after,
                        detail=f"{compacted.strategy} for source cap {cap}",
                    )
                )
                continue
            outcomes.append(
                SegmentOutcome(
                    segment_id=seg.id,
                    source=seg.source,
                    disposition=Disposition.DROPPED,
                    tokens_before=tokens,
                    tokens_after=0,
                    reason=DropReason.SOURCE_CAP,
                    detail=f"source {seg.source!r} over its {cap}-token cap",
                )
            )
        return kept, outcomes, rewritten

    async def _apply_segment_ceiling(
        self,
        evictable: list[tuple[ContextSegment, int]],
        budget: ContextBudget,
        query: str,
        outcomes: list[SegmentOutcome],
    ) -> tuple[list[tuple[ContextSegment, int]], list[SegmentOutcome], dict[str, str]]:
        """No single segment may eat more than ``max_segment_fraction``."""
        ceiling = budget.max_segment_tokens()
        kept: list[tuple[ContextSegment, int]] = []
        rewritten: dict[str, str] = {}
        for seg, tokens in evictable:
            if tokens <= ceiling:
                kept.append((seg, tokens))
                continue
            compacted = await self._try_compact(seg, query, budget, target=ceiling)
            if compacted is not None:
                kept.append((seg, compacted.tokens_after))
                rewritten[seg.id] = compacted.content
                outcomes.append(
                    SegmentOutcome(
                        segment_id=seg.id,
                        source=seg.source,
                        disposition=Disposition.COMPACTED,
                        tokens_before=tokens,
                        tokens_after=compacted.tokens_after,
                        detail=(
                            f"{compacted.strategy} for the {ceiling}-token per-segment "
                            "ceiling"
                        ),
                    )
                )
                continue
            kept.append((seg, tokens))
        return kept, outcomes, rewritten

    async def _try_compact(
        self,
        seg: ContextSegment,
        query: str,
        budget: ContextBudget,
        target: int | None = None,
    ) -> CompactionOutcome | None:
        if not seg.compactable or self.compactor is None:
            return None
        target = target or budget.max_segment_tokens()
        try:
            return await self.compactor.compact(seg, target_tokens=target, query=query)
        except Exception:  # noqa: BLE001
            # A failing compactor must not fail the assembly: the segment then
            # competes on its original size, which is the pre-compaction status
            # quo rather than a lost turn.
            return None

    def _emit(
        self,
        admitted: list[tuple[ContextSegment, int, bool]],
        compacted_content: dict[str, str],
    ) -> list[ContextMessage]:
        """Turn admitted segments into messages, framing first.

        ``compacted_content`` carries rewritten text for segments that were
        compacted on insert; a plain dict is threaded through rather than
        stashed on the model, because pydantic models are not scratch space and
        a private attribute would not survive serialisation anyway.
        """
        def rank(item: tuple[ContextSegment, int, bool]) -> tuple[int, int]:
            seg = item[0]
            # A system prompt and pinned memory are framing, not conversation:
            # they go first regardless of where they sat in the stream.
            framing = seg.role is ContextRole.SYSTEM or (
                seg.role is ContextRole.MEMORY and seg.pinned
            )
            return (0 if framing else 1, seg.sequence)

        messages: list[ContextMessage] = []
        for seg, _tokens, was_compacted in sorted(admitted, key=rank):
            # The rewritten text is authoritative whenever it exists — including
            # for segments compacted by the source-cap or per-segment-ceiling
            # passes, which are admitted with the flag still False. Keying off
            # the map keeps the emitted text and the debited size consistent.
            rewritten = compacted_content.get(seg.id)
            content = rewritten if rewritten is not None else seg.content
            messages.append(
                ContextMessage(
                    role=seg.role.to_wire(),
                    content=content,
                    tokens=self.counter.count_message(seg.role.to_wire(), content),
                    segment_id=seg.id,
                    compacted=was_compacted or rewritten is not None,
                )
            )
        return messages

    @staticmethod
    def _sort_outcomes(outcomes: list[SegmentOutcome]) -> list[SegmentOutcome]:
        order = {
            Disposition.DROPPED: 0,
            Disposition.DEDUPLICATED: 1,
            Disposition.COMPACTED: 2,
            Disposition.KEPT: 3,
        }
        return sorted(outcomes, key=lambda o: order.get(o.disposition, 4))


class SingleShotAssembler(ContextAssembler):
    """Convenience wrapper for the common 'one system prompt + history' shape.

    Exists so the 90% case does not require hand-building segments. It is a
    subclass, not a separate implementation, so the budget logic cannot drift.
    """

    async def build(
        self,
        system: str,
        history: list[dict[str, str]] | None = None,
        *,
        query: str = "",
        extra: list[ContextSegment] | None = None,
        budget: ContextBudget | None = None,
    ) -> ContextPackage:
        segments: list[ContextSegment] = [
            ContextSegment(
                role=ContextRole.SYSTEM,
                content=system,
                priority=SegmentPriority.CRITICAL,
                pinned=True,
                source="system",
                compactable=False,
                sequence=-1,
            )
        ]
        for i, msg in enumerate(history or []):
            segments.append(
                ContextSegment(
                    role=ContextRole(msg.get("role", "user")),
                    content=msg.get("content", ""),
                    priority=SegmentPriority.NORMAL,
                    source="history",
                    sequence=i,
                )
            )
        segments.extend(extra or [])
        return await self.assemble(segments, query=query, budget=budget)
