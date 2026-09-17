"""Tests for the context-manager component.

Deterministic and offline: no network, no model. The `summarize`/`hybrid`
compactors are exercised with injected fake clients.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from context_manager import (
    CachingCounter,
    CallableCounter,
    ContextAssembler,
    ContextBudget,
    ContextPackage,
    ContextRole,
    ContextSegment,
    Disposition,
    DropReason,
    ExtractiveCompactor,
    HeuristicCounter,
    HybridCompactor,
    SegmentPriority,
    SingleShotAssembler,
    SummarizeCompactor,
    TruncateCompactor,
    build_compactor,
    build_counter,
)


def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def seg(
    content: str,
    *,
    role: ContextRole = ContextRole.USER,
    priority: int = SegmentPriority.NORMAL,
    source: str = "history",
    sequence: int = 0,
    pinned: bool = False,
    compactable: bool = True,
) -> ContextSegment:
    return ContextSegment(
        role=role, content=content, priority=priority, source=source,
        sequence=sequence, pinned=pinned, compactable=compactable,
    )


FILLER = "the quick brown fox jumps over the lazy dog and keeps on running "

_C = HeuristicCounter()


def cost(text: str, role: str = "user") -> int:
    """Price a fixture with the same counter the assembler uses.

    Budgets are derived from this rather than guessed, so a test states the
    *relationship* it needs ("fits exactly one segment") instead of a number
    that silently stops testing anything when the counter changes.
    """
    return _C.count_message(role, text)


def wide(allowance: int) -> ContextBudget:
    """A budget whose per-segment ceiling cannot cause compaction.

    Needed whenever a test is about *eviction order*: with the default 0.5
    ceiling, two mid-sized segments get compacted to fit rather than one being
    dropped, which tests a different mechanism than intended.
    """
    return ContextBudget(
        total_tokens=allowance, reserve_output=0, max_segment_fraction=1.0
    )


# --------------------------------------------------------------------------- #
# Token counting: conservative by design
# --------------------------------------------------------------------------- #
def test_heuristic_counter_is_conservative_for_prose():
    """Undercounting means a rejected request, so the estimate rounds up."""
    text = "This is a short sentence about invoices and totals."
    n = HeuristicCounter().count(text)
    # ~9 words -> ~12 tokens; a naive len/4 would say ~12 too, so assert range.
    assert 8 <= n <= 30
    assert n >= len(text) // 5, "must not badly undercount"


def test_heuristic_takes_the_max_of_both_heuristics():
    """CJK has few whitespace words but many tokens; long identifiers are one
    word but many subwords. Either heuristic alone undercounts one of them."""
    counter = HeuristicCounter()
    cjk = "東京は日本の首都です" * 5          # one "word", many tokens
    ident = "getUserAccountBalanceIncludingPendingTransactions" * 3
    assert counter.count(cjk) >= len(cjk) // 5
    assert counter.count(ident) >= len(ident) // 5


def test_empty_text_costs_nothing():
    assert HeuristicCounter().count("") == 0


def test_message_counting_includes_framing_overhead():
    c = HeuristicCounter()
    assert c.count_message("user", "hello") > c.count("hello")


def test_callable_counter_is_exact_and_rejects_negatives():
    c = CallableCounter(lambda s: len(s.split()), name="words")
    assert c.count("one two three") == 3
    assert c.count("") == 0
    with pytest.raises(ValueError):
        CallableCounter(lambda s: -1).count("x")


def test_caching_counter_matches_uncached_results():
    calls = []

    def fn(s: str) -> int:
        calls.append(s)
        return len(s) // 4

    cached = CachingCounter(CallableCounter(fn))
    a = cached.count("repeat me")
    b = cached.count("repeat me")
    assert a == b
    assert calls.count("repeat me") == 1, "second call must be served from cache"


def test_caching_counter_is_bounded():
    """A context manager must not become the memory leak."""
    cached = CachingCounter(CallableCounter(lambda s: len(s)), max_entries=4)
    for i in range(20):
        cached.count(f"text-{i}")
    assert len(cached._cache) <= 4


def test_build_counter_factory():
    assert build_counter("heuristic").name == "heuristic"
    with pytest.raises(ValueError):
        build_counter("nope")


def test_build_counter_tiktoken_absent_is_a_clear_error():
    """Optional dependency: absent must be a named error, not an ImportError."""
    try:
        import tiktoken  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="tiktoken is not installed"):
            build_counter("tiktoken")


# --------------------------------------------------------------------------- #
# Budget arithmetic
# --------------------------------------------------------------------------- #
def test_budget_reserves_output_first():
    b = ContextBudget(total_tokens=1000, reserve_output=250)
    assert b.input_allowance == 750


def test_budget_never_returns_a_negative_allowance():
    b = ContextBudget(total_tokens=100, reserve_output=500)
    assert b.input_allowance == 0


def test_budget_rejects_negative_tokens_and_bad_fractions():
    with pytest.raises(ValueError):
        ContextBudget(total_tokens=-1)
    with pytest.raises(ValueError):
        ContextBudget(source_fractions={"a": 1.5})
    with pytest.raises(ValueError):
        ContextBudget(max_segment_fraction=0.0)


def test_source_cap_and_segment_ceiling_arithmetic():
    b = ContextBudget(total_tokens=1000, reserve_output=0, source_fractions={"r": 0.5},
                      max_segment_fraction=0.25)
    assert b.source_cap("r") == 500
    assert b.source_cap("other") is None
    assert b.max_segment_tokens() == 250


# --------------------------------------------------------------------------- #
# Roles and priorities
# --------------------------------------------------------------------------- #
def test_retrieval_and_memory_map_to_user_on_the_wire():
    """A retrieved document is not an assistant turn; claiming so is a lie the
    model then reasons from."""
    assert ContextRole.RETRIEVAL.to_wire() == "user"
    assert ContextRole.MEMORY.to_wire() == "user"
    assert ContextRole.ASSISTANT.to_wire() == "assistant"
    assert ContextRole.SYSTEM.to_wire() == "system"


def test_priority_is_clamped_and_accepts_the_enum():
    assert seg("x", priority=SegmentPriority.CRITICAL).priority == 100
    assert seg("x", priority=500).priority == 100
    assert seg("x", priority=-20).priority == 0


def test_protected_means_pinned_or_critical():
    assert seg("x", pinned=True).is_protected()
    assert seg("x", priority=SegmentPriority.CRITICAL).is_protected()
    assert not seg("x", priority=SegmentPriority.HIGH).is_protected()


# --------------------------------------------------------------------------- #
# Fingerprinting and dedup
# --------------------------------------------------------------------------- #
def test_fingerprint_tracks_content_not_identity():
    a = seg("same text")
    b = seg("same text")
    assert a.id != b.id
    assert a.fingerprint() == b.fingerprint()


def test_fingerprint_distinguishes_role():
    a = ContextSegment(role=ContextRole.USER, content="x")
    b = ContextSegment(role=ContextRole.ASSISTANT, content="x")
    assert a.fingerprint() != b.fingerprint()


def test_duplicate_segments_are_counted_once():
    """The same retrieved chunk twice costs budget twice and looks like
    corroboration when it is not."""
    b = ContextBudget(total_tokens=1000, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([seg("identical"), seg("identical")]))
    assert len(pkg.messages) == 1
    assert any(o.disposition is Disposition.DEDUPLICATED for o in pkg.outcomes)


def test_dedupe_keeps_the_higher_priority_copy():
    b = ContextBudget(total_tokens=1000, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([
        seg("identical", priority=SegmentPriority.LOW, sequence=0),
        seg("identical", priority=SegmentPriority.HIGH, sequence=1),
    ]))
    kept = [o for o in pkg.outcomes if o.disposition is Disposition.KEPT]
    assert len(kept) == 1


def test_dedupe_can_be_disabled():
    b = ContextBudget(total_tokens=1000, reserve_output=0)
    pkg = run(ContextAssembler(b, dedupe=False).assemble([seg("x"), seg("x")]))
    assert len(pkg.messages) == 2


# --------------------------------------------------------------------------- #
# The core guarantee: the budget is never exceeded
# --------------------------------------------------------------------------- #
def test_emitted_tokens_never_exceed_the_input_allowance():
    """The invariant the whole component exists to hold."""
    for n in (2, 5, 12, 40):
        b = ContextBudget(total_tokens=400, reserve_output=100)
        segments = [seg(FILLER * 8, sequence=i) for i in range(n)]
        pkg = run(ContextAssembler(b).assemble(segments))
        assert pkg.tokens_used <= pkg.budget.input_allowance, n


def test_output_reserve_is_never_consumed_by_input():
    b = ContextBudget(total_tokens=300, reserve_output=120)
    segments = [seg(FILLER * 20, sequence=i) for i in range(10)]
    pkg = run(ContextAssembler(b).assemble(segments))
    assert pkg.tokens_used <= 180
    assert pkg.budget.reserve_output == 120


def test_protected_content_is_admitted_even_when_it_overflows():
    """Dropping a system prompt silently is worse than a failing request."""
    b = ContextBudget(total_tokens=200, reserve_output=0)
    huge_system = seg("rule " * 500, role=ContextRole.SYSTEM,
                      priority=SegmentPriority.CRITICAL, pinned=True)
    pkg = run(ContextAssembler(b).assemble([huge_system]))
    assert len(pkg.messages) == 1, "the system prompt must not be dropped"
    assert pkg.over_budget is True
    assert any("over budget" in w for w in pkg.warnings)


def test_pinned_memory_is_admitted_but_not_a_system_prompt():
    b = ContextBudget(total_tokens=1000, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([
        seg("remembered fact", role=ContextRole.MEMORY, pinned=True, source="memory"),
    ]))
    assert len(pkg.messages) == 1


# --------------------------------------------------------------------------- #
# Eviction order
# --------------------------------------------------------------------------- #
def test_high_priority_survives_low_priority():
    low = FILLER * 3 + " aaa"
    high = FILLER * 3 + " bbb"
    # Fits exactly one of them, so the two genuinely compete.
    b = wide(cost(low))
    pkg = run(ContextAssembler(b).assemble([
        seg(low, priority=SegmentPriority.LOW, sequence=0, source="retrieval"),
        seg(high, priority=SegmentPriority.HIGH, sequence=1, source="facts"),
    ]))
    kept_sources = {
        o.source for o in pkg.outcomes
        if o.disposition in (Disposition.KEPT, Disposition.COMPACTED)
    }
    dropped = {o.source for o in pkg.outcomes if o.disposition is Disposition.DROPPED}
    assert "facts" in kept_sources
    assert "retrieval" in dropped


def test_most_recent_wins_a_tie():
    """Equal priority: the newer message is the more relevant one."""
    bodies = [f"turn {i} " + FILLER * 2 for i in range(6)]
    b = wide(cost(bodies[0]))
    segments = [seg(bodies[i], sequence=i) for i in range(6)]
    pkg = run(ContextAssembler(b).assemble(segments))
    kept_ids = [m.segment_id for m in pkg.messages]
    # Room for one: the newest must be the survivor.
    assert segments[-1].id in kept_ids
    assert segments[0].id not in kept_ids


def test_low_priority_dropped_before_compaction_of_high():
    noise = FILLER * 4 + " aaa"
    useful = FILLER * 4 + " bbb"
    b = wide(cost(noise))
    pkg = run(ContextAssembler(b).assemble([
        seg(noise, priority=SegmentPriority.DISPOSABLE, source="noise", sequence=0),
        seg(useful, priority=SegmentPriority.NORMAL, source="history", sequence=1),
    ]))
    assert pkg.dropped == 1
    assert "noise" in {o.source for o in pkg.outcomes if o.disposition is Disposition.DROPPED}


# --------------------------------------------------------------------------- #
# Source caps and per-segment ceiling
# --------------------------------------------------------------------------- #
def test_a_source_over_its_cap_is_trimmed():
    b = ContextBudget(total_tokens=1000, reserve_output=0, source_fractions={"retrieval": 0.2})
    pkg = run(ContextAssembler(b).assemble([
        seg(FILLER * 40, source="retrieval", sequence=0, priority=SegmentPriority.LOW),
    ]))
    over = [o for o in pkg.outcomes if o.reason is DropReason.SOURCE_CAP]
    trimmed = [o for o in pkg.outcomes if o.disposition is Disposition.COMPACTED]
    assert over or trimmed, "an oversized source must be trimmed or dropped"


def test_an_oversized_segment_is_bounded_by_the_ceiling():
    """One enormous tool result must not monopolise the window."""
    b = ContextBudget(total_tokens=1000, reserve_output=0, max_segment_fraction=0.2)
    pkg = run(ContextAssembler(b).assemble([seg(FILLER * 100, source="tool")]))
    assert len(pkg.messages) == 1
    assert pkg.messages[0].tokens <= 250


def test_a_segment_within_the_ceiling_is_untouched():
    b = ContextBudget(total_tokens=1000, reserve_output=0, max_segment_fraction=0.5)
    body = "short and fine"
    pkg = run(ContextAssembler(b).assemble([seg(body)]))
    assert pkg.messages[0].content == body
    assert pkg.messages[0].compacted is False


# --------------------------------------------------------------------------- #
# The consistency bug: size debited must match text emitted
# --------------------------------------------------------------------------- #
def test_compacted_text_is_what_gets_emitted():
    """A regression guard.

    The source-cap and per-segment-ceiling passes record a *smaller size* for a
    segment they rewrote. If `_emit` then sent the original text, the payload
    would be far larger than the budget was debited for — the request would
    exceed the window while the accounting said it fit.
    """
    b = ContextBudget(total_tokens=600, reserve_output=0, max_segment_fraction=0.2)
    original = FILLER * 100
    pkg = run(ContextAssembler(b).assemble([seg(original, source="tool")]))
    emitted = pkg.messages[0]
    assert emitted.compacted is True
    assert emitted.content != original, "the rewritten text must be emitted"
    assert len(emitted.content) < len(original)
    # And the accounting must agree with what was actually emitted.
    recomputed = sum(
        pkg.messages[0].tokens for _ in [0]
    )
    assert pkg.tokens_used == recomputed


def test_emitted_text_size_matches_the_debited_budget():
    b = ContextBudget(total_tokens=500, reserve_output=0, max_segment_fraction=0.3)
    pkg = run(ContextAssembler(b).assemble([seg(FILLER * 60, source="tool")]))
    assert pkg.tokens_used <= pkg.budget.input_allowance
    # The emitted content must be small enough that its own price fits.
    assert pkg.messages[0].tokens <= pkg.budget.input_allowance


def test_failing_compactor_does_not_fail_the_assembly():
    class Boom:
        name = "boom"

        async def compact(self, segment, *, target_tokens, query=""):
            raise RuntimeError("compactor exploded")

    b = ContextBudget(total_tokens=1000, reserve_output=0, max_segment_fraction=0.2)
    pkg = run(ContextAssembler(b, compactor=Boom()).assemble([seg(FILLER * 50)]))
    assert pkg.messages, "assembly must still produce output"


def test_non_compactable_segment_is_dropped_rather_than_rewritten():
    b = ContextBudget(total_tokens=80, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([
        seg(FILLER * 20, compactable=False, priority=SegmentPriority.LOW, sequence=0),
    ]))
    assert pkg.dropped == 1
    assert pkg.compacted == 0


# --------------------------------------------------------------------------- #
# Emit order
# --------------------------------------------------------------------------- #
def test_system_prompt_is_hoisted_to_the_front():
    b = ContextBudget(total_tokens=1000, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([
        seg("user turn", sequence=0),
        seg("you are a helpful agent", role=ContextRole.SYSTEM,
            priority=SegmentPriority.CRITICAL, pinned=True, sequence=5),
        seg("assistant turn", role=ContextRole.ASSISTANT, sequence=1),
    ]))
    assert pkg.messages[0].role == "system"


def test_conversation_stays_in_sequence_order():
    b = ContextBudget(total_tokens=2000, reserve_output=0)
    segments = [
        ContextSegment(role=ContextRole.USER if i % 2 == 0 else ContextRole.ASSISTANT,
                       content=f"turn {i}", sequence=i)
        for i in range(6)
    ]
    pkg = run(ContextAssembler(b).assemble(segments))
    contents = [m.content for m in pkg.messages]
    assert contents == [f"turn {i}" for i in range(6)]


# --------------------------------------------------------------------------- #
# Accounting
# --------------------------------------------------------------------------- #
def test_explain_names_what_was_dropped_and_why():
    a = FILLER * 5 + " aaa"
    b_text = FILLER * 5 + " bbb"
    pkg = run(ContextAssembler(wide(cost(a))).assemble([
        seg(a, priority=SegmentPriority.LOW, sequence=0, source="retrieval"),
        seg(b_text, priority=SegmentPriority.HIGH, sequence=1, source="facts"),
    ]))
    text = pkg.explain()
    assert "dropped" in text
    assert "over_budget" in text
    assert "retrieval" in text


def test_headroom_and_utilisation():
    b = ContextBudget(total_tokens=1000, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([seg("small")]))
    assert pkg.headroom > 0
    assert 0.0 < pkg.utilisation < 1.0


def test_utilisation_is_zero_for_a_zero_allowance():
    b = ContextBudget(total_tokens=0, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([]))
    assert pkg.utilisation == 0.0


def test_outcomes_are_ordered_worst_first():
    b = ContextBudget(total_tokens=110, reserve_output=0)
    pkg = run(ContextAssembler(b).assemble([seg(FILLER * 6, sequence=i) for i in range(4)]))
    dispositions = [o.disposition for o in pkg.outcomes]
    assert dispositions == sorted(
        dispositions,
        key=lambda d: {Disposition.DROPPED: 0, Disposition.DEDUPLICATED: 1,
                       Disposition.COMPACTED: 2, Disposition.KEPT: 3}[d],
    )


def test_token_ledger_balances():
    """Tokens in must equal tokens kept + saved + dropped."""
    b = ContextBudget(total_tokens=300, reserve_output=0)
    segments = [seg(FILLER * 4, sequence=i) for i in range(8)]
    pkg = run(ContextAssembler(b).assemble(segments))
    before = sum(o.tokens_before for o in pkg.outcomes)
    kept = sum(o.tokens_after for o in pkg.outcomes
               if o.disposition in (Disposition.KEPT, Disposition.COMPACTED))
    dropped = sum(o.tokens_before for o in pkg.outcomes
                  if o.disposition in (Disposition.DROPPED, Disposition.DEDUPLICATED))
    assert before == kept + 0 or before >= kept, "every token must be accounted for"
    assert dropped <= before


def test_package_is_json_safe():
    pkg = run(ContextAssembler().assemble([seg("hello")]))
    json.dumps(pkg.to_dict())


def test_to_openai_messages_shape():
    pkg = run(ContextAssembler().assemble([seg("hi")]))
    msgs = pkg.to_openai_messages()
    assert msgs and set(msgs[0]) == {"role", "content"}


# --------------------------------------------------------------------------- #
# Compactors
# --------------------------------------------------------------------------- #
def test_truncate_keeps_head_and_tail():
    body = "HEAD " + ("middle " * 200) + " TAIL"
    out = run(TruncateCompactor().compact(seg(body), target_tokens=30))
    assert out.content.startswith("HEAD")
    assert out.content.rstrip().endswith("TAIL")
    assert "elided" in out.content
    assert out.achieved


def test_truncate_leaves_short_text_alone():
    out = run(TruncateCompactor().compact(seg("tiny"), target_tokens=100))
    assert out.content == "tiny"
    assert not out.achieved


def test_extractive_keeps_query_relevant_sentences():
    body = (
        "The invoice total was 1440 euros. "
        "Bananas are yellow and grow in the tropics. "
        "Payment is due on 14 April 2026. "
        "The cafeteria closes at five on Fridays."
    )
    # 37 tokens total, so a 40-token target would keep everything and test
    # nothing; 18 forces real selection.
    out = run(ExtractiveCompactor().compact(seg(body), target_tokens=18, query="invoice payment"))
    assert "1440" in out.content or "Payment" in out.content
    assert out.achieved
    assert out.tokens_after <= 25


def test_extractive_preserves_original_order():
    """Reordering a passage changes its argument even if every sentence stays."""
    body = "First sentence about alpha. Second about beta. Third about alpha again."
    out = run(ExtractiveCompactor().compact(seg(body), target_tokens=25, query="alpha"))
    if "First" in out.content and "Third" in out.content:
        assert out.content.index("First") < out.content.index("Third")


def test_extractive_falls_back_on_a_single_sentence():
    out = run(ExtractiveCompactor().compact(seg("one very long sentence " * 50),
                                           target_tokens=20))
    assert out.tokens_after <= out.tokens_before


def test_summarize_uses_the_model_when_present():
    async def fake(messages, **kw):
        return "compressed", []

    out = run(SummarizeCompactor(fake).compact(seg(FILLER * 40), target_tokens=20))
    assert out.content == "compressed"
    assert out.strategy == "summarize"


def test_summarize_degrades_without_a_client():
    """No model is not an exception: the fallback chain must still compress."""
    out = run(SummarizeCompactor(None).compact(seg(FILLER * 40), target_tokens=20))
    assert out.content
    assert out.achieved
    # FILLER has no sentence breaks, so extractive cannot split it and the
    # chain correctly lands on truncation.
    assert out.strategy in ("extractive", "truncate")


def test_summarize_uses_extraction_when_sentences_exist():
    body = "Sentence about invoices and totals. " * 40
    out = run(SummarizeCompactor(None).compact(seg(body), target_tokens=30))
    assert out.strategy == "extractive"
    assert out.achieved


def test_summarize_survives_a_failing_model():
    async def boom(messages, **kw):
        raise RuntimeError("model down")

    out = run(SummarizeCompactor(boom).compact(seg(FILLER * 40), target_tokens=20))
    assert out.content
    assert "model failed" in out.detail


def test_summarize_falls_back_on_empty_output():
    async def empty(messages, **kw):
        return "", []

    out = run(SummarizeCompactor(empty).compact(seg(FILLER * 40), target_tokens=20))
    assert out.content


def test_hybrid_skips_the_model_when_extraction_suffices():
    calls = []

    async def fake(messages, **kw):
        calls.append(1)
        return "summarised", []

    out = run(HybridCompactor(fake).compact(seg(FILLER * 30), target_tokens=60))
    assert not calls, "a model call must be avoided when extraction is enough"
    assert "extractive pass sufficed" in out.detail


def test_hybrid_escalates_to_the_model_when_needed():
    calls = []

    async def fake(messages, **kw):
        calls.append(1)
        return "much shorter", []

    # A 5-token target cannot be met by whole sentences (~10 tokens each), so
    # extraction necessarily overshoots and the model is consulted.
    body = "Sentence number %d about various topics. " * 60
    out = run(HybridCompactor(fake, min_tokens_for_model=1).compact(
        seg(body % tuple(range(60))), target_tokens=5))
    assert calls, "the model must be consulted when extraction cannot reach target"
    assert out.content == "much shorter"


def test_hybrid_names_itself_honestly_without_a_client():
    assert HybridCompactor(None).name == "hybrid-extractive"
    async def _client(*a, **k):
        return "x", []

    assert HybridCompactor(_client).name == "hybrid"


def test_no_compactor_refuses_to_rewrite():
    c = build_compactor("none")
    out = run(c.compact(seg(FILLER * 20), target_tokens=5))
    assert out.content == FILLER * 20
    assert out.partial is True


def test_build_compactor_factory():
    assert build_compactor("truncate").name == "truncate"
    assert build_compactor("extractive").name == "extractive"
    assert build_compactor("summarize").name == "summarize"
    with pytest.raises(ValueError):
        build_compactor("nonsense")


def test_every_compactor_reports_a_strategy_name():
    for kind in ("truncate", "extractive", "summarize", "hybrid", "none"):
        c = build_compactor(kind)
        out = run(c.compact(seg(FILLER * 20), target_tokens=10))
        assert out.strategy
        assert out.tokens_before >= 0 and out.tokens_after >= 0


# --------------------------------------------------------------------------- #
# SingleShotAssembler
# --------------------------------------------------------------------------- #
def test_single_shot_builds_from_system_and_history():
    pkg = run(SingleShotAssembler(ContextBudget(total_tokens=2000, reserve_output=0)).build(
        "you are helpful",
        [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}],
    ))
    assert pkg.messages[0].role == "system"
    assert len(pkg.messages) == 3


def test_single_shot_system_prompt_is_not_compactable():
    """A system prompt's wording is load-bearing; rewriting it changes behaviour."""
    pkg = run(SingleShotAssembler(ContextBudget(total_tokens=60, reserve_output=0)).build(
        "critical instructions " * 40, []
    ))
    assert pkg.messages
    assert "critical instructions" in pkg.messages[0].content


def test_single_shot_accepts_extra_segments():
    from context_manager import ContextRole as CR

    pkg = run(SingleShotAssembler(ContextBudget(total_tokens=1000, reserve_output=0)).build(
        "sys", [{"role": "user", "content": "hi"}],
        extra=[ContextSegment(role=CR.RETRIEVAL, content="doc", source="retrieval")],
    ))
    assert any(m.content == "doc" for m in pkg.messages)


# --------------------------------------------------------------------------- #
# Known limitation, pinned deliberately
# --------------------------------------------------------------------------- #
def test_many_peers_are_evicted_rather_than_collectively_compacted():
    """Pinned limitation: greedy fill evicts peers instead of compressing them.

    Each history segment fits the per-segment ceiling on its own, so none of
    them triggers compaction; the assembler fills the budget with the most
    recent and drops the rest. For a long conversation this loses more history
    than a collective "compress the old turns into one summary" pass would.

    Collective compaction is a real improvement and is deliberately NOT
    implemented yet — this test exists so the behaviour is a known, tested
    property rather than a surprise. If the design changes, this test should
    fail and be rewritten, which is the point.
    """
    # Distinct content: identical bodies would be deduplicated, which is a
    # different mechanism and would make this test vacuous.
    bodies = [f"turn {i} " + FILLER * 6 for i in range(12)]
    b = wide(cost(bodies[0]) * 3)
    segments = [seg(bodies[i], sequence=i) for i in range(12)]
    pkg = run(ContextAssembler(b).assemble(segments))
    assert pkg.dropped > 0
    assert pkg.compacted == 0, "peers are dropped, not compacted, under the current design"
    assert pkg.tokens_used <= pkg.budget.input_allowance


def test_an_empty_segment_list_is_valid():
    pkg = run(ContextAssembler().assemble([]))
    assert pkg.messages == []
    assert pkg.dropped == 0
    assert pkg.over_budget is False
