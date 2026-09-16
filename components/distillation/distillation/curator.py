"""Curation: turn raw teacher output into a dataset worth training on.

Synthetic data is cheap and mostly bad. The curator is the gate that decides
what survives, and it composes the components that already own each concern
rather than reimplementing them:

============================  =============================================
Concern                       Provided by
============================  =============================================
PII redaction, injection      ``guardrails.GuardrailPipeline``
blocking, length limits
Groundedness / coherence /    ``eval_harness`` metrics (with a pluggable
fluency scoring               judge; heuristic defaults when none is bound)
Near-duplicate removal        ``memory_store.VectorMemory`` cosine similarity
Preference pairs for DPO      candidate completions from the synthesizer,
                              scored and paired here
============================  =============================================

Every decision is recorded in a :class:`CurationReport`, and quarantined samples
are returned rather than dropped, so a reviewer can see *why* a dataset is the
size it is.
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any, Sequence

from distillation.models import (
    CurationReport,
    ContentSample,
    DPOPair,
    DatasetFormat,
    DistillationConfig,
    QualityScore,
)

logger = logging.getLogger(__name__)

__all__ = ["ContentCurator", "readability_grade", "token_overlap"]

_WORD_RE = re.compile(r"[A-Za-zÀ-ÿ']+")
_SENTENCE_RE = re.compile(r"[.!?]+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text or "")


def _syllables(word: str) -> int:
    """Rough English syllable count (vowel groups, silent-final-e adjusted).

    Flesch-Kincaid only needs a good approximation; a dictionary-based counter
    would not change the grade enough to matter for a quality gate.
    """
    w = word.lower()
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)


def readability_grade(text: str) -> float:
    """Flesch-Kincaid grade level. Higher means harder to read.

    ``0.39*(words/sentences) + 11.8*(syllables/words) - 15.59``. Returns 0.0 for
    text too short to score, rather than a misleading number.
    """
    words = _words(text)
    if not words:
        return 0.0
    sentences = max(1, len(_SENTENCE_RE.findall(text or "")))
    syllables = sum(_syllables(w) for w in words)
    return max(
        0.0,
        0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59,
    )


#: Words too common, or too much a part of instruction boilerplate, to indicate
#: whether an answer addressed the question.
#:
#: The instructional verbs (`explain`, `summarise`, `describe`, ...) and
#: audience/pedagogy words (`beginner`, `competent`, `concept`, `step`, ...)
#: matter here specifically because the blueprints *generate* prompts from them:
#: "Explain the concept of X to a competent beginner" is mostly boilerplate, and
#: a correct answer properly never repeats the words "concept" or "beginner".
#: Leaving them in would penalise good answers for being on-topic, which is how
#: a whole batch scored 0.58 against a 0.6 gate and was quarantined.
_STOPWORDS = {
    # grammatical
    "the", "a", "an", "and", "or", "of", "to", "in", "is", "are", "was", "were",
    "for", "on", "with", "as", "by", "at", "it", "its", "this", "that", "these",
    "those", "be", "been", "from", "how", "what", "why", "when", "which", "you",
    "your", "can", "does", "do", "about", "into", "than", "then", "also", "not",
    # instructional boilerplate introduced by the style blueprints
    "explain", "describe", "summarise", "summarize", "construct", "demonstrate",
    "help", "understand", "work", "worked", "example", "concept", "topic",
    "beginner", "competent", "learner", "step", "steps", "sequence", "approach",
    "common", "mistake", "correct", "state", "present", "show", "shown", "give",
    "reason", "reasoning", "think", "first", "finally", "clear", "essential",
    "structured", "list", "fully", "each", "use", "using", "regarding", "people",
}


def token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of word sets — symmetric stylistic similarity.

    ``|A ∩ B| / |A ∪ B|``. Symmetric, so it is the right measure for comparing
    two answers of comparable length (the parity evaluator's job). It is the
    *wrong* measure for groundedness: Jaccard penalises a thorough answer simply
    for containing new words, so a short prompt against a complete explanation
    scores low even when every prompt term was addressed. Use
    :func:`prompt_coverage` for that.
    """
    sa = {w.lower() for w in _words(a)}
    sb = {w.lower() for w in _words(b)}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def prompt_coverage(prompt: str, answer: str) -> float:
    """Share of the prompt's content words the answer addresses.

    ``|prompt_terms ∩ answer| / |prompt_terms|``, ignoring stopwords. Asymmetric
    by design: it asks "did the answer engage with what was asked?", so a long
    and complete answer is not penalised for its length.
    """
    p_terms = {w.lower() for w in _words(prompt)} - _STOPWORDS
    if not p_terms:
        # Nothing substantive to cover (e.g. "explain this"); fall back to
        # symmetric overlap rather than reporting a meaningless 0.0.
        return token_overlap(prompt, answer)
    a_terms = {w.lower() for w in _words(answer)}
    return len(p_terms & a_terms) / len(p_terms)


class ContentCurator:
    """Filter, score, deduplicate and format synthetic samples.

    Parameters
    ----------
    pipeline:
        A ``guardrails.GuardrailPipeline``. When omitted, PII redaction and
        injection detection are configured inline so the curator is safe by
        default — a curator that silently skips safety because nothing was wired
        is worse than no curator.
    vector_memory:
        A ``memory_store.VectorMemory`` for dedup. Defaults to an in-memory
        store with a ``HashingEmbedder``, which is deterministic and needs no
        network. Inject a real embedder for semantic dedup.
    judge:
        Optional async ``(prompt, completion) -> dict`` groundedness judge. When
        absent, heuristic scores are used and the report says so.
    """

    def __init__(
        self,
        *,
        pipeline: Any | None = None,
        vector_memory: Any | None = None,
        judge: Any | None = None,
        embedder: Any | None = None,
    ) -> None:
        self.pipeline = pipeline or self._default_pipeline()
        self.vector_memory = vector_memory
        self._embedder = embedder
        self.judge = judge

    @staticmethod
    def _default_pipeline() -> Any:
        """Build a safe-by-default guardrail pipeline via the component's own
        settings (so env vars like ``GUARDRAIL_*`` still apply)."""
        try:
            from guardrails import build_pipeline

            return build_pipeline()
        except Exception as e:  # noqa: BLE001
            logger.warning("guardrails unavailable (%s); safety scoring disabled", e)
            return None

    async def _memory(self) -> Any:
        """Lazily build the dedup store (needs an await to embed)."""
        if self.vector_memory is not None:
            return self.vector_memory
        from memory_store import InMemoryVectorMemory

        if self._embedder is None:
            from memory_store.embeddings import HashingEmbedder

            self._embedder = HashingEmbedder(dim=256)
        self.vector_memory = InMemoryVectorMemory(embedder=self._embedder)
        return self.vector_memory

    # ---- hygiene ---------------------------------------------------------- #
    async def sanitize(self, sample: ContentSample) -> tuple[ContentSample, dict[str, Any]]:
        """Run guardrails over both the prompt and the completion.

        Both fields matter: a clean answer to a prompt that carried PII still
        leaks, and a teacher that echoes an injection into its answer would
        train the student to do the same.
        """
        info: dict[str, Any] = {"pii_redacted": False, "injection_blocked": False, "reasons": [], "flags": []}

        if self.pipeline is None:
            return sample, info

        for field_name in ("prompt", "completion", "teacher_reasoning"):
            value = getattr(sample, field_name, None)
            if not value:
                continue
            try:
                result = await self.pipeline.check_input(value)
            except Exception as e:  # noqa: BLE001 - never fail the batch on a guardrail
                logger.warning("guardrail error on %s: %s", field_name, e)
                continue
            if result.sanitized and result.sanitized != value:
                setattr(sample, field_name, result.sanitized)
                if "pii" in result.flags:
                    info["pii_redacted"] = True
            info["reasons"].extend(result.reasons or [])
            info["flags"].extend(result.flags or [])
            if result.blocked:
                info["injection_blocked"] = True
        return sample, info

    # ---- scoring ---------------------------------------------------------- #
    async def score(self, sample: ContentSample, config: DistillationConfig) -> QualityScore:
        """Score one sample on groundedness, coherence and fluency.

        * **groundedness** — is the answer supported? With a judge bound, the
          judge's verdict is used; otherwise it is approximated by topic overlap
          between prompt and completion (a weak but honest proxy, and the report
          labels it as heuristic).
        * **coherence** — does the reasoning (when present) actually connect to
          the answer? Measured as overlap between trace and completion.
        * **fluency** — readability within a sensible band. Both too-hard and
          too-simple output is penalised, since a student trained on either
          inherits the flaw.
        """
        reasons: list[str] = []
        flags: list[str] = []

        # -- groundedness --
        grounded = 0.0
        if self.judge is not None:
            try:
                verdict = await self.judge(sample.prompt, sample.completion)
                grounded = float(verdict.get("groundedness", 0.0))
                if verdict.get("unsupported_claims"):
                    flags.append("unsupported_claims")
                    reasons.append(f"{len(verdict['unsupported_claims'])} unsupported claim(s)")
            except Exception as e:  # noqa: BLE001
                logger.warning("judge failed, falling back to heuristic: %s", e)
                flags.append("judge_failed")
                grounded = prompt_coverage(sample.prompt, sample.completion)
        else:
            grounded = prompt_coverage(sample.prompt, sample.completion)
            flags.append("heuristic_groundedness")

        # -- coherence --
        if sample.teacher_reasoning:
            # Reasoning that shares nothing with the answer suggests the trace
            # was fabricated or the answer drifted away from the plan.
            coherence = prompt_coverage(sample.teacher_reasoning, sample.completion)
            # Overlap alone penalises correct concise answers, so scale it up:
            # any real connection implies a decent score.
            coherence = min(1.0, 0.4 + 0.6 * coherence)
        else:
            # No trace: cannot assess, so do not penalise — many tasks are
            # legitimately single-step.
            coherence = 0.7
            flags.append("no_reasoning_trace")

        # -- fluency --
        grade = readability_grade(sample.completion)
        if grade <= 4.0:
            fluency = 1.0
        elif grade >= 20.0:
            fluency = 0.2
        else:
            # Peak at grade ~10, decaying both ways.
            fluency = 1.0 - abs(grade - 10.0) / 10.0
        fluency = max(0.0, min(1.0, fluency))

        # -- length sanity --
        n_words = len(_words(sample.completion))
        if n_words < 5:
            flags.append("too_short")
            reasons.append(f"completion has only {n_words} words")
        elif n_words > 1200:
            flags.append("too_long")
            reasons.append(f"completion has {n_words} words")

        overall = (grounded + coherence + fluency) / 3.0
        if "too_short" in flags:
            overall *= 0.5
        elif "too_long" in flags:
            overall *= 0.8

        decision = "pass" if overall >= config.min_quality else "quarantine"
        if decision == "quarantine":
            reasons.append(f"overall {overall:.3f} below min_quality {config.min_quality}")

        return QualityScore(
            groundedness=round(grounded, 4),
            coherence=round(coherence, 4),
            fluency=round(fluency, 4),
            safety_passed=True,  # set by the caller; safety blocks are decided in sanitize
            overall=round(max(0.0, min(1.0, overall)), 4),
            decision=decision,
            reasons=reasons,
            flags=flags,
        )

    # ---- dedup ------------------------------------------------------------ #
    async def deduplicate(
        self, samples: Sequence[ContentSample], config: DistillationConfig
    ) -> tuple[list[ContentSample], int]:
        """Drop near-duplicates by cosine similarity; returns (kept, dropped).

        Uses the vector store rather than an O(n^2) string comparison, so this
        scales to the sizes distillation actually produces.
        """
        if not samples:
            return [], 0
        memory = await self._memory()
        from memory_store.models import MemoryQuery, MemoryRecord

        kept: list[ContentSample] = []
        dropped = 0
        for s in samples:
            try:
                hits = await memory.search(MemoryQuery(query=s.text, top_k=1, min_score=-1.0))
            except Exception as e:  # noqa: BLE001
                logger.warning("dedup search failed (%s); keeping sample", e)
                kept.append(s)
                continue
            if hits and hits[0][1] >= config.dedupe_threshold:
                dropped += 1
                continue
            kept.append(s)
            try:
                await memory.add([MemoryRecord(id=s.id, content=s.text, metadata={"topic": s.metadata.get("topic", "")})])
            except Exception as e:  # noqa: BLE001
                logger.debug("dedup add failed: %s", e)
        return kept, dropped

    # ---- the pipeline ----------------------------------------------------- #
    async def curate(
        self,
        samples: Sequence[ContentSample],
        config: DistillationConfig,
    ) -> tuple[list[ContentSample], list[dict[str, Any]], CurationReport]:
        """Run the full hygiene → score → dedup gate.

        Returns ``(accepted, quarantined, report)``. Quarantined entries carry
        their sample id and score so a reviewer can inspect them; they are never
        silently discarded.
        """
        start = time.time()
        report = CurationReport(total_in=len(samples))
        accepted: list[ContentSample] = []
        quarantined: list[dict[str, Any]] = []
        reason_counts: dict[str, int] = {}

        scored: list[ContentSample] = []
        for sample in samples:
            clean, info = await self.sanitize(sample)
            if info.get("pii_redacted"):
                report.pii_redacted += 1
            if info.get("injection_blocked"):
                report.injections_blocked += 1

            score = await self.score(clean, config)
            if info.get("injection_blocked"):
                score.safety_passed = False
                score.decision = "quarantine"
                score.flags = sorted(set([*score.flags, "injection"]))
                score.reasons.append("prompt-injection pattern detected")

            clean.metadata["quality"] = score.to_dict()

            if score.decision == "pass":
                scored.append(clean)
            else:
                report.quarantined += 1
                for reason in score.reasons or ["unspecified"]:
                    key = reason.split(":")[0][:60]
                    reason_counts[key] = reason_counts.get(key, 0) + 1
                quarantined.append({"sample_id": clean.id, "score": score.to_dict()})

        kept, dropped = await self.deduplicate(scored, config)
        report.deduplicated = dropped
        report.passed = len(kept)
        report.quarantine_reasons = reason_counts
        report.duration_ms = int((time.time() - start) * 1000)
        return kept, quarantined, report

    # ---- DPO pairs -------------------------------------------------------- #
    async def build_dpo_pairs(
        self,
        candidates: Sequence[tuple[str, list[str]]],
        config: DistillationConfig,
        *,
        min_margin: float = 0.05,
    ) -> tuple[list[DPOPair], int]:
        """Turn scored candidate completions into preference pairs.

        The best and worst candidate become ``chosen`` and ``rejected``. Pairs
        whose margin falls below ``min_margin`` are dropped: near-ties encode
        mostly noise, and DPO trains on exactly the difference it is shown.
        """
        pairs: list[DPOPair] = []
        dropped = 0
        for prompt, texts in candidates:
            if len(texts) < 2:
                dropped += 1
                continue
            scored: list[tuple[float, str]] = []
            for text in texts:
                probe = ContentSample(prompt=prompt, completion=text)
                score = await self.score(probe, config)
                scored.append((score.overall, text))
            scored.sort(key=lambda t: t[0], reverse=True)
            best_score, best = scored[0]
            worst_score, worst = scored[-1]
            margin = best_score - worst_score
            if margin < min_margin or best == worst:
                dropped += 1
                continue
            pairs.append(
                DPOPair(
                    prompt=prompt,
                    chosen=best,
                    rejected=worst,
                    margin=round(margin, 4),
                    metadata={"candidates": len(texts)},
                )
            )
        return pairs, dropped

    # ---- serialisation ---------------------------------------------------- #
    @staticmethod
    def to_records(samples: Sequence[ContentSample], config: DistillationConfig, *, system: str | None = None) -> list[dict[str, Any]]:
        """Render samples in the configured dataset format."""
        fmt = config.format
        if fmt is DatasetFormat.ALPACA:
            return [s.to_alpaca() for s in samples]
        if fmt is DatasetFormat.CHATML:
            return [s.to_chatml(system=system) for s in samples]
        if fmt is DatasetFormat.DPO:
            # A ContentSample has no chosen/rejected, so it cannot be expressed
            # as a DPO row. Returning raw records here would silently hand a
            # wrong-schema file to a preference-optimisation job, so fail loudly.
            raise ValueError(
                "format='dpo' cannot be rendered from ContentSamples: build DPOPairs "
                "via ContentCurator.build_dpo_pairs() and serialise those instead."
            )
        return [s.to_jsonl() for s in samples]
