"""Teacher-vs-student parity evaluation — the go/no-go before serving.

Distillation is only worth doing if the student is *close enough* to the teacher
on quality while being materially cheaper and faster. This module measures all
three, and deliberately reports the quality **gap** rather than only the
student's absolute score: a student at 0.85 quality is excellent news if the
teacher is 0.86 and alarming if the teacher is 0.99.

Two kinds of number appear here, and they are not equally trustworthy:

* **Measured** — latency, token counts and cost come from real calls (or from
  the injected clients in tests). These are facts about the run.
* **Scored** — quality, stylistic similarity and readability are derived. When
  no judge is bound, quality falls back to overlap heuristics, and the report
  says so in ``notes`` rather than presenting a heuristic as a verdict.

Cost is computed from explicit per-1M-token prices rather than a hardcoded
table, because prices change and a stale constant would quietly corrupt every
savings figure downstream.
"""
from __future__ import annotations

import asyncio
import logging
import statistics
import time
from typing import Any, Awaitable, Callable, Sequence

from distillation.curator import readability_grade, token_overlap
from distillation.models import (
    CostProfile,
    DistillationConfig,
    LatencyProfile,
    ParityReport,
)

logger = logging.getLogger(__name__)

__all__ = ["ParityEvaluator", "CallableClient", "DEFAULT_PRICES"]

#: A model call: (prompt, **kw) -> (content, latency_s, tokens_in, tokens_out).
#: Kept explicit so tests can inject deterministic numbers instead of mocking
#: an HTTP client.
CallableClient = Callable[..., Awaitable[tuple[str, float, int, int]]]


#: Illustrative list prices, USD per 1M tokens. Callers should override with
#: their own contracted rates — these exist so the evaluator has a sensible
#: default, not because they are authoritative.
DEFAULT_PRICES: dict[str, tuple[float, float]] = {
    "teacher": (2.50, 10.00),   # frontier model
    "student": (0.10, 0.40),    # small self-hosted-equivalent
}


class ParityEvaluator:
    """Compare a student model against its teacher.

    Parameters
    ----------
    teacher:
        Callable client for the teacher (alias/route ``config.teacher_model``).
    student:
        Callable client for the distilled student.
    judge:
        Optional async ``(prompt, a, b) -> dict`` comparing two answers. When
        absent, quality uses overlap heuristics and the report says so.
    prices:
        ``{"teacher": (in_per_mtok, out_per_mtok), "student": (...)}``.
    """

    def __init__(
        self,
        *,
        teacher: CallableClient | None = None,
        student: CallableClient | None = None,
        judge: Any | None = None,
        prices: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self.teacher = teacher
        self.student = student
        self.judge = judge
        self.prices = prices or DEFAULT_PRICES

    # ---- measurement ------------------------------------------------------ #
    @staticmethod
    def _profile(latencies: Sequence[float], ttfts: Sequence[float],
                 tin: int, tout: int, price: tuple[float, float]) -> tuple[LatencyProfile, CostProfile]:
        return (
            LatencyProfile(
                ttft_ms=statistics.fmean(ttfts) * 1000 if ttfts else 0.0,
                total_ms=statistics.fmean(latencies) * 1000 if latencies else 0.0,
                samples=len(latencies),
            ),
            CostProfile(
                input_cost_per_mtok=price[0],
                output_cost_per_mtok=price[1],
                tokens_in=tin,
                tokens_out=tout,
            ),
        )

    async def _score_quality(self, prompt: str, answer: str) -> float:
        """Quality of one answer, 0..1.

        With a judge this is the judge's verdict; without one it is topic overlap
        between prompt and answer, scaled so a real connection scores respectably
        (overlap alone would punish correct terse answers).
        """
        if self.judge is not None:
            try:
                verdict = await self.judge(prompt, answer)
                return float(verdict.get("quality", verdict.get("score", 0.0)))
            except Exception as e:  # noqa: BLE001
                logger.warning("parity judge failed, using heuristic: %s", e)
        return min(1.0, 0.4 + 0.6 * token_overlap(prompt, answer))

    # ---- the evaluation --------------------------------------------------- #
    async def evaluate(
        self,
        prompts: Sequence[str],
        config: DistillationConfig,
        *,
        concurrency: int = 4,
    ) -> ParityReport:
        """Run both models over `prompts` and build the parity report.

        Prompts the teacher answered but the student failed on are excluded from
        the pairwise metrics and counted in ``notes`` — averaging a missing
        student answer in as a zero would fabricate a quality gap that is really
        an availability problem.
        """
        report = ParityReport(
            teacher_model=config.teacher_model,
            student_model=config.student_base_model,
        )
        if not prompts:
            report.notes.append("no prompts supplied; nothing evaluated")
            return report
        if self.teacher is None or self.student is None:
            report.notes.append(
                "teacher and/or student client not bound; parity not measured"
            )
            return report

        sem = asyncio.Semaphore(max(1, concurrency))

        async def _call(client: CallableClient, prompt: str) -> tuple[str, float, int, int] | None:
            async with sem:
                try:
                    return await client(prompt)
                except Exception as e:  # noqa: BLE001
                    logger.warning("parity call failed for %r: %s", prompt[:40], e)
                    return None

        t_results = await asyncio.gather(*(_call(self.teacher, p) for p in prompts))
        s_results = await asyncio.gather(*(_call(self.student, p) for p in prompts))

        t_lat: list[float] = []
        s_lat: list[float] = []
        # TTFT is not separable from a single (content, latency) call, so the
        # end-to-end latency is used as the TTFT proxy and the report says so.
        t_ttft: list[float] = []
        s_ttft: list[float] = []
        t_in = t_out = s_in = s_out = 0
        t_scores: list[float] = []
        s_scores: list[float] = []
        similarities: list[float] = []
        failed = 0

        for prompt, tr, sr in zip(prompts, t_results, s_results):
            if tr is None or sr is None:
                failed += 1
                continue
            t_text, t_latency, t_tok_in, t_tok_out = tr
            s_text, s_latency, s_tok_in, s_tok_out = sr

            t_lat.append(t_latency); s_lat.append(s_latency)
            t_ttft.append(t_latency); s_ttft.append(s_latency)
            t_in += t_tok_in; t_out += t_tok_out
            s_in += s_tok_in; s_out += s_tok_out

            t_q = await self._score_quality(prompt, t_text)
            s_q = await self._score_quality(prompt, s_text)
            t_scores.append(t_q); s_scores.append(s_q)
            similarities.append(token_overlap(t_text, s_text))

            report.per_example.append({
                "prompt": prompt[:200],
                "teacher_quality": round(t_q, 4),
                "student_quality": round(s_q, 4),
                "similarity": round(similarities[-1], 4),
                "teacher_latency_ms": round(t_latency * 1000, 2),
                "student_latency_ms": round(s_latency * 1000, 2),
            })

        if not t_lat:
            report.notes.append("every comparison failed; no metrics computed")
            report.notes.append(f"{failed} of {len(prompts)} prompts had a failed call")
            return report

        report.examples = len(t_lat)
        report.teacher_quality = round(statistics.fmean(t_scores), 4)
        report.student_quality = round(statistics.fmean(s_scores), 4)
        report.stylistic_similarity = round(statistics.fmean(similarities), 4)

        report.teacher_latency, report.teacher_cost = self._profile(
            t_lat, t_ttft, t_in, t_out, self.prices.get("teacher", DEFAULT_PRICES["teacher"])
        )
        report.student_latency, report.student_cost = self._profile(
            s_lat, s_ttft, s_in, s_out, self.prices.get("student", DEFAULT_PRICES["student"])
        )

        # Readability of the student's prose vs the teacher's, over the answers
        # that survived, so a format drift is visible.
        s_texts = [str(r) for r in s_results if r is not None]
        t_texts = [str(r) for r in t_results if r is not None]
        if s_texts and t_texts:
            report.readability_grade = round(
                statistics.fmean([readability_grade(t[0]) for t in s_texts if t]), 2
            )
            report.teacher_readability_grade = round(
                statistics.fmean([readability_grade(t[0]) for t in t_texts if t]), 2
            )

        # Method honesty: state what kind of numbers these are.
        if self.judge is None:
            report.notes.append(
                "Quality is a token-overlap heuristic (no judge bound); treat the "
                "gap as indicative, not a verdict."
            )
        report.notes.append(
            "TTFT is approximated by end-to-end latency: a single completion call "
            "does not expose first-token timing. Bind a streaming client for a "
            "true TTFT."
        )
        if failed:
            report.notes.append(
                f"{failed} of {len(prompts)} prompts excluded (one side failed), so "
                "pairwise metrics cover only the {n} completed pairs".format(n=report.examples)
            )
        return report

    # ---- comparison ------------------------------------------------------- #
    @staticmethod
    def compare_reports(a: ParityReport, b: ParityReport) -> dict[str, Any]:
        """Diff two runs — useful for comparing LoRA r=8 against r=32.

        Returns the deltas, with the better run named per dimension. Naming the
        winner is the whole point: a table of deltas still leaves the reader to
        work out which is preferable and by how much it matters.
        """
        out: dict[str, Any] = {
            "quality_delta": round(b.student_quality - a.student_quality, 4),
            "retention_delta": round(b.quality_retention - a.quality_retention, 4),
            "stylistic_delta": round(b.stylistic_similarity - a.stylistic_similarity, 4),
            "speedup_delta": round(b.latency_speedup - a.latency_speedup, 4),
            "savings_delta": round(b.cost_savings_pct - a.cost_savings_pct, 2),
        }
        out["better_quality"] = "b" if out["quality_delta"] > 0 else "a" if out["quality_delta"] < 0 else "tie"
        out["better_stylistic"] = "b" if out["stylistic_delta"] > 0 else "a" if out["stylistic_delta"] < 0 else "tie"
        out["better_retention"] = "b" if out["retention_delta"] > 0 else "a" if out["retention_delta"] < 0 else "tie"
        return out

    # ---- convenience ------------------------------------------------------ #
    @staticmethod
    def make_router_client(alias_router: Any, alias: str) -> CallableClient:
        """Adapt a ``model_gateway`` Gateway into the evaluator's call shape.

        Measures wall-clock latency and counts tokens from the returned text, so
        the profiles are populated for a real gateway without the caller writing
        plumbing.
        """

        async def _client(prompt: str, **_: Any) -> tuple[str, float, int, int]:
            from components_core import Message

            t0 = time.perf_counter()
            content, _calls = await alias_router.chat(
                [Message(role="user", content=prompt)]
            )
            latency = time.perf_counter() - t0
            # Rough token estimate: a real run should report usage from the API.
            return content or "", latency, len(prompt.split()), len((content or "").split())

        return _client

    @staticmethod
    def make_static_client(response: str, *, latency: float = 0.05, tokens_in: int = 20, tokens_out: int = 40) -> CallableClient:
        """A deterministic client, for tests and offline demos."""

        async def _client(prompt: str, **_: Any) -> tuple[str, float, int, int]:
            return response, latency, tokens_in, tokens_out

        return _client
