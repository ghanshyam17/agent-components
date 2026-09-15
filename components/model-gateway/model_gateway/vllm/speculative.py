"""Speculative decoding: pair a small draft model with a large target.

Speculative decoding generates ``k`` candidate tokens with a cheap *draft*
model, then verifies all of them in a **single** forward pass of the expensive
*target* model. Accepted tokens are effectively free — one target pass produced
several tokens instead of one — so latency drops without changing the output
distribution, provided verification is exact.

The economics are entirely in the **acceptance rate**:

* High acceptance (say ≥0.7) → most drafted tokens survive, target passes are
  amortised, and throughput rises roughly ``1 + acceptance * k``-fold.
* Low acceptance → you pay draft cost *and* target cost for no gain, and
  speculation is strictly worse than plain decoding.

:class:`SpeculativeDecodingCoordinator` therefore (a) decides whether to
speculate at all, (b) picks ``k`` from the observed acceptance rate, and (c)
tracks per-pair statistics so the decision is evidence-based rather than a
static config. It is deliberately model-agnostic: the draft and target are
described by references (endpoint names), and the actual propose/verify
callables are injected by the caller — which keeps this unit-testable with no
GPU.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "SpeculationStats",
    "SpeculativeDecodingCoordinator",
    "Proposer",
    "Verifier",
]

# A proposal is a list of token ids/strings; the verifier returns, per
# position, whether the draft token was accepted, and the corrected tokens.
Proposer = Callable[[Sequence[Any], int], Awaitable[list[Any]]]
Verifier = Callable[[Sequence[Any], list[Any]], Awaitable[tuple[list[Any], int]]]


@dataclass
class SpeculationStats:
    """Acceptance statistics for one (draft, target) pair.

    ``acceptance_rate`` is a decaying average so a change in traffic mix is
    reflected instead of being averaged away over the process lifetime.
    """

    draft_ref: str
    target_ref: str
    rounds: int = 0
    proposed: int = 0
    accepted: int = 0
    target_passes: int = 0
    acceptance_rate: float = 0.0
    last_updated: float | None = None

    def record(self, proposed: int, accepted: int, *, decay: float = 0.8) -> None:
        self.rounds += 1
        self.proposed += proposed
        self.accepted += accepted
        self.target_passes += 1
        sample = (accepted / proposed) if proposed else 0.0
        if self.rounds == 1:
            self.acceptance_rate = sample
        else:
            self.acceptance_rate = decay * self.acceptance_rate + (1 - decay) * sample
        self.last_updated = time.time()

    @property
    def tokens_per_target_pass(self) -> float:
        """Average tokens emitted per expensive target forward pass."""
        if not self.target_passes:
            return 0.0
        return self.accepted / self.target_passes

    def to_dict(self) -> dict[str, Any]:
        return {
            "draft_ref": self.draft_ref,
            "target_ref": self.target_ref,
            "rounds": self.rounds,
            "proposed": self.proposed,
            "accepted": self.accepted,
            "target_passes": self.target_passes,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "tokens_per_target_pass": round(self.tokens_per_target_pass, 3),
            "last_updated": self.last_updated,
        }


class SpeculativeDecodingCoordinator:
    """Decide when to speculate and with how many draft tokens.

    Parameters
    ----------
    min_acceptance:
        Below this observed acceptance rate, speculation is switched off — the
        draft cost is not being repaid.
    max_draft_tokens / min_draft_tokens:
        Bounds on ``k``.
    decay:
        Weight on the previous acceptance estimate when blending a new sample.
    probe_rounds:
        Before this many observations, speculate anyway (to gather evidence)
        as long as a draft is configured.
    """

    def __init__(
        self,
        *,
        min_acceptance: float = 0.5,
        min_draft_tokens: int = 1,
        max_draft_tokens: int = 8,
        decay: float = 0.8,
        probe_rounds: int = 3,
        enabled: bool = True,
    ) -> None:
        if min_draft_tokens < 1:
            raise ValueError("min_draft_tokens must be >= 1")
        if max_draft_tokens < min_draft_tokens:
            raise ValueError("max_draft_tokens must be >= min_draft_tokens")
        self.min_acceptance = min_acceptance
        self.min_draft_tokens = min_draft_tokens
        self.max_draft_tokens = max_draft_tokens
        self.decay = decay
        self.probe_rounds = probe_rounds
        self.enabled = enabled
        self._stats: dict[tuple[str, str], SpeculationStats] = {}
        # Optional injected behaviour; absent callables mean "report only".
        self.proposer: Proposer | None = None
        self.verifier: Verifier | None = None

    # -- wiring ------------------------------------------------------------- #
    def bind(self, proposer: Proposer | None, verifier: Verifier | None) -> None:
        """Attach the draft-proposal and target-verification callables."""
        self.proposer = proposer
        self.verifier = verifier

    # -- statistics --------------------------------------------------------- #
    def stats_for(self, draft_ref: str, target_ref: str) -> SpeculationStats:
        key = (draft_ref, target_ref)
        if key not in self._stats:
            self._stats[key] = SpeculationStats(draft_ref=draft_ref, target_ref=target_ref)
        return self._stats[key]

    def stats(self) -> dict[str, dict[str, Any]]:
        return {
            f"{d}->{t}": s.to_dict() for (d, t), s in sorted(self._stats.items())
        }

    def record(self, draft_ref: str, target_ref: str, proposed: int, accepted: int) -> SpeculationStats:
        """Record the outcome of one speculative round."""
        stat = self.stats_for(draft_ref, target_ref)
        stat.record(proposed, accepted, decay=self.decay)
        return stat

    # -- the decision ------------------------------------------------------- #
    def should_speculate(self, draft_ref: str | None, target_ref: str) -> bool:
        """Whether to use the draft model for the next request."""
        if not self.enabled or draft_ref is None:
            return False
        stat = self._stats.get((draft_ref, target_ref))
        if stat is None or stat.rounds < self.probe_rounds:
            return True  # not enough evidence yet — gather some
        return stat.acceptance_rate >= self.min_acceptance

    def draft_tokens(self, draft_ref: str | None, target_ref: str) -> int:
        """Choose ``k``: scale draft length with measured acceptance.

        At the acceptance floor we draft minimally (spend little), and as
        acceptance approaches 1.0 we draft the maximum — because near-perfect
        acceptance makes every extra drafted token nearly free.
        """
        if not self.should_speculate(draft_ref, target_ref) or draft_ref is None:
            return 0
        stat = self._stats.get((draft_ref, target_ref))
        if stat is None or stat.rounds < self.probe_rounds:
            return self.min_draft_tokens
        span = self.max_draft_tokens - self.min_draft_tokens
        # Map acceptance in [min_acceptance, 1.0] onto [min_k, max_k].
        scaled = (stat.acceptance_rate - self.min_acceptance) / max(
            1e-9, 1.0 - self.min_acceptance
        )
        scaled = min(max(scaled, 0.0), 1.0)
        return self.min_draft_tokens + round(scaled * span)

    def estimate_speedup(self, draft_ref: str, target_ref: str) -> float:
        """Rough tokens-per-target-pass, relative to plain decoding (=1.0).

        Returns 1.0 when speculation is off, since plain decoding emits exactly
        one token per target pass.
        """
        if not self.should_speculate(draft_ref, target_ref):
            return 1.0
        stat = self._stats.get((draft_ref, target_ref))
        if stat is None or not stat.target_passes:
            return 1.0
        return max(1.0, stat.tokens_per_target_pass)

    # -- execution ---------------------------------------------------------- #
    async def generate(
        self,
        prefix: Sequence[Any],
        draft_ref: str,
        target_ref: str,
    ) -> tuple[list[Any], SpeculationStats] | None:
        """Run one speculative round through the bound callables.

        Returns ``(accepted_tokens, stats)``, or None when speculation is off
        or the callables are not bound — the caller then falls back to plain
        decoding. The *verifier* is authoritative: whatever it rejects is
        discarded, so correctness never depends on the draft model.
        """
        if self.proposer is None or self.verifier is None:
            return None
        k = self.draft_tokens(draft_ref, target_ref)
        if k <= 0:
            return None
        proposal = await self.proposer(prefix, k)
        if not proposal:
            return None
        accepted_tokens, n_accepted = await self.verifier(prefix, list(proposal))
        stat = self.record(draft_ref, target_ref, len(proposal), n_accepted)
        return accepted_tokens, stat

    def summary(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "min_acceptance": self.min_acceptance,
            "draft_tokens_range": [self.min_draft_tokens, self.max_draft_tokens],
            "pairs": self.stats(),
        }
