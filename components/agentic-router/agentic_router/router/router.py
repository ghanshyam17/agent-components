"""The hybrid router: heuristic first, classifier fallback for ambiguous cases.

Decision rule:
  * score <= lower_threshold          -> LOWER  (heuristic)
  * score >= higher_threshold         -> HIGHER (heuristic)
  * lower_threshold < score < higher_threshold -> classifier (LLM-as-judge)
  * classifier_band extends the uncertain region: a score within `band` of
    EITHER threshold is also sent to the classifier, so near-threshold scores
    (which the heuristic is least reliable about) get a second opinion.
"""
from __future__ import annotations

from agentic_router.clients import ModelClient
from agentic_router.config import Settings
from agentic_router.models import ModelTier, RouteDecision
from agentic_router.router.classifier import classify
from agentic_router.router.heuristic import score_task


class Router:
    def __init__(self, settings: Settings, lower_client: ModelClient):
        self.s = settings
        self.lower_client = lower_client

    def _needs_classifier(self, score: float) -> bool:
        # Between the thresholds, or within `band` of either threshold.
        if self.s.router_lower_threshold < score < self.s.router_higher_threshold:
            return True
        band = self.s.router_classifier_band
        near_lower = abs(score - self.s.router_lower_threshold) < band
        near_higher = abs(score - self.s.router_higher_threshold) < band
        return near_lower or near_higher

    async def route(self, task: str) -> RouteDecision:
        score, signals = score_task(task)

        if not self._needs_classifier(score):
            # Clear-cut: pure heuristic decision.
            if score <= self.s.router_lower_threshold:
                tier, model, reason = (
                    ModelTier.LOWER, self.s.lower_model, "below_lower_threshold"
                )
            else:
                tier, model, reason = (
                    ModelTier.HIGHER, self.s.higher_model, "above_higher_threshold"
                )
            return RouteDecision(
                tier=tier, score=score, method="heuristic", model=model,
                reason=reason, signals=signals,
            )

        # Ambiguous — ask the classifier (uses the lower model).
        tier, reason = await classify(task, self.lower_client)
        model = self.s.lower_model if tier is ModelTier.LOWER else self.s.higher_model
        return RouteDecision(
            tier=tier, score=score, method="classifier",
            model=model, reason=reason, signals=signals,
        )