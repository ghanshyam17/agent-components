"""Tests for the heuristic router and hybrid Router."""
from __future__ import annotations

import asyncio

from agentic_router.config import Settings
from agentic_router.models import ModelTier
from agentic_router.router.heuristic import score_task
from agentic_router.router.router import Router


# Prompts chosen to span the heuristic's score distribution (verified against
# the scorer): ~0.0 for greetings, ~0.2 for single-cue, ~0.5+ for loaded tasks.
GREETING = "hi there"
SINGLE_CUE = "debug and refactor the function"          # ~0.21  -> between thresholds -> classifier
LOADED = (                                            # ~0.53  -> >= higher threshold -> HIGHER heuristic
    "Debug this stack trace, design a step-by-step plan to refactor the "
    "authentication module, then implement an algorithm and run tests. "
    "```\ndef f(): pass\n```\n Please read the file, search the web, "
    "and finally optimize the regex and compare results."
)


class _FakeLowerClient:
    """Stub for the lower model client used by the classifier."""

    def __init__(self, tier_response: ModelTier = ModelTier.HIGHER):
        self.tier_response = tier_response
        self.calls: list[str] = []

    async def chat(self, messages, **kwargs):
        self.calls.append(messages[-1].content)
        return ('{"tier": "%s", "reason": "fake"}' % self.tier_response.value), []


def test_simple_greeting_scores_low():
    score, signals = score_task(GREETING)
    assert score <= 0.15, (score, signals)
    assert abs(signals["total"] - round(score, 3)) < 1e-9


def test_loaded_task_scores_high():
    score, signals = score_task(LOADED)
    assert score >= 0.45, (score, signals)


def test_single_cue_lands_in_middle():
    score, _ = score_task(SINGLE_CUE)
    assert 0.15 < score < 0.45, score


def test_code_fence_raises_code_signal():
    task = "```python\ndef f(x):\n    return x*2\n```\n explain"
    score, signals = score_task(task)
    assert signals["code"] > 0
    assert score > 0


def test_router_clear_cut_lower():
    s = Settings()  # defaults: 0.15 / 0.45
    router = Router(s, _FakeLowerClient())
    res = asyncio.get_event_loop().run_until_complete(router.route(GREETING))
    assert res.tier is ModelTier.LOWER
    assert res.method == "heuristic"


def test_router_clear_cut_higher():
    s = Settings()
    router = Router(s, _FakeLowerClient())
    res = asyncio.get_event_loop().run_until_complete(router.route(LOADED))
    assert res.tier is ModelTier.HIGHER
    assert res.method == "heuristic"


def test_router_uses_classifier_in_middle():
    s = Settings()
    fake = _FakeLowerClient(tier_response=ModelTier.HIGHER)
    router = Router(s, fake)
    res = asyncio.get_event_loop().run_until_complete(router.route(SINGLE_CUE))
    assert res.method == "classifier"
    assert res.tier is ModelTier.HIGHER
    assert len(fake.calls) == 1


def test_router_classifier_defaults_higher_on_error():
    class _ErrClient:
        async def chat(self, messages, **kwargs):
            raise RuntimeError("boom")

    s = Settings()
    router = Router(s, _ErrClient())
    res = asyncio.get_event_loop().run_until_complete(router.route(SINGLE_CUE))
    assert res.tier is ModelTier.HIGHER
    assert res.method == "classifier"