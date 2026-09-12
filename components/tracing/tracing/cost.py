"""Token usage and cost accounting."""
from __future__ import annotations

from pydantic import BaseModel, Field


class Usage(BaseModel):
    """Token usage for a single model call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    model: str = "unknown"


# Default cost table: model name -> (usd per 1k prompt tokens, usd per 1k
# completion tokens). Local/free models are seeded at 0.0.
_DEFAULT_RATES: dict[str, tuple[float, float]] = {
    "qwen2.5-1.5b-instruct": (0.0, 0.0),
    "qwen2.5-32b-instruct": (0.0, 0.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4o": (2.5, 10.0),
}


class CostTable:
    """Maps model names to USD per 1k prompt / 1k completion tokens."""

    def __init__(self, rates: dict[str, tuple[float, float]] | None = None) -> None:
        self.rates: dict[str, tuple[float, float]] = dict(rates or _DEFAULT_RATES)

    def set(self, model: str, prompt_per_1k: float, completion_per_1k: float) -> None:
        self.rates[model] = (prompt_per_1k, completion_per_1k)

    def account(self, usage: Usage) -> float:
        """Return the USD cost for a single ``Usage`` record."""
        rates = self.rates.get(usage.model, (0.0, 0.0))
        prompt_rate, completion_rate = rates
        return (
            (usage.prompt_tokens / 1000.0) * prompt_rate
            + (usage.completion_tokens / 1000.0) * completion_rate
        )


class CostAccountant:
    """Accumulates token usage and cost totals, broken down by model."""

    def __init__(self, cost_table: CostTable | None = None) -> None:
        self.cost_table: CostTable = cost_table or CostTable()
        self._by_model: dict[str, dict[str, float]] = {}

    def add(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Record one usage record; return its cost in USD."""
        usage = Usage(model=model, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        cost = self.cost_table.account(usage)
        bucket = self._by_model.setdefault(
            model, {"prompt_tokens": 0.0, "completion_tokens": 0.0, "cost": 0.0}
        )
        bucket["prompt_tokens"] += prompt_tokens
        bucket["completion_tokens"] += completion_tokens
        bucket["cost"] += cost
        return cost

    def by_model(self) -> dict[str, dict[str, float]]:
        """Per-model totals: ``{model: {prompt_tokens, completion_tokens, cost}}``."""
        # Return copies so callers can't mutate our state.
        return {m: dict(v) for m, v in self._by_model.items()}

    def total_cost(self) -> float:
        """Total USD cost across all models."""
        return sum(v["cost"] for v in self._by_model.values())

    def reset(self) -> None:
        self._by_model.clear()