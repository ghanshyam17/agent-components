"""Lightweight per-endpoint counters."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EndpointMetrics:
    name: str
    requests: int = 0
    errors: int = 0
    successes: int = 0
    total_latency: float = 0.0

    def record(self, latency: float, ok: bool) -> None:
        self.requests += 1
        self.total_latency += latency
        if ok:
            self.successes += 1
        else:
            self.errors += 1

    @property
    def avg_latency(self) -> float:
        return self.total_latency / self.requests if self.requests else 0.0

    def snapshot(self) -> dict:
        return {
            "name": self.name,
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
            "avg_latency": round(self.avg_latency, 4),
        }