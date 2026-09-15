"""Live vLLM telemetry: poll `/metrics` and `/health` for GPU-cache pressure.

vLLM exposes a Prometheus exposition endpoint at ``/metrics`` (and a liveness
``/health``). The values that matter for routing:

===============  ==========================================================
Metric           Why routing cares
===============  ==========================================================
``gpu_cache_usage_factor``
                 Fraction of the KV-cache blocks in use. At ~1.0 the replica
                 starts preempting sequences, so latency degrades sharply.
``num_requests_running``
                 Sequences currently decoding — real load, not just arrivals.
``num_requests_waiting``
                 Queue depth; the leading indicator of saturation.
``avg_prompt_throughput_tok_s`` / ``avg_generation_throughput_tok_s``
                 Throughput, used to compare replicas of unequal hardware.
===============  ==========================================================

:class:`VLLMMetricsCollector` polls asynchronously in the background and keeps
a snapshot per replica. Averages use **exponential decay** so a burst decays
instead of pinning the average for the process lifetime — the recent value
dominates, which is what a router wants. Health is tracked separately: a
replica that fails to answer ``/health`` is marked unhealthy even if its last
metrics looked fine.

The HTTP call is injected as an async callable, so this module imports and
unit-tests with no running vLLM and no HTTP dependency.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Mapping

logger = logging.getLogger(__name__)

__all__ = [
    "VLLMInstanceMetrics",
    "VLLMMetricsCollector",
    "parse_prometheus",
    "HEALTHY",
    "DEGRADED",
    "SATURATED",
    "UNHEALTHY",
]

DEFAULT_METRICS_URL = "http://localhost:8000/metrics"
DEFAULT_HEALTH_URL = "http://localhost:8000/health"

# Health states, coarse enough to drive routing and fine enough to explain it.
HEALTHY = "healthy"
DEGRADED = "degraded"
SATURATED = "saturated"
UNHEALTHY = "unhealthy"

# --- Prometheus exposition parsing ----------------------------------------- #
#   name{label="value",other="x"} 12.5
# Labels are optional; histogram/summary suffixes (_bucket/_sum/_count) are
# kept as distinct names, and only unlabelled or first-seen labelled samples
# are retained (vLLM's gauges of interest are unlabelled; the *_total counters
# are labelled by model_name, where any single sample is representative).
_SAMPLE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?"
    r"\s+(?P<value>[^\s]+)"
)


def parse_prometheus(text: str) -> dict[str, float]:
    """Parse Prometheus exposition text into ``{metric_name: value}``.

    Robust by design: comment lines (``#``) are skipped, unparseable values
    (``NaN``, ``+Inf``) are dropped rather than poisoning an average, and a
    repeated metric keeps its **first** sample so a multi-model replica gives
    a stable answer.
    """
    out: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if match is None:
            continue
        name = match.group("name")
        if name in out:
            continue  # keep the first sample for a repeated name
        try:
            value = float(match.group("value"))
        except (TypeError, ValueError):
            continue  # NaN / +Inf / garbage
        if value != value:  # NaN
            continue
        out[name] = value
    return out


@dataclass
class VLLMInstanceMetrics:
    """One replica's latest observed state.

    ``avg_*_throughput`` fields hold the *decayed* average maintained by the
    collector, not the raw sample; see :meth:`VLLMMetricsCollector._blend`.
    """

    name: str
    base_url: str = ""
    healthy: bool = True
    status: str = HEALTHY
    # KV-cache block utilisation, 0.0-1.0
    gpu_cache_usage_factor: float = 0.0
    # sequences decoding / queued
    num_requests_running: int = 0
    num_requests_waiting: int = 0
    avg_prompt_throughput_tok_s: float = 0.0
    avg_generation_throughput_tok_s: float = 0.0
    # bookkeeping
    last_success_at: float | None = None
    last_error: str | None = None
    polls: int = 0
    errors: int = 0
    samples: dict[str, float] = field(default_factory=dict)

    @property
    def queue_depth(self) -> int:
        """Waiting + running: total demand placed on the replica."""
        return self.num_requests_waiting + self.num_requests_running

    @property
    def pending(self) -> int:
        """Requests queued but not yet decoding — the saturation signal."""
        return self.num_requests_waiting

    def is_available(self) -> bool:
        return self.healthy and self.status != UNHEALTHY

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "base_url": self.base_url,
            "healthy": self.healthy,
            "status": self.status,
            "gpu_cache_usage_factor": round(self.gpu_cache_usage_factor, 4),
            "num_requests_running": self.num_requests_running,
            "num_requests_waiting": self.num_requests_waiting,
            "queue_depth": self.queue_depth,
            "avg_prompt_throughput_tok_s": round(self.avg_prompt_throughput_tok_s, 2),
            "avg_generation_throughput_tok_s": round(
                self.avg_generation_throughput_tok_s, 2
            ),
            "last_success_at": self.last_success_at,
            "last_error": self.last_error,
            "polls": self.polls,
            "errors": self.errors,
        }


# An injected async HTTP getter: fetch(url) -> text body.
Fetcher = Callable[[str], Awaitable[str]]


class VLLMMetricsCollector:
    """Background poller keeping per-replica vLLM telemetry.

    Usage::

        collector = VLLMMetricsCollector(
            {"vllm-a": "http://localhost:8000"}, fetcher=my_async_get
        )
        await collector.start()          # or: await collector.poll_once()
        ...
        best = collector.rank(["vllm-a", "vllm-b"])
        await collector.stop()

    ``decay`` is the exponential-decay factor for throughput averages: each new
    sample counts ``1 - decay`` and the running average counts ``decay``.
    ``0.0`` makes the average the last sample; ``0.9`` gives a ~10-sample
    effective window.
    """

    def __init__(
        self,
        endpoints: Mapping[str, str] | Iterable[tuple[str, str]],
        *,
        fetcher: Fetcher | None = None,
        interval: float = 2.0,
        decay: float = 0.7,
        cache_saturation: float = 0.85,
        queue_saturation: int = 8,
        metrics_path: str = "/metrics",
        health_path: str = "/health",
        timeout: float = 5.0,
    ) -> None:
        if isinstance(endpoints, Mapping):
            pairs = [(str(k), str(v)) for k, v in endpoints.items()]
        else:
            pairs = [(str(k), str(v)) for k, v in endpoints]
        self._metrics: dict[str, VLLMInstanceMetrics] = {
            name: VLLMInstanceMetrics(name=name, base_url=base_url.rstrip("/"))
            for name, base_url in pairs
        }
        self.interval = interval
        self.decay = min(max(decay, 0.0), 0.999)
        self.cache_saturation = cache_saturation
        self.queue_saturation = queue_saturation
        self.metrics_path = metrics_path
        self.health_path = health_path
        self.timeout = timeout
        self._fetcher = fetcher
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    # -- membership --------------------------------------------------------- #
    def register(self, name: str, base_url: str) -> None:
        self._metrics[name] = VLLMInstanceMetrics(name=name, base_url=base_url.rstrip("/"))

    def unregister(self, name: str) -> bool:
        return self._metrics.pop(name, None) is not None

    @property
    def names(self) -> list[str]:
        return list(self._metrics)

    def get(self, name: str) -> VLLMInstanceMetrics | None:
        return self._metrics.get(name)

    def snapshot(self) -> dict[str, dict[str, Any]]:
        return {n: m.to_dict() for n, m in self._metrics.items()}

    # -- the default fetcher (stdlib-free of new deps) ---------------------- #
    async def _default_fetcher(self, url: str) -> str:
        """Fetch `url` over HTTP using whatever client is installed.

        Tries ``httpx`` (already a model-gateway dependency via ``openai``),
        then falls back to ``urllib`` in a thread. Kept lazy so importing this
        module never pulls in a networking stack.
        """
        try:
            import httpx
        except ImportError:  # pragma: no cover - httpx ships with openai
            import urllib.request

            def _blocking() -> str:
                with urllib.request.urlopen(url, timeout=self.timeout) as resp:
                    return resp.read().decode("utf-8", "replace")

            return await asyncio.to_thread(_blocking)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.text

    async def _fetch(self, url: str) -> str:
        fetcher = self._fetcher or self._default_fetcher
        return await fetcher(url)

    # -- sampling ----------------------------------------------------------- #
    async def poll_once(self) -> dict[str, VLLMInstanceMetrics]:
        """Poll every registered replica once, concurrently."""
        await asyncio.gather(
            *(self._poll_one(name) for name in list(self._metrics)),
            return_exceptions=True,
        )
        return dict(self._metrics)

    async def _poll_one(self, name: str) -> None:
        state = self._metrics.get(name)
        if state is None:
            return
        state.polls += 1
        try:
            body = await self._fetch(f"{state.base_url}{self.metrics_path}")
        except Exception as e:  # noqa: BLE001 - any transport error is unhealthy
            self._mark_unhealthy(state, f"metrics fetch failed: {type(e).__name__}: {e}")
            return

        samples = parse_prometheus(body)
        if not samples:
            self._mark_unhealthy(state, "metrics endpoint returned no parseable samples")
            return

        self._apply(state, samples)
        # A metrics response implies the process is alive, but /health is
        # authoritative: vLLM reports readiness there (e.g. while loading
        # weights, /metrics may already answer while /health says 503).
        await self._poll_health(state)

    def _apply(self, state: VLLMInstanceMetrics, samples: dict[str, float]) -> None:
        state.samples = samples
        state.gpu_cache_usage_factor = float(
            samples.get("gpu_cache_usage_factor", state.gpu_cache_usage_factor)
        )
        state.num_requests_running = int(
            samples.get("num_requests_running", state.num_requests_running)
        )
        state.num_requests_waiting = int(
            samples.get("num_requests_waiting", state.num_requests_waiting)
        )
        state.avg_prompt_throughput_tok_s = self._blend(
            state.avg_prompt_throughput_tok_s,
            samples.get("avg_prompt_throughput_tok_s"),
        )
        state.avg_generation_throughput_tok_s = self._blend(
            state.avg_generation_throughput_tok_s,
            samples.get("avg_generation_throughput_tok_s"),
        )
        state.healthy = True
        state.last_error = None
        state.last_success_at = time.time()
        state.status = self._classify(state)

    def _blend(self, current: float, sample: float | None) -> float:
        """Exponential-decay average: recent samples dominate."""
        if sample is None:
            return current
        if not current:
            return float(sample)
        return self.decay * current + (1.0 - self.decay) * float(sample)

    async def _poll_health(self, state: VLLMInstanceMetrics) -> None:
        try:
            await self._fetch(f"{state.base_url}{self.health_path}")
        except Exception as e:  # noqa: BLE001
            self._mark_unhealthy(state, f"health check failed: {type(e).__name__}: {e}")

    def _mark_unhealthy(self, state: VLLMInstanceMetrics, reason: str) -> None:
        state.healthy = False
        state.status = UNHEALTHY
        state.last_error = reason
        state.errors += 1
        logger.debug("vllm replica %s unhealthy: %s", state.name, reason)

    def _classify(self, state: VLLMInstanceMetrics) -> str:
        """Derive the coarses health label from the current counters."""
        if state.gpu_cache_usage_factor >= self.cache_saturation:
            return SATURATED
        if state.num_requests_waiting >= self.queue_saturation:
            return SATURATED
        if state.gpu_cache_usage_factor >= self.cache_saturation * 0.8:
            return DEGRADED
        return HEALTHY

    # -- background loop ---------------------------------------------------- #
    async def start(self) -> None:
        """Start the background poll loop (idempotent)."""
        if self._task is not None and not self._task.done():
            return
        self._stop = asyncio.Event()
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """Stop the background poll loop and await its exit."""
        self._stop.set()
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._task = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                await self.poll_once()
            except Exception:  # noqa: BLE001 - the loop must not die
                logger.exception("vllm metrics poll failed")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                continue

    async def __aenter__(self) -> "VLLMMetricsCollector":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # -- routing helpers ---------------------------------------------------- #
    def is_saturated(self, name: str) -> bool:
        state = self._metrics.get(name)
        if state is None or not state.healthy:
            return True
        return state.status == SATURATED

    def is_overloaded(self, name: str, cache_threshold: float | None = None) -> bool:
        """True when `name` should not receive new prefix-affine traffic."""
        state = self._metrics.get(name)
        if state is None or not state.healthy:
            return True
        threshold = self.cache_saturation if cache_threshold is None else cache_threshold
        return state.gpu_cache_usage_factor >= threshold

    def rank(self, candidates: Iterable[str]) -> list[str]:
        """Order `candidates` by least load, best first.

        Sort key: health, then queue depth, then KV-cache utilisation, then
        running count — so a replica holding a full cache but with an empty
        queue still beats one that is merely busy. Unhealthy replicas sort
        last so they are only used when nothing else remains.
        """
        names = list(candidates)

        def key(name: str) -> tuple[int, int, float, int, str]:
            state = self._metrics.get(name)
            if state is None:
                # Unknown replica: usable, but ranked after known-healthy ones.
                return (1, 0, 0.0, 0, name)
            return (
                0 if state.healthy else 1,
                state.queue_depth,
                state.gpu_cache_usage_factor,
                state.num_requests_running,
                name,
            )

        return sorted(names, key=key)

    def best(self, candidates: Iterable[str]) -> str | None:
        ranked = self.rank(candidates)
        return ranked[0] if ranked else None
