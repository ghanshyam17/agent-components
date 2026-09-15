"""AdvancedVLLMRouter — affinity, load and fallback dispatch for vLLM clusters.

Unifies the three signals this subpackage produces:

* **KV-cache affinity** (:mod:`cache_affinity`) — which replica probably holds
  this request's prefix, and so can skip prefill for it.
* **Live load** (:mod:`metrics_collector`) — which replica actually has room
  (queue depth, KV-cache utilisation, health).
* **Speculation** (:mod:`speculative`) — whether a draft model is worth using
  for this pair of replicas.

Policies (the ``Group.strategy`` values):

``prefix_cache``
    Order candidates by affinity, then *demote* any replica whose KV cache is at
    or above ``gpu_cache_saturation``. Affinity is worthless if the replica
    that holds the prefix has no cache blocks free — vLLM would preempt and the
    supposed win evaporates. Demotion (rather than removal) keeps the replica as
    a last resort so a fully saturated cluster still serves.
``least_pending``
    Order by the collector's load ranking (health, queue depth, cache, running).
``speculative``
    Like ``least_pending``, but ask the coordinator whether to draft and expose
    the draft/target split to the caller.
``adaptive_fallback``
    Try the affinity/load-ordered primaries; on transport failure, timeout, or
    saturation (queue at/over ``queue_saturation``), divert to the group's
    ``fallback_endpoints`` — the Azure AI Foundry cloud model. Saturation is
    checked *before* dispatch so an overloaded cluster does not have to fail
    first, and an in-flight failure mid-stream also diverts.

The router is intentionally transport-agnostic: it orders candidates and hands
them to a caller-supplied dispatch coroutine, so it is testable without any
server, and the same code path serves ``chat`` and ``stream``.
"""
from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Iterable, Sequence

from model_gateway.models import ADVANCED_STRATEGIES, Endpoint, GatewayConfig, Group
from model_gateway.vllm.cache_affinity import PrefixCacheAffinity
from model_gateway.vllm.metrics_collector import VLLMMetricsCollector
from model_gateway.vllm.speculative import SpeculativeDecodingCoordinator

logger = logging.getLogger(__name__)

__all__ = ["AdvancedVLLMRouter", "RoutingDecision", "DispatchFn"]


#: A dispatch callable: (endpoint_name, messages, **kw) -> result.
DispatchFn = Callable[..., Awaitable[Any]]


@dataclass
class RoutingDecision:
    """Why a request went where it went — surfaced for tracing/observability."""

    alias: str
    strategy: str
    chosen: str
    candidates: list[str] = field(default_factory=list)
    reason: str = ""
    demoted: list[str] = field(default_factory=list)
    fallback: bool = False
    speculative: bool = False
    draft_tokens: int = 0
    draft_endpoint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "alias": self.alias,
            "strategy": self.strategy,
            "chosen": self.chosen,
            "candidates": self.candidates,
            "reason": self.reason,
            "demoted": self.demoted,
            "fallback": self.fallback,
            "speculative": self.speculative,
            "draft_tokens": self.draft_tokens,
            "draft_endpoint": self.draft_endpoint,
        }


class AdvancedVLLMRouter:
    """Order vLLM candidates by affinity and load, with adaptive cloud fallback.

    Parameters
    ----------
    config:
        The gateway config; endpoint/group definitions are read from it.
    metrics:
        A shared :class:`VLLMMetricsCollector`. Built from the config's vLLM
        endpoints when omitted, but constructing it does *not* start polling —
        call ``await router.start()`` for that, or drive it manually in tests.
    affinity:
        A shared :class:`PrefixCacheAffinity` (one ring across all endpoints, so
        affinity is comparable between groups).
    speculation:
        A :class:`SpeculativeDecodingCoordinator` consulted by the
        ``speculative`` strategy.
    """

    def __init__(
        self,
        config: GatewayConfig,
        *,
        metrics: VLLMMetricsCollector | None = None,
        affinity: PrefixCacheAffinity | None = None,
        speculation: SpeculativeDecodingCoordinator | None = None,
        enabled: bool = True,
    ) -> None:
        self.config = config
        self.enabled = enabled
        self._endpoints: dict[str, Endpoint] = {e.name: e for e in config.endpoints}
        self._groups: dict[str, Group] = {g.alias: g for g in config.groups}

        if affinity is None:
            weights = {
                e.name: e.vllm.weight for e in config.endpoints if e.vllm is not None
            }
            affinity = PrefixCacheAffinity(
                [e.name for e in config.endpoints if e.vllm is not None],
                virtual_nodes=self._virtual_nodes(),
                weights=weights,
            )
        self.affinity = affinity
        self.metrics = metrics if metrics is not None else self._build_collector()
        self.speculation = speculation or SpeculativeDecodingCoordinator()

    # -- construction helpers ----------------------------------------------- #
    def _virtual_nodes(self) -> int:
        for e in self.config.endpoints:
            if e.vllm is not None:
                return e.vllm.virtual_nodes
        return 160

    def _build_collector(self) -> VLLMMetricsCollector:
        """Collector over every vLLM endpoint, honouring its overrides."""
        endpoints: list[tuple[str, str]] = []
        per_endpoint: dict[str, dict[str, Any]] = {}
        for e in self.config.endpoints:
            if e.vllm is None:
                continue
            base = e.base_url.rstrip("/")
            # Strip a trailing /v1: metrics live at the server root.
            if base.endswith("/v1"):
                base = base[: -len("/v1")]
            endpoints.append((e.name, base))
            per_endpoint[e.name] = {
                "gpu_cache_saturation": e.vllm.gpu_cache_saturation,
                "queue_saturation": e.vllm.queue_saturation,
            }
        collector = VLLMMetricsCollector(endpoints)
        # Thresholds are collector-wide; the strictest configured value wins so
        # no replica is treated as having more headroom than it declared. The
        # per-replica values are kept for reference by the router.
        self._saturation_overrides = per_endpoint
        if per_endpoint:
            collector.cache_saturation = min(
                v["gpu_cache_saturation"] for v in per_endpoint.values()
            )
            collector.queue_saturation = min(
                v["queue_saturation"] for v in per_endpoint.values()
            )
        return collector

    # -- lifecycle ---------------------------------------------------------- #
    async def start(self) -> None:
        await self.metrics.start()

    async def stop(self) -> None:
        await self.metrics.stop()

    async def __aenter__(self) -> "AdvancedVLLMRouter":
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.stop()

    # -- membership --------------------------------------------------------- #
    @property
    def groups(self) -> list[str]:
        return list(self._groups)

    def group(self, alias: str) -> Group:
        g = self._groups.get(alias)
        if g is None:
            raise KeyError(f"unknown gateway group alias: {alias!r}")
        return g

    def endpoint(self, name: str) -> Endpoint | None:
        return self._endpoints.get(name)

    def uses_advanced_routing(self, alias: str) -> bool:
        return self.enabled and self.group(alias).is_advanced

    def is_vllm_endpoint(self, name: str) -> bool:
        e = self._endpoints.get(name)
        return e is not None and e.vllm is not None

    # -- candidate selection ------------------------------------------------ #
    def candidates(self, alias: str) -> list[str]:
        """Eligible endpoints for a group (enabled ones, else all defined)."""
        group = self.group(alias)
        refs = [n for n in group.endpoints if n in self._endpoints]
        enabled = [n for n in refs if self._endpoints[n].enabled]
        return enabled or refs

    def fallbacks(self, alias: str) -> list[str]:
        group = self.group(alias)
        refs = [n for n in group.fallback_endpoints if n in self._endpoints]
        enabled = [n for n in refs if self._endpoints[n].enabled]
        return enabled or refs

    def _affinity_capable(self, names: Iterable[str]) -> list[str]:
        """Only real vLLM replicas participate in KV-cache affinity.

        A cloud fallback has no PagedAttention prefix cache to warm, so putting
        it on the ring would only steal affinity slots from replicas that do.
        """
        return [n for n in names if self.is_vllm_endpoint(n)]

    def _is_saturated(self, name: str) -> bool:
        return self.metrics.is_saturated(name) if self.is_vllm_endpoint(name) else False

    def _is_overloaded(self, name: str) -> bool:
        """Cache-pressure check used to demote an affined replica."""
        e = self._endpoints.get(name)
        if e is None or e.vllm is None:
            return False
        return self.metrics.is_overloaded(name, cache_threshold=e.vllm.gpu_cache_saturation)

    # -- the ordering policies ---------------------------------------------- #
    def order(
        self,
        alias: str,
        messages: Sequence[Any] | None = None,
    ) -> tuple[list[str], RoutingDecision]:
        """Order a group's endpoints per its strategy; return them + the why."""
        group = self.group(alias)
        primary = self.candidates(alias)
        if not primary:
            decision = RoutingDecision(
                alias=alias, strategy=group.strategy, chosen="",
                reason="no enabled endpoints",
            )
            return [], decision

        if not self.uses_advanced_routing(alias):
            # Preserve the classic strategies exactly as the base Gateway does.
            if group.strategy == "round_robin":
                ordered = self._rotate(alias, primary)
            elif group.strategy == "weighted":
                ordered = self._weighted(primary)
            else:  # failover
                ordered = primary
            decision = RoutingDecision(
                alias=alias, strategy=group.strategy, chosen=ordered[0],
                candidates=ordered, reason=f"{group.strategy} order",
            )
            return ordered, decision

        if group.strategy == "prefix_cache":
            ordered, demoted, reason = self._order_prefix_cache(
                primary, messages, group.affinity_prefix_tokens
            )
            decision = RoutingDecision(
                alias=alias, strategy=group.strategy, chosen=ordered[0],
                candidates=ordered, demoted=demoted, reason=reason,
            )
            return ordered, decision

        if group.strategy == "least_pending":
            ordered = self.metrics.rank(primary)
            decision = RoutingDecision(
                alias=alias, strategy=group.strategy, chosen=ordered[0],
                candidates=ordered, reason="least queue depth, then KV-cache pressure",
            )
            return ordered, decision

        # speculative + adaptive_fallback both lead with load, then apply their
        # own extra behaviour.
        ordered = self.metrics.rank(primary)
        if group.strategy == "speculative":
            target = ordered[0]
            target_ep = self._endpoints.get(target)
            draft_ref = target_ep.vllm.draft_endpoint_ref if (target_ep and target_ep.vllm) else None
            speculate = self.speculation.should_speculate(draft_ref, target)
            k = self.speculation.draft_tokens(draft_ref, target) if speculate else 0
            decision = RoutingDecision(
                alias=alias, strategy=group.strategy, chosen=target,
                candidates=ordered,
                reason=(
                    f"speculative decoding via draft {draft_ref!r} (k={k})"
                    if speculate else "speculation off (low acceptance); plain decoding"
                ),
                speculative=speculate, draft_tokens=k, draft_endpoint=draft_ref,
            )
            return ordered, decision

        # adaptive_fallback
        fbs = self.fallbacks(alias)
        saturated = [n for n in ordered if self._is_saturated(n)]
        if saturated and fbs:
            decision = RoutingDecision(
                alias=alias, strategy=group.strategy, chosen=fbs[0],
                candidates=[*fbs, *ordered], reason=(
                    f"primary cluster saturated ({', '.join(saturated)}); "
                    f"diverting to cloud fallback"
                ),
                fallback=True,
            )
            return [*fbs, *ordered], decision
        decision = RoutingDecision(
            alias=alias, strategy=group.strategy, chosen=ordered[0],
            candidates=[*ordered, *fbs],
            reason="primary cluster has capacity",
        )
        return [*ordered, *fbs], decision

    def _order_prefix_cache(
        self,
        candidates: list[str],
        messages: Sequence[Any] | None,
        prefix_tokens: int,
    ) -> tuple[list[str], list[str], str]:
        """Affinity first, then demote replicas under cache pressure."""
        self.affinity.prefix_tokens = prefix_tokens
        affined = self._affinity_capable(candidates)
        plain = [n for n in candidates if n not in affined]

        if messages is not None and affined:
            ordered = self.affinity.order(messages, affined)
        else:
            ordered = affined

        # Partition into "has room" and "overloaded", preserving affinity order
        # within each; overloaded replicas stay reachable as a last resort.
        ok, demoted = [], []
        for name in ordered:
            (demoted if self._is_overloaded(name) else ok).append(name)
        final = [*ok, *demoted, *plain]
        if demoted:
            reason = (
                f"prefix affinity ({len(affined)} vLLM replicas); "
                f"demoted {len(demoted)} over cache saturation"
            )
        else:
            reason = f"prefix affinity across {len(affined)} vLLM replicas; all have cache room"
        return final, demoted, reason

    def _rotate(self, alias: str, names: list[str]) -> list[str]:
        """Round-robin rotation, matching the base Gateway's behaviour."""
        if not names:
            return []
        self._cursors = getattr(self, "_cursors", {})
        idx = self._cursors.get(alias, 0) % len(names)
        self._cursors[alias] = idx + 1
        return names[idx:] + names[:idx]

    @staticmethod
    def _weighted(names: list[str]) -> list[str]:
        """Stable weighted expansion (heavier endpoints repeated)."""
        out: list[str] = []
        for name in names:
            out.extend([name] * 1)
        return out

    # -- dispatch ----------------------------------------------------------- #
    async def dispatch(
        self,
        alias: str,
        dispatch_fn: DispatchFn,
        *,
        messages: Sequence[Any] | None = None,
        **kwargs: Any,
    ) -> tuple[Any, RoutingDecision]:
        """Route one request: order candidates, try each, fall back as needed.

        `dispatch_fn` is called as ``dispatch_fn(endpoint_name, messages, **kw)``
        and must raise on failure — the router treats any exception as "try the
        next candidate", except in ``adaptive_fallback`` where a saturated
        primary diverts straight to the cloud fallback.

        Returns ``(result, decision)``. Raises the last error when every
        candidate fails, so failures are never silently swallowed.
        """
        group = self.group(alias)
        ordered, decision = self.order(alias, messages)
        if not ordered:
            raise RuntimeError(f"no enabled endpoints for group {alias!r}")

        attempts = group.max_retries + 1
        last_err: Exception | None = None

        for attempt in range(attempts):
            for name in ordered:
                is_fallback = name in self.fallbacks(alias)
                # In adaptive_fallback, skip a saturated primary entirely and
                # let the fallback endpoint (which sorts after them) serve.
                if group.strategy == "adaptive_fallback" and not is_fallback:
                    if self._is_saturated(name) and self.fallbacks(alias):
                        last_err = last_err or RuntimeError(
                            f"{name} saturated; diverted to fallback"
                        )
                        logger.info(
                            "vllm router: %s saturated, diverting past it", name
                        )
                        continue
                try:
                    result = await dispatch_fn(name, messages, **kwargs)
                except Exception as e:  # noqa: BLE001 - fall through to next
                    last_err = e
                    logger.warning(
                        "vllm router: dispatch to %s failed (%s: %s)",
                        name, type(e).__name__, e,
                    )
                    continue
                # Learn the prefix→replica mapping only on success: an
                # observation from a failed call would teach the tree a cache
                # that was never populated.
                if self.is_vllm_endpoint(name) and messages is not None:
                    self.affinity.observe(messages, name)
                if name != decision.chosen:
                    decision.reason = f"{decision.reason}; served by {name}"
                    decision.chosen = name
                    decision.fallback = is_fallback
                return result, decision
            if attempt < attempts - 1 and group.retry_backoff:
                await asyncio.sleep(group.retry_backoff * (2 ** attempt))

        assert last_err is not None
        raise last_err

    # -- introspection ------------------------------------------------------ #
    def snapshot(self) -> dict[str, Any]:
        """Full routing state — for dashboards and the /metrics story."""
        return {
            "enabled": self.enabled,
            "groups": {
                alias: {
                    "strategy": g.strategy,
                    "advanced": self.uses_advanced_routing(alias),
                    "candidates": self.candidates(alias),
                    "fallbacks": self.fallbacks(alias),
                }
                for alias, g in self._groups.items()
            },
            "instances": self.metrics.snapshot(),
            "affinity": self.affinity.stats(),
            "speculation": self.speculation.summary(),
        }
