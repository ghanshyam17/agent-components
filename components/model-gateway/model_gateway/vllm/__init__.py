"""Advanced vLLM routing: KV-cache affinity, live GPU telemetry, speculation.

This subpackage specialises the gateway for clusters of **vLLM** instances,
where the interesting routing signal is not just load but *KV-cache locality*.
vLLM's PagedAttention keeps a cache of computed prompt prefixes; a request
whose prefix a replica has already seen skips prefill for that prefix and cuts
time-to-first-token dramatically. Routing therefore wants to be
*affinity-aware*, not merely round-robin.

Modules:

* :mod:`cache_affinity` — ``PrefixCacheAffinity``: a token-level radix tree plus
  a consistent-hash ring, used to send requests that share a system prompt or
  opening turns to the same replica.
* :mod:`metrics_collector` — ``VLLMMetricsCollector``: polls each replica's
  Prometheus ``/metrics`` and ``/health`` to expose live KV-cache pressure,
  queue depth and throughput.
* :mod:`speculative` — ``SpeculativeDecodingCoordinator``: pairs a small draft
  model with a large target verifier and tracks acceptance to decide when
  speculation is worth it.
* :mod:`router` — ``AdvancedVLLMRouter``: combines the three with load-aware
  dispatch and adaptive fallback to a cloud (Azure AI Foundry) endpoint.

Nothing here requires a live GPU to import or unit-test; the collector accepts
an injected HTTP client and the router accepts a gateway with injected clients.
"""
from __future__ import annotations

from model_gateway.vllm.cache_affinity import (
    ConsistentHashRing,
    PrefixCacheAffinity,
    RadixTree,
    tokens_for_messages,
)
from model_gateway.vllm.metrics_collector import (
    VLLMInstanceMetrics,
    VLLMMetricsCollector,
    parse_prometheus,
)
from model_gateway.vllm.speculative import (
    SpeculativeDecodingCoordinator,
    SpeculationStats,
)

__all__ = [
    "ConsistentHashRing",
    "PrefixCacheAffinity",
    "RadixTree",
    "tokens_for_messages",
    "VLLMInstanceMetrics",
    "VLLMMetricsCollector",
    "parse_prometheus",
    "SpeculativeDecodingCoordinator",
    "SpeculationStats",
]
