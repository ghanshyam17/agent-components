"""Tests for the advanced vLLM router subsystem.

Covers the three behaviours the implementation plan calls out explicitly:

* KV-cache prefix affinity — identical system prompts must resolve to the same
  replica, and the mapping must survive replica churn.
* Load-aware routing — requests go to the replica with the lowest queue depth.
* Adaptive fallback — a saturated or failing primary cluster diverts to the
  Azure AI Foundry cloud endpoint without losing the request.

Everything is offline: the metrics collector takes an injected async fetcher,
the gateway takes an injected client factory, and no vLLM server is required.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from components_core import Message
from model_gateway import Gateway
from model_gateway.models import Endpoint, GatewayConfig, Group, VLLMConfig
from model_gateway.vllm.cache_affinity import (
    ConsistentHashRing,
    PrefixCacheAffinity,
    RadixTree,
    tokens_for_messages,
)
from model_gateway.vllm.metrics_collector import (
    DEGRADED,
    HEALTHY,
    SATURATED,
    UNHEALTHY,
    VLLMMetricsCollector,
    parse_prometheus,
)
from model_gateway.vllm.router import AdvancedVLLMRouter
from model_gateway.vllm.speculative import SpeculativeDecodingCoordinator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _vllm(name: str, port: int, **cfg) -> Endpoint:
    return Endpoint(
        name=name,
        base_url=f"http://localhost:{port}/v1",
        model="qwen2.5-7b-instruct",
        vllm=VLLMConfig(**cfg),
    )


def _cloud(name: str = "cloud") -> Endpoint:
    return Endpoint(
        name=name,
        base_url="https://my-foundry-resource.openai.azure.com/openai/v1",
        model="gpt-5-mini",
        api_key="entra",
    )


def _msgs(system: str = "You are a TEF tutor.", user: str = "Bonjour") -> list[Message]:
    return [
        Message(role="system", content=system),
        Message(role="user", content=user),
    ]


def _metrics_body(
    cache: float = 0.1, running: int = 1, waiting: int = 0,
    prompt_tps: float = 100.0, gen_tps: float = 50.0,
) -> str:
    """A realistic vLLM Prometheus exposition body."""
    return "\n".join([
        "# HELP gpu_cache_usage_factor KV-cache block utilisation",
        "# TYPE gpu_cache_usage_factor gauge",
        f"gpu_cache_usage_factor {cache}",
        f"num_requests_running {running}",
        f"num_requests_waiting {waiting}",
        f"avg_prompt_throughput_tok_s {prompt_tps}",
        f"avg_generation_throughput_tok_s {gen_tps}",
        'vllm:num_requests_total{model_name="qwen"} 42',
    ])


class _Fetcher:
    """Injected async fetcher: maps URL -> body, or raises for failures."""

    def __init__(self, responses: dict[str, str], failures: set[str] | None = None):
        self.responses = responses
        self.failures = failures or set()
        self.calls: list[str] = []

    async def __call__(self, url: str) -> str:
        self.calls.append(url)
        for bad in self.failures:
            if bad in url:
                raise ConnectionError(f"refused: {url}")
        if url in self.responses:
            return self.responses[url]
        if url.endswith("/health"):
            return "OK"
        raise ConnectionError(f"no route for {url}")


# --------------------------------------------------------------------------- #
# Prometheus parsing
# --------------------------------------------------------------------------- #
def test_parse_prometheus_reads_gauges_and_skips_comments():
    parsed = parse_prometheus(_metrics_body(cache=0.42, running=3, waiting=7))
    assert parsed["gpu_cache_usage_factor"] == 0.42
    assert parsed["num_requests_running"] == 3
    assert parsed["num_requests_waiting"] == 7
    assert "HELP" not in parsed


def test_parse_prometheus_ignores_non_numeric_and_keeps_first_sample():
    body = (
        "gpu_cache_usage_factor NaN\n"          # invalid: skipped entirely
        "gpu_cache_usage_factor 0.5\n"          # first *valid* sample wins
        "gpu_cache_usage_factor 0.9\n"          # ignored (name already seen)
        'num_requests_waiting{model="a"} 4\n'
        "broken_line_without_value\n"
        "  \n"
        "# a comment\n"
    )
    parsed = parse_prometheus(body)
    assert parsed["num_requests_waiting"] == 4
    # NaN is dropped rather than poisoning the value, so the next valid sample
    # becomes the recorded one and the later duplicate is ignored.
    assert parsed["gpu_cache_usage_factor"] == 0.5


# --------------------------------------------------------------------------- #
# Metrics collector
# --------------------------------------------------------------------------- #
def test_collector_polls_and_classifies_health():
    fetcher = _Fetcher({
        "http://localhost:8001/metrics": _metrics_body(cache=0.2, waiting=0),
        "http://localhost:8001/health": "OK",
    })
    collector = VLLMMetricsCollector({"a": "http://localhost:8001"}, fetcher=fetcher)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    state = collector.get("a")
    assert state is not None
    assert state.healthy and state.status == HEALTHY
    assert state.gpu_cache_usage_factor == 0.2
    assert state.polls == 1


def test_collector_marks_saturated_when_cache_full():
    fetcher = _Fetcher({
        "http://localhost:8001/metrics": _metrics_body(cache=0.95),
        "http://localhost:8001/health": "OK",
    })
    collector = VLLMMetricsCollector({"a": "http://localhost:8001"}, fetcher=fetcher)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    assert collector.get("a").status == SATURATED
    assert collector.is_saturated("a") is True


def test_collector_marks_degraded_near_cache_limit():
    fetcher = _Fetcher({
        "http://localhost:8001/metrics": _metrics_body(cache=0.70),
        "http://localhost:8001/health": "OK",
    })
    # 0.70 >= 0.85*0.8 = 0.68 → degraded, but below the 0.85 saturation line.
    collector = VLLMMetricsCollector({"a": "http://localhost:8001"}, fetcher=fetcher)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    assert collector.get("a").status == DEGRADED


def test_collector_unhealthy_on_transport_error():
    fetcher = _Fetcher({}, failures={"8001"})
    collector = VLLMMetricsCollector({"a": "http://localhost:8001"}, fetcher=fetcher)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    state = collector.get("a")
    assert state.healthy is False
    assert state.status == UNHEALTHY
    assert state.errors == 1
    assert "metrics fetch failed" in (state.last_error or "")


def test_collector_unhealthy_when_metrics_empty():
    fetcher = _Fetcher({"http://localhost:8001/metrics": "# only comments\n"})
    collector = VLLMMetricsCollector({"a": "http://localhost:8001"}, fetcher=fetcher)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    assert collector.get("a").healthy is False
    assert "no parseable samples" in (collector.get("a").last_error or "")


def test_collector_health_endpoint_failure_beats_metrics_success():
    """vLLM can serve /metrics while still loading weights and failing /health."""
    fetcher = _Fetcher(
        {"http://localhost:8001/metrics": _metrics_body(cache=0.1)},
        failures={"/health"},
    )
    collector = VLLMMetricsCollector({"a": "http://localhost:8001"}, fetcher=fetcher)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    assert collector.get("a").healthy is False
    assert "health check failed" in (collector.get("a").last_error or "")


def test_collector_throughput_uses_exponential_decay():
    """A burst must decay rather than pin the average for the process lifetime."""
    fetcher = _Fetcher({
        "http://localhost:8001/metrics": _metrics_body(prompt_tps=100.0),
        "http://localhost:8001/health": "OK",
    })
    collector = VLLMMetricsCollector(
        {"a": "http://localhost:8001"}, fetcher=fetcher, decay=0.5
    )
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    assert collector.get("a").avg_prompt_throughput_tok_s == 100.0  # first sample

    # A slow sample must pull the average most of the way down.
    fetcher.responses["http://localhost:8001/metrics"] = _metrics_body(prompt_tps=0.0)
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    assert collector.get("a").avg_prompt_throughput_tok_s == pytest.approx(50.0)


def test_collector_rank_prefers_lowest_queue_then_cache():
    fetcher = _Fetcher({
        "http://localhost:8001/metrics": _metrics_body(cache=0.1, waiting=5),
        "http://localhost:8001/health": "OK",
        "http://localhost:8002/metrics": _metrics_body(cache=0.1, waiting=1),
        "http://localhost:8002/health": "OK",
        "http://localhost:8003/metrics": _metrics_body(cache=0.5, waiting=1),
        "http://localhost:8003/health": "OK",
    })
    collector = VLLMMetricsCollector(
        {"a": "http://localhost:8001", "b": "http://localhost:8002", "c": "http://localhost:8003"},
        fetcher=fetcher,
    )
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    # b (waiting=1, cache 0.1) and c (waiting=1, cache 0.5) tie on queue; cache
    # breaks the tie, so b first, then c, then a (waiting=5).
    assert collector.rank(["a", "b", "c"]) == ["b", "c", "a"]
    assert collector.best(["a", "b", "c"]) == "b"


def test_collector_rank_puts_unhealthy_last():
    fetcher = _Fetcher(
        {
            "http://localhost:8001/metrics": _metrics_body(waiting=9),
            "http://localhost:8001/health": "OK",
        },
        failures={"8002"},
    )
    collector = VLLMMetricsCollector(
        {"busy": "http://localhost:8001", "down": "http://localhost:8002"}, fetcher=fetcher
    )
    asyncio.get_event_loop().run_until_complete(collector.poll_once())
    # A healthy-but-busy replica still beats an unreachable one.
    assert collector.rank(["down", "busy"]) == ["busy", "down"]


# --------------------------------------------------------------------------- #
# Prefix affinity
# --------------------------------------------------------------------------- #
def test_tokens_for_messages_puts_system_prompt_first():
    msgs = [
        Message(role="user", content="question"),
        Message(role="system", content="SHARED INSTRUCTIONS"),
    ]
    tokens = tokens_for_messages(msgs)
    assert tokens[0] == "SHARED"
    assert tokens[1] == "INSTRUCTIONS"


def test_prefix_affinity_identical_system_prompts_share_a_replica():
    """The headline promise: same system prompt -> same replica."""
    aff = PrefixCacheAffinity(["a", "b", "c"])
    msgs = _msgs(system="You are a TEF tutor.")
    ordered = aff.order(msgs, ["a", "b", "c"])
    assert len(ordered) == 3
    # Deterministic for a given key, and a prefix of the ring's choice.
    assert aff.order(msgs, ["a", "b", "c"]) == ordered
    assert ordered[0] == aff.owner(aff.key_for(msgs))


def test_prefix_affinity_learns_from_observations():
    """A replica observed serving a prefix must be promoted above the ring pick."""
    aff = PrefixCacheAffinity(["a", "b", "c"])
    msgs = _msgs()
    ring_pick = aff.owner(aff.key_for(msgs))
    # Observe a *different* replica serving this prefix repeatedly.
    other = next(r for r in ["a", "b", "c"] if r != ring_pick)
    for _ in range(3):
        aff.observe(msgs, other)
    assert aff.order(msgs, ["a", "b", "c"])[0] == other


def test_prefix_affinity_different_prompts_can_differ():
    aff = PrefixCacheAffinity([f"r{i}" for i in range(8)])
    picks_a = {aff.order(_msgs(system=f"persona {i}"), aff.replicas)[0] for i in range(24)}
    # Eight replicas and 24 distinct prefixes: the ring must spread them.
    assert len(picks_a) >= 3


def test_radix_tree_longest_prefix_match():
    tree = RadixTree()
    tree.observe(["shared", "prefix", "one"], "a")
    tree.observe(["shared", "prefix", "one", "two"], "b")
    # Matching a longer stored prefix reaches the deeper observation.
    assert tree.lookup(["shared", "prefix", "one", "two"])["b"] == 1
    # A shorter query stops at the shallower node.
    assert tree.lookup(["shared", "prefix"]) == {}
    assert tree.lookup(["shared", "prefix", "one"])["a"] == 1
    assert tree.observations == 2


def test_radix_tree_ignores_empty_observation():
    tree = RadixTree()
    tree.observe([], "a")
    assert tree.observations == 0
    assert len(tree) == 0


def test_consistent_ring_is_stable_under_replica_churn():
    ring = ConsistentHashRing([f"r{i}" for i in range(6)])
    keys = [f"prefix-{i}" for i in range(200)]
    before = {k: ring.get(k) for k in keys}

    ring.add("r6")
    after = {k: ring.get(k) for k in keys}
    moved = sum(1 for k in keys if before[k] != after[k])
    # Consistent hashing moves ~1/7 of keys, not all of them. Allow slack for
    # hash variance but reject the hash%n behaviour (which would move ~6/7).
    assert moved < len(keys) * 0.5


def test_consistent_ring_remove_returns_false_when_absent():
    ring = ConsistentHashRing(["a"])
    assert ring.remove("nope") is False
    assert ring.remove("a") is True
    assert ring.get("key") is None


def test_consistent_ring_weights_shift_distribution():
    ring = ConsistentHashRing(virtual_nodes=200)
    ring.add("light", weight=1)
    ring.add("heavy", weight=9)
    dist = ring.distribution(f"k{i}" for i in range(2000))
    assert dist["heavy"] > dist["light"]


def test_prefix_affinity_stats_and_membership():
    aff = PrefixCacheAffinity(["a", "b"])
    assert aff.replicas == ["a", "b"]
    aff.add_replica("c")
    assert "c" in aff.replicas
    assert aff.remove_replica("c") is True
    aff.observe(_msgs(), "a")
    stats = aff.stats()
    assert stats["observations"] == 1
    assert stats["known_prefixes"] == 1


# --------------------------------------------------------------------------- #
# Speculative decoding
# --------------------------------------------------------------------------- #
def test_speculation_probes_then_requires_acceptance():
    spec = SpeculativeDecodingCoordinator(min_acceptance=0.6, probe_rounds=3)
    # No evidence yet -> speculate to gather some.
    assert spec.should_speculate("draft", "target") is True
    for _ in range(3):
        spec.record("draft", "target", proposed=4, accepted=1)  # 25% acceptance
    assert spec.should_speculate("draft", "target") is False


def test_speculation_enables_when_acceptance_high():
    spec = SpeculativeDecodingCoordinator(min_acceptance=0.5, probe_rounds=2)
    for _ in range(4):
        spec.record("draft", "target", proposed=4, accepted=4)
    assert spec.should_speculate("draft", "target") is True
    assert spec.draft_tokens("draft", "target") == 8  # acceptance 1.0 -> max k


def test_speculation_draft_tokens_scale_with_acceptance():
    spec = SpeculativeDecodingCoordinator(
        min_acceptance=0.5, min_draft_tokens=1, max_draft_tokens=8, probe_rounds=1
    )
    spec.record("draft", "target", proposed=10, accepted=5)   # 0.5 -> floor
    low = spec.draft_tokens("draft", "target")
    spec2 = SpeculativeDecodingCoordinator(
        min_acceptance=0.5, min_draft_tokens=1, max_draft_tokens=8, probe_rounds=1
    )
    spec2.record("draft", "target", proposed=10, accepted=9)  # 0.9 -> near max
    high = spec2.draft_tokens("draft", "target")
    assert low == 1 and high > low


def test_speculation_off_without_a_draft_endpoint():
    spec = SpeculativeDecodingCoordinator()
    assert spec.should_speculate(None, "target") is False
    assert spec.draft_tokens(None, "target") == 0


def test_speculation_disabled_flag():
    spec = SpeculativeDecodingCoordinator(enabled=False)
    assert spec.should_speculate("draft", "target") is False


def test_speculation_records_and_reports_stats():
    spec = SpeculativeDecodingCoordinator()
    spec.record("draft", "target", proposed=4, accepted=3)
    stats = spec.stats()["draft->target"]
    assert stats["proposed"] == 4 and stats["accepted"] == 3
    assert stats["tokens_per_target_pass"] == 3.0


def test_speculation_generate_requires_bound_callables():
    spec = SpeculativeDecodingCoordinator()
    result = asyncio.get_event_loop().run_until_complete(
        spec.generate([1, 2], "draft", "target")
    )
    assert result is None  # no proposer/verifier bound -> caller decodes plainly


def test_speculation_generate_verifies_via_verifier():
    async def proposer(prefix, k):
        return ["d1", "d2", "d3"]

    async def verifier(prefix, proposal):
        # accept the first two, reject the third (target is authoritative)
        return ["d1", "d2"], 2

    spec = SpeculativeDecodingCoordinator(probe_rounds=1, min_draft_tokens=3)
    spec.bind(proposer, verifier)
    result = asyncio.get_event_loop().run_until_complete(
        spec.generate(["p"], "draft", "target")
    )
    assert result is not None
    accepted, stats = result
    assert accepted == ["d1", "d2"]
    assert stats.accepted == 2 and stats.proposed == 3


# --------------------------------------------------------------------------- #
# Router: ordering policies
# --------------------------------------------------------------------------- #
def _router_config(strategy: str, *, with_fallback: bool = True) -> GatewayConfig:
    endpoints = [_vllm("v1", 8001), _vllm("v2", 8002)]
    groups = [Group(alias="g", endpoints=["v1", "v2"], strategy=strategy)]
    if with_fallback:
        endpoints.append(_cloud())
        groups[0].fallback_endpoints = ["cloud"]
    return GatewayConfig(endpoints=endpoints, groups=groups)


def test_router_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        Group(alias="g", endpoints=["v1"], strategy="nonsense")  # type: ignore[arg-type]


def test_router_requires_fallback_endpoints_to_exist():
    with pytest.raises(ValueError, match="unknown endpoint"):
        GatewayConfig(
            endpoints=[_vllm("v1", 8001)],
            groups=[Group(alias="g", endpoints=["v1"], fallback_endpoints=["ghost"])],
        )


def test_router_requires_draft_endpoint_to_exist():
    with pytest.raises(ValueError, match="unknown draft endpoint"):
        GatewayConfig(
            endpoints=[_vllm("v1", 8001, draft_endpoint_ref="ghost")],
            groups=[Group(alias="g", endpoints=["v1"])],
        )


def test_router_prefix_cache_strategy_picks_affinity_winner():
    router = AdvancedVLLMRouter(_router_config("prefix_cache"))
    msgs = _msgs()
    ordered, decision = router.order("g", msgs)
    assert decision.strategy == "prefix_cache"
    # The affined vLLM replica leads; the cloud endpoint cannot hold a prefix.
    assert ordered[0] in ("v1", "v2")
    assert "cloud" not in ordered[:2] or len(ordered) == 2
    assert "affinity" in decision.reason


def test_router_prefix_cache_demotes_saturated_replica():
    """Affinity is worthless if the affined replica has no cache blocks free."""
    router = AdvancedVLLMRouter(_router_config("prefix_cache"))
    # Make the affinity winner saturated, then re-order.
    winner = router.order("g", _msgs())[0][0]
    router.metrics.get(winner).gpu_cache_usage_factor = 0.99
    router.metrics.get(winner).status = SATURATED

    ordered, decision = router.order("g", _msgs())
    assert winner in decision.demoted
    assert ordered[0] != winner
    assert winner in ordered  # demoted, not removed — still a last resort


def test_router_least_pending_uses_lowest_queue_depth():
    router = AdvancedVLLMRouter(_router_config("least_pending"))
    router.metrics.get("v1").num_requests_waiting = 12
    router.metrics.get("v2").num_requests_waiting = 0
    ordered, decision = router.order("g", _msgs())
    assert decision.strategy == "least_pending"
    assert ordered[0] == "v2"


def test_router_adaptive_fallback_diverts_when_cluster_saturated():
    router = AdvancedVLLMRouter(_router_config("adaptive_fallback"))
    for name in ("v1", "v2"):
        state = router.metrics.get(name)
        state.num_requests_waiting = 99
        state.status = SATURATED
    ordered, decision = router.order("g", _msgs())
    assert decision.fallback is True
    assert decision.chosen == "cloud"
    assert ordered[0] == "cloud"
    assert "saturated" in decision.reason


def test_router_adaptive_fallback_uses_primaries_when_healthy():
    router = AdvancedVLLMRouter(_router_config("adaptive_fallback"))
    ordered, decision = router.order("g", _msgs())
    assert decision.fallback is False
    assert ordered[0] in ("v1", "v2")
    assert "capacity" in decision.reason


def test_router_speculative_exposes_draft_split():
    cfg = _router_config("speculative")
    # Give v2 a draft model so the decision can name it.
    for e in cfg.endpoints:
        if e.name == "v2" and e.vllm is not None:
            e.vllm.draft_endpoint_ref = "v1"
    router = AdvancedVLLMRouter(cfg)
    router.metrics.get("v2").num_requests_waiting = 0
    router.metrics.get("v1").num_requests_waiting = 9
    _, decision = router.order("g", _msgs())
    assert decision.strategy == "speculative"
    assert decision.chosen == "v2"
    assert decision.speculative is True
    assert decision.draft_endpoint == "v1"
    assert decision.draft_tokens >= 1


def test_router_classic_strategies_still_work():
    cfg = GatewayConfig(
        endpoints=[_vllm("v1", 8001), _vllm("v2", 8002)],
        groups=[Group(alias="rr", endpoints=["v1", "v2"], strategy="round_robin")],
    )
    router = AdvancedVLLMRouter(cfg)
    assert router.uses_advanced_routing("rr") is False
    first = router.order("rr", _msgs())[0]
    second = router.order("rr", _msgs())[0]
    assert first != second  # rotation actually rotates


def test_router_disabled_flag_disables_advanced_strategies():
    router = AdvancedVLLMRouter(_router_config("prefix_cache"), enabled=False)
    assert router.uses_advanced_routing("g") is False
    ordered, decision = router.order("g", _msgs())
    assert decision.strategy == "prefix_cache"
    assert ordered  # still ordered, just not by affinity


# --------------------------------------------------------------------------- #
# Router: dispatch + fallback
# --------------------------------------------------------------------------- #
def test_router_dispatch_returns_result_and_learns_affinity():
    router = AdvancedVLLMRouter(_router_config("prefix_cache"))
    seen: list[str] = []

    async def dispatch(name, messages, **kw):
        seen.append(name)
        return f"answered-by-{name}"

    result, decision = asyncio.get_event_loop().run_until_complete(
        router.dispatch("g", dispatch, messages=_msgs())
    )
    assert result == f"answered-by-{decision.chosen}"
    assert decision.chosen in ("v1", "v2")
    # A successful call must teach the tree, so the next identical request
    # prefers the same replica.
    assert router.affinity.order(_msgs(), ["v1", "v2"])[0] == decision.chosen


def test_router_dispatch_falls_over_to_next_replica():
    router = AdvancedVLLMRouter(_router_config("least_pending", with_fallback=False))
    router.metrics.get("v1").num_requests_waiting = 0
    router.metrics.get("v2").num_requests_waiting = 5

    async def dispatch(name, messages, **kw):
        if name == "v1":
            raise ConnectionError("v1 is down")
        return f"ok-{name}"

    result, decision = asyncio.get_event_loop().run_until_complete(
        router.dispatch("g", dispatch, messages=_msgs())
    )
    assert result == "ok-v2"
    assert decision.chosen == "v2"


def test_router_dispatch_diverts_to_cloud_on_failure():
    """Adaptive fallback must reach the cloud endpoint when vLLM is down."""
    router = AdvancedVLLMRouter(_router_config("adaptive_fallback"))

    async def dispatch(name, messages, **kw):
        if name == "cloud":
            return "served-by-foundry"
        raise RuntimeError(f"{name} unavailable")

    result, decision = asyncio.get_event_loop().run_until_complete(
        router.dispatch("g", dispatch, messages=_msgs())
    )
    assert result == "served-by-foundry"
    assert decision.chosen == "cloud"
    assert decision.fallback is True


def test_router_dispatch_raises_when_everything_fails():
    router = AdvancedVLLMRouter(_router_config("least_pending", with_fallback=False))

    async def dispatch(name, messages, **kw):
        raise RuntimeError(f"{name} unavailable")

    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.get_event_loop().run_until_complete(
            router.dispatch("g", dispatch, messages=_msgs())
        )


def test_router_dispatch_unknown_alias_raises():
    router = AdvancedVLLMRouter(_router_config("least_pending"))
    with pytest.raises(KeyError):
        router.order("nope", _msgs())


def test_router_snapshot_includes_all_planes():
    router = AdvancedVLLMRouter(_router_config("prefix_cache"))
    snap = router.snapshot()
    assert snap["groups"]["g"]["advanced"] is True
    assert "v1" in snap["instances"]
    assert "replicas" in snap["affinity"]
    assert "pairs" in snap["speculation"]


# --------------------------------------------------------------------------- #
# Gateway integration — the advanced path must be reachable through chat/stream
# --------------------------------------------------------------------------- #
class _RespMsg:
    def __init__(self, content, tool_calls=None):
        self.content = content
        self.tool_calls = tool_calls


class _FakeCompletions:
    """Minimal OpenAI-compatible completions object returning `content`."""

    def __init__(self, content: str):
        self.content = content
        self.calls = 0

    async def create(self, *, model, messages, tools=None, temperature=0.2,
                     max_tokens=None, stream=False):
        self.calls += 1
        if stream:
            async def gen():
                for word in self.content.split():
                    delta = type("D", (), {"content": word + " ", "tool_calls": None})()
                    choice = type("Ch", (), {"delta": delta})()
                    yield type("C", (), {"choices": [choice]})()
            return gen()
        choice = type("Ch", (), {"message": _RespMsg(self.content)})()
        return type("R", (), {"choices": [choice]})()


class _FakeClient:
    def __init__(self, content: str):
        self.chat = type("Chat", (), {"completions": _FakeCompletions(content)})()


def _content_factory(contents: dict[str, str], fail: set[str] | None = None):
    """client_factory mapping endpoint name -> client, with optional failures."""
    fail = fail or set()

    def factory(endpoint: Endpoint):
        if endpoint.name in fail:
            class _Boom:
                def __init__(self):
                    class C:
                        async def create(self, **kw):
                            raise ConnectionError(f"{endpoint.name} refused")
                    self.chat = type("Chat", (), {"completions": C()})()
            return _Boom()
        return _FakeClient(contents.get(endpoint.name, f"from-{endpoint.name}"))
    return factory


def test_gateway_routes_advanced_strategy_through_call_chat():
    """`Gateway.chat` must dispatch via the advanced router for advanced groups."""
    cfg = _router_config("prefix_cache", with_fallback=False)
    gw = Gateway(cfg, client_factory=_content_factory({"v1": "hi-from-v1", "v2": "hi-from-v2"}))
    assert gw.uses_advanced_routing("g") is True

    content, _calls = asyncio.get_event_loop().run_until_complete(
        gw.client("g").chat(_msgs())
    )
    assert content in ("hi-from-v1", "hi-from-v2")
    # The decision is exposed for tracing.
    assert gw.last_decision is not None
    assert gw.last_decision.strategy == "prefix_cache"
    assert gw.last_decision.chosen in ("v1", "v2")


def test_gateway_advanced_chat_prefers_same_replica_for_same_prefix():
    """Two calls with the same system prompt must land on one replica."""
    cfg = _router_config("prefix_cache", with_fallback=False)
    gw = Gateway(cfg, client_factory=_content_factory({
        "v1": "hi-from-v1", "v2": "hi-from-v2",
    }))
    loop = asyncio.get_event_loop()
    first, _ = loop.run_until_complete(gw.client("g").chat(_msgs()))
    second, _ = loop.run_until_complete(gw.client("g").chat(_msgs()))
    assert first == second  # affinity kept the prefix warm on one replica


def test_gateway_advanced_stream_yields_deltas_then_done():
    cfg = _router_config("least_pending", with_fallback=False)
    gw = Gateway(cfg, client_factory=_content_factory({"v1": "hello world", "v2": "hello world"}))

    async def run():
        return [ev async for ev in gw.client("g").stream(_msgs())]

    events = asyncio.get_event_loop().run_until_complete(run())
    assert events[-1]["type"] == "done"
    assert events[-1]["content"].strip() == "hello world"
    assert gw.last_decision is not None


def test_gateway_advanced_stream_falls_back_on_connection_error():
    cfg = _router_config("least_pending", with_fallback=False)
    # v1 is the least-loaded pick but refuses connections.
    gw = Gateway(cfg, client_factory=_content_factory(
        {"v2": "served by v2"}, fail={"v1"}
    ))
    gw.advanced.metrics.get("v1").num_requests_waiting = 0
    gw.advanced.metrics.get("v2").num_requests_waiting = 3

    async def run():
        return [ev async for ev in gw.client("g").stream(_msgs())]

    events = asyncio.get_event_loop().run_until_complete(run())
    assert events[-1]["type"] == "done"
    assert "served by v2" in events[-1]["content"]
    assert gw.last_decision.chosen == "v2"


def test_gateway_advanced_chat_diverts_to_cloud_when_vllm_down():
    cfg = _router_config("adaptive_fallback")
    gw = Gateway(cfg, client_factory=_content_factory(
        {"cloud": "served-by-foundry"}, fail={"v1", "v2"}
    ))
    content, _ = asyncio.get_event_loop().run_until_complete(
        gw.client("g").chat(_msgs())
    )
    assert content == "served-by-foundry"
    assert gw.last_decision.fallback is True
    assert gw.last_decision.chosen == "cloud"


def test_gateway_advanced_disabled_uses_classic_path():
    cfg = _router_config("prefix_cache", with_fallback=False)
    gw = Gateway(
        cfg, client_factory=_content_factory({"v1": "a", "v2": "b"}), advanced=False
    )
    assert gw.uses_advanced_routing("g") is False
    content, _ = asyncio.get_event_loop().run_until_complete(gw.client("g").chat(_msgs()))
    assert content in ("a", "b")


def test_gateway_classic_group_unaffected_by_advanced_machinery():
    """A non-advanced config must keep the exact classic behaviour."""
    cfg = GatewayConfig(
        endpoints=[_vllm("v1", 8001), _vllm("v2", 8002)],
        groups=[Group(alias="rr", endpoints=["v1", "v2"], strategy="round_robin")],
    )
    gw = Gateway(cfg, client_factory=_content_factory({"v1": "from-a", "v2": "from-b"}))
    assert gw.uses_advanced_routing("rr") is False
    loop = asyncio.get_event_loop()
    seen = {loop.run_until_complete(gw.client("rr").chat(_msgs()))[0] for _ in range(4)}
    assert seen == {"from-a", "from-b"}  # both endpoints used


def test_gateway_lazy_advanced_is_not_built_for_classic_configs():
    cfg = GatewayConfig(
        endpoints=[_vllm("v1", 8001)],
        groups=[Group(alias="rr", endpoints=["v1"], strategy="round_robin")],
    )
    gw = Gateway(cfg, client_factory=_content_factory({}))
    # Nothing advanced was requested, so no collector/ring was constructed.
    assert gw._advanced is None
    assert gw._advanced_started is False
