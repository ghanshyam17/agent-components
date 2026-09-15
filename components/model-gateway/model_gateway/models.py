"""Gateway configuration models.

A gateway is a set of named **endpoints** (real OpenAI-compatible backends —
vLLM, TGI, Ollama, cloud APIs) plus named **groups** that map an alias such as
`"lower"` or `"higher"` to one or more endpoints. The router asks for a client
by alias; the gateway load-balances across the group's endpoints and falls
back across them on error.
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class VLLMConfig(BaseModel):
    """vLLM-specific knobs for an endpoint.

    Only meaningful when the endpoint *is* a vLLM instance; harmless (and
    ignored) for TGI/Ollama/cloud backends, so a mixed group can carry both.
    """

    metrics_url: str | None = Field(
        default=None,
        description="Prometheus /metrics URL; falls back to base_url + /metrics.",
    )
    health_url: str | None = Field(
        default=None, description="Liveness URL; falls back to base_url + /health."
    )
    kv_cache_capacity_gb: float | None = Field(
        default=None, description="GPU KV-cache size, for capacity planning/telemetry."
    )
    draft_endpoint_ref: str | None = Field(
        default=None,
        description="Name of the small draft endpoint to pair with this instance "
        "for speculative decoding.",
    )
    # Thresholds that decide when this replica counts as saturated.
    gpu_cache_saturation: float = Field(
        default=0.85, ge=0.0, le=1.0,
        description="KV-cache utilisation at which the replica is 'saturated'.",
    )
    queue_saturation: int = Field(
        default=8, ge=1, description="Waiting-request count at which it is 'saturated'."
    )
    virtual_nodes: int = Field(
        default=160, ge=1, description="Consistent-hash ring points for prefix affinity."
    )
    weight: int = Field(default=1, ge=1, description="Relative ring weight.")


class Endpoint(BaseModel):
    """One OpenAI-compatible backend."""

    name: str
    base_url: str
    api_key: str = "EMPTY"
    model: str
    weight: int = Field(default=1, ge=1, description="relative weight for weighted strategy")
    rps: float | None = Field(default=None, description="requests/sec cap; None = unlimited")
    max_concurrency: int | None = Field(default=None, description="in-flight request cap")
    timeout: float = 60.0
    enabled: bool = True
    vllm: VLLMConfig | None = Field(
        default=None,
        description="vLLM-specific routing metadata (metrics URL, KV-cache size, "
        "draft model) used by the advanced routing strategies.",
    )

    @property
    def is_vllm(self) -> bool:
        return self.vllm is not None


#: Strategies that need the advanced vLLM router rather than plain ordering.
ADVANCED_STRATEGIES = (
    "prefix_cache",
    "least_pending",
    "speculative",
    "adaptive_fallback",
)


class Group(BaseModel):
    """An alias mapped to a set of endpoints."""

    alias: str
    endpoints: list[str]
    strategy: Literal[
        "round_robin",
        "weighted",
        "failover",
        # --- advanced vLLM strategies (see model_gateway.vllm.router) ---
        "prefix_cache",       # prefer the replica most likely to hold the prefix
        "least_pending",      # prefer the smallest queue depth
        "speculative",        # route drafts/targets for speculative decoding
        "adaptive_fallback",  # primaries first, divert to cloud when saturated
    ] = "round_robin"
    max_retries: int = Field(default=2, ge=0, description="extra attempts after the first")
    retry_backoff: float = Field(default=0.5, ge=0.0, description="base backoff seconds (exponential)")
    #: For `adaptive_fallback`: cloud endpoints to divert to when every primary
    #: is saturated or failing. Tried *after* `endpoints`.
    fallback_endpoints: list[str] = Field(default_factory=list)
    #: How many leading tokens define a cacheable prefix for affinity routing.
    affinity_prefix_tokens: int = Field(
        default=64, ge=1, description="prefix length used for KV-cache affinity."
    )

    @property
    def is_advanced(self) -> bool:
        return self.strategy in ADVANCED_STRATEGIES


class GatewayConfig(BaseModel):
    endpoints: list[Endpoint]
    groups: list[Group]

    @model_validator(mode="after")
    def _validate(self) -> "GatewayConfig":
        names = [e.name for e in self.endpoints]
        if len(names) != len(set(names)):
            raise ValueError("endpoint names must be unique")
        by_name = {e.name: e for e in self.endpoints}
        for g in self.groups:
            for ref in [*g.endpoints, *g.fallback_endpoints]:
                if ref not in by_name:
                    raise ValueError(f"group '{g.alias}' references unknown endpoint '{ref}'")
            if not ref_enabled(g, by_name):
                # allow but warn via config — not fatal
                pass
        # A draft endpoint reference must resolve, or speculation is a no-op
        # the operator cannot see.
        for e in self.endpoints:
            draft = e.vllm.draft_endpoint_ref if e.vllm else None
            if draft is not None and draft not in by_name:
                raise ValueError(
                    f"endpoint '{e.name}' references unknown draft endpoint '{draft}'"
                )
        return self


def ref_enabled(group: Group, by_name: dict[str, Endpoint]) -> bool:
    return any(by_name[n].enabled for n in group.endpoints)