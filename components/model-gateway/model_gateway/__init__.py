"""model-gateway: an OpenAI-compatible inference gateway.

Load balance, fall back, retry and rate-limit across multiple vLLM/TGI/
Ollama/cloud endpoints, exposed as router-compatible `GatewayClient`s keyed by
alias (e.g. "lower", "higher").

Quick start::

    from model_gateway import Gateway, load_config
    gw = Gateway(load_config("gateway.yaml"))
    lower = gw.client("lower")            # same .chat()/.stream() as the router's clients
    content, tool_calls = await lower.chat(messages)
"""
from __future__ import annotations

from model_gateway.config import GatewaySettings, get_config, load_config
from model_gateway.gateway import Gateway, GatewayClient, EndpointRunner
from model_gateway.models import ADVANCED_STRATEGIES, Endpoint, GatewayConfig, Group, VLLMConfig

__all__ = [
    "Gateway", "GatewayClient", "EndpointRunner",
    "Endpoint", "Group", "GatewayConfig", "VLLMConfig",
    "ADVANCED_STRATEGIES",
    "GatewaySettings", "load_config", "get_config", "build_gateway",
    "AdvancedVLLMRouter", "PrefixCacheAffinity", "VLLMMetricsCollector",
    "SpeculativeDecodingCoordinator",
]


def __getattr__(name: str):
    """Lazily expose the vLLM subsystem (keeps the base import dependency-free)."""
    if name == "AdvancedVLLMRouter":
        from model_gateway.vllm.router import AdvancedVLLMRouter

        return AdvancedVLLMRouter
    if name == "PrefixCacheAffinity":
        from model_gateway.vllm.cache_affinity import PrefixCacheAffinity

        return PrefixCacheAffinity
    if name == "VLLMMetricsCollector":
        from model_gateway.vllm.metrics_collector import VLLMMetricsCollector

        return VLLMMetricsCollector
    if name == "SpeculativeDecodingCoordinator":
        from model_gateway.vllm.speculative import SpeculativeDecodingCoordinator

        return SpeculativeDecodingCoordinator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def build_gateway(config: "GatewayConfig | None" = None) -> Gateway:
    """Convenience: build a Gateway from a config object, a path string, or env."""
    from model_gateway.models import GatewayConfig

    if config is None:
        return Gateway(get_config())
    if isinstance(config, GatewayConfig):
        return Gateway(config)
    if isinstance(config, str):
        return Gateway(load_config(config))
    raise TypeError(f"unsupported config type: {type(config)!r}")