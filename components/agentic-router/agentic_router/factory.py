"""Dependency wiring — build the Agent and its collaborators once.

By default session memory is the in-memory backend from `memory-store`; swap in
`memory_store.build_session_store()` (Redis backend) or your own `SessionStore`
implementation to persist across restarts.

If `gateway_config` is set (or the `GATEWAY_CONFIG` / `gateway_config` setting
is set), model calls route through a `model-gateway` Gateway instead of the
direct two-vLLM clients — enabling load balancing, fallback, retries and rate
limits across multiple OpenAI-compatible backends.
"""
from __future__ import annotations

from agentic_router.agent import Agent
from agentic_router.clients import ClientRegistry, GatewayClientRegistry
from agentic_router.config import Settings, get_settings
from agentic_router.models import ModelTier
from agentic_router.router.router import Router
from agentic_router.tools import build_default_registry
from memory_store import InMemorySessionStore, SessionStore


def _maybe_build_gateway(settings: Settings, gateway=None):
    """Return a model-gateway Gateway if configured, else None.

    Precedence: explicit `gateway` arg > settings.gateway_config file > the
    model-gateway component's own GATEWAY_CONFIG env (via get_config()).
    """
    if gateway is not None:
        return gateway
    from model_gateway import Gateway, load_config

    path = settings.gateway_config
    if path:
        return Gateway(load_config(path))
    # Fall back to the gateway component's env-driven config (GATEWAY_CONFIG).
    from model_gateway import get_config as _gw_get_config

    cfg = _gw_get_config()
    if cfg.endpoints:
        return Gateway(cfg)
    return None


def build_agent(
    settings: Settings | None = None,
    sessions: SessionStore | None = None,
    gateway=None,
) -> Agent:
    s = settings or get_settings()

    gw = _maybe_build_gateway(s, gateway)
    if gw is not None:
        registry: ClientRegistry | GatewayClientRegistry = GatewayClientRegistry(
            gw, s.lower_alias, s.higher_alias
        )
    else:
        registry = ClientRegistry(s)

    router = Router(s, registry.get(ModelTier.LOWER))
    tools = build_default_registry()
    sessions = sessions or InMemorySessionStore()
    return Agent(s, registry, router, tools, sessions)