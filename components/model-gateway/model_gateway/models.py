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


class Group(BaseModel):
    """An alias mapped to a set of endpoints."""

    alias: str
    endpoints: list[str]
    strategy: Literal["round_robin", "weighted", "failover"] = "round_robin"
    max_retries: int = Field(default=2, ge=0, description="extra attempts after the first")
    retry_backoff: float = Field(default=0.5, ge=0.0, description="base backoff seconds (exponential)")


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
            for ref in g.endpoints:
                if ref not in by_name:
                    raise ValueError(f"group '{g.alias}' references unknown endpoint '{ref}'")
            if not ref_enabled(g, by_name):
                # allow but warn via config — not fatal
                pass
        return self


def ref_enabled(group: Group, by_name: dict[str, Endpoint]) -> bool:
    return any(by_name[n].enabled for n in group.endpoints)