"""Loading gateway config from a JSON or YAML file, plus env-driven settings."""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from model_gateway.models import GatewayConfig


class GatewaySettings(BaseSettings):
    """Top-level env knobs. Nested endpoint/group config is loaded from a file
    (JSON or YAML) pointed to by `GATEWAY_CONFIG`, because env vars don't map
    well to lists of objects.
    """

    model_config = SettingsConfigDict(
        env_prefix="GATEWAY_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    config: str | None = None  # path to a JSON/YAML gateway config file


def load_config(path: str | os.PathLike[str] | None = None) -> GatewayConfig:
    """Load a GatewayConfig from a JSON or YAML file (by extension)."""
    p = Path(path) if path else None
    if p is None:
        return GatewayConfig(endpoints=[], groups=[])
    text = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "YAML config requires the 'yaml' extra: pip install model-gateway[yaml]"
            ) from e
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    return GatewayConfig.model_validate(data)


@lru_cache
def get_config() -> GatewayConfig:
    s = GatewaySettings()
    return load_config(s.config) if s.config else GatewayConfig(endpoints=[], groups=[])