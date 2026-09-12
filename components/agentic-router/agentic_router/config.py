"""Central configuration for the agentic router.

All values can be overridden via environment variables (or a `.env` file);
see `.env.example` for the full list.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- vLLM endpoints (OpenAI-compatible) ---
    lower_vllm_base_url: str = "http://localhost:8001/v1"
    lower_model: str = "qwen2.5-1.5b-instruct"

    higher_vllm_base_url: str = "http://localhost:8002/v1"
    higher_model: str = "qwen2.5-32b-instruct"

    vllm_api_key: str = "EMPTY"

    # --- Routing thresholds (hybrid router) ---
    # Calibrated to the heuristic's score distribution: clear-cut low prompts
    # score ~0.0, single-cue prompts ~0.2, loaded multi-cue/code prompts ~0.5+.
    router_lower_threshold: float = 0.15
    router_higher_threshold: float = 0.45
    # A score within `band` of either threshold is also sent to the classifier
    # (near-threshold decisions are the least reliable).
    router_classifier_band: float = 0.05

    # --- Agentic loop ---
    agent_max_iterations: int = 8
    agent_max_plan_subtasks: int = 5
    agent_enable_planning: bool = True
    agent_enable_tools: bool = True

    # --- model-gateway integration ---
    # If set, build_agent routes through a model-gateway Gateway loaded from
    # this config file (JSON/YAML) instead of direct vLLM clients. The tier
    # maps to a gateway group alias:
    lower_alias: str = "lower"
    higher_alias: str = "higher"
    gateway_config: str | None = None  # path to a gateway config file

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8000

    # --- Misc ---
    log_level: Literal["debug", "info", "warning", "error"] = "info"


@lru_cache
def get_settings() -> Settings:
    return Settings()