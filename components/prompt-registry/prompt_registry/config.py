"""Prompt-registry settings — env-driven via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class PromptSettings(BaseSettings):
    """Env knobs (prefix ``PROMPT_``)."""

    model_config = SettingsConfigDict(
        env_prefix="PROMPT_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    dir: str | None = None  # prompt directory to load on build
    default_version: str = "latest"  # "latest" or a concrete version


@lru_cache
def get_settings() -> PromptSettings:
    return PromptSettings()