"""Memory-store settings — env-driven via pydantic-settings."""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEMORY_", env_file=".env", env_file_encoding="utf-8",
        extra="ignore",
    )

    backend: str = Field(
        default="memory",
        description="memory | redis",
    )
    redis_url: str = "redis://localhost:6379/0"
    redis_session_ttl: int | None = None  # seconds; None = persist

    # Embedding endpoint for long-term memory (OpenAI-compatible).
    embed_base_url: str = "http://localhost:8001/v1"
    embed_model: str = "bge-small-en-v1.5"
    embed_api_key: str = "EMPTY"
    embed_dim: int | None = None


@lru_cache
def get_settings() -> MemorySettings:
    return MemorySettings()