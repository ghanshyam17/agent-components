"""Retriever settings — env-driven via pydantic-settings (prefix `RETRIEVER_`).

Embedding/backend configuration is delegated to `memory_store`: `build_retriever`
constructs the `VectorMemory` via `memory_store.build_vector_memory()`, so the
`MEMORY_*` env vars (embed endpoint, model, dim) govern the store. Only the
retriever-specific knobs (chunking + reranker) live here.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class RetrieverSettings(BaseSettings):
    """Retriever configuration (env prefix `RETRIEVER_`)."""

    model_config = SettingsConfigDict(
        env_prefix="RETRIEVER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    chunk_size: int = Field(default=800, description="Chunk size in characters.")
    chunk_overlap: int = Field(default=100, description="Overlap between chunks.")
    chunker: Literal["fixed", "recursive", "sentence"] = Field(
        default="recursive", description="Chunking strategy."
    )
    reranker: Literal["keyword", "none"] = Field(
        default="keyword", description="Reranker to apply after vector recall."
    )


@lru_cache
def get_settings() -> RetrieverSettings:
    return RetrieverSettings()