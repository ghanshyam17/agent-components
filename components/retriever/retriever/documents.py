"""Retriever documents and chunks.

A `Document` is the unit of ingestion — opaque content plus metadata. The
chunker turns each document into `Chunk`s, which are the unit of recall: a
slice of text with a deterministic id, a back-reference to its parent document,
and (after retrieval) an embedding and a relevance score.
"""
from __future__ import annotations

import uuid
from typing import Any

from pydantic import BaseModel, Field


class Document(BaseModel):
    """An ingestible document: content + metadata."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    """A slice of a document produced by a `Chunker`.

    `embedding` is populated by the vector memory on ingest; `score` is the
    relevance score assigned on retrieval (vector similarity and/or reranker).
    """

    id: str
    document_id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    embedding: list[float] | None = None
    score: float | None = None