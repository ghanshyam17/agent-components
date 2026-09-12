"""retriever: a RAG building block on top of memory-store's VectorMemory.

Chunk documents, embed and store them, recall by similarity, and optionally
rerank — the pieces of retrieval-augmented generation without any model-calling
of its own. Embedding and vector storage come from `memory_store`; this
package adds chunking, a reranker and a small pipeline that ties them
together.

Quick start::

    from retriever import build_retriever, Document

    r = build_retriever()
    await r.ingest([Document(content="..."), ...])
    chunks = await r.search("query", top_k=4)

For offline tests, construct a `Retriever` directly with a `HashingEmbedder`
backed `InMemoryVectorMemory` (see the tests).
"""
from __future__ import annotations

from retriever.chunking import (
    Chunker,
    FixedSizeChunker,
    RecursiveTextChunker,
    SentenceChunker,
)
from retriever.config import RetrieverSettings, get_settings
from retriever.documents import Chunk, Document
from retriever.pipeline import Retriever
from retriever.reranker import KeywordReranker, NoopReranker, Reranker


def _make_chunker(settings: RetrieverSettings) -> Chunker:
    if settings.chunker == "fixed":
        return FixedSizeChunker(size=settings.chunk_size, overlap=settings.chunk_overlap)
    if settings.chunker == "sentence":
        return SentenceChunker(size=settings.chunk_size, overlap=settings.chunk_overlap)
    return RecursiveTextChunker(
        size=settings.chunk_size, overlap=settings.chunk_overlap
    )


def _make_reranker(settings: RetrieverSettings) -> Reranker | None:
    if settings.reranker == "none":
        return NoopReranker()
    return KeywordReranker()


def build_retriever(settings: RetrieverSettings | None = None) -> Retriever:
    """Build a `Retriever` from settings.

    The `VectorMemory` is built via `memory_store.build_vector_memory()`, which
    defaults to an `OpenAIEmbedder` configured from `MEMORY_*` env vars. For
    offline tests, construct a `Retriever` directly with an
    `InMemoryVectorMemory(HashingEmbedder(...))`.
    """
    from memory_store import build_vector_memory

    s = settings or get_settings()
    vm = build_vector_memory()
    chunker = _make_chunker(s)
    reranker = _make_reranker(s)
    return Retriever(vm, chunker, reranker)


__all__ = [
    "Document",
    "Chunk",
    "Chunker",
    "FixedSizeChunker",
    "RecursiveTextChunker",
    "SentenceChunker",
    "Reranker",
    "KeywordReranker",
    "NoopReranker",
    "Retriever",
    "RetrieverSettings",
    "get_settings",
    "build_retriever",
]