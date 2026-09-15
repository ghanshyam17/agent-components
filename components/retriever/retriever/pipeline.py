"""The `Retriever` pipeline: chunk → embed/store → recall → rerank.

`Retriever` composes three pieces from `memory-store` and this package:

* a `VectorMemory` (from `memory_store`) for embedding and similarity recall,
* a `Chunker` (this package) for splitting ingested documents,
* an optional `Reranker` (this package) for a second-stage re-score.

It maps the `MemoryRecord`/`MemoryQuery` shape from `memory_store` to the
`Document`/`Chunk` shape this component exposes, so callers never touch the
memory-store types directly.
"""
from __future__ import annotations

from typing import Any

from components_core import get_logger
from memory_store.base import VectorMemory
from memory_store.models import MemoryQuery, MemoryRecord

from retriever.chunking import Chunker
from retriever.documents import Chunk, Document
from retriever.reranker import Reranker

logger = get_logger("retriever")


class Retriever:
    """A RAG pipeline over a `memory_store.VectorMemory`.

    Ingest chunks each document, embeds and stores the chunks as
    `MemoryRecord`s. Search runs vector recall then (optionally) reranks.
    """

    def __init__(
        self,
        vector_memory: VectorMemory,
        chunker: Chunker,
        reranker: Reranker | None = None,
    ):
        self.vector_memory = vector_memory
        self.chunker = chunker
        self.reranker = reranker

    async def ingest(self, documents: list[Document]) -> int:
        """Chunk, embed and store `documents`; return the number of chunks added."""
        total = 0
        for doc in documents:
            chunks = self.chunker.chunk(doc)
            records = [
                MemoryRecord(
                    content=c.text,
                    metadata={
                        "document_id": c.document_id,
                        "chunk_id": c.id,
                        **c.metadata,
                    },
                )
                for c in chunks
            ]
            if records:
                await self.vector_memory.add(records)
                total += len(records)
        logger.info("ingested %d documents -> %d chunks", len(documents), total)
        return total

    async def search(
        self,
        query: str,
        top_k: int = 4,
        filter: dict[str, Any] | None = None,
        min_score: float | None = None,
    ) -> list[Chunk]:
        """Recall candidates from the vector memory and rerank to `top_k`.

        `min_score` is a cosine floor applied during recall. It defaults to
        ``-1.0`` ("no floor"), *not* to the `MemoryQuery` default of ``0.0``:
        over-fetching exists to let the reranker reorder, and a 0.0 floor would
        silently drop unrelated-but-better-than-nothing candidates before the
        reranker ever sees them. Pass an explicit value to filter early.
        """
        if top_k <= 0:
            return []
        floor = -1.0 if min_score is None else min_score
        # Over-fetch so the reranker has room to reorder.
        hits = await self.vector_memory.search(
            MemoryQuery(query=query, top_k=top_k * 3, filter=filter, min_score=floor)
        )
        if self.reranker is not None:
            records = [r for r, _ in hits]
            reranked = await self.reranker.rerank(query, records, top_k)
            results = reranked
        else:
            results = [(r, s) for r, s in hits[:top_k]]

        chunks: list[Chunk] = []
        for rec, score in results:
            meta = dict(rec.metadata)
            doc_id = meta.pop("document_id", "")
            chunk_id = meta.pop("chunk_id", rec.id)
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=doc_id,
                    text=rec.content,
                    metadata=meta,
                    embedding=rec.embedding,
                    score=score,
                )
            )
        return chunks

    async def count(self) -> int:
        """Return the number of records stored in the underlying memory."""
        return await self.vector_memory.count()


__all__ = ["Retriever"]