"""Bridge: FrenchCase vector retrieval → chassis retriever VectorMemory.

FrenchCase's `app/core/vector.py::retrieve()` is a synchronous function that
queries ChromaDB (local) or Azure AI Search (prod) and returns a list of
`{"text", "metadata", "score"}` dicts. The chassis `retriever` pipeline wants a
`memory_store.VectorMemory` (async `add` / `search` / `count`).

This adapter exposes FrenchCase's existing corpus as a read-only VectorMemory,
so the chassis `Retriever` (with its chunker + reranker) works against the real
2,054-chunk TEF corpus without re-embedding anything.

Usage:
    from projects.frenchcase.bridges.retriever_bridge import FrenchCaseVectorMemory
    vm = FrenchCaseVectorMemory()
    hits = await vm.search(MemoryQuery(query="subjonctif", top_k=5, filter={"cefr": "B2"}))
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from memory_store.base import VectorMemory
from memory_store.models import MemoryQuery, MemoryRecord

logger = logging.getLogger(__name__)

# ── FrenchCase import (mounted at runtime) ───────────────────────────────
# `mount_frenchcase()` puts the repo root first (so `app.*` resolves to the
# package) and app/ last as a fallback (so bare `from config import ...` works).
from projects.frenchcase.mount import mount_frenchcase as _mount

_mount()

try:
    from app.core.vector import retrieve as _fc_retrieve  # type: ignore
    FRENCHCASE_AVAILABLE = True
except ImportError:
    try:
        from core.vector import retrieve as _fc_retrieve  # type: ignore
        FRENCHCASE_AVAILABLE = True
    except ImportError:
        _fc_retrieve = None  # type: ignore
        FRENCHCASE_AVAILABLE = False


class FrenchCaseVectorMemory(VectorMemory):
    """Read-only VectorMemory backed by FrenchCase's ChromaDB / Azure AI Search.

    `add()` is a no-op by design: the TEF corpus is authored in Markdown and
    indexed by FrenchCase's own ingestion pipeline (`vector_ingest.py`). The
    chassis retriever is used for *recall* over that corpus, not for creating
    a second copy of it.
    """

    def __init__(self, retrieve_fn: Any | None = None):
        self._retrieve = retrieve_fn or _fc_retrieve
        if self._retrieve is None:
            logger.warning("FrenchCase vector.retrieve unavailable — retriever bridge in no-op mode")

    def _search_sync(
        self,
        query: str,
        top_k: int,
        cefr: Optional[str],
        section: Optional[str],
        skill: Optional[str],
    ) -> list[dict[str, Any]]:
        if self._retrieve is None:
            return []
        try:
            return self._retrieve(
                query, cefr=cefr, section=section, skill=skill, top_k=top_k
            ) or []
        except Exception as exc:
            logger.error("FrenchCase retrieve() failed: %s", exc)
            return []

    async def add(self, records: list[MemoryRecord], embed: bool = True) -> list[str]:
        """No-op: the corpus is owned by FrenchCase's ingestion pipeline."""
        logger.debug("FrenchCaseVectorMemory.add ignored (%d records) — corpus is externally indexed", len(records))
        return []

    async def search(self, query: MemoryQuery) -> list[tuple[MemoryRecord, float]]:
        """Recall from the TEF corpus, mapping hits to (MemoryRecord, score)."""
        filters: dict[str, Any] = query.filter or {}
        hits = await asyncio.to_thread(
            self._search_sync,
            query.query,
            query.top_k,
            filters.get("cefr"),
            filters.get("section") or filters.get("tef_section"),
            filters.get("skill"),
        )

        out: list[tuple[MemoryRecord, float]] = []
        for h in hits:
            meta = dict(h.get("metadata", {}) or {})
            score = float(h.get("score", 0.0) or 0.0)
            if score < query.min_score:
                continue
            # Normalise FrenchCase metadata keys to the chassis vocabulary.
            if "tef_section" in meta and "section" not in meta:
                meta["section"] = meta["tef_section"]
            out.append(
                (
                    MemoryRecord(
                        id=meta.get("chunk_id") or meta.get("source", "")[:64] or "chunk",
                        content=h.get("text", "") or "",
                        metadata=meta,
                    ),
                    score,
                )
            )
        return out

    async def count(self) -> int:
        """Return the corpus size when FrenchCase can report it, else -1 (unknown)."""
        if self._retrieve is None:
            return 0
        try:
            # A broad query is the cheapest portable way to probe the corpus.
            hits = await asyncio.to_thread(self._search_sync, "", 1, None, None, None)
            return 1 if hits else 0
        except Exception:
            return -1


def build_vector_memory(retrieve_fn: Any | None = None) -> VectorMemory:
    """Factory matching `memory_store.build_vector_memory()` shape."""
    return FrenchCaseVectorMemory(retrieve_fn)


if __name__ == "__main__":
    import asyncio as _aio

    async def _main() -> None:
        vm = build_vector_memory()
        print(f"FrenchCase mounted: {FRENCHCASE_AVAILABLE}")
        hits = await vm.search(MemoryQuery(query="subjonctif présent", top_k=3, filter={"cefr": "B2"}))
        print(f"hits: {len(hits)}")
        for rec, score in hits:
            print(f"  [{score:.3f}] {rec.content[:90]}")

    _aio.run(_main())