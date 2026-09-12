"""Rerankers: reorder retrieved records by a (cheap) relevance signal.

A reranker sees the candidate records returned by the vector memory and
returns a re-scored, truncated list. We ship two deterministic, dependency-free
implementations:

* `KeywordReranker` — a BM25-ish TF-IDF-lite score over tokens. Because the
  `HashingEmbedder` used in tests is bag-of-words-ish, a keyword reranker is a
  useful second-stage signal that rewards exact term overlap with the query.
* `NoopReranker` — passthrough; preserves order, assigns a zero score.
"""
from __future__ import annotations

import abc
import math
import re
from collections import Counter
from typing import Sequence

from memory_store.models import MemoryRecord


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class Reranker(abc.ABC):
    """Re-score and truncate retrieved records."""

    @abc.abstractmethod
    async def rerank(
        self,
        query: str,
        records: Sequence[MemoryRecord],
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]:
        """Return the top_k records as (record, score), best first."""


class KeywordReranker(Reranker):
    """BM25-ish reranker over tokens, no external dependencies."""

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

    async def rerank(
        self,
        query: str,
        records: Sequence[MemoryRecord],
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]:
        docs_tokens = [_tokens(r.content) for r in records]
        n_docs = len(records)
        if n_docs == 0:
            return []
        df: Counter[str] = Counter()
        for toks in docs_tokens:
            for term in set(toks):
                df[term] += 1
        avgdl = sum(len(t) for t in docs_tokens) / n_docs
        q_terms = set(_tokens(query))
        scored: list[tuple[MemoryRecord, float]] = []
        for rec, toks in zip(records, docs_tokens):
            tf = Counter(toks)
            dl = len(toks)
            score = 0.0
            for term in q_terms:
                f = tf.get(term, 0)
                if f == 0:
                    continue
                n = df[term]
                idf = math.log(1 + (n_docs - n + 0.5) / (n + 0.5))
                denom = f + self.k1 * (1 - self.b + self.b * (dl / (avgdl or 1.0)))
                score += idf * (f * (self.k1 + 1)) / denom
            scored.append((rec, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        return scored[:top_k]


class NoopReranker(Reranker):
    """Passthrough reranker: preserves order, assigns a zero score."""

    async def rerank(
        self,
        query: str,
        records: Sequence[MemoryRecord],
        top_k: int,
    ) -> list[tuple[MemoryRecord, float]]:
        return [(r, 0.0) for r in list(records)[:top_k]]


__all__ = ["Reranker", "KeywordReranker", "NoopReranker"]