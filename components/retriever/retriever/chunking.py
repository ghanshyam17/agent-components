"""Text chunking strategies.

A `Chunker` turns a `Document` into a list of `Chunk`s. The chunkers here are
pure text operations — no I/O, no models — so they are cheap, deterministic and
easy to test. Chunk ids are a stable hash of (document id, index, text) so the
same document always chunks the same way.

Three strategies ship:

* `FixedSizeChunker` — character windows with overlap. Simple and predictable.
* `RecursiveTextChunker` — split on successively finer separators
  (paragraph → line → sentence → space), then merge to the size budget. Keeps
  semantic boundaries where it can; the usual default.
* `SentenceChunker` — split on sentence boundaries, then merge sentences to
  the size budget.
"""
from __future__ import annotations

import abc
import hashlib
import re
from typing import Any

from retriever.documents import Chunk, Document


def _chunk_id(document_id: str, text: str, index: int) -> str:
    """Deterministic chunk id: hash of (document id, index, text)."""
    h = hashlib.sha1(f"{document_id}:{index}:{text}".encode("utf-8")).hexdigest()
    return h[:16]


def _merge(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Greedily merge small pieces into chunks of <= `size` chars.

    When starting a new chunk, the trailing `overlap` characters of the
    previous chunk are prepended, so adjacent chunks share a little context.
    """
    if not pieces:
        return []
    chunks: list[str] = []
    current = ""
    for piece in pieces:
        if not piece:
            continue
        if not current:
            current = piece
        elif len(current) + len(piece) <= size:
            current += piece
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail + piece
    if current:
        chunks.append(current)
    return chunks


class Chunker(abc.ABC):
    """Splits a `Document` into `Chunk`s."""

    @abc.abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Return the chunks for `document`, in order."""


class FixedSizeChunker(Chunker):
    """Character-window chunker with overlap."""

    def __init__(self, size: int = 800, overlap: int = 100):
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        step = self.size - self.overlap
        out: list[Chunk] = []
        idx = 0
        i = 0
        n = len(text)
        while i < n:
            piece = text[i : i + self.size]
            out.append(
                Chunk(
                    id=_chunk_id(document.id, piece, idx),
                    document_id=document.id,
                    text=piece,
                    metadata=dict(document.metadata),
                )
            )
            idx += 1
            if i + self.size >= n:
                break
            i += step
        return out


class RecursiveTextChunker(Chunker):
    """Recursive separator chunker, langchain-style.

    Tries each separator in turn; the first one present in the text is used to
    split. Oversized pieces are recursed on with the remaining (finer)
    separators. The resulting small pieces are then merged to the size budget
    with overlap.
    """

    def __init__(
        self,
        size: int = 800,
        overlap: int = 100,
        separators: list[str] | None = None,
    ):
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap
        self.separators = list(separators) if separators else ["\n\n", "\n", ". ", " "]

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.size:
            return [text]
        for i, sep in enumerate(separators):
            if sep == "":
                # Character-level fallback for a piece no separator could break.
                return [text[j : j + self.size] for j in range(0, len(text), self.size)]
            if sep in text:
                parts = text.split(sep)
                pieces: list[str] = []
                for p in parts[:-1]:
                    pieces.append(p + sep)
                if parts[-1]:
                    pieces.append(parts[-1])
                rest = separators[i + 1 :] or [""]
                out: list[str] = []
                for p in pieces:
                    if len(p) > self.size:
                        out.extend(self._split(p, rest))
                    else:
                        out.append(p)
                return out
        # No separator matched; fall back to fixed windows.
        return [text[j : j + self.size] for j in range(0, len(text), self.size)]

    def chunk(self, document: Document) -> list[Chunk]:
        pieces = self._split(document.content, self.separators)
        texts = _merge(pieces, self.size, self.overlap)
        return [
            Chunk(
                id=_chunk_id(document.id, t, idx),
                document_id=document.id,
                text=t,
                metadata=dict(document.metadata),
            )
            for idx, t in enumerate(texts)
        ]


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(Chunker):
    """Sentence-aware chunker: split on sentence boundaries, then merge."""

    def __init__(self, size: int = 800, overlap: int = 100):
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap < 0 or overlap >= size:
            raise ValueError("overlap must be in [0, size)")
        self.size = size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        text = document.content
        if not text:
            return []
        sentences = [s for s in _SENT_SPLIT.split(text) if s]
        texts = _merge(sentences, self.size, self.overlap)
        return [
            Chunk(
                id=_chunk_id(document.id, t, idx),
                document_id=document.id,
                text=t,
                metadata=dict(document.metadata),
            )
            for idx, t in enumerate(texts)
        ]


__all__ = [
    "Chunker",
    "FixedSizeChunker",
    "RecursiveTextChunker",
    "SentenceChunker",
]