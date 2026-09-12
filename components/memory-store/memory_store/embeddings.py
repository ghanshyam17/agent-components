"""Embedding clients for long-term vector memory.

`Embedder` is a tiny protocol: anything that turns text into a fixed-length
float vector. We ship two implementations:

* `OpenAIEmbedder` — calls an OpenAI-compatible embeddings endpoint (vLLM
  serves these at `/v1/embeddings`), so it works against your local vLLM too.
* `HashingEmbedder` — a deterministic, dependency-free embedder used for tests
  and local demos. NOT semantic; only good enough to exercise the vector
  memory pipeline offline.
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol, runtime_checkable

import httpx


@runtime_checkable
class Embedder(Protocol):
    dim: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class OpenAIEmbedder:
    """Embeds via an OpenAI-compatible `/v1/embeddings` endpoint (vLLM/cloud)."""

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8001/v1",
        api_key: str = "EMPTY",
        dim: int | None = None,
        timeout: float = 30.0,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        # `dim` is resolved lazily on the first call if not provided.
        self._dim = dim

    @property
    def dim(self) -> int:
        if self._dim is None:
            raise RuntimeError(
                "OpenAIEmbedder.dim unknown until the first embed() call"
            )
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(
                f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={"model": self.model, "input": texts},
            )
            resp.raise_for_status()
            data = resp.json()["data"]
            # OpenAI returns objects sorted by index.
            vecs = [d["embedding"] for d in sorted(data, key=lambda x: x["index"])]
            if self._dim is None and vecs:
                self._dim = len(vecs[0])
            return vecs


class HashingEmbedder:
    """Deterministic hashing embedder — for tests / offline demos only.

    Maps each token to a bucket via SHA-1 and signs it with the token's
    presence. Two texts with overlapping tokens get a non-zero cosine
    similarity; totally unrelated text ≈ orthogonal. Not a real semantic
    embedding.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            vec = [0.0] * self.dim
            for tok in text.lower().split():
                h = int(hashlib.sha1(tok.encode("utf-8")).hexdigest(), 16)
                vec[h % self.dim] += 1.0
                vec[(h >> 1) % self.dim] -= 0.5
            # L2 normalize.
            norm = math.sqrt(sum(v * v for v in vec)) or 1.0
            out.append([v / norm for v in vec])
        return out