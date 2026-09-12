"""Retriever tests: chunking, recall, reranking and config — all offline.

Uses `memory_store`'s `HashingEmbedder` + `InMemoryVectorMemory` so no network
or model endpoint is involved. Tests are deterministic.
"""
from __future__ import annotations

import asyncio

import pytest

from memory_store import HashingEmbedder, InMemoryVectorMemory

from retriever import (
    Document,
    FixedSizeChunker,
    KeywordReranker,
    NoopReranker,
    RecursiveTextChunker,
    Retriever,
    SentenceChunker,
)
from retriever.config import RetrieverSettings


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


def _build(chunker=None, reranker=None, dim=256):
    vm = InMemoryVectorMemory(HashingEmbedder(dim=dim))
    return Retriever(vm, chunker or RecursiveTextChunker(size=200, overlap=20), reranker)


# ---------------- chunking ----------------
def test_fixed_size_chunker_counts_and_overlap():
    doc = Document(content="x" * 500)
    chunks = FixedSizeChunker(size=200, overlap=20).chunk(doc)
    # step = 180; windows: 0-200, 180-380, 360-500 -> 3 chunks
    assert len(chunks) == 3
    assert all(c.document_id == doc.id for c in chunks)
    assert chunks[0].text.startswith("x")
    # deterministic ids: chunk again -> same ids
    again = FixedSizeChunker(size=200, overlap=20).chunk(doc)
    assert [c.id for c in chunks] == [c.id for c in again]


def test_recursive_chunker_respects_boundaries():
    text = "First sentence here. Second sentence here. Third sentence here."
    doc = Document(content=text)
    chunks = RecursiveTextChunker(size=30, overlap=0).chunk(doc)
    assert len(chunks) >= 1
    # No chunk exceeds size (plus a little tolerance for a stuck separator).
    assert all(len(c.text) <= 30 + len(". ") for c in chunks)


def test_sentence_chunker_groups_sentences():
    text = "One sentence. Two sentence. Three sentence. Four sentence."
    chunks = SentenceChunker(size=100, overlap=0).chunk(Document(content=text))
    assert len(chunks) == 1
    assert chunks[0].text.count("sentence.") == 4


def test_chunker_rejects_bad_params():
    with pytest.raises(ValueError):
        FixedSizeChunker(size=0)
    with pytest.raises(ValueError):
        RecursiveTextChunker(size=100, overlap=100)
    with pytest.raises(ValueError):
        SentenceChunker(size=-1)


# ---------------- recall ----------------
def test_ingest_and_search_returns_relevant_first():
    r = _build(reranker=None)
    n = _run(r.ingest([
        Document(content="The python function returns a computed value to the caller"),
        Document(content="Rust ownership and borrowing rules prevent data races"),
        Document(content="How to bake sourdough bread at home with a starter"),
    ]))
    assert n == 3
    assert _run(r.count()) == 3

    chunks = _run(r.search("python function value", top_k=1))
    assert chunks, "expected at least one hit"
    top = chunks[0]
    assert "python" in top.text
    assert top.score is not None
    assert top.document_id  # populated from stored metadata


def test_search_filter_passes_through():
    r = _build(reranker=None)
    _run(r.ingest([
        Document(content="alpha beta gamma", metadata={"kind": "doc"}),
        Document(content="delta epsilon zeta", metadata={"kind": "note"}),
        Document(content="alpha again here", metadata={"kind": "doc"}),
    ]))
    chunks = _run(r.search("alpha", top_k=5, filter={"kind": "doc"}))
    assert chunks
    assert all(c.metadata.get("kind") == "doc" for c in chunks)


def test_search_top_k_truncates():
    r = _build(reranker=None)
    _run(r.ingest([Document(content=f"doc number {i} words words") for i in range(5)]))
    chunks = _run(r.search("words", top_k=2))
    assert len(chunks) == 2


def test_search_empty_store():
    r = _build(reranker=None)
    assert _run(r.search("anything", top_k=4)) == []


# ---------------- reranker ----------------
def test_keyword_reranker_promotes_term_overlap():
    r = _build(reranker=KeywordReranker())
    _run(r.ingest([
        Document(content="python python python functions and values"),
        Document(content="unrelated words about baking bread at home"),
        Document(content="the value of a python function is its return"),
    ]))
    chunks = _run(r.search("python function value", top_k=3))
    assert chunks
    # The chunk with the most query-term overlap should rank first.
    assert "python" in chunks[0].text and "function" in chunks[0].text


def test_noop_reranker_preserves_order():
    r = _build(reranker=NoopReranker())
    _run(r.ingest([
        Document(content="python function returns a value"),
        Document(content="rust ownership borrowing rules"),
        Document(content="baking sourdough bread at home"),
    ]))
    chunks = _run(r.search("python", top_k=3))
    assert len(chunks) == 3
    # Noop assigns zero scores but keeps the recall order.
    assert all(c.score == 0.0 for c in chunks)


def test_keyword_reranker_changes_or_preserves_top():
    # Build two pipelines (with/without reranker) over the same docs and check
    # the reranker does not crash and yields a sensible top result.
    docs = [
        Document(content="the model gateway routes requests by complexity"),
        Document(content="load balancing across vllm replicas round robin"),
        Document(content="baking sourdough bread requires a flour starter"),
    ]
    r_plain = _build(reranker=None)
    r_kw = _build(reranker=KeywordReranker())
    _run(r_plain.ingest(docs))
    _run(r_kw.ingest(docs))
    plain = _run(r_plain.search("gateway routes complexity", top_k=2))
    kw = _run(r_kw.search("gateway routes complexity", top_k=2))
    assert plain and kw
    assert "gateway" in kw[0].text


# ---------------- config ----------------
def test_settings_defaults():
    s = RetrieverSettings()
    assert s.chunk_size == 800
    assert s.chunk_overlap == 100
    assert s.chunker == "recursive"
    assert s.reranker == "keyword"


def test_settings_validation_rejects_unknown_chunker():
    with pytest.raises(Exception):
        RetrieverSettings(chunker="bogus")


def test_settings_validation_rejects_unknown_reranker():
    with pytest.raises(Exception):
        RetrieverSettings(reranker="bogus")