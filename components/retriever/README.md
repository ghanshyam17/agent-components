# retriever

A small **retrieval-augmented-generation (RAG)** building block that builds on
`memory-store`'s `VectorMemory`: chunk documents, embed and store them, recall
by similarity, and optionally rerank.

It adds the pieces `memory-store` deliberately leaves out — chunking and a
second-stage reranker — without any model-calling of its own. Embedding and
vector storage come from `memory-store`; this package composes them into a
pipeline.

Part of the [agent-components](../..) monorepo.

## What's inside

| Concept | Interface | Implementations |
|--------|-----------|-----------------|
| Documents | `Document`, `Chunk` | pydantic models |
| Chunking | `Chunker` (ABC) | `FixedSizeChunker`, `RecursiveTextChunker`, `SentenceChunker` |
| Reranking | `Reranker` (ABC) | `KeywordReranker` (BM25-ish), `NoopReranker` |
| Pipeline | `Retriever` | chunk → embed/store → recall → rerank |

## Install

```bash
uv sync --all-packages
# or once published:
pip install retriever
pip install "retriever[yaml]"   # if you load YAML config (future)
```

## Usage

```python
from retriever import build_retriever, Document

r = build_retriever()                                   # uses MEMORY_* for the store
await r.ingest([
    Document(content="The python function returns a value."),
    Document(content="Rust ownership and borrowing rules."),
    Document(content="How to bake sourdough bread at home."),
])
chunks = await r.search("python function value", top_k=4)
for c in chunks:
    print(c.score, c.text, c.metadata)
```

For tests/offline use, build a `Retriever` directly with a deterministic,
no-network embedder from `memory-store`:

```python
from memory_store import HashingEmbedder, InMemoryVectorMemory
from retriever import RecursiveTextChunker, KeywordReranker, Retriever

vm = InMemoryVectorMemory(HashingEmbedder(dim=256))
r = Retriever(vm, RecursiveTextChunker(size=800, overlap=100), KeywordReranker())
```

## Configuration (env, `RETRIEVER_` prefix)

Embedding/vector-store configuration is delegated to `memory-store` (see the
`MEMORY_*` table in the memory-store README). Only retriever-specific knobs
live here.

| Var | Default | Meaning |
|-----|---------|---------|
| `RETRIEVER_CHUNK_SIZE` | `800` | Chunk size in characters |
| `RETRIEVER_CHUNK_OVERLAP` | `100` | Overlap between adjacent chunks |
| `RETRIEVER_CHUNKER` | `recursive` | `fixed` \| `recursive` \| `sentence` |
| `RETRIEVER_RERANKER` | `keyword` | `keyword` \| `none` |

## Notes

- Chunk ids are a stable hash of `(document_id, index, text)` — the same
  document always chunks the same way.
- `KeywordReranker` is a dependency-free BM25-ish signal over tokens; it
  rewards exact term overlap with the query and is deterministic.
- `build_retriever()` defaults to `memory-store`'s `OpenAIEmbedder` (via
  `MEMORY_*`); pass a `HashingEmbedder`-backed `VectorMemory` directly for
  offline tests.

## License

MIT