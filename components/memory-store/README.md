# memory-store

Pluggable memory for AI agents: **short-term** session memory and **long-term**
vector memory, behind clean interfaces with swappable backends.

Part of the [agent-components](../..) monorepo.

## What's inside

| Concept | Interface | Backends |
|--------|-----------|----------|
| Short-term (session) | `SessionStore` | `InMemorySessionStore`, `RedisSessionStore` |
| Long-term (semantic) | `VectorMemory` | `InMemoryVectorMemory` (brute-force cosine) |
| Embeddings | `Embedder` (protocol) | `OpenAIEmbedder` (vLLM/cloud), `HashingEmbedder` (deterministic, tests) |

Short-term types (`Message`, `SessionState`, `Plan`, `SubTask`) are shared via
`core` and re-exported here.

## Install

```bash
uv sync                                # in the monorepo
# or, once published:
pip install memory-store               # in-memory backends
pip install "memory-store[redis]"      # + Redis session backend
```

## Usage

### Short-term session memory

```python
from memory_store import build_session_store, InMemorySessionStore
from components_core import Message, Plan, SubTask

sessions = build_session_store()          # backend="memory" by default; "redis" → Redis
state = await sessions.create()
await sessions.add_message(state.session_id, Message(role="user", content="hi"))
await sessions.set_plan(state.session_id, Plan(goal="...", subtasks=[SubTask(description="...")]))
got = await sessions.get(state.session_id)
```

Redis backend (set via env or directly):

```python
from memory_store.config import MemorySettings
sessions = build_session_store(MemorySettings(backend="redis", redis_url="redis://localhost:6379/0"))
```

### Long-term vector memory

```python
from memory_store import build_vector_memory, MemoryRecord, MemoryQuery
from memory_store.embeddings import OpenAIEmbedder

embedder = OpenAIEmbedder(model="bge-small-en-v1.5", base_url="http://localhost:8001/v1")
vm = build_vector_memory(embedder=embedder)
await vm.add([MemoryRecord(content="the model routes by complexity", metadata={"kind": "doc"})])
hits = await vm.search(MemoryQuery(query="how does routing decide?", top_k=4))
for record, score in hits:
    print(score, record.content)
```

For tests/demos without a model endpoint:

```python
from memory_store.embeddings import HashingEmbedder
vm = build_vector_memory(embedder=HashingEmbedder(dim=256))
```

## Configuration (env, `MEMORY_` prefix)

| Var | Default | Meaning |
|-----|---------|---------|
| `MEMORY_BACKEND` | `memory` | `memory` \| `redis` |
| `MEMORY_REDIS_URL` | `redis://localhost:6379/0` | Redis URL |
| `MEMORY_REDIS_SESSION_TTL` | unset (persist) | session TTL in seconds |
| `MEMORY_EMBED_BASE_URL` | `http://localhost:8001/v1` | embeddings endpoint |
| `MEMORY_EMBED_MODEL` | `bge-small-en-v1.5` | embedding model name |
| `MEMORY_EMBED_DIM` | unset (auto) | vector dimensionality |

## Notes

- `InMemoryVectorMemory` is brute-force — fine up to a few thousand records.
  For scale, a pgvector / Qdrant backend belongs in the future `retriever`
  component (which will build on this `VectorMemory` interface).
- `RedisSessionStore` stores each session as JSON under `sess:<id>`; set a TTL
  for ephemeral sessions.

## License

MIT