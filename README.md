# agent-components

A curated **monorepo of composable components** for building AI applications.
Each component is a small, focused package you can pull into most AI projects;
they share a thin `core` (common models, logging) so they compose cleanly.

> Status: early. `agentic-router` and `memory-store` exist; the rest of the
> taxonomy is planned (see [Roadmap](#roadmap)).

## Layout

```
agent-components/
  pyproject.toml                # uv workspace root
  core/                        # shared: models, logging (Message, SessionState, ...)
  components/
    agentic-router/            # route tasks to lower/higher model + agentic loop
    memory-store/              # pluggable session + vector memory
  examples/                    # reference compositions
  docs/
```

## Install

Requires Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                         # installs all workspace members + dev deps
uv run pytest                   # runs every component's tests
```

Use one component in another project (it's a normal package once published):

```bash
pip install agentic-router memory-store
```

## Components

### `agentic-router`
A high-level orchestrator that **routes** each task between a lower (fast) and
higher (capable) model — heuristic scorer + LLM-as-judge fallback — then runs a
ReAct loop (tools, multi-step planning, streaming, session memory) against the
chosen model. Works against two OpenAI-compatible vLLM instances.

See [`components/agentic-router/README.md`](components/agentic-router/README.md).

### `memory-store`
Pluggable memory for agents:

* **Short-term** `SessionStore` — conversation + plan keyed by session id,
  with in-memory and Redis backends. Drop-in replacement for ad-hoc session
  dicts; the router injects whichever backend you build.
* **Long-term** `VectorMemory` — store `MemoryRecord`s and recall by cosine
  similarity, with a pluggable `Embedder` (OpenAI-compatible endpoint, or a
  deterministic `HashingEmbedder` for tests).

```python
from memory_store import build_session_store, build_vector_memory
from memory_store.embeddings import HashingEmbedder

sessions = build_session_store()                       # in-memory; backend="redis" for Redis
vm = build_vector_memory(embedder=HashingEmbedder())    # or OpenAIEmbedder(...)
await vm.add([MemoryRecord(content="...")])
hits = await vm.search(MemoryQuery(query="...", top_k=4))
```

## Roadmap

The full curated taxonomy (in priority order). All 10 modular components and the enterprise Platform Engineering layer are completed.

| # | Component | Role | Status |
|---|-----------|------|--------|
| 1 | `agentic-router` | Orchestration: routing + agentic loop + tools | ✅ Done |
| 2 | `memory-store` | Session + vector memory | ✅ Done |
| 3 | `model-gateway` | OpenAI-compatible façade over vLLM/TGI/Ollama/cloud with LB, fallback, rate limits | ✅ Done |
| 4 | `toolkit` | Sandboxed, validated, permissioned tool registry | ✅ Done |
| 5 | `retriever` | RAG: ingest → chunk → embed → vector store → hybrid search → rerank | ✅ Done |
| 6 | `guardrails` | Input/output filtering, PII redaction, injection detection, schema validation | ✅ Done |
| 7 | `tracing` | Structured spans + cost/latency accounting | ✅ Done |
| 8 | `eval-harness` | Dataset-driven eval + LLM-judge metrics | ✅ Done |
| 9 | `prompt-registry` | Versioned, templated prompts + hot-reload | ✅ Done |
| 10 | `agent-ui` | Universal metadata-driven UI bridge (FastAPI + Fluent UI + SSE + Grafana + Power Automate) | ✅ Done |
| 11 | `platform` | Enterprise Platform Engineering: ADF & AML orchestrator, 8 design patterns, Medallion Lakehouse | ✅ Done |

## License

MIT