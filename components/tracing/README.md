# tracing

Structured tracing + token/cost accounting for AI apps. Keeps an in-memory span
tree by default, with an exporter interface (console / JSON / optional OTLP)
and auto-hooks for model and tool calls via context managers and a decorator.

Part of the [agent-components](../..) monorepo.

## What's inside

| Concept | Interface | Backends / Implementations |
|--------|-----------|------------|
| Span tree | `Span` (pydantic) | in-memory, parent links via `contextvars` |
| Tracer | `Tracer` | records spans, tracks current span per-task |
| Cost | `Usage`, `CostTable`, `CostAccountant` | seeded with common model rates |
| Exporters | `Exporter` (ABC) | `ConsoleExporter`, `JSONExporter`, `OTLPExporter` (stub) |
| Instrumentation | `trace_call`, `record_usage`, `trace_tool_call` | async + sync context managers |

## Install

```bash
uv sync                                # in the monorepo
# or, once published:
pip install tracing                    # core + console/json exporters
pip install "tracing[otel]"             # + opentelemetry-api/sdk for OTLPExporter
```

## Usage

### Span context manager

```python
from tracing import get_tracer

tracer = get_tracer()

with tracer.start_span("outer", {"kind": "root"}):
    with tracer.start_span("inner") as span:
        span.set_attribute("x", 1)
        span.add_event("computed", {"value": 42})

for span in tracer.spans():
    print(span.name, span.parent_id, span.duration)
```

### Tracing a model call (async)

```python
import asyncio
from tracing import get_tracer, trace_call, record_usage, Usage

tracer = get_tracer()

async def chat():
    async with trace_call(tracer, "model.chat", {"model": "gpt-4o"}) as span:
        # ... call the model ...
        record_usage(tracer, Usage(prompt_tokens=120, completion_tokens=80, model="gpt-4o"))

asyncio.run(chat())
print(tracer.cost.total_cost(), tracer.cost.by_model())
```

### Tracing a tool call (sync)

```python
from tracing import get_tracer, trace_tool_call

tracer = get_tracer()

with trace_tool_call(tracer, "search", {"q": "rag"}):
    results = search("rag")
```

### Exporters

```python
from tracing import get_tracer, ConsoleExporter, JSONExporter

tracer = get_tracer()
# ...record spans...
ConsoleExporter().export(tracer.spans())              # log a tree summary
JSONExporter("/tmp/spans.json").export(tracer.spans())  # write a JSON file
```

### Errors

Exceptions inside a span are recorded and re-raised:

```python
with tracer.start_span("risky"):
    raise ValueError("boom")   # span.status == "error", event captured
```

## Configuration (env, `TRACE_` prefix)

| Var | Default | Meaning |
|-----|---------|---------|
| `TRACE_EXPORTER` | `console` | `console` \| `json` \| `otlp` \| `none` |
| `TRACE_JSON_PATH` | `tracing-spans.json` | Path written by the json exporter |

`get_tracer()` builds a process-global `Tracer` with a `CostAccountant`
attached (as `tracer.cost`) and the configured exporter attached as
`tracer.exporter`.

## Notes

- Span times use `time.monotonic()` so durations are immune to system clock
  changes; ids are 16-char hex prefixes of `uuid4`.
- Current-span tracking uses `contextvars`, so concurrent asyncio tasks each
  get their own span chain.
- The `otel` extra is lazy: `OTLPExporter` raises a clear `RuntimeError` if
  `opentelemetry-api`/`opentelemetry-sdk` are missing. The package imports and
  works fully without the extra installed.

## License

MIT