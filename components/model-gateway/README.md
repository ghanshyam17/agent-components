# model-gateway

An **OpenAI-compatible inference gateway** that sits in front of your model
backends (vLLM, TGI, Ollama, cloud APIs) and gives you, per *alias* (e.g.
`"lower"`, `"higher"`):

- **Load balancing** across replicas — round-robin, weighted, or failover order.
- **Fallback** across endpoints in a group when one errors or times out.
- **Retries** with exponential backoff.
- **Rate limiting** per endpoint — rps token bucket + max in-flight concurrency.
- **Metrics** — per-endpoint request/success/error counts and latency.

The router's `ModelClient` interface is a drop-in: `gw.client(alias).chat(...)`
and `.stream(...)` have the same signatures and event shapes, so
`agentic-router` can call the gateway instead of vLLM directly — and the same
router then works whether models are local vLLM, cloud, or a mix.

Part of the [agent-components](../..) monorepo.

## Install

```bash
uv sync --all-packages
# or once published:
pip install model-gateway
pip install "model-gateway[yaml]"   # to load YAML configs
```

## Config

A gateway config lists `endpoints` (real backends) and `groups` (alias →
endpoints). Load from JSON or YAML — see [`gateway.example.yaml`](gateway.example.yaml).

```yaml
endpoints:
  - { name: vllm-small, base_url: http://localhost:8001/v1, api_key: EMPTY, model: qwen2.5-1.5b-instruct }
  - { name: vllm-large, base_url: http://localhost:8002/v1, api_key: EMPTY, model: qwen2.5-32b-instruct }
groups:
  - { alias: lower,  endpoints: [vllm-small], strategy: round_robin, max_retries: 2 }
  - { alias: higher, endpoints: [vllm-large], strategy: failover,    max_retries: 2 }
```

## Usage

```python
from model_gateway import build_gateway

gw = build_gateway("gateway.yaml")          # or Gateway(load_config(...))
lower = gw.client("lower")                   # router-compatible client
content, tool_calls = await lower.chat(messages, tools=tools_schema)

async for ev in lower.stream(messages):       # {"type": "delta"|"tool_calls"|"done", ...}
    ...

print(gw.metrics())                          # per-endpoint counters
```

## Wire the router to the gateway

In the router, set `GATEWAY_CONFIG=gateway.yaml` (and define `lower`/`higher`
groups). `build_agent()` then builds a gateway-backed registry; `LOWER` routes
to the `lower` group, `HIGHER` to `higher`. Without `GATEWAY_CONFIG` the
router falls back to its direct two-vLLM behavior (see the router README).

## Strategies

| Strategy | Behavior |
|----------|----------|
| `round_robin` | rotate the starting endpoint each call (default) |
| `weighted` | pick by `weight`; heavier endpoints get more traffic |
| `failover` | try endpoints in listed order; only move on on error |

Retries: `max_retries` extra attempts after the first, across the ordered
endpoints, with `retry_backoff * 2^i` seconds between attempts.

## License

MIT