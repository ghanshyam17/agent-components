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

## Advanced vLLM routing

For a cluster of **vLLM** instances the interesting signal is not just load but
**KV-cache locality**. vLLM's PagedAttention caches computed prompt prefixes per
instance; a request whose prefix a replica has already seen skips prefill for it
and its time-to-first-token drops sharply. The `model_gateway.vllm` subpackage
adds four strategies that route on that signal plus live telemetry:

| Strategy | Behavior |
|----------|----------|
| `prefix_cache` | Prefer the replica most likely to hold the request's prefix (radix tree + consistent-hash ring), demoting any replica above `gpu_cache_saturation` |
| `least_pending` | Smallest queue depth, then lowest KV-cache pressure, then fewest running requests |
| `speculative` | Load-aware, plus a draft/target decision from measured token acceptance |
| `adaptive_fallback` | Primaries while they have headroom; divert to `fallback_endpoints` (e.g. Azure AI Foundry) when every primary is saturated or errors |

Declare vLLM metadata per endpoint and pick a strategy per group:

```yaml
endpoints:
  - name: vllm-small
    base_url: http://localhost:8001/v1
    model: qwen2.5-1.5b-instruct
    vllm: { kv_cache_capacity_gb: 24 }
  - name: vllm-large
    base_url: http://localhost:8002/v1
    model: qwen2.5-32b-instruct
    vllm:
      draft_endpoint_ref: vllm-small   # speculative decoding partner
      gpu_cache_saturation: 0.85
      queue_saturation: 4
groups:
  - alias: lower
    endpoints: [vllm-small]
    strategy: prefix_cache
  - alias: higher
    endpoints: [vllm-large]
    fallback_endpoints: [cloud-fallback-large]
    strategy: adaptive_fallback
```

Advanced groups are dispatched through `AdvancedVLLMRouter` automatically from
`Gateway.client(alias).chat()` / `.stream()`; the routing decision is exposed as
`gateway.last_decision` for tracing. Start the telemetry poller with
`await gateway.start_advanced()`. A config that uses only classic strategies
never constructs the advanced machinery.

See [`gateway.example.yaml`](gateway.example.yaml) for a complete config and
`tests/test_vllm_router.py` for the routing/fallback behaviour.

## License

MIT