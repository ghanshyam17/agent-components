# Agentic Router

A high-level, **vLLM-backed** orchestration layer that **routes** each task
between a **lower** (small / fast) model and a **higher** (large / capable)
model — and then runs an **agentic loop** (tool use, multi-step planning,
streaming, session memory) against whichever model it chose.

```
                 ┌─────────────────────────────────────────────┐
   task ───────► │  Hybrid Router                              │
                 │   heuristic scorer ──► (uncertain?) ──►      │
                 │   LLM-as-judge classifier (lower model)       │
                 └───────────────┬─────────────────────────────┘
                                 │ RouteDecision{tier, score, method}
              ┌──────────────────┴──────────────────┐
              ▼                                     ▼
       lower vLLM :8001                       higher vLLM :8002
              └──────────────┬──────────────────┘
                             ▼
                     ReAct agent loop  (tools → plan → stream → memory)
                             ▼
                       streamed AgentEvents (SSE)
```

## Features

- **Hybrid routing** — fast heuristic scorer (length, keyword salience, code
  presence, tool/multi-step intent); an LLM-as-judge classifier (run on the
  *lower* model, so it's cheap) breaks ties in the uncertain band.
- **Two vLLM instances** — each is an OpenAI-compatible endpoint; the router
  picks one per (sub)task.
- **Agentic loop** — ReAct-style tool calling with up to N iterations.
- **Multi-step planning** — the higher model decomposes multi-step goals into
  subtasks, each routed independently.
- **Streaming** — token deltas, tool calls and results stream as `AgentEvent`s
  over SSE (`GET /agent/stream`) or the `stream` CLI command.
- **Session memory** — conversation + plan persisted per `session_id` (in-memory;
  swap in Redis behind the same `SessionStore` interface).
- **Built-in tools** — `run_shell`, `read_file`, `write_file`, `web_fetch`,
  `calculator` (safe, no `eval`).
- **FastAPI server** + **CLI**.

## Quick start

```bash
# 1. Install (editable)
pip install -e ".[dev]"

# 2. Configure (two vLLM instances)
cp .env.example .env
# edit .env to point at your lower & higher model endpoints

# 3. Launch vLLM instances (separate terminals):
#    Lower (small):
vllm serve Qwen/Qwen2.5-1.5B-Instruct --port 8001 \
  --enable-auto-tool-choice --tool-call-parser hermes
#    Higher (large):
vllm serve Qwen/Qwen2.5-32B-Instruct --port 8002 \
  --enable-auto-tool-choice --tool-call-parser hermes

# 4. Run the CLI
agentic-router route "debug this stack trace: ..."
agentic-router chat  "What is 6 * 7? Use the calculator tool."
agentic-router stream "Read README.md and list its top 3 topics, then summarize."
agentic-router serve  # starts FastAPI on :8000
```

## HTTP API

| Method | Path                  | Body / Params                         | Description |
|--------|-----------------------|---------------------------------------|-------------|
| GET    | `/health`             | –                                     | liveness |
| POST   | `/route`              | `{task}`                              | routing decision only |
| POST   | `/chat`               | `{task, session_id?}`                 | one-shot answer |
| GET    | `/agent/stream`       | `?task=...&session_id=...`            | SSE event stream |
| GET    | `/sessions`           | –                                     | list session ids |
| GET    | `/sessions/{id}`      | –                                     | session messages + plan |

### SSE event types
`plan`, `route`, `subtask_start`, `iteration`, `delta`, `tool_call`,
`tool_result`, `answer`, `final`, `error`.

```bash
curl -N "http://localhost:8000/agent/stream?task=$(python -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1]))' 'What can you do?')"
```

## Configuration

All settings are env-driven (see `.env.example`):

| Var | Default | Meaning |
|-----|---------|---------|
| `LOWER_VLLM_BASE_URL` / `LOWER_MODEL` | localhost:8001 / qwen2.5-1.5b | small/fast model |
| `HIGHER_VLLM_BASE_URL` / `HIGHER_MODEL` | localhost:8002 / qwen2.5-32b | large/capable model |
| `ROUTER_LOWER_THRESHOLD` | 0.35 | score ≤ this ⇒ lower (heuristic) |
| `ROUTER_HIGHER_THRESHOLD` | 0.65 | score ≥ this ⇒ higher (heuristic) |
| `ROUTER_CLASSIFIER_BAND` | 0.20 | ±band around midpoint ⇒ ask classifier |
| `AGENT_MAX_ITERATIONS` | 8 | ReAct tool rounds |
| `AGENT_MAX_PLAN_SUBTASKS` | 5 | plan size cap |
| `AGENT_ENABLE_PLANNING` / `AGENT_ENABLE_TOOLS` | true | toggle capabilities |

## Layout

```
agentic_router/
  config.py            # Settings (pydantic-settings)
  models.py            # RouteDecision, AgentEvent, SessionState, ...
  clients.py           # AsyncOpenAI wrappers for each vLLM instance
  router/
    heuristic.py       # deterministic complexity scorer
    classifier.py      # LLM-as-judge fallback (uses lower model)
    router.py          # hybrid Router
  tools/               # Tool interface + builtins (shell, file, web, calc)
  agent/
    loop.py            # Agent: plan → route → ReAct → stream
    planner.py         # higher-model planner
    memory.py          # in-memory SessionStore
  server/app.py        # FastAPI app
  cli.py               # typer CLI
  factory.py           # wiring
tests/                 # router + agent tests (no real vLLM)
```

## Tests

```bash
pytest
```

## Notes / safety

- `run_shell` and `write_file` are powerful — gate them behind auth in
  production and run the server sandboxed.
- The classifier deliberately falls back to the **higher** model on any parse
  error or timeout (correctness over latency).
- Routing thresholds are deliberately tunable; start with the defaults and
  adjust to your model pair's capability gap.

## License

MIT