# eval-harness

Dataset-driven evaluation for AI/agent systems: load examples, run an async
target function, score each prediction with metrics (including an async
**LLM-judge**), and produce an `EvaluationReport` with per-example scores and
aggregates.

Part of the [agent-components](../..) monorepo.

## What's inside

| Concept | Where | Notes |
|--------|------|-------|
| `Example` / `Dataset` | `eval_harness.dataset` | pydantic models; loaders for JSON / JSONL / CSV / YAML |
| Metrics | `eval_harness.metrics` | `exact_match`, `contains`, `regex_match`, `tool_call_accuracy`, `latency`, `llm_judge` |
| Runner | `eval_harness.runner.evaluate` | runs the target concurrently, scores, builds the report |
| Report | `eval_harness.report.EvaluationReport` | `to_json()`, `to_markdown()` |
| CLI | `eval-harness run` | offline `echo`/`stub` targets so you can try it without a model |

A metric is a callable `(example, prediction, context) -> float` in `[0, 1]`,
either sync or async. A `Metric` dataclass bundles the name, the function and an
`is_async` flag.

## Install

```bash
uv sync                                # in the monorepo
# or once published:
pip install eval-harness
pip install "eval-harness[yaml]"       # to load YAML datasets
```

## Usage

### Evaluate a custom async target

```python
import asyncio
from eval_harness import Dataset, Example, evaluate, exact_match, contains

dataset = Dataset(examples=[
    Example(id="1", input="What is 2+2?", expected="4"),
    Example(id="2", input="Capital of France?", expected="Paris"),
    Example(id="3", input="Say hi", expected="hi"),
])

async def target(example):
    # In real life this calls an LLM/agent. Here we echo a canned answer.
    answers = {"1": "4", "2": "Paris", "3": "hey"}
    return {"output": answers.get(example.id, ""), "latency": 0.05, "tool_calls": None}

async def main():
    report = await evaluate(target, dataset, [exact_match, contains])
    print(report.to_markdown())
    print(report.aggregate.overall)

asyncio.run(main())
```

### Metrics

```python
from eval_harness import exact_match, contains, regex_match, tool_call_accuracy, latency, llm_judge

metrics = [
    exact_match,                       # str equality after strip
    contains,                           # expected substring in output
    regex_match(r"\d+"),                 # regex matches output
    tool_call_accuracy,                  # expected tool names vs predicted
    latency(max_latency=2.0),            # 1.0 at 0s, 0.0 at 2s
    llm_judge(judge_client),              # async: PASS/FAIL from an LLM
]
```

`tool_call_accuracy` reads expected tool names from `example.expected` (a list
of names or dicts, or a dict with a `tool_calls` key) or from
`example.metadata["expected_tools"]`, and predicted tools from
`prediction["tool_calls"]` (a list of names or `{"name": ...}` dicts). The score
is `|expected ∩ predicted| / max(|expected|, |predicted|)` — order-insensitive,
penalizing both missing and extra tools.

`latency(max_latency)` normalizes the prediction latency against a threshold:
`1.0` at 0s, `0.0` at `max_latency`, clamped to `[0, 1]`.

### LLM-judge

`llm_judge` is async and accepts any client with `async chat(messages) -> (content, _)`
— the router's `ModelClient` and the gateway's `GatewayClient` both satisfy this,
so you can use any of them as the judge:

```python
from eval_harness import evaluate, llm_judge
from model_gateway import build_gateway

gw = build_gateway("gateway.yaml")
judge = gw.client("higher")             # or your router's ModelClient

report = await evaluate(
    target, dataset,
    [exact_match, llm_judge(judge, name="judge_higher")],
)
```

The default judge prompt asks for a single word `PASS`/`FAIL` with a short
rubric; pass your own `prompt_template` to customize it (it is `.format()`-ed
with `input`, `expected`, `prediction`). You can also pass the judge client to
`evaluate(..., judge_client=...)` instead of binding it to the metric.

### CLI

```bash
# needs a dataset on disk (json/jsonl/csv/yaml)
eval-harness run --dataset examples.json --target echo --metrics exact_match,contains
```

`echo` returns the input as the output, so `contains`/`exact_match` against an
`expected` that equals the input will pass — handy for a smoke test. `stub`
returns empty output (everything fails).

## Dataset formats

JSON (bare list or `{"examples": [...]}`):

```json
{"examples": [
  {"id": "1", "input": "hi", "expected": "hi", "metadata": {"topic": "greeting"}}
]}
```

JSONL (one object per line), CSV (`id,input,expected,metadata`; `expected` and
`metadata` may be JSON-encoded), and YAML (with the `yaml` extra):

```yaml
examples:
  - id: "1"
    input: hi
    expected: hi
```

## License

MIT