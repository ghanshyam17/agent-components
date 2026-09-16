# distillation

**Component #11** of the agent-components monorepo. Generates synthetic training
data from a frontier **teacher** model and distils it into a small, cheap
**student**.

A chassis *child extension*: it composes the foundational components and
reimplements none of them.

```
                      ┌──────────────── model-gateway ────────────────┐
                      │  route to the teacher (local vLLM | cloud)    │
                      └───────────────────────┬───────────────────────┘
                                              ▼
  prompt-registry ──────────────▶ ① SYNTHESISE  SyntheticContentGenerator
  (style/format blueprints)        CoT reasoning traces + completions
                                              │
                                              ▼
  guardrails ────────────────────▶ ② CURATE     ContentCurator
  (PII, injection, length)          ├── score groundedness / coherence / fluency
  eval-harness                         ├── dedup via memory-store cosine
  (groundedness metrics)               └── split train/val; mine DPO pairs
  memory-store                                        │
  (cosine dedup)                                      ▼
                                              ┌───────────────────────┐
                                              ③ TRAIN  Foundry job
                                              │  Azure AI Foundry      │
                                              │  Distillation Service  │
                                              │  — or —                │
                                              │  local mock runner     │
                                              └───────────┬───────────┘
                                                          ▼
                                              ④ EVALUATE  ParityEvaluator
                                              quality · style · TTFT · $/1M
                                                          │
                                                          ▼
                                              parity matrix → go / no-go
```

## Install

```bash
uv sync --all-packages            # offline: mock runner only
pip install "distillation[azure]" # + Azure AI Foundry fine-tuning
```

## Quick start — offline (no Azure, no GPU)

```python
import asyncio
from distillation import DistillationConfig, DistillationPipeline

cfg = DistillationConfig(
    teacher_model="higher",          # model-gateway alias, or a model name
    student_base_model="microsoft/Phi-4-mini-instruct",
    method="lora",
    num_samples=40,
)
result = asyncio.run(DistillationPipeline(cfg).run())
print(result.summary())
```

## Quick start — with a real teacher

```python
from model_gateway import build_gateway
from distillation import DistillationConfig, DistillationPipeline, ParityEvaluator

gw = build_gateway("gateway.yaml")
cfg = DistillationConfig(teacher_model="higher", num_samples=200, method="sft")

pipe = DistillationPipeline(
    cfg,
    teacher_client=gw.client("higher").chat,
    student_client=gw.client("lower").chat,      # parity baseline
)
result = await pipe.run()
```

## The four stages

| Stage | Engine | What it does |
|---|---|---|
| ① Synthesise | `SyntheticContentGenerator` | Prompts the teacher with style blueprints and captures the chain-of-thought alongside the answer. `generate_preference_pairs()` samples multiple completions per prompt to mine DPO pairs. |
| ② Curate | `ContentCurator` | Runs `guardrails` over prompt **and** completion, scores groundedness/coherence/fluency, drops near-duplicates via `memory-store` cosine, and emits Alpaca / ChatML / DPO records. Quarantined samples are returned with reasons — never silently dropped. |
| ③ Train | `FoundryDistillationClient` | Submits an Azure AI Foundry fine-tuning job and polls it. Without the SDKs or an endpoint it runs a **local mock** that walks the same state machine and writes the same artifacts. |
| ④ Evaluate | `ParityEvaluator` | Teacher-vs-student parity: quality retention, stylistic similarity, Flesch-Kincaid readability, TTFT speedup and cost savings %. |

## Why reasoning traces

A student trained only on `(prompt, answer)` learns surface form. Trained on the
teacher's *reasoning*, it learns the procedure — which is what generalises to
unseen prompts. The trace is captured separately from the completion so the
curator can score and filter the two independently: reasoning is where teacher
hallucinations hide.

## The parity matrix

```python
result.parity.to_dict()
# {
#   "quality_retention": 0.94,        student kept 94% of the teacher's quality
#   "stylistic_similarity": 0.71,    how teacher-like the answers are
#   "latency": {"ttft_speedup": 4.2},
#   "cost":    {"savings_pct": 96.0},
#   "meets_bar": true
# }
```

`meets_bar()` requires **both** retention and stylistic agreement. A student that
scores well but does not answer like the teacher has not been distilled — it has
been replaced with a different model.

## Honest measurement

Two kinds of number appear in a parity report, and the report labels which is
which rather than presenting them alike:

- **Measured** — latency, token counts, cost. Facts about the run, from real
  calls. Prices are supplied by the caller (`ParityEvaluator(prices=...)`), not
  hardcoded, because a stale price silently corrupts every savings figure.
- **Scored** — quality, stylistic similarity, readability. Derived. With no
  judge bound, quality falls back to token-overlap and the report says so in
  `notes`.

`TTFT` is approximated by end-to-end latency on a single non-streaming call —
stated in the report's notes rather than passed off as a first-token
measurement. Bind a streaming client for a true TTFT.

## Dataset formats

| `DatasetFormat` | Record shape |
|---|---|
| `alpaca` | `{"instruction", "input", "output"}` |
| `chatml` | `{"messages": [{"role", "content"}, ...]}` |
| `dpo` | `{"prompt", "chosen", "rejected"}` |
| `jsonl` | raw `ContentSample` |

`method="dpo"` requires `format="dpo"` — preference optimisation trains on
`(chosen, rejected)` pairs, which the Alpaca/ChatML schemas cannot express. The
config validator rejects the mismatch rather than failing later in training.

**DPO samples several completions per prompt.** The normal path emits exactly
one sample per `(topic, blueprint)` pair, so grouping the result by prompt
yields only singletons and no pair can be mined. In DPO mode the pipeline calls
`generate_preference_pairs()` instead (`dpo_candidates=3` by default) so each
prompt contributes a real candidate set.

DPO pairs whose margin falls below `min_margin` are dropped: near-ties encode
mostly noise, and DPO trains on exactly the difference it is shown.

If no pairs can be mined, the training stage **refuses to run** rather than
writing `ContentSample` records under a DPO method — a wrong-schema file is
worse than a clear failure.

## Artifacts

```
artifacts/<job_id>/
├── train.jsonl     curated dataset in the configured format
├── dpo.jsonl       preference pairs (only when method="dpo")
└── job.json        final DistillationJobStatus
```

## Foundry vs mock

`FoundryDistillationClient.describe()` reports which backend will run and why:

```python
{"backend": "mock", "azure_sdk_available": False,
 "note": "Local mock runner: no Azure calls, no GPU. Writes the same artifact
          layout and walks the same state machine."}
```

Degradation is logged once, not silent. A pipeline that believes it trained a
model when it did not is worse than one that reports it did not.

## Tests

```bash
uv run pytest components/distillation/tests/test_distillation.py -v
```

Part of the [agent-components](../../) monorepo.

## License

MIT
