# Functionality -> Foundry platform mapping (agent-components)

| Component (repo) | Foundry platform capability | Notes |
|---|---|---|
| `agentic-router` (planner, ReAct loop, sessions) | **Hosted agent** (`agcomps-router`, Responses protocol) | own code on platform-managed compute |
| `model-gateway` (LB, fallback, retries, rate limits) | **Model catalog deployments** per tier; platform rate limits per deployment | gateway stays in-code for local; in Foundry the deployment IS the endpoint group |
| `memory-store` (session + vector) | **Durable state store** (session KV) + **AI Search connection** (vector) via **Toolbox** | Redis backend optional for self-hosted |
| `guardrails` | **Content filtering (RAI policies)** on deployments + in-code filters as defense-in-depth | `rai_policy_name` in terraform |
| `prompt-registry` | **Agent versions** (immutable snapshots of config/env) | platform-side versioning |
| `retriever` (chunk/rank) | **AI Search via Toolbox connection** or in-sandbox execution | both supported |
| `toolkit` (sandboxed tools) | **Code Interpreter** / Toolbox tools | platform-sandboxed execution |
| `tracing` + `cost` | **OpenTelemetry -> App Insights** (auto-injected) + usage/billing APIs | no exporter needed in Foundry |
| `eval-harness` | **Foundry evaluations / agent evaluators** | run evals against the hosted agent endpoint |
| HTTP server (`server/app.py`) | **Azure Functions** ($0 idle) + **Toolbox OpenAPI tool** exposure | `azure/functions/` |

Shared platform resources: one hub (`my-foundry-resource`, S0), one project
(`agent-lab`), one model deployment (`gpt-5-mini`) - imported into Terraform,
referenced across all sibling repos.
