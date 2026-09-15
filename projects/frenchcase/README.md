# FrenchCase on agent-components chassis

> **FrenchCase is now a platform-engineered project.** Its 6 agents, 23 Azure
> resources, 5 tools, and content ingestion pipeline are declared in YAML and
> deployable via the agent-components `platform/deployer/` engine.

## What this is

This directory is the **declarative projection** of FrenchCase onto the
agent-components chassis. Every YAML file maps to a real engine in FrenchCase's
`app/` directory — nothing here is hypothetical.

```
projects/frenchcase/
├── project.yaml                     # Project root — 6 agents, 2 infra, 2 plugins, 3 envs
├── agents/
│   ├── tutor.yaml                   # RAG tutor (react)         → app/core/llm.py
│   ├── grader.yaml                  # Writing grader (react)    → app/core/llm.py::grade_text
│   ├── learncase-team.yaml          # 6-agent team (supervisor) → app/core/multi_agent_learncase.py
│   ├── exam-engine.yaml             # Mock exam (sequential)    → app/core/exam_engine.py
│   ├── recommender.yaml             # Adaptive rec (sequential) → app/core/case_recommender.py
│   └── conversation.yaml            # Role-play partner (react) → app/core/conversation.py
├── workers/                         # LearnCase team sub-agents
│   ├── grammar-agent.yaml
│   ├── podcast-agent.yaml
│   ├── content-agent.yaml
│   ├── exercise-agent.yaml
│   ├── quality-supervisor.yaml
│   └── json-mapper.yaml
├── infra/
│   ├── data-plane.yaml              # AI Search, SQL, Blob, Redis
│   └── compute.yaml                 # Functions, Container Apps, App Service
├── plugins/
│   ├── frenchcase-tools.yaml        # 5 tools (repo_search, conjugation, grade, tts, load_lesson)
│   └── content-ingestion.yaml       # Podcast/blog/news parsers + 4-axis quality gate
├── tools/
│   └── learncase_tools.py           # Adapter: FrenchCase tools → chassis Registry
└── README.md                        # This file
```

## Engine mapping (what's real vs what's declared)

| FrenchCase engine | Chassis component | Pattern | YAML spec |
|---|---|---|---|
| `app/core/llm.py` chat_stream() | agentic-router + model-gateway | react | `agents/tutor.yaml` |
| `app/core/llm.py` grade_text() | agentic-router (structured output) | react | `agents/grader.yaml` |
| `app/core/multi_agent_learncase.py` | platform/patterns/supervisor | supervisor | `agents/learncase-team.yaml` |
| `app/core/autogen_orchestrator.py` | platform/patterns/autogen | autogen | (future: AutoGen pattern spec) |
| `app/core/exam_engine.py` | platform/patterns/sequential | sequential | `agents/exam-engine.yaml` |
| `app/core/case_recommender.py` | platform/patterns/sequential | sequential | `agents/recommender.yaml` |
| `app/core/conversation.py` | agentic-router + model-gateway | react | `agents/conversation.yaml` |
| `app/core/llm_router.py` | model-gateway | — | (model.tier_strategy in each agent) |
| `app/core/vector.py` | retriever | — | (memory.vector in each agent) |
| `app/core/session_store.py` | memory-store (SessionStore) | — | (memory.session in each agent) |
| `app/agents/tools.py` | toolkit | — | `plugins/frenchcase-tools.yaml` |
| `app/core/content_syncer.py` | guardrails (output quality) | — | `plugins/content-ingestion.yaml` |
| `app/core/graph_engine.py` | (no chassis equivalent — FrenchCase-specific) | — | (data plane, not an agent) |
| `app/core/adaptive_engine.py` | (no chassis equivalent — FrenchCase-specific) | — | (data plane, not an agent) |

**3 engines have no chassis equivalent** (graph_engine, adaptive_engine,
gamification) — these are FrenchCase-specific learner-state logic that live in
the data plane, not in an agent. They're declared as SQL + Redis resources in
`infra/data-plane.yaml`.

## How to deploy

### Prerequisites
```bash
# In agent-components repo
cd /Users/ghanshyam/Projects/agent-components
uv sync

# Set Azure credentials
export AZURE_SUBSCRIPTION_ID=...
az login
```

### Dry run (validate specs, no deployment)
```bash
uv run python -c "
from platform.deployer.engine import DeployEngine
import asyncio
engine = DeployEngine()
result = asyncio.run(engine.deploy(
    'projects/frenchcase/project.yaml',
    env='dev',
    dry_run=True
))
print(f'Validated: {result.success}, steps: {len(result.plan.steps)}')
"
```

### Deploy to dev (Functions, $0 idle)
```bash
uv run python -c "
from platform.deployer.engine import DeployEngine
import asyncio
engine = DeployEngine()
result = asyncio.run(engine.deploy(
    'projects/frenchcase/project.yaml',
    env='dev'
))
print(f'Deployed: {result.success}')
print(f'Endpoints: {result.endpoints}')
"
```

### Local development (run agent without Azure)
```bash
# Mount FrenchCase's app/ into the platform's sys.path
export PYTHONPATH="/Users/ghanshyam/Documents/french-learning:$PYTHONPATH"
uv run python projects/frenchcase/tools/learncase_tools.py
# → FrenchCase tools: 5 tools, available=True
```

## Model routing strategy

FrenchCase's `llm_router.py` (5 backends, 4 strategies) maps to the chassis
`model-gateway` component. The tier_strategy in each agent YAML declares which
model handles which tier:

| Tier | Dev (speed) | Staging (balanced) | Prod (quality) |
|------|-------------|-------------------|----------------|
| lower | qwen2.5:3b (Ollama local) | gpt-4o-mini (Azure) | gpt-4o-mini (Azure) |
| higher | gemma4:cloud (Ollama cloud) | gpt-4o (Azure) | gpt-4o (Azure) |
| embedding | bge-m3 (Ollama local) | bge-m3 (Ollama) | text-embedding-3-large (Azure) |
| tts | edge-tts (free) | edge-tts (free) | edge-tts (free) |

## What the chassis gives FrenchCase

1. **Declarative config** — kills the `.azure.env` vs `config.py` drift we kept
   hitting. One YAML, validated by Pydantic, `${VAR}` resolved at load time.
2. **Pattern formalization** — the 6-agent LearnCase team is now a `supervisor`
   pattern, not 600 lines of bespoke sequential code.
3. **Single deploy command** — 3 hand-written shell scripts
   (`deploy-azure.sh`, `deploy-functions.sh`, `deploy-aca.sh`) collapse into
   `engine.deploy(project.yaml, env=...)`.
4. **Tool registry** — FrenchCase's 5 tools are now platform tools, callable by
   any agent in any project on the chassis.
5. **Plugin system** — content ingestion parsers become plugins, schedulable
   and composable.

## What FrenchCase keeps (chassis doesn't cover)

- **Content corpus** — 2,131 catalog entries, 1,816 LearnCases, 278 news
  articles, 323 podcast episodes. This is data-plane content, not agent logic.
- **Learner state** — FSRS-2 spaced repetition, case tracker, click analytics,
  gamification. FrenchCase-specific algorithms.
- **NetworkX graph** — 5,859 nodes, 18,622 edges. Prerequisite graph for
  recommendation. Not an agent pattern.
- **Frontend** — Next.js 14 app with 15 pages. The chassis has no UI component
  (agent-ui is planned but not built).

## Integration status

| Layer | Status |
|-------|--------|
| Project + agent YAML specs | ✅ Written and validated against schema |
| Worker agent specs (6) | ✅ Written |
| Infrastructure specs (2) | ✅ Written |
| Plugin specs (2) | ✅ Written |
| Tool adapter (Python) | ✅ Written (stub mode when FrenchCase not mounted) |
| DeployEngine.deploy() | 🟡 Stub — needs real Azure SDK calls (foundry.py has placeholder) |
| PatternFactory wiring | 🟡 Patterns registered but not yet calling FrenchCase engines |
| Model gateway integration | 🔴 Not yet — llm_router.py not wired to model-gateway |
| Retriever integration | 🔴 Not yet — vector.py not wired to retriever component |
| Memory-store integration | 🔴 Not yet — session_store.py not wired to memory-store |

## Next steps (to make the integration live)

1. **Wire DeployEngine to real Azure SDK** — `platform/deployer/foundry.py`
   has a stub `deploy()` that needs the actual `AIProjectClient` calls.
2. **Wire llm_router.py → model-gateway** — expose FrenchCase's 5-backend router
   as a model-gateway config, so the chassis routes through it.
3. **Wire vector.py → retriever** — map FrenchCase's ChromaDB retrieval to the
   retriever component's pipeline interface.
4. **Wire session_store.py → memory-store** — swap the in-memory SessionStore
   for the chassis's Redis-backed one in prod.
5. **Add `agcomps deploy frenchcase` CLI command** — one-command deploy via
   `platform/scaffold/cli.py`.