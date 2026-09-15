# agent-platform

The platform engineering layer for [agent-components](../README.md): a
declarative, YAML-driven system for defining, wiring, and deploying AI agents
on Azure.

Instead of hand-coding agent composition and Azure provisioning in Python, you
declare a **Project** (and optionally a **ComponentGraph**) in YAML and let the
platform resolve, validate, and deploy it.

## Why it exists

The chassis in `components/` gives you composable pieces (router, gateway,
memory, retriever, tools, guardrails, tracing, eval, prompts). The platform
gives you the layer above: schemas that describe an application, a pattern
library that implements agent topologies, and deployers that provision Azure
without bespoke scripts per project.

## Layout

```
platform/
├── schema/          Declarative specs (Pydantic-validated)
│   ├── agent_spec.py           Agent: pattern, model, tools, memory, guardrails
│   ├── project_spec.py         Project: agents + infra + plugins + environments
│   ├── infrastructure_spec.py  Azure resources (AI Search, SQL, ADF, AML, ...)
│   ├── pattern_spec.py         Per-pattern config (supervisor, network, autogen,
│   │                           langgraph, framework_router, ...)
│   ├── component_graph.py      End-to-end graph: data + ML + tools + agent + runtime
│   └── loader.py               YAML/JSON loader with ${VAR} resolution + validation
├── patterns/        Agentic design patterns
│   ├── base.py                 AgentPattern ABC + PatternFactory registry
│   ├── react.py  supervisor.py  network.py  sequential.py  map_reduce.py
│   ├── autogen.py  langgraph.py  framework_router.py
├── sandbox/         Code execution backends
│   ├── manager.py              Routes by task type
│   ├── dynamic_sessions.py     Azure Dynamic Sessions
│   ├── container_apps.py       Azure Container Apps Jobs
│   └── local_docker.py         Local Docker
├── deployer/        Azure deployment engine
│   ├── engine.py               Reads a project spec, executes the plan
│   ├── foundry.py              Azure AI Foundry Agent Service
│   ├── app_service.py  container_apps.py  functions.py
│   └── bicep_generator.py      YAML → Bicep
├── engineering/     Engineering infrastructure
│   ├── loop.py                 Loop control: iteration caps, backoff, circuit breakers
│   ├── harness.py              YAML-driven eval (metrics, LLM-judge, thresholds)
│   ├── data_infra.py  ai_infra.py
│   ├── graph_orchestrator.py   Runs a ComponentGraph end to end
│   ├── data/                   Data engineering
│   │   ├── models.py           Batch / Medallion / Streaming / VectorRAG / DataQuality
│   │   ├── manager.py          DataInfraManager
│   │   ├── tools.py            Data outputs exposed as agent tools
│   │   ├── adf/                Azure Data Factory (client, activities, triggers, templates)
│   │   └── pipelines/          Pipeline builders (batch, streaming, quality, vector_rag)
│   └── aml/                    Azure Machine Learning
│       ├── models.py           Compute / environment / data / jobs / models / endpoints
│       ├── client.py           azure-ai-ml SDK wrapper
│       ├── jobs.py             Command, Sweep, Pipeline jobs
│       ├── model_registry.py   Model registration + endpoint deploy
│       ├── pipeline_builder.py
│       └── manager.py          AMLInfraManager
├── plugins/         Plugin system (loader + registry)
└── scaffold/        Project scaffolding + CLI
```

## Quick start

Validate a project spec:

```python
from platform.schema.loader import load_spec

spec = load_spec("projects/frenchcase/project.yaml")
print(spec.kind, spec.metadata.name, len(spec.spec.agents), "agents")
```

Use a pattern:

```python
from platform.patterns.base import PatternFactory
import platform.patterns  # registers built-ins

pattern = PatternFactory.create("supervisor", strategy="plan_and_delegate")
print(PatternFactory.available())
```

Deploy:

```bash
uv run python -c "
import asyncio
from platform.deployer.engine import DeployEngine
engine = DeployEngine()
asyncio.run(engine.deploy('projects/frenchcase/project.yaml', env='dev', dry_run=True))
"
```

## Data & ML planes

`ComponentGraphSpec` unifies four planes plus the runtime:

| Plane | Purpose | Declared as |
|-------|---------|-------------|
| `data_plane` | ADF pipelines, Medallion lakehouse, streaming, vector RAG indexing, quality gates | `BatchPipelineSpec`, `MedallionPipelineSpec`, `StreamingPipelineSpec`, `VectorRAGPipelineSpec`, `DataQualitySpec` |
| `ml_plane` | AML compute, jobs, registered models, scoring endpoints | `AMLComputeSpec`, `AMLCommandJobSpec`, `AMLSweepJobSpec`, `AMLModelSpec`, `AMLEndpointSpec` |
| `tools_plane` | Which data/ML outputs become callable agent tools | `ToolsPlaneConfig` |
| `agent_plane` | The agent itself (pattern, model, tools, memory) | `AgentSpecModel` |
| `runtime_chassis` | Where it runs and over which protocol | `RuntimeChassisConfig` |

With `auto_generate_agent_tools: true` on either plane, data and ML capabilities
are surfaced to the agent automatically — an ADF pipeline trigger, a lakehouse
query, a vector search, or an AML model endpoint becomes a tool the agent can
call.

## Azure dependencies

`data/` and `aml/` import the Azure SDKs lazily and degrade to logged no-ops
when they are absent, so the whole package imports without credentials:

- `azure-mgmt-datafactory` — ADF client
- `azure-ai-ml` — AML client
- `azure-identity` — credentials

## Status

The schema, pattern library, data plane, and ML plane are implemented and
schema-validated. The deploy engine's Foundry path is the thinnest part — it
validates specs and emits a plan, but does not yet provision resources.
