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

## Platform Engineering Layer (NEW)

| Platform Module | Azure Service / Capability | Notes |
|---|---|---|
| `platform/schema/` (YAML specs) | **`azure.yaml`** (azd) pattern | Declarative agent + infra definitions |
| `platform/patterns/react` | **Foundry hosted agent** (ReAct loop) | Wraps existing agentic-router |
| `platform/patterns/autogen` | **Microsoft Agent Framework** / AutoGen | Conversational multi-agent group chat & debate |
| `platform/patterns/langgraph` | **LangChain Azure AI** / Foundry Hosted Agent | Stateful cyclical graphs & conditional branching |
| `platform/patterns/framework_router` | **Foundry Intelligent Router / First Routes** | Routes tasks to AutoGen, LangGraph, or ReAct first |
| `platform/patterns/supervisor` | **Multi-agent orchestration** (Agent Framework) | Supervisor → worker delegation |
| `platform/patterns/network` | **A2A protocol** / **Magentic One** | Peer-to-peer agent messaging |
| `platform/patterns/sequential` | **Prompt Flow** / chain orchestration | Pipeline-style agent chain |
| `platform/patterns/map_reduce` | **Batch processing** + parallel agent execution | Fan-out/fan-in pattern |

| `platform/sandbox/dynamic_sessions` | **Azure Dynamic Sessions** (Code Interpreter) | Managed sandboxed Python |
| `platform/sandbox/container_apps` | **Azure Container Apps Jobs** | Custom Docker, GPU workloads |
| `platform/sandbox/local_docker` | Local Docker | Development sandbox |
| `platform/deployer/foundry` | **Foundry Agent Service** (create_version_from_code) | Supersedes `deploy_hosted_agent.py` |
| `platform/deployer/app_service` | **Azure App Service** | Web app deployments |
| `platform/deployer/container_apps` | **Azure Container Apps** | Containerized agents |
| `platform/deployer/functions` | **Azure Functions** (Consumption) | Serverless agent endpoints |
| `platform/deployer/bicep_generator` | **Azure Bicep** templates | IaC generation from YAML |
| `platform/plugins/` | Plugin extensibility | YAML/JSON/text plugin loading |
| `platform/scaffold/` | **`azd ai agent init`** pattern | CLI project scaffolding |
| `platform/engineering/loop` | Loop controls + telemetry | Circuit breakers, guardrails |
| `platform/engineering/harness` | **Foundry evaluations** (YAML-driven) | Wraps eval-harness component |
| `platform/engineering/data/adf` | **Azure Data Factory** | Workflows, Copy activities, Data Flows, Triggers, Linked Services |
| `platform/engineering/data/pipelines/batch` | **Medallion Lakehouse** | Bronze/Silver/Gold Delta Lake on ADLS Gen2 |
| `platform/engineering/data/pipelines/vector_rag` | **Azure AI Search Vector RAG** | Document chunking, text-embedding-3, hybrid indexing |
| `platform/engineering/data/pipelines/streaming` | **Azure Event Hubs / Stream Analytics** | Real-time event-driven streaming ingestion |
| `platform/engineering/data/pipelines/quality` | **Azure Purview / Data Quality Gates** | Schema enforcement, assertions, quarantine dead-letter |
| `platform/engineering/data/tools` | **Agent Data & ML Tool Bridges** | Lakehouse query, ADF triggers, AML inference, RAG search |
| `platform/engineering/aml/` | **Azure Machine Learning v2** | Command jobs, Pipeline DAGs, Sweep hyperparameter tuning |
| `platform/engineering/aml/model_registry` | **Azure ML Model Registry & Endpoints** | MLflow tracking, online managed endpoints, traffic split |
| `platform/schema/component_graph` | **Component Graph Declarative Spec** | End-to-end multi-level pipeline-agent-infra definition |
| `platform/engineering/graph_orchestrator` | **Foundry SDK Hosted Component Graph** | Responses protocol execution of Data + ML + Agent graph |

Shared platform resources: one hub (`my-foundry-resource`, S0), one project
(`agent-lab`), one model deployment (`gpt-5-mini`) - imported into Terraform,
referenced across all sibling repos.

