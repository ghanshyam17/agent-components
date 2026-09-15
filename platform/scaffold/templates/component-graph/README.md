# {{ project_name }} (Component Graph Platform Architecture)

An end-to-end platform-engineered project plugging together:
- **Data Engineering Services**: Azure Data Factory (ADF) batch & medallion pipelines, Vector RAG indexing in Azure AI Search, and data quality gates.
- **Machine Learning Services**: Azure Machine Learning (AML) training jobs, registered models, and online scoring endpoints.
- **Tools Plane**: Automatic tool bridges connecting lakehouse tables, vector search, ADF triggers, and AML inference directly to agents.
- **Agent Component Graph**: Multi-agent collaboration with **AutoGen** (Data Analyst + ML Advisor team) and **LangGraph** (cyclical state machine workflow) under a first-route classifier.
- **Azure Chassis**: Azure AI Foundry SDK hosted agent runtime with zero idle cost.

## Quick Start

```bash
# Validate component graph specification
agcomps validate --path component_graph.yaml

# Run locally with component graph orchestrator
agcomps dev --agent component_graph.yaml

# Deploy to Azure AI Foundry Agent Service
agcomps deploy --project component_graph.yaml --env dev
```
