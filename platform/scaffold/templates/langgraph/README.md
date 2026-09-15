# {{ project_name }} (LangGraph Pattern)

This project uses the **LangGraph Pattern** to orchestrate cyclical stateful execution (draft -> verify -> refine) with conditional edge branching.

## Quick Start

```bash
# Validate YAML configuration
agcomps validate --path .

# Run locally
agcomps dev --agent agent.yaml

# Deploy to Azure AI Foundry Agent Service
agcomps deploy --project project.yaml --env dev
```
