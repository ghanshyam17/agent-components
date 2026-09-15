# {{ project_name }} (AutoGen Pattern)

This project uses the **AutoGen Multi-Agent Pattern** to coordinate multiple specialized agents (Coder, Critic, UserProxy) in a conversational group chat.

## Quick Start

```bash
# Validate YAML configuration
agcomps validate --path .

# Run locally
agcomps dev --agent agent.yaml

# Deploy to Azure AI Foundry Agent Service
agcomps deploy --project project.yaml --env dev
```
