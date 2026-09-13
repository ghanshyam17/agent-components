# Azure deployment (agent-components)

Zero-idle Azure footprint in the shared **`agent-lab`** Foundry project
(free S0 hub, eastus).

| Piece | Azure service | Idle cost | Purpose |
|---|---|---|---|
| `hosted-agent/` | Foundry hosted agent | $0 after idle timeout | Chat + delegate tasks to the real agentic-router loop |
| `functions/agent-api/` | Function app, Y1 Consumption (centralus) | $0 idle | `POST /api/agent {"task": ...}` -> route + answer |
| `terraform/` | project/model/Function stack | per-use only | Same resources as a2a-prototype (shared, imported) |

The Foundry project + model deployment are SHARED with the sibling repos
(one hub, one project, several agents); they are imported into Terraform
state, never recreated (see `terraform/README.md`).

## Deploy the hosted agent

```bash
python3 -m venv .venv-deploy && . .venv-deploy/bin/activate
pip install "azure-ai-projects>=2.3.0" azure-identity python-dotenv
export FOUNDRY_PROJECT_ENDPOINT=https://my-foundry-resource.services.ai.azure.com/api/projects/agent-lab
export FOUNDRY_MODEL_NAME=gpt-5-mini
python azure/deploy_hosted_agent.py --wait
```

## Deploy the Function app

```bash
func azure functionapp publish agcomps-agent-api --python
```

## Notes

- The agentic-router's model tiering needs OpenAI-compatible endpoints for
  LOWER/HIGHER tiers. In the sandbox both point at the project's gpt-5-mini
  deployment (zero extra spend); locally point them at your vLLM/Ollama as
  before.
- A `deploy_hosted_agent.py` identical to a2a's (agent name
  `agcomps-router`) is included below.
