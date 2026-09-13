# Terraform (agent-components)

Same shape as a2a-prototype: existing classic S0 hub account referenced via
data source; shared `agent-lab` project + `gpt-5-mini` deployment imported
(they already exist - do NOT let Terraform recreate them); a `agcomps-agent-api`
Function app on a Y1 Consumption plan in centralus (eastus Y1 quota is 0 on
this subscription).

```bash
SUB=3ef6aa8d-0594-4998-883f-9d5cc4953ee2
cd azure/terraform && terraform init
terraform import azurerm_cognitive_account_project.foundry_project \
  "/subscriptions/$SUB/resourceGroups/my-foundry-rg/providers/Microsoft.CognitiveServices/accounts/my-foundry-resource/projects/agent-lab"
terraform import azurerm_cognitive_deployment.chat_model \
  "/subscriptions/$SUB/resourceGroups/my-foundry-rg/providers/Microsoft.CognitiveServices/accounts/my-foundry-resource/deployments/gpt-5-mini"
terraform apply
```
