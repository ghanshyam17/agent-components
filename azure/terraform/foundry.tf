# Same shared Foundry project (agent-lab) as the sibling repos - one project,
# multiple hosted agents. Import command in README (already provisioned).
resource "azurerm_cognitive_account_project" "foundry_project" {
  name                 = var.project_name
  location             = var.location
  cognitive_account_id = data.azurerm_cognitive_account.hub.id
  tags                 = local.shared_tags

  identity {
    type = "SystemAssigned"
  }
}

resource "azurerm_cognitive_deployment" "chat_model" {
  name                 = var.model_name
  cognitive_account_id = data.azurerm_cognitive_account.hub.id
  rai_policy_name      = "Microsoft.DefaultV2"

  model {
    format  = "OpenAI"
    name    = var.model_name
    version = var.model_version
  }

  sku {
    name     = var.model_sku
    capacity = var.model_capacity
  }
}
