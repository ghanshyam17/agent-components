resource "azurerm_service_plan" "func" {
  name                = "${var.func_prefix}-asp"
  resource_group_name = var.hub_resource_group
  location            = var.func_location
  os_type             = "Linux"
  sku_name            = "Y1"
  tags                = local.common_tags
}

resource "azurerm_storage_account" "func" {
  name                     = "${var.func_prefix}func${substr(md5(var.project_name), 0, 6)}"
  resource_group_name      = var.hub_resource_group
  location                 = var.func_location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = local.common_tags
}

resource "azurerm_linux_function_app" "agent_api" {
  name                = "${var.func_prefix}-agent-api"
  resource_group_name = var.hub_resource_group
  location            = var.func_location
  service_plan_id     = azurerm_service_plan.func.id
  storage_account_name       = azurerm_storage_account.func.name
  storage_account_access_key = azurerm_storage_account.func.primary_access_key
  tags                = local.common_tags

  identity {
    type = "SystemAssigned"
  }

  site_config {
    application_stack {
      python_version = "3.11"
    }
  }

  app_settings = {
    "FUNCTIONS_WORKER_RUNTIME" = "python"
  }
}
