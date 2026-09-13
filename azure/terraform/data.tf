data "azurerm_cognitive_account" "hub" {
  name                = var.hub_account_name
  resource_group_name = var.hub_resource_group
}
