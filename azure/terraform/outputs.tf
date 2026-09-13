output "project_endpoint" {
  value = "https://${var.hub_account_name}.services.ai.azure.com/api/projects/${var.project_name}"
}

output "model_deployment_name" {
  value = azurerm_cognitive_deployment.chat_model.name
}

output "function_app_name" {
  value = azurerm_linux_function_app.agent_api.name
}
