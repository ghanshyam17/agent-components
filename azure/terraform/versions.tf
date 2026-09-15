terraform {
  required_version = ">= 1.6"

  backend "azurerm" {
    resource_group_name  = "rg-terraform-state"
    storage_account_name = "tfstateghanshyam"
    container_name       = "tfstate"
    key                  = "agent-components.tfstate"
  }

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.0"
    }
  }
}
