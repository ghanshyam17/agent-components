variable "location" {
  type    = string
  default = "eastus"
}

variable "func_location" {
  type        = string
  default     = "centralus"
  description = "Y1 (Consumption) quota is region-bound; centralus works, eastus is 0 on this subscription."
}

variable "hub_resource_group" {
  type    = string
  default = "my-foundry-rg"
}

variable "hub_account_name" {
  type    = string
  default = "my-foundry-resource"
}

variable "project_name" {
  type    = string
  default = "agent-lab"
}

variable "model_name" {
  type    = string
  default = "gpt-5-mini"
}

variable "model_version" {
  type    = string
  default = "2025-08-07"
}

variable "model_sku" {
  type    = string
  default = "GlobalStandard"
}

variable "model_capacity" {
  type    = number
  default = 1
}

variable "func_prefix" {
  type    = string
  default = "agcomps"
}

variable "tags" {
  type = map(string)
  default = {}
}

locals {
  common_tags = merge(
    {
      project      = "agent-components"
      managed-by   = "terraform"
      cost-posture = "zero-idle"
    },
    var.tags,
  )
}
