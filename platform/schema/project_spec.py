from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

class ProjectMetadata(BaseModel):
    """Metadata for a Project specification."""
    name: str = Field(..., description="Name of the project")
    subscription: Optional[str] = Field(None, description="Azure subscription ID")
    resource_group: Optional[str] = Field(None, description="Azure resource group")
    location: Optional[str] = Field(None, description="Azure location")

class FoundryConfig(BaseModel):
    """Foundry hub and project configuration."""
    hub_name: str = Field(..., description="Azure AI Foundry hub name")
    project_name: str = Field(..., description="Azure AI Foundry project name")

class ProjectSpecModel(BaseModel):
    """Main specification model for the Project."""
    foundry: FoundryConfig = Field(..., description="Foundry configuration")
    agents: List[str] = Field(default_factory=list, description="List of agent YAML file paths")
    infrastructure: List[str] = Field(default_factory=list, description="List of infra YAML file paths")
    plugins: List[str] = Field(default_factory=list, description="List of plugin file paths")
    environments: Dict[str, Dict[str, Any]] = Field(default_factory=dict, description="Environment configs (e.g. dev, staging, prod) with overrides")

class ProjectSpec(BaseModel):
    """Top-level Project specification."""
    apiVersion: Literal["agentcomponents/v1"] = Field("agentcomponents/v1", description="API Version")
    kind: Literal["Project"] = Field("Project", description="Resource kind")
    metadata: ProjectMetadata = Field(..., description="Project metadata")
    spec: ProjectSpecModel = Field(..., description="Project specification details")
