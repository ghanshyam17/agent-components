from __future__ import annotations
from typing import List, Literal, Optional, Dict, Any

from pydantic import BaseModel, Field

from platform.schema.pattern_spec import (
    SupervisorConfig,
    NetworkConfig,
    SequentialConfig,
    MapReduceConfig,
    AutoGenConfig,
    LangGraphConfig,
    FrameworkRouterConfig,
)

class Metadata(BaseModel):
    """Metadata for an Agent specification."""
    name: str = Field(..., description="Name of the agent")
    project: Optional[str] = Field(None, description="Project name")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags for the agent")
    description: Optional[str] = Field(None, description="Description of the agent")

class ModelConfig(BaseModel):
    """Model provider configuration."""
    provider: Literal["azure-foundry", "openai", "vllm", "ollama"] = Field(..., description="Model provider")
    deployment_name: str = Field(..., description="Deployment name or model name")
    tier_strategy: Optional[Dict[str, Any]] = Field(None, description="Lower/higher config tier strategy")

class CodeInterpreterConfig(BaseModel):
    """Code interpreter tool configuration."""
    enabled: bool = Field(False, description="Whether code interpreter is enabled")
    sandbox_type: Literal["dynamic-sessions", "container-apps", "local-docker"] = Field("local-docker", description="Sandbox type")
    packages: List[str] = Field(default_factory=list, description="List of packages to install")

class CustomTool(BaseModel):
    """Custom tool definition."""
    name: str = Field(..., description="Tool name")
    source_file: str = Field(..., description="Path to source file")
    function_name: str = Field(..., description="Function name in source file")

class ToolsConfig(BaseModel):
    """Agent tools configuration."""
    builtin: List[str] = Field(default_factory=list, description="List of builtin tools")
    code_interpreter: Optional[CodeInterpreterConfig] = Field(None, description="Code interpreter configuration")
    custom: List[CustomTool] = Field(default_factory=list, description="List of custom tools")

class SessionMemoryConfig(BaseModel):
    """Session memory configuration."""
    backend: Literal["redis", "in-memory"] = Field("in-memory", description="Session memory backend")
    ttl: int = Field(3600, description="Time to live in seconds")

class VectorMemoryConfig(BaseModel):
    """Vector memory configuration."""
    backend: Literal["ai-search", "in-memory"] = Field("in-memory", description="Vector memory backend")
    index_name: str = Field(..., description="Vector index name")

class MemoryConfig(BaseModel):
    """Agent memory configuration."""
    session: Optional[SessionMemoryConfig] = Field(None, description="Session memory config")
    vector: Optional[VectorMemoryConfig] = Field(None, description="Vector memory config")

class GuardrailsConfig(BaseModel):
    """Guardrails configuration."""
    input_filters: List[str] = Field(default_factory=list, description="List of input filters")
    output_filters: List[str] = Field(default_factory=list, description="List of output filters")

class InfrastructureConfig(BaseModel):
    """Agent infrastructure configuration."""
    compute_target: Literal["app-service", "container-apps", "functions", "foundry-hosted"] = Field("foundry-hosted", description="Compute target")
    sku: Optional[str] = Field(None, description="Compute SKU")
    scaling: Optional[Dict[str, int]] = Field(None, description="Scaling config with min, max keys")

class AgentSpecModel(BaseModel):
    """Main specification model for the Agent."""
    pattern: Literal["react", "supervisor", "network", "sequential", "map_reduce", "autogen", "langgraph", "framework_router"] = Field(..., description="Agent pattern")
    model: ModelConfig = Field(..., description="Model configuration")
    tools: Optional[ToolsConfig] = Field(None, description="Tools configuration")
    memory: Optional[MemoryConfig] = Field(None, description="Memory configuration")
    guardrails: Optional[GuardrailsConfig] = Field(None, description="Guardrails configuration")
    infrastructure: Optional[InfrastructureConfig] = Field(None, description="Infrastructure configuration")
    
    # Pattern specific configurations
    supervisor: Optional[SupervisorConfig] = Field(None, description="Supervisor pattern configuration")
    network: Optional[NetworkConfig] = Field(None, description="Network pattern configuration")
    sequential: Optional[SequentialConfig] = Field(None, description="Sequential pattern configuration")
    map_reduce: Optional[MapReduceConfig] = Field(None, description="MapReduce pattern configuration")
    autogen: Optional[AutoGenConfig] = Field(None, description="AutoGen pattern configuration")
    langgraph: Optional[LangGraphConfig] = Field(None, description="LangGraph pattern configuration")
    framework_router: Optional[FrameworkRouterConfig] = Field(None, description="Framework first-route pattern configuration")

class AgentSpec(BaseModel):
    """Top-level Agent specification."""
    apiVersion: Literal["agentcomponents/v1"] = Field("agentcomponents/v1", description="API Version")
    kind: Literal["Agent"] = Field("Agent", description="Resource kind")
    metadata: Metadata = Field(..., description="Resource metadata")
    spec: AgentSpecModel = Field(..., description="Agent specification details")
