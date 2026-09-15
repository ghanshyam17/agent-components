from __future__ import annotations

"""YAML Schema & Configuration Engine for the agent-components platform."""

from platform.schema.agent_spec import (
    AgentSpec,
    AgentSpecModel,
    Metadata,
    ModelConfig,
    ToolsConfig,
    MemoryConfig,
    GuardrailsConfig,
    InfrastructureConfig,
    CodeInterpreterConfig,
    CustomTool,
    SessionMemoryConfig,
    VectorMemoryConfig,
)
from platform.schema.project_spec import (
    ProjectSpec,
    ProjectSpecModel,
    ProjectMetadata,
    FoundryConfig,
)
from platform.schema.infrastructure_spec import (
    InfrastructureSpec,
    InfrastructureSpecModel,
    InfraMetadata,
    ResourceType,
    AISearchResource,
    RedisCacheResource,
    StorageAccountResource,
    DynamicSessionsResource,
    CosmosDBResource,
    AppServicePlanResource,
    ContainerAppsEnvResource,
)
from platform.schema.pattern_spec import (
    SupervisorConfig,
    NetworkConfig,
    SequentialConfig,
    MapReduceConfig,
    WorkerRef,
    AutoGenConfig,
    AutoGenAgentConfig,
    LangGraphConfig,
    LangGraphNodeConfig,
    LangGraphEdgeConfig,
    LangGraphConditionalEdgeConfig,
    FrameworkRouteRule,
    FrameworkRouterConfig,
)
from platform.schema.loader import (
    BaseSpec,
    load_spec,
    load_project,
    resolve_variables,
    resolve_includes,
    validate_spec,
)

__all__ = [
    # agent_spec
    "AgentSpec",
    "AgentSpecModel",
    "Metadata",
    "ModelConfig",
    "ToolsConfig",
    "MemoryConfig",
    "GuardrailsConfig",
    "InfrastructureConfig",
    "CodeInterpreterConfig",
    "CustomTool",
    "SessionMemoryConfig",
    "VectorMemoryConfig",
    # project_spec
    "ProjectSpec",
    "ProjectSpecModel",
    "ProjectMetadata",
    "FoundryConfig",
    # infrastructure_spec
    "InfrastructureSpec",
    "InfrastructureSpecModel",
    "InfraMetadata",
    "ResourceType",
    "AISearchResource",
    "RedisCacheResource",
    "StorageAccountResource",
    "DynamicSessionsResource",
    "CosmosDBResource",
    "AppServicePlanResource",
    "ContainerAppsEnvResource",
    # pattern_spec
    "SupervisorConfig",
    "NetworkConfig",
    "SequentialConfig",
    "MapReduceConfig",
    "WorkerRef",
    "AutoGenConfig",
    "AutoGenAgentConfig",
    "LangGraphConfig",
    "LangGraphNodeConfig",
    "LangGraphEdgeConfig",
    "LangGraphConditionalEdgeConfig",
    "FrameworkRouteRule",
    "FrameworkRouterConfig",
    # loader
    "BaseSpec",
    "load_spec",
    "load_project",
    "resolve_variables",
    "resolve_includes",
    "validate_spec",
]

