"""Component Graph Schema.

Defines the declarative specification for the end-to-end Component Graph:
wiring Data Engineering pipelines (ADF, Medallion, Vector RAG),
Azure Machine Learning workflows (models, endpoints, training),
Agentic Design Patterns (LangGraph, AutoGen, ReAct), and the
Azure AI Foundry SDK runtime chassis.
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

from platform.schema.agent_spec import AgentSpecModel, Metadata
from platform.schema.infrastructure_spec import ResourceType
from platform.engineering.data.models import (
    BatchPipelineSpec,
    MedallionPipelineSpec,
    StreamingPipelineSpec,
    VectorRAGPipelineSpec,
    DataQualitySpec,
)
from platform.engineering.aml.models import (
    AMLCommandJobSpec,
    AMLPipelineJobSpec,
    AMLSweepJobSpec,
    AMLModelSpec,
    AMLEndpointSpec,
)


class DataPlaneConfig(BaseModel):
    """Configuration for Data Engineering pipelines and services."""
    batch_pipelines: List[BatchPipelineSpec] = Field(default_factory=list, description="Batch ELT/ETL pipelines")
    medallion_pipelines: List[MedallionPipelineSpec] = Field(default_factory=list, description="Medallion Lakehouse pipelines")
    streaming_pipelines: List[StreamingPipelineSpec] = Field(default_factory=list, description="Streaming ingestion pipelines")
    rag_pipelines: List[VectorRAGPipelineSpec] = Field(default_factory=list, description="Vector RAG indexing pipelines")
    quality_gates: List[DataQualitySpec] = Field(default_factory=list, description="Data quality gate configurations")
    auto_generate_agent_tools: bool = Field(True, description="Automatically expose data outputs as agent tools")


class MLPlaneConfig(BaseModel):
    """Configuration for Azure Machine Learning workflows and endpoints."""
    command_jobs: List[AMLCommandJobSpec] = Field(default_factory=list, description="AML training command jobs")
    pipeline_jobs: List[AMLPipelineJobSpec] = Field(default_factory=list, description="AML multi-step training DAGs")
    sweep_jobs: List[AMLSweepJobSpec] = Field(default_factory=list, description="AML hyperparameter sweep jobs")
    models: List[AMLModelSpec] = Field(default_factory=list, description="Registered ML models")
    endpoints: List[AMLEndpointSpec] = Field(default_factory=list, description="Online/batch scoring endpoints")
    auto_generate_agent_tools: bool = Field(True, description="Automatically expose model endpoints as agent tools")


class ToolsPlaneConfig(BaseModel):
    """Configuration for tool registry wiring in the component graph."""
    enable_data_tools: bool = Field(True, description="Expose ADF and Lakehouse query tools to agents")
    enable_vector_search_tools: bool = Field(True, description="Expose Vector RAG AI Search tools to agents")
    enable_ml_inference_tools: bool = Field(True, description="Expose AML model scoring tools to agents")
    custom_plugin_files: List[str] = Field(default_factory=list, description="Paths to custom plugin YAML/JSON files")


class RuntimeChassisConfig(BaseModel):
    """Configuration for the deployment chassis on Azure."""
    provider: Literal["azure-foundry", "app-service", "container-apps", "functions"] = Field(
        "azure-foundry", description="Hosting runtime provider"
    )
    protocol: Literal["responses", "fastapi", "azure-functions"] = Field(
        "responses", description="Agent communication protocol"
    )
    sku: str = Field("Standard", description="Compute SKU")
    scale_to_zero: bool = Field(True, description="Enable scale-to-zero for idle cost reduction")


class ComponentGraphSpec(BaseModel):
    """Main specification model for the unified Component Graph."""
    data_plane: DataPlaneConfig = Field(default_factory=DataPlaneConfig, description="Data Engineering plane")
    ml_plane: MLPlaneConfig = Field(default_factory=MLPlaneConfig, description="Machine Learning plane")
    tools_plane: ToolsPlaneConfig = Field(default_factory=ToolsPlaneConfig, description="Tools plane")
    agent_plane: AgentSpecModel = Field(..., description="Agent configuration (LangGraph, AutoGen, ReAct, etc.)")
    runtime_chassis: RuntimeChassisConfig = Field(
        default_factory=RuntimeChassisConfig, description="Runtime hosting chassis"
    )
    infrastructure_resources: List[ResourceType] = Field(
        default_factory=list, description="Underlying Azure infrastructure resources"
    )


class ComponentGraph(BaseModel):
    """Top-level declarative specification for the end-to-end Component Graph."""
    apiVersion: Literal["agentcomponents/v1"] = Field("agentcomponents/v1", description="API Version")
    kind: Literal["ComponentGraph"] = Field("ComponentGraph", description="Resource kind")
    metadata: Metadata = Field(..., description="Component graph metadata")
    spec: ComponentGraphSpec = Field(..., description="Component graph specification details")
