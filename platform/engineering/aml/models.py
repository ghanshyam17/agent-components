from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

AMLComputeType = Literal["amlcompute", "computeinstance", "kubernetes", "serverless"]
AMLDataAssetType = Literal["uri_file", "uri_folder", "mltable"]
AMLSweepMetric = Literal["accuracy", "loss", "f1", "val_loss", "custom"]


class AMLWorkspaceSpec(BaseModel):
    """Specification for an Azure Machine Learning Workspace."""
    name: str
    subscription_id: str
    resource_group: str
    location: str
    key_vault: Optional[str] = None
    storage_account: Optional[str] = None
    container_registry: Optional[str] = None
    application_insights: Optional[str] = None


class AMLComputeSpec(BaseModel):
    """Specification for Azure ML compute resources."""
    name: str
    compute_type: AMLComputeType
    vm_size: str = "Standard_DS3_v2"
    min_instances: int = 0
    max_instances: int = 4
    idle_time_before_scale_down: int = 120


class AMLEnvironmentSpec(BaseModel):
    """Specification for an Azure ML Environment."""
    name: str
    version: str = "1.0"
    image: Optional[str] = None
    conda_file: Optional[str] = None
    dockerfile: Optional[str] = None
    build_context: Optional[str] = None


class AMLDataAssetSpec(BaseModel):
    """Specification for an Azure ML Data Asset."""
    name: str
    version: str = "1.0"
    asset_type: AMLDataAssetType
    path: str
    description: Optional[str] = None


class AMLCommandJobSpec(BaseModel):
    """Specification for an Azure ML Command Job."""
    name: str
    command: str
    code_path: Optional[str] = None
    environment: Optional[str] = None
    compute: Optional[str] = None
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    environment_variables: Dict[str, str] = Field(default_factory=dict)


class AMLSweepJobSpec(BaseModel):
    """Specification for an Azure ML Sweep (Hyperparameter Tuning) Job."""
    name: str
    trial_command_job: AMLCommandJobSpec
    sampling_algorithm: Literal["random", "bayesian", "grid"] = "random"
    primary_metric: str
    goal: Literal["maximize", "minimize"] = "maximize"
    max_total_trials: int = 20
    max_concurrent_trials: int = 4
    search_space: Dict[str, Any] = Field(default_factory=dict)


class AMLPipelineStepSpec(BaseModel):
    """Specification for a step in an Azure ML Pipeline Job."""
    name: str
    component_or_command: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)


class AMLPipelineJobSpec(BaseModel):
    """Specification for an Azure ML Pipeline Job."""
    name: str
    compute: Optional[str] = None
    steps: List[AMLPipelineStepSpec] = Field(default_factory=list)
    inputs: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)


class AMLModelSpec(BaseModel):
    """Specification for registering a model in Azure ML."""
    name: str
    version: str = "1.0"
    path: str
    model_format: Literal["custom", "mlflow", "onnx", "huggingface"] = "mlflow"
    tags: Dict[str, str] = Field(default_factory=dict)


class AMLEndpointSpec(BaseModel):
    """Specification for an Azure ML Endpoint."""
    name: str
    endpoint_type: Literal["online", "batch"] = "online"
    auth_mode: Literal["key", "aad_token"] = "key"
    traffic: Dict[str, int] = Field(default_factory=dict)


class AMLBatchDeploymentSpec(BaseModel):
    """Specification for an Azure ML Batch Deployment."""
    name: str
    endpoint_name: str
    model: str
    compute: str
    instance_count: int = 1
    max_concurrency_per_instance: int = 2
