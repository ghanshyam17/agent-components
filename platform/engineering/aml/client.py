from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from azure.ai.ml import MLClient
    from azure.identity import DefaultAzureCredential
    from azure.ai.ml.entities import (
        AmlCompute, ComputeInstance, KubernetesCompute,
        Environment, BuildContext,
        Data,
        CommandJob, SweepJob, PipelineJob,
        Model, ManagedOnlineEndpoint, BatchEndpoint
    )
    AZURE_ML_AVAILABLE = True
except ImportError:
    AZURE_ML_AVAILABLE = False
    logger.warning("azure-ai-ml or azure-identity not installed. AMLClient will use mock implementations.")

from platform.engineering.aml.models import (
    AMLComputeSpec, AMLEnvironmentSpec, AMLDataAssetSpec,
    AMLCommandJobSpec, AMLSweepJobSpec, AMLPipelineJobSpec,
    AMLModelSpec, AMLEndpointSpec
)

class AMLClient:
    """Wrapper for the Azure Machine Learning v2 SDK."""

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str, credential: Any = None):
        """Initialize the AMLClient."""
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.workspace_name = workspace_name
        self.credential = credential
        self.ml_client = None

        if AZURE_ML_AVAILABLE:
            if not self.credential:
                self.credential = DefaultAzureCredential()
            self.ml_client = MLClient(
                credential=self.credential,
                subscription_id=self.subscription_id,
                resource_group_name=self.resource_group,
                workspace_name=self.workspace_name,
            )
        else:
            logger.warning("Initializing AMLClient without Azure ML SDK available.")

    async def provision_compute(self, compute_spec: AMLComputeSpec) -> None:
        """Provision an Azure ML compute target."""
        logger.info(f"Provisioning compute: {compute_spec.name}")
        if not self.ml_client:
            return
        
        if compute_spec.compute_type == "amlcompute":
            compute = AmlCompute(
                name=compute_spec.name,
                size=compute_spec.vm_size,
                min_instances=compute_spec.min_instances,
                max_instances=compute_spec.max_instances,
                idle_time_before_scale_down=compute_spec.idle_time_before_scale_down
            )
            self.ml_client.compute.begin_create_or_update(compute)
            logger.info(f"Compute {compute_spec.name} provisioning started.")
        else:
            logger.warning(f"Compute type {compute_spec.compute_type} provisioning mapping not implemented.")

    async def create_environment(self, env_spec: AMLEnvironmentSpec) -> None:
        """Create an Azure ML environment."""
        logger.info(f"Creating environment: {env_spec.name}")
        if not self.ml_client:
            return

        env = Environment(
            name=env_spec.name,
            version=env_spec.version,
            image=env_spec.image,
            conda_file=env_spec.conda_file
        )
        self.ml_client.environments.create_or_update(env)
        logger.info(f"Environment {env_spec.name} created.")

    async def create_data_asset(self, data_spec: AMLDataAssetSpec) -> None:
        """Create an Azure ML data asset."""
        logger.info(f"Creating data asset: {data_spec.name}")
        if not self.ml_client:
            return
            
        my_data = Data(
            name=data_spec.name,
            version=data_spec.version,
            path=data_spec.path,
            type=data_spec.asset_type,
            description=data_spec.description
        )
        self.ml_client.data.create_or_update(my_data)
        logger.info(f"Data asset {data_spec.name} created.")

    async def submit_job(self, job_spec: Any) -> str:
        """Submit an Azure ML job."""
        logger.info(f"Submitting job: {job_spec.name}")
        if not self.ml_client:
            return f"mock_job_id_for_{job_spec.name}"
            
        # Simplified mapping
        job_entity = CommandJob(
            name=job_spec.name,
            command=job_spec.command,
            environment=job_spec.environment,
            compute=job_spec.compute
        )
        returned_job = self.ml_client.jobs.create_or_update(job_entity)
        return str(returned_job.name)

    async def get_job_status(self, job_name: str) -> dict:
        """Get the status of an Azure ML job."""
        logger.info(f"Getting status for job: {job_name}")
        if not self.ml_client:
            return {"status": "Completed"}
            
        job = self.ml_client.jobs.get(job_name)
        return {"status": job.status}

    async def register_model(self, model_spec: AMLModelSpec) -> dict:
        """Register a model in Azure ML."""
        logger.info(f"Registering model: {model_spec.name}")
        if not self.ml_client:
            return {"name": model_spec.name, "version": model_spec.version}
            
        model = Model(
            name=model_spec.name,
            version=model_spec.version,
            path=model_spec.path,
            type=model_spec.model_format,
            tags=model_spec.tags
        )
        returned_model = self.ml_client.models.create_or_update(model)
        return {"name": returned_model.name, "version": returned_model.version}

    async def create_endpoint(self, endpoint_spec: AMLEndpointSpec) -> dict:
        """Create an Azure ML endpoint."""
        logger.info(f"Creating endpoint: {endpoint_spec.name}")
        if not self.ml_client:
            return {"name": endpoint_spec.name, "status": "Succeeded"}
            
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_spec.name,
            auth_mode=endpoint_spec.auth_mode
        )
        returned_endpoint = self.ml_client.online_endpoints.begin_create_or_update(endpoint).result()
        return {"name": returned_endpoint.name, "status": "Succeeded"}
