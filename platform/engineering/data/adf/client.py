from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
import asyncio

logger = logging.getLogger(__name__)

try:
    from azure.mgmt.datafactory import DataFactoryManagementClient
    from azure.mgmt.datafactory.models import (
        PipelineResource, TriggerResource, LinkedServiceResource, DatasetResource
    )
    HAS_AZURE_ADF = True
except ImportError:
    HAS_AZURE_ADF = False
    logger.warning("azure-mgmt-datafactory is not installed. ADFClient will operate in mock mode.")

class ADFClient:
    """Azure Data Factory SDK client wrapper."""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        factory_name: str,
        credential: Any = None
    ) -> None:
        self.subscription_id = subscription_id
        self.resource_group = resource_group
        self.factory_name = factory_name
        self.credential = credential
        
        self.client = None
        if HAS_AZURE_ADF and self.credential:
            self.client = DataFactoryManagementClient(self.credential, self.subscription_id)

    async def create_or_update_pipeline(
        self, pipeline_name: str, activities: List[Dict[str, Any]], parameters: Optional[Dict[str, Any]] = None, variables: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Creates or updates an ADF pipeline."""
        logger.info(f"Creating pipeline {pipeline_name} in {self.factory_name}")
        
        if not HAS_AZURE_ADF or not self.client:
            return {"status": "mocked_success", "pipeline_name": pipeline_name, "activities_count": len(activities)}

        # Ensure we run sync methods in an executor if needed, though azure SDK has async equivalents via azure.mgmt.datafactory.aio
        pipeline_resource = PipelineResource(activities=activities, parameters=parameters, variables=variables)
        
        result = await asyncio.to_thread(
            self.client.pipelines.create_or_update,
            self.resource_group,
            self.factory_name,
            pipeline_name,
            pipeline_resource
        )
        return {"name": result.name, "type": result.type}

    async def create_linked_service(self, name: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Creating linked service {name}")
        if not HAS_AZURE_ADF or not self.client:
            return {"status": "mocked_success", "linked_service_name": name}
        ls_resource = LinkedServiceResource(properties=properties)
        result = await asyncio.to_thread(
            self.client.linked_services.create_or_update,
            self.resource_group,
            self.factory_name,
            name,
            ls_resource
        )
        return {"name": result.name}

    async def create_dataset(self, name: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Creating dataset {name}")
        if not HAS_AZURE_ADF or not self.client:
            return {"status": "mocked_success", "dataset_name": name}
        ds_resource = DatasetResource(properties=properties)
        result = await asyncio.to_thread(
            self.client.datasets.create_or_update,
            self.resource_group,
            self.factory_name,
            name,
            ds_resource
        )
        return {"name": result.name}

    async def create_trigger(self, trigger_name: str, trigger_properties: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Creating trigger {trigger_name}")
        if not HAS_AZURE_ADF or not self.client:
            return {"status": "mocked_success", "trigger_name": trigger_name}
        tr_resource = TriggerResource(properties=trigger_properties)
        result = await asyncio.to_thread(
            self.client.triggers.create_or_update,
            self.resource_group,
            self.factory_name,
            trigger_name,
            tr_resource
        )
        return {"name": result.name}

    async def trigger_pipeline(self, pipeline_name: str, parameters: Optional[Dict[str, Any]] = None) -> str:
        logger.info(f"Triggering pipeline {pipeline_name}")
        if not HAS_AZURE_ADF or not self.client:
            import uuid
            return f"mock_run_{uuid.uuid4().hex}"
        run_response = await asyncio.to_thread(
            self.client.pipelines.create_run,
            self.resource_group,
            self.factory_name,
            pipeline_name,
            parameters=parameters
        )
        return run_response.run_id

    async def get_pipeline_run_status(self, run_id: str) -> Dict[str, Any]:
        logger.info(f"Getting status for pipeline run {run_id}")
        if not HAS_AZURE_ADF or not self.client:
            return {"status": "Succeeded", "run_id": run_id}
        run_status = await asyncio.to_thread(
            self.client.pipeline_runs.get,
            self.resource_group,
            self.factory_name,
            run_id
        )
        return {"status": run_status.status, "run_id": run_status.run_id, "duration_ms": run_status.duration_in_ms}
