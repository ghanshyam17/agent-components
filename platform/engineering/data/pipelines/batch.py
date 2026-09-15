from __future__ import annotations
import logging
from typing import Any, Dict
from platform.engineering.data.models import BatchPipelineSpec, MedallionPipelineSpec
from platform.engineering.data.adf.client import ADFClient

logger = logging.getLogger(__name__)

class BatchPipelineOrchestrator:
    """Orchestrates Batch ELT/ETL and Medallion Lakehouse pipelines."""
    
    def __init__(self, adf_client: ADFClient) -> None:
        self.adf_client = adf_client
        
    async def deploy_batch_pipeline(self, spec: BatchPipelineSpec) -> Dict[str, Any]:
        logger.info(f"Deploying batch pipeline: {spec.name}")
        activities = [
            {
                "name": f"Copy_{spec.source.name}_to_{spec.sink.name}",
                "type": "Copy",
                "typeProperties": {
                    "source": {"type": "BinarySource"},
                    "sink": {"type": "BinarySink"}
                }
            }
        ]
        result = await self.adf_client.create_or_update_pipeline(spec.name, activities)
        return {"status": "success", "pipeline": result}

    async def deploy_medallion_pipeline(self, spec: MedallionPipelineSpec) -> Dict[str, Any]:
        logger.info(f"Deploying Medallion pipeline: {spec.name}")
        activities = [
            {"name": "LandingToBronze", "type": "DatabricksNotebook", "typeProperties": {"notebookPath": "/Shared/landing_to_bronze"}},
            {"name": "BronzeToSilver", "type": "DatabricksNotebook", "typeProperties": {"notebookPath": "/Shared/bronze_to_silver"}, "dependsOn": [{"activity": "LandingToBronze", "dependencyConditions": ["Succeeded"]}]},
            {"name": "SilverToGold", "type": "DatabricksNotebook", "typeProperties": {"notebookPath": "/Shared/silver_to_gold"}, "dependsOn": [{"activity": "BronzeToSilver", "dependencyConditions": ["Succeeded"]}]}
        ]
        result = await self.adf_client.create_or_update_pipeline(spec.name, activities)
        return {"status": "success", "pipeline": result}
