from __future__ import annotations

import logging
import asyncio
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class DataSourceConfig(BaseModel):
    type: str = Field(..., description="blob-storage, sql-database, cosmos-db, api")
    connection_details: Dict[str, Any] = Field(default_factory=dict)

class DataSinkConfig(BaseModel):
    type: str = Field(..., description="blob-storage, sql-database, data-lake, bigquery")
    connection_details: Dict[str, Any] = Field(default_factory=dict)

class DataPipelineSpec(BaseModel):
    name: str = Field(...)
    type: str = Field(..., description="adf, synapse-spark, fabric, custom")
    source_config: DataSourceConfig
    sink_config: DataSinkConfig
    schedule: Optional[str] = None
    transforms: List[Dict[str, Any]] = Field(default_factory=list)

class DataInfraManager:
    """
    Data Engineering Infrastructure — YAML-driven data pipeline setup.
    """
    async def provision(self, spec: DataPipelineSpec) -> Dict[str, Any]:
        logger.info(f"Provisioning data infrastructure for {spec.name}")
        await asyncio.sleep(0.1)
        return {"status": "provisioned", "pipeline": spec.name}

    async def deploy_pipeline(self, spec: DataPipelineSpec) -> Dict[str, Any]:
        logger.info(f"Deploying pipeline {spec.name}")
        await asyncio.sleep(0.1)
        return {"status": "deployed", "pipeline": spec.name}

    def generate_adf_template(self, spec: DataPipelineSpec) -> Dict[str, Any]:
        return {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "resources": [
                {
                    "type": "Microsoft.DataFactory/factories/pipelines",
                    "name": spec.name,
                    "properties": {
                        "activities": spec.transforms
                    }
                }
            ]
        }

    def generate_synapse_config(self, spec: DataPipelineSpec) -> Dict[str, Any]:
        return {
            "name": spec.name,
            "properties": {
                "activities": spec.transforms
            }
        }

    async def get_status(self, pipeline_name: str) -> Dict[str, Any]:
        return {"status": "running", "pipeline": pipeline_name}
