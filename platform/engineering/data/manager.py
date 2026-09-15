from __future__ import annotations
import logging
from typing import Any, Dict, Optional
from pydantic import BaseModel

from platform.engineering.data.adf.client import ADFClient
from platform.engineering.data.pipelines.batch import BatchPipelineOrchestrator
from platform.engineering.data.pipelines.streaming import StreamingPipelineOrchestrator
from platform.engineering.data.pipelines.vector_rag import VectorRAGPipelineOrchestrator
from platform.engineering.data.pipelines.quality import DataQualityGate
from platform.engineering.data.models import UnifiedDataPipelineSpec

logger = logging.getLogger(__name__)

class DataInfraManager:
    """Manager uniting all data infrastructure capabilities."""

    def __init__(
        self,
        subscription_id: str,
        resource_group: str,
        factory_name: str,
        credential: Any = None
    ) -> None:
        self.adf_client = ADFClient(
            subscription_id=subscription_id,
            resource_group=resource_group,
            factory_name=factory_name,
            credential=credential
        )
        self.batch_orchestrator = BatchPipelineOrchestrator(self.adf_client)
        self.streaming_orchestrator = StreamingPipelineOrchestrator(self.adf_client)
        self.rag_orchestrator = VectorRAGPipelineOrchestrator(self.adf_client)
        self.quality_gate = DataQualityGate(self.adf_client)

    async def deploy_pipeline(self, pipeline_spec: UnifiedDataPipelineSpec) -> Dict[str, Any]:
        """Deploys a data pipeline based on the unified specification."""
        spec_type = pipeline_spec.spec_type
        spec = pipeline_spec.spec
        
        logger.info(f"Deploying pipeline of type: {spec_type}")
        
        if spec_type == "batch":
            return await self.batch_orchestrator.deploy_batch_pipeline(spec) # type: ignore
        elif spec_type == "medallion":
            return await self.batch_orchestrator.deploy_medallion_pipeline(spec) # type: ignore
        elif spec_type == "streaming":
            return await self.streaming_orchestrator.deploy_streaming_pipeline(spec) # type: ignore
        elif spec_type == "vector-rag":
            return await self.rag_orchestrator.deploy_rag_pipeline(spec) # type: ignore
        elif spec_type == "quality":
            return await self.quality_gate.apply_quality_gate(spec) # type: ignore
        else:
            raise ValueError(f"Unsupported pipeline specification type: {spec_type}")

    async def get_run_status(self, run_id: str) -> Dict[str, Any]:
        """Gets the status of a pipeline run."""
        return await self.adf_client.get_pipeline_run_status(run_id)
