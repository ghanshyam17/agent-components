from __future__ import annotations
import logging
from typing import Any, Dict
from platform.engineering.data.models import VectorRAGPipelineSpec
from platform.engineering.data.adf.client import ADFClient

logger = logging.getLogger(__name__)

class VectorRAGPipelineOrchestrator:
    """Orchestrates doc chunking, embedding, and Azure AI Search indexing pipelines."""
    
    def __init__(self, adf_client: ADFClient) -> None:
        self.adf_client = adf_client
        
    async def deploy_rag_pipeline(self, spec: VectorRAGPipelineSpec) -> Dict[str, Any]:
        logger.info(f"Deploying Vector RAG pipeline: {spec.name}")
        activities = [
            {
                "name": "ExtractAndChunkDocs",
                "type": "AzureFunctionActivity",
                "typeProperties": {
                    "functionName": "chunk_documents",
                    "method": "POST"
                }
            },
            {
                "name": "EmbedAndIndex",
                "type": "AzureFunctionActivity",
                "dependsOn": [{"activity": "ExtractAndChunkDocs", "dependencyConditions": ["Succeeded"]}],
                "typeProperties": {
                    "functionName": "embed_and_index",
                    "method": "POST"
                }
            }
        ]
        result = await self.adf_client.create_or_update_pipeline(spec.name, activities)
        return {"status": "success", "pipeline": result}
