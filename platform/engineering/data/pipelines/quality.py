from __future__ import annotations
import logging
from typing import Any, Dict
from platform.engineering.data.models import DataQualitySpec
from platform.engineering.data.adf.client import ADFClient

logger = logging.getLogger(__name__)

class DataQualityGate:
    """Orchestrates data quality assertions and quarantine dead-letter routing."""

    def __init__(self, adf_client: ADFClient) -> None:
        self.adf_client = adf_client

    async def apply_quality_gate(self, spec: DataQualitySpec) -> Dict[str, Any]:
        logger.info(f"Applying quality gate for dataset: {spec.dataset_name}")
        activities = [
            {
                "name": f"CheckQuality_{spec.dataset_name}",
                "type": "SynapseNotebook",
                "typeProperties": {
                    "notebook": {"referenceName": "DataQualityChecker", "type": "NotebookReference"},
                    "parameters": {
                        "dataset": {"value": spec.dataset_name, "type": "String"},
                        "checks": {"value": str([c.model_dump() for c in spec.checks]), "type": "String"}
                    }
                }
            }
        ]
        result = await self.adf_client.create_or_update_pipeline(f"QualityGate_{spec.dataset_name}", activities)
        return {"status": "success", "pipeline": result}
