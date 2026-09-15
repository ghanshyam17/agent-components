from __future__ import annotations

import logging

from platform.engineering.aml.client import AMLClient
from platform.engineering.aml.jobs import AMLJobRunner
from platform.engineering.aml.model_registry import AMLModelRegistryManager

logger = logging.getLogger(__name__)

class AMLInfraManager:
    """Unified manager for all Azure Machine Learning capabilities."""

    def __init__(self, subscription_id: str, resource_group: str, workspace_name: str):
        """Initialize the AML infrastructure manager."""
        self.client = AMLClient(
            subscription_id=subscription_id,
            resource_group=resource_group,
            workspace_name=workspace_name
        )
        self.jobs = AMLJobRunner(self.client)
        self.models = AMLModelRegistryManager(self.client)
        logger.info(f"Initialized AMLInfraManager for workspace {workspace_name}")
