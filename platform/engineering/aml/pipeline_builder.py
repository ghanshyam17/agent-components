from __future__ import annotations

import logging
from typing import Dict, Any, List

from platform.engineering.aml.models import AMLPipelineJobSpec, AMLPipelineStepSpec

logger = logging.getLogger(__name__)

class AMLPipelineBuilder:
    """Builder for Azure ML pipeline DAGs."""

    def __init__(self, name: str, compute: str):
        """Initialize the pipeline builder."""
        self.name = name
        self.compute = compute
        self.steps: List[AMLPipelineStepSpec] = []
        self.inputs: Dict[str, Any] = {}
        self.outputs: Dict[str, Any] = {}

    def add_step(self, step: AMLPipelineStepSpec) -> AMLPipelineBuilder:
        """Add a step to the pipeline."""
        self.steps.append(step)
        return self

    def build(self) -> AMLPipelineJobSpec:
        """Build the pipeline specification."""
        logger.info(f"Building pipeline: {self.name} with {len(self.steps)} steps.")
        return AMLPipelineJobSpec(
            name=self.name,
            compute=self.compute,
            steps=self.steps,
            inputs=self.inputs,
            outputs=self.outputs
        )

    def generate_yaml(self) -> str:
        """Generate Azure ML v2 YAML specification for the pipeline."""
        yaml_content = f"name: {self.name}\n"
        yaml_content += "type: pipeline\n"
        yaml_content += f"compute: azureml:{self.compute}\n"
        yaml_content += "jobs:\n"
        for step in self.steps:
            yaml_content += f"  {step.name}:\n"
            yaml_content += f"    component: {step.component_or_command}\n"
        return yaml_content
