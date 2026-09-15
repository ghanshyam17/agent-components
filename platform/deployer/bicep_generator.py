from __future__ import annotations
import logging
from pathlib import Path
from typing import List

from platform.schema.infrastructure_spec import InfrastructureSpec
from platform.schema.project_spec import ProjectSpec

logger = logging.getLogger(__name__)

class BicepGenerator:
    """Bicep template generator from YAML infrastructure specs."""

    def generate(self, infra_spec: InfrastructureSpec, output_dir: Path) -> List[Path]:
        """Generates .bicep files for each resource in the spec."""
        logger.info(f"Generating bicep for {infra_spec.metadata.name} into {output_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        generated_files = []
        
        for resource in infra_spec.spec.resources:
            bicep_path = output_dir / f"{resource.name}.bicep"
            # Template generation logic based on resource.type
            # Example for aiSearch:
            # resource aiSearch 'Microsoft.Search/searchServices@2024-03-01-preview'
            
            bicep_content = f"// Generated Bicep for {resource.name} ({resource.type})\n"
            
            if resource.type == "azure/ai-search":
                bicep_content += "resource aiSearch 'Microsoft.Search/searchServices@2024-03-01-preview' = {\\n"
                bicep_content += f"  name: '{resource.name}'\\n"
                bicep_content += "  location: resourceGroup().location\\n"
                bicep_content += "}\\n"
                
            with bicep_path.open("w") as f:
                f.write(bicep_content)
                
            generated_files.append(bicep_path)
            
        return generated_files

    def generate_main(self, project_spec: ProjectSpec, output_dir: Path) -> Path:
        """Generates main.bicep with all modules referenced."""
        logger.info(f"Generating main bicep for project {project_spec.metadata.name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        main_bicep_path = output_dir / "main.bicep"
        with main_bicep_path.open("w") as f:
            f.write(f"// Main Bicep for Project {project_spec.metadata.name}\n")
            f.write("targetScope = 'subscription'\n")
            
        return main_bicep_path
