from __future__ import annotations
import json
from typing import Any, Dict, List

class ADFTemplateGenerator:
    """Synthesizes ARM JSON deployment templates and Bicep modules for Data Factory solutions."""

    def __init__(self, factory_name: str, location: str = "[resourceGroup().location]"):
        self.factory_name = factory_name
        self.location = location
        self.resources: List[Dict[str, Any]] = []
    
    def add_linked_service(self, name: str, properties: Dict[str, Any]) -> None:
        self.resources.append({
            "type": "Microsoft.DataFactory/factories/linkedservices",
            "apiVersion": "2018-06-01",
            "name": f"{self.factory_name}/{name}",
            "properties": properties,
            "dependsOn": [
                f"[resourceId('Microsoft.DataFactory/factories', '{self.factory_name}')]"
            ]
        })

    def add_dataset(self, name: str, properties: Dict[str, Any]) -> None:
        self.resources.append({
            "type": "Microsoft.DataFactory/factories/datasets",
            "apiVersion": "2018-06-01",
            "name": f"{self.factory_name}/{name}",
            "properties": properties,
            "dependsOn": [
                f"[resourceId('Microsoft.DataFactory/factories', '{self.factory_name}')]"
            ]
        })
        
    def add_pipeline(self, name: str, properties: Dict[str, Any]) -> None:
        self.resources.append({
            "type": "Microsoft.DataFactory/factories/pipelines",
            "apiVersion": "2018-06-01",
            "name": f"{self.factory_name}/{name}",
            "properties": properties,
            "dependsOn": [
                f"[resourceId('Microsoft.DataFactory/factories', '{self.factory_name}')]"
            ]
        })

    def add_trigger(self, name: str, properties: Dict[str, Any]) -> None:
        self.resources.append({
            "type": "Microsoft.DataFactory/factories/triggers",
            "apiVersion": "2018-06-01",
            "name": f"{self.factory_name}/{name}",
            "properties": properties,
            "dependsOn": [
                f"[resourceId('Microsoft.DataFactory/factories', '{self.factory_name}')]"
            ]
        })

    def generate_arm_template(self) -> Dict[str, Any]:
        """Generates the full ARM template for the ADF components."""
        return {
            "$schema": "https://schema.management.azure.com/schemas/2019-04-01/deploymentTemplate.json#",
            "contentVersion": "1.0.0.0",
            "resources": self.resources
        }

    def to_json(self) -> str:
        return json.dumps(self.generate_arm_template(), indent=2)
