from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class BaseActivityBuilder(BaseModel):
    name: str
    description: Optional[str] = None
    depends_on: List[Dict[str, Any]] = Field(default_factory=list)

    def build(self) -> Dict[str, Any]:
        raise NotImplementedError

class CopyActivityBuilder(BaseActivityBuilder):
    source: Dict[str, Any]
    sink: Dict[str, Any]
    translator: Optional[Dict[str, Any]] = None

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "Copy",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "source": self.source,
                "sink": self.sink,
                "translator": self.translator
            }
        }

class MappingDataFlowBuilder(BaseActivityBuilder):
    data_flow: Dict[str, Any]
    compute: Optional[Dict[str, Any]] = None

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "ExecuteDataFlow",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "dataFlow": self.data_flow,
                "compute": self.compute or {"coreCount": 8, "computeType": "General"}
            }
        }

class SynapseSparkActivityBuilder(BaseActivityBuilder):
    notebook: Dict[str, Any]
    parameters: Dict[str, Any] = Field(default_factory=dict)

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "SynapseNotebook",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "notebook": self.notebook,
                "parameters": self.parameters
            }
        }

class DatabricksNotebookActivityBuilder(BaseActivityBuilder):
    notebook_path: str
    base_parameters: Dict[str, Any] = Field(default_factory=dict)

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "DatabricksNotebook",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "notebookPath": self.notebook_path,
                "baseParameters": self.base_parameters
            }
        }

class AzureFunctionActivityBuilder(BaseActivityBuilder):
    function_name: str
    method: str = "POST"
    body: Optional[Dict[str, Any]] = None

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "AzureFunctionActivity",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "functionName": self.function_name,
                "method": self.method,
                "body": self.body
            }
        }

class WebActivityBuilder(BaseActivityBuilder):
    url: str
    method: str
    body: Optional[Any] = None
    headers: Dict[str, str] = Field(default_factory=dict)

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "WebActivity",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "url": self.url,
                "method": self.method,
                "body": self.body,
                "headers": self.headers
            }
        }

class ForEachActivityBuilder(BaseActivityBuilder):
    items: Any
    activities: List[Dict[str, Any]]

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "ForEach",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "items": self.items,
                "activities": self.activities
            }
        }

class IfConditionActivityBuilder(BaseActivityBuilder):
    expression: Any
    if_true_activities: List[Dict[str, Any]]
    if_false_activities: List[Dict[str, Any]] = Field(default_factory=list)

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "IfCondition",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "expression": self.expression,
                "ifTrueActivities": self.if_true_activities,
                "ifFalseActivities": self.if_false_activities
            }
        }

class LookupActivityBuilder(BaseActivityBuilder):
    source: Dict[str, Any]
    dataset: Dict[str, Any]

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "Lookup",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "source": self.source,
                "dataset": self.dataset
            }
        }

class GetMetadataActivityBuilder(BaseActivityBuilder):
    dataset: Dict[str, Any]
    field_list: List[str]

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "type": "GetMetadata",
            "description": self.description,
            "dependsOn": self.depends_on,
            "typeProperties": {
                "dataset": self.dataset,
                "fieldList": self.field_list
            }
        }
