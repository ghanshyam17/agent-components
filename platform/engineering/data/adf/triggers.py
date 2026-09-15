from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class BaseTriggerBuilder(BaseModel):
    name: str
    description: Optional[str] = None
    pipelines: List[Dict[str, Any]] = Field(default_factory=list)

    def build(self) -> Dict[str, Any]:
        raise NotImplementedError

class ScheduleTriggerBuilder(BaseTriggerBuilder):
    frequency: str = "Minute"
    interval: int = 15
    start_time: str

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "properties": {
                "type": "ScheduleTrigger",
                "description": self.description,
                "pipelines": self.pipelines,
                "typeProperties": {
                    "recurrence": {
                        "frequency": self.frequency,
                        "interval": self.interval,
                        "startTime": self.start_time
                    }
                }
            }
        }

class TumblingWindowTriggerBuilder(BaseTriggerBuilder):
    frequency: str = "Minute"
    interval: int = 15
    start_time: str
    delay: str = "00:00:00"
    max_concurrency: int = 1
    retry_policy: Dict[str, Any] = Field(default_factory=lambda: {"count": 3, "intervalInSeconds": 30})

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "properties": {
                "type": "TumblingWindowTrigger",
                "description": self.description,
                "pipeline": self.pipelines[0] if self.pipelines else {},
                "typeProperties": {
                    "frequency": self.frequency,
                    "interval": self.interval,
                    "startTime": self.start_time,
                    "delay": self.delay,
                    "maxConcurrency": self.max_concurrency,
                    "retryPolicy": self.retry_policy
                }
            }
        }

class BlobEventsTriggerBuilder(BaseTriggerBuilder):
    scope: str
    events: List[str] = Field(default_factory=lambda: ["Microsoft.Storage.BlobCreated"])
    blob_path_begins_with: Optional[str] = None
    blob_path_ends_with: Optional[str] = None
    ignore_empty_blobs: bool = True

    def build(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "properties": {
                "type": "BlobEventsTrigger",
                "description": self.description,
                "pipelines": self.pipelines,
                "typeProperties": {
                    "scope": self.scope,
                    "events": self.events,
                    "blobPathBeginsWith": self.blob_path_begins_with,
                    "blobPathEndsWith": self.blob_path_ends_with,
                    "ignoreEmptyBlobs": self.ignore_empty_blobs
                }
            }
        }
