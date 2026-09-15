from __future__ import annotations
from typing import Literal, Dict, Any, List, Optional
from pydantic import BaseModel, Field

DataSourceType = Literal["azure-blob", "adls-gen2", "azure-sql", "cosmos-db", "snowflake", "bigquery", "rest-api", "sftp", "event-hubs", "kafka"]
DataSinkType = Literal["adls-gen2", "azure-sql", "cosmos-db", "ai-search", "snowflake", "event-hubs", "delta-lake", "parquet"]
DatasetFormat = Literal["parquet", "delta", "json", "csv", "avro", "binary"]
MedallionStage = Literal["bronze", "silver", "gold"]

class DatasetSpec(BaseModel):
    name: str
    source_type: DataSourceType
    path_or_table: str
    format: DatasetFormat
    partition_keys: List[str] = Field(default_factory=list)
    schema_definition: Dict[str, Any] = Field(default_factory=dict)

class BatchPipelineSpec(BaseModel):
    name: str
    pipeline_type: Literal["batch"] = "batch"
    source: DatasetSpec
    sink: DatasetSpec
    schedule_cron: Optional[str] = None
    incremental_watermark_column: Optional[str] = None
    staging_path: Optional[str] = None
    column_mappings: Dict[str, str] = Field(default_factory=dict)

class MedallionPipelineSpec(BaseModel):
    name: str
    raw_landing_path: str
    bronze_path: str
    silver_path: str
    gold_path: str
    format: Literal["delta"] = "delta"
    deduplication_keys: List[str] = Field(default_factory=list)
    business_aggregations: List[Dict[str, Any]] = Field(default_factory=list)

class StreamingPipelineSpec(BaseModel):
    name: str
    pipeline_type: Literal["streaming"] = "streaming"
    event_hub_namespace: str
    event_hub_name: str
    consumer_group: str = "$Default"
    window_type: Literal["tumbling", "sliding", "session"] = "tumbling"
    window_duration: str = "5m"
    sink: DatasetSpec

class VectorRAGPipelineSpec(BaseModel):
    name: str
    pipeline_type: Literal["vector-rag"] = "vector-rag"
    document_source_path: str
    chunking_strategy: Literal["semantic", "fixed_size"] = "semantic"
    chunk_size: int = 512
    chunk_overlap: int = 50
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    ai_search_index_name: str = "agent-knowledge"
    hybrid_search: bool = True

class FeatureStoreSpec(BaseModel):
    name: str
    entity_keys: List[str]
    offline_store_path: str
    online_store_type: Literal["redis", "cosmos"] = "redis"
    sync_frequency_minutes: int = 60
    feature_definitions: List[Dict[str, Any]] = Field(default_factory=list)

class DataQualityCheck(BaseModel):
    column: str
    check_type: Literal["null_check", "unique_check", "range_check", "regex_check", "custom_sql"]
    threshold: Optional[float] = None
    severity: Literal["error", "warning", "info"] = "error"

class DataQualitySpec(BaseModel):
    dataset_name: str
    checks: List[DataQualityCheck] = Field(default_factory=list)
    quarantine_sink: Optional[DatasetSpec] = None
    fail_on_error: bool = True

class ADFActivitySpec(BaseModel):
    name: str
    type: str
    inputs: List[Dict[str, Any]] = Field(default_factory=list)
    outputs: List[Dict[str, Any]] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    depends_on: List[Dict[str, Any]] = Field(default_factory=list)

class UnifiedDataPipelineSpec(BaseModel):
    spec_type: Literal["batch", "medallion", "streaming", "vector-rag", "feature-store", "quality"]
    spec: BaseModel
