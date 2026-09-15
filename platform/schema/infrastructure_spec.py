from __future__ import annotations
from typing import List, Literal, Optional, Dict, Union, Annotated

from pydantic import BaseModel, Field

class InfraMetadata(BaseModel):
    """Metadata for an Infrastructure specification."""
    name: str = Field(..., description="Name of the infrastructure setup")
    tags: Dict[str, str] = Field(default_factory=dict, description="Tags for the resources")

class BaseResource(BaseModel):
    """Base model for an infrastructure resource."""
    name: str = Field(..., description="Resource name")
    sku: Optional[str] = Field(None, description="Resource SKU")

class AISearchResource(BaseResource):
    """Azure AI Search resource."""
    type: Literal["azure/ai-search"] = Field("azure/ai-search", description="Resource type")
    replica_count: Optional[int] = Field(1, description="Replica count")

class RedisCacheResource(BaseResource):
    """Azure Redis Cache resource."""
    type: Literal["azure/redis-cache"] = Field("azure/redis-cache", description="Resource type")
    capacity: Optional[int] = Field(1, description="Cache capacity")

class StorageAccountResource(BaseResource):
    """Azure Storage Account resource."""
    type: Literal["azure/storage-account"] = Field("azure/storage-account", description="Resource type")
    account_type: Optional[str] = Field(None, description="Account type")

class DynamicSessionsResource(BaseResource):
    """Azure Dynamic Sessions resource."""
    type: Literal["azure/dynamic-sessions"] = Field("azure/dynamic-sessions", description="Resource type")
    pool_size: Optional[int] = Field(None, description="Session pool size")

class CosmosDBResource(BaseResource):
    """Azure Cosmos DB resource."""
    type: Literal["azure/cosmos-db"] = Field("azure/cosmos-db", description="Resource type")
    database_name: Optional[str] = Field(None, description="Database name")

class AppServicePlanResource(BaseResource):
    """Azure App Service Plan resource."""
    type: Literal["azure/app-service-plan"] = Field("azure/app-service-plan", description="Resource type")
    worker_count: Optional[int] = Field(1, description="Worker count")

class ContainerAppsEnvResource(BaseResource):
    """Azure Container Apps Environment resource."""
    type: Literal["azure/container-apps-env"] = Field("azure/container-apps-env", description="Resource type")

class SQLDatabaseResource(BaseResource):
    """Azure SQL Database resource."""
    type: Literal["azure/sql-database"] = Field("azure/sql-database", description="Resource type")
    database_name: Optional[str] = Field(None, description="Database name")

class FunctionsAppResource(BaseResource):
    """Azure Functions App resource."""
    type: Literal["azure/functions"] = Field("azure/functions", description="Resource type")
    runtime: Optional[str] = Field("python-3.12", description="Runtime stack")
    scaling: Optional[Dict[str, int]] = Field(None, description="Scaling config with min, max keys")

class ApplicationInsightsResource(BaseResource):
    """Azure Application Insights resource."""
    type: Literal["azure/application-insights"] = Field("azure/application-insights", description="Resource type")

class DataFactoryResource(BaseResource):
    """Azure Data Factory resource."""
    type: Literal["azure/data-factory"] = Field("azure/data-factory", description="Resource type")
    managed_vnet: Optional[bool] = Field(True, description="Enable managed virtual network")
    integration_runtimes: Optional[List[str]] = Field(default_factory=list, description="Integration runtime names")

class AzureMLWorkspaceResource(BaseResource):
    """Azure Machine Learning Workspace resource."""
    type: Literal["azure/ml-workspace"] = Field("azure/ml-workspace", description="Resource type")
    key_vault: Optional[str] = Field(None, description="Linked Key Vault ID/name")
    storage_account: Optional[str] = Field(None, description="Linked Storage Account ID/name")
    container_registry: Optional[str] = Field(None, description="Linked Container Registry ID/name")

class EventHubsResource(BaseResource):
    """Azure Event Hubs namespace resource."""
    type: Literal["azure/event-hubs"] = Field("azure/event-hubs", description="Resource type")
    capacity: Optional[int] = Field(1, description="Throughput units")
    kafka_enabled: Optional[bool] = Field(True, description="Enable Kafka endpoint")

class SynapseWorkspaceResource(BaseResource):
    """Azure Synapse Analytics Workspace resource."""
    type: Literal["azure/synapse-workspace"] = Field("azure/synapse-workspace", description="Resource type")
    default_data_lake_storage_account: Optional[str] = Field(None, description="Default ADLS Gen2 account")

ResourceType = Annotated[
    Union[
        AISearchResource,
        RedisCacheResource,
        StorageAccountResource,
        DynamicSessionsResource,
        CosmosDBResource,
        AppServicePlanResource,
        ContainerAppsEnvResource,
        SQLDatabaseResource,
        FunctionsAppResource,
        ApplicationInsightsResource,
        DataFactoryResource,
        AzureMLWorkspaceResource,
        EventHubsResource,
        SynapseWorkspaceResource,
    ],
    Field(discriminator="type"),
]


class InfrastructureSpecModel(BaseModel):
    """Main specification model for Infrastructure."""
    resources: list[ResourceType] = Field(..., description="List of infrastructure resources")

class InfrastructureSpec(BaseModel):
    """Top-level Infrastructure specification."""
    apiVersion: Literal["agentcomponents/v1"] = Field("agentcomponents/v1", description="API Version")
    kind: Literal["Infrastructure"] = Field("Infrastructure", description="Resource kind")
    metadata: InfraMetadata = Field(..., description="Resource metadata")
    spec: InfrastructureSpecModel = Field(..., description="Infrastructure specification details")
