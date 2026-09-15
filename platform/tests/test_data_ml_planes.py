"""Tests for the data-engineering and ML planes.

Covers:
  * the new Azure resource types (ADF, AML, Event Hubs, Synapse) on the
    infrastructure schema,
  * the data-plane and ML-plane specs,
  * ComponentGraph validation (the end-to-end declarative spec),
  * the scaffold template rendering and validating,
  * the engineering package's lazy exports.

No Azure credentials or SDKs are required — every client degrades to a logged
no-op when `azure-ai-ml` / `azure-mgmt-datafactory` are absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

TEMPLATE = REPO_ROOT / "platform" / "scaffold" / "templates" / "component-graph" / "component_graph.yaml"


# ── infrastructure schema: new resource types ────────────────────────────

def test_infrastructure_accepts_adf_and_aml_resources() -> None:
    from platform.schema.infrastructure_spec import InfrastructureSpec

    spec = InfrastructureSpec.model_validate(
        {
            "apiVersion": "agentcomponents/v1",
            "kind": "Infrastructure",
            "metadata": {"name": "data-ml-infra"},
            "spec": {
                "resources": [
                    {"type": "azure/data-factory", "name": "adf-main", "managed_vnet": True},
                    {"type": "azure/ml-workspace", "name": "aml-main", "key_vault": "kv1"},
                    {"type": "azure/event-hubs", "name": "eh-stream", "capacity": 2},
                    {"type": "azure/synapse-workspace", "name": "syn-main"},
                ]
            },
        }
    )
    types = [r.type for r in spec.spec.resources]
    assert types == [
        "azure/data-factory",
        "azure/ml-workspace",
        "azure/event-hubs",
        "azure/synapse-workspace",
    ]
    adf, aml, eh, _syn = spec.spec.resources
    assert getattr(adf, "managed_vnet", None) is True
    assert getattr(aml, "key_vault", None) == "kv1"
    assert getattr(eh, "capacity", None) == 2


def test_infrastructure_rejects_unknown_resource_type() -> None:
    from pydantic import ValidationError

    from platform.schema.infrastructure_spec import InfrastructureSpec

    with pytest.raises(ValidationError):
        InfrastructureSpec.model_validate(
            {
                "apiVersion": "agentcomponents/v1",
                "kind": "Infrastructure",
                "metadata": {"name": "bad"},
                "spec": {"resources": [{"type": "azure/not-a-thing", "name": "x"}]},
            }
        )


# ── data plane ───────────────────────────────────────────────────────────

def test_batch_pipeline_requires_source_and_sink() -> None:
    from pydantic import ValidationError

    from platform.engineering.data.models import BatchPipelineSpec

    good = BatchPipelineSpec.model_validate(
        {
            "name": "ingest",
            "source": {
                "name": "raw", "source_type": "azure-blob",
                "path_or_table": "/landing/", "format": "json",
            },
            "sink": {
                "name": "bronze", "source_type": "adls-gen2",
                "path_or_table": "/bronze/", "format": "delta",
            },
        }
    )
    assert good.pipeline_type == "batch"

    with pytest.raises(ValidationError):
        BatchPipelineSpec.model_validate({"name": "ingest"})  # missing source/sink


def test_quality_gate_quarantine_sink_is_a_dataset_not_a_string() -> None:
    """Regression: the scaffold template once passed a bare path string here."""
    from pydantic import ValidationError

    from platform.engineering.data.models import DataQualitySpec

    gate = DataQualitySpec.model_validate(
        {
            "dataset_name": "silver_sales",
            "checks": [{"column": "id", "check_type": "unique_check", "threshold": 1.0}],
            "quarantine_sink": {
                "name": "quarantine_sales",
                "source_type": "adls-gen2",
                "path_or_table": "/quarantine/sales/",
                "format": "delta",
            },
        }
    )
    assert gate.quarantine_sink is not None
    assert gate.quarantine_sink.name == "quarantine_sales"

    with pytest.raises(ValidationError):
        DataQualitySpec.model_validate(
            {"dataset_name": "x", "quarantine_sink": "/quarantine/sales/"}
        )


def test_medallion_and_vector_rag_specs_validate() -> None:
    from platform.engineering.data.models import MedallionPipelineSpec, VectorRAGPipelineSpec

    medallion = MedallionPipelineSpec.model_validate(
        {
            "name": "sales_medallion",
            "raw_landing_path": "/landing/", "bronze_path": "/bronze/",
            "silver_path": "/silver/", "gold_path": "/gold/",
            "format": "delta", "deduplication_keys": ["transaction_id"],
        }
    )
    assert medallion.name == "sales_medallion"

    rag = VectorRAGPipelineSpec.model_validate(
        {
            "name": "kb_rag",
            "document_source_path": "/docs/",
            "chunking_strategy": "semantic",
            "chunk_size": 512, "chunk_overlap": 50,
            "embedding_model": "text-embedding-3-large",
            "embedding_dimensions": 3072,
            "ai_search_index_name": "kb-index",
            "hybrid_search": True,
        }
    )
    assert rag.ai_search_index_name == "kb-index"


# ── ML plane ─────────────────────────────────────────────────────────────

def test_aml_compute_and_job_specs_validate() -> None:
    from platform.engineering.aml.models import (
        AMLCommandJobSpec,
        AMLComputeSpec,
        AMLModelSpec,
    )

    compute = AMLComputeSpec.model_validate(
        {"name": "cpu-cluster", "compute_type": "amlcompute", "vm_size": "Standard_DS3_v2"}
    )
    assert compute.min_instances == 0 and compute.max_instances == 4

    job = AMLCommandJobSpec.model_validate(
        {"name": "train", "command": "python train.py", "code_path": "./src/train.py"}
    )
    assert job.command.startswith("python")

    model = AMLModelSpec.model_validate({"name": "forecaster", "path": "./model"})
    assert model.model_format == "mlflow"


def test_aml_endpoint_defaults_to_online() -> None:
    from platform.engineering.aml.models import AMLEndpointSpec

    ep = AMLEndpointSpec.model_validate({"name": "ep-forecaster"})
    assert ep.endpoint_type == "online"
    assert ep.auth_mode == "key"


# ── component graph ──────────────────────────────────────────────────────

def test_component_graph_spec_defaults() -> None:
    from platform.schema.component_graph import ComponentGraphSpec

    spec = ComponentGraphSpec.model_validate(
        {
            "agent_plane": {
                "pattern": "react",
                "model": {"provider": "azure-foundry", "deployment_name": "gpt-4o-mini"},
            }
        }
    )
    # Planes default to empty but enabled, and the runtime chassis is filled in.
    assert spec.data_plane.auto_generate_agent_tools is True
    assert spec.ml_plane.auto_generate_agent_tools is True
    assert spec.runtime_chassis.provider == "azure-foundry"
    assert spec.runtime_chassis.scale_to_zero is True
    assert spec.infrastructure_resources == []


def test_load_spec_dispatches_component_graph_by_kind() -> None:
    from platform.schema.component_graph import ComponentGraph
    from platform.schema.loader import validate_spec

    validated = validate_spec(
        {
            "apiVersion": "agentcomponents/v1",
            "kind": "ComponentGraph",
            "metadata": {"name": "g"},
            "spec": {
                "agent_plane": {
                    "pattern": "react",
                    "model": {"provider": "azure-foundry", "deployment_name": "gpt-4o-mini"},
                }
            },
        }
    )
    assert isinstance(validated, ComponentGraph)


def test_load_spec_rejects_unknown_kind() -> None:
    from platform.schema.loader import validate_spec

    with pytest.raises(ValueError, match="Unknown or missing kind"):
        validate_spec({"kind": "Nonsense", "metadata": {"name": "x"}, "spec": {}})


# ── scaffold template ────────────────────────────────────────────────────

def test_component_graph_template_renders_and_validates() -> None:
    """The scaffold template is Jinja2, so render it before validating."""
    from jinja2 import Template

    from platform.engineering.data.models import BatchPipelineSpec
    from platform.schema.component_graph import ComponentGraph

    assert TEMPLATE.is_file(), f"missing template at {TEMPLATE}"

    rendered = Template(TEMPLATE.read_text(encoding="utf-8")).render(project_name="demo")
    data = yaml.safe_load(rendered)
    assert data["kind"] == "ComponentGraph"

    graph = ComponentGraph.model_validate(data)
    spec = graph.spec

    # All four planes populated.
    assert len(spec.data_plane.batch_pipelines) >= 1
    assert len(spec.data_plane.medallion_pipelines) >= 1
    assert len(spec.data_plane.rag_pipelines) >= 1
    assert len(spec.data_plane.quality_gates) >= 1
    assert len(spec.ml_plane.command_jobs) >= 1
    assert len(spec.ml_plane.models) >= 1
    assert len(spec.ml_plane.endpoints) >= 1

    # The quarantine sink must be a DatasetSpec, not a bare path string.
    gate = spec.data_plane.quality_gates[0]
    assert gate.quarantine_sink is not None
    assert isinstance(gate.quarantine_sink.name, str)

    # Infrastructure covers both data and ML services.
    infra_types = {r.type for r in spec.infrastructure_resources}
    assert "azure/data-factory" in infra_types
    assert "azure/ml-workspace" in infra_types


def test_component_graph_template_batch_pipeline_shape() -> None:
    """Batch pipelines in the template must use DatasetSpec for source and sink."""
    from jinja2 import Template

    from platform.engineering.data.models import BatchPipelineSpec

    rendered = Template(TEMPLATE.read_text(encoding="utf-8")).render(project_name="demo")
    data = yaml.safe_load(rendered)

    for raw in data["spec"]["data_plane"]["batch_pipelines"]:
        pipe = BatchPipelineSpec.model_validate(raw)
        assert pipe.source.name and pipe.sink.name


# ── engineering package exports ──────────────────────────────────────────

def test_engineering_package_exports_and_lazy_graph() -> None:
    import platform.engineering as eng

    for name in (
        "LoopController", "EvalHarness", "DataInfraManager", "AMLInfraManager",
        "AIInfraManager", "ADFPipelineTool", "VectorSearchTool",
        "AMLModelInferenceTool",
    ):
        assert hasattr(eng, name), f"engineering should export {name}"

    # Lazily resolved (breaks the schema<->engineering import cycle).
    assert eng.ComponentGraphOrchestrator is not None
    assert eng.ComponentGraphExecutionResult is not None


def test_engineering_lazy_attr_raises_for_unknown_name() -> None:
    import platform.engineering as eng

    with pytest.raises(AttributeError):
        _ = eng.definitely_not_an_export


def test_data_infra_manager_collision_is_disambiguated() -> None:
    """`DataInfraManager` now comes from data.manager; the legacy one is kept."""
    import platform.engineering as eng
    from platform.engineering.data.manager import DataInfraManager as NewManager
    from platform.engineering.data_infra import DataInfraManager as OldManager

    assert eng.DataInfraManager is NewManager
    assert eng.LegacyDataInfraManager is OldManager
