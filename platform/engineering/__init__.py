"""Engineering infrastructure for the platform.

Two things are careful here:

1. **Import cycles.** ``platform.schema.component_graph`` imports the data/AML
   models, which imports this package, whose ``graph_orchestrator`` module
   imports ``platform.schema.component_graph`` back. Importing
   ``ComponentGraphOrchestrator`` eagerly therefore breaks
   ``import platform.schema``. It is exposed lazily via ``__getattr__``
   (PEP 562) so the cycle is never entered at import time.

2. **Name collision.** ``DataInfraManager`` exists in both the older
   ``data_infra`` module and the newer ``data.manager`` package. The newer one
   wins, and the legacy class is re-exported as ``LegacyDataInfraManager`` so
   neither is lost.
"""
from __future__ import annotations

from typing import Any

from platform.engineering.loop import LoopController, LoopConfig, LoopTelemetry, LoopGuardrail
from platform.engineering.harness import EvalHarness, EvalSpec, EvalResult
from platform.engineering.data_infra import (
    DataInfraManager as LegacyDataInfraManager,
    DataPipelineSpec,
    DataSourceConfig,
    DataSinkConfig,
)
from platform.engineering.ai_infra import AIInfraManager, ModelDeploymentSpec, FineTuneSpec, ABTestSpec

# The newer data/ and aml/ packages own the current managers.
from platform.engineering.data.manager import DataInfraManager
from platform.engineering.aml.manager import AMLInfraManager

from platform.engineering.data.tools import (
    ADFPipelineTool,
    VectorSearchTool,
    LakehouseQueryTool,
    AMLModelInferenceTool,
    DataQualityInspectionTool,
)

# Lazily loaded — see module docstring (import cycle with platform.schema).
_LAZY = {
    "ComponentGraphOrchestrator": ("platform.engineering.graph_orchestrator", "ComponentGraphOrchestrator"),
    "ComponentGraphExecutionResult": ("platform.engineering.graph_orchestrator", "ComponentGraphExecutionResult"),
}

__all__ = [
    "LoopController",
    "LoopConfig",
    "LoopTelemetry",
    "LoopGuardrail",
    "EvalHarness",
    "EvalSpec",
    "EvalResult",
    "DataInfraManager",
    "LegacyDataInfraManager",
    "DataPipelineSpec",
    "DataSourceConfig",
    "DataSinkConfig",
    "AIInfraManager",
    "AMLInfraManager",
    "ModelDeploymentSpec",
    "FineTuneSpec",
    "ABTestSpec",
    "ComponentGraphOrchestrator",
    "ComponentGraphExecutionResult",
    "ADFPipelineTool",
    "VectorSearchTool",
    "LakehouseQueryTool",
    "AMLModelInferenceTool",
    "DataQualityInspectionTool",
]


def __getattr__(name: str) -> Any:
    """Resolve the lazily-exported graph orchestrator symbols (PEP 562)."""
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_path, attr = target
    import importlib

    module = importlib.import_module(module_path)
    value = getattr(module, attr)
    globals()[name] = value  # cache so subsequent lookups are free
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
