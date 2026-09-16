"""distillation: synthetic data generation and LLM distillation.

Distils a frontier *teacher* into a small, cheap *student* in four stages:

    synthesise ──▶ curate ──▶ train ──▶ evaluate
    teacher CoT     safety +     Foundry /   parity matrix
    + completions   quality      mock         (go / no-go)
                    + dedup

The component is a **chassis child extension**: it consumes the foundational
components (``model-gateway`` for teacher calls, ``prompt-registry`` for style
blueprints, ``guardrails`` for hygiene, ``eval-harness`` for metrics,
``memory-store`` for dedup, ``retriever`` for grounding) and reimplements none
of them.

Quick start (offline — no Azure, no GPU)::

    from distillation import DistillationConfig, DistillationPipeline

    cfg = DistillationConfig(teacher_model="higher", num_samples=20, method="lora")
    pipe = DistillationPipeline(cfg)          # uses the mock runner
    result = await pipe.run()
    print(result["parity"]["cost"]["savings_pct"], "% cheaper")

With a real teacher over model-gateway::

    from model_gateway import build_gateway
    from distillation import SyntheticContentGenerator, DistillationPipeline

    teacher = build_gateway("gateway.yaml").client("higher")
    pipe = DistillationPipeline(cfg, teacher_client=teacher.chat)
    result = await pipe.run()

See ``README.md`` for the architecture, the Foundry path, and the dataset
formats.
"""
from __future__ import annotations

from distillation.curator import ContentCurator, readability_grade, token_overlap
from distillation.evaluator import ParityEvaluator
from distillation.foundry import FoundryDistillationClient, azure_available
from distillation.models import (
    ContentSample,
    CurationReport,
    DatasetFormat,
    DistillationConfig,
    DistillationJobStatus,
    DPOPair,
    JobState,
    LoRAConfig,
    ParityReport,
    QualityScore,
    TrainingMethod,
)
from distillation.synthesizer import (
    DEFAULT_SEED_PROMPTS,
    STYLE_BLUEPRINTS,
    SyntheticContentGenerator,
)
from distillation.pipeline import DistillationPipeline, DistillationResult

__all__ = [
    # models
    "DistillationConfig", "DistillationJobStatus", "DistillationResult",
    "ContentSample", "DPOPair", "QualityScore", "CurationReport", "ParityReport",
    "TrainingMethod", "DatasetFormat", "JobState", "LoRAConfig",
    # engines
    "SyntheticContentGenerator", "ContentCurator",
    "FoundryDistillationClient", "ParityEvaluator", "DistillationPipeline",
    # helpers
    "azure_available", "readability_grade", "token_overlap",
    "DEFAULT_SEED_PROMPTS", "STYLE_BLUEPRINTS",
]
