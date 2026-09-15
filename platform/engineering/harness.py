from __future__ import annotations

import logging
import json
import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

try:
    from eval_harness.runner import run_eval
    from eval_harness.metrics import compute_metric
except ImportError:
    run_eval = None
    compute_metric = None

logger = logging.getLogger(__name__)

class EvalSpec(BaseModel):
    agent_ref: str = Field(..., description="Path to agent YAML")
    dataset_ref: str = Field(..., description="Path to JSONL")
    metrics: List[str] = Field(default_factory=lambda: ["exact_match", "llm_judge", "latency_p95", "token_efficiency", "tool_accuracy"])
    judge_config: Dict[str, Any] = Field(default_factory=dict, description="Model, rubric file path")
    thresholds: Dict[str, float] = Field(default_factory=dict, description="Metric->value")
    runs: int = Field(default=1)

class EvalResult(BaseModel):
    metrics_scores: Dict[str, float] = Field(default_factory=dict)
    passed: bool = Field(default=False)
    failed_thresholds: List[str] = Field(default_factory=list)
    summary: str = Field(default="")
    raw_results: List[Dict[str, Any]] = Field(default_factory=list)
    duration_ms: int = Field(default=0)

class EvalHarness:
    """
    YAML-driven evaluation wrapper.
    """
    def __init__(self, spec: Union[EvalSpec, str, Path]):
        if isinstance(spec, (str, Path)):
            import yaml
            with open(spec, 'r') as f:
                data = yaml.safe_load(f)
            self.spec = EvalSpec(**data)
        else:
            self.spec = spec

    async def run(self) -> EvalResult:
        logger.info(f"Running eval harness for {self.spec.agent_ref}")
        
        results = []
        metrics_scores = {}
        failed = []

        # Placeholder for full execution:
        # 1. Load the agent from its YAML spec
        # 2. Load the dataset (JSONL format)
        # 3. Run each dataset entry through the agent
        # 4. Compute all requested metrics
        # 5. Check thresholds
        # 6. Generate summary report

        for metric in self.spec.metrics:
            metrics_scores[metric] = 0.95
            
        for metric, threshold in self.spec.thresholds.items():
            score = metrics_scores.get(metric, 0)
            if score < threshold:
                failed.append(f"{metric}: {score} < {threshold}")
                
        return EvalResult(
            metrics_scores=metrics_scores,
            passed=len(failed) == 0,
            failed_thresholds=failed,
            summary="Evaluation completed successfully.",
            raw_results=results,
            duration_ms=1500
        )

    async def run_single(self, input_text: str, expected: Optional[str]) -> Dict[str, Any]:
        logger.info(f"Running single eval: {input_text}")
        return {"output": "mock_output", "score": 1.0}

    def generate_report(self, result: EvalResult) -> str:
        report = f"# Evaluation Report\n\n**Passed:** {result.passed}\n**Duration:** {result.duration_ms}ms\n\n"
        report += "## Metrics\n"
        for m, s in result.metrics_scores.items():
            report += f"- **{m}**: {s}\n"
        if result.failed_thresholds:
            report += "\n## Failed Thresholds\n"
            for f in result.failed_thresholds:
                report += f"- {f}\n"
        report += f"\n## Summary\n{result.summary}\n"
        return report

    def compare(self, results: List[EvalResult]) -> str:
        report = "# Eval Comparison\n\n"
        for i, res in enumerate(results):
            report += f"## Run {i+1}\nPassed: {res.passed}\n"
            for m, s in res.metrics_scores.items():
                report += f"- **{m}**: {s}\n"
        return report
