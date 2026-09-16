"""Data models for the distillation component.

The pipeline this models has four stages, each with its own type:

    synthesise ──▶ curate ──▶ train ──▶ evaluate
    ContentSample  QualityScore  JobStatus  ParityReport

`ContentSample` carries both the final `completion` and the `teacher_reasoning`
chain-of-thought that produced it. Keeping them separate matters: reasoning
traces are what make distillation effective on small models, but they are also
where teacher hallucinations hide, so the curator scores them independently.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class TrainingMethod(str, Enum):
    """Which fine-tuning strategy the job will use."""

    SFT = "sft"      # supervised fine-tuning on (prompt, completion)
    DPO = "dpo"      # direct preference optimisation on (prompt, chosen, rejected)
    LORA = "lora"    # SFT with low-rank adapters — cheapest, the usual default


class DatasetFormat(str, Enum):
    """Serialisation format for a curated dataset."""

    ALPACA = "alpaca"    # {"instruction", "input", "output"}
    CHATML = "chatml"    # {"messages": [{role, content}, ...]}
    DPO = "dpo"          # {"prompt", "chosen", "rejected"}
    JSONL = "jsonl"      # raw ContentSample records


class JobState(str, Enum):
    """Lifecycle of a distillation job.

    Deliberately mirrors what Azure AI Foundry reports so a real job and a mock
    job are indistinguishable to callers.
    """

    PENDING = "pending"
    SYNTHESIZING = "synthesizing"
    CURATING = "curating"
    TRAINING = "training"
    COMPLETED = "completed"
    FAILED = "failed"

    @property
    def terminal(self) -> bool:
        return self in (JobState.COMPLETED, JobState.FAILED)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
class LoRAConfig(BaseModel):
    """Low-rank adapter settings. Only used when method is ``lora``."""

    r: int = Field(default=16, ge=1, description="Rank of the update matrices.")
    alpha: int = Field(default=32, ge=1, description="LoRA scaling factor.")
    dropout: float = Field(default=0.05, ge=0.0, le=1.0)
    target_modules: list[str] = Field(
        default_factory=lambda: ["q_proj", "v_proj"],
        description="Attention projections to adapt.",
    )

    @model_validator(mode="after")
    def _alpha_should_scale_with_r(self) -> "LoRAConfig":
        # alpha/r is the effective update magnitude. A wildly mismatched pair is
        # almost always a configuration mistake, and it silently ruins training.
        ratio = self.alpha / self.r
        if ratio > 8:
            raise ValueError(
                f"LoRA alpha/r = {ratio:.1f} is far above the usual 1-4 range; "
                "check alpha and r (a common default is alpha = 2*r)."
            )
        return self


class DistillationConfig(BaseModel):
    """Everything needed to run a distillation job.

    ``teacher_model`` is an alias resolved by ``model-gateway`` (e.g. ``"higher"``)
    or a concrete model name; keeping it an alias means the teacher can be
    swapped between a local vLLM replica and a cloud endpoint without touching
    the pipeline.
    """

    teacher_model: str = Field(..., description="Teacher alias or model name.")
    student_base_model: str = Field(
        default="microsoft/Phi-4-mini-instruct",
        description="Base model to fine-tune into the student.",
    )
    method: TrainingMethod = TrainingMethod.LORA
    format: DatasetFormat = DatasetFormat.CHATML
    # --- data ---
    num_samples: int = Field(default=50, ge=1, description="Target synthetic examples.")
    dataset_path: str | None = Field(None, description="Where curated data is written.")
    seed_prompts: list[str] = Field(default_factory=list)
    # --- quality gates ---
    min_quality: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Overall score below which a sample is quarantined.",
    )
    dedupe_threshold: float = Field(
        default=0.95, ge=0.0, le=1.0,
        description="Cosine similarity above which a sample is a near-duplicate.",
    )
    # --- training ---
    epochs: int = Field(default=3, ge=1)
    learning_rate: float = Field(default=2e-4, gt=0)
    batch_size: int = Field(default=4, ge=1)
    max_seq_length: int = Field(default=2048, ge=1)
    lora: LoRAConfig = Field(default_factory=LoRAConfig)
    # --- Azure ---
    foundry_endpoint: str | None = None
    # --- housekeeping ---
    job_name: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _dpo_needs_pairs(self) -> "DistillationConfig":
        if self.method is TrainingMethod.DPO and self.format is not DatasetFormat.DPO:
            raise ValueError(
                "method='dpo' requires format='dpo': preference optimisation "
                "trains on (chosen, rejected) pairs, which the alpaca/chatml "
                "schemas cannot express."
            )
        return self

    @property
    def resolved_job_name(self) -> str:
        return self.job_name or f"distill-{self.student_base_model.split('/')[-1].lower()}"

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["method"] = self.method.value
        d["format"] = self.format.value
        return d


# --------------------------------------------------------------------------- #
# Samples
# --------------------------------------------------------------------------- #
class ContentSample(BaseModel):
    """One synthetic example produced by the teacher.

    ``teacher_reasoning`` is the chain-of-thought; ``completion`` is the answer.
    A sample with no reasoning is still valid (not every task benefits from
    CoT), so it is optional rather than required.
    """

    id: str = Field(default_factory=lambda: _new_id("sample"))
    prompt: str
    completion: str
    teacher_reasoning: str | None = None
    format_type: DatasetFormat = DatasetFormat.CHATML
    teacher_model: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Everything a deduplicator should compare: prompt plus answer."""
        return f"{self.prompt}\n{self.completion}"

    def to_alpaca(self) -> dict[str, Any]:
        return {"instruction": self.prompt, "input": "", "output": self.completion}

    def to_chatml(self, *, system: str | None = None) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": self.prompt})
        messages.append({"role": "assistant", "content": self.completion})
        return {"messages": messages}

    def to_jsonl(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DPOPair(BaseModel):
    """A preference pair for DPO training.

    ``margin`` records how much better the chosen response was judged to be
    (``chosen_score - rejected_score``). It is not used by the DPO loss itself —
    which trains on the pair — but it lets the curator drop pairs that are
    essentially ties, where the preference signal is mostly noise.
    """

    prompt: str
    chosen: str
    rejected: str
    margin: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dpo(self) -> dict[str, Any]:
        return {
            "prompt": self.prompt,
            "chosen": self.chosen,
            "rejected": self.rejected,
        }


# --------------------------------------------------------------------------- #
# Quality
# --------------------------------------------------------------------------- #
class QualityScore(BaseModel):
    """Curator verdict on a single sample.

    The four sub-scores are deliberately kept separate rather than collapsed:
    a sample can be perfectly safe yet worthless (ungrounded), and reporting
    only the mean would hide which gate did the work.
    """

    groundedness: float = Field(default=0.0, ge=0.0, le=1.0)
    coherence: float = Field(default=0.0, ge=0.0, le=1.0)
    fluency: float = Field(default=0.0, ge=0.0, le=1.0)
    safety_passed: bool = True
    overall: float = Field(default=0.0, ge=0.0, le=1.0)
    decision: Literal["pass", "quarantine"] = "pass"
    reasons: list[str] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _safety_caps_the_overall(self) -> "QualityScore":
        # A sample that failed safety can never pass, whatever its other scores.
        if not self.safety_passed and self.decision == "pass":
            self.decision = "quarantine"
            if "safety" not in self.flags:
                self.flags.append("safety")
        return self

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class CurationReport(BaseModel):
    """What the curator did to a batch — the audit trail for a dataset."""

    total_in: int = 0
    passed: int = 0
    quarantined: int = 0
    deduplicated: int = 0
    dpo_pairs: int = 0
    pii_redacted: int = 0
    injections_blocked: int = 0
    quarantine_reasons: dict[str, int] = Field(default_factory=dict)
    duration_ms: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_in if self.total_in else 0.0

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["pass_rate"] = round(self.pass_rate, 4)
        return d


# --------------------------------------------------------------------------- #
# Jobs
# --------------------------------------------------------------------------- #
class DistillationJobStatus(BaseModel):
    """State of a distillation job, live or finished.

    ``artifact_uris`` maps a logical artifact name to where it lives, so a
    completed job is self-describing: callers do not need to know the layout of
    the workspace directory.
    """

    job_id: str = Field(default_factory=lambda: _new_id("job"))
    status: JobState = JobState.PENDING
    teacher_model: str = ""
    student_model: str = ""
    method: TrainingMethod = TrainingMethod.LORA
    config: DistillationConfig | None = None
    metrics: dict[str, float] = Field(default_factory=dict)
    artifact_uris: dict[str, str] = Field(default_factory=dict)
    error: str | None = None
    backend: Literal["foundry", "mock"] = "mock"
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    @property
    def terminal(self) -> bool:
        return self.status.terminal

    def to_dict(self) -> dict[str, Any]:
        d = self.model_dump(mode="json")
        d["status"] = self.status.value
        d["method"] = self.method.value
        return d


# --------------------------------------------------------------------------- #
# Parity
# --------------------------------------------------------------------------- #
class LatencyProfile(BaseModel):
    """Timing for one model: time-to-first-token and end-to-end latency."""

    ttft_ms: float = 0.0
    total_ms: float = 0.0
    samples: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "ttft_ms": round(self.ttft_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "samples": self.samples,
        }


class CostProfile(BaseModel):
    """Token pricing, expressed per 1M tokens (how cloud providers quote)."""

    input_cost_per_mtok: float = 0.0
    output_cost_per_mtok: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0

    def estimate_usd(self) -> float:
        return (
            self.tokens_in / 1_000_000 * self.input_cost_per_mtok
            + self.tokens_out / 1_000_000 * self.output_cost_per_mtok
        )

    def per_mtok_blended(self) -> float:
        """Blended $/1M tokens for this run's actual input:output mix."""
        total = self.tokens_in + self.tokens_out
        if not total:
            return 0.0
        return self.estimate_usd() / (total / 1_000_000)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_cost_per_mtok": self.input_cost_per_mtok,
            "output_cost_per_mtok": self.output_cost_per_mtok,
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "estimate_usd": round(self.estimate_usd(), 6),
            "blended_per_mtok": round(self.per_mtok_blended(), 4),
        }


class ParityReport(BaseModel):
    """Teacher-vs-student comparison — the go/no-go signal for serving.

    The headline numbers a reviewer wants are all derived, not stored:
    ``latency_speedup`` and ``cost_savings_pct`` are properties, so they can
    never contradict the profiles they come from.
    """

    teacher_model: str = ""
    student_model: str = ""
    examples: int = 0
    # quality, 0..1
    teacher_quality: float = 0.0
    student_quality: float = 0.0
    # style
    stylistic_similarity: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Token-overlap agreement between teacher and student answers.",
    )
    readability_grade: float = Field(
        default=0.0, description="Flesch-Kincaid grade of student output."
    )
    teacher_readability_grade: float = 0.0
    # performance
    teacher_latency: LatencyProfile = Field(default_factory=LatencyProfile)
    student_latency: LatencyProfile = Field(default_factory=LatencyProfile)
    teacher_cost: CostProfile = Field(default_factory=CostProfile)
    student_cost: CostProfile = Field(default_factory=CostProfile)
    per_example: list[dict[str, Any]] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def quality_retention(self) -> float:
        """Share of the teacher's quality the student kept."""
        if not self.teacher_quality:
            return 0.0
        return min(1.0, self.student_quality / self.teacher_quality)

    @property
    def quality_delta(self) -> float:
        return self.student_quality - self.teacher_quality

    @property
    def latency_speedup(self) -> float:
        """Teacher TTFT ÷ student TTFT. >1 means the student is faster."""
        if not self.student_latency.ttft_ms:
            return 0.0
        return self.teacher_latency.ttft_ms / self.student_latency.ttft_ms

    @property
    def end_to_end_speedup(self) -> float:
        if not self.student_latency.total_ms:
            return 0.0
        return self.teacher_latency.total_ms / self.student_latency.total_ms

    @property
    def cost_savings_pct(self) -> float:
        """Percent cheaper the student is, on blended $/1M tokens."""
        t = self.teacher_cost.per_mtok_blended()
        s = self.student_cost.per_mtok_blended()
        if not t:
            return 0.0
        return max(-1.0, (t - s) / t) * 100

    def meets_bar(self, *, min_retention: float = 0.9, min_stylistic: float = 0.6) -> bool:
        """Whether the student is good enough to serve.

        Requires BOTH retention and stylistic agreement: a student that scores
        well but does not answer like the teacher has not been distilled, it has
        been replaced with a different model.
        """
        return (
            self.quality_retention >= min_retention
            and self.stylistic_similarity >= min_stylistic
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "teacher_model": self.teacher_model,
            "student_model": self.student_model,
            "examples": self.examples,
            "teacher_quality": round(self.teacher_quality, 4),
            "student_quality": round(self.student_quality, 4),
            "quality_retention": round(self.quality_retention, 4),
            "quality_delta": round(self.quality_delta, 4),
            "stylistic_similarity": round(self.stylistic_similarity, 4),
            "readability_grade": round(self.readability_grade, 2),
            "teacher_readability_grade": round(self.teacher_readability_grade, 2),
            "latency": {
                "teacher": self.teacher_latency.to_dict(),
                "student": self.student_latency.to_dict(),
                "ttft_speedup": round(self.latency_speedup, 3),
                "end_to_end_speedup": round(self.end_to_end_speedup, 3),
            },
            "cost": {
                "teacher": self.teacher_cost.to_dict(),
                "student": self.student_cost.to_dict(),
                "savings_pct": round(self.cost_savings_pct, 2),
            },
            "meets_bar": self.meets_bar(),
            "notes": self.notes,
        }
