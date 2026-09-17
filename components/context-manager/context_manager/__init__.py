"""context-manager: budgeted assembly of a model's context window.

    segments ──▶ assemble ──▶ ContextPackage(messages, outcomes, accounting)
                 price / dedupe / evict / compact

The component answers one question — *what goes in the window this call* — and
records its reasoning so the answer can be audited afterwards.

It also owns **checkpointing** (``checkpoint.py``): durable, resumable run
state. The two concerns share a home because both are about recording what
happened across a boundary — one across the model's window, the other across a
process restart — and both must distinguish *absent* from *empty*.
"""
from context_manager.assembler import ContextAssembler, SingleShotAssembler
from context_manager.checkpoint import (
    BaseCheckpointer,
    CheckpointCapability,
    CheckpointedRun,
    Checkpointer,
    CheckpointStatus,
    CosmosCheckpointer,
    DegradedCheckpointer,
    FileCheckpointer,
    InMemoryCheckpointer,
    RedisCheckpointer,
    ResumePlan,
    RunCheckpoint,
    StepOutcome,
    build_checkpointer,
    plan_resume,
)
from context_manager.compactors import (
    BaseCompactor,
    ExtractiveCompactor,
    HybridCompactor,
    SummarizeCompactor,
    TruncateCompactor,
    build_compactor,
)
from context_manager.models import (
    CompactionOutcome,
    ContextBudget,
    ContextMessage,
    ContextPackage,
    ContextRole,
    ContextSegment,
    Disposition,
    DropReason,
    SegmentOutcome,
    SegmentPriority,
)
from context_manager.tokenizer import (
    CachingCounter,
    CallableCounter,
    HeuristicCounter,
    TokenCounter,
    build_counter,
)

__version__ = "0.1.0"
__all__ = [
    # assembler
    "ContextAssembler",
    "SingleShotAssembler",
    # models
    "ContextBudget",
    "ContextMessage",
    "ContextPackage",
    "ContextRole",
    "ContextSegment",
    "SegmentPriority",
    "SegmentOutcome",
    "Disposition",
    "DropReason",
    "CompactionOutcome",
    # compactors
    "BaseCompactor",
    "TruncateCompactor",
    "ExtractiveCompactor",
    "SummarizeCompactor",
    "HybridCompactor",
    "build_compactor",
    # tokenizers
    "TokenCounter",
    "HeuristicCounter",
    "CallableCounter",
    "CachingCounter",
    "build_counter",
    # checkpointing
    "Checkpointer",
    "BaseCheckpointer",
    "InMemoryCheckpointer",
    "FileCheckpointer",
    "RedisCheckpointer",
    "CosmosCheckpointer",
    "DegradedCheckpointer",
    "build_checkpointer",
    "RunCheckpoint",
    "CheckpointStatus",
    "CheckpointCapability",
    "StepOutcome",
    "ResumePlan",
    "plan_resume",
    "CheckpointedRun",
]
