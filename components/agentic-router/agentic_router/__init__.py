"""Agentic Router — a vLLM-backed orchestrator that routes tasks between a
lower (small/fast) and higher (large/capable) model and runs an agentic loop
with tools, planning, streaming and session memory."""

from agentic_router.config import Settings

__all__ = ["Settings", "__version__"]
__version__ = "0.1.0"