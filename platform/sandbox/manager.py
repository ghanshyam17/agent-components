from __future__ import annotations

import logging
from typing import Dict, Optional

from .base import ExecutionResult, SandboxBackend, SandboxConfig

logger = logging.getLogger(__name__)


class SandboxManager:
    """Routes execution requests to the appropriate sandbox backend based on task type."""
    
    def __init__(self) -> None:
        self._backends: Dict[str, SandboxBackend] = {}

    def register_backend(self, task_type: str, backend: SandboxBackend) -> None:
        """Register a custom backend for a specific task type."""
        self._backends[task_type] = backend
        logger.info(f"Registered backend '{backend.name}' for task type '{task_type}'")

    def get_backend(self, task_type: str) -> SandboxBackend:
        """Return the configured backend for the given task type."""
        if task_type in self._backends:
            return self._backends[task_type]
        logger.warning(f"No specific backend found for '{task_type}'. Falling back to default.")
        
        # Try to return general or any available
        if "general" in self._backends:
            return self._backends["general"]
        if self._backends:
            return next(iter(self._backends.values()))
            
        raise RuntimeError("No sandbox backends are registered.")

    def detect_task_type(self, code: str) -> str:
        """Heuristic detection of task type from code imports and content."""
        code_lower = code.lower()
        if any(pkg in code_lower for pkg in ["sklearn", "torch", "tensorflow"]):
            return "data_science"
        if any(pkg in code_lower for pkg in ["pandas", "matplotlib", "seaborn"]):
            return "data_analysis"
        if any(pkg in code_lower for pkg in ["requests", "selenium", "subprocess"]):
            return "automation"
        return "general"

    async def execute(self, code: str, task_type: Optional[str] = None, config: Optional[SandboxConfig] = None) -> ExecutionResult:
        """Execute code by routing to the appropriate backend."""
        detected_type = task_type or self.detect_task_type(code)
        backend = self.get_backend(detected_type)
        
        if config:
            await backend.initialize(config)
            
        logger.info(f"Executing code via backend '{backend.name}' (Task type: {detected_type})")
        return await backend.execute(code)
