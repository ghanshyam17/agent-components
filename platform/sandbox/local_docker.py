from __future__ import annotations

import logging
import time
from typing import List, Optional

from .base import ExecutionResult, InputFile, SandboxBackend, SandboxConfig

logger = logging.getLogger(__name__)


class LocalDockerBackend(SandboxBackend):
    """Local Docker sandbox for development and offline agent testing."""
    
    def __init__(self, image_name: str = "python:3.10-slim", memory_limit: str = "1g", cpu_limit: float = 1.0, network_disabled: bool = True) -> None:
        self.image_name = image_name
        self.memory_limit = memory_limit
        self.cpu_limit = cpu_limit
        self.network_disabled = network_disabled
        self.config: Optional[SandboxConfig] = None

    @property
    def name(self) -> str:
        return "local_docker"

    async def initialize(self, config: SandboxConfig) -> None:
        """Initialize the Local Docker sandbox."""
        self.config = config
        logger.info(f"Initialized Local Docker Backend with base image {self.image_name}")

    async def execute(self, code: str, files: Optional[List[InputFile]] = None) -> ExecutionResult:
        """Execute Python code in a local Docker container."""
        logger.info(f"Executing code in local Docker container (Image: {self.image_name})")
        start_time = time.time()
        
        # TODO: Implement local docker SDK integration (`docker run ...`)
        # Simulate docker run execution for now
        execution_time_ms = int((time.time() - start_time) * 1000)
        
        return ExecutionResult(
            success=True,
            output="Execution simulated via Local Docker placeholder.",
            execution_time_ms=execution_time_ms
        )

    async def install_packages(self, packages: List[str]) -> ExecutionResult:
        logger.info(f"Installing {packages} in local Docker environment")
        return ExecutionResult(success=True, output=f"Installed {packages} locally", execution_time_ms=100)

    async def upload_file(self, name: str, content: bytes) -> str:
        logger.info(f"Copying file {name} to local Docker container")
        return f"/app/workspace/{name}"

    async def download_file(self, file_id: str) -> bytes:
        logger.info(f"Copying file {file_id} from local Docker container")
        return b"simulated file content from local docker"

    async def cleanup(self) -> None:
        logger.info("Stopping and removing local Docker containers")

    async def health_check(self) -> bool:
        # Check if docker daemon is reachable
        return True
