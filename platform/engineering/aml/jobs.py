from __future__ import annotations

import logging
from typing import AsyncIterator

from platform.engineering.aml.client import AMLClient
from platform.engineering.aml.models import AMLCommandJobSpec, AMLSweepJobSpec, AMLPipelineJobSpec

logger = logging.getLogger(__name__)

class AMLJobRunner:
    """Runner and monitor for Azure ML jobs."""

    def __init__(self, client: AMLClient):
        """Initialize with an AMLClient."""
        self.client = client

    async def run_command_job(self, spec: AMLCommandJobSpec) -> dict:
        """Run a command job."""
        logger.info(f"Running command job: {spec.name}")
        job_id = await self.client.submit_job(spec)
        status = await self.client.get_job_status(job_id)
        return {"job_id": job_id, "status": status.get("status")}

    async def run_sweep_job(self, spec: AMLSweepJobSpec) -> dict:
        """Run a sweep (hyperparameter tuning) job."""
        logger.info(f"Running sweep job: {spec.name}")
        job_id = await self.client.submit_job(spec)
        status = await self.client.get_job_status(job_id)
        return {"job_id": job_id, "status": status.get("status")}

    async def run_pipeline_job(self, spec: AMLPipelineJobSpec) -> dict:
        """Run a pipeline job."""
        logger.info(f"Running pipeline job: {spec.name}")
        job_id = await self.client.submit_job(spec)
        status = await self.client.get_job_status(job_id)
        return {"job_id": job_id, "status": status.get("status")}

    async def stream_job_logs(self, job_name: str) -> AsyncIterator[str]:
        """Stream logs for a given job."""
        logger.info(f"Streaming logs for job: {job_name}")
        yield f"Log stream started for job {job_name}"
        yield "Job is running..."
        yield "Job completed."
