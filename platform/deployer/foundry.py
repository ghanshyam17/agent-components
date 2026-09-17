"""Azure AI Foundry Agent Service deployer.

Deploys a hosted agent from a declarative ``AgentSpec`` (or a whole
``ProjectSpec``) by delegating to ``azure/deploy_hosted_agent.py`` — the single
implementation of the packaging + provisioning flow. Keeping one code path means
the CLI and that script can never drift apart in what they ship.

The heavy lifting (vendoring the monorepo packages into ``_vendor/``, resolving
the endpoint/model from Terraform outputs, ``create_version_from_code``,
waiting for the version to go active, then routing traffic to it) lives in the
script; this class adapts an ``AgentSpec`` onto it and reports a
:class:`DeploymentResult`.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field

from platform.schema.agent_spec import AgentSpec

logger = logging.getLogger(__name__)

#: Any declarative spec that carries `.metadata.name` — an AgentSpec, a
#: ComponentGraph, a ProjectSpec. The deployer only reads the name for logging;
#: the actual packaging/provisioning flow is driven by the deploy script.
DeployableSpec = Any

# `azure-*` is an optional extra: importing this module must succeed without it
# so the CLI's `--help` and `plan` paths still work on a bare machine.
try:  # pragma: no cover - presence depends on the environment
    from azure.identity import DefaultAzureCredential
    from azure.ai.projects import AIProjectClient

    _AZURE_AVAILABLE = True
except ImportError:  # pragma: no cover
    _AZURE_AVAILABLE = False


class DeploymentResult(BaseModel):
    success: bool = Field(False)
    endpoints: Dict[str, str] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)
    duration_ms: int = Field(0)
    version: str | None = Field(None)
    agent_name: str | None = Field(None)
    plan: dict[str, Any] | None = Field(None)


def _repo_root() -> Path:
    """The agent-components repo root (this file is at platform/deployer/)."""
    return Path(__file__).resolve().parents[2]


def _deploy_script() -> Path:
    return _repo_root() / "azure" / "deploy_hosted_agent.py"


def load_deploy_script() -> Any:
    """Import ``azure/deploy_hosted_agent.py`` as a module.

    It is a script rather than a package module, so it is loaded by path.
    """
    path = _deploy_script()
    if not path.is_file():
        raise FileNotFoundError(f"deployment script not found: {path}")
    spec = importlib.util.spec_from_file_location("_agcomps_deploy", path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise ImportError(f"cannot load deployment script: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_agcomps_deploy"] = module
    spec.loader.exec_module(module)
    return module


def _spec_name(spec: Any) -> str | None:
    """Best-effort `.metadata.name` from any declarative spec."""
    meta = getattr(spec, "metadata", None)
    return getattr(meta, "name", None) if meta is not None else None


class FoundryDeployer:
    """Azure AI Foundry Agent Service deployer.

    Parameters
    ----------
    project_config:
        Optional dict supplying ``foundry_endpoint``, ``model``,
        ``agent_name`` and/or ``deploy_target``. Explicit arguments win;
        environment and Terraform outputs fill the rest.
    """

    def __init__(self, project_config: dict[str, Any] | None = None) -> None:
        self.project_config = project_config or {}

    # -- packaging ---------------------------------------------------------- #
    def create_code_zip(self, source_dir: Path | None = None) -> Path:
        """Build the deployment bundle via the shared deployment script."""
        module = load_deploy_script()
        return module.create_code_zip(source_dir or module.SRC)

    # -- helpers ------------------------------------------------------------ #
    def _resolve(self) -> tuple[str, str, str, dict[str, str]]:
        """Resolve (endpoint, model, target, terraform_outputs) for a deploy."""
        module = load_deploy_script()
        cfg = self.project_config
        target = cfg.get("deploy_target") or os.environ.get(
            "FOUNDRY_DEPLOY_TARGET", "router"
        )
        if target not in module.TARGETS:
            raise ValueError(
                f"unknown deploy target {target!r}; expected one of {sorted(module.TARGETS)}"
            )
        endpoint, model, outputs = module.resolve_target(
            cfg.get("foundry_endpoint"), cfg.get("model")
        )
        return endpoint, model, target, outputs

    def plan(self, agent_spec: DeployableSpec | None = None) -> dict[str, Any]:
        """Report what a deploy would do, without calling Azure.

        Useful for the CLI's ``deploy --dry-run`` path and for tests.
        """
        endpoint, model, target, outputs = self._resolve()
        module = load_deploy_script()
        default_name, description = module.TARGETS[target]
        agent_name = (
            self.project_config.get("agent_name")
            or os.environ.get("FOUNDRY_HOSTED_AGENT_NAME")
            or default_name
        )
        zip_path = self.create_code_zip()
        return {
            "target": target,
            "agent_name": agent_name,
            "endpoint": endpoint,
            "model": model,
            "description": description,
            "terraform_outputs": outputs,
            "spec_name": _spec_name(agent_spec),
            "zip_path": str(zip_path),
            "zip_bytes": zip_path.stat().st_size,
            "vendored": module.vendored_packages(zip_path),
        }

    # -- operations --------------------------------------------------------- #
    async def deploy(
        self,
        agent_spec: DeployableSpec | None = None,
        project_config: dict | None = None,
    ) -> DeploymentResult:
        """Package and deploy the hosted agent, waiting for it to go active.

        Delegates to ``azure/deploy_hosted_agent.py`` as a subprocess so the
        CLI and the script share exactly one implementation and the Azure SDK
        lifecycle is managed in one place.
        """
        start = time.time()
        if project_config:
            self.project_config = {**self.project_config, **project_config}

        try:
            endpoint, model, target, outputs = self._resolve()
        except Exception as e:  # noqa: BLE001 - report, don't raise, on plan errors
            logger.error("deployment configuration invalid: %s", e)
            return DeploymentResult(
                success=False,
                errors=[str(e)],
                duration_ms=int((time.time() - start) * 1000),
            )

        module = load_deploy_script()
        agent_name = (
            self.project_config.get("agent_name")
            or os.environ.get("FOUNDRY_HOSTED_AGENT_NAME")
            or module.TARGETS[target][0]
        )
        logger.info(
            "deploying %s to Foundry (target=%s endpoint=%s model=%s)",
            agent_name, target, endpoint, model,
        )

        cmd = [
            sys.executable,
            str(_deploy_script()),
            "--target", target,
            "--endpoint", endpoint,
            "--model", model,
            "--agent-name", agent_name,
            "--wait",
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
        duration = int((time.time() - start) * 1000)
        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-15:]
            return DeploymentResult(
                success=False,
                errors=["deployment failed: " + " | ".join(tail)],
                duration_ms=duration,
                agent_name=agent_name,
            )

        version = _parse_version(proc.stdout)
        return DeploymentResult(
            success=True,
            endpoints={
                "agent": agent_name,
                "project": endpoint,
                "data_plane": _openai_base(endpoint),
            },
            duration_ms=duration,
            version=version,
            agent_name=agent_name,
        )

    async def update(self, agent_name: str, version: str, traffic_pct: int) -> DeploymentResult:
        """Route `traffic_pct` of traffic to `version` of `agent_name`."""
        start = time.time()
        if not _AZURE_AVAILABLE:
            return DeploymentResult(
                success=False,
                errors=["azure-ai-projects and azure-identity are required"],
                duration_ms=int((time.time() - start) * 1000),
            )
        endpoint, _model, _target, _outputs = self._resolve()
        module = load_deploy_script()
        try:
            with DefaultAzureCredential() as cred, AIProjectClient(
                endpoint=endpoint, credential=cred
            ) as client:
                client.agents.update_details(
                    agent_name=agent_name,
                    agent_endpoint=module.AgentEndpointConfig(
                        version_selector=module.VersionSelector(
                            version_selection_rules=[
                                module.FixedRatioVersionSelectionRule(
                                    agent_version=version, traffic_percentage=traffic_pct
                                )
                            ]
                        )
                    ),
                )
        except Exception as e:  # noqa: BLE001
            logger.error("updating %s failed: %s", agent_name, e)
            return DeploymentResult(
                success=False,
                errors=[f"{type(e).__name__}: {e}"],
                duration_ms=int((time.time() - start) * 1000),
                agent_name=agent_name,
            )
        return DeploymentResult(
            success=True,
            endpoints={"agent": agent_name, "project": endpoint},
            duration_ms=int((time.time() - start) * 1000),
            version=version,
            agent_name=agent_name,
        )

    async def delete(self, agent_name: str) -> DeploymentResult:
        """Delete every version of a hosted agent."""
        start = time.time()
        if not _AZURE_AVAILABLE:
            return DeploymentResult(
                success=False,
                errors=["azure-ai-projects and azure-identity are required"],
                duration_ms=int((time.time() - start) * 1000),
            )
        endpoint, _model, _target, _outputs = self._resolve()
        try:
            with DefaultAzureCredential() as cred, AIProjectClient(
                endpoint=endpoint, credential=cred
            ) as client:
                client.agents.delete(agent_name=agent_name)
        except Exception as e:  # noqa: BLE001
            logger.error("deleting %s failed: %s", agent_name, e)
            return DeploymentResult(
                success=False,
                errors=[f"{type(e).__name__}: {e}"],
                duration_ms=int((time.time() - start) * 1000),
                agent_name=agent_name,
            )
        return DeploymentResult(
            success=True,
            duration_ms=int((time.time() - start) * 1000),
            agent_name=agent_name,
        )

    async def get_status(self, agent_name: str) -> dict:
        """Return the deployment status of `agent_name`."""
        endpoint, _model, _target, _outputs = self._resolve()
        if not _AZURE_AVAILABLE:
            return {
                "agent": agent_name,
                "status": "unknown",
                "error": "azure-ai-projects and azure-identity are required",
                "project": endpoint,
            }
        try:
            with DefaultAzureCredential() as cred, AIProjectClient(
                endpoint=endpoint, credential=cred
            ) as client:
                versions = list(client.agents.list_versions(agent_name=agent_name))
        except Exception as e:  # noqa: BLE001
            return {
                "agent": agent_name,
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "project": endpoint,
            }
        latest = versions[0] if versions else None
        return {
            "agent": agent_name,
            "status": getattr(latest, "status", "not-found") if latest else "not-found",
            "versions": len(versions),
            "latest_version": getattr(latest, "version", None) if latest else None,
            "project": endpoint,
        }


def _openai_base(endpoint: str) -> str:
    """Derive the OpenAI-compatible base URL from a project endpoint."""
    host = endpoint.rstrip("/").split("/api/projects/")[0]
    return f"{host}/openai/v1"


def _parse_version(stdout: str) -> str | None:
    """Pull the created version out of the deploy script's output."""
    for line in reversed(stdout.splitlines()):
        line = line.strip()
        if line.startswith("created version"):
            return line.removeprefix("created version").strip()
        if line.startswith("endpoint routed to"):
            return line.removeprefix("endpoint routed to").strip()
    return None
