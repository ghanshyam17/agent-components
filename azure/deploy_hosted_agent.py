"""Deploy the agent-components hosted agent to Azure AI Foundry Agent Service.

Vendors the monorepo packages the agent needs into ``_vendor/`` inside the
deployment zip:

    components_core, agentic_router, memory_store, model_gateway, platform

``platform`` is included so the hosted agent can bootstrap the
``ComponentGraphOrchestrator`` (Data + ML + Tools + Agents planes) alongside the
``agentic_router`` tiered loop and expose both through the Responses Protocol.

Two deployment targets are available via ``--target``:

``router`` (default)
    The tiered agentic router only (agent name ``agcomps-router``).
``component-graph``
    The full multi-level component graph (``agcomps-component-graph``).

Endpoint and model fall back to the Terraform outputs of ``azure/terraform``
when the corresponding environment variables are not set, so a fresh checkout
can deploy without exporting anything by hand.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

from azure.ai.projects import AIProjectClient
from azure.ai.projects.models import (
    AgentEndpointConfig,
    CodeConfiguration,
    CodeDependencyResolution,
    FixedRatioVersionSelectionRule,
    HostedAgentDefinition,
    ProtocolVersionRecord,
    VersionSelector,
)
from azure.identity import DefaultAzureCredential

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
SRC = (HERE / "hosted-agent" / "src").resolve()
TERRAFORM_DIR = HERE / "terraform"

EXCLUDED = {".git", ".venv", "__pycache__", ".env", ".pytest_cache"}

# --- defaults, used when neither env vars nor Terraform supply a value -------
DEFAULT_ENDPOINT = (
    "https://my-foundry-resource.services.ai.azure.com/api/projects/agent-lab"
)
DEFAULT_MODEL = "gpt-5-mini"

#: package source dir -> name inside `_vendor/`
VENDOR_SOURCES = [
    (REPO_ROOT / "core" / "src" / "components_core", "components_core"),
    (REPO_ROOT / "components" / "agentic-router" / "agentic_router", "agentic_router"),
    (REPO_ROOT / "components" / "memory-store" / "memory_store", "memory_store"),
    (REPO_ROOT / "components" / "model-gateway" / "model_gateway", "model_gateway"),
    # The platform package (schemas, orchestrators, deployers, patterns) is
    # required by the component-graph target: the hosted agent imports
    # platform.engineering.graph_orchestrator from `_vendor/platform`.
    (REPO_ROOT / "platform", "platform"),
]

#: target name -> (default agent name, description)
TARGETS: dict[str, tuple[str, str]] = {
    "router": (
        "agcomps-router",
        "agent-components: tiered agentic router as a Foundry hosted agent",
    ),
    "component-graph": (
        "agcomps-component-graph",
        "agent-components: multi-level component graph (Data + ML + Tools + Agents)",
    ),
}


# --------------------------------------------------------------------------- #
# Terraform output lookup
# --------------------------------------------------------------------------- #
def terraform_outputs(directory: Path = TERRAFORM_DIR) -> dict[str, str]:
    """Read Terraform outputs for this stack, or {} when unavailable.

    Best-effort by design: a missing binary, an uninitialised state, or a stack
    that was never applied must not block a deployment that passed explicit
    values.
    """
    if not directory.is_dir():
        return {}
    try:
        proc = subprocess.run(
            ["terraform", "output", "-json"],
            cwd=directory,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return {}
    if proc.returncode != 0 or not proc.stdout.strip():
        return {}
    try:
        raw = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {}
    # `terraform output -json` yields {name: {"value": ..., "type": ...}}
    return {
        k: str(v.get("value"))
        for k, v in raw.items()
        if isinstance(v, dict) and v.get("value") is not None
    }


def resolve_target(
    endpoint: str | None, model: str | None
) -> tuple[str, str, dict[str, str]]:
    """Resolve (endpoint, model, terraform_outputs): arg -> env -> tf -> default."""
    outputs = terraform_outputs()

    resolved_endpoint = (
        endpoint
        or os.environ.get("FOUNDRY_PROJECT_ENDPOINT")
        or outputs.get("project_endpoint")
        or DEFAULT_ENDPOINT
    )
    resolved_model = (
        model
        or os.environ.get("FOUNDRY_MODEL_NAME")
        or outputs.get("model_deployment_name")
        or DEFAULT_MODEL
    )
    return resolved_endpoint, resolved_model, outputs


# --------------------------------------------------------------------------- #
# Packaging
# --------------------------------------------------------------------------- #
def create_code_zip(
    source_dir: Path, *, vendor: list[tuple[Path, str]] | None = None
) -> Path:
    """Zip the agent sources plus the vendored monorepo packages.

    ``__pycache__`` and the other EXCLUDED names are skipped everywhere, so a
    local checkout with byte-code caches produces a clean bundle. The component
    graph spec is bundled alongside ``main.py`` so the component-graph target
    can load it without a repo checkout.
    """
    sources = VENDOR_SOURCES if vendor is None else vendor
    zip_path = Path(tempfile.gettempdir()) / "agcomps-hosted-agent.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in source_dir.rglob("*"):
            if not path.is_file():
                continue
            if any(part in EXCLUDED for part in path.parts):
                continue
            zf.write(path, path.relative_to(source_dir))
        graph_spec = REPO_ROOT / "projects" / "component_graph.yaml"
        if graph_spec.is_file():
            zf.write(graph_spec, Path("component_graph.yaml"))
        for src_dir, pkg_name in sources:
            if not src_dir.is_dir():
                raise RuntimeError(f"vendored source missing: {src_dir}")
            for path in src_dir.rglob("*"):
                if not path.is_file():
                    continue
                if any(part in EXCLUDED for part in path.parts):
                    continue
                zf.write(path, Path("_vendor") / pkg_name / path.relative_to(src_dir))
    return zip_path


def zip_manifest(zip_path: Path) -> list[str]:
    """Names inside a deployment zip — used by tests and `--dry-run`."""
    with zipfile.ZipFile(zip_path) as zf:
        return sorted(zf.namelist())


def vendored_packages(zip_path: Path) -> list[str]:
    """Top-level package names bundled under ``_vendor/`` in the zip."""
    return sorted({
        n.split("/")[1]
        for n in zip_manifest(zip_path)
        if n.startswith("_vendor/") and n.count("/") > 1
    })


def wait_for_active(client: AIProjectClient, agent_name: str, version: str) -> None:
    for attempt in range(60):
        time.sleep(10)
        details = client.agents.get_version(agent_name=agent_name, agent_version=version)
        status = details.get("status") if isinstance(details, dict) else getattr(details, "status", None)
        print(f"  provisioning: {status} ({attempt + 1}/60)")
        if status == "active":
            return
        if status == "failed":
            raise RuntimeError(f"provisioning failed: {details}")
    raise RuntimeError("timed out")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Deploy the agent-components hosted agent to Azure AI Foundry."
    )
    ap.add_argument(
        "--target",
        choices=sorted(TARGETS),
        default=os.environ.get("FOUNDRY_DEPLOY_TARGET", "router"),
        help="Which hosted agent to deploy (default: router).",
    )
    ap.add_argument("--endpoint", default=None, help="Foundry project endpoint.")
    ap.add_argument("--model", default=None, help="Model deployment name.")
    ap.add_argument("--agent-name", default=None, help="Override the agent name.")
    ap.add_argument("--wait", action="store_true", help="Wait for the version to go active.")
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and report the deployment bundle without calling Azure.",
    )
    return ap


def main() -> None:
    args = build_parser().parse_args()

    endpoint, model, outputs = resolve_target(args.endpoint, args.model)
    default_name, description = TARGETS[args.target]
    agent_name = (
        args.agent_name or os.environ.get("FOUNDRY_HOSTED_AGENT_NAME") or default_name
    )

    print(f"target      : {args.target}")
    print(f"agent       : {agent_name}")
    print(f"endpoint    : {endpoint}")
    print(f"model       : {model}")
    if outputs:
        print(f"terraform   : {sorted(outputs)}")

    zip_path = create_code_zip(SRC)
    names = zip_manifest(zip_path)
    print(f"code zip    : {zip_path} ({zip_path.stat().st_size} bytes)")
    print(f"vendored    : {', '.join(vendored_packages(zip_path))}")

    if args.dry_run:
        print(f"dry run: bundle built, Azure calls skipped ({len(names)} files)")
        return

    with zip_path.open("rb") as code_stream, DefaultAzureCredential() as cred, \
            AIProjectClient(endpoint=endpoint, credential=cred) as client:
        created = client.agents.create_version_from_code(
            agent_name=agent_name,
            description=description,
            definition=HostedAgentDefinition(
                cpu="0.5",
                memory="1Gi",
                code_configuration=CodeConfiguration(
                    runtime="python_3_13",
                    entry_point=["python", "main.py"],
                    dependency_resolution=CodeDependencyResolution.REMOTE_BUILD,
                ),
                environment_variables={
                    "FOUNDRY_PROJECT_ENDPOINT": endpoint,
                    "AZURE_AI_MODEL_DEPLOYMENT_NAME": model,
                    "PYTHONPATH": ".",
                    # Tells the hosted agent which graph to bootstrap.
                    "AGCOMPS_DEPLOY_TARGET": args.target,
                },
                protocol_versions=[ProtocolVersionRecord(protocol="responses", version="2.0.0")],
            ),
            code=code_stream,
        )
        print(f"created version {created.version}")

        if args.wait:
            wait_for_active(client, agent_name, created.version)

        client.agents.update_details(
            agent_name=agent_name,
            agent_endpoint=AgentEndpointConfig(
                version_selector=VersionSelector(
                    version_selection_rules=[
                        FixedRatioVersionSelectionRule(agent_version=created.version, traffic_percentage=100)
                    ]
                )
            ),
        )
        print("endpoint routed to", created.version)


if __name__ == "__main__":
    main()
