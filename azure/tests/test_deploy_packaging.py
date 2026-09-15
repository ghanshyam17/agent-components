"""Deployment-bundle tests for the Foundry hosted agent.

The implementation plan calls for two things to be verified without touching
Azure:

* **Monorepo vendoring & packaging** — every package the agent needs (including
  ``platform``) lands under ``_vendor/``, with no ``__pycache__`` or ``.git``.
* **Hosted-agent bootstrap** — ``azure/hosted-agent/src/main.py`` resolves its
  imports and initialises the Responses Protocol handler.

Both run offline: the zip is built on disk and inspected, and the bootstrap is
exercised by importing ``main`` from the repo checkout.
"""
from __future__ import annotations

import importlib.util
import json
import sys
import zipfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
AZURE = REPO_ROOT / "azure"
HOSTED_SRC = AZURE / "hosted-agent" / "src"
DEPLOY = AZURE / "deploy_hosted_agent.py"


# --------------------------------------------------------------------------- #
# Import the deploy script as a module (it is a script, not a package)
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def deploy_mod():
    spec = importlib.util.spec_from_file_location("_agcomps_deploy", DEPLOY)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_agcomps_deploy"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def bundle(deploy_mod) -> Path:
    return deploy_mod.create_code_zip(deploy_mod.SRC)


# --------------------------------------------------------------------------- #
# Vendoring & packaging
# --------------------------------------------------------------------------- #
def test_vendor_sources_include_the_platform_package(deploy_mod):
    """`platform` must be vendored, or the component graph cannot be imported."""
    names = [pkg for _src, pkg in deploy_mod.VENDOR_SOURCES]
    assert "platform" in names
    assert "components_core" in names
    assert "agentic_router" in names
    assert "model_gateway" in names
    assert "memory_store" in names


def test_vendor_source_paths_all_exist(deploy_mod):
    for src_dir, pkg in deploy_mod.VENDOR_SOURCES:
        assert src_dir.is_dir(), f"vendored source missing for {pkg}: {src_dir}"
        assert (src_dir / "__init__.py").is_file(), f"{pkg} has no __init__.py"


def test_bundle_contains_every_vendored_package(bundle, deploy_mod):
    vendored = deploy_mod.vendored_packages(bundle)
    for _src, pkg in deploy_mod.VENDOR_SOURCES:
        assert pkg in vendored, f"{pkg} missing from _vendor/ in the bundle"


def test_bundle_includes_the_agent_entry_point(bundle):
    names = set(zipfile.ZipFile(bundle).namelist())
    assert "main.py" in names


def test_bundle_includes_the_component_graph_spec(bundle):
    names = set(zipfile.ZipFile(bundle).namelist())
    assert "component_graph.yaml" in names


def test_bundle_excludes_pycache_and_git(bundle):
    names = zipfile.ZipFile(bundle).namelist()
    assert not [n for n in names if "__pycache__" in n], "byte-code caches leaked"
    assert not [n for n in names if n.startswith(".git/")], ".git leaked"
    assert not [n for n in names if n.endswith((".pyc", ".pyo"))]


def test_bundle_platform_package_has_its_submodules(bundle):
    """The graph orchestrator and schema loader live in submodules."""
    names = zipfile.ZipFile(bundle).namelist()
    assert "_vendor/platform/__init__.py" in names
    assert "_vendor/platform/schema/__init__.py" in names
    assert any(
        n.startswith("_vendor/platform/engineering/graph_orchestrator") for n in names
    )


def test_bundle_is_a_valid_zip(bundle):
    with zipfile.ZipFile(bundle) as zf:
        assert zf.testzip() is None  # no corrupt members


# --------------------------------------------------------------------------- #
# Terraform output defaults
# --------------------------------------------------------------------------- #
def test_resolve_target_uses_defaults_when_nothing_is_set(deploy_mod, monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_MODEL_NAME", raising=False)
    monkeypatch.setattr(deploy_mod, "terraform_outputs", lambda *a, **k: {})
    endpoint, model, outputs = deploy_mod.resolve_target(None, None)
    assert endpoint == deploy_mod.DEFAULT_ENDPOINT
    assert model == deploy_mod.DEFAULT_MODEL
    assert outputs == {}


def test_resolve_target_prefers_terraform_outputs(deploy_mod, monkeypatch):
    monkeypatch.delenv("FOUNDRY_PROJECT_ENDPOINT", raising=False)
    monkeypatch.delenv("FOUNDRY_MODEL_NAME", raising=False)
    monkeypatch.setattr(
        deploy_mod,
        "terraform_outputs",
        lambda *a, **k: {
            "project_endpoint": "https://tf.example.com/api/projects/tf-lab",
            "model_deployment_name": "tf-model",
        },
    )
    endpoint, model, _ = deploy_mod.resolve_target(None, None)
    assert endpoint == "https://tf.example.com/api/projects/tf-lab"
    assert model == "tf-model"


def test_resolve_target_env_beats_terraform(deploy_mod, monkeypatch):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://env.example.com/api/projects/env-lab")
    monkeypatch.setenv("FOUNDRY_MODEL_NAME", "env-model")
    monkeypatch.setattr(
        deploy_mod,
        "terraform_outputs",
        lambda *a, **k: {
            "project_endpoint": "https://tf.example.com",
            "model_deployment_name": "tf-model",
        },
    )
    endpoint, model, _ = deploy_mod.resolve_target(None, None)
    assert endpoint.endswith("/api/projects/env-lab")
    assert model == "env-model"


def test_resolve_target_explicit_argument_wins(deploy_mod, monkeypatch):
    monkeypatch.setenv("FOUNDRY_PROJECT_ENDPOINT", "https://env.example.com")
    endpoint, model, _ = deploy_mod.resolve_target("https://arg.example.com", "arg-model")
    assert endpoint == "https://arg.example.com"
    assert model == "arg-model"


def test_terraform_outputs_is_graceful_without_a_state(deploy_mod, tmp_path):
    """A missing/never-applied stack must not raise."""
    assert deploy_mod.terraform_outputs(tmp_path / "nonexistent") == {}


def test_terraform_outputs_parses_value_wrapper(deploy_mod, monkeypatch):
    raw = json.dumps({
        "project_endpoint": {"value": "https://x.example.com", "type": "string"},
        "empty": {"value": None, "type": "string"},
    })

    class _Proc:
        returncode = 0
        stdout = raw

    monkeypatch.setattr(deploy_mod.subprocess, "run", lambda *a, **k: _Proc())
    outs = deploy_mod.terraform_outputs(REPO_ROOT)
    assert outs == {"project_endpoint": "https://x.example.com"}


# --------------------------------------------------------------------------- #
# Deployment targets
# --------------------------------------------------------------------------- #
def test_targets_define_both_agents(deploy_mod):
    assert set(deploy_mod.TARGETS) == {"router", "component-graph"}
    assert deploy_mod.TARGETS["router"][0] == "agcomps-router"
    assert deploy_mod.TARGETS["component-graph"][0] == "agcomps-component-graph"


def test_parser_exposes_target_flag(deploy_mod):
    args = deploy_mod.build_parser().parse_args([])
    assert args.target == "router"
    assert deploy_mod.build_parser().parse_args(["--target", "component-graph"]).target == (
        "component-graph"
    )


def test_parser_rejects_unknown_target(deploy_mod):
    with pytest.raises(SystemExit):
        deploy_mod.build_parser().parse_args(["--target", "nope"])


# --------------------------------------------------------------------------- #
# Hosted-agent bootstrap
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hosted_main():
    """Import the hosted agent's `main` module from the checkout."""
    sys.path.insert(0, str(REPO_ROOT))
    sys.path.insert(0, str(HOSTED_SRC))
    spec = importlib.util.spec_from_file_location("hosted_main", HOSTED_SRC / "main.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["hosted_main"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bootstrap_puts_vendor_and_repo_on_path(hosted_main):
    assert str(HOSTED_SRC) in sys.path
    # The repo root must be importable so `platform.schema` resolves.
    assert str(REPO_ROOT) in sys.path


def test_bootstrap_resolves_monorepo_imports(hosted_main):
    """The imports the agent needs must actually resolve after bootstrap."""
    from agentic_router.config import Settings  # noqa: F401
    from agentic_router.factory import build_agent  # noqa: F401
    from components_core import Message  # noqa: F401
    from memory_store import InMemorySessionStore  # noqa: F401
    from platform.engineering.graph_orchestrator import ComponentGraphOrchestrator  # noqa: F401
    from platform.schema.loader import load_spec  # noqa: F401


def test_data_plane_base_derives_openai_base_url(hosted_main, monkeypatch):
    monkeypatch.setenv(
        "FOUNDRY_PROJECT_ENDPOINT",
        "https://my-foundry-resource.services.ai.azure.com/api/projects/agent-lab",
    )
    assert hosted_main._data_plane_base() == (
        "https://my-foundry-resource.services.ai.azure.com/openai/v1"
    )


def test_default_graph_spec_is_valid_and_uses_react(hosted_main):
    spec = hosted_main._default_graph_spec()
    assert spec.kind == "ComponentGraph"
    assert spec.spec.agent_plane.pattern == "react"
    assert spec.spec.runtime_chassis.protocol == "responses"


def test_hosted_agent_builds_the_component_graph(hosted_main):
    """`_graph()` must return a wired orchestrator, not None."""
    graph = hosted_main._graph()
    assert graph is not None
    assert type(graph).__name__ == "ComponentGraphOrchestrator"
    # The data/ML/tools planes must actually be wired.
    names = [t.name for t in graph.tool_registry]
    assert "query_lakehouse" in names
    assert "invoke_ml_model" in names


def test_graph_descriptor_reports_wired_planes(hosted_main):
    import asyncio

    payload = json.loads(asyncio.run(hosted_main.graph_descriptor()))
    assert payload["available"] is True
    assert payload["pattern"] == "react"
    assert payload["deployment"]["protocol"] == "responses"
    assert "query_lakehouse" in payload["tools"]


def test_run_graph_executes_end_to_end(hosted_main):
    """`run_graph` must drive the orchestrator and return a serialisable result."""
    import asyncio

    payload = json.loads(asyncio.run(hosted_main.run_graph("Audit the data plane.")))
    assert payload["success"] is True
    assert "duration_ms" in payload
    assert "data_plane" in payload and "ml_plane" in payload


def test_hosted_agent_builds_the_tiered_router(hosted_main):
    """The tiered agent must build without a live model endpoint."""
    agent = hosted_main._agent()
    assert agent is not None
    # A single instance is shared, so repeated calls are cheap and consistent.
    assert hosted_main._agent() is agent


def test_load_spec_resolves_the_bundled_graph(hosted_main):
    """The shipped component_graph.yaml must load and validate."""
    from platform.schema.loader import load_spec

    spec = load_spec(REPO_ROOT / "projects" / "component_graph.yaml")
    assert spec.kind == "ComponentGraph"
    assert spec.spec.agent_plane.pattern == "react"
    assert spec.spec.data_plane.batch_pipelines
    assert spec.spec.ml_plane.endpoints
