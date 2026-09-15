"""Platform engineering layer for agent-components.

Declarative, YAML-driven platform for defining, deploying, and managing
AI agents on Azure. Supports multiple agentic design patterns (ReAct,
Supervisor, Network, Sequential, Map-Reduce), sandboxed code execution,
plugin-based extensibility, and one-click deployment to Azure services.

--------------------------------------------------------------------------------
NOTE ON THE NAME `platform`
--------------------------------------------------------------------------------
This package is literally named ``platform``, which collides with the Python
standard-library module of the same name. Any process that has this repository
root on ``sys.path`` — which includes running tests from the repo root and
anything using the editable install — will resolve ``import platform`` to *this*
package.

That breaks third-party code which expects the stdlib module. pytest, for one,
calls ``platform.python_version()`` during session startup and dies with::

    AttributeError: module 'platform' has no attribute 'python_version'

The shim below re-exports the standard library's public API onto this package,
so ``platform.python_version()`` and friends keep working while
``platform.schema``, ``platform.engineering`` and the rest resolve here. It
loads the real stdlib module under a private alias to avoid recursing into
itself.

The genuinely robust fix is to rename this package; until then the shim keeps
both meanings coexisting.
"""
from __future__ import annotations

import importlib.util as _importlib_util
import sysconfig as _sysconfig
from pathlib import Path as _Path
from typing import Any as _Any

__version__ = "0.1.0"


def _load_stdlib_platform() -> _Any:
    """Load the real stdlib ``platform`` module under a private name."""
    stdlib = _sysconfig.get_paths().get("stdlib")
    if not stdlib:
        return None
    path = _Path(stdlib) / "platform.py"
    if not path.is_file():
        return None
    spec = _importlib_util.spec_from_file_location("_stdlib_platform", path)
    if spec is None or spec.loader is None:
        return None
    module = _importlib_util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _install_stdlib_shim() -> None:
    """Copy the stdlib module's public names onto this package.

    Only names not already defined here are copied, so this package's own
    attributes always win. A few private helpers (``_sysconfig``, etc.) are
    skipped by the leading-underscore rule.
    """
    stdlib = _load_stdlib_platform()
    if stdlib is None:
        return
    this = globals()
    for name in dir(stdlib):
        if name.startswith("__"):
            continue
        if name in this:
            continue
        try:
            this[name] = getattr(stdlib, name)
        except Exception:  # pragma: no cover - defensive
            continue


_install_stdlib_shim()
del _install_stdlib_shim
