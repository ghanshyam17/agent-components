"""Root conftest: make this repo's `platform` package importable under pytest.

`platform/` shadows the Python standard-library module of the same name.
pytest imports the stdlib `platform` during its own startup (before pytest
puts the rootdir on `sys.path`), so `sys.modules["platform"]` is already
bound to the stdlib by the time collection begins — a bare
`import platform.schema` then fails with "'platform' is not a package".

The package's own `__init__.py` already re-exports the stdlib API onto
itself (see `platform/__init__.py`), so swapping the entry in `sys.modules`
for the real package keeps both meanings working.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
_PKG = _ROOT / "platform"


def _install_repo_platform() -> None:
    if not (_PKG / "__init__.py").is_file():
        return
    current = sys.modules.get("platform")
    if current is not None and getattr(current, "__file__", None) == str(_PKG / "__init__.py"):
        return  # already ours
    spec = importlib.util.spec_from_file_location(
        "platform", _PKG / "__init__.py", submodule_search_locations=[str(_PKG)]
    )
    if spec is None or spec.loader is None:
        return
    module = importlib.util.module_from_spec(spec)
    sys.modules["platform"] = module
    spec.loader.exec_module(module)


_install_repo_platform()
