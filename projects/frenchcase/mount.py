"""Mount FrenchCase onto sys.path so its engines are importable.

FrenchCase's module layout has two quirks that make naive mounting fail:

1. `app/` has an `__init__.py`, so `app.agents.tools` must resolve from the
   repo root (`french-learning/`), not from `french-learning/app/`.
2. Several modules (e.g. `app/core/llm_router.py`) do bare `from config import
   ...`, which only resolves if `french-learning/app/` is *also* on the path.

Putting both entries on the path collides: with `app/` first, `import app`
finds `app/app.py` (a module) instead of the `app` package.

The fix: append `french-learning/app/` so it is a **fallback** entry. Python
tries path entries in order, so `import app` still resolves to the package via
the repo root, while `import config` (unavailable at the repo root) falls
through to the appended entry.

Call `mount_frenchcase()` once before importing FrenchCase engines.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Default location; override with the FRENCHCASE_HOME env var.
DEFAULT_HOME = Path("/Users/ghanshyam/Documents/french-learning")

_mounted: bool = False


def frenchcase_home() -> Path:
    """Resolve the FrenchCase repo root (env var wins over the default)."""
    return Path(os.environ.get("FRENCHCASE_HOME", str(DEFAULT_HOME))).expanduser().resolve()


def is_mounted() -> bool:
    """True once mount_frenchcase() has run successfully."""
    return _mounted


def mount_frenchcase(home: Path | str | None = None, *, verbose: bool = False) -> bool:
    """Put FrenchCase's paths on sys.path in the order its layout requires.

    Returns True when the repo root exists and was mounted (or already was).
    Safe to call repeatedly.
    """
    global _mounted
    if _mounted:
        return True

    root = Path(home).expanduser().resolve() if home else frenchcase_home()
    if not root.is_dir():
        if verbose:
            print(f"[frenchcase.mount] not found: {root}")
        return False

    # Repo root FIRST (so `app` resolves to the package, not app/app.py).
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    # app/ APPENDED as a fallback (bare `import config`, `import core`, ...).
    app_dir = root / "app"
    if app_dir.is_dir() and str(app_dir) not in sys.path:
        sys.path.append(str(app_dir))

    _mounted = True
    if verbose:
        print(f"[frenchcase.mount] mounted {root} (root) + {app_dir} (fallback)")
    return True


if __name__ == "__main__":
    ok = mount_frenchcase(verbose=True)
    print(f"mounted: {ok}")
    if ok:
        probes = [
            ("app.core.session_store", "SessionStore"),
            ("app.core.vector", "retrieve"),
            ("app.core.llm_router", "LLMRouter"),
            ("app.agents.tools", "repo_search"),
        ]
        for mod, attr in probes:
            try:
                m = __import__(mod, fromlist=[attr])
                getattr(m, attr)
                print(f"  ✅ {mod}.{attr}")
            except Exception as exc:
                print(f"  ❌ {mod}.{attr} → {type(exc).__name__}: {exc}")