#!/usr/bin/env bash
# Install the `agcomps` CLI onto the PATH of this repo's virtualenv.
#
# Why not a `[project.scripts]` entry point? The platform package is a top-level
# package named `platform`, colliding with the stdlib module of the same name.
# A generated console script imports its target before any project code runs, and
# the stdlib directory precedes site-packages on sys.path, so
# `from platform.scaffold.cli import agcomps` always resolves the stdlib and
# fails with "'platform' is not a package". `scripts/agcomps` instead bootstraps
# sys.path first, so it resolves the repo's package.
#
# Re-run this after recreating the venv.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_BIN="${1:-$REPO_ROOT/.venv/bin}"

if [ ! -d "$VENV_BIN" ]; then
  echo "no virtualenv bin dir at $VENV_BIN (run 'uv sync --all-packages' first)" >&2
  exit 1
fi

install -m 0755 "$REPO_ROOT/scripts/agcomps" "$VENV_BIN/agcomps"

# Point the shebang at this venv's interpreter. `#!/usr/bin/env python3` would
# pick up whatever python3 is on PATH, which may not be the venv (and on macOS
# plain `python` is still Python 2, which cannot parse this file at all).
if [ -x "$VENV_BIN/python3" ]; then
  PY="$VENV_BIN/python3"
elif [ -x "$VENV_BIN/python" ]; then
  PY="$VENV_BIN/python"
else
  echo "no interpreter in $VENV_BIN" >&2
  exit 1
fi
sed -i.bak "1s|^#!.*|#!$PY|" "$VENV_BIN/agcomps" && rm -f "$VENV_BIN/agcomps.bak"

echo "installed agcomps -> $VENV_BIN/agcomps (interpreter: $PY)"
"$VENV_BIN/agcomps" --help >/dev/null && echo "verified: agcomps --help works"
