#!/usr/bin/env bash
# Assemble + deploy the agentic-router Function app via `func publish` (remote build).
# Python v2 model: function_app.py sits at the app ROOT.
set -euo pipefail
RG="${RG:-my-foundry-rg}"
APP="${APP:-agcomps-agent-api}"
FUNC_BIN="${FUNC_BIN:-$HOME/bin/func-cli/func}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp "$ROOT/azure/functions/host.json" "$STAGE/"
cp "$ROOT/azure/functions/requirements.txt" "$STAGE/"
cp "$ROOT/azure/functions/agent-api/function_app.py" "$STAGE/"
mkdir -p "$STAGE/_vendor"
cp -R "$ROOT/core/src/components_core" "$STAGE/_vendor/components_core"
cp -R "$ROOT/components/agentic-router/agentic_router" "$STAGE/_vendor/agentic_router"
cp -R "$ROOT/components/memory-store/memory_store" "$STAGE/_vendor/memory_store"
cp -R "$ROOT/components/model-gateway/model_gateway" "$STAGE/_vendor/model_gateway" 2>/dev/null || true

find "$STAGE" -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true

echo "deploying $APP from $STAGE"
(cd "$STAGE" && "$FUNC_BIN" azure functionapp publish "$APP" --python) | tail -8
