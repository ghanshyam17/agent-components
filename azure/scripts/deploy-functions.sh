#!/usr/bin/env bash
# Assemble + deploy the agentic-router Function app (Y1 Consumption, $0 idle).
set -euo pipefail
RG="${RG:-my-foundry-rg}"
APP="${APP:-agcomps-agent-api}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

cp -R "$ROOT/azure/functions/"* "$STAGE/"
mkdir -p "$STAGE/_vendor"
# Monorepo sources the agent needs at runtime:
cp -R "$ROOT/core/src/components_core" "$STAGE/_vendor/components_core"
cp -R "$ROOT/components/agentic-router/agentic_router" "$STAGE/_vendor/agentic_router"
cp -R "$ROOT/components/memory-store/memory_store" "$STAGE/_vendor/memory_store"
cp -R "$ROOT/components/model-gateway/model_gateway" "$STAGE/_vendor/model_gateway" 2>/dev/null || true

echo "deploying $APP from $STAGE"
(cd "$STAGE" && func azure functionapp publish "$APP" --python)
