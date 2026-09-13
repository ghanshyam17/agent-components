#!/usr/bin/env bash
# Assemble + deploy the agentic-router Function app via zip-push (Kudu) - no func CLI.
# Python v2 model: function_app.py must sit at the app ROOT (not a subfolder).
set -euo pipefail
RG="${RG:-my-foundry-rg}"
APP="${APP:-agcomps-agent-api}"

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

# Flat layout at wwwroot root:
cp "$ROOT/azure/functions/host.json" "$STAGE/"
cp "$ROOT/azure/functions/requirements.txt" "$STAGE/"
cp "$ROOT/azure/functions/agent-api/function_app.py" "$STAGE/"
mkdir -p "$STAGE/_vendor"
cp -R "$ROOT/core/src/components_core" "$STAGE/_vendor/components_core"
cp -R "$ROOT/components/agentic-router/agentic_router" "$STAGE/_vendor/agentic_router"
cp -R "$ROOT/components/memory-store/memory_store" "$STAGE/_vendor/memory_store"
cp -R "$ROOT/components/model-gateway/model_gateway" "$STAGE/_vendor/model_gateway" 2>/dev/null || true

echo "deploying $APP from $STAGE"
ZIP="$(mktemp -u).zip"
(cd "$STAGE" && zip -qr "$ZIP" .)
az functionapp deployment source config-zip -g "$RG" -n "$APP" --src "$ZIP" >/dev/null
echo "uploaded. syncing + waiting for the worker to restart..."
sleep 20
az rest --method get --url "https://management.azure.com/subscriptions/$(
  az account show --query id -o tsv)/resourceGroups/$RG/providers/Microsoft.Web/sites/$APP/functions?api-version=2023-12-01" \
  --query '[].{name:name, status:properties.status}' -o table
