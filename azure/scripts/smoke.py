"""Smoke-test a deployed hosted agent (Responses protocol, Entra auth).

Usage:
    FOUNDRY_PROJECT_ENDPOINT=https://my-foundry-usecase.services.ai.azure.com/api/projects/agent-lab \
        python azure/scripts/smoke.py --message "Debate: tabs beat spaces"

Notes:
- Scope for the Entra token is https://ai.azure.com/.default (Cognitive
  Services scope is rejected by the agents data plane).
- First invocation after an idle period pays a sandbox cold start (minutes);
  use --timeout to raise the HTTP timeout beyond the default 540s.
- The endpoint host comes from the account's custom subdomain
  (my-foundry-usecase), not the resource name - always derive it from
  FOUNDRY_PROJECT_ENDPOINT.
"""

from __future__ import annotations

import argparse
import json
import os
import urllib.request

from azure.identity import DefaultAzureCredential


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--message", required=True)
    ap.add_argument("--agent", default=os.environ.get("FOUNDRY_HOSTED_AGENT_NAME", "agcomps-router"))
    ap.add_argument("--timeout", type=int, default=540)
    args = ap.parse_args()

    project = os.environ["FOUNDRY_PROJECT_ENDPOINT"].rstrip("/")
    url = f"{project}/agents/{args.agent}/endpoint/protocols/openai/responses?api-version=v1"

    token = DefaultAzureCredential().get_token("https://ai.azure.com/.default").token

    payload = json.dumps(
        {"model": os.environ.get("AZURE_AI_MODEL_DEPLOYMENT_NAME", "gpt-5-mini"),
         "input": args.message, "stream": False}
    ).encode()

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as resp:
            body = json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code}: {e.read().decode()[:500]}")

    texts = [
        part.get("text", "")
        for item in body.get("output", [])
        if item.get("type") == "message"
        for part in item.get("content", [])
        if part.get("type") in ("output_text", "text")
    ]
    print(json.dumps({"status": "ok", "reply": "\n".join(texts) or body}, indent=2)[:2000])


if __name__ == "__main__":
    main()
