"""LLM-as-judge classifier fallback.

When the heuristic score lands in the uncertain band, we ask the *lower*
model itself to classify the task as simple or complex. Using the lower model
for classification is cheap and fast; if it can't reliably decide, default to
the higher model (safer for correctness).
"""
from __future__ import annotations

import json
import re

from agentic_router.clients import ModelClient
from agentic_router.models import Message, ModelTier

CLASSIFIER_SYSTEM = """You are a routing classifier. Decide whether a task needs a SMALL/FAST model
or a LARGE/CAPABLE model.

Reply with a JSON object ONLY, no prose:
{"tier": "lower" | "higher", "reason": "<= 12 words"}

Rules of thumb:
- lower: greetings, simple Q&A, summaries, formatting, short translations, list/count tasks.
- higher: multi-step reasoning, coding, debugging, math/proofs, planning, tool use, long context.
When unsure, choose "higher"."""

CLASSIFIER_RE = re.compile(r'\{[^{}]*"tier"[^{}]*\}', re.DOTALL)


async def classify(
    task: str, lower_client: ModelClient
) -> tuple[ModelTier, str]:
    """Return (tier, reason). Falls back to HIGHER on any parse failure."""
    messages = [
        Message(role="system", content=CLASSIFIER_SYSTEM),
        Message(role="user", content=task[:2000]),
    ]
    try:
        content, _ = await lower_client.chat(
            messages, temperature=0.0, max_tokens=80
        )
    except Exception as e:  # noqa: BLE001
        return ModelTier.HIGHER, f"classifier_error: {e}"

    m = CLASSIFIER_RE.search(content)
    if not m:
        # Look for a bare tier word as a last resort.
        if re.search(r"\bhigher\b", content, re.IGNORECASE):
            return ModelTier.HIGHER, "classifier_keyword"
        if re.search(r"\blower\b", content, re.IGNORECASE):
            return ModelTier.LOWER, "classifier_keyword"
        return ModelTier.HIGHER, "classifier_unparseable"

    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return ModelTier.HIGHER, "classifier_unparseable"

    tier = str(obj.get("tier", "higher")).lower()
    reason = str(obj.get("reason", ""))[:120]
    if tier not in ("lower", "higher"):
        tier = "higher"
    return ModelTier(tier), reason or "classifier_decision"