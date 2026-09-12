"""Shared OpenAI-compatible helpers.

`messages_to_openai` converts our `core` `Message` list into the dict shape the
OpenAI / vLLM chat endpoint expects (including tool-call and tool-result
encoding). Both the router's clients and the model-gateway use this so the
encoding stays consistent across components.
"""
from __future__ import annotations

import json
from typing import Any

from components_core.models import Message


def messages_to_openai(messages: list[Message]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if m.role == "tool":
            out.append(
                {
                    "role": "tool",
                    "content": m.content,
                    "tool_call_id": m.tool_call_id or "",
                    "name": m.name or "",
                }
            )
            continue
        d: dict[str, Any] = {"role": m.role, "content": m.content}
        if m.tool_calls:
            d["tool_calls"] = [
                {
                    "id": f"call_{i}",
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(tc.arguments),
                    },
                }
                for i, tc in enumerate(m.tool_calls)
            ]
        out.append(d)
    return out