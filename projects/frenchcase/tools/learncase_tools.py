"""FrenchCase tool adapter — bridges FrenchCase's 5 agent tools to the
agent-components toolkit Registry.

This module is the glue between FrenchCase's existing tool implementations
(in app/agents/tools.py) and the platform's tool registry. It loads each
FrenchCase tool function and wraps it in a toolkit Tool so the chassis's
agentic-router, patterns, and Foundry deployment can call them.

Usage from the platform:
    from tools.learncase_tools import build_frenchcase_registry
    registry = build_frenchcase_tools()
    result = await registry.execute("repo_search", query="subjonctif", cefr="B2")

Usage from FrenchCase directly (unchanged):
    from app.agents.tools import repo_search, get_conjugation, grade_essay, ...
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── FrenchCase tool imports ──────────────────────────────────────────────
# These are the real implementations from app/agents/tools.py.
# `mount_frenchcase()` sets the two sys.path entries FrenchCase's layout needs.
from projects.frenchcase.mount import mount_frenchcase as _mount

_mount()

try:
    from app.agents.tools import (
        repo_search,
        get_conjugation,
        grade_essay,
        tts_pronounce,
        load_lesson,
    )
    FRENCHCASE_TOOLS_AVAILABLE = True
except ImportError:
    try:
        from agents.tools import (  # type: ignore
            repo_search,
            get_conjugation,
            grade_essay,
            tts_pronounce,
            load_lesson,
        )
        FRENCHCASE_TOOLS_AVAILABLE = True
    except ImportError:
        FRENCHCASE_TOOLS_AVAILABLE = False
        logger.warning("FrenchCase tools not importable — running in stub mode")

        async def repo_search(query: str, cefr: Optional[str] = None, section: Optional[str] = None, top_k: int = 5) -> dict:
            return {"error": "FrenchCase not mounted", "query": query}

        async def get_conjugation(verb: str, tense: Optional[str] = None) -> dict:
            return {"error": "FrenchCase not mounted", "verb": verb}

        async def grade_essay(text: str, section: str = "EE") -> dict:
            return {"error": "FrenchCase not mounted"}

        async def tts_pronounce(text: str, voice: str = "fr-FR-DeniseNeural") -> dict:
            return {"error": "FrenchCase not mounted"}

        async def load_lesson(lesson_id: str) -> dict:
            return {"error": "FrenchCase not mounted", "lesson_id": lesson_id}


# ── Tool metadata ────────────────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "name": "repo_search",
        "description": "Search the TEF learning corpus (2,131 catalog entries, 2,054 vector chunks). "
                       "Supports CEFR and section filtering.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query"},
                "cefr": {"type": "string", "enum": ["A1","A2","B1","B2","C1","C2"]},
                "section": {"type": "string", "enum": ["CO","CE","EE","EO","GRM","VOC","PHO"]},
                "top_k": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
        "handler": repo_search,
    },
    {
        "name": "get_conjugation",
        "description": "Get full conjugation table for a French verb across all tenses.",
        "parameters": {
            "type": "object",
            "properties": {
                "verb": {"type": "string", "description": "Infinitive form (e.g., aller)"},
                "tense": {"type": "string", "description": "Specific tense (optional)"},
            },
            "required": ["verb"],
        },
        "handler": get_conjugation,
    },
    {
        "name": "grade_essay",
        "description": "Grade a TEF Expression Écrite essay on the 4-criteria rubric "
                       "(task/coherence/vocab/grammar, each 0-9, total /36).",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "The essay text to grade"},
                "section": {"type": "string", "default": "EE"},
            },
            "required": ["text"],
        },
        "handler": grade_essay,
    },
    {
        "name": "tts_pronounce",
        "description": "Generate French pronunciation audio via edge-tts.",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "French text to pronounce"},
                "voice": {"type": "string", "default": "fr-FR-DeniseNeural"},
            },
            "required": ["text"],
        },
        "handler": tts_pronounce,
    },
    {
        "name": "load_lesson",
        "description": "Load a specific LearnCase by its catalog ID (1,816 available).",
        "parameters": {
            "type": "object",
            "properties": {
                "lesson_id": {"type": "string", "description": "LearnCase catalog ID"},
            },
            "required": ["lesson_id"],
        },
        "handler": load_lesson,
    },
]


# ── Registry builder ─────────────────────────────────────────────────────

def get_tool_definitions() -> list[dict]:
    """Return the tool definitions in OpenAI function-calling format."""
    defs = []
    for td in TOOL_DEFINITIONS:
        defs.append({
            "name": td["name"],
            "description": td["description"],
            "parameters": td["parameters"],
        })
    return defs


async def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute a FrenchCase tool by name with the given arguments."""
    for td in TOOL_DEFINITIONS:
        if td["name"] == name:
            handler = td["handler"]
            try:
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(**arguments)
                else:
                    result = handler(**arguments)
                return {"ok": True, "result": result, "tool": name}
            except Exception as exc:
                logger.error(f"Tool {name} failed: {exc}")
                return {"ok": False, "error": str(exc), "tool": name}
    return {"ok": False, "error": f"Unknown tool: {name}"}


def build_frenchcase_tools() -> dict:
    """Build the tool registry for the platform PatternContext.

    Returns a dict suitable for assignment to PatternContext.tools or
    injection into the agentic-router's tool registry.
    """
    return {
        "definitions": get_tool_definitions(),
        "execute": execute_tool,
        "available": FRENCHCASE_TOOLS_AVAILABLE,
        "count": len(TOOL_DEFINITIONS),
        "names": [td["name"] for td in TOOL_DEFINITIONS],
    }


if __name__ == "__main__":
    # Quick smoke test
    import json

    tools = build_frenchcase_tools()
    print(f"FrenchCase tools: {tools['count']} tools, available={tools['available']}")
    for d in tools["definitions"]:
        print(f"  - {d['name']}: {d['description'][:80]}...")