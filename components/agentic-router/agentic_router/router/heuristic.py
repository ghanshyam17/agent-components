"""Heuristic router — a fast, deterministic first-pass complexity scorer.

Scores a task on a 0..1 scale using cheap signals only:
  * prompt length (token estimate)
  * keyword salience (reasoning / code / multi-step cues)
  * code presence
  * explicit tool / multi-step intent

No model calls are made here. When the score lands in an uncertain band the
hybrid router falls back to the LLM classifier in ``classifier.py``.
"""
from __future__ import annotations

import re

# Cues that push a task toward the higher model.
HIGH_CUES = {
    # reasoning / planning
    "plan": 0.18, "design": 0.16, "architect": 0.20, "strategy": 0.16,
    "step by step": 0.15, "why": 0.08, "reason": 0.12, "analyze": 0.14,
    "debug": 0.15, "refactor": 0.15, "optimize": 0.14, "compare": 0.10,
    # coding / math
    "implement": 0.14, "function": 0.10, "class": 0.08, "algorithm": 0.16,
    "bug": 0.12, "stack trace": 0.14, "regex": 0.12, "sql": 0.10,
    "prove": 0.18, "derive": 0.16, "calculate": 0.10,
}

# Cues that suggest a simple task the lower model can handle.
LOW_CUES = {
    "hi": 0.12, "hello": 0.12, "hey": 0.10, "thanks": 0.10, "summarize": 0.08,
    "translate": 0.08, "list": 0.06, "format": 0.06, "rename": 0.08,
    "count words": 0.06, "echo": 0.10,
}

CODE_FENCE = re.compile(r"```", re.MULTILINE)
LONG_WORD_BOUND = re.compile(r"\b\w{15,}\b")


def _token_estimate(text: str) -> int:
    """Rough token estimate (~4 chars/token, floored)."""
    return max(1, len(text) // 4)


def score_task(task: str) -> tuple[float, dict[str, float]]:
    """Return (score, signals). Higher score => use the higher model."""
    text = task.strip()
    lowered = text.lower()
    signals: dict[str, float] = {}

    # Length: longer prompts tend to need more capable models. Saturates ~1k tokens.
    tokens = _token_estimate(text)
    length_score = min(1.0, tokens / 1000.0) * 0.25
    signals["length"] = round(length_score, 3)

    # Keyword salience.
    cue = 0.0
    for kw, w in HIGH_CUES.items():
        if kw in lowered:
            cue += w
    for kw, w in LOW_CUES.items():
        # Only count short prompts' low cues to avoid swamping long prompts.
        if tokens < 60 and kw in lowered:
            cue -= w
    keyword_score = max(0.0, min(1.0, cue))
    signals["keywords"] = round(keyword_score, 3)

    # Code presence — fences or long identifiers suggest a coding task.
    code_score = 0.0
    n_fences = len(CODE_FENCE.findall(text))
    if n_fences >= 2:
        code_score = 0.3
    elif n_fences == 1:
        code_score = 0.15
    if LONG_WORD_BOUND.search(text):
        code_score = max(code_score, 0.15)
    signals["code"] = round(code_score, 3)

    # Explicit tool / multi-step intent.
    tool_score = 0.0
    if re.search(r"\b(run|execute|search|fetch|read|write|file|shell|web)\b", lowered):
        tool_score += 0.2
    if re.search(r"\bthen\b|\bafter that\b|\bfinally\b|\bmulti.?step\b", lowered):
        tool_score += 0.15
    if "?" in text and text.count("?") >= 3:
        tool_score += 0.1  # multi-part question
    signals["tools_multistep"] = round(min(1.0, tool_score), 3)

    # Weighted blend. Length alone shouldn't dominate; cues carry the most signal.
    score = (
        length_score * 0.20
        + keyword_score * 0.40
        + code_score * 0.20
        + min(1.0, tool_score) * 0.20
    )
    score = max(0.0, min(1.0, score))
    signals["total"] = round(score, 3)
    return score, signals