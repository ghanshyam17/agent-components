"""LangGraph workflow definition for {{ project_name }}.

Defines nodes, conditional branching, and compilation for cyclical stateful execution.
"""
from __future__ import annotations
from typing import Dict, Any, TypedDict, List

class AgentState(TypedDict):
    task: str
    draft: str
    iterations: int
    quality_score: float
    feedback: str

def draft_node(state: AgentState) -> Dict[str, Any]:
    """Generate initial draft."""
    return {
        "draft": f"Draft response to: {state['task']}",
        "iterations": state.get("iterations", 0) + 1,
    }

def verify_node(state: AgentState) -> Dict[str, Any]:
    """Evaluate quality and provide verification feedback."""
    iterations = state.get("iterations", 1)
    # Passed after 2 iterations or if criteria met
    passed = iterations >= 2
    return {
        "quality_score": 0.95 if passed else 0.60,
        "feedback": "Approved" if passed else "Needs more depth",
    }

def check_quality(state: AgentState) -> str:
    """Conditional edge routing logic."""
    if state.get("quality_score", 0.0) >= 0.8:
        return "pass"
    return "needs_work"

def refine_node(state: AgentState) -> Dict[str, Any]:
    """Refine draft based on feedback."""
    return {
        "draft": f"{state.get('draft', '')} [Refined with: {state.get('feedback', '')}]",
        "iterations": state.get("iterations", 1) + 1,
    }
