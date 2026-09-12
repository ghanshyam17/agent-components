"""Planning: decompose a goal into subtasks using the higher model.

Disabled by default via `agent_enable_planning`; the agent loop calls this only
for multi-step-looking goals. Each subtask is later routed independently.
"""
from __future__ import annotations

import json
import re

from agentic_router.clients import ModelClient
from agentic_router.models import Message, Plan, SubTask

PLANNER_SYSTEM = """You are a planning module. Break the user's goal into a small ordered list
of concrete subtasks that an agent with shell, file, web and calculator tools
can execute. Reply with JSON only:
{"goal": "<the goal>", "subtasks": ["step 1", "step 2", ...]}
Use at most {max} subtasks. If the goal is already a single action, return one
subtask. Do not include subtasks you cannot verify."""


def _extract_json(content: str) -> dict | None:
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


async def plan_goal(
    goal: str, higher_client: ModelClient, max_subtasks: int = 5
) -> Plan:
    sys = PLANNER_SYSTEM.format(max=max_subtasks)
    messages = [
        Message(role="system", content=sys),
        Message(role="user", content=goal[:4000]),
    ]
    content, _ = await higher_client.chat(messages, temperature=0.1, max_tokens=400)
    obj = _extract_json(content) or {"goal": goal, "subtasks": [goal]}
    raw = obj.get("subtasks", []) or [goal]
    subtasks = [SubTask(description=str(s)) for s in raw[:max_subtasks] if str(s).strip()]
    if not subtasks:
        subtasks = [SubTask(description=goal)]
    return Plan(goal=str(obj.get("goal", goal)), subtasks=subtasks)