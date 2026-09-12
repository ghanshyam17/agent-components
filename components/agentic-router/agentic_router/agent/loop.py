"""The agentic loop: ReAct-style tool use with per-subtask routing.

Flow per user turn:
  1. (optional) Plan the goal into subtasks via the higher model.
  2. For each subtask, route it to lower or higher model.
  3. Run a ReAct loop: model emits tool calls -> execute -> feed back, up to
     max_iterations, streaming token deltas as they arrive.
  4. Emit a final answer event.

The loop is an async generator yielding AgentEvent objects, so the server can
pipe them straight to an SSE stream. A non-streaming `run()` wrapper is also
provided for the CLI / tests.
"""
from __future__ import annotations

from typing import AsyncIterator

from agentic_router.agent.planner import plan_goal
from agentic_router.clients import ClientRegistry, ModelClient
from agentic_router.config import Settings
from agentic_router.models import (
    AgentEvent, Message, ModelTier, RouteDecision, ToolResult,
)
from agentic_router.router.router import Router
from agentic_router.tools.base import Registry
from components_core import SessionState
from memory_store import SessionStore

AGENT_SYSTEM = """You are an autonomous agent. Use the provided tools when the task requires
external action, then answer concisely. To call a tool, emit a tool_call with
the tool's name and JSON arguments. After tools return, summarize their output
and continue until the task is complete. Do not call a tool you have already
called with identical arguments unless re-running for a reason."""


class Agent:
    def __init__(
        self,
        settings: Settings,
        registry: ClientRegistry,
        router: Router,
        tools: Registry,
        sessions: SessionStore,
    ):
        self.s = settings
        self.registry = registry
        self.router = router
        self.tools = tools
        self.sessions = sessions

    # ------------------------------------------------------------------ public
    async def run(
        self, task: str, session_id: str | None = None
    ) -> tuple[str, RouteDecision | None, SessionState]:
        """Non-streaming convenience wrapper. Collects all events and returns
        the final answer, the top-level route decision, and the session."""
        final = ""
        top_route: RouteDecision | None = None
        state = await self.sessions.get_or_create(session_id)
        async for ev in self.stream(task, state.session_id):
            if ev.type == "final":
                final = ev.data.get("content", "")
            elif ev.type == "route" and top_route is None:
                top_route = RouteDecision(**ev.data["decision"])
        if not final:
            # No explicit final event (e.g. error path); fall back to last answer.
            final = ""
        return final, top_route, state

    async def stream(
        self, task: str, session_id: str
    ) -> AsyncIterator[AgentEvent]:
        state = await self.sessions.get_or_create(session_id)
        await self.sessions.add_message(
            state.session_id, Message(role="user", content=task)
        )

        # --- 1. Plan (optional) ---
        subtask_descriptions: list[str] = []
        if self.s.agent_enable_planning and _looks_multi_step(task):
            try:
                plan = await plan_goal(
                    task,
                    self.registry.get(ModelTier.HIGHER),
                    self.s.agent_max_plan_subtasks,
                )
                await self.sessions.set_plan(state.session_id, plan)
                yield AgentEvent(type="plan", data={"plan": plan.model_dump()})
                subtask_descriptions = [st.description for st in plan.subtasks]
            except Exception as e:  # noqa: BLE001
                yield AgentEvent(
                    type="error", data={"stage": "plan", "error": str(e)}
                )
        if not subtask_descriptions:
            subtask_descriptions = [task]

        # --- 2. Execute each subtask ---
        running_msgs = list(state.messages)
        overall_answer_parts: list[str] = []

        for idx, desc in enumerate(subtask_descriptions, start=1):
            yield AgentEvent(
                type="subtask_start",
                data={"index": idx, "description": desc},
            )
            decision = await self.router.route(desc)
            yield AgentEvent(type="route", data={"decision": decision.model_dump()})

            client = self.registry.get(decision.tier)
            tools_schema = self.tools.to_openai() if self.s.agent_enable_tools else None

            answer: str = ""
            async for ev in self._react_loop(
                client, desc, running_msgs, tools_schema, decision.tier
            ):
                yield ev
                if ev.type == "answer":
                    answer = ev.data.get("content", "")
            running_msgs.append(
                Message(role="assistant", content=f"[subtask {idx}] {answer}")
            )
            overall_answer_parts.append(answer)
            await self.sessions.add_message(
                state.session_id,
                Message(role="assistant", content=f"[subtask {idx}] {answer}"),
            )

        final_text = (
            "\n\n".join(overall_answer_parts)
            if len(overall_answer_parts) > 1
            else (overall_answer_parts[0] if overall_answer_parts else "")
        )
        await self.sessions.add_message(
            state.session_id, Message(role="assistant", content=final_text)
        )
        yield AgentEvent(type="final", data={"content": final_text})

    # ------------------------------------------------------------------ react
    async def _react_loop(
        self,
        client: ModelClient,
        task: str,
        history: list[Message],
        tools_schema: list[dict] | None,
        tier: ModelTier,
    ) -> AsyncIterator[AgentEvent]:
        """ReAct loop for one subtask, streaming token deltas.

        Each round uses the streaming endpoint. If the model emits tool calls,
        we execute them and loop; otherwise the streamed text IS the answer.
        """
        msgs = _with_fresh_frame(history, task, tier)

        for iteration in range(self.s.agent_max_iterations):
            yield AgentEvent(type="iteration", data={"n": iteration + 1})
            content_chunks: list[str] = []
            tool_calls = None

            async for ev in client.stream(msgs, tools=tools_schema, temperature=0.2):
                if ev["type"] == "delta":
                    content_chunks.append(ev["content"])
                    yield AgentEvent(type="delta", data={"content": ev["content"]})
                elif ev["type"] == "tool_calls":
                    tool_calls = ev["tool_calls"]

            content = "".join(content_chunks)

            if not tool_calls:
                msgs.append(Message(role="assistant", content=content))
                yield AgentEvent(type="answer", data={"content": content})
                return

            # Tool calls -> execute, feed back, continue.
            msgs.append(
                Message(role="assistant", content=content, tool_calls=tool_calls)
            )
            for call in tool_calls:
                yield AgentEvent(
                    type="tool_call", data={"name": call.name, "arguments": call.arguments}
                )
                result: ToolResult = await self.tools.dispatch(call.name, call.arguments)
                yield AgentEvent(
                    type="tool_result",
                    data={
                        "name": call.name,
                        "ok": result.ok,
                        "output": result.output,
                        "error": result.error,
                    },
                )
                msgs.append(
                    Message(
                        role="tool",
                        content=result.output or result.error or "",
                        tool_call_id=f"call_{call.name}",
                        name=call.name,
                    )
                )

        # Exhausted iterations: one final non-tool summarize, streamed.
        content_chunks = []
        async for ev in client.stream(msgs, tools=None, temperature=0.2):
            if ev["type"] == "delta":
                content_chunks.append(ev["content"])
                yield AgentEvent(type="delta", data={"content": ev["content"]})
        content = "".join(content_chunks)
        yield AgentEvent(type="answer", data={"content": content})


# ----------------------------------------------------------------- helpers
def _with_fresh_frame(
    history: list[Message], task: str, tier: ModelTier
) -> list[Message]:
    """Re-seat the conversation on a fresh system + user frame, keeping the
    last few *user* turns for memory."""
    kept = [m for m in history if m.role == "user"]
    out: list[Message] = [Message(role="system", content=AGENT_SYSTEM)]
    out.extend(kept[-6:])
    out.append(
        Message(
            role="user",
            content=(
                f"Subtask to complete now: {task}\n"
                f"(Route: {tier.value} model. Use tools only if needed, "
                f"then answer.)"
            ),
        )
    )
    return out


def _looks_multi_step(task: str) -> bool:
    text = task.lower()
    markers = (
        " and then ", "after that", "step by step", "multi-step", "first ",
        "finally", "next,", "then,", " and also ",
    )
    return any(m in text for m in markers) or len(task) > 600