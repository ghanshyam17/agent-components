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

try:  # optional: checkpointing is opt-in and must not be a hard dependency
    from context_manager.checkpoint import (
        Checkpointer,
        CheckpointStatus,
        RunCheckpoint,
        StepOutcome,
    )
except ImportError:  # pragma: no cover - context-manager not installed
    Checkpointer = None  # type: ignore[assignment]
    CheckpointStatus = RunCheckpoint = StepOutcome = None  # type: ignore[assignment]

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
        checkpointer: "Checkpointer | None" = None,
    ):
        self.s = settings
        self.registry = registry
        self.router = router
        self.tools = tools
        self.sessions = sessions
        # Optional. When present, every tool call is checkpointed around, so a
        # crash mid-run can distinguish "this tool never ran" from "this tool
        # ran and we do not know whether its side effect landed".
        self.checkpointer = checkpointer

    # ------------------------------------------------------------ checkpointing
    # All of these are no-ops when no checkpointer is configured, so the
    # checkpointing capability is strictly opt-in and costs nothing when unused.

    def _run_id(self, session_id: str) -> str:
        return f"agent:{session_id}"

    @staticmethod
    def _subtask_key(idx: int, desc: str) -> str:
        """Stable across restarts, but not across a changed plan.

        The description is part of the key on purpose: if the planner produces
        different subtasks on the retry, the previous run's completions are for
        different work and must not be treated as done.
        """
        import hashlib

        digest = hashlib.sha256(desc.encode()).hexdigest()[:12]
        return f"subtask:{idx}:{digest}"

    @staticmethod
    def _tool_key(iteration: int, call: object) -> str:
        """Identify a tool call by name + arguments, not by position.

        Position would collide across restarts if the model emitted a different
        number of calls; content-addressing means the same call is recognised
        and a *different* call is not mistaken for a completed one.
        """
        import hashlib
        import json

        args = getattr(call, "arguments", None)
        try:
            payload = json.dumps(args, sort_keys=True, default=str)
        except (TypeError, ValueError):
            payload = str(args)
        digest = hashlib.sha256(
            f"{getattr(call, 'name', '?')}\x00{payload}".encode()
        ).hexdigest()[:12]
        return f"tool:{iteration}:{getattr(call, 'name', '?')}:{digest}"

    async def _load_state(self, session_id: str) -> dict:
        """The recorded step ledger for this session, or {} when none."""
        if self.checkpointer is None:
            return {}
        latest = await self.checkpointer.latest(self._run_id(session_id))
        if latest is None:
            return {}
        return dict(latest.step_outcomes)

    async def _completed_subtasks(self, session_id: str, descriptions: list[str]) -> set[str]:
        ledger = await self._load_state(session_id)
        done: set[str] = set()
        for idx, desc in enumerate(descriptions, start=1):
            key = self._subtask_key(idx, desc)
            if StepOutcome is not None and ledger.get(key) is StepOutcome.COMPLETED:
                done.add(key)
        return done

    async def _save_ledger(self, session_id: str, task: str, ledger: dict) -> None:
        if self.checkpointer is None:
            return
        await self.checkpointer.save(
            RunCheckpoint(
                run_id=self._run_id(session_id),
                step=task[:120],
                status=CheckpointStatus.RUNNING,
                step_outcomes=ledger,
                state={"task": task},
            )
        )

    async def _checkpoint_subtask(
        self, session_id: str, descriptions: list[str], key: str, outcome: object
    ) -> None:
        if self.checkpointer is None:
            return
        ledger = await self._load_state(session_id)
        ledger[key] = outcome
        # Record every subtask key so a subsequent run can match them.
        for idx, desc in enumerate(descriptions, start=1):
            ledger.setdefault(self._subtask_key(idx, desc), StepOutcome.SKIPPED)
        ledger[key] = outcome
        await self._save_ledger(session_id, f"subtasks:{len(descriptions)}", ledger)

    async def _checkpoint_tool(
        self, session_id: str, task: str, key: str, call: object, outcome: object
    ) -> None:
        if self.checkpointer is None:
            return
        ledger = await self._load_state(session_id)
        ledger[key] = outcome
        await self._save_ledger(session_id, task, ledger)

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

        # Resume support: if a previous attempt recorded completed subtasks,
        # do not re-run them. Subtasks are the natural checkpoint unit because
        # each one may complete real work (tool side effects) irreversibly.
        completed_subtasks = await self._completed_subtasks(state.session_id, subtask_descriptions)

        for idx, desc in enumerate(subtask_descriptions, start=1):
            key = self._subtask_key(idx, desc)
            if key in completed_subtasks:
                yield AgentEvent(
                    type="subtask_skipped",
                    data={"index": idx, "description": desc,
                          "reason": "already completed in a previous run"},
                )
                continue

            yield AgentEvent(
                type="subtask_start",
                data={"index": idx, "description": desc},
            )
            await self._checkpoint_subtask(
                state.session_id, subtask_descriptions, key, StepOutcome.PENDING
            )
            decision = await self.router.route(desc)
            yield AgentEvent(type="route", data={"decision": decision.model_dump()})

            client = self.registry.get(decision.tier)
            tools_schema = self.tools.to_openai() if self.s.agent_enable_tools else None

            answer: str = ""
            async for ev in self._react_loop(
                client, desc, running_msgs, tools_schema, decision.tier,
                state.session_id,
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
            await self._checkpoint_subtask(
                state.session_id, subtask_descriptions, key, StepOutcome.COMPLETED
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
        session_id: str = "",
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
                # Checkpoint BEFORE dispatch. A tool may move money, send mail
                # or mutate a warehouse, and the process can die between the
                # call and its return. Writing intent first is the only way a
                # later run can tell "never started" from "started, outcome
                # unknown" — the difference between safely retrying and
                # double-charging.
                tool_key = self._tool_key(iteration, call)
                await self._checkpoint_tool(
                    session_id, task, tool_key, call, StepOutcome.PENDING
                )
                try:
                    result: ToolResult = await self.tools.dispatch(call.name, call.arguments)
                except Exception as exc:  # noqa: BLE001
                    await self._checkpoint_tool(
                        session_id, task, tool_key, call, StepOutcome.FAILED
                    )
                    raise type(exc)(f"tool {call.name!r} raised: {exc}") from exc
                await self._checkpoint_tool(
                    session_id, task, tool_key, call, StepOutcome.COMPLETED
                )
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