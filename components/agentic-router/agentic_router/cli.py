"""CLI for the agentic router.

Usage:
  agentic-router chat "Summarize this article: ..."
  agentic-router stream "Plan and execute: read README.md and list its top 3 topics"
  agentic-router route "debug this stack trace: ..."
  agentic-router serve [--host 0.0.0.0 --port 8000]
"""
from __future__ import annotations

import json

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from agentic_router.factory import build_agent
from agentic_router.models import AgentEvent

app = typer.Typer(add_completion=False, help="vLLM-backed agentic router.")
console = Console()


@app.command()
def route(task: str) -> None:
    """Show the router's decision for a task without running the agent."""
    import asyncio

    agent = build_agent()
    decision = asyncio.run(agent.router.route(task))

    table = Table(title="Route Decision", show_header=False)
    table.add_column("field", style="cyan")
    table.add_column("value")
    table.add_row("tier", decision.tier.value)
    table.add_row("model", decision.model)
    table.add_row("method", decision.method)
    table.add_row("score", f"{decision.score:.3f}")
    table.add_row("reason", decision.reason)
    for k, v in decision.signals.items():
        table.add_row(f"signal:{k}", f"{v:.3f}")
    console.print(table)


@app.command()
def chat(task: str, session_id: str | None = None) -> None:
    """Run the agent once and print the answer + route."""
    import asyncio

    agent = build_agent()
    answer, decision, state = asyncio.run(agent.run(task, session_id))
    if decision:
        console.print(
            Panel(
                f"[bold]{decision.tier.value}[/bold] → {decision.model} "
                f"(method={decision.method}, score={decision.score:.2f})",
                title="Route",
            )
        )
    console.print(Panel(answer, title="Answer"))
    console.print(f"[dim]session_id={state.session_id}[/dim]")


@app.command()
def stream(task: str, session_id: str | None = None) -> None:
    """Run the agent and live-print streamed events."""
    import asyncio

    agent = build_agent()
    state = asyncio.run(agent.sessions.get_or_create(session_id))

    async def run():
        with Live(console=console, refresh_per_second=30) as live:
            buffer: list[str] = []
            async for ev in agent.stream(task, state.session_id):
                _render_event(ev, live, buffer)

    asyncio.run(run())
    console.print(f"\n[dim]session_id={state.session_id}[/dim]")


@app.command()
def serve(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Start the FastAPI server."""
    import uvicorn

    uvicorn.run("agentic_router.server.app:app", host=host, port=port, reload=False)


def _render_event(ev: AgentEvent, live: Live, buffer: list[str]) -> None:
    if ev.type == "route":
        d = ev.data.get("decision", {})
        console.print(
            f"[bold cyan]route[/bold cyan] -> {d.get('tier')} ({d.get('model')}, "
            f"method={d.get('method')}, score={d.get('score'):.2f})"
        )
    elif ev.type == "plan":
        st = ev.data["plan"]["subtasks"]
        console.print(f"[magenta]plan[/magenta] ({len(st)} subtasks):")
        for i, s in enumerate(st, 1):
            console.print(f"  {i}. {s['description']}")
    elif ev.type == "subtask_start":
        console.print(f"[yellow]subtask {ev.data['index']}[/yellow]: {ev.data['description']}")
    elif ev.type == "iteration":
        pass
    elif ev.type == "delta":
        buffer.append(ev.data["content"])
        live.update(Panel("".join(buffer), title="streaming"))
    elif ev.type == "tool_call":
        console.print(f"[green]tool_call[/green] {ev.data['name']} {json.dumps(ev.data['arguments'])}")
    elif ev.type == "tool_result":
        ok = ev.data.get("ok")
        out = ev.data.get("output") or ev.data.get("error")
        console.print(f"[{'green' if ok else 'red'}]tool_result[/] {ev.data['name']}: {out}")
    elif ev.type == "answer":
        buffer.clear()
        console.print(Panel(ev.data["content"], title="answer"))
    elif ev.type == "final":
        console.print(Panel(ev.data["content"], title="final"))
    elif ev.type == "error":
        console.print(f"[red]error[/red] {json.dumps(ev.data)}")


if __name__ == "__main__":
    app()