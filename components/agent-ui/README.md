# agent-ui

A universal, metadata-driven UI bridge and frontend for agent applications. Provides an embeddable Microsoft Fluent-styled web interface, Server-Sent Events (SSE) streaming for real-time agent responses and tool execution steps, an interactive Component Graph topology visualizer, and connectors for Power Automate and Azure Managed Grafana.

---

## Highlights

- **Metadata-Driven**: Exposes `/api/config` and `/api/graph` so the frontend adapts dynamically to any agent's name, description, capabilities, starter prompts, and component graph topology without writing custom frontend code.
- **Real-Time Streaming**: Server-Sent Events (SSE) over `/api/chat` delivers chunked token streaming alongside live tool execution call cards.
- **Enterprise Integrations**:
  - **Power Automate**: Pre-packaged `openapi.json` custom connector definition to trigger agent chat sessions or query graph configurations from Microsoft 365 / Power Platform flows.
  - **Azure Managed Grafana**: Pre-configured `dashboards/agent_observability.json` dashboard for tracking latency, token usage, error rates, and tool invocations.
- **Embeddable & Zero-Build**: Mounts directly onto any existing FastAPI application with one line of code (`mount_agent_ui(app)`), serving pure vanilla HTML/JS with Microsoft Fluent Design System styling.

---

## Quickstart

### 1. Minimal Standalone Server

```python
import asyncio
from agent_ui import create_ui_app

async def mock_handler(message: str, session_id: str):
    yield {"event": "tool", "data": {"name": "retriever", "query": message}}
    await asyncio.sleep(0.05)
    for word in f"Echoing back your message: {message}".split():
        yield {"event": "token", "data": word + " "}
        await asyncio.sleep(0.02)
    yield {"event": "done", "data": {"status": "success"}}

app = create_ui_app(
    name="Support Assistant",
    description="Customer support agent with real-time knowledge retrieval.",
    handler=mock_handler,
    starter_prompts=["How do I reset my password?", "Check system status"],
)

# Run with uvicorn:
# uvicorn my_module:app --port 8000
```

### 2. Mounting into an Existing FastAPI App

```python
from fastapi import FastAPI
from agent_ui import mount_agent_ui

app = FastAPI(title="My Corporate Enterprise AI")

# ... existing routes ...

mount_agent_ui(
    app,
    name="Corporate Copilot",
    handler=my_agent_stream_handler,
    mount_path="/ui",
)
```

---

## API Endpoints

| Method | Path | Description |
| --- | --- | --- |
| `GET` | `/` | Web UI single-page application (Microsoft Fluent Design) |
| `GET` | `/api/config` | Returns agent metadata, starter prompts, and UI configuration |
| `GET` | `/api/graph` | Returns nodes and edges representing the Component Graph topology |
| `POST` | `/api/chat` | SSE streaming endpoint accepting `{"message": str, "session_id": str}` |

---

## Power Automate Integration

Import `openapi.json` directly into **Power Apps / Power Automate -> Custom Connectors -> Create from OpenAPI file**. This enables automated workflows to invoke agent executions, query agent configuration, and receive structured JSON responses.

---

## Azure Managed Grafana Dashboard

Import `dashboards/agent_observability.json` into Azure Managed Grafana or self-hosted Grafana to instantly visualize:
- Request throughput and SSE connection count
- TTFT (Time to First Token) and P95/P99 latency
- Token consumption (Prompt vs Completion)
- Tool execution status and failure heatmaps
