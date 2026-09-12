# toolkit

A composable, hardened **tool-calling framework** for AI agents. It provides a
small, well-typed `Tool` / `Registry` abstraction that renders the OpenAI
function-calling schema, a set of sandboxed built-in tools (shell, file
read/write, web fetch, http request, calculator), a `tool_from_function`
decorator for one-line custom tools, and a factory that assembles everything
from env config.

Designed to be the reusable tool layer for any agent in the monorepo (the
`agentic-router` keeps its own inline tools for now; `toolkit` is the hardened
replacement available for new projects).

## Install

Part of the `agent-components` uv workspace — `uv sync --all-packages` makes it
importable as `toolkit`.

## Quick start

```python
import asyncio
from toolkit import build_registry

reg = build_registry()
print(reg.names())           # ['run_shell','read_file','write_file','web_fetch','http_request','calculator']

async def main():
    schema = reg.to_openai()
    res = await reg.dispatch("calculator", {"expression": "12*(3+4)/2"})
    print(res.ok, res.output)   # True '42.0'
asyncio.run(main())
```

## Custom tools

```python
from toolkit import tool_from_function, Registry

@tool_from_function
async def get_weather(city: str, units: str = "c") -> str:
    """Get the current weather for a city."""
    ...  # returns the string the model will see

reg = Registry()
reg.register(get_weather)
reg.to_openai()   # OpenAI tool schema derived from the signature
```

## Sandbox

- `FileReadTool` / `FileWriteTool` confine access to `sandbox_root`
  (`TOOLKIT_SANDBOX_ROOT`); paths escaping it raise `SandboxError` and return a
  failed `ToolResult`.
- `ShellTool` honours `TOOLKIT_SHELL_ALLOW` (comma-separated program names) and
  `TOOLKIT_SHELL_DENY` (substring blocklist; defaults to `rm -rf /`, `sudo`).

## Configuration

Environment variables (prefix `TOOLKIT_`):

| Var                    | Default | Description                                |
|------------------------|---------|--------------------------------------------|
| `TOOLKIT_SANDBOX_ROOT` | (none)  | Root directory for file tools (empty = off)|
| `TOOLKIT_SHELL_ALLOW`  | (none)  | Comma list of allowed program names         |
| `TOOLKIT_SHELL_DENY`   | (built) | Comma substring blocklist for shell commands |
| `TOOLKIT_SHELL_TIMEOUT`| `30`    | Default shell timeout (seconds)            |
| `TOOLKIT_MAX_OUTPUT`   | `8000`  | Max chars returned by shell/web/http tools  |
| `TOOLKIT_FILE_MAX_BYTES` | `200000` | Max bytes read by `read_file`            |