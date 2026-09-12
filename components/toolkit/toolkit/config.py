"""Settings and a factory that assembles a registry from env config."""
from __future__ import annotations

from typing import Any

from pydantic_settings import BaseSettings, SettingsConfigDict

from toolkit.base import Registry
from toolkit.builtin import default_tools


class ToolkitSettings(BaseSettings):
    """Environment configuration for the toolkit (prefix `TOOLKIT_`).

    `sandbox_root` confines file tools to a directory; an empty value disables
    sandboxing. `shell_allow` / `shell_deny` are comma-separated program names
    / substrings.
    """

    model_config = SettingsConfigDict(env_prefix="TOOLKIT_", extra="ignore")

    sandbox_root: str | None = None
    shell_allow: str | None = None
    shell_deny: str | None = None
    shell_timeout: float = 30.0
    max_output: int = 8000
    file_max_bytes: int = 200_000

    def _overrides(self) -> dict[str, Any]:
        return {
            "sandbox_root": self.sandbox_root,
            "shell_allow": [s for s in (self.shell_allow or "").split(",") if s] or None,
            "shell_deny": [s for s in (self.shell_deny or "").split(",") if s] or None,
            "shell_timeout": self.shell_timeout,
            "max_output": self.max_output,
            "file_max_bytes": self.file_max_bytes,
        }


def build_registry(settings: ToolkitSettings | None = None) -> Registry:
    """Assemble a registry pre-loaded with the hardened built-in tools."""
    s = settings or ToolkitSettings()
    reg = Registry()
    for tool in default_tools(**s._overrides()):
        reg.register(tool)
    return reg