"""prompt-registry: versioned, templated prompt management.

Register Jinja2 prompt templates by (name, version), resolve the latest
version by semver, render with strict variable checking, pick a version via
weighted A/B, and load a directory of prompts with hot-reload.

Quick start::

    from prompt_registry import build_registry, PromptTemplate

    reg = build_registry()
    reg.register(PromptTemplate(name="greet", version="1.0.0",
                                 template="Hello {{ name }}", variables=["name"]))
    out = reg.render("greet", name="Ada")
"""
from __future__ import annotations

from prompt_registry.config import PromptSettings, get_settings
from prompt_registry.errors import PromptError, PromptNotFound, PromptRenderError, PromptVersionError
from prompt_registry.prompt import PromptTemplate
from prompt_registry.registry import PromptRegistry

__all__ = [
    "PromptTemplate",
    "PromptRegistry",
    "PromptSettings",
    "PromptError", "PromptNotFound", "PromptRenderError", "PromptVersionError",
    "get_settings", "build_registry",
]


def build_registry(settings: PromptSettings | None = None) -> PromptRegistry:
    """Construct a PromptRegistry, loading from `settings.dir` if set."""
    s = settings or get_settings()
    reg = PromptRegistry()
    if s.dir:
        reg.load_dir(s.dir)
    return reg