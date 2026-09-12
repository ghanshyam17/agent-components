"""Prompt-registry errors."""
from __future__ import annotations


class PromptError(Exception):
    """Base class for prompt-registry errors."""


class PromptNotFound(PromptError):
    """Raised when no prompt with the given name (and optional version) exists."""


class PromptVersionError(PromptError):
    """Raised on invalid version strings or conflicting registrations."""


class PromptRenderError(PromptError):
    """Raised when rendering a template fails (e.g. missing variable)."""