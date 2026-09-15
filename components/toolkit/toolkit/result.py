"""Tool result and errors shared across the toolkit."""
from __future__ import annotations

import warnings
from typing import Any, TypeVar

from pydantic import BaseModel, Field

_ResultT = TypeVar("_ResultT", bound="_ToolResultFactories")


class _ToolResultFactories:
    """`ok()` / `fail()` constructors, kept *out* of the pydantic model body.

    These live on a separate base class because pydantic ≥2.13 discards a
    classmethod whose name collides with a model field. Declaring the factory
    inside the model instead::

        class ToolResult(BaseModel):
            ok: bool = True
            @classmethod
            def ok(cls, ...): ...      # silently dropped by pydantic

    leaves `ToolResult.ok` unbound, so every ``ToolResult.ok(...)`` call site
    raises ``AttributeError: ok``. Inheriting the factories from a plain
    (non-pydantic) base keeps the *field* named `ok` and the *constructor*
    available. On instances the field wins: ``res.ok`` is a ``bool``.

    The methods are typed loosely (``_ResultT``) because this base has no
    knowledge of the concrete model's fields; subclasses supply them.
    """

    @classmethod
    def ok(
        cls: type[_ResultT], name: str, arguments: dict[str, Any],
        output: str = "", **meta: Any,
    ) -> _ResultT:
        """Build a successful result; extra kwargs land in `metadata`."""
        # Built as a dict because this base cannot know the concrete model's
        # fields; the subclass declares them.
        fields: dict[str, Any] = {
            "name": name, "arguments": arguments,
            "output": output, "metadata": meta,
        }
        return cls(**fields)

    @classmethod
    def fail(
        cls: type[_ResultT],
        name: str,
        arguments: dict[str, Any],
        error: str,
        output: str = "",
        **meta: Any,
    ) -> _ResultT:
        """Build a failed result; extra kwargs land in `metadata`."""
        fields: dict[str, Any] = {
            "name": name, "arguments": arguments, "output": output,
            "ok": False, "error": error, "metadata": meta,
        }
        return cls(**fields)


# pydantic warns that the `ok` field shadows the inherited `ok` classmethod.
# That shadowing is deliberate (the field must win on instances, the method on
# the class), so suppress the warning for just this class definition.
with warnings.catch_warnings():
    warnings.filterwarnings(
        "ignore",
        message=r'Field name "ok" in "ToolResult" shadows an attribute in parent',
        category=UserWarning,
    )

    class ToolResult(_ToolResultFactories, BaseModel):
        """Outcome of a tool execution.

        `output` is the human/LLM-facing string. `ok=False` together with
        `error` signals a failure; the loop surfaces `error` (or `output`) to
        the model.
        """

        name: str
        arguments: dict[str, Any]
        output: str = ""
        # Shadows the inherited `ok()` classmethod on purpose: the field wins
        # on instances (`res.ok` is a bool), the factory on the class.
        ok: bool = True  # pyright: ignore[reportIncompatibleMethodOverride]
        error: str | None = None
        # Optional structured data a caller may inspect (not sent to the model).
        metadata: dict[str, Any] = Field(default_factory=dict)


class ToolError(Exception):
    """Raised by tool implementations for non-recoverable errors."""


class UnknownToolError(ToolError):
    def __init__(self, name: str) -> None:
        super().__init__(f"unknown tool: {name!r}")
        self.name = name
