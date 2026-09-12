"""PromptTemplate — a versioned, Jinja2-templated prompt."""
from __future__ import annotations

from typing import Any

from jinja2 import Environment, StrictUndefined, meta
from jinja2.exceptions import TemplateError, TemplateSyntaxError
from pydantic import BaseModel, Field, model_validator

from prompt_registry.errors import PromptRenderError, PromptVersionError

# A single shared environment: StrictUndefined so missing variables raise
# instead of rendering as the empty string.
_ENV = Environment(undefined=StrictUndefined, autoescape=False)


class PromptTemplate(BaseModel):
    """A versioned prompt template.

    ``template`` is a Jinja2 source string rendered with ``render(**vars)``.
    ``variables`` should list the names the template expects; on construction
    we best-effort check that it covers the Jinja2-undeclared names.
    """

    name: str
    version: str = Field(description="semver-ish, e.g. '1.2.0'")
    template: str
    description: str = ""
    variables: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate(self) -> "PromptTemplate":
        # Validate version shape loosely: dot-separated numeric parts.
        parts = self.version.split(".")
        if not parts or not all(p.isdigit() for p in parts):
            raise PromptVersionError(
                f"invalid version {self.version!r}: expected dot-separated "
                "numeric parts, e.g. '1.2.0'"
            )
        # Compile early to catch syntax errors and find undeclared variables.
        try:
            ast = _ENV.parse(self.template)
        except TemplateSyntaxError as e:
            raise PromptRenderError(
                f"template {self.name!r} v{self.version}: invalid Jinja2 syntax: {e.message}"
            ) from e
        undeclared = meta.find_undeclared_variables(ast)
        missing = sorted(undeclared - set(self.variables))
        if missing:
            raise PromptRenderError(
                f"template {self.name!r} v{self.version}: variables {missing} "
                "are used in the template but not declared in `variables`"
            )
        return self

    def render(self, **vars: Any) -> str:
        """Render the template, raising PromptRenderError on missing vars."""
        try:
            tpl = _ENV.from_string(self.template)
            return tpl.render(**vars)
        except TemplateError as e:
            raise PromptRenderError(
                f"failed to render template {self.name!r} v{self.version}: {e}"
            ) from e