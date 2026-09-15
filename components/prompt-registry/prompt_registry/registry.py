"""PromptRegistry — in-memory versioned prompt store with hot-reload and A/B."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any

from components_core import get_logger

from prompt_registry.errors import PromptNotFound, PromptVersionError
from prompt_registry.prompt import PromptTemplate

_log = get_logger("prompt_registry")

_VERSION_HEADER = re.compile(r"^#\s*version:\s*([^\s#]+)", re.MULTILINE)
_NAME_HEADER = re.compile(r"^#\s*name:\s*([^\s#]+)", re.MULTILINE)

_TEXT_SUFFIXES = {".j2", ".jinja", ".txt"}
_STRUCT_SUFFIXES = {".yaml", ".yml", ".json"}


def _parse_version(version: str) -> tuple[int, ...]:
    return tuple(int(p) for p in version.split("."))


class PromptRegistry:
    """In-memory registry of versioned prompt templates, keyed by (name, version).

    Supports directory loading (with hot-reload via ``reload()``) and a simple
    weighted A/B resolver that is deterministic given a seed.
    """

    def __init__(self, *, seed: int | None = None) -> None:
        self._store: dict[tuple[str, str], PromptTemplate] = {}
        self._dir: Path | None = None
        self._rng = random.Random(seed) if seed is not None else random.Random()
        self._seed = seed

    # ---------------- registration ----------------
    def register(self, template: PromptTemplate) -> None:
        """Register a template, replacing any existing (name, version)."""
        key = (template.name, template.version)
        self._store[key] = template
        _log.debug("registered %s v%s", template.name, template.version)

    def get(self, name: str, version: str) -> PromptTemplate:
        """Fetch an exact (name, version) or raise PromptNotFound."""
        key = (name, version)
        if key not in self._store:
            raise PromptNotFound(f"no prompt {name!r} v{version}")
        return self._store[key]

    def latest(self, name: str) -> PromptTemplate:
        """Return the highest semver among registered versions of `name`."""
        versions = self.versions(name)
        if not versions:
            raise PromptNotFound(f"no prompt named {name!r}")
        top = max(versions, key=_parse_version)
        return self.get(name, top)

    def list(self, name: str | None = None) -> list[PromptTemplate]:
        """List templates, optionally filtered by name."""
        if name is None:
            return sorted(
                self._store.values(),
                key=lambda t: (t.name, _parse_version(t.version)),
            )
        return sorted(
            (t for t in self._store.values() if t.name == name),
            key=lambda t: _parse_version(t.version),
        )

    def versions(self, name: str) -> list[str]:
        """Return all registered version strings for `name`, sorted ascending by semver."""
        vs = [k[1] for k in self._store if k[0] == name]
        return sorted(vs, key=_parse_version)

    # ---------------- rendering ----------------
    def render(
        self, prompt_name: str, version: str | None = None, **vars: Any
    ) -> str:
        """Render a prompt; version=None selects the latest registered version.

        The prompt is identified by `prompt_name` (not `name`) because `name`
        is a very common template *variable* and would otherwise collide::

            reg.render("greet", name="Ada")            # latest version
            reg.render("greet", version="1.0.0", name="Bo")
        """
        tpl = self.latest(prompt_name) if version is None else self.get(prompt_name, version)
        return tpl.render(**vars)

    # ---------------- A/B ----------------
    def ab(
        self, name: str, weights: dict[str, float] | None = None
    ) -> PromptTemplate:
        """Pick a version of `name` by weight (default uniform across versions).

        Deterministic given the registry's `seed`: the internal `random.Random`
        is reused, so consecutive calls advance the same stream. To reproduce a
        single draw, construct a fresh ``PromptRegistry(seed=...)`` per draw.
        """
        versions = self.versions(name)
        if not versions:
            raise PromptNotFound(f"no prompt named {name!r}")
        if weights is None:
            weights = {v: 1.0 for v in versions}
        # Validate: every version covered (extra keys are ignored with a warning).
        missing = [v for v in versions if v not in weights]
        if missing:
            raise PromptVersionError(
                f"weights missing for versions {missing} of prompt {name!r}"
            )
        population = list(versions)
        w = [max(0.0, float(weights[v])) for v in population]
        if sum(w) <= 0:
            raise PromptVersionError(f"weights sum to zero for prompt {name!r}")
        chosen = self._rng.choices(population, weights=w, k=1)[0]
        return self.get(name, chosen)

    # ---------------- file loading ----------------
    def load_dir(self, path: str | Path) -> None:
        """Load prompt templates from a directory (see README for file formats)."""
        p = Path(path)
        self._dir = p
        self._load_from(p)
        _log.info("loaded prompts from %s", p)

    def reload(self) -> None:
        """Re-read the directory supplied to ``load_dir`` (hot-reload)."""
        if self._dir is None:
            raise PromptNotFound("reload() called but no directory was loaded")
        # Clear current entries that came from disk; simplest: clear all and reload.
        self._store.clear()
        self._load_from(self._dir)
        _log.info("reloaded prompts from %s", self._dir)

    def _load_from(self, p: Path) -> None:
        if not p.is_dir():
            raise PromptNotFound(f"prompt directory not found: {p}")
        for f in sorted(p.iterdir()):
            if not f.is_file():
                continue
            suffix = f.suffix.lower()
            if suffix in _TEXT_SUFFIXES:
                self._load_text(f)
            elif suffix in _STRUCT_SUFFIXES:
                self._load_struct(f)

    # -- text files (.j2/.jinja/.txt) --
    # Format (tolerant):
    #   # name: greet          (optional; defaults to stem)
    #   # version: 1.0.0       (optional; defaults to "1.0.0")
    #   <jinja2 template body>
    # Alternatively, a v-prefixed filename like `greet.v2.j2` supplies version "2".
    def _load_text(self, f: Path) -> None:
        text = f.read_text(encoding="utf-8")
        name_match = _NAME_HEADER.search(text)
        version_match = _VERSION_HEADER.search(text)
        name = name_match.group(1) if name_match else f.stem.split(".")[0]
        version = version_match.group(1) if version_match else "1.0.0"
        # Strip header comment lines from the template body for cleanliness.
        body = _VERSION_HEADER.sub("", text)
        body = _NAME_HEADER.sub("", body)
        # Remove the blank lines left behind where headers were, plus a
        # trailing newline that text editors commonly append. Leading/trailing
        # newlines are rarely meaningful for a prompt body; preserve interior.
        body = body.strip("\n")
        try:
            tpl = PromptTemplate(
                name=name, version=version, template=body,
                variables=sorted(meta_find_undeclared(body)),
            )
        except Exception as e:  # noqa: BLE001 - tolerant loading
            _log.warning("skipping %s: %s", f, e)
            return
        self.register(tpl)

    # -- structured files (.yaml/.yml/.json) --
    # Format: a single object (or a list of objects) with PromptTemplate fields:
    #   name, version, template, description, variables, metadata, tags
    def _load_struct(self, f: Path) -> None:
        data: Any
        if f.suffix.lower() in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore
            except ImportError as e:  # pragma: no cover
                raise RuntimeError(
                    "YAML prompt files require the 'yaml' extra: "
                    "pip install prompt-registry[yaml]"
                ) from e
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        else:
            data = json.loads(f.read_text(encoding="utf-8"))
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                _log.warning("skipping non-object entry in %s", f)
                continue
            try:
                self.register(PromptTemplate.model_validate(item))
            except Exception as e:  # noqa: BLE001
                _log.warning("skipping entry in %s: %s", f, e)


def meta_find_undeclared(template_src: str) -> set[str]:
    """Best-effort: return the Jinja2-undeclared variable names in a source string."""
    from jinja2 import Environment, meta

    env = Environment()
    try:
        ast = env.parse(template_src)
    except Exception:
        return set()
    return meta.find_undeclared_variables(ast)