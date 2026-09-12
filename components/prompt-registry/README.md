# prompt-registry

Versioned, templated prompt management for AI agents: register Jinja2 prompt
templates by `(name, version)`, resolve the **latest** version by semver,
render with strict variable checking, pick a variant via weighted **A/B**, and
load a directory of prompt files with **hot-reload**.

Part of the [agent-components](../..) monorepo.

## What's inside

| Concept | Surface |
|--------|---------|
| Versioned template | `PromptTemplate` (pydantic v2 model) — `name`, `version` (semver-ish), Jinja2 `template`, `variables`, `metadata`, `tags` |
| Registry | `PromptRegistry` — in-memory dict keyed by `(name, version)` |
| Resolution | `latest(name)` (highest semver), `render(name, version=None, **vars)` |
| A/B | `ab(name, weights=None)` — weighted, deterministic with a seed |
| Loading | `load_dir(path)` — `.j2`/`.jinja`/`.txt`/`.yaml`/`.json` files; `reload()` for hot-reload |

Templates render with Jinja2 `StrictUndefined`: missing variables raise a
`PromptRenderError` instead of silently rendering empty. On construction, the
`variables` list is best-effort validated to cover the Jinja2-undeclared names.

## Install

```bash
uv sync                                # in the monorepo
# or, once published:
pip install prompt-registry             # core (jinja2 templating)
pip install "prompt-registry[yaml]"     # + YAML prompt files
```

## Usage

### Register + render

```python
from prompt_registry import PromptTemplate, PromptRegistry

reg = PromptRegistry()
reg.register(PromptTemplate(
    name="greet", version="1.0.0",
    template="Hello {{ name }}, you are {{ role }}.",
    variables=["name", "role"],
))
reg.register(PromptTemplate(
    name="greet", version="1.2.0",
    template="Hi {{ name }}! (role: {{ role }})",
    variables=["name", "role"],
))

# latest() picks the highest semver:
assert reg.latest("greet").version == "1.2.0"

# render resolves latest when version is omitted:
print(reg.render("greet", name="Ada", role="engineer"))
# Hi Ada! (role: engineer)

# exact version:
print(reg.render("greet", version="1.0.0", name="Ada", role="engineer"))

# missing variables raise:
from prompt_registry import PromptRenderError
try:
    reg.render("greet", name="Ada")
except PromptRenderError:
    ...
```

### A/B pick

```python
reg = PromptRegistry(seed=42)
reg.register(PromptTemplate(name="ad", version="1.0.0", template="A: {{ x }}", variables=["x"]))
reg.register(PromptTemplate(name="ad", version="2.0.0", template="B: {{ x }}", variables=["x"]))

# Default: uniform across versions. Deterministic given the seed.
pick = reg.ab("ad")

# Weighted:
pick = reg.ab("ad", weights={"1.0.0": 0.9, "2.0.0": 0.1})
```

> Determinism: `ab()` reuses the registry's internal `random.Random`. For a
> reproducible single draw, construct a fresh `PromptRegistry(seed=...)`.

### Load a directory

```python
from prompt_registry import build_registry

reg = build_registry()                 # uses PROMPT_DIR env if set
reg.load_dir("./prompts")               # or load explicitly
reg.reload()                            # hot-reload from the same directory
```

## File formats

`load_dir(path)` reads these extensions (others are ignored):

**Text templates** — `.j2`, `.jinja`, `.txt`. Name and version are derived from
optional header comments, with sensible defaults:

```
# name: greet
# version: 1.2.0
Hello {{ name }}!
```

| Header        | Default                                   |
|---------------|-------------------------------------------|
| `# name: X`   | filename stem (before the first `.`)      |
| `# version: X`| `1.0.0`                                   |

Header lines are stripped from the rendered template body. The `variables`
list is inferred (best-effort) from the Jinja2-undeclared names.

**Structured** — `.yaml`/`.yml` (requires the `yaml` extra) and `.json`. Each
file is a single object or a list of objects with `PromptTemplate` fields:

```yaml
- name: greet
  version: "1.2.0"
  template: "Hi {{ name }}!"
  variables: ["name"]
  description: "Greeting prompt"
  tags: ["v2"]
- name: summarize
  version: "1.0.0"
  template: "Summarize: {{ text }}"
  variables: ["text"]
```

```json
[
  {"name": "greet", "version": "1.0.0", "template": "Hello {{ name }}", "variables": ["name"]}
]
```

Malformed entries are skipped with a logged warning (tolerant loading).

## Configuration (env, `PROMPT_` prefix)

| Var                    | Default   | Meaning                                        |
|------------------------|-----------|------------------------------------------------|
| `PROMPT_DIR`           | unset     | directory of prompt files loaded by `build_registry` |
| `PROMPT_DEFAULT_VERSION` | `latest` | reserved — default version selector           |

## Notes

- Versions are compared as semver tuples (`1.12.0` > `1.2.0`); each dot-separated
  part must be numeric.
- The registry is in-memory; persistence belongs to a future component.
- `reload()` clears and re-reads the last `load_dir` path.

## License

MIT