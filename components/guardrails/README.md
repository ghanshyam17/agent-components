# guardrails

Input/output safety for LLM applications: keyword filtering, PII redaction,
prompt-injection detection, length limits, and output schema validation —
composable as a pipeline.

Part of the [agent-components](../..) monorepo.

## What's inside

| Guardrail | Direction | Behavior |
|----------|-----------|----------|
| `KeywordBlocklist` | input | block if any banned keyword is present (`flag="blocked_keyword"`) |
| `PIIRedactor` | input | replace email/phone/SSN/credit-card patterns with `[REDACTED:<kind>]`; always allowed (`flag="pii"`) |
| `InjectionDetector` | input | heuristic prompt-injection / jailbreak cues; blocks on match by default (`flag="injection"`) |
| `LengthGuard` | input | block input above a character budget (`flag="too_long"`) |
| `OutputSchemaValidator` | output | validate model output as JSON against a JSON Schema (`flag="schema"`) |

A `GuardrailPipeline` holds an ordered list of input guardrails and an
optional list of output guardrails. Guardrails run in order; each sees the
running sanitized text and the pipeline short-circuits on the first block.

## Install

```bash
uv sync                                # in the monorepo
# or, once published:
pip install guardrails                 # core guardrails
pip install "guardrails[jsonschema]"   # + OutputSchemaValidator
```

## Usage

### Input + output pipeline

```python
from guardrails import (
    GuardrailPipeline, KeywordBlocklist, PIIRedactor,
    InjectionDetector, LengthGuard, OutputSchemaValidator,
)

pipeline = GuardrailPipeline(
    input_guardrails=[
        LengthGuard(max_chars=4000),
        InjectionDetector(block_on_match=True),
        PIIRedactor(),
        KeywordBlocklist(keywords=["secret", "password"]),
    ],
    output_guardrails=[
        OutputSchemaValidator(schema={
            "type": "object",
            "properties": {"answer": {"type": "string"}},
            "required": ["answer"],
        }),
    ],
)

# On the way in:
result = await pipeline.check_input("Email me at alice@example.com")
assert result.allowed
print(result.sanitized)   # "Email me at [REDACTED:email]"
print(result.flags)       # ["pii"]

# On the way out:
out = await pipeline.check_output('{"answer": "42"}')
assert out.allowed
```

### From settings (env-driven)

```python
from guardrails import build_pipeline
pipeline = build_pipeline()   # reads GUARD_* env vars
```

## Configuration (env, `GUARD_` prefix)

| Var | Default | Meaning |
|-----|---------|---------|
| `GUARD_MAX_CHARS` | `8000` | input character budget |
| `GUARD_BLOCK_INJECTION` | `true` | block on injection cues (vs. flag only) |
| `GUARD_REDACT_PII` | `true` | run the PII redactor |
| `GUARD_BLOCKED_KEYWORDS` | unset | comma-separated list of banned keywords |

`build_pipeline()` assembles input guardrails in the order: length, injection,
pii (if enabled), keywords (if any). Output guardrails are empty by default —
add an `OutputSchemaValidator` explicitly when you have a schema to enforce.

## Notes

- `OutputSchemaValidator` imports `jsonschema` lazily; the rest of the package
  works without it. Install the `jsonschema` extra to use that guardrail.
- `InjectionDetector` is a heuristic regex set, not a classifier. It favors
  precision for obvious cues; tune `block_on_match=False` to log-only, or
  subclass and add patterns.

## License

MIT