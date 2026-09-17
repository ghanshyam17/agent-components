# doc-processing

**Intelligent document processing** for agent pipelines. Turns files into
validated, auditable structured data.

A chassis child extension: consumes `components-core`, `model-gateway`,
`prompt-registry`, `guardrails` and `eval-harness`, and reimplements none of
them.

```
   ┌──────────────┐      ┌──────────────┐      ┌──────────────┐
   │  ① INGEST    │      │ ② EXTRACT    │      │ ③ VALIDATE   │
   ├──────────────┤      ├──────────────┤      ├──────────────┤
   │ OCR + layout │─────▶│ schema fields│─────▶│ field rules  │
   │ tables, spans│      │ + provenance │      │ cross-checks │
   │ OCR conf.    │      │ + confidence │      │ document gates│
   └──────────────┘      └──────────────┘      └──────┬───────┘
                                                       ▼
                                        PASS │ REVIEW │ FAIL
                                        (the routing signal)
```

## Install

```bash
uv sync --all-packages                      # offline engines only
pip install "doc-processing[azure]"          # Azure AI Document Intelligence
pip install "doc-processing[local]"          # local PDF text-layer extraction
```

## Quick start — offline (no Azure, no OCR stack)

```python
import asyncio
from doc_processing import DocumentPipeline, INVOICE_SCHEMA, StubEngine

text = """INVOICE
Invoice No: INV-2024-0042
Invoice Date: 15/03/2026
Subtotal: 1,200.00
VAT: 240.00
Total: 1,440.00"""

pipe = DocumentPipeline(INVOICE_SCHEMA, engine=StubEngine(text=text))
result = asyncio.run(pipe.process("invoice.txt"))

result.validation.state.value          # 'pass'
result.extraction.get("total")         # 1440.0
result.extraction.fields["total"].evidence.method.value   # 'regex'
```

## Quick start — live Azure AI Document Intelligence

Entra auth by default (no key to rotate or leak):

```python
from doc_processing import AzureDocumentIntelligenceEngine, DocumentPipeline, INVOICE_SCHEMA

engine = AzureDocumentIntelligenceEngine(
    "https://<resource>.cognitiveservices.azure.com",
    model="prebuilt-invoice",       # or prebuilt-layout / a custom model id
)
pipe = DocumentPipeline(INVOICE_SCHEMA, engine=engine)
result = await pipe.process("scanned-invoice.pdf")
```

Authentication order: `DefaultAzureCredential` → explicit API key (only if you
pass one). The endpoint works from an `AIServices`-kind Cognitive Services
account — Document Intelligence is served on the same host, so **no separate
resource is needed**.

## The three engines

| Engine | OCR | Use when |
|---|---|---|
| `AzureDocumentIntelligenceEngine` | ✅ service-side | scans, photos, handwriting, prebuilt models |
| `TextLayerEngine` | ❌ | born-digital PDFs / text (exact, and warns on scanned pages) |
| `StubEngine` | ❌ | tests and CI — deterministic, dependency-free |

`select_engine()` picks the best available and **logs which and why**. Prefer
Azure → text layer → stub. The text-layer engine records a warning when a page
yields almost no text, because that is the signature of a scanned page — without
it, a scan parses to an empty document that looks like a valid extraction with
no fields.

## Provenance is not optional

Every extracted value carries a `FieldEvidence`: which strategy found it, the
page, the bounding polygon, the matched pattern, the source text, and the
underlying engine.

```python
f = result.extraction.fields["invoice_number"]
f.value           # 'INV-2024-0042'
f.confidence      # 0.85
f.evidence.method # ExtractionMethod.REGEX
f.evidence.pattern
f.evidence.source_text
f.evidence.bbox.as_dict   # {'page': 1, 'polygon': [...]}
```

An extracted value without provenance cannot be reviewed, audited, or safely
corrected. In a regulated workflow that makes it worthless — which is why
nothing in this component is extracted without it.

## Extraction strategies, in order of trust

The order **is** the design: strategies are tried most-trustworthy first, and a
later strategy only fills gaps a previous one left.

| # | Strategy | Confidence | Why there |
|---|---|---|---|
| 1 | `DOCUMENT_INTELLIGENCE` | service-reported | the service already located and typed the field |
| 2 | `REGEX` | 0.75 – 0.85 | deterministic and explainable — the pattern is recorded |
| 3 | `TABLE` | 0.70 | line items, keyed by header row |
| 4 | `HEURISTIC` | 0.55 | `"label: value"` proximity — an honest guess |
| 5 | `LLM` | ≤ 0.60 | the long tail, and **never overwrites** a deterministic match |

The LLM fallback is asked only for fields still missing, and its confidence is
capped below the regex tier: a plausible-sounding model answer must not outrank
an exact match.

The built-in regexes are deliberately conservative. A pattern that matches too
eagerly produces *confident wrong* values, which is worse than missing ones —
missing values get reviewed, wrong ones do not.

## Validation: three layers

Most real document errors are **relationships**, not values. A line-item total
that does not reconcile with the stated subtotal is invisible to any
single-field check, which is why cross-field rules exist.

```python
INVOICE_SCHEMA.cross_field
# amounts_reconcile    : subtotal + tax == total
# dates_ordered        : invoice_date <= due_date
# po_requires_customer : po_number present ⇒ customer_name present
# has_an_amount        : at least one of subtotal/total present
```

| Layer | Catches | Example |
|---|---|---|
| **Per field** | malformed values | type, pattern, range, length, allowed values, confidence floor |
| **Cross field** | inconsistent documents | amounts that don't reconcile; dates out of order |
| **Document** | failed parses | too few fields found; mean confidence too low |

Kinds available: `sum_equals`, `date_ordering`, `gte`, `equals`, `not_equals`,
`one_of`, `implies`.

**A check that cannot run is skipped, not failed.** With a missing operand the
report records `check_skipped` at INFO severity — flagging "cannot check" as an
error would drown real findings in noise on partially extracted documents. The
report tracks `cross_checks_run` and `cross_checks_failed` separately so a
skipped check is visible.

### The verdict is derived, never asserted

`ValidationReport` computes `state` from the issue severities, so it cannot
claim `PASS` while carrying an error:

| State | Meaning | Action |
|---|---|---|
| `PASS` | no errors, no warnings | use it |
| `REVIEW` | warnings only | route to a human |
| `FAIL` | at least one error | do not use |

## Preset schemas

`INVOICE_SCHEMA`, `RECEIPT_SCHEMA`, `CONTRACT_SCHEMA`, `ID_DOCUMENT_SCHEMA` —
also in `PRESETS` by name. Each is a plain data object, so copy and edit it
rather than subclassing.

Invoices, contracts and ID documents declare a `prebuilt_model`, so the extractor
can consume Document Intelligence's typed fields directly.

## Batch processing

```python
results = await pipe.process_batch(paths, concurrency=4)
DocumentPipeline.summarize(results)
# {'documents': 120, 'processed': 118, 'unreadable': 2,
#  'by_state': {'pass': 91, 'review': 22, 'fail': 5}, 'pass_rate': 0.77,
#  'top_failing_fields': {'total': 5, 'invoice_date': 3}}
```

One unreadable file does not abort the batch — it is returned with its error so
you can retry or quarantine it. `top_failing_fields` is the field that tells you
whether to fix the **schema** or the **documents**.

## Guardrails

Identity documents are PII by definition. Run the extracted values through
`guardrails` before persisting or logging them:

```python
from guardrails import GuardrailPipeline, PIIRedactor

pipeline = GuardrailPipeline(input_guardrails=[PIIRedactor()])
safe = await pipeline.check_input(str(result.extraction.get("document_number")))
```

## Tests

```bash
uv run pytest components/doc-processing/tests/test_doc_processing.py -v
```

Offline and deterministic by default; live Azure tests are marked and skipped
unless an endpoint is configured.

Part of the [agent-components](../../) monorepo.

## License

MIT
