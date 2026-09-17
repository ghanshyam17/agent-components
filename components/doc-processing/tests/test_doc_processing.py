"""Tests for the doc-processing component.

Offline and deterministic by default: the ingest engine is the injected
``StubEngine``, and the Azure path is exercised through an injected fetcher so
it needs no network or credentials. Tests marked ``live`` are skipped unless
``DOC_INTELLIGENCE_ENDPOINT`` is set.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from doc_processing import (
    AzureDocumentIntelligenceEngine,
    BoundingBox,
    CONTRACT_SCHEMA,
    CrossFieldRule,
    DocumentPipeline,
    DocumentSchema,
    DocumentValidator,
    ExtractedField,
    ExtractionMethod,
    ExtractionResult,
    FieldEvidence,
    FieldExtractor,
    FieldIssue,
    FieldRule,
    INVOICE_SCHEMA,
    ID_DOCUMENT_SCHEMA,
    Page,
    ParsedDocument,
    ParsedTable,
    PRESETS,
    RECEIPT_SCHEMA,
    Severity,
    SourceKind,
    StubEngine,
    TableCell,
    TextLayerEngine,
    TextSpan,
    ValidationState,
    normalize_value,
    select_engine,
)
from doc_processing.extractor import DATE_LABELS, DEFAULT_PATTERNS


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


GOOD_INVOICE = """INVOICE
Invoice No: INV-2024-0042
Invoice Date: 15/03/2026
Due Date: 14/04/2026
Vendor: Acme Industrial Supplies Ltd
Customer: Contoso Manufacturing
PO Number: PO-778812
Subtotal: 1,200.00
VAT: 240.00
Total: 1,440.00
Currency: EUR"""

BROKEN_INVOICE = """INVOICE
Invoice No: INV-9999
Invoice Date: 15/03/2026
Due Date: 01/03/2026
Vendor: Suspicious Supplies
Subtotal: 1,200.00
VAT: 240.00
Total: 9,999.00"""


def _doc(text: str, pages: int = 1) -> ParsedDocument:
    return run(StubEngine(text=text, lines_per_page=40).parse("memory.txt"))


def _infer(text: str) -> ExtractionResult:
    return run(FieldExtractor(INVOICE_SCHEMA).extract(_doc(text)))


def _process(text: str, schema=INVOICE_SCHEMA):
    return run(DocumentPipeline(schema, engine=StubEngine(text=text)).process("memory.txt"))


# --------------------------------------------------------------------------- #
# Models: provenance and confidence are mandatory
# --------------------------------------------------------------------------- #
def test_bounding_box_requires_four_corners():
    BoundingBox(polygon=[0, 0, 10, 0, 10, 10, 0, 10], page=1)
    with pytest.raises(ValueError, match="8 coordinates"):
        BoundingBox(polygon=[0, 0, 10, 10, 0], page=1)


def test_bounding_box_accepts_empty_polygon():
    """A text-layer parse has no geometry; that must not be an error."""
    assert BoundingBox().polygon == []


def test_extracted_field_present_and_effective():
    f = ExtractedField(name="total", value="1,440.00", normalized=1440.0)
    assert f.present is True
    assert f.effective == 1440.0          # normalised wins
    assert ExtractedField(name="x", value=None).present is False
    assert ExtractedField(name="x", value="").present is False
    assert ExtractedField(name="x", value=[]).present is False


def test_field_evidence_is_serialisable():
    f = ExtractedField(
        name="total", value=1440.0, confidence=0.9,
        evidence=FieldEvidence(
            method=ExtractionMethod.REGEX, page=1, pattern=r"\d+",
            bbox=BoundingBox(polygon=[0, 0, 1, 0, 1, 1, 0, 1], page=1),
        ),
    )
    d = f.to_dict()
    assert d["evidence"]["method"] == "regex"
    assert d["evidence"]["bbox"]["page"] == 1
    json.dumps(d)  # must not raise


def test_ocr_confidence_none_rather_than_zero_without_ocr():
    """A text-layer PDF has no OCR confidence; 0.0 would look like a bad scan."""
    doc = ParsedDocument(pages=[Page(number=1, lines=[TextSpan(text="hi")])])
    assert doc.ocr_confidence is None


def test_ocr_confidence_averages_reported_values():
    doc = ParsedDocument(pages=[Page(number=1, lines=[
        TextSpan(text="a", confidence=0.8), TextSpan(text="b", confidence=0.6),
        TextSpan(text="c"),  # no confidence — must be ignored, not counted as 0
    ])])
    assert doc.ocr_confidence == pytest.approx(0.7)


def test_parsed_document_to_dict_shape():
    doc = _doc("hello\nworld")
    d = doc.to_dict()
    assert {"id", "kind", "engine", "pages", "lines", "characters"} <= set(d)
    assert d["pages"] >= 1


# --------------------------------------------------------------------------- #
# Tables
# --------------------------------------------------------------------------- #
def _table() -> ParsedTable:
    cells = [
        TableCell(text="Description", row=0, column=0),
        TableCell(text="Amount", row=0, column=1),
        TableCell(text="Widgets", row=1, column=0),
        TableCell(text="100.00", row=1, column=1),
        TableCell(text="Gadgets", row=2, column=0),
        TableCell(text="200.00", row=2, column=1),
    ]
    return ParsedTable(page=1, cells=cells)


def test_table_grid_dimensions():
    t = _table()
    assert t.n_rows == 3 and t.n_columns == 2


def test_table_header_map_and_rows():
    t = _table()
    assert t.header_map() == {"description": 0, "amount": 1}
    rows = t.rows()
    assert len(rows) == 2
    assert rows[0]["description"] == "Widgets"
    assert rows[1]["amount"] == "200.00"


def test_table_cell_resolves_merged_spans():
    """A merged cell must be found at every (row, col) it covers."""
    t = ParsedTable(cells=[
        TableCell(text="Header", row=0, column=0, column_span=2),
        TableCell(text="b", row=1, column=0),
        TableCell(text="c", row=1, column=1),
    ])
    assert t.cell(0, 0).text == "Header"
    assert t.cell(0, 1).text == "Header"   # covered by the span
    assert t.cell(1, 1).text == "c"


def test_table_rows_empty_without_headers():
    t = ParsedTable(cells=[TableCell(text="x", row=1, column=0)])
    assert t.rows() == []


# --------------------------------------------------------------------------- #
# Normalisation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    ("1,440.00", 1440.0),
    ("1,234,567.89", 1234567.89),
    ("1.234,56", 1234.56),      # European decimal comma
    ("€240.00", 240.0),
    ("$1,200", 1200.0),
])
def test_normalize_money(raw, expected):
    assert normalize_value(raw, "total") == pytest.approx(expected)


@pytest.mark.parametrize("raw,expected", [
    ("2026-03-15", "2026-03-15"),
    ("15/03/2026", "2026-03-15"),
    ("15.03.2026", "2026-03-15"),
    ("15 March 2026", "2026-03-15"),
])
def test_normalize_dates(raw, expected):
    assert normalize_value(raw, "invoice_date") == expected


def test_normalize_leaves_unknown_shapes_alone():
    """A wrong normalisation is worse than none — it looks authoritative."""
    assert normalize_value("not a number", "total") == "not a number"
    assert normalize_value("", "total") == ""
    assert normalize_value(None, "total") is None
    assert normalize_value(1440.0, "total") == 1440.0


def test_normalize_does_not_treat_names_as_dates():
    assert normalize_value("Acme Ltd", "vendor_name") == "Acme Ltd"


# --------------------------------------------------------------------------- #
# Extraction: the field-level regressions
# --------------------------------------------------------------------------- #
def test_invoice_number_is_the_identifier_not_the_label():
    """Regression: IGNORECASE leaked into [A-Z0-9] and matched 'Invoice'."""
    f = _infer(GOOD_INVOICE).fields["invoice_number"]
    assert f.value == "INV-2024-0042"
    assert f.value.lower() != "invoice"


def test_total_is_not_the_subtotal():
    """Regression: `total` matched inside 'Subtotal' without a word boundary."""
    ext = _infer(GOOD_INVOICE)
    assert ext.fields["total"].effective == 1440.0
    assert ext.fields["subtotal"].effective == 1200.0
    assert ext.fields["total"].effective != ext.fields["subtotal"].effective


def test_due_date_is_not_the_invoice_date():
    """Regression: a generic date pattern returned the first date on the page."""
    ext = _infer(GOOD_INVOICE)
    assert ext.fields["invoice_date"].effective == "2026-03-15"
    assert ext.fields["due_date"].effective == "2026-04-14"
    assert ext.fields["invoice_date"].effective != ext.fields["due_date"].effective


def test_vendor_tax_id_does_not_borrow_the_tax_pattern():
    """Regression: substring lookup mapped vendor_tax_id -> the monetary tax regex."""
    ext = _infer(GOOD_INVOICE)
    f = ext.fields.get("vendor_tax_id")
    # Either absent, or not the VAT amount.
    assert f is None or f.effective != 240.0


def test_labelled_dates_beat_a_bare_date_pattern():
    ext = _infer(GOOD_INVOICE)
    assert ext.fields["due_date"].evidence.pattern.startswith("label:")


def test_every_extracted_field_carries_evidence():
    """The component's core promise: nothing is extracted without provenance."""
    ext = _infer(GOOD_INVOICE)
    for name, f in ext.fields.items():
        assert f.evidence is not None, name
        if f.present:
            assert f.evidence.method is not ExtractionMethod.NOT_FOUND, name


def test_provenance_includes_page_and_source_text():
    ext = _infer(GOOD_INVOICE)
    f = ext.fields["invoice_number"]
    assert f.evidence.page == 1
    assert f.evidence.source_text and "INV-2024-0042" in f.evidence.source_text


def test_missing_fields_are_recorded_as_not_found():
    """Absent must be distinguishable from never-attempted."""
    ext = _infer("nothing relevant here at all")
    assert ext.fields, "declared fields must still be present as NOT_FOUND"
    assert all(
        f.evidence.method is ExtractionMethod.NOT_FOUND
        for f in ext.fields.values() if not f.present
    )


def test_missing_required_is_surfaced():
    ext = _infer("Vendor: Acme Ltd")  # no invoice number, no total
    assert "invoice_number" in ext.missing_required
    assert any("missing required" in w for w in ext.warnings)


def test_extraction_reports_counts_and_weakest_field():
    ext = _infer(GOOD_INVOICE)
    assert ext.found >= 6
    assert ext.mean_confidence > 0.6
    assert ext.weakest_field in ext.fields


def test_extraction_get_returns_default_for_absent():
    ext = _infer("nothing")
    assert ext.get("total") is None
    assert ext.get("total", "fallback") == "fallback"


def test_table_extraction_uses_header_names():
    doc = _doc("ignored")
    doc.pages[0].tables = [ParsedTable(page=1, cells=[
        TableCell(text="invoice_number", row=0, column=0),
        TableCell(text="total", row=0, column=1),
        TableCell(text="INV-777", row=1, column=0),
        TableCell(text="500.00", row=1, column=1),
    ])]
    ext = run(FieldExtractor(INVOICE_SCHEMA).extract(doc))
    assert ext.fields["total"].evidence.method in (
        ExtractionMethod.TABLE, ExtractionMethod.REGEX, ExtractionMethod.HEURISTIC
    )


def test_llm_fallback_never_overwrites_a_deterministic_match():
    """A model's guess must not displace a regex's exact match.

    The fallback may still be *consulted* for the fields nothing found — that is
    its purpose — but it must not change a value that was already extracted.
    """
    async def _llm(messages, **kw):
        return '{"total": 9999, "invoice_number": "WRONG"}', []

    ext = run(FieldExtractor(INVOICE_SCHEMA, llm_client=_llm).extract(_doc(GOOD_INVOICE)))
    assert ext.fields["total"].effective == 1440.0
    assert ext.fields["invoice_number"].value == "INV-2024-0042"


def test_llm_fallback_is_skipped_entirely_when_nothing_is_missing():
    """With every declared field found, there is nothing to ask about."""
    called = []

    async def _llm(messages, **kw):
        called.append(1)
        return "{}", []

    schema = DocumentSchema(
        name="tiny",
        fields=[FieldRule(name="total", required=True, type="number")],
    )
    ext = run(FieldExtractor(schema, llm_client=_llm).extract(_doc("Total: 1,440.00")))
    assert ext.fields["total"].present
    assert not called, "LLM must not run when every field is already found"


def test_llm_fallback_fills_only_missing_fields_at_lower_confidence():
    async def _llm(messages, **kw):
        return '{"po_number": "PO-123", "customer_name": "Contoso"}', []

    ext = run(FieldExtractor(INVOICE_SCHEMA, llm_client=_llm).extract(
        _doc("Invoice No: INV-1\nTotal: 10.00")
    ))
    assert ext.fields["po_number"].evidence.method is ExtractionMethod.LLM
    # Capped below the regex tier by design.
    assert ext.fields["po_number"].confidence <= 0.6


def test_llm_fallback_survives_a_failing_client():
    async def _llm(messages, **kw):
        raise RuntimeError("model down")

    ext = run(FieldExtractor(INVOICE_SCHEMA, llm_client=_llm).extract(_doc("Total: 5")))
    assert ext.fields["total"].effective == 5.0


def test_llm_fallback_ignores_unparseable_output():
    async def _llm(messages, **kw):
        return "I could not find those fields, sorry.", []

    ext = run(FieldExtractor(INVOICE_SCHEMA, llm_client=_llm).extract(_doc("Total: 5")))
    assert ext.fields["total"].effective == 5.0


# --------------------------------------------------------------------------- #
# Regex hygiene
# --------------------------------------------------------------------------- #
def test_identifier_patterns_are_case_sensitive_in_their_capture():
    """`(?-i:...)` keeps IGNORECASE from matching lowercase identifiers."""
    import re

    pattern = DEFAULT_PATTERNS["invoice_number"]
    assert re.search(pattern, "Invoice No: INV-2024-1", re.IGNORECASE).group(1) == "INV-2024-1"
    # A lowercase word must not be captured as an identifier.
    m = re.search(pattern, "Invoice number: total", re.IGNORECASE)
    assert m is None or m.group(1) != "total"


def test_total_pattern_does_not_match_subtotal():
    import re

    assert re.search(DEFAULT_PATTERNS["total"], "Subtotal: 100.00", re.IGNORECASE) is None
    assert re.search(DEFAULT_PATTERNS["total"], "Total: 100.00", re.IGNORECASE) is not None


def test_pattern_lookup_refuses_identifiers():
    ex = FieldExtractor(INVOICE_SCHEMA)
    assert ex._pattern_for("invoice_number") is not None
    # Identifiers must not borrow a money/date pattern by substring.
    assert ex._pattern_for("vendor_tax_id") is None
    assert ex._pattern_for("some_random_ref") is None


def test_date_labels_cover_the_preset_dates():
    for name in ("invoice_date", "due_date", "effective_date", "expiry_date", "date_of_birth"):
        assert name in DATE_LABELS


# --------------------------------------------------------------------------- #
# Validation: per field
# --------------------------------------------------------------------------- #
def _validate(text: str, schema=INVOICE_SCHEMA):
    return run(
        DocumentPipeline(schema, engine=StubEngine(text=text)).process("memory.txt")
    ).validation


def test_valid_invoice_passes():
    r = _validate(GOOD_INVOICE)
    assert r.state is ValidationState.PASS
    assert r.errors == []
    assert not r.needs_review


def test_missing_required_field_is_an_error():
    r = _validate("Vendor: Acme Ltd\nSubtotal: 10.00")
    assert r.state is ValidationState.FAIL
    assert any(i.code == "missing_required" for i in r.errors)


def test_type_error_is_a_failure():
    schema = DocumentSchema(
        name="t", fields=[FieldRule(name="amount", required=True, type="number")]
    )
    ext = ExtractionResult(fields={
        "amount": ExtractedField(name="amount", value="not-a-number", confidence=0.9)
    })
    r = DocumentValidator(schema).validate(ext)
    assert any(i.code == "not_a_number" for i in r.errors)
    # `state` is derived by the report validator, so it must reflect the error.
    assert r.state is ValidationState.FAIL


def test_pattern_mismatch_is_a_failure():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="code", required=True, pattern=r"[A-Z]{3}-\d{4}")],
    )
    ext = ExtractionResult(fields={
        "code": ExtractedField(name="code", value="bad-code", confidence=0.9)
    })
    r = DocumentValidator(schema).validate(ext)
    assert any(i.code == "pattern_mismatch" for i in r.errors)


def test_numeric_bounds_are_enforced():
    schema = DocumentSchema(
        name="t", fields=[FieldRule(name="qty", type="number", min_value=1, max_value=10)]
    )
    ext = ExtractionResult(fields={
        "qty": ExtractedField(name="qty", value=50, confidence=0.9)
    })
    r = DocumentValidator(schema).validate(ext)
    assert any(i.code == "above_max" for i in r.errors)


def test_allowed_values_are_enforced():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="cur", allowed_values=["EUR", "USD"])],
    )
    ext = ExtractionResult(fields={
        "cur": ExtractedField(name="cur", value="XYZ", confidence=0.9)
    })
    r = DocumentValidator(schema).validate(ext)
    assert any(i.code == "not_allowed" for i in r.errors)


def test_low_confidence_produces_a_warning_not_an_error():
    """Low confidence means 'a human should look', not 'this is wrong'."""
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="x", required=True, type="string", min_confidence=0.9)],
    )
    ext = ExtractionResult(fields={
        "x": ExtractedField(name="x", value="ok", confidence=0.4)
    })
    r = DocumentValidator(schema).validate(ext)
    assert any(i.code == "low_confidence" for i in r.issues)
    assert any(i.severity is Severity.WARNING for i in r.issues)
    # Warnings only -> REVIEW, never FAIL.
    assert r.state is ValidationState.REVIEW


# --------------------------------------------------------------------------- #
# Validation: cross field — where real document errors live
# --------------------------------------------------------------------------- #
def test_amounts_that_do_not_reconcile_fail():
    r = _validate(BROKEN_INVOICE)
    assert r.state is ValidationState.FAIL
    assert any(i.code == "sum_mismatch" for i in r.errors)
    assert r.cross_checks_failed >= 2


def test_reversed_dates_fail():
    r = _validate(BROKEN_INVOICE)
    assert any(i.code == "date_order" for i in r.errors)


def test_preset_cross_rule_kinds_are_all_declared():
    """A rule kind absent from the Literal would fail at schema construction."""
    DocumentSchema(
        name="kinds",
        fields=[FieldRule(name="a", type="number"), FieldRule(name="b", type="number")],
        cross_field=[
            CrossFieldRule(name="s", kind="sum_equals", fields=["a", "b"], target="a"),
            CrossFieldRule(name="g", kind="gte", fields=["a", "b"]),
            CrossFieldRule(name="e", kind="equals", fields=["a", "b"]),
            CrossFieldRule(name="n", kind="not_equals", fields=["a", "b"]),
            CrossFieldRule(name="o", kind="one_of", fields=["a", "b"]),
            CrossFieldRule(name="i", kind="implies", fields=["a", "b"]),
        ],
    )


def test_reconciling_amounts_pass():
    r = _validate(GOOD_INVOICE)
    assert r.cross_checks_failed == 0
    assert r.cross_checks_run >= 2


def test_implies_rule_fires_when_consequent_is_missing():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="a"), FieldRule(name="b")],
        cross_field=[CrossFieldRule(name="a_implies_b", kind="implies", fields=["a", "b"])],
    )
    ext = ExtractionResult(fields={
        "a": ExtractedField(name="a", value="present", confidence=0.9)
    })
    r = DocumentValidator(schema).validate(ext)
    assert any(i.code == "implies_violated" for i in r.errors)


def test_one_of_rule_fires_when_none_present():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="a"), FieldRule(name="b")],
        cross_field=[CrossFieldRule(name="need_one", kind="one_of", fields=["a", "b"])],
    )
    r = DocumentValidator(schema).validate(ExtractionResult())
    assert any(i.code == "one_of_violated" for i in r.errors)


def test_one_of_rule_passes_when_one_present():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="a"), FieldRule(name="b")],
        cross_field=[CrossFieldRule(name="need_one", kind="one_of", fields=["a", "b"])],
    )
    ext = ExtractionResult(fields={"a": ExtractedField(name="a", value=1, confidence=0.9)})
    r = DocumentValidator(schema).validate(ext)
    assert r.state is ValidationState.PASS


def test_gte_numeric_ordering():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="total", type="number"), FieldRule(name="tax", type="number")],
        cross_field=[CrossFieldRule(name="total_ge_tax", kind="gte", fields=["total", "tax"])],
    )
    ok = ExtractionResult(fields={
        "total": ExtractedField(name="total", value=100, confidence=0.9),
        "tax": ExtractedField(name="tax", value=20, confidence=0.9),
    })
    assert DocumentValidator(schema).validate(ok).state is ValidationState.PASS

    bad = ExtractionResult(fields={
        "total": ExtractedField(name="total", value=10, confidence=0.9),
        "tax": ExtractedField(name="tax", value=20, confidence=0.9),
    })
    assert any(
        i.code == "numeric_order" for i in DocumentValidator(schema).validate(bad).errors
    )


def test_a_check_that_cannot_run_is_skipped_not_failed():
    """A missing operand means 'cannot judge', which must not read as a failure."""
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="a", type="number"), FieldRule(name="b", type="number"),
                FieldRule(name="c", type="number")],
        cross_field=[CrossFieldRule(name="sum", kind="sum_equals", fields=["a", "b"], target="c")],
    )
    ext = ExtractionResult(fields={
        "a": ExtractedField(name="a", value=1, confidence=0.9),
        # b and c absent -> the rule cannot be evaluated
    })
    # The document-level found-ratio gate is disabled so this test isolates the
    # cross-field behaviour: 1 of 3 fields found would otherwise trip it, which
    # is correct but a different assertion.
    r = DocumentValidator(schema, doc_min_found_ratio=0.0).validate(ext)
    assert r.cross_checks_run == 0, "an unevaluable check must not count as run"
    assert r.cross_checks_failed == 0
    assert any(i.code == "check_skipped" and i.severity is Severity.INFO for i in r.issues)
    assert r.state is ValidationState.PASS


def test_tolerance_absorbs_rounding():
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="subtotal", type="number"), FieldRule(name="tax", type="number"),
                FieldRule(name="total", type="number")],
        cross_field=[CrossFieldRule(name="sum", kind="sum_equals", fields=["subtotal", "tax"],
                                    target="total", tolerance=0.05)],
    )
    ext = ExtractionResult(fields={
        "subtotal": ExtractedField(name="subtotal", value=100.00, confidence=0.9),
        "tax": ExtractedField(name="tax", value=20.00, confidence=0.9),
        "total": ExtractedField(name="total", value=120.03, confidence=0.9),  # 3c out
    })
    assert DocumentValidator(schema).validate(ext).state is ValidationState.PASS


# --------------------------------------------------------------------------- #
# Validation: document level
# --------------------------------------------------------------------------- #
def test_empty_parse_is_a_failure_not_a_silent_pass():
    """The scanned-page case: the parse worked but found nothing."""
    r = _validate("")
    assert r.state is ValidationState.FAIL
    assert any(i.code == "empty_parse" for i in r.errors)


def test_document_confidence_gate_routes_to_review():
    """A weakly legible document routes to review even when fields parse."""
    schema = DocumentSchema(
        name="t", fields=[FieldRule(name="x", required=True, type="string")]
    )
    ext = ExtractionResult(fields={
        "x": ExtractedField(name="x", value="v", confidence=0.2)
    })
    v = DocumentValidator(schema, doc_min_confidence=0.5, doc_min_found_ratio=0.0)
    r = v.validate(ext)
    assert any(i.code == "low_mean_confidence" for i in r.issues)
    assert r.state is ValidationState.REVIEW


def test_report_state_is_derived_from_issues():
    """The report must never claim PASS while carrying an error."""
    from doc_processing import ValidationReport

    rep = ValidationReport(issues=[
        FieldIssue(field="x", severity=Severity.ERROR, code="e", message="bad")
    ])
    assert rep.state is ValidationState.FAIL

    rep2 = ValidationReport(issues=[
        FieldIssue(field="x", severity=Severity.WARNING, code="w", message="meh")
    ])
    assert rep2.state is ValidationState.REVIEW
    assert rep2.needs_review is True

    assert ValidationReport().state is ValidationState.PASS
    assert ValidationReport().needs_review is False


# --------------------------------------------------------------------------- #
# Schema definition and validation
# --------------------------------------------------------------------------- #
def test_schema_rejects_cross_rules_referencing_unknown_fields():
    with pytest.raises(ValueError, match="unknown field"):
        DocumentSchema(
            name="t",
            fields=[FieldRule(name="a")],
            cross_field=[CrossFieldRule(name="r", kind="equals", fields=["a", "ghost"])],
        )


def test_schema_rejects_hints_for_unknown_fields():
    with pytest.raises(ValueError, match="hints declared for unknown field"):
        DocumentSchema(name="t", fields=[FieldRule(name="a")], hints={"ghost": ["x"]})


def test_schema_rejects_rules_without_fields():
    with pytest.raises(ValueError, match="no fields"):
        DocumentSchema(name="t", cross_field=[CrossFieldRule(name="r", kind="one_of", fields=[])])
    with pytest.raises(ValueError, match="no fields"):
        DocumentSchema(name="t", hints={"a": ["x"]})


def test_preset_schemas_are_self_consistent():
    """Every preset must validate, and reference only its own fields."""
    for name, schema in PRESETS.items():
        assert schema.name == name
        known = {f.name for f in schema.fields}
        for rule in schema.cross_field:
            for ref in [*rule.fields, *([rule.target] if rule.target else [])]:
                assert ref in known, f"{name}: {rule.name} -> {ref}"
        for hint in schema.hints:
            assert hint in known, f"{name}: hint {hint}"


def test_preset_schemas_have_at_least_one_required_field():
    for name, schema in PRESETS.items():
        assert any(f.required for f in schema.fields), f"{name} requires nothing"


def test_every_preset_cross_field_kind_is_implemented():
    """A rule kind with no implementation would silently never run."""
    from doc_processing.validator import DocumentValidator as V

    supported = {"sum_equals", "date_ordering", "equals", "not_equals", "one_of", "implies", "gte"}
    for name, schema in PRESETS.items():
        for rule in schema.cross_field:
            assert rule.kind in supported, f"{name}: {rule.name} kind {rule.kind}"


def test_schema_to_dict_is_json_safe():
    json.dumps(INVOICE_SCHEMA.to_dict())


# --------------------------------------------------------------------------- #
# Engines
# --------------------------------------------------------------------------- #
def test_stub_engine_is_deterministic():
    a = run(StubEngine(text="x\ny").parse("s.txt"))
    b = run(StubEngine(text="x\ny").parse("s.txt"))
    assert [l.text for l in a.pages[0].lines] == [l.text for l in b.pages[0].lines]


def test_stub_engine_synthesises_bboxes():
    doc = run(StubEngine(text="one\ntwo").parse("s.txt"))
    assert all(l.bbox is not None for l in doc.pages[0].lines)


def test_source_kind_classification_by_suffix_and_magic():
    assert SourceKind.from_suffix(".pdf") is SourceKind.PDF
    assert SourceKind.from_suffix("png") is SourceKind.IMAGE
    assert SourceKind.from_suffix(".docx") is SourceKind.DOCX
    assert SourceKind.from_suffix(".weird") is SourceKind.UNKNOWN

    doc = run(StubEngine().parse("noext", b"%PDF-1.4 fake"))
    assert doc.kind is SourceKind.PDF, "magic bytes must win over the filename"


def test_azure_engine_reports_unconfigured_without_credentials():
    eng = AzureDocumentIntelligenceEngine(None)
    assert eng.configured is False
    d = eng.describe()
    assert d["configured"] is False
    assert d["auth"] == "entra"


def test_azure_engine_raises_when_not_configured():
    with pytest.raises(RuntimeError, match="no Document Intelligence endpoint"):
        run(AzureDocumentIntelligenceEngine(None).parse("x.pdf", b"%PDF"))


def test_select_engine_prefers_stub_when_asked():
    assert select_engine(prefer="stub").name == "stub"


def test_select_engine_refuses_azure_without_endpoint():
    with pytest.raises(RuntimeError, match="no endpoint"):
        select_engine(prefer="azure", endpoint=None)


def test_select_engine_falls_back_to_text_layer(monkeypatch):
    monkeypatch.delenv("DOC_INTELLIGENCE_ENDPOINT", raising=False)
    assert select_engine().name == "text-layer"


def test_select_engine_uses_azure_when_configured():
    assert select_engine(endpoint="https://x.cognitiveservices.azure.com").name == \
        "azure-document-intelligence"


# --------------------------------------------------------------------------- #
# Azure engine: response handling, via an injected fetcher
# --------------------------------------------------------------------------- #
class _Fetcher:
    """Records calls and replays canned responses, including headers."""

    def __init__(self, post_headers=None, post_body=None, poll_bodies=None, status=202):
        self.post_headers = post_headers or {}
        self.post_body = post_body or {}
        self.poll_bodies = list(poll_bodies or [])
        self.status = status
        self.calls: list[tuple[str, str]] = []

    async def __call__(self, method, url, headers, body):
        self.calls.append((method, url))
        if method == "POST":
            return self.status, self.post_body, self.post_headers
        return 200, self.poll_bodies.pop(0), {}


_DI_RESULT = {
    "status": "succeeded",
    "analyzeResult": {
        "content": "Invoice No: INV-1 Total: 10.00",
        "pages": [{
            "pageNumber": 1, "width": 612, "height": 792,
            "lines": [
                {"content": "Invoice No: INV-1", "polygon": [0, 0, 100, 0, 100, 10, 0, 10],
                 "words": [{"confidence": 0.99}, {"confidence": 0.98}]},
                {"content": "Total: 10.00", "polygon": [0, 0, 100, 0, 100, 10, 0, 10],
                 "words": [{"confidence": 0.95}]},
            ],
            "selectionMarks": [{"state": "selected"}],
        }],
        "tables": [{
            "cells": [
                {"content": "Item", "rowIndex": 0, "columnIndex": 0},
                {"content": "Amt", "rowIndex": 0, "columnIndex": 1},
                {"content": "Widget", "rowIndex": 1, "columnIndex": 0},
                {"content": "10.00", "rowIndex": 1, "columnIndex": 1,
                 "boundingRegions": [{"pageNumber": 1}]},
            ]
        }],
        "documents": [{"fields": {"InvoiceId": {"type": "string", "valueString": "INV-1",
                                                "confidence": 0.97, "content": "INV-1"}}}],
    },
}


def test_azure_engine_reads_operation_location_from_headers():
    """Regression: the 202 body is empty; operation-location is a HEADER.

    Discarding headers lost the operation URL, so the poll never ran and an
    empty document was returned that looked like a successful parse.
    """
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/operations/1"},
        poll_bodies=[_DI_RESULT],
    )
    eng = AzureDocumentIntelligenceEngine("https://x.cognitiveservices.azure.com", fetcher=fetcher)
    doc = run(eng.parse("inv.pdf", b"%PDF-1.4"))
    assert len(doc.pages) == 1
    assert [l.text for l in doc.pages[0].lines] == ["Invoice No: INV-1", "Total: 10.00"]
    # It must have polled the operation URL it received.
    assert any(m == "GET" and u == "https://x/operations/1" for m, u in fetcher.calls)


def test_azure_engine_accepts_operation_location_in_body():
    fetcher = _Fetcher(
        post_body={"operationLocation": "https://x/op2"}, poll_bodies=[_DI_RESULT]
    )
    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))
    assert len(doc.pages) == 1


def test_azure_engine_handles_inline_result():
    fetcher = _Fetcher(post_body={"analyzeResult": _DI_RESULT["analyzeResult"]})
    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))
    assert len(doc.pages) == 1


def test_azure_engine_fails_loudly_without_an_operation_or_result():
    """Neither present must raise, not return an empty-but-plausible document."""
    fetcher = _Fetcher(post_headers={}, post_body={})
    with pytest.raises(RuntimeError, match="no operation-location"):
        run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))


def test_azure_engine_surfaces_http_errors():
    fetcher = _Fetcher(status=401, post_body={"error": {"message": "unauthorized"}})
    with pytest.raises(RuntimeError, match="HTTP 401"):
        run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))


def test_azure_engine_surfaces_a_failed_operation():
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/op"},
        poll_bodies=[{"status": "failed", "error": {"code": "BadDoc", "message": "unreadable"}}],
    )
    with pytest.raises(RuntimeError, match="failed"):
        run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))


def test_azure_engine_polls_until_succeeded():
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/op"},
        poll_bodies=[{"status": "running"}, {"status": "running"}, _DI_RESULT],
    )
    eng = AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher)

    async def _no_sleep(_: float) -> None:
        return None

    eng._sleep = _no_sleep
    doc = run(eng.parse("i.pdf", b"%PDF"))
    assert len(doc.pages) == 1
    assert sum(1 for m, _ in fetcher.calls if m == "GET") == 3


def test_azure_engine_parses_tables_and_spans():
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/op"}, poll_bodies=[_DI_RESULT]
    )
    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))
    table = doc.all_tables[0]
    assert table.n_rows == 2 and table.n_columns == 2
    assert table.header_map() == {"item": 0, "amt": 1}
    assert table.rows()[0]["amt"] == "10.00"


def test_azure_engine_keeps_di_documents_for_the_extractor():
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/op"}, poll_bodies=[_DI_RESULT]
    )
    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))
    assert doc.metadata.get("di_documents")


def test_azure_engine_captures_selection_marks_and_confidence():
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/op"}, poll_bodies=[_DI_RESULT]
    )
    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))
    assert doc.pages[0].selection_marks == ["selected"]
    assert doc.pages[0].lines[0].confidence == pytest.approx(0.985)


def test_di_fields_are_used_as_the_most_trusted_source():
    """DI's typed fields must outrank regex when both could match.

    The schema is built with a field named for DI's own key (`invoice_number`
    matching `InvoiceId` is not how DI names things — `InvoiceId` is), so this
    asserts the DI path directly rather than hoping the name overlaps.
    """
    fetcher = _Fetcher(
        post_headers={"operation-location": "https://x/op"}, poll_bodies=[_DI_RESULT]
    )
    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=fetcher).parse("i.pdf", b"%PDF"))

    class _Schema(DocumentSchema):
        pass

    schema = DocumentSchema(
        name="di", fields=[FieldRule(name="invoiceid", type="string")]
    )
    ext = run(FieldExtractor(schema).extract(doc))
    f = ext.fields["invoiceid"]
    assert f.present
    assert f.value == "INV-1"
    assert f.evidence.method is ExtractionMethod.DOCUMENT_INTELLIGENCE
    assert f.evidence.engine == "azure-document-intelligence"


def test_two_tuple_fetchers_are_still_supported():
    """The simpler test seam must keep working."""

    async def _two_tuple(method, url, headers, body):
        if method == "POST":
            return 200, {"analyzeResult": _DI_RESULT["analyzeResult"]}
        return 200, {}

    doc = run(AzureDocumentIntelligenceEngine("https://x", fetcher=_two_tuple).parse("i.pdf", b"%PDF"))
    assert len(doc.pages) == 1


# --------------------------------------------------------------------------- #
# Pipeline
# --------------------------------------------------------------------------- #
def test_pipeline_returns_all_three_stage_outputs():
    r = _process(GOOD_INVOICE)
    assert r.document.pages
    assert r.extraction.fields
    assert r.validation.issues is not None
    assert r.usable is True
    assert r.routed_to_review is False


def test_pipeline_result_is_json_safe():
    json.dumps(_process(GOOD_INVOICE).to_dict())


def test_pipeline_routes_a_broken_document_to_review():
    r = _process(BROKEN_INVOICE)
    assert r.usable is False
    assert r.routed_to_review is True


def test_pipeline_raises_rather_than_returning_a_partial_result():
    """An empty extraction that 'succeeded' is the ambiguity this removes."""
    with pytest.raises(Exception):
        run(DocumentPipeline(INVOICE_SCHEMA, engine=AzureDocumentIntelligenceEngine(None))
            .process("x.pdf", b"%PDF"))


def test_pipeline_batch_isolates_failures():
    class _HalfBroken:
        name = "half"

        async def parse(self, source, data=None):
            if "bad" in source:
                raise RuntimeError("cannot read")
            return await StubEngine(text=GOOD_INVOICE).parse(source)

    pipe = DocumentPipeline(INVOICE_SCHEMA, engine=_HalfBroken())
    results = run(pipe.process_batch(["ok.txt", "bad.txt", "ok2.txt"]))
    assert len(results) == 3
    errors = [r for r in results if isinstance(r, dict)]
    assert len(errors) == 1 and "error" in errors[0]


def test_pipeline_summarize_reports_the_routing_picture():
    results = run(DocumentPipeline(INVOICE_SCHEMA, engine=StubEngine(text=GOOD_INVOICE))
                  .process_batch(["a.txt"]))
    s = DocumentPipeline.summarize(results)
    assert s["documents"] == 1
    assert s["by_state"].get("pass") == 1
    assert s["pass_rate"] == 1.0
    assert "top_failing_fields" in s


def test_pipeline_summarize_counts_unreadable():
    s = DocumentPipeline.summarize([{"source": "x", "error": "boom"}])
    assert s["unreadable"] == 1
    assert s["processed"] == 0
    assert s["pass_rate"] == 0.0


def test_pipeline_describe_reports_engine_and_schema():
    d = DocumentPipeline(INVOICE_SCHEMA, engine=StubEngine(text="x")).describe()
    assert d["schema"] == "invoice"
    assert d["fields"] == len(INVOICE_SCHEMA.fields)
    assert d["cross_field_rules"] == len(INVOICE_SCHEMA.cross_field)
    assert d["engine"]["engine"] == "stub"


def test_pipeline_works_for_every_preset_schema():
    """Each preset must run end-to-end without raising."""
    text = """CONTRACT between Acme Ltd and Contoso Inc
Effective Date: 01/01/2026
Expiry Date: 31/12/2026
Total: 50,000.00 EUR"""
    for name, schema in PRESETS.items():
        result = run(
            DocumentPipeline(schema, engine=StubEngine(text=text)).process(f"{name}.txt")
        )
        assert result.validation.state in tuple(ValidationState)
        json.dumps(result.to_dict())


# --------------------------------------------------------------------------- #
# Live Azure (skipped unless configured)
# --------------------------------------------------------------------------- #
LIVE_ENDPOINT = os.environ.get("DOC_INTELLIGENCE_ENDPOINT")

live = pytest.mark.skipif(
    not LIVE_ENDPOINT,
    reason="set DOC_INTELLIGENCE_ENDPOINT to run live Document Intelligence tests",
)


@live
def test_live_document_intelligence_reachable():
    eng = AzureDocumentIntelligenceEngine(LIVE_ENDPOINT)
    assert eng.configured


@live
def test_live_parse_a_pdf():
    """Requires a real PDF on disk; skipped when the fixture is absent."""
    pdf = Path(os.environ.get("DOC_INTELLIGENCE_PDF", "/tmp/real_invoice.pdf"))
    if not pdf.is_file():
        pytest.skip(f"no fixture PDF at {pdf}")
    doc = run(AzureDocumentIntelligenceEngine(LIVE_ENDPOINT).parse(str(pdf)))
    assert doc.pages, "live Document Intelligence returned no pages"
    assert doc.text.strip(), "live parse produced no text"


# --------------------------------------------------------------------------- #
# Document-level found-ratio gate
# --------------------------------------------------------------------------- #
def test_low_found_ratio_is_reported_as_a_failure():
    """Most fields missing means the parse failed, not that they are optional."""
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name=f"f{i}", type="number") for i in range(4)],
    )
    ext = ExtractionResult(fields={
        "f0": ExtractedField(name="f0", value=1, confidence=0.9),
    })
    r = DocumentValidator(schema, doc_min_found_ratio=0.5).validate(ext)
    assert any(i.code == "low_found_ratio" for i in r.issues)
    assert r.state is ValidationState.FAIL


def test_a_check_with_a_missing_operand_does_not_fail_the_document():
    """'Cannot judge' must not become 'wrong' once the ratio gate passes."""
    schema = DocumentSchema(
        name="t",
        fields=[FieldRule(name="a", type="number"), FieldRule(name="b", type="number")],
        cross_field=[CrossFieldRule(name="eq", kind="equals", fields=["a", "b"])],
    )
    ext = ExtractionResult(fields={"a": ExtractedField(name="a", value=1, confidence=0.9)})
    r = DocumentValidator(schema, doc_min_found_ratio=0.0).validate(ext)
    assert r.cross_checks_run == 0
    assert r.cross_checks_failed == 0
    assert r.state is ValidationState.PASS
