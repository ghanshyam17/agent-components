"""Data models for the document-processing component.

The pipeline is three stages, and the types mark the boundary between them:

    ingest ──▶ extract ──▶ validate
    ParsedDocument  ExtractionResult  ValidationReport

Two design commitments are encoded in these models rather than left to
convention:

**Provenance is not optional.** Every extracted field carries a
:class:`FieldEvidence` — where in the source it came from, how confident the
engine was, and how it was obtained. An extracted value with no provenance
cannot be reviewed, cannot be audited, and cannot be safely corrected, and in
a regulated workflow (invoices, IDs, contracts) that makes it worthless.

**Confidence is a first-class field, not a log line.** `confidence` travels with
the value through extraction and validation so the validator can gate on it and
the caller can route low-confidence documents to human review *without
re-parsing anything*.
"""
from __future__ import annotations

import time
import uuid
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


# --------------------------------------------------------------------------- #
# Enumerations
# --------------------------------------------------------------------------- #
class SourceKind(str, Enum):
    """What kind of input was ingested."""

    TEXT = "text"
    PDF = "pdf"
    IMAGE = "image"
    DOCX = "docx"
    UNKNOWN = "unknown"

    @classmethod
    def from_suffix(cls, suffix: str) -> "SourceKind":
        s = (suffix or "").lower().lstrip(".")
        return {
            "txt": cls.TEXT, "md": cls.TEXT, "csv": cls.TEXT,
            "pdf": cls.PDF,
            "png": cls.IMAGE, "jpg": cls.IMAGE, "jpeg": cls.IMAGE,
            "tif": cls.IMAGE, "tiff": cls.IMAGE, "bmp": cls.IMAGE, "webp": cls.IMAGE,
            "docx": cls.DOCX, "doc": cls.DOCX,
        }.get(s, cls.UNKNOWN)


class ExtractionMethod(str, Enum):
    """How a value was obtained — the 'why should I trust this?' field."""

    DOCUMENT_INTELLIGENCE = "document_intelligence"  # Azure prebuilt/custom model
    REGEX = "regex"                                  # deterministic pattern match
    TABLE = "table"                                  # lifted from a parsed table
    HEURISTIC = "heuristic"                          # keyword/position rule
    LLM = "llm"                                      # model-assisted extraction
    NOT_FOUND = "not_found"


class Severity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationState(str, Enum):
    """Overall verdict for a document."""

    PASS = "pass"
    REVIEW = "review"    # low confidence / soft failures — a human should look
    FAIL = "fail"        # a hard rule was violated


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class BoundingBox(BaseModel):
    """A region on a page, in Document Intelligence's convention.

    DI uses 8 floats: the four corners as x0,y0,x1,y1,x2,y2,x3,y3 (inches for
    PDF, pixels for images), in reading order from the top-left. Preserving the
    raw polygon rather than a normalised rectangle means a highlight overlay
    cannot drift.
    """

    polygon: list[float] = Field(default_factory=list, description="8 coords, DI convention.")
    page: int = Field(default=1, ge=1)

    @model_validator(mode="after")
    def _validate_polygon_length(self) -> "BoundingBox":
        if self.polygon and len(self.polygon) != 8:
            raise ValueError(
                f"polygon needs exactly 8 coordinates (4 corners), got {len(self.polygon)}"
            )
        return self

    @property
    def as_dict(self) -> dict[str, Any]:
        return {"page": self.page, "polygon": list(self.polygon)}


class TextSpan(BaseModel):
    """A run of recognised text and where it sits on the page."""

    text: str
    page: int = 1
    bbox: BoundingBox | None = None
    confidence: float | None = Field(
        default=None, ge=0.0, le=1.0,
        description="OCR confidence, when the engine reports it.",
    )


class TableCell(BaseModel):
    text: str
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    confidence: float | None = None


class ParsedTable(BaseModel):
    """A table recovered from the document, with its grid intact.

    Cells are addressable by (row, column) rather than stored as a list of rows,
    because merged cells mean rows are not rectangular — flattening loses the
    spans that make a table readable.
    """

    id: str = Field(default_factory=lambda: _new_id("table"))
    page: int = 1
    cells: list[TableCell] = Field(default_factory=list)
    caption: str | None = None

    @property
    def n_rows(self) -> int:
        return max((c.row + c.row_span for c in self.cells), default=0)

    @property
    def n_columns(self) -> int:
        return max((c.column + c.column_span for c in self.cells), default=0)

    def cell(self, row: int, column: int) -> TableCell | None:
        for c in self.cells:
            if c.row <= row < c.row + c.row_span and c.column <= column < c.column + c.column_span:
                return c
        return None

    def header_map(self, header_row: int = 0) -> dict[str, int]:
        """Column index by header label — the usual entry point for table extraction."""
        out: dict[str, int] = {}
        for c in self.cells:
            if c.row == header_row and c.text.strip():
                out[c.text.strip().lower()] = c.column
        return out

    def rows(self) -> list[dict[str, str]]:
        """Row dicts keyed by the header row, skipping the header itself."""
        headers = self.header_map()
        if not headers:
            return []
        by_col = {v: k for k, v in headers.items()}
        out: list[dict[str, str]] = []
        for r in range(1, self.n_rows):
            row: dict[str, str] = {}
            for col, name in by_col.items():
                cell = self.cell(r, col)
                row[name] = cell.text if cell else ""
            if any(v.strip() for v in row.values()):
                out.append(row)
        return out


class Page(BaseModel):
    """One page's recognised content."""

    number: int = Field(ge=1)
    width: float | None = None
    height: float | None = None
    lines: list[TextSpan] = Field(default_factory=list)
    tables: list[ParsedTable] = Field(default_factory=list)
    selection_marks: list[str] = Field(
        default_factory=list, description="Checkbox/radio states, when detected."
    )

    @property
    def text(self) -> str:
        return "\n".join(s.text for s in self.lines)


class ParsedDocument(BaseModel):
    """The result of ingestion: text, structure, and how it was obtained.

    ``engine`` and ``pages`` together answer "can I trust this parse?" — a
    text-layer PDF read by `pypdf` has no OCR error, while a scanned image run
    through OCR may. Callers should not have to guess.
    """

    id: str = Field(default_factory=lambda: _new_id("doc"))
    source: str = ""
    kind: SourceKind = SourceKind.UNKNOWN
    engine: str = ""
    pages: list[Page] = Field(default_factory=list)
    language: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0
    created_at: float = Field(default_factory=time.time)

    @property
    def text(self) -> str:
        """Full document text, pages joined by a form feed."""
        return "\n\f\n".join(p.text for p in self.pages)

    @property
    def all_tables(self) -> list[ParsedTable]:
        return [t for p in self.pages for t in p.tables]

    @property
    def ocr_confidence(self) -> float | None:
        """Mean OCR confidence across all lines that reported one.

        None rather than 0.0 when nothing reported confidence: a text-layer PDF
        has no OCR confidence at all, and reporting 0.0 would look like a
        catastrophically bad scan.
        """
        vals = [
            s.confidence for p in self.pages for s in p.lines
            if s.confidence is not None
        ]
        return sum(vals) / len(vals) if vals else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source": self.source,
            "kind": self.kind.value,
            "engine": self.engine,
            "pages": len(self.pages),
            "tables": len(self.all_tables),
            "lines": sum(len(p.lines) for p in self.pages),
            "characters": len(self.text),
            "ocr_confidence": self.ocr_confidence,
            "language": self.language,
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
        }


# --------------------------------------------------------------------------- #
# Extraction
# --------------------------------------------------------------------------- #
class FieldEvidence(BaseModel):
    """Where an extracted value came from and how much to trust it.

    The single most important model in this component. Without it an extracted
    field is an unfalsifiable assertion; with it, a reviewer can check the
    source region, and an automated repair can decide whether re-extraction is
    warranted.
    """

    method: ExtractionMethod = ExtractionMethod.NOT_FOUND
    page: int | None = None
    bbox: BoundingBox | None = None
    source_text: str | None = Field(
        default=None, description="The raw text this value was derived from."
    )
    pattern: str | None = Field(
        default=None, description="The regex/rule name that matched, when applicable."
    )
    table_ref: str | None = Field(
        default=None, description="Table id + cell, e.g. 'table_ab12:r3c2'."
    )
    engine: str | None = Field(default=None, description="Underlying model, if any.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method.value,
            "page": self.page,
            "bbox": self.bbox.as_dict if self.bbox else None,
            "source_text": self.source_text,
            "pattern": self.pattern,
            "table_ref": self.table_ref,
            "engine": self.engine,
        }


class ExtractedField(BaseModel):
    """One extracted value, its confidence, and its evidence.

    ``value`` is ``Any`` because a field may be a string, number, date, or list.
    ``confidence`` is required (0..1) rather than optional — an extractor that
    cannot state its confidence should say 0.0, not omit the question.
    """

    name: str
    value: Any = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    required: bool = False
    evidence: FieldEvidence = Field(default_factory=FieldEvidence)
    normalized: Any = Field(
        default=None, description="Value after normalisation (dates, amounts, casing)."
    )

    @property
    def present(self) -> bool:
        return self.value is not None and self.value != "" and self.value != []

    @property
    def effective(self) -> Any:
        """The value a consumer should use: the normalised form when available."""
        return self.normalized if self.normalized is not None else self.value

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "normalized": self.normalized,
            "confidence": round(self.confidence, 4),
            "required": self.required,
            "present": self.present,
            "evidence": self.evidence.to_dict(),
        }


class ExtractionResult(BaseModel):
    """All fields pulled from one document, plus the extraction's own telemetry."""

    id: str = Field(default_factory=lambda: _new_id("ext"))
    document_id: str = ""
    schema_name: str = ""
    fields: dict[str, ExtractedField] = Field(default_factory=dict)
    engine: str = ""
    method: ExtractionMethod = ExtractionMethod.NOT_FOUND
    warnings: list[str] = Field(default_factory=list)
    duration_ms: int = 0

    @property
    def found(self) -> int:
        return sum(1 for f in self.fields.values() if f.present)

    @property
    def missing_required(self) -> list[str]:
        return [n for n, f in self.fields.items() if f.required and not f.present]

    @property
    def mean_confidence(self) -> float:
        present = [f.confidence for f in self.fields.values() if f.present]
        return sum(present) / len(present) if present else 0.0

    @property
    def weakest_field(self) -> str | None:
        """The least-confident present field — usually where a reviewer should look first."""
        present = [(f.confidence, n) for n, f in self.fields.items() if f.present]
        return min(present)[1] if present else None

    def get(self, name: str, default: Any = None) -> Any:
        f = self.fields.get(name)
        return f.effective if f is not None and f.present else default

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "schema_name": self.schema_name,
            "engine": self.engine,
            "method": self.method.value,
            "counts": {
                "total": len(self.fields),
                "found": self.found,
                "missing_required": len(self.missing_required),
            },
            "mean_confidence": round(self.mean_confidence, 4),
            "weakest_field": self.weakest_field,
            "fields": {n: f.to_dict() for n, f in self.fields.items()},
            "warnings": self.warnings,
            "duration_ms": self.duration_ms,
        }


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
class FieldRule(BaseModel):
    """Declarative constraints for one field.

    Constraints are data rather than code so a schema can be versioned, diffed
    and reviewed by someone who does not read Python — which is the normal
    situation when a business analyst owns the document spec.
    """

    name: str
    required: bool = False
    type: Literal["string", "number", "integer", "date", "boolean", "list", "any"] = "any"
    pattern: str | None = Field(default=None, description="Regex the string form must match.")
    min_value: float | None = None
    max_value: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    allowed_values: list[str] | None = None
    min_confidence: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Below this the field is flagged for review even if valid.",
    )


class CrossFieldRule(BaseModel):
    """A constraint spanning fields — where most real document errors live.

    A line-item total that does not sum to the stated subtotal is invisible to
    any single-field check; it is a relationship. These rules are named so a
    failure message can say *which* business rule broke.
    """

    name: str
    description: str = ""
    kind: Literal[
        "sum_equals", "date_ordering", "equals", "not_equals", "one_of", "implies",
        # numeric ordering: fields[0] >= fields[1] (>= ...), within tolerance
        "gte",
    ]
    # Generic operands, interpreted per `kind`:
    #   sum_equals   : sum(fields) == target (within tolerance)
    #   date_ordering: fields[0] <= fields[1] (<= ...)
    #   equals / not_equals: fields[0] vs fields[1]
    #   one_of       : at least one of `fields` is present
    #   implies      : if `fields[0]` present then `fields[1]` required
    #   gte          : fields[0] >= fields[1] >= ... (numeric ordering)
    fields: list[str] = Field(default_factory=list)
    target: str | None = None
    tolerance: float = Field(default=0.01, ge=0.0)
    severity: Severity = Severity.ERROR


class FieldIssue(BaseModel):
    """One thing wrong with one field."""

    field: str
    severity: Severity
    code: str
    message: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "field": self.field, "severity": self.severity.value, "code": self.code,
            "message": self.message, "value": self.value,
        }


class ValidationReport(BaseModel):
    """The verdict on one extraction, with every reason recorded.

    ``state`` is derived from the issues rather than set independently, so it
    can never disagree with the list of problems.
    """

    id: str = Field(default_factory=lambda: _new_id("val"))
    extraction_id: str = ""
    schema_name: str = ""
    state: ValidationState = ValidationState.PASS
    issues: list[FieldIssue] = Field(default_factory=list)
    cross_checks_run: int = 0
    cross_checks_failed: int = 0
    duration_ms: int = 0

    @model_validator(mode="after")
    def _derive_state(self) -> "ValidationReport":
        return self.derive_state()

    def derive_state(self) -> "ValidationReport":
        """Recompute `state` from the current issues and return self.

        MUST be called after mutating ``issues``. A pydantic ``model_validator``
        only runs at *construction*, so a caller that builds a report and then
        assigns ``report.issues = [...]`` would leave ``state`` at its default
        ``PASS`` — a document with errors reported as passing, which is the one
        outcome this type exists to prevent. The validator calls this explicitly
        after setting issues.
        """
        errors = [i for i in self.issues if i.severity is Severity.ERROR]
        warnings = [i for i in self.issues if i.severity is Severity.WARNING]
        if errors:
            self.state = ValidationState.FAIL
        elif warnings:
            self.state = ValidationState.REVIEW
        else:
            self.state = ValidationState.PASS
        return self

    @property
    def errors(self) -> list[FieldIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def needs_review(self) -> bool:
        """Whether a human should look — the routing decision this exists for."""
        return self.state is not ValidationState.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "extraction_id": self.extraction_id,
            "schema_name": self.schema_name,
            "state": self.state.value,
            "needs_review": self.needs_review,
            "counts": {
                "errors": len(self.errors),
                "warnings": len([i for i in self.issues if i.severity is Severity.WARNING]),
                "issues": len(self.issues),
                "cross_checks_run": self.cross_checks_run,
                "cross_checks_failed": self.cross_checks_failed,
            },
            "issues": [i.to_dict() for i in self.issues],
            "duration_ms": self.duration_ms,
        }


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class DocumentSchema(BaseModel):
    """A complete extraction + validation spec for one document type.

    Bundling fields, rules and cross-checks together means an invoice schema is
    one reviewable artifact rather than three that can drift apart.

    ``hints`` are per-field extraction guidance (synonyms, regexes, table
    column names). They are separated from :class:`FieldRule` because extraction
    hints change when a supplier changes their layout, while validation rules
    change when the *business* changes — different owners, different cadence.
    """

    name: str
    description: str = ""
    fields: list[FieldRule] = Field(default_factory=list)
    cross_field: list[CrossFieldRule] = Field(default_factory=list)
    hints: dict[str, list[str]] = Field(
        default_factory=dict,
        description="field name -> synonyms / column headers to look for.",
    )
    #: Azure prebuilt model to use, when the schema maps onto one.
    prebuilt_model: str | None = Field(
        default=None, description="e.g. 'prebuilt-invoice'; None means layout-only."
    )

    @model_validator(mode="after")
    def _rules_reference_real_fields(self) -> "DocumentSchema":
        known = {f.name for f in self.fields}
        if not known and (self.hints or self.cross_field):
            raise ValueError(f"schema {self.name!r} declares rules/hints but no fields")
        for rule in self.cross_field:
            for ref in [*rule.fields, *([rule.target] if rule.target else [])]:
                if ref and ref not in known:
                    raise ValueError(
                        f"cross-field rule {rule.name!r} references unknown field {ref!r}; "
                        f"known fields: {sorted(known)}"
                    )
        for hint_field in self.hints:
            if hint_field not in known:
                raise ValueError(
                    f"hints declared for unknown field {hint_field!r}; "
                    f"known fields: {sorted(known)}"
                )
        return self

    def field(self, name: str) -> FieldRule | None:
        return next((f for f in self.fields if f.name == name), None)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


class DocumentResult(BaseModel):
    """The whole pipeline's output for one document — what a caller actually gets."""

    document: ParsedDocument
    extraction: ExtractionResult
    validation: ValidationReport

    @property
    def routed_to_review(self) -> bool:
        return self.validation.needs_review

    @property
    def usable(self) -> bool:
        """Whether the extraction passed; callers should not silently use a FAIL."""
        return self.validation.state is ValidationState.PASS

    def to_dict(self) -> dict[str, Any]:
        return {
            "document": self.document.to_dict(),
            "extraction": self.extraction.to_dict(),
            "validation": self.validation.to_dict(),
            "routed_to_review": self.routed_to_review,
            "usable": self.usable,
        }
