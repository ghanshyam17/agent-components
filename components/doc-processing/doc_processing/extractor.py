"""Extraction: pull schema fields out of a ``ParsedDocument``, with provenance.

Four strategies, tried in a deliberate order of trustworthiness. The order is
the design: a value read from a model-extracted field is more reliable than one
found by regex, which is more reliable than a positional guess, and the
confidence scores encode exactly that.

1. **Document Intelligence prebuilt/custom fields** (:attr:`ExtractionMethod.DOCUMENT_INTELLIGENCE`)
   The service already located and typed the field. Highest confidence.
2. **Regex over the full text** (``REGEX``)
   Deterministic and explainable — the evidence records the pattern that matched.
3. **Table lookup** (``TABLE``)
   Line items, where the header row names the column.
4. **Heuristic keyword/proximity scan** (``HEURISTIC``)
   "label: value" on the same line. Genuinely useful for invoices, and honestly
   labelled as a guess rather than dressed up as extraction.

An optional LLM strategy (:attr:`ExtractionMethod.LLM`) is available through a
``model_gateway`` client for the long tail that none of the above handle. It is
**last**, and it never overwrites a value another strategy already found: a
model's confident-sounding guess must not displace a regex's exact match.

Every field carries a :class:`FieldEvidence`. Nothing is extracted without it.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from datetime import date, datetime
from typing import Any, Awaitable, Callable, Sequence

from doc_processing.models import (
    BoundingBox,
    DocumentSchema,
    ExtractedField,
    ExtractionMethod,
    ExtractionResult,
    FieldEvidence,
    ParsedDocument,
    ParsedTable,
)

logger = logging.getLogger(__name__)

__all__ = ["FieldExtractor", "normalize_value", "DEFAULT_PATTERNS"]

#: Built-in patterns per normalised field-name fragment. These are deliberately
#: conservative: a pattern that matches too eagerly produces confident wrong
#: values, which is worse than a missing one because missing values get
#: reviewed and wrong ones do not.
# NOTE ON REGEX CASING. Patterns are applied with re.IGNORECASE, and that flag
# ALSO applies inside character classes — `[A-Z0-9]` then matches lowercase too,
# which silently turned "Invoice No: INV-2024-0042" into the value "Invoice".
# Identifier patterns therefore use `(?-i:...)` to switch case-sensitivity off
# for the capture group only, keeping the label match case-insensitive.
#
# NOTE ON WORD BOUNDARIES. `total` without a leading boundary matches inside
# "Subtotal", so the subtotal value was read as the total. Every label here is
# anchored with \b.
DEFAULT_PATTERNS: dict[str, str] = {
    "invoice_number": r"\b(?:invoice|facture|inv)\b\s*(?:no\.?|number|#|num)?\s*[:\-]?\s*(?-i:([A-Z0-9][A-Z0-9\-\/]{2,}))",
    "po_number": r"\b(?:p\.?o\.?|purchase\s*order)\b\s*(?:no\.?|number|#)?\s*[:\-]?\s*(?-i:([A-Z0-9][A-Z0-9\-\/]{2,}))",
    "date": r"\b(\d{4}-\d{2}-\d{2}|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})\b",
    # `total` must not match inside "subtotal": require the label to start a word
    # and exclude a preceding "sub".
    "total": r"(?<!sub)\b(?:grand\s+)?total\b\s*(?:ttc|due)?\s*[:\-]?\s*([€$£]?\s?\d[\d\s,\.]*\d)",
    "subtotal": r"\b(?:subtotal|sub-total|sous[-\s]?total|net)\b\s*[:\-]?\s*([€$£]?\s?\d[\d\s,\.]*\d)",
    "tax": r"\b(?:vat|tva|tax|gst)\b\s*(?:@\s*\d+%?)?\s*[:\-]?\s*([€$£]?\s?\d[\d\s,\.]*\d)",
    "email": r"([\w\.\-\+]+@[\w\-]+\.[\w\.\-]+)",
    "phone": r"(\+?\d[\d\s\-\(\)]{7,}\d)",
    "currency": r"\b(EUR|USD|GBP|CAD|CHF)\b",
}

#: Labels that identify *which* date a value is, for date disambiguation.
#: A generic date pattern returns the first date on the page, which is the
#: invoice date far more often than the due date — so a field asking for
#: "due_date" must look for its own label before settling for a bare date.
DATE_LABELS: dict[str, list[str]] = {
    "invoice_date": ["invoice date", "date of issue", "issued", "issue date"],
    "due_date": ["due date", "payment due", "due", "échéance", "echeance"],
    "transaction_date": ["date", "transaction date", "date of purchase"],
    "effective_date": ["effective date", "commencement", "start date", "entered into"],
    "expiry_date": ["expiry", "expiration", "expires", "end date", "valid until", "date of expiry"],
    "date_of_birth": ["date of birth", "dob", "born", "naissance"],
}

#: Field-name fragments that indicate a date, so `normalize_value` can parse
#: without the caller declaring a type for every field.
_DATE_HINTS = ("date", "due", "issued", "expiry", "expires", "start", "end")


def _strip_thousands(text: str) -> str:
    """Resolve the thousands/decimal separator ambiguity in a numeric string.

    Two conventions collide and the digits alone cannot disambiguate:

    * ``1,234.56`` — comma thousands, dot decimal (en-US)
    * ``1.234,56`` — dot thousands, comma decimal (most of Europe)

    The rule is the **rightmost** separator: it is the decimal one, because the
    fractional part is always last. Assuming a fixed convention produced
    ``1.234,56 -> 1.23456`` (a value 1000x too small) for European amounts,
    which then failed reconciliation against a correctly parsed total.
    """
    has_comma, has_dot = "," in text, "." in text
    if has_comma and has_dot:
        if text.rfind(",") > text.rfind("."):
            return text.replace(".", "").replace(",", ".")   # European
        return text.replace(",", "")                          # en-US
    if has_comma:
        # A single comma with exactly two trailing digits is a decimal comma.
        if text.count(",") == 1 and len(text.split(",")[-1]) == 2:
            return text.replace(",", ".")
        return text.replace(",", "")
    return text


def normalize_value(value: Any, name: str = "") -> Any:
    """Best-effort normalisation of an extracted value.

    Currency strings become floats, dates become ISO-8601, and everything else
    is stripped of surrounding whitespace. Returns the input unchanged when no
    confident interpretation exists — a wrong normalisation is worse than none,
    because it looks authoritative.
    """
    if value is None or isinstance(value, (int, float, bool, list, dict)):
        return value
    text = str(value).strip()
    if not text:
        return text

    low = name.lower()
    is_date = any(h in low for h in _DATE_HINTS)
    if is_date:
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
                    "%d/%m/%y", "%m/%d/%y", "%Y/%m/%d", "%d %B %Y", "%d %b %Y"):
            try:
                return datetime.strptime(text, fmt).date().isoformat()
            except ValueError:
                continue

    if any(k in low for k in ("total", "subtotal", "tax", "amount", "price", "vat", "net")):
        # Strip currency symbols and thin/non-breaking spaces, then decide
        # whether a comma is a decimal separator or a thousands separator.
        cleaned = re.sub(r"[^0-9,\.\-]", "", text)
        if cleaned:
            cleaned = _strip_thousands(cleaned)
            try:
                return float(cleaned)
            except ValueError:
                return text
    return text


class FieldExtractor:
    """Extract schema fields from a parsed document.

    Parameters
    ----------
    schema:
        The fields, hints and cross-checks to extract against.
    llm_client:
        Optional ``model_gateway``-shaped chat callable for the LLM fallback.
    min_confidence:
        Fields below this are still returned (the validator needs to see them to
        flag them) but the extractor records a warning.
    """

    def __init__(
        self,
        schema: DocumentSchema,
        *,
        llm_client: Callable[..., Awaitable[Any]] | None = None,
        min_confidence: float = 0.3,
    ) -> None:
        self.schema = schema
        self.llm_client = llm_client
        self.min_confidence = min_confidence

    # ---- helpers ---------------------------------------------------------- #
    def _pattern_for(self, name: str) -> str | None:
        """Find a pattern for a field name — by exact match, then by suffix.

        Deliberately *not* a loose substring search. `vendor_tax_id` contains
        "tax", so a substring fallback silently applied the monetary *tax*
        pattern and produced `vendor_tax_id = 240.0`. Identifier-style fields
        end in `_id` / `_number` / `_ref` and should never borrow a money or
        date pattern, so those suffixes are excluded from the fallback.
        """
        low = name.lower()
        if low in DEFAULT_PATTERNS:
            return DEFAULT_PATTERNS[low]
        if low.endswith(("_id", "_number", "_no", "_ref", "_code")):
            return None
        for key, pat in DEFAULT_PATTERNS.items():
            # Match only when the *whole* key appears as a token in the name.
            if key in low.split("_") or low.endswith(key):
                return pat
        return None

    @staticmethod
    def _evidence_for_line(
        doc: ParsedDocument, needle: str
    ) -> tuple[BoundingBox | None, int | None, str | None]:
        """Locate `needle` in the document, returning (bbox, page, source line)."""
        if not needle:
            return None, None, None
        probe = needle.strip().lower()
        for page in doc.pages:
            for span in page.lines:
                if probe and probe in span.text.lower():
                    return span.bbox, span.page, span.text
        return None, None, None

    # ---- strategy 1: Document Intelligence fields ------------------------- #
    def _di_fields(self, doc: ParsedDocument) -> dict[str, ExtractedField]:
        """Use the typed fields the service already extracted.

        DI returns these only for prebuilt/custom models. Each field may be a
        simple value or a structured object with its own confidence, so both
        shapes are handled.
        """
        out: dict[str, ExtractedField] = {}
        documents = (doc.metadata or {}).get("di_documents") or []
        if not documents:
            return out
        fields = documents[0].get("fields") or {}
        wanted = {f.name.lower(): f.name for f in self.schema.fields}
        for key, payload in fields.items():
            name = wanted.get(key.lower())
            if name is None:
                # Also accept a field whose normalised name contains the DI key.
                for cand in wanted:
                    if cand in key.lower() or key.lower() in cand:
                        name = wanted[cand]
                        break
            if name is None:
                continue
            value, conf, source = self._di_value(payload)
            if value is None:
                continue
            bbox, page, line = self._evidence_for_line(doc, str(source or value))
            rule = self.schema.field(name)
            out[name] = ExtractedField(
                name=name,
                value=value,
                confidence=conf,
                required=bool(rule.required) if rule else False,
                normalized=normalize_value(value, name),
                evidence=FieldEvidence(
                    method=ExtractionMethod.DOCUMENT_INTELLIGENCE,
                    page=page,
                    bbox=bbox,
                    source_text=line or (str(source) if source else None),
                    engine="azure-document-intelligence",
                ),
            )
        return out

    @staticmethod
    def _di_value(payload: Any) -> tuple[Any, float, Any]:
        """Unwrap a DI field payload into (value, confidence, raw_source)."""
        if not isinstance(payload, dict):
            return payload, 0.6, None
        conf = float(payload.get("confidence", 0.8) or 0.0)
        ftype = payload.get("type", "")
        key = {
            "string": "valueString", "date": "valueDate", "number": "valueNumber",
            "integer": "valueInteger", "currency": "valueCurrency",
            "boolean": "valueBoolean", "array": "valueArray", "object": "valueObject",
        }.get(ftype)
        if key and key in payload:
            val = payload[key]
        elif "content" in payload:
            val = payload["content"]
        else:
            val = None
        if isinstance(val, dict) and "amount" in val:
            val = val["amount"]
        source = payload.get("content")
        return val, conf, source

    # ---- strategy 2: regex ------------------------------------------------ #
    def _regex_fields(self, doc: ParsedDocument) -> dict[str, ExtractedField]:
        out: dict[str, ExtractedField] = {}
        text = doc.text
        for rule in self.schema.fields:
            pattern = rule.pattern or self._pattern_for(rule.name)
            if not pattern:
                continue
            try:
                match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
            except re.error as e:
                logger.warning("invalid pattern for %s: %s", rule.name, e)
                continue
            if not match:
                continue
            value = (match.group(1) if match.groups() else match.group(0)).strip()
            bbox, page, line = self._evidence_for_line(doc, value)
            confidence = None
            # A regex match is exact but the *field association* is assumed, so
            # it sits below a model that located the field explicitly.
            confidence = 0.85 if rule.pattern else 0.75
            if not line:
                confidence -= 0.15  # matched text not found in any line: weaker
            out[rule.name] = ExtractedField(
                name=rule.name,
                value=value,
                confidence=round(max(0.0, confidence), 3),
                required=rule.required,
                normalized=normalize_value(value, rule.name),
                evidence=FieldEvidence(
                    method=ExtractionMethod.REGEX,
                    page=page, bbox=bbox, source_text=line,
                    pattern=pattern,
                ),
            )
        return out

    def _labelled_date(self, doc: ParsedDocument, name: str) -> ExtractedField | None:
        """Find a date by its *label* rather than by taking the first date found.

        A bare date pattern returns whatever date appears first, which on an
        invoice is the invoice date — so `due_date` was silently populated with
        the invoice date and the `invoice_date <= due_date` cross-check then
        passed on a wrong value. Matching the label first is the only correct
        way to disambiguate two dates of the same shape.
        """
        labels = DATE_LABELS.get(name.lower())
        if not labels:
            return None
        date_re = r"(\d{4}-\d{2}-\d{2}|\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})"
        for page in doc.pages:
            for span in page.lines:
                low = span.text.lower()
                for label in labels:
                    if label not in low:
                        continue
                    m = re.search(rf"{re.escape(label)}\s*[:\-]?\s*{date_re}", low)
                    if not m:
                        continue
                    value = m.group(1)
                    return ExtractedField(
                        name=name,
                        value=value,
                        # Higher than a bare-date match: the label says what it is.
                        confidence=0.88,
                        normalized=normalize_value(value, name),
                        evidence=FieldEvidence(
                            method=ExtractionMethod.REGEX,
                            page=span.page, bbox=span.bbox, source_text=span.text,
                            pattern=f"label:{label}",
                        ),
                    )
        return None

    # ---- strategy 3: tables ----------------------------------------------- #
    def _table_fields(self, doc: ParsedDocument) -> dict[str, ExtractedField]:
        """Lift fields from tables using header names and schema hints."""
        out: dict[str, ExtractedField] = {}
        for table in doc.all_tables:
            headers = table.header_map()
            if not headers:
                continue
            for rule in self.schema.fields:
                if rule.name in out:
                    continue
                wanted = [rule.name.lower(), *[h.lower() for h in self.schema.hints.get(rule.name, [])]]
                col = next((headers[w] for w in wanted if w in headers), None)
                if col is None:
                    continue
                cell = table.cell(1, col) if table.n_rows > 1 else None
                if cell is None or not cell.text.strip():
                    continue
                out[rule.name] = ExtractedField(
                    name=rule.name,
                    value=cell.text.strip(),
                    confidence=0.7,
                    required=rule.required,
                    normalized=normalize_value(cell.text.strip(), rule.name),
                    evidence=FieldEvidence(
                        method=ExtractionMethod.TABLE,
                        page=table.page,
                        source_text=cell.text.strip(),
                        table_ref=f"{table.id}:r{cell.row}c{cell.column}",
                    ),
                )
        return out

    # ---- strategy 4: heuristic label scan --------------------------------- #
    def _heuristic_fields(self, doc: ParsedDocument) -> dict[str, ExtractedField]:
        """Find "label: value" on a line, using the schema's hints as labels."""
        out: dict[str, ExtractedField] = {}
        for rule in self.schema.fields:
            if rule.name in out:
                continue
            labels = [rule.name.replace("_", " "), *self.schema.hints.get(rule.name, [])]
            for page in doc.pages:
                hit = None
                for span in page.lines:
                    low = span.text.lower()
                    for label in labels:
                        if not label:
                            continue
                        m = re.search(
                            re.escape(label.lower()) + r"\s*[:\-]\s*(.+)$", low
                        )
                        if m:
                            hit = (span, m.group(1).strip())
                            break
                    if hit:
                        break
                if hit:
                    span, value = hit
                    out[rule.name] = ExtractedField(
                        name=rule.name,
                        value=value,
                        confidence=0.55,  # a positional guess, labelled as such
                        required=rule.required,
                        normalized=normalize_value(value, rule.name),
                        evidence=FieldEvidence(
                            method=ExtractionMethod.HEURISTIC,
                            page=span.page, bbox=span.bbox, source_text=span.text,
                        ),
                    )
                    break
        return out

    # ---- strategy 5: LLM fallback ----------------------------------------- #
    async def _llm_fields(
        self, doc: ParsedDocument, want: Sequence[str]
    ) -> dict[str, ExtractedField]:
        """Ask a model for the fields nothing else found.

        Deliberately last and deliberately narrow: it is asked only for fields
        that are still missing, so it can never overwrite a deterministic match.
        Confidence is capped below the regex tier for the same reason — a
        plausible-sounding model answer should not outrank an exact one.
        """
        if self.llm_client is None or not want:
            return {}
        from components_core import Message

        listing = "\n".join(f"- {name}" for name in want)
        prompt = (
            "Extract these fields from the document below. Reply with ONLY a JSON "
            "object mapping field name to the exact value found in the text. Use "
            "null for any field you cannot find verbatim. Never invent values.\n\n"
            f"Fields:\n{listing}\n\nDocument:\n{doc.text[:12000]}"
        )
        try:
            out = await self.llm_client([Message(role="user", content=prompt)])
        except Exception as e:  # noqa: BLE001 - a fallback must not fail the pipeline
            logger.warning("LLM extraction failed: %s", e)
            return {}
        content = out[0] if isinstance(out, (tuple, list)) and out else str(out or "")

        import json

        match = re.search(r"\{.*\}", content or "", re.DOTALL)
        if not match:
            return {}
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        if not isinstance(data, dict):
            return {}

        found: dict[str, ExtractedField] = {}
        for name, value in data.items():
            if value is None or name not in want:
                continue
            rule = self.schema.field(name)
            bbox, page, line = self._evidence_for_line(doc, str(value))
            found[name] = ExtractedField(
                name=name,
                value=value,
                # Capped at 0.6: below regex (0.75-0.85) by design.
                confidence=0.6 if line else 0.4,
                required=bool(rule and rule.required),
                normalized=normalize_value(value, name),
                evidence=FieldEvidence(
                    method=ExtractionMethod.LLM,
                    page=page, bbox=bbox, source_text=line,
                    engine="llm-fallback",
                ),
            )
        return found

    # ---- the pipeline ----------------------------------------------------- #
    async def extract(self, doc: ParsedDocument) -> ExtractionResult:
        """Run every strategy in order of trustworthiness and merge the results.

        Merge rule: the **first** strategy to produce a value for a field wins,
        because the strategies are ordered by trust. Later strategies fill only
        the gaps. Missing fields are recorded explicitly as ``NOT_FOUND`` so the
        validator can distinguish "absent" from "never looked for".
        """
        started = time.perf_counter()
        result = ExtractionResult(
            document_id=doc.id,
            schema_name=self.schema.name,
            engine=doc.engine,
        )

        di = self._di_fields(doc)
        # Labelled dates first: they disambiguate same-shaped dates that a
        # generic pattern cannot tell apart. Only fills fields the DI pass left.
        labelled: dict[str, ExtractedField] = {}
        for rule in self.schema.fields:
            if rule.name in di or rule.name in labelled:
                continue
            found = self._labelled_date(doc, rule.name)
            if found is not None:
                found.required = rule.required
                labelled[rule.name] = found
        regex = self._regex_fields(doc)
        tables = self._table_fields(doc)
        heuristic = self._heuristic_fields(doc)

        for strategy in (di, labelled, regex, tables, heuristic):
            for name, field in strategy.items():
                result.fields.setdefault(name, field)

        missing = [f.name for f in self.schema.fields if f.name not in result.fields]
        if missing and self.llm_client is not None:
            for name, field in (await self._llm_fields(doc, missing)).items():
                result.fields.setdefault(name, field)

        # Record the fields nothing found, so downstream can tell "absent" from
        # "not attempted".
        for rule in self.schema.fields:
            if rule.name not in result.fields:
                result.fields[rule.name] = ExtractedField(
                    name=rule.name,
                    value=None,
                    confidence=0.0,
                    required=rule.required,
                    evidence=FieldEvidence(method=ExtractionMethod.NOT_FOUND),
                )

        if result.missing_required:
            result.warnings.append(
                f"missing required field(s): {', '.join(result.missing_required)}"
            )
        low = [n for n, f in result.fields.items() if f.present and f.confidence < self.min_confidence]
        if low:
            result.warnings.append(f"low-confidence field(s): {', '.join(sorted(low))}")

        # The dominant method, for the result's headline label.
        methods = [f.evidence.method for f in result.fields.values() if f.present]
        if methods:
            result.method = max(set(methods), key=methods.count)

        result.duration_ms = int((time.perf_counter() - started) * 1000)
        return result
