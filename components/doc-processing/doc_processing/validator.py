"""Validation: decide whether an extraction can be trusted.

Three layers, and the layering is the point. Each catches a class of error the
others structurally cannot:

1. **Per-field** — type, pattern, range, length, enumerated values, and a
   confidence floor. Catches malformed values.
2. **Cross-field** — relationships between fields (``subtotal + tax == total``,
   ``invoice_date <= due_date``). This is where *most real document errors live*:
   a total that does not reconcile is invisible to any single-field check
   because it is a relationship, not a value.
3. **Document-level** — required-field completeness and mean-confidence gates.
   Catches "the parse worked but nothing was found", which is the classic
   scanned-page failure.

The verdict is derived from the issues (:class:`ValidationReport` computes
``state`` from severity), so the report can never claim PASS while carrying an
error. Routing is explicit: ``FAIL`` means do not use, ``REVIEW`` means a human
should look, ``PASS`` means usable.

Cross-field arithmetic uses a tolerance and **treats a missing operand as
skipped, not failed** — flagging "cannot check" as an error would drown real
findings in noise on partially-extracted documents.
"""
from __future__ import annotations

import logging
import re
import time
from datetime import date, datetime
from typing import Any

from doc_processing.models import (
    CrossFieldRule,
    DocumentSchema,
    ExtractionResult,
    FieldIssue,
    FieldRule,
    Severity,
    ValidationReport,
    ValidationState,
)

logger = logging.getLogger(__name__)

__all__ = ["DocumentValidator"]


def _as_number(value: Any) -> float | None:
    """Coerce to float, returning None when the value is not numeric.

    None (not 0.0) on failure, so a caller can distinguish "not a number" from
    "zero" — treating an unparseable amount as 0.0 would silently make sums
    reconcile against nothing.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if value is None:
        return None
    text = re.sub(r"[^0-9,\.\-]", "", str(value))
    if not text:
        return None
    # Share the separator logic with the extractor so a value cannot normalise
    # one way and validate another.
    from doc_processing.extractor import _strip_thousands

    text = _strip_thousands(text)
    try:
        return float(text)
    except ValueError:
        return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if value is None:
        return None
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%d.%m.%Y",
                "%d/%m/%y", "%m/%d/%y", "%Y/%m/%d", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


class DocumentValidator:
    """Validate an extraction against a schema.

    Parameters
    ----------
    schema:
        The fields and cross-field rules to enforce.
    doc_min_confidence:
        Mean-confidence floor for the whole document. Below it the document is
        routed to review even when every field individually passed — a document
        where everything was barely legible is not a confident extraction.
    doc_min_found_ratio:
        Minimum share of declared fields that must be present. Catches the
        empty-parse case.
    """

    def __init__(
        self,
        schema: DocumentSchema,
        *,
        doc_min_confidence: float = 0.5,
        doc_min_found_ratio: float = 0.5,
    ) -> None:
        self.schema = schema
        self.doc_min_confidence = doc_min_confidence
        self.doc_min_found_ratio = doc_min_found_ratio

    # ---- layer 1: per field ----------------------------------------------- #
    def _check_field(self, rule: FieldRule, field: Any) -> list[FieldIssue]:
        issues: list[FieldIssue] = []
        name = rule.name

        if not field.present:
            if rule.required:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="missing_required",
                        message=f"required field {name!r} was not found in the document",
                        value=None,
                    )
                )
            elif field.evidence.method.value == "not_found":
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.INFO, code="absent",
                        message=f"optional field {name!r} absent", value=None,
                    )
                )
            return issues

        value = field.effective

        # -- confidence floor --
        if field.confidence < rule.min_confidence:
            issues.append(
                FieldIssue(
                    field=name, severity=Severity.WARNING, code="low_confidence",
                    message=(
                        f"{name!r} confidence {field.confidence:.2f} is below "
                        f"the {rule.min_confidence:.2f} floor"
                    ),
                    value=value,
                )
            )

        # -- type --
        if rule.type == "number":
            if _as_number(value) is None:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="not_a_number",
                        message=f"{name!r} is not numeric: {value!r}", value=value,
                    )
                )
        elif rule.type == "integer":
            num = _as_number(value)
            if num is None or float(num).is_integer() is False:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="not_an_integer",
                        message=f"{name!r} is not an integer: {value!r}", value=value,
                    )
                )
        elif rule.type == "date":
            if _as_date(value) is None:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="not_a_date",
                        message=f"{name!r} is not a recognised date: {value!r}", value=value,
                    )
                )
        elif rule.type == "boolean":
            if not isinstance(value, bool) and str(value).strip().lower() not in (
                "true", "false", "yes", "no", "1", "0"
            ):
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="not_a_boolean",
                        message=f"{name!r} is not boolean-like: {value!r}", value=value,
                    )
                )
        elif rule.type == "list":
            if not isinstance(value, list):
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="not_a_list",
                        message=f"{name!r} is not a list: {type(value).__name__}",
                        value=value,
                    )
                )

        # -- pattern --
        if rule.pattern:
            try:
                if not re.fullmatch(rule.pattern, str(value).strip()):
                    issues.append(
                        FieldIssue(
                            field=name, severity=Severity.ERROR, code="pattern_mismatch",
                            message=f"{name!r} value {value!r} does not match {rule.pattern!r}",
                            value=value,
                        )
                    )
            except re.error as e:
                logger.warning("bad validation pattern for %s: %s", name, e)

        # -- numeric bounds --
        num = _as_number(value)
        if num is not None:
            if rule.min_value is not None and num < rule.min_value:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="below_min",
                        message=f"{name!r} {num} is below the minimum {rule.min_value}",
                        value=num,
                    )
                )
            if rule.max_value is not None and num > rule.max_value:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="above_max",
                        message=f"{name!r} {num} exceeds the maximum {rule.max_value}",
                        value=num,
                    )
                )

        # -- length --
        length = len(value) if isinstance(value, (str, list)) else None
        if length is not None:
            if rule.min_length is not None and length < rule.min_length:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="too_short",
                        message=f"{name!r} length {length} < {rule.min_length}",
                        value=value,
                    )
                )
            if rule.max_length is not None and length > rule.max_length:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="too_long",
                        message=f"{name!r} length {length} > {rule.max_length}",
                        value=value,
                    )
                )

        # -- enumerated values --
        if rule.allowed_values:
            if str(value).strip() not in rule.allowed_values:
                issues.append(
                    FieldIssue(
                        field=name, severity=Severity.ERROR, code="not_allowed",
                        message=(
                            f"{name!r} value {value!r} is not one of "
                            f"{rule.allowed_values}"
                        ),
                        value=value,
                    )
                )
        return issues

    # ---- layer 2: cross field --------------------------------------------- #
    def _check_cross(
        self, rule: CrossFieldRule, ext: ExtractionResult
    ) -> tuple[list[FieldIssue], bool]:
        """Evaluate one cross-field rule. Returns (issues, was_evaluated).

        ``was_evaluated`` is False when a required operand is missing, so the
        report can distinguish "the rule passed" from "the rule could not run".
        """
        values = {n: ext.get(n) for n in rule.fields}
        target = ext.get(rule.target) if rule.target else None

        if rule.kind == "one_of":
            if any(v is not None for v in values.values()):
                return [], True
            return [
                FieldIssue(
                    field=", ".join(rule.fields), severity=rule.severity,
                    code="one_of_violated",
                    message=(
                        f"{rule.name}: at least one of {rule.fields} must be present"
                    ),
                    value=None,
                )
            ], True

        if rule.kind == "implies":
            antecedent = values.get(rule.fields[0]) if rule.fields else None
            consequent = values.get(rule.fields[1]) if len(rule.fields) > 1 else None
            if antecedent is None:
                return [], False
            if consequent is not None:
                return [], True
            return [
                FieldIssue(
                    field=rule.fields[1], severity=rule.severity, code="implies_violated",
                    message=(
                        f"{rule.name}: {rule.fields[0]!r} is present but "
                        f"{rule.fields[1]!r} is missing"
                    ),
                    value=None,
                )
            ], True

        # The remaining kinds need every operand.
        if any(values.get(n) is None for n in rule.fields):
            return [], False

        if rule.kind == "sum_equals":
            total = sum(_as_number(values[n]) or 0.0 for n in rule.fields)
            expected = _as_number(target)
            if expected is None:
                return [], False
            if abs(total - expected) > rule.tolerance:
                return [
                    FieldIssue(
                        field=", ".join(rule.fields), severity=rule.severity,
                        code="sum_mismatch",
                        message=(
                            f"{rule.name}: sum of {rule.fields} = {total:.2f} "
                            f"but {rule.target} = {expected:.2f} "
                            f"(off by {abs(total - expected):.2f})"
                        ),
                        value=total,
                    )
                ], True
            return [], True

        if rule.kind == "date_ordering":
            dates = [_as_date(values[n]) for n in rule.fields]
            if any(d is None for d in dates):
                return [], False
            for i in range(len(dates) - 1):
                if dates[i] > dates[i + 1]:  # type: ignore[operator]
                    return [
                        FieldIssue(
                            field=", ".join(rule.fields), severity=rule.severity,
                            code="date_order",
                            message=(
                                f"{rule.name}: {rule.fields[i]} "
                                f"({dates[i]}) is after {rule.fields[i + 1]} "
                                f"({dates[i + 1]})"
                            ),
                            value=str(dates[i]),
                        )
                    ], True
            return [], True

        if rule.kind == "gte":
            nums = [_as_number(values[n]) for n in rule.fields]
            if any(n is None for n in nums):
                return [], False
            for i in range(len(nums) - 1):
                if nums[i] < nums[i + 1] - rule.tolerance:  # type: ignore[operator]
                    return [
                        FieldIssue(
                            field=", ".join(rule.fields), severity=rule.severity,
                            code="numeric_order",
                            message=(
                                f"{rule.name}: {rule.fields[i]} ({nums[i]:.2f}) is less "
                                f"than {rule.fields[i + 1]} ({nums[i + 1]:.2f})"
                            ),
                            value=nums[i],
                        )
                    ], True
            return [], True

        if rule.kind in ("equals", "not_equals"):
            a, b = values[rule.fields[0]], values[rule.fields[1]]
            same = (
                abs((_as_number(a) or 0) - (_as_number(b) or 0)) <= rule.tolerance
                if _as_number(a) is not None and _as_number(b) is not None
                else str(a).strip() == str(b).strip()
            )
            if rule.kind == "equals" and not same:
                return [
                    FieldIssue(
                        field=", ".join(rule.fields), severity=rule.severity,
                        code="not_equal",
                        message=f"{rule.name}: {a!r} != {b!r}", value=a,
                    )
                ], True
            if rule.kind == "not_equals" and same:
                return [
                    FieldIssue(
                        field=", ".join(rule.fields), severity=rule.severity,
                        code="equal_but_should_not_be",
                        message=f"{rule.name}: {a!r} should differ from {b!r}", value=a,
                    )
                ], True
            return [], True

        logger.warning("unknown cross-field kind %r in rule %r", rule.kind, rule.name)
        return [], False

    # ---- the pipeline ----------------------------------------------------- #
    def validate(self, extraction: ExtractionResult) -> ValidationReport:
        """Validate an extraction and return the report."""
        started = time.perf_counter()
        report = ValidationReport(
            extraction_id=extraction.id,
            schema_name=self.schema.name,
        )
        issues: list[FieldIssue] = []

        # Layer 1
        for rule in self.schema.fields:
            field = extraction.fields.get(rule.name)
            if field is None:
                if rule.required:
                    issues.append(
                        FieldIssue(
                            field=rule.name, severity=Severity.ERROR,
                            code="missing_required",
                            message=f"required field {rule.name!r} was never extracted",
                        )
                    )
                continue
            issues.extend(self._check_field(rule, field))

        # Layer 2
        for rule in self.schema.cross_field:
            report.cross_checks_run += 1
            found, evaluated = self._check_cross(rule, extraction)
            if not evaluated:
                report.cross_checks_run -= 1
                issues.append(
                    FieldIssue(
                        field=", ".join(rule.fields) or rule.name,
                        severity=Severity.INFO, code="check_skipped",
                        message=(
                            f"{rule.name}: not evaluated — a required operand is "
                            "missing, so no judgement is possible"
                        ),
                    )
                )
                continue
            if found:
                report.cross_checks_failed += 1
                issues.extend(found)

        # Layer 3 — document level
        total_declared = len(self.schema.fields)
        if total_declared:
            ratio = extraction.found / total_declared
            if ratio < self.doc_min_found_ratio:
                # Two structurally different situations, and the fix differs, so
                # they get different codes: nothing found at all usually means
                # the parse produced no usable text (blank or un-OCR'd scan),
                # whereas *some* fields found means the schema or the document
                # type is the likely problem.
                nothing = extraction.found == 0
                issues.append(
                    FieldIssue(
                        field="<document>",
                        severity=Severity.ERROR,
                        code="empty_parse" if nothing else "low_found_ratio",
                        message=(
                            f"only {extraction.found}/{total_declared} fields "
                            f"({ratio:.0%}) were found; below the "
                            f"{self.doc_min_found_ratio:.0%} floor — "
                            + (
                                "nothing was readable, so the parse likely failed "
                                "or the document is a scan without OCR"
                                if nothing
                                else "the document may not match this schema, or "
                                "extraction is failing for this layout"
                            )
                        ),
                    )
                )
        if extraction.found and extraction.mean_confidence < self.doc_min_confidence:
            issues.append(
                FieldIssue(
                    field="<document>", severity=Severity.WARNING, code="low_mean_confidence",
                    message=(
                        f"mean field confidence {extraction.mean_confidence:.2f} is below "
                        f"{self.doc_min_confidence:.2f}; the document is weakly legible"
                    ),
                )
            )

        report.issues = issues
        # Re-derive: the pydantic validator only ran at construction, when the
        # issue list was still empty. Without this the report would claim PASS
        # while carrying errors.
        report.derive_state()
        report.duration_ms = int((time.perf_counter() - started) * 1000)
        return report
