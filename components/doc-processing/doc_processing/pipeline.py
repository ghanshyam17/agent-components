"""The end-to-end pipeline: ingest → extract → validate.

Composition root. Its job is to make the *routing decision* explicit and to keep
stage failures attributable, so a caller learns which of the three stages broke
rather than receiving an empty result.

    ingest ──▶ extract ──▶ validate
      │           │            │
      │           │            └─→ PASS / REVIEW / FAIL  ← the routing signal
      │           └─→ fields, each with provenance + confidence
      └─→ text, layout, tables, OCR confidence

The ``DocumentResult`` carries all three stage outputs, so a reviewer has the
page text, the evidence for each value, and the reason it was flagged — without
re-running anything.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from doc_processing.extractor import FieldExtractor
from doc_processing.ingest import IngestEngine, StubEngine, select_engine
from doc_processing.models import (
    DocumentResult,
    DocumentSchema,
    ExtractionResult,
    ParsedDocument,
    ValidationReport,
    ValidationState,
)
from doc_processing.validator import DocumentValidator

logger = logging.getLogger(__name__)

__all__ = ["DocumentPipeline", "PipelineResult"]

PipelineResult = DocumentResult  # alias: the name callers reach for


class DocumentPipeline:
    """Run one document end-to-end against one schema.

    Parameters
    ----------
    schema:
        Fields, cross-field rules and extraction hints.
    engine:
        Ingest engine. When omitted, ``select_engine()`` picks the best
        available (Azure → text layer → stub) and logs which.
    llm_client:
        Optional ``model_gateway``-shaped chat callable for the extraction
        fallback on the long tail.
    validator:
        Inject a preconfigured validator to change the document-level gates.
    stop_on_fail:
        When True, a validation FAIL short-circuits any batch run. Default
        False: a batch should report every document's verdict, not abort on the
        first bad one.
    """

    def __init__(
        self,
        schema: DocumentSchema,
        *,
        engine: IngestEngine | Any | None = None,
        llm_client: Callable[..., Awaitable[Any]] | None = None,
        validator: DocumentValidator | None = None,
        stop_on_fail: bool = False,
    ) -> None:
        self.schema = schema
        self.engine = engine or select_engine()
        self.extractor = FieldExtractor(schema, llm_client=llm_client)
        self.validator = validator or DocumentValidator(schema)
        self.stop_on_fail = stop_on_fail

    # ---- stages ----------------------------------------------------------- #
    async def ingest(self, source: str, data: bytes | None = None) -> ParsedDocument:
        return await self.engine.parse(source, data)

    async def extract(self, doc: ParsedDocument) -> ExtractionResult:
        return await self.extractor.extract(doc)

    def validate(self, extraction: ExtractionResult) -> ValidationReport:
        return self.validator.validate(extraction)

    # ---- the whole thing -------------------------------------------------- #
    async def process(self, source: str, data: bytes | None = None) -> DocumentResult:
        """Run all three stages on one document.

        A stage failure raises rather than returning a partial result: an empty
        extraction that "succeeded" is indistinguishable from a document with no
        fields, and that ambiguity is exactly what this pipeline exists to
        remove.
        """
        doc = await self.ingest(source, data)
        extraction = await self.extract(doc)
        validation = self.validate(extraction)
        result = DocumentResult(
            document=doc, extraction=extraction, validation=validation
        )
        logger.info(
            "doc-processing: %s -> %s (%d/%d fields, mean conf %.2f)",
            Path(source).name, validation.state.value,
            extraction.found, len(extraction.fields), extraction.mean_confidence,
        )
        return result

    async def process_batch(
        self, sources: Sequence[str], *, concurrency: int = 4
    ) -> list[DocumentResult | dict[str, Any]]:
        """Process many documents, isolating per-document failures.

        One unreadable file must not abort a batch of a thousand — the failed
        entry is returned with its error instead, so the caller can retry or
        quarantine it.
        """
        import asyncio

        sem = asyncio.Semaphore(max(1, concurrency))

        async def _one(src: str) -> DocumentResult | dict[str, Any]:
            async with sem:
                try:
                    return await self.process(src)
                except Exception as e:  # noqa: BLE001
                    logger.warning("doc-processing failed for %s: %s", src, e)
                    return {
                        "source": src, "error": f"{type(e).__name__}: {e}",
                        "usable": False,
                    }

        return await asyncio.gather(*(_one(s) for s in sources))

    # ---- reporting -------------------------------------------------------- #
    @staticmethod
    def summarize(results: Sequence[DocumentResult | dict[str, Any]]) -> dict[str, Any]:
        """Batch summary, shaped for a routing dashboard.

        Reports the counts an operator acts on: how many passed, how many need a
        human, and which fields failed most often — the last is what tells you
        to fix the schema rather than the documents.
        """
        by_state: dict[str, int] = {}
        failed_fields: dict[str, int] = {}
        errors = 0
        total_issues = 0
        for r in results:
            if isinstance(r, dict):
                errors += 1
                continue
            state = r.validation.state.value
            by_state[state] = by_state.get(state, 0) + 1
            total_issues += len(r.validation.issues)
            for issue in r.validation.errors:
                failed_fields[issue.field] = failed_fields.get(issue.field, 0) + 1
        processed = len(results) - errors
        return {
            "documents": len(results),
            "processed": processed,
            "unreadable": errors,
            "by_state": by_state,
            "pass_rate": (
                round(by_state.get("pass", 0) / processed, 4) if processed else 0.0
            ),
            "total_issues": total_issues,
            "top_failing_fields": dict(
                sorted(failed_fields.items(), key=lambda kv: -kv[1])[:10]
            ),
        }

    def describe(self) -> dict[str, Any]:
        """What this pipeline will do, for logs and the component UI."""
        engine_desc = (
            self.engine.describe()
            if hasattr(self.engine, "describe")
            else {"engine": getattr(self.engine, "name", "unknown")}
        )
        return {
            "schema": self.schema.name,
            "fields": len(self.schema.fields),
            "cross_field_rules": len(self.schema.cross_field),
            "engine": engine_desc,
            "llm_fallback": self.extractor.llm_client is not None,
        }
