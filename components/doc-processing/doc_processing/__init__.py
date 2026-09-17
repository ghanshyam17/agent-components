"""doc-processing: intelligent document processing for agent pipelines.

Turns files into validated, auditable structured data in three stages:

    ingest ──▶ extract ──▶ validate
    OCR + layout   schema fields     rules + cross-checks
                   with provenance    → PASS / REVIEW / FAIL

A chassis child extension: it consumes ``components-core``, ``model-gateway``
(for the LLM extraction fallback), ``prompt-registry``, ``guardrails`` and
``eval-harness``, and reimplements none of them.

Quick start — offline (no Azure, no OCR stack)::

    import asyncio
    from doc_processing import DocumentPipeline, INVOICE_SCHEMA, StubEngine

    text = "Invoice No: INV-2024-001\\nTotal: 1,234.56 EUR"
    pipe = DocumentPipeline(INVOICE_SCHEMA, engine=StubEngine(text=text))
    result = asyncio.run(pipe.process("invoice.txt"))

    print(result.validation.state.value)          # 'pass' | 'review' | 'fail'
    print(result.extraction.get("total"))          # 1234.56
    print(result.extraction.fields["total"].evidence.method.value)  # 'regex'

Live Azure AI Document Intelligence (OCR + prebuilt models)::

    from doc_processing import AzureDocumentIntelligenceEngine, DocumentPipeline

    engine = AzureDocumentIntelligenceEngine(
        "https://<resource>.cognitiveservices.azure.com",   # Entra auth by default
        model="prebuilt-invoice",
    )
    pipe = DocumentPipeline(INVOICE_SCHEMA, engine=engine)
    result = await pipe.process("scan.pdf")

See ``README.md`` for the architecture, the provenance model, and the validation
layers.
"""
from __future__ import annotations

from doc_processing.extractor import DEFAULT_PATTERNS, FieldExtractor, normalize_value
from doc_processing.ingest import (
    DI_API_VERSION,
    AzureDocumentIntelligenceEngine,
    IngestEngine,
    StubEngine,
    TextLayerEngine,
    select_engine,
)
from doc_processing.models import (
    BoundingBox,
    CrossFieldRule,
    DocumentResult,
    DocumentSchema,
    ExtractedField,
    ExtractionMethod,
    ExtractionResult,
    FieldEvidence,
    FieldIssue,
    FieldRule,
    Page,
    ParsedDocument,
    ParsedTable,
    Severity,
    SourceKind,
    TableCell,
    TextSpan,
    ValidationReport,
    ValidationState,
)
from doc_processing.pipeline import DocumentPipeline, PipelineResult
from doc_processing.schemas import (
    CONTRACT_SCHEMA,
    ID_DOCUMENT_SCHEMA,
    INVOICE_SCHEMA,
    PRESETS,
    RECEIPT_SCHEMA,
)
from doc_processing.validator import DocumentValidator

__all__ = [
    # pipeline
    "DocumentPipeline", "PipelineResult",
    # engines
    "AzureDocumentIntelligenceEngine", "TextLayerEngine", "StubEngine",
    "IngestEngine", "select_engine", "DI_API_VERSION",
    # stages
    "FieldExtractor", "DocumentValidator", "normalize_value", "DEFAULT_PATTERNS",
    # models
    "ParsedDocument", "Page", "ParsedTable", "TableCell", "TextSpan", "BoundingBox",
    "DocumentSchema", "FieldRule", "CrossFieldRule",
    "ExtractionResult", "ExtractedField", "FieldEvidence", "ExtractionMethod",
    "ValidationReport", "FieldIssue", "ValidationState", "Severity",
    "DocumentResult", "SourceKind",
    # schemas
    "INVOICE_SCHEMA", "RECEIPT_SCHEMA", "CONTRACT_SCHEMA", "ID_DOCUMENT_SCHEMA",
    "PRESETS",
]
