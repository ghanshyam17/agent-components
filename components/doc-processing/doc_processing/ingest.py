"""Ingestion: turn a file or bytes into a structured ``ParsedDocument``.

Three engines behind one interface, selected by what is available:

``AzureDocumentIntelligenceEngine``
    The real thing. Calls Azure AI Document Intelligence, which returns layout,
    tables, selection marks and per-word OCR confidence. Requires no local OCR
    tooling because OCR happens service-side.
``TextLayerEngine``
    Reads a PDF's embedded text layer (``pypdf``) or a plain text file. No OCR,
    so it is exact for born-digital documents and useless for scans — and it says
    so rather than returning an empty document.
``StubEngine``
    Deterministic, dependency-free ingestion for tests and CI. Splits supplied
    or synthetic text into pages/lines/table so the rest of the pipeline can be
    exercised offline.

Authentication is Entra-first. Document Intelligence accepted API keys in the
past, but keys do not expire and cannot be scoped; the adapter defaults to
``DefaultAzureCredential`` and only falls back to a key when one is explicitly
supplied.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

from doc_processing.models import (
    BoundingBox,
    Page,
    ParsedDocument,
    ParsedTable,
    SourceKind,
    TableCell,
    TextSpan,
)

logger = logging.getLogger(__name__)

__all__ = [
    "IngestEngine",
    "AzureDocumentIntelligenceEngine",
    "TextLayerEngine",
    "StubEngine",
    "select_engine",
    "DI_API_VERSION",
]

#: Document Intelligence REST API version. Pinned because the response shape is
#: version-dependent and a silent bump would break parsing in production.
DI_API_VERSION = "2024-11-30"

#: Cognitive Services token scope for Entra auth.
COGNITIVE_SCOPE = "https://cognitiveservices.azure.com/.default"


def _classify(source: str, data: bytes | None = None) -> SourceKind:
    """Infer the source kind from the path, then the magic bytes.

    Content sniffing matters for streams and uploads where the filename lies
    (a `.pdf` that is really a PNG is common in scraped corpora).
    """
    kind = SourceKind.from_suffix(Path(source).suffix)
    if data:
        head = data[:8]
        if head.startswith(b"%PDF"):
            return SourceKind.PDF
        if head.startswith(b"\x89PNG") or head.startswith(b"\xff\xd8") or head.startswith(b"II*\x00"):
            return SourceKind.IMAGE
        if head.startswith(b"PK\x03\x04") and kind is SourceKind.UNKNOWN:
            return SourceKind.DOCX
    return kind


class IngestEngine(Protocol):
    """What every ingestion engine must provide."""

    name: str

    async def parse(self, source: str, data: bytes | None = None) -> ParsedDocument: ...


# --------------------------------------------------------------------------- #
# Azure AI Document Intelligence
# --------------------------------------------------------------------------- #
#: Maps DI's prebuilt model names onto the layout model that yields structure.
_LAYOUT_MODEL = "prebuilt-layout"


class AzureDocumentIntelligenceEngine:
    """OCR + layout via Azure AI Document Intelligence (REST).

    Uses the REST API directly rather than the SDK: the analyze operation is a
    long-running POST/poll pair, which is a few lines of HTTP, and doing it
    explicitly avoids pinning an SDK version whose model classes change between
    releases. It also means Entra tokens can be re-minted per request.

    Parameters
    ----------
    endpoint:
        Cognitive Services endpoint, e.g.
        ``https://<resource>.cognitiveservices.azure.com``. An ``AIServices``
        kind account serves Document Intelligence on the same host.
    model:
        DI model id. ``prebuilt-layout`` for structure; a prebuilt or custom
        model (e.g. ``prebuilt-invoice``) also returns its own ``documents``
        array of typed fields, which :mod:`doc_processing.extractor` consumes.
    api_key:
        Optional. When omitted, Entra (``DefaultAzureCredential``) is used.
    fetcher:
        Injected async ``(method, url, headers, body) -> (status, json)``. The
        seam that makes this class testable without network access.
    """

    name = "azure-document-intelligence"

    def __init__(
        self,
        endpoint: str | None = None,
        *,
        model: str = _LAYOUT_MODEL,
        api_key: str | None = None,
        # Accepts (status, body, headers) — or the older (status, body) two-tuple,
        # which _fetch pads with empty headers.
        fetcher: Callable[..., Awaitable[Any]] | None = None,
        poll_interval: float = 1.0,
        timeout_s: float = 120.0,
        sleep: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        self.endpoint = (endpoint or os.environ.get("DOC_INTELLIGENCE_ENDPOINT") or "").rstrip("/")
        self.model = model
        self.api_key = api_key or os.environ.get("DOC_INTELLIGENCE_KEY")
        self._fetcher = fetcher
        self.poll_interval = poll_interval
        self.timeout_s = timeout_s
        self._sleep = sleep or asyncio.sleep
        self._credential: Any | None = None

    # ---- availability ----------------------------------------------------- #
    @property
    def configured(self) -> bool:
        return bool(self.endpoint)

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "configured": self.configured,
            "endpoint": self.endpoint or None,
            "model": self.model,
            "auth": "api_key" if self.api_key else "entra",
            "api_version": DI_API_VERSION,
        }

    # ---- auth ------------------------------------------------------------- #
    def _token(self) -> str | None:
        """Mint an Entra token, caching the credential (not the token)."""
        if self.api_key:
            return None
        try:
            from azure.identity import DefaultAzureCredential
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "azure-identity is required for Entra auth: "
                "pip install 'doc-processing[azure]'"
            ) from e
        if self._credential is None:
            self._credential = DefaultAzureCredential()
        # A fresh token per request: the analyze+poll cycle can outlive one.
        return self._credential.get_token(COGNITIVE_SCOPE).token

    def _headers(self) -> dict[str, str]:
        token = self._token()
        if token:
            return {"Authorization": f"Bearer {token}", "Content-Type": "application/octet-stream"}
        return {"Ocp-Apim-Subscription-Key": self.api_key or "", "Content-Type": "application/octet-stream"}

    # ---- HTTP ------------------------------------------------------------- #
    async def _default_fetcher(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None
    ) -> tuple[int, dict, dict[str, str]]:
        import httpx

        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            resp = await client.request(method, url, headers=headers, content=body)
            try:
                payload = resp.json() if resp.content else {}
            except Exception:  # noqa: BLE001 - non-JSON error bodies are common
                payload = {"raw": resp.text[:2000]}
            # Headers are returned because the analyze POST is a 202 with an
            # EMPTY body: `operation-location` lives in the response *headers*.
            # Discarding them silently lost the operation URL, so the poll never
            # ran and the parse returned an empty document that looked valid.
            return resp.status_code, payload, dict(resp.headers)

    async def _fetch(
        self, method: str, url: str, headers: dict[str, str], body: bytes | None = None
    ) -> tuple[int, dict, dict[str, str]]:
        """Fetch, returning ``(status, json_body, response_headers)``.

        A two-tuple fetcher (the older, simpler test seam) is still accepted and
        padded with empty headers, so injecting one remains ergonomic.
        """
        fetcher = self._fetcher or self._default_fetcher
        result = await fetcher(method, url, headers, body)
        if len(result) == 2:  # type: ignore[arg-type]
            status, payload = result  # type: ignore[misc]
            return status, payload, {}
        return result  # type: ignore[return-value]

    # ---- the operation ---------------------------------------------------- #
    async def parse(self, source: str, data: bytes | None = None) -> ParsedDocument:
        """Analyse a document and convert the DI response into a ParsedDocument."""
        started = time.perf_counter()
        if not self.configured:
            raise RuntimeError(
                "no Document Intelligence endpoint configured; set "
                "DOC_INTELLIGENCE_ENDPOINT or pass endpoint="
            )

        payload = data if data is not None else Path(source).read_bytes()
        kind = _classify(source, payload)

        base = f"{self.endpoint}/documentintelligence/documentModels/{self.model}:analyze"
        url = f"{base}?api-version={DI_API_VERSION}&outputContentFormat=markdown"
        status, body, headers = await self._fetch("POST", url, self._headers(), payload)
        if status not in (200, 202):
            raise RuntimeError(
                f"Document Intelligence analyze failed: HTTP {status} {str(body)[:400]}"
            )

        # The operation URL is in the response HEADERS on a 202 (the body is
        # empty). Check the body too, for gateways that echo it there.
        operation = (
            headers.get("operation-location")
            or headers.get("Operation-Location")
            or body.get("operationLocation")
            or body.get("operation-location")
        )
        inline = body.get("analyzeResult")

        if operation:
            result = await self._poll(operation)
        elif inline:
            # A gateway inlined the finished result (small documents / mocks).
            result = inline
        else:
            # Neither. Returning an empty document here would present a failed
            # analyze as a valid parse with no fields — which downstream reads
            # as "the schema matched nothing" rather than "the service never
            # processed this". Fail loudly instead.
            raise RuntimeError(
                "Document Intelligence returned no operation-location and no inline "
                f"analyzeResult (HTTP {status}; headers: "
                f"{sorted(k for k in headers if 'peration' in k or 'ocation' in k)})"
            )

        doc = self._to_parsed(result, source, kind, started)
        return doc

    async def _poll(self, operation_url: str) -> dict:
        """Poll the long-running analyze operation to completion."""
        deadline = time.monotonic() + self.timeout_s
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Document Intelligence analyze exceeded {self.timeout_s}s"
                )
            status_code, body, _headers = await self._fetch(
                "GET", operation_url, self._headers()
            )
            state = str(body.get("status", "")).lower()
            if state == "succeeded":
                return body.get("analyzeResult") or body
            if state in ("failed", "canceled"):
                err = body.get("error") or {}
                raise RuntimeError(
                    f"Document Intelligence analyze {state}: "
                    f"{err.get('code')} {err.get('message')}"
                )
            await self._sleep(self.poll_interval)

    # ---- response -> model ------------------------------------------------ #
    @staticmethod
    def _to_parsed(
        result: dict, source: str, kind: SourceKind, started: float
    ) -> ParsedDocument:
        """Convert a DI ``analyzeResult`` into the component's document model.

        DI returns ``pages`` (with lines, words, selectionMarks), ``tables``
        (with cell spans) and ``documents`` (typed fields, for prebuilt models).
        Lines and tables are lifted into the model; the raw ``documents`` array
        is preserved in ``metadata`` so the extractor can use it without another
        service call.
        """
        pages: list[Page] = []
        for p in result.get("pages", []) or []:
            number = int(p.get("pageNumber", 1))
            lines: list[TextSpan] = []
            for line in p.get("lines", []) or []:
                poly = line.get("polygon") or []
                bbox = (
                    BoundingBox(polygon=[float(v) for v in poly], page=number)
                    if len(poly) == 8 else None
                )
                conf = None
                words = line.get("words") or []
                ws = [w.get("confidence") for w in words if w.get("confidence") is not None]
                if ws:
                    conf = sum(float(c) for c in ws) / len(ws)
                lines.append(
                    TextSpan(text=str(line.get("content", "")), page=number, bbox=bbox, confidence=conf)
                )

            marks = [
                str(m.get("state", "")) for m in (p.get("selectionMarks") or [])
            ]

            pages.append(
                Page(
                    number=number,
                    width=p.get("width"),
                    height=p.get("height"),
                    lines=lines,
                    selection_marks=marks,
                )
            )

        # Tables are document-level in DI, with pageNumber on each cell's
        # boundingRegion; group them back onto their page.
        by_page: dict[int, list[ParsedTable]] = {}
        for t in result.get("tables", []) or []:
            cells: list[TableCell] = []
            table_page = 1
            for c in t.get("cells", []) or []:
                regions = c.get("boundingRegions") or []
                if regions and regions[0].get("pageNumber"):
                    table_page = int(regions[0]["pageNumber"])
                cells.append(
                    TableCell(
                        text=str(c.get("content", "")),
                        row=int(c.get("rowIndex", 0)),
                        column=int(c.get("columnIndex", 0)),
                        row_span=int(c.get("rowSpan", 1) or 1),
                        column_span=int(c.get("columnSpan", 1) or 1),
                        confidence=c.get("confidence"),
                    )
                )
            table = ParsedTable(
                page=table_page,
                cells=cells,
                caption=str(t.get("caption", {}).get("content", "")) or None
                if isinstance(t.get("caption"), dict) else None,
            )
            by_page.setdefault(table_page, []).append(table)

        for page in pages:
            page.tables = by_page.get(page.number, [])

        doc = ParsedDocument(
            source=source,
            kind=kind,
            engine="azure-document-intelligence",
            pages=pages,
            language=(result.get("languages") or [{}])[0].get("locale") if result.get("languages") else None,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        # Keep the typed fields for the extractor; they are the reason to use a
        # prebuilt model at all, and re-fetching them would double the cost.
        if result.get("documents"):
            doc.metadata["di_documents"] = result["documents"]
        if result.get("content"):
            doc.metadata["markdown"] = result["content"]
        if result.get("keyValuePairs"):
            doc.metadata["di_key_value_pairs"] = result["keyValuePairs"]
        return doc


# --------------------------------------------------------------------------- #
# Local text layer
# --------------------------------------------------------------------------- #
class TextLayerEngine:
    """Read a PDF's embedded text layer, or a plain text file. No OCR.

    Exact for born-digital documents and *silently useless* for scans — so it
    records a warning when a page yields almost no text, which is the signature
    of a scanned page. Without that warning a scanned PDF would parse to an
    empty document and look like a valid extraction with no fields.
    """

    name = "text-layer"

    def __init__(self, *, min_chars_per_page: int = 20) -> None:
        self.min_chars_per_page = min_chars_per_page

    async def parse(self, source: str, data: bytes | None = None) -> ParsedDocument:
        started = time.perf_counter()
        payload = data if data is not None else Path(source).read_bytes()
        kind = _classify(source, payload)

        pages: list[Page] = []
        warnings: list[str] = []

        if kind is SourceKind.PDF:
            try:
                import io

                from pypdf import PdfReader
            except ImportError as e:
                raise RuntimeError(
                    "pypdf is required for PDF text extraction: "
                    "pip install 'doc-processing[local]'"
                ) from e
            reader = PdfReader(io.BytesIO(payload))
            for i, p in enumerate(reader.pages, start=1):
                text = (p.extract_text() or "").strip()
                lines = [TextSpan(text=ln, page=i) for ln in text.splitlines() if ln.strip()]
                pages.append(Page(number=i, lines=lines))
                if len(text) < self.min_chars_per_page:
                    warnings.append(
                        f"page {i} yielded {len(text)} characters — likely a scanned "
                        "page with no text layer; OCR is required for it"
                    )
        else:
            text = payload.decode("utf-8", errors="replace")
            lines = [TextSpan(text=ln, page=1) for ln in text.splitlines() if ln.strip()]
            pages.append(Page(number=1, lines=lines))

        return ParsedDocument(
            source=source,
            kind=kind,
            engine=self.name,
            pages=pages,
            warnings=warnings,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )


# --------------------------------------------------------------------------- #
# Deterministic stub
# --------------------------------------------------------------------------- #
class StubEngine:
    """Dependency-free, deterministic ingestion for tests and CI.

    Builds pages, lines and a table from either supplied ``text`` or the raw
    bytes. It is explicitly *not* OCR: it exists so the extraction and
    validation stages can be tested against known input without a network call
    or a local OCR stack.
    """

    name = "stub"

    def __init__(self, text: str | None = None, *, lines_per_page: int = 40) -> None:
        self.text = text
        self.lines_per_page = max(1, lines_per_page)

    async def parse(self, source: str, data: bytes | None = None) -> ParsedDocument:
        started = time.perf_counter()
        payload = data if data is not None else (
            Path(source).read_bytes() if Path(source).is_file() else b""
        )
        raw = self.text if self.text is not None else payload.decode("utf-8", errors="replace")
        kind = _classify(source, payload)

        all_lines = [ln for ln in raw.splitlines()]
        pages: list[Page] = []
        for start in range(0, max(1, len(all_lines)), self.lines_per_page):
            chunk = all_lines[start : start + self.lines_per_page]
            number = len(pages) + 1
            pages.append(
                Page(
                    number=number,
                    width=612.0,   # US Letter in points, so bboxes have a frame
                    height=792.0,
                    lines=[
                        TextSpan(
                            text=ln,
                            page=number,
                            # Synthesised boxes: enough for provenance tests
                            # without pretending to be real geometry.
                            bbox=BoundingBox(
                                polygon=[0.0, float(i * 12), 612.0, float(i * 12),
                                         612.0, float(i * 12 + 10), 0.0, float(i * 12 + 10)],
                                page=number,
                            ),
                            confidence=1.0,
                        )
                        for i, ln in enumerate(chunk) if ln.strip()
                    ],
                )
            )
        pages = [p for p in pages if p.lines] or [Page(number=1)]

        doc = ParsedDocument(
            source=source,
            kind=kind,
            engine=self.name,
            pages=pages,
            duration_ms=int((time.perf_counter() - started) * 1000),
        )
        doc.metadata["stub_text"] = raw
        return doc


# --------------------------------------------------------------------------- #
# Engine selection
# --------------------------------------------------------------------------- #
def select_engine(
    *,
    endpoint: str | None = None,
    prefer: str = "auto",
    model: str = _LAYOUT_MODEL,
    stub_text: str | None = None,
) -> Any:
    """Pick the best available engine, and never fail silently.

    ``auto`` prefers Azure (best OCR and layout), then the local text layer,
    then the stub — and logs which one it chose and why, because "my pipeline
    returned no fields" is almost always an engine-selection question.
    """
    if prefer == "stub":
        return StubEngine(text=stub_text)
    if prefer == "text-layer":
        return TextLayerEngine()
    if prefer == "azure":
        eng = AzureDocumentIntelligenceEngine(endpoint, model=model)
        if not eng.configured:
            raise RuntimeError("prefer='azure' but no endpoint is configured")
        return eng

    azure = AzureDocumentIntelligenceEngine(endpoint, model=model)
    if azure.configured:
        logger.info("doc-processing: using Azure Document Intelligence at %s", azure.endpoint)
        return azure
    logger.info(
        "doc-processing: no Document Intelligence endpoint; using the local text layer "
        "(no OCR — scanned documents will yield no text)"
    )
    return TextLayerEngine()
