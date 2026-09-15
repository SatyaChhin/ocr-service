"""FastAPI application exposing the OCR service.

Privacy posture, since these documents may contain personal health data:

* Uploads and extracted text live in memory only -- see ``_read_capped`` and
  the ``spool_max_size`` override below, which stops Starlette rolling large
  multipart uploads onto disk.
* Nothing logs document content.  The access log records method, path,
  status, duration and byte count; error handlers log exception *types*.
  Full tracebacks go to the log only when ``OCR_DEBUG=1``, and never to the
  client.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Iterable

from fastapi import FastAPI, File, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from . import db, engines, lab_results, ocr, pdf_utils

# --------------------------------------------------------------------------
# Limits and configuration
# --------------------------------------------------------------------------

MAX_UPLOAD_BYTES = int(os.getenv("OCR_MAX_UPLOAD_BYTES", str(25 * 1024 * 1024)))
READ_CHUNK_BYTES = 1024 * 1024
DEBUG_TRACEBACKS = os.getenv("OCR_DEBUG", "").lower() in {"1", "true", "yes"}

# Bound the number of documents being OCR'd at once.  Tesseract saturates a
# core per page, so an unbounded threadpool just thrashes.
MAX_CONCURRENCY = int(os.getenv("OCR_MAX_CONCURRENCY", str(min(4, (os.cpu_count() or 2)))))

PDF_MAGIC = b"%PDF-"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

SUPPORTED = "PDF, PNG, JPEG, WEBP"

# Keep every part of a 25 MB upload in RAM.  Starlette otherwise spools
# multipart parts over 1 MB to a SpooledTemporaryFile backed by disk.
try:
    from starlette.formparsers import MultiPartParser

    MultiPartParser.spool_max_size = MAX_UPLOAD_BYTES + READ_CHUNK_BYTES
except Exception:  # pragma: no cover - attribute may move between versions
    logging.getLogger(__name__).warning(
        "Could not raise Starlette's multipart spool threshold; large uploads "
        "may be buffered on disk."
    )

logging.basicConfig(
    level=os.getenv("OCR_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("ocr_service")


# --------------------------------------------------------------------------
# Response models
# --------------------------------------------------------------------------


class LanguageGuess(BaseModel):
    lang: str = Field(description="Tesseract-style code, e.g. 'eng'")
    iso639_1: str | None = None
    confidence: float = Field(description="0-1, from langdetect. Best effort.")


class WordBox(BaseModel):
    text: str
    conf: float = Field(description="Tesseract confidence, 0-100")
    bbox: list[int] = Field(description="[left, top, width, height] in preprocessed-image space")
    block: int
    par: int
    line: int
    word: int


class PageResult(BaseModel):
    page: int
    text: str
    word_count: int
    mean_confidence: float | None
    languages: list[LanguageGuess]
    width: int
    height: int
    scale: float = Field(description="Resize factor applied before OCR")
    skew_deg: float = Field(description="Deskew rotation applied, in degrees")
    preprocess_ms: float
    ocr_ms: float
    words: list[WordBox] | None = None


class LabNote(BaseModel):
    code: str = Field(description=(
        "low_confidence | flag_not_set | flag_unexpected | flag_wrong_direction | differential_mismatch | "
        "differential_sum | index_mismatch | unit_missing | unit_differs | name_fuzzy"
    ))
    params: dict[str, Any]


class LabReview(BaseModel):
    page: int
    confidence: float | None = Field(description="OCR confidence of the value, 0-100")
    source: str = Field(description="The report row the result was read from")
    cross_checked: list[str] = Field(description="Consistency checks this value passed")
    notes: list[LabNote]
    needs_review: bool = Field(description="True when any note is present: check against the document")


class LabUnparsed(BaseModel):
    page: int
    text: str


class LabReportHeader(BaseModel):
    patient_code: str | None = Field(None, max_length=64, description="e.g. KCM-260910054323")
    sample_no: str | None = Field(None, max_length=64, description="e.g. 0007-10092026")
    collected_at: datetime | None = None
    received_at: datetime | None = None
    conflicts: list[str] = Field(default_factory=list, description="Fields whose value differed between pages")


class LabExtraction(BaseModel):
    report: LabReportHeader = Field(description="Links the results to a patient and sample")
    results: list[dict[str, Any]] = Field(description=(
        "Database-ready records: test_name, [percent], value, flag, unit, ref_range, section. "
        "`percent` is present only on differential rows."
    ))
    review: list[LabReview] = Field(description="review[i] describes results[i]")
    unparsed: list[LabUnparsed] = Field(description="Table rows with numbers that were not read as results")


class OcrResponse(BaseModel):
    filename: str | None
    media_type: str
    engine: str = Field(description="OCR engine that produced the result: 'surya' or 'tesseract'")
    lang: str
    page_count: int
    languages: list[LanguageGuess]
    text: str = Field(description="All pages joined, separated by a form feed")
    pages: list[PageResult]
    lab: LabExtraction | None = Field(None, description="Structured lab results, when requested with lab=true")
    duration_ms: float


class LabResultIn(BaseModel):
    """One record of ``lab.results``, as returned by /ocr?lab=true."""

    test_name: str = Field(min_length=1, max_length=128)
    percent: float | None = None
    value: int | float | str
    flag: str | None = Field(None, max_length=4)
    unit: str | None = Field(None, max_length=32)
    ref_range: str | None = Field(None, max_length=64)
    section: str | None = Field(None, max_length=128)


class SaveLabReport(BaseModel):
    """Body of POST /lab-reports: the ``lab`` object of an /ocr response, plus context."""

    filename: str | None = Field(None, max_length=255)
    engine: str = Field(max_length=16)
    report: LabReportHeader = Field(default_factory=LabReportHeader)
    results: list[LabResultIn] = Field(min_length=1, max_length=500)
    review: list[dict[str, Any]] = Field(default_factory=list, description="lab.review, stored per result")
    replace: bool = Field(False, description="Overwrite a report already saved for the same patient + sample")


class SavedLabReport(BaseModel):
    id: int
    results_saved: int
    needs_review: int
    replaced: int | None = Field(description="Id of the report this one replaced, if any")


class ErrorResponse(BaseModel):
    detail: str
    request_id: str | None = None


# --------------------------------------------------------------------------
# Application
# --------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    pdf = pdf_utils.backend_info()
    app.state.semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    if engines.name() == "surya":
        # Loading the models takes ~15 s; do it off the event loop so the
        # service answers /health (state: loading) in the meantime.
        logger.info("OCR engine: Surya (loading models in the background)")
        threading.Thread(target=engines.warm_up, name="surya-warm-up", daemon=True).start()
    else:
        engine = ocr.engine_info()
        if engine["available"]:
            logger.info("OCR engine: Tesseract %s (%d languages)", engine["version"], len(engine["languages"]))
            if engines.missing_languages():
                logger.warning("Missing expected language data: %s", ", ".join(engines.missing_languages()))
        else:
            logger.error("Tesseract is unavailable (%s); /ocr will return 503", engine["error"])

    if pdf["available"]:
        logger.info("PDF backend: %s", pdf["backend"])
    else:
        logger.warning("No PDF backend; PDF uploads will return 503")
    yield


app = FastAPI(
    title="OCR Service",
    version="1.0.0",
    description=(
        "Extracts text from scanned or photographed documents (PNG/JPEG/WEBP/PDF) "
        "using Surya OCR on the GPU, with Tesseract available as a fallback "
        "(OCR_ENGINE=tesseract). Documents are processed in memory and never logged."
    ),
    lifespan=lifespan,
)


@app.middleware("http")
async def access_log(request: Request, call_next):
    """Log request metadata only -- never headers, query values or bodies."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        logger.exception("%s %s %s unhandled", request_id, request.method, request.url.path)
        raise
    duration_ms = (time.perf_counter() - started) * 1000
    logger.info(
        "%s %s %s -> %d in %.1fms",
        request_id,
        request.method,
        request.url.path,  # deliberately not request.url, which carries the query string
        response.status_code,
        duration_ms,
    )
    response.headers["X-Request-ID"] = request_id
    return response


# --------------------------------------------------------------------------
# Error handling -- clients get a message, never a stack trace
# --------------------------------------------------------------------------


def _error(request: Request, status: int, detail: str) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=status,
        content={"detail": detail, "request_id": request_id},
        headers={"X-Request-ID": request_id} if request_id else None,
    )


def _log_failure(request: Request, exc: Exception, status: int) -> None:
    request_id = getattr(request.state, "request_id", None)
    if DEBUG_TRACEBACKS:
        logger.exception("%s failed with %d (%s)", request_id, status, type(exc).__name__)
    else:
        logger.warning("%s failed with %d (%s)", request_id, status, type(exc).__name__)


@app.exception_handler(ocr.EngineUnavailable)
async def _engine_unavailable(request: Request, exc: ocr.EngineUnavailable):
    _log_failure(request, exc, 503)
    return _error(request, 503, str(exc) or "OCR engine unavailable")


@app.exception_handler(pdf_utils.PopplerUnavailable)
async def _poppler_unavailable(request: Request, exc: pdf_utils.PopplerUnavailable):
    _log_failure(request, exc, 503)
    return _error(request, 503, str(exc) or "PDF rendering backend unavailable")


@app.exception_handler(ocr.OcrError)
async def _ocr_error(request: Request, exc: ocr.OcrError):
    status = 400 if isinstance(exc, (ocr.UnsupportedLanguage, ocr.ImageDecodeError)) else 500
    _log_failure(request, exc, status)
    detail = str(exc) if status == 400 else "OCR failed while processing the document"
    return _error(request, status, detail)


@app.exception_handler(pdf_utils.PdfError)
async def _pdf_error(request: Request, exc: pdf_utils.PdfError):
    _log_failure(request, exc, 400)
    return _error(request, 400, str(exc) or "The PDF could not be processed")


@app.exception_handler(db.DatabaseUnavailable)
async def _database_unavailable(request: Request, exc: db.DatabaseUnavailable):
    _log_failure(request, exc, 503)
    return _error(request, 503, "The results database is not reachable. Check MySQL and OCR_DB_* in .env")


@app.exception_handler(db.DuplicateReport)
async def _duplicate_report(request: Request, exc: db.DuplicateReport):
    request_id = getattr(request.state, "request_id", None)
    return JSONResponse(
        status_code=409,
        content={
            "detail": f"This report is already saved (report {exc.report_id}). Send replace=true to overwrite it.",
            "request_id": request_id,
            "report_id": exc.report_id,
        },
        headers={"X-Request-ID": request_id} if request_id else None,
    )


@app.exception_handler(RequestValidationError)
async def _validation_error(request: Request, exc: RequestValidationError):
    # FastAPI's default 422 body echoes the offending input back to the
    # client and into any log that captures responses. Report the location
    # and the reason, never the value.
    problems = [
        {"loc": [str(part) for part in error.get("loc", [])], "msg": error.get("msg", "invalid")}
        for error in exc.errors()
    ]
    request_id = getattr(request.state, "request_id", None)
    logger.info("%s rejected: %d validation problem(s)", request_id, len(problems))
    return JSONResponse(
        status_code=422,
        content={"detail": "Request validation failed", "problems": problems, "request_id": request_id},
    )


@app.exception_handler(StarletteHTTPException)
async def _http_error(request: Request, exc: StarletteHTTPException):
    return _error(request, exc.status_code, str(exc.detail))


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    request_id = getattr(request.state, "request_id", None)
    logger.exception("%s unhandled %s", request_id, type(exc).__name__)
    return _error(request, 500, "Internal error while processing the document")


# --------------------------------------------------------------------------
# Upload validation
# --------------------------------------------------------------------------


async def _read_capped(upload: UploadFile) -> bytes:
    """Read the upload into memory, aborting once it exceeds the size cap."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(READ_CHUNK_BYTES)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_UPLOAD_BYTES:
            raise StarletteHTTPException(
                status_code=413,
                detail="File exceeds the " + str(MAX_UPLOAD_BYTES // (1024 * 1024)) + " MB limit",
            )
        chunks.append(chunk)
    if total == 0:
        raise StarletteHTTPException(status_code=400, detail="Uploaded file is empty")
    return b"".join(chunks)


def sniff_media_type(data: bytes) -> str:
    """Identify the upload from its magic bytes.

    The client-supplied filename and Content-Type are advisory; only the
    bytes decide what we hand to poppler or OpenCV.
    """
    if data.startswith(PDF_MAGIC):
        return "application/pdf"
    if data.startswith(PNG_MAGIC):
        return "image/png"
    if data.startswith(JPEG_MAGIC):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise StarletteHTTPException(
        status_code=400,
        detail="Unsupported file type. Supported formats: " + SUPPORTED,
    )


# --------------------------------------------------------------------------
# Document processing (blocking -- always called via run_in_threadpool)
# --------------------------------------------------------------------------


def _process(data: bytes, media_type: str, lang: str, detail: bool, dpi: int, lab: bool) -> dict[str, Any]:
    pages: list[dict[str, Any]] = []

    if media_type == "application/pdf":
        sources: Iterable[tuple[int, Any]] = pdf_utils.iter_pages(data, dpi=dpi)
    else:
        sources = [(1, ocr.decode_image(data))]

    for number, image in sources:
        # Lab extraction rebuilds table rows from word boxes, so it needs them
        # even when the client did not ask for detail.
        page = engines.ocr_page(image, lang=lang, detail=detail or lab)
        page["page"] = number
        pages.append(page)
        del image  # release the rendered page before the next one is built

    if not pages:
        raise ocr.OcrError("No pages could be processed")

    extraction = lab_results.extract(pages) if lab else None
    if lab and not detail:
        for page in pages:
            page["words"] = None

    return {
        "pages": pages,
        # Form feed is the conventional page separator in plain-text output.
        "text": "\f".join(page["text"] for page in pages),
        "languages": ocr.merge_languages([page["languages"] for page in pages]),
        "lab": extraction,
    }


# --------------------------------------------------------------------------
# Routes
# --------------------------------------------------------------------------


@app.post(
    "/ocr",
    response_model=OcrResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported or unreadable file, or bad lang"},
        413: {"model": ErrorResponse, "description": "File exceeds the size limit"},
        503: {"model": ErrorResponse, "description": "OCR engine or poppler unavailable"},
    },
    summary="Extract text from an uploaded image or PDF",
)
async def extract_text(
    request: Request,
    file: UploadFile = File(..., description="PNG, JPEG, WEBP or PDF"),
    lang: str = Query(ocr.DEFAULT_LANG, description="Tesseract language spec, e.g. 'eng+fra'"),
    detail: bool = Query(False, description="Include per-word boxes and confidences"),
    dpi: int = Query(
        pdf_utils.DEFAULT_DPI,
        ge=pdf_utils.MIN_DPI,
        le=pdf_utils.MAX_DPI,
        description="Rasterisation DPI for PDF pages",
    ),
    lab: bool = Query(False, description="Also extract structured lab results (the `lab` field)"),
) -> OcrResponse:
    started = time.perf_counter()
    # Order matters: reject a malformed request on the cheap checks first, so a
    # bad upload gets its 400/413 rather than a 503 about a missing engine.
    ocr.normalize_lang(lang)
    data = await _read_capped(file)
    media_type = sniff_media_type(data)
    validated_lang = engines.validate_lang(lang)

    logger.info(
        "%s ocr request type=%s bytes=%d lang=%s detail=%s",
        getattr(request.state, "request_id", None),
        media_type,
        len(data),
        validated_lang,
        detail,
    )

    semaphore: asyncio.Semaphore = request.app.state.semaphore
    async with semaphore:
        result = await run_in_threadpool(
            _process, data, media_type, validated_lang, detail, pdf_utils.clamp_dpi(dpi), lab
        )

    return OcrResponse(
        filename=file.filename,
        media_type=media_type,
        engine=engines.name(),
        lang=validated_lang,
        page_count=len(result["pages"]),
        languages=result["languages"],
        text=result["text"],
        pages=result["pages"],
        lab=result["lab"],
        duration_ms=round((time.perf_counter() - started) * 1000, 1),
    )


# --------------------------------------------------------------------------
# Saved lab reports (MySQL)
# --------------------------------------------------------------------------


@app.post(
    "/lab-reports",
    status_code=201,
    response_model=SavedLabReport,
    responses={
        409: {"model": ErrorResponse, "description": "Already saved for this patient + sample (see report_id)"},
        503: {"model": ErrorResponse, "description": "Database not reachable"},
    },
    summary="Save lab results into the database",
)
async def save_lab_report(body: SaveLabReport) -> SavedLabReport:
    """Store the ``lab`` object of an ``/ocr?lab=true`` response.

    Send ``report``, ``results`` and ``review`` as returned, plus the upload's
    ``filename`` and the ``engine`` that read it. Results that need review are
    saved with ``needs_review = 1`` so they can be checked later.
    """
    report = body.report.model_dump()
    for key in ("collected_at", "received_at"):
        report[key] = report[key].isoformat() if report[key] else None
    saved = await run_in_threadpool(
        db.save_report,
        report=report,
        results=[r.model_dump(exclude_unset=True) for r in body.results],  # no percent: null on plain rows
        review=body.review,
        filename=body.filename,
        engine=body.engine,
        replace=body.replace,
    )
    return SavedLabReport(**saved)


@app.get("/lab-reports", summary="Recently saved lab reports")
async def list_lab_reports(limit: int = Query(20, ge=1, le=200)) -> list[dict[str, Any]]:
    return await run_in_threadpool(db.list_reports, limit)


@app.get("/lab-reports/{report_id}", summary="One saved report, with its results in the JSON format")
async def get_lab_report(report_id: int) -> dict[str, Any]:
    report = await run_in_threadpool(db.get_report, report_id)
    if report is None:
        raise StarletteHTTPException(status_code=404, detail=f"No lab report {report_id}")
    return report


@app.get("/health", summary="Liveness and dependency status")
async def health() -> dict[str, Any]:
    """Always 200 so a liveness probe does not restart a running process.

    Use the ``ready`` flag for readiness: it is false while Surya's models are
    still loading, when the active engine failed to load, and (for Tesseract)
    when expected language data is missing.
    """
    engine = ocr.engine_info()
    pdf = pdf_utils.backend_info()
    missing = [code for code in ocr.EXPECTED_LANGS if code not in engine["languages"]]
    ready = engines.ready()

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "engine": {**engines.info(), "missing_expected": engines.missing_languages()},
        # Only saving lab reports needs it, so it does not affect `ready`.
        "database": await run_in_threadpool(db.info),
        # The fallback engine's status, reported whichever engine is active.
        "tesseract": {
            "available": engine["available"],
            "version": engine["version"],
            "languages": engine["languages"],
            "missing_expected": missing,
        },
        "pdf": pdf,
        "limits": {
            "max_upload_bytes": MAX_UPLOAD_BYTES,
            "max_pdf_pages": pdf_utils.MAX_PAGES,
            "max_concurrency": MAX_CONCURRENCY,
            "supported_formats": SUPPORTED,
        },
    }
