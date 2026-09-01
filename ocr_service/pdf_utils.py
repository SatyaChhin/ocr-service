"""PDF page rasterisation, in memory wherever poppler allows it.

Two paths exist, in preference order:

1. **stdin** -- ``pdfinfo -`` / ``pdftoppm -`` are fed the PDF on standard
   input and hand back PNG bytes on standard output.  The document never
   reaches the filesystem, which is what the privacy requirement asks for.
2. **pdf2image fallback** -- used only if the stdin path does not work on the
   installed poppler build.  ``pdf2image.convert_from_bytes`` writes the
   upload to a ``tempfile.mkstemp()`` file before calling poppler, so on this
   path the PDF *does* briefly touch disk.  The module logs a warning once and
   ``TMPDIR``/``OCR_PDF_TEMP_DIR`` can point that at a ramdisk.

Pages are rendered one at a time on purpose: a 50-page colour PDF at 300 DPI
is well over a gigabyte if you materialise every page up front.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
from functools import lru_cache
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_DPI = int(os.getenv("OCR_PDF_DPI", "300"))
MIN_DPI = 72
MAX_DPI = 600
MAX_PAGES = int(os.getenv("OCR_MAX_PDF_PAGES", "50"))

# Per-page render timeout.  A malformed PDF can otherwise pin a worker.
RENDER_TIMEOUT_S = int(os.getenv("OCR_PDF_TIMEOUT_S", "120"))

_PAGES_RE = re.compile(rb"^Pages:\s+(\d+)", re.MULTILINE)


class PdfError(Exception):
    """The PDF could not be read or rendered."""


class PopplerUnavailable(PdfError):
    """Neither the poppler binaries nor pdf2image are usable."""


class TooManyPages(PdfError):
    """The PDF exceeds the configured page cap."""


def clamp_dpi(dpi: int | None) -> int:
    if dpi is None:
        return DEFAULT_DPI
    return max(MIN_DPI, min(MAX_DPI, int(dpi)))


# --------------------------------------------------------------------------
# Capability probing
# --------------------------------------------------------------------------


# Windows installs rarely put poppler on PATH; let operators point at it
# without editing the system environment. Prepending to PATH covers both our
# own subprocess calls and pdf2image's.
_POPPLER_PATH = os.getenv("OCR_POPPLER_PATH")
if _POPPLER_PATH:
    os.environ["PATH"] = _POPPLER_PATH + os.pathsep + os.environ.get("PATH", "")


@lru_cache(maxsize=1)
def _poppler_binaries() -> bool:
    return bool(shutil.which("pdfinfo") and shutil.which("pdftoppm"))


@lru_cache(maxsize=1)
def _pdf2image_available() -> bool:
    try:
        import pdf2image  # noqa: F401
    except Exception:  # noqa: BLE001
        return False
    return True


def backend_info() -> dict[str, object]:
    """Describe which rasterisation path will be used, for /health.

    Note that pdf2image is *not* an alternative to poppler -- it shells out to
    the same binaries. Without them, no PDF can be rendered at all, and the
    probe has to say so rather than let the failure surface per request.
    """
    binaries = _poppler_binaries()
    return {
        "available": binaries,
        "backend": ("pdf2image" if _fallback_active else "poppler-stdin") if binaries else None,
        "poppler_binaries": binaries,
        "pdf2image": _pdf2image_available(),
        "writes_temp_file": _fallback_active,
        "max_pages": MAX_PAGES,
        "default_dpi": DEFAULT_DPI,
    }


def _run(args: list[str], data: bytes, *, what: str) -> bytes:
    try:
        completed = subprocess.run(
            args,
            input=data,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=RENDER_TIMEOUT_S,
            check=False,
        )
    except FileNotFoundError as exc:
        raise PopplerUnavailable("poppler-utils is not installed") from exc
    except subprocess.TimeoutExpired as exc:
        raise PdfError(what + " timed out") from exc

    if completed.returncode != 0 or not completed.stdout:
        # poppler's stderr describes the *document structure* problem
        # (encryption, damage), not its contents, so it is safe to surface.
        detail = completed.stderr.decode("utf-8", "replace").strip().splitlines()
        message = detail[-1] if detail else "exit code " + str(completed.returncode)
        raise PdfError(what + " failed: " + message)
    return completed.stdout


# --------------------------------------------------------------------------
# Page count
# --------------------------------------------------------------------------


def _require_poppler() -> None:
    """Both code paths shell out to poppler; without it neither can work."""
    if not _poppler_binaries():
        raise PopplerUnavailable(
            "poppler-utils is not installed (pdfinfo/pdftoppm not found on PATH). "
            "Set OCR_POPPLER_PATH if it is installed elsewhere."
        )


def page_count(data: bytes) -> int:
    """Number of pages in the PDF, without rendering anything."""
    _require_poppler()

    try:
        output = _run(["pdfinfo", "-"], data, what="pdfinfo")
        match = _PAGES_RE.search(output)
        if match:
            return int(match.group(1))
        raise PdfError("could not determine page count")
    except PdfError:
        # An older poppler build may not accept '-' for stdin. Retry through
        # pdf2image where it is installed; otherwise the error stands.
        if not _pdf2image_available():
            raise
        logger.warning("pdfinfo stdin path unusable; falling back to pdf2image")

    from pdf2image import pdfinfo_from_bytes
    from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError

    _warn_temp_file()
    try:
        return int(pdfinfo_from_bytes(data)["Pages"])
    except (PDFPageCountError, PDFSyntaxError, KeyError, ValueError) as exc:
        raise PdfError("could not determine page count: " + type(exc).__name__) from exc


_fallback_active = False


def _warn_temp_file() -> None:
    """Record that the temp-file path is in use, and say so once."""
    global _fallback_active
    if not _fallback_active:
        _fallback_active = True
        logger.warning(
            "Falling back to pdf2image: uploaded PDFs are written to a short-lived "
            "temp file. Point OCR_PDF_TEMP_DIR at a ramdisk, or install poppler-utils "
            "so the stdin path can be used instead."
        )


_TEMP_DIR = os.getenv("OCR_PDF_TEMP_DIR")
if _TEMP_DIR:
    # pdf2image's mkstemp honours TMPDIR/TEMP; keep both in sync.
    os.environ.setdefault("TMPDIR", _TEMP_DIR)
    os.environ.setdefault("TEMP", _TEMP_DIR)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render_stdin(data: bytes, page: int, dpi: int) -> np.ndarray:
    """Render one page via ``pdftoppm`` reading stdin and writing stdout."""
    png = _run(
        [
            "pdftoppm",
            "-png",
            "-r",
            str(dpi),
            "-f",
            str(page),
            "-l",
            str(page),
            "-singlefile",
            "-",
        ],
        data,
        what="pdftoppm (page " + str(page) + ")",
    )
    image = cv2.imdecode(np.frombuffer(png, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise PdfError("page " + str(page) + " could not be rasterised")
    return image


def _render_pdf2image(data: bytes, page: int, dpi: int) -> np.ndarray:
    from pdf2image import convert_from_bytes
    from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError

    _warn_temp_file()
    try:
        pages = convert_from_bytes(data, dpi=dpi, first_page=page, last_page=page, fmt="ppm")
    except (PDFPageCountError, PDFSyntaxError) as exc:
        raise PdfError("page " + str(page) + " could not be rasterised") from exc
    except Exception as exc:  # noqa: BLE001 - poppler surfaces many error types
        raise PdfError("page " + str(page) + " could not be rasterised") from exc
    if not pages:
        raise PdfError("page " + str(page) + " could not be rasterised")
    # PIL RGB -> OpenCV BGR
    return cv2.cvtColor(np.asarray(pages[0].convert("RGB")), cv2.COLOR_RGB2BGR)


def iter_pages(data: bytes, *, dpi: int = DEFAULT_DPI, max_pages: int = MAX_PAGES) -> Iterator[tuple[int, np.ndarray]]:
    """Yield ``(page_number, bgr_image)`` one page at a time.

    Lazy by design so peak memory stays at roughly one rendered page.
    """
    total = page_count(data)
    if total < 1:
        raise PdfError("PDF contains no pages")
    if total > max_pages:
        raise TooManyPages(
            "PDF has " + str(total) + " pages; the limit is " + str(max_pages)
        )

    # `_fallback_active` means page_count already found the stdin path unusable
    # on this poppler build; no point retrying it per page.
    use_stdin = not _fallback_active
    for page in range(1, total + 1):
        if use_stdin:
            try:
                yield page, _render_stdin(data, page, dpi)
                continue
            except PdfError:
                # Only the first page tells us the stdin path is unsupported.
                # A failure later in the document is a real document problem.
                if page > 1 or not _pdf2image_available():
                    raise
                logger.warning("pdftoppm stdin path unusable; switching to pdf2image")
                use_stdin = False
        yield page, _render_pdf2image(data, page, dpi)
