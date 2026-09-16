"""Surya OCR engine (surya-ocr 0.17, in-process PyTorch).

Pages arrive as BGR numpy arrays and results leave as plain dicts, the shape
the API and the frontend consume.

Surya downloads its model weights once, into its own cache directory; after
that, inference is entirely local. Documents never leave this process, never
touch the filesystem, and nothing here logs their content -- the logger
records shapes, timings and exception *types* only.

The models take ~15 s to load onto the GPU, so ``warm_up`` is started in a
background thread at startup and ``/health`` reports ``state: loading`` until
it finishes. A request that arrives earlier waits for the load rather than
failing.
"""

from __future__ import annotations

import html
import logging
import os
import re
import threading
import time
from importlib import metadata
from typing import Any

import cv2
import numpy as np

from . import ocr

# Surya reads its settings from the environment when it is first imported.
# Progress bars would only clutter the service log.
os.environ.setdefault("DISABLE_TQDM", "true")

logger = logging.getLogger(__name__)

NAME = "surya"

# Pages above this are downscaled first, bounding VRAM and time. A 300 DPI
# A4 page is 8.7 MP, so ordinary scans are never touched.
MAX_PIXELS = int(os.getenv("OCR_SURYA_MAX_PIXELS", str(20_000_000)))

# Straighten pages tilted by at least this much before OCR. Surya itself
# reads tilted lines fine, but a 1-2 degree tilt drops a table row's right
# end (unit, range) a full line lower than its name, and rows fall apart.
# Uses the projection-profile estimator in ``ocr.py`` (~40 ms a page).
DESKEW_MIN_DEG = float(os.getenv("OCR_SURYA_DESKEW_MIN_DEG", "0.3"))

# Inline formatting the model emits (<b>, <i>, <sup>, <br>, <math>, ...).
_TAG = re.compile(r"</?[a-zA-Z][^>]*>")

_load_lock = threading.Lock()
# One page on the GPU at a time: concurrent calls would just contend for the
# same device and multiply peak memory.
_infer_lock = threading.Lock()

_models: tuple[Any, Any] | None = None  # (recognition, detection) predictors
_state = "idle"  # idle -> loading -> ready | error
_error: str | None = None
_device: str | None = None
_device_name: str | None = None


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------


def _version() -> str | None:
    try:
        return metadata.version("surya-ocr")
    except metadata.PackageNotFoundError:
        return None


def _load() -> tuple[Any, Any]:
    """Load the predictors once; later callers get the cached pair.

    Callers that arrive while another thread is loading block on the lock
    and then return the result, so the first request after startup waits
    for the warm-up instead of loading a second copy.
    """
    global _models, _state, _error, _device, _device_name
    with _load_lock:
        if _models is not None:
            return _models
        if _state == "error":
            raise ocr.EngineUnavailable(f"Surya failed to load ({_error}); see the service log")

        _state = "loading"
        started = time.perf_counter()
        try:
            from surya.detection import DetectionPredictor
            from surya.foundation import FoundationPredictor
            from surya.recognition import RecognitionPredictor
            from surya.settings import settings

            recognition = RecognitionPredictor(FoundationPredictor())
            detection = DetectionPredictor()
            _device = str(settings.TORCH_DEVICE_MODEL)
            if _device.startswith("cuda"):
                import torch

                _device_name = torch.cuda.get_device_name(0)
        except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
            _state, _error = "error", type(exc).__name__
            logger.exception("Surya failed to load")
            raise ocr.EngineUnavailable(f"Surya failed to load ({_error}); see the service log") from exc

        _models = (recognition, detection)
        _state = "ready"
        logger.info(
            "Surya %s ready on %s in %.1fs", _version(), _device_name or _device, time.perf_counter() - started
        )
        return _models


def warm_up() -> None:
    """Load the models and run one tiny page so CUDA kernels are initialised.

    Meant for a background thread at startup. Never raises: a failure is
    recorded for ``/health`` and re-raised to requests as EngineUnavailable.
    """
    try:
        _load()
        page = np.full((96, 480, 3), 255, dtype=np.uint8)
        cv2.putText(page, "warm up", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3, cv2.LINE_AA)
        ocr_page(page)
    except Exception:  # noqa: BLE001
        pass  # already logged by _load, or harmless if only the dry run failed


def engine_info() -> dict[str, Any]:
    """Cheap status snapshot for ``/health``; never triggers a load."""
    version = _version()
    return {
        "name": NAME,
        "available": version is not None and _state != "error",
        "state": _state if version is not None else "error",
        "version": version,
        "device": _device,
        "device_name": _device_name,
        # Surya's recognition model reads every script it knows at once;
        # there is no language parameter to pass.
        "uses_language_hints": False,
        "languages": [],
        "error": _error if version is not None else "surya-ocr is not installed",
    }


def validate_lang(lang: str) -> str:
    """Accept any well-formed spec: Surya does not use it, but the API echoes it."""
    return ocr.normalize_lang(lang)


# --------------------------------------------------------------------------
# Result mapping (pure functions, tested without a GPU)
# --------------------------------------------------------------------------


def clean_text(text: str) -> str:
    """Drop the model's inline formatting tags and decode HTML entities."""
    return re.sub(r"\s+", " ", html.unescape(_TAG.sub("", text or ""))).strip()


def words_from_chars(chars: list[Any]) -> list[dict[str, Any]]:
    """Split a line's characters into words with boxes and confidences.

    Surya's own ``words_from_chars`` gives each word the confidence of its
    *first* character; here it is the mean over all of them, rescaled to
    0-100. Characters without a valid box are special or formatting tokens
    and are skipped.
    """
    words: list[dict[str, Any]] = []
    text: list[str] = []
    confs: list[float] = []
    box: list[float] | None = None

    def flush() -> None:
        nonlocal text, confs, box
        word = clean_text("".join(text))
        if word and box is not None:
            left, top = int(round(box[0])), int(round(box[1]))
            words.append(
                {
                    "text": word,
                    "conf": round(100.0 * sum(confs) / len(confs), 2),
                    "bbox": [left, top, max(1, int(round(box[2])) - left), max(1, int(round(box[3])) - top)],
                }
            )
        text, confs, box = [], [], None

    for char in chars:
        if not getattr(char, "bbox_valid", True):
            continue
        piece = char.text or ""
        if not piece.strip():
            flush()
            continue
        if piece[0].isspace():
            flush()
        x0, y0, x1, y1 = char.bbox
        box = [x0, y0, x1, y1] if box is None else [min(box[0], x0), min(box[1], y0), max(box[2], x1), max(box[3], y1)]
        text.append(piece.strip())
        confs.append(float(char.confidence or 0.0))
        if piece[-1].isspace():
            flush()
    flush()
    return words


def build_page(text_lines: list[Any]) -> tuple[str, list[dict[str, Any]]]:
    """Turn Surya ``TextLine`` objects into the page text and numbered words."""
    lines = []
    for line in text_lines:
        text = clean_text(line.text)
        if text:
            lines.append({"text": text, "bbox": list(line.bbox), "words": words_from_chars(line.chars)})

    # Surya detects a form's label and its value as separate lines; reading
    # them as one row keeps "Patient ID" next to ": SMP-000-0001".
    rows = ocr.group_rows(lines)
    words: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=1):
        position = 0
        for line in row:
            for word in line["words"]:
                position += 1
                words.append({**word, "block": 1, "par": 1, "line": row_number, "word": position})
    text = "\n".join(" ".join(line["text"] for line in row) for row in rows)
    return text, words


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


def _prepare(image: np.ndarray) -> tuple[np.ndarray, float, float]:
    """BGR/gray/BGRA -> RGB, downscaled past ``MAX_PIXELS``, deskewed.

    Returns the page, the scale applied and the deskew rotation in degrees.
    """
    if image.ndim == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    else:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    height, width = rgb.shape[:2]
    scale = 1.0
    if height * width > MAX_PIXELS:
        scale = (MAX_PIXELS / (height * width)) ** 0.5
        rgb = cv2.resize(rgb, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    angle = ocr.estimate_skew(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
    if abs(angle) < DESKEW_MIN_DEG:
        return rgb, scale, 0.0
    # Pad with white so the rotated corners do not read as ink.
    return ocr._rotate(rgb, angle, (255, 255, 255)), scale, angle


def ocr_page(image: np.ndarray, *, lang: str = ocr.DEFAULT_LANG, detail: bool = False) -> dict[str, Any]:
    """OCR a single page image. Blocking -- callers run it off the event loop.

    ``lang`` is syntax-checked and echoed back on the response; Surya reads
    every script it knows without a hint, so it is otherwise ignored.
    """
    from PIL import Image

    recognition, detection = _load()

    started = time.perf_counter()
    rgb, scale, skew = _prepare(image)
    height, width = rgb.shape[:2]
    preprocess_ms = round((time.perf_counter() - started) * 1000, 1)

    started = time.perf_counter()
    try:
        with _infer_lock:
            [result] = recognition([Image.fromarray(rgb)], det_predictor=detection, math_mode=False)
    except Exception as exc:  # noqa: BLE001 - e.g. torch.cuda.OutOfMemoryError
        raise ocr.OcrError(f"Surya failed ({type(exc).__name__})") from exc

    text, words = build_page(result.text_lines)
    confidences = [word["conf"] for word in words]
    page: dict[str, Any] = {
        "text": text,
        "word_count": len(words),
        "mean_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "languages": ocr.detect_languages(text),
        "ocr_ms": round((time.perf_counter() - started) * 1000, 1),
        "width": width,
        "height": height,
        "scale": round(scale, 4),
        "skew_deg": skew,
        "preprocess_ms": preprocess_ms,
    }
    if detail:
        page["words"] = words
    logger.debug("surya page %dx%d words=%d conf=%s", width, height, len(words), page["mean_confidence"])
    return page
