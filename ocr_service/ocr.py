"""Image preprocessing and Tesseract OCR extraction.

Nothing here touches the filesystem: pages arrive as ``bytes`` or as numpy
arrays and results leave as plain dicts.  Nothing here logs document content
either -- the logger records shapes, angles, timings and error *types* only,
because these documents may carry personal health information.
"""

from __future__ import annotations

import logging
import os
import re
import time
from functools import lru_cache
from typing import Any, Iterable

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

DEFAULT_LANG = "eng+fra"
LANG_RE = re.compile(r"^[a-z]{3,4}(?:_[a-z]+)?(?:\+[a-z]{3,4}(?:_[a-z]+)?)*$", re.I)

# Language data the deployment is expected to ship.  /health turns `ready`
# false when one is absent, so a half-provisioned image is visible before a
# request fails on it rather than after.  Override for a slimmer install:
#   OCR_EXPECTED_LANGS=eng,khm
EXPECTED_LANGS = tuple(
    code.strip()
    for code in os.getenv("OCR_EXPECTED_LANGS", "eng,fra,khm").split(",")
    if code.strip()
)

# oem 3 = default LSTM engine, psm 3 = fully automatic page segmentation.
# psm 3 handles the multi-column / table-ish layouts on medical forms far
# better than the single-block modes.
TESSERACT_CONFIG = os.getenv("OCR_TESSERACT_CONFIG", "--oem 3 --psm 3")

# Tesseract is trained on ~300 DPI text.  Small photos get upscaled, and
# absurdly large scans get downscaled so one page cannot exhaust memory.
MIN_LONG_SIDE = int(os.getenv("OCR_MIN_LONG_SIDE", "1600"))
MAX_PIXELS = int(os.getenv("OCR_MAX_PIXELS", str(40_000_000)))

# "auto" picks fast median filtering for big scans and non-local-means for
# small ones; nlmeans on a full 300 DPI page costs seconds per page.
DENOISE_MODE = os.getenv("OCR_DENOISE", "auto").lower()
NLMEANS_PIXEL_LIMIT = int(os.getenv("OCR_NLMEANS_PIXEL_LIMIT", str(4_000_000)))

MAX_DESKEW_DEG = float(os.getenv("OCR_MAX_DESKEW_DEG", "12"))

# Binarisation strategy.  See _binarise() for why the default is not a plain
# cv2.adaptiveThreshold: "flatten-otsu" | "adaptive" | "otsu" | "off".
BINARISE_MODE = os.getenv("OCR_BINARISE", "flatten-otsu").lower()

# Erase ruled table borders before OCR; see _strip_rules for why.
STRIP_RULES = os.getenv("OCR_STRIP_RULES", "1").lower() not in {"0", "false", "no"}

# Long side of the downscaled copy the illumination field is estimated on.
BACKGROUND_ESTIMATE_PX = int(os.getenv("OCR_BACKGROUND_ESTIMATE_PX", "512"))

# Window for the "adaptive" mode.  Must be odd; 31px suits body text at 300 DPI.
ADAPTIVE_BLOCK_SIZE = int(os.getenv("OCR_ADAPTIVE_BLOCK", "31"))
ADAPTIVE_C = int(os.getenv("OCR_ADAPTIVE_C", "15"))

# Allow operators to point at a non-PATH binary (common on Windows).
_TESSERACT_CMD = os.getenv("TESSERACT_CMD")
if _TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class OcrError(Exception):
    """Base class for OCR failures that map to a client-visible response."""


class EngineUnavailable(OcrError):
    """The Tesseract binary is missing or unusable."""


class UnsupportedLanguage(OcrError):
    """The requested `lang` string is malformed or not installed."""


class ImageDecodeError(OcrError):
    """The bytes could not be decoded as a raster image."""


# --------------------------------------------------------------------------
# Engine introspection
# --------------------------------------------------------------------------


@lru_cache(maxsize=1)
def engine_info() -> dict[str, Any]:
    """Probe Tesseract once and cache the result.

    Never raises: a missing binary is reported as ``available: False`` so the
    health endpoint can describe the problem instead of crashing.
    """
    try:
        version = str(pytesseract.get_tesseract_version())
        languages = sorted(pytesseract.get_languages(config=""))
    except Exception as exc:  # noqa: BLE001 - any failure means "unusable"
        logger.warning("Tesseract unavailable: %s", type(exc).__name__)
        return {"available": False, "version": None, "languages": [], "error": type(exc).__name__}
    return {"available": True, "version": version, "languages": languages, "error": None}


def normalize_lang(lang: str) -> str:
    """Check the *syntax* of a `lang` spec such as ``eng+fra``.

    Deliberately separate from :func:`validate_lang`: this needs no engine, so
    a caller can reject a malformed request before deciding whether the
    service itself is healthy.
    """
    lang = (lang or "").strip()
    if not lang:
        return DEFAULT_LANG
    if not LANG_RE.match(lang):
        raise UnsupportedLanguage(
            "lang must be one or more '+'-joined Tesseract codes, e.g. 'eng', 'fra', 'eng+fra'"
        )
    return lang


def validate_lang(lang: str) -> str:
    """Check syntax *and* that the language data is actually installed."""
    lang = normalize_lang(lang)
    info = engine_info()
    if not info["available"]:
        raise EngineUnavailable("Tesseract is not installed or not on PATH")
    installed = set(info["languages"])
    missing = [part for part in lang.split("+") if part not in installed]
    if missing:
        raise UnsupportedLanguage(
            "language data not installed for: "
            + ", ".join(sorted(missing))
            + ". Installed: "
            + (", ".join(sorted(installed)) or "none")
        )
    return lang


# --------------------------------------------------------------------------
# Decoding
# --------------------------------------------------------------------------


def decode_image(data: bytes) -> np.ndarray:
    """Decode PNG/JPEG/WEBP bytes into a BGR numpy array."""
    buf = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise ImageDecodeError("File could not be decoded as an image")
    return image


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------


def _to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _rescale(gray: np.ndarray) -> tuple[np.ndarray, float]:
    """Bring the page into Tesseract's comfort zone, returning the factor used."""
    height, width = gray.shape[:2]
    long_side = max(height, width)
    scale = 1.0

    if long_side < MIN_LONG_SIDE:
        scale = min(4.0, MIN_LONG_SIDE / long_side)
    pixels = height * width * scale * scale
    if pixels > MAX_PIXELS:
        scale *= (MAX_PIXELS / pixels) ** 0.5

    if abs(scale - 1.0) < 0.01:
        return gray, 1.0
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    resized = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=interpolation)
    return resized, scale


def _denoise(gray: np.ndarray) -> np.ndarray:
    mode = DENOISE_MODE
    if mode == "off":
        return gray
    if mode == "auto":
        mode = "nlmeans" if gray.size <= NLMEANS_PIXEL_LIMIT else "fast"
    if mode == "nlmeans":
        # h=7 removes scanner grain and JPEG mush without eating thin strokes.
        return cv2.fastNlMeansDenoising(gray, None, h=7, templateWindowSize=7, searchWindowSize=21)
    return cv2.medianBlur(gray, 3)


def _flatten_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide out the lighting field estimated by a heavy Gaussian blur.

    The blur, at a radius far larger than any glyph, captures the page
    background -- shadow gradients, uneven scanner lamps, phone-camera
    vignetting -- while text averages out. Dividing by it leaves a page with a
    flat white background, which a single global threshold can then handle.

    Estimated on a downscaled copy and resized back. The field is
    low-frequency by construction, so this changes it very little, but it
    keeps the cost flat: blurring a 24 MP page directly needs a ~1100px
    kernel and took 24 seconds per page.
    """
    height, width = gray.shape[:2]
    scale = min(1.0, BACKGROUND_ESTIMATE_PX / max(height, width))
    if scale < 1.0:
        small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    else:
        small = gray
    # Radius stays the same fraction of the page either way.
    sigma = max(small.shape[:2]) / 30.0
    background = cv2.GaussianBlur(small, (0, 0), sigmaX=sigma)
    if scale < 1.0:
        background = cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)
    return cv2.divide(gray, background, scale=255)


def _binarise(gray: np.ndarray) -> np.ndarray:
    """Convert a grayscale page to black text on white.

    The default is *not* ``cv2.adaptiveThreshold``, despite that being the
    usual recommendation, because it measurably loses text on forms with
    ruled tables. Adaptive thresholding sets each pixel's threshold from its
    local mean, and next to a heavy black table rule that local mean is
    dragged down -- so extra pixels flip to black and the nearby characters
    thicken until they merge. On a bilingual medical form with a ruled
    medication table, that cost 14 of 51 ground-truth tokens: the entire
    table read as nothing.

    Flattening the illumination and then applying a global Otsu threshold
    keeps the lighting invariance that adaptive thresholding is chosen for
    without touching stroke weight. Measured token recall over the same page
    under three lighting conditions (see README):

        flatten + Otsu   150/153
        adaptive 31/15   113/153
        plain Otsu       124/153   (collapses under a hard shadow)

    ``OCR_BINARISE`` selects a different mode if a corpus disagrees.
    """
    if BINARISE_MODE == "off":
        return gray
    if BINARISE_MODE == "adaptive":
        return cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            ADAPTIVE_BLOCK_SIZE, ADAPTIVE_C,
        )
    if BINARISE_MODE == "otsu":
        return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return cv2.threshold(
        _flatten_illumination(gray), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )[1]


def _strip_rules(gray: np.ndarray) -> np.ndarray:
    """Erase long horizontal and vertical rules, keeping the glyphs.

    Tesseract's layout analysis can classify a fully-bordered table as a
    non-text region and skip every cell in it -- silently, with no drop in
    reported confidence, and in *every* page segmentation mode. Painting the
    rules white leaves the cell contents looking like ordinary text lines.

    Measured on a ruled medication table, token recall went 3/10 -> 10/10;
    on a form whose table was already read correctly it changed nothing
    (98 words either way). Set ``OCR_STRIP_RULES=0`` to disable.

    Must run *after* deskewing, since the directional kernels only match
    rules that are level.
    """
    height, width = gray.shape[:2]
    inverted = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # Size the kernels from the *text*, not the page. Deriving them from page
    # dimensions looks reasonable but breaks on a short page or large type:
    # a kernel shorter than a capital letter matches the letter's own stems
    # and erases the glyph. Six times the median glyph height is far longer
    # than any stroke and far shorter than a real rule.
    minimum_length = max(20, int(6 * _estimate_text_height(inverted)))
    lengths = (max(minimum_length, width // 30), max(minimum_length, height // 30))

    out = gray.copy()
    for axis, length in enumerate(lengths):
        if length >= (width if axis == 0 else height):
            continue  # no rule could be that long; nothing to find
        kernel_size = (length, 1) if axis == 0 else (1, length)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        rules = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel)
        rules = cv2.dilate(rules, np.ones((3, 3), np.uint8))
        out[rules > 0] = 255
    return out


def _estimate_text_height(inverted: np.ndarray) -> float:
    """Median height of the glyph-sized connected components on the page."""
    count, _, stats, _ = cv2.connectedComponentsWithStats(inverted, connectivity=8)
    if count < 2:
        return 20.0
    heights = stats[1:, cv2.CC_STAT_HEIGHT]
    # Drop specks and page-spanning components (rules, borders, margins).
    plausible = heights[(heights > 3) & (heights < inverted.shape[0] // 8)]
    return float(np.median(plausible)) if plausible.size else 20.0


def _projection_score(binary: np.ndarray) -> float:
    """Sharpness of the horizontal projection profile.

    When text lines are level, row sums alternate hard between ink rows and
    gaps, so the squared first difference peaks.  Scoring candidate rotations
    this way sidesteps the sign ambiguity of ``minAreaRect`` angles.
    """
    profile = binary.sum(axis=1, dtype=np.float64)
    return float(np.square(np.diff(profile)).sum())


def _rotate(image: np.ndarray, angle: float, border: int) -> np.ndarray:
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border,
    )


def estimate_skew(gray: np.ndarray) -> float:
    """Estimate page skew in degrees (positive angle = correcting rotation).

    Runs on a downscaled copy: skew is a global property and full resolution
    buys nothing but time.
    """
    long_side = max(gray.shape[:2])
    scale = 800.0 / long_side if long_side > 800 else 1.0
    small = (
        cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        if scale != 1.0
        else gray
    )
    binary = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    if cv2.countNonZero(binary) < 50:
        return 0.0

    def best_of(candidates: Iterable[float]) -> float:
        best_angle, best_score = 0.0, -1.0
        for candidate in candidates:
            angle = round(float(candidate), 2)
            rotated = binary if angle == 0.0 else _rotate(binary, angle, 0)
            score = _projection_score(rotated)
            if score > best_score:
                best_angle, best_score = angle, score
        return best_angle

    limit = MAX_DESKEW_DEG
    coarse = best_of(np.arange(-limit, limit + 0.5, 1.0))
    fine = best_of(np.arange(coarse - 0.9, coarse + 0.95, 0.2))
    return 0.0 if abs(fine) > limit else fine


def preprocess(image: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Grayscale -> rescale -> denoise -> deskew -> strip rules -> binarise.

    Returns the binarised page plus the geometry metadata a caller needs to
    map word boxes back onto the original upload.
    """
    started = time.perf_counter()
    gray = _to_gray(image)
    gray, scale = _rescale(gray)
    gray = _denoise(gray)

    angle = estimate_skew(gray)
    if abs(angle) >= 0.1:
        # Pad with white so rotation corners do not read as ink.
        gray = _rotate(gray, angle, 255)

    if STRIP_RULES:
        gray = _strip_rules(gray)

    binary = _binarise(gray)
    height, width = binary.shape[:2]
    meta = {
        "width": width,
        "height": height,
        "scale": round(scale, 4),
        "skew_deg": angle,
        "preprocess_ms": round((time.perf_counter() - started) * 1000, 1),
    }
    logger.debug("preprocessed page %dx%d scale=%.2f skew=%.2f", width, height, scale, angle)
    return binary, meta


# --------------------------------------------------------------------------
# OCR
# --------------------------------------------------------------------------


def _rows(data: dict[str, list[Any]]) -> Iterable[dict[str, Any]]:
    """Yield the real word rows from ``image_to_data`` output.

    Tesseract emits structural rows (page/block/paragraph/line) with an empty
    string and conf ``-1``; those carry no text and are dropped here.
    """
    for index, raw_text in enumerate(data.get("text", [])):
        text = (raw_text or "").strip()
        if not text:
            continue
        try:
            conf = float(data["conf"][index])
        except (TypeError, ValueError):
            continue
        if conf < 0:
            continue
        yield {
            "text": text,
            "conf": conf,
            "block": int(data["block_num"][index]),
            "par": int(data["par_num"][index]),
            "line": int(data["line_num"][index]),
            "word": int(data["word_num"][index]),
            "bbox": [
                int(data["left"][index]),
                int(data["top"][index]),
                int(data["width"][index]),
                int(data["height"][index]),
            ],
        }


def _reassemble(words: list[dict[str, Any]]) -> str:
    """Rebuild readable text, preserving line and paragraph breaks.

    Layout matters on forms: a label and its value sit on one line, and losing
    that boundary would glue unrelated fields together.
    """
    lines: list[str] = []
    previous_block: tuple[int, int] | None = None
    current_key: tuple[int, int, int] | None = None
    current: list[str] = []

    for word in sorted(words, key=lambda w: (w["block"], w["par"], w["line"], w["word"])):
        key = (word["block"], word["par"], word["line"])
        if key != current_key:
            if current:
                lines.append(" ".join(current))
            block = (word["block"], word["par"])
            if previous_block is not None and block != previous_block:
                lines.append("")  # blank line between paragraphs/blocks
            previous_block, current_key, current = block, key, []
        current.append(word["text"])

    if current:
        lines.append(" ".join(current))
    return "\n".join(lines).strip()


def ocr_page(
    image: np.ndarray,
    *,
    lang: str = DEFAULT_LANG,
    detail: bool = False,
    preprocess_page: bool = True,
) -> dict[str, Any]:
    """Preprocess and OCR a single page image.

    Blocking and CPU-bound -- callers must run this off the event loop.
    """
    if preprocess_page:
        prepared, meta = preprocess(image)
    else:
        prepared = _to_gray(image)
        height, width = prepared.shape[:2]
        meta = {
            "width": width,
            "height": height,
            "scale": 1.0,
            "skew_deg": 0.0,
            "preprocess_ms": 0.0,
        }

    started = time.perf_counter()
    try:
        data = pytesseract.image_to_data(
            prepared, lang=lang, config=TESSERACT_CONFIG, output_type=Output.DICT
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise EngineUnavailable("Tesseract is not installed or not on PATH") from exc
    except pytesseract.TesseractError as exc:
        # exc.message is Tesseract's stderr, which describes configuration
        # problems rather than document content.
        raise OcrError("Tesseract failed: " + str(getattr(exc, "message", "unknown error"))) from exc

    words = list(_rows(data))
    text = _reassemble(words)
    confidences = [word["conf"] for word in words]

    result: dict[str, Any] = {
        "text": text,
        "word_count": len(words),
        "mean_confidence": round(sum(confidences) / len(confidences), 2) if confidences else None,
        "languages": detect_languages(text),
        "ocr_ms": round((time.perf_counter() - started) * 1000, 1),
        **meta,
    }
    if detail:
        # Boxes are in preprocessed-image space; `scale` and `skew_deg` above
        # describe the transform applied to the original upload.
        result["words"] = words
    logger.debug("ocr page words=%d conf=%s", len(words), result["mean_confidence"])
    return result


# --------------------------------------------------------------------------
# Language detection (best effort)
# --------------------------------------------------------------------------

# langdetect speaks ISO 639-1; Tesseract speaks 639-2/T.
_ISO1_TO_ISO2 = {
    "en": "eng",
    "fr": "fra",
    "km": "khm",
    "de": "deu",
    "es": "spa",
    "it": "ita",
    "nl": "nld",
    "pt": "por",
}

MIN_CHARS_FOR_DETECTION = int(os.getenv("OCR_MIN_CHARS_FOR_LANGDETECT", "40"))

# langdetect ships 55 profiles and Khmer is not among them.  Worse, it raises
# "No features in text" on Khmer -- including on a *bilingual* page, because
# its n-gram features come out empty -- so relying on it alone would report no
# language at all for exactly the documents this service was extended to read.
#
# Khmer has its own Unicode block and shares it with no other language, so the
# script itself is the identification.  Counting characters is deterministic,
# needs no model, and works on the two-word fragment that langdetect's 40-char
# floor would reject.
_KHMER_BLOCK = (0x1780, 0x17FF)  # U+1780-17FF Khmer (U+19E0-19FF is symbols only)

# Below this share of the page's letters, Khmer is more likely to be a stray
# glyph hallucinated out of a stamp or a logo than real text.
MIN_KHMER_RATIO = float(os.getenv("OCR_MIN_KHMER_RATIO", "0.10"))


def _khmer_ratio(text: str) -> float:
    """Share of the alphabetic characters that sit in the Khmer block."""
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    khmer = sum(1 for char in letters if _KHMER_BLOCK[0] <= ord(char) <= _KHMER_BLOCK[1])
    return khmer / len(letters)


def _strip_khmer(text: str) -> str:
    """Drop Khmer characters so langdetect sees only what it can model."""
    return "".join(
        " " if _KHMER_BLOCK[0] <= ord(char) <= _KHMER_BLOCK[1] else char for char in text
    )

try:  # pragma: no cover - import-time capability probe
    from langdetect import DetectorFactory, detect_langs
    from langdetect.lang_detect_exception import LangDetectException

    DetectorFactory.seed = 0  # langdetect is non-deterministic without a seed
    _LANGDETECT = True
except Exception:  # noqa: BLE001
    _LANGDETECT = False
    LangDetectException = Exception  # type: ignore[assignment,misc]


def detect_languages(text: str, *, top: int = 3) -> list[dict[str, Any]]:
    """Best-effort language guess for a page of OCR output.

    Tesseract will not report which of ``eng+khm`` it actually matched, so this
    runs over the extracted text instead.  Two signals are combined:

    * **Khmer** by script, since it owns its Unicode block outright.
    * **Everything else** by langdetect, over the text with Khmer removed --
      otherwise a bilingual page yields no features and langdetect gives up on
      the Latin half too.

    Confidences are the share of the page each language accounts for, so a
    half-Khmer form reports roughly ``khm 0.5 / eng 0.5``.  Still best effort:
    unreliable on short or mixed form text, hence the explicit confidence and
    an empty list when there is too little to go on.
    """
    stripped = (text or "").strip()
    if not stripped:
        return []

    guesses: list[dict[str, Any]] = []

    khmer_share = _khmer_ratio(stripped)
    if khmer_share >= MIN_KHMER_RATIO:
        guesses.append(
            {"lang": "khm", "iso639_1": "km", "confidence": round(khmer_share, 4)}
        )

    # Whatever is left over for the model-based detector.
    latin_share = 1.0 - (khmer_share if khmer_share >= MIN_KHMER_RATIO else 0.0)
    remainder = _strip_khmer(stripped).strip() if khmer_share else stripped

    if _LANGDETECT and latin_share > 0 and len(remainder) >= MIN_CHARS_FOR_DETECTION:
        try:
            for guess in detect_langs(remainder)[:top]:
                guesses.append(
                    {
                        "lang": _ISO1_TO_ISO2.get(guess.lang, guess.lang),
                        "iso639_1": guess.lang,
                        # Scale into the page as a whole, not just the Latin part.
                        "confidence": round(float(guess.prob) * latin_share, 4),
                    }
                )
        except LangDetectException:
            pass  # too little Latin text to model; the Khmer guess may still stand
        except Exception:  # noqa: BLE001 - detection must never fail a request
            logger.debug("language detection failed")

    guesses.sort(key=lambda item: item["confidence"], reverse=True)
    return guesses[:top]


def merge_languages(per_page: list[list[dict[str, Any]]], *, top: int = 3) -> list[dict[str, Any]]:
    """Combine per-page guesses into a document-level answer."""
    totals: dict[str, dict[str, Any]] = {}
    for page in per_page:
        for guess in page:
            entry = totals.setdefault(
                guess["lang"],
                {
                    "lang": guess["lang"],
                    "iso639_1": guess.get("iso639_1"),
                    "confidence": 0.0,
                    "_n": 0,
                },
            )
            entry["confidence"] += guess["confidence"]
            entry["_n"] += 1
    merged = []
    for entry in totals.values():
        count = entry.pop("_n")
        entry["confidence"] = round(entry["confidence"] / count, 4)
        merged.append(entry)
    merged.sort(key=lambda item: item["confidence"], reverse=True)
    return merged[:top]
