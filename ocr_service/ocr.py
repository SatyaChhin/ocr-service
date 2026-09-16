"""Shared OCR helpers: decoding, deskew, row grouping and language detection.

The engine itself lives in ``surya_engine``; this module holds the pieces
around it that are not specific to any engine.

Nothing here touches the filesystem: pages arrive as ``bytes`` or as numpy
arrays and results leave as plain dicts.  Nothing here logs document content
either -- the logger records shapes, angles, timings and error *types* only,
because these documents may carry personal health information.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Iterable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

#: Default `lang` spec.  Surya reads every script it knows without a hint,
#: so this is syntax-checked and echoed back rather than acted on.
DEFAULT_LANG = "eng+fra+khm"
LANG_RE = re.compile(r"^[a-z]{3,4}(?:_[a-z]+)?(?:\+[a-z]{3,4}(?:_[a-z]+)?)*$", re.I)

MAX_DESKEW_DEG = float(os.getenv("OCR_MAX_DESKEW_DEG", "12"))


# --------------------------------------------------------------------------
# Errors
# --------------------------------------------------------------------------


class OcrError(Exception):
    """Base class for OCR failures that map to a client-visible response."""


class EngineUnavailable(OcrError):
    """The OCR engine is missing, still loading, or failed to load."""


class UnsupportedLanguage(OcrError):
    """The requested `lang` string is malformed."""


class ImageDecodeError(OcrError):
    """The bytes could not be decoded as a raster image."""


# --------------------------------------------------------------------------
# Language spec
# --------------------------------------------------------------------------


def normalize_lang(lang: str) -> str:
    """Check the *syntax* of a `lang` spec such as ``eng+fra``.

    Syntax only: no engine is consulted.  Surya takes no language hint, so
    a well-formed spec is simply echoed back on the response.
    """
    lang = (lang or "").strip()
    if not lang:
        return DEFAULT_LANG
    if not LANG_RE.match(lang):
        raise UnsupportedLanguage(
            "lang must be one or more '+'-joined ISO 639-2/T codes, e.g. 'eng', 'fra', 'eng+khm'"
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


#: Two boxes whose vertical extents overlap by at least this share of the
#: shorter one sit on the same visual row.
ROW_OVERLAP = float(os.getenv("OCR_ROW_OVERLAP", "0.5"))


def group_rows(items: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group boxes (``bbox`` = ``[x0, y0, x1, y1]``) into visual rows, each left to right.

    Items are taken in order of vertical centre and compared with the first
    item of the current row, not the row's growing extent, so a slightly
    sloped page cannot chain every line into one row.
    """
    rows: list[list[dict[str, Any]]] = []
    span: tuple[float, float] = (0.0, 0.0)
    for item in sorted(items, key=lambda it: (it["bbox"][1] + it["bbox"][3]) / 2):
        top, bottom = item["bbox"][1], item["bbox"][3]
        if rows:
            overlap = min(bottom, span[1]) - max(top, span[0])
            if overlap >= ROW_OVERLAP * min(bottom - top, span[1] - span[0]):
                rows[-1].append(item)
                continue
        rows.append([item])
        span = (top, bottom)
    return [sorted(row, key=lambda it: it["bbox"][0]) for row in rows]


# --------------------------------------------------------------------------
# Language detection (best effort)
# --------------------------------------------------------------------------

# langdetect speaks ISO 639-1; the API reports 639-2/T.
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

    The engine does not report which languages it read, so this runs over the
    extracted text instead.  Two signals are combined:

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
