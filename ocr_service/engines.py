"""Chooses the OCR engine behind ``/ocr``.

``OCR_ENGINE=surya`` (default) runs Surya on the GPU; ``OCR_ENGINE=tesseract``
falls back to the Tesseract pipeline in ``ocr.py``. Both return pages with the
same keys, so nothing downstream branches on the engine.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import ocr, surya_engine

ENGINES = ("surya", "tesseract")

ENGINE_NAME = os.getenv("OCR_ENGINE", "surya").strip().lower()
if ENGINE_NAME not in ENGINES:
    raise ValueError(f"OCR_ENGINE must be one of {', '.join(ENGINES)}; got {ENGINE_NAME!r}")


def name() -> str:
    return ENGINE_NAME


def info() -> dict[str, Any]:
    """Status of the active engine, in one shape for both engines."""
    if ENGINE_NAME == "surya":
        return surya_engine.engine_info()

    engine = ocr.engine_info()
    return {
        "name": "tesseract",
        "available": engine["available"],
        "state": "ready" if engine["available"] else "error",
        "version": engine["version"],
        "device": "cpu",
        "device_name": None,
        "uses_language_hints": True,
        "languages": engine["languages"],
        "error": engine["error"],
    }


def missing_languages() -> list[str]:
    """Expected language data the active engine lacks (Tesseract only)."""
    if ENGINE_NAME != "tesseract":
        return []
    installed = ocr.engine_info()["languages"]
    return [code for code in ocr.EXPECTED_LANGS if code not in installed]


def ready() -> bool:
    return info()["state"] == "ready" and not missing_languages()


def warm_up() -> None:
    """Load models ahead of the first request. Blocking; run it in a thread."""
    if ENGINE_NAME == "surya":
        surya_engine.warm_up()


def validate_lang(lang: str) -> str:
    if ENGINE_NAME == "surya":
        return surya_engine.validate_lang(lang)
    return ocr.validate_lang(lang)


def ocr_page(image: np.ndarray, *, lang: str, detail: bool) -> dict[str, Any]:
    if ENGINE_NAME == "surya":
        return surya_engine.ocr_page(image, lang=lang, detail=detail)
    return ocr.ocr_page(image, lang=lang, detail=detail)
