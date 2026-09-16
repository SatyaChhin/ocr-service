"""The OCR engine behind ``/ocr``.

Surya is the only engine: it runs in-process on the GPU and reads every script
it knows, Khmer included, without a language hint. This module is the seam the
rest of the service talks to, so nothing else imports the engine directly.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np

from . import surya_engine

ENGINES = ("surya",)

ENGINE_NAME = os.getenv("OCR_ENGINE", "surya").strip().lower()
if ENGINE_NAME not in ENGINES:
    # Tesseract used to be selectable here. Say so rather than fall through to
    # a generic error: an old .env or `run.ps1 -Engine tesseract` lands here.
    hint = " (Tesseract support was removed; Surya reads Khmer without it)" if ENGINE_NAME == "tesseract" else ""
    raise ValueError(f"OCR_ENGINE must be {ENGINES[0]!r}; got {ENGINE_NAME!r}{hint}")


def name() -> str:
    return ENGINE_NAME


def info() -> dict[str, Any]:
    """Status of the active engine."""
    return surya_engine.engine_info()


def ready() -> bool:
    return info()["state"] == "ready"


def warm_up() -> None:
    """Load models ahead of the first request. Blocking; run it in a thread."""
    surya_engine.warm_up()


def validate_lang(lang: str) -> str:
    return surya_engine.validate_lang(lang)


def ocr_page(image: np.ndarray, *, lang: str, detail: bool) -> dict[str, Any]:
    return surya_engine.ocr_page(image, lang=lang, detail=detail)
