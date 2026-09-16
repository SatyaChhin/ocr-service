"""Tests for the Surya engine.

The result mapping and the request path run against fake Surya predictions,
so they need no GPU and no model download. One end-to-end test runs the real
models; it is opt-in (``OCR_TEST_SURYA=1``) because loading them takes ~15 s
and a CUDA device.
"""

from __future__ import annotations

import os

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocr_service import ocr, surya_engine
from ocr_service.main import app
from ocr_service.tests.conftest import fake_char as _char
from ocr_service.tests.conftest import fake_chars_for as _chars_for
from ocr_service.tests.conftest import fake_line as _line

requires_surya_models = pytest.mark.skipif(
    not os.getenv("OCR_TEST_SURYA"), reason="set OCR_TEST_SURYA=1 to run Surya on real models"
)


# --------------------------------------------------------------------------
# Mapping
# --------------------------------------------------------------------------


def test_clean_text_drops_formatting_tags_and_decodes_entities() -> None:
    assert surya_engine.clean_text("<b>Patient ID</b>") == "Patient ID"
    assert surya_engine.clean_text("A &amp; B<br/>  C") == "A & B C"
    assert surya_engine.clean_text("x<sup>2</sup>") == "x2"


def test_words_from_chars_splits_on_spaces_and_averages_confidence() -> None:
    chars = [
        _char("", 0, 0, 0.99, valid=False),  # special token, no usable box
        _char("<b>", 0, 0, 0.5, valid=False),  # formatting token
        _char("N", 10, 20, 0.8),
        _char("o", 20, 30, 1.0),
        _char(" ", 30, 40, 1.0),
        _char("1", 40, 50, 0.6),
    ]

    words = surya_engine.words_from_chars(chars)

    assert [w["text"] for w in words] == ["No", "1"]
    # Mean of all the word's characters, not just the first (Surya's own
    # words_from_chars would report 80 here).
    assert words[0]["conf"] == pytest.approx(90.0)
    assert words[0]["bbox"] == [10, 10, 20, 20]  # [left, top, width, height]
    assert words[1]["conf"] == pytest.approx(60.0)


def test_group_rows_keeps_a_label_next_to_its_value() -> None:
    """Surya detects them as two lines, the value sitting a few px higher."""
    value = {"text": ": SMP-000-0001", "bbox": [505, 283, 731, 313]}
    label = {"text": "Patient ID", "bbox": [126, 286, 272, 315]}
    below = {"text": "Name", "bbox": [126, 339, 219, 368]}

    rows = ocr.group_rows([value, below, label])

    assert [[line["text"] for line in row] for row in rows] == [
        ["Patient ID", ": SMP-000-0001"],
        ["Name"],
    ]


def test_group_rows_does_not_chain_a_sloped_page_into_one_row() -> None:
    # Each line overlaps the next by a little; none overlaps the first by half.
    lines = [{"text": str(i), "bbox": [0, 30 * i, 100, 30 * i + 40]} for i in range(5)]

    assert len(ocr.group_rows(lines)) >= 3


def test_build_page_numbers_words_by_row() -> None:
    text_lines = [
        _line(": SMP-1", [300, 98, 370, 120], _chars_for(": SMP-1", 300, 98, 120)),
        _line("<b>Patient ID</b>", [100, 100, 200, 122], _chars_for("Patient ID", 100, 100, 122)),
        _line("Name", [100, 150, 140, 172], _chars_for("Name", 100, 150, 172)),
        _line("<br>", [0, 0, 5, 5]),  # nothing left after cleaning -> dropped
    ]

    text, words = surya_engine.build_page(text_lines)

    assert text == "Patient ID : SMP-1\nName"
    assert [(w["text"], w["line"], w["word"]) for w in words] == [
        ("Patient", 1, 1),
        ("ID", 1, 2),
        (":", 1, 3),
        ("SMP-1", 1, 4),
        ("Name", 2, 1),
    ]


# --------------------------------------------------------------------------
# Request path with fake predictions
# --------------------------------------------------------------------------


def _png(width: int = 400, height: int = 120) -> bytes:
    ok, buffer = cv2.imencode(".png", np.full((height, width, 3), 255, dtype=np.uint8))
    assert ok
    return buffer.tobytes()


def test_ocr_through_surya_returns_the_same_shape(fake_surya: list) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/ocr",
            params={"detail": "true", "lang": "eng+khm"},
            files={"file": ("scan.png", _png(), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["engine"] == "surya"
    assert body["lang"] == "eng+khm"  # echoed; Surya takes no language hint
    assert body["text"] == "Hemoglobin 9.7"
    page = body["pages"][0]
    assert (page["width"], page["height"], page["scale"], page["skew_deg"]) == (400, 120, 1.0, 0.0)
    assert [(w["text"], w["conf"]) for w in page["words"]] == [("Hemoglobin", 95.0), ("9.7", 50.0)]
    assert page["mean_confidence"] == pytest.approx(72.5)

    # Surya receives an RGB PIL image with math mode off.
    images, kwargs = fake_surya[0]
    assert images[0].mode == "RGB" and kwargs["math_mode"] is False


def test_health_reports_the_active_engine(fake_surya: list) -> None:
    with TestClient(app) as client:
        body = client.get("/health").json()

    assert body["engine"]["name"] == "surya"
    assert body["engine"]["uses_language_hints"] is False
    assert body["ready"] is True


def test_prepare_downscales_only_oversized_pages(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(surya_engine, "MAX_PIXELS", 10_000)

    small, scale, _ = surya_engine._prepare(np.zeros((50, 100, 3), dtype=np.uint8))
    assert small.shape == (50, 100, 3) and scale == 1.0

    big, scale, _ = surya_engine._prepare(np.zeros((400, 400), dtype=np.uint8))
    assert big.shape[2] == 3 and big.shape[0] * big.shape[1] <= 10_000
    assert scale == pytest.approx(0.25)


def test_prepare_straightens_a_tilted_page() -> None:
    """A tilted phone photo would otherwise split table rows in two."""
    page = np.full((900, 900, 3), 255, dtype=np.uint8)
    for top in range(80, 820, 60):
        cv2.rectangle(page, (120, top), (780, top + 18), (0, 0, 0), -1)
    tilted = cv2.warpAffine(page, cv2.getRotationMatrix2D((450, 450), 2.0, 1.0), (900, 900),
                            borderValue=(255, 255, 255))

    _, _, skew = surya_engine._prepare(tilted)
    assert skew == pytest.approx(-2.0, abs=0.5)

    _, _, skew = surya_engine._prepare(page)
    assert skew == 0.0  # below OCR_SURYA_DESKEW_MIN_DEG: left alone


# --------------------------------------------------------------------------
# Real models (opt-in)
# --------------------------------------------------------------------------


@requires_surya_models
def test_surya_reads_a_rendered_page() -> None:
    page = np.full((260, 1100, 3), 255, dtype=np.uint8)
    cv2.putText(page, "Invoice total: 1234.56", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(page, "Patient ID: SMP-0001", (30, 210), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 4, cv2.LINE_AA)

    result = surya_engine.ocr_page(page, detail=True)

    assert "1234.56" in result["text"]
    assert "SMP-0001" in result["text"]
    assert result["words"] and all(len(word["bbox"]) == 4 for word in result["words"])
