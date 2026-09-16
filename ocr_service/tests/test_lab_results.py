"""Tests for structured lab-result extraction.

The fixture page reproduces the layout of the lab's report -- results tables
under a "Test Name" header, department banners, a footer -- with synthetic
patient details. The table rows are the text Surya read from a real report,
and ``EXPECTED`` is the database format the lab system inserts.
"""

from __future__ import annotations

import copy

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocr_service import engines, lab_results
from ocr_service.main import app

PAGE_ONE = [
    "LABORATORY RESULT",
    "Patient ID : SMP-000-0001 Sample : MAT",
    "Name : Test Patient Sex : F Age : 24",
    "HEAMATOLOGY លេខសំណាក អ្នកស្នើសុំ ថ្ងៃប្រមូលសំណាក ថ្ងៃទទួលសំណាក",
    "Blood-EDTA 0001-01012026 01-Jan-2026 08:00 01-Jan-2026 09:00",
    "Test Name លទ្ធផល ខ្នាត តំលៃយោង",
    "COMPLETE BLOOD COUNT",
    "WBC 10.88 x109/L 5 - 13",
    "RBC ..... 5.31 H x1012/L 3.8 - 4.8",
    "Hemoglobin ..... 9.9 L g/dL 12 - 15",
    "Hematocrit ..... 32.3 L % 36 - 46",
    "MCV 60.8 L fl 83 - 101",
    "MCH ..... 18.6 L pg 27 - 32",
    "MCHC ..... 30.7 L g/dL 31.5 - 34.5",
    "Platelets ..... 744 H x109/L 150 - 400",
    "RDW-CV 15.4 H % 11.5 - 14",
    "Differential White Cell Count",
    "Neutrophils (%) 65.7% 7.15 H x109/L 2 - 7",
    "Lymphocytes (%) 25.3% 2.75 x109/L 1 - 3",
    "Monocytes (%) 7.0% 0.76 x109/L 0.2 - 1",
    "Eosinophils (%) 1.8% 0.20 x109/L 0.02 - 0.5",
    "Basophils (%) 0.2% 0.02 x109/L 0.02 - 0.1",
    "Blood Group O Rh (D): Positive",
    "HEAMATOLOGY លេខសំណាក អ្នកស្នើសុំ ថ្ងៃប្រមូលសំណាក ថ្ងៃទទួលសំណាក",
    "Blood-Sodium-Citrate 0001-01012026 01-Jan-2026 08:00 01-Jan-2026 09:00",
    "Test Name លទ្ធផល ខ្នាត តំលៃយោង",
    "Prothrombin Time",
    "PT (Sec) 10.7 seconds 10.4 - 14.4",
    "PT (INR) 0.94 INR 0.8 - 1.2",
    "aPTT ..... 26 seconds 23 - 34",
    "ថ្ងៃពិសោធន៍ចុងក្រោយ : 01-Jan-2026 ថ្ងៃចេញលទ្ធផលដំបូង : 01-Jan-2026 10:00",
]

PAGE_TWO = [
    "Patient ID : SMP-000-0001 Sample : MAT",
    "BIOCHEMISTRY លេខសំណាក អ្នកស្នើសុំ ថ្ងៃប្រមូលសំណាក ថ្ងៃទទួលសំណាក",
    "Blood-Clotted 0001-01012026 01-Jan-2026 08:00 01-Jan-2026 09:00",
    "Test Name លទ្ធផល ខ្នាត តំលៃយោង",
    "LIVER FUNCTIONS",
    "Transaminase",
    "AST ..... 22 U/L < 31",
    "ALT ..... 19 U/L < 32",
    "RENAL FUNCTIONS",
    "Urea 13 mg/dL 10 - 50",
    "Creatinine 0.60 mg/dL 0.5 - 0.9",
    "ថ្ងៃពិសោធន៍ចុងក្រោយ : 01-Jan-2026 ថ្ងៃចេញលទ្ធផលដំបូង : 01-Jan-2026 10:00",
]

CBC = "COMPLETE BLOOD COUNT"
DIFF = "Differential White Cell Count"
EXPECTED = [
    {"name": "WBC", "value": 10.88, "flag": None, "unit": "x10^9/L", "ref_range": "5 - 13", "category": CBC},
    {"name": "RBC", "value": 5.31, "flag": "H", "unit": "x10^12/L", "ref_range": "3.8 - 4.8", "category": CBC},
    {"name": "Hemoglobin", "value": 9.9, "flag": "L", "unit": "g/dL", "ref_range": "12 - 15", "category": CBC},
    {"name": "Hematocrit", "value": 32.3, "flag": "L", "unit": "%", "ref_range": "36 - 46", "category": CBC},
    {"name": "MCV", "value": 60.8, "flag": "L", "unit": "fl", "ref_range": "83 - 101", "category": CBC},
    {"name": "MCH", "value": 18.6, "flag": "L", "unit": "pg", "ref_range": "27 - 32", "category": CBC},
    {"name": "MCHC", "value": 30.7, "flag": "L", "unit": "g/dL", "ref_range": "31.5 - 34.5", "category": CBC},
    {"name": "Platelets", "value": 744, "flag": "H", "unit": "x10^9/L", "ref_range": "150 - 400", "category": CBC},
    {"name": "RDW-CV", "value": 15.4, "flag": "H", "unit": "%", "ref_range": "11.5 - 14", "category": CBC},
    {"name": "Neutrophils", "percent": 65.7, "value": 7.15, "flag": "H", "unit": "x10^9/L", "ref_range": "2 - 7", "category": DIFF},
    {"name": "Lymphocytes", "percent": 25.3, "value": 2.75, "flag": None, "unit": "x10^9/L", "ref_range": "1 - 3", "category": DIFF},
    {"name": "Monocytes", "percent": 7, "value": 0.76, "flag": None, "unit": "x10^9/L", "ref_range": "0.2 - 1", "category": DIFF},
    {"name": "Eosinophils", "percent": 1.8, "value": 0.2, "flag": None, "unit": "x10^9/L", "ref_range": "0.02 - 0.5", "category": DIFF},
    {"name": "Basophils", "percent": 0.2, "value": 0.02, "flag": None, "unit": "x10^9/L", "ref_range": "0.02 - 0.1", "category": DIFF},
    {"name": "Blood Group", "value": "O Rh (D): Positive", "flag": None, "unit": None, "ref_range": None, "category": CBC},
    {"name": "PT (Sec)", "value": 10.7, "flag": None, "unit": "seconds", "ref_range": "10.4 - 14.4", "category": "Prothrombin Time"},
    {"name": "PT (INR)", "value": 0.94, "flag": None, "unit": "INR", "ref_range": "0.8 - 1.2", "category": "Prothrombin Time"},
    {"name": "aPTT", "value": 26, "flag": None, "unit": "seconds", "ref_range": "23 - 34", "category": "Prothrombin Time"},
    {"name": "AST", "value": 22, "flag": None, "unit": "U/L", "ref_range": "< 31", "category": "LIVER FUNCTIONS"},
    {"name": "ALT", "value": 19, "flag": None, "unit": "U/L", "ref_range": "< 32", "category": "LIVER FUNCTIONS"},
    {"name": "Urea", "value": 13, "flag": None, "unit": "mg/dL", "ref_range": "10 - 50", "category": "RENAL FUNCTIONS"},
    {"name": "Creatinine", "value": 0.6, "flag": None, "unit": "mg/dL", "ref_range": "0.5 - 0.9", "category": "RENAL FUNCTIONS"},
]


def _page(rows: list[str], number: int = 1, conf: dict[str, float] | None = None) -> dict:
    """An OCR page whose words sit on the given rows, as an engine returns it."""
    words = []
    for index, row in enumerate(rows):
        x = 40
        for position, text in enumerate(row.split(), start=1):
            width = 12 * len(text)
            words.append({"text": text, "conf": (conf or {}).get(text, 98.0),
                          "bbox": [x, 100 + 40 * index, width, 24],
                          "block": 1, "par": 1, "line": index + 1, "word": position})
            x += width + 14
    return {"page": number, "text": "\n".join(rows), "words": words}


def _extract(*pages: list[str], conf: dict[str, float] | None = None) -> dict:
    return lab_results.extract([_page(rows, n, conf) for n, rows in enumerate(pages, start=1)])


def _notes(lab: dict, name: str) -> list[str]:
    index = [r["name"] for r in lab["results"]].index(name)
    return [note["code"] for note in lab["review"][index]["notes"]]


def _replace(rows: list[str], old: str, new: str) -> list[str]:
    return [new if row == old else row for row in rows]


# --------------------------------------------------------------------------
# The database format
# --------------------------------------------------------------------------


def test_extract_matches_the_database_format_exactly() -> None:
    lab = _extract(PAGE_ONE, PAGE_TWO)

    assert lab["results"] == EXPECTED
    # Key order matters to the importer: percent sits right after name.
    assert [list(r) for r in lab["results"]] == [list(r) for r in EXPECTED]
    assert lab["unparsed"] == []
    assert not any(entry["needs_review"] for entry in lab["review"])


def test_patient_details_and_banners_are_not_read_as_results() -> None:
    """"Age : 24" and "0001-01012026" have numbers; neither is a test."""
    names = {r["name"] for r in _extract(PAGE_ONE)["results"]}

    assert not names & {"Age", "Name", "Patient ID", "Blood-EDTA", "Blood-Sodium-Citrate"}


def test_categories_come_from_the_catalog_not_the_nearest_heading() -> None:
    """AST/ALT sit under "Transaminase" but belong to LIVER FUNCTIONS."""
    lab = _extract(PAGE_TWO)

    assert {r["name"]: r["category"] for r in lab["results"]}["AST"] == "LIVER FUNCTIONS"


def test_unknown_tests_take_the_nearest_heading() -> None:
    rows = ["Test Name", "GLYCEMIA", "Fasting Glucose ..... 110 H mg/dL 70 - 100"]

    [record] = _extract(rows)["results"]

    assert record == {"name": "Fasting Glucose", "value": 110, "flag": "H", "unit": "mg/dL",
                      "ref_range": "70 - 100", "category": "GLYCEMIA"}


def test_a_name_ocr_turned_to_noise_is_not_a_result() -> None:
    rows = ["Test Name", "ឯ]ខ169 , -.- .---- 744 H xl0°/L 150 - 400"]

    lab = _extract(rows)

    assert lab["results"] == []
    assert lab["unparsed"]  # still surfaced for a person to look at


# --------------------------------------------------------------------------
# Normalisation
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("printed", "expected"),
    [
        ("x109/L", "x10^9/L"),  # OCR flattened the superscript
        ("x1012/L", "x10^12/L"),
        ("x10⁹/L", "x10^9/L"),
        ("x10¹²/L", "x10^12/L"),  # ¹² are Latin-1, not in the U+2070 block
        ("×10^9/L", "x10^9/L"),
        ("cells/mm³", "cells/mm^3"),
        ("g/dL", "g/dL"),
    ],
)
def test_normalize_unit(printed: str, expected: str) -> None:
    assert lab_results.normalize_unit(printed) == expected


def test_catalog_names_match_whole_words_and_tolerate_typos() -> None:
    rows = ["Test Name", "MCHC 30.7 L g/dL 31.5 - 34.5", "Hemoglobln 9.9 L g/dL 12 - 15"]

    lab = _extract(rows)

    assert [r["name"] for r in lab["results"]] == ["MCHC", "Hemoglobin"]  # not "MCH"
    assert _notes(lab, "Hemoglobin") == ["name_fuzzy"]


def test_blood_group_spacing_is_restored() -> None:
    rows = ["Test Name", "Blood Group ORh (D) : Positive"]

    assert _extract(rows)["results"][0]["value"] == "O Rh (D): Positive"


# --------------------------------------------------------------------------
# Checks that catch misread digits
# --------------------------------------------------------------------------


def test_red_cell_indices_catch_a_misread_digit() -> None:
    """MCV 60.8 read as 68.8: Hct / RBC x 10 says it should be ~60.8."""
    lab = _extract(_replace(PAGE_ONE, "MCV 60.8 L fl 83 - 101", "MCV 68.8 L fl 83 - 101"))

    assert "index_mismatch" in _notes(lab, "MCV")
    assert _notes(lab, "MCH") == []  # the other indices still agree


def test_differential_catches_a_percent_sign_read_as_digits() -> None:
    """"0.2%" read as "0.296": flagged, with the reading that fits WBC."""
    lab = _extract(_replace(PAGE_ONE, "Basophils (%) 0.2% 0.02 x109/L 0.02 - 0.1",
                            "Basophils (%) 0.296 0.02 x109/L 0.02 - 0.1"))

    index = [r["name"] for r in lab["results"]].index("Basophils")
    notes = {note["code"]: note["params"] for note in lab["review"][index]["notes"]}
    assert "differential_mismatch" in notes
    assert notes["percent_sign_missing"] == {"read": "0.296", "suggested": 0.2}
    assert lab["results"][index]["percent"] == 0.296  # never corrected silently


@pytest.mark.parametrize(
    ("row", "code"),
    [
        ("RBC ..... 5.31 x1012/L 3.8 - 4.8", "flag_not_set"),  # H lost
        ("RBC ..... 4.31 H x1012/L 3.8 - 4.8", "flag_unexpected"),  # 5 read as 4
        ("RBC ..... 5.31 L x1012/L 3.8 - 4.8", "flag_wrong_direction"),
    ],
)
def test_flag_is_checked_against_the_reference_range(row: str, code: str) -> None:
    lab = _extract(["Test Name", row])

    assert code in _notes(lab, "RBC")


def test_low_confidence_is_waived_only_for_cross_checked_values() -> None:
    """Surya scores bold print low; this report prints flagged values in bold."""
    lab = _extract(PAGE_ONE, conf={"60.8": 50.0, "744": 55.0})

    assert _notes(lab, "MCV") == []  # confirmed by Hct / RBC x 10
    assert _notes(lab, "Platelets") == ["low_confidence"]  # nothing to confirm it


def test_a_missing_unit_is_filled_from_the_catalog_and_flagged() -> None:
    lab = _extract(["Test Name", "Urea 13 10 - 50"])

    assert lab["results"][0]["unit"] == "mg/dL"
    assert _notes(lab, "Urea") == ["unit_missing"]


# --------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------


def test_ocr_lab_param_returns_results_without_word_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    page = _page(PAGE_TWO)
    page.update(word_count=len(page["words"]), mean_confidence=98.0, languages=[], ocr_ms=1.0,
                width=1000, height=1400, scale=1.0, skew_deg=0.0, preprocess_ms=1.0)
    requested = []

    def fake_ocr_page(image, *, lang, detail):
        requested.append(detail)
        return copy.deepcopy(page)

    monkeypatch.setattr(engines, "ocr_page", fake_ocr_page)
    monkeypatch.setattr(engines, "validate_lang", lambda lang: lang)  # no engine install needed
    ok, png = cv2.imencode(".png", np.full((60, 60, 3), 255, dtype=np.uint8))

    with TestClient(app) as client:
        response = client.post("/ocr", params={"lab": "true"}, files={"file": ("r.png", png.tobytes(), "image/png")})

    assert response.status_code == 200
    body = response.json()
    assert requested == [True]  # words are needed to rebuild table rows...
    assert body["pages"][0]["words"] is None  # ...but not returned unless asked for
    assert body["lab"]["results"] == EXPECTED[-4:]
    assert body["lab"]["review"][0]["source"] == "AST 22 U/L < 31"
