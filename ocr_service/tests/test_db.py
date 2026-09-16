"""Tests for saving lab reports to MySQL/MariaDB.

Results are stored as JSON on the report row, so the storage test is a
round-trip through that column. The /lab-reports endpoints run without a
database (the queries are stubbed). One round-trip test uses the real
database named in .env; it is opt-in (``OCR_TEST_DB=1``) and deletes what it
writes.
"""

from __future__ import annotations

import json
import os
import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from ocr_service import db
from ocr_service.main import app
from ocr_service.tests.test_lab_results import EXPECTED

requires_db = pytest.mark.skipif(not os.getenv("OCR_TEST_DB"), reason="set OCR_TEST_DB=1 to use the real database")


def test_results_round_trip_through_the_json_column() -> None:
    """What is stored comes back identical -- key order and number types included."""
    back = db._load(db._dump(EXPECTED), [])

    assert back == EXPECTED
    assert [list(r) for r in back] == [list(r) for r in EXPECTED]
    platelets = next(r for r in back if r["name"] == "Platelets")
    assert isinstance(platelets["value"], int)  # 744 does not become 744.0
    blood_group = next(r for r in back if r["name"] == "Blood Group")
    assert blood_group["value"] == "O Rh (D): Positive"  # text values stay text


def test_stored_json_keeps_unicode_as_characters() -> None:
    """utf8mb4 columns hold the characters themselves, not \\uXXXX escapes."""
    stored = db._dump([{"name": "Urea", "value": 5.2, "unit": "µmol/L"}])

    assert "µmol/L" in stored
    assert db._load(stored, [])[0]["unit"] == "µmol/L"


def test_load_tolerates_an_already_decoded_column() -> None:
    """Some drivers decode JSON columns for you; MariaDB's LONGTEXT does not."""
    assert db._load([{"name": "WBC"}], []) == [{"name": "WBC"}]
    assert db._load(None, []) == []


def test_needs_review_counts_only_flagged_results() -> None:
    review = [{"needs_review": False}, {"needs_review": True}, {}, {"needs_review": True}]

    assert db.count_needs_review(review) == 2
    assert db.count_needs_review([]) == 0


# --------------------------------------------------------------------------
# Migration off the pre-JSON lab_results table
# --------------------------------------------------------------------------


def test_legacy_rows_become_the_json_format() -> None:
    """A row of the dropped lab_results table maps back to the JSON record."""
    row = {"test_name": "Neutrophils", "percent": Decimal("65.7"), "value": "7.15",
           "value_numeric": Decimal("7.15"), "flag": "H", "unit": "x10^9/L",
           "ref_range": "2 - 7", "section": "Differential White Cell Count"}

    assert db._legacy_result_json(row) == {
        "name": "Neutrophils", "percent": 65.7, "value": 7.15, "flag": "H",
        "unit": "x10^9/L", "ref_range": "2 - 7", "category": "Differential White Cell Count",
    }


def test_legacy_text_values_keep_their_text() -> None:
    row = {"test_name": "Blood Group", "percent": None, "value": "O Rh (D): Positive",
           "value_numeric": None, "flag": None, "unit": None, "ref_range": None,
           "section": "COMPLETE BLOOD COUNT"}

    assert db._legacy_result_json(row)["value"] == "O Rh (D): Positive"


def test_stored_results_rename_test_name_and_section() -> None:
    """Reports saved before the rename come back under the current keys."""
    old = [{"test_name": "Neutrophils", "percent": 65.7, "value": 7.15, "flag": "H",
            "unit": "x10^9/L", "ref_range": "2 - 7", "section": "Differential White Cell Count"}]

    [record] = db.rename_result_keys(old)

    assert record == {"name": "Neutrophils", "percent": 65.7, "value": 7.15, "flag": "H",
                      "unit": "x10^9/L", "ref_range": "2 - 7",
                      "category": "Differential White Cell Count"}
    # Re-emitted in the documented order, so a migrated row looks freshly saved.
    assert list(record) == ["name", "percent", "value", "flag", "unit", "ref_range", "category"]


def test_renaming_is_a_no_op_on_current_results() -> None:
    assert db.rename_result_keys(EXPECTED) == EXPECTED
    assert [list(r) for r in db.rename_result_keys(EXPECTED)] == [list(r) for r in EXPECTED]


# --------------------------------------------------------------------------
# API, database stubbed
# --------------------------------------------------------------------------


BODY = {
    "filename": "report.pdf",
    "engine": "surya",
    "report": {"patient_code": "SMP-000-0001", "sample_no": "0001-01012026",
               "collected_at": "2026-01-01T08:00:00", "received_at": None, "conflicts": []},
    "results": EXPECTED,
    "review": [],
}


def test_save_passes_the_lab_object_through(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_save(**kwargs):
        calls.append(kwargs)
        return {"id": 7, "results_saved": len(kwargs["results"]), "needs_review": 0, "replaced": None}

    monkeypatch.setattr(db, "save_report", fake_save)
    with TestClient(app) as client:
        response = client.post("/lab-reports", json=BODY)

    assert response.status_code == 201
    assert response.json() == {"id": 7, "results_saved": 22, "needs_review": 0, "replaced": None}
    saved = calls[0]
    assert saved["results"] == EXPECTED  # values keep their types: 744 stays an int
    assert saved["report"]["collected_at"] == "2026-01-01T08:00:00"
    assert saved["replace"] is False


def test_saving_twice_is_a_409_with_the_existing_report(monkeypatch: pytest.MonkeyPatch) -> None:
    def duplicate(**kwargs):
        raise db.DuplicateReport(7)

    monkeypatch.setattr(db, "save_report", duplicate)
    with TestClient(app) as client:
        response = client.post("/lab-reports", json=BODY)

    assert response.status_code == 409
    assert response.json()["report_id"] == 7
    assert response.json()["request_id"] == response.headers["X-Request-ID"]


def test_database_down_is_a_503(monkeypatch: pytest.MonkeyPatch) -> None:
    def down(**kwargs):
        raise db.DatabaseUnavailable("cannot connect")

    monkeypatch.setattr(db, "save_report", down)
    with TestClient(app) as client:
        response = client.post("/lab-reports", json=BODY)

    assert response.status_code == 503
    assert "database" in response.json()["detail"].lower()


def test_save_rejects_an_empty_or_malformed_body() -> None:
    with TestClient(app) as client:
        assert client.post("/lab-reports", json={**BODY, "results": []}).status_code == 422
        assert client.post("/lab-reports", json={**BODY, "results": [{"value": 1}]}).status_code == 422


# --------------------------------------------------------------------------
# Real database (opt-in)
# --------------------------------------------------------------------------


@requires_db
def test_save_and_read_back_from_mysql() -> None:
    report = {**BODY["report"], "patient_code": f"TEST-{uuid.uuid4().hex[:10]}"}
    review = [{"needs_review": r["name"] == "Platelets", "notes": [], "confidence": 90.0} for r in EXPECTED]

    saved = db.save_report(report=report, results=EXPECTED, review=review, filename="t.pdf", engine="surya")
    try:
        assert saved["results_saved"] == 22 and saved["needs_review"] == 1
        with pytest.raises(db.DuplicateReport):
            db.save_report(report=report, results=EXPECTED, review=review, filename="t.pdf", engine="surya")
        again = db.save_report(report=report, results=EXPECTED, review=review, filename="t.pdf",
                               engine="surya", replace=True)
        assert again["replaced"] == saved["id"]

        stored = db.get_report(again["id"])
        assert stored["results"] == EXPECTED  # byte-for-byte the format /ocr returned
        assert [list(r) for r in stored["results"]] == [list(r) for r in EXPECTED]  # key order too
        assert stored["review"] == review
        assert stored["collected_at"] == "2026-01-01T08:00:00"
        assert db.get_report(saved["id"]) is None  # replaced report is gone
    finally:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM lab_reports WHERE patient_code = %s", (report["patient_code"],))
