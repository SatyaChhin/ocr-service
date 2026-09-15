"""Tests for saving lab reports to MySQL/MariaDB.

The row mapping and the /lab-reports endpoints run without a database (the
queries are stubbed). One round-trip test uses the real database named in
.env; it is opt-in (``OCR_TEST_DB=1``) and deletes what it writes.
"""

from __future__ import annotations

import os
import uuid
from decimal import Decimal

import pytest
from fastapi.testclient import TestClient

from ocr_service import db
from ocr_service.main import app
from ocr_service.tests.test_lab_results import EXPECTED

requires_db = pytest.mark.skipif(not os.getenv("OCR_TEST_DB"), reason="set OCR_TEST_DB=1 to use the real database")


def _as_fetched(row: tuple) -> dict:
    """A result_rows() tuple as PyMySQL's DictCursor returns it (DECIMAL -> Decimal)."""
    (position, test_name, percent, value, value_numeric, flag, unit, ref_range, section,
     needs_review, review_notes, ocr_confidence) = row
    dec = lambda v: None if v is None else Decimal(str(v))  # noqa: E731
    return {"position": position, "test_name": test_name, "percent": dec(percent), "value": value,
            "value_numeric": dec(value_numeric), "flag": flag, "unit": unit, "ref_range": ref_range,
            "section": section, "needs_review": needs_review, "review_notes": review_notes,
            "ocr_confidence": dec(ocr_confidence)}


def test_rows_round_trip_to_the_exact_json_format() -> None:
    """What goes into lab_results comes back out as the same JSON, key order included."""
    rows = db.result_rows(EXPECTED, review=[])

    back = [db.result_json(_as_fetched(row)) for row in rows]

    assert back == EXPECTED
    assert [list(r) for r in back] == [list(r) for r in EXPECTED]


def test_result_rows_keep_text_values_and_review_state() -> None:
    blood_group = next(r for r in EXPECTED if r["test_name"] == "Blood Group")
    review = [{"needs_review": True, "confidence": 59.9, "notes": [{"code": "low_confidence", "params": {}}]}]

    [row] = db.result_rows([blood_group], review)

    assert row[3] == "O Rh (D): Positive" and row[4] is None  # value, value_numeric
    assert row[9] == 1 and '"low_confidence"' in row[10] and row[11] == 59.9


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
    review = [{"needs_review": r["test_name"] == "Platelets", "notes": [], "confidence": 90.0} for r in EXPECTED]

    saved = db.save_report(report=report, results=EXPECTED, review=review, filename="t.pdf", engine="surya")
    try:
        assert saved["results_saved"] == 22 and saved["needs_review"] == 1
        with pytest.raises(db.DuplicateReport):
            db.save_report(report=report, results=EXPECTED, review=review, filename="t.pdf", engine="surya")
        again = db.save_report(report=report, results=EXPECTED, review=review, filename="t.pdf",
                               engine="surya", replace=True)
        assert again["replaced"] == saved["id"]

        stored = db.get_report(again["id"])
        assert stored["results"] == EXPECTED
        assert stored["collected_at"] == "2026-01-01T08:00:00"
        assert db.get_report(saved["id"]) is None  # replaced, results cascaded away
    finally:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("DELETE FROM lab_reports WHERE patient_code = %s", (report["patient_code"],))
