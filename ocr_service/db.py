"""MySQL/MariaDB storage for lab results.

Connection settings come from the environment, or from ``ocr-service/.env``
(git-ignored)::

    OCR_DB_HOST=127.0.0.1
    OCR_DB_PORT=3306
    OCR_DB_USER=root
    OCR_DB_PASSWORD=
    OCR_DB_NAME=ocr

The tables in ``schema.sql`` are created on first use. Every query is
parameterised, and nothing here logs document content.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterator

import pymysql
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Real environment variables win over the file.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


class DatabaseUnavailable(Exception):
    """The database cannot be reached or rejected the connection."""


class DuplicateReport(Exception):
    """A report with the same patient_code and sample_no is already saved."""

    def __init__(self, report_id: int):
        super().__init__(f"already saved as report {report_id}")
        self.report_id = report_id


def settings() -> dict[str, Any]:
    return {
        "host": os.getenv("OCR_DB_HOST", "127.0.0.1"),
        "port": int(os.getenv("OCR_DB_PORT", "3306")),
        "user": os.getenv("OCR_DB_USER", "root"),
        "password": os.getenv("OCR_DB_PASSWORD", ""),
        "database": os.getenv("OCR_DB_NAME", "ocr"),
    }


_schema_ready = False
_schema_lock = threading.Lock()


@contextmanager
def connect(timeout: int = 5) -> Iterator[pymysql.connections.Connection]:
    """A short-lived connection in one transaction: committed on success."""
    try:
        conn = pymysql.connect(**settings(), charset="utf8mb4", autocommit=False,
                               connect_timeout=timeout, cursorclass=pymysql.cursors.DictCursor)
    except pymysql.MySQLError as exc:
        raise DatabaseUnavailable(f"cannot connect to MySQL ({type(exc).__name__}: {exc.args[0] if exc.args else ''})") from exc
    try:
        _ensure_schema(conn)
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _ensure_schema(conn: pymysql.connections.Connection) -> None:
    global _schema_ready
    if _schema_ready:
        return
    with _schema_lock:
        if _schema_ready:
            return
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        statements = [s.strip() for s in re.sub(r"--[^\n]*", "", sql).split(";") if s.strip()]
        with conn.cursor() as cur:
            for statement in statements:
                cur.execute(statement)
        conn.commit()
        _schema_ready = True


def info() -> dict[str, Any]:
    """Status for /health; never raises."""
    config = settings()
    base = {"name": config["database"], "host": f"{config['host']}:{config['port']}"}
    try:
        with connect(timeout=2) as conn, conn.cursor() as cur:
            cur.execute("SELECT VERSION() AS version, COUNT(*) AS reports FROM lab_reports")
            row = cur.fetchone()
        return {**base, "available": True, "version": row["version"], "reports": row["reports"], "error": None}
    except Exception as exc:  # noqa: BLE001 - health must not fail
        return {**base, "available": False, "version": None, "reports": None, "error": str(exc)}


# --------------------------------------------------------------------------
# Mapping between the JSON format and table rows (pure; tested without a DB)
# --------------------------------------------------------------------------


def _mysql_datetime(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")


def result_rows(results: list[dict[str, Any]], review: list[dict[str, Any]]) -> list[tuple]:
    """``lab.results`` (+ ``lab.review``) -> ``lab_results`` column tuples."""
    rows = []
    for position, result in enumerate(results, start=1):
        entry = review[position - 1] if position - 1 < len(review) else {}
        value = result["value"]
        numeric = value if isinstance(value, (int, float)) and not isinstance(value, bool) else None
        notes = entry.get("notes") or []
        rows.append((
            position,
            result["test_name"],
            result.get("percent"),
            str(value),
            numeric,
            result.get("flag"),
            result.get("unit"),
            result.get("ref_range"),
            result.get("section"),
            1 if entry.get("needs_review") else 0,
            json.dumps(notes, ensure_ascii=False) if notes else None,
            entry.get("confidence"),
        ))
    return rows


def _number(value: Decimal | None) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def result_json(row: dict[str, Any]) -> dict[str, Any]:
    """A ``lab_results`` row -> the JSON format, keys in the original order."""
    record: dict[str, Any] = {"test_name": row["test_name"]}
    if row["percent"] is not None:
        record["percent"] = _number(row["percent"])
    value_numeric = _number(row["value_numeric"])
    record.update(
        value=value_numeric if value_numeric is not None else row["value"],
        flag=row["flag"], unit=row["unit"], ref_range=row["ref_range"], section=row["section"],
    )
    return record


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

_RESULT_COLUMNS = ("position, test_name, percent, value, value_numeric, flag, unit, ref_range, section, "
                   "needs_review, review_notes, ocr_confidence")


def save_report(*, report: dict[str, Any], results: list[dict[str, Any]], review: list[dict[str, Any]],
                filename: str | None, engine: str, replace: bool = False) -> dict[str, Any]:
    """Insert a report and its results in one transaction.

    Raises DuplicateReport when patient_code + sample_no is already saved and
    ``replace`` is false; with ``replace`` the old report and its results are
    deleted first.
    """
    rows = result_rows(results, review)
    needs_review = sum(row[9] for row in rows)
    with connect() as conn, conn.cursor() as cur:
        patient, sample = report.get("patient_code"), report.get("sample_no")
        replaced = None
        if patient and sample:
            cur.execute("SELECT id FROM lab_reports WHERE patient_code = %s AND sample_no = %s FOR UPDATE",
                        (patient, sample))
            existing = cur.fetchone()
            if existing and not replace:
                raise DuplicateReport(existing["id"])
            if existing:
                cur.execute("DELETE FROM lab_reports WHERE id = %s", (existing["id"],))
                replaced = existing["id"]
        cur.execute(
            "INSERT INTO lab_reports (patient_code, sample_no, collected_at, received_at, source_filename,"
            " engine, results_count, needs_review_count) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
            (patient, sample, _mysql_datetime(report.get("collected_at")), _mysql_datetime(report.get("received_at")),
             filename, engine, len(rows), needs_review),
        )
        report_id = cur.lastrowid
        cur.executemany(
            f"INSERT INTO lab_results (report_id, {_RESULT_COLUMNS})"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            [(report_id, *row) for row in rows],
        )
    return {"id": report_id, "results_saved": len(rows), "needs_review": needs_review, "replaced": replaced}


def _report_json(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": row["id"],
        "patient_code": row["patient_code"],
        "sample_no": row["sample_no"],
        "collected_at": row["collected_at"].isoformat() if row["collected_at"] else None,
        "received_at": row["received_at"].isoformat() if row["received_at"] else None,
        "source_filename": row["source_filename"],
        "engine": row["engine"],
        "results_count": row["results_count"],
        "needs_review_count": row["needs_review_count"],
        "created_at": row["created_at"].isoformat(),
    }


def get_report(report_id: int) -> dict[str, Any] | None:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM lab_reports WHERE id = %s", (report_id,))
        report = cur.fetchone()
        if report is None:
            return None
        cur.execute(f"SELECT {_RESULT_COLUMNS} FROM lab_results WHERE report_id = %s ORDER BY position",
                    (report_id,))
        rows = cur.fetchall()
    return {
        **_report_json(report),
        "results": [result_json(row) for row in rows],
        "review": [
            {"needs_review": bool(row["needs_review"]),
             "notes": json.loads(row["review_notes"]) if row["review_notes"] else [],
             "confidence": float(row["ocr_confidence"]) if row["ocr_confidence"] is not None else None}
            for row in rows
        ],
    }


def list_reports(limit: int = 20) -> list[dict[str, Any]]:
    with connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT * FROM lab_reports ORDER BY id DESC LIMIT %s", (limit,))
        return [_report_json(row) for row in cur.fetchall()]
