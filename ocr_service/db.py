"""MySQL/MariaDB storage for lab results.

Connection settings come from the environment, or from ``ocr-service/.env``
(git-ignored)::

    OCR_DB_HOST=127.0.0.1
    OCR_DB_PORT=3306
    OCR_DB_USER=root
    OCR_DB_PASSWORD=
    OCR_DB_NAME=ocr

The table in ``schema.sql`` is created on first use. One row per report holds
its results as JSON, exactly as ``/ocr?lab=true`` returned them. Every query
is parameterised, and nothing here logs document content.
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
            _migrate_results_into_json(cur)
            _migrate_result_keys(cur)
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
# JSON columns
# --------------------------------------------------------------------------


def _mysql_datetime(value: str | None) -> str | None:
    if not value:
        return None
    return datetime.fromisoformat(value).strftime("%Y-%m-%d %H:%M:%S")


def _dump(value: Any) -> str:
    """A list/dict as it is stored: compact, Unicode kept as characters."""
    return json.dumps(value, ensure_ascii=False)


def _load(value: Any, default: Any) -> Any:
    """A JSON column as Python. Drivers hand these back as ``str``; MariaDB's
    JSON is an alias for LONGTEXT, so never assume it is decoded already."""
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    return json.loads(value)


def count_needs_review(review: list[dict[str, Any]]) -> int:
    """How many results a person still has to check."""
    return sum(1 for entry in review if entry.get("needs_review"))


# --------------------------------------------------------------------------
# One-time migration off the old lab_results table
# --------------------------------------------------------------------------

_LEGACY_COLUMNS = ("position, test_name, percent, value, value_numeric, flag, unit, ref_range, section, "
                   "needs_review, review_notes, ocr_confidence")


def _number(value: Decimal | None) -> int | float | None:
    if value is None:
        return None
    number = float(value)
    return int(number) if number.is_integer() else number


def _legacy_result_json(row: dict[str, Any]) -> dict[str, Any]:
    """An old ``lab_results`` row -> the JSON format, keys in the original order."""
    record: dict[str, Any] = {"name": row["test_name"]}
    if row["percent"] is not None:
        record["percent"] = _number(row["percent"])
    value_numeric = _number(row["value_numeric"])
    record.update(
        value=value_numeric if value_numeric is not None else row["value"],
        flag=row["flag"], unit=row["unit"], ref_range=row["ref_range"], category=row["section"],
    )
    return record


_RENAMED_KEYS = {"test_name": "name", "section": "category"}
_KEY_ORDER = ("name", "percent", "value", "flag", "unit", "ref_range", "category")


def rename_result_keys(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Results stored under the old key names -> the current ones.

    ``test_name`` became ``name`` and ``section`` became ``category``. Keys are
    re-emitted in the documented order so a migrated row is indistinguishable
    from a freshly saved one.
    """
    renamed = []
    for result in results:
        record = {_RENAMED_KEYS.get(key, key): value for key, value in result.items()}
        ordered = {key: record[key] for key in _KEY_ORDER if key in record}
        ordered.update({k: v for k, v in record.items() if k not in ordered})  # anything unexpected
        renamed.append(ordered)
    return renamed


def _migrate_result_keys(cur: Any) -> None:
    """Rewrite reports saved before ``test_name``/``section`` were renamed."""
    cur.execute("SELECT id, results FROM lab_reports WHERE results LIKE %s", ('%"test_name"%',))
    rows = cur.fetchall()
    for row in rows:
        results = rename_result_keys(_load(row["results"], []))
        cur.execute("UPDATE lab_reports SET results = %s WHERE id = %s", (_dump(results), row["id"]))
    if rows:
        logger.info("renamed test_name/section to name/category in %d report(s)", len(rows))


def _has_table(cur: Any, name: str) -> bool:
    cur.execute("SELECT COUNT(*) AS n FROM information_schema.TABLES"
                " WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (name,))
    return bool(cur.fetchone()["n"])


def _column_names(cur: Any, table: str) -> set[str]:
    cur.execute("SELECT COLUMN_NAME AS c FROM information_schema.COLUMNS"
                " WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s", (table,))
    return {row["c"] for row in cur.fetchall()}


def _migrate_results_into_json(cur: Any) -> None:
    """Fold a pre-JSON ``lab_results`` table into ``lab_reports.results``.

    Results used to be one row per test in a child table. They are now stored
    as the JSON the API returned. This runs inside the schema check, so a
    database created before the change upgrades itself on first use; on a
    fresh database it does nothing.
    """
    columns = _column_names(cur, "lab_reports")
    if "results" not in columns:  # added NULL, made NOT NULL once backfilled
        cur.execute("ALTER TABLE lab_reports ADD COLUMN results JSON NULL AFTER needs_review_count")
    if "review" not in columns:
        cur.execute("ALTER TABLE lab_reports ADD COLUMN review JSON NULL AFTER results")
    if not _has_table(cur, "lab_results"):
        return

    # Only reports that have not been folded in yet, so re-running this (when
    # the DROP below was refused) never overwrites the JSON with stale rows.
    cur.execute(f"SELECT report_id, {_LEGACY_COLUMNS} FROM lab_results"
                " WHERE report_id IN (SELECT id FROM lab_reports WHERE results IS NULL)"
                " ORDER BY report_id, position")
    by_report: dict[int, list[dict[str, Any]]] = {}
    for row in cur.fetchall():
        by_report.setdefault(row["report_id"], []).append(row)

    for report_id, rows in by_report.items():
        results = [_legacy_result_json(row) for row in rows]
        review = [
            {"confidence": float(row["ocr_confidence"]) if row["ocr_confidence"] is not None else None,
             "notes": _load(row["review_notes"], []),
             "needs_review": bool(row["needs_review"])}
            for row in rows
        ]
        cur.execute("UPDATE lab_reports SET results = %s, review = %s WHERE id = %s",
                    (_dump(results), _dump(review), report_id))

    cur.execute("UPDATE lab_reports SET results = '[]' WHERE results IS NULL")
    cur.execute("ALTER TABLE lab_reports MODIFY results JSON NOT NULL")
    logger.info("migrated %d report(s) from lab_results into lab_reports.results", len(by_report))

    # The service runs as a least-privilege user that may not hold DROP. The
    # data is already copied at this point, so a refusal here is not fatal --
    # the leftover table is simply no longer read or written.
    try:
        cur.execute("DROP TABLE lab_results")
    except pymysql.MySQLError as exc:
        logger.warning("lab_results is migrated but could not be dropped (%s); "
                       "drop it by hand: DROP TABLE lab_results;", exc.args[-1] if exc.args else exc)


# --------------------------------------------------------------------------
# Queries
# --------------------------------------------------------------------------

def save_report(*, report: dict[str, Any], results: list[dict[str, Any]], review: list[dict[str, Any]],
                filename: str | None, engine: str, replace: bool = False) -> dict[str, Any]:
    """Insert one report row holding its results as JSON.

    Raises DuplicateReport when patient_code + sample_no is already saved and
    ``replace`` is false; with ``replace`` the old report is deleted first.
    """
    needs_review = count_needs_review(review)
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
            " engine, results_count, needs_review_count, results, review)"
            " VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (patient, sample, _mysql_datetime(report.get("collected_at")), _mysql_datetime(report.get("received_at")),
             filename, engine, len(results), needs_review, _dump(results), _dump(review) if review else None),
        )
        report_id = cur.lastrowid
    return {"id": report_id, "results_saved": len(results), "needs_review": needs_review, "replaced": replaced}


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
    return {
        **_report_json(report),
        "results": _load(report["results"], []),
        "review": _load(report["review"], []),
    }


_HEADER_COLUMNS = ("id, patient_code, sample_no, collected_at, received_at, source_filename, engine, "
                   "results_count, needs_review_count, created_at")


def list_reports(limit: int = 20) -> list[dict[str, Any]]:
    """Report headers only -- the results JSON is not read."""
    with connect() as conn, conn.cursor() as cur:
        cur.execute(f"SELECT {_HEADER_COLUMNS} FROM lab_reports ORDER BY id DESC LIMIT %s", (limit,))
        return [_report_json(row) for row in cur.fetchall()]
