"""Tests for the OCR service.

Everything here runs without a GPU and without the Surya models: the request
path goes through the fake_surya fixture in conftest, which feeds canned
predictions in at the predictor. The whole suite is therefore useful on a CI
image that has installed nothing but the Python dependencies.
"""

from __future__ import annotations

import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from ocr_service import ocr
from ocr_service.main import MAX_UPLOAD_BYTES, app, sniff_media_type


@pytest.fixture(scope="module")
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def _png_bytes(image: np.ndarray) -> bytes:
    ok, buffer = cv2.imencode(".png", image)
    assert ok
    return buffer.tobytes()


def _text_page(text: str = "HELLO WORLD") -> np.ndarray:
    page = np.full((320, 1200), 255, dtype=np.uint8)
    cv2.putText(page, text, (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 3.0, 0, 6, cv2.LINE_AA)
    return page


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def test_health_returns_200_with_dependency_status(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    # 200 is liveness; `ready` is the readiness signal, so this passes whether
    # or not the system packages are installed.
    assert body["status"] in {"ok", "degraded"}
    assert isinstance(body["ready"], bool)
    assert body["engine"]["name"] == "surya"
    assert "backend" in body["pdf"]
    assert body["limits"]["max_upload_bytes"] == MAX_UPLOAD_BYTES


# --------------------------------------------------------------------------
# /ocr input validation
# --------------------------------------------------------------------------


def test_ocr_rejects_unsupported_file_type(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("notes.txt", b"just some plain text", "text/plain")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_ocr_rejects_a_file_whose_bytes_contradict_its_name(client: TestClient) -> None:
    # An attacker-friendly upload: image extension and content type, but the
    # bytes are not an image. Sniffing, not the filename, must decide.
    response = client.post(
        "/ocr",
        files={"file": ("scan.png", b"<html>not a png</html>", "image/png")},
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_ocr_rejects_an_empty_file(client: TestClient) -> None:
    response = client.post("/ocr", files={"file": ("empty.png", b"", "image/png")})

    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ocr_requires_a_file(client: TestClient) -> None:
    response = client.post("/ocr")

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Request validation failed"
    # The generic handler must not echo submitted values back to the client.
    assert all("input" not in problem for problem in body["problems"])


def test_ocr_rejects_a_malformed_lang(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        params={"lang": "not-a-language"},
        files={"file": ("scan.png", _png_bytes(_text_page()), "image/png")},
    )

    assert response.status_code == 400
    assert "lang" in response.json()["detail"]


def test_ocr_rejects_an_oversized_upload(client: TestClient) -> None:
    oversized = b"\x89PNG\r\n\x1a\n" + b"\0" * (MAX_UPLOAD_BYTES + 1024)

    response = client.post("/ocr", files={"file": ("huge.png", oversized, "image/png")})

    assert response.status_code == 413
    assert "limit" in response.json()["detail"].lower()


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        (b"%PDF-1.7\n%...", "application/pdf"),
        (b"\x89PNG\r\n\x1a\n" + b"\0" * 16, "image/png"),
        (b"\xff\xd8\xff\xe0" + b"\0" * 16, "image/jpeg"),
        (b"RIFF\x00\x00\x00\x00WEBPVP8 ", "image/webp"),
    ],
)
def test_sniff_media_type_recognises_supported_formats(payload: bytes, expected: str) -> None:
    assert sniff_media_type(payload) == expected


# --------------------------------------------------------------------------
# Deskew (shared: Surya straightens tilted pages with this)
# --------------------------------------------------------------------------


def test_estimate_skew_recovers_the_correcting_rotation() -> None:
    page = np.full((900, 900), 255, dtype=np.uint8)
    for top in range(80, 820, 60):
        cv2.rectangle(page, (120, top), (780, top + 18), 0, -1)

    tilt = 4.0
    matrix = cv2.getRotationMatrix2D((450.0, 450.0), tilt, 1.0)
    tilted = cv2.warpAffine(
        page, matrix, (900, 900), flags=cv2.INTER_CUBIC, borderValue=255
    )

    # The correction is the opposite of the tilt that was applied.
    assert ocr.estimate_skew(tilted) == pytest.approx(-tilt, abs=1.0)


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------

KHMER_TEXT = "នេះជាការសាកល្បងអត្ថបទភាសាខ្មែរសម្រាប់ការធ្វើតេស្តនៃប្រព័ន្ធអានអក្សរ។"
ENGLISH_TEXT = "This is an English invoice with a total amount payable of 1250.00 US dollars."


def test_detect_languages_identifies_khmer_by_script() -> None:
    """langdetect has no Khmer profile, so this must come from the block check."""
    guesses = ocr.detect_languages(KHMER_TEXT)

    assert [g["lang"] for g in guesses] == ["khm"]
    assert guesses[0]["iso639_1"] == "km"
    assert guesses[0]["confidence"] > 0.9


def test_detect_languages_reports_both_halves_of_a_bilingual_page() -> None:
    """The whole point of stripping Khmer before langdetect: on mixed text it
    raises "No features in text" and would otherwise lose English too."""
    guesses = ocr.detect_languages(f"{ENGLISH_TEXT} {KHMER_TEXT}")
    found = {g["lang"]: g["confidence"] for g in guesses}

    assert "khm" in found and "eng" in found
    assert 0.2 < found["khm"] < 0.8, found  # roughly the Khmer share of the page
    assert sum(found.values()) == pytest.approx(1.0, abs=0.05)


def test_detect_languages_still_handles_latin_only_text() -> None:
    guesses = ocr.detect_languages(ENGLISH_TEXT)

    assert guesses[0]["lang"] == "eng"
    assert guesses[0]["confidence"] > 0.9


def test_detect_languages_ignores_a_stray_khmer_glyph() -> None:
    """A single character out of a stamp or logo is not a Khmer document."""
    assert all(g["lang"] != "khm" for g in ocr.detect_languages(ENGLISH_TEXT + " ក"))


def test_detect_languages_returns_empty_for_no_text() -> None:
    assert ocr.detect_languages("") == []
    assert ocr.detect_languages("   \n  ") == []


# --------------------------------------------------------------------------
# End-to-end with canned predictions
#
# Covers the whole request path -- validation, threadpool hop, response model
# -- without a GPU or the Surya models.
# --------------------------------------------------------------------------


def test_ocr_returns_pages_and_word_boxes(fake_surya: list) -> None:
    with TestClient(app) as client:
        response = client.post(
            "/ocr",
            params={"detail": "true"},
            files={"file": ("scan.png", _png_bytes(_text_page()), "image/png")},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image/png"
    assert body["engine"] == "surya"
    assert body["lang"] == ocr.DEFAULT_LANG  # echoed back, whatever the default is
    assert body["page_count"] == 1
    assert body["text"] == "Hemoglobin 9.7"

    page = body["pages"][0]
    assert page["page"] == 1
    assert page["word_count"] == 2
    assert [word["text"] for word in page["words"]] == ["Hemoglobin", "9.7"]
    # Geometry needed to map boxes back onto the original upload.
    assert page["scale"] > 0 and abs(page["skew_deg"]) <= ocr.MAX_DESKEW_DEG


def test_large_upload_is_not_spooled_to_disk(
    fake_surya: list, tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Requirement: documents never touch the filesystem.

    Starlette spools multipart parts over 1 MB to a temp file by default;
    main.py raises that threshold above the upload cap. Point tempfile at an
    empty directory and assert nothing lands in it.
    """
    from starlette.formparsers import MultiPartParser

    assert MultiPartParser.spool_max_size > MAX_UPLOAD_BYTES

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    rng = np.random.default_rng(0)
    noisy = rng.integers(0, 256, size=(1500, 1500), dtype=np.uint8)
    payload = _png_bytes(noisy)
    assert len(payload) > 1024 * 1024, "payload must exceed the default spool threshold"

    with TestClient(app) as client:
        response = client.post("/ocr", files={"file": ("big.png", payload, "image/png")})

    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_error_responses_carry_a_request_id_and_no_traceback(client: TestClient) -> None:
    response = client.post("/ocr", files={"file": ("notes.txt", b"plain text", "text/plain")})

    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "Traceback" not in response.text
