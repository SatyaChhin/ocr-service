"""Tests for the OCR service.

Everything here runs without a Tesseract binary except the two tests at the
bottom, which skip themselves when the engine is missing. The request path is
still covered on such machines via a stubbed engine, so the suite is useful in
CI images that have not installed the system packages yet.
"""

from __future__ import annotations

import tempfile

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw, ImageFont

from ocr_service import ocr
from ocr_service.main import MAX_UPLOAD_BYTES, app, sniff_media_type

TESSERACT_AVAILABLE = ocr.engine_info()["available"]
requires_tesseract = pytest.mark.skipif(
    not TESSERACT_AVAILABLE, reason="Tesseract binary is not installed"
)


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
    assert "available" in body["tesseract"]
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
# Preprocessing
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


def test_preprocess_returns_a_binary_page_with_geometry() -> None:
    binary, meta = ocr.preprocess(cv2.cvtColor(_text_page(), cv2.COLOR_GRAY2BGR))

    assert binary.ndim == 2
    assert set(np.unique(binary)).issubset({0, 255})
    assert meta["width"] > 0 and meta["height"] > 0
    assert meta["scale"] >= 1.0  # a 1200px-wide page is upscaled toward 1600
    assert abs(meta["skew_deg"]) <= ocr.MAX_DESKEW_DEG


def test_flatten_illumination_removes_a_lighting_gradient() -> None:
    """The mechanism that lets a global threshold cope with uneven lighting."""
    page = np.full((900, 700), 220, dtype=np.uint8)
    yy, xx = np.mgrid[0:900, 0:700]
    shadowed = np.clip(page * (1.0 - 0.6 * (xx / 700)), 0, 255).astype(np.uint8)

    # Background brightness varies hugely across the shadowed page...
    left, right = shadowed[:, :100].mean(), shadowed[:, -100:].mean()
    assert left - right > 100

    flattened = ocr._flatten_illumination(shadowed)

    # ...and barely at all once the illumination field is divided out.
    assert abs(flattened[:, :100].mean() - flattened[:, -100:].mean()) < 5


def test_strip_rules_erases_borders_but_keeps_glyphs() -> None:
    page = np.full((900, 1400), 255, dtype=np.uint8)
    cv2.putText(page, "Metformin", (120, 300), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
    ink_before = int((page < 128).sum())
    cv2.line(page, (60, 200), (1340, 200), 0, 5)  # horizontal rule
    cv2.line(page, (60, 200), (60, 700), 0, 5)  # vertical rule

    stripped = ocr._strip_rules(page)

    # Both rules gone...
    assert (stripped[195:210, 600:1300] > 200).all()
    assert (stripped[300:600, 55:70] > 200).all()
    # ...and the glyphs essentially untouched.
    assert int((stripped < 128).sum()) >= ink_before * 0.9


# A real typeface is needed: cv2.putText draws a Hershey *stroke* font, which
# Tesseract's LSTM reads poorly at cell-sized text regardless of preprocessing.
_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "LiberationSans-Regular.ttf",
    "Arial.ttf",
    "arial.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
)


def _truetype(size: int):
    for path in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return None


def _form_with_ruled_table() -> np.ndarray | None:
    """A form page whose medication table is fully bordered."""
    heading, body, cell = _truetype(58), _truetype(40), _truetype(36)
    if not all((heading, body, cell)):
        return None

    image = Image.new("L", (1700, 2200), 255)
    draw = ImageDraw.Draw(image)
    draw.text((110, 100), "PATIENT MEDICATION RECORD", font=heading, fill=0)
    draw.text((110, 230), "Surname : TREMBLAY", font=body, fill=0)
    draw.text((110, 300), "Date of birth : 12/03/1978", font=body, fill=0)
    draw.text((110, 430), "CURRENT MEDICATIONS", font=body, fill=0)

    rows = [
        ["DRUG", "DOSE", "FREQUENCY"],
        ["Metformin", "500 mg", "twice daily"],
        ["Lisinopril", "10 mg", "once daily"],
        ["Atorvastatin", "20 mg", "at bedtime"],
    ]
    xs = [110, 700, 1050, 1580]
    top, row_height = 520, 110
    for index, row in enumerate(rows):
        y = top + index * row_height
        draw.line([(110, y), (1580, y)], fill=0, width=5)
        for column, text in enumerate(row):
            draw.text((xs[column] + 25, y + 32), text, font=cell, fill=0)
    bottom = top + len(rows) * row_height
    draw.line([(110, bottom), (1580, bottom)], fill=0, width=5)
    for x in xs:
        draw.line([(x, top), (x, bottom)], fill=0, width=5)

    draw.text((110, 1100), "Reviewed by the attending physician on 14 August 2026.",
              font=body, fill=0)
    return np.asarray(image)


@requires_tesseract
def test_ruled_table_cells_are_not_dropped(client: TestClient) -> None:
    """Regression: a fully-bordered table used to read as nothing at all.

    Two independent causes, both silent -- overall confidence stayed high
    because the words that *were* read were read well:

    1. cv2.adaptiveThreshold thickened strokes next to the heavy rules until
       characters merged.
    2. Tesseract's layout analysis classified the bordered table as a
       non-text region and skipped every cell, in every psm mode.

    Token recall on this page: 3/10 before, 10/10 now.
    """
    page = _form_with_ruled_table()
    if page is None:
        pytest.skip("no TrueType font available to render the fixture")

    response = client.post(
        "/ocr", files={"file": ("record.png", _png_bytes(page), "image/png")}
    )

    assert response.status_code == 200
    text = response.json()["text"]
    for expected in ("PATIENT MEDICATION RECORD", "TREMBLAY", "Metformin", "Lisinopril",
                     "Atorvastatin", "500 mg", "10 mg", "20 mg", "bedtime"):
        assert expected in text, f"{expected!r} missing from:\n{text}"


def test_reassemble_keeps_lines_and_paragraphs_apart() -> None:
    words = [
        {"text": "Nom", "block": 1, "par": 1, "line": 1, "word": 1},
        {"text": "Dupont", "block": 1, "par": 1, "line": 1, "word": 2},
        {"text": "Date", "block": 1, "par": 1, "line": 2, "word": 1},
        {"text": "Diagnosis", "block": 2, "par": 1, "line": 1, "word": 1},
    ]

    assert ocr._reassemble(words) == "Nom Dupont\nDate\n\nDiagnosis"


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
# End-to-end with a stubbed engine
#
# Covers the whole request path -- validation, threadpool hop, response model
# -- on CI images that have no Tesseract binary.
# --------------------------------------------------------------------------


_FAKE_TSV = {
    "text": ["", "Nom", "Dupont", "Date", ""],
    "conf": ["-1", "94.0", "88.5", "91.0", "-1"],
    "block_num": [1, 1, 1, 1, 1],
    "par_num": [1, 1, 1, 1, 1],
    "line_num": [1, 1, 1, 2, 2],
    "word_num": [0, 1, 2, 1, 0],
    "left": [0, 212, 320, 212, 0],
    "top": [0, 480, 480, 540, 0],
    "width": [0, 96, 140, 88, 0],
    "height": [0, 34, 34, 34, 0],
}


@pytest.fixture
def stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ocr,
        "engine_info",
        # The three languages this deployment ships (see ocr.EXPECTED_LANGS),
        # so the default lang spec validates against the stub.
        lambda: {"available": True, "version": "5.3.4", "languages": ["eng", "fra", "khm"], "error": None},
    )
    monkeypatch.setattr(ocr.pytesseract, "image_to_data", lambda *a, **kw: dict(_FAKE_TSV))


def test_ocr_returns_pages_and_word_boxes(client: TestClient, stub_engine: None) -> None:
    response = client.post(
        "/ocr",
        params={"detail": "true"},
        files={"file": ("scan.png", _png_bytes(_text_page()), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image/png"
    assert body["lang"] == ocr.DEFAULT_LANG  # echoed back, whatever the default is
    assert body["page_count"] == 1
    assert body["text"] == "Nom Dupont\nDate"

    page = body["pages"][0]
    assert page["page"] == 1
    assert page["word_count"] == 3
    assert page["mean_confidence"] == pytest.approx(91.17, abs=0.01)
    assert [word["text"] for word in page["words"]] == ["Nom", "Dupont", "Date"]
    assert page["words"][0]["bbox"] == [212, 480, 96, 34]
    # Geometry needed to map boxes back onto the original upload.
    assert page["scale"] > 0 and abs(page["skew_deg"]) <= ocr.MAX_DESKEW_DEG


def test_ocr_rejects_a_language_that_is_not_installed(
    client: TestClient, stub_engine: None
) -> None:
    response = client.post(
        "/ocr",
        params={"lang": "deu"},
        files={"file": ("scan.png", _png_bytes(_text_page()), "image/png")},
    )

    assert response.status_code == 400
    assert "deu" in response.json()["detail"]


def test_large_upload_is_not_spooled_to_disk(
    client: TestClient, stub_engine: None, tmp_path, monkeypatch: pytest.MonkeyPatch
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

    response = client.post("/ocr", files={"file": ("big.png", payload, "image/png")})

    assert response.status_code == 200
    assert list(tmp_path.iterdir()) == []


def test_error_responses_carry_a_request_id_and_no_traceback(client: TestClient) -> None:
    response = client.post("/ocr", files={"file": ("notes.txt", b"plain text", "text/plain")})

    body = response.json()
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert "Traceback" not in response.text


# --------------------------------------------------------------------------
# End-to-end (needs the system packages)
# --------------------------------------------------------------------------


@requires_tesseract
def test_ocr_extracts_text_from_an_image(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        params={"detail": "true"},
        files={"file": ("scan.png", _png_bytes(_text_page("HELLO WORLD")), "image/png")},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image/png"
    assert body["page_count"] == 1
    assert "HELLO" in body["text"].upper()

    page = body["pages"][0]
    assert page["page"] == 1
    assert page["word_count"] >= 1
    assert page["words"] and len(page["words"][0]["bbox"]) == 4


@requires_tesseract
def test_ocr_omits_word_boxes_unless_requested(client: TestClient) -> None:
    response = client.post(
        "/ocr",
        files={"file": ("scan.png", _png_bytes(_text_page()), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["pages"][0]["words"] is None
