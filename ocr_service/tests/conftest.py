import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make the `ocr_service` package importable however pytest is invoked
# (from the repo root, or from inside ocr_service/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ocr_service import engines, surya_engine  # noqa: E402


# --------------------------------------------------------------------------
# Fake Surya predictions
#
# Surya is the only engine, so every test that goes through /ocr needs one.
# Feeding canned predictions in at the predictor keeps the whole request path
# under test -- routing, validation, mapping, response shape -- with no GPU
# and no 1.5 GB model download.
# --------------------------------------------------------------------------


def fake_char(text: str, x0: float, x1: float, conf: float = 0.9, valid: bool = True,
              y0: float = 10, y1: float = 30) -> SimpleNamespace:
    return SimpleNamespace(text=text, bbox=[x0, y0, x1, y1], confidence=conf, bbox_valid=valid)


def fake_line(text: str, bbox: list[float], chars: list | None = None) -> SimpleNamespace:
    return SimpleNamespace(text=text, bbox=bbox, chars=chars or [])


def fake_chars_for(text: str, x: float, y0: float, y1: float, conf: float = 0.9) -> list:
    """One fake character per letter, 10 px wide, starting at ``x``."""
    return [fake_char(ch, x + 10 * i, x + 10 * i + 9, conf, y0=y0, y1=y1) for i, ch in enumerate(text)]


@pytest.fixture(scope="session", autouse=True)
def never_load_real_models() -> None:
    """Stop app startup from loading Surya for real.

    Surya is the only engine now, so *any* test that starts the app would
    otherwise kick off a background load -- and a 1.5 GB download on a machine
    that has never run it. Session-scoped so it is in place before the
    module-scoped client fixtures build their apps. The opt-in real-model test
    calls ``surya_engine`` directly and is unaffected.
    """
    original = engines.warm_up
    engines.warm_up = lambda: None
    yield
    engines.warm_up = original


@pytest.fixture
def fake_surya(monkeypatch: pytest.MonkeyPatch) -> list:
    """Route /ocr through Surya with canned predictions.

    Returns the list the predictor appends its calls to, so a test can assert
    on what the engine was handed.
    """
    calls: list = []
    lines = [
        fake_line("Hemoglobin", [40, 40, 140, 62], fake_chars_for("Hemoglobin", 40, 40, 62, 0.95)),
        fake_line("9.7", [300, 41, 330, 63], fake_chars_for("9.7", 300, 41, 63, 0.5)),
    ]

    def recognition(images, **kwargs):
        calls.append((images, kwargs))
        return [SimpleNamespace(text_lines=lines, image_bbox=[0, 0, *images[0].size])]

    monkeypatch.setattr(engines, "warm_up", lambda: None)  # no background load at startup
    monkeypatch.setattr(surya_engine, "_load", lambda: (recognition, object()))
    monkeypatch.setattr(surya_engine, "_state", "ready")
    return calls
