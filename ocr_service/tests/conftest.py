import os
import sys
from pathlib import Path

# Make the `ocr_service` package importable however pytest is invoked
# (from the repo root, or from inside ocr_service/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# The suite exercises the request path through a stubbed Tesseract, so it
# runs without a GPU or model download. Surya tests switch the engine
# themselves (see test_surya.py). Must be set before ocr_service is imported.
os.environ.setdefault("OCR_ENGINE", "tesseract")
