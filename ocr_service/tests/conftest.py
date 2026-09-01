import sys
from pathlib import Path

# Make the `ocr_service` package importable however pytest is invoked
# (from the repo root, or from inside ocr_service/).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
