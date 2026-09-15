# OCR Service

FastAPI service that extracts text from scanned forms, photographed documents
and PDFs. Two engines sit behind the same API:

| `OCR_ENGINE` | Engine | Notes |
| --- | --- | --- |
| `surya` *(default)* | [Surya](https://github.com/datalab-to/surya) 0.17.1, in-process PyTorch on the GPU | Reads every script it knows at once, Khmer included — no language hint. |
| `tesseract` | Tesseract 5 with `eng`/`fra`/`khm` data | CPU. Needs `lang` (e.g. `eng+khm`). Kept as a fallback and for comparison. |

Built for medical-style forms: OCR output keeps a label and its value on one
line, and `?detail=true` returns per-word bounding boxes and confidences you
can build key-value or table extraction on top of. Responses have the same
shape whichever engine ran; the `engine` field says which one did.

### Surya vs Tesseract on this service's test pages

| | Surya (RTX 5080 Laptop) | Tesseract |
| --- | --- | --- |
| Clean 900×200 PNG, "Hello from the OCR service / Invoice total: 1234.56" | exact, ~0.9 s | first letter of each line lost (upscaling in the preprocessing), ~0.8 s |
| Synthetic A4 lab report (English, 104 words) | exact rows, 93% mean conf, ~2.6 s | `2F/55` for `F / 55`, `x10412/L` for `x10^12/L` |
| Bilingual Khmer/English form | 94% conf; 3 Khmer character errors, incl. age `៥៥` → `៥៤` | 61% conf; bold Khmer headings read as Latin gibberish |

Surya is far better on these pages but not error-free — it misread a Khmer
digit in a patient's age. Treat the Low-confidence review as a hint, not a
guarantee: that digit was read with high confidence.

**Confidence is not comparable across engines.** Surya's is the model's
token probability (0–100 here); it scores **bold** text noticeably lower even
when it is read correctly.

### Licensing (Surya)

The Surya code is GPL-3.0. The model weights use a modified AI Pubs Open
Rail-M licence: free for research, personal use, and organisations under
$2M in funding/revenue; broader commercial use needs a licence from
[Datalab](https://www.datalab.to/pricing). Check this before deploying
commercially. Tesseract (Apache-2.0) has no such restriction.

---

## System dependencies

These are **not** installable with `pip` and must be present on the host or in
the container image.

| Dependency | Why | Check |
| --- | --- | --- |
| NVIDIA GPU + driver | Surya on CUDA (it falls back to CPU, which is far slower) | `nvidia-smi` |
| `poppler-utils` | `pdfinfo` / `pdftoppm`, used to rasterise PDF pages | `pdftoppm -v` |
| `tesseract-ocr` | fallback engine that `pytesseract` shells out to | `tesseract --version` |
| `tesseract-ocr-eng` / `-fra` / `-khm` | language data for the fallback engine | `tesseract --list-langs` |

Tesseract is optional while `OCR_ENGINE=surya`; `/health` reports its status
either way.

**Debian / Ubuntu**

```bash
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra \
                        tesseract-ocr-khm poppler-utils
```

**macOS (Homebrew)** — the `tesseract-lang` formula carries all language data:

```bash
brew install tesseract tesseract-lang poppler
```

**Windows**

1. Install the [UB Mannheim Tesseract build](https://github.com/UB-Mannheim/tesseract/wiki),
   ticking **French** and **Khmer** under "Additional language data" during setup.
2. Download the [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
   release and add its `Library\bin` directory to `PATH`.
3. If either is not on `PATH`, point the service at it directly:
   ```
   set TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
   set OCR_POPPLER_PATH=C:\poppler\Library\bin
   ```

**Docker**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        tesseract-ocr tesseract-ocr-eng tesseract-ocr-fra tesseract-ocr-khm \
        poppler-utils \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . /app/ocr_service
WORKDIR /app
CMD ["uvicorn", "ocr_service.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

`GET /health` reports whether each of these was found, so check it first when
something fails.

---

## Install and run

Python **3.11–3.13**. (3.14 is not recommended yet: OpenCV publishes no wheels
for it at the pinned version and pip will try to build from source.)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# PyTorch first, from the CUDA index -- PyPI's Windows wheels are CPU-only.
# cu128 covers RTX 50-series (Blackwell, sm_120).
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# from the directory *containing* ocr_service/
uvicorn ocr_service.main:app --reload --port 8000
```

Interactive docs: <http://localhost:8000/docs>

**Pins worth knowing about** (see `requirements.txt`):

* `surya-ocr==0.17.1` — the last in-process release. 0.20+ ("Surya 2") runs a
  vision-language model behind a separate vLLM (Docker) or llama.cpp server and
  returns text per layout *block*, without word or line boxes.
* `transformers==4.57.6` — Surya 0.17.1 declares `>=4.56.1` but fails on 5.x
  (`SuryaDecoderConfig has no attribute pad_token_id`).
* `requests` — imported by Surya's model downloader but not declared.
* `pillow==10.4.0`, `opencv-python-headless==4.11.0.86` — pinned by Surya.

**First start downloads the models** (~1.5 GB) from `models.datalab.to` into
`%LOCALAPPDATA%\datalab\datalab\Cache\models` (`MODEL_CACHE_DIR` overrides).
After that, startup loads them onto the GPU in ~5–15 s, in a background
thread: `/health` answers immediately with `engine.state: "loading"` and
`ready: false`, and an `/ocr` request that arrives early waits for the load
instead of failing.

**On this machine**, Tesseract is installed but not on `PATH`, and its `fra` /
`khm` language data lives in a project-local `tessdata/` directory because
`C:\Program Files\Tesseract-OCR\tessdata` is not writable without admin.
`run.ps1` sets `TESSERACT_CMD` and `TESSDATA_PREFIX` for you:

```powershell
.\run.ps1                          # http://localhost:8000, Surya
.\run.ps1 -Port 8080 -Reload
.\run.ps1 -Engine tesseract        # the fallback engine
```

Check `GET /health` first — `"ready": true` means the active engine is loaded
(for Tesseract: found, with all expected languages).

---

## API

### `POST /ocr`

`multipart/form-data` with a single `file` part.

| Query param | Default | Notes |
| --- | --- | --- |
| `lang` | `eng+fra` | Tesseract: any `+`-joined codes (`eng`, `fra`, `khm`, `eng+khm`, ...), validated against installed data. Surya: syntax-checked and echoed back, otherwise ignored. |
| `detail` | `false` | Include per-word boxes and confidences. |
| `dpi` | `300` | PDF rasterisation DPI, 72–600. Ignored for images. |
| `lab` | `false` | Also extract structured lab results into `lab` — see [Lab results](#lab-results). |

```bash
curl -F "file=@form.pdf" "http://localhost:8000/ocr?lang=eng+fra&detail=true"
```

Response:

```jsonc
{
  "filename": "form.pdf",
  "media_type": "application/pdf",
  "engine": "surya",                   // or "tesseract"
  "lang": "eng+fra",
  "page_count": 2,
  "languages": [                       // document-level, averaged over pages
    { "lang": "fra", "iso639_1": "fr", "confidence": 0.87 },
    { "lang": "eng", "iso639_1": "en", "confidence": 0.13 }
  ],
  "text": "page one text\fpage two text",   // pages joined with \f (form feed)
  "pages": [
    {
      "page": 1,
      "text": "Nom Dupont\nDate de naissance 12/03/1978",
      "word_count": 42,
      "mean_confidence": 87.4,
      "languages": [{ "lang": "fra", "iso639_1": "fr", "confidence": 0.99 }],
      "width": 2480, "height": 3508,
      "scale": 1.0,                    // resize applied before OCR
      "skew_deg": -1.4,                // deskew rotation applied
      "preprocess_ms": 412.6,
      "ocr_ms": 1893.2,
      "words": [                       // only when detail=true, else null
        { "text": "Nom", "conf": 94.0, "bbox": [212, 480, 96, 34],
          "block": 1, "par": 1, "line": 3, "word": 1 }
      ]
    }
  ],
  "duration_ms": 2410.8
}
```

**Word box coordinate space.** Boxes are in *preprocessed*-image space, not
original-upload space. Use the page's `scale` and `skew_deg` to map back:
divide by `scale`, then rotate by `-skew_deg` about the image centre. With
Surya, `scale` is 1 unless the page exceeded `OCR_SURYA_MAX_PIXELS`, and
`skew_deg` is non-zero only for pages tilted by at least
`OCR_SURYA_DESKEW_MIN_DEG` (0.3°) — typically phone photos.

**How Surya results are mapped.** Surya returns detected lines with
per-character boxes. The service rebuilds words from the characters (word
confidence = mean of its characters; Surya's own word builder uses only the
first), strips the model's inline formatting tags (`<b>`, `<sup>`, `<math>`,
...), and groups lines whose vertical extents overlap by half into one row
read left to right — Surya detects a form label and its value as separate
lines, and its own sort can put the value first. For Surya words, `line` is
that row number and `block`/`par` are always 1.

**Language detection is best effort.** Tesseract will not tell you which of
`eng+khm` it matched, so detection runs over the extracted text instead, from
two signals:

* **Khmer by script.** `langdetect` ships 55 profiles and Khmer is not one of
  them — worse, it raises `No features in text` on Khmer, *including on a
  bilingual page*, which would lose the English half too. Khmer owns the
  `U+1780–17FF` block outright, so the script itself is the identification.
  Anything below `OCR_MIN_KHMER_RATIO` (default 10%) of the page's letters is
  treated as a stray glyph from a stamp or logo rather than Khmer text.
* **Everything else by `langdetect`**, run over the text with Khmer stripped
  out, its confidence scaled by the non-Khmer share of the page.

Confidences are therefore the share of the page each language accounts for: a
half-Khmer form reports roughly `khm 0.5 / eng 0.5`. Still unreliable on short
or mixed form text — the Latin half needs 40 characters before `langdetect`
will look at it. Treat `confidence` as a hint, not a verdict.

### Lab results

`POST /ocr?lab=true` adds a `lab` object built from the report's results
tables (`lab_results.py`). `lab.results` is the **database-ready array**, in
exactly the format the lab system inserts:

```jsonc
{
  "lab": {
    "results": [
      { "test_name": "RBC", "value": 5.31, "flag": "H", "unit": "x10^12/L",
        "ref_range": "3.8 - 4.8", "section": "COMPLETE BLOOD COUNT" },
      { "test_name": "Neutrophils", "percent": 65.7, "value": 7.15, "flag": "H",   // percent: differential rows only
        "unit": "x10^9/L", "ref_range": "2 - 7", "section": "Differential White Cell Count" },
      { "test_name": "Blood Group", "value": "O Rh (D): Positive", "flag": null,
        "unit": null, "ref_range": null, "section": "COMPLETE BLOOD COUNT" }
    ],
    "review": [                         // review[i] describes results[i]
      { "page": 1, "confidence": 100.0, "source": "RBC 5.31 H x1012/L 3.8 - 4.8",
        "cross_checked": ["MCV = Hct / RBC × 10", "MCH = Hb / RBC × 10"],
        "notes": [], "needs_review": false }
    ],
    "unparsed": [ { "page": 1, "text": "..." } ]   // table rows with numbers that did not parse
  }
}
```

On the lab's two-page CBC / coagulation / biochemistry report this reproduces
the reference JSON exactly — 22 records, same values, number types, key order
and sections. Insert `results[i]` only when `review[i].needs_review` is false,
or after a person has checked it against the document.

**How rows are read.** Word boxes are regrouped into visual rows (both
engines), and only rows between a `Test Name` header and the next department
banner (`HEAMATOLOGY`, `BIOCHEMISTRY`, ...) or the report footer count, so
patient details never become results. Dot leaders are dropped, units are
normalised (`x10⁹/L`, OCR's flattened `x109/L` → `x10^9/L`), numbers are typed
(`0.60` → `0.6`, `744` → `744`), and `"65.7% 7.15"` splits into `percent` and
`value`. Surya pages tilted by 0.3° or more are straightened first: on a
phone photo a 1–2° tilt drops a row's unit and range a full line below its
name.

**Test catalog** (`lab_catalog.json`). Canonical `test_name`, `section` and
unit per test, plus aliases (`Haemoglobin`, `HGB`, `Neutrophils (%)`, ...).
Sections follow the catalog because the report's own headings do not map
one-to-one onto the database groups (AST/ALT sit under "Transaminase" on the
report but belong to LIVER FUNCTIONS). Tests not in the catalog are still
extracted, with the nearest heading as their section; add them to the catalog
to control their names. Deliberately **not** an alias: BUN for Urea (urea ≈
BUN × 2.14 — a different number).

**Review notes** — any note sets `needs_review`:

| Code | Meaning |
| --- | --- |
| `flag_not_set` / `flag_unexpected` / `flag_wrong_direction` | The H/L flag disagrees with the value and reference range. |
| `differential_mismatch` | Absolute count ≠ percentage × WBC. |
| `differential_sum` | The five differential percentages do not add up to 100 ± 1. |
| `index_mismatch` | MCV ≠ Hct/RBC×10, MCH ≠ Hb/RBC×10 or MCHC ≠ Hb/Hct×100 (the analyser computes these, so they agree to rounding unless a digit was misread). |
| `percent_sign_missing` | No `%` after a differential percentage; OCR reads `0.2%` as `0.296`. `params.suggested` gives the reading that fits WBC, when one does. Values are never corrected automatically. |
| `unit_missing` / `unit_differs` | Unit not read (filled from the catalog) or not the catalog's. |
| `name_fuzzy` | Name matched approximately (`Hemoglobln` → Hemoglobin). |
| `low_confidence` | The value was read below `OCR_LAB_MIN_CONFIDENCE` (80) **and** no cross-check confirmed it. |

Why the last one is waived for cross-checked values: Surya scores **bold**
print low even when it reads it correctly, and this lab prints every flagged
value in bold (32.3 → 54%, 60.8 → 50%). Values confirmed by a formula are
trustworthy whatever their confidence; a misread one fails the formula and
gets its own note. On the reference report that leaves one note — Platelets
744, bold, 60%, with nothing to confirm it.

Measured on a simulated phone photo of the same report (tilted 1.3°, uneven
light, blur, noise, JPEG 55): 20 of 22 records exact, and both errors flagged
— `x10°/L` for `x10⁹/L`, and Basophils `0.2%` read as `0.296` (with
`suggested: 0.2`). Tesseract reads only 13 of 22 from the clean PDF (it turns
dot leaders and flags into noise), so use Surya for lab extraction.

### `GET /health`

Always returns **200** so a liveness probe will not restart a running
process. Use the `ready` boolean for readiness — it is false while Surya's
models are loading, when the active engine failed to load, and (Tesseract)
when an expected language is missing (`eng`, `fra`, `khm` by default — see
`OCR_EXPECTED_LANGS`). `engine` describes the active engine (`name`, `state`,
`version`, `device`, `device_name`, `uses_language_hints`); `tesseract`
reports the fallback's status whichever engine is active. The body also
reports the PDF backend in use and the configured limits.

### Errors

Errors return `{"detail": "...", "request_id": "..."}` — never a stack trace.
`X-Request-ID` is set on every response; quote it when reporting a problem.

| Status | Cause |
| --- | --- |
| `400` | Unsupported/undecodable file, empty upload, bad `lang`, unreadable or over-long PDF |
| `413` | Upload exceeds 25 MB |
| `422` | Missing `file` part or out-of-range `dpi` |
| `500` | Unexpected failure (details logged, not returned) |
| `503` | OCR engine (Tesseract binary, or Surya models) or poppler unavailable |

File type is decided by **magic bytes**, not by the filename or the
client-supplied `Content-Type`.

---

## Tesseract preprocessing pipeline

Applied to every page before Tesseract OCR (`ocr.preprocess`). Surya gets the
page as-is (RGB, downscaled only past `OCR_SURYA_MAX_PIXELS`): it is a neural
model trained on raw scans, and binarisation would only remove information.

1. **Grayscale** — including a 4-channel alpha path.
2. **Rescale** — upscale toward a 1600px long side (Tesseract is trained near
   300 DPI and does poorly on small phone photos); downscale anything over 40
   MP so one page cannot exhaust memory.
3. **Denoise** — non-local-means below 4 MP, median blur above it. Non-local
   means costs seconds per full-resolution page, which is untenable across 50
   of them.
4. **Deskew** — candidate rotations are scored by the sharpness of the
   horizontal projection profile (coarse 1° sweep over ±12°, then a 0.2°
   refinement). This is slower than reading an angle off `cv2.minAreaRect` but
   avoids that function's sign ambiguity across OpenCV versions.
5. **Strip table rules** — long horizontal/vertical lines are painted white.
6. **Binarise** — illumination flattening followed by a global Otsu threshold.

Steps 5 and 6 are not the obvious choices, so here is why.

### Why not `cv2.adaptiveThreshold`

Adaptive thresholding sets each pixel's threshold from its local mean. Next to
a heavy black table rule that local mean is dragged down, so extra pixels flip
to black and nearby characters thicken until they merge. On a bilingual
medical form with a ruled medication table this cost **14 of 51 ground-truth
tokens — the entire table read as nothing** — while reported confidence stayed
above 90%, because the words that *were* read were read well.

Flattening the illumination (divide by a heavily-blurred copy of the page,
which captures the lighting field but not the glyphs) and then applying a
single global Otsu threshold keeps the lighting invariance adaptive
thresholding is chosen for, without touching stroke weight. Token recall on
the same page under three lighting conditions:

| method | mild gradient | heavy shadow | severe shadow | total |
| --- | --- | --- | --- | --- |
| **flatten + Otsu** (default) | 50/51 | 50/51 | 50/51 | **150/153** |
| `cv2.adaptiveThreshold` 31/15 | 37/51 | 37/51 | 39/51 | 113/153 |
| plain Otsu | 50/51 | 38/51 | 36/51 | 124/153 |

Plain Otsu collapses under a hard shadow, which is exactly why adaptive
thresholding gets recommended — the flattening step is what makes a global
threshold safe. Set `OCR_BINARISE=adaptive` to restore the literal behaviour.

The illumination field is estimated on a 512px copy and resized back. It is
low-frequency by construction, so this differs from the full-resolution blur
by a mean of 0.02/255, but blurring a 24 MP page directly needs a ~1100px
kernel and took **24 seconds per page**.

### Why strip table rules

Independently of binarisation, Tesseract's layout analysis can classify a
fully-bordered table as a non-text region and skip every cell in it — silently,
in *every* page segmentation mode (3, 4, 6, 11 and 12 were all tested). Erasing
the rules leaves the cell contents looking like ordinary text lines. On a form
page whose medication table is fully boxed, token recall went **3/10 → 10/10**;
on a form whose table was already read correctly it changed nothing.

Rules are found by directional morphological opening. The kernel length is
derived from the **median glyph height**, not from the page dimensions: a
kernel shorter than a capital letter matches the letter's own stems and erases
the glyph, which is a real failure on short pages or large type. Disable with
`OCR_STRIP_RULES=0`.

### Combined effect

Token recall, same two documents, all four combinations:

| | boxed table page | real bilingual form |
| --- | --- | --- |
| adaptive, rules kept *(the obvious build)* | 3/10 | 5/14 |
| adaptive, rules stripped | 10/10 | 12/14 |
| flatten+Otsu, rules kept | 3/10 | 14/14 |
| **flatten+Otsu, rules stripped** *(shipped)* | **10/10** | **14/14** |

---

## Privacy and logging

The service handles documents that may contain personal health information.

* **Documents never leave the machine.** Surya runs in-process; its only
  network access is the one-time model download into its cache.
* **Nothing is written to disk by this service.** Uploads, rendered pages and
  extracted text exist only in memory. Starlette's multipart spool threshold
  is raised above the upload cap in `main.py` so large uploads are not rolled
  onto a temp file, and PDFs are streamed to poppler over stdin.
* **No document content is logged.** The access log records request id,
  method, path, status and duration — no query string, headers or bodies.
  Error handlers log exception *types*; tracebacks are logged only when
  `OCR_DEBUG=1`, and never returned to the client.
* The `422` handler is custom because FastAPI's default echoes the submitted
  value back in the response body.

### The one caveat

`pdf2image` **cannot** honour the no-disk rule: `convert_from_bytes` writes the
PDF to a `tempfile.mkstemp()` file before invoking poppler. So `pdf_utils`
prefers a direct path that pipes the PDF to `pdfinfo -` / `pdftoppm -` over
stdin, which never touches the filesystem, and falls back to `pdf2image` only
if the installed poppler build rejects stdin. The fallback logs a warning
once, and `GET /health` reports `pdf.writes_temp_file`.

Note that `pdf2image` is not an *alternative* to poppler — it shells out to the
same binaries. Without `poppler-utils` installed, no PDF can be rendered at
all, and `GET /health` reports `pdf.available: false` rather than letting that
surface as a failure on the first PDF request.

If you end up on the fallback path and the temp write is unacceptable, either
set `OCR_PDF_TEMP_DIR` to a `tmpfs`/ramdisk, or replace `pdf_utils` with
PyMuPDF (`fitz`), which rasterises fully in memory.

---

## Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `OCR_ENGINE` | `surya` | `surya` or `tesseract` |
| `OCR_SURYA_MAX_PIXELS` | `20000000` | Pages above this are downscaled before Surya (bounds VRAM and time) |
| `OCR_ROW_OVERLAP` | `0.5` | Vertical overlap (share of the shorter box) for two boxes to count as one row — Surya lines, and words in lab tables |
| `OCR_SURYA_DESKEW_MIN_DEG` | `0.3` | Straighten Surya pages tilted by at least this many degrees |
| `OCR_LAB_CATALOG` | `ocr_service/lab_catalog.json` | Test catalog for `lab=true` |
| `OCR_LAB_MIN_CONFIDENCE` | `80` | Values read below this (and not cross-checked) get a `low_confidence` note |
| `TORCH_DEVICE` | auto (`cuda` if available) | Surya setting: force `cpu` / `cuda` |
| `MODEL_CACHE_DIR` | `%LOCALAPPDATA%\datalab\datalab\Cache\models` | Surya setting: where the weights are cached |
| `RECOGNITION_BATCH_SIZE` / `DETECTOR_BATCH_SIZE` | Surya auto | Surya settings: lower them if the GPU runs out of memory |
| `TESSERACT_CMD` | — | Path to `tesseract` if it is not on `PATH` |
| `OCR_MAX_UPLOAD_BYTES` | `26214400` | Upload cap (25 MB) |
| `OCR_MAX_PDF_PAGES` | `50` | Page cap for PDFs |
| `OCR_MAX_CONCURRENCY` | `min(4, cpu_count)` | Documents OCR'd at once |
| `OCR_PDF_DPI` | `300` | Default PDF rasterisation DPI |
| `OCR_PDF_TIMEOUT_S` | `120` | Per-page poppler timeout |
| `OCR_POPPLER_PATH` | — | Directory holding `pdfinfo`/`pdftoppm` if not on `PATH` |
| `OCR_PDF_TEMP_DIR` | — | Temp dir for the `pdf2image` fallback |
| `OCR_TESSERACT_CONFIG` | `--oem 3 --psm 3` | Raw Tesseract flags |
| `OCR_EXPECTED_LANGS` | `eng,fra,khm` | Languages `/health` requires before reporting `ready` |
| `OCR_MIN_KHMER_RATIO` | `0.10` | Share of letters in the Khmer block before `khm` is reported |
| `OCR_MIN_CHARS_FOR_LANGDETECT` | `40` | Latin characters needed before `langdetect` runs |
| `OCR_DENOISE` | `auto` | `auto` / `nlmeans` / `fast` / `off` |
| `OCR_BINARISE` | `flatten-otsu` | `flatten-otsu` / `adaptive` / `otsu` / `off` |
| `OCR_STRIP_RULES` | `1` | `0` to keep table borders |
| `OCR_ADAPTIVE_BLOCK` / `OCR_ADAPTIVE_C` | `31` / `15` | Window for `OCR_BINARISE=adaptive` |
| `OCR_BACKGROUND_ESTIMATE_PX` | `512` | Resolution the illumination field is estimated at |
| `OCR_MAX_DESKEW_DEG` | `12` | Deskew search range |
| `OCR_LOG_LEVEL` | `INFO` | Log level |
| `OCR_DEBUG` | — | `1` to log full tracebacks |

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest ocr_service/tests -v      # from the directory containing ocr_service/
```

The suite runs with `OCR_ENGINE=tesseract` (set in `conftest.py`) and needs
neither a GPU nor the Surya models: the Surya mapping and request path are
tested against fake predictions. Tesseract end-to-end tests skip themselves
when the binary is missing. Lab extraction is tested on a synthetic page with
the report's layout (`test_lab_results.py`). 59 tests, ~5 s. To also run
Surya on the real models (~15 s, needs the download):

```bash
OCR_TEST_SURYA=1 pytest ocr_service/tests/test_surya.py
```

On Windows, if `fra.traineddata` / `khm.traineddata` cannot be written into
`C:\Program Files\Tesseract-OCR\tessdata` (that needs an elevated shell), put
the language files in a writable directory and point `TESSDATA_PREFIX` at it:

```
set TESSDATA_PREFIX=D:\path\to\tessdata
```

## Performance notes

Both engines are blocking, so `/ocr` runs them via `run_in_threadpool` and
an `asyncio.Semaphore` caps concurrent documents.

**Surya** on an RTX 5080 Laptop GPU: ~0.9 s for a small image, ~2.6 s for a
dense A4 page, ~2.4 GB VRAM. The first request or two after startup are
slower while the GPU clocks up. Pages go through the GPU one at a time (a
lock in `surya_engine`); PDF rendering of the next page still overlaps.

**Tesseract** is CPU-bound: budget roughly 1–3 s per page at 300 DPI on one
core. Without the semaphore, parallel requests just thrash the cores.

Either way a 50-page PDF is a slow request, so consider a job queue rather
than a synchronous call if you expect those routinely.

PDF pages are rendered and OCR'd one at a time. Materialising 50 pages of
300 DPI colour up front would cost well over a gigabyte.
