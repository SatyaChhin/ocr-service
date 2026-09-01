# OCR Service

FastAPI service that extracts text from scanned forms, photographed documents
and PDFs using Tesseract with **English, French and Khmer** language data
(e.g. `lang="eng+khm"` — every language in the spec is matched in a single
pass).

Built for medical-style forms: OCR output preserves line and paragraph
structure so a label and its value stay on one line, and `?detail=true`
returns per-word bounding boxes and confidences you can build key-value or
table extraction on top of.

---

## System dependencies

These are **not** installable with `pip` and must be present on the host or in
the container image.

| Dependency | Why | Check |
| --- | --- | --- |
| `tesseract-ocr` | the OCR engine `pytesseract` shells out to | `tesseract --version` |
| `tesseract-ocr-eng` | English language data | `tesseract --list-langs` |
| `tesseract-ocr-fra` | French language data | `tesseract --list-langs` |
| `tesseract-ocr-khm` | Khmer language data | `tesseract --list-langs` |
| `poppler-utils` | `pdfinfo` / `pdftoppm`, used to rasterise PDF pages | `pdftoppm -v` |

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
pip install -r requirements.txt

# from the directory *containing* ocr_service/
uvicorn ocr_service.main:app --reload --port 8000
```

Interactive docs: <http://localhost:8000/docs>

**On this machine**, Tesseract is installed but not on `PATH`, and its `fra` /
`khm` language data lives in a project-local `tessdata/` directory because
`C:\Program Files\Tesseract-OCR\tessdata` is not writable without admin.
`run.ps1` sets `TESSERACT_CMD` and `TESSDATA_PREFIX` for you:

```powershell
.\run.ps1                 # http://localhost:8000
.\run.ps1 -Port 8080 -Reload
```

Check `GET /health` first — `"ready": true` means Tesseract and all expected
languages were found. `"ready": false` with a non-empty `missing_expected`
means language data is missing and `/ocr` will reject those languages.

---

## API

### `POST /ocr`

`multipart/form-data` with a single `file` part.

| Query param | Default | Notes |
| --- | --- | --- |
| `lang` | `eng+fra` | Any `+`-joined Tesseract codes (`eng`, `fra`, `khm`, `eng+khm`, ...). Validated against installed language data. |
| `detail` | `false` | Include per-word boxes and confidences. |
| `dpi` | `300` | PDF rasterisation DPI, 72–600. Ignored for images. |

```bash
curl -F "file=@form.pdf" "http://localhost:8000/ocr?lang=eng+fra&detail=true"
```

Response:

```jsonc
{
  "filename": "form.pdf",
  "media_type": "application/pdf",
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
divide by `scale`, then rotate by `-skew_deg` about the image centre.

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

### `GET /health`

Always returns **200** so a liveness probe will not restart a running
process. Use the `ready` boolean for readiness — it is false when Tesseract
or any of the expected languages is missing (`eng`, `fra`, `khm` by default —
see `OCR_EXPECTED_LANGS`). The body also reports the PDF backend in
use and the configured limits.

### Errors

Errors return `{"detail": "...", "request_id": "..."}` — never a stack trace.
`X-Request-ID` is set on every response; quote it when reporting a problem.

| Status | Cause |
| --- | --- |
| `400` | Unsupported/undecodable file, empty upload, bad `lang`, unreadable or over-long PDF |
| `413` | Upload exceeds 25 MB |
| `422` | Missing `file` part or out-of-range `dpi` |
| `500` | Unexpected failure (details logged, not returned) |
| `503` | Tesseract or poppler unavailable |

File type is decided by **magic bytes**, not by the filename or the
client-supplied `Content-Type`.

---

## Preprocessing pipeline

Applied to every page before OCR (`ocr.preprocess`):

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

Everything runs without Tesseract installed except the end-to-end tests, which
skip themselves when the binary is missing; the request path is still covered
on such machines through a stubbed engine. 28 tests, ~5 s.

On Windows, if `fra.traineddata` / `khm.traineddata` cannot be written into
`C:\Program Files\Tesseract-OCR\tessdata` (that needs an elevated shell), put
the language files in a writable directory and point `TESSDATA_PREFIX` at it:

```
set TESSDATA_PREFIX=D:\path\to\tessdata
```

## Performance notes

Tesseract is CPU-bound and blocking, so `/ocr` runs it via
`run_in_threadpool` and an `asyncio.Semaphore` caps concurrent documents —
without that cap, parallel requests just thrash the cores. Budget roughly
1–3 s per page at 300 DPI on one core; a 50-page PDF is a slow request, so
consider a job queue rather than a synchronous call if you expect those
routinely.

PDF pages are rendered and OCR'd one at a time. Materialising 50 pages of
300 DPI colour up front would cost well over a gigabyte.
