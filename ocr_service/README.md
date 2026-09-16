# OCR Service

FastAPI service that extracts text from scanned forms, photographed documents
and PDFs, using [Surya](https://github.com/datalab-to/surya) 0.17.1 — an
in-process PyTorch model on the GPU. It reads every script it knows at once,
Khmer included, so there is no language to choose.

Built for medical-style forms: OCR output keeps a label and its value on one
line, and `?detail=true` returns per-word bounding boxes and confidences you
can build key-value or table extraction on top of.

### Accuracy on this service's test pages

| Page | Result (RTX 5080 Laptop) |
| --- | --- |
| Clean 900×200 PNG, "Hello from the OCR service / Invoice total: 1234.56" | exact, ~0.9 s |
| Synthetic A4 lab report (English, 104 words) | exact rows, 93% mean confidence, ~2.6 s |
| Bilingual Khmer/English form | 94% confidence; 3 Khmer character errors, incl. age `៥៥` → `៥៤` |

Good, but not error-free — it misread a Khmer digit in a patient's age.
Treat the low-confidence review as a hint, not a guarantee: that digit was
read with high confidence.

**Confidence is the model's token probability** (0–100 here). It scores
**bold** text noticeably lower even when it is read correctly, which is why
a cross-checked value is trusted regardless of its score — see
[Lab results](#lab-results).

### Licensing (Surya)

The Surya code is GPL-3.0. The model weights use a modified AI Pubs Open
Rail-M licence: free for research, personal use, and organisations under
$2M in funding/revenue; broader commercial use needs a licence from
[Datalab](https://www.datalab.to/pricing). Check this before deploying
commercially.

---

## System dependencies

These are **not** installable with `pip` and must be present on the host or in
the container image.

| Dependency | Why | Check |
| --- | --- | --- |
| NVIDIA GPU + driver | Surya on CUDA (it falls back to CPU, which is far slower) | `nvidia-smi` |
| `poppler-utils` | `pdfinfo` / `pdftoppm`, used to rasterise PDF pages | `pdftoppm -v` |

Surya needs no language data: it reads every script it knows without a hint.

**Debian / Ubuntu**

```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

**macOS (Homebrew)**

```bash
brew install poppler
```

**Windows**

1. Download the [poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases)
   release and add its `Library\bin` directory to `PATH`.
2. If it is not on `PATH`, point the service at it directly:
   ```
   set OCR_POPPLER_PATH=C:\poppler\Library\bin
   ```

**Docker**

```dockerfile
FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
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

`run.ps1` starts the service on Surya:

```powershell
.\run.ps1 -Port 8001                 # http://localhost:8001, Surya
.\run.ps1 -Port 8001 -Reload         # auto-reload while editing
```

Port 8000 is taken by a PHP app on this machine, so run the service on
**8001** (`.\run.ps1 -Port 8001`) — that is the port the `ocr-site` frontend's
git-ignored `.env` points `NUXT_OCR_API_BASE` at. The default stays 8000
everywhere else; only the local `.env` files override it.

Check `GET /health` first — `"ready": true` means the active engine is
loaded. Surya loads its models in a background thread, so expect
`engine.state: "loading"` for the first ~5–15 s after startup.

---

## API

### `POST /ocr`

`multipart/form-data` with a single `file` part.

| Query param | Default | Notes |
| --- | --- | --- |
| `lang` | `eng+fra+khm` | Syntax-checked and echoed back on the response, otherwise **ignored**: Surya reads every script it knows without a hint. Kept so a client can record what it expected. |
| `detail` | `false` | Include per-word boxes and confidences. |
| `dpi` | `300` | PDF rasterisation DPI, 72–600. Ignored for images. |
| `lab` | `false` | Also extract structured lab results into `lab` — see [Lab results](#lab-results). |

```bash
curl -F "file=@form.pdf" "http://localhost:8000/ocr?lang=eng+fra+khm&detail=true"
```

Response:

```jsonc
{
  "filename": "form.pdf",
  "media_type": "application/pdf",
  "engine": "surya",
  "lang": "eng+fra+khm",
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

**Language detection is best effort.** The engine does not report which
languages it read, so detection runs over the extracted text instead, from
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
      { "name": "RBC", "value": 5.31, "flag": "H", "unit": "x10^12/L",
        "ref_range": "3.8 - 4.8", "category": "COMPLETE BLOOD COUNT" },
      { "name": "Neutrophils", "percent": 65.7, "value": 7.15, "flag": "H",   // percent: differential rows only
        "unit": "x10^9/L", "ref_range": "2 - 7", "category": "Differential White Cell Count" },
      { "name": "Blood Group", "value": "O Rh (D): Positive", "flag": null,
        "unit": null, "ref_range": null, "category": "COMPLETE BLOOD COUNT" }
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
and categories. Insert `results[i]` only when `review[i].needs_review` is false,
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

**Test catalog** (`lab_catalog.json`). Canonical `name`, `category` and
unit per test, plus aliases (`Haemoglobin`, `HGB`, `Neutrophils (%)`, ...).
Sections follow the catalog because the report's own headings do not map
one-to-one onto the database groups (AST/ALT sit under "Transaminase" on the
report but belong to LIVER FUNCTIONS). Tests not in the catalog are still
extracted, with the nearest heading as their category; add them to the catalog
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
`suggested: 0.2`).

### Saving lab reports to MySQL

`lab.report` carries the header that links the results to a patient:
`patient_code`, `sample_no`, `collected_at`, `received_at` (ISO 8601). The
patient's name and address are deliberately not extracted or stored.

| Endpoint | Purpose |
| --- | --- |
| `POST /lab-reports` | Save `{filename, engine, report, results, review}` — the `lab` object of an `/ocr?lab=true` response plus context. `201 {id, results_saved, needs_review, replaced}`. `409` with `report_id` if this `patient_code` + `sample_no` is already saved; resend with `"replace": true` to overwrite. `503` if the database is down. |
| `GET /lab-reports?limit=20` | Recently saved reports (header and counts). |
| `GET /lab-reports/{id}` | One report, `results` in exactly the JSON format above, plus per-result `review`. |

One table (`schema.sql`, created on first use in the database named by
`OCR_DB_NAME`): **`lab_reports`**, one row per saved report, unique on
`(patient_code, sample_no)`. The header is columns — `patient_code`,
`sample_no`, `collected_at`, `received_at`, `source_filename`, `engine`,
`results_count`, `needs_review_count`, `created_at` — and the results
themselves are two JSON columns:

* `results` — `lab.results` exactly as `/ocr?lab=true` returned it.
* `review` — `lab.review`, where `review[i]` describes `results[i]`.

Storing the array rather than a row per test means what comes back out of
`GET /lab-reports/{id}` is the format the API produced, key order and number
types included: `744` is still an int, `"O Rh (D): Positive"` is still text.
There is no column mapping to drift. The price is that a result is not a row,
so aggregate queries go through `JSON_TABLE` instead of a plain `WHERE`.

Cheap things stay cheap — the per-report counts are real columns, and single
fields need no join:

```sql
SELECT patient_code, sample_no, results_count, needs_review_count,
       JSON_VALUE(results, '$[0].name') AS first_test
FROM lab_reports
WHERE needs_review_count > 0;
```

To query across individual results, expand the arrays. `FOR ORDINALITY` is
what pairs a result with its review entry, since `review[i]` describes
`results[i]`:

```sql
SELECT r.patient_code, r.sample_no, x.name, x.value, v.notes
FROM lab_reports r
JOIN JSON_TABLE(r.results, '$[*]' COLUMNS (
       seq       FOR ORDINALITY,
       name      VARCHAR(128) PATH '$.name',
       value     VARCHAR(255) PATH '$.value'
     )) x
JOIN JSON_TABLE(r.review, '$[*]' COLUMNS (
       seq          FOR ORDINALITY,
       needs_review INT  PATH '$.needs_review',
       notes        JSON PATH '$.notes'
     )) v ON v.seq = x.seq
WHERE v.needs_review = 1;
```

Results that still need review are saved with `needs_review: true` in
`review` (the web UI asks before saving them) and counted in
`needs_review_count`, so the query above finds them later.

**Upgrading an existing database.** Results used to live in a separate
`lab_results` table, one row per test. `db.py` folds those rows into
`lab_reports.results` on first use and then drops the table — no manual step,
and it only touches reports whose `results` is still `NULL`, so it cannot
overwrite newer JSON. If the service's database user lacks `DROP` (the
least-privilege `ocr_app` does), the data is still migrated and a warning
names the leftover table; remove it as an admin:

```sql
DROP TABLE lab_results;
```

**Renamed keys.** A result's `test_name` is now `name` and its `section` is
now `category`. Reports saved under the old names are rewritten on first use
(`rename_result_keys`), keys re-emitted in the documented order so a migrated
row is indistinguishable from a freshly saved one. The rename reaches the
whole chain — `lab_catalog.json`, the `/ocr?lab=true` response, the
`POST /lab-reports` body and the `ocr-site` frontend — so an older client
reading `test_name` sees `undefined`.

**On this machine** the database is the `ocr` schema on the local MariaDB
12.3 service, reached as a dedicated user `ocr_app` that has rights on `ocr.*`
only (its password is in the git-ignored `ocr-service/.env`; see
`.env.example`). To remove it: `DROP USER 'ocr_app'@'localhost', 'ocr_app'@'127.0.0.1';`.

### `GET /health`

Always returns **200** so a liveness probe will not restart a running
process. Use the `ready` boolean for readiness — it is false while Surya's
models are loading and when the engine failed to load. `engine` describes it
(`name`, `state`, `version`, `device`, `device_name`, `uses_language_hints`).
The body also reports the database, the PDF backend in use and the
configured limits.

### Errors

Errors return `{"detail": "...", "request_id": "..."}` — never a stack trace.
`X-Request-ID` is set on every response; quote it when reporting a problem.

| Status | Cause |
| --- | --- |
| `400` | Unsupported/undecodable file, empty upload, bad `lang`, unreadable or over-long PDF |
| `413` | Upload exceeds 25 MB |
| `422` | Missing `file` part or out-of-range `dpi` |
| `500` | Unexpected failure (details logged, not returned) |
| `503` | OCR engine (Surya models) or poppler unavailable |

File type is decided by **magic bytes**, not by the filename or the
client-supplied `Content-Type`.

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
| `OCR_ENGINE` | `surya` | Only `surya`. Kept so an old value fails loudly at startup. |
| `OCR_SURYA_MAX_PIXELS` | `20000000` | Pages above this are downscaled before Surya (bounds VRAM and time) |
| `OCR_ROW_OVERLAP` | `0.5` | Vertical overlap (share of the shorter box) for two boxes to count as one row — Surya lines, and words in lab tables |
| `OCR_SURYA_DESKEW_MIN_DEG` | `0.3` | Straighten Surya pages tilted by at least this many degrees |
| `OCR_LAB_CATALOG` | `ocr_service/lab_catalog.json` | Test catalog for `lab=true` |
| `OCR_DB_HOST` / `OCR_DB_PORT` | `127.0.0.1` / `3306` | MySQL/MariaDB for saved lab reports |
| `OCR_DB_USER` / `OCR_DB_PASSWORD` | `root` / empty | Database login (set in `.env`) |
| `OCR_DB_NAME` | `ocr` | Database; must exist, tables are created on first use |
| `OCR_LAB_MIN_CONFIDENCE` | `80` | Values read below this (and not cross-checked) get a `low_confidence` note |
| `TORCH_DEVICE` | auto (`cuda` if available) | Surya setting: force `cpu` / `cuda` |
| `MODEL_CACHE_DIR` | `%LOCALAPPDATA%\datalab\datalab\Cache\models` | Surya setting: where the weights are cached |
| `RECOGNITION_BATCH_SIZE` / `DETECTOR_BATCH_SIZE` | Surya auto | Surya settings: lower them if the GPU runs out of memory |
| `OCR_MAX_UPLOAD_BYTES` | `26214400` | Upload cap (25 MB) |
| `OCR_MAX_PDF_PAGES` | `50` | Page cap for PDFs |
| `OCR_MAX_CONCURRENCY` | `min(4, cpu_count)` | Documents OCR'd at once |
| `OCR_PDF_DPI` | `300` | Default PDF rasterisation DPI |
| `OCR_PDF_TIMEOUT_S` | `120` | Per-page poppler timeout |
| `OCR_POPPLER_PATH` | — | Directory holding `pdfinfo`/`pdftoppm` if not on `PATH` |
| `OCR_PDF_TEMP_DIR` | — | Temp dir for the `pdf2image` fallback |
| `OCR_MIN_KHMER_RATIO` | `0.10` | Share of letters in the Khmer block before `khm` is reported |
| `OCR_MIN_CHARS_FOR_LANGDETECT` | `40` | Latin characters needed before `langdetect` runs |
| `OCR_MAX_DESKEW_DEG` | `12` | Deskew search range |
| `OCR_LOG_LEVEL` | `INFO` | Log level |
| `OCR_DEBUG` | — | `1` to log full tracebacks |

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest ocr_service/tests -v      # from the directory containing ocr_service/
```

The suite needs neither a GPU nor the Surya models: the mapping and the whole
request path run against canned predictions fed in at the predictor (the
`fake_surya` fixture in `conftest.py`). Lab extraction is tested on a
synthetic page with the report's layout (`test_lab_results.py`). 65 tests,
a couple of seconds. Two skip unless you opt into the real resources:

```bash
OCR_TEST_SURYA=1 pytest ocr_service/tests/test_surya.py   # Surya models (~15 s, needs the download)
OCR_TEST_DB=1 pytest ocr_service/tests/test_db.py         # the database in .env; deletes what it writes
```

## Performance notes

Both engines are blocking, so `/ocr` runs them via `run_in_threadpool` and
an `asyncio.Semaphore` caps concurrent documents.

On an RTX 5080 Laptop GPU: ~0.9 s for a small image, ~2.6 s for a dense A4
page, ~2.4 GB VRAM. The first request or two after startup are slower while
the GPU clocks up. Pages go through the GPU one at a time (a lock in
`surya_engine`); PDF rendering of the next page still overlaps.

A 50-page PDF is therefore a slow request, so consider a job queue rather
than a synchronous call if you expect those routinely.

PDF pages are rendered and OCR'd one at a time. Materialising 50 pages of
300 DPI colour up front would cost well over a gigabyte.
