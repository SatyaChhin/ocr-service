# OCR Service — Setup Guide

Extracts text and structured lab results from scanned or photographed medical
forms. Two pieces make up the system:

| Part | Repo | What it is |
| --- | --- | --- |
| **Backend** | `Backend/ocr-service` (this one) | FastAPI service running Surya OCR on the GPU, plus MySQL storage for saved lab reports |
| **Frontend** | `Frontend/ocr-site` | Nuxt web UI for uploading a scan and reviewing what came out |

The backend works on its own — you can drive it entirely from `/docs` or
`curl`. Set up the frontend only if you want the web UI.

This file gets you running from nothing. For what each setting, endpoint and
field means, see **[`ocr_service/README.md`](ocr_service/README.md)** — the
reference documentation.

---

## Before you start

| Need | Version | Check | Required? |
| --- | --- | --- | --- |
| Python | 3.11 – 3.13 | `python --version` | **Yes** |
| NVIDIA GPU + driver | CUDA 12.8 capable | `nvidia-smi` | Strongly recommended — CPU works but is far slower |
| poppler | any recent | `pdftoppm -v` | Only if you feed it PDFs |
| MariaDB / MySQL | 10.6+ / 8.0+ | `mysql --version` | Only to save lab results |
| Node.js | 20+ | `node --version` | Only for the frontend |

Python **3.14 is not supported yet**: OpenCV publishes no wheels for it at the
pinned version, so pip tries to build from source and fails.

> Verified on this machine with Python 3.12.10, Node 24.20.0, MariaDB 12.3.3
> and an RTX 5080 Laptop GPU.

---

## Step 1 — Get the code

```powershell
cd D:\projects_sky\Backend
git clone <ocr-service-repo-url> ocr-service
cd ocr-service
```

The frontend is a **separate repository**; clone it only if you need the UI:

```powershell
cd D:\projects_sky\Frontend
git clone <ocr-site-repo-url> ocr-site
```

---

## Step 2 — Python environment

Create the virtualenv **inside** `ocr-service` — `run.ps1` expects it at
`.venv`:

```powershell
cd D:\projects_sky\Backend\ocr-service
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install PyTorch **first**, from the CUDA index. PyPI's Windows wheels are
CPU-only, and `cu128` is what covers RTX 50-series (Blackwell, sm_120):

```powershell
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r ocr_service\requirements.txt
```

Confirm the GPU is visible before going further — if this prints `False`,
Surya will fall back to the CPU and every request will take seconds longer:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

---

## Step 3 — System dependencies

These are **not** pip-installable.

**poppler** (PDF rasterising). Without it, PDF uploads return `503`; images
still work.

- *Windows*: install the [poppler-windows](https://github.com/oschwartz10612/poppler-windows/releases)
  release, then either add its `Library\bin` to `PATH` or set
  `OCR_POPPLER_PATH` to that directory. `run.ps1` also auto-detects the winget
  package.
- *Debian/Ubuntu*: `sudo apt-get install -y poppler-utils`
- *macOS*: `brew install poppler`

That is the only system dependency. Surya is the OCR engine and needs no
language data — it reads every script it knows, Khmer included, with no
configuration.

---

## Step 4 — Database

Only needed for `POST /lab-reports`. Skip to Step 5 if you just want OCR.

As an admin user (`mysql -u root -p`):

```sql
CREATE DATABASE ocr CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

CREATE USER 'ocr_app'@'localhost'  IDENTIFIED BY 'choose-a-strong-password';
CREATE USER 'ocr_app'@'127.0.0.1'  IDENTIFIED BY 'choose-a-strong-password';

GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, REFERENCES, INDEX, ALTER
  ON ocr.* TO 'ocr_app'@'localhost';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, REFERENCES, INDEX, ALTER
  ON ocr.* TO 'ocr_app'@'127.0.0.1';

FLUSH PRIVILEGES;
```

Create both hosts: whether `127.0.0.1` matches `@localhost` or `@127.0.0.1`
depends on the platform and connection type, and a mismatch shows up as a
confusing access-denied error.

`CREATE`/`ALTER` are needed because the service creates and upgrades its own
table on first use. `DROP` is deliberately **not** granted — the app never
needs to drop anything, and a least-privilege user shouldn't hold DDL it
doesn't use.

Now write the credentials into a git-ignored `.env` in the repo root:

```powershell
copy .env.example .env
notepad .env          # set OCR_DB_USER=ocr_app and OCR_DB_PASSWORD=...
```

The table itself is created automatically on the first save — you do not run
`schema.sql` by hand.

---

## Step 5 — Start the backend

```powershell
.\run.ps1
```

`run.ps1` handles the venv and poppler discovery for you.

**The first start downloads ~1.5 GB of Surya models** from `models.datalab.to`
into `%LOCALAPPDATA%\datalab\datalab\Cache\models`. That happens once. After
that, startup loads them onto the GPU in about 5–15 seconds.

Loading happens in a background thread, so the service answers immediately
with `engine.state: "loading"` and `ready: false`. Wait for `ready: true`:

```powershell
curl.exe http://localhost:8000/health
```

Then open the interactive API docs and try an upload:

**<http://localhost:8000/docs>**

> There is no route at `/` — opening `http://localhost:8000/` returns
> `{"detail":"Not Found"}`. That is the service working correctly. Use `/docs`.

Useful variants:

```powershell
.\run.ps1 -Port 8001              # if 8000 is taken (a PHP app owns it on some machines)
.\run.ps1 -Reload                 # auto-reload while editing code
```

---

## Step 6 — Start the frontend (optional)

```powershell
cd D:\projects_sky\Frontend\ocr-site
npm install
copy .env.example .env
```

Point `.env` at whichever port the backend is on — this is the single most
common setup mistake:

```
NUXT_OCR_API_BASE=http://localhost:8000
```

Then:

```powershell
npm run dev
```

Open **<http://localhost:3000>**. The browser talks to `/api/ocr` on the Nuxt
server, which forwards to the backend, so the OCR service needs no CORS
configuration.

---

## Step 7 — Check it end to end

1. Open <http://localhost:3000> (or `/docs` if you skipped the frontend).
2. Drop in a scan — PNG, JPEG, WEBP or PDF, up to 25 MB.
3. Press **Extract text**. You should get the text, per-word confidences and
   timings back.
4. For a lab report, the **Lab results** table appears; rows the service is
   unsure about are highlighted for review.
5. Press **Save** to write it to the database, then confirm:

```sql
SELECT id, sample_no, results_count, needs_review_count FROM ocr.lab_reports;
```

There is also a one-command helper that does OCR and the insert together:

```powershell
.\save-lab.ps1 report.jpg -DryRun    # show what would be inserted
.\save-lab.ps1 report.jpg            # actually insert
```

---

## When something goes wrong

Check **`GET /health` first** — it reports every dependency and is the fastest
way to tell a setup problem from a code problem.

| Symptom | Cause and fix |
| --- | --- |
| `{"detail":"Not Found"}` in the browser | You hit `/`, which has no route. Use `/docs`. |
| `ready: false`, `engine.state: "loading"` | Models still loading. Wait; the first run also downloads them. |
| `503` on a PDF, images fine | poppler missing. `pdftoppm -v`, then set `OCR_POPPLER_PATH`. |
| `503` on everything | The engine failed to load. `health.engine.error` says why. |
| Frontend shows "Backend offline" | `NUXT_OCR_API_BASE` points at the wrong port. |
| `503` with "database" in the message | MySQL unreachable, or the `.env` credentials are wrong. |
| Access denied for `ocr_app` | The user exists for one host but not the other — see Step 4. |
| Surya running on the CPU | `torch.cuda.is_available()` is `False`. PyTorch was installed from PyPI instead of the cu128 index. Reinstall per Step 2. |
| CUDA out of memory | Lower `RECOGNITION_BATCH_SIZE` / `DETECTOR_BATCH_SIZE`, or `OCR_SURYA_MAX_PIXELS`. |

Every error response carries a `request_id`, also sent as the `X-Request-ID`
header. Quote it when reporting a problem. For full tracebacks in the log, set
`OCR_DEBUG=1`.

---

## Running the tests

```powershell
pip install -r ocr_service\requirements-dev.txt
python -m pytest ocr_service\tests -q
```

65 tests, a couple of seconds. They need neither a GPU nor the Surya models:
the request path runs against canned predictions fed in at the predictor.
Two skip unless you opt into the real models and the real database, below.

Two opt-in suites hit real resources:

```powershell
$env:OCR_TEST_SURYA=1; python -m pytest ocr_service\tests\test_surya.py   # real models
$env:OCR_TEST_DB=1;    python -m pytest ocr_service\tests\test_db.py      # real database
```

The database one deletes everything it writes.

---

## Where things live

| Path | What |
| --- | --- |
| `ocr_service/main.py` | FastAPI app: routes, validation, error handling |
| `ocr_service/surya_engine.py` | Surya (default engine) |
| `ocr_service/ocr.py` | Shared helpers: decoding, deskew, row grouping, language detection |
| `ocr_service/lab_results.py` | Turns OCR output into structured lab records |
| `ocr_service/lab_catalog.json` | Known tests, their categories, units and aliases |
| `ocr_service/db.py` | MySQL storage and schema migrations |
| `ocr_service/schema.sql` | The `lab_reports` table |
| `run.ps1` | Start the service |
| `save-lab.ps1` | OCR a report and insert it in one command |
| `.env` | Database credentials (git-ignored; copy from `.env.example`) |

**[`ocr_service/README.md`](ocr_service/README.md)** documents the API,
every configuration variable, how lab extraction and review notes work, the
preprocessing choices and the privacy guarantees.
