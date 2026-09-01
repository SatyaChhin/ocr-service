# Start the OCR service.
#   .\run.ps1            -> http://localhost:8000
#   .\run.ps1 -Port 8080 -Reload
param(
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

# Tesseract is installed but not on PATH, and its fra/khm language data lives in
# the project-local tessdata/ dir because Program Files is not writable.
$env:TESSERACT_CMD   = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:TESSDATA_PREFIX = Join-Path $root "tessdata"

if (-not (Test-Path $env:TESSERACT_CMD)) {
    throw "Tesseract not found at $($env:TESSERACT_CMD) - install it or edit run.ps1"
}

# uvicorn resolves "ocr_service.main" from the working directory.
Push-Location $root
try {

$python = Join-Path $root ".venv\Scripts\python.exe"
$args = @("-m", "uvicorn", "ocr_service.main:app", "--port", $Port)
if ($Reload) { $args += "--reload" }

Write-Host "OCR service -> http://localhost:$Port  (docs: /docs, health: /health)"
& $python @args

} finally { Pop-Location }
