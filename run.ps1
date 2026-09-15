# Start the OCR service.
#   .\run.ps1                       -> http://localhost:8000, Surya on the GPU
#   .\run.ps1 -Port 8080 -Reload
#   .\run.ps1 -Engine tesseract     -> fall back to Tesseract
param(
    [int]$Port = 8000,
    [switch]$Reload,
    [ValidateSet("surya", "tesseract")]
    [string]$Engine = "surya"
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$env:OCR_ENGINE = $Engine

# Tesseract is installed but not on PATH, and its fra/khm language data lives in
# the project-local tessdata/ dir because Program Files is not writable. Set
# either way: /health reports the fallback engine's status too.
$env:TESSERACT_CMD   = "C:\Program Files\Tesseract-OCR\tesseract.exe"
$env:TESSDATA_PREFIX = Join-Path $root "tessdata"

if (-not (Test-Path $env:TESSERACT_CMD)) {
    if ($Engine -eq "tesseract") {
        throw "Tesseract not found at $($env:TESSERACT_CMD) - install it or edit run.ps1"
    }
    Write-Warning "Tesseract not found - the fallback engine (-Engine tesseract) will not work"
}

# The winget poppler package does not put pdftoppm on PATH; find it there.
if (-not $env:OCR_POPPLER_PATH -and -not (Get-Command pdftoppm -ErrorAction SilentlyContinue)) {
    $pdftoppm = Get-ChildItem "$env:LOCALAPPDATA\Microsoft\WinGet\Packages\oschwartz10612.Poppler_*" `
        -Filter pdftoppm.exe -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($pdftoppm) { $env:OCR_POPPLER_PATH = $pdftoppm.DirectoryName }
    else { Write-Warning "poppler not found - PDF uploads will return 503 (images still work)" }
}

$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
    throw "No virtualenv at $root\.venv - create it with Python 3.11-3.13 and install ocr_service\requirements.txt"
}

# uvicorn resolves "ocr_service.main" from the working directory.
Push-Location $root
try {

$uvicornArgs = @("-m", "uvicorn", "ocr_service.main:app", "--port", $Port)
if ($Reload) { $uvicornArgs += "--reload" }

Write-Host "OCR service ($Engine) -> http://localhost:$Port  (docs: /docs, health: /health)"
& $python @uvicornArgs

} finally { Pop-Location }
