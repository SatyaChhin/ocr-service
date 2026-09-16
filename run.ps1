# Start the OCR service.
#   .\run.ps1                       -> http://localhost:8000, Surya on the GPU
#   .\run.ps1 -Port 8001 -Reload
param(
    [int]$Port = 8000,
    [switch]$Reload
)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

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

Write-Host "OCR service (Surya) -> http://localhost:$Port  (docs: /docs, health: /health)"
& $python @uvicornArgs

} finally { Pop-Location }
