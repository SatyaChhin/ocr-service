# OCR a lab report and save its results to the database.
#
#   .\save-lab.ps1 report.jpg                  # OCR, then insert
#   .\save-lab.ps1 report.jpg -DryRun          # show the JSON, insert nothing
#   .\save-lab.ps1 report.jpg -OutFile body.json -DryRun
#   .\save-lab.ps1 report.jpg -Replace         # overwrite the same patient + sample
#   .\save-lab.ps1 report.pdf -Port 8001
param(
    [Parameter(Mandatory, Position = 0)][string]$File,
    [int]$Port = 8000,
    [switch]$Replace,
    [switch]$DryRun,
    [string]$OutFile
)

$ErrorActionPreference = "Stop"
if (-not (Test-Path -LiteralPath $File)) { throw "No such file: $File" }
$base = "http://localhost:$Port"

# 1. OCR with lab=true -- this already returns database-ready records.
Write-Host "OCR  $File ..." -ForegroundColor Cyan
$ocr = Invoke-RestMethod -Method Post -Uri "$base/ocr?lab=true" `
                         -Form @{ file = Get-Item -LiteralPath $File }

if (-not $ocr.lab) { throw "No lab data extracted -- is this a lab report?" }

# 2. The save body IS the lab object plus filename/engine from the response.
$body = [ordered]@{
    filename = $ocr.filename
    engine   = $ocr.engine
    report   = $ocr.lab.report
    results  = $ocr.lab.results
    review   = $ocr.lab.review
    replace  = [bool]$Replace
}

$json = $body | ConvertTo-Json -Depth 12
if ($OutFile) { $json | Set-Content -LiteralPath $OutFile -Encoding utf8; Write-Host "Wrote $OutFile" }

# 3. Show what will be inserted.
$needs = @($ocr.lab.review | Where-Object { $_.needs_review }).Count
Write-Host ("`n{0} results, {1} need review  (patient {2}, sample {3})" -f `
    $ocr.lab.results.Count, $needs, $ocr.lab.report.patient_code, $ocr.lab.report.sample_no) -ForegroundColor Yellow
$ocr.lab.results | Format-Table name, value, flag, unit, ref_range -AutoSize

if ($DryRun) { Write-Host "-DryRun: nothing inserted." -ForegroundColor DarkGray; return }

# 4. Insert.
try {
    $saved = Invoke-RestMethod -Method Post -Uri "$base/lab-reports" `
                               -ContentType "application/json; charset=utf-8" `
                               -Body ([Text.Encoding]::UTF8.GetBytes($json))
    Write-Host ("Saved report id {0}: {1} results, {2} flagged for review." -f `
        $saved.id, $saved.results_saved, $saved.needs_review) -ForegroundColor Green
    if ($saved.replaced) { Write-Host "Replaced report id $($saved.replaced)." }
}
catch {
    $r = $_.Exception.Response
    if ($r -and $r.StatusCode.value__ -eq 409) {
        Write-Warning "Already saved for this patient + sample. Re-run with -Replace to overwrite."
    }
    else { throw }
}
