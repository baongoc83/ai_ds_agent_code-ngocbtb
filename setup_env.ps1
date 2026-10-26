# One-shot environment setup (Windows PowerShell).
#
#   .\setup_env.ps1            # runtime only  → .venv
#   .\setup_env.ps1 -Dev       # + pytest
#
# Creates .venv with Python 3.12 (.python-version), installs the exact pinned
# versions from requirements-lock.txt, then verifies them with tools/check_env.py.
# Afterwards ALWAYS run the project with this interpreter:
#   .\.venv\Scripts\python.exe main.py ...
#   .\.venv\Scripts\python.exe -m streamlit run ui/app.py
param([switch]$Dev)
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$pyVer = (Get-Content .python-version).Trim()
if (-not (Test-Path .venv\Scripts\python.exe)) {
    Write-Host "Creating .venv with Python $pyVer ..."
    py -$pyVer -m venv .venv
    if ($LASTEXITCODE -ne 0) { throw "Python $pyVer not found. Install it from python.org (py launcher), then re-run." }
}
$py = ".\.venv\Scripts\python.exe"
& $py -m pip install --upgrade pip
& $py -m pip install -r requirements-lock.txt
if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
if ($Dev) {
    & $py -m pip install -r requirements-dev.txt
    if ($LASTEXITCODE -ne 0) { throw "pip install (dev) failed" }
    & $py tools\check_env.py --dev
} else {
    & $py tools\check_env.py
}
if ($LASTEXITCODE -ne 0) { throw "environment check failed" }
Write-Host "`nDone. Use: $py -m streamlit run ui/app.py"
