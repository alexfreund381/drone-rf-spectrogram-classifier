# Usage: .\make.ps1 verify
param([string]$Target = "verify")
if ($Target -eq "verify") {
    & .venv\Scripts\python -m ruff check .
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & .venv\Scripts\python -m pytest -q
    exit $LASTEXITCODE
}
Write-Error "unknown target: $Target"
