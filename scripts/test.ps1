# Run the whole test suite (Python + JavaScript parity) on Windows.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
$py = if (Test-Path ".venv\Scripts\python.exe") { ".venv\Scripts\python.exe" } else { "python" }
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Error "Node.js is required (JavaScript parity tests)"
}
Get-ChildItem web\*.js | ForEach-Object {
    node --check $_.FullName
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}
& $py -m pytest -q @args
exit $LASTEXITCODE
