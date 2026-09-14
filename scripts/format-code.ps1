# Defaults to read-only; pass --write explicitly to change files.
& python (Join-Path $PSScriptRoot "format_code.py") @args
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
