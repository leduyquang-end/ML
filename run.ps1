# Chạy bằng Python ổ E; đặt cache/thư mục tạm trước cả khi Python khởi động.
$ErrorActionPreference = "Stop"
$projectCache = Join-Path $PSScriptRoot ".cache"
$projectTemp = Join-Path $projectCache "tmp"
New-Item -ItemType Directory -Force -Path $projectTemp | Out-Null
$env:TEMP = $projectTemp
$env:TMP = $projectTemp
$env:TMPDIR = $projectTemp
$env:PIP_CACHE_DIR = Join-Path $projectCache "pip"

Push-Location -LiteralPath $PSScriptRoot
try {
    & "$PSScriptRoot\.venv\Scripts\python.exe" -B "$PSScriptRoot\phobert_features.py" @args
    $result = $LASTEXITCODE
}
finally {
    Pop-Location
}
exit $result
