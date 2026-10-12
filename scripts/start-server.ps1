# 以 .env 設定啟動 laya-deploy 伺服器（Windows）。
# 用法：powershell -ExecutionPolicy Bypass -File scripts\start-server.ps1 [-Fake]
param([switch]$Fake)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$env:USE_TF = '0'   # TensorFlow 的 runtime 可能讓模型建構卡死
$exe = Join-Path $root '.venv\Scripts\laya-deploy.exe'
if (-not (Test-Path $exe)) {
    Write-Host '找不到 .venv，先執行：uv sync --extra model' -ForegroundColor Yellow
    exit 1
}
$serveArgs = @('serve')
if ($Fake) { $serveArgs += '--fake' }
& $exe @serveArgs
