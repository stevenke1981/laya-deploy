# Run the real local Laya model and its Pac-Man window without a separate server.
param(
    [ValidateSet('english', 'multilingual', 'typed-decisions', 'auto')]
    [string]$Model = 'english',
    [ValidateRange(0, 60000)]
    [int]$Delay = 400,
    [int]$Seed = 0,
    [ValidateRange(1, 1000000)]
    [int]$Steps = 600
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$launcher = Join-Path $projectRoot '.venv\Scripts\laya-deploy.exe'
if (-not (Test-Path -LiteralPath $launcher)) {
    Write-Error 'Missing .venv. Run: uv sync --extra model --extra cua'
    exit 1
}
Push-Location -LiteralPath $projectRoot
try {
    & $launcher pacman --direct --model $Model --delay $Delay --seed $Seed --steps $Steps
    $resultCode = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $resultCode
