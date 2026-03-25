param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('demo', 'full')]
    [string]$Mode,

    [switch]$NoRestart
)

$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = Resolve-Path (Join-Path $ScriptDir '..')
$TemplatePath = Join-Path $RepoRoot ("env.$Mode")
$EnvPath = Join-Path $RepoRoot '.env'

if (-not (Test-Path $TemplatePath)) {
    throw "Template not found: $TemplatePath"
}

Copy-Item -Path $TemplatePath -Destination $EnvPath -Force
Write-Host "[ok] switched .env -> env.$Mode"

if ($NoRestart) {
    Write-Host '[skip] restart skipped because -NoRestart is set'
    exit 0
}

Push-Location $RepoRoot
try {
    docker compose up -d --force-recreate api worker
    if ($LASTEXITCODE -ne 0) {
        throw "docker compose failed with exit code $LASTEXITCODE"
    }
} finally {
    Pop-Location
}

Write-Host '[ok] api/worker recreated'
$verifyCmd = 'docker compose exec -T api /app/.venv/bin/python -c "from app.core.config import DATA_SCHEMA,DEMO_DATA_DIR,DEMO_MANIFEST_PATH; print(DATA_SCHEMA); print(DEMO_DATA_DIR); print(DEMO_MANIFEST_PATH)"'
Write-Host ("[hint] verify active mode: " + $verifyCmd)
