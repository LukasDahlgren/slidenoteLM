[CmdletBinding()]
param(
    [string]$ProjectId = "",
    [string]$HostAddress = "",
    [int]$Port = 0,
    [switch]$NoReload,
    [switch]$SkipBws
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path

function Get-DotEnvValue {
    param(
        [string]$FilePath,
        [string]$Key
    )
    if (-not (Test-Path -LiteralPath $FilePath)) {
        return ""
    }
    foreach ($line in Get-Content -LiteralPath $FilePath) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $parts = $trimmed -split "=", 2
        if ($parts.Count -eq 2 -and $parts[0].Trim() -eq $Key) {
            return $parts[1].Trim().Trim('"').Trim("'")
        }
    }
    return ""
}

$envFile = Join-Path $RepoRoot ".env"

if (-not $HostAddress) {
    $HostAddress = $env:BACKEND_HOST
    if (-not $HostAddress) {
        $HostAddress = Get-DotEnvValue -FilePath $envFile -Key "BACKEND_HOST"
    }
    if (-not $HostAddress) {
        $HostAddress = "127.0.0.1"
    }
}

if ($Port -le 0) {
    $rawPort = $env:BACKEND_PORT
    if (-not $rawPort) {
        $rawPort = Get-DotEnvValue -FilePath $envFile -Key "BACKEND_PORT"
    }
    if ($rawPort -and ($rawPort -as [int]) -gt 0) {
        $Port = [int]$rawPort
    } else {
        $Port = 8000
    }
}

$uvicornRel = ".\.venv\Scripts\uvicorn.exe"
$uvicornFull = Join-Path $RepoRoot ".venv\Scripts\uvicorn.exe"
if (-not (Test-Path -LiteralPath $uvicornFull)) {
    Write-Error "Virtual environment uvicorn not found at '$uvicornFull'. Create .venv and run 'pip install -r backend\requirements.txt' first."
    exit 1
}

$uvicornArgs = @(
    "backend.app.main:app",
    "--host", $HostAddress,
    "--port", "$Port"
)
if (-not $NoReload) {
    $uvicornArgs += "--reload"
}

Push-Location $RepoRoot
try {
    if ($SkipBws) {
        & $uvicornRel @uvicornArgs
        exit $LASTEXITCODE
    }

    if (-not (Get-Command bws -ErrorAction SilentlyContinue)) {
        Write-Error "Bitwarden Secrets Manager CLI ('bws') not found on PATH. Install it with: winget install -e --id Bitwarden.BWS (then reopen your terminal), or run with -SkipBws."
        exit 1
    }

    if (-not $env:BWS_ACCESS_TOKEN) {
        Write-Error "BWS_ACCESS_TOKEN is not set. Set it as a Windows User environment variable (not inside .env) and reopen your terminal."
        exit 1
    }

    if (-not $ProjectId) {
        $ProjectId = $env:BWS_PROJECT_ID
    }
    if (-not $ProjectId) {
        $ProjectId = Get-DotEnvValue -FilePath $envFile -Key "BWS_PROJECT_ID"
    }
    if (-not $ProjectId) {
        Write-Error "Bitwarden project ID not provided. Pass -ProjectId <UUID>, set `$env:BWS_PROJECT_ID, or add BWS_PROJECT_ID=<UUID> to .env."
        exit 1
    }

    & bws run --project-id $ProjectId -- $uvicornRel @uvicornArgs
    exit $LASTEXITCODE
} finally {
    Pop-Location
}
