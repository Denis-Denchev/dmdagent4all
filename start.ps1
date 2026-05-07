param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$Arguments
)

$ErrorActionPreference = "Stop"

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$VenvPython = Join-Path $RootDir ".venv\Scripts\python.exe"
$DmdAgent = Join-Path $RootDir ".venv\Scripts\dmdagent.exe"

function Get-PythonCommand {
    $candidates = @(
        @{ Exe = "py"; Args = @("-3.13") },
        @{ Exe = "py"; Args = @("-3.12") },
        @{ Exe = "py"; Args = @("-3.11") },
        @{ Exe = "python"; Args = @() },
        @{ Exe = "python3"; Args = @() }
    )
    foreach ($candidate in $candidates) {
        if (-not (Get-Command $candidate.Exe -ErrorAction SilentlyContinue)) {
            continue
        }
        & $candidate.Exe @($candidate.Args + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3, 11) else 1)")) *> $null
        if ($LASTEXITCODE -eq 0) {
            return $candidate
        }
    }
    return $null
}

function Invoke-BasePython {
    param(
        [hashtable]$PythonCommand,
        [string[]]$PythonArguments
    )
    & $PythonCommand.Exe @($PythonCommand.Args + $PythonArguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed."
    }
}

function Ensure-Package {
    if (-not (Test-Path $VenvPython)) {
        $PythonCommand = Get-PythonCommand
        if (-not $PythonCommand) {
            Write-Error "Python 3.11+ is required. Install Python from https://www.python.org/downloads/ and run this again."
            exit 1
        }
        Write-Host "Creating local Python environment..."
        Invoke-BasePython $PythonCommand @("-m", "venv", ".venv")
    }
    if (-not (Test-Path $DmdAgent)) {
        Write-Host "Installing DMD Agent locally..."
        & $VenvPython -m pip install --upgrade pip
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
        & $VenvPython -m pip install -e .
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    }
}

function Print-Help {
    Write-Host @"
DMD Agent 4 All launcher

Beginner commands from the project folder:
  .\start.ps1 session              Open terminal chat
  .\start.ps1 ask "hello"          Ask once and exit
  .\start.ps1 web                  Start API + dashboard + Telegram if configured
  .\start.ps1 model fast --pull    Set and pull the default local model
  .\start.ps1 model light --pull   Set and pull a smaller local model
  .\start.ps1 doctor               Check health/security
"@
}

Set-Location $RootDir

if (-not $Arguments -or $Arguments.Count -eq 0) {
    $CommandName = "session"
    $Rest = @()
} else {
    $CommandName = $Arguments[0]
    $Rest = @()
    if ($Arguments.Count -gt 1) {
        $Rest = $Arguments[1..($Arguments.Count - 1)]
    }
}

switch ($CommandName) {
    { $_ -in @("help", "-h", "--help") } {
        Print-Help
        exit 0
    }
    { $_ -in @("session", "chat") } {
        Ensure-Package
        & $DmdAgent chat @Rest
        exit $LASTEXITCODE
    }
    "ask" {
        Ensure-Package
        & $DmdAgent ask @Rest
        exit $LASTEXITCODE
    }
    { $_ -in @("web", "dashboard") } {
        Ensure-Package
        & $DmdAgent start @Rest
        exit $LASTEXITCODE
    }
    "model" {
        Ensure-Package
        & $DmdAgent model @Rest
        exit $LASTEXITCODE
    }
    { $_ -in @("doctor", "status", "open") } {
        Ensure-Package
        & $DmdAgent $CommandName @Rest
        exit $LASTEXITCODE
    }
    default {
        Ensure-Package
        & $DmdAgent $CommandName @Rest
        exit $LASTEXITCODE
    }
}
