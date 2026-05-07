param(
    [string]$InstallDir = $(if ($env:DMDAGENT_INSTALL_DIR) { $env:DMDAGENT_INSTALL_DIR } else { "dmdagent4all" }),
    [string]$RepoUrl = $(if ($env:DMDAGENT_REPO_URL) { $env:DMDAGENT_REPO_URL } else { "https://github.com/Denis-Denchev/dmdagent4all.git" }),
    [switch]$Docker,
    [switch]$Detached
)

$ErrorActionPreference = "Stop"

function Write-Step {
    param([string]$Message)
    Write-Host $Message
}

function Fail {
    param([string]$Message)
    Write-Error $Message
    exit 1
}

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
        [string[]]$Arguments
    )
    & $PythonCommand.Exe @($PythonCommand.Args + $Arguments)
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed."
    }
}

$Current = Get-Location
if ((Test-Path "pyproject.toml") -and (Test-Path "src/dmdagent4all")) {
    $ProjectDir = $Current.Path
} else {
    if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
        Fail "git is required to clone the project. Install Git for Windows, then run this command again."
    }
    if (Test-Path (Join-Path $InstallDir ".git")) {
        Write-Step "Updating existing checkout: $InstallDir"
        git -C $InstallDir pull --ff-only
        if ($LASTEXITCODE -ne 0) { Fail "git pull failed." }
    } elseif (Test-Path $InstallDir) {
        Fail "$InstallDir already exists and is not a git checkout. Choose another folder with DMDAGENT_INSTALL_DIR."
    } else {
        Write-Step "Cloning $RepoUrl into $InstallDir"
        git clone $RepoUrl $InstallDir
        if ($LASTEXITCODE -ne 0) { Fail "git clone failed." }
    }
    $ProjectDir = (Resolve-Path $InstallDir).Path
}

Set-Location $ProjectDir

if ($Docker) {
    if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
        Fail "Docker is required for -Docker. Install Docker Desktop, then run this again."
    }
    New-Item -ItemType Directory -Force -Path "workspace" | Out-Null
    Write-Step "Starting Docker stack: DMD Agent + Ollama + local model"
    docker compose version *> $null
    if ($LASTEXITCODE -ne 0) {
        Fail "docker compose is required. Install/update Docker Desktop, then run this again."
    }
    if ($Detached) {
        docker compose up --build -d
        if ($LASTEXITCODE -ne 0) { Fail "docker compose up failed." }
        Write-Step "Dashboard: http://127.0.0.1:8765"
        exit 0
    }
    docker compose up --build
    exit $LASTEXITCODE
}

$PythonCommand = Get-PythonCommand
if (-not $PythonCommand) {
    Fail "Python 3.11+ is required. Install Python from https://www.python.org/downloads/ and run this again."
}

Write-Step "Using Python:"
Invoke-BasePython $PythonCommand @("-c", "import sys; print(sys.version.split()[0])")

$VenvPython = Join-Path $ProjectDir ".venv\Scripts\python.exe"
$DmdAgent = Join-Path $ProjectDir ".venv\Scripts\dmdagent.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Step "Creating local Python environment: .venv"
    Invoke-BasePython $PythonCommand @("-m", "venv", ".venv")
}

Write-Step "Installing Python package dependencies"
& $VenvPython -m pip install --upgrade pip
if ($LASTEXITCODE -ne 0) { Fail "pip upgrade failed." }
& $VenvPython -m pip install -e .
if ($LASTEXITCODE -ne 0) { Fail "Python package install failed." }

Write-Step "Initializing local app data"
& $DmdAgent init
if ($LASTEXITCODE -ne 0) { Fail "dmdagent init failed." }

if (Test-Path "frontend") {
    if (Get-Command npm -ErrorAction SilentlyContinue) {
        Write-Step "Installing dashboard packages"
        Push-Location "frontend"
        npm install
        if ($LASTEXITCODE -ne 0) { Fail "npm install failed." }
        Pop-Location
    } else {
        Write-Step "npm was not found. Terminal chat will work; dashboard needs Node.js/npm from https://nodejs.org/"
    }
}

if ($env:DMDAGENT_SKIP_WIZARD -ne "1") {
    Write-Step "Running first-time setup wizard"
    & $DmdAgent wizard
}

Write-Step ""
Write-Step "Install complete."
Write-Step "Project folder:"
Write-Step "  $ProjectDir"
Write-Step ""
Write-Step "Start from this folder in PowerShell:"
Write-Step "  .\start.ps1 session"
Write-Step "  .\start.ps1 web"
Write-Step ""
Write-Step "Or from cmd.exe:"
Write-Step "  start.cmd session"
Write-Step "  start.cmd web"
