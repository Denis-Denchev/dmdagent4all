# DMD Sentinel one-line installer for Windows.
# (Formerly DMD Core — Python package and env var names kept as `dmdcore` / `DMDCORE_*` for backwards compatibility.)
#
# Usage (PowerShell):
#   irm https://raw.githubusercontent.com/Denis-Denchev/dmd-sentinel/main/install.ps1 | iex
#
# Optional environment variables:
#   DMDCORE_INSTALL_DIR   Target install folder (default: $HOME\dmd-sentinel)
#   DMDCORE_REPO_URL      Source git URL (default: official repo)
#   DMDCORE_SKIP_OLLAMA   Set to 1 to skip Ollama installation
#   DMDCORE_SKIP_LAUNCH   Set to 1 to skip auto-launching the dashboard
#   DMDCORE_SKIP_WIZARD   Set to 1 to skip the interactive first-time wizard

$ErrorActionPreference = "Stop"

$RepoUrl    = if ($env:DMDCORE_REPO_URL) { $env:DMDCORE_REPO_URL } else { "https://github.com/Denis-Denchev/dmd-sentinel.git" }
$InstallDir = if ($env:DMDCORE_INSTALL_DIR) { $env:DMDCORE_INSTALL_DIR } else { Join-Path $HOME "dmd-sentinel" }
$SkipOllama = ($env:DMDCORE_SKIP_OLLAMA -eq "1")
$SkipLaunch = ($env:DMDCORE_SKIP_LAUNCH -eq "1")

function Step($msg)  { Write-Host ""; Write-Host "== $msg" -ForegroundColor Cyan }
function Info($msg)  { Write-Host "   $msg" }
function Warn($msg)  { Write-Host "   warn: $msg" -ForegroundColor Yellow }
function Fail($msg)  { Write-Host "error: $msg" -ForegroundColor Red; exit 1 }

function Have($cmd)  { [bool](Get-Command $cmd -ErrorAction SilentlyContinue) }

function Refresh-PathFromMachine {
    $userPath    = [Environment]::GetEnvironmentVariable("Path", "User")
    $machinePath = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $env:Path    = "$machinePath;$userPath"
}

function Ensure-Winget {
    if (Have winget) { return }
    Fail "winget is not available. Update Windows (Microsoft Store > App Installer) and re-run."
}

function Winget-Install($id) {
    Info "winget install $id"
    & winget install --exact --silent --accept-package-agreements --accept-source-agreements --id $id
    if ($LASTEXITCODE -ne 0 -and $LASTEXITCODE -ne -1978335189) {
        Fail "winget install $id failed (exit $LASTEXITCODE)"
    }
    Refresh-PathFromMachine
}

function Python-Ok($exe, [string[]]$prefix = @()) {
    if (-not (Have $exe)) { return $false }
    & $exe @($prefix + @("-c", "import sys; raise SystemExit(0 if sys.version_info >= (3,11) else 1)")) *> $null
    return ($LASTEXITCODE -eq 0)
}

function Find-Python {
    if (Python-Ok "py" @("-3.13")) { return @{ Exe = "py"; Args = @("-3.13") } }
    if (Python-Ok "py" @("-3.12")) { return @{ Exe = "py"; Args = @("-3.12") } }
    if (Python-Ok "py" @("-3.11")) { return @{ Exe = "py"; Args = @("-3.11") } }
    if (Python-Ok "python")        { return @{ Exe = "python"; Args = @() } }
    if (Python-Ok "python3")       { return @{ Exe = "python3"; Args = @() } }
    return $null
}

function Ensure-Python {
    $p = Find-Python
    if ($p) {
        Info ("Python found: " + (& $p.Exe @($p.Args + @("-c", "import sys; print(sys.version.split()[0])"))))
        return
    }
    Step "Installing Python 3.13"
    Winget-Install "Python.Python.3.13"
    $p = Find-Python
    if (-not $p) { Fail "Python install completed but no compatible interpreter is on PATH. Open a new PowerShell and re-run." }
}

function Ensure-Git {
    if (Have git) { return }
    Step "Installing Git"
    Winget-Install "Git.Git"
    if (-not (Have git)) { Fail "Git install completed but `git` is not on PATH. Open a new PowerShell and re-run." }
}

function Ensure-Node {
    if ((Have node) -and (Have npm)) {
        Info ("Node.js found: " + (node --version))
        return
    }
    Step "Installing Node.js LTS"
    Winget-Install "OpenJS.NodeJS.LTS"
    if (-not (Have node)) { Warn "Node install completed but `node` is not on PATH. Dashboard UI will be skipped." }
}

function Ensure-Ollama {
    if ($SkipOllama) { Info "Skipping Ollama (DMDCORE_SKIP_OLLAMA=1)"; return }
    if (Have ollama) {
        Info ("Ollama found: " + ((ollama --version 2>$null) | Select-Object -First 1))
        return
    }
    Step "Installing Ollama (local LLM runtime)"
    try {
        Winget-Install "Ollama.Ollama"
    } catch {
        Warn "winget install Ollama failed; falling back to the official installer download."
        $url = "https://ollama.com/download/OllamaSetup.exe"
        $tmp = Join-Path $env:TEMP "OllamaSetup.exe"
        Invoke-WebRequest -Uri $url -OutFile $tmp
        Start-Process -FilePath $tmp -ArgumentList "/SILENT" -Wait
        Refresh-PathFromMachine
    }
}

function Clone-Or-Update-Repo {
    $isCheckout = (Test-Path (Join-Path $InstallDir "pyproject.toml")) -and (Test-Path (Join-Path $InstallDir "src\dmdcore"))
    if ($isCheckout) {
        Step "Updating existing checkout at $InstallDir"
        if (Test-Path (Join-Path $InstallDir ".git")) {
            try { git -C $InstallDir pull --ff-only } catch { Warn "git pull failed; continuing." }
        }
        return
    }
    if (Test-Path $InstallDir) {
        Fail "$InstallDir exists but is not a DMD Core checkout. Move it or set DMDCORE_INSTALL_DIR."
    }
    Step "Cloning DMD Core into $InstallDir"
    git clone $RepoUrl $InstallDir
    if ($LASTEXITCODE -ne 0) { Fail "git clone failed." }
}

Step "DMD Core installer (Windows)"

$ScriptDir = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
if ((Test-Path (Join-Path $ScriptDir "pyproject.toml")) -and (Test-Path (Join-Path $ScriptDir "src\dmdcore"))) {
    Info "Detected existing DMD Core checkout at $ScriptDir; using it instead of cloning."
    $InstallDir = $ScriptDir
}

Ensure-Winget
Ensure-Git
Ensure-Python
Ensure-Node
Ensure-Ollama
Clone-Or-Update-Repo

Set-Location $InstallDir

Step "Running project bootstrap"
$env:DMDCORE_INSTALL_SHORTCUT = "0"
& powershell -NoProfile -ExecutionPolicy Bypass -File "scripts\bootstrap.ps1"
if ($LASTEXITCODE -ne 0) { Fail "bootstrap.ps1 failed (exit $LASTEXITCODE)." }

if (-not $SkipLaunch) {
    Step "Launching DMD Core"
    Start-Job -ScriptBlock {
        Start-Sleep -Seconds 4
        Start-Process "http://127.0.0.1:5174/"
    } | Out-Null
    Info "If the browser does not open automatically, visit http://127.0.0.1:5174"
    Info "Press Ctrl+C to stop."
    & ".\start.ps1" web
    exit $LASTEXITCODE
}

Write-Host ""
Write-Host "DMD Core install finished." -ForegroundColor Green
Write-Host "Project folder: $InstallDir"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  cd `"$InstallDir`""
Write-Host "  .\start.ps1 web         # dashboard + API"
Write-Host "  .\start.ps1 session     # terminal chat"
