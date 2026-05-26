# Install

All install steps and terminal output are English by default.

## Local Development Install

For a beginner-friendly Git install, run one command from the folder where the
project should be downloaded.

Recommended Docker mode for normal users:

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/Denis-Denchev/dmdcore/main/scripts/bootstrap.sh | bash -s -- --docker
```

Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=irm https://raw.githubusercontent.com/Denis-Denchev/dmdcore/main/scripts/bootstrap.ps1; & ([scriptblock]::Create($s)) -Docker"
```

Open:

```text
http://127.0.0.1:8765
```

This starts Docker Compose with:

- DMD Agent backend and built dashboard in one container
- Ollama in a second container
- persistent app data in Docker volumes
- `./workspace` mounted as the only normal working folder
- default model `qwen3:4b`, pulled on first start

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/Denis-Denchev/dmdcore/main/scripts/bootstrap.sh | bash
cd dmdcore
./start web
```

Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Denis-Denchev/dmdcore/main/scripts/bootstrap.ps1 | iex"
cd dmdcore
.\start.ps1 web
```

The bootstrap script clones the repo, creates `.venv`, installs Python package
dependencies, initializes local app data, runs the first-time wizard, and runs
`npm install` for the dashboard when Node.js/npm is available.

Required system runtimes:

- Git
- Python 3.11+
- Node.js/npm for the dashboard
- Ollama for local models, unless an API provider is selected

Docker mode only requires Git and Docker Desktop / Docker Engine. The app
containers provide Python, Node-built dashboard assets, backend dependencies,
Ollama, and the default model pull.

If Node.js is not installed, terminal chat still works:

```bash
./start session
```

On Windows:

```powershell
.\start.ps1 session
```

## Manual Local Development Install

```bash
git clone https://github.com/Denis-Denchev/dmdcore.git
cd dmdcore
./scripts/install.sh
```

The script:

- checks Python
- creates `.venv`
- installs the package in editable mode
- creates local app directories
- creates `~/.local/share/dmdcore/config.yaml`
- runs the hardware/model wizard

After install:

```bash
./start session
```

`./start session` manages `.venv` for you and opens the terminal chat. On first
run it asks for the assistant name, your name, response language, and whether to
use a local model or an API provider. Type `/panel` inside chat for Telegram,
permissions, logs, model, and approval controls.

Optional shortcut:

```bash
./start install-command
```

After opening a new terminal:

```bash
start session
```

## Ollama

Install Ollama from:

```text
https://ollama.com/download
```

Pull the recommended model:

```bash
start model fast --pull
```

If Ollama is not already running:

```bash
start web
```

The wizard recommends a model based on detected system RAM.
The terminal chat can also select the first model during `start session`.

For faster responses on 8GB or 16GB machines:

```bash
start model light --pull
```

## One-Terminal Start

Terminal chat only:

```bash
start session
```

On Windows PowerShell:

```powershell
.\start.ps1 session
```

Ask once:

```bash
start ask "What can you do?"
```

On Windows PowerShell:

```powershell
.\start.ps1 ask "What can you do?"
```

Start API and dashboard together:

```bash
start web
```

On Windows PowerShell:

```powershell
.\start.ps1 web
```

Stop both with `Ctrl+C`.

Open the dashboard:

```bash
start open
```

## API

Developer-only API start:

```bash
dmdcore serve
```

Default URL:

```text
http://127.0.0.1:8765
```

Health check:

```bash
curl http://127.0.0.1:8765/health
```

## Web Dashboard

Normal users should use:

```bash
start web
```

Open:

```text
http://127.0.0.1:5174
```

The dashboard includes chat, tools, approvals, memory, audit logs, and model settings.
