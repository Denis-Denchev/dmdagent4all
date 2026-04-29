# Install

All install steps and terminal output are English by default.

## Local Development Install

```bash
git clone https://github.com/Denis-Denchev/dmdagent4all.git
cd dmdagent4all
./scripts/install.sh
```

The script:

- checks Python
- creates `.venv`
- installs the package in editable mode
- creates local app directories
- creates `~/.local/share/dmdagent4all/config.yaml`
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

Ask once:

```bash
start ask "What can you do?"
```

Start API and dashboard together:

```bash
start web
```

Stop both with `Ctrl+C`.

Open the dashboard:

```bash
start open
```

## API

Developer-only API start:

```bash
dmdagent serve
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
