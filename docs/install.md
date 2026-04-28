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
source .venv/bin/activate
dmdagent status
dmdagent wizard
dmdagent chat "Show my local memory files"
dmdagent models list
dmdagent serve
```

## Ollama

Install Ollama from:

```text
https://ollama.com/download
```

Pull the recommended model:

```bash
ollama pull qwen3:8b
```

If Ollama is not already running:

```bash
ollama serve
```

The wizard recommends a model based on detected system RAM.

For faster responses on 8GB or 16GB machines:

```bash
dmdagent models set-mode light
ollama pull qwen3:4b
```

## API

Start the API:

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

Start the API in one terminal:

```bash
source .venv/bin/activate
dmdagent serve
```

Start the dashboard in another terminal:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5174
```

The dashboard includes chat, tools, approvals, memory, audit logs, and model settings.
