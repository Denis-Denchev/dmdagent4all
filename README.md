# DMD Agent 4 All

Open-source local AI control center for personal computers and small servers.

DMD Agent 4 All is designed around one core rule:

> The model is never trusted for security.

The local model can propose structured tool calls, but the backend decides what is allowed. The model never receives direct operating system access, raw tokens, passwords, `.env` files, SSH keys, or a shell.

## What It Is

DMD Agent 4 All is a local-first personal AI assistant that can be installed on machines such as Mac mini devices. Users interact through a web dashboard, CLI, and later locked remote channels such as Telegram or WhatsApp. Every connector and tool is explicitly enabled by the user.

```text
User Interfaces
Web UI / Telegram / CLI
        |
Agent Core
Planner / Router / Memory
        |
Permission Engine
Risk levels / Approval / Logs
        |
Tool Layer
Gmail / Calendar / Terminal / Files / Browser / GitHub
        |
Sandbox / Connectors
Docker / OAuth / Local APIs
```

## Current Status

This repository is in early foundation stage. The first public milestone focuses on:

- security model and threat model
- local storage layout
- permission engine
- tool manifests
- audit logging
- Markdown memory
- local model presets
- CLI install wizard
- local Ollama planner loop
- approval queue basics
- React web dashboard MVP
- FastAPI backend skeleton

High-risk connectors such as terminal, browser automation, email sending, deployment, SSH, and production access are disabled by default.

## Install For Local Development

```bash
git clone https://github.com/Denis-Denchev/dmdagent4all.git
cd dmdagent4all
./scripts/install.sh
```

Run the wizard:

```bash
dmdagent wizard
```

Run the API:

```bash
dmdagent serve
```

Run the web dashboard:

```bash
cd frontend
npm install
npm run dev
```

Open:

```text
http://localhost:5174
```

## CLI

```bash
dmdagent status
dmdagent wizard
dmdagent chat "Show my local memory files"
dmdagent models list
dmdagent tools list
dmdagent memory path
dmdagent memory list
dmdagent approvals list
dmdagent approvals approve <id>
```

## Local Chat

Start Ollama first, then pull the recommended model:

```bash
ollama serve
ollama pull qwen3:8b
```

Send a message:

```bash
dmdagent chat "Show my local memory files"
```

The model can only propose a structured tool request. The backend still validates the tool, risk level, permissions, approval requirement, and cloud-context policy before anything runs.

For slower machines, switch to Light Mode:

```bash
dmdagent models set-mode light
ollama pull qwen3:4b
```

This also switches the planner model to the selected lightweight model.

## Model Presets

| Mode | Model |
|---|---|
| Light Mode | `qwen3:4b` or `phi4-mini` |
| Fast Mode | `qwen3:8b` |
| Balanced Mode | `qwen3:14b` |
| Power Mode | `qwen3:30b` |

Mac mini recommendations are documented in [docs/model-presets.md](docs/model-presets.md).

## Security Principles

- The LLM is an untrusted planner, not a security boundary.
- Tools are explicit, named backend capabilities.
- Every tool has a manifest with risk level, permissions, approval requirements, and cloud-context rules.
- Secrets are stored through OS secret stores where possible, never exposed to the model.
- Markdown memory is the source of truth. Vector indexes are optional derived data.
- Terminal and browser automation are disabled by default.
- Cloud models are disabled by default for private connector data.

Read:

- [SECURITY.md](SECURITY.md)
- [PRIVACY.md](PRIVACY.md)
- [THREAT_MODEL.md](THREAT_MODEL.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/permissions.md](docs/permissions.md)

## Language Policy

All install flow, terminal output, documentation, approval text, and product UI copy are English by default.

The assistant can answer in any language supported by the selected model. The default response mode is auto-detect from the user's message.

## License

MIT
