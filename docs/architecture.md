# Architecture

DMD Agent 4 All is a local AI control center. It is not an AI process with unrestricted computer access.

## Runtime Flow

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

## Agent Core

The agent core:

- receives user messages
- loads approved local memory
- calls the selected LLM provider
- interprets structured tool requests
- sends tool requests to the permission engine
- returns sanitized tool results to the user

The core does not bypass the permission engine.

## LLM Provider Layer

Providers are isolated behind a common interface:

- Ollama provider for local models
- OpenAI provider later
- Anthropic provider later
- OpenRouter provider later

Default:

```yaml
llm:
  provider: ollama
  model: qwen3:8b
```

Cloud providers require explicit user setup and privacy approvals before private connector context is sent.

## Permission Engine

Every tool has a manifest:

```json
{
  "name": "gmail.summarize_inbox",
  "description": "Summarize recent Gmail messages.",
  "risk": 1,
  "permissions": ["gmail.readonly"],
  "approval_required": false,
  "cloud_allowed": false,
  "default_enabled": false
}
```

The backend decides whether a request is allowed, denied, or requires approval.

## Tool Layer

Tools are small named backend modules. The model cannot invent new capabilities.

Initial tool groups:

- `memory.*`
- `system.*`
- `gmail.*` manifests
- `calendar.*` manifests
- `terminal.*` manifest
- `browser.*` manifests

High-risk tools are disabled by default.

## Storage

Default path:

```text
~/.local/share/dmdagent4all/
  config.yaml
  memory/
  workspace/
  logs/
  audit.db
  vector_index/
```

SQLite stores audit logs, approvals, tool calls, connector status, LLM metadata, and memory events.

Markdown files are the source of truth for user memory.
