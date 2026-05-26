# DMD Agent 4 All

Open-source, local-first AI control center for personal computers, developer
workstations, Mac minis, and small private servers.

Current backend/package version: `1.0.0`.

Current dashboard visual generation: `v2.0 cockpit UI`.

This README is intentionally detailed. Treat it as:

- product documentation
- developer onboarding
- architecture notes
- safety policy
- handoff context for the next coding agent

The goal is that a junior developer can open this repository, read this file,
and understand what the project does, how requests move through the code, which
parts are security critical, how to run it, and how to extend it without
breaking the local-first safety model.

## Table Of Contents

- [What This Project Is](#what-this-project-is)
- [What Changed Recently](#what-changed-recently)
- [Non-Negotiable Safety Contract](#non-negotiable-safety-contract)
- [Mental Model](#mental-model)
- [Architecture](#architecture)
- [Request Flow](#request-flow)
- [Repository Map](#repository-map)
- [Runtime Data Locations](#runtime-data-locations)
- [Installation](#installation)
- [Running The App](#running-the-app)
- [CLI Cheat Sheet](#cli-cheat-sheet)
- [Dashboard v2 Cockpit UI](#dashboard-v2-cockpit-ui)
- [Configuration](#configuration)
- [Models And Providers](#models-and-providers)
- [Tools, Permissions, Risk, And Approvals](#tools-permissions-risk-and-approvals)
- [Email System](#email-system)
- [Telegram Interface](#telegram-interface)
- [Terminal Safety](#terminal-safety)
- [Browser Tools](#browser-tools)
- [Workspace And Files](#workspace-and-files)
- [Memory](#memory)
- [Reminders](#reminders)
- [Calendar](#calendar)
- [Emergency Stop](#emergency-stop)
- [API Surface](#api-surface)
- [Frontend Code Guide](#frontend-code-guide)
- [Backend Code Guide](#backend-code-guide)
- [Testing And Verification](#testing-and-verification)
- [Common Debugging Paths](#common-debugging-paths)
- [How To Add A New Tool](#how-to-add-a-new-tool)
- [Known Limitations And Next Work](#known-limitations-and-next-work)
- [Handoff Prompt For The Next Agent](#handoff-prompt-for-the-next-agent)
- [License](#license)

## What This Project Is

DMD Agent 4 All is a private AI operations cockpit.

It runs locally and gives the user one place to control:

- chat with the agent
- local and cloud model selection
- tool execution
- approvals for risky actions
- terminal command policy
- local memory files
- workspace files and downloads
- Gmail and Outlook email connectors
- Telegram remote access
- audit logs
- emergency stop
- configuration and safety settings

The important product idea is:

> The model can plan, but backend code decides what is allowed.

The LLM is not allowed to directly control the operating system. It can ask the
backend to run registered tools. The backend checks every tool request against
manifests, permissions, risk levels, workspace policy, approval state, and
emergency stop state.

## What Changed Recently

This repository has moved beyond the original MVP. The current state includes
several important updates that a new developer should know before making
changes.

### Dashboard v2 Cockpit Redesign

The React dashboard has been redesigned from a flat admin-style interface into a
dark, professional local agent cockpit.

The v2 UI adds:

- command-deck left sidebar
- product identity block: `DMD Agent 4 All`, `Local-first command system`, `v2.0`
- bottom sidebar runtime/status block
- top cockpit status strip
- clear Danger Mode and Emergency Stop controls
- redesigned chat message cards
- tool trace metadata inside agent messages
- email approval preview blocks
- professional approval cards with risk badges
- right-side AI core panel
- recent activity feed
- tools overview grid
- models runtime control view
- config control center styling
- responsive layout for desktop, laptop, tablet, and mobile

New reusable frontend components live in:

```text
frontend/src/components/cockpit.tsx
```

They include:

- `AppShell`
- `Sidebar`
- `TopStatusBar`
- `StatusBadge`
- `CommandCard`
- `ApprovalCard`
- `ToolCard`
- `AgentCorePanel`
- `ActivityFeed`
- `ChatMessage`
- `ToolTrace`

Theme tokens and v2 styling live in:

```text
frontend/src/styles.css
```

The v2 redesign did not intentionally change backend API contracts, routes,
tool execution flow, approval logic, or state management behavior.

### Email Send Preview And Approval Flow

The email workflow now supports the safer product behavior the user wanted:

1. User asks the agent to write and send an email.
2. Backend creates a local draft.
3. Backend immediately creates a send approval when the user clearly requested
   sending.
4. Chat shows a preview before sending:
   - recipient
   - subject
   - body preview
5. User presses `Approve`.
6. Backend sends the stored local draft through Gmail or Outlook SMTP.

This matters because the user should see the generated message before any real
email leaves the machine.

Relevant backend logic is in:

```text
src/dmdcore/agent/core.py
src/dmdcore/tools/email_connector.py
src/dmdcore/server.py
```

Relevant frontend preview rendering is in:

```text
frontend/src/components/cockpit.tsx
```

### Gmail App Password Whitespace Fix

Google app passwords are often copied with spaces or non-breaking spaces. A real
failure was seen:

```text
'ascii' codec can't encode character '\xa0'
```

The backend now normalizes Gmail app passwords and login values before IMAP/SMTP
auth. This prevents hidden NBSP characters from breaking SMTP login.

Relevant code:

```text
src/dmdcore/tools/email_connector.py
src/dmdcore/server.py
tests/test_dashboard_controls.py
```

### Gmail OAuth Support

The codebase now contains Google OAuth helpers and dashboard/API endpoints for a
Gmail OAuth flow.

Important files:

```text
src/dmdcore/email_oauth.py
src/dmdcore/server.py
src/dmdcore/tools/email_connector.py
```

Important endpoints:

```text
POST /v1/email/oauth/google/client-secret
POST /v1/email/oauth/google/start
GET  /v1/email/oauth/google/callback
POST /v1/email/oauth/google/disconnect
```

OAuth secrets are not written into `config.yaml`. They are loaded from
environment variables or local secret stores.

## Non-Negotiable Safety Contract

This section is the most important part of the project.

Do not weaken these rules without a deliberate security review.

### Core Rule

The LLM is an untrusted planner. It is never a security boundary.

### What The LLM May Do

The LLM may:

- answer normal questions
- ask clarifying questions
- propose a registered tool call
- propose a short multi-tool plan
- summarize safe tool output
- route memory searches
- help draft text before the user approves an action

### What The LLM Must Not Do

The LLM must not:

- invent backend tool names
- bypass tool manifests
- directly execute shell commands
- directly read arbitrary files
- directly access `.env`, private keys, OAuth tokens, SSH keys, or credentials
- decide by itself that a risky action is safe
- send email without approval
- submit browser forms without approval
- delete files without approval
- run terminal commands without backend policy and approval
- expose secrets in prompts, logs, or model context

### Backend-Enforced Rules

Backend code must enforce:

- registered tool manifests only
- enabled/disabled tool state
- granted permissions
- risk threshold
- approval queue
- workspace path safety
- secret path blocking
- command allowlists
- timeout and output limits
- redaction
- audit logging
- emergency stop

Read these security documents before changing policy:

- [SECURITY.md](SECURITY.md)
- [PRIVACY.md](PRIVACY.md)
- [THREAT_MODEL.md](THREAT_MODEL.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/permissions.md](docs/permissions.md)
- [docs/product-runtime.md](docs/product-runtime.md)
- [docs/connectors.md](docs/connectors.md)

## Mental Model

Think of the app as five layers.

```text
1. Interface layer
   Web dashboard, terminal CLI, Telegram

2. Agent layer
   AgentCore, planner, deterministic command handling, response shaping

3. Policy layer
   ToolSafetyPolicy, PermissionEngine, approval queue, emergency stop

4. Tool layer
   Files, memory, browser, terminal, email, reminders, calendar, workspace

5. Storage/runtime layer
   Local config, SQLite audit DB, workspace files, memory files, local drafts,
   local secret store, environment variables, Ollama or API provider
```

A junior developer should remember:

> UI and CLI send requests. AgentCore interprets requests. Policy decides what
> can happen. Tools execute only after policy allows them.

## Architecture

High-level architecture:

```text
User
 |
 | Web dashboard / CLI / Telegram
 v
AgentCore
 |
 | deterministic controls
 | LLM structured planner
 | fallback router
 v
ToolSafetyPolicy
 |
 | path checks
 | secret checks
 | destructive action checks
 | emergency stop checks
 v
PermissionEngine
 |
 | enabled tool?
 | required permission granted?
 | risk requires approval?
 | cloud/private context allowed?
 v
Approval Queue / Tool Executor
 |
 | pending approval or execution
 v
Tool Result
 |
 | audit event
 | optional answer synthesis
 v
User response
```

Main backend package:

```text
src/dmdcore/
```

Main frontend package:

```text
frontend/
```

## Request Flow

This is the normal request path when a user types into the dashboard chat.

1. User submits text in `frontend/src/App.tsx`.
2. Frontend calls `POST /v1/chat` or `POST /v1/chat/stream`.
3. FastAPI route in `src/dmdcore/server.py` forwards the message to
   `AgentCore`.
4. `AgentCore` checks minimal deterministic controls first:
   - approve/deny
   - emergency stop/reset
   - explicit slash-like commands
   - setup/help style commands
   - direct JSON tool requests
5. For normal requests, `AgentCore` asks the planner for a structured decision.
6. Planner returns one of:
   - final answer
   - tool call
   - multi-tool plan
   - clarification question
   - memory search
7. A tool request goes through backend safety policy.
8. The permission engine checks manifest, risk, permission, and approval rules.
9. If safe and allowed, tool executes.
10. If approval is needed, a pending approval is stored in SQLite.
11. Frontend displays approval card.
12. User approves or denies.
13. Approval route executes or cancels the stored action.
14. Audit log records what happened.
15. User sees the final result.

## Repository Map

This section explains the important files and folders.

```text
.
├── README.md
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
├── start
├── start.ps1
├── start.cmd
├── config/
├── docs/
├── frontend/
├── scripts/
├── src/dmdcore/
└── tests/
```

### Root Files

`README.md`

This file. It is the main project handoff document.

`pyproject.toml`

Python package metadata. It defines:

- package name: `dmdcore`
- version: `1.0.0`
- Python requirement: `>=3.11`
- runtime dependencies:
  - FastAPI
  - PyYAML
  - Uvicorn
- optional browser dependency:
  - Playwright
- console script:
  - `dmdcore = dmdcore.cli:main`

`start`

macOS/Linux convenience launcher. It creates `.venv` if needed, installs the
package in editable mode if needed, and forwards friendly commands to
`dmdcore`.

Examples:

```bash
./start session
./start ask "hello"
./start web
./start model fast --pull
./start doctor
```

`start.ps1`

Windows PowerShell launcher.

`start.cmd`

Windows command launcher shim.

`Dockerfile`

Builds the production container. It:

1. builds the Vite dashboard in a Node stage
2. installs the Python package in a Python slim image
3. copies the built dashboard into the image
4. exposes port `8765`

`docker-compose.yml`

Runs:

- `dmdcore` container
- `ollama` container

It mounts `./workspace` as the safe user workspace and stores application data
in Docker volumes.

`.env.example`

Example environment variables. Never commit real secrets.

### `config/`

`config/default.yaml`

Human-readable default config file used as reference. The real runtime config is
created in the local app data directory.

### `docs/`

Additional project notes:

- `docs/architecture.md`
- `docs/connectors.md`
- `docs/install.md`
- `docs/model-presets.md`
- `docs/permissions.md`
- `docs/product-runtime.md`
- `docs/roadmap.md`

### `frontend/`

React + Vite dashboard.

Important files:

```text
frontend/package.json
frontend/src/main.tsx
frontend/src/App.tsx
frontend/src/api.ts
frontend/src/styles.css
frontend/src/components/layout.tsx
frontend/src/components/settings.tsx
frontend/src/components/cockpit.tsx
```

`frontend/src/main.tsx`

React entrypoint.

`frontend/src/App.tsx`

Main dashboard state and routing file. It owns API calls, page selection,
message state, approvals, settings drafts, and handlers.

`frontend/src/api.ts`

Typed frontend wrapper around backend `/v1/*` endpoints.

`frontend/src/components/cockpit.tsx`

Reusable cockpit UI components introduced for dashboard v2.

`frontend/src/components/settings.tsx`

Reusable config/settings UI components.

`frontend/src/components/layout.tsx`

Small layout compatibility helpers.

`frontend/src/styles.css`

All current dashboard styling. The v2 cockpit theme tokens are appended under:

```text
/* DMD Agent 4 All cockpit UI v2 */
```

### `scripts/`

Install and bootstrap scripts:

- `scripts/install.sh`
- `scripts/bootstrap.sh`
- `scripts/bootstrap.ps1`
- `scripts/docker-entrypoint.sh`

### `src/dmdcore/`

Main Python package.

Important files and folders:

```text
src/dmdcore/__init__.py
src/dmdcore/app_paths.py
src/dmdcore/audit.py
src/dmdcore/autonomy.py
src/dmdcore/cli.py
src/dmdcore/config.py
src/dmdcore/doctor.py
src/dmdcore/email_oauth.py
src/dmdcore/model_presets.py
src/dmdcore/runtime.py
src/dmdcore/server.py
src/dmdcore/workspace.py
src/dmdcore/agent/
src/dmdcore/interfaces/
src/dmdcore/llm/
src/dmdcore/manifests/tools/
src/dmdcore/memory/
src/dmdcore/permissions/
src/dmdcore/sandbox/
src/dmdcore/secrets/
src/dmdcore/security/
src/dmdcore/tools/
```

`app_paths.py`

Computes local data paths such as config, memory, workspace, and audit DB.

`audit.py`

SQLite audit store. Records approvals, tool calls, audit events, connector
status, model requests, and related runtime events.

`autonomy.py`

Autonomy mode definitions and safety invariants.

`cli.py`

Command-line interface. This file defines `dmdcore` commands.

`config.py`

Default config, deep merge logic, config loading, and config saving.

`doctor.py`

Local readiness and safety diagnostics.

`email_oauth.py`

Google OAuth helper functions and email secret lookup/storage helpers.

`model_presets.py`

Local model modes and hardware recommendations.

`runtime.py`

Builds the runtime objects used by CLI/server: config, memory, tools,
permissions, agent core, and contexts.

`server.py`

FastAPI app. Owns HTTP routes, dashboard serving, API models, configuration
updates, token loading, provider setup, terminal control, Telegram control, and
workspace file APIs.

`workspace.py`

Workspace path policy helpers.

### `agent/`

Agent decision and routing code.

Important files:

- `core.py`: main agent core
- `planner.py`: LLM planner prompt/schema/parse layer
- `router.py`: fallback routing
- `context.py`: context helpers
- `history.py`: chat history helpers
- `runtime_state.py`: runtime state helpers

### `interfaces/`

External user interfaces beyond web/CLI.

Current important file:

- `telegram.py`

### `llm/`

Model provider layer.

Current providers:

- Ollama
- OpenAI-compatible API
- DeepSeek through OpenAI-compatible API shape

### `manifests/tools/`

Every tool has a JSON manifest. The model may only call registered tools.

The manifest declares:

- tool name
- description
- risk
- required permissions
- whether approval is required
- whether cloud context is allowed
- whether enabled by default
- argument schema

### `memory/`

Markdown memory manager.

### `permissions/`

Permission engine and tool request models.

### `sandbox/`

Terminal sandbox/policy.

### `secrets/`

Local and OS secret store helpers.

### `security/`

Redaction and security policy helpers.

### `tools/`

Tool handler implementations:

- browser
- built-in local tools
- email connector
- reminders
- storage
- web helpers

### `tests/`

Unit tests. The current suite covers agent core behavior, browser tools, CLI,
dashboard controls, doctor checks, memory, OpenAI usage, permissions, planner
behavior, redaction, Telegram, terminal policy, terminal tool execution, and
workspace file APIs.

## Runtime Data Locations

Default local data lives under:

```text
~/.local/share/dmdcore/
```

Important runtime files/directories:

```text
~/.local/share/dmdcore/config.yaml
~/.local/share/dmdcore/memory/
~/.local/share/dmdcore/workspace/
~/.local/share/dmdcore/audit.db
```

The code should treat these as local user data, not repository files.

Do not commit runtime config, memory, workspace data, audit DBs, tokens, API
keys, app passwords, OAuth refresh tokens, or downloaded user files.

## Installation

### Docker Mode For Normal Users

Docker mode is the easiest path for non-developers.

macOS/Linux:

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

Docker mode starts:

- backend plus built dashboard on port `8765`
- Ollama local model runtime

Docker mode mounts only:

```text
./workspace -> /workspace
```

Do not mount the whole home directory. The project intentionally keeps Docker
workspace access narrow.

### Local Development Install

macOS/Linux:

```bash
git clone https://github.com/Denis-Denchev/dmdcore.git
cd dmdcore
./scripts/install.sh
```

Or use the project launcher directly:

```bash
./start session
```

The launcher creates `.venv` and installs the package if needed.

Windows PowerShell:

```powershell
git clone https://github.com/Denis-Denchev/dmdcore.git
cd dmdcore
.\start.ps1 session
```

### System Requirements

Required:

- Python 3.11+
- Git
- Node.js/npm for dashboard development
- Ollama for local models, unless using an API provider

Optional:

- Docker Desktop for Docker mode
- Playwright and Chromium for browser interaction tools

## Running The App

### Terminal Chat

```bash
./start session
```

Ask once and exit:

```bash
./start ask "What can you do?"
```

### Dashboard And API

```bash
./start web
```

This starts:

- local API server
- Vite dashboard
- Telegram polling when Telegram is enabled and a token is available

Default URLs:

```text
Dashboard: http://127.0.0.1:5174
API:       http://127.0.0.1:8765
```

Stop with:

```text
Ctrl+C
```

### Developer Servers

Backend only:

```bash
.venv/bin/dmdcore serve
```

Frontend only:

```bash
cd frontend
npm run dev
```

Production frontend build:

```bash
cd frontend
npm run build
```

## CLI Cheat Sheet

The `./start` launcher forwards to `dmdcore`.

Everyday commands:

```bash
./start session
./start ask "Show my local memory files"
./start web
./start open
./start doctor
./start status
```

Direct CLI commands:

```bash
dmdcore
dmdcore chat
dmdcore ask "Show my local memory files"
dmdcore start
dmdcore open
dmdcore status
dmdcore doctor
dmdcore doctor --json
dmdcore wizard
```

Model commands:

```bash
dmdcore model
dmdcore model light --pull
dmdcore model fast --pull
dmdcore model use qwen3:8b --pull
dmdcore models list
dmdcore models set-mode light
dmdcore models set qwen3:8b
dmdcore models provider ollama
dmdcore models provider openai --api-key-env DMDCORE_OPENAI_API_KEY --base-url https://api.openai.com/v1 --model gpt-4o-mini
dmdcore models provider deepseek --api-key-env DMDCORE_DEEPSEEK_API_KEY --base-url https://api.deepseek.com --model deepseek-v4-flash
dmdcore models provider lmstudio --base-url http://localhost:1234/v1
```

Tools and permissions:

```bash
dmdcore tools list
dmdcore tools enable memory.write
dmdcore tools disable memory.write
dmdcore permissions list
dmdcore permissions grant gmail.readonly
dmdcore permissions revoke gmail.readonly
```

Terminal:

```bash
dmdcore terminal status
dmdcore terminal workspace /path/to/project
dmdcore terminal allow git status
dmdcore terminal remove git status
dmdcore terminal enable --tool --grant-permission
dmdcore terminal disable
dmdcore terminal auto-approve on
dmdcore terminal run -- git status
```

Memory:

```bash
dmdcore memory path
dmdcore memory list
dmdcore memory read profile.md
dmdcore memory open
```

Approvals:

```bash
dmdcore approvals list
dmdcore approvals approve <id>
dmdcore approvals deny <id>
```

Telegram:

```bash
dmdcore telegram status
dmdcore telegram allow <telegram_user_id>
dmdcore telegram remove <telegram_user_id>
dmdcore telegram enable
dmdcore telegram disable
dmdcore telegram run
dmdcore telegram run --once
```

## Dashboard v2 Cockpit UI

The dashboard is the main visual control surface.

Routes are preserved:

```text
/chat
/downloads
/approvals
/models
/config
/config/general
/config/models
/config/tools
/config/security
/config/workspace
/config/memory
/config/telegram
/config/emergency
/config/advanced
/telegram
/memory
/tools
/logs
/help
```

### Layout

The dashboard has:

1. Left command rail
2. Top cockpit status strip
3. Main route content
4. Right operational panel on wide chat screens

### Left Sidebar

The sidebar shows:

- product identity
- navigation
- active route glow
- model/provider
- enabled tools count
- pending approvals count
- safety mode
- runtime version

### Top Status Bar

The top status bar shows:

- current page
- mode
- runtime model
- safety state
- pending approvals
- active tools count

It also contains:

- Danger Mode toggle
- Emergency Stop
- Reset Emergency when active
- Start Tour
- Refresh

### Chat Page

The chat page contains:

- central message log
- distinct user/agent/system message cards
- status metadata for agent messages
- expandable `ToolTrace`
- email preview blocks when available
- approval buttons inside the chat
- command-style composer with quick-action icons
- right-side `AgentCorePanel`
- pending approvals preview
- recent activity feed

### Approvals Page

Approvals are shown as risk-control cards:

- approval id
- tool name
- risk badge
- reason
- requested time
- status
- expandable action args
- approve/deny buttons

### Models Page

Models page shows runtime controls:

- current provider
- active model
- planner model
- response language
- planner token settings
- planner temperature
- provider key env
- local model mode cards
- OpenAI key/model/usage controls
- DeepSeek key/model controls

### Tools Page

Tools page has two layers:

1. Human-friendly tool deck:
   - Browser
   - Gmail
   - Calendar
   - Terminal
   - Memory
   - Filesystem
   - Telegram
   - Web Search
   - Code

2. Actual backend tool list with enable/disable controls.

### Config Page

Config page is a settings control center with sections:

- Core Setup
- Models
- Tools
- Workspace
- Security
- System

Each row has:

- compact icon
- title
- description
- status badge
- open action

## Configuration

Runtime config is loaded from local app data:

```text
~/.local/share/dmdcore/config.yaml
```

Defaults are defined in:

```text
src/dmdcore/config.py
config/default.yaml
```

The config loader deep-merges saved config over defaults. This means new config
keys can be added safely without breaking older user configs.

### Main Config Sections

`llm`

Controls model provider, model names, base URL, API key env var name, token
budgets, response language, planner temperature, and planner prompts.

`privacy`

Controls local-first expectations and cloud-context policy.

`terminal`

Controls terminal execution, allowlist, workspace root, timeouts, output limits,
and auto-approval for exact allowlist entries.

`workspace`

Controls default/current workspace, allowed roots, and blocked paths.

`browser`

Controls browser tool enablement, isolated profile behavior, scrape limits, and
submit approval policy.

`email`

Controls Gmail/Outlook connector settings, auth method, IMAP/SMTP hosts,
environment variable names, OAuth settings, mailboxes, and max body size.

`storage`

Controls downloads root.

`tools`

Stores per-tool enabled/disabled overrides.

`openai_usage`

Stores local OpenAI spend limit.

`permissions`

Stores granted permissions and approval risk threshold.

`runtime`

Stores emergency stop state.

`setup`

Stores local assistant identity and user preferences.

`interfaces`

Stores Telegram and future remote interface settings.

## Models And Providers

Default local provider:

```yaml
llm:
  provider: ollama
  model: qwen3:8b
  base_url: http://localhost:11434
```

Recommended local modes:

| Mode | Typical model | Use case |
|---|---|---|
| Light | `qwen3:4b` or `phi4-mini` | low-memory machines |
| Fast | `qwen3:8b` | default local use |
| Balanced | `qwen3:14b` | better quality |
| Power | `qwen3:30b` | stronger machines |

OpenAI-compatible providers are opt-in.

OpenAI example:

```bash
export DMDCORE_OPENAI_API_KEY="sk-..."
dmdcore models provider openai --api-key-env DMDCORE_OPENAI_API_KEY --base-url https://api.openai.com/v1 --model gpt-4o-mini
```

DeepSeek example:

```bash
export DMDCORE_DEEPSEEK_API_KEY="sk-..."
dmdcore models provider deepseek --api-key-env DMDCORE_DEEPSEEK_API_KEY --base-url https://api.deepseek.com --model deepseek-v4-flash
```

LM Studio example:

```bash
dmdcore models provider lmstudio --base-url http://localhost:1234/v1
dmdcore models set local-model-name
```

Important:

- API key values should stay in environment variables or process-local dashboard
  state.
- Config stores env var names, not secret values.
- Private connector context still goes through approval before cloud use.

## Tools, Permissions, Risk, And Approvals

Every tool has a manifest in:

```text
src/dmdcore/manifests/tools/
```

Each manifest declares:

- `name`
- `description`
- `risk`
- `permissions`
- `approval_required`
- `cloud_allowed`
- `default_enabled`
- argument schema

Risk is a numeric level:

| Risk | Meaning |
|---|---|
| 0 | local read/status or harmless context |
| 1 | low-risk read/query |
| 2 | local write or moderate local state change |
| 3 | external modification or meaningful side effect |
| 4 | high-risk send/write/submit/scaffold |
| 5 | highest risk: terminal execution, delete, destructive action |

Default approval threshold is:

```yaml
permissions:
  approval_required_at_risk: 3
```

That means risk `3+` normally requires approval unless a tool is explicitly
marked otherwise by policy. Some tools require approval regardless.

### Current Tool Manifest Summary

| Tool | Risk | Permission | Approval | Default | Description |
|---|---:|---|---|---|---|
| `browser.click` | 3 | `browser.interact` | yes | no | Click an element in the isolated browser. |
| `browser.extract_text` | 1 | `browser.read` | no | no | Fetch and extract readable text from an HTTP/HTTPS URL. |
| `browser.fill_form` | 3 | `browser.interact` | yes | no | Fill a form without submitting. |
| `browser.open` | 1 | `browser.read` | no | no | Fetch safe metadata for an HTTP/HTTPS URL. |
| `browser.scrape_markdown` | 1 | `browser.read` | no | no | Scrape a page, convert to Markdown, and save locally. |
| `browser.submit` | 4 | `browser.submit` | yes | no | Submit a form in the isolated browser. |
| `calendar.create_event` | 3 | `calendar.events` | yes | no | Create a calendar event. |
| `calendar.delete_event` | 5 | `calendar.events` | yes | no | Delete a calendar event. |
| `calendar.find_free_slots` | 1 | `calendar.freebusy` | no | no | Find free calendar slots. |
| `calendar.today` | 1 | `calendar.readonly` | no | no | Read today's calendar events. |
| `calendar.update_event` | 3 | `calendar.events` | yes | no | Update a calendar event. |
| `calendar.week` | 1 | `calendar.readonly` | no | no | Read this week's calendar events. |
| `developer.context` | 0 | none | no | yes | Load safe coding workspace context. |
| `files.delete` | 5 | none | yes | yes | Delete a file or directory inside allowed workspace roots. |
| `files.list` | 0 | none | no | yes | List workspace files and filter blocked secrets. |
| `files.mkdir` | 2 | none | yes | yes | Create a directory inside an allowed workspace root. |
| `files.read` | 0 | none | no | yes | Read a non-secret text file. |
| `files.write` | 4 | none | yes | yes | Create, append, or overwrite text/code files. |
| `files.write_many` | 4 | none | yes | yes | Batch write multiple text/code files. |
| `gmail.archive` | 3 | `gmail.modify` | yes | no | Archive Gmail messages. |
| `gmail.create_draft` | 2 | `gmail.compose` | no | no | Create a local Gmail draft. |
| `gmail.label` | 3 | `gmail.modify` | yes | no | Apply Gmail labels. |
| `gmail.read_thread` | 1 | `gmail.readonly` | no | no | Read one Gmail thread. |
| `gmail.reply_draft` | 2 | `gmail.compose` | no | no | Create a local Gmail reply draft. |
| `gmail.search` | 1 | `gmail.readonly` | no | no | Search Gmail messages. |
| `gmail.send_draft` | 4 | `gmail.send` | yes | no | Send an existing Gmail draft. |
| `gmail.summarize_inbox` | 1 | `gmail.readonly` | no | no | Summarize recent Gmail messages. |
| `memory.list` | 0 | none | no | yes | List local Markdown memory files. |
| `memory.organize_long_term` | 2 | none | yes | yes | Split a large memory file into category files. |
| `memory.read` | 0 | none | no | yes | Read a local Markdown memory file. |
| `memory.write` | 2 | none | yes | yes | Write an approved memory file. |
| `outlook.archive` | 3 | `outlook.modify` | yes | no | Archive Outlook messages. |
| `outlook.create_draft` | 2 | `outlook.compose` | no | no | Create a local Outlook draft. |
| `outlook.read_thread` | 1 | `outlook.readonly` | no | no | Read one Outlook message/thread. |
| `outlook.reply_draft` | 2 | `outlook.compose` | no | no | Create a local Outlook reply draft. |
| `outlook.search` | 1 | `outlook.readonly` | no | no | Search Outlook messages. |
| `outlook.send_draft` | 4 | `outlook.send` | yes | no | Send an Outlook draft through SMTP. |
| `outlook.summarize_inbox` | 1 | `outlook.readonly` | no | no | Summarize recent Outlook inbox messages. |
| `profile.update` | 2 | none | yes | yes | Update local agent/user identity settings. |
| `project.scaffold_one_page_app` | 4 | none | yes | yes | Scaffold a one-page React + Node project. |
| `reminders.complete` | 2 | none | yes | yes | Mark a reminder completed. |
| `reminders.create` | 2 | none | yes | yes | Create a local reminder. |
| `reminders.list` | 0 | none | no | yes | List local reminders. |
| `system.list_enabled_tools` | 0 | none | no | yes | List enabled tools for current policy. |
| `terminal.run` | 5 | `terminal.run` | yes | no | Run an allowlisted workspace command. |
| `workspace.status` | 0 | none | no | yes | Show workspace status and policy. |
| `workspace.switch` | 2 | none | no | yes | Switch current workspace to a validated path. |

## Email System

Email is implemented through provider-specific tools:

- Gmail
- Outlook

The current practical transport is IMAP/SMTP. Gmail also has OAuth2 support in
the codebase.

### Email Safety Rules

- Email connectors are disabled by default.
- Read-only email actions require the correct permission.
- Draft creation creates local drafts only.
- Sending always requires approval.
- The chat approval should show preview before sending when the backend has
  draft metadata.
- SMTP credentials are not sent to the model.
- App passwords and OAuth tokens must not be committed.

### Gmail App Password Setup

Environment variables:

```text
DMDCORE_GMAIL_USERNAME
DMDCORE_GMAIL_APP_PASSWORD
DMDCORE_GMAIL_FROM optional
```

Dashboard setup:

1. Open `/config/tools`.
2. Enable Gmail.
3. Choose `App password`.
4. Enter Gmail address.
5. Enter Google app password.
6. Optional: enter From address.
7. Click `Load Gmail Credentials`.
8. Click `Test Gmail Connection`.

The connection test logs into IMAP and SMTP but does not send an email.

Important: Google app passwords may contain spaces or hidden non-breaking
spaces when copied. The backend normalizes this input before auth.

### Gmail OAuth Setup

Gmail OAuth endpoints exist:

```text
POST /v1/email/oauth/google/client-secret
POST /v1/email/oauth/google/start
GET  /v1/email/oauth/google/callback
POST /v1/email/oauth/google/disconnect
```

Default callback URL:

```text
http://127.0.0.1:8765/v1/email/oauth/google/callback
```

OAuth secret env names:

```text
DMDCORE_GMAIL_OAUTH_CLIENT_SECRET
DMDCORE_GMAIL_OAUTH_REFRESH_TOKEN
```

### Send Flow

When the user says something like:

```text
Use my Gmail and send an email to denis.denchev@outlook.com saying hello world.
```

The intended flow is:

1. Agent creates local draft.
2. Agent returns approval-required response.
3. Chat displays:
   - To
   - Subject
   - Body preview
4. User presses `Approve`.
5. Backend sends the draft.
6. Audit log records the action.

If the user only asks to create a draft, the draft is created but not sent.

### Email Troubleshooting

`connector_not_configured`

Provider is disabled or required credentials are missing.

`authentication_failed`

The username/password/token was rejected. For Gmail app passwords, create a new
app password and paste it again.

`ascii codec can't encode character '\xa0'`

This was caused by hidden non-breaking spaces in the app password. The code now
normalizes app passwords, but if it appears again, retype the password manually.

Outlook `5.7.139 Authentication unsuccessful`

Microsoft may have disabled basic auth/SMTP for that account or tenant. Use
Gmail SMTP or implement Microsoft Graph OAuth for Outlook.

## Telegram Interface

Telegram is a remote control interface into the same local agent core.

It is disabled by default.

Rules:

- only allowlisted Telegram user IDs can control the agent
- bot token is not stored in `config.yaml`
- `/id` is available for discovering user ID
- every useful command requires allowlist
- risky actions produce approval buttons
- approving in Telegram executes the same stored approval as dashboard/CLI
- Telegram does not bypass permissions, disabled tools, or emergency stop

Setup from terminal chat:

```text
/telegram setup
```

Telegram chat commands:

```text
/id
/help
/approvals
/approve <id>
/deny <id>
```

Terminal setup controls:

```text
/telegram token <bot_token>
/telegram once
/telegram allow <telegram_user_id>
/telegram enable
/telegram disable
/telegram run
/telegram status
/back
```

CLI controls:

```bash
dmdcore telegram status
dmdcore telegram allow <telegram_user_id>
dmdcore telegram remove <telegram_user_id>
dmdcore telegram enable
dmdcore telegram disable
dmdcore telegram run
```

Dashboard controls:

- enable/disable Telegram
- set token env var name
- load token into current API process
- allow/remove user IDs
- start/stop polling

## Terminal Safety

Terminal execution is intentionally narrow.

Rules:

- disabled by default
- requires `terminal.run` tool enabled
- requires `terminal.run` permission
- commands are arrays, not shell strings
- exact allowlist is enforced
- cwd must stay inside workspace policy
- timeout is enforced
- max output limit is enforced
- obvious secrets are redacted
- risky execution requires approval
- emergency stop terminates active terminal processes

Example setup:

```bash
dmdcore terminal workspace /path/to/project
dmdcore terminal allow git status
dmdcore terminal enable --tool --grant-permission
dmdcore terminal run -- git status
dmdcore approvals approve <id>
```

Auto-approval is only for exact allowlist matches:

```bash
dmdcore terminal auto-approve on
```

Never replace this with broad shell access.

## Browser Tools

Browser tools are disabled by default.

Read tools:

- `browser.open`
- `browser.extract_text`
- `browser.scrape_markdown`

Interaction tools:

- `browser.click`
- `browser.fill_form`
- `browser.submit`

Read tools use guarded HTTP/HTTPS access with:

- URL validation
- local/private network blocking
- timeouts
- response size limits
- text size limits
- secret redaction
- local Markdown saves under `scrapefiles`

Interaction tools require optional Playwright support:

```bash
pip install -e ".[browser]"
python -m playwright install chromium
```

Rules:

- use isolated profile
- do not use personal Chrome profile
- do not use saved personal passwords
- require approval for clicks, form fills, submits, logins, and purchases

## Workspace And Files

Workspace policy protects local files.

The agent should only operate inside allowed workspace roots.

Blocked paths include sensitive/system areas such as:

```text
~/.ssh
~/.aws
~/.config/gcloud
~/.kube
/etc
/var
/private
/Library
/System
```

Secret filenames are blocked even inside allowed roots.

Examples of blocked secret-like names:

```text
.env
.env.*
*.pem
*.key
id_rsa
id_ed25519
credentials.json
token.json
secret.*
secrets.*
*.p12
*.pfx
*.kubeconfig
```

File tool behavior:

- `files.list` is safe local listing.
- `files.read` reads non-secret text files.
- `files.mkdir` requires approval.
- `files.write` requires approval.
- `files.write_many` requires approval.
- `files.delete` is risk 5 and requires approval.

## Memory

Memory is local Markdown storage.

Default location:

```text
~/.local/share/dmdcore/memory/
```

Memory tools:

- `memory.list`
- `memory.read`
- `memory.write`
- `memory.organize_long_term`

Rules:

- reads are low risk
- writes require approval
- memory is local
- memory files must stay inside memory root
- short-term memory can have TTL behavior

Terminal chat helpers:

```text
/memory
/read profile.md
```

CLI:

```bash
dmdcore memory list
dmdcore memory read profile.md
dmdcore memory open
```

## Reminders

Reminders are local, private, and stored in the workspace/runtime data area.

Tools:

- `reminders.create`
- `reminders.list`
- `reminders.complete`

Reminder writes require approval.

Telegram can notify due reminders when Telegram is configured.

Reminder action buttons can include:

- Done
- Snooze
- Repeat +1d
- Cancel
- Maps link when an action URL exists

## Calendar

Calendar support currently uses a local workspace JSONL event store.

Tools:

- `calendar.today`
- `calendar.week`
- `calendar.find_free_slots`
- `calendar.create_event`
- `calendar.update_event`
- `calendar.delete_event`

Rules:

- reads/free-busy are lower risk
- create/update require approval
- delete is risk 5

Google Calendar OAuth sync is not fully implemented yet.

## Emergency Stop

Emergency Stop is backend enforced.

It is not just UI state and not just a prompt instruction.

When triggered, it:

- terminates active terminal subprocesses
- force-kills subprocesses that do not exit after a short grace period
- cancels pending approvals
- stops background Telegram/reminder polling
- stores emergency mode in config
- blocks further tool execution
- blocks approval execution until reset

Dashboard controls:

- `Emergency Stop`
- `Reset Emergency`

API:

```text
GET  /v1/emergency
POST /v1/emergency/stop
POST /v1/emergency/reset
```

## API Surface

FastAPI routes live in:

```text
src/dmdcore/server.py
```

Health:

```text
GET /health
```

Chat:

```text
POST /v1/chat
POST /v1/chat/stream
```

Tools and permissions:

```text
GET  /v1/tools
POST /v1/tools/{tool_name}/enable
POST /v1/tools/{tool_name}/disable
GET  /v1/permissions
POST /v1/permissions/grant
POST /v1/permissions/revoke
```

Audit, doctor, approvals:

```text
GET  /v1/audit
GET  /v1/doctor
GET  /v1/approvals
POST /v1/approvals/{approval_id}/approve
POST /v1/approvals/{approval_id}/deny
```

Emergency:

```text
GET  /v1/emergency
POST /v1/emergency/stop
POST /v1/emergency/reset
```

Memory:

```text
GET  /v1/memory
GET  /v1/memory/file
POST /v1/memory/file
```

Workspace files:

```text
GET /v1/workspace-files
GET /v1/workspace-files/file
GET /v1/workspace-files/download
```

Configuration:

```text
GET  /v1/configuration
POST /v1/configuration
```

Email:

```text
POST /v1/email/credentials
POST /v1/email/test
POST /v1/email/oauth/google/client-secret
POST /v1/email/oauth/google/start
GET  /v1/email/oauth/google/callback
POST /v1/email/oauth/google/disconnect
```

Status and models:

```text
GET  /v1/status
GET  /v1/models
POST /v1/models/mode
POST /v1/models/model
GET  /v1/openai
POST /v1/openai/key
POST /v1/openai/model
GET  /v1/openai/models
POST /v1/openai/limit
POST /v1/openai/usage/reset
GET  /v1/deepseek
POST /v1/deepseek/key
POST /v1/deepseek/model
GET  /v1/deepseek/models
```

Connectors, terminal, Telegram:

```text
GET  /v1/connectors
GET  /v1/terminal
POST /v1/terminal/enable
POST /v1/terminal/disable
POST /v1/terminal/settings
POST /v1/terminal/allow
POST /v1/terminal/remove
POST /v1/terminal/run
GET  /v1/telegram
POST /v1/telegram/enable
POST /v1/telegram/disable
POST /v1/telegram/allow
POST /v1/telegram/remove
POST /v1/telegram/token-env
POST /v1/telegram/token
POST /v1/telegram/start
POST /v1/telegram/stop
```

Dashboard routes served by backend/static fallback:

```text
/chat
/downloads
/approvals
/models
/memory
/telegram
/tools
/logs
/help
/config
/config/{section}
```

## Frontend Code Guide

The frontend is a React 19 + Vite app.

Package scripts:

```bash
cd frontend
npm run dev
npm run build
npm run preview
```

### State Owner

`frontend/src/App.tsx` owns:

- active route
- configuration drafts
- chat messages
- active trace events
- approval list
- tool list
- permissions
- connectors
- terminal state
- Telegram state
- OpenAI/DeepSeek state
- memory files
- workspace files
- emergency state
- tour state

### API Wrapper

`frontend/src/api.ts` defines TypeScript types and functions for backend calls.

If backend API shapes change, update this file first, then update consuming UI.

### Cockpit Components

`frontend/src/components/cockpit.tsx` contains the v2 reusable UI layer.

Important components:

- `AppShell`: top-level layout wrapper
- `Sidebar`: left command rail
- `TopStatusBar`: cockpit status strip
- `StatusBadge`: status/risk pill
- `ApprovalCard`: approval/risk card
- `ToolCard`: tool status card
- `AgentCorePanel`: right-side AI core visual and stats
- `ActivityFeed`: recent audit activity
- `ChatMessage`: message card with metadata and approval preview
- `ToolTrace`: expandable tool/reasoning trace

### Settings Components

`frontend/src/components/settings.tsx` contains:

- `ConfigHub`
- `ConfigSectionPage`
- `SettingGroup`
- `SettingRow`
- `DangerZone`

### Styling

`frontend/src/styles.css` contains legacy base styles plus v2 cockpit overrides.

Key v2 theme tokens:

```css
--bg-main: #05080d;
--bg-panel: rgba(10, 18, 28, 0.82);
--bg-panel-strong: rgba(14, 25, 38, 0.95);
--border-soft: rgba(0, 255, 220, 0.14);
--border-strong: rgba(0, 255, 220, 0.38);
--text-main: #eef7ff;
--text-muted: #8fa6b8;
--accent-cyan: #00f5ff;
--accent-teal: #00ffc6;
--accent-green: #2cff9a;
--accent-amber: #ffb020;
--danger: #ff4d4d;
--danger-soft: rgba(255, 77, 77, 0.14);
```

UI design rules:

- keep the dashboard operational, not marketing-like
- preserve routes
- preserve backend API contracts
- preserve approval flow
- avoid hiding safety controls
- keep Emergency Stop visually serious
- avoid horizontal overflow on mobile

## Backend Code Guide

### `runtime.py`

Use this when you need to understand how the app is assembled.

It builds:

- config
- app paths
- memory manager
- audit store
- permissions
- tool registry
- agent core

### `server.py`

Use this when changing dashboard/API behavior.

Do not change API contracts casually. Frontend and tests depend on them.

### `agent/core.py`

Use this when changing agent behavior.

This file contains:

- deterministic command handling
- planner calls
- approval creation
- approval execution
- response shaping
- email draft/send flow
- fallback behavior
- emergency stop checks

Be careful: this file is central.

### `permissions/engine.py`

Use this when changing tool policy.

It decides whether a tool request is:

- allowed
- denied
- approval required

### `security/policy.py`

Use this when changing path, secret, destructive action, and workspace safety
checks.

### `tools/`

Use this when changing actual tool behavior.

Tool code should be deterministic, defensive, and redacted.

### `manifests/tools/`

Add or update manifests before exposing a tool to the planner.

## Testing And Verification

Backend tests:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Python import/syntax pass:

```bash
.venv/bin/python -m compileall -q src tests
```

Frontend build:

```bash
cd frontend
npm run build
```

Optional pytest:

```bash
.venv/bin/python -m pytest -q
```

Current frontend note:

There is no lint script yet. `npm run build` is the main frontend verification.

## Common Debugging Paths

### Dashboard does not load

Check API:

```bash
curl http://127.0.0.1:8765/health
```

Check frontend dev server:

```bash
curl -I http://127.0.0.1:5174/chat
```

Run:

```bash
./start web
```

### Tool is denied

Check:

1. Is the tool enabled?
2. Is required permission granted?
3. Is risk above approval threshold?
4. Is emergency stop active?
5. Is workspace path allowed?
6. Is the request touching a secret path?

Useful commands:

```bash
dmdcore tools list
dmdcore permissions list
dmdcore approvals list
dmdcore doctor
```

### Email will not send

Check:

1. Gmail/Outlook provider enabled?
2. Correct auth method?
3. Credentials loaded in current API process?
4. IMAP/SMTP test passes?
5. `gmail.send` or `outlook.send` permission granted?
6. Approval was approved?
7. App password has hidden spaces?

Dashboard path:

```text
/config/tools
```

### Approval appears but nothing happens

Approvals are one-time stored actions.

Check:

```bash
dmdcore approvals list
```

If emergency stop is active, reset it only when safe.

### Terminal command will not run

Check:

```bash
dmdcore terminal status
dmdcore terminal allow git status
dmdcore terminal enable --tool --grant-permission
```

Remember: command matching is exact.

`git status` is not the same as `git status --short`.

### Model is not responding

For local Ollama:

```bash
ollama list
ollama pull qwen3:8b
dmdcore doctor
```

For cloud/API provider:

```bash
echo "$DMDCORE_OPENAI_API_KEY"
dmdcore status
dmdcore doctor
```

## How To Add A New Tool

Use this process for new tools.

1. Add a manifest in:

```text
src/dmdcore/manifests/tools/
```

2. Pick a risk level.

Use risk 3+ for external writes, sends, submits, meaningful modifications, and
anything that could cost money or damage data.

3. Define required permissions.

Example:

```json
{
  "name": "example.send",
  "description": "Send something to an external service.",
  "risk": 4,
  "permissions": ["example.send"],
  "approval_required": true,
  "cloud_allowed": false,
  "default_enabled": false
}
```

4. Implement the handler in `src/dmdcore/tools/`.

5. Register the handler in the tool registry/runtime path.

6. Make sure secrets are not included in tool output.

7. Add tests for:

- allowed behavior
- denied behavior
- approval-required behavior
- audit behavior
- secret redaction
- emergency stop behavior if relevant

8. Run:

```bash
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m compileall -q src tests
cd frontend && npm run build
```

## Known Limitations And Next Work

Recommended next implementation targets:

1. Add a frontend lint/type quality script beyond `npm run build`.
2. Add component-level frontend tests for cockpit components.
3. Harden operational deployment examples for Telegram polling with launchd or
   systemd.
4. Finish Microsoft Graph OAuth for Outlook accounts where IMAP/SMTP basic auth
   is disabled.
5. Continue hardening Gmail OAuth UX and secret-store behavior.
6. Package optional Playwright browser runtime setup into a smoother installer.
7. Add signed release packaging for macOS/Windows/Linux.
8. Expand dashboard visual tests with Playwright screenshots once a browser test
   stack is added.

## Handoff Prompt For The Next Agent

Use this prompt when continuing the project in another coding-agent session:

```text
You are continuing the DMD Agent 4 All repository.

First read README.md completely. Treat it as product brief, architecture
contract, safety policy, and implementation map.

Core facts:
- This is a local-first AI agent cockpit.
- The LLM is an untrusted planner, never a security boundary.
- Tool execution must go through registered manifests, ToolSafetyPolicy,
  PermissionEngine, approvals, and audit logging.
- Secrets must never be stored in config, committed files, prompts, logs, or
  model context.
- High-risk tools and remote interfaces are disabled by default.
- Sending email requires approval and should show preview before send.
- Terminal execution is workspace-scoped, allowlisted, approval-gated, timed,
  output-limited, and redacted.
- Telegram is allowlisted and must not bypass the same approval engine.
- Emergency Stop is backend-enforced and blocks tools/approvals until reset.
- Dashboard v2 cockpit UI lives in frontend/src/components/cockpit.tsx and
  frontend/src/styles.css.

Before editing:
1. Run `git status --short`.
2. Inspect files related to the task.
3. Do not revert user changes or unrelated work.
4. Preserve backend API contracts unless the task explicitly requires changing
   them.
5. Keep docs and UI copy in English unless the user explicitly requests another
   language.

Useful verification:
- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall -q src tests`
- `cd frontend && npm run build`

When adding capabilities:
- Add or update tool manifests first.
- Keep tools disabled by default unless they are harmless local reads.
- Add tests for allowed, denied, approval-required, and audit behavior.
- Never let Telegram, CLI, web UI, or the planner bypass AgentCore and
  PermissionEngine.
- Prefer small complete increments over broad rewrites.
```

## License

MIT
