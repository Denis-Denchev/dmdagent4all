# DMD Agent 4 All

Open-source, local-first AI control center for personal computers and small
servers.

Current version: `1.0.0`.

Read this file as both product documentation and a handoff prompt for another
coding agent. It explains what the project is, what must stay true, what already
works, and what should be built next.

## Core Idea

DMD Agent 4 All is a personal AI assistant that runs on a user's own machine,
such as a Mac mini, home server, or workstation. The user can talk to it through
a web dashboard, CLI, and locked remote channels such as Telegram.

The assistant is allowed to plan and suggest actions, but it is never trusted as
a security boundary.

> The model is an untrusted planner. Backend code enforces security.

The model can propose structured tool calls. The backend decides whether each
tool call is allowed, denied, or requires approval. The model must never receive
raw secrets, unrestricted shell access, `.env` files, SSH keys, OAuth refresh
tokens, private browser profiles, or direct operating system control.

## Product Vision

The long-term goal is a local AI operator that helps a user manage routine
digital work without handing control of their computer or private data to a
cloud agent.

Target capabilities:

- Chat with a local model through Ollama by default.
- Read and write local Markdown memory with approval for writes.
- Manage explicit tools through a permission engine.
- Connect to Gmail, Calendar, browser automation, terminal commands, and future
  services only after the user enables each connector.
- Use remote interfaces such as Telegram only for allowlisted user IDs.
- Require approval for risky actions such as sending, modifying external data,
  browser submits, terminal execution, or memory writes.
- Keep a local audit trail of requests, tool calls, approvals, connector status,
  LLM requests, and memory events.

## Architecture

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
Docker / OAuth / Local APIs / OS Secret Store
```

Request flow:

1. A user sends a message through CLI, web UI, or Telegram.
2. `AgentCore` handles deterministic commands first.
3. If needed, the Ollama planner proposes either a final answer or a structured
   `ToolRequest`.
4. `PermissionEngine` evaluates the request against tool manifests, enabled
   tools, granted permissions, risk level, approval requirements, and cloud
   context policy.
5. Allowed tools execute through registered backend handlers.
6. Risky tools create a pending approval in SQLite and execute only after the
   stored approval is approved.
7. Audit logs record remote requests, tool calls, approvals, and execution
   results.

## Non-Negotiable Security Rules

- Never trust the LLM for security decisions.
- Never let the model invent tool names or bypass registered tool manifests.
- Never expose secrets to model context or logs.
- Never store bot tokens, OAuth refresh tokens, API keys, or passwords in
  `config.yaml` or `.env` files.
- Every connector and high-risk tool must be disabled by default.
- Every external write/send/submit/delete/execute action must require approval.
- Telegram must remain an allowlisted remote interface, not a public bot mode.
- Terminal commands must use command arrays, exact allowlists, workspace-only
  execution, timeouts, and blocked secret/system paths.
- Browser automation must use an isolated profile, not the user's personal
  browser profile.
- Cloud models must stay disabled by default for private connector context.

Read the security docs before changing these rules:

- [SECURITY.md](SECURITY.md)
- [PRIVACY.md](PRIVACY.md)
- [THREAT_MODEL.md](THREAT_MODEL.md)
- [docs/architecture.md](docs/architecture.md)
- [docs/permissions.md](docs/permissions.md)
- [docs/product-runtime.md](docs/product-runtime.md)
- [docs/connectors.md](docs/connectors.md)

## Current Implementation Status

Foundation implemented:

- Python package and `dmdagent` CLI entry point.
- Local data layout under `~/.local/share/dmdagent4all/`.
- YAML config with deep-merged defaults.
- Tool manifest loader with compact argument schemas for planner guidance.
- Permission engine with risk levels, enabled/disabled tools, required
  permissions, approval thresholds, and cloud context checks.
- SQLite audit store with audit logs, tool calls, approvals, connector status,
  LLM requests, and memory events tables.
- Markdown memory manager with path safety, long-term/short-term scopes, and
  automatic TTL purging for approved short-term memory.
- Deterministic chat routes for help, greetings, memory listing, explicit
  long-term and short-term memory writes, local reminder fallback parsing, and
  simple browser-open requests.
- Ollama planner integration for structured tool planning, including
  model-first reminder extraction that can use local memory context and
  model-guided Markdown memory routing into long-term or short-term files.
- Approval queue with approve-and-execute semantics.
- FastAPI backend skeleton.
- React/Vite dashboard MVP.
- Local model mode presets for light, fast, balanced, and power modes.
- OpenAI-compatible provider support for API-key and local compatible servers,
  with API key values kept in the running process environment only from the
  dashboard when entered there.
- `dmdagent doctor` readiness and security diagnostics, also exposed in the
  dashboard Doctor view.
- Dashboard connector, terminal, and Telegram control views backed by
  `/v1/connectors`, `/v1/terminal`, and `/v1/telegram` API endpoints.
- Terminal policy module with exact command allowlist and blocked dangerous
  commands/secret paths.
- `terminal.run` handler wired through manifests, permissions, approval queue,
  workspace-only cwd, timeouts, output limits, and redaction.
- Local `reminders.create`, `reminders.list`, and `reminders.complete` tools
  with approval-gated writes to the private workspace reminder store, adaptive
  low-resource scheduling, event/reminder time separation, optional
  location/action URLs, and Telegram notifications for due reminders when
  Telegram is configured.
- Local calendar store handlers for create, update, delete, today, week, and
  free-slot queries. External calendar sync is still a connector task.
- Guarded `browser.open`, `browser.extract_text`, and
  `browser.scrape_markdown` read tools for HTTP/HTTPS pages with URL
  validation, private/local network blocking, timeouts, output limits, secret
  redaction, and local Markdown saves under `scrapefiles`.
- Approval-gated `browser.click`, `browser.fill_form`, and `browser.submit`
  handlers for an isolated Playwright profile when optional browser runtime
  support is installed.
- Gmail and Outlook IMAP/SMTP email handlers for search, read, summarize,
  local draft creation, reply drafts, send-draft, archive, and Gmail label copy.
  They fail closed with `not_configured` until provider config and required env
  vars are present.
- Telegram polling interface with allowlist, `/id`, `/help`, `/approvals`,
  `/approve <id>`, `/deny <id>`, audit logging, approve/deny inline buttons,
  and reminder action buttons for Done, Snooze, Repeat +1d, Cancel, and Maps
  links when a reminder has an action URL.

Partially implemented or stubbed:

- Calendar has a local workspace store; real Google Calendar OAuth sync is not
  implemented yet.
- Email currently uses IMAP/SMTP env-var configuration. Gmail/Outlook OAuth and
  Microsoft Graph are not implemented yet.
- Browser interaction tools require the optional Playwright runtime and Chromium
  browser install. Without that runtime, they fail closed with setup guidance.
- Web dashboard can view chat, connectors, tools, permissions, approvals,
  terminal controls, Telegram setup controls, OpenAI key/model/usage controls,
  memory, audit, models, and Doctor diagnostics. `start web` starts Telegram
  polling in the API process when Telegram is enabled and the bot token is
  available.
- Telegram and dashboard-entered OpenAI token values are process-only by
  default. Restarting the API requires entering them again or exporting the
  relevant environment variable before launch.

## Repository Map

```text
.
├── config/default.yaml                 # Human-readable default config
├── docs/                               # Architecture, connectors, roadmap
├── frontend/                           # React + Vite dashboard
├── scripts/install.sh                  # Local install script
├── src/dmdagent4all/
│   ├── agent/core.py                   # AgentCore, deterministic routing, approvals
│   ├── agent/planner.py                # Planner prompt and structured parse layer
│   ├── app_paths.py                    # Local data paths
│   ├── audit.py                        # SQLite audit and approvals store
│   ├── cli.py                          # dmdagent CLI
│   ├── config.py                       # Config defaults and persistence
│   ├── doctor.py                       # Local readiness and security diagnostics
│   ├── interfaces/telegram.py          # Telegram Bot API polling interface
│   ├── llm/                            # Ollama and OpenAI-compatible providers
│   ├── manifests/tools/                # Tool manifests and risk metadata
│   ├── memory/manager.py               # Markdown memory manager
│   ├── permissions/                    # Permission models and engine
│   ├── sandbox/terminal.py             # Terminal command policy
│   ├── secrets/store.py                # Secret store protocol
│   ├── server.py                       # FastAPI app
│   └── tools/                          # Tool registry and built-in handlers
└── tests/                              # Unit tests
```

## Beginner One-Command Install

Recommended for non-technical users: Docker mode. It installs the app stack,
dashboard, backend, Ollama service, and a small default local model.

macOS / Linux with Docker:

```bash
curl -fsSL https://raw.githubusercontent.com/Denis-Denchev/dmdagent4all/main/scripts/bootstrap.sh | bash -s -- --docker
```

Windows PowerShell with Docker:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "$s=irm https://raw.githubusercontent.com/Denis-Denchev/dmdagent4all/main/scripts/bootstrap.ps1; & ([scriptblock]::Create($s)) -Docker"
```

Open:

```text
http://127.0.0.1:8765
```

Docker mode uses these containers:

- `dmdagent` for backend + built dashboard
- `ollama` for the local model runtime

Persistent data is stored in Docker volumes. The host folder `./workspace` is
mounted into the container as the safe working folder. Do not mount your whole
home directory.

macOS / Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/Denis-Denchev/dmdagent4all/main/scripts/bootstrap.sh | bash
cd dmdagent4all
./start web
```

Windows PowerShell:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/Denis-Denchev/dmdagent4all/main/scripts/bootstrap.ps1 | iex"
cd dmdagent4all
.\start.ps1 web
```

The bootstrap command clones the GitHub repo into the current folder, creates a
local `.venv`, installs Python package dependencies, initializes local app data,
runs the first-time wizard, and installs frontend packages when `npm` is
available.

System runtimes are still required:

- Git to download the repo
- Python 3.11+
- Node.js/npm for the dashboard
- Ollama for local models, unless using an API provider

If Node.js is not installed, terminal chat still works:

```bash
./start session
```

On Windows:

```powershell
.\start.ps1 session
```

## Install For Local Development

```bash
git clone https://github.com/Denis-Denchev/dmdagent4all.git
cd dmdagent4all
./scripts/install.sh
```

The install script creates `.venv`, installs the package in editable mode, and
initializes local app data.

No manual `.venv` activation is required for normal use. From the project
folder:

```bash
./start session
```

On the first run, terminal setup asks for:

- assistant name
- user name
- response language
- local model or API provider choice

Optional one-time shortcut:

```bash
./start install-command
```

After opening a new terminal, this works from the project folder:

```bash
start session
```

Without the shortcut, use `./start session`.

On Windows, use PowerShell from the project folder:

```powershell
.\start.ps1 session
.\start.ps1 web
```

## Running Locally

For terminal chat only:

```bash
start session
```

Ask one question and exit:

```bash
start ask "What can you do?"
```

For normal use, start everything from one terminal:

```bash
start web
```

This starts the local API and dashboard, opens the browser, and also starts
Telegram polling when Telegram is enabled and a bot token is available from the
current process environment or `DMDAGENT_TELEGRAM_BOT_TOKEN`. Stop everything
with `Ctrl+C`.

First-time setup is built into `start session`. To change model later:

```bash
start model fast --pull
start model light --pull
start model use qwen3:14b --pull
```

Open:

```text
http://127.0.0.1:5174
```

Useful everyday controls:

```bash
start session                   # terminal chat
start ask "Show memory"         # one terminal answer
start web                       # API + dashboard + Telegram if configured
start open                      # open the dashboard
start doctor                    # check local health/security
start model                     # show current model and modes
start model light --pull        # small local model
start model fast --pull         # default local model
start model use qwen3:14b --pull
start telegram status           # check Telegram interface
```

Developer commands still exist when needed:

```bash
dmdagent serve
cd frontend && npm run dev
```

## CLI Cheat Sheet

```bash
dmdagent
dmdagent ask "Show my local memory files"
dmdagent start
dmdagent open
dmdagent status
dmdagent doctor
dmdagent doctor --json
dmdagent wizard
dmdagent chat "Show my local memory files"
dmdagent model
dmdagent model light --pull
dmdagent model fast --pull
dmdagent model use qwen3:8b --pull
dmdagent models list
dmdagent models set-mode light
dmdagent models set qwen3:8b
dmdagent models provider ollama
dmdagent models provider openai --api-key-env DMDAGENT_OPENAI_API_KEY --base-url https://api.openai.com/v1 --model gpt-4o-mini
dmdagent models provider deepseek --api-key-env DMDAGENT_DEEPSEEK_API_KEY --base-url https://api.deepseek.com --model deepseek-v4-flash
dmdagent models provider lmstudio --base-url http://localhost:1234/v1
dmdagent tools list
dmdagent tools enable memory.write
dmdagent tools disable memory.write
dmdagent permissions list
dmdagent permissions grant gmail.readonly
dmdagent permissions revoke gmail.readonly
dmdagent terminal status
dmdagent terminal workspace /path/to/dmdagent4all
dmdagent terminal allow git status
dmdagent terminal enable --tool --grant-permission
dmdagent terminal auto-approve on
dmdagent terminal run -- git status
dmdagent memory path
dmdagent memory list
dmdagent memory read profile.md
dmdagent approvals list
dmdagent approvals approve <id>
dmdagent approvals deny <id>
dmdagent telegram status
dmdagent telegram allow <telegram_user_id>
dmdagent telegram remove <telegram_user_id>
dmdagent telegram enable
dmdagent telegram disable
dmdagent telegram run
```

## Terminal Chat

The terminal chat is the simplest interface:

```bash
start session
```

Inside chat:

```text
/panel
/help
/help telegram
/telegram
/telegram setup
/permissions
/tool enable <tool>
/tool disable <tool>
/permission grant <permission>
/permission revoke <permission>
/back
/model fast --pull
/model light --pull
/doctor
/setup
/memory
/read profile.md
/tools
/approvals
/logs
/approve <id>
/deny <id>
/exit
```

Plain-language setup requests are routed before the LLM is used. Inside
interactive chat, `set up telegram bot` starts the Telegram setup wizard in the
same terminal panel. `start ask "set up telegram bot"` prints the non-interactive
guide.

Identity is stored in local setup config and Markdown memory. The assistant can
answer `who are you` and `who am i` without a model call. To update names in
chat:

```text
call yourself Jarvis
my name is Alex
```

The chat starts Ollama automatically when possible. If a model is missing, run:

```bash
start model fast --pull
```

## Local Chat And Models

Recommended first local setup:

```bash
start session
```

The first terminal session asks whether to use a local Ollama model or an
OpenAI-compatible API provider. It stores only provider settings and environment
variable names. API key values stay in the shell environment.

Send a message:

```bash
start ask "Show my local memory files"
```

For slower machines, switch to Light Mode:

```bash
start model light --pull
```

Model presets:

| Mode | Model |
|---|---|
| Light Mode | `qwen3:4b` or `phi4-mini` |
| Fast Mode | `qwen3:8b` |
| Balanced Mode | `qwen3:14b` |
| Power Mode | `qwen3:30b` |

Mac mini recommendations are documented in
[docs/model-presets.md](docs/model-presets.md).

Cloud and OpenAI-compatible providers are opt-in. Store only the env var name
in config; never store the key value:

```bash
export DMDAGENT_OPENAI_API_KEY="sk-..."
start models provider openai --api-key-env DMDAGENT_OPENAI_API_KEY --model gpt-4o-mini
start doctor
```

DeepSeek is also supported through its OpenAI-compatible API:

```bash
export DMDAGENT_DEEPSEEK_API_KEY="sk-..."
start models provider deepseek --api-key-env DMDAGENT_DEEPSEEK_API_KEY --model deepseek-v4-flash
```

The dashboard has OpenAI and DeepSeek views. Paste the API key there to load it
into the current API process, fetch the model dropdown, and choose the model.
The OpenAI view also has a local project spending limit. Local spend is
estimated from token usage returned to this project; provider billing remains
the source of truth.

For a local OpenAI-compatible server:

```bash
start models provider lmstudio --base-url http://localhost:1234/v1
start models set local-model-name
```

Private tool context still requires backend approval before being used with a
cloud model.

## Doctor And Terminal Safety

Run diagnostics:

```bash
dmdagent doctor
```

Terminal execution is disabled by default and requires all of these:

```bash
dmdagent terminal allow git status
dmdagent terminal workspace /path/to/dmdagent4all
dmdagent terminal enable --tool --grant-permission
dmdagent terminal run -- git status
dmdagent approvals approve <id>
dmdagent terminal auto-approve on
```

`terminal.run` accepts command arrays only, matches exact allowlist entries,
forces workspace-only cwd, applies a timeout, redacts obvious secrets from
output, and requires approval by default. The dashboard Terminal view and
`dmdagent terminal auto-approve on` can opt in to automatic execution for exact
allowlist matches only.

## Telegram Interface

Telegram is a locked remote interface into the same local agent core and
permission engine. It is disabled by default.

Rules:

- Bot token comes from the current process environment, normally
  `DMDAGENT_TELEGRAM_BOT_TOKEN`.
- Bot token is not stored in `config.yaml` and dashboard-entered tokens are lost
  on API process restart.
- `/id` is available so a user can discover their Telegram user ID.
- All useful access requires an allowlisted Telegram user ID.
- Every Telegram request is audited as `telegram.request`.
- Risky tool requests return approve/deny buttons.
- Approving from Telegram executes the stored approval request once. It does not
  bypass disabled tools or missing permissions.

Setup:

```bash
/telegram setup
```

Useful in-chat controls:

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

The token command loads the token into the current process. It is never stored
in `config.yaml`. After a restart, paste it in the dashboard again or export
`DMDAGENT_TELEGRAM_BOT_TOKEN` before running `start web`.

Available Telegram commands:

```text
/id
/help
/approvals
/approve <id>
/deny <id>
```

## Testing And Verification

Run backend tests with the standard library test runner:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

Run a Python syntax/import pass:

```bash
.venv/bin/python -m compileall -q src tests
```

Build the frontend:

```bash
cd frontend
npm run build
```

Optional, if `pytest` is installed:

```bash
.venv/bin/python -m pytest -q
```

Current note: the frontend package has `dev`, `build`, and `preview` scripts.
There is no `lint` script yet.

## Language Policy

All install flow, terminal output, documentation, approval text, and product UI
copy are English by default.

The assistant can answer in any language supported by the selected model. The
default response mode is auto-detect from the user's message.

## Recommended Next Implementation Targets

The next Codex should choose one narrow target, implement it end to end, and
verify it with tests.

Recommended order:

1. Harden Telegram operational behavior.
   Document a launchd/systemd service example and consider dashboard-driven
   lifecycle controls for starting/stopping polling.
2. Broaden OS-backed secret storage.
   Extend the local secret store beyond Telegram and add OS-backed secure
   backends where needed. Keep secrets out of config and model context.
3. Add OAuth/Graph email connector modes.
   IMAP/SMTP works for app-password accounts. Gmail OAuth and Microsoft Graph
   should be added for accounts where basic IMAP/SMTP auth is disabled.
4. Package optional browser runtime setup.
   Add an installer path for `.[browser]` and Playwright Chromium so browser
   interaction tools are easy to enable.
5. Prepare release packaging.
   Add signed macOS packaging or another release artifact once signing identity
   and distribution target are available.
6. Improve frontend coverage and scripts.
   Add a lint script, consider component tests, and keep the dashboard focused
   on operational control rather than marketing UI.

## Handoff Prompt For The Next Codex

Use this prompt when continuing the project in another Codex session:

```text
You are continuing the DMD Agent 4 All repository.

First read README.md completely. Treat it as the product brief, architecture
contract, and security policy.

Project summary:
- This is a local-first AI control center for personal machines and small
  servers.
- The LLM is an untrusted planner, never a security boundary.
- All actions must go through registered backend tools, tool manifests, the
  permission engine, approval queue, and audit logging.
- Secrets must never be stored in config, committed files, prompts, logs, or
  model context.
- High-risk tools and remote interfaces are disabled by default.
- Telegram is now implemented as an allowlisted polling interface with approval
  buttons, but it still needs operational hardening and/or dashboard controls.
- terminal.run is now wired through the same manifest, permission, approval,
  workspace, timeout, and audit path. Keep it disabled unless explicitly enabled.
- dmdagent doctor is the local readiness/security check and should stay aligned
  with backend and dashboard behavior.

Before editing:
1. Run `git status --short`.
2. Inspect the files related to the task.
3. Do not revert user changes or unrelated work.
4. Keep documentation and UI copy in English.

Useful verification commands:
- `.venv/bin/python -m unittest discover -s tests -v`
- `.venv/bin/python -m compileall -q src tests`
- `cd frontend && npm run build`

When adding capabilities:
- Add or update tool manifests first.
- Keep tools disabled by default unless they are harmless local reads.
- Add tests for allowed, denied, approval-required, and audit behavior.
- Never let Telegram, CLI, web UI, or the planner bypass AgentCore and
  PermissionEngine.
- Prefer small, complete increments over broad rewrites.

Recommended next task:
Pick one remaining item from "Recommended Next Implementation Targets" in
README.md, implement it end to end, run the verification commands, and summarize
exactly what changed.
```

## License

MIT
