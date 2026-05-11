# Architecture

DMD Agent 4 All is a local AI control center. It is not an AI process with unrestricted computer access.

## Runtime Flow

```text
User Interfaces
Web UI / Telegram / CLI
        |
Agent Core
Minimal Controls / LLM Decision / Policy-Gated Tools / Memory
        |
WorkspaceManager / ToolSafetyPolicy
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
- handles only minimal deterministic controls before model routing: slash/help-style commands, approve/deny, emergency stop/reset, explicit JSON tool requests, chat-history lookup, and session corrections
- asks the LLM for a structured decision for normal chat, tool calls, multi-tool plans, clarification questions, and memory search
- lets the LLM choose `files.write` for natural code/file edit requests instead of relying on phrase regexes
- treats `cd <folder>`, `open <folder>`, and "go into folder" as workspace changes when selected by the LLM or fallback router, not as stateless shell `cd`
- supports short LLM multi-tool plans while approving the first risky step before any continuation runs
- blocks tool execution when emergency stop mode is active
- keeps older regex/deterministic parsers only as local fallback when the LLM is unavailable
- sends tool requests through backend safety policy and then the permission engine
- returns natural-language tool results to the user

The core does not bypass the safety policy or permission engine.

## Runtime Routing

```text
User message
  -> minimal deterministic controls
  -> LLM structured decision
  -> answer | tool_call | multi_tool_plan | ask_clarification | memory_search
  -> ToolSafetyPolicy
  -> PermissionEngine / approvals
  -> Tool executor
  -> Answer synthesis / response shaping
```

The LLM decision schema supports:

- `answer` for normal conversation and simple Q&A
- `tool_call` for one backend tool
- `multi_tool_plan` for short ordered tool plans
- `ask_clarification` when required inputs are missing
- `memory_search` when the user asks about saved local facts

Explicit browser, terminal, email, file, reminder, and workspace intents are not
answered by early memory regexes. Memory lookup is now a decision, so a request
such as `scrape sportal.bg` routes to `browser.scrape_markdown` instead of being
stolen by unrelated saved memory.

Planner mode uses a smaller planner history window and `llm.planner_max_tokens`.
Tool result synthesis uses `llm.synthesis_max_tokens`. Planner JSON repair uses
`llm.repair_max_tokens`.

If the LLM is unavailable, the backend falls back to the older deterministic
router for local/offline usability. The fallback still goes through the same
workspace, safety, permission, and approval gates.

## Workspace Manager

Workspace policy is config driven:

```yaml
workspace:
  default_path: ""
  current_path: ""
  allowed_roots: []
  blocked_paths:
    - ~/.ssh
    - ~/.aws
    - ~/.config/gcloud
    - ~/.kube
    - /etc
    - /var
    - /private
    - /Library
    - /System
```

Empty `allowed_roots` falls back to the current/default workspace plus common
user project folders when present. Every user path is expanded, normalized, and
resolved through symlinks before policy checks.

`workspace.switch` can change the current workspace only when the target is
inside an allowed root and outside blocked paths.

Relative workspace targets are resolved against the current workspace. This keeps
terminal, file, and workspace tools on the same path base.

## Safety Policy

`ToolSafetyPolicy` runs before approvals and before tool execution. It blocks:

- secret filenames such as `.env`, `.env.*`, `*.pem`, `*.key`, `id_rsa`,
  `id_ed25519`, `credentials.json`, `token.json`, `secret.*`, `secrets.*`,
  `*.p12`, `*.pfx`, and `*.kubeconfig`
- blocked system paths configured through `workspace.blocked_paths`
- destructive SQL in terminal or SQL/query tool arguments
- file reads/deletes outside allowed workspace roots
- file writes outside allowed workspace roots
- workspace switches that escape allowed roots, including symlink escapes

## Developer Context Tool

`developer.context` gives the planner safe coding context without granting broad
filesystem access. It returns the current workspace, a filtered file tree,
optional previews of explicitly requested non-secret files, and the available
developer actions (`files.read`, `files.write`, `terminal.run`,
`workspace.switch`).

It excludes secret files and common generated folders. It does not edit files or
run commands itself. Code changes still go through `files.write` approval, and
terminal commands still go through `terminal.run` policy.

## LLM Provider Layer

Providers are isolated behind a common interface:

- Ollama provider for local models
- OpenAI-compatible provider for OpenAI-style cloud or local API servers
- DeepSeek provider via `https://api.deepseek.com` and `DMDAGENT_DEEPSEEK_API_KEY`
- Cloud-provider stubs for providers that still need explicit implementation

Default:

```yaml
llm:
  provider: ollama
  model: qwen3:8b
  base_url: http://localhost:11434
  api_key_env: null
  chat_max_tokens: 1024
  planner_max_tokens: 512
  synthesis_max_tokens: 1024
  repair_max_tokens: 256
  chat_history_turns: 24
  chat_history_char_limit: 12000
  planner_history_turns: 4
  planner_history_char_limit: 3000
```

Cloud providers require explicit user setup, API key values in environment
variables, and privacy approvals before private connector context is sent.

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

- `files.*`
- `memory.*`
- `workspace.*`
- `system.*`
- `gmail.*` IMAP/SMTP email tools
- `outlook.*` IMAP/SMTP email tools
- `calendar.*` manifests
- `terminal.*` manifest and approval-gated workspace runner
- `browser.*` manifests

High-risk tools are disabled by default.

`browser.scrape_markdown` supports `raw_page` mode for full page Markdown and
`targeted` mode for article extraction. Targeted article mode uses static HTML
only in the default runtime: JSON-LD `NewsArticle`, `<article>` elements,
headings with nearby links, and article-like links are converted into compact
Markdown sections. If targeted extraction cannot find candidates, the tool marks
the result as a raw fallback instead of pretending the targeted scrape worked.
Playwright/rendered scraping remains optional and is not the default path.

`terminal.run` is intentionally narrow: command arrays only, exact allowlist,
workspace-only cwd, timeout, output limit, redaction, audit, and approval.

`files.delete` is always approval-gated. Directory deletion is risk 5.
Secret paths are denied, even inside an allowed workspace.

`files.write` creates or overwrites text files only after approval and only
inside allowed workspace roots. Interactive terminal editors such as `nano` and
`vim` are not run from the dashboard; the agent should offer `files.write`
instead.

## Product Runtime

Normal users can run the project through Docker Compose. In Docker mode, the
backend serves the built React dashboard directly, so the user opens only:

```text
http://127.0.0.1:8765
```

The Compose stack starts the app container and an Ollama container. The default
workspace is the host `./workspace` folder mounted into the app container.

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
