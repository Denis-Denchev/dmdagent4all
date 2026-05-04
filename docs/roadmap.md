# Roadmap

## Milestone 0: Foundation

- security docs
- threat model
- permission engine
- tool manifests
- audit DB
- Markdown memory
- model presets
- CLI wizard
- FastAPI skeleton
- doctor diagnostics

## Milestone 1: Local MVP

- chat endpoint connected to Ollama: started
- structured tool-call parsing: started
- approval queue with approve-and-execute: started
- memory viewer/editor API
- local reminder tools
- local workspace file tools
- web dashboard shell with connectors, terminal, Telegram, tools, approvals,
  memory, audit, models, and Doctor views
- OpenAI-compatible provider support
- terminal.run through approval-gated workspace policy

## Milestone 2: Connectors

- Gmail read-only connector
- Gmail draft connector
- Calendar free/busy connector
- Calendar read-only connector
- local calendar store create/update/delete/read/free-slot handlers
- connector status UI
- OS secret store integration

## Milestone 3: Remote Interfaces

- Telegram allowlisted bot
- approval buttons
- remote status commands
- dashboard setup controls

## Milestone 4: Sandboxed Automation

- terminal workspace sandbox: started
- Docker sandbox executor
- optional isolated browser profile through Playwright
- guarded browser read/extraction: started
- approval-gated browser interactions through optional runtime

## Milestone 5: Public Release

- signed macOS package or app bundle
- GitHub Releases
- website download page
- security review
- contributor guide
