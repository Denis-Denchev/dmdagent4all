# Threat Model

## Core Assumption

The LLM is an untrusted planner. It can be wrong, manipulated, or maliciously instructed by content it reads.

Security must be enforced by backend code, not by prompts.

## Assets

- OAuth refresh tokens
- API keys
- SSH keys
- local files
- email content
- calendar details
- browser cookies
- local memory
- audit logs
- user approvals
- shell access
- production infrastructure access

## Untrusted Inputs

- user chat messages
- email bodies and attachments
- calendar event text
- web pages
- browser extraction results
- repository content
- tool outputs
- model output
- remote chat messages

## Main Threats

### Prompt Injection

Untrusted content may try to instruct the model to ignore policies, reveal secrets, call tools, or modify memory.

Mitigations:

- treat retrieved content as data, not instructions
- route every tool call through permission engine
- require approval for risky actions
- label source/provenance in context
- never expose secrets in context

### Excessive Agency

The model may attempt actions beyond user intent.

Mitigations:

- tool allowlists
- risk levels
- explicit permissions
- approvals
- audit logs
- disabled-by-default dangerous tools

### Sensitive Information Disclosure

The model may leak private information in responses or send it to cloud providers.

Mitigations:

- redaction layer
- cloud-disabled default
- per-tool cloud allowance
- approval before cloud context
- connector result minimization

### Memory Poisoning

Untrusted content may try to write false or malicious long-term memory.

Mitigations:

- no automatic memory writes from untrusted sources
- memory update proposals require approval by default
- Markdown frontmatter records source and confidence
- audit memory events

### Tool Misuse

Tools could be invoked with unsafe arguments.

Mitigations:

- typed tool schemas
- backend validation
- workspace path enforcement
- command specs instead of shell strings
- timeouts and resource limits

### Remote Interface Abuse

A public Telegram bot or compromised chat account could control the agent.

Mitigations:

- allowlisted user IDs
- no public bot mode
- approval buttons for risky actions
- short status commands
- audit all remote requests

## Out Of Scope For MVP

- autonomous production deployments
- SSH to production
- browser purchases
- destructive filesystem actions
- unrestricted terminal
- personal Chrome profile automation
