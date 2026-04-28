# Security Policy

## Security Model

DMD Agent 4 All treats the language model as untrusted.

The model can propose a structured request:

```json
{
  "tool": "gmail.summarize_inbox",
  "args": {
    "limit": 10
  },
  "reason": "User asked for today's email summary"
}
```

The backend then validates the request through the permission engine. The model does not get raw access to the operating system, secrets, tokens, shell, SSH keys, `.env` files, browser profiles, or connector credentials.

## Hard Security Rules

- No direct `LLM -> shell`.
- No raw tokens or credentials in model context.
- No personal browser profile access.
- No host home directory mount into sandboxed tools.
- No `docker.sock` access from sandboxes.
- No terminal access by default.
- No destructive tool defaults.
- No automatic durable memory write from untrusted content.
- Every external write/send/submit action requires backend enforcement and audit logging.

## Risk Levels

| Level | Meaning | Examples |
|---|---|---|
| 0 | Harmless local read | list enabled tools, show local memory |
| 1 | Read-only external data | read Gmail, read Calendar, read GitHub issue |
| 2 | Local draft/write | create email draft, save markdown note |
| 3 | Modify external resource | create calendar event, archive email |
| 4 | Send or execute action | send email, run command, push commit |
| 5 | Dangerous/system/security action | sudo, delete files, deploy, SSH production |

## Reporting Vulnerabilities

This project is early-stage. For now, report security issues privately to the repository owner before opening public issues.

Please include:

- affected version or commit
- reproducible steps
- expected impact
- logs with secrets removed

Do not include real tokens, private keys, passwords, private email bodies, or personal calendar data in reports.
