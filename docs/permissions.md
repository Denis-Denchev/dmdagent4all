# Permissions

The permission engine is the security boundary for tool usage.

## Tool Request Shape

The model may propose:

```json
{
  "tool": "gmail.summarize_inbox",
  "args": {
    "limit": 10
  },
  "reason": "User asked for today's email summary"
}
```

The backend decides whether the request is allowed.

## Risk Levels

| Level | Meaning | Examples |
|---|---|---|
| 0 | Harmless local read | show memory, list enabled tools |
| 1 | Read-only external data | read Gmail, read Calendar |
| 2 | Local draft/write | create draft, save markdown note |
| 3 | Modify external resource | create calendar event, archive email |
| 4 | Send or execute action | send email, push commit |
| 5 | Dangerous/system/security action | sudo, delete files, deploy, SSH production |

## Defaults

- Risk 0 can run if tool is enabled.
- Risk 1 can run if connector permission is granted.
- Risk 2 may require approval depending on the tool.
- Risk 3+ requires approval by default.
- Risk 5 tools are disabled by default.

## Approval Flow

When a tool requires approval, the backend stores the exact pending request:

```text
dmdagent approvals list
dmdagent approvals approve <id>
dmdagent approvals deny <id>
```

Approving a request executes that stored request once. Approval does not bypass tool enablement or missing connector permissions.
In the dashboard, clicking Approve or typing `approve <id>` executes the stored
request and posts the tool result back into the chat log. Typing `готово` or
bare `approve` approves the newest pending request.

## Terminal

Terminal access is disabled by default.

Allowed terminal execution must use:

- command arrays, not shell strings
- exact allowlisted commands
- workspace-only current directory
- timeout
- no `sudo`
- no SSH
- no `.env`
- no `.ssh`
- no `docker.sock`
- no host home access

Operational flow:

```bash
dmdagent terminal status
dmdagent terminal workspace /Users/Apple/PycharmProjects/dmdagent4all
dmdagent terminal allow git status
dmdagent terminal enable --tool --grant-permission
dmdagent terminal run -- git status
dmdagent approvals approve <id>
dmdagent terminal auto-approve on
```

`terminal.run` is a risk 5 tool. Enabling the terminal policy, enabling the
tool manifest, and granting `terminal.run` permission still keeps execution
approval-gated by default. If exact allowlist auto-approve is enabled, only
commands that exactly match the allowlist can run without creating a new
approval. Blocked commands, non-allowlisted commands, and unsafe cwd values stay
blocked.

The web dashboard exposes the same policy state, exact allowlist editing,
workspace root, settings, the auto-approve toggle, and run requests through the
Terminal view.

## Cloud Context

If a cloud model is active and a tool is marked `cloud_allowed: false`, the backend must require explicit approval before private tool output is sent into model context.
