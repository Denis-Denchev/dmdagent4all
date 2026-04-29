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
dmdagent terminal allow git status
dmdagent terminal enable --tool --grant-permission
dmdagent terminal run -- git status
dmdagent approvals approve <id>
```

`terminal.run` is a risk 5 tool. Enabling the terminal policy, enabling the
tool manifest, and granting `terminal.run` permission still does not execute
commands automatically. Each request is stored as an approval and executed once
only after approval.

## Cloud Context

If a cloud model is active and a tool is marked `cloud_allowed: false`, the backend must require explicit approval before private tool output is sent into model context.
