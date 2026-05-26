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

Before the permission engine runs, `ToolSafetyPolicy` checks the request for
blocked paths, secret filenames, destructive SQL, and workspace escapes. A
safety denial cannot be bypassed by approval.

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
dmdcore approvals list
dmdcore approvals approve <id>
dmdcore approvals deny <id>
```

Approving a request executes that stored request once. Approval does not bypass tool enablement or missing connector permissions.
Approval also does not bypass safety policy. A destructive SQL request or secret
file read remains blocked even if the stored approval is executed.
In the dashboard, clicking Approve or typing `approve <id>` executes the stored
request and posts the tool result back into the chat log. Typing `готово` or
bare `approve` approves the newest pending request.

Emergency stop cancels pending approvals and blocks approval execution until the
user resets emergency mode.

## Workspace Policy

Workspace configuration controls where local file and terminal tools can work:

```yaml
workspace:
  default_path: "/path/to/default/project"
  current_path: "/path/to/current/project"
  allowed_roots:
    - "/path/to/projects"
    - "~/Desktop"
    - "~/Documents"
  blocked_paths:
    - "~/.ssh"
    - "~/.aws"
    - "~/.config/gcloud"
    - "~/.kube"
    - "/etc"
    - "/var"
    - "/private"
    - "/Library"
    - "/System"
```

All paths are expanded, normalized, and resolved through symlinks before checks.
Path traversal and symlink escapes are denied.

The user can ask to switch workspace, but the backend validates the target
before storing it.

Requests such as `cd test`, `open test`, or "go into test" are interpreted as a
validated workspace switch. They are not executed as shell `cd`, because the
dashboard terminal runner is stateless between commands.

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
- no destructive SQL

Operational flow:

```bash
dmdcore terminal status
dmdcore terminal workspace /path/to/dmdcore
dmdcore terminal allow git status
dmdcore terminal enable --tool --grant-permission
dmdcore terminal run -- git status
dmdcore approvals approve <id>
dmdcore terminal auto-approve on
```

`terminal.run` is a risk 5 tool. Enabling the terminal policy, enabling the
tool manifest, and granting `terminal.run` permission still keeps execution
approval-gated by default. If exact allowlist auto-approve is enabled, only
commands that exactly match the allowlist can run without creating a new
approval. Blocked commands, non-allowlisted commands, and unsafe cwd values stay
blocked.

Commands such as `mkdir relative/path` may pass terminal validation as a safe
workspace command, but they are not exact allowlist auto-approved. They create an
approval request unless explicitly added to the exact allowlist.

Interactive terminal apps such as `nano`, `vim`, `vi`, and `emacs` are not
allowed in the dashboard command runner. Use file tools for create/edit flows.

## Developer Tooling

`developer.context` is a read-only coding helper. It can list the validated
workspace, return a filtered project tree, and preview explicitly requested
non-secret files so the planner can understand code tasks.

It does not bypass policy:

- `.env` and secret paths stay blocked
- code edits still use `files.write` and require approval
- terminal commands still use `terminal.run` and require policy/approval
- destructive SQL remains blocked before approval

Emergency stop terminates active terminal subprocesses. If a process ignores the
first terminate signal, the backend force-kills it after a short grace period.

## File Tools

`files.read` reads only non-secret text files inside allowed workspace roots.

`files.write` creates or overwrites text files only after approval. It is denied
for secret paths and paths outside allowed workspace roots.

`files.delete` always requires approval. File deletion does not execute
immediately. Directory deletion is risk 5 and also requires approval. Secret
paths are denied instead of approved.

## Database Safety

Database tools must be read-only by default. SQL containing destructive or
mutating operations is blocked by backend policy:

```text
DELETE DROP TRUNCATE ALTER UPDATE INSERT CREATE REPLACE MERGE GRANT REVOKE VACUUM FULL
```

The block applies to terminal commands and SQL/query tool arguments before the
permission engine and before approvals.

## Emergency Stop

Emergency stop is a backend safety control. It is available through the
dashboard and API:

```text
POST /v1/emergency/stop
POST /v1/emergency/reset
GET  /v1/emergency
```

When active, tool execution and approval execution are denied even if the model
or user asks to continue. Reset restores normal policy and approval behavior.

The web dashboard exposes the same policy state, exact allowlist editing,
workspace root, settings, the auto-approve toggle, and run requests through the
Terminal view.

## Cloud Context

If a cloud model is active and a tool is marked `cloud_allowed: false`, the backend must require explicit approval before private tool output is sent into model context.
