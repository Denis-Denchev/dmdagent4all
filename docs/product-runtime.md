# Product Runtime

DMD Agent borrows useful product patterns from projects like OpenClaw while
keeping a stricter local safety model.

## Borrowed Product Patterns

- one-command bootstrap from GitHub
- first-run onboarding wizard
- long-running local control center
- dashboard and API started together
- Docker Compose deployment for users who do not want Python/Node setup
- local model runtime through Ollama
- channel-style interfaces such as Telegram behind allowlists
- doctor/diagnostics commands
- tool registry instead of model-invented actions

## Differences

DMD Agent intentionally stays more conservative:

- no whole-home filesystem mount in Docker mode
- default Docker workspace is only `./workspace`
- `.env`, private keys, token files, and secret/system paths are blocked by backend policy
- destructive or mutating SQL is blocked before approval and before execution
- risky tool execution needs explicit approval
- emergency stop kills active terminal processes and blocks further tool execution

## Docker Runtime

`docker-compose.yml` starts:

- `dmdagent`: FastAPI backend plus built React dashboard
- `ollama`: local model runtime

The app is available at:

```text
http://127.0.0.1:8765
```

Persistent state lives in Docker volumes. User work happens through the mounted
`./workspace` folder.

## Emergency Stop

Emergency stop is a backend control, not a prompt instruction.

When triggered, it:

- terminates active terminal subprocesses
- force-kills subprocesses that do not exit after a short grace period
- cancels pending approvals
- stops background Telegram and reminder polling
- stores emergency mode in config
- blocks further tool and approval execution until reset

The dashboard exposes Emergency Stop and Reset Emergency buttons.
