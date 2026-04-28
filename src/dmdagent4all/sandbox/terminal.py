from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


ALWAYS_BLOCKED = {
    "sudo",
    "su",
    "rm",
    "chmod",
    "chown",
    "dd",
    "mkfs",
    "ssh",
    "scp",
}

BLOCKED_ARG_FRAGMENTS = (
    ".env",
    ".ssh",
    "id_rsa",
    "id_ed25519",
    "docker.sock",
    "/etc/passwd",
    "/etc/shadow",
)


@dataclass(frozen=True)
class TerminalPolicy:
    enabled: bool = False
    allowed_commands: tuple[tuple[str, ...], ...] = (
        ("pwd",),
        ("ls",),
        ("git", "status"),
        ("git", "diff"),
        ("npm", "test"),
        ("pytest",),
    )
    timeout_seconds: int = 30

    @classmethod
    def from_config(cls, config: dict) -> "TerminalPolicy":
        terminal = config.get("terminal", {})
        allowed = tuple(tuple(item) for item in terminal.get("allowed_commands", []))
        return cls(
            enabled=bool(terminal.get("enabled", False)),
            allowed_commands=allowed or cls.allowed_commands,
            timeout_seconds=int(terminal.get("timeout_seconds", 30)),
        )

    def validate(self, command: list[str]) -> None:
        if not self.enabled:
            raise PermissionError("Terminal access is disabled.")
        if not command:
            raise PermissionError("Command is empty.")
        if command[0] in ALWAYS_BLOCKED:
            raise PermissionError(f"Command is always blocked: {command[0]}")
        if any(fragment in arg for arg in command for fragment in BLOCKED_ARG_FRAGMENTS):
            raise PermissionError("Command references a blocked secret or system path.")
        if tuple(command) not in self.allowed_commands:
            raise PermissionError("Command is not in the allowlist.")


def run_workspace_command(
    command: list[str],
    *,
    workspace: Path,
    policy: TerminalPolicy,
) -> dict[str, object]:
    policy.validate(command)
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        cwd=workspace,
        text=True,
        capture_output=True,
        timeout=policy.timeout_seconds,
        check=False,
    )
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
