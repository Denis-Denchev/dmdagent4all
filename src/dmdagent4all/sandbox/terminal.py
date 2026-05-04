from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
        ("cat", "README.md"),
        ("cat", "readme.md"),
        ("npm", "test"),
        ("pytest",),
    )
    timeout_seconds: int = 30
    workspace_only: bool = True
    max_output_chars: int = 20000

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TerminalPolicy":
        terminal = config.get("terminal", {})
        allowed = tuple(
            tuple(str(part) for part in item)
            for item in terminal.get("allowed_commands", [])
            if isinstance(item, list | tuple) and item
        )
        return cls(
            enabled=bool(terminal.get("enabled", False)),
            allowed_commands=allowed or cls.allowed_commands,
            timeout_seconds=int(terminal.get("timeout_seconds", 30)),
            workspace_only=bool(terminal.get("workspace_only", True)),
            max_output_chars=int(terminal.get("max_output_chars", 20000)),
        )

    def validate(self, command: list[str]) -> None:
        if not self.enabled:
            raise PermissionError("Terminal access is disabled.")
        if not command:
            raise PermissionError("Command is empty.")
        if not all(isinstance(part, str) and part for part in command):
            raise PermissionError("Command must be a non-empty string array.")
        binary_name = Path(command[0]).name
        if binary_name in ALWAYS_BLOCKED:
            raise PermissionError(f"Command is always blocked: {binary_name}")
        lowered = [arg.casefold() for arg in command]
        if any(fragment.casefold() in arg for arg in lowered for fragment in BLOCKED_ARG_FRAGMENTS):
            raise PermissionError("Command references a blocked secret or system path.")
        if tuple(command) not in self.allowed_commands:
            raise PermissionError("Command is not in the allowlist.")


def run_workspace_command(
    command: list[str],
    *,
    workspace: Path,
    policy: TerminalPolicy,
    cwd: str | None = None,
) -> dict[str, object]:
    policy.validate(command)
    workspace = workspace.expanduser().resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    run_cwd = _resolve_cwd(workspace, cwd, policy)
    completed = subprocess.run(
        command,
        cwd=run_cwd,
        text=True,
        capture_output=True,
        timeout=policy.timeout_seconds,
        check=False,
    )
    return {
        "command": command,
        "cwd": str(run_cwd),
        "returncode": completed.returncode,
        "stdout": _limit_output(completed.stdout, policy.max_output_chars),
        "stderr": _limit_output(completed.stderr, policy.max_output_chars),
    }


def _resolve_cwd(workspace: Path, cwd: str | None, policy: TerminalPolicy) -> Path:
    if cwd is None or not cwd.strip():
        return workspace
    requested = Path(cwd).expanduser()
    if not requested.is_absolute():
        requested = workspace / requested
    resolved = requested.resolve()
    if policy.workspace_only and resolved != workspace and workspace not in resolved.parents:
        raise PermissionError("Terminal cwd must stay inside the configured workspace.")
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _limit_output(value: str, max_chars: int) -> str:
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + "\n[output truncated]"
