from __future__ import annotations

import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmdagent4all.security.policy import command_contains_destructive_sql, command_references_secret


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
    "/var",
    "/private",
    "/Library",
    "/System",
)


_ACTIVE_PROCESSES: dict[str, subprocess.Popen[str]] = {}
_ACTIVE_LOCK = threading.RLock()


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
    allow_safe_workspace_commands: bool = True

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
            allow_safe_workspace_commands=bool(
                terminal.get("allow_safe_workspace_commands", True)
            ),
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
        if command_references_secret(command):
            raise PermissionError("Command references a blocked secret or system path.")
        if command_contains_destructive_sql(command):
            raise PermissionError("Command contains a blocked destructive SQL operation.")
        if tuple(command) not in self.allowed_commands and not self._is_safe_workspace_command(command):
            raise PermissionError("Command is not in the allowlist.")

    def _is_safe_workspace_command(self, command: list[str]) -> bool:
        if not self.allow_safe_workspace_commands or not self.workspace_only:
            return False
        binary_name = Path(command[0]).name
        if binary_name != "mkdir":
            return False
        args = command[1:]
        if not args:
            return False
        if args[0] == "-p":
            args = args[1:]
        elif any(arg.startswith("-") for arg in args):
            return False
        return bool(args) and all(_is_safe_relative_path(arg) for arg in args)


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
    command_id = uuid.uuid4().hex
    process = subprocess.Popen(
        command,
        cwd=run_cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    with _ACTIVE_LOCK:
        _ACTIVE_PROCESSES[command_id] = process
    try:
        try:
            stdout, stderr = process.communicate(timeout=policy.timeout_seconds)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            stderr = f"{stderr}\nCommand timed out after {policy.timeout_seconds} seconds.".strip()
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.pop(command_id, None)
    return {
        "command_id": command_id,
        "command": command,
        "cwd": str(run_cwd),
        "returncode": process.returncode,
        "stdout": _limit_output(stdout, policy.max_output_chars),
        "stderr": _limit_output(stderr, policy.max_output_chars),
    }


def active_terminal_processes() -> list[dict[str, object]]:
    with _ACTIVE_LOCK:
        return [
            {
                "command_id": command_id,
                "pid": process.pid,
                "running": process.poll() is None,
            }
            for command_id, process in _ACTIVE_PROCESSES.items()
        ]


def emergency_stop_terminal_processes(*, grace_seconds: float = 1.5) -> dict[str, object]:
    with _ACTIVE_LOCK:
        processes = list(_ACTIVE_PROCESSES.items())
    terminated: list[dict[str, object]] = []
    for command_id, process in processes:
        if process.poll() is not None:
            continue
        process.terminate()
        terminated.append({"command_id": command_id, "pid": process.pid, "signal": "terminate"})
    deadline = time.monotonic() + grace_seconds
    while time.monotonic() < deadline:
        if all(process.poll() is not None for _, process in processes):
            break
        time.sleep(0.05)
    killed: list[dict[str, object]] = []
    for command_id, process in processes:
        if process.poll() is None:
            process.kill()
            killed.append({"command_id": command_id, "pid": process.pid, "signal": "kill"})
    return {
        "terminated": terminated,
        "killed": killed,
        "active_before": len(processes),
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


def _is_safe_relative_path(value: str) -> bool:
    if not value.strip() or value.startswith("~"):
        return False
    path = Path(value)
    if path.is_absolute():
        return False
    if any(part in {"", ".", ".."} for part in path.parts):
        return False
    return True
