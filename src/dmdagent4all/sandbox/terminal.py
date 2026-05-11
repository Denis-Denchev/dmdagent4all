from __future__ import annotations

import re
import os
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dmdagent4all.autonomy import local_dev_autonomy_enabled
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

SECRET_ENV_MARKERS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "PASSWORD",
    "SECRET",
    "TOKEN",
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
    local_dev_autonomy: bool = False
    sanitize_environment: bool = False

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TerminalPolicy":
        terminal = config.get("terminal", {})
        if not isinstance(terminal, dict):
            terminal = {}
        allowed = tuple(
            tuple(str(part) for part in item)
            for item in terminal.get("allowed_commands", [])
            if isinstance(item, list | tuple) and item
        )
        autonomy_active = local_dev_autonomy_enabled(config)
        return cls(
            enabled=bool(terminal.get("enabled", False)) or autonomy_active,
            allowed_commands=allowed or cls.allowed_commands,
            timeout_seconds=int(terminal.get("timeout_seconds", 30)),
            workspace_only=bool(terminal.get("workspace_only", True)),
            max_output_chars=int(terminal.get("max_output_chars", 20000)),
            allow_safe_workspace_commands=bool(
                terminal.get("allow_safe_workspace_commands", True)
            ),
            local_dev_autonomy=autonomy_active,
            sanitize_environment=bool(terminal.get("sanitize_environment", autonomy_active)),
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
        if (
            tuple(command) not in self.allowed_commands
            and not self._is_safe_workspace_command(command)
            and not self._is_local_dev_readonly_command(command)
            and not self._is_local_dev_workspace_script_command(command)
        ):
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

    def _is_local_dev_readonly_command(self, command: list[str]) -> bool:
        if not self.local_dev_autonomy or not self.workspace_only:
            return False
        if _has_shell_metacharacter(command):
            return False
        binary_name = Path(command[0]).name
        args = command[1:]
        if binary_name == "pwd":
            return not args
        if binary_name == "ls":
            return _readonly_ls_args(args)
        if binary_name == "find":
            return _readonly_find_args(args)
        if binary_name in {"cat", "head", "tail"}:
            return _readonly_file_read_args(args)
        if binary_name == "sed":
            return _readonly_sed_args(args)
        if binary_name in {"grep", "rg"}:
            return _readonly_search_args(args)
        if binary_name == "git":
            return _readonly_git_args(args)
        return False

    def is_local_dev_readonly_command(self, command: list[str]) -> bool:
        return self._is_local_dev_readonly_command(command)

    def _is_local_dev_workspace_script_command(self, command: list[str]) -> bool:
        if not self.local_dev_autonomy or not self.workspace_only:
            return False
        if _has_shell_metacharacter(command):
            return False
        script_index = _workspace_script_arg_index(command)
        if script_index is None:
            return False
        script = command[script_index]
        if not _is_safe_terminal_path(script, allow_current=False):
            return False
        if not _script_extension_allowed_for_command(Path(command[0]).name, Path(script).suffix.casefold()):
            return False
        return all(_is_safe_script_arg(arg) for arg in command[script_index + 1 :])

    def is_local_dev_workspace_script_command(self, command: list[str]) -> bool:
        return self._is_local_dev_workspace_script_command(command)


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
    if policy.local_dev_autonomy and policy.is_local_dev_workspace_script_command(command):
        _validate_workspace_script_execution(command, workspace=workspace, cwd=run_cwd)
    command_id = uuid.uuid4().hex
    process = subprocess.Popen(
        command,
        cwd=run_cwd,
        env=_subprocess_env(policy),
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


def _is_safe_terminal_path(value: str, *, allow_current: bool = True) -> bool:
    if not value.strip() or value.startswith("~"):
        return False
    if value == ".":
        return allow_current
    path = Path(value)
    if path.is_absolute():
        return False
    if any(part in {"", ".."} for part in path.parts):
        return False
    return True


def _workspace_script_arg_index(command: list[str]) -> int | None:
    if not command:
        return None
    binary_name = Path(command[0]).name.casefold()
    if binary_name.startswith("python"):
        if len(command) >= 2 and command[1] == "-u":
            return 2 if len(command) >= 3 else None
        if len(command) >= 2 and command[1] not in {"-c", "-m"} and not command[1].startswith("-"):
            return 1
        return None
    if binary_name == "node":
        if len(command) >= 2 and not command[1].startswith("-"):
            return 1
        return None
    return None


def _script_extension_allowed_for_command(binary_name: str, suffix: str) -> bool:
    lowered = binary_name.casefold()
    if lowered.startswith("python"):
        return suffix == ".py"
    if lowered == "node":
        return suffix in {".js", ".mjs", ".cjs"}
    return False


def _is_safe_script_arg(value: str) -> bool:
    if not value or _has_shell_metacharacter([value]):
        return False
    lowered = value.casefold()
    if any(fragment.casefold() in lowered for fragment in BLOCKED_ARG_FRAGMENTS):
        return False
    if value.startswith("~"):
        return False
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        return False
    return True


def _validate_workspace_script_execution(command: list[str], *, workspace: Path, cwd: Path) -> None:
    script_index = _workspace_script_arg_index(command)
    if script_index is None:
        return
    script = Path(command[script_index])
    resolved = script if script.is_absolute() else cwd / script
    resolved = resolved.resolve(strict=False)
    if workspace != resolved and workspace not in resolved.parents:
        raise PermissionError("Local dev script execution must stay inside the configured workspace.")
    if not resolved.is_file():
        raise PermissionError("Local dev script execution requires an existing workspace script file.")
    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")[:200_000]
    except OSError as exc:
        raise PermissionError(f"Could not inspect workspace script before execution: {exc}") from exc
    lowered = content.casefold()
    if _script_references_blocked_content(lowered):
        raise PermissionError("Workspace script references a blocked secret or system path.")


def _subprocess_env(policy: TerminalPolicy) -> dict[str, str] | None:
    if not policy.sanitize_environment:
        return None
    sanitized: dict[str, str] = {}
    for key, value in os.environ.items():
        upper = key.upper()
        if any(marker in upper for marker in SECRET_ENV_MARKERS):
            continue
        sanitized[key] = value
    return sanitized


def _script_references_blocked_content(lowered_content: str) -> bool:
    if re.search(r"(?<![A-Za-z0-9_])\.env(?![A-Za-z0-9_])", lowered_content):
        return True
    for fragment in BLOCKED_ARG_FRAGMENTS:
        lowered_fragment = fragment.casefold()
        if lowered_fragment == ".env":
            continue
        if lowered_fragment in lowered_content:
            return True
    return False


def _has_shell_metacharacter(command: list[str]) -> bool:
    blocked = {";", "|", "&", ">", "<", "`", "$(", "${"}
    return any(any(marker in part for marker in blocked) for part in command)


def _readonly_ls_args(args: list[str]) -> bool:
    return all(arg.startswith("-") or _is_safe_terminal_path(arg) for arg in args)


def _readonly_find_args(args: list[str]) -> bool:
    if not args:
        return True
    blocked = {"-exec", "-execdir", "-delete", "-ok", "-okdir", "-fdelete"}
    allowed_options = {
        "-maxdepth",
        "-mindepth",
        "-type",
        "-name",
        "-iname",
        "-path",
        "-not",
        "!",
        "-print",
        "-print0",
    }
    expect_value_for = ""
    for arg in args:
        if arg in blocked:
            return False
        if expect_value_for:
            if expect_value_for in {"-maxdepth", "-mindepth"} and not arg.isdigit():
                return False
            if expect_value_for == "-type" and arg not in {"f", "d", "l"}:
                return False
            if expect_value_for in {"-name", "-iname", "-path"} and (arg.startswith("/") or ".." in Path(arg).parts):
                return False
            expect_value_for = ""
            continue
        if arg in {"-maxdepth", "-mindepth", "-type", "-name", "-iname", "-path"}:
            expect_value_for = arg
            continue
        if arg in allowed_options:
            continue
        if not _is_safe_terminal_path(arg):
            return False
    return not expect_value_for


def _readonly_file_read_args(args: list[str]) -> bool:
    if not args:
        return False
    has_path = False
    skip_next = False
    for arg in args:
        if skip_next:
            if not arg.isdigit():
                return False
            skip_next = False
            continue
        if arg in {"-n", "-c"}:
            skip_next = True
            continue
        if arg.startswith("-"):
            if not arg[1:].isdigit():
                return False
            continue
        if not _is_safe_terminal_path(arg, allow_current=False):
            return False
        has_path = True
    return has_path and not skip_next


def _readonly_sed_args(args: list[str]) -> bool:
    if len(args) < 3 or args[0] != "-n":
        return False
    if args[1].startswith("-") or not _is_readonly_sed_address(args[1]):
        return False
    return all(_is_safe_terminal_path(arg, allow_current=False) for arg in args[2:])


def _is_readonly_sed_address(value: str) -> bool:
    return bool(re.fullmatch(r"\d+(?:,\d+)?p", value))


def _readonly_search_args(args: list[str]) -> bool:
    if len(args) < 2:
        return False
    paths_seen = 0
    skip_next = False
    for index, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg in {"-e", "-m", "-A", "-B", "-C", "--max-count", "--after-context", "--before-context", "--context"}:
            skip_next = True
            continue
        if arg.startswith("-"):
            if arg in {"--files-with-matches", "--line-number", "--ignore-case", "--hidden", "--no-ignore"}:
                continue
            if set(arg[1:]) <= set("RrInHhis"):
                continue
            return False
        if index == 0:
            continue
        if _is_safe_terminal_path(arg):
            paths_seen += 1
    return paths_seen > 0 and not skip_next


def _readonly_git_args(args: list[str]) -> bool:
    if not args:
        return False
    safe_subcommands = {"status", "diff", "log", "show", "branch", "rev-parse", "ls-files"}
    if args[0] not in safe_subcommands:
        return False
    blocked_fragments = ("--output", "--exec", "--upload-pack", "-c", "--config")
    if any(any(fragment in arg for fragment in blocked_fragments) for arg in args[1:]):
        return False
    return all(not arg.startswith("/") and ".." not in Path(arg).parts for arg in args[1:])
