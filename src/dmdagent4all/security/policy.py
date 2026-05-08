from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from dmdagent4all.permissions import RiskLevel, ToolRequest
from dmdagent4all.workspace import SECRET_NAME_PATTERNS, WorkspaceError, WorkspaceManager

if TYPE_CHECKING:
    from dmdagent4all.tools.base import ToolRuntimeContext


DESTRUCTIVE_SQL_RE = re.compile(
    r"(?is)\b("
    r"delete|drop|truncate|alter|update|insert|create|replace|merge|grant|revoke"
    r")\b|vacuum\s+full"
)

READ_ONLY_SQL_RE = re.compile(r"(?is)^\s*(?:with\b.*?\bselect\b|select\b|show\b|explain\b)")


@dataclass(frozen=True)
class SafetyDecision:
    allowed: bool
    reason: str
    risk: RiskLevel | None = None

    @classmethod
    def allow(cls) -> "SafetyDecision":
        return cls(allowed=True, reason="Allowed by safety policy.")

    @classmethod
    def deny(cls, reason: str, *, risk: RiskLevel | None = None) -> "SafetyDecision":
        return cls(allowed=False, reason=reason, risk=risk)


class ToolSafetyPolicy:
    def evaluate(self, request: ToolRequest, context: ToolRuntimeContext) -> SafetyDecision:
        if not request.tool.startswith(("gmail.", "outlook.")):
            sql_reason = _destructive_sql_reason(request.args)
            if sql_reason is not None:
                return SafetyDecision.deny(sql_reason, risk=RiskLevel.DANGEROUS_SYSTEM)

        secret_reason = _secret_reference_reason(request.args)
        if secret_reason is not None:
            return SafetyDecision.deny(secret_reason, risk=RiskLevel.DANGEROUS_SYSTEM)

        manager = WorkspaceManager.from_config(
            context.config,
            fallback_workspace=context.workspace_root,
        )

        if request.tool in {"files.read", "files.write", "files.delete"}:
            path = request.args.get("path")
            if not isinstance(path, str) or not path.strip():
                return SafetyDecision.deny(f"{request.tool} requires a path.")
            try:
                manager.validate_user_path(
                    path,
                    allow_missing=request.tool == "files.write",
                )
            except WorkspaceError as exc:
                return SafetyDecision.deny(str(exc), risk=RiskLevel.DANGEROUS_SYSTEM)

        if request.tool == "workspace.switch":
            path = request.args.get("path")
            if not isinstance(path, str) or not path.strip():
                return SafetyDecision.deny("workspace.switch requires a path.")
            try:
                manager.validate_workspace(path)
            except WorkspaceError as exc:
                return SafetyDecision.deny(str(exc), risk=RiskLevel.DANGEROUS_SYSTEM)

        if request.tool == "developer.context":
            try:
                raw_workspace = request.args.get("workspace")
                base = (
                    manager.validate_workspace(raw_workspace)
                    if isinstance(raw_workspace, str) and raw_workspace.strip()
                    else manager.current_workspace
                )
                focus_paths = request.args.get("focus_paths")
                if isinstance(focus_paths, list):
                    for focus_path in focus_paths:
                        if isinstance(focus_path, str) and focus_path.strip():
                            manager.validate_user_path(focus_path, base=base)
            except WorkspaceError as exc:
                return SafetyDecision.deny(str(exc), risk=RiskLevel.DANGEROUS_SYSTEM)

        return SafetyDecision.allow()


def contains_destructive_sql(value: str) -> bool:
    return bool(DESTRUCTIVE_SQL_RE.search(value))


def is_read_only_sql(value: str) -> bool:
    stripped = _strip_sql_comments(value).strip()
    if not stripped:
        return False
    if contains_destructive_sql(stripped):
        return False
    return bool(READ_ONLY_SQL_RE.match(stripped))


def command_references_secret(command: Iterable[str]) -> bool:
    return _secret_reference_reason({"command": list(command)}) is not None


def command_contains_destructive_sql(command: Iterable[str]) -> bool:
    parts = [str(part) for part in command]
    if not parts:
        return False
    binary = Path(parts[0]).name.casefold()
    joined = " ".join(parts)
    if binary in {"psql", "sqlite3", "mysql", "mariadb", "duckdb"}:
        return contains_destructive_sql(joined)
    if any(flag in parts for flag in {"-c", "-e", "--execute", "--command"}):
        return contains_destructive_sql(joined)
    sql_context = re.search(r"(?is)\b(from|table|database|schema|where|into|values)\b", joined)
    return bool(sql_context and contains_destructive_sql(joined))


def _destructive_sql_reason(args: Any) -> str | None:
    for key, value in _iter_keyed_strings(args):
        if key.casefold() not in {"command", "sql", "query", "statement"}:
            continue
        values = value if isinstance(value, list) else [value]
        joined = " ".join(str(item) for item in values)
        if contains_destructive_sql(joined):
            return "Blocked destructive or mutating SQL operation."
    return None


def _secret_reference_reason(args: Any) -> str | None:
    for key, value in _iter_keyed_strings(args):
        if not _is_path_like_key(key) and key != "command":
            continue
        values = value if isinstance(value, list) else [value]
        for item in values:
            if not isinstance(item, str):
                continue
            if _looks_like_secret_reference(item):
                return "Blocked access to a secret or system path."
    return None


def _looks_like_secret_reference(value: str) -> bool:
    text = value.strip()
    if not text:
        return False
    normalized = text.replace("\\", "/")
    lowered = normalized.casefold()
    blocked_fragments = (
        "/.ssh",
        "/.aws",
        "/.config/gcloud",
        "/.kube",
        "node_modules/.cache",
    )
    if any(fragment in lowered for fragment in blocked_fragments):
        return True
    parts = [part for part in Path(normalized).parts if part not in {"/", ""}]
    return any(
        re.fullmatch(_glob_to_regex(pattern), part, flags=re.IGNORECASE)
        for part in parts
        for pattern in SECRET_NAME_PATTERNS
    )


def _is_path_like_key(key: str) -> bool:
    lowered = key.casefold()
    return any(
        marker in lowered
        for marker in {
            "path",
            "file",
            "filename",
            "cwd",
            "directory",
            "dir",
            "root",
            "workspace",
            "url",
        }
    )


def _iter_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, list | tuple):
        for item in value:
            yield from _iter_strings(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_strings(item)


def _iter_keyed_strings(value: Any, *, key: str = "") -> Iterable[tuple[str, Any]]:
    if isinstance(value, str):
        yield key, value
    elif isinstance(value, list | tuple):
        if key:
            yield key, list(value)
        for item in value:
            yield from _iter_keyed_strings(item, key=key)
    elif isinstance(value, dict):
        for item_key, item_value in value.items():
            yield from _iter_keyed_strings(item_value, key=str(item_key))


def _glob_to_regex(pattern: str) -> str:
    escaped = re.escape(pattern).replace(r"\*", ".*")
    return f"^{escaped}$"


def _strip_sql_comments(value: str) -> str:
    without_block = re.sub(r"/\*.*?\*/", "", value, flags=re.DOTALL)
    return re.sub(r"--.*?$", "", without_block, flags=re.MULTILINE)
