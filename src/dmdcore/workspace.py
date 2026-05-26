from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SECRET_NAME_PATTERNS = (
    ".env",
    ".env.*",
    "*.env",
    "*.pem",
    "*.key",
    "id_rsa",
    "id_ed25519",
    "credentials.json",
    "token.json",
    "secrets.*",
    "secret.*",
    "*.p12",
    "*.pfx",
    "*.kubeconfig",
)

DEFAULT_BLOCKED_PATHS = (
    "~/.ssh",
    "~/.aws",
    "~/.config/gcloud",
    "~/.kube",
    "/etc",
    "/var",
    "/private",
    "/Library",
    "/System",
)


class WorkspaceError(ValueError):
    pass


@dataclass(frozen=True)
class WorkspaceManager:
    default_workspace: Path
    current_workspace: Path
    allowed_roots: tuple[Path, ...]
    blocked_paths: tuple[Path, ...]
    secret_name_patterns: tuple[str, ...] = SECRET_NAME_PATTERNS

    @classmethod
    def from_config(cls, config: dict[str, Any], *, fallback_workspace: Path) -> "WorkspaceManager":
        workspace = config.get("workspace", {})
        if not isinstance(workspace, dict):
            workspace = {}
        terminal = config.get("terminal", {})
        if not isinstance(terminal, dict):
            terminal = {}

        default_raw = workspace.get("default_path") or terminal.get("workspace_root") or str(fallback_workspace)
        default_workspace = _resolve_path(default_raw)

        current_raw = workspace.get("current_path") or terminal.get("workspace_root") or str(default_workspace)
        current_workspace = _resolve_path(current_raw)

        allowed_roots = _configured_paths(workspace.get("allowed_roots"))
        if not allowed_roots:
            allowed_roots = _default_allowed_roots(default_workspace)

        if "blocked_paths" in workspace:
            blocked_paths = _configured_paths(workspace.get("blocked_paths"))
        else:
            blocked_paths = _configured_paths(DEFAULT_BLOCKED_PATHS)

        return cls(
            default_workspace=default_workspace,
            current_workspace=current_workspace,
            allowed_roots=allowed_roots,
            blocked_paths=blocked_paths,
        )

    def workspace_info(self) -> dict[str, Any]:
        return {
            "default_workspace": str(self.default_workspace),
            "current_workspace": str(self.current_workspace),
            "allowed_roots": [str(path) for path in self.allowed_roots],
            "blocked_paths": [str(path) for path in self.blocked_paths],
        }

    def validate_workspace(self, raw_path: str | Path) -> Path:
        resolved = self.resolve_user_path(raw_path)
        if not self._is_inside_any(resolved, self.allowed_roots):
            raise WorkspaceError("Workspace path is outside allowed roots.")
        if self.is_blocked_path(resolved):
            raise WorkspaceError("Workspace path is blocked by policy.")
        if self.is_secret_path(resolved):
            raise WorkspaceError("Workspace path references a blocked secret path.")
        return resolved

    def resolve_user_path(self, raw_path: str | Path, *, base: Path | None = None) -> Path:
        raw = Path(str(raw_path)).expanduser()
        if raw.is_absolute():
            candidate = raw
        else:
            candidate = (base or self.current_workspace) / raw
        return _resolve_path(candidate)

    def validate_user_path(
        self,
        raw_path: str | Path,
        *,
        base: Path | None = None,
        allow_missing: bool = False,
    ) -> Path:
        resolved = self.resolve_user_path(raw_path, base=base)
        if not allow_missing and not resolved.exists():
            raise WorkspaceError("Path does not exist.")
        if not self._is_inside_any(resolved, self.allowed_roots):
            raise WorkspaceError("Path is outside allowed roots.")
        if self.is_blocked_path(resolved):
            raise WorkspaceError("Path is blocked by policy.")
        if self.is_secret_path(resolved):
            raise WorkspaceError("Path references a blocked secret file or directory.")
        return resolved

    def is_secret_path(self, path: Path | str) -> bool:
        resolved = _resolve_path(path)
        if self.is_blocked_path(resolved):
            return True
        for part in resolved.parts:
            if any(fnmatch.fnmatchcase(part, pattern) for pattern in self.secret_name_patterns):
                return True
        return False

    def is_blocked_path(self, path: Path | str) -> bool:
        resolved = _resolve_path(path)
        for blocked in self.blocked_paths:
            if resolved == blocked or blocked in resolved.parents:
                return True
        return False

    @staticmethod
    def _is_inside_any(path: Path, roots: tuple[Path, ...]) -> bool:
        for root in roots:
            if path == root or root in path.parents:
                return True
        return False


def _configured_paths(value: Any) -> tuple[Path, ...]:
    if isinstance(value, tuple):
        value = list(value)
    if not isinstance(value, list):
        return ()
    paths: list[Path] = []
    for item in value:
        if isinstance(item, Path):
            paths.append(_resolve_path(item))
        elif isinstance(item, str) and item.strip():
            paths.append(_resolve_path(item))
    return tuple(paths)


def _resolve_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve(strict=False)


def _default_allowed_roots(default_workspace: Path) -> tuple[Path, ...]:
    roots = [default_workspace]
    home = Path.home()
    for name in ("PycharmProjects", "Desktop", "Documents"):
        candidate = home / name
        if candidate.exists():
            roots.append(_resolve_path(candidate))
    unique: list[Path] = []
    for root in roots:
        resolved = _resolve_path(root)
        if resolved not in unique:
            unique.append(resolved)
    return tuple(unique)
