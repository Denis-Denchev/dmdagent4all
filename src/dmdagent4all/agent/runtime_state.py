from __future__ import annotations

import os
import platform
import time
from pathlib import Path
from typing import Any

from dmdagent4all.autonomy import local_dev_autonomy_enabled
from dmdagent4all.security.policy import command_references_secret
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.workspace import WorkspaceManager


_CACHE_TTL_SECONDS = 2.0
_MAX_INDEXED_ITEMS = 80
_TEXT_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".log",
    ".markdown",
    ".md",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
_CACHE: dict[str, Any] = {"key": None, "expires_at": 0.0, "snapshot": None}


def runtime_state_snapshot(
    *,
    runtime_context: ToolRuntimeContext,
    manager: WorkspaceManager,
    downloads_root: Path,
    memory_files: list[str],
    enabled_tools: list[str],
    disabled_tools: list[str],
    planner_active: bool,
) -> dict[str, Any]:
    config = runtime_context.config
    now = time.monotonic()
    key = _cache_key(runtime_context, manager, downloads_root, enabled_tools, disabled_tools, planner_active)
    if _CACHE.get("key") == key and float(_CACHE.get("expires_at") or 0.0) > now:
        cached = dict(_CACHE.get("snapshot") or {})
        cached["cache"] = {"ttl_seconds": _CACHE_TTL_SECONDS, "hit": True}
        return cached

    snapshot = {
        "schema_version": 1,
        "cache": {"ttl_seconds": _CACHE_TTL_SECONDS, "hit": False},
        "system": {
            "os": platform.platform(),
            "python": platform.python_version(),
            "process_cwd": os.getcwd(),
        },
        "mode": "full_llm_first_autonomy" if local_dev_autonomy_enabled(config) else "standard_safe",
        "planner": {"active": planner_active},
        "workspace": _folder_state(manager.current_workspace, manager=manager),
        "downloads": _folder_state(downloads_root, manager=manager),
        "memory": {
            "root": str(runtime_context.memory_root.resolve()),
            "markdown_files": memory_files[:_MAX_INDEXED_ITEMS],
            "markdown_file_count": len(memory_files),
        },
        "tools": {
            "enabled_count": len(enabled_tools),
            "disabled_count": len(disabled_tools),
            "enabled": enabled_tools[:_MAX_INDEXED_ITEMS],
            "disabled": disabled_tools[:_MAX_INDEXED_ITEMS],
        },
        "config": {
            "path": str(runtime_context.config_path or ""),
            "sections": sorted(str(key) for key in config.keys()),
        },
    }
    _CACHE.update({"key": key, "expires_at": now + _CACHE_TTL_SECONDS, "snapshot": snapshot})
    return snapshot


def invalidate_runtime_state_cache() -> None:
    _CACHE.update({"key": None, "expires_at": 0.0, "snapshot": None})


def _cache_key(
    runtime_context: ToolRuntimeContext,
    manager: WorkspaceManager,
    downloads_root: Path,
    enabled_tools: list[str],
    disabled_tools: list[str],
    planner_active: bool,
) -> tuple[Any, ...]:
    return (
        str(runtime_context.config_path or ""),
        str(manager.current_workspace),
        str(downloads_root),
        str(runtime_context.memory_root),
        tuple(enabled_tools),
        tuple(disabled_tools),
        planner_active,
        local_dev_autonomy_enabled(runtime_context.config),
    )


def _folder_state(path: Path, *, manager: WorkspaceManager) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    allowed = _is_inside_allowed_roots(resolved, manager)
    state: dict[str, Any] = {
        "path": str(resolved),
        "exists": resolved.exists(),
        "allowed": allowed,
        "folders": [],
        "files": [],
        "file_count": 0,
    }
    if not resolved.exists() or not resolved.is_dir() or not allowed:
        return state
    files: list[dict[str, Any]] = []
    folders: list[str] = []
    try:
        for item in resolved.iterdir():
            if command_references_secret([str(item)]):
                continue
            try:
                if item.is_dir():
                    folders.append(item.name)
                    continue
                if item.suffix.casefold() not in _TEXT_SUFFIXES:
                    continue
                stat = item.stat()
                files.append(
                    {
                        "name": item.name,
                        "path": str(item.relative_to(resolved)),
                        "size": stat.st_size,
                        "modified_at": stat.st_mtime,
                    }
                )
            except OSError:
                continue
    except OSError:
        return state
    files.sort(key=lambda entry: float(entry.get("modified_at") or 0.0), reverse=True)
    folders.sort(key=str.casefold)
    state["folders"] = folders[:_MAX_INDEXED_ITEMS]
    state["files"] = files[:_MAX_INDEXED_ITEMS]
    state["file_count"] = len(files)
    return state


def _is_inside_allowed_roots(path: Path, manager: WorkspaceManager) -> bool:
    resolved = path.expanduser().resolve()
    for root in manager.allowed_roots:
        root_resolved = root.expanduser().resolve()
        if resolved == root_resolved or root_resolved in resolved.parents:
            return True
    return False
