from __future__ import annotations

import json
from datetime import datetime, timezone
from urllib.parse import urlparse
from typing import Any

from dmdagent4all.memory import MemoryManager
from dmdagent4all.sandbox import TerminalPolicy, run_workspace_command
from dmdagent4all.security import redact_text
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.registry import ToolRegistry, load_builtin_manifests


def build_builtin_registry() -> ToolRegistry:
    registry = ToolRegistry(load_builtin_manifests())
    registry.register_handler("system.list_enabled_tools", _list_enabled_tools)
    registry.register_handler("memory.list", _memory_list)
    registry.register_handler("memory.read", _memory_read)
    registry.register_handler("memory.write", _memory_write)
    registry.register_handler("terminal.run", _terminal_run)
    registry.register_handler("browser.open", _browser_not_implemented)
    registry.register_handler("browser.extract_text", _browser_not_implemented)
    registry.register_handler("browser.click", _browser_not_implemented)
    registry.register_handler("browser.fill_form", _browser_not_implemented)
    registry.register_handler("browser.submit", _browser_not_implemented)
    registry.register_handler("calendar.create_event", _calendar_create_event)
    return registry


def _list_enabled_tools(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    del args
    tool_overrides = context.config.get("tools", {})
    manifests = load_builtin_manifests()
    tools = []
    for name, manifest in manifests.items():
        override = tool_overrides.get(name, {})
        enabled = bool(override.get("enabled", manifest.default_enabled))
        tools.append(
            {
                "name": name,
                "risk": int(manifest.risk),
                "enabled": enabled,
                "approval_required": manifest.approval_required,
                "permissions": list(manifest.permissions),
            }
        )
    return {"tools": tools}


def _memory_list(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    manager = MemoryManager(context.memory_root)
    manager.bootstrap()
    return {"files": manager.list_files()}


def _memory_read(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path", "profile.md"))
    manager = MemoryManager(context.memory_root)
    return {"path": path, "content": manager.read(path)}


def _memory_write(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "")
    body = str(args.get("body") or "")
    if not path or not body:
        raise ValueError("memory.write requires path and body.")
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided")
    manager = MemoryManager(context.memory_root)
    written = manager.write(path, body, metadata=metadata)
    return {"path": str(written.relative_to(context.memory_root.resolve()))}


def _terminal_run(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    raw_command = args.get("command")
    if not isinstance(raw_command, list):
        raise ValueError("terminal.run requires command as a string array")
    command = [str(part) for part in raw_command]
    cwd = args.get("cwd")
    if cwd is not None and not isinstance(cwd, str):
        raise ValueError("terminal.run cwd must be a string when provided")

    result = run_workspace_command(
        command,
        workspace=context.workspace_root,
        policy=TerminalPolicy.from_config(context.config),
        cwd=cwd,
    )
    return {
        **result,
        "stdout": redact_text(str(result.get("stdout", ""))),
        "stderr": redact_text(str(result.get("stderr", ""))),
    }


def _browser_not_implemented(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    del context
    url = args.get("url")
    if isinstance(url, str) and url:
        parsed = urlparse(url if "://" in url else f"https://{url}")
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("browser tools require a valid http or https URL")
        url = parsed.geturl()
    return {
        "status": "not_implemented",
        "message": (
            "Browser sandbox execution is not implemented yet. "
            "The request passed tool and permission policy, but no browser session was started."
        ),
        "url": url,
    }


def _calendar_create_event(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    context.workspace_root.mkdir(parents=True, exist_ok=True)
    title = (
        args.get("title")
        or args.get("summary")
        or args.get("name")
        or args.get("description")
        or "Untitled event"
    )
    event = {
        "id": datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "title": str(title),
        "start": args.get("start") or args.get("start_time") or args.get("datetime"),
        "end": args.get("end") or args.get("end_time"),
        "reminder": args.get("reminder") or args.get("remind_at") or args.get("reminder_offset"),
        "raw_args": args,
    }
    calendar_file = context.workspace_root / "local_calendar_events.jsonl"
    with calendar_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
    return {
        "status": "saved_local",
        "message": (
            "Saved to the local calendar store. External calendar sync and active "
            "notification delivery are not implemented yet."
        ),
        "event": event,
        "path": str(calendar_file),
    }
