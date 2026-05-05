from __future__ import annotations

import json
import re
from datetime import datetime, time, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse
from typing import Any

from dmdagent4all.memory import MemoryManager
from dmdagent4all.sandbox import TerminalPolicy, run_workspace_command
from dmdagent4all.security import redact_text
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.browser_automation import browser_click, browser_fill_form, browser_submit
from dmdagent4all.tools.reminders import complete_reminder, create_reminder, list_reminders
from dmdagent4all.tools.registry import ToolRegistry, load_builtin_manifests
from dmdagent4all.tools.web import browser_extract_text, browser_open


def build_builtin_registry() -> ToolRegistry:
    registry = ToolRegistry(load_builtin_manifests())
    registry.register_handler("system.list_enabled_tools", _list_enabled_tools)
    registry.register_handler("memory.list", _memory_list)
    registry.register_handler("memory.read", _memory_read)
    registry.register_handler("memory.write", _memory_write)
    registry.register_handler("reminders.create", create_reminder)
    registry.register_handler("reminders.list", list_reminders)
    registry.register_handler("reminders.complete", complete_reminder)
    registry.register_handler("terminal.run", _terminal_run)
    registry.register_handler("browser.open", browser_open)
    registry.register_handler("browser.extract_text", browser_extract_text)
    registry.register_handler("browser.click", browser_click)
    registry.register_handler("browser.fill_form", browser_fill_form)
    registry.register_handler("browser.submit", browser_submit)
    registry.register_handler("calendar.create_event", _calendar_create_event)
    registry.register_handler("calendar.delete_event", _calendar_delete_event)
    registry.register_handler("calendar.find_free_slots", _calendar_find_free_slots)
    registry.register_handler("calendar.today", _calendar_today)
    registry.register_handler("calendar.update_event", _calendar_update_event)
    registry.register_handler("calendar.week", _calendar_week)
    for name in {
        "gmail.archive",
        "gmail.create_draft",
        "gmail.label",
        "gmail.read_thread",
        "gmail.search",
        "gmail.send_draft",
        "gmail.summarize_inbox",
    }:
        registry.register_handler(name, _gmail_not_configured)
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
    path = str(args.get("path", "long-term/profile.md"))
    manager = MemoryManager(context.memory_root)
    return {"path": path, "content": manager.read(path)}


def _memory_write(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args.get("path") or "").strip()
    body = str(args.get("body") or "")
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided")
    if path.lower() == "auto":
        path = ""
    if not path or not body:
        title = str(args.get("title") or "").strip()
        if not body:
            raise ValueError("memory.write requires body.")
        path = _memory_auto_path(title=title, body=body, args=args)
    metadata = dict(metadata or {})
    scope = _memory_scope(args, metadata)
    metadata["memory_scope"] = scope
    if "ttl_hours" in args and "ttl_hours" not in metadata:
        metadata["ttl_hours"] = args["ttl_hours"]
    if scope == "short-term" and not path.startswith("short-term/"):
        path = f"short-term/{path}"
    manager = MemoryManager(context.memory_root)
    written = manager.write(path, body, metadata=metadata)
    return {"path": str(written.relative_to(context.memory_root.resolve()))}


def _memory_scope(args: dict[str, Any], metadata: dict[str, Any]) -> str:
    raw_scope = (
        args.get("memory_scope")
        or args.get("scope")
        or metadata.get("memory_scope")
        or metadata.get("scope")
        or "long-term"
    )
    scope = str(raw_scope).strip().lower().replace("_", "-")
    return "short-term" if scope in {"short", "short-term", "temporary", "temp"} else "long-term"


def _memory_auto_path(*, title: str, body: str, args: dict[str, Any]) -> str:
    scope = _memory_scope(args, dict(args.get("metadata") or {}))
    raw_title = title or _first_meaningful_line(body) or "memory"
    slug = _slugify(raw_title) or "memory"
    prefix = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    folder = "short-term" if scope == "short-term" else "long-term/notes"
    return f"{folder}/{prefix}-{slug}.md"


def _first_meaningful_line(body: str) -> str:
    for line in body.splitlines():
        stripped = line.strip().strip("# ").strip()
        if stripped:
            return stripped
    return ""


def _slugify(value: str) -> str:
    slug = re.sub(r"[^\w]+", "-", value.lower(), flags=re.UNICODE).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:80].strip("-")


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
        workspace=_terminal_workspace_root(context),
        policy=TerminalPolicy.from_config(context.config),
        cwd=cwd,
    )
    return {
        **result,
        "stdout": redact_text(str(result.get("stdout", ""))),
        "stderr": redact_text(str(result.get("stderr", ""))),
    }


def _terminal_workspace_root(context: ToolRuntimeContext) -> Path:
    raw_root = context.config.get("terminal", {}).get("workspace_root")
    if isinstance(raw_root, str) and raw_root.strip():
        return Path(raw_root).expanduser()
    return context.workspace_root


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


def _gmail_not_configured(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    del args, context
    return {
        "status": "connector_not_configured",
        "connector": "gmail",
        "message": (
            "Gmail OAuth is not configured in this build. Enablement and permissions "
            "are enforced, but real Gmail API access still requires connector setup."
        ),
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


def _calendar_update_event(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    event_id = str(args.get("id") or args.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("calendar.update_event requires id.")
    events = _read_local_calendar_events(context)
    updated_event: dict[str, Any] | None = None
    for event in events:
        if str(event.get("id") or "") != event_id:
            continue
        for source_key, target_key in {
            "title": "title",
            "summary": "title",
            "start": "start",
            "start_time": "start",
            "end": "end",
            "end_time": "end",
            "reminder": "reminder",
            "remind_at": "reminder",
        }.items():
            if source_key in args:
                event[target_key] = args[source_key]
        event["updated_at"] = datetime.now(timezone.utc).isoformat()
        event["raw_update_args"] = args
        updated_event = event
        break
    if updated_event is None:
        raise ValueError(f"Calendar event not found: {event_id}")
    _write_local_calendar_events(context, events)
    return {
        "status": "updated_local",
        "message": "Updated the local calendar store. External calendar sync is not implemented yet.",
        "event": updated_event,
        "path": str(_calendar_store_path(context)),
    }


def _calendar_delete_event(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    event_id = str(args.get("id") or args.get("event_id") or "").strip()
    if not event_id:
        raise ValueError("calendar.delete_event requires id.")
    events = _read_local_calendar_events(context)
    remaining = [event for event in events if str(event.get("id") or "") != event_id]
    if len(remaining) == len(events):
        raise ValueError(f"Calendar event not found: {event_id}")
    _write_local_calendar_events(context, remaining)
    return {
        "status": "deleted_local",
        "message": "Deleted the event from the local calendar store. External calendar sync is not implemented yet.",
        "event_id": event_id,
        "path": str(_calendar_store_path(context)),
    }


def _calendar_find_free_slots(
    args: dict[str, Any],
    context: ToolRuntimeContext,
) -> dict[str, Any]:
    start = _parse_datetime_arg(args.get("start")) or datetime.now().astimezone()
    end = _parse_datetime_arg(args.get("end")) or (start + timedelta(days=7))
    if end <= start:
        raise ValueError("calendar.find_free_slots end must be after start.")
    duration_minutes = _bounded_int(args.get("duration_minutes"), default=30, minimum=5, maximum=480)
    limit = _bounded_int(args.get("limit"), default=20, minimum=1, maximum=100)
    workday_start = _parse_time_arg(args.get("workday_start"), default=time(9, 0))
    workday_end = _parse_time_arg(args.get("workday_end"), default=time(17, 0))
    if workday_end <= workday_start:
        raise ValueError("workday_end must be after workday_start.")

    busy_ranges = sorted(
        range_
        for range_ in (_event_datetime_range(event) for event in _read_local_calendar_events(context))
        if range_ is not None and range_[1] > start and range_[0] < end
    )
    slots: list[dict[str, str]] = []
    current_day = start.date()
    while len(slots) < limit and current_day <= end.date():
        window_start = datetime.combine(current_day, workday_start, tzinfo=start.tzinfo)
        window_end = datetime.combine(current_day, workday_end, tzinfo=start.tzinfo)
        cursor = max(window_start, start)
        day_end = min(window_end, end)
        if cursor < day_end:
            for busy_start, busy_end in busy_ranges:
                if busy_end <= cursor or busy_start >= day_end:
                    continue
                if _minutes_between(cursor, busy_start) >= duration_minutes:
                    slots.append({"start": cursor.isoformat(), "end": busy_start.isoformat()})
                    if len(slots) >= limit:
                        break
                cursor = max(cursor, busy_end)
            if len(slots) >= limit:
                break
            if _minutes_between(cursor, day_end) >= duration_minutes:
                slots.append({"start": cursor.isoformat(), "end": day_end.isoformat()})
        current_day += timedelta(days=1)

    return {
        "slots": slots[:limit],
        "source": "local_calendar_store",
        "duration_minutes": duration_minutes,
    }


def _calendar_today(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    now = datetime.now().astimezone()
    events = [
        event
        for event in _read_local_calendar_events(context)
        if _event_start_date(event) == now.date()
    ]
    return {"events": events, "source": "local_calendar_store"}


def _calendar_week(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    del args
    now = datetime.now().astimezone()
    today = now.date()
    events = []
    for event in _read_local_calendar_events(context):
        start_date = _event_start_date(event)
        if start_date is not None and 0 <= (start_date - today).days <= 6:
            events.append(event)
    return {"events": events, "source": "local_calendar_store"}


def _read_local_calendar_events(context: ToolRuntimeContext) -> list[dict[str, Any]]:
    calendar_file = _calendar_store_path(context)
    if not calendar_file.exists():
        return []
    events: list[dict[str, Any]] = []
    for line in calendar_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            loaded = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(loaded, dict):
            events.append(loaded)
    return events


def _write_local_calendar_events(context: ToolRuntimeContext, events: list[dict[str, Any]]) -> None:
    calendar_file = _calendar_store_path(context)
    calendar_file.parent.mkdir(parents=True, exist_ok=True)
    with calendar_file.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")


def _calendar_store_path(context: ToolRuntimeContext):
    context.workspace_root.mkdir(parents=True, exist_ok=True)
    return context.workspace_root / "local_calendar_events.jsonl"


def _event_start_date(event: dict[str, Any]):
    raw = event.get("start")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00")).astimezone().date()
    except ValueError:
        return None


def _event_datetime_range(event: dict[str, Any]):
    start = _parse_datetime_arg(event.get("start"))
    if start is None:
        return None
    end = _parse_datetime_arg(event.get("end")) or (start + timedelta(hours=1))
    if end <= start:
        end = start + timedelta(hours=1)
    return start, end


def _parse_datetime_arg(value: Any) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone()


def _parse_time_arg(value: Any, *, default: time) -> time:
    if value is None or str(value).strip() == "":
        return default
    raw = str(value).strip()
    try:
        parsed = time.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"Invalid time value: {raw}") from exc
    return parsed.replace(tzinfo=None)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    if not minimum <= parsed <= maximum:
        raise ValueError(f"Value must be between {minimum} and {maximum}.")
    return parsed


def _minutes_between(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() / 60
