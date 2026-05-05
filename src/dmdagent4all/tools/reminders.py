from __future__ import annotations

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

from dmdagent4all.tools.base import ToolRuntimeContext

_REMINDER_CHANGE_CONDITION = threading.Condition()
_REMINDER_CHANGE_VERSION = 0


def create_reminder(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    title = str(args.get("title") or "").strip()
    due_at = str(args.get("due_at") or "").strip()
    event_at = str(args.get("event_at") or "").strip()
    remind_before = str(args.get("remind_before") or "").strip()
    location = str(args.get("location") or "").strip()
    action_url = str(args.get("action_url") or "").strip()
    notes = str(args.get("notes") or "").strip()
    if not title:
        raise ValueError("reminders.create requires title.")
    if not due_at:
        raise ValueError("reminders.create requires due_at.")
    normalized_due_at = _normalize_datetime(due_at)
    normalized_event_at = _normalize_datetime(event_at) if event_at else ""
    reminders = _load_reminders(_store_path(context))
    reminder = {
        "id": _next_id(reminders),
        "title": title[:300],
        "due_at": normalized_due_at,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes[:1000],
    }
    if normalized_event_at:
        reminder["event_at"] = normalized_event_at
    if remind_before:
        reminder["remind_before"] = remind_before[:100]
    if location:
        reminder["location"] = location[:300]
    if location and not action_url:
        action_url = _maps_search_url(location)
    if action_url:
        reminder["action_url"] = action_url[:1000]
    reminders.append(reminder)
    _save_reminders(_store_path(context), reminders)
    return {"reminder": reminder, "path": str(_store_path(context))}


def list_reminders(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    status = str(args.get("status") or "pending").strip().lower()
    if status not in {"pending", "notified", "completed", "canceled", "all"}:
        raise ValueError("reminders.list status must be pending, notified, completed, canceled, or all.")
    reminders = _load_reminders(_store_path(context))
    if status != "all":
        reminders = [item for item in reminders if item.get("status") == status]
    return {
        "reminders": sorted(
            reminders,
            key=lambda item: (str(item.get("due_at") or ""), int(item.get("id") or 0)),
        )
    }


def complete_reminder(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    try:
        reminder_id = int(args.get("id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("reminders.complete requires numeric id.") from exc

    reminders = _load_reminders(_store_path(context))
    for reminder in reminders:
        if int(reminder.get("id") or 0) != reminder_id:
            continue
        reminder["status"] = "completed"
        reminder["completed_at"] = datetime.now(timezone.utc).isoformat()
        _save_reminders(_store_path(context), reminders)
        return {"reminder": reminder, "path": str(_store_path(context))}
    raise ValueError(f"Reminder not found: {reminder_id}")


def cancel_reminder(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    reminder_id = _require_reminder_id(args)
    reminders = _load_reminders(_store_path(context))
    for reminder in reminders:
        if int(reminder.get("id") or 0) != reminder_id:
            continue
        reminder["status"] = "canceled"
        reminder["canceled_at"] = datetime.now(timezone.utc).isoformat()
        _save_reminders(_store_path(context), reminders)
        return {"reminder": reminder, "path": str(_store_path(context))}
    raise ValueError(f"Reminder not found: {reminder_id}")


def snooze_reminder(
    context: ToolRuntimeContext,
    reminder_id: int,
    *,
    seconds: int,
    now: datetime | None = None,
) -> dict[str, Any] | None:
    if not (1 <= int(seconds) <= 60 * 60 * 24 * 365):
        raise ValueError("Snooze seconds must be between 1 second and 365 days.")
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    reminders = _load_reminders(_store_path(context))
    for reminder in reminders:
        if int(reminder.get("id") or 0) != int(reminder_id):
            continue
        reminder["status"] = "pending"
        reminder["due_at"] = (current + timedelta(seconds=int(seconds))).isoformat()
        reminder["snoozed_at"] = current.isoformat()
        reminder["snooze_seconds"] = int(seconds)
        reminder.pop("completed_at", None)
        reminder.pop("canceled_at", None)
        _save_reminders(_store_path(context), reminders)
        return reminder
    return None


def due_reminders(context: ToolRuntimeContext, *, now: datetime | None = None) -> list[dict[str, Any]]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    due: list[dict[str, Any]] = []
    for reminder in _load_reminders(_store_path(context)):
        if reminder.get("status") != "pending":
            continue
        due_at = _parse_datetime(str(reminder.get("due_at") or ""))
        if due_at is not None and due_at <= current:
            due.append(reminder)
    return sorted(due, key=lambda item: (str(item.get("due_at") or ""), int(item.get("id") or 0)))


def next_pending_reminder_due_at(context: ToolRuntimeContext) -> datetime | None:
    next_due: datetime | None = None
    for reminder in _load_reminders(_store_path(context)):
        if reminder.get("status") != "pending":
            continue
        due_at = _parse_datetime(str(reminder.get("due_at") or ""))
        if due_at is None:
            continue
        if next_due is None or due_at < next_due:
            next_due = due_at
    return next_due


def mark_reminder_notified(
    context: ToolRuntimeContext,
    reminder_id: int,
    *,
    channel: str,
) -> dict[str, Any] | None:
    reminders = _load_reminders(_store_path(context))
    for reminder in reminders:
        if int(reminder.get("id") or 0) != int(reminder_id):
            continue
        reminder["status"] = "notified"
        reminder["notified_at"] = datetime.now(timezone.utc).isoformat()
        reminder["notification_channel"] = channel
        _save_reminders(_store_path(context), reminders)
        return reminder
    return None


def reminder_change_version() -> int:
    with _REMINDER_CHANGE_CONDITION:
        return _REMINDER_CHANGE_VERSION


def wait_for_reminder_change_since(version: int, timeout_seconds: float) -> int:
    timeout_seconds = max(0.0, timeout_seconds)
    with _REMINDER_CHANGE_CONDITION:
        if _REMINDER_CHANGE_VERSION == version:
            _REMINDER_CHANGE_CONDITION.wait(timeout=timeout_seconds)
        return _REMINDER_CHANGE_VERSION


def wake_reminder_waiters() -> None:
    with _REMINDER_CHANGE_CONDITION:
        _REMINDER_CHANGE_CONDITION.notify_all()


def _store_path(context: ToolRuntimeContext) -> Path:
    context.workspace_root.mkdir(parents=True, exist_ok=True)
    return context.workspace_root / "local_reminders.json"


def _load_reminders(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, list):
        raise ValueError("Reminder store is corrupted.")
    return [item for item in loaded if isinstance(item, dict)]


def _save_reminders(path: Path, reminders: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(reminders, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _notify_reminder_store_changed()


def _notify_reminder_store_changed() -> None:
    global _REMINDER_CHANGE_VERSION
    with _REMINDER_CHANGE_CONDITION:
        _REMINDER_CHANGE_VERSION += 1
        _REMINDER_CHANGE_CONDITION.notify_all()


def _next_id(reminders: list[dict[str, Any]]) -> int:
    ids = [int(item.get("id") or 0) for item in reminders]
    return max(ids, default=0) + 1


def _require_reminder_id(args: dict[str, Any]) -> int:
    try:
        return int(args.get("id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Reminder id must be numeric.") from exc


def _maps_search_url(location: str) -> str:
    return f"https://www.google.com/maps/search/?api=1&query={quote_plus(location)}"


def _normalize_datetime(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    parsed = _parse_datetime(normalized)
    if parsed is None:
        raise ValueError("due_at must be an ISO datetime.")
    return parsed.isoformat()


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.astimezone(timezone.utc)
