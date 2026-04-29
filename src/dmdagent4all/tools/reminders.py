from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dmdagent4all.tools.base import ToolRuntimeContext


def create_reminder(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    title = str(args.get("title") or "").strip()
    due_at = str(args.get("due_at") or "").strip()
    notes = str(args.get("notes") or "").strip()
    if not title:
        raise ValueError("reminders.create requires title.")
    if not due_at:
        raise ValueError("reminders.create requires due_at.")
    normalized_due_at = _normalize_datetime(due_at)
    reminders = _load_reminders(_store_path(context))
    reminder = {
        "id": _next_id(reminders),
        "title": title[:300],
        "due_at": normalized_due_at,
        "status": "pending",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "notes": notes[:1000],
    }
    reminders.append(reminder)
    _save_reminders(_store_path(context), reminders)
    return {"reminder": reminder, "path": str(_store_path(context))}


def list_reminders(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    status = str(args.get("status") or "pending").strip().lower()
    if status not in {"pending", "completed", "all"}:
        raise ValueError("reminders.list status must be pending, completed, or all.")
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


def _next_id(reminders: list[dict[str, Any]]) -> int:
    ids = [int(item.get("id") or 0) for item in reminders]
    return max(ids, default=0) + 1


def _normalize_datetime(value: str) -> str:
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError("due_at must be an ISO datetime.") from exc
    if parsed.tzinfo is None:
        parsed = parsed.astimezone()
    return parsed.isoformat()
