from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ChatTurn:
    role: str
    content: str
    created_at: str


class ChatHistory:
    def __init__(self, path: Path, *, max_turns: int = 40, max_chars_per_turn: int = 2000) -> None:
        self.path = path.expanduser().resolve()
        self.max_turns = max(4, int(max_turns))
        self.max_chars_per_turn = max(200, int(max_chars_per_turn))
        self._lock = threading.Lock()

    def recent(self, session_id: str = "default", *, limit: int = 12) -> list[ChatTurn]:
        with self._lock:
            data = self._load()
        raw_turns = data.get(_safe_session_id(session_id), [])
        if not isinstance(raw_turns, list):
            return []
        turns: list[ChatTurn] = []
        for item in raw_turns[-max(1, int(limit)) :]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip()
            content = str(item.get("content") or "").strip()
            created_at = str(item.get("created_at") or "").strip()
            if role in {"user", "assistant"} and content:
                turns.append(ChatTurn(role=role, content=content, created_at=created_at))
        return turns

    def append(self, session_id: str, role: str, content: str) -> None:
        if role not in {"user", "assistant"}:
            raise ValueError("Chat history role must be user or assistant.")
        cleaned = " ".join(str(content or "").split()).strip()
        if not cleaned:
            return
        if len(cleaned) > self.max_chars_per_turn:
            cleaned = cleaned[: self.max_chars_per_turn].rstrip() + "..."
        turn = ChatTurn(
            role=role,
            content=cleaned,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        safe_session = _safe_session_id(session_id)
        with self._lock:
            data = self._load()
            raw_turns = data.get(safe_session)
            turns = raw_turns if isinstance(raw_turns, list) else []
            turns.append(asdict(turn))
            data[safe_session] = turns[-self.max_turns :]
            self._save(data)

    def format_recent(self, session_id: str = "default", *, limit: int = 12, max_chars: int = 6000) -> str:
        lines: list[str] = []
        for turn in self.recent(session_id, limit=limit):
            prefix = "User" if turn.role == "user" else "Assistant"
            lines.append(f"{prefix}: {turn.content}")
        text = "\n".join(lines).strip()
        if len(text) > max_chars:
            return text[-max_chars:].lstrip()
        return text

    def _load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {}
        try:
            loaded = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}
        return loaded if isinstance(loaded, dict) else {}

    def _save(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)


def _safe_session_id(session_id: str) -> str:
    cleaned = "".join(char for char in str(session_id or "default") if char.isalnum() or char in {"-", "_"})
    return cleaned[:80] or "default"
