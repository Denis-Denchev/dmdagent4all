from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any


class MemoryPathError(ValueError):
    pass


class MemoryManager:
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def bootstrap(self) -> None:
        defaults = {
            "long-term/profile.md": "User profile notes live here.",
            "long-term/preferences.md": "User preferences live here.",
            "long-term/facts/personal.md": "Personal facts approved by the user live here.",
            "long-term/facts/technical.md": "Technical facts approved by the user live here.",
            "long-term/facts/business.md": "Business facts approved by the user live here.",
        }
        for relative_path, body in defaults.items():
            target = self._safe_path(relative_path)
            if not target.exists():
                self.write(
                    relative_path,
                    body,
                    metadata={
                        "type": "seed",
                        "source": "system",
                        "confidence": "medium",
                    },
                )

    def list_files(self) -> list[str]:
        self.purge_expired()
        return sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*.md")
            if path.is_file()
        )

    def read(self, relative_path: str) -> str:
        self.purge_expired()
        target = self._safe_path(relative_path)
        if not target.exists():
            raise FileNotFoundError(relative_path)
        if self._is_expired(target):
            target.unlink(missing_ok=True)
            raise FileNotFoundError(relative_path)
        return target.read_text(encoding="utf-8")

    def write(
        self,
        relative_path: str,
        body: str,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        target = self._safe_path(relative_path)
        if target.suffix != ".md":
            raise MemoryPathError("Memory files must be Markdown files.")
        target.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        metadata = {
            "type": "note",
            "created_at": now,
            "updated_at": now,
            "source": "user",
            "confidence": "medium",
            **(metadata or {}),
        }
        memory_scope = str(metadata.get("memory_scope") or metadata.get("scope") or "").strip()
        if not memory_scope:
            memory_scope = "short-term" if relative_path.startswith("short-term/") else "long-term"
        metadata["memory_scope"] = memory_scope
        if memory_scope == "short-term":
            ttl_hours = _bounded_ttl_hours(metadata.get("ttl_hours"))
            metadata["ttl_hours"] = ttl_hours
            metadata["expires_at"] = (datetime.now(timezone.utc) + timedelta(hours=ttl_hours)).isoformat()
        target.write_text(_render_markdown(metadata, body), encoding="utf-8")
        return target

    def purge_expired(self, *, now: datetime | None = None) -> list[str]:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        removed: list[str] = []
        for path in self.root.rglob("*.md"):
            if not path.is_file():
                continue
            if self._is_expired(path, now=current):
                removed.append(str(path.relative_to(self.root)))
                path.unlink(missing_ok=True)
        return sorted(removed)

    def _safe_path(self, relative_path: str) -> Path:
        if relative_path.startswith("/") or relative_path.startswith("~"):
            raise MemoryPathError("Memory paths must be relative.")
        target = (self.root / relative_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise MemoryPathError("Memory path escapes the memory directory.")
        return target

    def _is_expired(self, path: Path, *, now: datetime | None = None) -> bool:
        metadata = _parse_frontmatter(path.read_text(encoding="utf-8"))
        expires_at = str(metadata.get("expires_at") or "").strip()
        if not expires_at:
            return False
        try:
            expires = datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
        except ValueError:
            return False
        return expires.astimezone(timezone.utc) <= (now or datetime.now(timezone.utc)).astimezone(timezone.utc)


def _render_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    return f"---\n{frontmatter}\n---\n\n{body.rstrip()}\n"


def _parse_frontmatter(markdown: str) -> dict[str, str]:
    if not markdown.startswith("---\n"):
        return {}
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return {}
    metadata: dict[str, str] = {}
    for line in markdown[4:end].splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _bounded_ttl_hours(value: Any) -> int:
    try:
        ttl_hours = int(value)
    except (TypeError, ValueError):
        ttl_hours = 24
    return min(48, max(1, ttl_hours))
