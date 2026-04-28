from __future__ import annotations

from datetime import datetime, timezone
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
            "profile.md": "User profile notes live here.",
            "preferences.md": "User preferences live here.",
            "facts/technical.md": "Technical facts approved by the user live here.",
            "facts/business.md": "Business facts approved by the user live here.",
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
        return sorted(
            str(path.relative_to(self.root))
            for path in self.root.rglob("*.md")
            if path.is_file()
        )

    def read(self, relative_path: str) -> str:
        target = self._safe_path(relative_path)
        if not target.exists():
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
        target.write_text(_render_markdown(metadata, body), encoding="utf-8")
        return target

    def _safe_path(self, relative_path: str) -> Path:
        if relative_path.startswith("/") or relative_path.startswith("~"):
            raise MemoryPathError("Memory paths must be relative.")
        target = (self.root / relative_path).resolve()
        if target != self.root and self.root not in target.parents:
            raise MemoryPathError("Memory path escapes the memory directory.")
        return target


def _render_markdown(metadata: dict[str, Any], body: str) -> str:
    frontmatter = "\n".join(f"{key}: {value}" for key, value in metadata.items())
    return f"---\n{frontmatter}\n---\n\n{body.rstrip()}\n"
