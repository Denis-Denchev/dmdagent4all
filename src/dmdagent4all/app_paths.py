from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


APP_NAME = "dmdagent4all"


def data_home() -> Path:
    configured = os.environ.get("XDG_DATA_HOME")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share"


@dataclass(frozen=True)
class AppPaths:
    root: Path
    config: Path
    memory: Path
    workspace: Path
    logs: Path
    audit_db: Path
    vector_index: Path
    tokens_marker: Path

    @classmethod
    def default(cls) -> "AppPaths":
        root = data_home() / APP_NAME
        return cls(
            root=root,
            config=root / "config.yaml",
            memory=root / "memory",
            workspace=root / "workspace",
            logs=root / "logs",
            audit_db=root / "audit.db",
            vector_index=root / "vector_index",
            tokens_marker=root / "tokens.enc",
        )

    def ensure(self) -> None:
        for directory in (
            self.root,
            self.memory,
            self.workspace,
            self.logs,
            self.vector_index,
        ):
            directory.mkdir(parents=True, exist_ok=True)
