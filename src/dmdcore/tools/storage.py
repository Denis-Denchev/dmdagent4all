from __future__ import annotations

from pathlib import Path
from typing import Any


def downloads_root_from_config(config: dict[str, Any], *, default_root: Path) -> Path:
    raw_root = config.get("storage", {}).get("downloads_root")
    if isinstance(raw_root, str) and raw_root.strip():
        return Path(raw_root).expanduser().resolve()
    return default_root.resolve()


def internet_files_dir(config: dict[str, Any], *, default_root: Path, folder: str) -> Path:
    return downloads_root_from_config(config, default_root=default_root) / folder
