from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class ToolRuntimeContext:
    memory_root: Path
    workspace_root: Path
    config: dict[str, Any]
    config_path: Path | None = None


ToolHandler = Callable[[dict[str, Any], ToolRuntimeContext], dict[str, Any]]
