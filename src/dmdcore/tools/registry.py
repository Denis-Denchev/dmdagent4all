from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from dmdcore.permissions.models import ToolManifest
from dmdcore.tools.base import ToolHandler, ToolRuntimeContext


class ToolExecutionError(RuntimeError):
    pass


class ToolRegistry:
    def __init__(self, manifests: dict[str, ToolManifest]) -> None:
        self._manifests = manifests
        self._handlers: dict[str, ToolHandler] = {}

    @property
    def manifests(self) -> dict[str, ToolManifest]:
        return dict(self._manifests)

    def register_handler(self, name: str, handler: ToolHandler) -> None:
        if name not in self._manifests:
            raise KeyError(f"Cannot register handler for unknown tool: {name}")
        self._handlers[name] = handler

    def execute(
        self,
        name: str,
        args: dict[str, Any],
        context: ToolRuntimeContext,
    ) -> dict[str, Any]:
        handler = self._handlers.get(name)
        if handler is None:
            raise ToolExecutionError(f"No handler is implemented for tool: {name}")
        return handler(args, context)


def load_builtin_manifests() -> dict[str, ToolManifest]:
    manifest_root = files("dmdcore").joinpath("manifests/tools")
    manifests: dict[str, ToolManifest] = {}
    for manifest_file in manifest_root.iterdir():
        if not manifest_file.name.endswith(".json"):
            continue
        data = json.loads(manifest_file.read_text(encoding="utf-8"))
        manifest = ToolManifest.from_dict(data)
        manifests[manifest.name] = manifest
    return dict(sorted(manifests.items()))
