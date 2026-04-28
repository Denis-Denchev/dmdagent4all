from __future__ import annotations

from typing import Any

from dmdagent4all.memory import MemoryManager
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.registry import ToolRegistry, load_builtin_manifests


def build_builtin_registry() -> ToolRegistry:
    registry = ToolRegistry(load_builtin_manifests())
    registry.register_handler("system.list_enabled_tools", _list_enabled_tools)
    registry.register_handler("memory.list", _memory_list)
    registry.register_handler("memory.read", _memory_read)
    registry.register_handler("memory.write", _memory_write)
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
    path = str(args.get("path", "profile.md"))
    manager = MemoryManager(context.memory_root)
    return {"path": path, "content": manager.read(path)}


def _memory_write(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
    path = str(args["path"])
    body = str(args["body"])
    metadata = args.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        raise ValueError("metadata must be an object when provided")
    manager = MemoryManager(context.memory_root)
    written = manager.write(path, body, metadata=metadata)
    return {"path": str(written.relative_to(context.memory_root))}
