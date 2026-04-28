from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditStore
from dmdagent4all.config import load_config, save_config, write_default_config
from dmdagent4all.memory import MemoryManager
from dmdagent4all.memory.manager import MemoryPathError
from dmdagent4all.model_presets import MODEL_MODES
from dmdagent4all.permissions import ToolRequest
from dmdagent4all.runtime import build_agent_core
from dmdagent4all.tools import build_builtin_registry


class ChatRequest(BaseModel):
    message: str


class MemoryWriteRequest(BaseModel):
    path: str
    body: str
    metadata: dict[str, Any] | None = None


class ModelModeRequest(BaseModel):
    mode: str


class ModelRequest(BaseModel):
    model: str


def create_app() -> FastAPI:
    paths = AppPaths.default()
    paths.ensure()
    write_default_config(paths.config)
    registry = build_builtin_registry()

    app = FastAPI(title="DMD Agent 4 All", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/tools")
    def tools() -> list[dict[str, Any]]:
        config = load_config(paths.config)
        overrides = config.get("tools", {})
        return [
            {
                "name": manifest.name,
                "description": manifest.description,
                "risk": int(manifest.risk),
                "permissions": list(manifest.permissions),
                "approval_required": manifest.approval_required,
                "cloud_allowed": manifest.cloud_allowed,
                "default_enabled": manifest.default_enabled,
                "enabled": bool(
                    overrides.get(manifest.name, {}).get(
                        "enabled",
                        manifest.default_enabled,
                    )
                ),
            }
            for manifest in registry.manifests.values()
        ]

    @app.post("/v1/tools/{tool_name:path}/enable")
    def enable_tool(tool_name: str) -> dict[str, Any]:
        return _set_tool_enabled(paths, tool_name, True)

    @app.post("/v1/tools/{tool_name:path}/disable")
    def disable_tool(tool_name: str) -> dict[str, Any]:
        return _set_tool_enabled(paths, tool_name, False)

    @app.post("/v1/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        return asdict(build_agent_core().handle_text(request.message))

    @app.get("/v1/audit")
    def audit(limit: int = 20) -> list[dict[str, Any]]:
        return AuditStore(paths.audit_db).list_recent_events(limit=limit)

    @app.get("/v1/approvals")
    def approvals(status: str | None = "pending", limit: int = 50) -> list[dict[str, Any]]:
        return AuditStore(paths.audit_db).list_approvals(status=status, limit=limit)

    @app.post("/v1/approvals/{approval_id}/approve")
    def approve(approval_id: int) -> dict[str, Any]:
        return asdict(build_agent_core().approve_and_execute(approval_id))

    @app.post("/v1/approvals/{approval_id}/deny")
    def deny(approval_id: int) -> dict[str, Any]:
        changed = AuditStore(paths.audit_db).set_approval_status(approval_id, "denied")
        if not changed:
            return {
                "status": "not_found",
                "message": "No pending approval found.",
                "data": {"approval_id": approval_id},
            }
        return {
            "status": "ok",
            "message": "Approval denied.",
            "data": {"approval_id": approval_id},
        }

    @app.get("/v1/memory")
    def memory_list() -> dict[str, Any]:
        manager = MemoryManager(paths.memory)
        manager.bootstrap()
        return {"files": manager.list_files()}

    @app.get("/v1/memory/file")
    def memory_file(path: str) -> dict[str, Any]:
        manager = MemoryManager(paths.memory)
        try:
            return {"path": path, "content": manager.read(path)}
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail="Memory file not found.") from exc
        except MemoryPathError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/v1/memory/file")
    def memory_write(request: MemoryWriteRequest) -> dict[str, Any]:
        response = build_agent_core().handle_tool_request(
            ToolRequest(
                tool="memory.write",
                args={
                    "path": request.path,
                    "body": request.body,
                    "metadata": request.metadata or {},
                },
                reason="User requested memory update from the web dashboard.",
            )
        )
        return asdict(response)

    @app.get("/v1/status")
    def status() -> dict[str, Any]:
        config = load_config(paths.config)
        return {
            "data_dir": str(paths.root),
            "config": str(paths.config),
            "memory": str(paths.memory),
            "workspace": str(paths.workspace),
            "audit_db": str(paths.audit_db),
            "llm": config.get("llm", {}),
            "privacy": config.get("privacy", {}),
            "terminal_enabled": bool(config.get("terminal", {}).get("enabled", False)),
            "browser_enabled": bool(config.get("browser", {}).get("enabled", False)),
        }

    @app.get("/v1/models")
    def models() -> dict[str, Any]:
        config = load_config(paths.config)
        return {
            "current": config.get("llm", {}),
            "modes": [
                {
                    "key": mode.key,
                    "label": mode.label,
                    "default_model": mode.default_model,
                    "alternatives": list(mode.alternatives),
                    "description": mode.description,
                }
                for mode in MODEL_MODES
            ],
        }

    @app.post("/v1/models/mode")
    def set_model_mode(request: ModelModeRequest) -> dict[str, Any]:
        selected = next((mode for mode in MODEL_MODES if mode.key == request.mode), None)
        if selected is None:
            return {
                "status": "error",
                "message": f"Unknown model mode: {request.mode}",
                "data": None,
            }
        config = load_config(paths.config)
        config.setdefault("llm", {})["mode"] = selected.key
        config["llm"]["model"] = selected.default_model
        config["llm"]["planner_model"] = selected.default_model
        save_config(config, paths.config)
        return {
            "status": "ok",
            "message": f"Model mode set to {selected.label}.",
            "data": config["llm"],
        }

    @app.post("/v1/models/model")
    def set_model(request: ModelRequest) -> dict[str, Any]:
        config = load_config(paths.config)
        config.setdefault("llm", {})["mode"] = "custom"
        config["llm"]["model"] = request.model
        config["llm"]["planner_model"] = request.model
        save_config(config, paths.config)
        return {
            "status": "ok",
            "message": f"Model set to {request.model}.",
            "data": config["llm"],
        }

    return app


def _set_tool_enabled(paths: AppPaths, tool_name: str, enabled: bool) -> dict[str, Any]:
    registry = build_builtin_registry()
    if tool_name not in registry.manifests:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}",
            "data": None,
        }
    config = load_config(paths.config)
    config.setdefault("tools", {}).setdefault(tool_name, {})["enabled"] = enabled
    save_config(config, paths.config)
    return {
        "status": "ok",
        "message": f"Tool {tool_name} {'enabled' if enabled else 'disabled'}.",
        "data": {"tool": tool_name, "enabled": enabled},
    }


app = create_app()
