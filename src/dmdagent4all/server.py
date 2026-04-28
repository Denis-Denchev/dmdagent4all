from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditStore
from dmdagent4all.config import write_default_config
from dmdagent4all.runtime import build_agent_core
from dmdagent4all.tools import build_builtin_registry


class ChatRequest(BaseModel):
    message: str


def create_app() -> FastAPI:
    paths = AppPaths.default()
    paths.ensure()
    write_default_config(paths.config)
    registry = build_builtin_registry()
    core = build_agent_core()

    app = FastAPI(title="DMD Agent 4 All", version="0.1.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/tools")
    def tools() -> list[dict[str, Any]]:
        return [
            {
                "name": manifest.name,
                "description": manifest.description,
                "risk": int(manifest.risk),
                "permissions": list(manifest.permissions),
                "approval_required": manifest.approval_required,
                "cloud_allowed": manifest.cloud_allowed,
                "default_enabled": manifest.default_enabled,
            }
            for manifest in registry.manifests.values()
        ]

    @app.post("/v1/chat")
    def chat(request: ChatRequest) -> dict[str, Any]:
        return asdict(core.handle_text(request.message))

    @app.get("/v1/audit")
    def audit(limit: int = 20) -> list[dict[str, Any]]:
        return AuditStore(paths.audit_db).list_recent_events(limit=limit)

    @app.get("/v1/approvals")
    def approvals(status: str | None = "pending", limit: int = 50) -> list[dict[str, Any]]:
        return AuditStore(paths.audit_db).list_approvals(status=status, limit=limit)

    return app


app = create_app()
