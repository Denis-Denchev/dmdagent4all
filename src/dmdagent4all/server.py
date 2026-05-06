from __future__ import annotations

import os
import json
import re
import shlex
import threading
import urllib.error
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from dmdagent4all import __version__
from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.config import load_config, update_config, write_default_config
from dmdagent4all.doctor import doctor_summary, run_doctor
from dmdagent4all.interfaces.telegram import (
    DEFAULT_TOKEN_ENV,
    TelegramConfigError,
    TelegramError,
    TelegramInterface,
    reminder_keyboard,
    reminder_notification_text,
    settings_from_config,
    store_telegram_token,
    telegram_token_available,
)
from dmdagent4all.memory import MemoryManager
from dmdagent4all.memory.manager import MemoryPathError
from dmdagent4all.model_presets import MODEL_MODES
from dmdagent4all.llm.openai_usage import (
    DEFAULT_OPENAI_API_KEY_ENV,
    openai_usage_summary,
    reset_openai_usage,
)
from dmdagent4all.permissions import ToolRequest
from dmdagent4all.runtime import build_agent_core
from dmdagent4all.sandbox import TerminalPolicy
from dmdagent4all.tools import build_builtin_registry
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.reminders import (
    due_reminders,
    mark_reminder_notified,
    next_pending_reminder_due_at,
    reminder_change_version,
    wait_for_reminder_change_since,
    wake_reminder_waiters,
)


DEFAULT_DEEPSEEK_API_KEY_ENV = "DMDAGENT_DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODEL_FALLBACKS = ["deepseek-v4-flash", "deepseek-v4-pro"]


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class MemoryWriteRequest(BaseModel):
    path: str
    body: str
    metadata: dict[str, Any] | None = None


class ModelModeRequest(BaseModel):
    mode: str


class ModelRequest(BaseModel):
    model: str


class PermissionRequest(BaseModel):
    permission: str


class TerminalSettingsRequest(BaseModel):
    workspace_only: bool | None = None
    workspace_root: str | None = None
    timeout_seconds: int | None = None
    max_output_chars: int | None = None
    auto_approve_allowlisted: bool | None = None


class TerminalCommandRequest(BaseModel):
    command: str | list[str]
    cwd: str | None = None


class TelegramUserRequest(BaseModel):
    user_id: int


class TelegramTokenEnvRequest(BaseModel):
    bot_token_env: str


class TelegramTokenRequest(BaseModel):
    token: str


class OpenAIKeyRequest(BaseModel):
    api_key: str


class OpenAILimitRequest(BaseModel):
    limit_usd: float | None = None


def create_app() -> FastAPI:
    paths = AppPaths.default()
    paths.ensure()
    write_default_config(paths.config)
    registry = build_builtin_registry()
    telegram_runtime = TelegramRuntime()
    reminder_runtime = ReminderRuntime(paths, telegram_runtime)

    app = FastAPI(title="DMD Agent 4 All", version="1.0.0")

    @app.on_event("startup")
    def start_background_services() -> None:
        if os.environ.get("DMDAGENT_NO_TELEGRAM") != "1":
            telegram_runtime.start(load_config(paths.config))
        reminder_runtime.start()

    @app.on_event("shutdown")
    def stop_background_services() -> None:
        reminder_runtime.stop()
        telegram_runtime.stop()

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
        return asdict(build_agent_core().handle_text(request.message, session_id=request.session_id or "dashboard"))

    @app.get("/v1/permissions")
    def permissions() -> dict[str, Any]:
        config = load_config(paths.config)
        granted = sorted(config.get("permissions", {}).get("granted", []))
        required = sorted(
            {
                permission
                for manifest in registry.manifests.values()
                for permission in manifest.permissions
            }
        )
        return {
            "granted": granted,
            "available": [
                {
                    "name": permission,
                    "granted": permission in granted,
                    "tools": [
                        manifest.name
                        for manifest in registry.manifests.values()
                        if permission in manifest.permissions
                    ],
                }
                for permission in required
            ],
        }

    @app.post("/v1/permissions/grant")
    def grant_permission(request: PermissionRequest) -> dict[str, Any]:
        return _set_permission(paths, request.permission, True)

    @app.post("/v1/permissions/revoke")
    def revoke_permission(request: PermissionRequest) -> dict[str, Any]:
        return _set_permission(paths, request.permission, False)

    @app.get("/v1/audit")
    def audit(limit: int = 20) -> list[dict[str, Any]]:
        return AuditStore(paths.audit_db).list_recent_events(limit=limit)

    @app.get("/v1/doctor")
    def doctor(check_network: bool = False) -> dict[str, Any]:
        config = load_config(paths.config)
        checks = run_doctor(paths=paths, config=config, check_network=check_network)
        return {
            "summary": doctor_summary(checks),
            "checks": [check.to_dict() for check in checks],
        }

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
        telegram_config = config.get("interfaces", {}).get("telegram", {})
        return {
            "version": __version__,
            "data_dir": str(paths.root),
            "config": str(paths.config),
            "memory": str(paths.memory),
            "workspace": str(paths.workspace),
            "audit_db": str(paths.audit_db),
            "llm": config.get("llm", {}),
            "privacy": config.get("privacy", {}),
            "terminal": config.get("terminal", {}),
            "terminal_enabled": bool(config.get("terminal", {}).get("enabled", False)),
            "browser_enabled": bool(config.get("browser", {}).get("enabled", False)),
            "telegram": {
                "enabled": bool(telegram_config.get("enabled", False)),
                "allowed_user_ids": list(telegram_config.get("allowed_user_ids", [])),
                "bot_token_env": telegram_config.get(
                    "bot_token_env",
                    "DMDAGENT_TELEGRAM_BOT_TOKEN",
                ),
            },
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
        config = update_config(
            lambda current: _set_model_config(
                current,
                mode=selected.key,
                model=selected.default_model,
                planner_model=selected.default_model,
            ),
            paths.config,
        )
        return {
            "status": "ok",
            "message": f"Model mode set to {selected.label}.",
            "data": config["llm"],
        }

    @app.post("/v1/models/model")
    def set_model(request: ModelRequest) -> dict[str, Any]:
        config = update_config(
            lambda current: _set_model_config(
                current,
                mode="custom",
                model=request.model,
                planner_model=request.model,
            ),
            paths.config,
        )
        return {
            "status": "ok",
            "message": f"Model set to {request.model}.",
            "data": config["llm"],
        }

    @app.get("/v1/openai")
    def openai_status() -> dict[str, Any]:
        return _openai_status(load_config(paths.config), paths)

    @app.post("/v1/openai/key")
    def openai_key(request: OpenAIKeyRequest) -> dict[str, Any]:
        api_key = request.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="OpenAI API key is required.")
        os.environ[DEFAULT_OPENAI_API_KEY_ENV] = api_key
        config = update_config(
            lambda current: _set_openai_config(
                current,
                model=_default_openai_model(current),
            ),
            paths.config,
        )
        return {
            "status": "ok",
            "message": "OpenAI API key loaded into this API process environment.",
            "data": _openai_status(config, paths),
        }

    @app.post("/v1/openai/model")
    def openai_model(request: ModelRequest) -> dict[str, Any]:
        model = request.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="OpenAI model is required.")
        config = update_config(lambda current: _set_openai_config(current, model=model), paths.config)
        return {
            "status": "ok",
            "message": f"OpenAI model set to {model}.",
            "data": _openai_status(config, paths),
        }

    @app.get("/v1/openai/models")
    def openai_models() -> dict[str, Any]:
        config = load_config(paths.config)
        return {
            "status": "ok",
            "models": _fetch_openai_models(config),
            "data": _openai_status(config, paths),
        }

    @app.post("/v1/openai/limit")
    def openai_limit(request: OpenAILimitRequest) -> dict[str, Any]:
        config = update_config(
            lambda current: _set_openai_limit(current, request.limit_usd),
            paths.config,
        )
        return {
            "status": "ok",
            "message": "OpenAI local project limit updated.",
            "data": _openai_status(config, paths),
        }

    @app.post("/v1/openai/usage/reset")
    def openai_usage_reset() -> dict[str, Any]:
        reset_openai_usage(paths.audit_db)
        return {
            "status": "ok",
            "message": "OpenAI local usage counters reset.",
            "data": _openai_status(load_config(paths.config), paths),
        }

    @app.get("/v1/deepseek")
    def deepseek_status() -> dict[str, Any]:
        return _deepseek_status(load_config(paths.config))

    @app.post("/v1/deepseek/key")
    def deepseek_key(request: OpenAIKeyRequest) -> dict[str, Any]:
        api_key = request.api_key.strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="DeepSeek API key is required.")
        os.environ[DEFAULT_DEEPSEEK_API_KEY_ENV] = api_key
        config = update_config(
            lambda current: _set_deepseek_config(
                current,
                model=_default_deepseek_model(current),
            ),
            paths.config,
        )
        return {
            "status": "ok",
            "message": "DeepSeek API key loaded into this API process environment.",
            "data": _deepseek_status(config),
        }

    @app.post("/v1/deepseek/model")
    def deepseek_model(request: ModelRequest) -> dict[str, Any]:
        model = request.model.strip()
        if not model:
            raise HTTPException(status_code=400, detail="DeepSeek model is required.")
        config = update_config(lambda current: _set_deepseek_config(current, model=model), paths.config)
        return {
            "status": "ok",
            "message": f"DeepSeek model set to {model}.",
            "data": _deepseek_status(config),
        }

    @app.get("/v1/deepseek/models")
    def deepseek_models() -> dict[str, Any]:
        config = load_config(paths.config)
        return {
            "status": "ok",
            "models": _fetch_deepseek_models(config),
            "data": _deepseek_status(config),
        }

    @app.get("/v1/connectors")
    def connectors() -> list[dict[str, Any]]:
        config = load_config(paths.config)
        return _connector_statuses(config, registry.manifests)

    @app.get("/v1/terminal")
    def terminal_status() -> dict[str, Any]:
        config = load_config(paths.config)
        return _terminal_status(config)

    @app.post("/v1/terminal/enable")
    def terminal_enable() -> dict[str, Any]:
        config = update_config(lambda current: _set_terminal_enabled(current, True), paths.config)
        return {
            "status": "ok",
            "message": "Terminal policy, terminal.run tool, and terminal.run permission enabled.",
            "data": _terminal_status(config),
        }

    @app.post("/v1/terminal/disable")
    def terminal_disable() -> dict[str, Any]:
        config = update_config(lambda current: _set_terminal_enabled(current, False), paths.config)
        return {
            "status": "ok",
            "message": "Terminal policy and terminal.run tool disabled.",
            "data": _terminal_status(config),
        }

    @app.post("/v1/terminal/settings")
    def terminal_settings(request: TerminalSettingsRequest) -> dict[str, Any]:
        config = update_config(
            lambda current: _update_terminal_settings(current, request),
            paths.config,
        )
        return {
            "status": "ok",
            "message": "Terminal settings updated.",
            "data": _terminal_status(config),
        }

    @app.post("/v1/terminal/allow")
    def terminal_allow(request: TerminalCommandRequest) -> dict[str, Any]:
        command = _terminal_command_from_request(request.command)
        _validate_terminal_allowlist_command(command)
        config = update_config(
            lambda current: _set_terminal_allowlist_command(current, command, True),
            paths.config,
        )
        return {
            "status": "ok",
            "message": f"Allowlisted exact command: {' '.join(command)}",
            "data": _terminal_status(config),
        }

    @app.post("/v1/terminal/remove")
    def terminal_remove(request: TerminalCommandRequest) -> dict[str, Any]:
        command = _terminal_command_from_request(request.command)
        config = update_config(
            lambda current: _set_terminal_allowlist_command(current, command, False),
            paths.config,
        )
        return {
            "status": "ok",
            "message": f"Removed exact command if present: {' '.join(command)}",
            "data": _terminal_status(config),
        }

    @app.post("/v1/terminal/run")
    def terminal_run(request: TerminalCommandRequest) -> dict[str, Any]:
        command = _terminal_command_from_request(request.command)
        response = build_agent_core().handle_tool_request(
            ToolRequest(
                tool="terminal.run",
                args={"command": command, "cwd": request.cwd},
                reason="User requested a terminal command from the web dashboard.",
            )
        )
        return asdict(response)

    @app.get("/v1/telegram")
    def telegram_status() -> dict[str, Any]:
        config = load_config(paths.config)
        return telegram_runtime.status(config)

    @app.post("/v1/telegram/enable")
    def telegram_enable() -> dict[str, Any]:
        config = update_config(lambda current: _set_telegram_enabled(current, True), paths.config)
        telegram_runtime.start(config)
        return {
            "status": "ok",
            "message": "Telegram interface enabled.",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/disable")
    def telegram_disable() -> dict[str, Any]:
        config = update_config(lambda current: _set_telegram_enabled(current, False), paths.config)
        telegram_runtime.stop()
        return {
            "status": "ok",
            "message": "Telegram interface disabled.",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/allow")
    def telegram_allow(request: TelegramUserRequest) -> dict[str, Any]:
        config = update_config(
            lambda current: _set_telegram_user_allowed(current, request.user_id, True),
            paths.config,
        )
        telegram_runtime.start(config)
        return {
            "status": "ok",
            "message": f"Telegram user allowed: {request.user_id}",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/remove")
    def telegram_remove(request: TelegramUserRequest) -> dict[str, Any]:
        config = update_config(
            lambda current: _set_telegram_user_allowed(current, request.user_id, False),
            paths.config,
        )
        if not settings_from_config(config).allowed_user_ids:
            telegram_runtime.stop()
        return {
            "status": "ok",
            "message": f"Telegram user removed: {request.user_id}",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/token-env")
    def telegram_token_env(request: TelegramTokenEnvRequest) -> dict[str, Any]:
        config = update_config(
            lambda current: _set_telegram_token_env(current, request.bot_token_env),
            paths.config,
        )
        telegram_runtime.stop()
        telegram_runtime.start(config)
        return {
            "status": "ok",
            "message": f"Telegram token environment variable set to {request.bot_token_env}.",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/token")
    def telegram_token(request: TelegramTokenRequest) -> dict[str, Any]:
        token = request.token.strip()
        if not token:
            raise HTTPException(status_code=400, detail="Telegram token cannot be empty.")
        config = load_config(paths.config)
        token_env = str(
            config.get("interfaces", {})
            .get("telegram", {})
            .get("bot_token_env", DEFAULT_TOKEN_ENV)
        )
        settings = settings_from_config(config)
        storage = store_telegram_token(settings, token)
        telegram_runtime.start(config)
        return {
            "status": "ok",
            "message": f"Telegram token loaded into {storage} as {token_env}.",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/start")
    def telegram_start() -> dict[str, Any]:
        config = load_config(paths.config)
        started = telegram_runtime.start(config)
        return {
            "status": "ok" if started else "not_ready",
            "message": "Telegram polling started." if started else "Telegram polling is not ready.",
            "data": telegram_runtime.status(config),
        }

    @app.post("/v1/telegram/stop")
    def telegram_stop() -> dict[str, Any]:
        telegram_runtime.stop()
        config = load_config(paths.config)
        return {
            "status": "ok",
            "message": "Telegram polling stopped.",
            "data": telegram_runtime.status(config),
        }

    return app


class TelegramRuntime:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._last_error: str | None = None

    def start(self, config: dict[str, Any]) -> bool:
        settings = settings_from_config(config)
        if not settings.enabled or not settings.allowed_user_ids or not telegram_token_available(settings):
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return True
            self._last_error = None
            self._stop_event = threading.Event()
            try:
                interface = TelegramInterface.from_current_config()
            except TelegramConfigError as exc:
                self._last_error = str(exc)
                return False

            def run() -> None:
                try:
                    interface.run_polling(timeout_seconds=2, stop_event=self._stop_event)
                except TelegramError as exc:
                    self._last_error = str(exc)
                except Exception as exc:  # pragma: no cover - defensive guard
                    self._last_error = str(exc)

            self._thread = threading.Thread(target=run, name="dmdagent-api-telegram", daemon=True)
            self._thread.start()
            return True

    def stop(self) -> None:
        with self._lock:
            stop_event = self._stop_event
            thread = self._thread
            self._stop_event = None
            self._thread = None
        if stop_event is not None:
            stop_event.set()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)

    def status(self, config: dict[str, Any]) -> dict[str, Any]:
        status = _telegram_status(config)
        thread = self._thread
        polling = thread is not None and thread.is_alive()
        status["polling"] = polling
        status["polling_error"] = self._last_error
        return status

    def send_to_allowed_users(
        self,
        config: dict[str, Any],
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> bool:
        settings = settings_from_config(config)
        if not settings.enabled or not settings.allowed_user_ids or not telegram_token_available(settings):
            return False
        try:
            interface = TelegramInterface.from_current_config()
        except TelegramConfigError as exc:
            self._last_error = str(exc)
            return False
        sent = False
        for user_id in sorted(settings.allowed_user_ids):
            try:
                interface.api.send_message(user_id, text, reply_markup=reply_markup)
                sent = True
            except TelegramError as exc:
                self._last_error = str(exc)
        return sent


class ReminderRuntime:
    idle_sleep_seconds = 3600.0
    max_sleep_seconds = 3600.0
    precision_window_seconds = 120.0
    due_retry_sleep_seconds = 30.0
    immediate_due_sleep_seconds = 0.5

    def __init__(self, paths: AppPaths, telegram_runtime: TelegramRuntime) -> None:
        self._paths = paths
        self._telegram_runtime = telegram_runtime
        self._thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._last_due_attempt_count = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, name="dmdagent-reminders", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        stop_event = self._stop_event
        thread = self._thread
        self._stop_event = None
        self._thread = None
        if stop_event is not None:
            stop_event.set()
            wake_reminder_waiters()
        if thread is not None and thread.is_alive():
            thread.join(timeout=3)

    def _run(self) -> None:
        while self._stop_event is None or not self._stop_event.is_set():
            self.tick()
            if self._stop_event is None:
                return
            version = reminder_change_version()
            sleep_seconds = self.next_sleep_seconds()
            wait_for_reminder_change_since(version, sleep_seconds)

    def next_sleep_seconds(self, *, now: datetime | None = None) -> float:
        current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
        config = load_config(self._paths.config)
        context = ToolRuntimeContext(
            memory_root=self._paths.memory,
            workspace_root=self._paths.workspace,
            config=config,
            config_path=self._paths.config,
        )
        next_due = next_pending_reminder_due_at(context)
        if next_due is None:
            return self.idle_sleep_seconds

        seconds_until_due = (next_due - current).total_seconds()
        if seconds_until_due <= 0:
            if self._last_due_attempt_count > 0:
                return self.due_retry_sleep_seconds
            return self.immediate_due_sleep_seconds
        if seconds_until_due <= self.precision_window_seconds:
            return seconds_until_due
        seconds_until_precision_window = seconds_until_due - self.precision_window_seconds
        if seconds_until_precision_window <= 0:
            return self.immediate_due_sleep_seconds
        return min(seconds_until_precision_window, self.max_sleep_seconds)

    def tick(self) -> int:
        config = load_config(self._paths.config)
        context = ToolRuntimeContext(
            memory_root=self._paths.memory,
            workspace_root=self._paths.workspace,
            config=config,
            config_path=self._paths.config,
        )
        sent_count = 0
        due_attempt_count = 0
        for reminder in due_reminders(context):
            reminder_id = int(reminder.get("id") or 0)
            if reminder_id <= 0:
                continue
            due_attempt_count += 1
            text = reminder_notification_text(reminder)
            reply_markup = reminder_keyboard(reminder)
            if self._telegram_runtime.send_to_allowed_users(config, text, reply_markup=reply_markup):
                mark_reminder_notified(context, reminder_id, channel="telegram")
                AuditStore(self._paths.audit_db).record_event(
                    AuditEvent(
                        event_type="reminder.notification",
                        tool="reminders.create",
                        approved=True,
                        result_status="sent",
                        metadata={"reminder_id": reminder_id, "channel": "telegram"},
                    )
                )
                sent_count += 1
        self._last_due_attempt_count = due_attempt_count
        return sent_count


def _set_tool_enabled(paths: AppPaths, tool_name: str, enabled: bool) -> dict[str, Any]:
    registry = build_builtin_registry()
    if tool_name not in registry.manifests:
        return {
            "status": "error",
            "message": f"Unknown tool: {tool_name}",
            "data": None,
        }
    update_config(
        lambda config: config.setdefault("tools", {}).setdefault(tool_name, {}).__setitem__(
            "enabled",
            enabled,
        ),
        paths.config,
    )
    return {
        "status": "ok",
        "message": f"Tool {tool_name} {'enabled' if enabled else 'disabled'}.",
        "data": {"tool": tool_name, "enabled": enabled},
    }


def _set_permission(paths: AppPaths, permission: str, granted: bool) -> dict[str, Any]:
    registry = build_builtin_registry()
    known_permissions = {
        item
        for manifest in registry.manifests.values()
        for item in manifest.permissions
    }
    if permission not in known_permissions:
        return {
            "status": "error",
            "message": f"Unknown permission: {permission}",
            "data": None,
        }
    update_config(
        lambda config: _set_permission_config(config, permission, granted),
        paths.config,
    )
    return {
        "status": "ok",
        "message": f"Permission {permission} {'granted' if granted else 'revoked'}.",
        "data": {"permission": permission, "granted": granted},
    }


def _set_model_config(
    config: dict[str, Any],
    *,
    mode: str,
    model: str,
    planner_model: str,
) -> None:
    llm = config.setdefault("llm", {})
    llm["provider"] = "ollama"
    llm["mode"] = mode
    llm["model"] = model
    llm["planner_model"] = planner_model
    llm["base_url"] = "http://localhost:11434"
    llm["api_key_env"] = None


def _set_openai_config(config: dict[str, Any], *, model: str) -> None:
    llm = config.setdefault("llm", {})
    llm["provider"] = "openai"
    llm["mode"] = "openai"
    llm["model"] = model
    llm["planner_model"] = model
    llm["base_url"] = "https://api.openai.com/v1"
    llm["api_key_env"] = DEFAULT_OPENAI_API_KEY_ENV


def _set_openai_limit(config: dict[str, Any], limit_usd: float | None) -> None:
    usage = config.setdefault("openai_usage", {})
    usage["limit_usd"] = None if limit_usd is None else max(0.0, float(limit_usd))


def _set_deepseek_config(config: dict[str, Any], *, model: str) -> None:
    llm = config.setdefault("llm", {})
    llm["provider"] = "deepseek"
    llm["mode"] = "deepseek"
    llm["model"] = model
    llm["planner_model"] = model
    llm["base_url"] = DEEPSEEK_BASE_URL
    llm["api_key_env"] = DEFAULT_DEEPSEEK_API_KEY_ENV


def _openai_status(config: dict[str, Any], paths: AppPaths) -> dict[str, Any]:
    llm = config.get("llm", {})
    api_key_env = _openai_api_key_env(config)
    return {
        "provider": str(llm.get("provider", "")),
        "model": str(llm.get("model", "")),
        "planner_model": llm.get("planner_model"),
        "base_url": str(llm.get("base_url") or "https://api.openai.com/v1"),
        "api_key_env": api_key_env,
        "api_key_available": bool(os.environ.get(api_key_env, "").strip()),
        "usage": openai_usage_summary(paths.audit_db, config=config),
    }


def _deepseek_status(config: dict[str, Any]) -> dict[str, Any]:
    llm = config.get("llm", {})
    api_key_env = _deepseek_api_key_env(config)
    return {
        "provider": str(llm.get("provider", "")),
        "model": str(llm.get("model", "")),
        "planner_model": llm.get("planner_model"),
        "base_url": _deepseek_base_url(config),
        "api_key_env": api_key_env,
        "api_key_available": bool(os.environ.get(api_key_env, "").strip()),
        "default_models": list(DEEPSEEK_MODEL_FALLBACKS),
    }


def _fetch_openai_models(config: dict[str, Any]) -> list[str]:
    api_key_env = _openai_api_key_env(config)
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"OpenAI API key is not loaded in this process: {api_key_env}.",
        )
    request = urllib.request.Request(
        "https://api.openai.com/v1/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise HTTPException(
            status_code=502,
            detail=f"OpenAI models request failed with HTTP {exc.code}: {detail}",
        ) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to OpenAI models API: {exc.reason}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="OpenAI models response was not JSON.") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="OpenAI models response did not include a data list.")
    models = {
        str(item.get("id")).strip()
        for item in data
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    return sorted(models, key=_openai_model_sort_key)


def _fetch_deepseek_models(config: dict[str, Any]) -> list[str]:
    api_key_env = _deepseek_api_key_env(config)
    api_key = os.environ.get(api_key_env, "").strip()
    if not api_key:
        raise HTTPException(
            status_code=400,
            detail=f"DeepSeek API key is not loaded in this process: {api_key_env}.",
        )
    request = urllib.request.Request(
        f"{_deepseek_base_url(config).rstrip('/')}/models",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="GET",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise HTTPException(
            status_code=502,
            detail=f"DeepSeek models request failed with HTTP {exc.code}: {detail}",
        ) from exc
    except urllib.error.URLError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Cannot connect to DeepSeek models API: {exc.reason}",
        ) from exc
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=502, detail="DeepSeek models response was not JSON.") from exc

    data = payload.get("data")
    if not isinstance(data, list):
        raise HTTPException(status_code=502, detail="DeepSeek models response did not include a data list.")
    models = {
        str(item.get("id")).strip()
        for item in data
        if isinstance(item, dict) and str(item.get("id", "")).strip()
    }
    return sorted(models, key=_deepseek_model_sort_key)


def _openai_api_key_env(config: dict[str, Any]) -> str:
    value = config.get("llm", {}).get("api_key_env")
    return str(value or DEFAULT_OPENAI_API_KEY_ENV)


def _deepseek_api_key_env(config: dict[str, Any]) -> str:
    llm = config.get("llm", {})
    if str(llm.get("provider", "")).lower() == "deepseek":
        value = llm.get("api_key_env")
        if value:
            return str(value)
    return DEFAULT_DEEPSEEK_API_KEY_ENV


def _deepseek_base_url(config: dict[str, Any]) -> str:
    llm = config.get("llm", {})
    if str(llm.get("provider", "")).lower() == "deepseek":
        value = llm.get("base_url")
        if value:
            return str(value)
    return DEEPSEEK_BASE_URL


def _default_openai_model(config: dict[str, Any]) -> str:
    llm = config.get("llm", {})
    provider = str(llm.get("provider", "")).lower()
    model = str(llm.get("model") or "").strip()
    if provider == "openai" and _looks_like_openai_chat_model(model):
        return model
    return "gpt-4o-mini"


def _default_deepseek_model(config: dict[str, Any]) -> str:
    llm = config.get("llm", {})
    provider = str(llm.get("provider", "")).lower()
    model = str(llm.get("model") or "").strip()
    if provider == "deepseek" and _looks_like_deepseek_model(model):
        return model
    return DEFAULT_DEEPSEEK_MODEL


def _openai_model_sort_key(model: str) -> tuple[int, str]:
    normalized = model.lower()
    if _looks_like_openai_chat_model(normalized):
        return (0, normalized)
    return (1, normalized)


def _deepseek_model_sort_key(model: str) -> tuple[int, str]:
    normalized = model.lower()
    try:
        return (0, DEEPSEEK_MODEL_FALLBACKS.index(normalized))
    except ValueError:
        return (1, normalized)


def _looks_like_openai_chat_model(model: str) -> bool:
    normalized = model.lower()
    return normalized.startswith(("gpt-", "chatgpt-")) or re.match(r"o\d", normalized) is not None


def _looks_like_deepseek_model(model: str) -> bool:
    return model.lower().startswith("deepseek-")


def _set_permission_config(
    config: dict[str, Any],
    permission: str,
    granted: bool,
) -> None:
    permissions = set(config.setdefault("permissions", {}).setdefault("granted", []))
    if granted:
        permissions.add(permission)
    else:
        permissions.discard(permission)
    config["permissions"]["granted"] = sorted(permissions)


def _connector_statuses(
    config: dict[str, Any],
    manifests: dict[str, Any],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Any]] = {}
    for manifest in manifests.values():
        connector = str(manifest.name).split(".", 1)[0]
        grouped.setdefault(connector, []).append(manifest)

    granted_permissions = set(config.get("permissions", {}).get("granted", []))
    tool_overrides = config.get("tools", {})
    statuses: list[dict[str, Any]] = []
    for name in sorted(grouped):
        tools = grouped[name]
        enabled_tools = [
            tool.name
            for tool in tools
            if bool(tool_overrides.get(tool.name, {}).get("enabled", tool.default_enabled))
        ]
        required_permissions = sorted(
            {
                permission
                for tool in tools
                for permission in getattr(tool, "permissions", ())
            }
        )
        status = "enabled" if enabled_tools else "disabled"
        detail = ""
        if name == "gmail":
            status = "not_configured"
            detail = "Gmail OAuth connector is not implemented in this build."
        elif name == "calendar":
            status = "local_store" if enabled_tools else "disabled"
            detail = "Calendar tools use the local workspace event store until OAuth sync is added."
        elif name == "browser":
            status = "read_ready" if enabled_tools else "disabled"
            detail = "Read/extract tools use guarded HTTP fetches; click/fill/submit use optional isolated Playwright runtime."
        elif name == "terminal":
            terminal = _terminal_status(config)
            status = "enabled" if terminal["ready"] else "disabled"
            detail = "Terminal commands require policy enablement, tool enablement, permission grant, and an exact allowlist. Approval can be per-run or opt-in automatic for allowlisted commands."
        statuses.append(
            {
                "name": name,
                "status": status,
                "detail": detail,
                "tools_total": len(tools),
                "enabled_tools": enabled_tools,
                "permissions_required": required_permissions,
                "permissions_granted": sorted(
                    permission for permission in required_permissions if permission in granted_permissions
                ),
            }
        )

    telegram = _telegram_status(config)
    statuses.append(
        {
            "name": "telegram",
            "status": "enabled" if telegram["ready"] else "disabled",
            "detail": "Remote Telegram access requires enablement, a loaded token, and at least one allowlisted user ID.",
            "tools_total": 0,
            "enabled_tools": [],
            "permissions_required": [],
            "permissions_granted": [],
            "interface": telegram,
        }
    )
    return statuses


def _terminal_status(config: dict[str, Any]) -> dict[str, Any]:
    terminal = _terminal_config_section(config)
    permissions = set(config.get("permissions", {}).get("granted", []))
    tool_enabled = bool(config.get("tools", {}).get("terminal.run", {}).get("enabled", False))
    policy_enabled = bool(terminal.get("enabled", False))
    permission_granted = "terminal.run" in permissions
    return {
        "enabled": policy_enabled,
        "tool_enabled": tool_enabled,
        "permission_granted": permission_granted,
        "ready": policy_enabled and tool_enabled and permission_granted,
        "workspace_only": bool(terminal.get("workspace_only", True)),
        "workspace_root": _terminal_workspace_root(config),
        "timeout_seconds": int(terminal.get("timeout_seconds", 30)),
        "max_output_chars": int(terminal.get("max_output_chars", 20000)),
        "auto_approve_allowlisted": bool(terminal.get("auto_approve_allowlisted", False)),
        "allowed_commands": [list(command) for command in _terminal_allowed_commands(terminal)],
    }


def _terminal_config_section(config: dict[str, Any]) -> dict[str, Any]:
    terminal = config.setdefault("terminal", {})
    terminal.setdefault("enabled", False)
    terminal.setdefault("mode", "allowlist")
    terminal.setdefault("workspace_only", True)
    terminal.setdefault("workspace_root", "")
    terminal.setdefault("timeout_seconds", 30)
    terminal.setdefault("max_output_chars", 20000)
    terminal.setdefault("auto_approve_allowlisted", False)
    terminal.setdefault(
        "allowed_commands",
        [
            ["pwd"],
            ["ls"],
            ["git", "status"],
            ["git", "diff"],
            ["cat", "README.md"],
            ["cat", "readme.md"],
            ["npm", "test"],
            ["pytest"],
        ],
    )
    return terminal


def _terminal_workspace_root(config: dict[str, Any]) -> str:
    raw_root = _terminal_config_section(config).get("workspace_root")
    if isinstance(raw_root, str) and raw_root.strip():
        return str(Path(raw_root).expanduser())
    return str(AppPaths.default().workspace)


def _terminal_allowed_commands(terminal: dict[str, Any]) -> list[tuple[str, ...]]:
    commands: list[tuple[str, ...]] = []
    raw_commands = terminal.setdefault("allowed_commands", [])
    if isinstance(raw_commands, list):
        for command in raw_commands:
            if isinstance(command, list | tuple) and command:
                commands.append(tuple(str(part) for part in command if str(part)))
    return commands


def _set_terminal_enabled(config: dict[str, Any], enabled: bool) -> None:
    terminal = _terminal_config_section(config)
    terminal["enabled"] = enabled
    config.setdefault("tools", {}).setdefault("terminal.run", {})["enabled"] = enabled
    if enabled:
        permissions = set(config.setdefault("permissions", {}).setdefault("granted", []))
        permissions.add("terminal.run")
        config["permissions"]["granted"] = sorted(permissions)


def _update_terminal_settings(
    config: dict[str, Any],
    request: TerminalSettingsRequest,
) -> None:
    terminal = _terminal_config_section(config)
    if request.workspace_only is not None:
        terminal["workspace_only"] = bool(request.workspace_only)
    if request.workspace_root is not None:
        value = request.workspace_root.strip()
        if value:
            root = Path(value).expanduser()
            if not root.is_absolute():
                raise HTTPException(status_code=400, detail="workspace_root must be an absolute path.")
            terminal["workspace_root"] = str(root)
        else:
            terminal["workspace_root"] = ""
    if request.auto_approve_allowlisted is not None:
        terminal["auto_approve_allowlisted"] = bool(request.auto_approve_allowlisted)
    if request.timeout_seconds is not None:
        if not 1 <= request.timeout_seconds <= 600:
            raise HTTPException(status_code=400, detail="timeout_seconds must be between 1 and 600.")
        terminal["timeout_seconds"] = request.timeout_seconds
    if request.max_output_chars is not None:
        if not 100 <= request.max_output_chars <= 1_000_000:
            raise HTTPException(
                status_code=400,
                detail="max_output_chars must be between 100 and 1000000.",
            )
        terminal["max_output_chars"] = request.max_output_chars


def _terminal_command_from_request(raw: str | list[str]) -> list[str]:
    try:
        command = shlex.split(raw) if isinstance(raw, str) else [str(part) for part in raw]
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid command syntax: {exc}") from exc
    command = [part for part in command if part]
    if not command:
        raise HTTPException(status_code=400, detail="Command is required.")
    return command


def _validate_terminal_allowlist_command(command: list[str]) -> None:
    try:
        TerminalPolicy(enabled=True, allowed_commands=(tuple(command),)).validate(command)
    except PermissionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _set_terminal_allowlist_command(
    config: dict[str, Any],
    command: list[str],
    allowed: bool,
) -> None:
    terminal = _terminal_config_section(config)
    commands = _terminal_allowed_commands(terminal)
    command_tuple = tuple(command)
    if allowed and command_tuple not in commands:
        commands.append(command_tuple)
    if not allowed:
        commands = [item for item in commands if item != command_tuple]
    terminal["allowed_commands"] = [list(item) for item in commands]


def _telegram_status(config: dict[str, Any]) -> dict[str, Any]:
    settings = settings_from_config(config)
    token_available = telegram_token_available(settings)
    allowed_user_ids = sorted(settings.allowed_user_ids)
    return {
        "enabled": settings.enabled,
        "allowed_user_ids": allowed_user_ids,
        "bot_token_env": settings.bot_token_env,
        "bot_token_available": token_available,
        "polling_timeout_seconds": settings.polling_timeout_seconds,
        "ready": settings.enabled and token_available and bool(allowed_user_ids),
    }


def _telegram_config_section(config: dict[str, Any]) -> dict[str, Any]:
    telegram = config.setdefault("interfaces", {}).setdefault("telegram", {})
    telegram.setdefault("enabled", False)
    telegram.setdefault("allowed_user_ids", [])
    telegram.setdefault("bot_token_env", DEFAULT_TOKEN_ENV)
    telegram.setdefault("polling_timeout_seconds", 30)
    return telegram


def _set_telegram_enabled(config: dict[str, Any], enabled: bool) -> None:
    _telegram_config_section(config)["enabled"] = enabled


def _set_telegram_user_allowed(config: dict[str, Any], user_id: int, allowed: bool) -> None:
    telegram = _telegram_config_section(config)
    user_ids = _telegram_allowed_user_ids(telegram)
    if allowed:
        user_ids.add(int(user_id))
    else:
        user_ids.discard(int(user_id))
    telegram["allowed_user_ids"] = sorted(user_ids)


def _set_telegram_token_env(config: dict[str, Any], token_env: str) -> None:
    value = token_env.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", value):
        raise HTTPException(
            status_code=400,
            detail="bot_token_env must be a valid environment variable name.",
        )
    _telegram_config_section(config)["bot_token_env"] = value


def _telegram_allowed_user_ids(telegram: dict[str, Any]) -> set[int]:
    allowed: set[int] = set()
    raw_allowed = telegram.setdefault("allowed_user_ids", [])
    if isinstance(raw_allowed, list):
        for item in raw_allowed:
            try:
                allowed.add(int(item))
            except (TypeError, ValueError):
                continue
    return allowed


app = create_app()
