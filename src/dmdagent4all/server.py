from __future__ import annotations

import os
import html
import json
import mimetypes
import re
import secrets as token_secrets
import shlex
import threading
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from dmdagent4all import __version__
from dmdagent4all.agent.planner import ANSWER_PROMPT, CHAT_PROMPT, SYSTEM_PROMPT
from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.config import load_config, update_config, write_default_config
from dmdagent4all.doctor import doctor_summary, run_doctor
from dmdagent4all.email_oauth import (
    DEFAULT_GMAIL_OAUTH_REDIRECT_URI,
    GMAIL_OAUTH_SCOPE,
    delete_email_secret,
    exchange_google_code,
    google_authorization_url,
    load_email_secret,
    store_email_secret,
)
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
from dmdagent4all.sandbox import TerminalPolicy, active_terminal_processes, emergency_stop_terminal_processes
from dmdagent4all.tools import build_builtin_registry
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.storage import downloads_root_from_config
from dmdagent4all.tools.reminders import (
    due_reminders,
    mark_reminder_notified,
    next_pending_reminder_due_at,
    reminder_change_version,
    wait_for_reminder_change_since,
    wake_reminder_waiters,
)
from dmdagent4all.workspace import WorkspaceError, WorkspaceManager


DEFAULT_DEEPSEEK_API_KEY_ENV = "DMDAGENT_DEEPSEEK_API_KEY"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEEPSEEK_MODEL_FALLBACKS = ["deepseek-v4-flash", "deepseek-v4-pro"]
EMAIL_MAX_BODY_CHARS_DEFAULT = 20_000
EMAIL_PROVIDER_DEFAULTS = {
    "gmail": {
        "enabled": False,
        "auth_method": "app_password",
        "imap_host": "imap.gmail.com",
        "imap_port": 993,
        "smtp_host": "smtp.gmail.com",
        "smtp_port": 587,
        "username_env": "DMDAGENT_GMAIL_USERNAME",
        "password_env": "DMDAGENT_GMAIL_APP_PASSWORD",
        "from_env": "DMDAGENT_GMAIL_FROM",
        "oauth_client_id": "",
        "oauth_redirect_uri": DEFAULT_GMAIL_OAUTH_REDIRECT_URI,
        "oauth_email": "",
        "oauth_from_address": "",
        "mailbox": "INBOX",
        "archive_mailbox": "[Gmail]/All Mail",
    },
    "outlook": {
        "enabled": False,
        "auth_method": "app_password",
        "imap_host": "outlook.office365.com",
        "imap_port": 993,
        "smtp_host": "smtp.office365.com",
        "smtp_port": 587,
        "username_env": "DMDAGENT_OUTLOOK_USERNAME",
        "password_env": "DMDAGENT_OUTLOOK_APP_PASSWORD",
        "from_env": "DMDAGENT_OUTLOOK_FROM",
        "mailbox": "INBOX",
        "archive_mailbox": "Archive",
    },
}
WORKSPACE_FILE_ROOTS = {
    "scrapefiles": "Scraped Markdown",
    "browser-downloads": "Browser Downloads",
}
WORKSPACE_TEXT_SUFFIXES = {
    ".csv",
    ".htm",
    ".html",
    ".json",
    ".log",
    ".markdown",
    ".md",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
}
WORKSPACE_MAX_PREVIEW_BYTES = 200_000


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
    allow_safe_workspace_commands: bool | None = None


class EmailProviderConfigurationRequest(BaseModel):
    enabled: bool | None = None
    auth_method: str | None = None
    imap_host: str | None = None
    imap_port: int | None = None
    smtp_host: str | None = None
    smtp_port: int | None = None
    username_env: str | None = None
    password_env: str | None = None
    from_env: str | None = None
    oauth_client_id: str | None = None
    oauth_redirect_uri: str | None = None
    oauth_email: str | None = None
    oauth_from_address: str | None = None
    mailbox: str | None = None
    archive_mailbox: str | None = None


class EmailConfigurationRequest(BaseModel):
    max_body_chars: int | None = None
    gmail: EmailProviderConfigurationRequest | None = None
    outlook: EmailProviderConfigurationRequest | None = None


class AgentConfigurationRequest(BaseModel):
    downloads_root: str | None = None
    agent_name: str | None = None
    user_name: str | None = None
    preferred_language: str | None = None
    response_language: str | None = None
    chat_max_tokens: int | None = None
    planner_max_tokens: int | None = None
    synthesis_max_tokens: int | None = None
    repair_max_tokens: int | None = None
    chat_history_turns: int | None = None
    chat_history_char_limit: int | None = None
    planner_history_turns: int | None = None
    planner_history_char_limit: int | None = None
    planner_temperature: float | None = None
    planner_think: bool | None = None
    chat_system_prompt: str | None = None
    planner_system_prompt: str | None = None
    answer_system_prompt: str | None = None
    send_chat_history_to_cloud: bool | None = None
    browser_timeout_seconds: int | None = None
    browser_max_response_bytes: int | None = None
    browser_max_text_chars: int | None = None
    approval_required_at_risk: int | None = None
    email: EmailConfigurationRequest | None = None


class TerminalCommandRequest(BaseModel):
    command: str | list[str]
    cwd: str | None = None


class TelegramUserRequest(BaseModel):
    user_id: int


class TelegramTokenEnvRequest(BaseModel):
    bot_token_env: str


class TelegramTokenRequest(BaseModel):
    token: str


class EmailCredentialsRequest(BaseModel):
    provider: str
    username: str
    app_password: str
    from_address: str | None = None


class GmailOAuthStartRequest(BaseModel):
    client_id: str
    client_secret: str | None = None
    email: str
    from_address: str | None = None
    redirect_uri: str | None = None


class GmailOAuthSecretRequest(BaseModel):
    client_secret: str
    client_id: str | None = None
    email: str | None = None
    from_address: str | None = None
    redirect_uri: str | None = None


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
    google_oauth_states: dict[str, datetime] = {}
    google_oauth_lock = threading.RLock()

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

    @app.get("/v1/emergency")
    def emergency_status() -> dict[str, Any]:
        config = load_config(paths.config)
        return _emergency_status(config)

    @app.post("/v1/emergency/stop")
    def emergency_stop() -> dict[str, Any]:
        terminal_result = emergency_stop_terminal_processes()
        telegram_runtime.stop()
        reminder_runtime.stop()
        cancelled = _cancel_pending_approvals(paths)

        def mutate(config: dict[str, Any]) -> None:
            emergency = config.setdefault("runtime", {}).setdefault("emergency_stop", {})
            emergency["active"] = True
            emergency["triggered_at"] = datetime.now(timezone.utc).isoformat()
            emergency["reason"] = "Emergency stop triggered from dashboard/API."

        config = update_config(mutate, paths.config)
        AuditStore(paths.audit_db).record_event(
            AuditEvent(
                event_type="emergency_stop",
                tool=None,
                risk=5,
                approved=True,
                result_status="stopped",
                metadata={"terminal": terminal_result, "cancelled_approvals": cancelled},
            )
        )
        return {
            "status": "ok",
            "message": "Emergency stop active. Terminal processes stopped, approvals cancelled, and tool execution blocked until reset.",
            "data": {
                **_emergency_status(config),
                "terminal": terminal_result,
                "cancelled_approvals": cancelled,
            },
        }

    @app.post("/v1/emergency/reset")
    def emergency_reset() -> dict[str, Any]:
        def mutate(config: dict[str, Any]) -> None:
            emergency = config.setdefault("runtime", {}).setdefault("emergency_stop", {})
            emergency["active"] = False
            emergency["triggered_at"] = ""
            emergency["reason"] = ""

        config = update_config(mutate, paths.config)
        telegram_runtime.start(config)
        reminder_runtime.start()
        AuditStore(paths.audit_db).record_event(
            AuditEvent(
                event_type="emergency_reset",
                tool=None,
                risk=0,
                approved=True,
                result_status="reset",
                metadata={},
            )
        )
        return {
            "status": "ok",
            "message": "Emergency stop reset. Normal policy and approvals are active again.",
            "data": _emergency_status(config),
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

    @app.get("/v1/workspace-files")
    def workspace_files() -> dict[str, Any]:
        return _workspace_files_response(paths, load_config(paths.config))

    @app.get("/v1/workspace-files/file")
    def workspace_file(path: str) -> dict[str, Any]:
        config = load_config(paths.config)
        file_path = _resolve_workspace_file(paths, config, path)
        if not _is_previewable_workspace_file(file_path):
            raise HTTPException(status_code=415, detail="Workspace file is not text-previewable.")
        content, truncated = _read_workspace_text_preview(file_path)
        item = _workspace_file_item(paths, config, file_path)
        return {
            **item,
            "content": content,
            "truncated": truncated,
        }

    @app.get("/v1/workspace-files/download")
    def workspace_file_download(path: str) -> FileResponse:
        file_path = _resolve_workspace_file(paths, load_config(paths.config), path)
        return FileResponse(file_path, filename=file_path.name)

    @app.get("/v1/configuration")
    def configuration() -> dict[str, Any]:
        return _configuration_response(paths, load_config(paths.config))

    @app.post("/v1/configuration")
    def configuration_update(request: AgentConfigurationRequest) -> dict[str, Any]:
        config = update_config(lambda current: _update_agent_configuration(current, request), paths.config)
        return {
            "status": "ok",
            "message": "Agent configuration updated.",
            "data": _configuration_response(paths, config),
        }

    @app.post("/v1/email/credentials")
    def email_credentials(request: EmailCredentialsRequest) -> dict[str, Any]:
        provider = request.provider.strip().lower()
        if provider not in EMAIL_PROVIDER_DEFAULTS:
            raise HTTPException(status_code=400, detail="provider must be gmail or outlook.")
        def update(current: dict[str, Any]) -> None:
            _set_email_provider_enabled(current, provider, True)
            current.setdefault("email", {}).setdefault(provider, {})["auth_method"] = "app_password"

        config = update_config(update, paths.config)
        return _load_email_credentials(config, request)

    @app.post("/v1/email/oauth/google/client-secret")
    def gmail_oauth_client_secret(request: GmailOAuthSecretRequest) -> dict[str, Any]:
        try:
            storage = store_email_secret("gmail", "oauth_client_secret", request.client_secret)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if request.client_id or request.email or request.from_address or request.redirect_uri:
            update_config(
                lambda current: _configure_gmail_oauth(
                    current,
                    GmailOAuthStartRequest(
                        client_id=request.client_id or "",
                        email=request.email or "",
                        from_address=request.from_address,
                        redirect_uri=request.redirect_uri,
                    ),
                    require_ready_fields=False,
                ),
                paths.config,
            )
        config = load_config(paths.config)
        return {
            "status": "ok",
            "message": "Gmail OAuth client secret loaded into the local secret store.",
            "data": {
                "provider": "gmail",
                "secret_loaded": True,
                "storage": storage,
                "gmail": _email_configuration_response(config)["gmail"],
            },
        }

    @app.post("/v1/email/oauth/google/start")
    def gmail_oauth_start(request: GmailOAuthStartRequest) -> dict[str, Any]:
        state = token_secrets.token_urlsafe(32)
        with google_oauth_lock:
            google_oauth_states[state] = datetime.now(timezone.utc)
        config = update_config(lambda current: _configure_gmail_oauth(current, request), paths.config)
        if request.client_secret and request.client_secret.strip():
            try:
                store_email_secret("gmail", "oauth_client_secret", request.client_secret)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
        gmail = _email_configuration_response(config)["gmail"]
        if not gmail["oauth_client_secret_loaded"]:
            raise HTTPException(status_code=400, detail="Gmail OAuth client secret is required before starting authorization.")
        authorization_url = google_authorization_url(
            client_id=gmail["oauth_client_id"],
            redirect_uri=gmail["oauth_redirect_uri"],
            state=state,
        )
        return {
            "status": "ok",
            "message": "Open the Google authorization URL and approve Gmail access.",
            "data": {
                "authorization_url": authorization_url,
                "redirect_uri": gmail["oauth_redirect_uri"],
                "scope": GMAIL_OAUTH_SCOPE,
            },
        }

    @app.get("/v1/email/oauth/google/callback")
    def gmail_oauth_callback(code: str | None = None, state: str | None = None, error: str | None = None) -> HTMLResponse:
        if error:
            return _oauth_callback_page("Google authorization was denied.", error, success=False)
        if not code or not state:
            return _oauth_callback_page("Google authorization failed.", "Missing code or state.", success=False)
        with google_oauth_lock:
            state_created = google_oauth_states.pop(state, None)
        if state_created is None:
            return _oauth_callback_page("Google authorization failed.", "OAuth state was not recognized. Start again from Config.", success=False)
        config = load_config(paths.config)
        gmail = _email_configuration_response(config)["gmail"]
        client_secret = load_email_secret("gmail", "oauth_client_secret")
        if not gmail["oauth_client_id"] or not client_secret:
            return _oauth_callback_page("Google authorization failed.", "Client ID or client secret is missing.", success=False)
        try:
            token = exchange_google_code(
                client_id=gmail["oauth_client_id"],
                client_secret=client_secret,
                redirect_uri=gmail["oauth_redirect_uri"],
                code=code,
            )
        except ValueError as exc:
            return _oauth_callback_page("Google token exchange failed.", str(exc), success=False)
        refresh_token = str(token.get("refresh_token") or "").strip()
        if refresh_token:
            try:
                store_email_secret("gmail", "oauth_refresh_token", refresh_token)
            except ValueError as exc:
                return _oauth_callback_page("Could not store Gmail OAuth token.", str(exc), success=False)
        elif not load_email_secret("gmail", "oauth_refresh_token"):
            return _oauth_callback_page(
                "Google authorization did not return a refresh token.",
                "Start again and make sure the consent screen is shown.",
                success=False,
            )
        update_config(lambda current: _set_email_provider_enabled(current, "gmail", True), paths.config)
        return _oauth_callback_page(
            "Gmail OAuth2 is connected.",
            "You can close this tab and return to DMD Agent.",
            success=True,
        )

    @app.post("/v1/email/oauth/google/disconnect")
    def gmail_oauth_disconnect() -> dict[str, Any]:
        delete_email_secret("gmail", "oauth_refresh_token")
        config = update_config(lambda current: _disconnect_gmail_oauth(current), paths.config)
        return {
            "status": "ok",
            "message": "Gmail OAuth refresh token disconnected.",
            "data": _email_configuration_response(config)["gmail"],
        }

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

    _register_spa_routes(app)
    _mount_static_dashboard(app)
    return app


def _emergency_status(config: dict[str, Any]) -> dict[str, Any]:
    emergency = config.get("runtime", {}).get("emergency_stop", {})
    if not isinstance(emergency, dict):
        emergency = {}
    return {
        "active": bool(emergency.get("active") is True),
        "triggered_at": str(emergency.get("triggered_at") or ""),
        "reason": str(emergency.get("reason") or ""),
        "active_terminal_processes": active_terminal_processes(),
    }


def _cancel_pending_approvals(paths: AppPaths) -> list[int]:
    store = AuditStore(paths.audit_db)
    cancelled: list[int] = []
    for approval in store.list_approvals(status="pending", limit=1000):
        approval_id = approval.get("id")
        if isinstance(approval_id, int) and store.set_approval_status(approval_id, "cancelled"):
            cancelled.append(approval_id)
    return cancelled


def _mount_static_dashboard(app: FastAPI) -> None:
    static_dir = _dashboard_static_dir()
    if static_dir is not None:
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="dashboard")


def _register_spa_routes(app: FastAPI) -> None:
    def index_response() -> FileResponse:
        static_dir = _dashboard_static_dir()
        if static_dir is None:
            raise HTTPException(status_code=404, detail="Dashboard build is not available.")
        return FileResponse(static_dir / "index.html")

    @app.get("/chat", include_in_schema=False)
    @app.get("/downloads", include_in_schema=False)
    @app.get("/approvals", include_in_schema=False)
    @app.get("/models", include_in_schema=False)
    @app.get("/memory", include_in_schema=False)
    @app.get("/telegram", include_in_schema=False)
    @app.get("/tools", include_in_schema=False)
    @app.get("/logs", include_in_schema=False)
    @app.get("/help", include_in_schema=False)
    @app.get("/config", include_in_schema=False)
    @app.get("/config/{section}", include_in_schema=False)
    def dashboard_route(section: str | None = None) -> FileResponse:
        del section
        return index_response()


def _dashboard_static_dir() -> Path | None:
    configured = os.environ.get("DMDAGENT_STATIC_DIR", "").strip()
    candidates = [
        Path(configured).expanduser() if configured else None,
        Path(__file__).resolve().parents[2] / "frontend" / "dist",
    ]
    for candidate in candidates:
        if candidate is None:
            continue
        if (candidate / "index.html").exists():
            return candidate
    return None


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


def _configuration_response(paths: AppPaths, config: dict[str, Any]) -> dict[str, Any]:
    downloads_root = downloads_root_from_config(config, default_root=paths.workspace)
    llm = config.get("llm", {})
    return {
        "paths": {
            "data_dir": str(paths.root),
            "config": str(paths.config),
            "memory": str(paths.memory),
            "workspace": str(paths.workspace),
            "current_workspace": str(
                WorkspaceManager.from_config(config, fallback_workspace=paths.workspace).current_workspace
            ),
            "downloads_root": str(downloads_root),
            "downloads_root_custom": bool(str(config.get("storage", {}).get("downloads_root") or "").strip()),
            "audit_db": str(paths.audit_db),
        },
        "setup": config.get("setup", {}),
        "llm": llm,
        "system_prompts": _system_prompts_response(llm),
        "browser": config.get("browser", {}),
        "email": _email_configuration_response(config),
        "terminal": config.get("terminal", {}),
        "privacy": config.get("privacy", {}),
        "permissions": config.get("permissions", {}),
        "storage": config.get("storage", {}),
        "workspace": WorkspaceManager.from_config(config, fallback_workspace=paths.workspace).workspace_info(),
    }


def _email_configuration_response(config: dict[str, Any]) -> dict[str, Any]:
    email = config.get("email", {})
    if not isinstance(email, dict):
        email = {}
    max_body_chars = email.get("max_body_chars", EMAIL_MAX_BODY_CHARS_DEFAULT)
    return {
        "max_body_chars": max(1_000, min(_coerce_int(max_body_chars, EMAIL_MAX_BODY_CHARS_DEFAULT), 200_000)),
        "gmail": _email_provider_configuration_response(email, "gmail"),
        "outlook": _email_provider_configuration_response(email, "outlook"),
    }


def _email_provider_configuration_response(email: dict[str, Any], provider: str) -> dict[str, Any]:
    provider_config = email.get(provider, {})
    if not isinstance(provider_config, dict):
        provider_config = {}
    defaults = EMAIL_PROVIDER_DEFAULTS[provider]
    username_env = _email_provider_string(provider_config, defaults, "username_env")
    password_env = _email_provider_string(provider_config, defaults, "password_env")
    from_env = _email_provider_string(provider_config, defaults, "from_env")
    auth_method = str(provider_config.get("auth_method") or defaults.get("auth_method") or "app_password").strip()
    if auth_method not in {"app_password", "oauth2"}:
        auth_method = "app_password"
    oauth_client_secret_loaded = bool(load_email_secret(provider, "oauth_client_secret")) if provider == "gmail" else False
    oauth_refresh_token_loaded = bool(load_email_secret(provider, "oauth_refresh_token")) if provider == "gmail" else False
    oauth_email = _email_provider_string(provider_config, defaults, "oauth_email") if "oauth_email" in defaults else ""
    oauth_connected = provider == "gmail" and bool(
        auth_method == "oauth2"
        and provider_config.get("oauth_client_id")
        and oauth_email
        and oauth_client_secret_loaded
        and oauth_refresh_token_loaded
    )
    app_password_loaded = bool(os.environ.get(username_env, "").strip() and os.environ.get(password_env, "").strip())
    return {
        "enabled": bool(provider_config.get("enabled", defaults["enabled"])),
        "auth_method": auth_method,
        "imap_host": _email_provider_string(provider_config, defaults, "imap_host"),
        "imap_port": _coerce_int(provider_config.get("imap_port"), int(defaults["imap_port"])),
        "smtp_host": _email_provider_string(provider_config, defaults, "smtp_host"),
        "smtp_port": _coerce_int(provider_config.get("smtp_port"), int(defaults["smtp_port"])),
        "username_env": username_env,
        "password_env": password_env,
        "from_env": from_env,
        "oauth_client_id": _email_provider_string(provider_config, defaults, "oauth_client_id") if "oauth_client_id" in defaults else "",
        "oauth_redirect_uri": _email_provider_string(provider_config, defaults, "oauth_redirect_uri") if "oauth_redirect_uri" in defaults else "",
        "oauth_email": oauth_email,
        "oauth_from_address": _email_provider_string(provider_config, defaults, "oauth_from_address") if "oauth_from_address" in defaults else "",
        "oauth_client_secret_loaded": oauth_client_secret_loaded,
        "oauth_refresh_token_loaded": oauth_refresh_token_loaded,
        "oauth_connected": oauth_connected,
        "mailbox": _email_provider_string(provider_config, defaults, "mailbox"),
        "archive_mailbox": _email_provider_string(provider_config, defaults, "archive_mailbox"),
        "credentials_loaded": oauth_connected if auth_method == "oauth2" else app_password_loaded,
        "from_loaded": bool(os.environ.get(from_env, "").strip()),
    }


def _email_provider_string(
    provider_config: dict[str, Any],
    defaults: dict[str, Any],
    key: str,
) -> str:
    value = str(provider_config.get(key) or defaults[key]).strip()
    return value or str(defaults[key])


def _coerce_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _update_email_configuration(config: dict[str, Any], request: EmailConfigurationRequest) -> None:
    email = config.get("email")
    if not isinstance(email, dict):
        email = {}
        config["email"] = email
    if request.max_body_chars is not None:
        email["max_body_chars"] = max(1_000, min(int(request.max_body_chars), 200_000))
    for provider in ("gmail", "outlook"):
        provider_request = getattr(request, provider)
        if provider_request is not None:
            _update_email_provider_configuration(config, email, provider, provider_request)


def _update_email_provider_configuration(
    config: dict[str, Any],
    email: dict[str, Any],
    provider: str,
    request: EmailProviderConfigurationRequest,
) -> None:
    provider_config = email.get(provider)
    if not isinstance(provider_config, dict):
        provider_config = {}
        email[provider] = provider_config
    defaults = EMAIL_PROVIDER_DEFAULTS[provider]
    if request.enabled is not None:
        _set_email_provider_enabled(config, provider, bool(request.enabled))
    if request.auth_method is not None:
        provider_config["auth_method"] = _validated_email_auth_method(request.auth_method, provider)
    for key in ("imap_host", "smtp_host", "mailbox", "archive_mailbox"):
        value = getattr(request, key)
        if value is not None:
            provider_config[key] = value.strip() or defaults[key]
    for key in ("imap_port", "smtp_port"):
        value = getattr(request, key)
        if value is not None:
            provider_config[key] = max(1, min(int(value), 65_535))
    for key in ("username_env", "password_env", "from_env"):
        value = getattr(request, key)
        if value is not None:
            provider_config[key] = _validated_env_var_name(value, key)
    if provider == "gmail":
        if request.oauth_client_id is not None:
            provider_config["oauth_client_id"] = request.oauth_client_id.strip()
        if request.oauth_redirect_uri is not None:
            provider_config["oauth_redirect_uri"] = _validated_oauth_redirect_uri(request.oauth_redirect_uri)
        if request.oauth_email is not None:
            provider_config["oauth_email"] = request.oauth_email.strip()
        if request.oauth_from_address is not None:
            provider_config["oauth_from_address"] = request.oauth_from_address.strip()


def _set_email_provider_enabled(config: dict[str, Any], provider: str, enabled: bool) -> None:
    email = config.get("email")
    if not isinstance(email, dict):
        email = {}
        config["email"] = email
    provider_config = email.get(provider)
    if not isinstance(provider_config, dict):
        provider_config = {}
        email[provider] = provider_config
    provider_config["enabled"] = bool(enabled)
    tools = config.setdefault("tools", {})
    for tool in _email_tool_names(provider):
        tools.setdefault(tool, {})["enabled"] = bool(enabled)
    if enabled:
        permissions = config.setdefault("permissions", {})
        granted = permissions.setdefault("granted", [])
        if not isinstance(granted, list):
            granted = []
            permissions["granted"] = granted
        existing = {str(item) for item in granted}
        for permission in _email_permission_names(provider):
            if permission not in existing:
                granted.append(permission)
                existing.add(permission)


def _email_tool_names(provider: str) -> tuple[str, ...]:
    return (
        f"{provider}.search",
        f"{provider}.read_thread",
        f"{provider}.summarize_inbox",
        f"{provider}.create_draft",
        f"{provider}.reply_draft",
        f"{provider}.send_draft",
        f"{provider}.archive",
        *((f"{provider}.label",) if provider == "gmail" else ()),
    )


def _email_permission_names(provider: str) -> tuple[str, ...]:
    return (
        f"{provider}.readonly",
        f"{provider}.compose",
        f"{provider}.send",
        f"{provider}.modify",
    )


def _validated_env_var_name(value: str, label: str) -> str:
    env_var = value.strip()
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", env_var):
        raise HTTPException(
            status_code=400,
            detail=f"{label} must be a valid environment variable name.",
        )
    return env_var


def _validated_email_auth_method(value: str, provider: str) -> str:
    auth_method = value.strip().lower()
    if auth_method not in {"app_password", "oauth2"}:
        raise HTTPException(status_code=400, detail="auth_method must be app_password or oauth2.")
    if auth_method == "oauth2" and provider != "gmail":
        raise HTTPException(status_code=400, detail="OAuth2 is currently available for Gmail only.")
    return auth_method


def _validated_oauth_redirect_uri(value: str) -> str:
    redirect_uri = value.strip() or DEFAULT_GMAIL_OAUTH_REDIRECT_URI
    parsed = urllib.parse.urlparse(redirect_uri)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise HTTPException(status_code=400, detail="oauth_redirect_uri must be an absolute http(s) URL.")
    return redirect_uri


def _configure_gmail_oauth(
    config: dict[str, Any],
    request: GmailOAuthStartRequest,
    *,
    require_ready_fields: bool = True,
) -> None:
    client_id = request.client_id.strip()
    email = request.email.strip()
    if require_ready_fields and not client_id:
        raise HTTPException(status_code=400, detail="Google OAuth client ID is required.")
    if require_ready_fields and not email:
        raise HTTPException(status_code=400, detail="Gmail address is required.")
    _set_email_provider_enabled(config, "gmail", True)
    provider_config = config.setdefault("email", {}).setdefault("gmail", {})
    if not isinstance(provider_config, dict):
        provider_config = {}
        config["email"]["gmail"] = provider_config
    provider_config["auth_method"] = "oauth2"
    if client_id:
        provider_config["oauth_client_id"] = client_id
    provider_config["oauth_redirect_uri"] = _validated_oauth_redirect_uri(request.redirect_uri or DEFAULT_GMAIL_OAUTH_REDIRECT_URI)
    if email:
        provider_config["oauth_email"] = email
    if request.from_address is not None:
        provider_config["oauth_from_address"] = request.from_address.strip()


def _disconnect_gmail_oauth(config: dict[str, Any]) -> None:
    email = config.setdefault("email", {})
    provider_config = email.setdefault("gmail", {})
    if not isinstance(provider_config, dict):
        provider_config = {}
        email["gmail"] = provider_config
    provider_config["auth_method"] = "app_password"


def _oauth_callback_page(title: str, detail: str, *, success: bool) -> HTMLResponse:
    color = "#41d0a2" if success else "#e35e4f"
    safe_title = html.escape(title)
    safe_detail = html.escape(detail)
    return HTMLResponse(
        f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{safe_title}</title>
    <style>
      body {{ margin: 0; background: #080b0f; color: #d9fff1; font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
      main {{ max-width: 720px; margin: 12vh auto; border: 1px solid #24323d; border-radius: 8px; background: #111820; padding: 28px; }}
      h1 {{ margin: 0 0 12px; color: {color}; }}
      p {{ color: #9db1aa; line-height: 1.55; }}
    </style>
  </head>
  <body><main><h1>{safe_title}</h1><p>{safe_detail}</p></main></body>
</html>"""
    )


def _load_email_credentials(config: dict[str, Any], request: EmailCredentialsRequest) -> dict[str, Any]:
    provider = request.provider.strip().lower()
    if provider not in EMAIL_PROVIDER_DEFAULTS:
        raise HTTPException(status_code=400, detail="provider must be gmail or outlook.")
    username = request.username.strip()
    app_password = request.app_password.strip()
    if not username:
        raise HTTPException(status_code=400, detail="Email username is required.")
    if not app_password:
        raise HTTPException(status_code=400, detail="Email app password is required.")
    provider_config = _email_configuration_response(config)[provider]
    config.setdefault("email", {}).setdefault(provider, {})["auth_method"] = "app_password"
    username_env = provider_config["username_env"]
    password_env = provider_config["password_env"]
    from_env = provider_config["from_env"]
    os.environ[username_env] = username
    os.environ[password_env] = app_password
    from_address = (request.from_address or "").strip()
    if from_address:
        os.environ[from_env] = from_address
    return {
        "status": "ok",
        "message": (
            f"{provider.title()} credentials loaded into this API process environment. "
            "They are not written to the agent config file."
        ),
        "data": {
            "provider": provider,
            "username_env": username_env,
            "password_env": password_env,
            "from_env": from_env,
            "credentials_loaded": True,
            "from_loaded": bool(from_address or os.environ.get(from_env, "").strip()),
        },
    }


def _update_agent_configuration(config: dict[str, Any], request: AgentConfigurationRequest) -> None:
    if request.downloads_root is not None:
        value = request.downloads_root.strip()
        if value:
            root = Path(value).expanduser()
            if not root.is_absolute():
                raise HTTPException(status_code=400, detail="downloads_root must be an absolute path or empty.")
            root.mkdir(parents=True, exist_ok=True)
            config.setdefault("storage", {})["downloads_root"] = str(root.resolve())
        else:
            config.setdefault("storage", {})["downloads_root"] = ""
    setup = config.setdefault("setup", {})
    for key, value in {
        "agent_name": request.agent_name,
        "user_name": request.user_name,
        "preferred_language": request.preferred_language,
    }.items():
        if value is not None:
            setup[key] = value.strip()
    llm = config.setdefault("llm", {})
    if request.response_language is not None:
        llm["response_language"] = request.response_language.strip() or "auto"
    if request.chat_max_tokens is not None:
        llm["chat_max_tokens"] = max(128, min(int(request.chat_max_tokens), 8192))
    if request.planner_max_tokens is not None:
        llm["planner_max_tokens"] = max(128, min(int(request.planner_max_tokens), 8192))
    if request.synthesis_max_tokens is not None:
        llm["synthesis_max_tokens"] = max(128, min(int(request.synthesis_max_tokens), 8192))
    if request.repair_max_tokens is not None:
        llm["repair_max_tokens"] = max(64, min(int(request.repair_max_tokens), 2048))
    if request.chat_history_turns is not None:
        llm["chat_history_turns"] = max(1, min(int(request.chat_history_turns), 200))
    if request.chat_history_char_limit is not None:
        llm["chat_history_char_limit"] = max(500, min(int(request.chat_history_char_limit), 100_000))
    if request.planner_history_turns is not None:
        llm["planner_history_turns"] = max(0, min(int(request.planner_history_turns), 50))
    if request.planner_history_char_limit is not None:
        llm["planner_history_char_limit"] = max(0, min(int(request.planner_history_char_limit), 50_000))
    if request.planner_temperature is not None:
        llm["planner_temperature"] = max(0.0, min(float(request.planner_temperature), 2.0))
    if request.planner_think is not None:
        llm["planner_think"] = bool(request.planner_think)
    prompts = llm.setdefault("system_prompts", {})
    if request.chat_system_prompt is not None:
        prompts["chat"] = _stored_system_prompt(request.chat_system_prompt, CHAT_PROMPT)
    if request.planner_system_prompt is not None:
        prompts["planner"] = _stored_system_prompt(request.planner_system_prompt, SYSTEM_PROMPT)
    if request.answer_system_prompt is not None:
        prompts["answer"] = _stored_system_prompt(request.answer_system_prompt, ANSWER_PROMPT)
    if request.send_chat_history_to_cloud is not None:
        config.setdefault("privacy", {})["send_chat_history_to_cloud"] = bool(request.send_chat_history_to_cloud)
    browser = config.setdefault("browser", {})
    if request.browser_timeout_seconds is not None:
        browser["timeout_seconds"] = max(1, min(int(request.browser_timeout_seconds), 120))
    if request.browser_max_response_bytes is not None:
        browser["max_response_bytes"] = max(100_000, min(int(request.browser_max_response_bytes), 20_000_000))
    if request.browser_max_text_chars is not None:
        browser["max_text_chars"] = max(500, min(int(request.browser_max_text_chars), 1_000_000))
    if request.approval_required_at_risk is not None:
        config.setdefault("permissions", {})["approval_required_at_risk"] = max(
            0,
            min(int(request.approval_required_at_risk), 5),
        )
    if request.email is not None:
        _update_email_configuration(config, request.email)


def _system_prompts_response(llm: dict[str, Any]) -> dict[str, Any]:
    prompts = llm.get("system_prompts", {})
    if not isinstance(prompts, dict):
        prompts = {}
    chat_custom = str(prompts.get("chat") or "")
    planner_custom = str(prompts.get("planner") or "")
    answer_custom = str(prompts.get("answer") or "")
    return {
        "chat": {
            "default": CHAT_PROMPT,
            "custom": chat_custom,
            "effective": chat_custom.strip() or CHAT_PROMPT,
            "customized": bool(chat_custom.strip()),
        },
        "planner": {
            "default": SYSTEM_PROMPT,
            "custom": planner_custom,
            "effective": planner_custom.strip() or SYSTEM_PROMPT,
            "customized": bool(planner_custom.strip()),
        },
        "answer": {
            "default": ANSWER_PROMPT,
            "custom": answer_custom,
            "effective": answer_custom.strip() or ANSWER_PROMPT,
            "customized": bool(answer_custom.strip()),
        },
    }


def _stored_system_prompt(value: str, default: str) -> str:
    stripped = value.strip()
    return "" if stripped == default.strip() else stripped


def _workspace_files_response(paths: AppPaths, config: dict[str, Any]) -> dict[str, Any]:
    paths.ensure()
    files = _list_workspace_files(paths, config)
    grouped_counts = {name: 0 for name in WORKSPACE_FILE_ROOTS}
    for item in files:
        folder = str(item["folder"])
        grouped_counts[folder] = grouped_counts.get(folder, 0) + 1
    storage_root = downloads_root_from_config(config, default_root=paths.workspace)
    return {
        "workspace": str(storage_root),
        "roots": [
            {
                "name": name,
                "label": label,
                "path": str(storage_root / name),
                "exists": (storage_root / name).exists(),
                "count": grouped_counts.get(name, 0),
            }
            for name, label in WORKSPACE_FILE_ROOTS.items()
        ],
        "files": files,
    }


def _list_workspace_files(paths: AppPaths, config: dict[str, Any]) -> list[dict[str, Any]]:
    storage_root = downloads_root_from_config(config, default_root=paths.workspace)
    items: list[dict[str, Any]] = []
    for folder in WORKSPACE_FILE_ROOTS:
        folder_path = (storage_root / folder).resolve()
        if not folder_path.is_dir():
            continue
        for file_path in folder_path.rglob("*"):
            try:
                resolved = file_path.resolve()
                resolved.relative_to(folder_path)
            except (OSError, ValueError):
                continue
            if resolved.is_file():
                items.append(_workspace_file_item(paths, config, resolved))
    items.sort(key=lambda item: str(item["modified_at"]), reverse=True)
    return items[:500]


def _workspace_file_item(paths: AppPaths, config: dict[str, Any], file_path: Path) -> dict[str, Any]:
    storage_root = downloads_root_from_config(config, default_root=paths.workspace)
    resolved = file_path.resolve()
    relative_path = resolved.relative_to(storage_root).as_posix()
    stat = resolved.stat()
    folder = relative_path.split("/", 1)[0]
    content_type, _ = mimetypes.guess_type(resolved.name)
    if resolved.suffix.lower() in {".md", ".markdown"}:
        content_type = "text/markdown"
    return {
        "path": relative_path,
        "name": resolved.name,
        "folder": folder,
        "label": WORKSPACE_FILE_ROOTS.get(folder, folder),
        "size": stat.st_size,
        "modified_at": datetime.fromtimestamp(stat.st_mtime, timezone.utc).isoformat(timespec="seconds"),
        "content_type": content_type or "application/octet-stream",
        "previewable": _is_previewable_workspace_file(resolved),
    }


def _resolve_workspace_file(paths: AppPaths, config: dict[str, Any], raw_path: str) -> Path:
    requested = Path(raw_path)
    if requested.is_absolute():
        raise HTTPException(status_code=400, detail="Workspace file path must be relative.")
    storage_root = downloads_root_from_config(config, default_root=paths.workspace)
    candidate = (storage_root / requested).resolve()
    allowed = False
    for folder in WORKSPACE_FILE_ROOTS:
        root = (storage_root / folder).resolve()
        try:
            candidate.relative_to(root)
            allowed = True
            break
        except ValueError:
            continue
    if not allowed:
        raise HTTPException(status_code=400, detail="Workspace file path is not in an allowed download folder.")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail="Workspace file not found.")
    return candidate


def _is_previewable_workspace_file(file_path: Path) -> bool:
    content_type, _ = mimetypes.guess_type(file_path.name)
    if content_type and content_type.startswith("text/"):
        return True
    return file_path.suffix.lower() in WORKSPACE_TEXT_SUFFIXES


def _read_workspace_text_preview(file_path: Path) -> tuple[str, bool]:
    body = file_path.read_bytes()
    truncated = len(body) > WORKSPACE_MAX_PREVIEW_BYTES
    if truncated:
        body = body[:WORKSPACE_MAX_PREVIEW_BYTES]
    return body.decode("utf-8", errors="replace"), truncated


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
        if name in {"gmail", "outlook"}:
            email_status = _email_connector_status(config, name, enabled_tools)
            status = email_status["status"]
            detail = email_status["detail"]
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


def _email_connector_status(
    config: dict[str, Any],
    provider: str,
    enabled_tools: list[str],
) -> dict[str, str]:
    defaults = EMAIL_PROVIDER_DEFAULTS[provider]
    email_config = config.get("email", {})
    provider_config = email_config.get(provider, {}) if isinstance(email_config, dict) else {}
    if not isinstance(provider_config, dict):
        provider_config = {}
    username_env = str(provider_config.get("username_env") or defaults["username_env"])
    password_env = str(provider_config.get("password_env") or defaults["password_env"])
    enabled = bool(provider_config.get("enabled", False))
    auth_method = str(provider_config.get("auth_method") or defaults.get("auth_method") or "app_password")
    if provider == "gmail" and auth_method == "oauth2":
        has_oauth = bool(
            provider_config.get("oauth_client_id")
            and provider_config.get("oauth_email")
            and load_email_secret("gmail", "oauth_client_secret")
            and load_email_secret("gmail", "oauth_refresh_token")
        )
        if enabled and has_oauth:
            return {
                "status": "configured" if enabled_tools else "disabled",
                "detail": "Gmail OAuth2 connector is configured through the local secret store.",
            }
        return {
            "status": "not_configured",
            "detail": "Gmail OAuth2 needs a client ID, client secret, Gmail address, and completed Google authorization.",
        }
    has_env = bool(os.environ.get(username_env) and os.environ.get(password_env))
    if enabled and has_env:
        return {
            "status": "configured" if enabled_tools else "disabled",
            "detail": f"{provider.title()} IMAP/SMTP connector is configured through environment variables.",
        }
    return {
        "status": "not_configured",
        "detail": (
            f"{provider.title()} uses IMAP/SMTP. Enable email.{provider}.enabled and set "
            f"{username_env} plus {password_env} in the API process environment."
        ),
    }


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
        "allow_safe_workspace_commands": bool(terminal.get("allow_safe_workspace_commands", True)),
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
    terminal.setdefault("allow_safe_workspace_commands", True)
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
    manager = WorkspaceManager.from_config(config, fallback_workspace=AppPaths.default().workspace)
    return str(manager.current_workspace)


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
            try:
                root = WorkspaceManager.from_config(
                    config,
                    fallback_workspace=AppPaths.default().workspace,
                ).validate_workspace(value)
            except WorkspaceError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            terminal["workspace_root"] = str(root)
            workspace = config.setdefault("workspace", {})
            workspace["current_path"] = str(root)
            workspace.setdefault("default_path", str(root))
        else:
            terminal["workspace_root"] = ""
    if request.auto_approve_allowlisted is not None:
        terminal["auto_approve_allowlisted"] = bool(request.auto_approve_allowlisted)
    if request.allow_safe_workspace_commands is not None:
        terminal["allow_safe_workspace_commands"] = bool(request.allow_safe_workspace_commands)
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
