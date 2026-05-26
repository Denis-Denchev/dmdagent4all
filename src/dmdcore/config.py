from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from threading import RLock
from typing import Any, Callable

import yaml

from dmdcore.app_paths import AppPaths


_CONFIG_LOCK = RLock()


DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model": "qwen3:8b",
        "planner_model": None,
        "base_url": "http://localhost:11434",
        "api_key_env": None,
        "mode": "fast",
        "response_language": "auto",
        "chat_max_tokens": 1024,
        "planner_max_tokens": 192,
        "synthesis_max_tokens": 1024,
        "repair_max_tokens": 256,
        "chat_history_turns": 24,
        "chat_history_char_limit": 12000,
        "planner_history_turns": 4,
        "planner_history_char_limit": 3000,
        "planner_temperature": 0.0,
        "planner_think": False,
        "system_prompts": {
            "planner": "",
            "answer": "",
        },
    },
    "privacy": {
        "local_first": True,
        "cloud_models_disabled_by_default": True,
        "never_send_email_body_to_cloud": True,
        "never_send_calendar_details_to_cloud": True,
        "redact_secrets": True,
        "require_approval_for_cloud_context": True,
        "send_chat_history_to_cloud": True,
    },
    "terminal": {
        "enabled": False,
        "mode": "allowlist",
        "workspace_only": True,
        "timeout_seconds": 30,
        "max_output_chars": 20000,
        "auto_approve_allowlisted": False,
        "allow_safe_workspace_commands": True,
        "workspace_root": "",
        "allowed_commands": [
            ["pwd"],
            ["ls"],
            ["git", "status"],
            ["git", "diff"],
            ["cat", "README.md"],
            ["cat", "readme.md"],
            ["npm", "test"],
            ["pytest"],
        ],
    },
    "workspace": {
        "default_path": "",
        "current_path": "",
        "allowed_roots": [],
        "blocked_paths": [
            "~/.ssh",
            "~/.aws",
            "~/.config/gcloud",
            "~/.kube",
            "/etc",
            "/var",
            "/private",
            "/Library",
            "/System",
        ],
    },
    "browser": {
        "enabled": False,
        "isolated_profile": True,
        "downloads_to_workspace": True,
        "approval_required_for_submit": True,
        "timeout_seconds": 15,
        "max_response_bytes": 1000000,
        "max_text_chars": 12000,
    },
    "email": {
        "max_body_chars": 20000,
        "gmail": {
            "enabled": False,
            "auth_method": "app_password",
            "imap_host": "imap.gmail.com",
            "imap_port": 993,
            "smtp_host": "smtp.gmail.com",
            "smtp_port": 587,
            "username_env": "DMDCORE_GMAIL_USERNAME",
            "password_env": "DMDCORE_GMAIL_APP_PASSWORD",
            "from_env": "DMDCORE_GMAIL_FROM",
            "oauth_client_id": "",
            "oauth_redirect_uri": "http://127.0.0.1:8765/v1/email/oauth/google/callback",
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
            "username_env": "DMDCORE_OUTLOOK_USERNAME",
            "password_env": "DMDCORE_OUTLOOK_APP_PASSWORD",
            "from_env": "DMDCORE_OUTLOOK_FROM",
            "mailbox": "INBOX",
            "archive_mailbox": "Archive",
        },
    },
    "storage": {
        "downloads_root": "",
    },
    "tools": {},
    "openai_usage": {
        "limit_usd": None,
    },
    "permissions": {
        "granted": [],
        "approval_required_at_risk": 3,
    },
    "runtime": {
        "emergency_stop": {
            "active": False,
            "triggered_at": "",
            "reason": "",
        },
    },
    "setup": {
        "completed": False,
        "agent_name": "DMD Agent",
        "user_name": "",
        "preferred_language": "auto",
    },
    "interfaces": {
        "ui_language": "en",
        "telegram": {
            "enabled": False,
            "allowed_user_ids": [],
            "bot_token_env": "DMDCORE_TELEGRAM_BOT_TOKEN",
            "polling_timeout_seconds": 30,
        },
        "whatsapp": {
            "enabled": False,
            "allowed_user_ids": [],
        },
    },
}


def load_config(path: Path | None = None) -> dict[str, Any]:
    config_path = path or AppPaths.default().config
    if not config_path.exists():
        return deepcopy(DEFAULT_CONFIG)

    loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"Config must be a mapping: {config_path}")

    return deep_merge(deepcopy(DEFAULT_CONFIG), loaded)


def write_default_config(path: Path | None = None) -> Path:
    paths = AppPaths.default()
    paths.ensure()
    config_path = path or paths.config
    with _CONFIG_LOCK:
        if not config_path.exists():
            config_path.write_text(
                yaml.safe_dump(DEFAULT_CONFIG, sort_keys=False),
                encoding="utf-8",
            )
    return config_path


def save_config(config: dict[str, Any], path: Path | None = None) -> Path:
    paths = AppPaths.default()
    paths.ensure()
    config_path = path or paths.config
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path


def update_config(
    mutator: Callable[[dict[str, Any]], None],
    path: Path | None = None,
) -> dict[str, Any]:
    paths = AppPaths.default()
    paths.ensure()
    config_path = path or paths.config
    with _CONFIG_LOCK:
        config = load_config(config_path)
        mutator(config)
        save_config(config, config_path)
    return config


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base
