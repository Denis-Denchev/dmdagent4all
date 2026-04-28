from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from dmdagent4all.app_paths import AppPaths


DEFAULT_CONFIG: dict[str, Any] = {
    "llm": {
        "provider": "ollama",
        "model": "qwen3:8b",
        "mode": "fast",
        "response_language": "auto",
    },
    "privacy": {
        "local_first": True,
        "cloud_models_disabled_by_default": True,
        "never_send_email_body_to_cloud": True,
        "never_send_calendar_details_to_cloud": True,
        "redact_secrets": True,
        "require_approval_for_cloud_context": True,
    },
    "terminal": {
        "enabled": False,
        "mode": "allowlist",
        "workspace_only": True,
        "allowed_commands": [
            ["pwd"],
            ["ls"],
            ["git", "status"],
            ["git", "diff"],
            ["npm", "test"],
            ["pytest"],
        ],
    },
    "browser": {
        "enabled": False,
        "isolated_profile": True,
        "downloads_to_workspace": True,
        "approval_required_for_submit": True,
    },
    "tools": {},
    "permissions": {
        "granted": [],
        "approval_required_at_risk": 3,
    },
    "interfaces": {
        "ui_language": "en",
        "telegram": {
            "enabled": False,
            "allowed_user_ids": [],
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


def deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            base[key] = deep_merge(base[key], value)
        else:
            base[key] = value
    return base
