from __future__ import annotations

import json
import os
from pathlib import Path

from dmdagent4all.app_paths import AppPaths


def get_local_secret(account: str) -> str | None:
    data = _read_store()
    value = data.get(account)
    return value if isinstance(value, str) and value else None


def set_local_secret(account: str, value: str) -> bool:
    if not account or not value:
        return False
    data = _read_store()
    data[account] = value
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def delete_local_secret(account: str) -> bool:
    if not account:
        return False
    data = _read_store()
    if account not in data:
        return False
    data.pop(account, None)
    path = _store_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".tmp")
    tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    try:
        os.chmod(tmp_path, 0o600)
    except OSError:
        pass
    os.replace(tmp_path, path)
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass
    return True


def _read_store() -> dict[str, str]:
    path = _store_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(key): str(value) for key, value in raw.items() if isinstance(value, str)}


def _store_path() -> Path:
    return AppPaths.default().root / "secrets.json"
