from __future__ import annotations

import shutil
import subprocess
import sys


KEYCHAIN_SERVICE = "dmdcore"


def macos_keychain_available() -> bool:
    return sys.platform == "darwin" and shutil.which("security") is not None


def get_macos_keychain_secret(account: str) -> str | None:
    if not macos_keychain_available():
        return None
    try:
        completed = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                account,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=2,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    token = completed.stdout.rstrip("\n")
    return token or None


def set_macos_keychain_secret(account: str, value: str) -> bool:
    if not macos_keychain_available():
        return False
    try:
        completed = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-U",
                "-a",
                account,
                "-s",
                KEYCHAIN_SERVICE,
                "-w",
                value,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def delete_macos_keychain_secret(account: str) -> bool:
    if not macos_keychain_available():
        return False
    try:
        completed = subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-a",
                account,
                "-s",
                KEYCHAIN_SERVICE,
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0
