from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any

from dmdcore.secrets import (
    delete_local_secret,
    delete_macos_keychain_secret,
    get_local_secret,
    get_macos_keychain_secret,
    set_local_secret,
    set_macos_keychain_secret,
)
from dmdcore.security import redact_text


GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GMAIL_OAUTH_SCOPE = "https://mail.google.com/"
DEFAULT_GMAIL_OAUTH_REDIRECT_URI = "http://127.0.0.1:8765/v1/email/oauth/google/callback"

GMAIL_OAUTH_CLIENT_SECRET_ENV = "DMDCORE_GMAIL_OAUTH_CLIENT_SECRET"
GMAIL_OAUTH_REFRESH_TOKEN_ENV = "DMDCORE_GMAIL_OAUTH_REFRESH_TOKEN"


def email_secret_account(provider: str, name: str) -> str:
    return f"email.{provider}.{name}"


def load_email_secret(provider: str, name: str) -> str | None:
    env_name = _secret_env_name(provider, name)
    if env_name:
        env_value = os.environ.get(env_name, "").strip()
        if env_value:
            return env_value
    account = email_secret_account(provider, name)
    return get_macos_keychain_secret(account) or get_local_secret(account)


def store_email_secret(provider: str, name: str, value: str) -> str:
    secret = value.strip()
    if not secret:
        raise ValueError("Secret value is required.")
    account = email_secret_account(provider, name)
    if set_macos_keychain_secret(account, secret):
        return "macos_keychain"
    if set_local_secret(account, secret):
        return "local_secret_store"
    raise ValueError("Could not store secret.")


def delete_email_secret(provider: str, name: str) -> bool:
    account = email_secret_account(provider, name)
    return delete_macos_keychain_secret(account) or delete_local_secret(account)


def google_authorization_url(*, client_id: str, redirect_uri: str, state: str) -> str:
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": GMAIL_OAUTH_SCOPE,
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{GOOGLE_AUTH_URL}?{urllib.parse.urlencode(params)}"


def exchange_google_code(
    *,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    code: str,
) -> dict[str, Any]:
    return _google_token_request(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "redirect_uri": redirect_uri,
            "code": code,
            "grant_type": "authorization_code",
        }
    )


def refresh_google_access_token(
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> str:
    token = _google_token_request(
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
    )
    access_token = str(token.get("access_token") or "").strip()
    if not access_token:
        raise ValueError("Google did not return an access token.")
    return access_token


def _google_token_request(payload: dict[str, str]) -> dict[str, Any]:
    data = urllib.parse.urlencode(payload).encode("utf-8")
    request = urllib.request.Request(
        GOOGLE_TOKEN_URL,
        data=data,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise ValueError(f"Google OAuth token request failed: {redact_text(detail)}") from exc
    except urllib.error.URLError as exc:
        raise ValueError(f"Google OAuth token request failed: {redact_text(str(exc.reason))}") from exc
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("Google OAuth token response was not an object.")
    return parsed


def _secret_env_name(provider: str, name: str) -> str | None:
    if provider == "gmail" and name == "oauth_client_secret":
        return GMAIL_OAUTH_CLIENT_SECRET_ENV
    if provider == "gmail" and name == "oauth_refresh_token":
        return GMAIL_OAUTH_REFRESH_TOKEN_ENV
    return None
