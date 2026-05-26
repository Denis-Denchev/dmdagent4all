from __future__ import annotations

import imaplib
import json
import os
import smtplib
import uuid
import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from email import policy
from email.message import EmailMessage, Message
from email.parser import BytesParser
from email.utils import getaddresses, parseaddr
from pathlib import Path
from typing import Any, Callable

from dmdcore.email_oauth import load_email_secret, refresh_google_access_token
from dmdcore.security import redact_text
from dmdcore.tools.base import ToolRuntimeContext


Handler = Callable[[dict[str, Any], ToolRuntimeContext], dict[str, Any]]


PROVIDER_DEFAULTS: dict[str, dict[str, Any]] = {
    "gmail": {
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
}


@dataclass(frozen=True)
class EmailSettings:
    provider: str
    imap_host: str
    imap_port: int
    smtp_host: str
    smtp_port: int
    username: str
    password: str
    from_addr: str
    auth_method: str
    oauth_access_token: str
    mailbox: str
    archive_mailbox: str
    max_body_chars: int


def make_email_handler(provider: str, action: str) -> Handler:
    def handler(args: dict[str, Any], context: ToolRuntimeContext) -> dict[str, Any]:
        settings = _settings(provider, context)
        if settings is None:
            return _not_configured(provider, context)
        if action == "search":
            return _search(args, settings)
        if action == "read_thread":
            return _read_thread(args, settings)
        if action == "summarize_inbox":
            return _summarize_inbox(args, settings)
        if action == "create_draft":
            return _create_draft(args, context, settings)
        if action == "reply_draft":
            return _reply_draft(args, context, settings)
        if action == "send_draft":
            return _send_draft(args, context, settings)
        if action == "archive":
            return _archive(args, settings)
        if action == "label":
            return _label(args, settings)
        raise ValueError(f"Unknown email action: {action}")

    return handler


def test_email_connection(provider: str, context: ToolRuntimeContext) -> dict[str, Any]:
    if provider not in PROVIDER_DEFAULTS:
        raise ValueError("provider must be gmail or outlook.")
    try:
        settings = _settings(provider, context)
    except Exception as exc:
        return {
            "provider": provider,
            "status": "authentication_failed",
            "configured": False,
            "imap": {"ok": False, "detail": "Not tested because credentials could not be loaded."},
            "smtp": {"ok": False, "detail": "Not tested because credentials could not be loaded."},
            "message": _connection_error_message(provider, str(exc)),
            "detail": redact_text(str(exc)),
        }
    if settings is None:
        return _not_configured(provider, context)

    imap_result = _test_imap(settings)
    smtp_result = _test_smtp(settings)
    ok = bool(imap_result["ok"] and smtp_result["ok"])
    failed_detail = _first_failed_detail(imap_result, smtp_result)
    return {
        "provider": provider,
        "status": "ok" if ok else "connection_failed",
        "configured": True,
        "auth_method": settings.auth_method,
        "mailbox": settings.mailbox,
        "imap": imap_result,
        "smtp": smtp_result,
        "message": (
            f"{provider.title()} IMAP and SMTP are working."
            if ok
            else _connection_error_message(provider, failed_detail)
        ),
    }


def _settings(provider: str, context: ToolRuntimeContext) -> EmailSettings | None:
    defaults = PROVIDER_DEFAULTS[provider]
    email_config = context.config.get("email", {})
    if not isinstance(email_config, dict):
        email_config = {}
    provider_config = email_config.get(provider, {})
    if not isinstance(provider_config, dict):
        provider_config = {}
    if not bool(provider_config.get("enabled", False)):
        return None

    auth_method = str(provider_config.get("auth_method") or defaults.get("auth_method") or "app_password").strip()
    if auth_method == "oauth2":
        if provider != "gmail":
            return None
        username = _normalize_email_login_value(str(provider_config.get("oauth_email") or ""))
        client_id = str(provider_config.get("oauth_client_id") or "").strip()
        client_secret = load_email_secret(provider, "oauth_client_secret") or ""
        refresh_token = load_email_secret(provider, "oauth_refresh_token") or ""
        if not username or not client_id or not client_secret or not refresh_token:
            return None
        access_token = refresh_google_access_token(
            client_id=client_id,
            client_secret=client_secret,
            refresh_token=refresh_token,
        )
        max_body_chars = int(email_config.get("max_body_chars", 20_000))
        return EmailSettings(
            provider=provider,
            imap_host=str(provider_config.get("imap_host") or defaults["imap_host"]),
            imap_port=int(provider_config.get("imap_port") or defaults["imap_port"]),
            smtp_host=str(provider_config.get("smtp_host") or defaults["smtp_host"]),
            smtp_port=int(provider_config.get("smtp_port") or defaults["smtp_port"]),
            username=username,
            password="",
            from_addr=_sanitize_header_value(str(provider_config.get("oauth_from_address") or "")) or username,
            auth_method="oauth2",
            oauth_access_token=access_token,
            mailbox=str(provider_config.get("mailbox") or defaults["mailbox"]),
            archive_mailbox=str(provider_config.get("archive_mailbox") or defaults["archive_mailbox"]),
            max_body_chars=max_body_chars,
        )

    username_env = str(provider_config.get("username_env") or defaults["username_env"])
    password_env = str(provider_config.get("password_env") or defaults["password_env"])
    from_env = str(provider_config.get("from_env") or defaults["from_env"])
    username = _normalize_email_login_value(os.environ.get(username_env, ""))
    password = _normalize_app_password(os.environ.get(password_env, ""))
    from_addr = _sanitize_header_value(os.environ.get(from_env, "")) or username
    if not username or not password:
        return None

    max_body_chars = int(email_config.get("max_body_chars", 20_000))
    return EmailSettings(
        provider=provider,
        imap_host=str(provider_config.get("imap_host") or defaults["imap_host"]),
        imap_port=int(provider_config.get("imap_port") or defaults["imap_port"]),
        smtp_host=str(provider_config.get("smtp_host") or defaults["smtp_host"]),
        smtp_port=int(provider_config.get("smtp_port") or defaults["smtp_port"]),
        username=username,
        password=password,
        from_addr=from_addr,
        auth_method="app_password",
        oauth_access_token="",
        mailbox=str(provider_config.get("mailbox") or defaults["mailbox"]),
        archive_mailbox=str(provider_config.get("archive_mailbox") or defaults["archive_mailbox"]),
        max_body_chars=max_body_chars,
    )


def _not_configured(provider: str, context: ToolRuntimeContext) -> dict[str, Any]:
    defaults = PROVIDER_DEFAULTS[provider]
    configured = context.config.get("email", {}).get(provider, {})
    if isinstance(configured, dict) and configured.get("auth_method") == "oauth2":
        return {
            "status": "connector_not_configured",
            "connector": provider,
            "message": (
                "Gmail OAuth2 is not fully configured. For the simplest setup, switch Gmail "
                "Authentication to App password, then load your Gmail address and Google app password. "
                "If you keep OAuth2, add the Google client ID/client secret, complete authorization, "
                "and make sure a refresh token is stored."
            ),
            "required": ["oauth_client_id", "oauth_client_secret", "oauth_refresh_token", "oauth_email"],
        }
    username_env = str(configured.get("username_env") or defaults["username_env"]) if isinstance(configured, dict) else str(defaults["username_env"])
    password_env = str(configured.get("password_env") or defaults["password_env"]) if isinstance(configured, dict) else str(defaults["password_env"])
    return {
        "status": "connector_not_configured",
        "connector": provider,
        "message": (
            f"{provider.title()} email is not configured. Enable email.{provider}.enabled "
            f"and set {username_env} plus {password_env} in the API process environment."
        ),
        "required_env": [username_env, password_env],
    }


def _search(args: dict[str, Any], settings: EmailSettings) -> dict[str, Any]:
    query = str(args.get("query") or "").strip()
    limit = _bounded_int(args.get("limit"), default=10, minimum=1, maximum=50)
    mailbox = str(args.get("mailbox") or settings.mailbox)
    with _imap(settings) as client:
        _select(client, mailbox)
        ids = _imap_search(client, query)
        selected = list(reversed(ids[-limit:]))
        messages = [_message_summary(_fetch_message(client, uid), uid=uid) for uid in selected]
    return {
        "provider": settings.provider,
        "query": query,
        "mailbox": mailbox,
        "messages": messages,
        "count": len(messages),
    }


def _read_thread(args: dict[str, Any], settings: EmailSettings) -> dict[str, Any]:
    thread_id = str(args.get("thread_id") or args.get("message_id") or args.get("uid") or "").strip()
    latest = bool(args.get("latest", False))
    mailbox = str(args.get("mailbox") or settings.mailbox)
    with _imap(settings) as client:
        _select(client, mailbox)
        if not thread_id and latest:
            ids = _imap_search(client, "")
            if not ids:
                return {
                    "provider": settings.provider,
                    "thread_id": "",
                    "mailbox": mailbox,
                    "messages": [],
                    "threading": "imap_uid",
                }
            thread_id = ids[-1]
        if not thread_id:
            raise ValueError("read_thread requires thread_id/message uid.")
        message = _fetch_message(client, thread_id)
    parsed = _message_detail(message, uid=thread_id, max_body_chars=settings.max_body_chars)
    return {
        "provider": settings.provider,
        "thread_id": thread_id,
        "mailbox": mailbox,
        "messages": [parsed],
        "threading": "imap_uid",
    }


def _summarize_inbox(args: dict[str, Any], settings: EmailSettings) -> dict[str, Any]:
    result = _search(args, settings)
    summaries = []
    for message in result["messages"]:
        summaries.append(
            {
                "id": message["id"],
                "from": message["from"],
                "date": message["date"],
                "subject": message["subject"],
                "summary": message["snippet"],
            }
        )
    return {
        "provider": settings.provider,
        "query": result["query"],
        "count": result["count"],
        "summary": summaries,
    }


def _create_draft(args: dict[str, Any], context: ToolRuntimeContext, settings: EmailSettings) -> dict[str, Any]:
    draft = _draft_from_args(args, settings)
    return _store_draft(context, draft)


def _reply_draft(args: dict[str, Any], context: ToolRuntimeContext, settings: EmailSettings) -> dict[str, Any]:
    thread_id = str(args.get("thread_id") or args.get("message_id") or args.get("uid") or "").strip()
    body = str(args.get("body") or "").strip()
    if not thread_id:
        raise ValueError("reply_draft requires thread_id/message uid.")
    if not body:
        raise ValueError("reply_draft requires body.")
    original = _read_thread({"thread_id": thread_id, "mailbox": args.get("mailbox")}, settings)["messages"][0]
    to_addr = parseaddr(str(original.get("from") or ""))[1]
    if not to_addr:
        raise ValueError("Could not determine reply recipient.")
    subject = str(original.get("subject") or "")
    if not subject.casefold().startswith("re:"):
        subject = f"Re: {subject}"
    draft = {
        "provider": settings.provider,
        "from": settings.from_addr,
        "to": [to_addr],
        "cc": _address_list(args.get("cc")),
        "bcc": _address_list(args.get("bcc")),
        "subject": subject,
        "body": body,
        "reply_to_uid": thread_id,
        "in_reply_to": original.get("message_id", ""),
        "references": original.get("references", ""),
        "created_at": _now(),
        "status": "draft",
    }
    return _store_draft(context, draft)


def _send_draft(args: dict[str, Any], context: ToolRuntimeContext, settings: EmailSettings) -> dict[str, Any]:
    draft_id = str(args.get("draft_id") or "").strip()
    if not draft_id:
        raise ValueError("send_draft requires draft_id.")
    draft_path = _draft_path(context, settings.provider, draft_id)
    if not draft_path.exists():
        raise FileNotFoundError(f"Draft not found: {draft_id}")
    draft = json.loads(draft_path.read_text(encoding="utf-8"))
    if draft.get("provider") != settings.provider:
        raise ValueError("Draft provider does not match tool provider.")
    message = _email_message_from_draft(draft)
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as client:
            client.starttls()
            _smtp_login(client, settings)
            client.send_message(message)
    except smtplib.SMTPAuthenticationError as exc:
        return _smtp_authentication_failed(settings, draft_id, exc)
    except smtplib.SMTPException as exc:
        return {
            "provider": settings.provider,
            "draft_id": draft_id,
            "status": "smtp_failed",
            "sent": False,
            "message": f"{settings.provider.title()} SMTP failed before sending. Draft remains local.",
            "detail": redact_text(str(exc)),
        }
    except UnicodeEncodeError as exc:
        return _smtp_credential_encoding_failed(settings, draft_id, exc)
    draft["status"] = "sent"
    draft["sent_at"] = _now()
    draft_path.write_text(json.dumps(draft, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "provider": settings.provider,
        "draft_id": draft_id,
        "sent": True,
        "to": draft.get("to", []),
        "subject": draft.get("subject", ""),
    }


def _smtp_authentication_failed(
    settings: EmailSettings,
    draft_id: str,
    exc: smtplib.SMTPAuthenticationError,
) -> dict[str, Any]:
    if settings.provider == "outlook":
        message = (
            "Outlook SMTP authentication failed because Basic Authentication is disabled for this account or tenant. "
            "The draft remains local and was not sent. Use Gmail SMTP for now, or configure Outlook through Microsoft Graph/OAuth."
        )
    else:
        message = (
            f"{settings.provider.title()} SMTP authentication failed. "
            "Check that the account allows SMTP and that the app password is correct. The draft remains local and was not sent."
        )
    return {
        "provider": settings.provider,
        "draft_id": draft_id,
        "status": "authentication_failed",
        "sent": False,
        "message": message,
        "smtp_code": getattr(exc, "smtp_code", None),
        "smtp_error": redact_text(str(getattr(exc, "smtp_error", b""))),
    }


def _smtp_credential_encoding_failed(
    settings: EmailSettings,
    draft_id: str,
    exc: UnicodeEncodeError,
) -> dict[str, Any]:
    return {
        "provider": settings.provider,
        "draft_id": draft_id,
        "status": "authentication_failed",
        "sent": False,
        "message": (
            f"{settings.provider.title()} SMTP authentication failed before sending because the "
            "username or app password contains a non-ASCII character. Re-load the credentials; "
            "Google app passwords should be pasted as the 16-character token without spaces."
        ),
        "detail": redact_text(str(exc)),
    }


def _archive(args: dict[str, Any], settings: EmailSettings) -> dict[str, Any]:
    message_ids = _string_list(args.get("message_ids") or args.get("ids"))
    if not message_ids:
        raise ValueError("archive requires message_ids.")
    mailbox = str(args.get("mailbox") or settings.mailbox)
    with _imap(settings) as client:
        _select(client, mailbox)
        for uid in message_ids:
            status, _ = client.uid("COPY", uid, _quote_imap(settings.archive_mailbox))
            if status != "OK":
                raise ValueError(f"Could not copy message {uid} to archive mailbox.")
            client.uid("STORE", uid, "+FLAGS", r"(\Deleted)")
        client.expunge()
    return {
        "provider": settings.provider,
        "message_ids": message_ids,
        "archived": True,
        "archive_mailbox": settings.archive_mailbox,
    }


def _label(args: dict[str, Any], settings: EmailSettings) -> dict[str, Any]:
    labels = _string_list(args.get("labels"))
    message_ids = _string_list(args.get("message_ids") or args.get("ids"))
    if not message_ids:
        raise ValueError("label requires message_ids.")
    if not labels:
        raise ValueError("label requires labels.")
    with _imap(settings) as client:
        _select(client, settings.mailbox)
        for label in labels:
            client.create(label)
            for uid in message_ids:
                client.uid("COPY", uid, _quote_imap(label))
    return {
        "provider": settings.provider,
        "message_ids": message_ids,
        "labels": labels,
        "applied": True,
    }


def _imap(settings: EmailSettings) -> imaplib.IMAP4_SSL:
    client = imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port, timeout=30)
    if settings.auth_method == "oauth2":
        client.authenticate("XOAUTH2", lambda _: _xoauth2_string(settings))
    else:
        client.login(settings.username, settings.password)
    return client


def _smtp_login(client: smtplib.SMTP, settings: EmailSettings) -> None:
    if settings.auth_method != "oauth2":
        client.login(settings.username, settings.password)
        return
    auth = base64.b64encode(_xoauth2_string(settings)).decode("ascii")
    code, response = client.docmd("AUTH", f"XOAUTH2 {auth}")
    if code != 235:
        raise smtplib.SMTPAuthenticationError(code, response)


def _xoauth2_string(settings: EmailSettings) -> bytes:
    return f"user={settings.username}\x01auth=Bearer {settings.oauth_access_token}\x01\x01".encode("utf-8")


def _test_imap(settings: EmailSettings) -> dict[str, Any]:
    try:
        with _imap(settings) as client:
            _select(client, settings.mailbox)
    except imaplib.IMAP4.error as exc:
        return {"ok": False, "stage": "imap_login_or_select", "detail": _email_error_detail(exc)}
    except OSError as exc:
        return {"ok": False, "stage": "imap_connect", "detail": _email_error_detail(exc)}
    except Exception as exc:
        return {"ok": False, "stage": "imap", "detail": _email_error_detail(exc)}
    return {
        "ok": True,
        "stage": "imap_login_and_select",
        "detail": f"Connected to {settings.imap_host}:{settings.imap_port} and opened {settings.mailbox}.",
    }


def _test_smtp(settings: EmailSettings) -> dict[str, Any]:
    try:
        with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=30) as client:
            client.starttls()
            _smtp_login(client, settings)
    except smtplib.SMTPAuthenticationError as exc:
        return {
            "ok": False,
            "stage": "smtp_login",
            "detail": _email_error_detail(exc),
            "smtp_code": getattr(exc, "smtp_code", None),
        }
    except smtplib.SMTPException as exc:
        return {"ok": False, "stage": "smtp", "detail": _email_error_detail(exc)}
    except UnicodeEncodeError as exc:
        return {
            "ok": False,
            "stage": "smtp_login",
            "detail": (
                f"{settings.provider.title()} username or app password contains a non-ASCII character. "
                "Re-load the credentials; app passwords should not include spaces copied from the display."
            ),
            "encoding_error": redact_text(str(exc)),
        }
    except OSError as exc:
        return {"ok": False, "stage": "smtp_connect", "detail": _email_error_detail(exc)}
    except Exception as exc:
        return {"ok": False, "stage": "smtp", "detail": _email_error_detail(exc)}
    return {
        "ok": True,
        "stage": "smtp_login",
        "detail": f"Connected to {settings.smtp_host}:{settings.smtp_port} and authenticated.",
    }


def _email_error_detail(exc: BaseException) -> str:
    return redact_text(str(exc) or exc.__class__.__name__)


def _first_failed_detail(*results: dict[str, Any]) -> str:
    for result in results:
        if result.get("ok") is False:
            return str(result.get("detail") or "")
    return ""


def _connection_error_message(provider: str, detail: str) -> str:
    normalized = detail.casefold()
    if provider == "gmail":
        if "non-ascii" in normalized:
            return (
                "Gmail authentication failed because the loaded username or app password contains a non-ASCII "
                "character. Re-load the Gmail credentials and paste the app password as the 16-character token."
            )
        if any(marker in normalized for marker in {"invalid credentials", "application-specific password", "534", "535"}):
            return (
                "Gmail authentication failed. Use a Google app password, not your normal Google password, "
                "and make sure IMAP is enabled in Gmail settings."
            )
        return (
            "Gmail connection failed. Check that Gmail is set to App password mode, IMAP is enabled, "
            "and the app password is loaded in the running API process."
        )
    if provider == "outlook":
        if any(marker in normalized for marker in {"basic authentication", "5.7.139", "535"}):
            return (
                "Outlook authentication failed. This account or tenant may block IMAP/SMTP basic auth; "
                "Microsoft Graph OAuth is needed for that mailbox."
            )
        return "Outlook connection failed. Check IMAP/SMTP settings and whether the mailbox allows app-password access."
    return "Email connection failed."


def _select(client: imaplib.IMAP4_SSL, mailbox: str) -> None:
    status, _ = client.select(mailbox, readonly=False)
    if status != "OK":
        raise ValueError(f"Could not open mailbox: {mailbox}")


def _imap_search(client: imaplib.IMAP4_SSL, query: str) -> list[str]:
    criteria = _query_to_imap(query)
    status, data = client.uid("SEARCH", None, *criteria)
    if status != "OK" or not data:
        return []
    return [item.decode("ascii", errors="ignore") for item in data[0].split()]


def _query_to_imap(query: str) -> list[str]:
    if not query:
        return ["ALL"]
    lowered = query.casefold()
    if lowered.startswith("from:"):
        return ["FROM", _quote_imap(query[5:].strip())]
    if lowered.startswith("to:"):
        return ["TO", _quote_imap(query[3:].strip())]
    if lowered.startswith("subject:"):
        return ["SUBJECT", _quote_imap(query[8:].strip())]
    return ["TEXT", _quote_imap(query)]


def _quote_imap(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', r"\"")
    return f'"{escaped}"'


def _fetch_message(client: imaplib.IMAP4_SSL, uid: str) -> Message:
    status, data = client.uid("FETCH", uid, "(RFC822)")
    if status != "OK" or not data:
        raise FileNotFoundError(f"Message not found: {uid}")
    for item in data:
        if isinstance(item, tuple) and item[1]:
            return BytesParser(policy=policy.default).parsebytes(item[1])
    raise FileNotFoundError(f"Message not found: {uid}")


def _message_summary(message: Message, *, uid: str) -> dict[str, Any]:
    body = _message_body(message)
    return {
        "id": uid,
        "thread_id": uid,
        "message_id": str(message.get("Message-ID") or ""),
        "from": _header(message, "From"),
        "to": _header(message, "To"),
        "date": _header(message, "Date"),
        "subject": _header(message, "Subject"),
        "snippet": redact_text(_collapse_ws(body)[:500]),
    }


def _message_detail(message: Message, *, uid: str, max_body_chars: int) -> dict[str, Any]:
    body = _message_body(message)
    truncated = len(body) > max_body_chars
    if truncated:
        body = body[:max_body_chars]
    return {
        **_message_summary(message, uid=uid),
        "cc": _header(message, "Cc"),
        "in_reply_to": str(message.get("In-Reply-To") or ""),
        "references": str(message.get("References") or ""),
        "body": redact_text(body),
        "truncated": truncated,
    }


def _message_body(message: Message) -> str:
    if message.is_multipart():
        html_fallback = ""
        for part in message.walk():
            content_disposition = str(part.get("Content-Disposition") or "")
            if "attachment" in content_disposition.casefold():
                continue
            content_type = part.get_content_type()
            try:
                payload = part.get_content()
            except Exception:
                continue
            if content_type == "text/plain" and isinstance(payload, str):
                return payload.strip()
            if content_type == "text/html" and isinstance(payload, str) and not html_fallback:
                html_fallback = _html_to_text(payload)
        return html_fallback.strip()
    try:
        payload = message.get_content()
    except Exception:
        payload = message.get_payload(decode=True) or b""
        if isinstance(payload, bytes):
            return payload.decode("utf-8", errors="replace").strip()
    return str(payload).strip()


def _html_to_text(html: str) -> str:
    import re

    text = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    text = re.sub(r"(?s)<br\s*/?>", "\n", text)
    text = re.sub(r"(?s)</p>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    return _collapse_ws(text)


def _collapse_ws(value: str) -> str:
    return " ".join(value.split())


def _header(message: Message, name: str) -> str:
    return redact_text(str(message.get(name) or ""))


def _draft_from_args(args: dict[str, Any], settings: EmailSettings) -> dict[str, Any]:
    to = _address_list(args.get("to"))
    if not to:
        raise ValueError("create_draft requires to.")
    subject = _sanitize_header_value(str(args.get("subject") or ""))
    body = str(args.get("body") or "").strip()
    if not subject:
        raise ValueError("create_draft requires subject.")
    if not body:
        raise ValueError("create_draft requires body.")
    return {
        "provider": settings.provider,
        "from": settings.from_addr,
        "to": to,
        "cc": _address_list(args.get("cc")),
        "bcc": _address_list(args.get("bcc")),
        "subject": subject,
        "body": body,
        "created_at": _now(),
        "status": "draft",
    }


def _store_draft(context: ToolRuntimeContext, draft: dict[str, Any]) -> dict[str, Any]:
    draft_id = uuid.uuid4().hex
    draft["draft_id"] = draft_id
    path = _draft_path(context, str(draft["provider"]), draft_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(draft, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "provider": draft["provider"],
        "draft_id": draft_id,
        "draft_path": str(path),
        "to": draft.get("to", []),
        "subject": draft.get("subject", ""),
        "status": "draft",
    }


def _draft_path(context: ToolRuntimeContext, provider: str, draft_id: str) -> Path:
    safe_id = "".join(ch for ch in draft_id if ch.isalnum() or ch in {"-", "_"})
    if not safe_id:
        raise ValueError("Invalid draft_id.")
    return context.memory_root.parent / "email-drafts" / provider / f"{safe_id}.json"


def _email_message_from_draft(draft: dict[str, Any]) -> EmailMessage:
    message = EmailMessage()
    message["From"] = _sanitize_header_value(str(draft["from"]))
    message["To"] = ", ".join(_string_list(draft.get("to")))
    cc = _string_list(draft.get("cc"))
    bcc = _string_list(draft.get("bcc"))
    if cc:
        message["Cc"] = _sanitize_header_value(", ".join(cc))
    if bcc:
        message["Bcc"] = _sanitize_header_value(", ".join(bcc))
    message["Subject"] = _sanitize_header_value(str(draft.get("subject") or ""))
    if draft.get("in_reply_to"):
        message["In-Reply-To"] = str(draft["in_reply_to"])
    if draft.get("references"):
        message["References"] = str(draft["references"])
    message.set_content(str(draft.get("body") or ""))
    return message


def _sanitize_header_value(value: str) -> str:
    return " ".join(value.split())


def _normalize_email_login_value(value: str) -> str:
    return "".join(str(value or "").split())


def _normalize_app_password(value: str) -> str:
    return "".join(str(value or "").split())


def _address_list(value: Any) -> list[str]:
    raw_values = _string_list(value)
    addresses = [addr for _, addr in getaddresses(raw_values) if addr]
    return addresses


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list | tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
