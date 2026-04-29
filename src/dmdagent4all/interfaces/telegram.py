from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable, Protocol

from dmdagent4all.agent import AgentCore, AgentResponse
from dmdagent4all.app_paths import AppPaths
from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.config import load_config, write_default_config
from dmdagent4all.runtime import build_agent_core
from dmdagent4all.security import redact_text


DEFAULT_TOKEN_ENV = "DMDAGENT_TELEGRAM_BOT_TOKEN"
TELEGRAM_MESSAGE_LIMIT = 4096


class TelegramError(RuntimeError):
    pass


class TelegramConfigError(TelegramError):
    pass


class TelegramAPI(Protocol):
    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        ...

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        ...

    def answer_callback_query(self, callback_query_id: str, text: str) -> None:
        ...


@dataclass(frozen=True)
class TelegramSettings:
    enabled: bool
    allowed_user_ids: frozenset[int]
    bot_token_env: str = DEFAULT_TOKEN_ENV
    polling_timeout_seconds: int = 30


class TelegramBotAPI:
    def __init__(self, token: str, *, api_root: str = "https://api.telegram.org") -> None:
        if not token.strip():
            raise TelegramConfigError("Telegram bot token is empty.")
        self._base_url = f"{api_root.rstrip('/')}/bot{token}"

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        payload: dict[str, Any] = {
            "timeout": timeout_seconds,
            "allowed_updates": ["message", "callback_query"],
        }
        if offset is not None:
            payload["offset"] = offset
        result = self._call("getUpdates", payload)
        if not isinstance(result, list):
            raise TelegramError("Telegram getUpdates returned an unexpected payload.")
        return [update for update in result if isinstance(update, dict)]

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": _fit_message(text),
            "disable_web_page_preview": True,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup
        self._call("sendMessage", payload)

    def answer_callback_query(self, callback_query_id: str, text: str) -> None:
        self._call(
            "answerCallbackQuery",
            {
                "callback_query_id": callback_query_id,
                "text": _fit_message(text, limit=180),
            },
        )

    def _call(self, method: str, payload: dict[str, Any]) -> Any:
        request = urllib.request.Request(
            f"{self._base_url}/{method}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                raw = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise TelegramError(f"Telegram API HTTP {exc.code}: {_safe_api_error(body)}") from exc
        except urllib.error.URLError as exc:
            raise TelegramError(f"Telegram API request failed: {exc.reason}") from exc

        data = json.loads(raw)
        if not data.get("ok"):
            description = str(data.get("description", "unknown error"))
            raise TelegramError(f"Telegram API error: {_safe_api_error(description)}")
        return data.get("result")


class TelegramInterface:
    def __init__(
        self,
        *,
        settings: TelegramSettings,
        api: TelegramAPI,
        audit_store: AuditStore,
        core_factory: Callable[[], AgentCore] = build_agent_core,
    ) -> None:
        self.settings = settings
        self.api = api
        self.audit_store = audit_store
        self.core_factory = core_factory

    @classmethod
    def from_current_config(cls) -> "TelegramInterface":
        paths = AppPaths.default()
        paths.ensure()
        write_default_config(paths.config)
        config = load_config(paths.config)
        settings = settings_from_config(config)
        token = os.environ.get(settings.bot_token_env)
        if not token:
            raise TelegramConfigError(
                f"Missing Telegram bot token. Export {settings.bot_token_env} before running."
            )
        return cls(
            settings=settings,
            api=TelegramBotAPI(token),
            audit_store=AuditStore(paths.audit_db),
        )

    def run_polling(
        self,
        *,
        once: bool = False,
        timeout_seconds: int | None = None,
        stop_event: Any | None = None,
    ) -> None:
        timeout = timeout_seconds or self.settings.polling_timeout_seconds
        offset: int | None = None
        while stop_event is None or not stop_event.is_set():
            try:
                updates = self.api.get_updates(offset=offset, timeout_seconds=timeout)
            except TelegramError:
                if once:
                    raise
                if stop_event is None:
                    time.sleep(2)
                else:
                    stop_event.wait(2)
                continue
            offset = self.process_updates(updates, offset=offset)
            if once:
                return
            if not updates:
                if stop_event is None:
                    time.sleep(0.2)
                else:
                    stop_event.wait(0.2)

    def process_updates(
        self,
        updates: list[dict[str, Any]],
        *,
        offset: int | None = None,
    ) -> int | None:
        next_offset = offset
        for update in updates:
            update_id = _as_int(update.get("update_id"))
            if update_id is not None:
                next_offset = update_id + 1
            try:
                self.handle_update(update)
            except TelegramError:
                continue
        return next_offset

    def handle_update(self, update: dict[str, Any]) -> None:
        if isinstance(update.get("callback_query"), dict):
            self._handle_callback(update, update["callback_query"])
            return
        if isinstance(update.get("message"), dict):
            self._handle_message(update, update["message"])

    def _handle_message(self, update: dict[str, Any], message: dict[str, Any]) -> None:
        chat = message.get("chat")
        sender = message.get("from")
        if not isinstance(chat, dict) or not isinstance(sender, dict):
            return
        chat_id = _as_int(chat.get("id"))
        user_id = _as_int(sender.get("id"))
        text = str(message.get("text") or "").strip()
        if chat_id is None or user_id is None or not text:
            return

        if text == "/id":
            self.api.send_message(chat_id, f"Telegram user ID: {user_id}")
            self._record_request(
                update=update,
                user_id=user_id,
                chat_id=chat_id,
                text=text,
                allowed=True,
                result_status="id",
                request_type="message",
            )
            return

        denial = self._authorization_denial(user_id)
        if denial is not None:
            self.api.send_message(chat_id, denial)
            self._record_request(
                update=update,
                user_id=user_id,
                chat_id=chat_id,
                text=text,
                allowed=False,
                result_status="denied",
                request_type="message",
            )
            return

        response = self._dispatch_message(text)
        reply_markup = _approval_keyboard(response)
        self.api.send_message(
            chat_id,
            _format_agent_response(response),
            reply_markup=reply_markup,
        )
        self._record_request(
            update=update,
            user_id=user_id,
            chat_id=chat_id,
            text=text,
            allowed=True,
            result_status=response.status,
            request_type="message",
        )

    def _handle_callback(self, update: dict[str, Any], callback: dict[str, Any]) -> None:
        sender = callback.get("from")
        message = callback.get("message")
        if not isinstance(sender, dict) or not isinstance(message, dict):
            return
        chat = message.get("chat")
        if not isinstance(chat, dict):
            return
        user_id = _as_int(sender.get("id"))
        chat_id = _as_int(chat.get("id"))
        callback_id = str(callback.get("id") or "")
        data = str(callback.get("data") or "")
        if user_id is None or chat_id is None or not callback_id:
            return

        denial = self._authorization_denial(user_id)
        if denial is not None:
            self.api.answer_callback_query(callback_id, "Access denied.")
            self.api.send_message(chat_id, denial)
            self._record_request(
                update=update,
                user_id=user_id,
                chat_id=chat_id,
                text=data,
                allowed=False,
                result_status="denied",
                request_type="callback",
            )
            return

        action, approval_id = _parse_approval_callback(data)
        if action is None or approval_id is None:
            self.api.answer_callback_query(callback_id, "Unsupported action.")
            self._record_request(
                update=update,
                user_id=user_id,
                chat_id=chat_id,
                text=data,
                allowed=True,
                result_status="unsupported_callback",
                request_type="callback",
            )
            return

        response = self._handle_approval_action(action, approval_id)
        self.api.answer_callback_query(callback_id, response.message)
        self.api.send_message(chat_id, _format_agent_response(response))
        self._record_request(
            update=update,
            user_id=user_id,
            chat_id=chat_id,
            text=data,
            allowed=True,
            result_status=response.status,
            request_type="callback",
        )

    def _dispatch_message(self, text: str) -> AgentResponse:
        if text in {"/start", "/help"}:
            return AgentResponse(
                status="ok",
                message=(
                    "DMD Agent 4 All Telegram interface is active.\n"
                    "Commands: /id, /help, /approvals, /approve <id>, /deny <id>.\n"
                    "Any other message is routed through the local agent and permission engine."
                ),
            )
        if text == "/approvals":
            approvals = self.audit_store.list_approvals(status="pending", limit=10)
            if not approvals:
                return AgentResponse(status="ok", message="No pending approvals.")
            return AgentResponse(
                status="ok",
                message="Pending approvals:",
                data={"approvals": approvals},
            )
        if text.startswith("/approve "):
            approval_id = _as_int(text.removeprefix("/approve ").strip())
            if approval_id is None:
                return AgentResponse(status="error", message="Usage: /approve <id>")
            return self._handle_approval_action("approve", approval_id)
        if text.startswith("/deny "):
            approval_id = _as_int(text.removeprefix("/deny ").strip())
            if approval_id is None:
                return AgentResponse(status="error", message="Usage: /deny <id>")
            return self._handle_approval_action("deny", approval_id)
        return self.core_factory().handle_text(text)

    def _handle_approval_action(self, action: str, approval_id: int) -> AgentResponse:
        if action == "approve":
            return self.core_factory().approve_and_execute(approval_id)
        if action == "deny":
            changed = self.audit_store.set_approval_status(approval_id, "denied")
            if not changed:
                return AgentResponse(
                    status="not_found",
                    message="No pending approval found.",
                    data={"approval_id": approval_id},
                )
            return AgentResponse(
                status="ok",
                message="Approval denied.",
                data={"approval_id": approval_id},
            )
        return AgentResponse(status="error", message=f"Unknown approval action: {action}")

    def _authorization_denial(self, user_id: int) -> str | None:
        if not self.settings.enabled:
            return "Telegram interface is disabled locally. In terminal chat run: /telegram enable"
        if not self.settings.allowed_user_ids:
            return (
                "Telegram interface has no allowlisted users. "
                f"In terminal chat run: /telegram allow {user_id}"
            )
        if user_id not in self.settings.allowed_user_ids:
            return (
                "Access denied. This Telegram user is not allowlisted. "
                f"User ID: {user_id}"
            )
        return None

    def _record_request(
        self,
        *,
        update: dict[str, Any],
        user_id: int,
        chat_id: int,
        text: str,
        allowed: bool,
        result_status: str,
        request_type: str,
    ) -> None:
        self.audit_store.record_event(
            AuditEvent(
                event_type="telegram.request",
                user_request=redact_text(text),
                approved=allowed,
                result_status=result_status,
                metadata={
                    "telegram_user_id": user_id,
                    "telegram_chat_id": chat_id,
                    "telegram_update_id": update.get("update_id"),
                    "request_type": request_type,
                },
            )
        )


def settings_from_config(config: dict[str, Any]) -> TelegramSettings:
    telegram = config.get("interfaces", {}).get("telegram", {})
    raw_user_ids = telegram.get("allowed_user_ids", [])
    allowed_user_ids: set[int] = set()
    if isinstance(raw_user_ids, list):
        for value in raw_user_ids:
            user_id = _as_int(value)
            if user_id is not None:
                allowed_user_ids.add(user_id)
    timeout = _as_int(telegram.get("polling_timeout_seconds")) or 30
    return TelegramSettings(
        enabled=bool(telegram.get("enabled", False)),
        allowed_user_ids=frozenset(allowed_user_ids),
        bot_token_env=str(telegram.get("bot_token_env") or DEFAULT_TOKEN_ENV),
        polling_timeout_seconds=max(1, timeout),
    )


def _format_agent_response(response: AgentResponse) -> str:
    if response.status == "ok":
        lines = [response.message]
    else:
        lines = [f"[{response.status}] {response.message}"]
    if response.status == "error" and "Cannot connect to Ollama" in response.message:
        lines.append("")
        lines.append("Fix from the project terminal:")
        lines.append("  start model fast --pull")
        lines.append("  start telegram run")
    hidden_data_shapes = (
        {"planner"},
        {"approval_id"},
        {"approval_id", "tool"},
        {"approval_id", "risk", "tool"},
        {"path"},
    )
    if response.data == {"missing_permissions": []}:
        return _fit_message(redact_text("\n".join(lines)))
    if response.data and set(response.data.keys()) not in hidden_data_shapes:
        lines.append(json.dumps(response.data, indent=2, sort_keys=True, ensure_ascii=False))
    return _fit_message(redact_text("\n".join(lines)))


def _approval_keyboard(response: AgentResponse) -> dict[str, Any] | None:
    if response.status != "approval_required" or not response.data:
        return None
    approval_id = _as_int(response.data.get("approval_id"))
    if approval_id is None:
        return None
    return {
        "inline_keyboard": [
            [
                {"text": f"Approve #{approval_id}", "callback_data": f"approve:{approval_id}"},
                {"text": f"Deny #{approval_id}", "callback_data": f"deny:{approval_id}"},
            ]
        ]
    }


def _parse_approval_callback(data: str) -> tuple[str | None, int | None]:
    if ":" not in data:
        return None, None
    action, raw_id = data.split(":", 1)
    if action not in {"approve", "deny"}:
        return None, None
    approval_id = _as_int(raw_id)
    return action, approval_id


def _fit_message(text: str, *, limit: int = TELEGRAM_MESSAGE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    suffix = "\n\n[truncated]"
    return text[: max(0, limit - len(suffix))] + suffix


def _as_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _safe_api_error(text: str) -> str:
    return redact_text(text).replace("\n", " ")[:500]
