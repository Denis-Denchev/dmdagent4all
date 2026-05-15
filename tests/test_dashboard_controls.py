import os
import smtplib
import tempfile
import unittest
import urllib.parse
from unittest import mock
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import HTTPException

from dmdagent4all.config import DEFAULT_CONFIG
from dmdagent4all.llm.openai_usage import DEFAULT_OPENAI_API_KEY_ENV
from dmdagent4all.audit import AuditStore
from dmdagent4all.server import (
    DEFAULT_DEEPSEEK_API_KEY_ENV,
    DEEPSEEK_BASE_URL,
    DEFAULT_DEEPSEEK_MODEL,
    AgentConfigurationRequest,
    EmailConfigurationRequest,
    EmailCredentialsRequest,
    EmailProviderConfigurationRequest,
    GmailOAuthStartRequest,
    ReminderRuntime,
    _configuration_response,
    _configure_gmail_oauth,
    _connector_statuses,
    _deepseek_status,
    _load_email_credentials,
    _openai_status,
    _set_deepseek_config,
    _set_model_config,
    _set_openai_config,
    _set_openai_limit,
    _set_telegram_token_env,
    _set_telegram_user_allowed,
    _set_terminal_enabled,
    _telegram_status,
    _terminal_command_from_request,
    _terminal_status,
    _update_agent_configuration,
    _validate_terminal_allowlist_command,
)
from dmdagent4all.email_oauth import GMAIL_OAUTH_SCOPE, google_authorization_url
from dmdagent4all.tools import build_builtin_registry
from dmdagent4all.tools.email_connector import _settings, test_email_connection
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.reminders import (
    create_reminder,
    list_reminders,
    reminder_change_version,
    wait_for_reminder_change_since,
)


class DashboardControlsTest(unittest.TestCase):
    def test_terminal_controls_enable_full_policy_path(self) -> None:
        config = {**DEFAULT_CONFIG, "permissions": {"granted": []}, "tools": {}}

        _set_terminal_enabled(config, True)

        status = _terminal_status(config)
        self.assertTrue(status["enabled"])
        self.assertTrue(status["tool_enabled"])
        self.assertTrue(status["permission_granted"])
        self.assertTrue(status["ready"])

    def test_terminal_command_parser_and_validator_block_dangerous_commands(self) -> None:
        self.assertEqual(_terminal_command_from_request("git status"), ["git", "status"])
        with self.assertRaises(HTTPException):
            _validate_terminal_allowlist_command(["rm", "-rf", "."])

    def test_telegram_controls_keep_token_as_environment_reference(self) -> None:
        config = {**DEFAULT_CONFIG, "interfaces": {"telegram": {}}}

        _set_telegram_token_env(config, "DMDAGENT_TEST_TOKEN")
        _set_telegram_user_allowed(config, 12345, True)

        status = _telegram_status(config)
        self.assertEqual(status["bot_token_env"], "DMDAGENT_TEST_TOKEN")
        self.assertEqual(status["allowed_user_ids"], [12345])
        with self.assertRaises(HTTPException):
            _set_telegram_token_env(config, "not valid")

    def test_openai_controls_use_process_key_and_local_limit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _FakePaths(Path(tmp))
            config = deepcopy(DEFAULT_CONFIG)
            os.environ.pop(DEFAULT_OPENAI_API_KEY_ENV, None)

            _set_openai_config(config, model="gpt-test")
            _set_openai_limit(config, 3.5)
            status = _openai_status(config, paths)

            self.assertEqual(status["provider"], "openai")
            self.assertEqual(status["model"], "gpt-test")
            self.assertEqual(status["api_key_env"], DEFAULT_OPENAI_API_KEY_ENV)
            self.assertFalse(status["api_key_available"])
            self.assertEqual(status["usage"]["limit_usd"], 3.5)

            os.environ[DEFAULT_OPENAI_API_KEY_ENV] = "sk-test"
            try:
                status = _openai_status(config, paths)
            finally:
                os.environ.pop(DEFAULT_OPENAI_API_KEY_ENV, None)

        self.assertTrue(status["api_key_available"])

    def test_deepseek_controls_use_process_key_and_official_defaults(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        os.environ.pop(DEFAULT_DEEPSEEK_API_KEY_ENV, None)

        _set_deepseek_config(config, model=DEFAULT_DEEPSEEK_MODEL)
        status = _deepseek_status(config)

        self.assertEqual(status["provider"], "deepseek")
        self.assertEqual(status["model"], DEFAULT_DEEPSEEK_MODEL)
        self.assertEqual(status["base_url"], DEEPSEEK_BASE_URL)
        self.assertEqual(status["api_key_env"], DEFAULT_DEEPSEEK_API_KEY_ENV)
        self.assertFalse(status["api_key_available"])
        self.assertIn("deepseek-v4-pro", status["default_models"])

        os.environ[DEFAULT_DEEPSEEK_API_KEY_ENV] = "sk-deepseek-test"
        try:
            status = _deepseek_status(config)
        finally:
            os.environ.pop(DEFAULT_DEEPSEEK_API_KEY_ENV, None)

        self.assertTrue(status["api_key_available"])

    def test_local_model_selection_resets_openai_provider(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        _set_openai_config(config, model="gpt-test")

        _set_model_config(
            config,
            mode="fast",
            model="qwen3:8b",
            planner_model="qwen3:8b",
        )

        self.assertEqual(config["llm"]["provider"], "ollama")
        self.assertEqual(config["llm"]["mode"], "fast")
        self.assertEqual(config["llm"]["model"], "qwen3:8b")
        self.assertEqual(config["llm"]["planner_model"], "qwen3:8b")
        self.assertEqual(config["llm"]["base_url"], "http://localhost:11434")
        self.assertIsNone(config["llm"]["api_key_env"])

    def test_connector_statuses_include_gmail_as_not_configured(self) -> None:
        registry = build_builtin_registry()

        statuses = {
            item["name"]: item
            for item in _connector_statuses(DEFAULT_CONFIG.copy(), registry.manifests)
        }

        self.assertEqual(statuses["gmail"]["status"], "not_configured")
        self.assertEqual(statuses["outlook"]["status"], "not_configured")
        self.assertIn("calendar", statuses)
        self.assertIn("telegram", statuses)

    def test_email_configuration_can_be_managed_from_config_api(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _FakePaths(Path(tmp))
            config = deepcopy(DEFAULT_CONFIG)

            _update_agent_configuration(
                config,
                AgentConfigurationRequest(
                    email=EmailConfigurationRequest(
                        max_body_chars=12345,
                        gmail=EmailProviderConfigurationRequest(
                            enabled=True,
                            username_env="DMDAGENT_TEST_GMAIL_USER",
                            password_env="DMDAGENT_TEST_GMAIL_PASSWORD",
                            from_env="DMDAGENT_TEST_GMAIL_FROM",
                            mailbox="Primary",
                        ),
                    ),
                ),
            )
            response = _configuration_response(paths, config)

        self.assertEqual(response["email"]["max_body_chars"], 12345)
        self.assertTrue(response["email"]["gmail"]["enabled"])
        self.assertEqual(response["email"]["gmail"]["username_env"], "DMDAGENT_TEST_GMAIL_USER")
        self.assertEqual(response["email"]["gmail"]["password_env"], "DMDAGENT_TEST_GMAIL_PASSWORD")
        self.assertEqual(response["email"]["gmail"]["from_env"], "DMDAGENT_TEST_GMAIL_FROM")
        self.assertEqual(response["email"]["gmail"]["mailbox"], "Primary")
        self.assertTrue(config["tools"]["gmail.search"]["enabled"])
        self.assertTrue(config["tools"]["gmail.read_thread"]["enabled"])
        self.assertTrue(config["tools"]["gmail.send_draft"]["enabled"])
        self.assertIn("gmail.send", config["permissions"]["granted"])

    def test_email_credentials_loader_sets_process_env_without_returning_secret(self) -> None:
        config = deepcopy(DEFAULT_CONFIG)
        config["email"]["gmail"]["username_env"] = "DMDAGENT_TEST_GMAIL_USER"
        config["email"]["gmail"]["password_env"] = "DMDAGENT_TEST_GMAIL_PASSWORD"
        config["email"]["gmail"]["from_env"] = "DMDAGENT_TEST_GMAIL_FROM"
        for key in (
            "DMDAGENT_TEST_GMAIL_USER",
            "DMDAGENT_TEST_GMAIL_PASSWORD",
            "DMDAGENT_TEST_GMAIL_FROM",
        ):
            os.environ.pop(key, None)

        try:
            result = _load_email_credentials(
                config,
                EmailCredentialsRequest(
                    provider="gmail",
                    username="sender@example.com",
                    app_password="app-password",
                    from_address="from@example.com",
                ),
            )
        finally:
            env_values = {
                key: os.environ.pop(key, None)
                for key in (
                    "DMDAGENT_TEST_GMAIL_USER",
                    "DMDAGENT_TEST_GMAIL_PASSWORD",
                    "DMDAGENT_TEST_GMAIL_FROM",
                )
            }

        self.assertEqual(env_values["DMDAGENT_TEST_GMAIL_USER"], "sender@example.com")
        self.assertEqual(env_values["DMDAGENT_TEST_GMAIL_PASSWORD"], "app-password")
        self.assertEqual(env_values["DMDAGENT_TEST_GMAIL_FROM"], "from@example.com")
        self.assertTrue(result["data"]["credentials_loaded"])
        self.assertNotIn("app-password", str(result))

    def test_gmail_oauth_configuration_does_not_expose_secret(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            paths = _FakePaths(Path(tmp))
            config = deepcopy(DEFAULT_CONFIG)
            _configure_gmail_oauth(
                config,
                GmailOAuthStartRequest(
                    client_id="client-id.apps.googleusercontent.com",
                    email="sender@gmail.com",
                    from_address="from@gmail.com",
                    redirect_uri="http://127.0.0.1:8765/v1/email/oauth/google/callback",
                ),
            )
            with mock.patch("dmdagent4all.server.load_email_secret", return_value="stored-secret"):
                response = _configuration_response(paths, config)

        gmail = response["email"]["gmail"]
        self.assertTrue(gmail["enabled"])
        self.assertEqual(gmail["auth_method"], "oauth2")
        self.assertEqual(gmail["oauth_client_id"], "client-id.apps.googleusercontent.com")
        self.assertEqual(gmail["oauth_email"], "sender@gmail.com")
        self.assertTrue(gmail["oauth_client_secret_loaded"])
        self.assertTrue(gmail["oauth_refresh_token_loaded"])
        self.assertTrue(gmail["oauth_connected"])
        self.assertNotIn("stored-secret", str(response))

    def test_gmail_oauth_authorization_url_uses_gmail_scope_and_offline_access(self) -> None:
        url = google_authorization_url(
            client_id="client-id.apps.googleusercontent.com",
            redirect_uri="http://127.0.0.1:8765/v1/email/oauth/google/callback",
            state="state-123",
        )
        parsed = urllib.parse.urlparse(url)
        query = urllib.parse.parse_qs(parsed.query)

        self.assertEqual(parsed.netloc, "accounts.google.com")
        self.assertEqual(query["scope"], [GMAIL_OAUTH_SCOPE])
        self.assertEqual(query["access_type"], ["offline"])
        self.assertEqual(query["prompt"], ["consent"])
        self.assertEqual(query["state"], ["state-123"])

    def test_gmail_oauth_settings_refresh_token_without_app_password(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["gmail"].update(
                {
                    "enabled": True,
                    "auth_method": "oauth2",
                    "oauth_client_id": "client-id.apps.googleusercontent.com",
                    "oauth_email": "sender@gmail.com",
                    "oauth_from_address": "from@gmail.com",
                }
            )
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            with mock.patch("dmdagent4all.tools.email_connector.load_email_secret", return_value="secret"):
                with mock.patch("dmdagent4all.tools.email_connector.refresh_google_access_token", return_value="access-token") as refresh:
                    settings = _settings("gmail", context)

        self.assertIsNotNone(settings)
        assert settings is not None
        self.assertEqual(settings.auth_method, "oauth2")
        self.assertEqual(settings.username, "sender@gmail.com")
        self.assertEqual(settings.from_addr, "from@gmail.com")
        self.assertEqual(settings.password, "")
        self.assertEqual(settings.oauth_access_token, "access-token")
        refresh.assert_called_once()

    def test_local_calendar_handlers_cover_update_delete_and_free_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config={},
            )
            registry = build_builtin_registry()
            base = (datetime.now().astimezone() + timedelta(days=7)).replace(
                hour=10,
                minute=0,
                second=0,
                microsecond=0,
            )
            created = registry.execute(
                "calendar.create_event",
                {
                    "title": "Busy",
                    "start": base.isoformat(),
                    "end": (base + timedelta(hours=1)).isoformat(),
                },
                context,
            )
            event_id = created["event"]["id"]

            updated = registry.execute(
                "calendar.update_event",
                {"id": event_id, "title": "Updated Busy"},
                context,
            )
            slots = registry.execute(
                "calendar.find_free_slots",
                {
                    "start": base.replace(hour=9).isoformat(),
                    "end": base.replace(hour=12).isoformat(),
                    "duration_minutes": 30,
                },
                context,
            )
            deleted = registry.execute("calendar.delete_event", {"id": event_id}, context)

        self.assertEqual(updated["event"]["title"], "Updated Busy")
        self.assertEqual(deleted["event_id"], event_id)
        self.assertTrue(slots["slots"])

    def test_gmail_handlers_fail_as_not_configured_instead_of_missing_handler(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config={},
            )
            result = build_builtin_registry().execute("gmail.search", {"query": "from:test"}, context)

        self.assertEqual(result["status"], "connector_not_configured")

    def test_email_draft_and_send_use_env_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["gmail"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            registry = build_builtin_registry()
            env = {
                "DMDAGENT_GMAIL_USERNAME": "sender@example.com",
                "DMDAGENT_GMAIL_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                draft = registry.execute(
                    "gmail.create_draft",
                    {
                        "to": "recipient@example.com",
                        "subject": "Test subject",
                        "body": "Hello from draft",
                    },
                    context,
                )
                with mock.patch("dmdagent4all.tools.email_connector.smtplib.SMTP", FakeSMTP):
                    FakeSMTP.sent_messages.clear()
                    sent = registry.execute("gmail.send_draft", {"draft_id": draft["draft_id"]}, context)

        self.assertEqual(draft["status"], "draft")
        self.assertTrue(sent["sent"])
        self.assertEqual(sent["to"], ["recipient@example.com"])
        self.assertEqual(FakeSMTP.sent_messages[0]["To"], "recipient@example.com")

    def test_email_draft_sanitizes_multiline_subject_before_send(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["gmail"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            registry = build_builtin_registry()
            env = {
                "DMDAGENT_GMAIL_USERNAME": "sender@example.com",
                "DMDAGENT_GMAIL_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                draft = registry.execute(
                    "gmail.create_draft",
                    {
                        "to": "recipient@example.com",
                        "subject": "Line one\nLine two",
                        "body": "Hello from draft",
                    },
                    context,
                )
                with mock.patch("dmdagent4all.tools.email_connector.smtplib.SMTP", FakeSMTP):
                    FakeSMTP.sent_messages.clear()
                    sent = registry.execute("gmail.send_draft", {"draft_id": draft["draft_id"]}, context)

        self.assertTrue(sent["sent"])
        self.assertEqual(draft["subject"], "Line one Line two")
        self.assertEqual(FakeSMTP.sent_messages[0]["Subject"], "Line one Line two")

    def test_gmail_oauth_send_uses_xoauth2_instead_of_password_login(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["gmail"].update(
                {
                    "enabled": True,
                    "auth_method": "oauth2",
                    "oauth_client_id": "client-id.apps.googleusercontent.com",
                    "oauth_email": "sender@gmail.com",
                }
            )
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            registry = build_builtin_registry()
            with mock.patch("dmdagent4all.tools.email_connector.load_email_secret", return_value="secret"):
                with mock.patch("dmdagent4all.tools.email_connector.refresh_google_access_token", return_value="access-token"):
                    draft = registry.execute(
                        "gmail.create_draft",
                        {
                            "to": "recipient@example.com",
                            "subject": "OAuth subject",
                            "body": "Hello from OAuth",
                        },
                        context,
                    )
                    with mock.patch("dmdagent4all.tools.email_connector.smtplib.SMTP", FakeOAuthSMTP):
                        FakeOAuthSMTP.sent_messages.clear()
                        FakeOAuthSMTP.auth_commands.clear()
                        FakeOAuthSMTP.login_called = False
                        sent = registry.execute("gmail.send_draft", {"draft_id": draft["draft_id"]}, context)

        self.assertTrue(sent["sent"])
        self.assertEqual(FakeOAuthSMTP.sent_messages[0]["To"], "recipient@example.com")
        self.assertFalse(FakeOAuthSMTP.login_called)
        self.assertTrue(FakeOAuthSMTP.auth_commands)
        self.assertTrue(FakeOAuthSMTP.auth_commands[0].startswith("AUTH XOAUTH2 "))

    def test_outlook_basic_auth_failure_returns_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["outlook"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            registry = build_builtin_registry()
            env = {
                "DMDAGENT_OUTLOOK_USERNAME": "sender@outlook.com",
                "DMDAGENT_OUTLOOK_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                draft = registry.execute(
                    "outlook.create_draft",
                    {
                        "to": "recipient@example.com",
                        "subject": "Test subject",
                        "body": "Hello from draft",
                    },
                    context,
                )
                with mock.patch("dmdagent4all.tools.email_connector.smtplib.SMTP", FailingAuthSMTP):
                    result = registry.execute("outlook.send_draft", {"draft_id": draft["draft_id"]}, context)

        self.assertEqual(result["status"], "authentication_failed")
        self.assertFalse(result["sent"])
        self.assertIn("Basic Authentication is disabled", result["message"])
        self.assertNotIn("app-password", str(result))

    def test_outlook_search_uses_imap_connector(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["outlook"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            env = {
                "DMDAGENT_OUTLOOK_USERNAME": "sender@outlook.com",
                "DMDAGENT_OUTLOOK_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                with mock.patch("dmdagent4all.tools.email_connector.imaplib.IMAP4_SSL", FakeIMAP):
                    result = build_builtin_registry().execute(
                        "outlook.summarize_inbox",
                        {"query": "subject:Invoice", "limit": 1},
                        context,
                    )

        self.assertEqual(result["provider"], "outlook")
        self.assertEqual(result["count"], 1)
        self.assertEqual(result["summary"][0]["subject"], "Invoice 123")

    def test_email_read_thread_can_read_latest_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["gmail"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            env = {
                "DMDAGENT_GMAIL_USERNAME": "sender@gmail.com",
                "DMDAGENT_GMAIL_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                with mock.patch("dmdagent4all.tools.email_connector.imaplib.IMAP4_SSL", FakeIMAP):
                    result = build_builtin_registry().execute(
                        "gmail.read_thread",
                        {"latest": True},
                        context,
                    )

        self.assertEqual(result["provider"], "gmail")
        self.assertEqual(result["thread_id"], "99")
        self.assertEqual(result["messages"][0]["subject"], "Invoice 123")

    def test_email_connection_test_reports_imap_and_smtp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["gmail"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            env = {
                "DMDAGENT_GMAIL_USERNAME": "sender@gmail.com",
                "DMDAGENT_GMAIL_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                with mock.patch("dmdagent4all.tools.email_connector.imaplib.IMAP4_SSL", FakeIMAP):
                    with mock.patch("dmdagent4all.tools.email_connector.smtplib.SMTP", FakeSMTP):
                        result = test_email_connection("gmail", context)

        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["imap"]["ok"])
        self.assertTrue(result["smtp"]["ok"])
        self.assertEqual(result["auth_method"], "app_password")

    def test_email_connection_test_returns_actionable_outlook_auth_failure(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = deepcopy(DEFAULT_CONFIG)
            config["email"]["outlook"]["enabled"] = True
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config=config,
            )
            env = {
                "DMDAGENT_OUTLOOK_USERNAME": "sender@outlook.com",
                "DMDAGENT_OUTLOOK_APP_PASSWORD": "app-password",
            }
            with mock.patch.dict(os.environ, env):
                with mock.patch("dmdagent4all.tools.email_connector.imaplib.IMAP4_SSL", FakeIMAP):
                    with mock.patch("dmdagent4all.tools.email_connector.smtplib.SMTP", FailingAuthSMTP):
                        result = test_email_connection("outlook", context)

        self.assertEqual(result["status"], "connection_failed")
        self.assertTrue(result["imap"]["ok"])
        self.assertFalse(result["smtp"]["ok"])
        self.assertIn("Microsoft Graph OAuth", result["message"])

    def test_memory_write_can_create_auto_short_term_markdown(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config={},
            )

            result = build_builtin_registry().execute(
                "memory.write",
                {
                    "title": "Current Stripe bug",
                    "body": "Investigating webhook retries.",
                    "memory_scope": "short-term",
                    "ttl_hours": 48,
                },
                context,
            )
            path = root / "memory" / result["path"]
            content = path.read_text(encoding="utf-8")

        self.assertTrue(result["path"].startswith("short-term/"))
        self.assertIn("memory_scope: short-term", content)
        self.assertIn("ttl_hours: 48", content)
        self.assertIn("expires_at:", content)

    def test_reminder_runtime_sends_due_reminders_to_telegram_and_marks_notified(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _FakePaths(root)
            context = ToolRuntimeContext(
                memory_root=paths.memory,
                workspace_root=paths.workspace,
                config={},
            )
            created = create_reminder(
                {
                    "title": "Check the oven",
                    "due_at": "2020-01-01T00:00:00+00:00",
                },
                context,
            )
            telegram = _FakeTelegramRuntime()
            runtime = ReminderRuntime(paths, telegram)

            sent = runtime.tick()
            listed = list_reminders({"status": "all"}, context)["reminders"]
            events = AuditStore(paths.audit_db).list_recent_events()

        self.assertEqual(sent, 1)
        self.assertEqual(telegram.messages, ["Reminder: Check the oven"])
        self.assertIsNotNone(telegram.reply_markups[0])
        self.assertEqual(listed[0]["id"], created["reminder"]["id"])
        self.assertEqual(listed[0]["status"], "notified")
        self.assertEqual(listed[0]["notification_channel"], "telegram")
        self.assertEqual(events[0]["event_type"], "reminder.notification")

    def test_reminder_runtime_sleeps_until_next_due_with_long_cap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _FakePaths(root)
            context = ToolRuntimeContext(
                memory_root=paths.memory,
                workspace_root=paths.workspace,
                config={},
            )
            now = datetime(2026, 5, 4, 12, 0, 0, tzinfo=timezone.utc)
            runtime = ReminderRuntime(paths, _FakeTelegramRuntime())

            self.assertEqual(runtime.next_sleep_seconds(now=now), runtime.idle_sleep_seconds)

            create_reminder(
                {
                    "title": "Much later",
                    "due_at": (now + timedelta(days=2)).isoformat(),
                },
                context,
            )
            self.assertEqual(runtime.next_sleep_seconds(now=now), runtime.max_sleep_seconds)

            create_reminder(
                {
                    "title": "Later",
                    "due_at": (now + timedelta(minutes=10)).isoformat(),
                },
                context,
            )
            self.assertEqual(
                runtime.next_sleep_seconds(now=now),
                600 - runtime.precision_window_seconds,
            )

            create_reminder(
                {
                    "title": "Soon",
                    "due_at": (now + timedelta(seconds=45)).isoformat(),
                },
                context,
            )
            self.assertEqual(runtime.next_sleep_seconds(now=now), 45)

    def test_reminder_store_change_wakes_waiters(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _FakePaths(root)
            context = ToolRuntimeContext(
                memory_root=paths.memory,
                workspace_root=paths.workspace,
                config={},
            )
            version = reminder_change_version()

            create_reminder(
                {
                    "title": "Wake runtime",
                    "due_at": "2026-05-04T12:00:00+00:00",
                },
                context,
            )

            self.assertGreater(wait_for_reminder_change_since(version, 0), version)

    def test_reminder_location_generates_maps_action_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config={},
            )

            created = create_reminder(
                {
                    "title": "Dentist",
                    "due_at": "2026-05-04T12:00:00+00:00",
                    "location": "бул. България 10",
                },
                context,
            )

        self.assertEqual(created["reminder"]["location"], "бул. България 10")
        self.assertIn("google.com/maps", created["reminder"]["action_url"])


class _FakePaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = root / "config.yaml"
        self.memory = root / "memory"
        self.workspace = root / "workspace"
        self.audit_db = root / "audit.db"
        self.config.write_text("llm:\n  provider: ollama\n", encoding="utf-8")
        self.memory.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)


class _FakeTelegramRuntime:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.reply_markups: list[dict | None] = []

    def send_to_allowed_users(
        self,
        config: dict,
        text: str,
        *,
        reply_markup: dict | None = None,
    ) -> bool:
        del config
        self.messages.append(text)
        self.reply_markups.append(reply_markup)
        return True


class FakeSMTP:
    sent_messages: list = []

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.logged_in = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def starttls(self) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        self.logged_in = bool(username and password)

    def send_message(self, message) -> None:
        self.sent_messages.append(message)


class FailingAuthSMTP(FakeSMTP):
    def login(self, username: str, password: str) -> None:
        del username, password
        raise smtplib.SMTPAuthenticationError(
            535,
            b"5.7.139 Authentication unsuccessful, basic authentication is disabled.",
        )


class FakeOAuthSMTP(FakeSMTP):
    auth_commands: list[str] = []
    login_called = False

    def login(self, username: str, password: str) -> None:
        del username, password
        type(self).login_called = True

    def docmd(self, command: str, args: str):
        self.auth_commands.append(f"{command} {args}")
        return 235, b"2.7.0 Accepted"


class FakeIMAP:
    raw_message = (
        b"From: Sender <sender@example.com>\r\n"
        b"To: Receiver <receiver@example.com>\r\n"
        b"Subject: Invoice 123\r\n"
        b"Date: Thu, 07 May 2026 10:00:00 +0000\r\n"
        b"Message-ID: <invoice-123@example.com>\r\n"
        b"\r\n"
        b"Invoice body text"
    )

    def __init__(self, host: str, port: int, timeout: int) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def login(self, username: str, password: str) -> None:
        if not username or not password:
            raise RuntimeError("missing login")

    def select(self, mailbox: str, readonly: bool = False):
        del mailbox, readonly
        return "OK", [b""]

    def uid(self, command: str, *args):
        del args
        if command == "SEARCH":
            return "OK", [b"99"]
        if command == "FETCH":
            return "OK", [(b"99 (RFC822)", self.raw_message)]
        if command in {"COPY", "STORE"}:
            return "OK", [b""]
        return "NO", [b""]

    def expunge(self):
        return "OK", [b""]


if __name__ == "__main__":
    unittest.main()
