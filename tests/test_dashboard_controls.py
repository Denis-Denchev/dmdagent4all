import os
import tempfile
import unittest
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
    ReminderRuntime,
    _connector_statuses,
    _deepseek_status,
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
    _validate_terminal_allowlist_command,
)
from dmdagent4all.tools import build_builtin_registry
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
        self.assertIn("calendar", statuses)
        self.assertIn("telegram", statuses)

    def test_local_calendar_handlers_cover_update_delete_and_free_slots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            context = ToolRuntimeContext(
                memory_root=root / "memory",
                workspace_root=root / "workspace",
                config={},
            )
            registry = build_builtin_registry()
            created = registry.execute(
                "calendar.create_event",
                {
                    "title": "Busy",
                    "start": "2026-05-04T10:00:00-07:00",
                    "end": "2026-05-04T11:00:00-07:00",
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
                    "start": "2026-05-04T09:00:00-07:00",
                    "end": "2026-05-04T12:00:00-07:00",
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


if __name__ == "__main__":
    unittest.main()
