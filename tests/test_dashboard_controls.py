import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from dmdagent4all.config import DEFAULT_CONFIG
from dmdagent4all.server import (
    _connector_statuses,
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


if __name__ == "__main__":
    unittest.main()
