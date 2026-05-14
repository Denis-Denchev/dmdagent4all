import os
import tempfile
import unittest
from pathlib import Path
from typing import Any

from dmdagent4all.agent import AgentResponse
from dmdagent4all.audit import AuditStore
from dmdagent4all.interfaces.telegram import (
    TelegramInterface,
    TelegramSettings,
    load_telegram_token,
    store_telegram_token,
)
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.reminders import create_reminder, list_reminders


class FakeTelegramAPI:
    def __init__(self) -> None:
        self.sent_messages: list[dict[str, Any]] = []
        self.callback_answers: list[dict[str, Any]] = []

    def get_updates(
        self,
        *,
        offset: int | None,
        timeout_seconds: int,
    ) -> list[dict[str, Any]]:
        del offset, timeout_seconds
        return []

    def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_markup: dict[str, Any] | None = None,
    ) -> None:
        self.sent_messages.append(
            {"chat_id": chat_id, "text": text, "reply_markup": reply_markup}
        )

    def answer_callback_query(self, callback_query_id: str, text: str) -> None:
        self.callback_answers.append({"id": callback_query_id, "text": text})


class FakeCore:
    def __init__(self, response: AgentResponse) -> None:
        self.response = response
        self.messages: list[str] = []
        self.approved: list[int] = []

    def handle_text(self, text: str, *, session_id: str = "default") -> AgentResponse:
        del session_id
        self.messages.append(text)
        return self.response

    def approve_and_execute(self, approval_id: int) -> AgentResponse:
        self.approved.append(approval_id)
        return AgentResponse(
            status="ok",
            message="Approval executed.",
            data={"approval_id": approval_id},
        )


class TelegramInterfaceTest(unittest.TestCase):
    def test_unauthorized_user_is_denied_and_audited(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(AgentResponse(status="ok", message="unused"))
            audit = AuditStore(Path(tmp) / "audit.db")
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=audit,
                core_factory=lambda: core,
            )

            interface.handle_update(_message_update(user_id=200, text="hello"))

            self.assertEqual(core.messages, [])
            self.assertIn("Access denied", api.sent_messages[0]["text"])
            events = audit.list_recent_events()
            self.assertEqual(events[0]["event_type"], "telegram.request")
            self.assertFalse(events[0]["approved"])
            self.assertEqual(events[0]["result_status"], "denied")

    def test_allowed_message_routes_to_agent_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(
                AgentResponse(
                    status="ok",
                    message="Hello from the local agent.",
                    data={"planner": "deterministic"},
                )
            )
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(Path(tmp) / "audit.db"),
                core_factory=lambda: core,
            )

            interface.handle_update(_message_update(user_id=100, text="hi"))

            self.assertEqual(core.messages, ["hi"])
            self.assertEqual(api.sent_messages[0]["text"], "Hello from the local agent.")
            self.assertIsNone(api.sent_messages[0]["reply_markup"])

    def test_tool_data_is_summarized_without_raw_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(
                AgentResponse(
                    status="ok",
                    message="Files written.",
                    data={
                        "tool": "files.write_many",
                        "files": [{"path": "src/app.tsx"}, {"path": "src/styles.css"}],
                        "count": 2,
                    },
                )
            )
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(Path(tmp) / "audit.db"),
                core_factory=lambda: core,
            )

            interface.handle_update(_message_update(user_id=100, text="write files"))

            text = api.sent_messages[0]["text"]
            self.assertIn("Files written.", text)
            self.assertIn("Tool: files.write_many", text)
            self.assertIn("Files: 2", text)
            self.assertIn("- src/app.tsx", text)
            self.assertNotIn('"files"', text)
            self.assertNotIn("{", text)

    def test_approval_required_adds_inline_buttons(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(
                AgentResponse(
                    status="approval_required",
                    message="Tool manifest requires approval.",
                    data={"approval_id": 7, "tool": "memory.write"},
                )
            )
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(Path(tmp) / "audit.db"),
                core_factory=lambda: core,
            )

            interface.handle_update(_message_update(user_id=100, text="remember this"))

            reply_markup = api.sent_messages[0]["reply_markup"]
            self.assertIsNotNone(reply_markup)
            buttons = reply_markup["inline_keyboard"][0]
            self.assertEqual(buttons[0]["callback_data"], "approve:7")
            self.assertEqual(buttons[1]["callback_data"], "deny:7")
            self.assertNotIn('"approval_id"', api.sent_messages[0]["text"])

    def test_approvals_command_is_readable_without_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(AgentResponse(status="ok", message="unused"))
            audit = AuditStore(Path(tmp) / "audit.db")
            approval_id = audit.record_approval(
                tool="files.write",
                risk=3,
                args={"path": "README.md"},
                request_reason="write requested",
                decision_reason="requires approval",
            )
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=audit,
                core_factory=lambda: core,
            )

            interface.handle_update(_message_update(user_id=100, text="/approvals"))

            text = api.sent_messages[0]["text"]
            self.assertIn("Pending approvals:", text)
            self.assertIn(f"#{approval_id}: files.write", text)
            self.assertNotIn('"args"', text)
            self.assertNotIn("{", text)

    def test_approve_callback_executes_stored_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(AgentResponse(status="ok", message="unused"))
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(Path(tmp) / "audit.db"),
                core_factory=lambda: core,
            )

            interface.handle_update(_callback_update(user_id=100, data="approve:42"))

            self.assertEqual(core.approved, [42])
            self.assertIn("Approval executed", api.callback_answers[0]["text"])
            self.assertEqual(api.sent_messages[0]["text"], "Approval executed.")

    def test_reminder_done_callback_completes_notified_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _FakePaths(root)
            api = FakeTelegramAPI()
            created = create_reminder(
                {
                    "title": "Check tickets",
                    "due_at": "2026-05-04T12:00:00+00:00",
                },
                ToolRuntimeContext(
                    memory_root=paths.memory,
                    workspace_root=paths.workspace,
                    config={},
                    config_path=paths.config,
                ),
            )
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(paths.audit_db),
                core_factory=lambda: FakeCore(AgentResponse(status="ok", message="unused")),
                paths=paths,
            )

            interface.handle_update(
                _callback_update(user_id=100, data=f"reminder:done:{created['reminder']['id']}")
            )
            listed = list_reminders(
                {"status": "all"},
                ToolRuntimeContext(
                    memory_root=paths.memory,
                    workspace_root=paths.workspace,
                    config={},
                    config_path=paths.config,
                ),
            )["reminders"]

        self.assertIn("completed", api.callback_answers[0]["text"].lower())
        self.assertEqual(api.sent_messages[0]["text"], "Reminder completed.")
        self.assertEqual(listed[0]["status"], "completed")

    def test_reminder_snooze_callback_moves_due_time_back_to_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = _FakePaths(root)
            context = ToolRuntimeContext(
                memory_root=paths.memory,
                workspace_root=paths.workspace,
                config={},
                config_path=paths.config,
            )
            api = FakeTelegramAPI()
            created = create_reminder(
                {
                    "title": "Stretch",
                    "due_at": "2026-05-04T12:00:00+00:00",
                },
                context,
            )
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=True,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(paths.audit_db),
                core_factory=lambda: FakeCore(AgentResponse(status="ok", message="unused")),
                paths=paths,
            )

            interface.handle_update(
                _callback_update(user_id=100, data=f"reminder:snooze:{created['reminder']['id']}:300")
            )
            listed = list_reminders({"status": "all"}, context)["reminders"]

        self.assertIn("snoozed", api.callback_answers[0]["text"].lower())
        self.assertEqual(listed[0]["status"], "pending")
        self.assertEqual(listed[0]["snooze_seconds"], 300)

    def test_disabled_interface_does_not_route_messages(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            api = FakeTelegramAPI()
            core = FakeCore(AgentResponse(status="ok", message="unused"))
            interface = TelegramInterface(
                settings=TelegramSettings(
                    enabled=False,
                    allowed_user_ids=frozenset({100}),
                ),
                api=api,
                audit_store=AuditStore(Path(tmp) / "audit.db"),
                core_factory=lambda: core,
            )

            interface.handle_update(_message_update(user_id=100, text="hi"))

            self.assertEqual(core.messages, [])
            self.assertIn("disabled locally", api.sent_messages[0]["text"])

    def test_token_helpers_store_and_load_from_process_environment(self) -> None:
        settings = TelegramSettings(
            enabled=True,
            allowed_user_ids=frozenset({100}),
            bot_token_env="DMDAGENT_TEST_TELEGRAM_TOKEN",
        )
        os.environ.pop(settings.bot_token_env, None)

        storage = store_telegram_token(settings, "secret-token")

        self.assertEqual(storage, "process environment")
        self.assertEqual(os.environ[settings.bot_token_env], "secret-token")
        self.assertEqual(load_telegram_token(settings), "secret-token")
        os.environ.pop(settings.bot_token_env, None)
        self.assertIsNone(load_telegram_token(settings))


def _message_update(user_id: int, text: str) -> dict[str, Any]:
    return {
        "update_id": 1,
        "message": {
            "message_id": 10,
            "from": {"id": user_id},
            "chat": {"id": user_id},
            "text": text,
        },
    }


def _callback_update(user_id: int, data: str) -> dict[str, Any]:
    return {
        "update_id": 2,
        "callback_query": {
            "id": "callback-id",
            "from": {"id": user_id},
            "message": {
                "message_id": 11,
                "chat": {"id": user_id},
            },
            "data": data,
        },
    }


class _FakePaths:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = root / "config.yaml"
        self.memory = root / "memory"
        self.workspace = root / "workspace"
        self.audit_db = root / "audit.db"
        self.ensure()
        self.config.write_text("llm:\n  provider: ollama\n", encoding="utf-8")

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.memory.mkdir(parents=True, exist_ok=True)
        self.workspace.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    unittest.main()
