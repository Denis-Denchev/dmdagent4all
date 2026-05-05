import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from dmdagent4all.agent import AgentCore
from dmdagent4all.agent.planner import PlanResult
from dmdagent4all.audit import AuditStore
from dmdagent4all.memory import MemoryManager
from dmdagent4all.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdagent4all.tools import build_builtin_registry
from dmdagent4all.tools.base import ToolRuntimeContext


class FakePlanner:
    def __init__(self, result: PlanResult) -> None:
        self.result = result

    def plan(self, **kwargs) -> PlanResult:
        del kwargs
        return self.result


class ExplodingPlanner:
    def plan(self, **kwargs) -> PlanResult:
        del kwargs
        raise AssertionError("planner should not be called")


class AnsweringPlanner(FakePlanner):
    def __init__(self, result: PlanResult, answer: str) -> None:
        super().__init__(result)
        self.answer_text = answer

    def answer(self, **kwargs) -> str:
        del kwargs
        return self.answer_text


class AgentCoreTest(unittest.TestCase):
    def test_planner_final_response_is_returned(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                FakePlanner(PlanResult(final_message="Hello from local model")),
            )
            response = core.handle_text("tell me one short sentence")
            self.assertEqual(response.status, "ok")
            self.assertEqual(response.message, "Hello from local model")

    def test_planner_tool_request_runs_through_permission_engine(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.list",
                            args={},
                            reason="User asked for memory files.",
                        )
                    )
                ),
            )
            response = core.handle_text("show memory")
            self.assertEqual(response.status, "ok")
            self.assertIn("files", response.data)

    def test_obvious_memory_list_request_does_not_need_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), ExplodingPlanner())
            response = core.handle_text("Show my local memory files")
            self.assertEqual(response.status, "ok")
            self.assertIn("files", response.data)

    def test_help_request_does_not_need_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), ExplodingPlanner())
            response = core.handle_text("Hello, what can you do?")
            self.assertEqual(response.status, "ok")
            self.assertIn("permission engine", response.message)

    def test_bulgarian_greeting_does_not_echo_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), ExplodingPlanner())
            response = core.handle_text("как си")
            self.assertEqual(response.status, "ok")
            self.assertNotEqual(response.message, "как си")
            self.assertIn("permission engine", response.message)

    def test_browser_request_routes_to_policy_before_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), ExplodingPlanner())
            response = core.handle_text("може ли да отвориш гугъл")
            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: browser.open", response.message)

    def test_identity_answers_use_setup_config_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                ExplodingPlanner(),
                config={
                    "llm": {"response_language": "auto"},
                    "setup": {
                        "agent_name": "jarvis",
                        "user_name": "Denis",
                        "preferred_language": "auto",
                    },
                },
            )
            who_are_you = core.handle_text("who are you")
            who_am_i = core.handle_text("who am i")
            self.assertEqual(who_are_you.status, "ok")
            self.assertIn("jarvis", who_are_you.message)
            self.assertEqual(who_am_i.message, "You are Denis.")

    def test_identity_changes_update_config_and_memory_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "llm": {"response_language": "auto"},
                "setup": {
                    "agent_name": "DMD Agent",
                    "user_name": "",
                    "preferred_language": "auto",
                },
            }
            core = _build_core(root, ExplodingPlanner(), config=config)

            agent_response = core.handle_text("call yourself Jarvis")
            user_response = core.handle_text("my name is Denis")

            self.assertEqual(agent_response.status, "ok")
            self.assertEqual(user_response.status, "ok")
            self.assertEqual(config["setup"]["agent_name"], "Jarvis")
            self.assertEqual(config["setup"]["user_name"], "Denis")
            profile = (root / "memory" / "long-term" / "profile.md").read_text(encoding="utf-8")
            self.assertIn("Assistant name: Jarvis", profile)
            self.assertIn("User name: Denis", profile)

    def test_short_im_name_phrase_updates_identity_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "llm": {"response_language": "auto"},
                "setup": {
                    "agent_name": "DMD Agent",
                    "user_name": "",
                    "preferred_language": "auto",
                },
            }
            core = _build_core(root, ExplodingPlanner(), config=config)

            response = core.handle_text("im Denis")

            self.assertEqual(response.status, "ok")
            self.assertEqual(config["setup"]["user_name"], "Denis")

    def test_simple_terminal_phrase_routes_to_terminal_policy_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), ExplodingPlanner())

            response = core.handle_text("tell me can you type ls in terminal")

            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: terminal.run", response.message)

    def test_bulgarian_terminal_ls_phrase_runs_allowlisted_command_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "marker.txt").write_text("ok\n", encoding="utf-8")
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["ls"]],
                        "auto_approve_allowlisted": True,
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_text("Отвори терминала и напиши ls върни ми резултат")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.data["command"], ["ls"])
            self.assertIn("marker.txt", response.data["stdout"])

    def test_open_terminal_phrase_does_not_create_unrunnable_terminal_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("run open new terminal")

            self.assertEqual(response.status, "ok")
            self.assertIn("cannot open", response.message)
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_non_allowlisted_terminal_command_is_denied_before_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["ls"]],
                        "auto_approve_allowlisted": False,
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_tool_request(
                ToolRequest(tool="terminal.run", args={"command": ["open", "new", "terminal"]})
            )

            self.assertEqual(response.status, "denied")
            self.assertIn("allowlist", response.message)
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_approval_required_is_persisted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.write",
                            args={"path": "profile.md", "body": "test"},
                            reason="User asked to save memory.",
                        )
                    )
                ),
                audit,
            )
            response = core.handle_text("remember this")
            self.assertEqual(response.status, "approval_required")
            approvals = audit.list_approvals(status="pending")
            self.assertEqual(len(approvals), 1)
            self.assertEqual(approvals[0]["tool"], "memory.write")

    def test_approved_memory_write_executes_pending_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.write",
                            args={"path": "facts/test.md", "body": "Approved fact"},
                            reason="User asked to save memory.",
                        )
                    )
                ),
                audit,
            )
            pending = core.handle_text("remember this")
            approval_id = pending.data["approval_id"]
            approved = core.approve_and_execute(approval_id)
            self.assertEqual(approved.status, "ok")
            self.assertIn("facts/test.md", approved.data["path"])
            self.assertEqual(audit.get_approval(approval_id)["status"], "executed")
            self.assertTrue((root / "memory" / "facts" / "test.md").exists())

    def test_explicit_remember_requires_real_memory_write_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Remember that I like ice cream")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "memory.write")
            self.assertEqual(approval["args"]["path"], "long-term/facts/personal.md")
            self.assertIn("I like ice cream", approval["args"]["body"])

    def test_short_term_remember_phrase_requires_approval_and_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("запомни временно че работя по Stripe bug за 48 часа")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "memory.write")
            self.assertEqual(approval["args"]["memory_scope"], "short-term")
            self.assertEqual(approval["args"]["ttl_hours"], 48)
            self.assertIn("Stripe bug", approval["args"]["body"])

    def test_natural_remember_phrases_require_real_memory_write_approval(self) -> None:
        examples = [
            ("And also remember that I like BMW cars", "I like BMW cars"),
            ("I like BMW cars remember that", "I like BMW cars"),
            ("Can you remember that I have Lenovo servers at home", "I have Lenovo servers at home"),
        ]
        for message, expected_fact in examples:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    audit = AuditStore(root / "audit.db")
                    core = _build_core(
                        root,
                        ExplodingPlanner(),
                        audit,
                        config={"llm": {"provider": "ollama", "response_language": "auto"}},
                    )

                    response = core.handle_text(message)

                    self.assertEqual(response.status, "approval_required")
                    approval = audit.list_approvals(status="pending")[0]
                    self.assertEqual(approval["tool"], "memory.write")
                    self.assertIn(expected_fact, approval["args"]["body"])

    def test_bulgarian_new_long_term_memory_file_request_uses_auto_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text(
                "Моля те да запомниш в нов .md дългосрочно че имам два сървъра Lenovo m700s"
            )

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "memory.write")
            self.assertEqual(approval["args"]["path"], "auto")
            self.assertEqual(approval["args"]["memory_scope"], "long-term")
            self.assertEqual(approval["args"]["body"], "имам два сървъра Lenovo m700s")

    def test_bulgarian_memory_recall_answers_from_local_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "long-term/facts/computers.md",
                "Имам два сървъра Lenovo m700s.",
                metadata={"type": "long_term_note", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Какви сървъри имам")

            self.assertEqual(response.status, "ok")
            self.assertIn("Имаш два сървъра Lenovo m700s", response.message)
            self.assertEqual(response.data, {"planner": "deterministic"})

    def test_broad_bulgarian_memory_recall_lists_saved_facts_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "long-term/facts/computers.md",
                "Имам два сървъра Lenovo m700s.",
                metadata={"type": "long_term_note", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Искам да ми кажеш всичко което знаеш за мен")

            self.assertEqual(response.status, "ok")
            self.assertIn("В локалната long-term memory знам това:", response.message)
            self.assertIn("Имаш два сървъра Lenovo m700s", response.message)

    def test_remind_me_phrase_creates_approval_gated_local_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Remind me that I have a dentist appointment after 1 hour")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "reminders.create")
            self.assertIn("dentist appointment", approval["args"]["title"])
            self.assertIn("due_at", approval["args"])

    def test_bulgarian_short_timer_can_place_action_after_delay(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )
            before = datetime.now().astimezone()

            response = core.handle_text("Сложих да се вари яйце напомни ми след 8 минути да го извадя")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            due_at = datetime.fromisoformat(approval["args"]["due_at"])
            self.assertEqual(approval["tool"], "reminders.create")
            self.assertIn("извадя", approval["args"]["title"])
            self.assertGreaterEqual(due_at, before + timedelta(minutes=7, seconds=50))

    def test_bulgarian_event_reminder_can_notify_at_time_on_event_day(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text(
                "запомни и ми напомни че съм на зъболекар след 2 дни в 14:00 "
                "искам да ми напомниш в 12:00 в деня за часа"
            )

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            event_at = datetime.fromisoformat(approval["args"]["event_at"])
            due_at = datetime.fromisoformat(approval["args"]["due_at"])
            self.assertEqual(approval["tool"], "reminders.create")
            self.assertIn("зъболекар", approval["args"]["title"])
            self.assertEqual((event_at.hour, event_at.minute), (14, 0))
            self.assertEqual((due_at.hour, due_at.minute), (12, 0))
            self.assertEqual(event_at.date(), due_at.date())

    def test_bulgarian_event_reminder_can_notify_before_event(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("след 2 месеца съм на концерт напомни ми 1 ден преди това")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            event_at = datetime.fromisoformat(approval["args"]["event_at"])
            due_at = datetime.fromisoformat(approval["args"]["due_at"])
            self.assertEqual(approval["tool"], "reminders.create")
            self.assertIn("концерт", approval["args"]["title"])
            self.assertEqual(event_at - due_at, timedelta(days=1))
            self.assertEqual(approval["args"]["remind_before"], "1 day(s) before")

    def test_bulgarian_address_reminder_extracts_location_and_maps_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text(
                "Напомни ми че след 1 минута трябва да тръгна за адрес булевард България номер 1"
            )
            approval = audit.list_approvals(status="pending")[0]
            approved = core.approve_and_execute(response.data["approval_id"])

            self.assertEqual(response.status, "approval_required")
            self.assertEqual(approval["args"]["title"], "трябва да тръгна")
            self.assertEqual(approval["args"]["location"], "булевард България No. 1")
            self.assertEqual(approved.status, "ok")
            self.assertIn("google.com/maps", approved.data["reminder"]["action_url"])

    def test_reminder_prefers_planner_enrichment_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="reminders.create",
                            args={
                                "title": "зъболекар",
                                "due_at": "2026-05-06T12:00:00-07:00",
                                "event_at": "2026-05-06T14:00:00-07:00",
                                "location": "бул. България 10",
                                "action_url": "https://www.google.com/maps/search/?api=1&query=бул.%20България%2010",
                            },
                            reason="Use saved dentist location from memory.",
                        )
                    )
                ),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("напомни ми за зъболекаря след 2 дни")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "reminders.create")
            self.assertEqual(approval["args"]["location"], "бул. България 10")
            self.assertIn("maps", approval["args"]["action_url"])

    def test_approved_reminder_is_saved_locally(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            pending = core.handle_text("Remind me to call the dentist in 1 hour")
            approved = core.approve_and_execute(pending.data["approval_id"])
            listed = core.handle_tool_request(ToolRequest(tool="reminders.list", args={}))

            self.assertEqual(approved.status, "ok")
            self.assertEqual(listed.status, "ok")
            self.assertEqual(listed.data["reminders"][0]["status"], "pending")
            self.assertIn("call the dentist", listed.data["reminders"][0]["title"])

    def test_ready_text_approves_latest_pending_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            pending = core.handle_text("Remind me to start work in 15 minutes")
            approved = core.handle_text("готово")
            listed = core.handle_tool_request(ToolRequest(tool="reminders.list", args={}))

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approved.status, "ok")
            self.assertEqual(audit.list_approvals(limit=1)[0]["status"], "executed")
            self.assertIn("start work", listed.data["reminders"][0]["title"])

    def test_auto_approve_allowlisted_terminal_command_runs_without_pending_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ExplodingPlanner(),
                audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["pwd"]],
                        "auto_approve_allowlisted": True,
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_tool_request(ToolRequest(tool="terminal.run", args={"command": ["pwd"]}))

            self.assertEqual(response.status, "ok")
            self.assertEqual(audit.list_approvals(status="pending"), [])
            self.assertIn(str(root / "workspace"), response.data["stdout"])

    def test_terminal_workspace_root_can_point_at_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            (project / "README.md").write_text("Project readme\n", encoding="utf-8")
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "workspace_root": str(project),
                        "allowed_commands": [["cat", "README.md"]],
                        "auto_approve_allowlisted": True,
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_text("type cat README.md")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.data["cwd"], str(project.resolve()))
            self.assertEqual(response.data["stdout"], "Project readme\n")

    def test_bulgarian_identity_uses_saved_nickname_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "llm": {"response_language": "auto"},
                "setup": {
                    "agent_name": "jarvis",
                    "user_name": "Denis",
                    "preferred_language": "auto",
                },
            }
            core = _build_core(root, ExplodingPlanner(), config=config)

            saved = core.handle_text(
                "i want you to call me buddy or if i type in Bulgarian you can also call me маняк"
            )
            answered = core.handle_text("аз кой съм")

            self.assertEqual(saved.status, "ok")
            self.assertEqual(config["setup"]["nickname"], "buddy")
            self.assertEqual(config["setup"]["nickname_bg"], "маняк")
            self.assertEqual(answered.status, "ok")
            self.assertIn("маняк", answered.message)
            self.assertIn("Denis", answered.message)

    def test_memory_tool_result_can_be_synthesized_into_human_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- Denis likes ice cream.",
                metadata={"type": "personal_fact"},
            )
            core = _build_core(
                root,
                AnsweringPlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.list",
                            args={},
                            reason="Look up memory.",
                        )
                    ),
                    "You like ice cream.",
                ),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("What do I like?")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.message, "You like ice cream.")

    def test_approved_cloud_memory_read_is_synthesized_into_human_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- Denis likes ice cream.",
                metadata={"type": "personal_fact"},
            )
            core = _build_core(
                root,
                AnsweringPlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.list",
                            args={},
                            reason="Look up memory.",
                        )
                    ),
                    "You like ice cream.",
                ),
                audit=audit,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            pending = core.handle_text("What do I like?")
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approved.status, "ok")
            self.assertEqual(approved.message, "You like ice cream.")

    def test_cloud_memory_approval_applies_for_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- Denis likes ice cream.",
                metadata={"type": "personal_fact"},
            )
            core = _build_core(
                root,
                AnsweringPlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.list",
                            args={},
                            reason="Look up memory.",
                        )
                    ),
                    "You like ice cream.",
                ),
                audit=audit,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            pending = core.handle_text("What do I like?")
            core.approve_and_execute(pending.data["approval_id"])
            second = core.handle_text("What do I like?")

            self.assertEqual(second.status, "ok")
            self.assertEqual(second.message, "You like ice cream.")
            self.assertEqual(len(audit.list_approvals(status="pending")), 0)


def _build_core(
    root: Path,
    planner,
    audit: AuditStore | None = None,
    config: dict | None = None,
    permission_context: PermissionContext | None = None,
) -> AgentCore:
    registry = build_builtin_registry()
    runtime_config = config or {"llm": {"response_language": "auto"}}
    return AgentCore(
        permission_engine=PermissionEngine(registry.manifests),
        tool_registry=registry,
        permission_context=permission_context or PermissionContext(),
        runtime_context=ToolRuntimeContext(
            memory_root=root / "memory",
            workspace_root=root / "workspace",
            config=runtime_config,
        ),
        audit_store=audit or AuditStore(root / "audit.db"),
        planner=planner,
    )


if __name__ == "__main__":
    unittest.main()
