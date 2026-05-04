import tempfile
import unittest
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
            profile = (root / "memory" / "profile.md").read_text(encoding="utf-8")
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
            self.assertEqual(approval["args"]["path"], "facts/personal.md")
            self.assertIn("I like ice cream", approval["args"]["body"])

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
