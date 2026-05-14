import tempfile
import unittest
import os
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest import mock

from dmdagent4all.agent import AgentCore
from dmdagent4all.agent.planner import PlanResult, PlannerError
from dmdagent4all.agent.router import ConversationRouter
from dmdagent4all.audit import AuditStore
from dmdagent4all.memory import MemoryManager
from dmdagent4all.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdagent4all.tools import build_builtin_registry
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.web import FetchedPage


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


class FailingPlanner:
    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def plan(self, **kwargs) -> PlanResult:
        del kwargs
        raise self.exc


class AnsweringPlanner(FakePlanner):
    def __init__(self, result: PlanResult, answer: str) -> None:
        super().__init__(result)
        self.answer_text = answer

    def answer(self, **kwargs) -> str:
        del kwargs
        return self.answer_text


class RecordingPlanner:
    def __init__(self, answer: str = "noted") -> None:
        self.answer = answer
        self.calls: list[dict] = []

    def plan(self, **kwargs) -> PlanResult:
        self.calls.append(kwargs)
        return PlanResult(final_message=self.answer)


class ChatOnlyPlanner:
    def __init__(self, answer: str) -> None:
        self.answer = answer
        self.chat_calls: list[dict] = []

    def plan(self, **kwargs) -> PlanResult:
        del kwargs
        raise AssertionError("planner JSON mode should not be called")

    def chat(self, **kwargs) -> str:
        self.chat_calls.append(kwargs)
        return self.answer


class ContextAnsweringPlanner:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def plan(self, **kwargs) -> PlanResult:
        self.calls.append(kwargs)
        context = kwargs.get("agent_context") or {}
        text = str(kwargs.get("user_message") or "").casefold()
        if "model" in text:
            runtime = context.get("runtime") or {}
            return PlanResult(
                final_message=(
                    f"Provider: {runtime.get('provider', '')}; "
                    f"model: {runtime.get('model', '')}"
                )
            )
        if "tools" in text:
            tools = context.get("tools") or {}
            return PlanResult(
                final_message=(
                    f"Enabled: {', '.join(tools.get('enabled') or [])}\n"
                    f"Disabled: {', '.join(tools.get('disabled') or [])}"
                )
            )
        if "downloads" in text or "download" in text:
            workspace = context.get("workspace") or {}
            mounts = workspace.get("mounted_roots") or []
            downloads = next(
                (
                    item
                    for item in mounts
                    if isinstance(item, dict) and item.get("label") == "downloads"
                ),
                {},
            )
            if not downloads.get("allowed"):
                return PlanResult(
                    final_message=(
                        "Downloads is not mounted under an allowed root. Configure "
                        "storage.downloads_root or workspace.allowed_roots before I can inspect it."
                    )
                )
            return PlanResult(final_message=f"Downloads mounted at {downloads.get('path', '')}")
        return PlanResult(final_message="ok")


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

    def test_recent_conversation_is_passed_to_local_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = RecordingPlanner(answer="разбрах")
            core = _build_core(
                Path(tmp),
                planner,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            core.handle_text("Говорим за настройката на memory organizer", session_id="test")
            core.handle_text("Какво казах преди малко?", session_id="test")

            self.assertGreaterEqual(len(planner.calls), 2)
            self.assertIn("Говорим за настройката", planner.calls[1]["conversation_context"])
            self.assertIn("Assistant: разбрах", planner.calls[1]["conversation_context"])

    def test_agent_context_passed_to_planner_includes_runtime_workspace_tools_and_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            downloads = root / "Downloads"
            workspace.mkdir()
            downloads.mkdir()
            MemoryManager(root / "memory").write(
                "services/platform.md",
                "Platform context file.",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                root,
                planner,
                config={
                    "setup": {"agent_name": "DMD Runtime"},
                    "llm": {
                        "provider": "ollama",
                        "model": "qwen-context",
                        "planner_model": "qwen-planner",
                        "response_language": "auto",
                    },
                    "storage": {"downloads_root": str(downloads)},
                    "workspace": {
                        "default_path": str(workspace),
                        "current_path": str(workspace),
                        "allowed_roots": [str(root)],
                        "blocked_paths": [],
                    },
                    "tools": {"browser.scrape_markdown": {"enabled": True}},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["ls"], ["git", "status"]],
                        "auto_approve_allowlisted": True,
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"gmail.create_draft"}),
                    granted_permissions=frozenset({"browser.read", "gmail.compose"}),
                ),
            )

            core.handle_text("what model and tools are configured?")
            context = planner.calls[0]["agent_context"]

            self.assertEqual(context["agent_name"], "DMD Runtime")
            self.assertEqual(context["runtime"]["provider"], "ollama")
            self.assertEqual(context["runtime"]["model"], "qwen-context")
            self.assertEqual(context["runtime"]["planner_model"], "qwen-planner")
            self.assertEqual(context["workspace"]["current"], str(workspace.resolve()))
            self.assertIn(str(root.resolve()), context["workspace"]["allowed_roots"])
            self.assertIn("services/platform.md", context["memory"]["files"])
            self.assertIn("browser.scrape_markdown", context["tools"]["enabled"])
            self.assertIn("gmail.create_draft", context["tools"]["enabled"])
            self.assertIn("browser.read", context["permissions"]["granted"])
            self.assertIn("ls", context["terminal"]["allowed_commands"])
            self.assertTrue(
                any(
                    mount["label"] == "downloads" and mount["allowed"]
                    for mount in context["workspace"]["mounted_roots"]
                )
            )

    def test_local_dev_autonomy_context_includes_runtime_locations_and_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "Downloads"
            downloads.mkdir()
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                root,
                planner,
                config={
                    "llm": {"provider": "deepseek", "model": "deepseek-chat", "response_language": "auto"},
                    "storage": {"downloads_root": str(downloads)},
                    "tools": {"browser.scrape_markdown": {"enabled": False}},
                    "workspace": {
                        "default_path": str(root / "workspace"),
                        "current_path": str(root / "workspace"),
                        "allowed_roots": [str(root)],
                        "blocked_paths": [],
                    },
                },
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                core.handle_text("what tools are enabled?")

            context = planner.calls[0]["agent_context"]
            self.assertTrue(context["autonomy"]["local_dev_autonomy"])
            self.assertEqual(context["runtime"]["provider"], "deepseek")
            self.assertEqual(context["workspace"]["downloads_path"], str(downloads.resolve()))
            self.assertIn(str((root / "memory").resolve()), context["memory"]["locations"])
            self.assertIn("sections", context["config"])
            self.assertIn("browser.scrape_markdown", context["tools"]["enabled"])
            self.assertIn("browser.read", context["permissions"]["granted"])
            self.assertEqual(context["runtime_state"]["mode"], "full_llm_first_autonomy")
            self.assertEqual(context["runtime_state"]["downloads"]["path"], str(downloads.resolve()))
            self.assertIn("browser.scrape_markdown", context["runtime_state"]["tools"]["enabled"])

    def test_autonomy_config_toggle_can_disable_env_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                root,
                planner,
                config={
                    "runtime": {"autonomy": {"enabled": False, "mode": "standard"}},
                    "tools": {"browser.scrape_markdown": {"enabled": False}},
                },
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                core.handle_text("what tools are enabled?")

            context = planner.calls[0]["agent_context"]
            self.assertFalse(context["autonomy"]["enabled"])
            self.assertEqual(context["runtime_state"]["mode"], "standard_safe")
            self.assertIn("browser.scrape_markdown", context["tools"]["disabled"])

    def test_model_identity_uses_runtime_agent_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = ContextAnsweringPlanner()
            core = _build_core(
                Path(tmp),
                planner,
                config={
                    "llm": {
                        "provider": "deepseek",
                        "model": "deepseek-chat",
                        "response_language": "auto",
                    }
                },
            )

            response = core.handle_text("what model are you?")

            self.assertEqual(response.status, "ok")
            self.assertIn("deepseek", response.message)
            self.assertIn("deepseek-chat", response.message)
            self.assertEqual(planner.calls[0]["agent_context"]["runtime"]["model"], "deepseek-chat")

    def test_tools_question_uses_tool_registry_state_from_agent_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = ContextAnsweringPlanner()
            core = _build_core(
                Path(tmp),
                planner,
                config={
                    "llm": {"provider": "ollama", "model": "qwen3:8b", "response_language": "auto"},
                    "tools": {
                        "browser.scrape_markdown": {"enabled": True},
                        "terminal.run": {"enabled": False},
                    },
                },
            )

            response = core.handle_text("what tools do you have?")

            self.assertEqual(response.status, "ok")
            self.assertIn("browser.scrape_markdown", response.message)
            self.assertIn("terminal.run", response.message)
            self.assertIn("terminal.run", planner.calls[0]["agent_context"]["tools"]["disabled"])

    def test_downloads_unavailable_context_answer_is_clear(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside_downloads = root.parent / f"{root.name}-outside-downloads"
            planner = ContextAnsweringPlanner()
            core = _build_core(
                root,
                planner,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "storage": {"downloads_root": str(outside_downloads)},
                    "workspace": {
                        "default_path": str(root / "workspace"),
                        "current_path": str(root / "workspace"),
                        "allowed_roots": [str(root)],
                        "blocked_paths": [],
                    },
                },
            )

            response = core.handle_text("look in Downloads for the football file")

            self.assertEqual(response.status, "ok")
            self.assertIn("Downloads is not mounted", response.message)
            self.assertIn("allowed root", response.message)
            downloads = next(
                mount
                for mount in planner.calls[0]["agent_context"]["workspace"]["mounted_roots"]
                if mount["label"] == "downloads"
            )
            self.assertFalse(downloads["allowed"])

    def test_memory_file_names_question_uses_agent_context_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/immich.md",
                "Immich memory file.",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(root, ExplodingPlanner())

            response = core.handle_text(
                "кажи ми в memory какви .md имаш, искам само имената на файловете"
            )

            self.assertEqual(response.status, "ok")
            self.assertIn("services/immich.md", response.message)
            self.assertNotIn("LLM planner", response.message)
            self.assertEqual(response.data["source"], "agent_context")
            self.assertEqual(response.data["kind"], "memory_files")

    def test_missing_tool_permissions_question_uses_agent_context_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), ExplodingPlanner())

            response = core.handle_text("кажи ми на кои тулове не сме дали разрешение")

            self.assertEqual(response.status, "ok")
            self.assertIn("terminal.run", response.message)
            self.assertIn("terminal.run", response.data["missing_permissions_by_tool"])
            self.assertNotIn("LLM planner", response.message)
            self.assertEqual(response.data["source"], "agent_context")
            self.assertEqual(response.data["kind"], "missing_tool_permissions")

    def test_scraped_downloads_question_uses_local_downloads_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            downloads = root / "internet-files"
            scrape_dir = downloads / "scrapefiles"
            scrape_dir.mkdir(parents=True)
            (scrape_dir / "fibank.md").write_text("# Fibank\n", encoding="utf-8")
            (scrape_dir / "sportal.md").write_text("# Sportal\n", encoding="utf-8")
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "storage": {"downloads_root": str(downloads)},
                },
            )

            response = core.handle_text("кажи какво имаме в downloads от скрейпнатите сайтове")

            self.assertEqual(response.status, "ok")
            self.assertIn("scrapefiles/fibank.md", response.message)
            self.assertIn("scrapefiles/sportal.md", response.message)
            self.assertNotIn("LLM planner", response.message)
            self.assertEqual(response.data["source"], "agent_context")
            self.assertEqual(response.data["kind"], "scraped_downloads")

    def test_cloud_planner_does_not_receive_chat_history_without_context_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                Path(tmp),
                planner,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            core.handle_text("private previous message", session_id="cloud")
            core.handle_text("what did I just say?", session_id="cloud")

            self.assertEqual(planner.calls[1]["conversation_context"], "")

    def test_cloud_planner_receives_chat_history_when_privacy_setting_allows_it(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                Path(tmp),
                planner,
                config={
                    "llm": {"provider": "openai", "response_language": "auto"},
                    "privacy": {"send_chat_history_to_cloud": True},
                },
                permission_context=PermissionContext(cloud_model_active=True),
            )

            core.handle_text("private previous message", session_id="cloud")
            core.handle_text("what did I just say?", session_id="cloud")

            self.assertIn("First user message in this session: private previous message", planner.calls[1]["conversation_context"])
            self.assertIn("User: private previous message", planner.calls[1]["conversation_context"])

    def test_first_message_question_is_answered_from_local_chat_history(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                Path(tmp),
                planner,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            core.handle_text("първият ми въпрос", session_id="history")
            response = core.handle_text("кой беше първия въпрос който те питах в този чат", session_id="history")

            self.assertEqual(response.status, "ok")
            self.assertIn("първият ми въпрос", response.message)
            self.assertEqual(response.data, {"planner": "deterministic", "source": "chat_history"})

    def test_missing_openai_key_acknowledgement_falls_back_without_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                FailingPlanner(RuntimeError("openai requires an API key in DMDAGENT_OPENAI_API_KEY.")),
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            response = core.handle_text("i know that")

            self.assertEqual(response.status, "ok")
            self.assertIn("Got it", response.message)
            self.assertEqual(response.data, {"planner": "deterministic", "fallback": "llm_unavailable"})

    def test_missing_openai_key_returns_actionable_message_instead_of_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                FailingPlanner(RuntimeError("openai requires an API key in DMDAGENT_OPENAI_API_KEY.")),
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            response = core.handle_text("summarize my current projects")

            self.assertEqual(response.status, "ok")
            self.assertIn("API key is not loaded", response.message)
            self.assertIn("DMDAGENT_OPENAI_API_KEY", response.message)
            self.assertEqual(response.data, {"planner": "deterministic", "fallback": "missing_api_key"})

    def test_planner_unavailable_mode_mentions_context_aware_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)

            response = core.handle_text("inspect mounted roots and decide the next workflow")

            self.assertEqual(response.status, "ok")
            self.assertIn("LLM planner is not active", response.message)
            self.assertIn("context-aware tasks", response.message)
            self.assertEqual(response.data, {"planner": "inactive"})

    def test_local_planner_receives_retrieved_memory_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "README.md",
                "\n".join(
                    [
                        "`homelab/homelab-primary.md` — Current primary server role, IPs, paths, services and ports.",
                        "`services/adguard.md` — AdGuard container, config path, ports and DNS exposure rules.",
                    ]
                ),
                metadata={"type": "memory_index", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "services/immich.md",
                "\n".join(
                    [
                        "# Immich",
                        "Important Immich lesson:",
                        "- Safe update pattern: cd ~/docker-data/immich && docker compose pull && docker compose up -d",
                    ]
                ),
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            planner = RecordingPlanner(answer="готово")
            core = _build_core(
                root,
                planner,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Направи ми кратък план за Immich update")

            self.assertEqual(response.status, "ok")
            self.assertIn("[services/immich.md]", planner.calls[0]["memory_context"])
            self.assertIn("docker compose pull", planner.calls[0]["memory_context"])

    def test_cloud_planner_gets_no_retrieved_memory_without_context_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/immich.md",
                "Safe update pattern: cd ~/docker-data/immich && docker compose pull && docker compose up -d",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                root,
                planner,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            core.handle_text("Направи ми кратък план за Immich update")

            self.assertEqual(planner.calls[0]["memory_context"], "")

    def test_cloud_planner_receives_retrieved_memory_after_context_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/immich.md",
                "Safe update pattern: cd ~/docker-data/immich && docker compose pull && docker compose up -d",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            planner = RecordingPlanner(answer="ok")
            core = _build_core(
                root,
                planner,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True, cloud_context_approved=True),
            )

            core.handle_text("Направи ми кратък план за Immich update")

            self.assertIn("[services/immich.md]", planner.calls[0]["memory_context"])
            self.assertIn("docker compose pull", planner.calls[0]["memory_context"])

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
            core = _build_core(Path(tmp), None)
            response = core.handle_text("Show my local memory files")
            self.assertEqual(response.status, "ok")
            self.assertIn("files", response.data)

    def test_help_request_does_not_need_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)
            response = core.handle_text("Hello, what can you do?")
            self.assertEqual(response.status, "ok")
            self.assertIn("permission engine", response.message)

    def test_bulgarian_greeting_does_not_echo_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)
            response = core.handle_text("как си")
            self.assertEqual(response.status, "ok")
            self.assertNotEqual(response.message, "как си")
            self.assertIn("permission engine", response.message)

    def test_browser_request_routes_to_policy_before_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)
            response = core.handle_text("може ли да отвориш гугъл")
            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: browser.open", response.message)

    def test_browser_scrape_request_routes_to_policy_before_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)
            response = core.handle_text(
                "събери информация от https://example.com за цените и запази в markdown"
            )
            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: browser.scrape_markdown", response.message)

    def test_llm_first_scrape_request_is_not_stolen_by_memory_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "preferences.md",
                "Preferences is on port 192.",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="browser.scrape_markdown",
                            args={"url": "sportal.bg", "instructions": "scrape sportal.bg"},
                            reason="User asked to scrape a web page.",
                        )
                    )
                ),
            )

            response = core.handle_text("i want you to scrape sportal.bg")

            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: browser.scrape_markdown", response.message)
            self.assertNotIn("Preferences", response.message)
            self.assertNotIn("port 192", response.message)

    def test_llm_first_open_domain_routes_to_browser_open_not_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "preferences.md",
                "Preferences is on port 192.",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="browser.open",
                            args={"url": "sportal.bg"},
                            reason="User asked to open a web page.",
                        )
                    )
                ),
            )

            response = core.handle_text("open sportal.bg")

            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: browser.open", response.message)
            self.assertNotIn("Preferences", response.message)

    def test_browser_scrape_chat_response_includes_count_path_and_preview(self) -> None:
        html = """
        <html>
          <head><title>Daily Example</title></head>
          <body>
            <main>
              <article>
                <h2><a href="/news/first-real-story-1001">First real article title today</a></h2>
                <p>First article summary.</p>
              </article>
              <article>
                <h2><a href="/news/second-real-story-1002">Second real article headline here</a></h2>
                <p>Second article summary.</p>
              </article>
            </main>
          </body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="browser.scrape_markdown",
                            args={
                                "url": "https://news.example.test",
                                "mode": "targeted",
                                "content_type": "articles",
                                "limit": 2,
                                "format": "clean_markdown",
                            },
                            reason="User asked to scrape two articles.",
                        )
                    )
                ),
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"browser.scrape_markdown"}),
                    granted_permissions=frozenset({"browser.read"}),
                ),
            )
            page = FetchedPage(
                url="https://news.example.test",
                final_url="https://news.example.test/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=1024,
                truncated=False,
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page):
                response = core.handle_text("scrape first 2 articles from news.example.test")

            self.assertEqual(response.status, "ok")
            self.assertIn("Extracted 2 article", response.message)
            self.assertIn("scrapefiles/", response.message)
            self.assertIn("First real article title today", response.message)
            self.assertIn("Second real article headline here", response.message)

    def test_llm_memory_search_decision_can_still_answer_plex(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/plex.md",
                "Plex port: 32400",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "homelab/homelab-primary.md",
                "LAN IP: 192.0.2.43\nTailscale IP: 198.51.100.43",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "homelab/proxmox.md",
                "Proxmox web UI: https://192.0.2.25:8006",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                FakePlanner(PlanResult(memory_query="на кой адрес е plex")),
            )
            core.chat_history.append("default", "user", "а проксмокс на кой адрес беше")
            core.chat_history.append("default", "assistant", "Proxmox web UI: https://192.0.2.25:8006")

            response = core.handle_text("на кой адрес е plex")

            self.assertEqual(response.status, "ok")
            self.assertIn("Plex", response.message)
            self.assertIn("http://192.0.2.43:32400/web", response.message)
            self.assertNotIn("Proxmox", response.message)
            self.assertEqual(response.data, {"planner": "llm", "decision": "memory_search"})

    def test_llm_multi_tool_plan_still_uses_backend_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "terminal": {
                    "enabled": True,
                    "allowed_commands": [["ls"]],
                    "auto_approve_allowlisted": True,
                    "workspace_root": str(workspace),
                },
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="terminal.run",
                                args={"command": ["mkdir", "test1"]},
                                reason="Create requested folder.",
                            ),
                            ToolRequest(
                                tool="workspace.switch",
                                args={"path": "test1"},
                                reason="Move into the new folder.",
                            ),
                        )
                    )
                ),
                audit=audit,
                config=config,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            pending = core.handle_text("mkdir test1 then open the folder")
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(audit.list_approvals(status="pending"), [])
            self.assertEqual(approved.status, "ok")
            self.assertTrue((workspace / "test1").exists())
            self.assertEqual(config["workspace"]["current_path"], str((workspace / "test1").resolve()))

    def test_scrape_to_file_plan_passes_markdown_to_files_write_approval(self) -> None:
        html = """
        <html>
          <head><title>DMD Flow</title></head>
          <body><main><h1>DMD Flow</h1><p>Planner regression content.</p></main></body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "readme123.md").write_text("old content\n", encoding="utf-8")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="browser.scrape_markdown",
                                args={
                                    "url": "dmdflow.com",
                                    "mode": "raw_page",
                                    "format": "clean_markdown",
                                    "instructions": "Scrape dmdflow.com",
                                },
                                reason="Scrape the requested page.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={
                                    "path": "readme123.md",
                                    "content_from_previous_step": True,
                                },
                                reason="Write the scraped Markdown to the requested file.",
                            ),
                        )
                    )
                ),
                audit=audit,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"browser.scrape_markdown", "files.write"}),
                    granted_permissions=frozenset({"browser.read"}),
                ),
            )
            page = FetchedPage(
                url="https://dmdflow.com",
                final_url="https://dmdflow.com/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=512,
                truncated=False,
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page):
                pending = core.handle_text("scrape dmdflow.com and place the results in readme123.md")

            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approval["tool"], "files.write")
            self.assertEqual(approval["args"]["path"], "readme123.md")
            self.assertTrue(approval["args"]["overwrite"])
            self.assertIn("Planner regression content", approval["args"]["content"])
            self.assertNotIn("content_from_previous_step", approval["args"])
            self.assertIn("write file readme123.md", pending.data["plan"])

    def test_scrape_to_file_approval_writes_target_under_workspace(self) -> None:
        html = """
        <html>
          <head><title>DMD Flow</title></head>
          <body><main><h1>DMD Flow</h1><p>Validated workspace write.</p></main></body>
        </html>
        """
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="browser.scrape_markdown",
                                args={"url": "https://dmdflow.com"},
                                reason="Scrape the requested page.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={"path": "readme123.md", "content_from_previous_step": True},
                                reason="Write the scraped Markdown.",
                            ),
                        )
                    )
                ),
                audit=audit,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"browser.scrape_markdown", "files.write"}),
                    granted_permissions=frozenset({"browser.read"}),
                ),
            )
            page = FetchedPage(
                url="https://dmdflow.com",
                final_url="https://dmdflow.com/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=512,
                truncated=False,
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page):
                pending = core.handle_text("scrape dmdflow.com and place the results in readme123.md")
            approved = core.approve_and_execute(pending.data["approval_id"])

            target = workspace / "readme123.md"
            self.assertEqual(approved.status, "ok")
            self.assertTrue(target.exists())
            self.assertEqual(approved.data["path"], str(target.resolve()))
            self.assertIn("Validated workspace write", target.read_text(encoding="utf-8"))

    def test_downloads_file_move_workflow_uses_multi_tool_plan_and_delete_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            downloads = root / "Downloads"
            workspace.mkdir()
            downloads.mkdir()
            source = downloads / "football-notes.md"
            source.write_text("Football file contents\n", encoding="utf-8")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="files.list",
                                args={"path": str(downloads)},
                                reason="Inspect mounted Downloads.",
                            ),
                            ToolRequest(
                                tool="files.read",
                                args={"path_from_previous_step_match": "football"},
                                reason="Read the matching football file.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={"path": "readme.md", "content_from_previous_step": True},
                                reason="Write the selected file contents to readme.md.",
                            ),
                            ToolRequest(
                                tool="files.delete",
                                args={"path_from_selected_step": True},
                                reason="Delete the original after the write is approved.",
                            ),
                        )
                    )
                ),
                audit=audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "storage": {"downloads_root": str(downloads)},
                    "workspace": {
                        "default_path": str(workspace),
                        "current_path": str(workspace),
                        "allowed_roots": [str(root)],
                        "blocked_paths": [],
                    },
                },
            )

            pending_write = core.handle_text(
                "виж в downloads имам един файл за футбол искам да го преместиш в readme.md"
            )
            approved_write = core.approve_and_execute(pending_write.data["approval_id"])
            approved_delete = core.approve_and_execute(approved_write.data["approval_id"])

            self.assertEqual(pending_write.status, "approval_required")
            self.assertEqual(pending_write.data["pending_tool"], "files.write")
            self.assertEqual(approved_write.status, "approval_required")
            self.assertEqual(approved_write.data["pending_tool"], "files.delete")
            self.assertTrue((workspace / "readme.md").exists())
            self.assertIn(
                "Football file contents",
                (workspace / "readme.md").read_text(encoding="utf-8"),
            )
            self.assertEqual(approved_delete.status, "ok")
            self.assertFalse(source.exists())

    def test_llm_file_to_email_workflow_creates_draft_then_requires_send_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            source = workspace / "report.md"
            source.write_text("Project report body\n", encoding="utf-8")
            env = {
                "DMDAGENT_GMAIL_USERNAME": "sender@example.com",
                "DMDAGENT_GMAIL_APP_PASSWORD": "app-password",
            }
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="files.read",
                                args={"path": "report.md"},
                                reason="Read the file to send.",
                            ),
                            ToolRequest(
                                tool="gmail.create_draft",
                                args={
                                    "to": "person@example.com",
                                    "subject": "Requested file",
                                    "body_from_previous_step": True,
                                },
                                reason="Create a local email draft with the file body.",
                            ),
                            ToolRequest(
                                tool="gmail.send_draft",
                                args={"draft_id_from_previous_step": True},
                                reason="Send only after approval.",
                            ),
                        )
                    )
                ),
                audit=audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "email": {"gmail": {"enabled": True}, "max_body_chars": 20000},
                    "workspace": {
                        "default_path": str(workspace),
                        "current_path": str(workspace),
                        "allowed_roots": [str(root)],
                        "blocked_paths": [],
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"gmail.create_draft", "gmail.send_draft"}),
                    granted_permissions=frozenset({"gmail.compose", "gmail.send"}),
                    approval_risk_threshold=3,
                ),
            )

            with mock.patch.dict(os.environ, env):
                response = core.handle_text("прати този файл на person@example.com по мейл")

            approval = audit.list_approvals(status="pending")[0]
            draft_id = str(approval["args"]["draft_id"])
            draft_path = root / "email-drafts" / "gmail" / f"{draft_id}.json"
            draft = json.loads(draft_path.read_text(encoding="utf-8"))

            self.assertEqual(response.status, "approval_required")
            self.assertEqual(response.data["pending_tool"], "gmail.send_draft")
            self.assertEqual(approval["tool"], "gmail.send_draft")
            self.assertEqual(draft["to"], ["person@example.com"])
            self.assertEqual(draft["subject"], "Requested file")
            self.assertEqual(draft["body"], "Project report body")

    def test_scrape_to_env_file_is_denied_by_path_policy(self) -> None:
        html = "<html><head><title>DMD Flow</title></head><body><p>Secret target test.</p></body></html>"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="browser.scrape_markdown",
                                args={"url": "https://dmdflow.com"},
                                reason="Scrape the requested page.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={"path": ".env", "content_from_previous_step": True},
                                reason="Write the scraped Markdown.",
                            ),
                        )
                    )
                ),
                audit=audit,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"browser.scrape_markdown", "files.write"}),
                    granted_permissions=frozenset({"browser.read"}),
                ),
            )
            page = FetchedPage(
                url="https://dmdflow.com",
                final_url="https://dmdflow.com/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=256,
                truncated=False,
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page):
                response = core.handle_text("scrape dmdflow.com and place the results in .env")

            self.assertEqual(response.status, "denied")
            self.assertIn("secret", response.message.casefold())
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_llm_selected_secret_read_and_write_are_blocked_by_backend_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / ".env").write_text("SECRET_TOKEN=abc\n", encoding="utf-8")
            read_core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="files.read",
                            args={"path": ".env"},
                            reason="LLM attempted to read a secret file.",
                        )
                    )
                ),
                audit=audit,
            )
            write_core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="files.write",
                            args={"path": ".env", "content": "new secret"},
                            reason="LLM attempted to write a secret file.",
                        )
                    )
                ),
                audit=audit,
            )

            read_response = read_core.handle_text("read .env")
            write_response = write_core.handle_text("write to .env")

            self.assertEqual(read_response.status, "denied")
            self.assertEqual(write_response.status, "denied")
            self.assertIn("secret", read_response.message.casefold())
            self.assertIn("secret", write_response.message.casefold())
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_emergency_stop_blocks_scrape_to_file_plan_before_any_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="browser.scrape_markdown",
                                args={"url": "https://dmdflow.com"},
                                reason="Scrape the requested page.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={"path": "readme123.md", "content_from_previous_step": True},
                                reason="Write the scraped Markdown.",
                            ),
                        )
                    )
                ),
                audit=audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "runtime": {"emergency_stop": {"active": True, "triggered_at": "now", "reason": "test"}},
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"browser.scrape_markdown", "files.write"}),
                    granted_permissions=frozenset({"browser.read"}),
                ),
            )
            with mock.patch("dmdagent4all.tools.web.fetch_page") as fetch:
                response = core.handle_text("scrape dmdflow.com and place the results in readme123.md")

            self.assertEqual(response.status, "denied")
            self.assertIn("Emergency stop", response.message)
            fetch.assert_not_called()
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_unrelated_visual_planner_clarification_is_not_shown_to_user(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        clarification_message=(
                            "I need the image or description of the rotating objects to determine "
                            "which number is the rotating one."
                        )
                    )
                ),
            )

            response = core.handle_text("scrape dmdflow.com and place the results in readme123.md")

            self.assertNotIn("rotating objects", response.message)
            self.assertNotIn("which number", response.message)
            self.assertNotIn("image or description", response.message)
            self.assertEqual(response.status, "denied")
            self.assertIn("Tool is disabled: browser.scrape_markdown", response.message)

    def test_planner_error_fallback_does_not_leak_repair_debug_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = _build_core(
                root,
                FailingPlanner(
                    PlannerError(
                        "Return valid JSON only. rotating objects repair prompt debug"
                    )
                ),
            )

            response = core.handle_text("scrape dmdflow.com and place the results in readme123.md")

            self.assertNotIn("Return valid JSON", response.message)
            self.assertNotIn("rotating objects", response.message)
            self.assertNotIn("repair prompt", response.message)
            self.assertEqual(response.status, "denied")

    def test_planner_error_on_normal_question_does_not_claim_planner_inactive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                FailingPlanner(PlannerError("Return valid JSON only. internal repair text")),
            )

            response = core.handle_text("искам да ми кажеш какво знаеш за конфиг")

            self.assertEqual(response.status, "ok")
            self.assertIn("planner is active", response.message)
            self.assertNotIn("planner is not active", response.message)
            self.assertNotIn("Return valid JSON", response.message)
            self.assertEqual(response.data, {"planner": "llm", "fallback": "planner_error"})

    def test_local_dev_autonomy_recovers_malformed_planner_output_with_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = ChatOnlyPlanner("Config sections are available in runtime context.")
            core = _build_core(
                Path(tmp),
                planner,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_text("искам да ми кажеш какво знаеш за конфиг")

            self.assertEqual(response.status, "ok")
            self.assertIn("Config sections", response.message)
            self.assertNotIn("planner is active", response.message)
            self.assertEqual(response.data["fallback"], "local_dev_planner_chat")
            self.assertEqual(response.data["runtime_mode"], "full_llm_first_autonomy")
            self.assertGreaterEqual(len(response.data["trace"]), 2)
            self.assertEqual(len(planner.chat_calls), 1)

    def test_local_dev_autonomy_auto_approves_safe_file_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            core = _build_core(root, None, audit=audit)

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="files.write",
                        args={"path": "readme.md", "content": "# Local dev\n", "overwrite": True},
                        reason="Local dev write.",
                    )
                )

            self.assertEqual(response.status, "ok")
            self.assertEqual(audit.list_approvals(status="pending"), [])
            self.assertEqual((workspace / "readme.md").read_text(encoding="utf-8"), "# Local dev\n")

    def test_local_dev_autonomy_still_blocks_env_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(root, None, audit=audit)

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="files.write",
                        args={"path": ".env", "content": "TOKEN=x"},
                        reason="Secret write should remain blocked.",
                    )
                )

            self.assertEqual(response.status, "denied")
            self.assertIn("secret", response.message.casefold())
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_local_dev_autonomy_delete_still_requires_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "old.md").write_text("remove me\n", encoding="utf-8")
            core = _build_core(root, None, audit=audit)

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="files.delete",
                        args={"path": "old.md"},
                        reason="Delete still needs approval.",
                    )
                )

            self.assertEqual(response.status, "approval_required")
            self.assertEqual(audit.list_approvals(status="pending")[0]["tool"], "files.delete")
            self.assertTrue((workspace / "old.md").exists())

    def test_local_dev_autonomy_auto_enables_and_runs_scrape(self) -> None:
        html = "<html><head><title>DMD Flow</title></head><body><main><p>Autonomy scrape.</p></main></body></html>"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(root, None, audit=audit)
            page = FetchedPage(
                url="https://dmdflow.com",
                final_url="https://dmdflow.com/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=128,
                truncated=False,
            )

            with (
                mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}),
                mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page),
            ):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="browser.scrape_markdown",
                        args={"url": "dmdflow.com", "mode": "raw_page", "format": "clean_markdown"},
                        reason="Safe scrape.",
                    )
                )

            self.assertEqual(response.status, "ok")
            self.assertIn("Autonomy scrape", response.data["markdown"])
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_local_dev_autonomy_scrape_to_file_chain_writes_without_approval(self) -> None:
        html = "<html><head><title>DMD Flow</title></head><body><main><p>Autonomous chain.</p></main></body></html>"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="browser.scrape_markdown",
                                args={"url": "dmdflow.com", "mode": "raw_page", "format": "clean_markdown"},
                                reason="Scrape requested page.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={"path": "readme123.md", "content_from_previous_step": True},
                                reason="Write previous markdown.",
                            ),
                        )
                    )
                ),
                audit=audit,
            )
            page = FetchedPage(
                url="https://dmdflow.com",
                final_url="https://dmdflow.com/",
                status=200,
                content_type="text/html",
                body=html,
                bytes_read=128,
                truncated=False,
            )

            with (
                mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}),
                mock.patch("dmdagent4all.tools.web.fetch_page", return_value=page),
            ):
                response = core.handle_text("scrape dmdflow.com and place the results in readme123.md")

            self.assertEqual(response.status, "ok")
            self.assertEqual(audit.list_approvals(status="pending"), [])
            self.assertIn("Autonomous chain", (workspace / "readme123.md").read_text(encoding="utf-8"))

    def test_local_dev_autonomy_scaffolds_one_page_project_without_terminal_mkdir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="project.scaffold_one_page_app",
                            args={
                                "path": "test",
                                "owner_name": "Денис Денчев",
                                "role": "AI Developer",
                                "theme": "developer tech dark",
                                "project_summary": "Work on DMD Agent autonomous runtime.",
                                "include_backend": True,
                                "overwrite": True,
                            },
                            reason="Scaffold requested one-page site.",
                        )
                    )
                ),
                audit=audit,
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_text("създай one pager react node.js сайт в папка test")

            self.assertEqual(response.status, "ok")
            self.assertEqual(audit.list_approvals(status="pending"), [])
            self.assertTrue((workspace / "test" / "package.json").exists())
            self.assertTrue((workspace / "test" / "src" / "App.jsx").exists())
            self.assertTrue((workspace / "test" / "server" / "index.js").exists())
            self.assertIn("Денис Денчев", (workspace / "test" / "src" / "App.jsx").read_text(encoding="utf-8"))
            self.assertNotIn("terminal.run", response.message)

    def test_write_many_still_blocks_env_targets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(root, None, audit=audit)

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="files.write_many",
                        args={"files": [{"path": ".env", "content": "TOKEN=x", "overwrite": True}]},
                        reason="Secret batch write should remain blocked.",
                    )
                )

            self.assertEqual(response.status, "denied")
            self.assertIn("secret", response.message.casefold())
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_local_dev_autonomy_recovers_prose_file_append_as_real_write(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            readme = workspace / "README.md"
            readme.write_text("# README\n\nThis is an empty README file.\n", encoding="utf-8")
            core = _build_core(
                root,
                FakePlanner(PlanResult(final_message='Готово, Денис. Добавих "ТЕСТ" в README.md.')),
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_text('в README.md искам да добавиш текст "ТЕСТ"')

            self.assertEqual(response.status, "ok")
            self.assertIn("Добавих", response.message)
            self.assertIn("ТЕСТ", readme.read_text(encoding="utf-8"))

    def test_file_chain_falls_back_to_scraped_downloads_when_previous_list_misses_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            downloads = root / "Downloads"
            scrape_dir = downloads / "scrapefiles"
            workspace.mkdir()
            scrape_dir.mkdir(parents=True)
            (workspace / "README.md").write_text("# README\n\n", encoding="utf-8")
            source = scrape_dir / "футбол-спорт-спортни-новини-sportal.bg.md"
            source.write_text("# Sportal\n\nFootball scrape body\n", encoding="utf-8")
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="files.list",
                                args={"path": "."},
                                reason="Planner incorrectly inspected the active workspace first.",
                            ),
                            ToolRequest(
                                tool="files.read",
                                args={"path_from_previous_step_match": "футбол-спорт-спортни-новини-sportal.bg.md"},
                                reason="Read the named scraped file.",
                            ),
                            ToolRequest(
                                tool="files.write",
                                args={"path": "README.md", "content_from_previous_step": True},
                                reason="Write the scraped file content into README.md.",
                            ),
                        )
                    )
                ),
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "storage": {"downloads_root": str(downloads)},
                    "workspace": {
                        "default_path": str(workspace),
                        "current_path": str(workspace),
                        "allowed_roots": [str(root)],
                        "blocked_paths": [],
                    },
                },
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_text(
                    "може ли да вземеш текста от футбол-спорт-спортни-новини-sportal.bg.md и да го преместиш в този празен README file"
                )

            self.assertEqual(response.status, "ok")
            self.assertIn("Football scrape body", (workspace / "README.md").read_text(encoding="utf-8"))

    def test_local_dev_autonomy_emergency_stop_still_blocks_safe_actions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
                audit=audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "runtime": {"emergency_stop": {"active": True, "reason": "test"}},
                },
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="files.write",
                        args={"path": "readme.md", "content": "blocked"},
                        reason="Should be blocked by emergency stop.",
                    )
                )

            self.assertEqual(response.status, "denied")
            self.assertIn("Emergency stop", response.message)
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_local_dev_autonomy_allows_readonly_terminal_inspection_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "marker.txt").write_text("ok\n", encoding="utf-8")
            audit = AuditStore(root / "audit.db")
            core = _build_core(root, None, audit=audit)

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="terminal.run",
                        args={"command": ["find", ".", "-maxdepth", "1", "-type", "f", "-print"]},
                        reason="Readonly workspace inspection.",
                    )
                )
                blocked = core.handle_tool_request(
                    ToolRequest(
                        tool="terminal.run",
                        args={"command": ["git", "push"]},
                        reason="Write-impact command.",
                    )
                )

            self.assertEqual(response.status, "ok")
            self.assertIn("marker.txt", response.data["stdout"])
            self.assertEqual(blocked.status, "denied")
            self.assertIn("allowlist", blocked.message)
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_local_dev_autonomy_executes_planned_workspace_python_script_without_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="files.write",
                                args={
                                    "path": "time_sofia.py",
                                    "content": (
                                        "from datetime import datetime\n"
                                        "from zoneinfo import ZoneInfo\n"
                                        "print(datetime.now(ZoneInfo('Europe/Sofia')).strftime('%H:%M'))\n"
                                    ),
                                    "overwrite": True,
                                },
                                reason="Create a local script requested by the user.",
                            ),
                            ToolRequest(
                                tool="terminal.run",
                                args={"command": [sys.executable, "time_sofia.py"]},
                                reason="Execute the script to verify it works.",
                            ),
                        )
                    )
                ),
                audit=audit,
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_text(
                    "създай питонски скрипт за часа в София и го изпълни"
                )

            self.assertEqual(response.status, "ok")
            self.assertTrue((workspace / "time_sofia.py").exists())
            self.assertRegex(response.data["steps"][-1]["data"]["stdout"], r"\d{2}:\d{2}")
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_local_dev_autonomy_does_not_auto_approve_write_impact_allowlisted_terminal(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
                audit=audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["git", "push"]],
                        "auto_approve_allowlisted": True,
                    },
                },
            )

            with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
                response = core.handle_tool_request(
                    ToolRequest(
                        tool="terminal.run",
                        args={"command": ["git", "push"]},
                        reason="Write-impact command must still require approval.",
                    )
                )

            self.assertEqual(response.status, "approval_required")
            self.assertEqual(audit.list_approvals(status="pending")[0]["tool"], "terminal.run")

    def test_identity_answers_use_setup_config_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                None,
                config={
                    "llm": {"response_language": "auto"},
                    "setup": {
                        "agent_name": "jarvis",
                        "user_name": "Test User",
                        "preferred_language": "auto",
                    },
                },
            )
            who_are_you = core.handle_text("who are you")
            who_am_i = core.handle_text("who am i")
            self.assertEqual(who_are_you.status, "ok")
            self.assertIn("jarvis", who_are_you.message)
            self.assertEqual(who_am_i.message, "You are Test User.")

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
            core = _build_core(root, None, config=config)

            agent_response = core.handle_text("call yourself Jarvis")
            user_response = core.handle_text("my name is Test User")

            self.assertEqual(agent_response.status, "ok")
            self.assertEqual(user_response.status, "ok")
            self.assertEqual(config["setup"]["agent_name"], "Jarvis")
            self.assertEqual(config["setup"]["user_name"], "Test User")
            profile = (root / "memory" / "long-term" / "profile.md").read_text(encoding="utf-8")
            self.assertIn("Assistant name: Jarvis", profile)
            self.assertIn("User name: Test User", profile)

    def test_llm_first_does_not_parse_correction_as_identity_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "setup": {
                    "agent_name": "DMD Agent",
                    "user_name": "Denis",
                    "preferred_language": "auto",
                },
            }
            planner = RecordingPlanner(answer="Immich, not Proxmox.")
            core = _build_core(Path(tmp), planner, config=config)

            response = core.handle_text("im asking for immich not for proxmox")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.data, {"planner": "deterministic", "source": "session_correction"})
            self.assertEqual(config["setup"]["user_name"], "Denis")
            self.assertEqual(planner.calls, [])

    def test_model_requests_profile_update_tool_instead_of_regex_identity_update(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "setup": {
                    "agent_name": "DMD Agent",
                    "user_name": "",
                    "preferred_language": "auto",
                },
            }
            planner = FakePlanner(
                PlanResult(
                    tool_request=ToolRequest(
                        tool="profile.update",
                        args={"user_name": "Denis"},
                        reason="User explicitly gave their name.",
                    )
                )
            )
            core = _build_core(root, planner, config=config)

            pending = core.handle_text("my name is Denis")
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approved.status, "ok")
            self.assertEqual(config["setup"]["user_name"], "Denis")
            profile = (root / "memory" / "long-term" / "profile.md").read_text(encoding="utf-8")
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
            core = _build_core(root, None, config=config)

            response = core.handle_text("im Test User")

            self.assertEqual(response.status, "ok")
            self.assertEqual(config["setup"]["user_name"], "Test User")

    def test_simple_terminal_phrase_routes_to_terminal_policy_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)

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
                None,
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
                None,
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
                None,
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
                None,
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Remember that I like green tea")

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "memory.write")
            self.assertEqual(approval["args"]["path"], "long-term/facts/personal.md")
            self.assertIn("I like green tea", approval["args"]["body"])

    def test_short_term_remember_phrase_requires_approval_and_ttl(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
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
            ("And also remember that I like red bicycles", "I like red bicycles"),
            ("I like red bicycles remember that", "I like red bicycles"),
            ("Can you remember that I have lab servers at home", "I have lab servers at home"),
        ]
        for message, expected_fact in examples:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    audit = AuditStore(root / "audit.db")
                    core = _build_core(
                        root,
                        None,
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
                None,
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text(
                "Моля те да запомниш в нов .md дългосрочно че имам два тестови сървъра модел TestBox X1"
            )

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "memory.write")
            self.assertEqual(approval["args"]["path"], "auto")
            self.assertEqual(approval["args"]["memory_scope"], "long-term")
            self.assertEqual(approval["args"]["body"], "имам два тестови сървъра модел TestBox X1")

    def test_bulk_memory_organizer_prompt_does_not_route_to_browser_or_reminders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
                audit,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
            )
            prompt = """
Искам да разделиш long-term memory markdown файла на категории.
Препоръчителна структура: memory/ owner/ profile.md projects/ primary-product/ failover.md
High priority файлове: memory/README.md
Не прави един огромен файл.
and this is the knowlage
# Long-Term Memory — Test User
## Owner / User
Name: Test User
## Project: DMD Agent 4 All
Useful scenario: “Напомни ми след 8 минути да извадя яйцата.”
"""

            response = core.handle_text(prompt)

            self.assertEqual(response.status, "approval_required")
            approval = audit.list_approvals(status="pending")[0]
            self.assertEqual(approval["tool"], "memory.organize_long_term")
            self.assertIn("# Long-Term Memory", approval["args"]["source_markdown"])

    def test_bulk_memory_organizer_writes_modular_files_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
                audit,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )
            prompt = """
Препоръчителна структура: memory/ owner/ profile.md memory/README.md
High priority файлове: memory/owner/profile.md
and this is the knowlage
# Long-Term Memory — Test User ## Owner / User Name: Test User Main working language: Bulgarian ## Project: Example Marketplace Example Marketplace is the user's major marketplace platform. ## Primary Product High Availability / Failover Architecture Avoid split-brain at all costs. Primary server: primary-node Backup server: backup-node ## Security Preferences / Rules Do not expose admin panels publicly. Never run docker compose down -v unless absolutely sure.
"""

            pending = core.handle_text(prompt)
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(approved.status, "ok")
            self.assertIn("owner/profile.md", approved.data["files"])
            self.assertIn("projects/primary-product/failover.md", approved.data["files"])
            self.assertTrue((root / "memory" / "README.md").exists())
            self.assertTrue((root / "memory" / "owner" / "profile.md").exists())
            self.assertIn(
                "Avoid split-brain",
                (root / "memory" / "projects" / "primary-product" / "failover.md").read_text(encoding="utf-8"),
            )
            self.assertNotIn(
                "No source notes were found",
                (root / "memory" / "owner" / "profile.md").read_text(encoding="utf-8"),
            )

    def test_memory_organizer_correction_without_source_does_not_open_markdown_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
                audit,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
            )

            response = core.handle_text(
                "Имаш грешка в организирането на memory файловете. "
                "Не оставяй No source notes were found в memory/owner/profile.md."
            )

            self.assertEqual(response.status, "ok")
            self.assertIn("Нямам достъп до source memory content", response.message)
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_bulgarian_what_are_you_answer_does_not_need_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(
                Path(tmp),
                None,
                config={
                    "llm": {"provider": "openai", "response_language": "auto"},
                    "setup": {"agent_name": "jarvis", "user_name": "Test User"},
                },
            )

            response = core.handle_text("здравей, имаш ли инфо какво си ти")

            self.assertEqual(response.status, "ok")
            self.assertIn("jarvis", response.message)

    def test_bulgarian_memory_recall_answers_from_local_long_term_memory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "long-term/facts/computers.md",
                "Имам два тестови сървъра модел TestBox X1.",
                metadata={"type": "long_term_note", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "homelab/homelab-primary.md",
                "Hostname: primary-node\nPrimary server runs many Docker services.",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- I like green tea.\n- I like red bicycles.",
                metadata={"type": "personal_fact", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Какви сървъри имам")

            self.assertEqual(response.status, "ok")
            self.assertIn("Имаш два тестови сървъра модел TestBox X1", response.message)
            self.assertNotIn("Primary server runs many Docker services", response.message)
            self.assertNotIn("green tea", response.message)
            self.assertEqual(response.data, {"planner": "deterministic"})

    def test_bulgarian_immich_port_lookup_answers_from_memory_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/immich.md",
                "\n".join(
                    [
                        "# Immich",
                        "Immich port:",
                        "- 2283",
                    ]
                ),
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            response = core.handle_text("на кой порт е имич")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.message, "Immich е на порт 2283.")
            self.assertEqual(response.data, {"planner": "deterministic"})

    def test_bulgarian_proxmox_access_recall_uses_local_memory_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "homelab/homelab2-backup.md",
                "\n".join(
                    [
                        "Hostname: backup-node",
                        "LAN IP: 192.0.2.22",
                        "Tailscale IP: 198.51.100.22",
                        "Backup Proxmox web UI: https://192.0.2.25:8006",
                    ]
                ),
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "long-term/facts/computers.md",
                "Имам два тестови сървъра модел TestBox X1.",
                metadata={"type": "long_term_note", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "projects/dmd-agent-4-all/overview.md",
                "GitHub: https://github.com/example/local-agent",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            response = core.handle_text("не помня как да си вляза в проксмокс в сървъра")

            self.assertEqual(response.status, "ok")
            self.assertIn("Proxmox web UI адресът ти е: https://192.0.2.25:8006", response.message)
            self.assertNotIn("TestBox X1", response.message)
            self.assertNotIn("github.com", response.message)
            self.assertEqual(response.data, {"planner": "deterministic"})

    def test_bulgarian_followup_machine_address_uses_recent_context_for_memory_recall(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "homelab/homelab2-backup.md",
                "\n".join(
                    [
                        "Hostname: backup-node",
                        "LAN IP: 192.0.2.22",
                        "Tailscale IP: 198.51.100.22",
                        "Backup Proxmox web UI: https://192.0.2.25:8006",
                    ]
                ),
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "long-term/facts/computers.md",
                "Имам два тестови сървъра модел TestBox X1.",
                metadata={"type": "long_term_note", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "projects/dmd-agent-4-all/overview.md",
                "GitHub: https://github.com/example/local-agent",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            core.handle_text("не помня как да си вляза в проксмокс в сървъра", session_id="proxmox")
            response = core.handle_text("не, кажи ми на кой адрес беше машината ми", session_id="proxmox")

            self.assertEqual(response.status, "ok")
            self.assertIn("https://192.0.2.25:8006", response.message)
            self.assertNotIn("TestBox X1", response.message)
            self.assertNotIn("github.com", response.message)
            self.assertEqual(response.data, {"planner": "deterministic"})

    def test_bulgarian_machine_address_without_context_prefers_machine_endpoint_over_public_url(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "projects/dmd-agent-4-all/overview.md",
                "GitHub: https://github.com/example/local-agent",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "homelab/homelab-primary.md",
                "\n".join(
                    [
                        "Hostname: primary-node",
                        "LAN IP: 192.0.2.43",
                        "Tailscale IP: 198.51.100.43",
                    ]
                ),
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            response = core.handle_text("кажи ми на кой адрес беше машината ми")

            self.assertEqual(response.status, "ok")
            self.assertIn("192.0.2.43", response.message)
            self.assertNotIn("github.com", response.message)
            self.assertEqual(response.data, {"planner": "deterministic"})

    def test_broad_bulgarian_memory_recall_lists_saved_facts_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "README.md",
                "\n".join(
                    [
                        "# Memory Index",
                        "Purpose: Index for modular long-term memory files.",
                        "## High Priority For Retrieval",
                        "- owner/profile.md",
                        "- homelab/security.md",
                    ]
                ),
                metadata={"type": "memory_index", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "profile.md",
                "User name: Test User\nPreferred nickname: buddy\nBulgarian nickname: маняк",
                metadata={"type": "profile", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "long-term/facts/computers.md",
                "Имам два тестови сървъра модел TestBox X1.",
                metadata={"type": "long_term_note", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Искам да ми кажеш всичко което знаеш за мен")

            self.assertEqual(response.status, "ok")
            self.assertIn("В локалната long-term memory знам това за теб:", response.message)
            self.assertIn("Име: Test User", response.message)
            self.assertIn("Български прякор: маняк", response.message)
            self.assertIn("Имаш два тестови сървъра модел TestBox X1", response.message)
            self.assertNotIn("Memory Index", response.message)
            self.assertNotIn("High Priority", response.message)

    def test_short_bulgarian_clarification_uses_previous_assistant_turn_without_planner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/plex.md",
                "Plex port: 32400",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(
                root,
                None,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            core.handle_text("не помня на кой порт беше плекс", session_id="clarify")
            response = core.handle_text("не разбах", session_id="clarify")

            self.assertEqual(response.status, "ok")
            self.assertIn("Казано по-просто:", response.message)
            self.assertIn("Plex е на порт 32400", response.message)
            self.assertEqual(response.data, {"planner": "deterministic", "source": "recent_conversation"})

    def test_remind_me_phrase_creates_approval_gated_local_reminder(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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
                None,
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

    def test_safe_workspace_mkdir_requires_approval_before_running(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                None,
                audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["pwd"]],
                        "auto_approve_allowlisted": True,
                        "workspace_root": str(root / "workspace"),
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_tool_request(
                ToolRequest(tool="terminal.run", args={"command": ["mkdir", "test"]})
            )

            self.assertEqual(response.status, "approval_required")
            self.assertEqual(len(audit.list_approvals(status="pending")), 1)
            self.assertFalse((root / "workspace" / "test").exists())

    def test_terminal_workspace_root_can_point_at_project_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            (project / "README.md").write_text("Project readme\n", encoding="utf-8")
            core = _build_core(
                root,
                None,
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
                    "user_name": "Test User",
                    "preferred_language": "auto",
                },
            }
            core = _build_core(root, None, config=config)

            saved = core.handle_text(
                "i want you to call me buddy or if i type in Bulgarian you can also call me маняк"
            )
            answered = core.handle_text("аз кой съм")

            self.assertEqual(saved.status, "ok")
            self.assertEqual(config["setup"]["nickname"], "buddy")
            self.assertEqual(config["setup"]["nickname_bg"], "маняк")
            self.assertEqual(answered.status, "ok")
            self.assertIn("маняк", answered.message)
            self.assertIn("Test User", answered.message)

    def test_memory_tool_result_can_be_synthesized_into_human_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- Test User likes green tea.",
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
                    "You like green tea.",
                ),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("What do I like?")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.message, "You like green tea.")

    def test_multi_memory_read_plan_is_synthesized_into_final_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MemoryManager(root / "memory")
            manager.write(
                "business/profile.md",
                "Denis runs an AI automation business.",
                metadata={"type": "business"},
            )
            manager.write(
                "business/services.md",
                "Services: local AI agents, automation, Proxmox homelab operations.",
                metadata={"type": "business"},
            )
            core = _build_core(
                root,
                AnsweringPlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="memory.read",
                                args={"path": "business/profile.md"},
                                reason="Read business profile.",
                            ),
                            ToolRequest(
                                tool="memory.read",
                                args={"path": "business/services.md"},
                                reason="Read business services.",
                            ),
                        )
                    ),
                    "Start with a focused AI automation offer and package it for small businesses.",
                ),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Can you help me with a business plan?")

            self.assertEqual(response.status, "ok")
            self.assertIn("focused AI automation offer", response.message)
            self.assertNotIn("Tool executed", response.message)
            self.assertNotIn("Denis runs", response.message)
            self.assertEqual(response.data, {"planner": "llm"})

    def test_multi_memory_read_synthesis_failure_does_not_dump_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manager = MemoryManager(root / "memory")
            manager.write(
                "business/private.md",
                "Private business context that should not be dumped raw.",
                metadata={"type": "business"},
            )
            core = _build_core(
                root,
                AnsweringPlanner(
                    PlanResult(
                        tool_plan=(
                            ToolRequest(
                                tool="memory.read",
                                args={"path": "business/private.md"},
                                reason="Read business context.",
                            ),
                        )
                    ),
                    "",
                ),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Give me business advice from my memory.")

            self.assertEqual(response.status, "ok")
            self.assertIn("business/private.md", response.message)
            self.assertNotIn("Tool executed", response.message)
            self.assertNotIn("Private business context", response.message)
            self.assertNotIn("content", json.dumps(response.data or {}))

    def test_memory_read_synthesis_failure_does_not_dump_file_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/immich.md",
                "# Immich\nSecret-ish long local notes that should not be dumped raw.",
                metadata={"type": "service"},
            )
            core = _build_core(
                root,
                AnsweringPlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="memory.read",
                            args={"path": "services/immich.md"},
                            reason="Look up Immich.",
                        )
                    ),
                    "",
                ),
                config={"llm": {"provider": "ollama", "response_language": "auto"}},
            )

            response = core.handle_text("Where do I open Immich?")

            self.assertEqual(response.status, "ok")
            self.assertIn("services/immich.md", response.message)
            self.assertNotIn("Secret-ish long local notes", response.message)
            self.assertNotIn("content", response.data or {})

    def test_cloud_low_risk_memory_list_is_synthesized_without_extra_cloud_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- Test User likes green tea.",
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
                    "You like green tea.",
                ),
                audit=audit,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            response = core.handle_text("What do I like?")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.message, "You like green tea.")
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_cloud_low_risk_memory_list_does_not_create_repeated_approvals(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            MemoryManager(root / "memory").write(
                "facts/personal.md",
                "- Test User likes green tea.",
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
                    "You like green tea.",
                ),
                audit=audit,
                config={"llm": {"provider": "openai", "response_language": "auto"}},
                permission_context=PermissionContext(cloud_model_active=True),
            )

            first = core.handle_text("What do I like?")
            second = core.handle_text("What do I like?")

            self.assertEqual(first.status, "ok")
            self.assertEqual(second.status, "ok")
            self.assertEqual(second.message, "You like green tea.")
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_normal_bulgarian_translation_uses_chat_mode_without_planner_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            planner = ChatOnlyPlanner("Здравей, как си?")
            core = _build_core(root, planner)

            response = core.handle_text("може ли да ми преведеш това на български: Hello how are you")

            self.assertEqual(response.status, "ok")
            self.assertIn("Здравей", response.message)
            self.assertEqual(len(planner.chat_calls), 1)
            self.assertNotIn("planner", response.message.lower())
            self.assertNotIn("{", response.message)

    def test_bulgarian_thanks_uses_normal_chat_without_debug_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = ChatOnlyPlanner("Няма проблем.")
            core = _build_core(Path(tmp), planner)

            response = core.handle_text("благодаря ти")

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.message, "Няма проблем.")
            self.assertEqual(response.data, {"mode": "normal_chat"})

    def test_bulgarian_casual_status_uses_normal_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            planner = ChatOnlyPlanner("Добре съм, готов съм да помагам.")
            core = _build_core(Path(tmp), planner)

            response = core.handle_text("джарвис как си")

            self.assertEqual(response.status, "ok")
            self.assertIn("Добре", response.message)
            self.assertEqual(len(planner.chat_calls), 1)

    def test_terminal_ls_request_routes_deterministically_and_returns_clean_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "workspace" / "note.txt").parent.mkdir(parents=True, exist_ok=True)
            (root / "workspace" / "note.txt").write_text("hello", encoding="utf-8")
            core = _build_core(
                root,
                ChatOnlyPlanner("should not be used"),
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

            response = core.handle_text("изпълни в терминала ls")

            self.assertEqual(response.status, "ok")
            self.assertIn("Изпълних `ls`", response.message)
            self.assertIn("note.txt", response.message)
            self.assertNotIn("returncode", response.message)

    def test_mkdir_request_requires_approval_and_does_not_execute_silently(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            core = _build_core(
                root,
                ChatOnlyPlanner("should not be used"),
                audit=audit,
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

            response = core.handle_text("изпълни mkdir test")

            self.assertEqual(response.status, "approval_required")
            self.assertFalse((root / "workspace" / "test").exists())
            self.assertEqual(len(audit.list_approvals(status="pending")), 1)

    def test_secret_file_reads_are_denied_by_backend_policy(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / ".env").write_text("API_KEY=secret-value", encoding="utf-8")
            core = _build_core(root, ChatOnlyPlanner("should not be used"))

            bg = core.handle_text("прочети .env")
            cat = core.handle_text("cat .env")

            self.assertEqual(bg.status, "denied")
            self.assertEqual(cat.status, "denied")
            self.assertIn("secret", bg.message.lower())
            self.assertNotIn("secret-value", bg.message)

    def test_latest_email_read_is_not_routed_as_file_read(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = _build_core(root, ExplodingPlanner())

            response = core.handle_text("прочети ми последно получения мейл")

            self.assertIsNone(ConversationRouter().route("прочети ми последно получения мейл"))
            self.assertEqual(response.status, "not_configured")
            self.assertIn("Email", response.message)
            self.assertNotEqual((response.data or {}).get("tool"), "files.read")

    def test_bulgarian_google_email_read_routes_to_gmail(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            enabled_tools = frozenset({"gmail.read_thread"})
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "email": {"gmail": {"enabled": True}, "max_body_chars": 20000},
                },
                permission_context=PermissionContext(
                    enabled_tools=enabled_tools,
                    granted_permissions=frozenset({"gmail.readonly"}),
                    approval_risk_threshold=3,
                ),
            )

            response = core.handle_text("искам да прочетеш последния ми мейл в гугъл")

            self.assertEqual(response.status, "not_configured")
            self.assertIn("Gmail", response.message)
            self.assertNotIn("Outlook", response.message)

    def test_email_send_request_creates_draft_then_requires_send_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            env = {
                "DMDAGENT_GMAIL_USERNAME": "sender@example.com",
                "DMDAGENT_GMAIL_APP_PASSWORD": "app-password",
            }
            enabled_tools = frozenset(
                {
                    "gmail.search",
                    "gmail.read_thread",
                    "gmail.summarize_inbox",
                    "gmail.create_draft",
                    "gmail.reply_draft",
                    "gmail.send_draft",
                    "gmail.archive",
                    "gmail.label",
                }
            )
            core = _build_core(
                root,
                ExplodingPlanner(),
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "email": {"gmail": {"enabled": True}, "max_body_chars": 20000},
                },
                permission_context=PermissionContext(
                    enabled_tools=enabled_tools,
                    granted_permissions=frozenset(
                        {"gmail.readonly", "gmail.compose", "gmail.send", "gmail.modify"}
                    ),
                    approval_risk_threshold=3,
                ),
            )

            with mock.patch.dict(os.environ, env):
                response = core.handle_text(
                    "прати мейл на d.d.denchev94@gmail.com сам избери тема а имейла е да му кажа че проекта работи и е онлайн"
                )

            self.assertEqual(response.status, "approval_required")
            self.assertEqual((response.data or {}).get("tool"), "gmail.send_draft")
            self.assertEqual((response.data or {}).get("to"), "d.d.denchev94@gmail.com")
            self.assertEqual((response.data or {}).get("subject"), "Проектът работи и е онлайн")
            draft_id = str((response.data or {}).get("draft_id"))
            draft_path = root / "email-drafts" / "gmail" / f"{draft_id}.json"
            self.assertTrue(draft_path.exists())
            draft = json.loads(draft_path.read_text(encoding="utf-8"))
            self.assertEqual(draft["body"], "Проекта работи и е онлайн.")

    def test_destructive_sql_is_blocked_before_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            core = _build_core(
                root,
                None,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["psql", "-c", "DELETE FROM users"]],
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_tool_request(
                ToolRequest(
                    tool="terminal.run",
                    args={"command": ["psql", "-c", "DELETE FROM users"]},
                )
            )

            self.assertEqual(response.status, "denied")
            self.assertIn("SQL", response.message)

    def test_file_and_directory_deletion_requests_create_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "old.txt").write_text("remove me", encoding="utf-8")
            (workspace / "old-dir").mkdir()
            core = _build_core(root, None, audit=audit)

            file_response = core.handle_text("delete old.txt")
            dir_response = core.handle_text("delete folder old-dir")

            self.assertEqual(file_response.status, "approval_required")
            self.assertEqual(dir_response.status, "approval_required")
            self.assertTrue((workspace / "old.txt").exists())
            self.assertTrue((workspace / "old-dir").exists())
            self.assertEqual(dir_response.data["risk"], 5)
            self.assertEqual(len(audit.list_approvals(status="pending")), 2)

    def test_file_read_outside_allowed_roots_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            core = _build_core(
                root,
                None,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "workspace": {
                        "default_path": str(root / "workspace"),
                        "current_path": str(root / "workspace"),
                        "allowed_roots": [str(root / "workspace")],
                        "blocked_paths": [],
                    },
                },
            )

            response = core.handle_tool_request(
                ToolRequest(tool="files.read", args={"path": str(outside)})
            )

            self.assertEqual(response.status, "denied")
            self.assertIn("outside allowed roots", response.message)

    def test_workspace_switch_allows_allowed_root_and_blocks_system_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / "project"
            project.mkdir()
            core = _build_core(
                root,
                None,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "workspace": {
                        "default_path": str(root / "workspace"),
                        "current_path": str(root / "workspace"),
                        "allowed_roots": [str(root)],
                        "blocked_paths": ["/etc", "/System"],
                    },
                },
            )

            allowed = core.handle_text(f"switch workspace to {project}")
            blocked = core.handle_text("switch workspace to /etc")

            self.assertEqual(allowed.status, "ok")
            self.assertIn(str(project), allowed.message)
            self.assertEqual(blocked.status, "denied")

    def test_workspace_switch_blocks_symlink_escape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            allowed_root = root / "allowed"
            allowed_root.mkdir()
            link = allowed_root / "escape"
            link.symlink_to("/etc")
            core = _build_core(
                root,
                None,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "workspace": {
                        "default_path": str(allowed_root),
                        "current_path": str(allowed_root),
                        "allowed_roots": [str(allowed_root)],
                        "blocked_paths": ["/etc"],
                    },
                },
            )

            response = core.handle_text(f"switch workspace to {link}")

            self.assertEqual(response.status, "denied")

    def test_plex_address_lookup_does_not_return_proxmox(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            MemoryManager(root / "memory").write(
                "services/plex.md",
                "Plex port: 32400",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "homelab/homelab-primary.md",
                "LAN IP: 192.0.2.43\nTailscale IP: 198.51.100.43",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            MemoryManager(root / "memory").write(
                "homelab/homelab2-backup.md",
                "Backup Proxmox web UI: https://192.0.2.25:8006",
                metadata={"type": "organized_long_term_memory", "memory_scope": "long-term"},
            )
            core = _build_core(root, None)

            response = core.handle_text("на кой адрес зареждах от домашния сървър плекс")

            self.assertEqual(response.status, "ok")
            self.assertIn("Plex", response.message)
            self.assertIn("http://192.0.2.43:32400/web", response.message)
            self.assertNotIn("Proxmox", response.message)
            self.assertNotIn("8006", response.message)

    def test_service_correction_is_kept_for_current_session(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)

            correction = core.handle_text("питам за плекс не за проксмокс", session_id="correction")
            feedback = core.handle_text("да не се повтаря тая грешка", session_id="correction")

            self.assertEqual(correction.status, "ok")
            self.assertEqual(feedback.status, "ok")
            self.assertIn("Plex", correction.message)
            self.assertIn("Proxmox", feedback.message)
            self.assertEqual(correction.data["source"], "session_correction")

    def test_mkdir_then_delete_uses_same_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            core = _build_core(
                root,
                None,
                audit=audit,
                config={
                    "llm": {"provider": "ollama", "response_language": "auto"},
                    "terminal": {
                        "enabled": True,
                        "allowed_commands": [["ls"]],
                        "auto_approve_allowlisted": True,
                        "workspace_root": str(workspace),
                    },
                },
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            pending_mkdir = core.handle_text("execute mkdir test")
            approved_mkdir = core.approve_and_execute(pending_mkdir.data["approval_id"])
            self.assertTrue((workspace / "test").exists())
            pending_delete = core.handle_text("delete this test folder")
            approved_delete = core.approve_and_execute(pending_delete.data["approval_id"])

            self.assertEqual(pending_mkdir.status, "approval_required")
            self.assertEqual(approved_mkdir.status, "ok")
            self.assertEqual(pending_delete.status, "approval_required")
            self.assertEqual(approved_delete.status, "ok")
            self.assertFalse((workspace / "test").exists())

    def test_cd_requests_switch_workspace_instead_of_terminal_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            target = workspace / "test"
            target.mkdir(parents=True)
            audit = AuditStore(root / "audit.db")
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "terminal": {"enabled": True, "allowed_commands": [["ls"]]},
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(
                root,
                None,
                audit=audit,
                config=config,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_text("in my terminal execute cd test")

            self.assertEqual(response.status, "ok")
            self.assertIn(str(target.resolve()), response.message)
            self.assertEqual(config["workspace"]["current_path"], str(target.resolve()))
            self.assertEqual(audit.list_approvals(status="pending"), [])

    def test_open_folder_request_switches_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            target = workspace / "test"
            target.mkdir(parents=True)
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(root, None, config=config)

            response = core.handle_text("open test")

            self.assertEqual(response.status, "ok")
            self.assertIn(str(target.resolve()), response.message)
            self.assertEqual(config["workspace"]["current_path"], str(target.resolve()))

    def test_llm_file_write_intent_can_place_text_in_readme(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            readme = workspace / "readme.md"
            readme.write_text("# README\nThis is an empty README file.\n", encoding="utf-8")
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            message = 'in this readme.md place text "Test"'
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="files.write",
                            args={"path": "readme.md", "content": "Test", "overwrite": True},
                            reason="The user asked to edit a local README file.",
                        )
                    )
                ),
                audit=audit,
                config=config,
            )

            self.assertIsNone(ConversationRouter().route(message))
            pending = core.handle_text(message)
            approval = audit.get_approval(pending.data["approval_id"])
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approval["tool"], "files.write")
            self.assertEqual(approval["args"]["path"], "readme.md")
            self.assertEqual(approval["args"]["content"], "Test")
            self.assertTrue(approval["args"]["overwrite"])
            self.assertEqual(approved.status, "ok")
            self.assertEqual(readme.read_text(encoding="utf-8"), "Test")

    def test_llm_file_write_intent_can_create_file_with_content(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            message = 'run touch test.md and place "12345" inside'
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="files.write",
                            args={"path": "test.md", "content": "12345", "overwrite": False},
                            reason="The user asked to create a text file with content.",
                        )
                    )
                ),
                audit=audit,
                config=config,
            )

            self.assertIsNone(ConversationRouter().route(message))
            pending = core.handle_text(message)
            approval = audit.get_approval(pending.data["approval_id"])
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approval["tool"], "files.write")
            self.assertEqual(approval["args"]["path"], "test.md")
            self.assertEqual(approval["args"]["content"], "12345")
            self.assertEqual(approved.status, "ok")
            self.assertEqual((workspace / "test.md").read_text(encoding="utf-8"), "12345")

    def test_developer_context_tool_returns_safe_code_workspace_context(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / "README.md").write_text("# Project\n", encoding="utf-8")
            (workspace / ".env").write_text("SECRET_TOKEN=abc\n", encoding="utf-8")
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(root, None, config=config)

            response = core.handle_tool_request(
                ToolRequest(
                    tool="developer.context",
                    args={"focus_paths": ["README.md"], "max_files": 50},
                )
            )

            self.assertEqual(response.status, "ok")
            self.assertEqual(response.data["scan_root"], str(workspace.resolve()))
            self.assertIn("README.md", {item["path"] for item in response.data["files"]})
            self.assertNotIn(".env", {item["path"] for item in response.data["files"]})
            self.assertEqual(response.data["focus_files"][0]["path"], "README.md")
            self.assertIn("files.write", {item["tool"] for item in response.data["available_actions"]})
            self.assertIn("terminal.run", {item["tool"] for item in response.data["available_actions"]})

    def test_developer_context_blocks_secret_focus_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            workspace = root / "workspace"
            workspace.mkdir()
            (workspace / ".env").write_text("SECRET_TOKEN=abc\n", encoding="utf-8")
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(root, None, config=config)

            response = core.handle_tool_request(
                ToolRequest(tool="developer.context", args={"focus_paths": [".env"]})
            )

            self.assertEqual(response.status, "denied")
            self.assertIn("secret", response.message.lower())

    def test_multi_step_mkdir_then_open_folder_continues_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "terminal": {
                    "enabled": True,
                    "allowed_commands": [["ls"]],
                    "auto_approve_allowlisted": True,
                    "workspace_root": str(workspace),
                },
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(
                root,
                None,
                audit=audit,
                config=config,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            pending = core.handle_text("i want you to execute mkdir test1 after that open the folder")
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approved.status, "ok")
            self.assertTrue((workspace / "test1").exists())
            self.assertEqual(config["workspace"]["current_path"], str((workspace / "test1").resolve()))
            self.assertIn("Switched workspace", approved.message)

    def test_multi_step_mkdir_then_nano_does_not_treat_file_as_host(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            workspace = root / "workspace"
            workspace.mkdir()
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "terminal": {
                    "enabled": True,
                    "allowed_commands": [["ls"]],
                    "auto_approve_allowlisted": True,
                    "workspace_root": str(workspace),
                },
                "workspace": {
                    "default_path": str(workspace),
                    "current_path": str(workspace),
                    "allowed_roots": [str(root)],
                    "blocked_paths": [],
                },
            }
            core = _build_core(
                root,
                None,
                audit=audit,
                config=config,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            pending = core.handle_text(
                "i want you to execute the following in my terminal mkdir test1 after that open the folder and execute nano realtest.py"
            )
            approved = core.approve_and_execute(pending.data["approval_id"])

            self.assertEqual(pending.status, "approval_required")
            self.assertEqual(approved.status, "ok")
            self.assertIn("realtest.py", approved.message)
            self.assertIn("interactive", approved.message)
            self.assertNotIn("host", approved.message.lower())
            self.assertEqual(config["workspace"]["current_path"], str((workspace / "test1").resolve()))

    def test_interactive_editor_response_includes_helpful_alternative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            core = _build_core(Path(tmp), None)

            response = core.handle_text("execute nano realtest.py")

            self.assertEqual(response.status, "ok")
            self.assertIn("interactive", response.message)
            self.assertIn("files.write", response.message)
            self.assertIn("realtest.py", response.message)

    def test_emergency_stop_blocks_tool_and_approval_execution(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            audit = AuditStore(root / "audit.db")
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "runtime": {"emergency_stop": {"active": True, "triggered_at": "now", "reason": "test"}},
                "terminal": {
                    "enabled": True,
                    "allowed_commands": [["ls"]],
                    "auto_approve_allowlisted": True,
                },
            }
            core = _build_core(
                root,
                None,
                audit=audit,
                config=config,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run", "memory.write"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            tool_response = core.handle_text("execute ls")
            approval_config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "runtime": {"emergency_stop": {"active": False, "triggered_at": "", "reason": ""}},
            }
            approval_core = _build_core(root, None, audit=AuditStore(root / "audit2.db"), config=approval_config)
            approval = approval_core.handle_text("Remember that I like test data")
            approval_config["runtime"]["emergency_stop"]["active"] = True
            approved = approval_core.approve_and_execute(approval.data["approval_id"])

            self.assertEqual(tool_response.status, "denied")
            self.assertIn("Emergency stop", tool_response.message)
            self.assertEqual(approved.status, "denied")
            self.assertIn("Emergency stop", approved.message)

    def test_emergency_stop_blocks_llm_selected_tool(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {
                "llm": {"provider": "ollama", "response_language": "auto"},
                "runtime": {"emergency_stop": {"active": True, "triggered_at": "now", "reason": "test"}},
                "terminal": {
                    "enabled": True,
                    "allowed_commands": [["ls"]],
                    "auto_approve_allowlisted": True,
                },
            }
            core = _build_core(
                root,
                FakePlanner(
                    PlanResult(
                        tool_request=ToolRequest(
                            tool="terminal.run",
                            args={"command": ["ls"]},
                            reason="User asked to list files.",
                        )
                    )
                ),
                config=config,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
            )

            response = core.handle_text("run ls")

            self.assertEqual(response.status, "denied")
            self.assertIn("Emergency stop", response.message)


def _build_core(
    root: Path,
    planner,
    audit: AuditStore | None = None,
    config: dict | None = None,
    permission_context: PermissionContext | None = None,
) -> AgentCore:
    registry = build_builtin_registry()
    runtime_config = config or {"llm": {"response_language": "auto"}}
    configured_workspace = runtime_config.get("terminal", {}).get("workspace_root") or str(root / "workspace")
    runtime_config.setdefault(
        "workspace",
        {
            "default_path": str(configured_workspace),
            "current_path": str(configured_workspace),
            "allowed_roots": [str(root)],
            "blocked_paths": [],
        },
    )
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
