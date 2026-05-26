import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from dmdcore.cli import main
from dmdcore.config import DEFAULT_CONFIG, load_config


class CliTest(unittest.TestCase):
    def test_help_imports_and_exits(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as context:
                main(["--help"])
        self.assertEqual(context.exception.code, 0)

    def test_simple_model_command_lists_modes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    result = main(["model"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        self.assertIn("Fast Mode", output.getvalue())

    def test_simple_model_command_sets_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    result = main(["model", "light"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        self.assertIn("Model mode set to Light Mode", output.getvalue())

    def test_custom_model_command_switches_back_to_ollama_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    provider_result = main(["models", "provider", "openai"])
                    model_result = main(["model", "use", "qwen3:8b"])
                config = load_config()
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home

        self.assertEqual(provider_result, 0)
        self.assertEqual(model_result, 0)
        self.assertEqual(config["llm"]["provider"], "ollama")
        self.assertEqual(config["llm"]["model"], "qwen3:8b")
        self.assertEqual(config["llm"]["base_url"], "http://localhost:11434")
        self.assertIsNone(config["llm"]["api_key_env"])

    def test_provider_command_switches_models_between_openai_and_local(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    openai_result = main(["models", "provider", "openai"])
                openai_config = load_config()
                with contextlib.redirect_stdout(io.StringIO()):
                    local_result = main(["models", "provider", "ollama"])
                local_config = load_config()
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home

        self.assertEqual(openai_result, 0)
        self.assertEqual(openai_config["llm"]["provider"], "openai")
        self.assertEqual(openai_config["llm"]["model"], "gpt-4o-mini")
        self.assertEqual(openai_config["llm"]["api_key_env"], "DMDCORE_OPENAI_API_KEY")
        self.assertEqual(local_result, 0)
        self.assertEqual(local_config["llm"]["provider"], "ollama")
        self.assertEqual(local_config["llm"]["model"], "qwen3:8b")
        self.assertEqual(local_config["llm"]["base_url"], "http://localhost:11434")
        self.assertIsNone(local_config["llm"]["api_key_env"])

    def test_provider_command_configures_deepseek_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    deepseek_result = main(["models", "provider", "deepseek", "--model", "deepseek-v4-pro"])
                deepseek_config = load_config()
                with contextlib.redirect_stdout(io.StringIO()):
                    local_result = main(["models", "provider", "ollama"])
                local_config = load_config()
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home

        self.assertEqual(deepseek_result, 0)
        self.assertEqual(deepseek_config["llm"]["provider"], "deepseek")
        self.assertEqual(deepseek_config["llm"]["model"], "deepseek-v4-pro")
        self.assertEqual(deepseek_config["llm"]["base_url"], "https://api.deepseek.com")
        self.assertEqual(deepseek_config["llm"]["api_key_env"], "DMDCORE_DEEPSEEK_API_KEY")
        self.assertEqual(local_result, 0)
        self.assertEqual(local_config["llm"]["provider"], "ollama")
        self.assertEqual(local_config["llm"]["model"], "qwen3:8b")

    def test_ask_alias_prints_friendly_answer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    result = main(["ask", "--no-ollama", "hello"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        self.assertIn("agent>", output.getvalue())

    def test_telegram_setup_request_uses_builtin_guide(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    mock.patch("dmdcore.cli.build_agent_core") as build_core,
                ):
                    result = main(["ask", "--no-ollama", "set", "up", "telegram", "bot"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        self.assertIn("Telegram Bot Setup", output.getvalue())
        build_core.assert_not_called()

    def test_first_chat_run_saves_terminal_onboarding(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    mock.patch("builtins.input", side_effect=["DMD", "Test User", "1", "1", EOFError]),
                    mock.patch("dmdcore.cli._ensure_ollama", return_value=None),
                ):
                    result = main(["chat", "--no-ollama"])
                with open(
                    os.path.join(tmp, "dmdcore", "memory", "long-term", "profile.md"),
                    encoding="utf-8",
                ) as profile_file:
                    profile = profile_file.read()
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        self.assertIn("DMD Terminal Chat", output.getvalue())
        self.assertIn("User name: Test User", profile)
        self.assertIn("Assistant name: DMD", profile)

    def test_no_args_opens_terminal_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    mock.patch("builtins.input", side_effect=EOFError),
                    mock.patch("dmdcore.cli._ensure_ollama", return_value=None),
                    mock.patch(
                        "dmdcore.cli._run_terminal_onboarding",
                        return_value={
                            **DEFAULT_CONFIG,
                            "setup": {
                                **DEFAULT_CONFIG["setup"],
                                "completed": True,
                            },
                        },
                    ),
                ):
                    result = main([])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        self.assertIn("Terminal Chat", output.getvalue())

    def test_chat_tool_and_permission_changes_refresh_live_core(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    mock.patch(
                        "builtins.input",
                        side_effect=[
                            "DMD",
                            "Test User",
                            "1",
                            "1",
                            "/tool enable calendar.create_event",
                            "/permission grant calendar.create_event",
                            '{"tool":"calendar.create_event","args":{"title":"Dentist"}}',
                            EOFError,
                        ],
                    ),
                    mock.patch("dmdcore.cli._ensure_ollama", return_value=None),
                ):
                    result = main(["chat", "--no-ollama"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home
        self.assertEqual(result, 0)
        value = output.getvalue()
        self.assertIn("Tool calendar.create_event enabled.", value)
        self.assertIn("Permission granted: calendar.events", value)
        self.assertIn("approval required>", value)
        self.assertNotIn("Tool is disabled: calendar.create_event", value)

    def test_start_web_reports_api_managed_telegram_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    mock.patch("dmdcore.cli._ensure_api", return_value=None),
                    mock.patch("dmdcore.cli._ensure_dashboard", return_value=None),
                    mock.patch("dmdcore.cli._http_json", return_value={"polling": True}) as http_json,
                ):
                    result = main(["start", "--no-open", "--no-ollama"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home

        self.assertEqual(result, 0)
        http_json.assert_called_once()
        self.assertIn("Telegram: running", output.getvalue())

    def test_start_web_can_skip_telegram(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()),
                    mock.patch("dmdcore.cli._ensure_api", return_value=None),
                    mock.patch("dmdcore.cli._ensure_dashboard", return_value=None),
                    mock.patch("dmdcore.cli._http_json") as http_json,
                ):
                    result = main(["start", "--no-open", "--no-ollama", "--no-telegram"])
            finally:
                if old_home is None:
                    os.environ.pop("XDG_DATA_HOME", None)
                else:
                    os.environ["XDG_DATA_HOME"] = old_home

        self.assertEqual(result, 0)
        http_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
