import contextlib
import io
import os
import tempfile
import unittest
from unittest import mock

from dmdagent4all.cli import main
from dmdagent4all.config import DEFAULT_CONFIG


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
                    mock.patch("dmdagent4all.cli.build_agent_core") as build_core,
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
                    mock.patch("builtins.input", side_effect=["DMD", "Denis", "1", "1", EOFError]),
                    mock.patch("dmdagent4all.cli._ensure_ollama", return_value=None),
                ):
                    result = main(["chat", "--no-ollama"])
                with open(
                    os.path.join(tmp, "dmdagent4all", "memory", "profile.md"),
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
        self.assertIn("User name: Denis", profile)
        self.assertIn("Assistant name: DMD", profile)

    def test_no_args_opens_terminal_chat(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            old_home = os.environ.get("XDG_DATA_HOME")
            os.environ["XDG_DATA_HOME"] = tmp
            try:
                with (
                    contextlib.redirect_stdout(io.StringIO()) as output,
                    mock.patch("builtins.input", side_effect=EOFError),
                    mock.patch("dmdagent4all.cli._ensure_ollama", return_value=None),
                    mock.patch(
                        "dmdagent4all.cli._run_terminal_onboarding",
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
                            "Denis",
                            "1",
                            "1",
                            "/tool enable calendar.create_event",
                            "/permission grant calendar.create_event",
                            '{"tool":"calendar.create_event","args":{"title":"Dentist"}}',
                            EOFError,
                        ],
                    ),
                    mock.patch("dmdagent4all.cli._ensure_ollama", return_value=None),
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


if __name__ == "__main__":
    unittest.main()
