import unittest
from pathlib import Path
import tempfile

from dmdagent4all.sandbox import TerminalPolicy, run_workspace_command


class TerminalPolicyTest(unittest.TestCase):
    def test_terminal_disabled_by_default(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy().validate(["git", "status"])

    def test_allowed_exact_command_when_enabled(self) -> None:
        TerminalPolicy(enabled=True).validate(["git", "status"])

    def test_blocks_dangerous_binary(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True).validate(["sudo"])

    def test_blocks_secret_paths(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True, allowed_commands=(("cat", ".env"),)).validate(["cat", ".env"])

    def test_workspace_cwd_must_stay_inside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(PermissionError):
                run_workspace_command(
                    ["pwd"],
                    workspace=root / "workspace",
                    policy=TerminalPolicy(enabled=True),
                    cwd=str(root / "outside"),
                )


if __name__ == "__main__":
    unittest.main()
