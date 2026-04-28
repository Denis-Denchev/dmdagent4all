import unittest

from dmdagent4all.sandbox import TerminalPolicy


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


if __name__ == "__main__":
    unittest.main()
