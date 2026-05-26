import unittest
from pathlib import Path
import tempfile
import os
import sys
from unittest import mock

from dmdcore.sandbox import TerminalPolicy, run_workspace_command


class TerminalPolicyTest(unittest.TestCase):
    def test_terminal_disabled_by_default(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy().validate(["git", "status"])

    def test_allowed_exact_command_when_enabled(self) -> None:
        TerminalPolicy(enabled=True).validate(["git", "status"])

    def test_local_dev_autonomy_allows_readonly_non_allowlisted_commands(self) -> None:
        policy = TerminalPolicy(
            enabled=True,
            allowed_commands=(("ls",),),
            local_dev_autonomy=True,
        )

        policy.validate(["find", ".", "-maxdepth", "1", "-type", "f", "-print"])
        policy.validate(["git", "log", "--oneline", "-1"])

    def test_local_dev_autonomy_does_not_allow_write_impact_commands(self) -> None:
        policy = TerminalPolicy(
            enabled=True,
            allowed_commands=(("ls",),),
            local_dev_autonomy=True,
        )

        with self.assertRaises(PermissionError):
            policy.validate(["git", "push"])
        with self.assertRaises(PermissionError):
            policy.validate(["docker", "ps"])
        with self.assertRaises(PermissionError):
            policy.validate(["npm", "install"])

    def test_local_dev_autonomy_allows_validated_workspace_python_script(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "time_sofia.py"
            script.write_text("print('ok')\n", encoding="utf-8")

            result = run_workspace_command(
                [sys.executable, "time_sofia.py"],
                workspace=root,
                policy=TerminalPolicy(enabled=True, local_dev_autonomy=True),
            )

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["stdout"], "ok\n")

    def test_local_dev_autonomy_blocks_script_outside_workspace_and_inline_code(self) -> None:
        policy = TerminalPolicy(enabled=True, local_dev_autonomy=True)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outside = root.parent / f"{root.name}-outside.py"
            outside.write_text("print('outside')\n", encoding="utf-8")
            with self.assertRaises(PermissionError):
                run_workspace_command(
                    [sys.executable, str(outside)],
                    workspace=root,
                    policy=policy,
                )
        with self.assertRaises(PermissionError):
            policy.validate([sys.executable, "-c", "print('no')"])

    def test_local_dev_workspace_script_execution_sanitizes_secret_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "env_check.py"
            script.write_text(
                "import os\nprint(os.environ.get('DMDCORE_DEEPSEEK_API_KEY', 'missing'))\n",
                encoding="utf-8",
            )

            with mock.patch.dict(os.environ, {"DMDCORE_DEEPSEEK_API_KEY": "secret-value"}):
                result = run_workspace_command(
                    [sys.executable, "env_check.py"],
                    workspace=root,
                    policy=TerminalPolicy(
                        enabled=True,
                        local_dev_autonomy=True,
                        sanitize_environment=True,
                    ),
                )

            self.assertEqual(result["returncode"], 0)
            self.assertEqual(result["stdout"], "missing\n")

    def test_local_dev_autonomy_flag_enables_terminal_policy_from_config(self) -> None:
        with mock.patch.dict(os.environ, {"LOCAL_DEV_AUTONOMY": "true"}):
            policy = TerminalPolicy.from_config({"terminal": {"enabled": False}})

        self.assertTrue(policy.enabled)
        policy.validate(["pwd"])

    def test_allows_safe_workspace_mkdir_when_enabled(self) -> None:
        TerminalPolicy(enabled=True).validate(["mkdir", "test"])
        TerminalPolicy(enabled=True).validate(["mkdir", "-p", "tmp/nested"])

    def test_blocks_unsafe_mkdir_paths(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True).validate(["mkdir", "../outside"])
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True).validate(["mkdir", "/tmp/outside"])
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True).validate(["mkdir", "--mode=777", "test"])

    def test_blocks_dangerous_binary(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True).validate(["sudo"])

    def test_blocks_secret_paths(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy(enabled=True, allowed_commands=(("cat", ".env"),)).validate(["cat", ".env"])

    def test_blocks_destructive_sql_commands(self) -> None:
        with self.assertRaises(PermissionError):
            TerminalPolicy(
                enabled=True,
                allowed_commands=(("psql", "-c", "DROP TABLE users"),),
            ).validate(["psql", "-c", "DROP TABLE users"])

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
