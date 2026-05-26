import tempfile
import unittest
from pathlib import Path

from dmdcore.agent import AgentCore
from dmdcore.audit import AuditStore
from dmdcore.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdcore.tools import build_builtin_registry
from dmdcore.tools.base import ToolRuntimeContext


class TerminalToolTest(unittest.TestCase):
    def test_terminal_run_executes_only_after_approval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            registry = build_builtin_registry()
            audit = AuditStore(root / "audit.db")
            core = AgentCore(
                permission_engine=PermissionEngine(registry.manifests),
                tool_registry=registry,
                permission_context=PermissionContext(
                    enabled_tools=frozenset({"terminal.run"}),
                    granted_permissions=frozenset({"terminal.run"}),
                ),
                runtime_context=ToolRuntimeContext(
                    memory_root=root / "memory",
                    workspace_root=root / "workspace",
                    config={
                        "terminal": {
                            "enabled": True,
                            "workspace_only": True,
                            "allowed_commands": [["pwd"]],
                        },
                        "workspace": {
                            "default_path": str(root / "workspace"),
                            "current_path": str(root / "workspace"),
                            "allowed_roots": [str(root)],
                            "blocked_paths": [],
                        },
                        "llm": {"response_language": "auto"},
                    },
                ),
                audit_store=audit,
                planner=None,
            )

            pending = core.handle_tool_request(
                ToolRequest(
                    tool="terminal.run",
                    args={"command": ["pwd"]},
                    reason="test",
                )
            )
            self.assertEqual(pending.status, "approval_required")

            approved = core.approve_and_execute(pending.data["approval_id"])
            self.assertEqual(approved.status, "ok")
            self.assertEqual(approved.data["returncode"], 0)
            self.assertIn(str(root / "workspace"), approved.data["stdout"])


if __name__ == "__main__":
    unittest.main()
