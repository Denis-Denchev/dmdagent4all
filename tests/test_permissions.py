import unittest

from dmdcore.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdcore.tools import load_builtin_manifests


class PermissionEngineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PermissionEngine(load_builtin_manifests())

    def test_default_system_tool_is_allowed(self) -> None:
        decision = self.engine.evaluate(ToolRequest(tool="system.list_enabled_tools"))
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.approval_required)

    def test_disabled_connector_tool_is_denied(self) -> None:
        decision = self.engine.evaluate(ToolRequest(tool="gmail.search"))
        self.assertFalse(decision.allowed)
        self.assertIn("disabled", decision.reason)

    def test_missing_permission_is_denied(self) -> None:
        context = PermissionContext(enabled_tools=frozenset({"gmail.search"}))
        decision = self.engine.evaluate(ToolRequest(tool="gmail.search"), context)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.missing_permissions, ("gmail.readonly",))

    def test_granted_read_permission_allows_low_risk_tool(self) -> None:
        context = PermissionContext(
            enabled_tools=frozenset({"gmail.search"}),
            granted_permissions=frozenset({"gmail.readonly"}),
        )
        decision = self.engine.evaluate(ToolRequest(tool="gmail.search"), context)
        self.assertTrue(decision.allowed)

    def test_memory_write_requires_approval(self) -> None:
        decision = self.engine.evaluate(ToolRequest(tool="memory.write"))
        self.assertFalse(decision.allowed)
        self.assertTrue(decision.approval_required)

    def test_approval_granted_allows_approval_required_tool(self) -> None:
        decision = self.engine.evaluate(
            ToolRequest(tool="memory.write"),
            approval_granted=True,
        )
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.approval_required)

    def test_cloud_context_does_not_add_approval_for_low_risk_tool(self) -> None:
        context = PermissionContext(cloud_model_active=True)
        decision = self.engine.evaluate(ToolRequest(tool="memory.read"), context)
        self.assertTrue(decision.allowed)
        self.assertFalse(decision.approval_required)


if __name__ == "__main__":
    unittest.main()
