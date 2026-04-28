from __future__ import annotations

from dmdagent4all.permissions.models import (
    PermissionContext,
    PermissionDecision,
    ToolManifest,
    ToolRequest,
)


class PermissionEngine:
    def __init__(self, manifests: dict[str, ToolManifest]) -> None:
        self._manifests = manifests

    @property
    def manifests(self) -> dict[str, ToolManifest]:
        return dict(self._manifests)

    def evaluate(
        self,
        request: ToolRequest,
        context: PermissionContext | None = None,
        *,
        approval_granted: bool = False,
    ) -> PermissionDecision:
        context = context or PermissionContext()
        manifest = self._manifests.get(request.tool)
        if manifest is None:
            return PermissionDecision.deny(f"Unknown tool: {request.tool}")

        enabled = manifest.default_enabled or manifest.name in context.enabled_tools
        if manifest.name in context.disabled_tools:
            enabled = False
        if not enabled:
            return PermissionDecision.deny(
                f"Tool is disabled: {manifest.name}",
                risk=manifest.risk,
            )

        missing = tuple(
            permission
            for permission in manifest.permissions
            if permission not in context.granted_permissions
        )
        if missing:
            return PermissionDecision.deny(
                "Missing required permissions.",
                risk=manifest.risk,
                missing_permissions=missing,
            )

        if (
            context.cloud_model_active
            and not manifest.cloud_allowed
            and not context.cloud_context_approved
            and not approval_granted
        ):
            return PermissionDecision.require_approval(
                "This tool may expose private context to a cloud model.",
                risk=manifest.risk,
            )

        if manifest.approval_required and not approval_granted:
            return PermissionDecision.require_approval(
                "Tool manifest requires approval.",
                risk=manifest.risk,
            )

        if int(manifest.risk) >= context.approval_risk_threshold and not approval_granted:
            return PermissionDecision.require_approval(
                f"Risk level {int(manifest.risk)} requires approval.",
                risk=manifest.risk,
            )

        return PermissionDecision.allow("Allowed by policy.", risk=manifest.risk)
