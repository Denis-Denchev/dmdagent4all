from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class RiskLevel(IntEnum):
    HARMLESS_LOCAL_READ = 0
    READ_ONLY_EXTERNAL = 1
    LOCAL_DRAFT_OR_WRITE = 2
    MODIFY_EXTERNAL = 3
    SEND_OR_EXECUTE = 4
    DANGEROUS_SYSTEM = 5


@dataclass(frozen=True)
class ToolManifest:
    name: str
    description: str
    risk: RiskLevel
    permissions: tuple[str, ...] = ()
    argument_schema: dict[str, Any] = field(default_factory=dict)
    approval_required: bool = False
    cloud_allowed: bool = False
    default_enabled: bool = False
    sandbox_required: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ToolManifest":
        return cls(
            name=str(data["name"]),
            description=str(data.get("description", "")),
            risk=RiskLevel(int(data.get("risk", 0))),
            permissions=tuple(str(item) for item in data.get("permissions", [])),
            argument_schema=dict(data.get("args_schema") or data.get("parameters") or {}),
            approval_required=bool(data.get("approval_required", False)),
            cloud_allowed=bool(data.get("cloud_allowed", False)),
            default_enabled=bool(data.get("default_enabled", False)),
            sandbox_required=bool(data.get("sandbox_required", False)),
        )


@dataclass(frozen=True)
class ToolRequest:
    tool: str
    args: dict[str, Any] = field(default_factory=dict)
    reason: str = ""


@dataclass(frozen=True)
class PermissionContext:
    granted_permissions: frozenset[str] = frozenset()
    enabled_tools: frozenset[str] = frozenset()
    disabled_tools: frozenset[str] = frozenset()
    cloud_model_active: bool = False
    cloud_context_approved: bool = False
    approval_risk_threshold: int = 3


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    approval_required: bool
    reason: str
    risk: RiskLevel | None = None
    missing_permissions: tuple[str, ...] = ()

    @classmethod
    def allow(cls, reason: str, risk: RiskLevel) -> "PermissionDecision":
        return cls(allowed=True, approval_required=False, reason=reason, risk=risk)

    @classmethod
    def require_approval(cls, reason: str, risk: RiskLevel) -> "PermissionDecision":
        return cls(allowed=False, approval_required=True, reason=reason, risk=risk)

    @classmethod
    def deny(
        cls,
        reason: str,
        risk: RiskLevel | None = None,
        missing_permissions: tuple[str, ...] = (),
    ) -> "PermissionDecision":
        return cls(
            allowed=False,
            approval_required=False,
            reason=reason,
            risk=risk,
            missing_permissions=missing_permissions,
        )
