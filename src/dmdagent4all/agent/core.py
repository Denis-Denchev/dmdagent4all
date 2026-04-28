from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.agent.planner import LLMPlanner, PlannerError
from dmdagent4all.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.registry import ToolExecutionError, ToolRegistry


@dataclass(frozen=True)
class AgentResponse:
    status: str
    message: str
    data: dict[str, Any] | None = None


class AgentCore:
    def __init__(
        self,
        *,
        permission_engine: PermissionEngine,
        tool_registry: ToolRegistry,
        permission_context: PermissionContext,
        runtime_context: ToolRuntimeContext,
        audit_store: AuditStore,
        planner: LLMPlanner | None = None,
    ) -> None:
        self.permission_engine = permission_engine
        self.tool_registry = tool_registry
        self.permission_context = permission_context
        self.runtime_context = runtime_context
        self.audit_store = audit_store
        self.planner = planner

    def handle_text(self, text: str) -> AgentResponse:
        stripped = text.strip()
        if stripped == "/tools":
            return self.handle_tool_request(
                ToolRequest(
                    tool="system.list_enabled_tools",
                    args={},
                    reason="User requested enabled tools.",
                )
            )
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
                return self.handle_tool_request(
                    ToolRequest(
                        tool=str(payload["tool"]),
                        args=dict(payload.get("args", {})),
                        reason=str(payload.get("reason", "")),
                    )
                )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return AgentResponse(
                    status="error",
                    message=f"Invalid tool request JSON: {exc}",
                )

        if self.planner is None:
            return AgentResponse(
                status="ok",
                message=(
                    "Agent core is initialized without a planner. Use /tools or send a "
                    "structured tool request JSON."
                ),
            )

        try:
            plan = self.planner.plan(
                user_message=text,
                manifests=self.tool_registry.manifests,
                response_language=self.runtime_context.config.get("llm", {}).get(
                    "response_language",
                    "auto",
                ),
            )
        except (PlannerError, OSError, RuntimeError) as exc:
            return AgentResponse(
                status="error",
                message=f"LLM planner failed: {exc}",
            )

        if plan.tool_request is not None:
            return self.handle_tool_request(plan.tool_request)

        return AgentResponse(
            status="ok",
            message=plan.final_message or "",
            data={"planner": "llm"},
        )

    def handle_tool_request(self, request: ToolRequest) -> AgentResponse:
        decision = self.permission_engine.evaluate(request, self.permission_context)
        self.audit_store.record_tool_call(
            tool=request.tool,
            risk=None if decision.risk is None else int(decision.risk),
            args=request.args,
            decision=decision.reason,
            result_status="blocked" if not decision.allowed else None,
        )

        if decision.approval_required:
            approval_id = self.audit_store.record_approval(
                tool=request.tool,
                risk=None if decision.risk is None else int(decision.risk),
                args=request.args,
                request_reason=request.reason,
                decision_reason=decision.reason,
            )
            self.audit_store.record_event(
                AuditEvent(
                    event_type="approval_required",
                    tool=request.tool,
                    risk=None if decision.risk is None else int(decision.risk),
                    approved=False,
                    result_status="pending",
                    metadata={"reason": decision.reason},
                )
            )
            return AgentResponse(
                status="approval_required",
                message=decision.reason,
                data={
                    "approval_id": approval_id,
                    "tool": request.tool,
                    "risk": None if decision.risk is None else int(decision.risk),
                },
            )

        if not decision.allowed:
            return AgentResponse(
                status="denied",
                message=decision.reason,
                data={"missing_permissions": list(decision.missing_permissions)},
            )

        try:
            result = self.tool_registry.execute(
                request.tool,
                request.args,
                self.runtime_context,
            )
        except ToolExecutionError as exc:
            return AgentResponse(status="error", message=str(exc))

        self.audit_store.record_event(
            AuditEvent(
                event_type="tool_call",
                tool=request.tool,
                risk=None if decision.risk is None else int(decision.risk),
                approved=True,
                result_status="success",
            )
        )
        return AgentResponse(status="ok", message="Tool executed.", data=result)
