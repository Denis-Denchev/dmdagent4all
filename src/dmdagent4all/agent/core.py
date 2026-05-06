from __future__ import annotations

import json
import calendar
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.agent.history import ChatHistory
from dmdagent4all.agent.planner import LLMPlanner, PlannerError
from dmdagent4all.config import save_config
from dmdagent4all.memory import MemoryManager
from dmdagent4all.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdagent4all.sandbox import TerminalPolicy
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.registry import ToolExecutionError, ToolRegistry


@dataclass(frozen=True)
class AgentResponse:
    status: str
    message: str
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryEntry:
    path: str
    text: str


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
        chat_history: ChatHistory | None = None,
    ) -> None:
        self.permission_engine = permission_engine
        self.tool_registry = tool_registry
        self.permission_context = permission_context
        self.runtime_context = runtime_context
        self.audit_store = audit_store
        self.planner = planner
        self._cloud_context_approved = permission_context.cloud_context_approved
        self.chat_history = chat_history or ChatHistory(runtime_context.workspace_root / "chat_history.json")

    def handle_text(self, text: str, *, session_id: str = "default") -> AgentResponse:
        stripped = text.strip()
        response = self._handle_text(stripped, text, session_id=session_id)
        self._record_chat_turn(session_id, stripped, response)
        return response

    def _handle_text(self, stripped: str, text: str, *, session_id: str) -> AgentResponse:
        conversation_context = self._conversation_context(session_id)
        if stripped == "/tools":
            return self.handle_tool_request(
                ToolRequest(
                    tool="system.list_enabled_tools",
                    args={},
                    reason="User requested enabled tools.",
                )
            )
        approval_action = self._handle_approval_action_from_text(stripped)
        if approval_action is not None:
            return approval_action
        identity_update = _handle_identity_update(stripped, self.runtime_context)
        if identity_update is not None:
            return identity_update
        memory_organize = _memory_organize_request_from_text(stripped)
        if memory_organize is not None:
            return self.handle_tool_request(memory_organize)
        memory_organize_missing_source = _memory_organize_missing_source_response(stripped)
        if memory_organize_missing_source is not None:
            return memory_organize_missing_source
        llm_reminder_request = self._plan_reminder_request(stripped, session_id=session_id)
        if llm_reminder_request is not None:
            return self.handle_tool_request(llm_reminder_request)
        reminder_request = _reminder_request_from_text(stripped)
        if reminder_request is not None:
            return self.handle_tool_request(reminder_request)
        memory_update = _memory_write_request_from_text(stripped, self.runtime_context)
        if memory_update is not None:
            return self.handle_tool_request(memory_update)
        memory_answer = _answer_memory_recall_question(
            stripped,
            self.runtime_context,
            conversation_context=self.chat_history.format_recent(session_id, limit=6, max_chars=2500),
        )
        if memory_answer is not None:
            if self._can_send_private_context_to_llm():
                synthesized = self._synthesize_memory_recall_response(
                    stripped,
                    memory_answer,
                    conversation_context=conversation_context,
                )
                if synthesized is not None:
                    return synthesized
            return memory_answer
        clarification = self._answer_clarification_followup(stripped, session_id=session_id)
        if clarification is not None:
            return clarification
        browser_request = _browser_request_from_text(stripped)
        if browser_request is not None:
            return self.handle_tool_request(browser_request)
        terminal_request = _terminal_request_from_text(stripped)
        if terminal_request is not None:
            return self.handle_tool_request(terminal_request)
        routed = _route_without_llm(stripped)
        if routed is not None:
            return self.handle_tool_request(routed)
        fast_answer = _answer_without_llm(stripped, self.runtime_context.config)
        if fast_answer is not None and _is_fast_control_answer(stripped):
            return AgentResponse(
                status="ok",
                message=fast_answer,
                data={"planner": "deterministic"},
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
            if fast_answer is not None:
                return AgentResponse(
                    status="ok",
                    message=fast_answer,
                    data={"planner": "deterministic"},
                )
            return AgentResponse(
                status="ok",
                message=(
                    "Agent core is initialized without a planner. Use /tools or send a "
                    "structured tool request JSON."
                ),
            )

        try:
            llm_config = self.runtime_context.config.get("llm", {})
            memory_context = _load_memory_context(
                self.runtime_context,
                query=text,
                conversation_context=conversation_context,
                allow_cloud_context=self._cloud_context_approved,
            )
            now = datetime.now().astimezone()
            plan = self.planner.plan(
                user_message=text,
                manifests=self.tool_registry.manifests,
                profile=_profile_from_config(self.runtime_context.config),
                memory_context=memory_context,
                conversation_context=conversation_context,
                enabled_tools=_enabled_tools_from_config(
                    self.tool_registry.manifests,
                    self.runtime_context.config,
                ),
                response_language=llm_config.get("response_language", "auto"),
                current_time=now.isoformat(timespec="seconds"),
                timezone_name=now.tzname() or "",
                max_tokens=max(512, int(llm_config.get("planner_max_tokens", 192))),
                temperature=float(llm_config.get("planner_temperature", 0.0)),
                think=bool(llm_config.get("planner_think", False)),
            )
        except Exception as exc:
            fast_answer = _answer_without_llm(stripped, self.runtime_context.config)
            if fast_answer is not None:
                return AgentResponse(
                    status="ok",
                    message=fast_answer,
                    data={"planner": "deterministic", "fallback": "llm_unavailable"},
                )
            unavailable_answer = _answer_llm_unavailable(stripped, self.runtime_context.config, exc)
            if unavailable_answer is not None:
                return unavailable_answer
            return AgentResponse(
                status="error",
                message=f"LLM planner failed: {exc}",
        )

        if plan.tool_request is not None:
            request = _request_with_original_message(plan.tool_request, stripped)
            tool_response = self.handle_tool_request(request)
            if self._should_synthesize_tool_response(stripped, request, tool_response):
                return self._synthesize_tool_response(
                    stripped,
                    tool_response,
                    conversation_context=conversation_context,
                )
            return tool_response

        return AgentResponse(
            status="ok",
            message=plan.final_message or "",
            data={"planner": "llm"},
        )

    def _handle_approval_action_from_text(self, text: str) -> AgentResponse | None:
        action = _approval_action_from_text(text)
        if action is None:
            return None
        verb, approval_id = action
        if approval_id is None:
            pending = self.audit_store.list_approvals(status="pending", limit=1)
            if not pending:
                return AgentResponse(
                    status="ok",
                    message="No pending approvals.",
                    data={"planner": "deterministic"},
                )
            approval_id = int(pending[0]["id"])
        if verb == "approve":
            return self.approve_and_execute(approval_id)
        changed = self.audit_store.set_approval_status(approval_id, "denied")
        if not changed:
            return AgentResponse(
                status="not_found",
                message=f"No pending approval found: {approval_id}",
                data={"approval_id": approval_id},
            )
        return AgentResponse(
            status="ok",
            message=f"Approval denied: {approval_id}",
            data={"approval_id": approval_id},
        )

    def _should_synthesize_tool_response(
        self,
        user_message: str,
        request: ToolRequest,
        response: AgentResponse,
    ) -> bool:
        if self.planner is None or not hasattr(self.planner, "answer") or response.status != "ok":
            return False
        if request.tool not in {"memory.list", "memory.read"}:
            return False
        normalized = _normalize_for_match(user_message)
        explicit_file_request = any(
            phrase in normalized
            for phrase in {
                "list memory",
                "show memory",
                "show memory files",
                "memory files",
                "local memory files",
                "покажи файловете",
                "списък с памет",
            }
        )
        return not explicit_file_request

    def _synthesize_tool_response(
        self,
        user_message: str,
        tool_response: AgentResponse,
        *,
        allow_cloud_memory: bool = False,
        conversation_context: str = "",
    ) -> AgentResponse:
        if self.planner is None:
            return tool_response
        try:
            llm_config = self.runtime_context.config.get("llm", {})
            answer = self.planner.answer(
                user_message=user_message,
                profile=_profile_from_config(self.runtime_context.config),
                memory_context=_load_memory_context(
                    self.runtime_context,
                    allow_cloud_context=allow_cloud_memory,
                ),
                conversation_context=conversation_context,
                tool_result=tool_response.data or {},
                response_language=llm_config.get("response_language", "auto"),
                max_tokens=max(256, int(llm_config.get("planner_max_tokens", 192))),
                temperature=0.2,
                think=bool(llm_config.get("planner_think", False)),
            )
            if answer:
                return AgentResponse(status="ok", message=answer, data={"planner": "llm"})
        except (PlannerError, OSError, RuntimeError):
            return tool_response
        return tool_response

    def _synthesize_memory_recall_response(
        self,
        user_message: str,
        memory_answer: AgentResponse,
        *,
        conversation_context: str,
    ) -> AgentResponse | None:
        if self.planner is None or not hasattr(self.planner, "answer") or memory_answer.status != "ok":
            return None
        try:
            llm_config = self.runtime_context.config.get("llm", {})
            answer = self.planner.answer(
                user_message=user_message,
                profile=_profile_from_config(self.runtime_context.config),
                memory_context=_load_memory_context(
                    self.runtime_context,
                    allow_cloud_context=self._cloud_context_approved,
                ),
                conversation_context=conversation_context,
                tool_result={
                    "kind": "memory_recall",
                    "answer": memory_answer.message,
                    "data": memory_answer.data or {},
                },
                response_language=llm_config.get("response_language", "auto"),
                max_tokens=max(384, int(llm_config.get("planner_max_tokens", 192))),
                temperature=0.25,
                think=bool(llm_config.get("planner_think", False)),
            )
            if answer:
                return AgentResponse(status="ok", message=answer, data={"planner": "llm", "source": "memory_recall"})
        except (PlannerError, OSError, RuntimeError):
            return None
        return None

    def _plan_reminder_request(self, text: str, *, session_id: str = "default") -> ToolRequest | None:
        if self.planner is None or not _is_reminder_creation_text(text):
            return None
        if self.runtime_context.config.get("reminders", {}).get("prefer_llm_parser") is False:
            return None
        try:
            llm_config = self.runtime_context.config.get("llm", {})
            now = datetime.now().astimezone()
            plan = self.planner.plan(
                user_message=text,
                manifests=self.tool_registry.manifests,
                profile=_profile_from_config(self.runtime_context.config),
                memory_context=_load_memory_context(
                    self.runtime_context,
                    query=text,
                    conversation_context=self._conversation_context(session_id),
                    allow_cloud_context=self._cloud_context_approved,
                ),
                conversation_context=self._conversation_context(session_id),
                enabled_tools=_enabled_tools_from_config(
                    self.tool_registry.manifests,
                    self.runtime_context.config,
                ),
                response_language=llm_config.get("response_language", "auto"),
                current_time=now.isoformat(timespec="seconds"),
                timezone_name=now.tzname() or "",
                max_tokens=max(512, int(llm_config.get("planner_max_tokens", 192))),
                temperature=float(llm_config.get("planner_temperature", 0.0)),
                think=bool(llm_config.get("planner_think", False)),
            )
        except Exception:
            return None
        request = plan.tool_request
        if request is None or request.tool != "reminders.create":
            return None
        if not _valid_reminder_args(request.args):
            return None
        args = dict(request.args)
        args.setdefault("notes", text)
        _enrich_reminder_args_from_text(args, text)
        return _request_with_original_message(
            ToolRequest(tool=request.tool, args=args, reason=request.reason),
            text,
        )

    def _answer_clarification_followup(self, text: str, *, session_id: str = "default") -> AgentResponse | None:
        if not _is_clarification_followup(text):
            return None
        previous = _last_assistant_message(self.chat_history.recent(session_id, limit=8))
        if previous is None:
            return None
        if _looks_bulgarian(text):
            return AgentResponse(
                status="ok",
                message=f"Казано по-просто: {previous}",
                data={"planner": "deterministic", "source": "recent_conversation"},
            )
        return AgentResponse(
            status="ok",
            message=f"In simpler terms: {previous}",
            data={"planner": "deterministic", "source": "recent_conversation"},
        )

    def _can_send_private_context_to_llm(self) -> bool:
        return not self.permission_context.cloud_model_active or self._cloud_context_approved

    def _conversation_context(self, session_id: str) -> str:
        if self.permission_context.cloud_model_active and not self._cloud_context_approved:
            return ""
        return self.chat_history.format_recent(session_id, limit=12, max_chars=6000)

    def _record_chat_turn(self, session_id: str, user_message: str, response: AgentResponse) -> None:
        try:
            self.chat_history.append(session_id, "user", user_message)
            self.chat_history.append(session_id, "assistant", _history_text_from_response(response))
        except (OSError, ValueError):
            return

    def handle_tool_request(self, request: ToolRequest) -> AgentResponse:
        auto_approved = _is_auto_approved_terminal_request(request, self.runtime_context.config)
        decision = self.permission_engine.evaluate(
            request,
            self._permission_context(),
            approval_granted=auto_approved,
        )
        self.audit_store.record_tool_call(
            tool=request.tool,
            risk=None if decision.risk is None else int(decision.risk),
            args=request.args,
            decision=(
                f"auto_approved_allowlisted:{decision.reason}"
                if auto_approved
                else decision.reason
            ),
            result_status="blocked" if not decision.allowed else None,
        )

        if decision.approval_required or decision.allowed:
            terminal_policy_error = _terminal_policy_error(request, self.runtime_context.config)
            if terminal_policy_error is not None:
                return AgentResponse(
                    status="denied",
                    message=terminal_policy_error,
                    data={"missing_permissions": []},
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
                message=_approval_message(request, decision.reason),
                data={
                    "approval_id": approval_id,
                    "tool": request.tool,
                    "risk": None if decision.risk is None else int(decision.risk),
                },
            )

        if not decision.allowed:
            return AgentResponse(
                status="denied",
                message=_denial_message(request, decision.reason),
                data={"missing_permissions": list(decision.missing_permissions)},
            )

        return self._execute_tool_request(
            request,
            risk=None if decision.risk is None else int(decision.risk),
        )

    def approve_and_execute(self, approval_id: int) -> AgentResponse:
        approval = self.audit_store.get_approval(approval_id)
        if approval is None:
            return AgentResponse(
                status="not_found",
                message=f"Approval not found: {approval_id}",
            )
        if approval["status"] != "pending":
            return AgentResponse(
                status="denied",
                message=f"Approval is not pending. Current status: {approval['status']}",
                data={"approval_id": approval_id},
            )

        request = ToolRequest(
            tool=str(approval["tool"]),
            args=dict(approval["args"]),
            reason=str(approval.get("request_reason") or "Approved by user."),
        )
        decision = self.permission_engine.evaluate(
            request,
            self._permission_context(),
            approval_granted=True,
        )
        self.audit_store.record_tool_call(
            tool=request.tool,
            risk=None if decision.risk is None else int(decision.risk),
            args=request.args,
            decision=f"approved:{decision.reason}",
            result_status="blocked" if not decision.allowed else None,
        )
        if not decision.allowed:
            return AgentResponse(
                status="denied",
                message=decision.reason,
                data={
                    "approval_id": approval_id,
                    "missing_permissions": list(decision.missing_permissions),
                },
            )

        self.audit_store.set_approval_status(approval_id, "approved")
        response = self._execute_tool_request(
            request,
            risk=None if decision.risk is None else int(decision.risk),
            approval_id=approval_id,
        )
        if approval.get("decision_reason") == "This tool may expose private context to a cloud model.":
            self._cloud_context_approved = True
        self.audit_store.set_approval_status(
            approval_id,
            "executed" if response.status == "ok" else "failed",
        )
        original_message = _original_message_from_reason(str(approval.get("request_reason") or ""))
        if (
            response.status == "ok"
            and original_message
            and request.tool in {"memory.list", "memory.read"}
        ):
            return self._synthesize_tool_response(
                original_message,
                response,
                allow_cloud_memory=True,
            )
        return response

    def _permission_context(self) -> PermissionContext:
        return replace(
            self.permission_context,
            cloud_context_approved=self._cloud_context_approved,
        )

    def _execute_tool_request(
        self,
        request: ToolRequest,
        *,
        risk: int | None,
        approval_id: int | None = None,
    ) -> AgentResponse:
        try:
            result = self.tool_registry.execute(
                request.tool,
                request.args,
                self.runtime_context,
            )
        except (ToolExecutionError, ValueError, KeyError, OSError) as exc:
            return AgentResponse(status="error", message=str(exc))

        if result.get("status") == "not_implemented":
            return AgentResponse(
                status="not_implemented",
                message=str(result.get("message", "Tool is not implemented yet.")),
                data=result,
            )
        if result.get("status") == "connector_not_configured":
            return AgentResponse(
                status="not_configured",
                message=str(result.get("message", "Connector is not configured yet.")),
                data=result,
            )

        self.audit_store.record_event(
            AuditEvent(
                event_type="tool_call",
                tool=request.tool,
                risk=risk,
                approved=True,
                result_status="success",
                metadata={} if approval_id is None else {"approval_id": approval_id},
            )
        )
        return AgentResponse(status="ok", message=_tool_success_message(request), data=result)


def _route_without_llm(text: str) -> ToolRequest | None:
    normalized = _normalize_for_match(text)
    if normalized in {
        "/tools",
        "tools",
        "show tools",
        "list tools",
        "what tools do you have",
        "what tools are available",
        "какви инструменти имаш",
        "покажи инструментите",
    }:
        return ToolRequest(
            tool="system.list_enabled_tools",
            args={},
            reason="User requested available tools.",
        )
    if normalized in {"/memory", "memory list", "list memory"}:
        return ToolRequest(
            tool="memory.list",
            args={},
            reason="User requested local memory files.",
        )
    if normalized in {
        "/reminders",
        "reminders",
        "list reminders",
        "show reminders",
        "напомняния",
        "покажи напомняния",
    }:
        return ToolRequest(
            tool="reminders.list",
            args={"status": "pending"},
            reason="User requested local reminders.",
        )
    if "memory files" in normalized or "local memory" in normalized:
        if any(word in normalized for word in {"show", "list", "see", "display"}):
            return ToolRequest(
                tool="memory.list",
                args={},
                reason="User requested local memory files.",
            )
    return None


def _approval_action_from_text(text: str) -> tuple[str, int | None] | None:
    normalized = _normalize_for_match(text)
    approve_match = re.match(
        r"^(?:/approve|approve|approved|одобри|одобрявам)(?:\s+#?(?P<id>\d+))?$",
        normalized,
        flags=re.IGNORECASE,
    )
    if approve_match:
        raw_id = approve_match.group("id")
        return "approve", int(raw_id) if raw_id else None
    deny_match = re.match(
        r"^(?:/deny|deny|denied|откажи|отказвам)(?:\s+#?(?P<id>\d+))?$",
        normalized,
        flags=re.IGNORECASE,
    )
    if deny_match:
        raw_id = deny_match.group("id")
        return "deny", int(raw_id) if raw_id else None
    if normalized in {"approve", "approved", "одобри", "одобрявам", "готово"}:
        return "approve", None
    if normalized in {"deny", "denied", "откажи", "отказвам"}:
        return "deny", None
    return None


def _reminder_request_from_text(text: str) -> ToolRequest | None:
    stripped = text.strip()
    if not _is_reminder_creation_text(stripped):
        return None

    reminder = _parse_reminder_details(stripped)
    if reminder is None:
        return None
    args: dict[str, Any] = {
        "title": reminder["title"],
        "due_at": reminder["due_at"],
        "notes": stripped,
    }
    if reminder.get("event_at"):
        args["event_at"] = reminder["event_at"]
    if reminder.get("remind_before"):
        args["remind_before"] = reminder["remind_before"]
    if reminder.get("location"):
        args["location"] = reminder["location"]
    if reminder.get("action_url"):
        args["action_url"] = reminder["action_url"]
    return ToolRequest(
        tool="reminders.create",
        args=args,
        reason="User asked the agent to create a local reminder.",
    )


def _is_reminder_creation_text(text: str) -> bool:
    if _is_bulk_memory_organization_text(text):
        return False
    normalized = _normalize_for_match(text)
    if normalized in {
        "/reminders",
        "reminders",
        "list reminders",
        "show reminders",
        "напомняния",
        "покажи напомняния",
    }:
        return False
    return any(
        phrase in normalized
        for phrase in {
            "remind me",
            "notify me",
            "remind",
            "напомни",
            "напомниш",
        }
    )


def _valid_reminder_args(args: dict[str, Any]) -> bool:
    if not isinstance(args, dict):
        return False
    due_at = str(args.get("due_at") or "").strip()
    event_at = str(args.get("event_at") or "").strip()
    if not str(args.get("title") or "").strip() or not _looks_like_iso_datetime(due_at):
        return False
    return not event_at or _looks_like_iso_datetime(event_at)


def _enrich_reminder_args_from_text(args: dict[str, Any], text: str) -> None:
    if not str(args.get("location") or "").strip():
        location = _extract_location_from_reminder_text(text)
        if location:
            args["location"] = location


def _looks_like_iso_datetime(value: str) -> bool:
    if not value.strip():
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True


def _extract_location_from_reminder_text(text: str) -> str | None:
    patterns = [
        r"\b(?:за|до|на)\s+адрес\s+(?P<location>.+)$",
        r"\baddress\s+(?P<location>.+)$",
        r"\b(?:to|at)\s+(?P<location>(?:boulevard|blvd\.?|street|st\.?)\s+.+)$",
        r"\b(?P<location>(?:булевард|бул\.?|улица|ул\.?|площад|пл\.?)\s+.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        location = _clean_extracted_location(match.group("location"))
        if location:
            return location
    return None


def _clean_extracted_location(value: str) -> str:
    location = value.strip().strip(" .,!?:;\"'")
    location = re.split(
        r"\s+(?:и\s+)?(?:искам\s+да\s+ми\s+)?(?:напомни|напомниш|remind|notify)\b",
        location,
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0]
    location = re.sub(r"\bномер\s+", "No. ", location, flags=re.IGNORECASE)
    location = re.sub(r"\s+", " ", location).strip().strip(" .,!?:;\"'")
    return location[:300]


def _parse_reminder_details(text: str) -> dict[str, str] | None:
    event = _relative_event_from_text(text)
    if event is not None:
        event_at, match, unit, explicit_event_time = event
        due_at = _same_day_notification_due_at(text, event_at)
        remind_before: str | None = None
        if due_at is None:
            before_delta = _before_event_notification_delta(text)
            if before_delta is not None:
                delta, label = before_delta
                due_at = event_at - delta
                remind_before = label
        if due_at is None:
            due_at = event_at

        title = _title_after_relative_delay(text, match) or _clean_reminder_title(text)
        if not title:
            return None
        result = {
            "title": title,
            "due_at": due_at.isoformat(timespec="seconds"),
        }
        if explicit_event_time or remind_before or due_at != event_at:
            result["event_at"] = event_at.isoformat(timespec="seconds")
        if remind_before:
            result["remind_before"] = remind_before
        _enrich_reminder_args_from_text(result, text)
        return result

    due_at = _tomorrow_reminder_due_at(text)
    if due_at is None:
        return None
    title = _clean_reminder_title(text)
    if not title:
        return None
    result = {"title": title, "due_at": due_at}
    _enrich_reminder_args_from_text(result, text)
    return result


def _relative_event_from_text(
    text: str,
) -> tuple[datetime, re.Match[str], str, bool] | None:
    pattern = re.compile(
        r"(?:\b(?:after|in)\s+(?P<amount_en>\d+|one|a|an)\s+"
        r"(?P<unit_en>seconds?|minutes?|mins?|hours?|days?|weeks?|months?)|"
        r"\bслед\s+(?P<amount_bg>\d+|един|една)\s+"
        r"(?P<unit_bg>секунда|секунди?|минута|минути?|часа?|дни|ден|седмици?|седмица|месеца?|месец))",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text.strip())
    if not match:
        return None
    amount = _parse_amount(match.group("amount_en") or match.group("amount_bg") or "")
    unit = (match.group("unit_en") or match.group("unit_bg") or "").lower()
    if amount is None:
        return None

    now = datetime.now().astimezone()
    event_at = _add_relative_time(now, amount, unit)
    if event_at is None:
        return None

    explicit_event_time = False
    time_match = re.match(
        r"\s*(?:at|в)\s+(?P<time>\d{1,2}(?:[:.]\d{2})?)",
        text[match.end() :],
        flags=re.IGNORECASE,
    )
    if time_match and _unit_is_day_or_larger(unit):
        parsed_time = _parse_time_of_day(time_match.group("time"))
        if parsed_time is not None:
            hour, minute = parsed_time
            event_at = event_at.replace(hour=hour, minute=minute, second=0, microsecond=0)
            explicit_event_time = True

    return event_at, match, unit, explicit_event_time


def _add_relative_time(current: datetime, amount: int, unit: str) -> datetime | None:
    if unit.startswith(("second", "секунд")):
        return current + timedelta(seconds=amount)
    if unit.startswith(("minute", "min", "минут")):
        return current + timedelta(minutes=amount)
    if unit.startswith(("hour", "час")):
        return current + timedelta(hours=amount)
    if unit.startswith(("day", "д")) or unit == "ден":
        return current + timedelta(days=amount)
    if unit.startswith(("week", "седмиц")):
        return current + timedelta(weeks=amount)
    if unit.startswith(("month", "месец", "месеца")):
        return _add_months(current, amount)
    return None


def _add_months(current: datetime, months: int) -> datetime:
    month_index = current.month - 1 + months
    year = current.year + month_index // 12
    month = month_index % 12 + 1
    day = min(current.day, calendar.monthrange(year, month)[1])
    return current.replace(year=year, month=month, day=day)


def _unit_is_day_or_larger(unit: str) -> bool:
    return (
        unit.startswith(("day", "week", "month", "д", "седмиц", "месец", "месеца"))
        or unit == "ден"
    )


def _same_day_notification_due_at(text: str, event_at: datetime) -> datetime | None:
    patterns = [
        r"(?:i\s+want\s+you\s+to\s+remind\s+me|remind\s+me|notify\s+me)"
        r"\s+at\s+(?P<time>\d{1,2}(?:[:.]\d{2})?)"
        r".*?\b(?:same\s+day|that\s+day|day\s+of)",
        r"(?:искам\s+да\s+ми\s+напомниш|напомн(?:и|иш)\s+ми)"
        r"\s+в\s+(?P<time>\d{1,2}(?:[:.]\d{2})?)"
        r".*?\b(?:в\s+деня|същия\s+ден)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        parsed_time = _parse_time_of_day(match.group("time"))
        if parsed_time is None:
            continue
        hour, minute = parsed_time
        return event_at.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return None


def _before_event_notification_delta(text: str) -> tuple[timedelta, str] | None:
    patterns = [
        r"(?:remind\s+me|notify\s+me)\s+(?P<amount_en>\d+|one|a|an)\s+"
        r"(?P<unit_en>minutes?|mins?|hours?|days?|weeks?)\s+before",
        r"напомн(?:и|иш)(?:\s+ми)?\s+(?P<amount_bg>\d+|един|една)\s+"
        r"(?P<unit_bg>минути?|часа?|дни|ден|седмици?|седмица)\s+преди(?:\s+това)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        amount = _parse_amount(_match_group(match, "amount_en") or _match_group(match, "amount_bg") or "")
        unit = (_match_group(match, "unit_en") or _match_group(match, "unit_bg") or "").lower()
        if amount is None:
            return None
        if unit.startswith(("minute", "min", "минут")):
            return timedelta(minutes=amount), f"{amount} minute(s) before"
        if unit.startswith(("hour", "час")):
            return timedelta(hours=amount), f"{amount} hour(s) before"
        if unit.startswith(("day", "д")) or unit == "ден":
            return timedelta(days=amount), f"{amount} day(s) before"
        if unit.startswith(("week", "седмиц")):
            return timedelta(weeks=amount), f"{amount} week(s) before"
    return None


def _match_group(match: re.Match[str], name: str) -> str | None:
    try:
        return match.group(name)
    except IndexError:
        return None


def _parse_time_of_day(value: str) -> tuple[int, int] | None:
    match = re.match(r"^(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?$", value.strip())
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour, minute


def _title_after_relative_delay(text: str, match: re.Match[str]) -> str | None:
    before = _normalize_for_match(text[: match.start()])
    if not any(word in before for word in {"remind", "напомни"}):
        return None
    tail = text[match.end() :].strip()
    tail = re.sub(
        r"^(?:at|в)\s+\d{1,2}(?:[:.]\d{2})?\s*",
        "",
        tail,
        flags=re.IGNORECASE,
    ).strip(" ,.!?:;")
    if not tail:
        return None
    normalized_tail = _normalize_for_match(tail)
    if any(
        marker in normalized_tail
        for marker in {"same day", "that day", "в деня", "същия ден", "преди това", "искам да ми напомниш"}
    ):
        return None
    return _remove_location_phrase_from_title(tail.strip().strip(" .,!?:;\"'"))[:300]


def _relative_reminder_due_at(text: str) -> str | None:
    pattern = re.compile(
        r"(?:\b(?:after|in)\s+(?P<amount_en>\d+|one|a|an)\s+"
        r"(?P<unit_en>seconds?|minutes?|mins?|hours?|days?)|"
        r"\bслед\s+(?P<amount_bg>\d+|един|една)\s+"
        r"(?P<unit_bg>секунда|секунди?|минута|минути?|часа?|дни?))\s*$",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text.strip())
    if not match:
        return None
    amount = _parse_amount(match.group("amount_en") or match.group("amount_bg") or "")
    unit = (match.group("unit_en") or match.group("unit_bg") or "").lower()
    if amount is None:
        return None
    if unit.startswith(("second", "секунд")):
        delta = timedelta(seconds=amount)
    elif unit.startswith(("minute", "min", "минут")):
        delta = timedelta(minutes=amount)
    elif unit.startswith(("hour", "час")):
        delta = timedelta(hours=amount)
    elif unit.startswith(("day", "д")):
        delta = timedelta(days=amount)
    else:
        return None
    return (datetime.now().astimezone() + delta).isoformat(timespec="seconds")


def _tomorrow_reminder_due_at(text: str) -> str | None:
    pattern = re.compile(
        r"(?:tomorrow\s+(?:at\s+)?|утре\s+(?:в\s+)?)"
        r"(?P<hour>\d{1,2})(?:[:.](?P<minute>\d{2}))?",
        flags=re.IGNORECASE,
    )
    match = pattern.search(text)
    if not match:
        return None
    hour = int(match.group("hour"))
    minute = int(match.group("minute") or 0)
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    now = datetime.now().astimezone()
    due = (now + timedelta(days=1)).replace(hour=hour, minute=minute, second=0, microsecond=0)
    return due.isoformat(timespec="seconds")


def _clean_reminder_title(text: str) -> str:
    title = re.sub(
        r"\s+(?:and\s+)?(?:i\s+want\s+you\s+to\s+)?(?:remind\s+me|notify\s+me)"
        r"\s+(?:\d+|one|a|an)\s+"
        r"(?:minutes?|mins?|hours?|days?|weeks?)\s+before.*$",
        "",
        text.strip(),
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\s+(?:и\s+)?(?:искам\s+да\s+ми\s+)?напомн(?:и|иш)(?:\s+ми)?\s+"
        r"(?:\d+|един|една)\s+(?:минути?|минута|часа?|дни|ден|седмици?|седмица)"
        r"\s+преди(?:\s+това)?.*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\s+(?:and\s+)?(?:i\s+want\s+you\s+to\s+remind\s+me|remind\s+me|notify\s+me)"
        r"\s+at\s+\d{1,2}(?:[:.]\d{2})?.*?\b(?:same\s+day|that\s+day|day\s+of).*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"\s+(?:и\s+)?(?:искам\s+да\s+ми\s+напомниш|напомн(?:и|иш)\s+ми)"
        r"\s+в\s+\d{1,2}(?:[:.]\d{2})?.*?\b(?:в\s+деня|същия\s+ден).*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(
        r"(?:\b(?:after|in)\s+(?:\d+|one|a|an)\s+"
        r"(?:seconds?|minutes?|mins?|hours?|days?|weeks?|months?)|"
        r"\bслед\s+(?:\d+|един|една)\s+"
        r"(?:секунда|секунди?|минута|минути?|часа?|дни|ден|седмици?|седмица|месеца?|месец)"
        r"(?:\s+(?:at|в)\s+\d{1,2}(?:[:.]\d{2})?)?)\s*",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^(?:запомни\s+)?(?:и\s+)?ми\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^запомни\s+и\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(
        r"^(?:please\s+)?remind\s+me(?:\s+(?:that|to))?\s+",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"^напомни(?:\s+ми)?(?:\s+(?:че|да))?\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(r"^съм\s+на\s+", "", title, flags=re.IGNORECASE)
    title = re.sub(
        r"\s+and\s+i\s+want\s+you\s+to\s+remind\s+me\s*$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    title = re.sub(r"\s+и\s+искам\s+да\s+ми\s+напомниш\s*$", "", title, flags=re.IGNORECASE)
    return _remove_location_phrase_from_title(title.strip().strip(" .,!?:;\"'"))


def _remove_location_phrase_from_title(title: str) -> str:
    cleaned = re.sub(
        r"\s+(?:за|до|на)\s+адрес\s+.+$",
        "",
        title,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s+(?:to|at)\s+address\s+.+$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip().strip(" .,!?:;\"'") or title


def _parse_amount(value: str) -> int | None:
    normalized = value.strip().lower()
    if normalized in {"one", "a", "an", "един", "една"}:
        return 1
    if normalized.isdigit():
        amount = int(normalized)
        return amount if 1 <= amount <= 365 else None
    return None


def _browser_request_from_text(text: str) -> ToolRequest | None:
    if _is_bulk_memory_organization_text(text):
        return None
    scrape_request = _browser_scrape_markdown_request_from_text(text)
    if scrape_request is not None:
        return scrape_request
    normalized = _normalize_for_match(text)
    if not any(word in normalized for word in {"open", "visit", "отвори", "отвориш"}):
        return None
    url = _extract_urlish_target(text)
    if url is None:
        return None
    return ToolRequest(
        tool="browser.open",
        args={"url": url},
        reason="User asked the agent to open a web page through the guarded browser-read tool.",
    )


def _browser_scrape_markdown_request_from_text(text: str) -> ToolRequest | None:
    normalized = _normalize_for_match(text)
    if not _is_browser_scrape_markdown_text(normalized):
        return None
    url = _extract_urlish_target(text)
    if url is None:
        return None
    args: dict[str, Any] = {
        "url": url,
        "instructions": text.strip(),
    }
    filename = _extract_requested_markdown_filename(text)
    if filename:
        args["filename"] = filename
    return ToolRequest(
        tool="browser.scrape_markdown",
        args=args,
        reason="User asked the agent to scrape a web page and save the result as Markdown.",
    )


def _is_browser_scrape_markdown_text(normalized: str) -> bool:
    return any(
        phrase in normalized
        for phrase in {
            "scrape",
            "scraping",
            "scraped",
            "scrapefiles",
            "collect information",
            "collect info",
            "extract information",
            "extract info",
            "extract content",
            "save as markdown",
            "convert to markdown",
            "markdown format",
            "md format",
            "скрейп",
            "скрейпва",
            "скрейпни",
            "изскрейп",
            "събери информация",
            "събереш информация",
            "събери инфо",
            "извлечи информация",
            "извади информация",
            "запази като markdown",
            "запази в markdown",
            "md формат",
        }
    )


def _extract_requested_markdown_filename(text: str) -> str | None:
    match = re.search(
        r"(?:filename|file|файл|име)\s*[:=]\s*(?P<name>[\w.\-]+\.md)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return match.group("name").strip(" .,!?:;\"'")


def _extract_urlish_target(text: str) -> str | None:
    lower = text.lower()
    if "google" in lower or "гугъл" in lower:
        return "https://www.google.com"
    explicit_match = re.search(r"https?://[^\s]+", text, flags=re.IGNORECASE)
    if explicit_match:
        return explicit_match.group(0).strip(" .,!?:;\"'")
    domain_match = re.search(r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?", text, flags=re.IGNORECASE)
    if domain_match:
        candidate = domain_match.group(0).strip(" .,!?:;\"'")
        if candidate.casefold().endswith(".md"):
            return None
        return candidate
    return None


def _terminal_request_from_text(text: str) -> ToolRequest | None:
    normalized = _normalize_for_match(text)
    mentions_terminal = "terminal" in normalized or "терминал" in normalized
    mentions_readme_cat = re.search(r"\bcat\s+(readme\.md|README\.md)\b", text, flags=re.IGNORECASE)
    if not mentions_terminal and mentions_readme_cat is None:
        return None
    command: list[str] | None = None
    if re.search(r"\b(?:type|run|execute|напиши|изпълни|пусни|въведи)\s+ls\b", text, flags=re.IGNORECASE):
        command = ["ls"]
    elif mentions_terminal and re.search(r"\bls\b", text, flags=re.IGNORECASE):
        command = ["ls"]
    elif re.search(r"\b(?:type|run|execute|напиши|изпълни|пусни|въведи)\s+pwd\b", text, flags=re.IGNORECASE):
        command = ["pwd"]
    elif mentions_terminal and re.search(r"\bpwd\b", text, flags=re.IGNORECASE):
        command = ["pwd"]
    elif re.search(r"\bgit\s+status\b", text, flags=re.IGNORECASE):
        command = ["git", "status"]
    elif re.search(r"\bgit\s+diff\b", text, flags=re.IGNORECASE):
        command = ["git", "diff"]
    else:
        if mentions_readme_cat:
            command = ["cat", mentions_readme_cat.group(1)]
    if command is None:
        return None
    return ToolRequest(
        tool="terminal.run",
        args={"command": command},
        reason="User asked to run a simple allowlist-style terminal command.",
    )


def _terminal_policy_error(request: ToolRequest, config: dict[str, Any]) -> str | None:
    if request.tool != "terminal.run":
        return None
    raw_command = request.args.get("command")
    if not isinstance(raw_command, list):
        return "terminal.run requires command as a string array"
    try:
        TerminalPolicy.from_config(config).validate([str(part) for part in raw_command])
    except PermissionError as exc:
        return str(exc)
    return None


def _is_auto_approved_terminal_request(request: ToolRequest, config: dict[str, Any]) -> bool:
    if request.tool != "terminal.run":
        return False
    terminal = config.get("terminal", {})
    if not bool(terminal.get("auto_approve_allowlisted", False)):
        return False
    raw_command = request.args.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        return False
    command = tuple(str(part) for part in raw_command if str(part))
    allowed = {
        tuple(str(part) for part in item)
        for item in terminal.get("allowed_commands", [])
        if isinstance(item, list) and item
    }
    return command in allowed


def _memory_write_request_from_text(
    text: str,
    runtime_context: ToolRuntimeContext,
) -> ToolRequest | None:
    fact = _extract_memory_fact(text)
    if fact is None:
        return None
    if _is_short_term_memory_text(text):
        fact = _clean_short_term_memory_fact(fact)
        ttl_hours = _short_term_ttl_hours_from_text(text)
        return ToolRequest(
            tool="memory.write",
            args={
                "path": "auto",
                "title": fact[:80],
                "body": fact,
                "memory_scope": "short-term",
                "ttl_hours": ttl_hours,
                "metadata": {
                    "type": "short_term_note",
                    "memory_scope": "short-term",
                    "ttl_hours": ttl_hours,
                    "source": "chat",
                    "confidence": "medium",
                },
            },
            reason="User asked the agent to remember temporary context.",
        )
    fact = _clean_long_term_memory_fact(fact)
    if _wants_new_long_term_memory_file(text):
        return ToolRequest(
            tool="memory.write",
            args={
                "path": "auto",
                "title": fact[:80],
                "body": fact,
                "memory_scope": "long-term",
                "metadata": {
                    "type": "long_term_note",
                    "memory_scope": "long-term",
                    "source": "chat",
                    "confidence": "high",
                },
            },
            reason="User asked the agent to remember a long-term fact in a new memory file.",
        )
    path = "long-term/facts/personal.md"
    manager = MemoryManager(runtime_context.memory_root)
    existing_body = ""
    try:
        existing_body = _strip_frontmatter(manager.read(path)).strip()
    except FileNotFoundError:
        existing_body = ""

    lines = [line.rstrip() for line in existing_body.splitlines() if line.strip()]
    note = f"- {fact}"
    if note not in lines:
        lines.append(note)
    return ToolRequest(
        tool="memory.write",
        args={
            "path": path,
            "body": "\n".join(lines),
            "metadata": {
                "type": "personal_fact",
                "memory_scope": "long-term",
                "source": "chat",
                "confidence": "high",
            },
        },
        reason="User asked the agent to remember a personal fact.",
    )


def _memory_organize_request_from_text(text: str) -> ToolRequest | None:
    if not _is_bulk_memory_organization_text(text):
        return None
    if not _has_embedded_bulk_memory_source(text):
        return None
    source = _extract_bulk_memory_source(text)
    if len(source.strip()) < 100:
        return None
    return ToolRequest(
        tool="memory.organize_long_term",
        args={
            "source_markdown": source,
            "today": datetime.now().astimezone().date().isoformat(),
        },
        reason="User asked to split a large long-term memory Markdown source into modular category files.",
    )


def _is_bulk_memory_organization_text(text: str) -> bool:
    normalized = _normalize_for_match(text)
    strong_markers = {
        "# long-term memory",
        "long-term memory —",
        "memory/readme.md",
        "high priority файлове",
        "high-priority for retrieval",
        "препоръчителна структура",
        "разделиш на ясни категории",
        "модулна long-term memory",
        "modular long-term memory",
        "грешка в организирането на memory",
        "празни категории",
        "празни лонг търм",
        "no source notes were found",
        "source memory content",
        "не създавай празни категории",
        "не оставяй",
        "разпредели съществуващите реални факти",
    }
    return any(marker in normalized for marker in strong_markers) and any(
        marker in normalized
        for marker in {
            "markdown",
            ".md",
            "memory/",
            "памет",
            "memory",
        }
    )


def _has_embedded_bulk_memory_source(text: str) -> bool:
    return any(
        marker in text
        for marker in {
            "# Long-Term Memory",
            "# Long-term Memory",
            "# long-term Memory",
            "and this is the knowlage",
            "and this is the knowledge",
        }
    )


def _memory_organize_missing_source_response(text: str) -> AgentResponse | None:
    if not _is_bulk_memory_organization_text(text) or _has_embedded_bulk_memory_source(text):
        return None
    return AgentResponse(
        status="ok",
        message="Нямам достъп до source memory content. Моля, дай ми го отново.",
        data={"planner": "deterministic", "missing": "source_memory_content"},
    )


def _extract_bulk_memory_source(text: str) -> str:
    markers = [
        "# Long-Term Memory",
        "# Long-term Memory",
        "and this is the knowlage",
        "and this is the knowledge",
    ]
    for marker in markers:
        index = text.find(marker)
        if index != -1:
            if marker.startswith("#"):
                return text[index:]
            return text[index + len(marker) :].strip()
    return text


def _answer_memory_recall_question(
    text: str,
    runtime_context: ToolRuntimeContext,
    *,
    conversation_context: str = "",
) -> AgentResponse | None:
    recall_text = _memory_recall_query_text(text, conversation_context)
    explicit_recall = _is_memory_recall_question(recall_text)

    entries = _load_memory_entries(runtime_context)
    bulgarian = _looks_bulgarian(text)
    if not entries:
        if not explicit_recall:
            return None
        return AgentResponse(
            status="ok",
            message=(
                "Не намирам запазени факти в локалната memory още."
                if bulgarian
                else "I do not have saved facts in local memory yet."
            ),
            data={"planner": "deterministic"},
        )

    ranked = _rank_memory_entries(recall_text, entries)
    if not explicit_recall and not _is_likely_memory_lookup_question(recall_text, ranked):
        return None

    if _is_broad_memory_recall_question(recall_text):
        matches = _broad_memory_matches(entries)
    else:
        if _asks_about_proxmox(recall_text):
            proxmox_matches = [
                entry
                for entry in ranked
                if any(marker in _normalize_for_match(f"{entry.path} {entry.text}") for marker in {"proxmox", "проксмокс"})
            ]
            matches = (proxmox_matches or ranked)[:3]
        elif _asks_for_port(recall_text):
            port_matches = [entry for entry in ranked if _memory_entry_contains_port(entry)]
            subject_terms = _memory_subject_query_terms(recall_text)
            subject_port_matches = [
                entry
                for entry in port_matches
                if any(term in _normalize_for_match(f"{entry.path} {entry.text}") for term in subject_terms)
            ]
            matches = (subject_port_matches or port_matches or ranked)[:4]
        elif _asks_for_address_or_url(recall_text) and _asks_about_servers(recall_text):
            access_matches = [entry for entry in ranked if _memory_entry_contains_machine_access_endpoint(entry)]
            address_matches = [entry for entry in ranked if _memory_entry_contains_address_or_url(entry)]
            matches = (access_matches or address_matches or ranked)[:3]
        elif _asks_for_address_or_url(recall_text) or _asks_about_machine_access(recall_text):
            address_matches = [entry for entry in ranked if _memory_entry_contains_address_or_url(entry)]
            matches = (address_matches or ranked)[:3]
        elif _asks_about_servers(recall_text) or _asks_about_models(recall_text):
            owned_matches = [entry for entry in ranked if _is_relevant_owned_inventory_memory_entry(entry, recall_text)]
            matches = (owned_matches or ranked)[:3]
        else:
            matches = ranked[:6]

    if not matches:
        if not explicit_recall:
            return None
        return AgentResponse(
            status="ok",
            message=(
                "Не намирам това в локалната memory още."
                if bulgarian
                else "I do not have that in local memory yet."
            ),
            data={"planner": "deterministic"},
        )

    return AgentResponse(
        status="ok",
        message=_format_memory_recall_answer(matches, recall_text),
        data={"planner": "deterministic"},
    )


def _memory_recall_query_text(text: str, conversation_context: str) -> str:
    normalized = _normalize_for_match(text)
    if not conversation_context.strip():
        return text
    if any(
        marker in normalized
        for marker in {
            "на кой адрес",
            "кой адрес",
            "адреса",
            "адресът",
            "ip",
            "url",
            "линк",
            "link",
            "машината",
            "machine",
            "вляза",
            "login",
            "log in",
        }
    ):
        return f"{conversation_context}\nUser: {text}"
    return text


def _load_memory_entries(runtime_context: ToolRuntimeContext) -> list[MemoryEntry]:
    manager = MemoryManager(runtime_context.memory_root)
    manager.bootstrap()
    entries: list[MemoryEntry] = []
    seen: set[str] = set()
    for relative_path in manager.list_files():
        try:
            content = _strip_frontmatter(manager.read(relative_path)).strip()
        except (FileNotFoundError, ValueError):
            continue
        for line in _memory_content_lines(content):
            normalized = _normalize_for_match(line)
            if normalized in seen:
                continue
            seen.add(normalized)
            entries.append(MemoryEntry(path=relative_path, text=line))
    return entries


def _broad_memory_matches(entries: list[MemoryEntry], *, max_entries: int = 14) -> list[MemoryEntry]:
    preferred_paths = [
        "profile.md",
        "facts/personal.md",
        "long-term/facts/computers.md",
        "owner/profile.md",
        "owner/motivation.md",
        "preferences.md",
        "long-term/facts/personal.md",
        "long-term/preferences.md",
        "business/brand.md",
        "business/positioning.md",
        "projects/dmd-agent-4-all/overview.md",
        "projects/dmd-agent-4-all/current-features.md",
    ]
    path_limits = {
        "profile.md": 5,
        "facts/personal.md": 5,
        "long-term/facts/computers.md": 3,
        "owner/profile.md": 4,
        "owner/motivation.md": 2,
        "preferences.md": 4,
        "business/brand.md": 3,
        "business/positioning.md": 2,
    }
    by_path: dict[str, list[MemoryEntry]] = {}
    for entry in entries:
        by_path.setdefault(entry.path, []).append(entry)

    selected: list[MemoryEntry] = []
    seen: set[str] = set()

    def add(entry: MemoryEntry) -> None:
        if len(selected) >= max_entries:
            return
        if _is_broad_memory_noise(entry):
            return
        normalized = _normalize_for_match(entry.text)
        if normalized.startswith("bulgaria / sofia area") and any(
            _normalize_for_match(selected_entry.text).startswith("location: bulgaria / sofia area")
            for selected_entry in selected
        ):
            return
        if normalized in seen:
            return
        seen.add(normalized)
        selected.append(entry)

    for path in preferred_paths:
        per_path = 0
        for entry in by_path.get(path, []):
            before = len(selected)
            add(entry)
            if len(selected) > before:
                per_path += 1
            if per_path >= path_limits.get(path, 2) or len(selected) >= max_entries:
                break

    if len(selected) < max_entries:
        for entry in entries:
            if entry.path == "README.md" or entry.path.startswith(
                ("assistant-behavior/", "github/", "security/", "owner/preferences.md")
            ):
                continue
            add(entry)
            if len(selected) >= max_entries:
                break
    return selected


def _is_broad_memory_noise(entry: MemoryEntry) -> bool:
    normalized_path = _normalize_for_match(entry.path)
    normalized = _normalize_for_match(entry.text)
    if normalized_path == "readme.md":
        return True
    if not normalized:
        return True
    if normalized.startswith("#") or normalized.startswith("##"):
        return True
    if normalized.startswith(("purpose:", "last updated:", "important rules", "notes:", "## ")):
        return True
    if normalized in {"location", "domains", "focus areas", "role", "known system"}:
        return True
    if normalized in {
        "memory index",
        "high priority for retrieval",
        "high priority files",
    }:
        return True
    if normalized.endswith(".md") and "/" in normalized:
        return True
    return False


def _memory_content_lines(content: str) -> list[str]:
    cleaned_lines: list[str] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"^[-*]\s+", "", line).strip()
        if not line or _is_placeholder_memory_line(line):
            continue
        cleaned_lines.append(line)
    lines: list[str] = []
    for index, line in enumerate(cleaned_lines):
        lines.append(line)
        if line.endswith(":") and index + 1 < len(cleaned_lines):
            next_line = cleaned_lines[index + 1]
            if next_line and not next_line.startswith("#"):
                lines.append(f"{line} {next_line}")
    if lines:
        return lines
    compact = re.sub(r"\s+", " ", content).strip()
    if compact and not _is_placeholder_memory_line(compact):
        return [compact]
    return []


def _is_placeholder_memory_line(line: str) -> bool:
    normalized = _normalize_for_match(line)
    if normalized.endswith("not set"):
        return True
    return normalized in {
        "user profile notes live here",
        "user preferences live here",
        "personal facts approved by the user live here",
        "technical facts approved by the user live here",
        "business facts approved by the user live here",
    }


def _rank_memory_entries(text: str, entries: list[MemoryEntry]) -> list[MemoryEntry]:
    terms = _memory_query_terms(text)
    if not terms:
        return []
    scored: list[tuple[int, int, MemoryEntry]] = []
    for index, entry in enumerate(entries):
        haystack = f"{_normalize_for_match(entry.path)} {_normalize_for_match(entry.text)}"
        score = 0
        for term in terms:
            if term in haystack:
                score += 3
            elif len(term) >= 5 and any(token.startswith(term[:5]) for token in _tokenize(haystack)):
                score += 1
        if _asks_about_servers(text) and any(marker in haystack for marker in {"server", "servers", "сърв"}):
            score += 4
            if re.search(r"\b(?:имам|i have|you have)\b", entry.text, flags=re.IGNORECASE):
                score += 8
            if any(marker in haystack for marker in {"model", "модел"}):
                score += 6
        if _asks_about_models(text) and any(marker in haystack for marker in {"model", "модел"}):
            score += 2
        if _is_owned_inventory_memory_entry(entry):
            score += 6
        if _asks_for_port(text) and _memory_entry_contains_port(entry):
            score += 12
        if _asks_about_proxmox(text) and any(marker in haystack for marker in {"proxmox", "проксмокс"}):
            score += 14
        if _asks_for_address_or_url(text) and _memory_entry_contains_address_or_url(entry):
            score += 10
        if _asks_about_machine_access(text) and any(
            marker in haystack for marker in {"web ui", "ui", "https://", "http://", "ssh", "tailscale", "lan ip"}
        ):
            score += 5
        if score > 0:
            scored.append((score, -index, entry))
    scored.sort(reverse=True)
    return [entry for _, __, entry in scored]


def _is_owned_inventory_memory_entry(entry: MemoryEntry) -> bool:
    path = _normalize_for_match(entry.path)
    text = entry.text.strip()
    if path.startswith("long-term/facts/") or path.startswith("facts/"):
        return True
    return bool(re.match(r"^(?:аз\s+)?имам\b|^i\s+have\b|^you\s+have\b", text, flags=re.IGNORECASE))


def _is_relevant_owned_inventory_memory_entry(entry: MemoryEntry, text: str) -> bool:
    if not _is_owned_inventory_memory_entry(entry):
        return False
    haystack = f"{_normalize_for_match(entry.path)} {_normalize_for_match(entry.text)}"
    if _asks_about_servers(text):
        return any(marker in haystack for marker in {"server", "servers", "сърв", "model", "модел"})
    if _asks_about_models(text):
        return any(marker in haystack for marker in {"model", "models", "модел", "server", "servers", "сърв"})
    return True


def _memory_query_terms(text: str) -> set[str]:
    stopwords = {
        "a",
        "an",
        "are",
        "about",
        "do",
        "for",
        "have",
        "i",
        "me",
        "my",
        "it",
        "that",
        "was",
        "where",
        "the",
        "what",
        "which",
        "you",
        "аз",
        "адрес",
        "адреса",
        "адресът",
        "бяха",
        "беше",
        "вече",
        "да",
        "е",
        "за",
        "знам",
        "знаеш",
        "имам",
        "какви",
        "какво",
        "кажи",
        "казах",
        "кой",
        "коя",
        "къде",
        "което",
        "машина",
        "машината",
        "ми",
        "ме",
        "на",
        "моля",
        "напомни",
        "помниш",
        "съм",
        "ти",
    }
    normalized = _normalize_for_match(text)
    all_tokens = set(_tokenize(normalized))
    terms = {token for token in all_tokens if token not in stopwords and len(token) >= 2}
    aliases: set[str] = set()
    if any(marker in normalized for marker in {"ай пи", "айпи", "ай пито"}):
        aliases.update({"ip", "lan", "tailscale", "адрес"})
    for term in all_tokens:
        service_aliases = {
            "имич": "immich",
            "иммич": "immich",
            "хоумлаб": "homelab",
            "хомелаб": "homelab",
            "графана": "grafana",
            "адгард": "adguard",
            "адгуард": "adguard",
            "плекс": "plex",
            "пейпърлес": "paperless",
            "прометеус": "prometheus",
            "тейлскейл": "tailscale",
            "клаудфлеър": "cloudflare",
            "основенпродукт": "primary-product",
        }
        if term in service_aliases:
            aliases.add(service_aliases[term])
        if term in {"proxmox", "проксмокс", "prox"} or term.startswith("проксм"):
            aliases.update({"proxmox", "проксмокс"})
        if term in {"port", "ports"} or term.startswith("порт"):
            aliases.update({"port", "ports", "порт"})
        if term in {"address", "url", "link", "линк"} or term.startswith("адрес"):
            aliases.update({"address", "url", "link", "адрес", "https", "http"})
        if term in {"ip", "айпи", "ип"}:
            aliases.update({"ip", "lan", "tailscale", "адрес"})
        if term in {"machine"} or term.startswith("машин"):
            aliases.update({"machine", "машина", "server", "servers", "сърв"})
        if term.startswith("сърв"):
            aliases.update({"сърв", "server", "servers"})
        if term in {"server", "servers"}:
            aliases.update({"сърв", "server", "servers"})
        if term.startswith("модел"):
            aliases.update({"модел", "model"})
        if term.startswith("комп"):
            aliases.update({"комп", "computer", "computers"})
    return terms | aliases


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zа-я0-9]+", text, flags=re.IGNORECASE)


def _format_memory_recall_answer(matches: list[MemoryEntry], text: str) -> str:
    bulgarian = _looks_bulgarian(text)
    if _is_broad_memory_recall_question(text):
        return _format_broad_memory_answer(matches, bulgarian=bulgarian)
    port_answer = _format_port_answer(matches, text, bulgarian=bulgarian)
    if port_answer and _asks_for_port(text):
        return port_answer
    proxmox_answer = _format_proxmox_access_answer(matches, bulgarian=bulgarian)
    if proxmox_answer and (_asks_about_proxmox(text) or _asks_for_address_or_url(text) or _asks_about_machine_access(text)):
        return proxmox_answer
    facts = [_memory_fact_for_user(entry.text, bulgarian=bulgarian) for entry in matches]
    if len(facts) == 1 and not _is_broad_memory_recall_question(text):
        return facts[0]
    prefix = (
        "В локалната long-term memory знам това:"
        if bulgarian
        else "Here is what I have in local long-term memory:"
    )
    return "\n".join([prefix, *[f"- {fact}" for fact in facts]])


def _format_broad_memory_answer(matches: list[MemoryEntry], *, bulgarian: bool) -> str:
    prefix = (
        "В локалната long-term memory знам това за теб:"
        if bulgarian
        else "Here is what I know about you from local long-term memory:"
    )
    facts: list[str] = []
    seen: set[str] = set()
    for entry in matches:
        fact = _compact_broad_memory_fact(_memory_fact_for_user(entry.text, bulgarian=bulgarian))
        normalized = _normalize_for_match(fact[:180])
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        facts.append(fact)
        if len(facts) >= 14:
            break
    if not facts:
        return (
            "Имам memory файлове, но не намирам полезни лични факти в тях още."
            if bulgarian
            else "I have memory files, but I do not find useful personal facts in them yet."
        )
    return "\n".join([prefix, *[f"- {fact}" for fact in facts]])


def _compact_broad_memory_fact(fact: str, *, max_chars: int = 360) -> str:
    compact = " ".join(fact.split()).strip()
    if len(compact) <= max_chars:
        return compact
    sentences = re.split(r"(?<=[.!?])\s+", compact)
    selected: list[str] = []
    length = 0
    for sentence in sentences:
        if not sentence or sentence.startswith("##"):
            continue
        if length + len(sentence) + 1 > max_chars:
            break
        selected.append(sentence)
        length += len(sentence) + 1
    if selected:
        return " ".join(selected).rstrip(" .") + "."
    return compact[:max_chars].rstrip(" ,.;") + "..."


def _format_port_answer(matches: list[MemoryEntry], text: str, *, bulgarian: bool) -> str | None:
    combined = "\n".join(entry.text for entry in matches)
    port_match = re.search(r"\b(?:port|ports|порт)[^0-9]{0,80}([0-9]{2,5})\b", combined, flags=re.IGNORECASE)
    if port_match is None:
        for number in re.findall(r"\b([0-9]{2,5})\b", combined):
            value = int(number)
            if value < 1900 or value > 2100:
                port_match = re.match(r".*", number)
                break
    if port_match is None:
        return None
    port = port_match.group(1) if port_match.lastindex else port_match.group(0)
    subject = _memory_subject_label(matches, text)
    if bulgarian:
        return f"{subject} е на порт {port}."
    return f"{subject} is on port {port}."


def _memory_subject_label(matches: list[MemoryEntry], text: str) -> str:
    normalized = _normalize_for_match(text)
    labels = {
        "immich": "Immich",
        "имич": "Immich",
        "иммич": "Immich",
        "homelab": "homelab",
        "хоумлаб": "homelab",
        "хомелаб": "homelab",
        "grafana": "Grafana",
        "графана": "Grafana",
        "adguard": "AdGuard",
        "адгард": "AdGuard",
        "адгуард": "AdGuard",
        "plex": "Plex",
        "плекс": "Plex",
        "paperless": "Paperless",
        "пейпърлес": "Paperless",
        "proxmox": "Proxmox",
        "проксмокс": "Proxmox",
    }
    for marker, label in labels.items():
        if marker in normalized:
            return label
    for entry in matches:
        stem = entry.path.rsplit("/", 1)[-1].removesuffix(".md")
        if stem and stem not in {"overview", "services", "commands", "security", "networking"}:
            return stem.replace("-", " ").title()
    return "Това" if _looks_bulgarian(text) else "That"


def _format_proxmox_access_answer(matches: list[MemoryEntry], *, bulgarian: bool) -> str | None:
    combined = "\n".join(entry.text for entry in matches)
    url_match = re.search(r"https?://[^\s,)]+", combined)
    lan_match = re.search(r"\bLAN IP:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", combined, flags=re.IGNORECASE)
    tailscale_match = re.search(r"\bTailscale IP:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", combined, flags=re.IGNORECASE)
    if not url_match and not lan_match and not tailscale_match:
        return None
    if bulgarian:
        lines = []
        if url_match:
            lines.append(f"Proxmox web UI адресът ти е: {url_match.group(0)}")
        if lan_match:
            lines.append(f"LAN IP на машината: {lan_match.group(1)}")
        if tailscale_match:
            lines.append(f"Tailscale IP: {tailscale_match.group(1)}")
        lines.append("Отваря се през браузър; ако не зареди, трябва да си в LAN/VPN мрежата или през Tailscale.")
        return "\n".join(lines)
    lines = []
    if url_match:
        lines.append(f"Your Proxmox web UI address is: {url_match.group(0)}")
    if lan_match:
        lines.append(f"Machine LAN IP: {lan_match.group(1)}")
    if tailscale_match:
        lines.append(f"Tailscale IP: {tailscale_match.group(1)}")
    lines.append("Open it in a browser; if it does not load, connect through LAN/VPN or Tailscale.")
    return "\n".join(lines)


def _memory_fact_for_user(line: str, *, bulgarian: bool) -> str:
    value = line.strip().strip(" .")
    if bulgarian:
        profile_labels = {
            "User name": "Име",
            "Preferred nickname": "Предпочитан прякор",
            "Bulgarian nickname": "Български прякор",
            "Assistant name": "Име на асистента",
            "Preferred response language": "Предпочитан език за отговор",
        }
        for source, label in profile_labels.items():
            profile_match = re.match(rf"{re.escape(source)}:\s*(.+)$", value, flags=re.IGNORECASE)
            if profile_match:
                return f"{label}: {profile_match.group(1).strip().rstrip('.')}"
        value = re.sub(r"^Аз\s+имам\b", "Имаш", value, flags=re.IGNORECASE)
        value = re.sub(r"^Имам\b", "Имаш", value, flags=re.IGNORECASE)
        value = re.sub(r"^I\s+have\b", "Имаш", value, flags=re.IGNORECASE)
        value = re.sub(r"^I\s+like\b", "Харесваш", value, flags=re.IGNORECASE)
        return value.rstrip(".") + "."
    value = re.sub(r"^I\s+have\b", "You have", value, flags=re.IGNORECASE)
    value = re.sub(r"^My\b", "Your", value, flags=re.IGNORECASE)
    return value.rstrip(".") + "."


def _is_memory_recall_question(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if _is_broad_memory_recall_question(text):
        return True
    recall_markers = {
        "what do i have",
        "what have i told you",
        "what do you remember",
        "remind me what",
        "use long term memo",
        "long term memo",
        "long-term memory",
        "какви сървъри имам",
        "какви модели",
        "имаш инфо в long term memo",
        "long term memo",
        "дългосрочната памет",
        "локалната memory",
        "какво съм ти казал",
        "какво помниш",
        "не помня как",
        "на кой адрес",
        "кой адрес",
        "къде беше",
        "адреса на машината",
        "напомни ми какви",
        "напомни ми какво",
    }
    if any(marker in normalized for marker in recall_markers):
        return True
    return bool(re.search(r"\b(?:what|which)\b.+\b(?:do i have|are mine)\b", normalized)) or bool(
        re.search(r"\bкакви\b.+\bимам\b", normalized)
        or re.search(r"\b(?:къде|кой|какъв|как)\b.+\b(?:адрес|ip|url|линк|вляза|машин)", normalized)
    )


def _is_broad_memory_recall_question(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(
        marker in normalized
        for marker in {
            "what do you know about me",
            "what do you remember about me",
            "tell me everything you know about me",
            "какво знаеш за мен",
            "какво помниш за мен",
            "кажи ми всичко което знаеш за мен",
            "всичко което знаеш за мен",
        }
    )


def _asks_about_servers(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in {"server", "servers", "сърв", "машина", "машината", "machine"})


def _asks_about_models(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in {"model", "models", "модел"})


def _asks_about_proxmox(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in {"proxmox", "проксмокс", "proxm"})


def _asks_for_port(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in {"port", "ports", "порт", "порта", "портът"})


def _asks_for_address_or_url(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(
        marker in normalized
        for marker in {"адрес", "address", "url", "линк", "link", "ip", "айпи", "ай пи", "ай пито", "https", "http"}
    )


def _asks_about_machine_access(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in {"вляза", "login", "log in", "достъп", "access", "web ui", "ui"})


def _memory_entry_contains_address_or_url(entry: MemoryEntry) -> bool:
    haystack = f"{entry.path}\n{entry.text}".casefold()
    return bool(re.search(r"https?://|(?:lan|tailscale)\s+ip:|\bip:\s*\d", haystack, flags=re.IGNORECASE))


def _memory_entry_contains_port(entry: MemoryEntry) -> bool:
    haystack = f"{entry.path}\n{entry.text}".casefold()
    return bool(re.search(r"\b(?:port|ports|порт)\b|:\s*[0-9]{2,5}\b|\b[0-9]{2,5}\s+[a-zа-я]", haystack))


def _memory_entry_contains_machine_access_endpoint(entry: MemoryEntry) -> bool:
    haystack = f"{entry.path}\n{entry.text}".casefold()
    if re.search(r"(?:lan|tailscale)\s+ip:|\bip:\s*\d", haystack, flags=re.IGNORECASE):
        return True
    return bool(re.search(r"https?://[^\s)]*(?::\d{2,5}|proxmox|homelab|tailscale|192\.168\.|10\.)", haystack))


def _memory_subject_query_terms(text: str) -> set[str]:
    normalized = _normalize_for_match(text)
    groups = [
        ({"immich", "имич", "иммич"}, "immich"),
        ({"homelab", "хоумлаб", "хомелаб"}, "homelab"),
        ({"grafana", "графана"}, "grafana"),
        ({"adguard", "адгард", "адгуард"}, "adguard"),
        ({"plex", "плекс"}, "plex"),
        ({"paperless", "пейпърлес"}, "paperless"),
        ({"proxmox", "проксмокс"}, "proxmox"),
        ({"prometheus", "прометеус"}, "prometheus"),
    ]
    terms: set[str] = set()
    for markers, canonical in groups:
        if any(marker in normalized for marker in markers):
            terms.add(canonical)
    return terms


def _is_likely_memory_lookup_question(text: str, ranked: list[MemoryEntry]) -> bool:
    if not ranked:
        return False
    normalized = _normalize_for_match(text)
    direct_lookup_markers = {
        "порт",
        "порта",
        "портът",
        "адрес",
        "айпи",
        "ай пи",
        "ай пито",
        "ip",
        "url",
        "линк",
        "port",
        "address",
    }
    if any(marker in normalized for marker in direct_lookup_markers):
        return True
    question_markers = {"кой", "коя", "кое", "къде", "какъв", "каква", "where", "which"}
    entity_markers = {
        "proxmox",
        "проксмокс",
        "homelab",
        "хоумлаб",
        "хомелаб",
        "immich",
        "имич",
        "иммич",
        "grafana",
        "графана",
        "adguard",
        "адгард",
        "адгуард",
        "plex",
        "плекс",
        "paperless",
        "пейпърлес",
        "машина",
        "machine",
        "server",
        "сърв",
    }
    return any(marker in normalized for marker in question_markers) and any(
        marker in normalized for marker in entity_markers
    )


def _wants_new_long_term_memory_file(text: str) -> bool:
    normalized = _normalize_for_match(text)
    wants_new_file = any(
        marker in normalized
        for marker in {"new md", "new .md", "new markdown", "нов md", "нов .md", "нов мд", "нов markdown"}
    )
    wants_long_term = any(marker in normalized for marker in {"long term", "long-term", "дългосрочно"})
    return wants_new_file or wants_long_term


def _clean_long_term_memory_fact(fact: str) -> str:
    cleaned = fact.strip()
    cleaned = re.sub(
        r"^(?:в\s+)?(?:нов\s+)?(?:\.?md|мд|markdown)\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^дългосрочно\s+(?:че\s+)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^(?:long[-\s]?term)\s+(?:that\s+)?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^че\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().strip(" .,!?:;\"'") or fact


def _is_short_term_memory_text(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(
        marker in normalized
        for marker in {
            "short term",
            "short-term",
            "temporary",
            "temporarily",
            "временно",
            "краткосрочно",
            "за 24 часа",
            "за 48 часа",
        }
    )


def _short_term_ttl_hours_from_text(text: str) -> int:
    normalized = _normalize_for_match(text)
    if "48" in normalized:
        return 48
    return 24


def _clean_short_term_memory_fact(fact: str) -> str:
    cleaned = re.sub(
        r"^(?:short[-\s]?term|temporary|temporarily|временно|краткосрочно)(?:\s+че|\s+that)?\s+",
        "",
        fact.strip(),
        flags=re.IGNORECASE,
    )
    return cleaned.strip().strip(" .,!?:;\"'") or fact


def _extract_memory_fact(text: str) -> str | None:
    patterns = [
        r"^(?:and\s+also\s+)?(?:please\s+)?remember(?:\s+that)?\s+(.+)$",
        r"^(?:can|could)\s+you\s+(?:please\s+)?remember(?:\s+that)?\s+(.+)$",
        r"^(?:please\s+)?save(?:\s+that)?\s+(.+)$",
        r"^(?:please\s+)?keep\s+in\s+mind(?:\s+that)?\s+(.+)$",
        r"^(.+?)\s*,?\s+(?:please\s+)?remember(?:\s+that|this)?$",
        r"^(?:и\s+)?(?:също\s+)?запомни(?:\s+че)?\s+(.+)$",
        r"^(?:моля\s+те\s+да\s+)?запомни(?:ш)?(?:\s+че)?\s+(.+)$",
        r"^(?:можеш\s+ли\s+да\s+)?запомниш(?:\s+че)?\s+(.+)$",
        r"^запази(?:\s+че)?\s+(.+)$",
        r"^(.+?)\s*,?\s+запомни(?:\s+това)?$",
    ]
    for pattern in patterns:
        match = re.match(pattern, text.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        fact = match.group(1).strip().strip(" .,!?:;\"'")
        if fact.lower() in {"this", "that", "това"}:
            return None
        if 1 <= len(fact) <= 500 and "\n" not in fact:
            return fact
    return None


def _answer_without_llm(text: str, config: dict[str, Any]) -> str | None:
    normalized = _normalize_for_match(text)
    profile = _profile_from_config(config)
    agent_name = profile["agent_name"]
    user_name = profile["user_name"]
    nickname = profile["nickname_bg"] if _looks_bulgarian(text) else profile["nickname"]
    if _is_open_terminal_window_request(normalized):
        if _looks_bulgarian(text):
            return (
                "Не мога да отворя нов GUI прозорец на Terminal от dashboard-а. "
                "Мога да изпълня allowlisted команда през Terminal tool, например: ls, pwd, git status."
            )
        return (
            "I cannot open a new GUI Terminal window from the dashboard. "
            "I can run allowlisted commands through the Terminal tool, for example: ls, pwd, git status."
        )
    if _is_unimplemented_browser_request(normalized):
        if _looks_bulgarian(text):
            return "Browser sandbox още не е имплементиран, затова не мога реално да отварям Google или уеб страници оттук."
        return "Browser sandbox is not implemented yet, so I cannot actually open Google or browse pages from here yet."
    if normalized in {"start status", "status"}:
        return "Use /status inside chat to show local agent status."
    if normalized in {"kak si", "kak si brat"}:
        return (
            f"Dobre sum. Az sum {agent_name}, rabotya kato agenten runtime, "
            "a security/action chasta minava prez permission engine."
        )
    if normalized in {"какво можеш да правиш", "какво можеш", "какво можеш ти"}:
        return (
            f"Аз съм {agent_name}. Мога да говоря с теб през LLM, да чета локалната memory с одобрение "
            "при cloud модел, да записвам спомени след approval, да работя с Telegram във фон и да минавам "
            "рисковите действия през permission engine. Browser и Calendar още не са реално имплементирани."
        )
    if _looks_bulgarian(text) and "какво си ти" in normalized:
        return f"Аз съм {agent_name}. Работя локално и пазя действията зад permission engine."
    if normalized in {"кой модел си ти", "какъв модел си", "кой модел използваш"}:
        llm = config.get("llm", {})
        return f"В момента съм настроен да използвам {llm.get('provider', '-')} модел: {llm.get('model', '-')}."
    if normalized in {
        "who are you",
        "what is your name",
        "what's your name",
        "your name?",
        "кой си",
        "коя си",
        "как се казваш",
        "името ти",
    }:
        if _looks_bulgarian(text):
            return f"Аз съм {agent_name}. Работя локално и пазя действията зад permission engine."
        return f"I am {agent_name}. I run locally and keep actions behind the permission engine."
    if normalized in {
        "who am i",
        "what is my name",
        "what's my name",
        "my name?",
        "кой съм аз",
        "коя съм аз",
        "аз кой съм",
        "аз коя съм",
        "как се казвам",
        "името ми",
    }:
        if user_name:
            if _looks_bulgarian(text):
                return f"{nickname}, ти си {user_name}." if nickname else f"Ти си {user_name}."
            return f"{nickname}, you are {user_name}." if nickname else f"You are {user_name}."
        if _looks_bulgarian(text):
            return "Още не знам името ти. Напиши: казвам се <име>."
        return "I do not know your name yet. Write: my name is <name>."
    if normalized in {"как си", "здравей", "здрасти"}:
        return (
            f"Добре съм. Аз съм {agent_name}, работя локално, пазя действията зад permission engine, "
            "и мога да помагам с memory, tools и бъдещи connectors."
        )
    if normalized in {"hello", "hi", "hey", "how are you"}:
        return (
            f"I am {agent_name}, running locally and ready. I can help with memory, tools, "
            "and connector-driven tasks once you enable them."
        )
    if normalized in {"help", "/help"} or "what can you do" in normalized:
        return (
            "I can chat through a local model, list and read local Markdown memory, "
            "show enabled tools, and route tool requests through the permission engine. "
            "Gmail, Calendar, Browser, and Terminal tools exist as manifests but are "
            "disabled until explicitly configured."
        )
    return None


def _answer_llm_unavailable(text: str, config: dict[str, Any], exc: Exception) -> AgentResponse | None:
    if not _is_missing_llm_api_key_error(exc) and not _is_local_llm_connection_error(exc):
        return None
    normalized = _normalize_for_match(text)
    bulgarian = _looks_bulgarian(text)
    llm = config.get("llm", {})
    provider = str(llm.get("provider") or "LLM")
    if _is_acknowledgement_text(normalized):
        return AgentResponse(
            status="ok",
            message=(
                "Ясно, маняк. Кажи какво искаш да направим."
                if bulgarian
                else "Got it. Tell me what you want to do next."
            ),
            data={"planner": "deterministic", "fallback": "llm_unavailable"},
        )
    if _is_missing_llm_api_key_error(exc):
        env_name = _extract_missing_api_key_env(str(exc))
        if bulgarian:
            message = (
                f"{provider} моделът е избран, но API key не е зареден в текущия процес"
                f"{f' ({env_name})' if env_name else ''}. "
                "Локалната memory/tools част още работи; добави ключа от provider таба или превключи към локален модел."
            )
        else:
            message = (
                f"The {provider} model is selected, but its API key is not loaded in this running process"
                f"{f' ({env_name})' if env_name else ''}. "
                "Local memory/tools still work; add the key in the provider tab or switch to a local model."
            )
        return AgentResponse(
            status="ok",
            message=message,
            data={"planner": "deterministic", "fallback": "missing_api_key"},
        )
    return AgentResponse(
        status="ok",
        message=(
            f"Не мога да се свържа с {provider} модела в момента. Локалните memory/tools fallback-и още работят."
            if bulgarian
            else f"I cannot connect to the {provider} model right now. Local memory/tools fallbacks still work."
        ),
        data={"planner": "deterministic", "fallback": "llm_unavailable"},
    )


def _is_acknowledgement_text(normalized: str) -> bool:
    return normalized in {
        "i know",
        "i know that",
        "i know this",
        "ok",
        "okay",
        "got it",
        "gotcha",
        "yes",
        "yep",
        "yeah",
        "thanks",
        "thank you",
        "sure",
        "cool",
        "fine",
        "разбрах",
        "ясно",
        "ок",
        "окей",
        "да",
        "добре",
        "мерси",
        "благодаря",
    }


def _is_missing_llm_api_key_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    return "requires an api key" in text or "api key" in text and "requires" in text


def _is_local_llm_connection_error(exc: Exception) -> bool:
    text = str(exc).casefold()
    return "cannot connect" in text or "connection refused" in text


def _extract_missing_api_key_env(message: str) -> str:
    match = re.search(r"\b([A-Z][A-Z0-9_]*API_KEY[A-Z0-9_]*)\b", message)
    return match.group(1) if match else ""


def _is_fast_control_answer(text: str) -> bool:
    normalized = _normalize_for_match(text)
    exact = {
        "help",
        "/help",
        "hello",
        "hi",
        "hey",
        "how are you",
        "как си",
        "здравей",
        "здрасти",
        "who are you",
        "what is your name",
        "what's your name",
        "your name?",
        "кой си",
        "коя си",
        "как се казваш",
        "името ти",
        "who am i",
        "what is my name",
        "what's my name",
        "my name?",
        "кой съм аз",
        "коя съм аз",
        "аз кой съм",
        "аз коя съм",
        "как се казвам",
        "името ми",
        "кой модел си ти",
        "какъв модел си",
        "кой модел използваш",
        "какво можеш да правиш",
        "какво можеш",
        "какво можеш ти",
    }
    return (
        normalized in exact
        or "what can you do" in normalized
        or _is_open_terminal_window_request(normalized)
        or (_looks_bulgarian(text) and "какво си ти" in normalized)
    )


def _is_open_terminal_window_request(normalized: str) -> bool:
    if "terminal" not in normalized and "терминал" not in normalized:
        return False
    if "open" not in normalized and "отвори" not in normalized:
        return False
    runnable_markers = {" ls", " pwd", "git status", "git diff", "cat "}
    return not any(marker in f" {normalized}" for marker in runnable_markers)


def _handle_identity_update(text: str, runtime_context: ToolRuntimeContext) -> AgentResponse | None:
    assistant_name = _extract_name(
        text,
        [
            r"^call yourself\s+(.+)$",
            r"^your name is\s+(.+)$",
            r"^you are now\s+(.+)$",
            r"^rename yourself to\s+(.+)$",
            r"^казвай се\s+(.+)$",
            r"^ще се казваш\s+(.+)$",
            r"^смени си името на\s+(.+)$",
            r"^името ти е\s+(.+)$",
            r"^наричай се\s+(.+)$",
        ],
    )
    if assistant_name:
        _update_profile(runtime_context, agent_name=assistant_name)
        if _looks_bulgarian(text):
            return AgentResponse(
                status="ok",
                message=f"Готово. Аз вече се казвам {assistant_name}.",
                data={"planner": "deterministic"},
            )
        return AgentResponse(
            status="ok",
            message=f"Done. My name is now {assistant_name}.",
            data={"planner": "deterministic"},
        )

    nickname_update = _extract_nickname_update(text)
    if nickname_update is not None:
        nickname, nickname_bg = nickname_update
        _update_profile(runtime_context, nickname=nickname, nickname_bg=nickname_bg)
        if _looks_bulgarian(text):
            display = nickname_bg or nickname
            return AgentResponse(
                status="ok",
                message=f"Готово. Ще ти казвам {display}.",
                data={"planner": "deterministic"},
            )
        return AgentResponse(
            status="ok",
            message=f"Done. I will call you {nickname}.",
            data={"planner": "deterministic"},
        )

    user_name = _extract_name(
        text,
        [
            r"^my name is\s+(.+)$",
            r"^i['’]?m\s+(.+)$",
            r"^call me\s+(.+)$",
            r"^remember my name is\s+(.+)$",
            r"^казвам се\s+(.+)$",
            r"^викай ми\s+(.+)$",
            r"^името ми е\s+(.+)$",
            r"^запомни че се казвам\s+(.+)$",
        ],
    )
    if user_name:
        _update_profile(runtime_context, user_name=user_name)
        if _looks_bulgarian(text):
            return AgentResponse(
                status="ok",
                message=f"Готово, {user_name}. Запомних името ти.",
                data={"planner": "deterministic"},
            )
        return AgentResponse(
            status="ok",
            message=f"Done, {user_name}. I remembered your name.",
            data={"planner": "deterministic"},
        )

    return None


def _extract_nickname_update(text: str) -> tuple[str, str] | None:
    match = re.match(
        r"^i\s+want\s+you\s+to\s+call\s+me\s+(.+?)\s+or\s+if\s+i\s+type\s+in\s+bulgarian\s+"
        r"you\s+can\s+also\s+call\s+me\s+(.+)$",
        text.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    nickname = match.group(1).strip().strip(" .,!?:;\"'")
    nickname_bg = match.group(2).strip().strip(" .,!?:;\"'")
    if not nickname or len(nickname) > 80 or "\n" in nickname:
        return None
    if not nickname_bg or len(nickname_bg) > 80 or "\n" in nickname_bg:
        return None
    return nickname, nickname_bg


def _extract_name(text: str, patterns: list[str]) -> str | None:
    for pattern in patterns:
        match = re.match(pattern, text.strip(), flags=re.IGNORECASE)
        if not match:
            continue
        name = match.group(1).strip().strip(" .,!?:;\"'")
        if 1 <= len(name) <= 80 and "\n" not in name:
            return name
    return None


def _profile_from_config(config: dict[str, Any]) -> dict[str, str]:
    setup = config.get("setup", {})
    return {
        "agent_name": str(setup.get("agent_name") or "DMD Agent"),
        "user_name": str(setup.get("user_name") or ""),
        "preferred_language": str(setup.get("preferred_language") or "auto"),
        "nickname": str(setup.get("nickname") or ""),
        "nickname_bg": str(setup.get("nickname_bg") or setup.get("nickname") or ""),
    }


def _enabled_tools_from_config(
    manifests: dict[str, Any],
    config: dict[str, Any],
) -> frozenset[str]:
    overrides = config.get("tools", {})
    enabled: set[str] = set()
    for name, manifest in manifests.items():
        override = overrides.get(name, {})
        if bool(override.get("enabled", getattr(manifest, "default_enabled", False))):
            enabled.add(name)
    return frozenset(enabled)


def _request_with_original_message(request: ToolRequest, user_message: str) -> ToolRequest:
    return ToolRequest(
        tool=request.tool,
        args=request.args,
        reason=f"User message: {user_message}\nPlanner reason: {request.reason}".strip(),
    )


def _original_message_from_reason(reason: str) -> str:
    prefix = "User message: "
    for line in reason.splitlines():
        if line.startswith(prefix):
            return line.removeprefix(prefix).strip()
    return ""


def _update_profile(
    runtime_context: ToolRuntimeContext,
    *,
    agent_name: str | None = None,
    user_name: str | None = None,
    nickname: str | None = None,
    nickname_bg: str | None = None,
) -> None:
    config = runtime_context.config
    setup = config.setdefault("setup", {})
    if agent_name is not None:
        setup["agent_name"] = agent_name
    if user_name is not None:
        setup["user_name"] = user_name
    if nickname is not None:
        setup["nickname"] = nickname
    if nickname_bg is not None:
        setup["nickname_bg"] = nickname_bg
    setup["completed"] = True

    if runtime_context.config_path is not None:
        save_config(config, runtime_context.config_path)

    profile = _profile_from_config(config)
    MemoryManager(runtime_context.memory_root).write(
        "long-term/profile.md",
        "\n".join(
            [
                f"User name: {profile['user_name'] or 'not set'}",
                f"Preferred nickname: {profile['nickname'] or 'not set'}",
                f"Bulgarian nickname: {profile['nickname_bg'] or 'not set'}",
                f"Assistant name: {profile['agent_name']}",
                f"Preferred response language: {profile['preferred_language']}",
            ]
        ),
        metadata={
            "type": "profile",
            "memory_scope": "long-term",
            "source": "chat_identity",
            "confidence": "high",
        },
    )


def _load_memory_context(
    runtime_context: ToolRuntimeContext,
    *,
    max_chars: int = 12000,
    allow_cloud_context: bool = False,
    query: str = "",
    conversation_context: str = "",
) -> str:
    if (
        runtime_context.config.get("llm", {}).get("provider") not in {"ollama", "local"}
        and not allow_cloud_context
    ):
        return ""
    if query.strip():
        return _load_relevant_memory_context(
            runtime_context,
            query=query,
            conversation_context=conversation_context,
            max_chars=max_chars,
        )
    manager = MemoryManager(runtime_context.memory_root)
    manager.bootstrap()
    chunks: list[str] = []
    remaining = max_chars
    for relative_path in manager.list_files():
        if remaining <= 0:
            break
        try:
            content = _strip_frontmatter(manager.read(relative_path)).strip()
        except (FileNotFoundError, ValueError):
            continue
        if not content:
            continue
        chunk = f"[{relative_path}]\n{content}"
        if len(chunk) > remaining:
            chunk = chunk[:remaining]
        chunks.append(chunk)
        remaining -= len(chunk)
    return "\n\n".join(chunks)


def _load_relevant_memory_context(
    runtime_context: ToolRuntimeContext,
    *,
    query: str,
    conversation_context: str = "",
    max_chars: int = 12000,
    max_entries: int = 14,
) -> str:
    query_text = _memory_recall_query_text(query, conversation_context)
    ranked = _rank_memory_entries(query_text, _load_memory_entries(runtime_context))
    if not ranked:
        return ""
    remaining = max_chars
    chunks: list[str] = []
    used: set[tuple[str, str]] = set()
    for entry in ranked[:max_entries]:
        key = (entry.path, entry.text)
        if key in used:
            continue
        used.add(key)
        chunk = f"[{entry.path}]\n{entry.text}"
        if len(chunk) > remaining:
            chunk = chunk[:remaining].rstrip()
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk) + 2
        if remaining <= 0:
            break
    return "\n\n".join(chunks)


def _strip_frontmatter(markdown: str) -> str:
    if not markdown.startswith("---\n"):
        return markdown
    end = markdown.find("\n---\n", 4)
    if end == -1:
        return markdown
    return markdown[end + len("\n---\n") :]


def _approval_message(request: ToolRequest, default: str) -> str:
    if request.tool == "memory.write":
        return "I can save that to memory after you approve it."
    if request.tool == "memory.organize_long_term":
        return "I can split that long-term memory into modular Markdown files after you approve it."
    if request.tool == "reminders.create":
        return "I can create that local reminder after you approve it."
    return default


def _denial_message(request: ToolRequest, default: str) -> str:
    if default.startswith("Tool is disabled:"):
        return (
            f"{default}. Enable it from the dashboard Tools/Terminal tabs or from terminal chat "
            f"with /tool enable {request.tool}. If the tool needs a permission, open Permissions."
        )
    return default


def _tool_success_message(request: ToolRequest) -> str:
    if request.tool == "memory.write":
        return "Saved to memory."
    if request.tool == "memory.organize_long_term":
        return "Long-term memory organized into modular Markdown files."
    if request.tool == "browser.scrape_markdown":
        return "Scraped page saved as Markdown."
    if request.tool == "reminders.create":
        return "Reminder saved. I will notify you in Telegram when it is due if Telegram is configured."
    if request.tool == "reminders.complete":
        return "Reminder completed."
    if request.tool == "calendar.create_event":
        return "Saved to local calendar store. Active reminder notifications are not implemented yet."
    return "Tool executed."


def _history_text_from_response(response: AgentResponse) -> str:
    text = response.message.strip()
    data = response.data or {}
    if response.status == "approval_required":
        tool = data.get("tool")
        approval_id = data.get("approval_id")
        suffix = f" approval_id={approval_id}" if approval_id is not None else ""
        return f"{text} [approval_required tool={tool}{suffix}]".strip()
    if response.status == "ok" and isinstance(data, dict):
        stdout = str(data.get("stdout") or "").strip()
        if stdout:
            return f"{text}\n{stdout[:1200]}"
        command = data.get("command")
        if command:
            return f"{text} command={command}"
    return text or response.status


def _is_clarification_followup(text: str) -> bool:
    normalized = _normalize_for_match(text)
    if len(normalized) > 80:
        return False
    return any(
        marker in normalized
        for marker in {
            "не разбрах",
            "не разбах",
            "не схванах",
            "не разбрах това",
            "какво имаш предвид",
            "обясни по просто",
            "обясни по-просто",
            "i don't understand",
            "i didnt understand",
            "i didn't get it",
            "explain simpler",
        }
    )


def _last_assistant_message(turns: list[Any]) -> str | None:
    for turn in reversed(turns):
        if getattr(turn, "role", "") != "assistant":
            continue
        content = " ".join(str(getattr(turn, "content", "")).split()).strip()
        if not content:
            continue
        if len(content) > 700:
            content = content[:700].rstrip() + "..."
        return content
    return None


def _is_unimplemented_browser_request(normalized: str) -> bool:
    browser_words = {
        "google",
        "browser",
        "web page",
        "website",
        "url",
        "гугъл",
        "браузър",
        "сайт",
        "страница",
    }
    action_words = {
        "open",
        "browse",
        "search",
        "visit",
        "отвори",
        "отваряш",
        "потърси",
        "търси",
    }
    return any(word in normalized for word in browser_words) and any(
        word in normalized for word in action_words
    )


def _looks_bulgarian(text: str) -> bool:
    return any("а" <= char.lower() <= "я" for char in text)


def _normalize_for_match(text: str) -> str:
    return text.lower().strip().strip(" .,!?:;")
