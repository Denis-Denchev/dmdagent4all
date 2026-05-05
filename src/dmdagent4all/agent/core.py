from __future__ import annotations

import json
import calendar
import re
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

from dmdagent4all.audit import AuditEvent, AuditStore
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
        self._cloud_context_approved = permission_context.cloud_context_approved

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
        approval_action = self._handle_approval_action_from_text(stripped)
        if approval_action is not None:
            return approval_action
        identity_update = _handle_identity_update(stripped, self.runtime_context)
        if identity_update is not None:
            return identity_update
        llm_reminder_request = self._plan_reminder_request(stripped)
        if llm_reminder_request is not None:
            return self.handle_tool_request(llm_reminder_request)
        reminder_request = _reminder_request_from_text(stripped)
        if reminder_request is not None:
            return self.handle_tool_request(reminder_request)
        memory_update = _memory_write_request_from_text(stripped, self.runtime_context)
        if memory_update is not None:
            return self.handle_tool_request(memory_update)
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
        if fast_answer is not None:
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
            return AgentResponse(
                status="ok",
                message=(
                    "Agent core is initialized without a planner. Use /tools or send a "
                    "structured tool request JSON."
                ),
            )

        try:
            llm_config = self.runtime_context.config.get("llm", {})
            memory_context = _load_memory_context(self.runtime_context)
            now = datetime.now().astimezone()
            plan = self.planner.plan(
                user_message=text,
                manifests=self.tool_registry.manifests,
                profile=_profile_from_config(self.runtime_context.config),
                memory_context=memory_context,
                enabled_tools=_enabled_tools_from_config(
                    self.tool_registry.manifests,
                    self.runtime_context.config,
                ),
                response_language=llm_config.get("response_language", "auto"),
                current_time=now.isoformat(timespec="seconds"),
                timezone_name=now.tzname() or "",
                max_tokens=int(llm_config.get("planner_max_tokens", 192)),
                temperature=float(llm_config.get("planner_temperature", 0.0)),
                think=bool(llm_config.get("planner_think", False)),
            )
        except (PlannerError, OSError, RuntimeError) as exc:
            return AgentResponse(
                status="error",
                message=f"LLM planner failed: {exc}",
        )

        if plan.tool_request is not None:
            request = _request_with_original_message(plan.tool_request, stripped)
            tool_response = self.handle_tool_request(request)
            if self._should_synthesize_tool_response(stripped, request, tool_response):
                return self._synthesize_tool_response(stripped, tool_response)
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

    def _plan_reminder_request(self, text: str) -> ToolRequest | None:
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
                memory_context=_load_memory_context(self.runtime_context),
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


def _extract_urlish_target(text: str) -> str | None:
    lower = text.lower()
    if "google" in lower or "гугъл" in lower:
        return "https://www.google.com"
    patterns = [
        r"https?://[^\s]+",
        r"\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(0).strip(" .,!?:;\"'")
    return None


def _terminal_request_from_text(text: str) -> ToolRequest | None:
    normalized = _normalize_for_match(text)
    mentions_terminal = "terminal" in normalized
    mentions_readme_cat = re.search(r"\bcat\s+(readme\.md|README\.md)\b", text, flags=re.IGNORECASE)
    if not mentions_terminal and mentions_readme_cat is None:
        return None
    command: list[str] | None = None
    if re.search(r"\b(?:type|run|execute)\s+ls\b", text, flags=re.IGNORECASE):
        command = ["ls"]
    elif re.search(r"\b(?:type|run|execute)\s+pwd\b", text, flags=re.IGNORECASE):
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
) -> str:
    if (
        runtime_context.config.get("llm", {}).get("provider") not in {"ollama", "local"}
        and not allow_cloud_context
    ):
        return ""
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
    if request.tool == "reminders.create":
        return "Reminder saved. I will notify you in Telegram when it is due if Telegram is configured."
    if request.tool == "reminders.complete":
        return "Reminder completed."
    if request.tool == "calendar.create_event":
        return "Saved to local calendar store. Active reminder notifications are not implemented yet."
    return "Tool executed."


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
