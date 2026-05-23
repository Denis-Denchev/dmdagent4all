from __future__ import annotations

import json
import calendar
import os
import re
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.agent.context import AgentContextProvider
from dmdagent4all.agent.history import ChatHistory
from dmdagent4all.agent.planner import LLMPlanner, MemoryWriteIntent, PlannerError, is_low_relevance_plan
from dmdagent4all.agent.router import ConversationRouter
from dmdagent4all.autonomy import local_dev_autonomy_enabled, log_local_dev_autonomy
from dmdagent4all.config import save_config
from dmdagent4all.memory import MemoryManager
from dmdagent4all.permissions import PermissionContext, PermissionEngine, ToolRequest
from dmdagent4all.sandbox import TerminalPolicy, emergency_stop_terminal_processes
from dmdagent4all.security import redact_text
from dmdagent4all.security.policy import ToolSafetyPolicy
from dmdagent4all.tools.base import ToolRuntimeContext
from dmdagent4all.tools.registry import ToolExecutionError, ToolRegistry
from dmdagent4all.tools.storage import downloads_root_from_config
from dmdagent4all.workspace import WorkspaceError, WorkspaceManager


LOCAL_DEV_AUTONOMY_ENABLED_TOOLS = frozenset(
    {
        "browser.extract_text",
        "browser.open",
        "browser.scrape_markdown",
        "developer.context",
        "files.list",
        "files.mkdir",
        "files.read",
        "files.write",
        "files.write_many",
        "memory.list",
        "memory.read",
        "project.scaffold_one_page_app",
        "system.list_enabled_tools",
        "terminal.run",
        "workspace.status",
    }
)

LOCAL_DEV_AUTONOMY_PERMISSIONS = frozenset({"browser.read", "terminal.run"})

LOCAL_DEV_AUTO_APPROVED_TOOLS = frozenset(
    {
        "browser.extract_text",
        "browser.open",
        "browser.scrape_markdown",
        "developer.context",
        "files.list",
        "files.mkdir",
        "files.read",
        "files.write",
        "files.write_many",
        "memory.list",
        "memory.read",
        "project.scaffold_one_page_app",
        "system.list_enabled_tools",
        "workspace.status",
    }
)

CONTEXT_GATHERING_SYNTHESIS_TOOLS = frozenset(
    {
        "developer.context",
        "files.list",
        "files.read",
        "memory.list",
        "memory.read",
        "system.list_enabled_tools",
        "workspace.status",
    }
)


@dataclass(frozen=True)
class AgentResponse:
    status: str
    message: str
    data: dict[str, Any] | None = None


@dataclass(frozen=True)
class MemoryEntry:
    path: str
    text: str


@dataclass(frozen=True)
class UnsupportedAction:
    kind: str
    target: str = ""


@dataclass(frozen=True)
class EmailSendIntent:
    provider: str
    to: str
    subject: str
    body: str


@dataclass(frozen=True)
class PendingContinuation:
    original_message: str
    steps: tuple[ToolRequest | UnsupportedAction, ...]
    chain_state: dict[str, Any] = field(default_factory=dict)


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
        event_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        self.permission_engine = permission_engine
        self.tool_registry = tool_registry
        self.permission_context = permission_context
        self.runtime_context = runtime_context
        self.audit_store = audit_store
        self.planner = planner
        self._cloud_context_approved = permission_context.cloud_context_approved
        self.chat_history = chat_history or ChatHistory(runtime_context.workspace_root / "chat_history.json")
        self.router = ConversationRouter()
        self.safety_policy = ToolSafetyPolicy()
        self._session_corrections: dict[str, list[str]] = {}
        self._pending_continuations: dict[int, PendingContinuation] = {}
        self._last_denial_reason = ""
        self._last_tool_result_summary: dict[str, Any] = {}
        self._event_sink = event_sink
        self._trace_events: list[dict[str, Any]] = []

    def _trace_event(
        self,
        kind: str,
        title: str,
        *,
        status: str = "running",
        detail: str = "",
        tool: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event: dict[str, Any] = {
            "at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "kind": kind,
            "title": title,
            "status": status,
        }
        if detail:
            event["detail"] = redact_text(detail[:1000])
        if tool:
            event["tool"] = tool
        if metadata:
            event["metadata"] = _safe_tool_result_summary(metadata)
        self._trace_events.append(event)
        if self._event_sink is not None:
            try:
                self._event_sink(dict(event))
            except Exception:
                pass

    def _attach_trace(self, response: AgentResponse) -> AgentResponse:
        if not local_dev_autonomy_enabled(self.runtime_context.config):
            return response
        data = dict(response.data or {})
        data.setdefault(
            "runtime_mode",
            "full_llm_first_autonomy"
            if local_dev_autonomy_enabled(self.runtime_context.config)
            else "standard_safe",
        )
        if self._trace_events:
            data["trace"] = list(self._trace_events)
        return replace(response, data=data)

    def _agent_context_snapshot(self) -> dict[str, Any]:
        return AgentContextProvider(
            runtime_context=self.runtime_context,
            manifests=self.tool_registry.manifests,
            permission_context=self._permission_context(),
            audit_store=self.audit_store,
        ).snapshot(
            planner_active=self.planner is not None,
            last_denial_reason=self._last_denial_reason,
            last_tool_result_summary=self._last_tool_result_summary,
        )

    def _answer_agent_context_question(self, text: str) -> AgentResponse | None:
        if _asks_memory_file_name_list(text):
            files = list(self._agent_context_snapshot().get("memory", {}).get("files") or [])
            message = "\n".join(str(path) for path in files) if files else (
                "Няма Markdown файлове в memory."
                if _looks_bulgarian(text)
                else "There are no Markdown files in memory."
            )
            return AgentResponse(
                status="ok",
                message=message,
                data={
                    "planner": "deterministic",
                    "source": "agent_context",
                    "kind": "memory_files",
                    "files": files,
                },
            )
        if _asks_scraped_downloads_list(text):
            files = _scraped_download_files(self.runtime_context)
            message = "\n".join(item["path"] for item in files) if files else (
                "Няма scrape Markdown файлове в downloads/scrapefiles."
                if _looks_bulgarian(text)
                else "There are no scraped Markdown files in downloads/scrapefiles."
            )
            return AgentResponse(
                status="ok",
                message=message,
                data={
                    "planner": "deterministic",
                    "source": "agent_context",
                    "kind": "scraped_downloads",
                    "files": files,
                },
            )
        if _asks_missing_tool_permissions(text):
            missing = _missing_tool_permissions(
                self.tool_registry.manifests,
                self._permission_context(),
            )
            if not missing:
                message = (
                    "Няма tools с липсващи required permissions."
                    if _looks_bulgarian(text)
                    else "No tools have missing required permissions."
                )
            else:
                intro = (
                    "Tools с липсващи permissions:"
                    if _looks_bulgarian(text)
                    else "Tools with missing permissions:"
                )
                lines = [intro]
                lines.extend(
                    f"- {tool}: {', '.join(permissions)}"
                    for tool, permissions in missing.items()
                )
                message = "\n".join(lines)
            return AgentResponse(
                status="ok",
                message=message,
                data={
                    "planner": "deterministic",
                    "source": "agent_context",
                    "kind": "missing_tool_permissions",
                    "missing_permissions_by_tool": missing,
                },
            )
        return None

    def handle_text(self, text: str, *, session_id: str = "default") -> AgentResponse:
        stripped = text.strip()
        self._trace_event(
            "runtime",
            "Reading runtime state",
            status="ok",
            metadata={
                "mode": "full_llm_first_autonomy"
                if local_dev_autonomy_enabled(self.runtime_context.config)
                else "standard_safe",
                "workspace": str(self.runtime_context.workspace_root),
                "planner_active": self.planner is not None,
            },
        )
        response = self._handle_text(stripped, text, session_id=session_id)
        response = self._attach_trace(response)
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
        if stripped == "/help":
            return AgentResponse(
                status="ok",
                message=_answer_without_llm(stripped, self.runtime_context.config)
                or "Use /tools to list tools and /status to show the current workspace.",
                data={"planner": "deterministic", "control": "help"},
            )
        if stripped == "/status":
            request = ToolRequest(
                tool="workspace.status",
                args={},
                reason="User requested current local agent status.",
            )
            response = self.handle_tool_request(request)
            return self._shape_tool_response(stripped, request, response)
        emergency_response = self._handle_emergency_text(stripped)
        if emergency_response is not None:
            return emergency_response
        approval_action = self._handle_approval_action_from_text(stripped)
        if approval_action is not None:
            return approval_action
        chat_history_answer = self._answer_chat_history_question(stripped, session_id=session_id)
        if chat_history_answer is not None:
            return chat_history_answer
        feedback_answer = self._handle_session_feedback(stripped, session_id=session_id)
        if feedback_answer is not None:
            return feedback_answer
        context_answer = self._answer_agent_context_question(stripped)
        if context_answer is not None:
            return context_answer
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
                request = ToolRequest(
                    tool=str(payload["tool"]),
                    args=dict(payload.get("args", {})),
                    reason=str(payload.get("reason", "")),
                )
                response = self.handle_tool_request(request)
                return self._shape_tool_response(stripped, request, response)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                return AgentResponse(
                    status="error",
                    message=f"Invalid tool request JSON: {exc}",
                )

        email_action = _email_action_from_text(stripped, self.runtime_context.config)
        if isinstance(email_action, EmailSendIntent):
            if not self._should_defer_email_send_to_planner(stripped):
                return self._handle_email_send_intent(stripped, email_action)
        if isinstance(email_action, ToolRequest):
            return self._run_user_tool_request(
                stripped,
                email_action,
                conversation_context=conversation_context,
            )
        if isinstance(email_action, AgentResponse) and email_action.status == "not_configured":
            return email_action

        if self.planner is not None:
            return self._handle_with_planner(
                stripped,
                text,
                session_id=session_id,
                conversation_context=self._planner_conversation_context(session_id),
            )

        return self._handle_without_llm(
            stripped,
            text,
            session_id=session_id,
            conversation_context=conversation_context,
        )

    def _handle_without_llm(
        self,
        stripped: str,
        text: str,
        *,
        session_id: str,
        conversation_context: str,
        planner_error: Exception | None = None,
    ) -> AgentResponse:
        multi_step_answer = self._handle_multi_step_request(
            stripped,
            conversation_context=conversation_context,
        )
        if multi_step_answer is not None:
            return multi_step_answer
        interactive_answer = _interactive_terminal_response(stripped)
        if interactive_answer is not None:
            return interactive_answer

        routed = self.router.route(stripped)
        if routed is not None:
            if routed.kind == "normal_chat":
                return self._handle_normal_chat(
                    stripped,
                    conversation_context=conversation_context,
                )
            if routed.request is not None:
                return self._run_user_tool_request(
                    stripped,
                    routed.request,
                    conversation_context=conversation_context,
                )

        email_action = _email_action_from_text(stripped, self.runtime_context.config)
        if isinstance(email_action, AgentResponse):
            return email_action
        if isinstance(email_action, EmailSendIntent):
            return self._handle_email_send_intent(stripped, email_action)
        if isinstance(email_action, ToolRequest):
            return self._run_user_tool_request(
                stripped,
                email_action,
                conversation_context=conversation_context,
            )

        if self.planner is None or planner_error is not None:
            identity_update = _handle_identity_update(stripped, self.runtime_context)
            if identity_update is not None:
                return identity_update
        local_file_write = _local_dev_file_write_request_from_text(stripped, self.runtime_context.config)
        if local_file_write is not None:
            return self._run_user_tool_request(
                stripped,
                local_file_write,
                conversation_context=conversation_context,
            )
        memory_organize = _memory_organize_request_from_text(stripped)
        if memory_organize is not None:
            return self._run_user_tool_request(stripped, memory_organize, conversation_context=conversation_context)
        memory_organize_missing_source = _memory_organize_missing_source_response(stripped)
        if memory_organize_missing_source is not None:
            return memory_organize_missing_source
        llm_reminder_request = self._plan_reminder_request(stripped, session_id=session_id)
        if llm_reminder_request is not None:
            return self._run_user_tool_request(stripped, llm_reminder_request, conversation_context=conversation_context)
        reminder_request = _reminder_request_from_text(stripped)
        if reminder_request is not None:
            return self._run_user_tool_request(stripped, reminder_request, conversation_context=conversation_context)
        memory_update = _memory_write_request_from_text(stripped, self.runtime_context)
        if memory_update is not None:
            return self._run_user_tool_request(stripped, memory_update, conversation_context=conversation_context)
        memory_answer = _answer_memory_recall_question(
            stripped,
            self.runtime_context,
            conversation_context=self.chat_history.format_recent(session_id, limit=6, max_chars=2500),
        )
        if memory_answer is not None:
            if _service_from_text(stripped) is not None:
                return memory_answer
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
            return self._run_user_tool_request(stripped, browser_request, conversation_context=conversation_context)
        terminal_request = _terminal_request_from_text(stripped)
        if terminal_request is not None:
            return self._run_user_tool_request(stripped, terminal_request, conversation_context=conversation_context)
        route_without_llm = _route_without_llm(stripped)
        if route_without_llm is not None:
            return self._run_user_tool_request(stripped, route_without_llm, conversation_context=conversation_context)
        fast_answer = _answer_without_llm(stripped, self.runtime_context.config)
        if fast_answer is not None and _is_fast_control_answer(stripped):
            data: dict[str, Any] = {"planner": "deterministic"}
            if planner_error is not None:
                data["fallback"] = "llm_unavailable"
            return AgentResponse(
                status="ok",
                message=fast_answer,
                data=data,
            )

        if fast_answer is not None:
            data = {"planner": "deterministic"}
            if planner_error is not None:
                data["fallback"] = "llm_unavailable"
            return AgentResponse(
                status="ok",
                message=fast_answer,
                data=data,
            )
        if planner_error is not None:
            unavailable_answer = _answer_llm_unavailable(stripped, self.runtime_context.config, planner_error)
            if unavailable_answer is not None:
                return unavailable_answer
            if local_dev_autonomy_enabled(self.runtime_context.config) and self.planner is not None and hasattr(self.planner, "chat"):
                log_local_dev_autonomy(
                    "planner_recovery_chat_fallback",
                    config=self.runtime_context.config,
                    user_message=stripped,
                    error_type=type(planner_error).__name__,
                )
                chat_response = self._handle_normal_chat(
                    stripped,
                    conversation_context=conversation_context,
                )
                if chat_response.status == "ok" and chat_response.message:
                    data = dict(chat_response.data or {})
                    data["fallback"] = "local_dev_planner_chat"
                    return AgentResponse(
                        status=chat_response.status,
                        message=chat_response.message,
                        data=data,
                    )
            return AgentResponse(
                status="ok",
                message=(
                    "The LLM planner is active, but it did not return a usable structured response for that request. "
                    "I can still show basic system status and tools."
                ),
                data={"planner": "llm", "fallback": "planner_error"},
            )
        return AgentResponse(
            status="ok",
            message=(
                "The LLM planner is not active, so I cannot perform context-aware tasks. "
                "I can still show basic system status and tools."
            ),
            data={"planner": "inactive"},
        )

    def _handle_with_planner(
        self,
        stripped: str,
        text: str,
        *,
        session_id: str,
        conversation_context: str,
    ) -> AgentResponse:
        if local_dev_autonomy_enabled(self.runtime_context.config):
            return self._handle_with_autonomy_loop(
                stripped,
                text,
                session_id=session_id,
                conversation_context=conversation_context,
            )
        try:
            plan = self._build_planner_plan(
                text=text,
                conversation_context=conversation_context,
            )
        except Exception as exc:
            log_local_dev_autonomy(
                "planner_exception",
                config=self.runtime_context.config,
                error_type=type(exc).__name__,
                message=str(exc),
                user_message=stripped,
            )
            return self._handle_without_llm(
                stripped,
                text,
                session_id=session_id,
                conversation_context=self._conversation_context(session_id),
                planner_error=exc,
            )

        if is_low_relevance_plan(plan, stripped):
            log_local_dev_autonomy(
                "planner_low_relevance",
                config=self.runtime_context.config,
                user_message=stripped,
                final_message=plan.final_message or "",
                clarification_message=plan.clarification_message or "",
            )
            return self._handle_without_llm(
                stripped,
                text,
                session_id=session_id,
                conversation_context=self._conversation_context(session_id),
                planner_error=PlannerError("Planner returned an unrelated response for an explicit tool request."),
            )

        response = self._execute_planner_decision(
            stripped,
            plan,
            conversation_context=conversation_context,
        )
        self._run_autonomous_memory_writes(plan, stripped, response)
        return response

    def _handle_with_autonomy_loop(
        self,
        stripped: str,
        text: str,
        *,
        session_id: str,
        conversation_context: str,
    ) -> AgentResponse:
        runtime = self.runtime_context.config.get("runtime", {})
        autonomy = runtime.get("autonomy", {}) if isinstance(runtime, dict) else {}
        depth = _execution_depth_for_request(text)
        configured_iterations = max(1, min(int(autonomy.get("max_iterations", 3)), 6)) if isinstance(autonomy, dict) else 3
        max_iterations = min(configured_iterations, 1 if depth == "light" else 2 if depth == "medium" else 3)
        observations: list[str] = []
        last_error: Exception | None = None
        for iteration in range(max_iterations):
            self._trace_event(
                "loop",
                "Choosing execution strategy",
                status="running",
                metadata={"iteration": iteration + 1, "max_iterations": max_iterations, "depth": depth},
            )
            try:
                plan = self._build_planner_plan(
                    text=text,
                    conversation_context=conversation_context,
                    recovery_note="\n".join(observations[-3:]),
                    depth=depth,
                )
            except Exception as exc:
                last_error = exc
                self._trace_event(
                    "planner_retry",
                    "Refining execution plan",
                    status="retry" if iteration + 1 < max_iterations else "error",
                    detail=str(exc),
                    metadata={"iteration": iteration + 1, "visibility": "debug"},
                )
                log_local_dev_autonomy(
                    "planner_retry",
                    config=self.runtime_context.config,
                    error_type=type(exc).__name__,
                    user_message=stripped,
                    iteration=iteration + 1,
                )
                observations.append(
                    "Previous planner call failed. Return exactly one valid JSON decision for the current user message."
                )
                if iteration + 1 < max_iterations:
                    continue
                break

            if is_low_relevance_plan(plan, stripped):
                self._trace_event(
                    "planner_retry",
                    "Refining execution plan",
                    status="retry" if iteration + 1 < max_iterations else "error",
                    detail=plan.final_message or plan.clarification_message or "",
                    metadata={"iteration": iteration + 1, "visibility": "debug"},
                )
                observations.append(
                    "Previous planner output was unrelated to the current user message. Ignore stale benchmark/repair context."
                )
                if iteration + 1 < max_iterations:
                    continue
                last_error = PlannerError("Planner returned an unrelated response for an explicit tool request.")
                break

            response = self._execute_planner_decision(
                stripped,
                plan,
                conversation_context=conversation_context,
            )
            if not _autonomy_should_retry_after_response(response):
                self._trace_event(
                    "loop",
                    "Finished autonomous execution",
                    status=response.status,
                    metadata={"iteration": iteration + 1},
                )
                self._run_autonomous_memory_writes(plan, stripped, response)
                return response
            observations.append(f"Previous action failed: {response.message[:500]}")
            self._trace_event(
                "loop",
                "Retrying failed execution step",
                status="retry",
                detail=response.message,
                metadata={"iteration": iteration + 1, "status": response.status},
            )

        return self._handle_without_llm(
            stripped,
            text,
            session_id=session_id,
            conversation_context=self._conversation_context(session_id),
            planner_error=last_error or PlannerError("Autonomy loop did not produce a usable response."),
        )

    def _build_planner_plan(
        self,
        *,
        text: str,
        conversation_context: str,
        recovery_note: str = "",
        depth: str = "medium",
    ) -> Any:
        llm_config = self.runtime_context.config.get("llm", {})
        memory_context = _load_memory_context(
            self.runtime_context,
            query=text,
            conversation_context=conversation_context,
            allow_cloud_context=self._cloud_context_approved,
        )
        if recovery_note:
            conversation_context = (
                f"{conversation_context}\n\nRuntime recovery observations:\n{recovery_note}"
                if conversation_context
                else f"Runtime recovery observations:\n{recovery_note}"
            )
        now = datetime.now().astimezone()
        planner_max_tokens = max(512, int(llm_config.get("planner_max_tokens", 192)))
        if local_dev_autonomy_enabled(self.runtime_context.config) and depth == "deep":
            planner_max_tokens = max(planner_max_tokens, 2048)
        elif local_dev_autonomy_enabled(self.runtime_context.config):
            planner_max_tokens = max(planner_max_tokens, 768)
        self._trace_event(
            "planner",
            "Planning workspace action",
            status="running",
            metadata={
                "provider": str(llm_config.get("provider") or ""),
                "model": str(llm_config.get("planner_model") or llm_config.get("model") or ""),
                "autonomy": local_dev_autonomy_enabled(self.runtime_context.config),
                "depth": depth,
            },
        )
        plan = self.planner.plan(
            user_message=text,
            manifests=self.tool_registry.manifests,
            profile=_profile_from_config(self.runtime_context.config),
            memory_context=memory_context,
            conversation_context=conversation_context,
            agent_context=self._agent_context_snapshot(),
            enabled_tools=_enabled_tools_from_config(
                self.tool_registry.manifests,
                self.runtime_context.config,
                self._permission_context(),
            ),
            response_language=llm_config.get("response_language", "auto"),
            current_time=now.isoformat(timespec="seconds"),
            timezone_name=now.tzname() or "",
            max_tokens=planner_max_tokens,
            repair_max_tokens=int(llm_config.get("repair_max_tokens", 256)),
            temperature=float(llm_config.get("planner_temperature", 0.0)),
            think=bool(llm_config.get("planner_think", False)),
            system_prompt=_llm_system_prompt(llm_config, "planner"),
            context_detail="deep" if depth == "deep" else "compact",
        )
        self._trace_event(
            "planner",
            "Selected execution action",
            status="ok",
            metadata=_planner_decision_summary(plan),
        )
        return plan

    def _run_autonomous_memory_writes(
        self,
        plan: Any,
        user_message: str,
        response: AgentResponse,
    ) -> None:
        intents = getattr(plan, "memory_writes", ()) or ()
        if not intents:
            return
        if response.status not in {"ok", "needs_input"}:
            return
        try:
            manager = MemoryManager(self.runtime_context.memory_root)
        except Exception:
            return
        for intent in intents:
            if intent.confidence != "high":
                continue
            try:
                self._apply_memory_write_intent(manager, intent)
            except Exception as exc:
                log_local_dev_autonomy(
                    "autonomous_scribe_error",
                    config=self.runtime_context.config,
                    slug=intent.slug,
                    error_type=type(exc).__name__,
                    message=str(exc)[:200],
                )
                continue

    def _apply_memory_write_intent(
        self,
        manager: MemoryManager,
        intent: MemoryWriteIntent,
    ) -> None:
        path = intent.path.strip() or f"long-term/topics/{intent.memory_type}_{intent.slug}.md"
        if not path.endswith(".md"):
            path = f"{path}.md"
        try:
            existing_full = manager.read(path)
            existing = _strip_frontmatter(existing_full).strip()
        except FileNotFoundError:
            existing = ""
        new_body = intent.body.strip()
        if not new_body:
            return
        if existing and new_body in existing:
            return
        timestamp = datetime.now().astimezone().strftime("%Y-%m-%d")
        if existing:
            combined = f"{existing}\n\n## {timestamp}\n\n{new_body}".strip()
        else:
            heading = intent.description.strip() or intent.slug.replace("-", " ").title()
            combined = f"# {heading}\n\n{new_body}".strip()
        metadata = {
            "name": intent.slug,
            "description": (intent.description or intent.slug).replace("\n", " ")[:200],
            "type": intent.memory_type,
            "memory_scope": "long-term",
            "source": "autonomous_scribe",
            "confidence": intent.confidence,
        }
        manager.write(path, combined, metadata=metadata)
        log_local_dev_autonomy(
            "autonomous_scribe_write",
            config=self.runtime_context.config,
            slug=intent.slug,
            path=path,
            confidence=intent.confidence,
            type=intent.memory_type,
        )

    def _execute_planner_decision(
        self,
        stripped: str,
        plan: Any,
        *,
        conversation_context: str,
    ) -> AgentResponse:
        if plan.tool_request is not None:
            request = _normalize_planner_tool_request(
                plan.tool_request,
                stripped,
                self.runtime_context.config,
            )
            if _is_email_create_draft_tool(request.tool) and _looks_email_send_request(_normalize_for_match(stripped)):
                return self._handle_email_create_then_send_request(
                    stripped,
                    request,
                    conversation_context=conversation_context,
                )
            log_local_dev_autonomy(
                "planner_selected_tool",
                config=self.runtime_context.config,
                tool=request.tool,
                args=_redact_step_args_for_response(request.args),
            )
            return self._run_user_tool_request(
                stripped,
                request,
                conversation_context=conversation_context,
            )

        if plan.tool_plan:
            log_local_dev_autonomy(
                "planner_selected_chain",
                config=self.runtime_context.config,
                tools=[request.tool for request in plan.tool_plan],
            )
            return self._run_user_tool_plan(
                stripped,
                plan.tool_plan,
                conversation_context=conversation_context,
            )

        if plan.memory_query:
            return self._handle_memory_search_decision(
                stripped,
                plan.memory_query,
                conversation_context=conversation_context,
            )

        if plan.clarification_message:
            return AgentResponse(
                status="needs_input",
                message=plan.clarification_message,
                data={"planner": "llm", "decision": "ask_clarification"},
            )

        return AgentResponse(
            status="ok",
            message=plan.final_message or "",
            data={"planner": "llm", "decision": "answer"},
        )

    def _run_user_tool_plan(
        self,
        user_message: str,
        requests: tuple[ToolRequest, ...],
        *,
        conversation_context: str,
    ) -> AgentResponse:
        if not requests:
            return AgentResponse(
                status="ok",
                message=_tool_selection_failure_message(user_message),
                data={"planner": "llm", "decision": "empty_tool_plan"},
            )
        normalized_requests = tuple(
            _normalize_planner_tool_request(request, user_message, self.runtime_context.config)
            for request in requests
        )
        if (
            len(normalized_requests) == 1
            and _is_email_create_draft_tool(normalized_requests[0].tool)
            and _looks_email_send_request(_normalize_for_match(user_message))
        ):
            return self._handle_email_create_then_send_request(
                user_message,
                normalized_requests[0],
                conversation_context=conversation_context,
            )
        log_local_dev_autonomy(
            "chain_start",
            config=self.runtime_context.config,
            tools=[request.tool for request in normalized_requests],
            user_message=user_message,
        )
        self._trace_event(
            "chain",
            "Starting action sequence",
            status="running",
            metadata={"tools": [request.tool for request in normalized_requests]},
        )
        first, *continuations = normalized_requests
        first = _prepare_tool_request_for_execution(first, self.runtime_context)
        first_request = _request_with_original_message(first, user_message)
        response = self.handle_tool_request(first_request)
        if response.status == "approval_required":
            approval_id = (response.data or {}).get("approval_id")
            if isinstance(approval_id, int) and continuations:
                self._pending_continuations[approval_id] = PendingContinuation(
                    original_message=user_message,
                    steps=tuple(continuations),
                )
                return AgentResponse(
                    status=response.status,
                    message=_multi_step_approval_message(user_message, first_request, continuations),
                    data={
                        **(response.data or {}),
                        "plan": _continuation_plan_for_user(continuations),
                        "planner": "llm",
                        "decision": "multi_tool_plan",
                    },
                )
        if response.status != "ok" or not continuations:
            shaped = self._shape_tool_response(user_message, first_request, response)
            if self._should_synthesize_tool_response(user_message, first_request, response):
                return self._synthesize_tool_response(
                    user_message,
                    response,
                    conversation_context=conversation_context,
                )
            return shaped
        return self._execute_continuation_steps(
            user_message,
            first_request=first_request,
            first_response=response,
            steps=tuple(continuations),
            conversation_context=conversation_context,
        )

    def _should_defer_email_send_to_planner(self, user_message: str) -> bool:
        if self.planner is None:
            return False
        return _email_send_needs_llm_composition(user_message)

    def _handle_memory_search_decision(
        self,
        user_message: str,
        query: str,
        *,
        conversation_context: str,
    ) -> AgentResponse:
        memory_context_for_query = "" if _service_from_text(query) is not None else conversation_context
        memory_answer = _answer_memory_recall_question(
            query,
            self.runtime_context,
            conversation_context=memory_context_for_query,
        )
        if memory_answer is None:
            return AgentResponse(
                status="ok",
                message=(
                    "Не намирам това в локалната memory още."
                    if _looks_bulgarian(user_message)
                    else "I do not have that in local memory yet."
                ),
                data={"planner": "llm", "decision": "memory_search"},
            )
        if _service_from_text(query) is not None:
            return AgentResponse(
                status=memory_answer.status,
                message=memory_answer.message,
                data={"planner": "llm", "decision": "memory_search"},
            )
        if self._can_send_private_context_to_llm():
            synthesized = self._synthesize_memory_recall_response(
                user_message,
                memory_answer,
                conversation_context=conversation_context,
            )
            if synthesized is not None:
                return AgentResponse(
                    status=synthesized.status,
                    message=synthesized.message,
                    data={"planner": "llm", "decision": "memory_search"},
                )
        return AgentResponse(
            status=memory_answer.status,
            message=memory_answer.message,
            data={"planner": "llm", "decision": "memory_search"},
        )

    def _handle_normal_chat(
        self,
        user_message: str,
        *,
        conversation_context: str,
    ) -> AgentResponse:
        fast_answer = _answer_without_llm(user_message, self.runtime_context.config)
        if fast_answer is not None and _is_fast_control_answer(user_message):
            return AgentResponse(status="ok", message=fast_answer, data={"mode": "normal_chat"})
        if self.planner is None or not hasattr(self.planner, "chat"):
            return AgentResponse(
                status="ok",
                message=fast_answer or ("Кажи ми какво ти трябва." if _looks_bulgarian(user_message) else "Tell me what you need."),
                data={"mode": "normal_chat"},
            )
        try:
            llm_config = self.runtime_context.config.get("llm", {})
            answer = self.planner.chat(
                user_message=user_message,
                profile=_profile_from_config(self.runtime_context.config),
                memory_context=_load_memory_context(
                    self.runtime_context,
                    query=user_message,
                    conversation_context=conversation_context,
                    allow_cloud_context=self._cloud_context_approved,
                    max_chars=4000,
                ),
                conversation_context=conversation_context,
                agent_context=self._agent_context_snapshot(),
                response_language=llm_config.get("response_language", "auto"),
                max_tokens=int(llm_config.get("chat_max_tokens", 1024)),
                temperature=0.4,
                think=bool(llm_config.get("planner_think", False)),
                system_prompt=_llm_system_prompt(llm_config, "chat"),
            )
        except (PlannerError, OSError, RuntimeError) as exc:
            unavailable_answer = _answer_llm_unavailable(user_message, self.runtime_context.config, exc)
            if unavailable_answer is not None:
                return unavailable_answer
            return AgentResponse(
                status="ok",
                message=(
                    "Не успях да отговоря през модела в момента."
                    if _looks_bulgarian(user_message)
                    else "I could not answer through the model right now."
                ),
                data={"debug": {"chat_error": str(exc)}},
            )
        return AgentResponse(status="ok", message=answer, data={"mode": "normal_chat"})

    def _run_user_tool_request(
        self,
        user_message: str,
        request: ToolRequest,
        *,
        conversation_context: str,
    ) -> AgentResponse:
        request = _prepare_tool_request_for_execution(request, self.runtime_context)
        request = _request_with_original_message(request, user_message)
        log_local_dev_autonomy(
            "selected_tool",
            config=self.runtime_context.config,
            tool=request.tool,
            args=_redact_step_args_for_response(request.args),
        )
        self._trace_event(
            "tool",
            _tool_action_title(request.tool),
            status="running",
            tool=request.tool,
            metadata={"args": _redact_step_args_for_response(request.args)},
        )
        tool_response = self.handle_tool_request(request)
        if self._should_synthesize_tool_response(user_message, request, tool_response):
            return self._synthesize_tool_response(
                user_message,
                tool_response,
                conversation_context=conversation_context,
            )
        return self._shape_tool_response(user_message, request, tool_response)

    def _handle_email_send_intent(self, user_message: str, intent: EmailSendIntent) -> AgentResponse:
        create_request = _request_with_original_message(
            ToolRequest(
                tool=f"{intent.provider}.create_draft",
                args={"to": intent.to, "subject": intent.subject, "body": intent.body},
                reason="User asked to send an email; create a local draft before approval-gated send.",
            ),
            user_message,
        )
        draft_response = self.handle_tool_request(create_request)
        if draft_response.status != "ok":
            return self._shape_tool_response(user_message, create_request, draft_response)
        draft_id = str((draft_response.data or {}).get("draft_id") or "").strip()
        if not draft_id:
            return AgentResponse(
                status="error",
                message=(
                    "Създаването на чернова не върна draft_id."
                    if _looks_bulgarian(user_message)
                    else "Draft creation did not return a draft_id."
                ),
                data=draft_response.data,
            )
        send_request = _request_with_original_message(
            ToolRequest(
                tool=f"{intent.provider}.send_draft",
                args={"draft_id": draft_id},
                reason="User asked to send this email. Sending requires explicit approval.",
            ),
            user_message,
            )
        send_response = self.handle_tool_request(send_request)
        if send_response.status == "approval_required":
            return _email_send_approval_response(
                user_message,
                draft=_local_email_draft(self.runtime_context, intent.provider, draft_id)
                or {
                    "draft_id": draft_id,
                    "to": [intent.to],
                    "subject": intent.subject,
                    "body": intent.body,
                },
                send_response=send_response,
            )
        return self._shape_tool_response(user_message, send_request, send_response)

    def _handle_email_create_then_send_request(
        self,
        user_message: str,
        create_request: ToolRequest,
        *,
        conversation_context: str,
    ) -> AgentResponse:
        del conversation_context
        create_request = _prepare_tool_request_for_execution(create_request, self.runtime_context)
        create_request = _request_with_original_message(create_request, user_message)
        draft_response = self.handle_tool_request(create_request)
        if draft_response.status != "ok":
            return self._shape_tool_response(user_message, create_request, draft_response)
        provider = create_request.tool.split(".", 1)[0]
        draft_id = str((draft_response.data or {}).get("draft_id") or "").strip()
        if not draft_id:
            return AgentResponse(
                status="error",
                message=(
                    "Създаването на чернова не върна draft_id."
                    if _looks_bulgarian(user_message)
                    else "Draft creation did not return a draft_id."
                ),
                data=draft_response.data,
            )
        send_request = _request_with_original_message(
            ToolRequest(
                tool=f"{provider}.send_draft",
                args={"draft_id": draft_id},
                reason="User asked to send this generated email. Sending requires explicit approval.",
            ),
            user_message,
        )
        send_response = self.handle_tool_request(send_request)
        if send_response.status == "approval_required":
            return _email_send_approval_response(
                user_message,
                draft=_local_email_draft(self.runtime_context, provider, draft_id)
                or {
                    "draft_id": draft_id,
                    "to": (draft_response.data or {}).get("to", []),
                    "subject": (draft_response.data or {}).get("subject", ""),
                    "body": str(create_request.args.get("body") or ""),
                },
                send_response=send_response,
            )
        return self._shape_tool_response(user_message, send_request, send_response)

    def _handle_multi_step_request(
        self,
        user_message: str,
        *,
        conversation_context: str,
    ) -> AgentResponse | None:
        del conversation_context
        plan = _multi_step_plan_from_text(user_message)
        if plan is None:
            return None
        first, *continuations = plan
        if isinstance(first, UnsupportedAction):
            return _unsupported_action_response(user_message, first)
        first = _prepare_tool_request_for_execution(first, self.runtime_context)
        first_request = _request_with_original_message(first, user_message)
        response = self.handle_tool_request(first_request)
        if response.status == "approval_required":
            approval_id = (response.data or {}).get("approval_id")
            if isinstance(approval_id, int) and continuations:
                self._pending_continuations[approval_id] = PendingContinuation(
                    original_message=user_message,
                    steps=tuple(continuations),
                )
                return AgentResponse(
                    status=response.status,
                    message=_multi_step_approval_message(user_message, first_request, continuations),
                    data={
                        **(response.data or {}),
                        "plan": _continuation_plan_for_user(continuations),
                    },
                )
        if response.status != "ok" or not continuations:
            return self._shape_tool_response(user_message, first_request, response)
        return self._execute_continuation_steps(
            user_message,
            first_request=first_request,
            first_response=response,
            steps=tuple(continuations),
        )

    def _execute_continuation_steps(
        self,
        user_message: str,
        *,
        first_request: ToolRequest,
        first_response: AgentResponse,
        steps: tuple[ToolRequest | UnsupportedAction, ...],
        initial_chain_state: dict[str, Any] | None = None,
        conversation_context: str = "",
    ) -> AgentResponse:
        messages: list[str] = []
        executed_steps: list[ToolRequest] = [first_request]
        data: dict[str, Any] = {
            "tool": "multi_step",
            "first_tool": first_request.tool,
            "plan": _continuation_plan_for_user(steps),
            "steps": [],
        }
        shaped_first = self._shape_tool_response(user_message, first_request, first_response)
        if shaped_first.message:
            messages.append(shaped_first.message)
        data["steps"].append(
            {
                "tool": first_request.tool,
                "args": _redact_step_args_for_response(first_request.args),
                "status": shaped_first.status,
                "data": shaped_first.data or {},
            }
        )
        previous_response = first_response
        chain_state: dict[str, Any] = dict(initial_chain_state or {})
        _update_chain_state(chain_state, first_request, first_response)
        for index, step in enumerate(steps):
            if isinstance(step, UnsupportedAction):
                unsupported = _unsupported_action_response(user_message, step)
                messages.append(unsupported.message)
                data["steps"].append({"kind": step.kind, "target": step.target, "status": "unsupported"})
                continue
            try:
                runnable_step = _resolve_previous_step_content(
                    step,
                    previous_response=previous_response,
                    chain_state=chain_state,
                    runtime_context=self.runtime_context,
                )
            except ValueError as exc:
                return AgentResponse(
                    status="error",
                    message="\n\n".join([*messages, str(exc)]).strip(),
                    data=data,
                )
            runnable_step = _prepare_tool_request_for_execution(runnable_step, self.runtime_context)
            log_local_dev_autonomy(
                "chain_step",
                config=self.runtime_context.config,
                index=index + 2,
                tool=runnable_step.tool,
                args=_redact_step_args_for_response(runnable_step.args),
            )
            self._trace_event(
                "chain",
                _tool_action_title(runnable_step.tool),
                status="running",
                tool=runnable_step.tool,
                metadata={"args": _redact_step_args_for_response(runnable_step.args)},
            )
            response = self.handle_tool_request(_request_with_original_message(runnable_step, user_message))
            shaped = self._shape_tool_response(user_message, runnable_step, response)
            messages.append(shaped.message)
            executed_steps.append(runnable_step)
            data["steps"].append(
                {
                    "tool": runnable_step.tool,
                    "args": _redact_step_args_for_response(runnable_step.args),
                    "status": shaped.status,
                    "data": shaped.data or {},
                }
            )
            if shaped.status != "ok":
                if isinstance(shaped.data, dict):
                    for key in ("approval_id", "risk", "missing_permissions", "emergency_stop"):
                        if key in shaped.data:
                            data[key] = shaped.data[key]
                    data["pending_tool"] = shaped.data.get("tool", runnable_step.tool)
                    approval_id = shaped.data.get("approval_id")
                    remaining_steps = tuple(steps[index + 1 :])
                    if (
                        shaped.status == "approval_required"
                        and isinstance(approval_id, int)
                        and remaining_steps
                    ):
                        self._pending_continuations[approval_id] = PendingContinuation(
                            original_message=user_message,
                            steps=remaining_steps,
                            chain_state=dict(chain_state),
                        )
                return AgentResponse(
                    status=shaped.status,
                    message="\n\n".join(message for message in messages if message),
                    data=data,
                )
            previous_response = response
            _update_chain_state(chain_state, runnable_step, response)
        final_response = AgentResponse(
            status="ok",
            message="\n\n".join(message for message in messages if message),
            data=data,
        )
        if self._should_synthesize_tool_plan_response(
            user_message,
            tuple(executed_steps),
            final_response,
        ):
            return self._synthesize_tool_response(
                user_message,
                final_response,
                conversation_context=conversation_context,
            )
        return final_response

    def _shape_tool_response(
        self,
        user_message: str,
        request: ToolRequest,
        response: AgentResponse,
    ) -> AgentResponse:
        if response.status != "ok":
            return response
        data = response.data or {}
        if request.tool == "terminal.run":
            return _terminal_user_response(user_message, request, data)
        if request.tool == "memory.list":
            return _memory_list_user_response(user_message, data)
        if request.tool == "memory.read":
            return _memory_read_user_response(user_message, data)
        if request.tool == "files.list":
            path = str(data.get("path") or request.args.get("path") or "")
            count = int(data.get("count") or 0)
            message = (
                f"Намерих {count} елемента в: {path}"
                if _looks_bulgarian(user_message)
                else f"Found {count} item(s) in: {path}"
            )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, **data})
        if request.tool == "files.read":
            return _file_read_user_response(user_message, data)
        if request.tool.startswith(("gmail.", "outlook.")):
            return _email_user_response(user_message, request, data)
        if request.tool == "files.mkdir":
            path = str(data.get("path") or request.args.get("path") or "")
            message = (
                f"Създадох папката: {path}"
                if _looks_bulgarian(user_message)
                else f"Created folder: {path}"
            )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, "path": path})
        if request.tool == "files.write":
            path = str(data.get("path") or request.args.get("path") or "")
            appended = bool(data.get("appended") or request.args.get("append"))
            if appended:
                message = (
                    f"Добавих текста във файла: {path}"
                    if _looks_bulgarian(user_message)
                    else f"Appended to file: {path}"
                )
            else:
                message = (
                    f"Записах файла: {path}"
                    if _looks_bulgarian(user_message)
                    else f"Wrote file: {path}"
                )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, "path": path})
        if request.tool == "files.write_many":
            count = int(data.get("count") or 0)
            message = (
                f"Записах {count} файла."
                if _looks_bulgarian(user_message)
                else f"Wrote {count} files."
            )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, **data})
        if request.tool == "project.scaffold_one_page_app":
            path = str(data.get("path") or request.args.get("path") or "")
            count = int(data.get("count") or 0)
            commands = data.get("next_commands") if isinstance(data.get("next_commands"), list) else []
            command_text = "\n".join(f"- {command}" for command in commands)
            message = (
                f"Създадох one-page React/Node проект в: {path}\nФайлове: {count}\nСледващи команди:\n{command_text}"
                if _looks_bulgarian(user_message)
                else f"Scaffolded one-page React/Node project in: {path}\nFiles: {count}\nNext commands:\n{command_text}"
            )
            return AgentResponse(status="ok", message=message.strip(), data={"tool": request.tool, **data})
        if request.tool == "files.delete":
            path = str(data.get("path") or request.args.get("path") or "")
            kind = str(data.get("kind") or "path")
            message = (
                f"Изтрих {kind}: {path}"
                if _looks_bulgarian(user_message)
                else f"Deleted {kind}: {path}"
            )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, "path": path})
        if request.tool == "workspace.switch":
            current = str(data.get("current_workspace") or "")
            message = (
                f"Смених workspace на: {current}"
                if _looks_bulgarian(user_message)
                else f"Switched workspace to: {current}"
            )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, "current_workspace": current})
        if request.tool == "workspace.status":
            current = str(data.get("current_workspace") or "")
            message = (
                f"Текущият workspace е: {current}"
                if _looks_bulgarian(user_message)
                else f"Current workspace: {current}"
            )
            return AgentResponse(status="ok", message=message, data={"tool": request.tool, "current_workspace": current})
        if request.tool == "developer.context":
            scan_root = str(data.get("scan_root") or "")
            file_count = int(data.get("file_count") or 0)
            message = (
                f"Заредих developer context за {file_count} файла в: {scan_root}"
                if _looks_bulgarian(user_message)
                else f"Loaded developer context for {file_count} files in: {scan_root}"
            )
            return AgentResponse(status="ok", message=message, data=data)
        if request.tool == "browser.scrape_markdown":
            return _browser_scrape_user_response(user_message, data)
        return response

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

    def _handle_session_feedback(self, text: str, *, session_id: str = "default") -> AgentResponse | None:
        correction = _entity_correction_from_text(text)
        if correction is not None:
            wanted, wrong = correction
            note = f"{wanted} != {wrong}. If the user asks for {wanted}, do not return {wrong} details."
            corrections = self._session_corrections.setdefault(session_id, [])
            if note not in corrections:
                corrections.append(note)
            message = (
                f"Записах го за тази сесия: {wanted} не е {wrong}. При въпрос за {wanted} няма да връщам {wrong} адрес."
                if _looks_bulgarian(text)
                else f"Noted for this session: {wanted} is not {wrong}. If you ask for {wanted}, I will not return {wrong} details."
            )
            return AgentResponse(
                status="ok",
                message=message,
                data={"planner": "deterministic", "source": "session_correction"},
            )
        if _is_do_not_repeat_error_feedback(text):
            corrections = self._session_corrections.get(session_id, [])
            if corrections:
                latest = corrections[-1]
                message = (
                    f"Разбрано. Активната correction за тази сесия е: {latest}"
                    if _looks_bulgarian(text)
                    else f"Understood. Active session correction: {latest}"
                )
            else:
                message = (
                    "Разбрано. Ако става дума за конкретен service или адрес, кажи ми кое беше грешното и кое е правилното."
                    if _looks_bulgarian(text)
                    else "Understood. If this is about a specific service or address, tell me what was wrong and what is correct."
                )
            return AgentResponse(
                status="ok",
                message=message,
                data={"planner": "deterministic", "source": "session_feedback"},
            )
        return None

    def _should_synthesize_tool_plan_response(
        self,
        user_message: str,
        requests: tuple[ToolRequest, ...],
        response: AgentResponse,
    ) -> bool:
        if self.planner is None or not hasattr(self.planner, "answer") or response.status != "ok":
            return False
        if not requests:
            return False
        if not all(request.tool in CONTEXT_GATHERING_SYNTHESIS_TOOLS for request in requests):
            return False
        if not any(request.tool in {"memory.list", "memory.read", "developer.context"} for request in requests):
            return False
        normalized = _normalize_for_match(user_message)
        if _explicit_raw_memory_request(normalized):
            return False
        return True

    def _should_synthesize_tool_response(
        self,
        user_message: str,
        request: ToolRequest,
        response: AgentResponse,
    ) -> bool:
        if self.planner is None or not hasattr(self.planner, "answer") or response.status != "ok":
            return False
        if request.tool == "developer.context":
            return self._can_send_private_context_to_llm()
        if request.tool not in {"memory.list", "memory.read"}:
            return False
        normalized = _normalize_for_match(user_message)
        return not _explicit_raw_memory_request(normalized)

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
                agent_context=self._agent_context_snapshot(),
                tool_result=tool_response.data or {},
                response_language=llm_config.get("response_language", "auto"),
                max_tokens=max(256, int(llm_config.get("synthesis_max_tokens", 1024))),
                temperature=0.2,
                think=bool(llm_config.get("planner_think", False)),
                system_prompt=_llm_system_prompt(llm_config, "answer"),
            )
            if answer:
                return AgentResponse(status="ok", message=answer, data={"planner": "llm"})
        except (PlannerError, OSError, RuntimeError):
            return _safe_tool_synthesis_fallback(tool_response, user_message)
        return _safe_tool_synthesis_fallback(tool_response, user_message)

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
                agent_context=self._agent_context_snapshot(),
                tool_result={
                    "kind": "memory_recall",
                    "answer": memory_answer.message,
                    "data": memory_answer.data or {},
                },
                response_language=llm_config.get("response_language", "auto"),
                max_tokens=max(384, int(llm_config.get("synthesis_max_tokens", 1024))),
                temperature=0.25,
                think=bool(llm_config.get("planner_think", False)),
                system_prompt=_llm_system_prompt(llm_config, "answer"),
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
            planner_max_tokens = max(512, int(llm_config.get("planner_max_tokens", 192)))
            if local_dev_autonomy_enabled(self.runtime_context.config):
                planner_max_tokens = max(planner_max_tokens, 2048)
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
                agent_context=self._agent_context_snapshot(),
                enabled_tools=_enabled_tools_from_config(
                    self.tool_registry.manifests,
                    self.runtime_context.config,
                    self._permission_context(),
                ),
                response_language=llm_config.get("response_language", "auto"),
                current_time=now.isoformat(timespec="seconds"),
                timezone_name=now.tzname() or "",
                max_tokens=planner_max_tokens,
                repair_max_tokens=int(llm_config.get("repair_max_tokens", 256)),
                temperature=float(llm_config.get("planner_temperature", 0.0)),
                think=bool(llm_config.get("planner_think", False)),
                system_prompt=_llm_system_prompt(llm_config, "planner"),
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

    def _answer_chat_history_question(self, text: str, *, session_id: str = "default") -> AgentResponse | None:
        normalized = _normalize_for_match(text)
        asks_first = any(
            marker in normalized
            for marker in {
                "първото съобщение",
                "първия въпрос",
                "първият въпрос",
                "първо ме попита",
                "first message",
                "first question",
            }
        )
        if not asks_first:
            return None
        first = self.chat_history.first_user_message(session_id)
        if first is None:
            return AgentResponse(
                status="ok",
                message=(
                    "Нямам записано първо потребителско съобщение за тази chat session."
                    if _looks_bulgarian(text)
                    else "I do not have a recorded first user message for this chat session."
                ),
                data={"planner": "deterministic", "source": "chat_history"},
            )
        if _looks_bulgarian(text):
            message = f"Първото ти съобщение в тази chat session беше: {first.content}"
        else:
            message = f"Your first message in this chat session was: {first.content}"
        return AgentResponse(
            status="ok",
            message=message,
            data={"planner": "deterministic", "source": "chat_history"},
        )

    def _can_send_private_context_to_llm(self) -> bool:
        return not self.permission_context.cloud_model_active or self._cloud_context_approved

    def _conversation_context(self, session_id: str) -> str:
        if (
            self.permission_context.cloud_model_active
            and not self._cloud_context_approved
            and not _chat_history_allowed_to_cloud(self.runtime_context.config)
        ):
            return ""
        llm_config = self.runtime_context.config.get("llm", {})
        return self.chat_history.format_recent(
            session_id,
            limit=int(llm_config.get("chat_history_turns", 24)),
            max_chars=int(llm_config.get("chat_history_char_limit", 12000)),
        )

    def _planner_conversation_context(self, session_id: str) -> str:
        if (
            self.permission_context.cloud_model_active
            and not self._cloud_context_approved
            and not _chat_history_allowed_to_cloud(self.runtime_context.config)
        ):
            return ""
        llm_config = self.runtime_context.config.get("llm", {})
        return self.chat_history.format_recent(
            session_id,
            limit=int(llm_config.get("planner_history_turns", 4)),
            max_chars=int(llm_config.get("planner_history_char_limit", 3000)),
        )

    def _record_chat_turn(self, session_id: str, user_message: str, response: AgentResponse) -> None:
        try:
            self.chat_history.append(session_id, "user", user_message)
            self.chat_history.append(session_id, "assistant", _history_text_from_response(response))
        except (OSError, ValueError):
            return

    def _remember_tool_response(self, request: ToolRequest, response: AgentResponse) -> AgentResponse:
        if response.status in {"denied", "approval_required", "error", "not_configured", "not_implemented"}:
            self._last_denial_reason = response.message
        if response.status == "ok":
            self._last_denial_reason = ""
        self._last_tool_result_summary = {
            "tool": request.tool,
            "status": response.status,
            "message": response.message[:500],
            "data": _safe_tool_result_summary(response.data or {}),
        }
        return response

    def handle_tool_request(self, request: ToolRequest) -> AgentResponse:
        request = _prepare_tool_request_for_execution(request, self.runtime_context)
        self._trace_event(
            "policy",
            _tool_action_title(request.tool, phase="validate"),
            status="running",
            tool=request.tool,
            metadata={"args": _redact_step_args_for_response(request.args)},
        )
        if _emergency_stop_active(self.runtime_context.config) and request.tool not in {
            "system.list_enabled_tools",
            "workspace.status",
        }:
            self._trace_event("policy", "Emergency stop blocked tool", status="denied", tool=request.tool)
            return self._remember_tool_response(request, AgentResponse(
                status="denied",
                message=(
                    "Emergency stop is active. Tool execution is blocked until you reset emergency mode."
                ),
                data={"missing_permissions": [], "emergency_stop": True},
            ))
        safety = self.safety_policy.evaluate(request, self.runtime_context)
        if not safety.allowed:
            self._trace_event("policy", "Safety policy blocked tool", status="denied", detail=safety.reason, tool=request.tool)
            self.audit_store.record_tool_call(
                tool=request.tool,
                risk=None if safety.risk is None else int(safety.risk),
                args=request.args,
                decision=safety.reason,
                result_status="blocked",
            )
            return self._remember_tool_response(request, AgentResponse(
                status="denied",
                message=safety.reason,
                data={"missing_permissions": []},
            ))

        auto_approved = _is_auto_approved_terminal_request(request, self.runtime_context.config)
        auto_approval_reason = "allowlisted_terminal" if auto_approved else ""
        if local_dev_autonomy_enabled(self.runtime_context.config) and _terminal_request_requires_explicit_approval_in_local_dev(request):
            auto_approved = False
            auto_approval_reason = ""
        local_auto_approval_reason = _local_dev_auto_approval_reason(request, self.runtime_context.config)
        if local_auto_approval_reason:
            auto_approved = True
            auto_approval_reason = local_auto_approval_reason
            log_local_dev_autonomy(
                "approval_bypassed",
                config=self.runtime_context.config,
                tool=request.tool,
                reason=local_auto_approval_reason,
                args=_redact_step_args_for_response(request.args),
            )
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
                f"auto_approved:{auto_approval_reason}:{decision.reason}"
                if auto_approved
                else decision.reason
            ),
            result_status="blocked" if not decision.allowed else None,
        )

        if decision.approval_required or decision.allowed:
            terminal_policy_error = _terminal_policy_error(request, self.runtime_context.config)
            if terminal_policy_error is not None:
                self._trace_event("policy", "Terminal policy blocked command", status="denied", detail=terminal_policy_error, tool=request.tool)
                return self._remember_tool_response(request, AgentResponse(
                    status="denied",
                    message=terminal_policy_error,
                    data={"missing_permissions": []},
                ))

        if decision.approval_required:
            self._trace_event("approval", "Approval required", status="approval_required", detail=decision.reason, tool=request.tool)
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
            return self._remember_tool_response(request, AgentResponse(
                status="approval_required",
                message=_approval_message(request, decision.reason),
                data={
                    "approval_id": approval_id,
                    "tool": request.tool,
                    "risk": None if decision.risk is None else int(decision.risk),
                },
            ))

        if not decision.allowed:
            self._trace_event("policy", "Permission engine denied tool", status="denied", detail=decision.reason, tool=request.tool)
            return self._remember_tool_response(request, AgentResponse(
                status="denied",
                message=_denial_message(request, decision.reason),
                data={"missing_permissions": list(decision.missing_permissions)},
            ))

        self._trace_event("tool", _tool_action_title(request.tool), status="running", tool=request.tool)
        response = self._execute_tool_request(
            request,
            risk=None if decision.risk is None else int(decision.risk),
        )
        self._trace_event(
            "tool",
            _tool_action_title(request.tool, phase="observe"),
            status=response.status,
            tool=request.tool,
            detail=response.message,
        )
        return self._remember_tool_response(request, response)

    def _handle_emergency_text(self, text: str) -> AgentResponse | None:
        normalized = _normalize_for_match(text)
        if any(
            marker in normalized
            for marker in {
                "emergency stop",
                "panic stop",
                "stop everything",
                "abort everything",
                "спешен стоп",
                "авариен стоп",
                "спри всичко",
                "прекрати всичко",
            }
        ):
            result = emergency_stop_terminal_processes()
            _set_emergency_stop(self.runtime_context.config, active=True, reason=text)
            if self.runtime_context.config_path is not None:
                save_config(self.runtime_context.config, self.runtime_context.config_path)
            return AgentResponse(
                status="ok",
                message=(
                    "Emergency stop is active. I stopped active terminal processes and blocked further tool execution until reset."
                    if not _looks_bulgarian(text)
                    else "Emergency stop е активен. Спрях активните terminal процеси и блокирах tool execution до reset."
                ),
                data={"emergency_stop": True, "terminal": result},
            )
        if normalized in {"reset emergency", "clear emergency", "emergency reset", "махни emergency stop", "reset emergency stop"}:
            _set_emergency_stop(self.runtime_context.config, active=False, reason="")
            if self.runtime_context.config_path is not None:
                save_config(self.runtime_context.config, self.runtime_context.config_path)
            return AgentResponse(
                status="ok",
                message="Emergency stop reset. Tool execution can continue through normal policy and approvals.",
                data={"emergency_stop": False},
            )
        return None

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
        request = _prepare_tool_request_for_execution(request, self.runtime_context)
        if _emergency_stop_active(self.runtime_context.config):
            return AgentResponse(
                status="denied",
                message="Emergency stop is active. Approval execution is blocked until you reset emergency mode.",
                data={"approval_id": approval_id, "missing_permissions": [], "emergency_stop": True},
            )
        safety = self.safety_policy.evaluate(request, self.runtime_context)
        if not safety.allowed:
            self.audit_store.record_tool_call(
                tool=request.tool,
                risk=None if safety.risk is None else int(safety.risk),
                args=request.args,
                decision=f"approved_blocked:{safety.reason}",
                result_status="blocked",
            )
            return AgentResponse(
                status="denied",
                message=safety.reason,
                data={"approval_id": approval_id, "missing_permissions": []},
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
        continuation = self._pending_continuations.pop(approval_id, None)
        if response.status == "ok" and continuation is not None:
            return self._execute_continuation_steps(
                continuation.original_message,
                first_request=request,
                first_response=response,
                steps=continuation.steps,
                initial_chain_state=continuation.chain_state,
            )
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
        if response.status == "ok" and original_message:
            return self._shape_tool_response(original_message, request, response)
        return response

    def _permission_context(self) -> PermissionContext:
        context = replace(
            self.permission_context,
            cloud_context_approved=self._cloud_context_approved,
        )
        if not local_dev_autonomy_enabled(self.runtime_context.config):
            return context
        return replace(
            context,
            enabled_tools=context.enabled_tools | LOCAL_DEV_AUTONOMY_ENABLED_TOOLS,
            disabled_tools=context.disabled_tools - LOCAL_DEV_AUTONOMY_ENABLED_TOOLS,
            granted_permissions=context.granted_permissions | LOCAL_DEV_AUTONOMY_PERMISSIONS,
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
        except (ToolExecutionError, PermissionError, ValueError, KeyError, OSError) as exc:
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
        if result.get("status") in {"authentication_failed", "smtp_failed"}:
            self.audit_store.record_event(
                AuditEvent(
                    event_type="tool_call",
                    tool=request.tool,
                    risk=risk,
                    approved=True,
                    result_status="failed",
                    metadata={} if approval_id is None else {"approval_id": approval_id},
                )
            )
            return AgentResponse(
                status="error",
                message=str(result.get("message", "Email delivery failed.")),
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


def _emergency_stop_active(config: dict[str, Any]) -> bool:
    runtime = config.get("runtime", {})
    if not isinstance(runtime, dict):
        return False
    emergency = runtime.get("emergency_stop", {})
    return bool(isinstance(emergency, dict) and emergency.get("active") is True)


def _set_emergency_stop(config: dict[str, Any], *, active: bool, reason: str) -> None:
    emergency = config.setdefault("runtime", {}).setdefault("emergency_stop", {})
    emergency["active"] = bool(active)
    emergency["triggered_at"] = datetime.now().astimezone().isoformat(timespec="seconds") if active else ""
    emergency["reason"] = reason.strip()[:500] if active else ""


def _email_action_from_text(
    text: str,
    config: dict[str, Any],
) -> ToolRequest | EmailSendIntent | AgentResponse | None:
    if not _looks_email_request(text):
        return None
    provider = _email_provider_from_text(text, config)
    if provider is None:
        return AgentResponse(
            status="not_configured",
            message=(
                "Email не е включен. Отвори Config -> Email Connectors, включи Gmail или Outlook и зареди credentials."
                if _looks_bulgarian(text)
                else "Email is not enabled. Open Config -> Email Connectors, enable Gmail or Outlook, and load credentials."
            ),
            data={"connector": "email", "configuration": "Config -> Email Connectors"},
        )
    normalized = _normalize_for_match(text)
    if _looks_email_send_request(normalized):
        return _email_send_intent_from_text(text, provider)
    if _looks_email_read_latest_request(normalized):
        return ToolRequest(
            tool=f"{provider}.read_thread",
            args={"latest": True, "limit": 1},
            reason="User asked to read the latest received email.",
        )
    if _looks_email_summary_request(normalized):
        return ToolRequest(
            tool=f"{provider}.summarize_inbox",
            args={"limit": 5},
            reason="User asked to summarize recent email.",
        )
    if _looks_email_search_request(normalized):
        return ToolRequest(
            tool=f"{provider}.search",
            args={"query": "", "limit": 10},
            reason="User asked to inspect email messages.",
        )
    return None


def _looks_email_request(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(
        marker in normalized
        for marker in {
            "email",
            "e-mail",
            "mail",
            "gmail",
            "google mail",
            "google",
            "гугъл",
            "джимейл",
            "outlook",
            "hotmail",
            "имейл",
            "имейла",
            "мейл",
            "мейла",
            "поща",
        }
    )


def _email_provider_from_text(text: str, config: dict[str, Any]) -> str | None:
    normalized = re.sub(
        r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}",
        "",
        _normalize_for_match(text),
        flags=re.IGNORECASE,
    )
    if any(marker in normalized for marker in {"outlook", "hotmail", "microsoft"}):
        return "outlook"
    if any(marker in normalized for marker in {"gmail", "google mail", "google", "гугъл", "джимейл"}):
        return "gmail"
    enabled = [
        provider
        for provider in ("gmail", "outlook")
        if bool(config.get("email", {}).get(provider, {}).get("enabled", False))
    ]
    if enabled:
        return enabled[0]
    return None


def _looks_email_send_request(normalized: str) -> bool:
    return any(
        marker in normalized
        for marker in {
            "send email",
            "send mail",
            "email to",
            "mail to",
            "send it to",
            "прати имейл",
            "пратиш имейл",
            "прати мейл",
            "пратиш мейл",
            "пратиш на",
            "да го пратиш",
            "го пратиш",
            "изпрати имейл",
            "изпратиш имейл",
            "изпрати мейл",
            "изпратиш мейл",
            "изпратиш на",
            "да го изпратиш",
            "го изпратиш",
            "прати email",
            "пратиш email",
            "изпрати email",
            "изпратиш email",
        }
    )


def _email_send_needs_llm_composition(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(
        marker in normalized
        for marker in {
            "generate email",
            "compose email",
            "write email",
            "генерирай",
            "генерираш",
            "състави",
            "напиши мейл",
            "напиши имейл",
            "business tone",
            "professional tone",
            "formal tone",
            "respectful tone",
            "бизнес",
            "уважител",
            "професионал",
            "официал",
            "делови",
            "тон",
            "формулирай",
            "напиши го",
        }
    )


def _looks_email_read_latest_request(normalized: str) -> bool:
    has_read = any(
        marker in normalized
        for marker in {
            "read",
            "open",
            "tell",
            "show",
            "прочети",
            "прочет",
            "отвори",
            "покажи",
            "кажи",
            "кажеш",
            "кой е",
        }
    )
    has_latest = any(marker in normalized for marker in {"latest", "last", "newest", "послед", "нов", "получен"})
    return has_read and has_latest


def _looks_email_summary_request(normalized: str) -> bool:
    return any(marker in normalized for marker in {"summarize", "summary", "съмари", "обобщи", "обобщение"})


def _looks_email_search_request(normalized: str) -> bool:
    return any(marker in normalized for marker in {"search", "find", "list", "show", "търси", "намери", "покажи"})


def _email_send_intent_from_text(text: str, provider: str) -> EmailSendIntent | AgentResponse:
    recipient = _first_email_address(text)
    if recipient is None:
        return AgentResponse(
            status="needs_input",
            message=(
                "Кажи ми до кой email адрес да го изпратя, каква тема и какво съдържание да има."
                if _looks_bulgarian(text)
                else "Tell me the recipient email address, subject, and body."
            ),
        )
    body = _email_body_from_text(text)
    if not body:
        return AgentResponse(
            status="needs_input",
            message=(
                "Имам получател, но ми трябва съдържанието на имейла."
                if _looks_bulgarian(text)
                else "I have the recipient, but I need the email body."
            ),
        )
    body = _maybe_business_email_body(text, body)
    subject = _email_subject_from_text(text, body)
    return EmailSendIntent(provider=provider, to=recipient, subject=subject, body=body)


def _first_email_address(text: str) -> str | None:
    match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE)
    return match.group(0) if match else None


def _email_body_from_text(text: str) -> str:
    patterns = [
        r"(?:искам\s+)?(?:в|във)\s+(?:имейла|имейлът|мейла|мейлът)\s+(?:искам\s+)?(?:да\s+)?(?:пише|напише|има|оповестим|уведомим|кажем)\s+(?P<body>.+)$",
        r"(?:да\s+му\s+кажеш|да\s+кажеш|кажеш)\s+(?:че\s+)?(?P<body>.+)$",
        r"(?:имейлът|имейла|мейлът|мейла|съдържанието)\s+(?:е|да бъде|да е)\s+(?P<body>.+)$",
        r"(?:body|message|email)\s+(?:is|=|:)\s+(?P<body>.+)$",
        r"(?:кажи му|кажи|напиши)\s+(?:че\s+)?(?P<body>.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return _clean_email_body(match.group("body"))
    email_match = re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, flags=re.IGNORECASE)
    if email_match:
        tail = text[email_match.end() :].strip(" ,.;:-")
        return _clean_email_body(tail)
    return ""


def _clean_email_body(body: str) -> str:
    cleaned = body.strip().strip(" \"'")
    add_personal_note = bool(
        re.search(
            r"(?:и\s+)?добави\s+нещо\s+от\s+себе\s+си\b|add\s+something\s+(?:from\s+yourself|of\s+your\s+own)",
            cleaned,
            flags=re.IGNORECASE,
        )
    )
    cleaned = re.sub(
        r"\s*(?:и\s+)?добави\s+нещо\s+от\s+себе\s+си\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    cleaned = re.sub(
        r"\s*(?:and\s+)?add\s+something\s+(?:from\s+yourself|of\s+your\s+own)\b.*$",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip(" ,.;:-")
    cleaned = re.sub(r",?\s*той\s+знае\b.*$", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"\s+кажи\s+го\s+с\s+[^.?!,]*?\s+тон\s+и\s+кажи\s+че\s+",
        " и ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+кажи\s+го\s+с\s+[^.?!,]*?\s+тон\b", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+и\s+кажи\s+че\s+", " и ", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(
        r"^(?:а\s+)?(?:имейлът|имейла|мейлът|мейла)?\s*(?:е\s+)?(?:да\s+)?(?:му\s+)?(?:кажа|кажеш|кажем)\s+че\s+",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^(?:че|that)\s+", "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\bгит\b", "Git", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"^проекта\b", "проектът", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bперефектно\b", "перфектно", cleaned, flags=re.IGNORECASE)
    if not cleaned:
        return ""
    if _looks_bulgarian(cleaned):
        cleaned = _capitalize_first(cleaned)
    else:
        cleaned = cleaned[0].upper() + cleaned[1:]
    if add_personal_note:
        if cleaned[-1] not in ".!?":
            cleaned += "."
        note = (
            "Добавям и кратка бележка: изпратено е през локалния ми агент."
            if _looks_bulgarian(cleaned)
            else "Sent through a local assistant."
        )
        return f"{cleaned} {note}"
    if cleaned[-1] not in ".!?":
        cleaned += "."
    return cleaned


def _maybe_business_email_body(text: str, body: str) -> str:
    if not _looks_bulgarian(text) or not _email_send_needs_llm_composition(text):
        return body
    core = re.sub(r"\s+", " ", body.strip()).rstrip(".")
    if not core:
        return body
    return (
        "Здравейте,\n\n"
        f"Информирам Ви, че {_lowercase_first(core)}.\n\n"
        "Оставам на разположение при нужда от допълнителна информация.\n\n"
        "С уважение"
    )


def _email_subject_from_text(text: str, body: str) -> str:
    default = "Съобщение" if _looks_bulgarian(text) or _looks_bulgarian(body) else "Message"
    explicit = _explicit_email_subject_from_text(text)
    if explicit:
        return _sanitize_email_subject(explicit, default)
    generated = _generated_email_subject_from_content(text, body)
    if generated:
        return _sanitize_email_subject(generated, default)
    return _email_subject_fallback(body, default)


def _is_email_create_draft_tool(tool: str) -> bool:
    return tool in {"gmail.create_draft", "outlook.create_draft"}


def _local_email_draft(runtime_context: ToolRuntimeContext, provider: str, draft_id: str) -> dict[str, Any]:
    safe_id = "".join(ch for ch in draft_id if ch.isalnum() or ch in {"-", "_"})
    if not safe_id:
        return {}
    path = runtime_context.memory_root.parent / "email-drafts" / provider / f"{safe_id}.json"
    try:
        draft = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return draft if isinstance(draft, dict) else {}


def _email_send_approval_response(
    user_message: str,
    *,
    draft: dict[str, Any],
    send_response: AgentResponse,
) -> AgentResponse:
    draft_id = str(draft.get("draft_id") or (send_response.data or {}).get("draft_id") or "").strip()
    to_values = draft.get("to") if isinstance(draft.get("to"), list) else []
    to = ", ".join(str(item) for item in to_values) or str(draft.get("to") or "")
    subject = str(draft.get("subject") or "").strip()
    body = str(draft.get("body") or "").strip()
    body_preview = body if len(body) <= 1800 else f"{body[:1800].rstrip()}\n[truncated]"
    if _looks_bulgarian(user_message):
        message = (
            f"Подготвих имейла и го записах като локална чернова `{draft_id}`.\n\n"
            f"До: {to}\n"
            f"Тема: {subject or '-'}\n\n"
            f"Съдържание:\n{body_preview or '-'}\n\n"
            "Ако изглежда добре, натисни Approve, за да го изпратя."
        )
    else:
        message = (
            f"Prepared the email as local draft `{draft_id}`.\n\n"
            f"To: {to}\n"
            f"Subject: {subject or '-'}\n\n"
            f"Body:\n{body_preview or '-'}\n\n"
            "If it looks good, approve it and I will send it."
        )
    return AgentResponse(
        status="approval_required",
        message=message,
        data={
            **(send_response.data or {}),
            "draft_id": draft_id,
            "to": to,
            "to_list": to_values or ([to] if to else []),
            "subject": subject,
            "body_preview": body_preview,
        },
    )


def _explicit_email_subject_from_text(text: str) -> str:
    subject_match = re.search(
        r"(?:subject|тема(?:та)?)\s*(?:is|е|:|=)\s*(?P<subject>[^.;\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not subject_match:
        return ""
    subject = subject_match.group("subject")
    if re.search(r"сам\s+избери|choose", subject, flags=re.IGNORECASE):
        return ""
    return subject


def _generated_email_subject_from_content(text: str, body: str) -> str:
    source = _collapse_spaces(f"{text} {body}")
    normalized = _normalize_for_match(source)
    if _looks_bulgarian(source):
        if "проект" in normalized:
            if "маркетинг" in normalized:
                return "Проектът е готов за маркетинг отдела"
            if "git" in normalized:
                return "Проектът е готов и качен в Git"
            if "онлайн" in normalized:
                return "Проектът работи и е онлайн"
            if any(marker in normalized for marker in {"готов", "завършен", "перфект"}):
                return "Проектът е готов"
        if "маркетинг" in normalized:
            return "Готово за маркетинг отдела"
        if "срещ" in normalized:
            if any(marker in normalized for marker in {"мести", "премести", "промен", "нов час"}):
                return "Промяна на срещата"
            return "Относно срещата"
        if "фактур" in normalized:
            return "Относно фактурата"
        if "оферт" in normalized:
            return "Относно офертата"
        if "договор" in normalized:
            return "Относно договора"
        if "напомн" in normalized:
            return "Напомняне"
        if "благодар" in normalized:
            return "Благодарност"
        return ""
    if "project" in normalized:
        if "marketing" in normalized:
            return "Project ready for marketing"
        if "git" in normalized:
            return "Project ready in Git"
        if any(marker in normalized for marker in {"ready", "finished", "complete", "completed"}):
            return "Project ready"
    if "meeting" in normalized:
        if any(marker in normalized for marker in {"reschedule", "moved", "new time", "change"}):
            return "Meeting update"
        return "About the meeting"
    if "invoice" in normalized:
        return "Invoice update"
    if "proposal" in normalized or "offer" in normalized:
        return "Proposal update"
    if "contract" in normalized:
        return "Contract update"
    if "reminder" in normalized:
        return "Reminder"
    if "thank" in normalized:
        return "Thank you"
    return ""


def _email_subject_fallback(body: str, default: str) -> str:
    compact = _subject_fallback_source(body)
    if not compact:
        return default
    return _sanitize_email_subject(compact, default)


def _subject_fallback_source(body: str) -> str:
    compact = _collapse_spaces(body)
    compact = re.sub(r"^здравейте,?\s*", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"^информирам\s+ви,\s+че\s+", "", compact, flags=re.IGNORECASE)
    compact = re.sub(r"^пиша\s+ви,\s+за\s+да\s+", "", compact, flags=re.IGNORECASE)
    compact = re.split(r"\b(?:оставам на разположение|с уважение|поздрави)\b", compact, flags=re.IGNORECASE)[0]
    return compact.strip(" ,.;:!?")


def _sanitize_email_subject(subject: str, default: str, *, max_chars: int = 78) -> str:
    compact = _collapse_spaces(subject).strip(" \"'`.,;:-")
    if not compact:
        return default
    if len(compact) <= max_chars:
        return compact
    truncated = compact[:max_chars].rstrip(" ,.;:-")
    if " " in truncated:
        truncated = truncated.rsplit(" ", 1)[0].rstrip(" ,.;:-")
    return truncated or default


def _collapse_spaces(text: str) -> str:
    return " ".join(str(text).split())


def _capitalize_first(text: str) -> str:
    return text[:1].upper() + text[1:]


def _lowercase_first(text: str) -> str:
    return text[:1].lower() + text[1:]


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


def _multi_step_plan_from_text(text: str) -> tuple[ToolRequest | UnsupportedAction, ...] | None:
    if not _looks_multi_step_request(text):
        return None
    steps = _split_multi_step_text(text)
    if len(steps) < 2:
        return None
    planned: list[ToolRequest | UnsupportedAction] = []
    last_created_path = ""
    for step in steps:
        planned_step, created_path = _plan_single_step(step, last_created_path=last_created_path)
        if planned_step is None:
            continue
        planned.append(planned_step)
        if created_path:
            last_created_path = created_path
    if len(planned) < 2:
        return None
    return tuple(planned)


def _looks_multi_step_request(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(marker in normalized for marker in {"after that", "then", "след това", "после"})


def _split_multi_step_text(text: str) -> list[str]:
    rough = [
        part.strip(" ,.;")
        for part in re.split(r"\b(?:after\s+that|then|след\s+това|после)\b", text, flags=re.IGNORECASE)
        if part.strip(" ,.;")
    ]
    steps: list[str] = []
    for part in rough:
        subparts = re.split(
            r"\s+(?:and|и)\s+(?=(?:execute|run|open|cd|nano|vim|vi|emacs|изпълни|пусни|отвори|влез)\b)",
            part,
            flags=re.IGNORECASE,
        )
        steps.extend(subpart.strip(" ,.;") for subpart in subparts if subpart.strip(" ,.;"))
    return steps


def _plan_single_step(
    step: str,
    *,
    last_created_path: str,
) -> tuple[ToolRequest | UnsupportedAction | None, str]:
    interactive = _interactive_action_from_text(step)
    if interactive is not None:
        return interactive, ""
    mkdir = re.search(r"\bmkdir(?:\s+-p)?\s+(?P<path>[A-Za-z0-9_.\-/]+)", step, flags=re.IGNORECASE)
    if mkdir:
        path = mkdir.group("path").strip(" ,.;\"'")
        command = ["mkdir", path]
        if "-p" in mkdir.group(0).split():
            command = ["mkdir", "-p", path]
        return (
            ToolRequest(
                tool="terminal.run",
                args={"command": command},
                reason="User asked to create a workspace folder as part of a multi-step request.",
            ),
            path,
        )
    workspace_path = _workspace_path_from_step(step, last_created_path=last_created_path)
    if workspace_path:
        return (
            ToolRequest(
                tool="workspace.switch",
                args={"path": workspace_path},
                reason="User asked to move into a folder as part of a multi-step request.",
            ),
            "",
        )
    return None, ""


def _workspace_path_from_step(step: str, *, last_created_path: str) -> str:
    normalized = _normalize_for_match(step)
    if any(phrase in normalized for phrase in {"open the folder", "open this folder", "отвори папката", "отвори тази папка"}):
        return last_created_path
    match = re.search(
        r"^(?:open|cd|go\s+into|enter|switch\s+to|отвори|влез\s+в|иди\s+в)\s+(?P<path>.+)$",
        step.strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        terminal_cd = re.search(
            r"(?:execute|run|exeute|изпълни|пусни)\s+cd\s+(?P<path>.+)$",
            step.strip(),
            flags=re.IGNORECASE,
        )
        if terminal_cd:
            match = terminal_cd
    if not match:
        return ""
    path = str(match.group("path")).strip(" ,.;\"'")
    if path.casefold() in {"the folder", "this folder", "folder", "папката", "тази папка"}:
        return last_created_path
    path = re.sub(r"^(?:folder|directory|папка|директория)\s+", "", path, flags=re.IGNORECASE).strip()
    if re.search(r"https?://|(?:[a-z0-9-]+\.)+[a-z]{2,}", path, flags=re.IGNORECASE):
        return ""
    return path


def _multi_step_approval_message(
    user_message: str,
    first_request: ToolRequest,
    continuations: list[ToolRequest | UnsupportedAction],
) -> str:
    plan = "; ".join(_continuation_plan_for_user(continuations))
    command = first_request.args.get("command")
    command_text = " ".join(str(part) for part in command) if isinstance(command, list) else first_request.tool
    if _looks_bulgarian(user_message):
        return f"Това започва с `{command_text}` и иска approval. След одобрение ще продължа с: {plan}."
    return f"This starts with `{command_text}` and needs approval. After approval I will continue with: {plan}."


def _continuation_plan_for_user(
    steps: list[ToolRequest | UnsupportedAction] | tuple[ToolRequest | UnsupportedAction, ...],
) -> list[str]:
    plan: list[str] = []
    for step in steps:
        if isinstance(step, UnsupportedAction):
            if step.kind == "interactive_editor":
                plan.append(f"skip interactive editor for {step.target}")
            else:
                plan.append(f"unsupported action: {step.kind}")
        elif step.tool == "workspace.switch":
            plan.append(f"switch workspace to {step.args.get('path')}")
        elif step.tool == "files.write":
            plan.append(f"write file {step.args.get('path')}")
        else:
            plan.append(step.tool)
    return plan


def _interactive_terminal_response(text: str) -> AgentResponse | None:
    action = _interactive_action_from_text(text)
    if action is None:
        return None
    return _unsupported_action_response(text, action)


def _interactive_action_from_text(text: str) -> UnsupportedAction | None:
    match = re.search(
        r"\b(?:execute|run|exeute|start|open|изпълни|пусни|стартирай)?\s*(?P<editor>nano|vim|vi|emacs)\s+(?P<target>[^\s;&|]+)",
        text,
        flags=re.IGNORECASE,
    )
    if not match:
        return None
    return UnsupportedAction(kind="interactive_editor", target=match.group("target").strip(" ,.;\"'"))


def _unsupported_action_response(user_message: str, action: UnsupportedAction) -> AgentResponse:
    if action.kind == "interactive_editor":
        if _looks_bulgarian(user_message):
            message = (
                f"Не мога да стартирам interactive editor като nano/vim в dashboard terminal. "
                f"Мога да създам или редактирам `{action.target}` през `files.write`, след approval."
            )
        else:
            message = (
                f"I cannot run interactive editors like nano/vim in the dashboard terminal. "
                f"I can create or edit `{action.target}` through `files.write` after approval."
            )
        return AgentResponse(
            status="ok",
            message=message,
            data={"planner": "deterministic", "unsupported": action.kind, "target": action.target},
        )
    message = (
        "Не мога да изпълня това действие от dashboard-а, но мога да сменя workspace, да пускам allowlisted команди и да работя с файлове през tools."
        if _looks_bulgarian(user_message)
        else "I cannot execute that action from the dashboard, but I can switch workspace, run allowlisted commands, and work with files through tools."
    )
    return AgentResponse(status="ok", message=message, data={"planner": "deterministic", "unsupported": action.kind})


def _tool_selection_failure_message(text: str) -> str:
    if _looks_bulgarian(text):
        return (
            "Не успях да избера надежден инструмент за това. Мога да сменя текущия workspace към папка, "
            "да пускам allowlisted terminal команди там и да създавам/редактирам файлове през file tools с approval."
        )
    return (
        "I could not choose a reliable tool for that. I can switch the current workspace to a folder, "
        "run allowlisted terminal commands there, and create/edit files through file tools with approval."
    )


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
    try:
        TerminalPolicy.from_config(config).validate(list(command))
    except PermissionError:
        return False
    allowed = {
        tuple(str(part) for part in item)
        for item in terminal.get("allowed_commands", [])
        if isinstance(item, list) and item
    }
    return command in allowed


def _local_dev_auto_approval_reason(request: ToolRequest, config: dict[str, Any]) -> str:
    if not local_dev_autonomy_enabled(config):
        return ""
    if request.tool in LOCAL_DEV_AUTO_APPROVED_TOOLS:
        return "local_dev_safe_tool"
    if request.tool == "terminal.run" and _is_local_dev_readonly_terminal_request(request, config):
        return "local_dev_readonly_terminal"
    if request.tool == "terminal.run" and _is_local_dev_workspace_script_terminal_request(request, config):
        return "local_dev_workspace_script"
    return ""


def _planner_decision_summary(plan: Any) -> dict[str, Any]:
    objective = str(getattr(plan, "objective", "") or "").strip()
    summary = list(getattr(plan, "plan_summary", ()) or ())
    if getattr(plan, "tool_request", None) is not None:
        request = plan.tool_request
        data = {
            "action": "tool_call",
            "tool": getattr(request, "tool", ""),
            "args": _redact_step_args_for_response(getattr(request, "args", {}) or {}),
        }
        if objective:
            data["objective"] = objective
        if summary:
            data["plan"] = summary
        return data
    tool_plan = tuple(getattr(plan, "tool_plan", ()) or ())
    if tool_plan:
        data = {
            "action": "multi_tool_plan",
            "tools": [getattr(request, "tool", "") for request in tool_plan],
        }
        if objective:
            data["objective"] = objective
        if summary:
            data["plan"] = summary
        return data
    if getattr(plan, "memory_query", None):
        return {"action": "memory_search"}
    if getattr(plan, "clarification_message", None):
        return {"action": "ask_clarification"}
    return {"action": "answer"}


def _tool_action_title(tool: str, *, phase: str = "run") -> str:
    base = {
        "files.list": "Listing workspace files",
        "files.mkdir": "Creating folder",
        "files.read": "Reading file",
        "files.write": "Writing file",
        "files.write_many": "Writing project files",
        "files.delete": "Deleting file",
        "project.scaffold_one_page_app": "Scaffolding web project",
        "browser.scrape_markdown": "Scraping page to Markdown",
        "browser.open": "Opening page",
        "browser.extract_text": "Extracting page text",
        "terminal.run": "Running terminal command",
        "developer.context": "Analyzing project context",
        "memory.list": "Reading memory index",
        "memory.read": "Reading memory file",
        "memory.write": "Writing memory file",
        "workspace.status": "Checking workspace status",
        "system.list_enabled_tools": "Checking enabled tools",
    }.get(tool, f"Using {tool}")
    if phase == "validate":
        return "Checking safety gates"
    if phase == "observe":
        return f"Completed: {base}"
    return base


def _execution_depth_for_request(text: str) -> str:
    normalized = _normalize_for_match(text)
    deep_markers = {
        "app",
        "application",
        "backend",
        "frontend",
        "full stack",
        "node",
        "node.js",
        "project",
        "react",
        "scaffold",
        "site",
        "website",
        "страница",
        "сайт",
        "уеб",
        "уебстраница",
        "проект",
    }
    if len(text) > 220 or any(marker in normalized for marker in deep_markers):
        return "deep"
    medium_markers = {
        "browser",
        "compare",
        "email",
        "find",
        "scrape",
        "summarize",
        "travel",
        "обобщи",
        "намери",
        "скрейп",
    }
    if any(marker in normalized for marker in medium_markers):
        return "medium"
    return "light"


def _autonomy_should_retry_after_response(response: AgentResponse) -> bool:
    if response.status != "error":
        return False
    data = response.data or {}
    if isinstance(data, dict) and (data.get("emergency_stop") or data.get("approval_id")):
        return False
    message = response.message.casefold()
    blocked_markers = ("secret", ".env", "emergency stop", "approval", "permission")
    return not any(marker in message for marker in blocked_markers)


def _is_local_dev_readonly_terminal_request(request: ToolRequest, config: dict[str, Any]) -> bool:
    raw_command = request.args.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        return False
    command = [str(part) for part in raw_command if str(part)]
    if len(command) != len(raw_command):
        return False
    if _terminal_request_requires_explicit_approval_in_local_dev(request):
        return False
    try:
        policy = TerminalPolicy.from_config(config)
        policy.validate(command)
    except PermissionError:
        return False
    return policy.is_local_dev_readonly_command(command)


def _is_local_dev_workspace_script_terminal_request(request: ToolRequest, config: dict[str, Any]) -> bool:
    raw_command = request.args.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        return False
    command = [str(part) for part in raw_command if str(part)]
    if len(command) != len(raw_command):
        return False
    if _terminal_request_requires_explicit_approval_in_local_dev(request):
        return False
    try:
        policy = TerminalPolicy.from_config(config)
        policy.validate(command)
    except PermissionError:
        return False
    return policy.is_local_dev_workspace_script_command(command)


def _terminal_request_requires_explicit_approval_in_local_dev(request: ToolRequest) -> bool:
    if request.tool != "terminal.run":
        return False
    raw_command = request.args.get("command")
    if not isinstance(raw_command, list) or not raw_command:
        return False
    command = [str(part).casefold() for part in raw_command if str(part)]
    if not command:
        return False
    binary = command[0].rsplit("/", 1)[-1]
    if binary in {
        "apt",
        "apt-get",
        "brew",
        "chmod",
        "chown",
        "docker",
        "docker-compose",
        "kubectl",
        "pip",
        "pip3",
        "rm",
        "scp",
        "ssh",
        "sudo",
    }:
        return True
    if binary in {"npm", "pnpm", "yarn"} and any(
        part in {"add", "install", "remove", "uninstall", "update", "upgrade"}
        for part in command[1:]
    ):
        return True
    if binary in {"python", "python3"} and "-m" in command and "pip" in command and "install" in command:
        return True
    if binary == "git" and len(command) > 1 and command[1] in {
        "checkout",
        "clean",
        "commit",
        "merge",
        "push",
        "rebase",
        "reset",
        "restore",
        "switch",
        "tag",
    }:
        return True
    return False


def _local_dev_file_write_request_from_text(text: str, config: dict[str, Any]) -> ToolRequest | None:
    if not local_dev_autonomy_enabled(config):
        return None
    normalized = _normalize_for_match(text)
    if not any(
        marker in normalized
        for marker in {
            "add text",
            "append text",
            "write text",
            "добави текст",
            "добавиш текст",
            "добави текста",
            "добавиш текста",
            "запиши текст",
        }
    ):
        return None
    path = _extract_file_write_target_path(text)
    content = _extract_file_write_literal_text(text)
    if not path or content is None:
        return None
    append = any(marker in normalized for marker in {"add", "append", "добави", "добавиш"})
    args: dict[str, Any] = {
        "path": path,
        "content": content,
    }
    if append:
        args["append"] = True
        args["append_newline"] = True
    else:
        args["overwrite"] = True
    return ToolRequest(
        tool="files.write",
        args=args,
        reason="Local dev planner recovery for an explicit file text edit request.",
    )


def _extract_file_write_target_path(text: str) -> str:
    matches = [
        match.group(0).strip(" .,!?:;\"'")
        for match in re.finditer(
            r"\b[\w./-]+\.(?:md|txt|json|py|js|ts|tsx|jsx|html|css|yaml|yml|toml)\b",
            text,
            flags=re.IGNORECASE,
        )
    ]
    if not matches:
        return ""
    for candidate in matches:
        if candidate.casefold() in {"readme.md", "README.md".casefold()}:
            return candidate
    return matches[0]


def _extract_file_write_literal_text(text: str) -> str | None:
    quote_match = re.search(r"[\"“„](?P<content>.*?)[\"”]", text)
    if quote_match:
        return quote_match.group("content")
    marker_match = re.search(
        r"(?:add|append|write|добави|добавиш|запиши)\s+(?:text|текст|текста)?\s*[:=-]?\s*(?P<content>.+)$",
        text,
        flags=re.IGNORECASE,
    )
    if not marker_match:
        return None
    content = marker_match.group("content").strip(" .,:;\"'")
    return content or None


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

    requested_service = _service_from_text(recall_text)
    service_ranked: list[MemoryEntry] = []
    if requested_service is not None:
        service_ranked = [
            entry for entry in ranked if _memory_entry_matches_service(entry, requested_service)
        ]
        if not service_ranked and explicit_recall:
            return AgentResponse(
                status="ok",
                message=(
                    f"Не намирам достатъчно сигурен запис за {requested_service} в локалната memory. Няма да връщам адрес за друг service."
                    if bulgarian
                    else f"I do not find a confident local memory entry for {requested_service}. I will not return another service's address."
                ),
                data={"planner": "deterministic"},
            )
        if service_ranked and (_asks_for_address_or_url(recall_text) or _asks_for_port(recall_text)):
            service_answer = _format_service_access_answer(
                requested_service,
                service_ranked,
                entries,
                recall_text,
                bulgarian=bulgarian,
            )
            if service_answer is not None:
                return AgentResponse(
                    status="ok",
                    message=service_answer,
                    data={"planner": "deterministic"},
                )
        if service_ranked:
            ranked = service_ranked

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


SERVICE_ALIASES: dict[str, tuple[str, ...]] = {
    "Immich": ("immich", "имич", "иммич"),
    "Plex": ("plex", "плекс"),
    "Proxmox": ("proxmox", "проксмокс", "proxm"),
    "Grafana": ("grafana", "графана"),
    "AdGuard": ("adguard", "адгард", "адгуард"),
    "Paperless": ("paperless", "пейпърлес"),
    "Homepage": ("homepage", "хоумпейдж"),
}


def _entity_correction_from_text(text: str) -> tuple[str, str] | None:
    normalized = _normalize_for_match(text)
    match = re.search(
        r"\b(?P<wanted>immich|имич|иммич|plex|плекс|proxmox|проксмокс|grafana|графана|adguard|адгард|адгуард)"
        r"\b.{0,40}(?:\b(?:not|не)\b|!=).{0,40}\b"
        r"(?P<wrong>immich|имич|иммич|plex|плекс|proxmox|проксмокс|grafana|графана|adguard|адгард|адгуард)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if match:
        wanted_label = _service_from_text(match.group("wanted"))
        wrong_label = _service_from_text(match.group("wrong"))
        if wanted_label and wrong_label and wanted_label != wrong_label:
            return wanted_label, wrong_label
    reverse = re.search(
        r"\b(?:not|не)\b.{0,20}\b"
        r"(?P<wrong>immich|имич|иммич|plex|плекс|proxmox|проксмокс|grafana|графана|adguard|адгард|адгуард)"
        r"\b.{0,40}\b"
        r"(?P<wanted>immich|имич|иммич|plex|плекс|proxmox|проксмокс|grafana|графана|adguard|адгард|адгуард)\b",
        normalized,
        flags=re.IGNORECASE,
    )
    if reverse:
        wanted_label = _service_from_text(reverse.group("wanted"))
        wrong_label = _service_from_text(reverse.group("wrong"))
        if wanted_label and wrong_label and wanted_label != wrong_label:
            return wanted_label, wrong_label
    return None


def _is_do_not_repeat_error_feedback(text: str) -> bool:
    normalized = _normalize_for_match(text)
    return any(
        marker in normalized
        for marker in {
            "да не се повтаря",
            "не повтаряй",
            "не прави пак",
            "dont repeat",
            "don't repeat",
            "do not repeat",
            "dont make this mistake",
            "don't make this mistake",
        }
    )


def _service_from_text(text: str) -> str | None:
    normalized = _normalize_for_match(text)
    for label, aliases in SERVICE_ALIASES.items():
        if any(alias in normalized for alias in aliases):
            return label
    return None


def _memory_entry_matches_service(entry: MemoryEntry, service: str) -> bool:
    aliases = SERVICE_ALIASES.get(service, (service.casefold(),))
    haystack = _normalize_for_match(f"{entry.path} {entry.text}")
    return any(alias in haystack for alias in aliases)


def _format_service_access_answer(
    service: str,
    service_matches: list[MemoryEntry],
    all_entries: list[MemoryEntry],
    text: str,
    *,
    bulgarian: bool,
) -> str | None:
    if service == "Proxmox":
        return _format_proxmox_access_answer(service_matches, bulgarian=bulgarian)
    urls = _urls_from_entries(service_matches)
    if urls:
        if bulgarian:
            return "\n".join([f"{service} адрес:", *[f"- {url}" for url in urls[:3]]])
        return "\n".join([f"{service} address:", *[f"- {url}" for url in urls[:3]]])
    port = _port_from_entries(service_matches)
    if not port:
        return None
    if _asks_for_port(text) and not _asks_for_address_or_url(text):
        return f"{service} е на порт {port}." if bulgarian else f"{service} is on port {port}."
    lan_ip, tailscale_ip = _machine_ips_from_entries(all_entries)
    suffix = "/web" if service == "Plex" else ""
    if lan_ip or tailscale_ip:
        if bulgarian:
            lines = [f"{service} можеш да отвориш така:"]
            if lan_ip:
                lines.append(f"- LAN: http://{lan_ip}:{port}{suffix}")
            if tailscale_ip:
                lines.append(f"- Tailscale: http://{tailscale_ip}:{port}{suffix}")
            return "\n".join(lines)
        lines = [f"You can open {service} here:"]
        if lan_ip:
            lines.append(f"- LAN: http://{lan_ip}:{port}{suffix}")
        if tailscale_ip:
            lines.append(f"- Tailscale: http://{tailscale_ip}:{port}{suffix}")
        return "\n".join(lines)
    if bulgarian:
        return f"{service} е на порт {port}, но не виждам записан LAN/Tailscale IP в локалната memory."
    return f"{service} is on port {port}, but I do not see a saved LAN/Tailscale IP in local memory."


def _urls_from_entries(entries: list[MemoryEntry]) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for entry in entries:
        for url in re.findall(r"https?://[^\s,)]+", entry.text):
            cleaned = url.rstrip(" .")
            if cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
    return urls


def _port_from_entries(entries: list[MemoryEntry]) -> str | None:
    combined = "\n".join(entry.text for entry in entries)
    match = re.search(r"\b(?:port|ports|порт)[^0-9]{0,80}([0-9]{2,5})\b", combined, flags=re.IGNORECASE)
    if match:
        return match.group(1)
    for number in re.findall(r"\b([0-9]{2,5})\b", combined):
        value = int(number)
        if 1000 <= value <= 65535:
            return number
    return None


def _machine_ips_from_entries(entries: list[MemoryEntry]) -> tuple[str, str]:
    combined = "\n".join(entry.text for entry in entries)
    lan_match = re.search(r"\bLAN IP:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", combined, flags=re.IGNORECASE)
    tailscale_match = re.search(r"\bTailscale IP:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", combined, flags=re.IGNORECASE)
    return (
        lan_match.group(1) if lan_match else "",
        tailscale_match.group(1) if tailscale_match else "",
    )


def _format_memory_recall_answer(matches: list[MemoryEntry], text: str) -> str:
    bulgarian = _looks_bulgarian(text)
    if _is_broad_memory_recall_question(text):
        return _format_broad_memory_answer(matches, bulgarian=bulgarian)
    port_answer = _format_port_answer(matches, text, bulgarian=bulgarian)
    if port_answer and _asks_for_port(text):
        return port_answer
    proxmox_answer = _format_proxmox_access_answer(matches, bulgarian=bulgarian)
    if proxmox_answer and _asks_about_proxmox(text):
        return proxmox_answer
    machine_answer = _format_machine_access_answer(matches, bulgarian=bulgarian)
    if machine_answer and (_asks_for_address_or_url(text) or _asks_about_machine_access(text)):
        return machine_answer
    if _asks_for_address_or_url(text):
        paths = sorted({entry.path for entry in matches})
        if paths:
            path_text = ", ".join(paths[:3])
            return (
                f"Не виждам изрично записан адрес/URL. Намерих свързана memory в: {path_text}."
                if bulgarian
                else f"I do not see an explicit address/URL. I found related memory in: {path_text}."
            )
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


def _format_machine_access_answer(matches: list[MemoryEntry], *, bulgarian: bool) -> str | None:
    combined = "\n".join(entry.text for entry in matches)
    url_match = re.search(r"https?://[^\s,)]+", combined)
    lan_match = re.search(r"\bLAN IP:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", combined, flags=re.IGNORECASE)
    tailscale_match = re.search(r"\bTailscale IP:\s*([0-9]{1,3}(?:\.[0-9]{1,3}){3})", combined, flags=re.IGNORECASE)
    if not url_match and not lan_match and not tailscale_match:
        return None
    if bulgarian:
        lines = []
        if url_match:
            lines.append(f"URL: {url_match.group(0)}")
        if lan_match:
            lines.append(f"LAN IP: {lan_match.group(1)}")
        if tailscale_match:
            lines.append(f"Tailscale IP: {tailscale_match.group(1)}")
        return "\n".join(lines)
    lines = []
    if url_match:
        lines.append(f"URL: {url_match.group(0)}")
    if lan_match:
        lines.append(f"LAN IP: {lan_match.group(1)}")
    if tailscale_match:
        lines.append(f"Tailscale IP: {tailscale_match.group(1)}")
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
        for marker in {
            "адрес",
            "address",
            "url",
            "линк",
            "link",
            "ip",
            "айпи",
            "ай пи",
            "ай пито",
            "https",
            "http",
            "where do i open",
            "къде",
        }
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
        if name.casefold().split(maxsplit=1)[0] in {"asking", "ask", "looking", "trying"}:
            continue
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


def _llm_system_prompt(llm_config: dict[str, Any], key: str) -> str | None:
    prompts = llm_config.get("system_prompts", {})
    if not isinstance(prompts, dict):
        return None
    value = prompts.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _chat_history_allowed_to_cloud(config: dict[str, Any]) -> bool:
    privacy = config.get("privacy", {})
    return bool(isinstance(privacy, dict) and privacy.get("send_chat_history_to_cloud") is True)


def _asks_memory_file_name_list(text: str) -> bool:
    normalized = _normalize_for_match(text)
    mentions_memory = "memory" in normalized or "памет" in normalized
    mentions_markdown_files = any(
        marker in normalized
        for marker in {".md", " markdown", "файл", "файлов", "files", "file"}
    )
    asks_for_names = any(
        marker in normalized
        for marker in {"имен", "какви", "кои", "list", "show", "покажи", "кажи"}
    )
    return mentions_memory and mentions_markdown_files and asks_for_names


def _asks_scraped_downloads_list(text: str) -> bool:
    normalized = _normalize_for_match(text)
    mentions_downloads = any(
        marker in normalized
        for marker in {"download", "downloads", "свален", "свалени", "изтеглен"}
    )
    mentions_scrapes = any(
        marker in normalized
        for marker in {"scrape", "scraped", "scrapefiles", "скрейп", "скрейпнат", "скрейпнати"}
    )
    asks_inventory = any(
        marker in normalized
        for marker in {"какво", "кои", "какви", "имаме", "list", "show", "what"}
    )
    return mentions_downloads and mentions_scrapes and asks_inventory


def _asks_missing_tool_permissions(text: str) -> bool:
    normalized = _normalize_for_match(text)
    mentions_tools = any(
        marker in normalized
        for marker in {"tool", "tools", "тул", "тулове", "инструмент", "инструменти"}
    )
    mentions_permissions = any(
        marker in normalized
        for marker in {"permission", "permissions", "разреш", "право", "достъп"}
    )
    asks_missing = any(
        marker in normalized
        for marker in {"missing", "without", "not granted", "липс", "не сме", "не е", "няма", "не"}
    )
    return mentions_tools and mentions_permissions and asks_missing


def _scraped_download_files(context: ToolRuntimeContext, *, limit: int = 200) -> list[dict[str, Any]]:
    storage_root = downloads_root_from_config(context.config, default_root=context.workspace_root)
    scrape_root = (storage_root / "scrapefiles").resolve(strict=False)
    if not scrape_root.is_dir():
        return []
    items: list[dict[str, Any]] = []
    for path in scrape_root.rglob("*"):
        try:
            resolved = path.resolve(strict=False)
            resolved.relative_to(scrape_root)
        except (OSError, ValueError):
            continue
        if not resolved.is_file() or resolved.suffix.lower() not in {".md", ".markdown"}:
            continue
        stat = resolved.stat()
        try:
            relative_path = resolved.relative_to(storage_root).as_posix()
        except ValueError:
            relative_path = resolved.name
        items.append(
            {
                "path": relative_path,
                "name": resolved.name,
                "size": stat.st_size,
                "modified_at": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
            }
        )
    items.sort(key=lambda item: str(item["modified_at"]), reverse=True)
    return items[:limit]


def _missing_tool_permissions(
    manifests: dict[str, Any],
    permission_context: PermissionContext,
) -> dict[str, list[str]]:
    granted = set(permission_context.granted_permissions)
    missing: dict[str, list[str]] = {}
    for name, manifest in sorted(manifests.items()):
        required = set(getattr(manifest, "permissions", ()))
        missing_permissions = sorted(required - granted)
        if missing_permissions:
            missing[name] = missing_permissions
    return missing


def _enabled_tools_from_config(
    manifests: dict[str, Any],
    config: dict[str, Any],
    permission_context: PermissionContext | None = None,
) -> frozenset[str]:
    overrides = config.get("tools", {})
    enabled: set[str] = set()
    for name, manifest in manifests.items():
        override = overrides.get(name, {})
        is_enabled = bool(getattr(manifest, "default_enabled", False))
        if permission_context is not None and name in permission_context.enabled_tools:
            is_enabled = True
        if isinstance(override, dict) and override.get("enabled") is True:
            is_enabled = True
        if permission_context is not None and name in permission_context.disabled_tools:
            is_enabled = False
        if isinstance(override, dict) and override.get("enabled") is False:
            is_enabled = False
        if local_dev_autonomy_enabled(config) and name in LOCAL_DEV_AUTONOMY_ENABLED_TOOLS:
            is_enabled = True
        if is_enabled:
            enabled.add(name)
    return frozenset(enabled)


def _normalize_planner_tool_request(
    request: ToolRequest,
    user_message: str,
    config: dict[str, Any] | None = None,
) -> ToolRequest:
    args = dict(request.args)
    tool = request.tool
    if config is not None:
        tool = _normalize_planner_email_tool_provider(tool, user_message, config)
    if request.tool in {"browser.open", "browser.scrape_markdown", "browser.extract_text"}:
        url = args.get("url")
        if isinstance(url, str):
            args["url"] = _normalize_browser_url_arg(url)
        if request.tool == "browser.scrape_markdown":
            args.setdefault("instructions", user_message)
            args.setdefault("format", "clean_markdown")
    if request.tool == "files.write" and args.get("content_from_previous_step") is True:
        args.setdefault("overwrite", True)
    return ToolRequest(tool=tool, args=args, reason=request.reason)


def _normalize_planner_email_tool_provider(
    tool: str,
    user_message: str,
    config: dict[str, Any],
) -> str:
    if not tool.startswith(("gmail.", "outlook.")) or "." not in tool:
        return tool
    action = tool.split(".", 1)[1]
    provider = _email_provider_from_text(user_message, config)
    if provider is None:
        return tool
    return f"{provider}.{action}"


def _prepare_tool_request_for_execution(
    request: ToolRequest,
    runtime_context: ToolRuntimeContext,
) -> ToolRequest:
    if not _is_email_send_draft_tool(request.tool):
        return request
    args = dict(request.args)
    if str(args.get("draft_id") or "").strip():
        return request
    requested_provider = request.tool.split(".", 1)[0]
    provider, draft_id = _latest_local_email_draft(runtime_context, requested_provider)
    if not draft_id:
        provider, draft_id = _latest_local_email_draft(runtime_context)
    if not draft_id:
        return request
    args["draft_id"] = draft_id
    tool = request.tool if provider == requested_provider else f"{provider}.send_draft"
    return ToolRequest(tool=tool, args=args, reason=request.reason)


def _is_email_send_draft_tool(tool: str) -> bool:
    return tool in {"gmail.send_draft", "outlook.send_draft"}


def _latest_local_email_draft(
    runtime_context: ToolRuntimeContext,
    provider: str = "",
) -> tuple[str, str]:
    providers = (provider.casefold(),) if provider else ("gmail", "outlook")
    candidates: list[tuple[str, float, str, str]] = []
    for provider_name in providers:
        draft_root = runtime_context.memory_root.parent / "email-drafts" / provider_name
        if not draft_root.is_dir():
            continue
        for path in draft_root.glob("*.json"):
            try:
                draft = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(draft, dict):
                continue
            draft_provider = str(draft.get("provider") or provider_name).casefold()
            if draft_provider != provider_name:
                continue
            if str(draft.get("status") or "draft").casefold() != "draft":
                continue
            draft_id = str(draft.get("draft_id") or path.stem).strip()
            if not draft_id:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                mtime = 0.0
            candidates.append((str(draft.get("created_at") or ""), mtime, provider_name, draft_id))
    if not candidates:
        return "", ""
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    latest = candidates[0]
    return latest[2], latest[3]


def _resolve_previous_step_content(
    request: ToolRequest,
    *,
    previous_response: AgentResponse,
    chain_state: dict[str, Any] | None = None,
    runtime_context: ToolRuntimeContext,
) -> ToolRequest:
    args = dict(request.args)
    chain_state = chain_state if chain_state is not None else {}
    path_match = args.get("path_from_previous_step_match")
    if isinstance(path_match, str) and path_match.strip():
        selected_path = _path_from_previous_list_result(previous_response, path_match.strip())
        if not selected_path:
            selected_path = _path_from_known_runtime_locations(path_match.strip(), runtime_context)
        if not selected_path:
            raise ValueError(f"Could not find a previous listed file matching: {path_match.strip()}")
        args["path"] = selected_path
        args.pop("path_from_previous_step_match", None)
        chain_state["selected_path"] = selected_path
    if args.get("path_from_selected_step") is True:
        selected_path = str(chain_state.get("selected_path") or "").strip()
        if not selected_path:
            raise ValueError("No previously selected file path is available.")
        args["path"] = selected_path
        args.pop("path_from_selected_step", None)
    if args.get("body_from_previous_step") is True:
        content = _content_from_previous_tool_response(previous_response)
        if content is None:
            raise ValueError("Previous tool result did not include text content for the draft body.")
        args.pop("body_from_previous_step", None)
        args["body"] = _cap_chained_content(content, runtime_context.config)
    if args.get("draft_id_from_previous_step") is True:
        draft_id = _draft_id_from_previous_tool_response(previous_response)
        if not draft_id:
            raise ValueError("Previous tool result did not include a draft_id.")
        args.pop("draft_id_from_previous_step", None)
        args["draft_id"] = draft_id
    if args.get("content_from_previous_step") is not True:
        return ToolRequest(tool=request.tool, args=args, reason=request.reason)
    if request.tool != "files.write":
        raise ValueError("content_from_previous_step is only supported for files.write.")
    content = _content_from_previous_tool_response(previous_response)
    if content is None:
        raise ValueError("Previous tool result did not include text content to write.")
    content = _cap_chained_content(content, runtime_context.config)
    args.pop("content_from_previous_step", None)
    args["content"] = content
    args.setdefault("overwrite", True)
    return ToolRequest(tool=request.tool, args=args, reason=request.reason)


def _content_from_previous_tool_response(response: AgentResponse) -> str | None:
    data = response.data if isinstance(response.data, dict) else {}
    for key in ("markdown", "content", "text"):
        value = data.get(key)
        if isinstance(value, str):
            return value
    return None


def _draft_id_from_previous_tool_response(response: AgentResponse) -> str:
    data = response.data if isinstance(response.data, dict) else {}
    draft_id = data.get("draft_id")
    return draft_id if isinstance(draft_id, str) else ""


def _path_from_previous_list_result(response: AgentResponse, match_text: str) -> str:
    data = response.data if isinstance(response.data, dict) else {}
    entries = data.get("entries")
    if not isinstance(entries, list):
        return ""
    normalized_match = _normalize_for_match(match_text)
    best_path = ""
    best_score = -1
    for entry in entries:
        if not isinstance(entry, dict) or bool(entry.get("is_dir")):
            continue
        name = str(entry.get("name") or "")
        relative_path = str(entry.get("relative_path") or "")
        path = str(entry.get("path") or relative_path)
        haystack = _normalize_for_match(f"{name} {relative_path} {path}")
        score = _match_score(normalized_match, haystack)
        if score > best_score:
            best_score = score
            best_path = path
    return best_path if best_score > 0 else ""


def _path_from_known_runtime_locations(match_text: str, runtime_context: ToolRuntimeContext) -> str:
    manager = WorkspaceManager.from_config(
        runtime_context.config,
        fallback_workspace=runtime_context.workspace_root,
    )
    roots = _candidate_file_search_roots(runtime_context, manager)
    normalized_match = _normalize_for_match(match_text)
    best_path = ""
    best_score = -1
    inspected = 0
    for root in roots:
        try:
            resolved_root = manager.validate_user_path(str(root))
        except (OSError, WorkspaceError, ValueError):
            continue
        if resolved_root.is_file():
            candidates = [resolved_root]
        elif resolved_root.is_dir():
            candidates = _iter_candidate_files(resolved_root, manager)
        else:
            continue
        for candidate in candidates:
            inspected += 1
            if inspected > 3000:
                return best_path if best_score > 0 else ""
            try:
                if manager.is_secret_path(candidate) or not candidate.is_file():
                    continue
                relative = candidate.relative_to(resolved_root).as_posix()
            except (OSError, ValueError):
                relative = candidate.name
            haystack = _normalize_for_match(f"{candidate.name} {relative} {candidate}")
            score = _match_score(normalized_match, haystack)
            if candidate.name.casefold() == Path(match_text).name.casefold():
                score += 200
            if score > best_score:
                best_score = score
                best_path = str(candidate)
    return best_path if best_score > 0 else ""


def _candidate_file_search_roots(
    runtime_context: ToolRuntimeContext,
    manager: WorkspaceManager,
) -> list[Path]:
    roots: list[Path] = []
    downloads = downloads_root_from_config(
        runtime_context.config,
        default_root=runtime_context.workspace_root,
    )
    roots.extend([downloads / "scrapefiles", downloads, manager.current_workspace])
    roots.extend(manager.allowed_roots)
    unique: list[Path] = []
    for root in roots:
        resolved = root.expanduser().resolve(strict=False)
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _iter_candidate_files(root: Path, manager: WorkspaceManager) -> list[Path]:
    files: list[Path] = []
    excluded_dirs = {".git", ".venv", "__pycache__", "node_modules", "dist", "build"}
    for current_root, dirnames, filenames in os.walk(root):
        root_path = Path(current_root)
        kept_dirs: list[str] = []
        for dirname in sorted(dirnames):
            path = root_path / dirname
            if dirname in excluded_dirs or manager.is_secret_path(path):
                continue
            kept_dirs.append(dirname)
        dirnames[:] = kept_dirs
        for filename in sorted(filenames):
            path = root_path / filename
            if not manager.is_secret_path(path):
                files.append(path)
            if len(files) >= 3000:
                return files
    return files


def _match_score(needle: str, haystack: str) -> int:
    if not needle or not haystack:
        return 0
    if needle in haystack:
        return 100 + len(needle)
    score = 0
    for token in needle.split():
        if len(token) >= 3 and token in haystack:
            score += 10 + len(token)
    return score


def _update_chain_state(chain_state: dict[str, Any], request: ToolRequest, response: AgentResponse) -> None:
    if response.status != "ok":
        return
    data = response.data if isinstance(response.data, dict) else {}
    if request.tool == "files.read":
        path = data.get("path") or request.args.get("path")
        if isinstance(path, str) and path:
            chain_state["selected_path"] = path


def _cap_chained_content(content: str, config: dict[str, Any]) -> str:
    files_config = config.get("files", {})
    browser_config = config.get("browser", {})
    raw_cap = (
        files_config.get("max_chained_content_chars")
        if isinstance(files_config, dict)
        else None
    )
    if raw_cap is None and isinstance(browser_config, dict):
        raw_cap = browser_config.get("max_text_chars")
    try:
        cap = int(raw_cap if raw_cap is not None else 12_000)
    except (TypeError, ValueError):
        cap = 12_000
    cap = max(500, min(cap, 200_000))
    if len(content) <= cap:
        return content
    return content[:cap].rstrip() + "\n\n[content truncated]\n"


def _redact_step_args_for_response(args: dict[str, Any]) -> dict[str, Any]:
    redacted = dict(args)
    for key in ("content", "body"):
        if key in redacted:
            content = str(redacted[key])
            redacted[f"{key}_preview"] = content[:500]
            redacted[f"{key}_chars"] = len(content)
            redacted.pop(key, None)
    return redacted


def _safe_tool_result_summary(data: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in data.items():
        if key in {"content", "markdown", "text", "body", "source_markdown"}:
            text = str(value)
            safe[f"{key}_chars"] = len(text)
            safe[f"{key}_preview"] = redact_text(text[:500])
            continue
        if key in {"entries", "files", "items", "messages", "summary"} and isinstance(value, list):
            safe[key] = value[:10]
            safe[f"{key}_count"] = len(value)
            continue
        if isinstance(value, str):
            safe[key] = redact_text(value[:500])
        else:
            safe[key] = value
    return safe


def _normalize_browser_url_arg(url: str) -> str:
    value = url.strip()
    if not value or "://" in value:
        return value
    if value.startswith("www.") or ("." in value and " " not in value):
        return f"https://{value}"
    return value


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
        and not _chat_history_allowed_to_cloud(runtime_context.config)
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
    if request.tool == "profile.update":
        return "I can update your local profile after you approve it."
    if request.tool == "files.write":
        return "I can create or edit that file after you approve it."
    if request.tool == "files.mkdir":
        return "I can create that folder after you approve it."
    if request.tool == "files.write_many":
        return "I can write those files after you approve it."
    if request.tool == "project.scaffold_one_page_app":
        return "I can scaffold that project after you approve it."
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
    if request.tool == "profile.update":
        return "Profile updated."
    if request.tool == "files.read":
        return "File read."
    if request.tool == "files.list":
        return "Files listed."
    if request.tool == "files.write":
        return "File written."
    if request.tool == "files.mkdir":
        return "Folder created."
    if request.tool == "files.write_many":
        return "Files written."
    if request.tool == "project.scaffold_one_page_app":
        return "Project scaffolded."
    if request.tool == "files.delete":
        return "File deleted."
    if request.tool == "workspace.switch":
        return "Workspace switched."
    if request.tool == "workspace.status":
        return "Workspace status loaded."
    if request.tool == "developer.context":
        return "Developer workspace context loaded."
    if request.tool == "browser.scrape_markdown":
        return "Scraped page saved as Markdown."
    if request.tool == "reminders.create":
        return "Reminder saved. I will notify you in Telegram when it is due if Telegram is configured."
    if request.tool == "reminders.complete":
        return "Reminder completed."
    if request.tool == "calendar.create_event":
        return "Saved to local calendar store. Active reminder notifications are not implemented yet."
    if request.tool.startswith(("gmail.", "outlook.")):
        return "Email tool executed."
    return "Tool executed."


def _safe_tool_synthesis_fallback(response: AgentResponse, user_message: str) -> AgentResponse:
    data = response.data or {}
    bulgarian = _looks_bulgarian(user_message)
    if isinstance(data, dict) and data.get("tool") == "multi_step":
        steps = data.get("steps") if isinstance(data.get("steps"), list) else []
        paths: list[str] = []
        tools: list[str] = []
        for step in steps:
            if not isinstance(step, dict):
                continue
            tool = str(step.get("tool") or "").strip()
            if tool:
                tools.append(tool)
            step_data = step.get("data")
            if not isinstance(step_data, dict):
                continue
            path = str(step_data.get("path") or step_data.get("relative_path") or "").strip()
            if path:
                paths.append(path)
        if paths:
            preview = ", ".join(paths[:4])
            if len(paths) > 4:
                preview = f"{preview}, +{len(paths) - 4} more"
            message = (
                f"Прочетох {len(paths)} context файла ({preview}), но не успях да ги синтезирам в кратък отговор. "
                "Няма да изливам суровото memory съдържание в чата."
                if bulgarian
                else f"I read {len(paths)} context file(s) ({preview}), but could not synthesize a concise answer. "
                "I will not dump raw memory content into chat."
            )
        else:
            message = (
                "Изпълних context стъпките, но не успях да ги синтезирам в кратък отговор."
                if bulgarian
                else "I ran the context steps, but could not synthesize a concise answer."
            )
        return AgentResponse(
            status=response.status,
            message=message,
            data={
                "planner": "deterministic",
                "source": "multi_step_context",
                "tools": tools,
                "paths": paths,
            },
        )
    path = data.get("path") if isinstance(data, dict) else None
    if isinstance(path, str) and path:
        if bulgarian:
            message = (
                f"Прочетох {path}, но не успях да го превърна в кратък отговор. "
                "Няма да изливам целия memory файл в чата."
            )
        else:
            message = (
                f"I read {path}, but could not turn it into a concise answer. "
                "I will not dump the whole memory file into chat."
            )
        return AgentResponse(
            status=response.status,
            message=message,
            data={"planner": "deterministic", "source": "memory.read", "path": path},
        )
    if isinstance(data, dict) and "files" in data:
        files = data.get("files")
        count = len(files) if isinstance(files, list) else 0
        message = (
            f"Намерих {count} memory файла, но не успях да синтезирам кратък отговор."
            if bulgarian
            else f"I found {count} memory files, but could not synthesize a concise answer."
        )
        return AgentResponse(
            status=response.status,
            message=message,
            data={"planner": "deterministic", "source": "memory.list", "file_count": count},
        )
    return response


def _explicit_raw_memory_request(normalized_text: str) -> bool:
    return any(
        phrase in normalized_text
        for phrase in {
            "list memory",
            "show memory",
            "show memory files",
            "memory files",
            "local memory files",
            "show raw memory",
            "print the memory file",
            "raw memory file",
            "покажи файловете",
            "списък с памет",
            "покажи суровия memory файл",
            "суровия memory файл",
            "принтирай memory файла",
        }
    )


def _memory_list_user_response(user_message: str, data: dict[str, Any]) -> AgentResponse:
    files = data.get("files") if isinstance(data.get("files"), list) else []
    lines = [str(path) for path in files]
    if not lines:
        message = (
            "Няма записани memory файлове."
            if _looks_bulgarian(user_message)
            else "No memory files are saved."
        )
    else:
        intro = (
            f"Memory файлове ({len(lines)}):"
            if _looks_bulgarian(user_message)
            else f"Memory files ({len(lines)}):"
        )
        message = "\n".join([intro, *[f"- {path}" for path in lines]])
    return AgentResponse(
        status="ok",
        message=message,
        data={"tool": "memory.list", "files": files, "count": len(files)},
    )


def _memory_read_user_response(user_message: str, data: dict[str, Any]) -> AgentResponse:
    path = str(data.get("path") or "")
    content = str(data.get("content") or "")
    truncated = bool(data.get("truncated", False))
    suffix = "\n\n[truncated]" if truncated else ""
    prefix = f"Прочетох memory файла `{path}`:" if _looks_bulgarian(user_message) else f"Read memory file `{path}`:"
    return AgentResponse(
        status="ok",
        message=f"{prefix}\n\n{content}{suffix}".strip(),
        data={"tool": "memory.read", "path": path, "truncated": truncated},
    )


def _browser_scrape_user_response(user_message: str, data: dict[str, Any]) -> AgentResponse:
    bulgarian = _looks_bulgarian(user_message)
    path = str(data.get("relative_path") or data.get("path") or "")
    final_url = str(data.get("final_url") or data.get("url") or "")
    mode = str(data.get("scrape_mode") or "raw_page")
    requested_mode = str(data.get("requested_mode") or mode)
    count = int(data.get("extracted_count") or 0)
    items = data.get("items") if isinstance(data.get("items"), list) else []
    if bool(data.get("raw_fallback")):
        message = (
            f"Targeted extraction failed for {final_url}; saved a raw page fallback to {path}."
            if not bulgarian
            else f"Targeted extraction не намери достатъчно статии от {final_url}; запазих raw fallback в {path}."
        )
        return AgentResponse(status="ok", message=message, data={**data, "tool": "browser.scrape_markdown"})
    if mode == "targeted" and count:
        intro = (
            f"Extracted {count} article(s) from {final_url} and saved them to {path}:"
            if not bulgarian
            else f"Извлякох {count} статии от {final_url} и ги запазих в {path}:"
        )
        lines = [intro]
        for index, item in enumerate(items[:count], start=1):
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            if title:
                lines.append(f"{index}. {title}")
        return AgentResponse(
            status="ok",
            message="\n".join(lines),
            data={**data, "tool": "browser.scrape_markdown"},
        )
    message = (
        f"Scraped page in {requested_mode} mode and saved Markdown to {path}."
        if not bulgarian
        else f"Скрейпнах страницата в режим {requested_mode} и запазих Markdown в {path}."
    )
    return AgentResponse(status="ok", message=message, data={**data, "tool": "browser.scrape_markdown"})


def _terminal_user_response(user_message: str, request: ToolRequest, data: dict[str, Any]) -> AgentResponse:
    command = request.args.get("command")
    command_text = " ".join(str(part) for part in command) if isinstance(command, list) else "command"
    returncode = int(data.get("returncode") or 0)
    stdout = str(data.get("stdout") or "").strip()
    stderr = str(data.get("stderr") or "").strip()
    output = stdout or stderr
    if len(output) > 4000:
        output = output[:4000].rstrip() + "\n[output truncated]"
    bulgarian = _looks_bulgarian(user_message)
    if returncode == 0:
        prefix = f"Изпълних `{command_text}`." if bulgarian else f"Ran `{command_text}`."
    else:
        prefix = (
            f"`{command_text}` приключи с код {returncode}."
            if bulgarian
            else f"`{command_text}` exited with code {returncode}."
        )
    message = f"{prefix}\n\n{output}" if output else prefix
    return AgentResponse(
        status="ok",
        message=message,
        data={
            **data,
            "tool": "terminal.run",
            "command": command,
            "returncode": returncode,
        },
    )


def _email_user_response(user_message: str, request: ToolRequest, data: dict[str, Any]) -> AgentResponse:
    del user_message
    provider = str(data.get("provider") or request.tool.split(".", 1)[0]).title()
    action = request.tool.split(".", 1)[1] if "." in request.tool else request.tool
    if data.get("status") == "connector_not_configured":
        return AgentResponse(
            status="not_configured",
            message=str(data.get("message") or "Email connector is not configured."),
            data=data,
        )
    if action == "search":
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        lines = [f"{provider}: found {len(messages)} message(s)."]
        lines.extend(_email_message_line(message) for message in messages[:10] if isinstance(message, dict))
        return AgentResponse(status="ok", message="\n".join(lines), data=data)
    if action == "summarize_inbox":
        summaries = data.get("summary") if isinstance(data.get("summary"), list) else []
        lines = [f"{provider}: summarized {len(summaries)} message(s)."]
        lines.extend(_email_summary_line(item) for item in summaries[:10] if isinstance(item, dict))
        return AgentResponse(status="ok", message="\n".join(lines), data=data)
    if action == "read_thread":
        messages = data.get("messages") if isinstance(data.get("messages"), list) else []
        if not messages or not isinstance(messages[0], dict):
            return AgentResponse(status="ok", message=f"{provider}: message loaded.", data=data)
        message = messages[0]
        body = str(message.get("body") or "")
        if len(body) > 3000:
            body = f"{body[:3000]}\n...[truncated]"
        text = (
            f"{provider}: {message.get('subject', '(no subject)')}\n"
            f"From: {message.get('from', '-')}\n"
            f"Date: {message.get('date', '-')}\n\n"
            f"{body}"
        )
        return AgentResponse(status="ok", message=text.strip(), data=data)
    if action in {"create_draft", "reply_draft"}:
        to = ", ".join(str(item) for item in data.get("to", []))
        return AgentResponse(
            status="ok",
            message=f"{provider}: created local draft {data.get('draft_id')} for {to}.",
            data=data,
        )
    if action == "send_draft":
        return AgentResponse(status="ok", message=f"{provider}: sent draft {data.get('draft_id')}.", data=data)
    if action == "archive":
        return AgentResponse(status="ok", message=f"{provider}: archived {len(data.get('message_ids', []))} message(s).", data=data)
    if action == "label":
        return AgentResponse(status="ok", message=f"{provider}: applied labels to {len(data.get('message_ids', []))} message(s).", data=data)
    return AgentResponse(status="ok", message=f"{provider}: email action completed.", data=data)


def _email_message_line(message: dict[str, Any]) -> str:
    return (
        f"- {message.get('id')}: {message.get('subject', '(no subject)')} "
        f"from {message.get('from', '-')} | {message.get('date', '-')}"
    )


def _email_summary_line(item: dict[str, Any]) -> str:
    return (
        f"- {item.get('id')}: {item.get('subject', '(no subject)')} "
        f"from {item.get('from', '-')} - {item.get('summary', '')}"
    )


def _file_read_user_response(user_message: str, data: dict[str, Any]) -> AgentResponse:
    path = str(data.get("path") or "")
    content = str(data.get("content") or "")
    truncated = bool(data.get("truncated", False))
    suffix = "\n\n[truncated]" if truncated else ""
    prefix = f"Прочетох `{path}`:" if _looks_bulgarian(user_message) else f"Read `{path}`:"
    return AgentResponse(
        status="ok",
        message=f"{prefix}\n\n{content}{suffix}",
        data={"tool": "files.read", "path": path, "truncated": truncated},
    )


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
