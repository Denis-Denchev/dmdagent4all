from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any

from dmdagent4all.audit import AuditEvent, AuditStore
from dmdagent4all.agent.planner import LLMPlanner, PlannerError
from dmdagent4all.config import save_config
from dmdagent4all.memory import MemoryManager
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
        identity_update = _handle_identity_update(stripped, self.runtime_context)
        if identity_update is not None:
            return identity_update
        memory_update = _memory_write_request_from_text(stripped, self.runtime_context)
        if memory_update is not None:
            return self.handle_tool_request(memory_update)
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
                current_time=datetime.now().astimezone().isoformat(timespec="seconds"),
                timezone_name=datetime.now().astimezone().tzname() or "",
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

    def handle_tool_request(self, request: ToolRequest) -> AgentResponse:
        decision = self.permission_engine.evaluate(request, self._permission_context())
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
    if "memory files" in normalized or "local memory" in normalized:
        if any(word in normalized for word in {"show", "list", "see", "display"}):
            return ToolRequest(
                tool="memory.list",
                args={},
                reason="User requested local memory files.",
            )
    return None


def _memory_write_request_from_text(
    text: str,
    runtime_context: ToolRuntimeContext,
) -> ToolRequest | None:
    fact = _extract_memory_fact(text)
    if fact is None:
        return None
    path = "facts/personal.md"
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
                "source": "chat",
                "confidence": "high",
            },
        },
        reason="User asked the agent to remember a personal fact.",
    )


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
        "как се казвам",
        "името ми",
    }:
        if user_name:
            if _looks_bulgarian(text):
                return f"Ти си {user_name}."
            return f"You are {user_name}."
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

    user_name = _extract_name(
        text,
        [
            r"^my name is\s+(.+)$",
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
) -> None:
    config = runtime_context.config
    setup = config.setdefault("setup", {})
    if agent_name is not None:
        setup["agent_name"] = agent_name
    if user_name is not None:
        setup["user_name"] = user_name
    setup["completed"] = True

    if runtime_context.config_path is not None:
        save_config(config, runtime_context.config_path)

    profile = _profile_from_config(config)
    MemoryManager(runtime_context.memory_root).write(
        "profile.md",
        "\n".join(
            [
                f"User name: {profile['user_name'] or 'not set'}",
                f"Assistant name: {profile['agent_name']}",
                f"Preferred response language: {profile['preferred_language']}",
            ]
        ),
        metadata={
            "type": "profile",
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
    return default


def _denial_message(request: ToolRequest, default: str) -> str:
    if default.startswith("Tool is disabled:"):
        return (
            f"{default}. Enable it first from the terminal chat with /tool enable {request.tool}. "
            "If the tool needs a permission, open /permissions."
        )
    return default


def _tool_success_message(request: ToolRequest) -> str:
    if request.tool == "memory.write":
        return "Saved to memory."
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
