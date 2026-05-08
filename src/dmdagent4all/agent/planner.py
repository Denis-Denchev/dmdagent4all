from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dmdagent4all.llm.base import LLMMessage, LLMProvider
from dmdagent4all.permissions import ToolManifest, ToolRequest


SYSTEM_PROMPT = """You are Jarvis, the user's personal assistant and everyday chat companion. /no_think

Act naturally. If the user is just talking, chatting, asking a general knowledge question, or asking for advice, answer directly from your own knowledge and the provided context. Be friendly, practical, and concise.

If the user asks you to do something that needs the computer, local memory, files, browser, reminders, calendar, terminal, or an integration, choose the best available tool and request exactly one tool call. The backend will validate safety, permissions, paths, secrets, and approvals before anything executes.

For developer/coding work:
- Use developer.context when the user asks to inspect a project, understand code, prepare a coding task, or needs project context.
- Use files.read for a specific non-secret file read.
- Use files.write when the user asks to create/edit/overwrite a text/code file, including natural requests like "put this text in README.md" or "create test.md with 12345".
- Use terminal.run for explicit terminal commands. Put commands as an array of parts, for example {"command":["ls"]}.
- Never invent a result after choosing a tool. Return only the tool_request JSON and let the backend execute or ask approval.

For email:
- Use gmail.* tools for Gmail and outlook.* tools for Outlook.
- Search/read/summarize are read-only. Draft tools create local drafts. send_draft sends and always requires approval.
- Do not include email bodies in final answers unless the user explicitly asked to read the message.

Use recent_conversation and memory context when they help. Answer in the user's language. Do not use emoji.

Return JSON:
- For normal conversation:
  {"type":"final","message":"your answer"}
- For an action:
  {"type":"tool_request","tool":"tool.name","args":{},"reason":"why this tool is useful"}
"""


ANSWER_PROMPT = """You are Jarvis, the user's personal assistant. /no_think

Turn the tool result into a natural, useful answer in the user's language. Be concise and friendly. Do not use emoji. If the tool failed or lacks the answer, say that plainly.
"""

CHAT_PROMPT = """You are Jarvis, the user's local desktop assistant and chat companion. /no_think

Answer naturally in the user's language. Do not return JSON. Do not mention planner internals. Do not use emoji.
"""

REPAIR_PROMPT = """Return valid JSON only. Repair the planner output into exactly one of these shapes:
{"type":"final","message":"answer"}
{"type":"tool_request","tool":"tool.name","args":{},"reason":"short reason"}
"""


@dataclass(frozen=True)
class PlanResult:
    tool_request: ToolRequest | None = None
    final_message: str | None = None
    raw: str = ""


class PlannerError(RuntimeError):
    pass


class LLMPlanner:
    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider

    def plan(
        self,
        *,
        user_message: str,
        manifests: dict[str, ToolManifest],
        profile: dict[str, str] | None = None,
        memory_context: str = "",
        conversation_context: str = "",
        enabled_tools: set[str] | frozenset[str] | None = None,
        response_language: str = "auto",
        current_time: str = "",
        timezone_name: str = "",
        max_tokens: int = 192,
        repair_max_tokens: int = 256,
        temperature: float = 0.0,
        think: bool = False,
        system_prompt: str | None = None,
    ) -> PlanResult:
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=system_prompt or SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=_build_planning_prompt(
                        user_message=user_message,
                        manifests=manifests,
                        profile=profile or {},
                        memory_context=memory_context,
                        conversation_context=conversation_context,
                        enabled_tools=enabled_tools,
                        response_language=response_language,
                        current_time=current_time,
                        timezone_name=timezone_name,
                    ),
                ),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            think=think,
        )
        try:
            return parse_plan_response(response.content)
        except PlannerError as exc:
            repaired = self._repair_plan_response(
                raw=response.content,
                max_tokens=repair_max_tokens,
                think=think,
            )
            try:
                return parse_plan_response(repaired)
            except PlannerError:
                raise exc from None

    def chat(
        self,
        *,
        user_message: str,
        profile: dict[str, str] | None = None,
        memory_context: str = "",
        conversation_context: str = "",
        response_language: str = "auto",
        max_tokens: int = 1024,
        temperature: float = 0.4,
        think: bool = False,
        system_prompt: str | None = None,
    ) -> str:
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=system_prompt or CHAT_PROMPT),
                LLMMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "lang": response_language,
                            "profile": {
                                "assistant_name": (profile or {}).get("agent_name", "DMD Agent"),
                                "user_name": (profile or {}).get("user_name", ""),
                            },
                            "memory": memory_context,
                            "recent_conversation": conversation_context,
                            "user": user_message,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                ),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            think=think,
        )
        return response.content.strip()

    def answer(
        self,
        *,
        user_message: str,
        profile: dict[str, str] | None = None,
        memory_context: str = "",
        conversation_context: str = "",
        tool_result: dict[str, Any] | None = None,
        response_language: str = "auto",
        max_tokens: int = 384,
        temperature: float = 0.2,
        think: bool = False,
        system_prompt: str | None = None,
    ) -> str:
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=system_prompt or ANSWER_PROMPT),
                LLMMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "lang": response_language,
                            "profile": {
                                "assistant_name": (profile or {}).get("agent_name", "DMD Agent"),
                                "user_name": (profile or {}).get("user_name", ""),
                            },
                            "memory": memory_context,
                            "recent_conversation": conversation_context,
                            "tool_result": tool_result or {},
                            "user": user_message,
                        },
                        ensure_ascii=True,
                        separators=(",", ":"),
                    ),
                ),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            think=think,
        )
        return response.content.strip()

    def _repair_plan_response(self, *, raw: str, max_tokens: int, think: bool) -> str:
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=REPAIR_PROMPT),
                LLMMessage(role="user", content=raw),
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            think=think,
        )
        return response.content.strip()


def parse_plan_response(raw: str) -> PlanResult:
    payload = _load_json(raw)
    plan_type = payload.get("type")
    if plan_type == "final":
        message = str(payload.get("message", "")).strip()
        if not message:
            raise PlannerError("Planner returned an empty final message.")
        return PlanResult(final_message=message, raw=raw)

    if plan_type in {"tool_request", "tool", "tool_call", "function_call"} or (
        isinstance(payload.get("tool"), str) and not plan_type
    ):
        tool = str(payload.get("tool", "")).strip()
        if not tool and isinstance(payload.get("name"), str):
            tool = str(payload.get("name", "")).strip()
        if not tool:
            raise PlannerError("Planner returned a tool request without a tool name.")
        args = (
            payload.get("arguments", {})
            if plan_type == "function_call"
            else payload.get("args", {})
        )
        if not isinstance(args, dict) and isinstance(payload.get("arguments"), dict):
            args = payload.get("arguments", {})
        if not isinstance(args, dict):
            raise PlannerError("Planner tool request args must be an object.")
        return PlanResult(
            tool_request=ToolRequest(
                tool=tool,
                args=args,
                reason=str(payload.get("reason", "")).strip(),
            ),
            raw=raw,
        )

    raise PlannerError("Planner response type must be final or tool_request.")


def _load_json(raw: str) -> dict[str, Any]:
    try:
        loaded = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1:
            stripped = raw.strip()
            if stripped:
                return {"type": "final", "message": stripped}
            raise PlannerError("Planner response was not JSON.") from None
        if end == -1 or end <= start:
            raise PlannerError("Planner response JSON could not be parsed.") from None
        try:
            loaded = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PlannerError(f"Planner response JSON could not be parsed: {exc}") from exc

    if not isinstance(loaded, dict):
        raise PlannerError("Planner response must be a JSON object.")
    if "type" not in loaded and isinstance(loaded.get("message"), str):
        loaded["type"] = "final"
    return loaded


def _build_planning_prompt(
    *,
    user_message: str,
    manifests: dict[str, ToolManifest],
    profile: dict[str, str],
    memory_context: str,
    conversation_context: str,
    enabled_tools: set[str] | frozenset[str] | None,
    response_language: str,
    current_time: str,
    timezone_name: str,
) -> str:
    tool_rows: list[dict[str, Any]] = []
    effective_enabled = set(enabled_tools or ())
    for manifest in manifests.values():
        enabled = (
            manifest.name in effective_enabled
            if enabled_tools is not None
            else manifest.default_enabled
        )
        tool_rows.append(
            {
                "name": manifest.name,
                "description": manifest.description,
                "risk": int(manifest.risk),
                "enabled": enabled,
                "approval_required": manifest.approval_required,
                "permissions": list(manifest.permissions),
                "args": manifest.argument_schema,
            }
        )
    return json.dumps(
        {
            "lang": response_language,
            "now": current_time,
            "timezone": timezone_name,
            "profile": {
                "assistant_name": profile.get("agent_name", "DMD Agent"),
                "user_name": profile.get("user_name", ""),
            },
            "memory": memory_context,
            "recent_conversation": conversation_context,
            "tools": tool_rows,
            "routing_hints": [
                "Natural file edit/create requests should become files.write with path, content, and overwrite=true only when replacing existing content is intended.",
                "Developer project inspection requests should become developer.context before detailed coding advice.",
                "Explicit terminal commands should become terminal.run with command as an array.",
                "The backend validates workspace paths, secrets, approvals, and destructive operations.",
            ],
            "user": user_message,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
