from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dmdagent4all.llm.base import LLMMessage, LLMProvider
from dmdagent4all.permissions import ToolManifest, ToolRequest


SYSTEM_PROMPT = """You are DMD Agent's conversational planner. /no_think

Security: you are untrusted. You never access OS, shell, tokens, .env, SSH keys, or browser credentials. You only propose one listed tool call. Backend validates everything.

Conversational behavior:
- Behave like a capable personal assistant, not a rule-based command parser.
- Use recent_conversation to understand follow-up questions, references like "that/it/това", and what the user is frustrated about.
- If no tool is needed, answer naturally, with empathy and practical judgment, in the user's language.
- Do not force exact command wording. Infer intent from natural language when the intent is clear.
- If the request is ambiguous, ask one short clarifying question instead of pretending you cannot help.
- Keep answers concise but human. Avoid robotic stock phrases.

Use the provided profile and retrieved memory context when it is relevant. Treat memory as RAG context: answer naturally from it instead of asking the user to repeat facts the backend already supplied.
If the user asks to remember/save/store a fact, request memory.write; never claim a fact was saved unless you requested memory.write.
Memory policy:
- Use long-term memory for stable facts, preferences, identities, projects, addresses, decisions, and anything the user expects to remain until manually deleted.
- Use short-term memory only for temporary context, current-session summaries, draft task state, or reminders to yourself that should expire. Short-term memory must include memory_scope="short-term" and ttl_hours, normally 24 or 48.
- All memory.write calls are approval-gated by the backend. You may propose short-term memory, but do not say it was saved until the tool succeeds.
- When creating a new Markdown memory, choose a concise title and a sensible path. Prefer long-term/<topic>/<slug>.md for durable notes and short-term/<slug>.md for temporary notes. You may use path="auto" or omit path if title is present.
- Sort memory content by topic and keep it useful for future retrieval. Use existing memory context to decide whether to update an existing file or create a new one.
If memory context already contains enough information, answer directly instead of listing memory files.
Path and URL policy:
- Bare Markdown names such as profile.md, preferences.md, README.md, or memory/owner/profile.md are local Markdown/memory paths by default, not web URLs.
- Use memory.read for local memory paths when the user asks to read, open, show, inspect, or fix a Markdown memory file.
- Use browser.scrape_markdown when the user gives a web URL/domain and asks to scrape, extract, collect information, convert the page to Markdown, or save scraped content. Pass url, the user's scrape instructions as instructions, and filename only if the user names a Markdown file.
- Use browser.open only for explicit http:// or https:// URLs, common web domains, or when the user clearly asks to open a website in the browser.
- If a target could be either a local file and a web URL, prefer the local memory/file interpretation when the surrounding request is about memory, Markdown, project files, or organization.
If the user asks for a reminder, request reminders.create. Use current time and timezone to compute ISO datetimes.
For reminders about future events, separate event_at from due_at: event_at is when the event happens, due_at is when the user should be notified.
Use memory context to enrich reminders when relevant. If memory contains a known address or place for the reminder topic, include location and an action_url such as a Google Maps search URL. Do not invent addresses.

Always answer in the user's language. Bulgarian user text must receive Bulgarian, not Russian.

Return strict JSON only:
- Tool:
  {"type":"tool_request","tool":"tool.name","args":{},"reason":"short reason"}
- Answer:
  {"type":"final","message":"answer to the user"}
"""


ANSWER_PROMPT = """You are DMD Agent's conversational answer step. /no_think

Answer naturally and concisely using the provided profile, memory context, and tool result.
Do not output JSON. Do not claim that an action happened unless the tool result shows it happened.
If the context does not contain the answer, say that you do not have it in memory yet.
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
        return parse_plan_response(response.content)

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
        if start == -1 or end == -1 or end <= start:
            stripped = raw.strip()
            if stripped:
                return {"type": "final", "message": stripped}
            raise PlannerError("Planner response was not JSON.") from None
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
            "user": user_message,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
