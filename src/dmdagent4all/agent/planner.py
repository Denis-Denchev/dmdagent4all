from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from dmdagent4all.llm.base import LLMMessage, LLMProvider
from dmdagent4all.permissions import ToolManifest, ToolRequest


SYSTEM_PROMPT = """You are Jarvis, the user's personal assistant and everyday chat companion. /no_think

Act naturally. If the user is just talking, chatting, asking a general knowledge question, or asking for advice, answer directly from your own knowledge and the provided context. Be friendly, practical, and concise.

If the user asks you to do something that needs the computer, local memory, files, browser, reminders, calendar, terminal, or an integration, choose the best available tool. The backend will validate safety, permissions, paths, secrets, and approvals before anything executes.

For developer/coding work:
- Use developer.context when the user asks to inspect a project, understand code, prepare a coding task, or needs project context.
- Use files.read for a specific non-secret file read.
- Use files.list to inspect an allowed workspace/downloads folder before choosing among files.
- Use files.write when the user asks to create/edit/overwrite a text/code file, including natural requests like "put this text in README.md" or "create test.md with 12345".
- Use files.delete only when the user explicitly wants deletion or a move that removes the original; deletion requires backend approval.
- Use terminal.run for explicit terminal commands. Put commands as an array of parts, for example {"command":["ls"]}.
- Never invent a result after choosing a tool. Return only the tool_request JSON and let the backend execute or ask approval.

For email:
- Use gmail.* tools for Gmail and outlook.* tools for Outlook.
- Search/read/summarize are read-only. Draft tools create local drafts. send_draft sends and always requires approval.
- Do not include email bodies in final answers unless the user explicitly asked to read the message.

For browser work:
- If the user asks to scrape, extract, download, collect information from, or convert a web page to Markdown, use browser.scrape_markdown.
- If the user asks to scrape a page and place/write/save the results into a file, return a multi_tool_plan: first browser.scrape_markdown, then files.write with {"path":"target file","content_from_previous_step":true}.
- For targeted article/news requests, include args like {"mode":"targeted","content_type":"articles","limit":3,"format":"clean_markdown"}.
- For full-page dumps, include {"mode":"raw_page"}.
- If the user asks to open, visit, or inspect a web page, use browser.open.
- If the user gives a domain without a scheme in a browser request, put https:// in front of it.
- Explicit browser/tool intents must not be answered from memory.

For file workflows:
- If the user asks for a file from Downloads or another mounted folder, check agent_context.workspace.mounted_roots and allowed_roots.
- If the folder is not mounted/allowed, answer with the missing configuration instead of guessing paths.
- For "find the file about X", plan files.list first, then files.read with {"path_from_previous_step_match":"X"}.
- To copy/write the selected file into another file, follow files.read with files.write and {"content_from_previous_step":true}.
- To move the original, add files.delete with {"path_from_selected_step":true}; backend approval is required.

For email workflows:
- Check agent_context.email and enabled tools before planning email work.
- To send a file by email, read the file, create a draft using body_from_previous_step, then call send_draft with draft_id_from_previous_step. Sending requires backend approval.
- If no email connector is configured/enabled, explain what must be configured.

For memory:
- If the user asks about saved personal/project/service facts and the answer is not already obvious from provided memory context, use memory_search.
- Do not use memory_search for explicit browser, file, terminal, email, reminder, or workspace actions.

Use recent_conversation and memory context when they help. Answer in the user's language. Do not use emoji.
The current user message is authoritative. Ignore unrelated prior tasks, examples, benchmark questions, and stale repair text.
Use agent_context as the source of truth for identity, provider/model, tools, permissions, workspace, memory files, pending approvals, and recent tool results. Do not claim to be a provider/model that is not in agent_context.runtime.

Return exactly one JSON object:
- For normal conversation:
  {"action":"answer","message":"your answer"}
- For one tool action:
  {"action":"tool_call","tool":"tool.name","args":{},"reason":"why this tool is useful"}
- For a short multi-step tool plan:
  {"action":"multi_tool_plan","steps":[{"tool":"tool.name","args":{},"reason":"step reason"}],"reason":"why this plan is useful"}
- To ask a clarification:
  {"action":"ask_clarification","message":"your question"}
- To search local memory:
  {"action":"memory_search","query":"what to look up","reason":"why memory is needed"}
"""


ANSWER_PROMPT = """You are Jarvis, the user's personal assistant. /no_think

Turn the tool result into a natural, useful answer in the user's language. Be concise and friendly. Do not use emoji. If the tool failed or lacks the answer, say that plainly.
"""

CHAT_PROMPT = """You are Jarvis, the user's local desktop assistant and chat companion. /no_think

Answer naturally in the user's language. Do not return JSON. Do not mention planner internals. Do not use emoji.
"""

REPAIR_PROMPT = """Return valid JSON only. Repair the planner output into exactly one of these shapes:
{"action":"answer","message":"answer"}
{"action":"tool_call","tool":"tool.name","args":{},"reason":"short reason"}
{"action":"multi_tool_plan","steps":[{"tool":"tool.name","args":{},"reason":"step reason"}],"reason":"short reason"}
{"action":"ask_clarification","message":"question"}
{"action":"memory_search","query":"query","reason":"short reason"}
Do not preserve unrelated benchmark/task text. If the user asked for a tool action, return the tool action schema instead of prose.
"""


STRICT_PLANNER_PROMPT = SYSTEM_PROMPT + """

Your previous response was rejected as unrelated or underspecified for the current user message.
Return only one valid JSON object matching the schema. For explicit scrape/open/file/terminal requests, choose tool_call or multi_tool_plan unless a required value is truly missing.
"""


_TOOL_INTENT_MARKERS = (
    "approval",
    "browser",
    "cat",
    "collect",
    "convert",
    "create",
    "delete",
    "download",
    "edit",
    "execute",
    "extract",
    "file",
    "ls",
    "markdown",
    "open",
    "permission",
    "place",
    "pwd",
    "read",
    "run",
    "save",
    "scrape",
    "scraping",
    "terminal",
    "tool",
    "visit",
    "write",
    "запази",
    "изпълни",
    "отвори",
    "прочети",
    "пусни",
    "скрейп",
)


@dataclass(frozen=True)
class PlanResult:
    tool_request: ToolRequest | None = None
    tool_plan: tuple[ToolRequest, ...] = ()
    final_message: str | None = None
    clarification_message: str | None = None
    memory_query: str | None = None
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
        agent_context: dict[str, Any] | None = None,
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
        planning_prompt = _build_planning_prompt(
            user_message=user_message,
            manifests=manifests,
            profile=profile or {},
            memory_context=memory_context,
            conversation_context=conversation_context,
            agent_context=agent_context or {},
            enabled_tools=enabled_tools,
            response_language=response_language,
            current_time=current_time,
            timezone_name=timezone_name,
        )
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=system_prompt or SYSTEM_PROMPT),
                LLMMessage(role="user", content=planning_prompt),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            think=think,
        )
        result = self._parse_or_repair_plan_response(
            response.content,
            repair_max_tokens=repair_max_tokens,
            think=think,
        )
        if not is_low_relevance_plan(result, user_message):
            return result

        retry_response = self.provider.chat(
            [
                LLMMessage(role="system", content=STRICT_PLANNER_PROMPT),
                LLMMessage(role="user", content=planning_prompt),
            ],
            max_tokens=max_tokens,
            temperature=0.0,
            think=think,
        )
        retry_result = self._parse_or_repair_plan_response(
            retry_response.content,
            repair_max_tokens=repair_max_tokens,
            think=think,
        )
        if is_low_relevance_plan(retry_result, user_message):
            raise PlannerError("Planner returned an unrelated response for an explicit tool request.")
        return retry_result

    def chat(
        self,
        *,
        user_message: str,
        profile: dict[str, str] | None = None,
        memory_context: str = "",
        conversation_context: str = "",
        agent_context: dict[str, Any] | None = None,
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
                            "agent_context": agent_context or {},
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
        agent_context: dict[str, Any] | None = None,
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
                            "agent_context": agent_context or {},
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

    def _parse_or_repair_plan_response(
        self,
        raw: str,
        *,
        repair_max_tokens: int,
        think: bool,
    ) -> PlanResult:
        try:
            return parse_plan_response(raw)
        except PlannerError as exc:
            repaired = self._repair_plan_response(
                raw=raw,
                max_tokens=repair_max_tokens,
                think=think,
            )
            try:
                return parse_plan_response(repaired)
            except PlannerError:
                raise exc from None


def parse_plan_response(raw: str) -> PlanResult:
    payload = _load_json(raw)
    plan_type = _plan_action(payload)
    if plan_type in {"answer", "final"}:
        message = str(payload.get("message", "")).strip()
        if not message:
            raise PlannerError("Planner returned an empty final message.")
        return PlanResult(final_message=message, raw=raw)

    if plan_type == "ask_clarification":
        message = str(payload.get("message", "")).strip()
        if not message:
            raise PlannerError("Planner returned an empty clarification message.")
        return PlanResult(clarification_message=message, raw=raw)

    if plan_type == "memory_search":
        query = str(payload.get("query") or payload.get("message") or "").strip()
        if not query:
            raise PlannerError("Planner returned memory_search without a query.")
        return PlanResult(memory_query=query, raw=raw)

    if plan_type == "multi_tool_plan":
        raw_steps = payload.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            raise PlannerError("Planner multi_tool_plan requires a non-empty steps array.")
        steps: list[ToolRequest] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                raise PlannerError("Planner multi_tool_plan steps must be objects.")
            steps.append(_tool_request_from_payload(step, default_reason=str(step.get("reason", "")).strip()))
        return PlanResult(tool_plan=tuple(steps), raw=raw)

    if plan_type in {"tool_request", "tool", "tool_call", "function_call"} or (
        isinstance(payload.get("tool"), str) and not plan_type
    ):
        return PlanResult(
            tool_request=_tool_request_from_payload(payload, default_reason=str(payload.get("reason", "")).strip()),
            raw=raw,
        )

    raise PlannerError("Planner response action must be answer, tool_call, multi_tool_plan, ask_clarification, or memory_search.")


def is_low_relevance_plan(result: PlanResult, user_message: str) -> bool:
    message = (result.clarification_message or result.final_message or "").strip()
    if not message:
        return False
    if not _looks_like_explicit_tool_request(user_message):
        return False
    return not _message_overlaps_tool_request(message, user_message)


def _plan_action(payload: dict[str, Any]) -> str:
    action = payload.get("action")
    if isinstance(action, str) and action.strip():
        return action.strip()
    plan_type = payload.get("type")
    if isinstance(plan_type, str) and plan_type.strip():
        return plan_type.strip()
    return ""


def _tool_request_from_payload(payload: dict[str, Any], *, default_reason: str = "") -> ToolRequest:
    action = _plan_action(payload)
    tool = str(payload.get("tool", "")).strip()
    if not tool and isinstance(payload.get("name"), str):
        tool = str(payload.get("name", "")).strip()
    if not tool:
        raise PlannerError("Planner returned a tool request without a tool name.")
    args = (
        payload.get("arguments", {})
        if action == "function_call"
        else payload.get("args", {})
    )
    if not isinstance(args, dict) and isinstance(payload.get("arguments"), dict):
        args = payload.get("arguments", {})
    if not isinstance(args, dict):
        raise PlannerError("Planner tool request args must be an object.")
    return ToolRequest(
        tool=tool,
        args=args,
        reason=str(payload.get("reason") or default_reason or "").strip(),
    )


def _looks_like_explicit_tool_request(user_message: str) -> bool:
    normalized = user_message.casefold()
    has_action = any(marker in normalized for marker in _TOOL_INTENT_MARKERS)
    if not has_action:
        return False
    return bool(_request_relevance_terms(user_message))


def _message_overlaps_tool_request(message: str, user_message: str) -> bool:
    normalized_message = message.casefold()
    for term in _request_relevance_terms(user_message):
        if term and term.casefold() in normalized_message:
            return True
    return False


def _request_relevance_terms(user_message: str) -> tuple[str, ...]:
    terms: list[str] = []
    normalized = user_message.casefold()
    for marker in _TOOL_INTENT_MARKERS:
        if marker in normalized:
            terms.append(marker)
    terms.extend(
        match.group(0).strip(" .,!?:;\"'").casefold()
        for match in re.finditer(r"https?://[^\s]+|\b(?:[a-z0-9-]+\.)+[a-z]{2,}(?:/[^\s]*)?", user_message, flags=re.IGNORECASE)
    )
    terms.extend(
        match.group(0).strip(" .,!?:;\"'").casefold()
        for match in re.finditer(r"\b[\w.-]+\.(?:md|txt|json|py|js|ts|tsx|jsx|html|css|yaml|yml|toml)\b", user_message, flags=re.IGNORECASE)
    )
    command_match = re.search(r"\b(?:execute|run|type|пусни|изпълни)\s+([a-z][\w.-]*)\b", user_message, flags=re.IGNORECASE)
    if command_match:
        terms.append(command_match.group(1).casefold())
    unique: list[str] = []
    for term in terms:
        if len(term) < 2 or term in unique:
            continue
        unique.append(term)
    return tuple(unique)


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
    if "type" not in loaded and "action" not in loaded and isinstance(loaded.get("message"), str):
        loaded["action"] = "answer"
    return loaded


def _build_planning_prompt(
    *,
    user_message: str,
    manifests: dict[str, ToolManifest],
    profile: dict[str, str],
    memory_context: str,
    conversation_context: str,
    agent_context: dict[str, Any],
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
            "agent_context": agent_context,
            "tools": tool_rows,
            "routing_hints": [
                "Natural file edit/create requests should become files.write with path, content, and overwrite=true only when replacing existing content is intended.",
                "Folder inspection and file selection requests should become files.list before files.read/files.write.",
                "Use path_from_previous_step_match to read/delete a file chosen from a previous files.list result.",
                "Use body_from_previous_step for email drafts based on a previous files.read result and draft_id_from_previous_step for send_draft after create_draft.",
                "Developer project inspection requests should become developer.context before detailed coding advice.",
                "Explicit terminal commands should become terminal.run with command as an array.",
                "Browser scrape/extract/collect/download-to-Markdown requests should become browser.scrape_markdown with url and instructions.",
                "When scraping first/top/latest N articles, set mode=targeted, content_type=articles, limit=N, and format=clean_markdown.",
                "Browser open/visit requests should become browser.open with url.",
                "Saved-fact/service/address questions may become memory_search, but memory must never preempt explicit browser, terminal, email, file, reminder, or workspace actions.",
                "The backend validates workspace paths, secrets, approvals, and destructive operations.",
            ],
            "user": user_message,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
