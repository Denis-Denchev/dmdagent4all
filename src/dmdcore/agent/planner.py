from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from dmdcore.autonomy import log_local_dev_autonomy
from dmdcore.llm.base import LLMMessage, LLMProvider
from dmdcore.permissions import ToolManifest, ToolRequest


SYSTEM_PROMPT = """You are Jarvis, the user's personal assistant and everyday chat companion. /no_think

Act naturally. If the user is just talking, chatting, asking a general knowledge question, or asking for advice, answer directly from your own knowledge and the provided context. Be friendly, practical, and concise.

If the user asks you to do something that needs the computer, local memory, files, browser, reminders, calendar, terminal, or an integration, choose the best available tool. The backend will validate safety, permissions, paths, secrets, and approvals before anything executes.

For developer/coding work:
- Use developer.context when the user asks to inspect a project, understand code, prepare a coding task, or needs project context.
- Use files.read for a specific non-secret file read.
- Use files.list to inspect an allowed workspace/downloads folder before choosing among files.
- Use files.mkdir instead of terminal.run mkdir for safe folder creation.
- Use files.write when the user asks to create/edit/overwrite a text/code file, including natural requests like "put this text in README.md" or "create test.md with 12345".
- Use files.write with {"append":true} when the user asks to add/append text to an existing file.
- Use files.write_many for compact multi-file scaffolding when you know the exact files and content.
- Use files.delete only when the user explicitly wants deletion or a move that removes the original; deletion requires backend approval.
- Use project.scaffold_one_page_app for portfolio, landing page, one-page website, React + Node, frontend/backend, or business-card site scaffolds when the user gives high-level page details.
- Use terminal.run for explicit terminal commands. Put commands as an array of parts, for example {"command":["ls"]}.
- Never claim that a file was created, edited, appended, moved, copied, or overwritten unless you are summarizing an actual tool result. For requested file/project work, include executable ACTIONS, not prose-only progress text.

For local scripts in LOCAL_DEV_AUTONOMY:
- If the user asks you to create a script/program, write the script with files.write in the active workspace.
- If the user asks to execute or test the script, follow with terminal.run using an interpreter command array such as {"command":["python3","script.py"]}.
- If a previous turn established a local script for a future user question and the user now asks that matching question, use terminal.run to execute that workspace script instead of answering from memory.
- Script execution is still backend-gated: workspace path validation, secret blocking, timeout, no shell=True, emergency stop, and command policy remain active.

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
- If the user names a scraped/downloaded file, especially one from downloads/scrapefiles or recent conversation, inspect agent_context.workspace.downloads_path/scrapefiles before searching the active workspace.
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

For autonomous memory growth (scribe mode):
- After choosing your action, scan the current user message for new durable facts about the user, their projects, preferences, places, people, or context that should outlive this conversation.
- If you find any, append a single MEMORY_WRITES block at the very end of your response (after ACTIONS, or as the only addition to a plain answer):

MEMORY_WRITES: [{"slug":"house-build-project","type":"project","confidence":"high","description":"Denis-built wooden house in Gorna Malina","body":"- Location: село Горна Малина (~30km east of Sofia)\n- Material: wood\n- Mode: DIY (Denis builds himself)"}]

Rules:
- The block is a JSON array on a single line, starting with literal `MEMORY_WRITES:`.
- Each item: slug (kebab-case), type (user|feedback|project|reference), confidence (high|medium|low), description (one short sentence), body (markdown — bullet points preferred).
- Only include facts that appeared in THIS user turn and are not already in the memory context. Do not echo old facts.
- Never write placeholders ("unknown", "TBD", "not set", "TBD"). If a field is unknown, omit it entirely.
- confidence=high only if the user stated the fact explicitly. Use medium for reasonable inferences. Skip low.
- If no new durable fact appeared this turn, omit the MEMORY_WRITES block entirely.
- The MEMORY_WRITES block is a side-channel — your main reply must still be a normal answer or ACTIONS plan as usual.

CRITICAL distinction — memory.write tool vs MEMORY_WRITES side-channel:
- The `memory.write` TOOL is ONLY for cases where the user explicitly asks you to save a specific block of text to memory — verbs like "save this", "write this into memory", "create a memory file with X", "запомни това като файл", "запиши го в паметта като...".
- When the user simply mentions a fact about themselves, their projects, places, decisions, preferences, plans — even with phrases like "I decided X", "my house will be in Y", "remember", "запомни", "реших че X", "къщата ми ще е X" — do NOT use memory.write tool. Use the MEMORY_WRITES side-channel instead. The scribe will save it silently.
- Your main response in these cases should be a normal conversational answer (no ACTIONS block). Just chat naturally and append MEMORY_WRITES at the end. The user expects you to acknowledge the fact like a friend, not run a tool that needs approval.
- Never wrap a casual fact-mention in an ACTIONS/memory.write call. That triggers an approval prompt the user does NOT want for ambient facts.

Use recent_conversation and memory context when they help. Answer in the user's language. Do not use emoji.
The current user message is authoritative. Ignore unrelated prior tasks, examples, benchmark questions, and stale repair text.
Use agent_context as the source of truth for identity, provider/model, tools, permissions, workspace, memory files, pending approvals, and recent tool results. Do not claim to be a provider/model that is not in agent_context.runtime.

Use a tolerant action protocol. You may answer naturally for normal conversation.
For executable work, keep reasoning separate from actions and include an ACTIONS block at the end:

OBJECTIVE: short user-facing objective
PLAN:
- short semantic step
ACTIONS:
- tool: files.mkdir
  args: {"path":"test","parents":true}
  reason: Create the project folder.
- tool: project.scaffold_one_page_app
  args: {"path":"test","owner_name":"Denis Denchev","role":"AI Developer","theme":"developer tech dark","project_summary":"Portfolio page about DMD Agent work","include_backend":true,"overwrite":true}
  reason: Scaffold the requested one-page React + Node site.

JSON is also accepted, but do not depend on fenced JSON. If you use JSON, return one normal object. Do not mix hidden reasoning into tool args.
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


STRICT_PLANNER_PROMPT = SYSTEM_PROMPT


_TOOL_INTENT_MARKERS = (
    "approval",
    "append",
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
    "вземи",
    "вземеш",
    "добави",
    "добавиш",
    "запази",
    "запиши",
    "изпълни",
    "копирай",
    "копираш",
    "отвори",
    "премести",
    "преместиш",
    "прочети",
    "пусни",
    "сложи",
    "сложиш",
    "скрейп",
)


@dataclass(frozen=True)
class MemoryWriteIntent:
    slug: str
    body: str
    confidence: str = "medium"
    memory_type: str = "project"
    description: str = ""
    path: str = ""


@dataclass(frozen=True)
class PlanResult:
    tool_request: ToolRequest | None = None
    tool_plan: tuple[ToolRequest, ...] = ()
    final_message: str | None = None
    clarification_message: str | None = None
    memory_query: str | None = None
    objective: str = ""
    plan_summary: tuple[str, ...] = ()
    memory_writes: tuple[MemoryWriteIntent, ...] = ()
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
        context_detail: str = "compact",
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
            context_detail=context_detail,
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
        del repair_max_tokens
        return parse_plan_response(response.content)

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
        del repair_max_tokens, think
        return parse_plan_response(raw)


def parse_plan_response(raw: str) -> PlanResult:
    action_block = _parse_action_protocol(raw)
    if action_block is not None:
        return action_block

    block_writes = _parse_memory_writes_block(raw)
    payload = _load_json(raw)
    if payload is None:
        return PlanResult(final_message=raw.strip(), memory_writes=block_writes, raw=raw)
    payload_writes = _memory_writes_from_payload(payload) if isinstance(payload, dict) else ()
    writes = payload_writes or block_writes
    if isinstance(payload, list):
        steps = [
            request
            for item in payload
            if isinstance(item, dict) and (item.get("tool") or item.get("name"))
            for request in [_tool_request_from_payload_or_none(item, default_reason=str(item.get("reason", "")).strip())]
            if request is not None
        ]
        if steps:
            return PlanResult(tool_plan=tuple(steps), memory_writes=writes, raw=raw)
        return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)
    if not isinstance(payload, dict):
        return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)
    plan_type = _plan_action(payload)
    if plan_type in {"answer", "final"}:
        message = str(payload.get("message", "")).strip()
        if not message:
            return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)
        return PlanResult(final_message=message, memory_writes=writes, raw=raw)

    if plan_type == "ask_clarification":
        message = str(payload.get("message", "")).strip()
        if not message:
            return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)
        return PlanResult(clarification_message=message, memory_writes=writes, raw=raw)

    if plan_type == "memory_search":
        query = str(payload.get("query") or payload.get("message") or "").strip()
        if not query:
            return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)
        return PlanResult(memory_query=query, memory_writes=writes, raw=raw)

    if plan_type == "multi_tool_plan":
        raw_steps = payload.get("steps", [])
        if not isinstance(raw_steps, list) or not raw_steps:
            return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)
        steps: list[ToolRequest] = []
        for step in raw_steps:
            if not isinstance(step, dict):
                continue
            request = _tool_request_from_payload_or_none(step, default_reason=str(step.get("reason", "")).strip())
            if request is not None:
                steps.append(request)
        if steps:
            return PlanResult(tool_plan=tuple(steps), memory_writes=writes, raw=raw)
        return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)

    if plan_type in {"tool_request", "tool", "tool_call", "function_call"} or (
        isinstance(payload.get("tool"), str) and not plan_type
    ):
        request = _tool_request_from_payload_or_none(payload, default_reason=str(payload.get("reason", "")).strip())
        if request is not None:
            return PlanResult(tool_request=request, memory_writes=writes, raw=raw)

    if isinstance(payload.get("message"), str) and str(payload.get("message")).strip():
        return PlanResult(final_message=str(payload.get("message")).strip(), memory_writes=writes, raw=raw)
    return PlanResult(final_message=raw.strip(), memory_writes=writes, raw=raw)


def is_low_relevance_plan(result: PlanResult, user_message: str) -> bool:
    message = (result.clarification_message or result.final_message or "").strip()
    if not message:
        return False
    if result.final_message and _looks_like_explicit_file_mutation_request(user_message):
        return True
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


def _tool_request_from_payload_or_none(payload: dict[str, Any], *, default_reason: str = "") -> ToolRequest | None:
    try:
        return _tool_request_from_payload(payload, default_reason=default_reason)
    except PlannerError:
        return None


def _parse_action_protocol(raw: str) -> PlanResult | None:
    stripped = raw.strip()
    if not stripped:
        return PlanResult(final_message="", raw=raw)
    objective = _prefixed_line(stripped, "OBJECTIVE")
    summary = tuple(_plan_lines(stripped))
    requests: list[ToolRequest] = []
    lines = stripped.splitlines()
    for index, line in enumerate(lines):
        parsed = _request_from_action_line(line, following="\n".join(lines[index + 1 : index + 6]))
        if parsed is not None:
            requests.append(parsed)
    requests.extend(_requests_from_yaml_tool_blocks(stripped))
    requests = _dedupe_requests(requests)
    writes = _parse_memory_writes_block(stripped)
    if not requests:
        if writes:
            return PlanResult(
                final_message=_strip_memory_writes_block(stripped),
                objective=objective,
                plan_summary=summary,
                memory_writes=writes,
                raw=raw,
            )
        return None
    if len(requests) == 1:
        return PlanResult(
            tool_request=requests[0],
            objective=objective,
            plan_summary=summary,
            memory_writes=writes,
            raw=raw,
        )
    return PlanResult(
        tool_plan=tuple(requests),
        objective=objective,
        plan_summary=summary,
        memory_writes=writes,
        raw=raw,
    )


def _parse_memory_writes_block(raw: str) -> tuple[MemoryWriteIntent, ...]:
    match = re.search(r"(?im)^\s*MEMORY_WRITES\s*:\s*(\[.*?\])\s*$", raw, flags=re.DOTALL)
    if match is None:
        return ()
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return ()
    if not isinstance(payload, list):
        return ()
    return _memory_writes_from_list(payload)


def _strip_memory_writes_block(raw: str) -> str:
    return re.sub(r"(?im)^\s*MEMORY_WRITES\s*:\s*\[.*?\]\s*$", "", raw, flags=re.DOTALL).strip()


def _memory_writes_from_payload(payload: dict[str, Any]) -> tuple[MemoryWriteIntent, ...]:
    items = payload.get("memory_writes")
    if not isinstance(items, list):
        return ()
    return _memory_writes_from_list(items)


def _memory_writes_from_list(items: list[Any]) -> tuple[MemoryWriteIntent, ...]:
    result: list[MemoryWriteIntent] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "").strip()
        body = str(item.get("body") or "").strip()
        if not slug or not body:
            continue
        confidence = str(item.get("confidence") or "medium").strip().lower()
        if confidence not in {"high", "medium", "low"}:
            confidence = "medium"
        memory_type = str(item.get("type") or item.get("memory_type") or "project").strip().lower()
        if memory_type not in {"user", "feedback", "project", "reference"}:
            memory_type = "project"
        description = str(item.get("description") or "").strip()
        path = str(item.get("path") or "").strip()
        result.append(
            MemoryWriteIntent(
                slug=_slug_for_memory(slug),
                body=body,
                confidence=confidence,
                memory_type=memory_type,
                description=description,
                path=path,
            )
        )
    return tuple(result)


def _slug_for_memory(value: str) -> str:
    slug = re.sub(r"[^\w]+", "-", value.lower(), flags=re.UNICODE).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug[:80].strip("-") or "memory"


def _request_from_action_line(line: str, *, following: str) -> ToolRequest | None:
    match = re.match(
        r"^\s*(?:[-*]\s*)?(?:@?tool|action)\s*:?\s+([a-zA-Z_][\w.-]*\.[\w.-]+)\s*(.*)$",
        line,
        flags=re.IGNORECASE,
    )
    if match is None:
        return None
    tool = match.group(1).strip()
    rest = match.group(2).strip()
    args = _first_json_object(rest) or _args_from_following_lines(following) or {}
    reason = _reason_from_text(following)
    return ToolRequest(tool=tool, args=args, reason=reason)


def _requests_from_yaml_tool_blocks(raw: str) -> list[ToolRequest]:
    requests: list[ToolRequest] = []
    pattern = re.compile(
        r"(?ims)^\s*[-*]\s*tool\s*:\s*([a-zA-Z_][\w.-]*\.[\w.-]+)\s*$"
        r"(?P<body>.*?)(?=^\s*[-*]\s*tool\s*:|\Z)"
    )
    for match in pattern.finditer(raw):
        body = match.group("body") or ""
        args = _args_from_following_lines(body) or {}
        reason = _reason_from_text(body)
        requests.append(ToolRequest(tool=match.group(1).strip(), args=args, reason=reason))
    return requests


def _args_from_following_lines(text: str) -> dict[str, Any] | None:
    args_match = re.search(r"(?ims)^\s*args\s*:\s*(.+)$", text)
    if args_match is None:
        return _first_json_object(text)
    return _first_json_object(args_match.group(1)) or _first_json_object(text[args_match.start() :])


def _first_json_object(text: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            loaded, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            start = text.find("{", start + 1)
            continue
        if isinstance(loaded, dict):
            return loaded
        start = text.find("{", start + 1)
    return None


def _reason_from_text(text: str) -> str:
    match = re.search(r"(?im)^\s*reason\s*:\s*(.+)$", text)
    return match.group(1).strip()[:240] if match else ""


def _prefixed_line(text: str, prefix: str) -> str:
    match = re.search(rf"(?im)^\s*{re.escape(prefix)}\s*:\s*(.+)$", text)
    return match.group(1).strip()[:240] if match else ""


def _plan_lines(text: str) -> list[str]:
    lines: list[str] = []
    in_plan = False
    for line in text.splitlines():
        stripped = line.strip()
        if re.match(r"(?i)^plan\s*:\s*$", stripped):
            in_plan = True
            continue
        if re.match(r"(?i)^actions\s*:\s*$", stripped):
            break
        if in_plan and stripped.startswith(("-", "*")):
            lines.append(stripped.lstrip("-* ").strip()[:240])
    return lines[:8]


def _dedupe_requests(requests: list[ToolRequest]) -> list[ToolRequest]:
    unique: list[ToolRequest] = []
    seen: set[tuple[str, str]] = set()
    for request in requests:
        key = (request.tool, json.dumps(request.args, sort_keys=True, ensure_ascii=True, default=str))
        if key in seen:
            continue
        seen.add(key)
        unique.append(request)
    return unique


def _looks_like_explicit_tool_request(user_message: str) -> bool:
    normalized = user_message.casefold()
    has_action = any(marker in normalized for marker in _TOOL_INTENT_MARKERS)
    if not has_action:
        return False
    return bool(_request_relevance_terms(user_message))


def _looks_like_explicit_file_mutation_request(user_message: str) -> bool:
    if not re.search(
        r"\b[\w./-]+\.(?:md|txt|json|py|js|ts|tsx|jsx|html|css|yaml|yml|toml)\b",
        user_message,
        flags=re.IGNORECASE,
    ):
        return False
    normalized = user_message.casefold()
    mutation_markers = {
        "add",
        "append",
        "copy",
        "create",
        "edit",
        "move",
        "overwrite",
        "place",
        "put",
        "save",
        "write",
        "вземи",
        "вземеш",
        "добави",
        "добавиш",
        "запази",
        "запиши",
        "копирай",
        "копираш",
        "премести",
        "преместиш",
        "сложи",
        "сложиш",
    }
    return any(marker in normalized for marker in mutation_markers)


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


def _load_json(raw: str) -> Any | None:
    stripped = raw.strip()
    try:
        loaded = json.loads(stripped)
    except json.JSONDecodeError:
        if not stripped:
            return None
        if _looks_like_json_fence(stripped):
            stripped = _strip_json_fence(stripped)
            try:
                loaded = json.loads(stripped)
            except json.JSONDecodeError:
                return None
        elif stripped.startswith("```"):
            return None
        elif stripped.startswith("{"):
            embedded = _first_embedded_planner_payload(stripped)
            return embedded
        else:
            embedded = _first_embedded_planner_payload(stripped)
            if embedded is None:
                return None
            loaded = embedded

    if isinstance(loaded, dict) and "type" not in loaded and "action" not in loaded and isinstance(loaded.get("message"), str):
        loaded["action"] = "answer"
    return loaded


def _first_embedded_planner_payload(raw: str) -> dict[str, Any] | None:
    decoder = json.JSONDecoder()
    index = raw.find("{")
    while index != -1:
        try:
            loaded, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            index = raw.find("{", index + 1)
            continue
        if isinstance(loaded, dict) and _looks_like_planner_payload(loaded):
            return loaded
        index = raw.find("{", index + 1)
    return None


def _looks_like_planner_payload(payload: dict[str, Any]) -> bool:
    if isinstance(payload.get("action"), str) or isinstance(payload.get("type"), str):
        return True
    if isinstance(payload.get("tool"), str) or isinstance(payload.get("name"), str):
        return True
    keys = set(payload)
    return bool("message" in keys and keys <= {"message", "reason"})


def _looks_like_json_fence(raw: str) -> bool:
    return raw.startswith("```") and raw.endswith("```")


def _strip_json_fence(raw: str) -> str:
    lines = raw.splitlines()
    if len(lines) < 2:
        return raw
    first = lines[0].strip().casefold()
    if not first.startswith("```"):
        return raw
    if lines[-1].strip() != "```":
        return raw
    return "\n".join(lines[1:-1]).strip()


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
    context_detail: str = "compact",
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
                "args": manifest.argument_schema if context_detail == "deep" else list(manifest.argument_schema.keys()),
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
            "agent_context": _agent_context_for_prompt(agent_context, detail=context_detail),
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


def _agent_context_for_prompt(agent_context: dict[str, Any], *, detail: str) -> dict[str, Any]:
    if detail == "full":
        return agent_context
    runtime = _dict_value(agent_context, "runtime")
    workspace = _dict_value(agent_context, "workspace")
    memory = _dict_value(agent_context, "memory")
    tools = _dict_value(agent_context, "tools")
    autonomy = _dict_value(agent_context, "autonomy")
    runtime_state = _dict_value(agent_context, "runtime_state")
    compact = {
        "agent_name": agent_context.get("agent_name", "DMD Agent"),
        "autonomy": {
            "enabled": autonomy.get("enabled"),
            "mode": autonomy.get("mode"),
        },
        "runtime": runtime,
        "workspace": {
            "current": workspace.get("current"),
            "downloads_path": workspace.get("downloads_path"),
            "allowed_roots": workspace.get("allowed_roots", [])[:8],
            "mounted_roots": workspace.get("mounted_roots", [])[:8],
        },
        "memory": {
            "root": memory.get("root"),
            "file_count": memory.get("file_count"),
            "files": list(memory.get("files") or [])[:40],
        },
        "tools": {
            "enabled": list(tools.get("enabled") or [])[:80],
            "disabled": list(tools.get("disabled") or [])[:80],
            "count": tools.get("count"),
        },
        "permissions": agent_context.get("permissions", {}),
        "terminal": agent_context.get("terminal", {}),
        "approvals": agent_context.get("approvals", {}),
        "recent_activity": agent_context.get("recent_activity", {}),
        "runtime_state": {
            "mode": runtime_state.get("mode"),
            "cache": runtime_state.get("cache"),
            "workspace": _compact_folder_state(_dict_value(runtime_state, "workspace")),
            "downloads": _compact_folder_state(_dict_value(runtime_state, "downloads")),
        },
    }
    if detail == "deep":
        compact["config"] = agent_context.get("config", {})
        compact["email"] = agent_context.get("email", {})
        compact["security_invariants"] = agent_context.get("security_invariants", [])
    return compact


def _compact_folder_state(folder: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": folder.get("path"),
        "exists": folder.get("exists"),
        "allowed": folder.get("allowed"),
        "folders": list(folder.get("folders") or [])[:40],
        "files": list(folder.get("files") or [])[:40],
        "file_count": folder.get("file_count"),
    }


def _dict_value(mapping: dict[str, Any], key: str) -> dict[str, Any]:
    value = mapping.get(key)
    return value if isinstance(value, dict) else {}
