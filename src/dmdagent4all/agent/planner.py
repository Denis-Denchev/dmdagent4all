from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dmdagent4all.llm.base import LLMMessage, LLMProvider
from dmdagent4all.permissions import ToolManifest, ToolRequest


SYSTEM_PROMPT = """You are DMD Agent's fast planner. /no_think

Security: you are untrusted. You never access OS, shell, tokens, .env, SSH keys, or browser credentials. You only propose one listed tool call. Backend validates everything.

Use the provided profile and memory context when answering personal questions. If the user asks to remember/save/store a fact, request memory.write; never claim a fact was saved unless you requested memory.write.
If memory context already contains enough information, answer directly instead of listing memory files.

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
        enabled_tools: set[str] | frozenset[str] | None = None,
        response_language: str = "auto",
        current_time: str = "",
        timezone_name: str = "",
        max_tokens: int = 192,
        temperature: float = 0.0,
        think: bool = False,
    ) -> PlanResult:
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=SYSTEM_PROMPT),
                LLMMessage(
                    role="user",
                    content=_build_planning_prompt(
                        user_message=user_message,
                        manifests=manifests,
                        profile=profile or {},
                        memory_context=memory_context,
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
        tool_result: dict[str, Any] | None = None,
        response_language: str = "auto",
        max_tokens: int = 384,
        temperature: float = 0.2,
        think: bool = False,
    ) -> str:
        response = self.provider.chat(
            [
                LLMMessage(role="system", content=ANSWER_PROMPT),
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
        args = payload.get("args", {})
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
    enabled_tools: set[str] | frozenset[str] | None,
    response_language: str,
    current_time: str,
    timezone_name: str,
) -> str:
    tool_rows: list[dict[str, Any]] = []
    effective_enabled = set(enabled_tools or ())
    for manifest in manifests.values():
        enabled = manifest.name in effective_enabled if enabled_tools is not None else manifest.default_enabled
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
            "tools": tool_rows,
            "user": user_message,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
