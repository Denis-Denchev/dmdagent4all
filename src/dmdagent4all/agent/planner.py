from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from dmdagent4all.llm.base import LLMMessage, LLMProvider
from dmdagent4all.permissions import ToolManifest, ToolRequest


SYSTEM_PROMPT = """You are DMD Agent's fast planner. /no_think

Security: you are untrusted. You never access OS, shell, tokens, .env, SSH keys, or browser credentials. You only propose one listed tool call. Backend validates everything.

Return strict JSON only:
- Tool:
  {"type":"tool_request","tool":"tool.name","args":{},"reason":"short reason"}
- Answer:
  {"type":"final","message":"answer to the user"}
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
        response_language: str = "auto",
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
                        response_language=response_language,
                    ),
                ),
            ],
            max_tokens=max_tokens,
            temperature=temperature,
            think=think,
        )
        return parse_plan_response(response.content)


def parse_plan_response(raw: str) -> PlanResult:
    payload = _load_json(raw)
    plan_type = payload.get("type")
    if plan_type == "final":
        message = str(payload.get("message", "")).strip()
        if not message:
            raise PlannerError("Planner returned an empty final message.")
        return PlanResult(final_message=message, raw=raw)

    if plan_type == "tool_request":
        tool = str(payload.get("tool", "")).strip()
        if not tool:
            raise PlannerError("Planner returned a tool request without a tool name.")
        args = payload.get("args", {})
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
            raise PlannerError("Planner response was not JSON.") from None
        try:
            loaded = json.loads(raw[start : end + 1])
        except json.JSONDecodeError as exc:
            raise PlannerError(f"Planner response JSON could not be parsed: {exc}") from exc

    if not isinstance(loaded, dict):
        raise PlannerError("Planner response must be a JSON object.")
    return loaded


def _build_planning_prompt(
    *,
    user_message: str,
    manifests: dict[str, ToolManifest],
    response_language: str,
) -> str:
    tool_rows = []
    for manifest in manifests.values():
        enabled = "on" if manifest.default_enabled else "off"
        approval = ",approval" if manifest.approval_required else ""
        tool_rows.append(f"{manifest.name}(r{int(manifest.risk)},{enabled}{approval})")
    return json.dumps(
        {
            "lang": response_language,
            "tools": "; ".join(tool_rows),
            "user": user_message,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    )
