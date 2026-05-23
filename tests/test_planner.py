import json
import unittest

from dmdagent4all.agent.planner import (
    LLMPlanner,
    PlannerError,
    REPAIR_PROMPT,
    SYSTEM_PROMPT,
    is_low_relevance_plan,
    parse_plan_response,
)
from dmdagent4all.llm.base import LLMMessage, LLMResponse


class PlannerTest(unittest.TestCase):
    def test_parses_final_response(self) -> None:
        result = parse_plan_response('{"type":"final","message":"Hello"}')
        self.assertEqual(result.final_message, "Hello")
        self.assertIsNone(result.tool_request)

    def test_memory_writes_block_in_action_protocol(self) -> None:
        raw = (
            "OBJECTIVE: Discuss the house\n"
            "PLAN:\n"
            "- Acknowledge\n"
            "ACTIONS:\n"
            "MEMORY_WRITES: ["
            '{"slug":"house-build-project","type":"project","confidence":"high",'
            '"description":"Wooden house in Gorna Malina",'
            r'"body":"- Location: Gorna Malina\n- Material: wood"}'
            "]\n"
        )
        result = parse_plan_response(raw)
        self.assertEqual(len(result.memory_writes), 1)
        intent = result.memory_writes[0]
        self.assertEqual(intent.slug, "house-build-project")
        self.assertEqual(intent.memory_type, "project")
        self.assertEqual(intent.confidence, "high")
        self.assertIn("Gorna Malina", intent.body)
        self.assertNotIn("MEMORY_WRITES", result.final_message or "")

    def test_memory_writes_in_json_payload(self) -> None:
        raw = json.dumps(
            {
                "action": "answer",
                "message": "OK",
                "memory_writes": [
                    {
                        "slug": "coffee-preference",
                        "type": "user",
                        "confidence": "high",
                        "description": "Coffee preference",
                        "body": "- Likes ice coffee",
                    }
                ],
            }
        )
        result = parse_plan_response(raw)
        self.assertEqual(result.final_message, "OK")
        self.assertEqual(len(result.memory_writes), 1)
        intent = result.memory_writes[0]
        self.assertEqual(intent.slug, "coffee-preference")
        self.assertEqual(intent.memory_type, "user")
        self.assertEqual(intent.body, "- Likes ice coffee")

    def test_memory_writes_skipped_when_no_block(self) -> None:
        result = parse_plan_response('{"type":"final","message":"Just chatting"}')
        self.assertEqual(result.memory_writes, ())

    def test_memory_writes_filters_invalid_entries(self) -> None:
        raw = json.dumps(
            {
                "action": "answer",
                "message": "OK",
                "memory_writes": [
                    {"slug": "", "body": "no slug"},
                    {"slug": "valid", "body": ""},
                    {
                        "slug": "valid-slug",
                        "type": "junk",
                        "confidence": "weird",
                        "body": "- Fact",
                    },
                ],
            }
        )
        result = parse_plan_response(raw)
        self.assertEqual(len(result.memory_writes), 1)
        self.assertEqual(result.memory_writes[0].slug, "valid-slug")
        self.assertEqual(result.memory_writes[0].memory_type, "project")
        self.assertEqual(result.memory_writes[0].confidence, "medium")

    def test_parses_tool_request(self) -> None:
        result = parse_plan_response(
            '{"type":"tool_request","tool":"memory.list","args":{},"reason":"List memory"}'
        )
        self.assertIsNotNone(result.tool_request)
        self.assertEqual(result.tool_request.tool, "memory.list")

    def test_accepts_common_tool_type_aliases(self) -> None:
        result = parse_plan_response('{"type":"tool","tool":"memory.list","args":{}}')
        self.assertIsNotNone(result.tool_request)
        self.assertEqual(result.tool_request.tool, "memory.list")

    def test_accepts_function_call_shape(self) -> None:
        result = parse_plan_response(
            '{"type":"function_call","name":"memory.read","arguments":{"path":"profile.md"}}'
        )
        self.assertIsNotNone(result.tool_request)
        self.assertEqual(result.tool_request.tool, "memory.read")
        self.assertEqual(result.tool_request.args["path"], "profile.md")

    def test_extracts_json_from_wrapped_text(self) -> None:
        result = parse_plan_response('text before {"type":"final","message":"Ok"} text after')
        self.assertEqual(result.final_message, "Ok")

    def test_plain_text_response_becomes_final_answer(self) -> None:
        result = parse_plan_response("Sure, I can help with that.")
        self.assertEqual(result.final_message, "Sure, I can help with that.")

    def test_plain_text_with_config_json_snippet_becomes_final_answer(self) -> None:
        raw = (
            'За config знам, че llm изглежда така: {"provider":"deepseek","model":"deepseek-v4-pro"}. '
            "Workspace и tools идват от runtime контекста."
        )

        result = parse_plan_response(raw)

        self.assertEqual(result.final_message, raw)

    def test_malformed_json_response_becomes_final_without_repair(self) -> None:
        raw = '{"type":"final","message":"Still answer naturally"'
        result = parse_plan_response(raw)
        self.assertEqual(result.final_message, raw)

    def test_unclosed_json_fence_becomes_final_without_repair(self) -> None:
        raw = '```json\n{"action":"answer","message":"partial"'
        result = parse_plan_response(raw)
        self.assertEqual(result.final_message, raw)

    def test_message_without_type_becomes_final_answer(self) -> None:
        result = parse_plan_response('{"message":"Hello without type"}')
        self.assertEqual(result.final_message, "Hello without type")

    def test_unknown_type_becomes_final_answer(self) -> None:
        raw = '{"type":"unknown"}'
        result = parse_plan_response(raw)
        self.assertEqual(result.final_message, raw)

    def test_custom_system_prompt_is_sent_to_provider(self) -> None:
        provider = RecordingProvider('{"type":"final","message":"ok"}')
        planner = LLMPlanner(provider)

        result = planner.plan(
            user_message="hello",
            manifests={},
            system_prompt="Custom planner instructions",
        )

        self.assertEqual(result.final_message, "ok")
        self.assertEqual(provider.messages[0].content, "Custom planner instructions")

    def test_custom_answer_prompt_is_sent_to_provider(self) -> None:
        provider = RecordingProvider("final answer")
        planner = LLMPlanner(provider)

        answer = planner.answer(user_message="hello", system_prompt="Custom answer instructions")

        self.assertEqual(answer, "final answer")
        self.assertEqual(provider.messages[0].content, "Custom answer instructions")

    def test_planner_does_not_repair_malformed_json(self) -> None:
        provider = SequenceProvider(
            [
                '{"type":"tool_request","tool":"memory.list","args":{}',
                '{"type":"tool_request","tool":"memory.list","args":{},"reason":"fixed"}',
            ]
        )
        planner = LLMPlanner(provider)

        result = planner.plan(user_message="list memory", manifests={})

        self.assertIsNone(result.tool_request)
        self.assertEqual(provider.call_count, 1)

    def test_parses_action_protocol_multi_tool_plan(self) -> None:
        result = parse_plan_response(
            """OBJECTIVE: Create project
PLAN:
- Create folder
- Scaffold app
ACTIONS:
- tool: files.mkdir
  args: {"path":"test","parents":true}
  reason: Create folder.
- tool: project.scaffold_one_page_app
  args: {"path":"test","owner_name":"Denis Denchev","role":"AI Developer","theme":"developer tech dark","include_backend":true,"overwrite":true}
  reason: Scaffold site.
"""
        )

        self.assertEqual(result.objective, "Create project")
        self.assertEqual(len(result.tool_plan), 2)
        self.assertEqual(result.tool_plan[0].tool, "files.mkdir")
        self.assertEqual(result.tool_plan[1].tool, "project.scaffold_one_page_app")

    def test_parses_scrape_to_file_multi_tool_plan(self) -> None:
        raw = json.dumps(
            {
                "action": "multi_tool_plan",
                "steps": [
                    {
                        "tool": "browser.scrape_markdown",
                        "args": {
                            "url": "https://dmdflow.com",
                            "mode": "raw_page",
                            "format": "clean_markdown",
                            "instructions": "Scrape dmdflow.com",
                        },
                    },
                    {
                        "tool": "files.write",
                        "args": {
                            "path": "readme123.md",
                            "content_from_previous_step": True,
                        },
                    },
                ],
            }
        )

        result = parse_plan_response(raw)

        self.assertEqual(len(result.tool_plan), 2)
        self.assertEqual(result.tool_plan[0].tool, "browser.scrape_markdown")
        self.assertEqual(result.tool_plan[0].args["url"], "https://dmdflow.com")
        self.assertEqual(result.tool_plan[1].tool, "files.write")
        self.assertTrue(result.tool_plan[1].args["content_from_previous_step"])

    def test_scrape_to_file_prompt_guides_multi_tool_plan(self) -> None:
        self.assertIn("content_from_previous_step", SYSTEM_PROMPT)
        self.assertIn("multi_tool_plan", SYSTEM_PROMPT)
        self.assertIn("browser.scrape_markdown", SYSTEM_PROMPT)
        self.assertIn("files.write", SYSTEM_PROMPT)

    def test_prompt_receives_agent_context_snapshot(self) -> None:
        provider = RecordingProvider('{"type":"final","message":"ok"}')
        planner = LLMPlanner(provider)
        agent_context = {
            "runtime": {"provider": "ollama", "model": "qwen-context"},
            "workspace": {"current": "/tmp/workspace"},
            "tools": {"enabled": ["files.read"], "disabled": ["terminal.run"]},
            "memory": {"files": ["long-term/profile.md"]},
        }

        planner.plan(
            user_message="what model are you",
            manifests={},
            agent_context=agent_context,
        )
        payload = json.loads(provider.messages[1].content)

        self.assertEqual(payload["agent_context"]["runtime"]["model"], "qwen-context")
        self.assertIn("files.read", payload["agent_context"]["tools"]["enabled"])
        self.assertIn("workspace", payload["agent_context"])

    def test_prompt_guides_context_aware_file_and_email_workflows(self) -> None:
        self.assertIn("agent_context.workspace.mounted_roots", SYSTEM_PROMPT)
        self.assertIn("files.list", SYSTEM_PROMPT)
        self.assertIn("path_from_previous_step_match", SYSTEM_PROMPT)
        self.assertIn("path_from_selected_step", SYSTEM_PROMPT)
        self.assertIn("body_from_previous_step", SYSTEM_PROMPT)
        self.assertIn("draft_id_from_previous_step", SYSTEM_PROMPT)

    def test_prompt_guides_local_dev_script_workflows(self) -> None:
        self.assertIn("LOCAL_DEV_AUTONOMY", SYSTEM_PROMPT)
        self.assertIn("create a script", SYSTEM_PROMPT)
        self.assertIn("terminal.run", SYSTEM_PROMPT)
        self.assertIn("python3", SYSTEM_PROMPT)

    def test_unrelated_visual_clarification_is_low_relevance_for_scrape_request(self) -> None:
        result = parse_plan_response(
            '{"action":"ask_clarification","message":"I need the image or description of the rotating objects to determine which number is the rotating one."}'
        )

        self.assertTrue(
            is_low_relevance_plan(
                result,
                "scrape dmdflow.com and place the results in readme123.md",
            )
        )

    def test_planner_does_not_retry_unrelated_output_with_strict_schema(self) -> None:
        provider = SequenceProvider(
            [
                '{"action":"ask_clarification","message":"I need the image or description of the rotating objects to determine which number is the rotating one."}',
                json.dumps(
                    {
                        "action": "multi_tool_plan",
                        "steps": [
                            {
                                "tool": "browser.scrape_markdown",
                                "args": {"url": "https://dmdflow.com"},
                                "reason": "Scrape the requested page.",
                            },
                            {
                                "tool": "files.write",
                                "args": {"path": "readme123.md", "content_from_previous_step": True},
                                "reason": "Write the scraped Markdown to the requested file.",
                            },
                        ],
                    }
                ),
            ]
        )
        planner = LLMPlanner(provider)

        result = planner.plan(
            user_message="scrape dmdflow.com and place the results in readme123.md",
            manifests={},
        )

        self.assertEqual(provider.call_count, 1)
        self.assertEqual(result.clarification_message, "I need the image or description of the rotating objects to determine which number is the rotating one.")

    def test_planner_prompts_do_not_contain_visual_task_leakage(self) -> None:
        prompt_text = f"{SYSTEM_PROMPT}\n{REPAIR_PROMPT}".casefold()
        for marker in (
            "rotating objects",
            "rotating one",
            "image or description",
            "determine which number",
        ):
            self.assertNotIn(marker, prompt_text)


class RecordingProvider:
    provider_name = "test"
    model = "test-model"
    is_local = True

    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[LLMMessage] = []

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature, think
        self.messages = messages
        return LLMResponse(content=self.content, model=self.model, provider=self.provider_name)


class SequenceProvider(RecordingProvider):
    def __init__(self, contents: list[str]) -> None:
        super().__init__(contents[0])
        self.contents = contents
        self.call_count = 0

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        del max_tokens, temperature, think
        self.messages = messages
        content = self.contents[min(self.call_count, len(self.contents) - 1)]
        self.call_count += 1
        return LLMResponse(content=content, model=self.model, provider=self.provider_name)


if __name__ == "__main__":
    unittest.main()
