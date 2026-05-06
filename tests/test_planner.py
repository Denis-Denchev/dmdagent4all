import unittest

from dmdagent4all.agent.planner import LLMPlanner, PlannerError, parse_plan_response
from dmdagent4all.llm.base import LLMMessage, LLMResponse


class PlannerTest(unittest.TestCase):
    def test_parses_final_response(self) -> None:
        result = parse_plan_response('{"type":"final","message":"Hello"}')
        self.assertEqual(result.final_message, "Hello")
        self.assertIsNone(result.tool_request)

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

    def test_message_without_type_becomes_final_answer(self) -> None:
        result = parse_plan_response('{"message":"Hello without type"}')
        self.assertEqual(result.final_message, "Hello without type")

    def test_rejects_unknown_type(self) -> None:
        with self.assertRaises(PlannerError):
            parse_plan_response('{"type":"unknown"}')

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


if __name__ == "__main__":
    unittest.main()
