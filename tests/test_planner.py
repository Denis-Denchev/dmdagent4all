import unittest

from dmdagent4all.agent.planner import PlannerError, parse_plan_response


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

    def test_extracts_json_from_wrapped_text(self) -> None:
        result = parse_plan_response('text before {"type":"final","message":"Ok"} text after')
        self.assertEqual(result.final_message, "Ok")

    def test_rejects_unknown_type(self) -> None:
        with self.assertRaises(PlannerError):
            parse_plan_response('{"type":"unknown"}')


if __name__ == "__main__":
    unittest.main()
