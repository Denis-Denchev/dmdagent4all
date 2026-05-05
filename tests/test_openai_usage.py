import tempfile
import unittest
from pathlib import Path

from dmdagent4all.llm.openai_usage import (
    estimate_openai_cost,
    openai_usage_summary,
    record_openai_usage,
    reset_openai_usage,
)


class OpenAIUsageTest(unittest.TestCase):
    def test_records_and_summarizes_estimated_usage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_db = Path(tmp) / "audit.db"

            recorded = record_openai_usage(
                provider="openai",
                model="gpt-4o-mini",
                usage={"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000},
                audit_db=audit_db,
            )
            summary = openai_usage_summary(
                audit_db,
                config={"openai_usage": {"limit_usd": 1.0}},
            )

        self.assertEqual(recorded["total_tokens"], 2_000_000)
        self.assertEqual(summary["requests"], 1)
        self.assertEqual(summary["prompt_tokens"], 1_000_000)
        self.assertEqual(summary["completion_tokens"], 1_000_000)
        self.assertEqual(summary["estimated_cost_usd"], 0.75)
        self.assertEqual(summary["remaining_usd"], 0.25)
        self.assertFalse(summary["limit_reached"])

    def test_unknown_model_cost_is_zero(self) -> None:
        self.assertEqual(estimate_openai_cost("unknown-model", 1000, 1000), 0.0)

    def test_reset_usage_clears_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            audit_db = Path(tmp) / "audit.db"
            record_openai_usage(
                provider="openai",
                model="gpt-4o-mini",
                usage={"prompt_tokens": 10, "completion_tokens": 10},
                audit_db=audit_db,
            )

            reset_openai_usage(audit_db)
            summary = openai_usage_summary(audit_db, config={})

        self.assertEqual(summary["requests"], 0)
        self.assertEqual(summary["estimated_cost_usd"], 0.0)


if __name__ == "__main__":
    unittest.main()
