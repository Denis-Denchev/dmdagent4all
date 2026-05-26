import unittest

from dmdcore.security import redact_text


class RedactionTest(unittest.TestCase):
    def test_redacts_api_keys_and_passwords(self) -> None:
        text = "token=abcdefghijklmnopqrstuvwxyz and sk-abcdefghijklmnopqrstuvwxyz123456"
        redacted = redact_text(text)
        self.assertIn("[REDACTED_SECRET]", redacted)
        self.assertIn("[REDACTED_API_KEY]", redacted)

    def test_email_redaction_is_optional(self) -> None:
        text = "Contact user@example.com"
        self.assertIn("user@example.com", redact_text(text))
        self.assertIn("[REDACTED_EMAIL]", redact_text(text, redact_emails=True))


if __name__ == "__main__":
    unittest.main()
