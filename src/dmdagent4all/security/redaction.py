from __future__ import annotations

import re


PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
    re.DOTALL,
)
API_KEY_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{20,}|xox[baprs]-[A-Za-z0-9-]{20,}|gh[pousr]_[A-Za-z0-9_]{20,})\b"
)
GENERIC_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|secret|token|password)\s*[:=]\s*['\"]?([A-Za-z0-9_./+=:-]{12,})['\"]?"
)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
PHONE_RE = re.compile(r"(?<!\d)(?:\+?\d[\d .()-]{7,}\d)(?!\d)")
CARD_RE = re.compile(r"(?<![\w/-])(?:\d[ -]*?){13,19}(?![\w/-])")


def redact_text(
    text: str,
    *,
    redact_emails: bool = False,
    redact_phone_numbers: bool = False,
    redact_addresses: bool = False,
) -> str:
    redacted = PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", text)
    redacted = API_KEY_RE.sub("[REDACTED_API_KEY]", redacted)
    redacted = GENERIC_SECRET_RE.sub(
        lambda match: f"{match.group(1)}=[REDACTED_SECRET]",
        redacted,
    )
    redacted = CARD_RE.sub("[REDACTED_PAYMENT]", redacted)
    if redact_emails:
        redacted = EMAIL_RE.sub("[REDACTED_EMAIL]", redacted)
    if redact_phone_numbers:
        redacted = PHONE_RE.sub("[REDACTED_PHONE]", redacted)
    if redact_addresses:
        redacted = re.sub(r"\b\d{1,6}\s+[A-Za-z0-9 .'-]+\b", "[REDACTED_ADDRESS]", redacted)
    return redacted
