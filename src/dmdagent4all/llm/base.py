from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class LLMMessage:
    role: str
    content: str


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    provider: str


class LLMProvider(Protocol):
    provider_name: str
    model: str
    is_local: bool

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        """Return a model response. Providers must never receive raw secrets."""
