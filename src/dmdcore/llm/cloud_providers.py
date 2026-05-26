from __future__ import annotations

from dataclasses import dataclass

from dmdcore.llm.base import LLMMessage, LLMResponse


class CloudProviderNotConfigured(RuntimeError):
    pass


@dataclass
class CloudProviderStub:
    provider_name: str
    model: str
    is_local: bool = False

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        del messages
        del max_tokens
        del temperature
        del think
        raise CloudProviderNotConfigured(
            f"{self.provider_name} is not configured. Cloud providers require explicit setup and privacy approval."
        )
