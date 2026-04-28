from __future__ import annotations

from dataclasses import dataclass

from dmdagent4all.llm.base import LLMMessage, LLMResponse


class CloudProviderNotConfigured(RuntimeError):
    pass


@dataclass
class CloudProviderStub:
    provider_name: str
    model: str
    is_local: bool = False

    def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        del messages
        raise CloudProviderNotConfigured(
            f"{self.provider_name} is not configured. Cloud providers require explicit setup and privacy approval."
        )
