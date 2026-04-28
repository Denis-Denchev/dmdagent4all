from __future__ import annotations

import json
import urllib.request
from urllib.error import URLError
from dataclasses import dataclass

from dmdagent4all.llm.base import LLMMessage, LLMResponse


@dataclass
class OllamaProvider:
    model: str = "qwen3:8b"
    base_url: str = "http://localhost:11434"

    provider_name: str = "ollama"
    is_local: bool = True

    def chat(self, messages: list[LLMMessage]) -> LLMResponse:
        payload = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
            "stream": False,
        }
        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except URLError as exc:
            raise RuntimeError(
                "Cannot connect to Ollama. Start Ollama with `ollama serve` "
                f"and make sure `{self.model}` is installed with "
                f"`ollama pull {self.model}`."
            ) from exc
        content = data.get("message", {}).get("content", "")
        return LLMResponse(
            content=content,
            model=self.model,
            provider=self.provider_name,
        )
