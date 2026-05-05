from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse

from dmdagent4all.llm.base import LLMMessage, LLMResponse
from dmdagent4all.llm.openai_usage import check_openai_budget_available, record_openai_usage


@dataclass
class OpenAICompatibleProvider:
    model: str
    base_url: str
    api_key_env: str | None = None
    provider_name: str = "openai-compatible"

    @property
    def is_local(self) -> bool:
        return _is_local_base_url(self.base_url)

    def chat(
        self,
        messages: list[LLMMessage],
        *,
        max_tokens: int | None = None,
        temperature: float | None = None,
        think: bool | None = None,
    ) -> LLMResponse:
        del think
        payload: dict[str, object] = {
            "model": self.model,
            "messages": [
                {"role": message.role, "content": message.content}
                for message in messages
            ],
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if temperature is not None:
            payload["temperature"] = temperature

        headers = {"Content-Type": "application/json"}
        api_key = _api_key_from_env(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        elif not self.is_local:
            raise RuntimeError(
                f"{self.provider_name} requires an API key in "
                f"{self.api_key_env or 'the configured api_key_env'}."
            )
        if self.provider_name == "openai":
            check_openai_budget_available()

        request = urllib.request.Request(
            f"{self.base_url.rstrip('/')}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:400]
            raise RuntimeError(
                f"{self.provider_name} request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Cannot connect to {self.provider_name} at {self.base_url}."
            ) from exc

        choices = data.get("choices")
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        content = first_choice.get("message", {}).get("content", "")
        usage = data.get("usage") if isinstance(data.get("usage"), dict) else None
        if self.provider_name == "openai" and usage is not None:
            record_openai_usage(provider=self.provider_name, model=self.model, usage=usage)
        return LLMResponse(
            content=str(content),
            model=self.model,
            provider=self.provider_name,
            usage=usage,
        )


def _api_key_from_env(api_key_env: str | None) -> str:
    if not api_key_env:
        return ""
    return os.environ.get(api_key_env, "").strip()


def _is_local_base_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.hostname in {"127.0.0.1", "localhost", "::1"}
