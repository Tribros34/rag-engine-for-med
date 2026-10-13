"""OpenAI uyumlu /chat/completions istemcisi (Ollama, vLLM, LM Studio vb.)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import httpx

from anatomy_ks.config import Settings
from anatomy_ks.llm.base import LLMUnavailable


class OpenAICompatLLM:
    def __init__(self, settings: Settings, transport: httpx.BaseTransport | None = None) -> None:
        self.model = settings.llm_model
        base = settings.llm_base_url.rstrip("/")
        self.url = f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if settings.llm_api_key:
            headers["Authorization"] = f"Bearer {settings.llm_api_key}"
        self.client = httpx.Client(
            headers=headers, timeout=settings.llm_timeout_s, transport=transport
        )

    def _body(self, system: str, user: str, max_tokens: int, stream: bool) -> dict[str, Any]:
        return {
            "model": self.model,
            "max_tokens": max_tokens,
            "temperature": 0.1,
            "stream": stream,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        try:
            with self.client.stream(
                "POST", self.url, json=self._body(system, user, max_tokens, True)
            ) as r:
                r.raise_for_status()
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        delta = json.loads(data)["choices"][0].get("delta", {})
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue
                    text = delta.get("content")
                    if text:
                        yield text
        except httpx.HTTPError as e:
            raise LLMUnavailable(f"openai_compatible: {type(e).__name__}") from e

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        try:
            r = self.client.post(self.url, json=self._body(system, user, max_tokens, False))
            r.raise_for_status()
            return str(r.json()["choices"][0]["message"]["content"] or "")
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as e:
            raise LLMUnavailable(f"openai_compatible: {type(e).__name__}") from e
