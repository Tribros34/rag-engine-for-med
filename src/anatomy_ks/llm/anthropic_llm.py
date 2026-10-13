"""Anthropic Messages API istemcisi."""

from __future__ import annotations

from collections.abc import Iterator

import anthropic

from anatomy_ks.config import Settings
from anatomy_ks.llm.base import LLMUnavailable


class AnthropicLLM:
    def __init__(self, settings: Settings) -> None:
        self.model = settings.llm_model
        self.client = anthropic.Anthropic(
            api_key=settings.llm_api_key,
            timeout=settings.llm_timeout_s,
            max_retries=1,
            base_url=settings.llm_base_url or None,
        )

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        try:
            with self.client.messages.stream(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            ) as s:
                yield from s.text_stream
        except anthropic.APIError as e:
            raise LLMUnavailable(f"anthropic: {type(e).__name__}") from e

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        try:
            msg = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
        except anthropic.APIError as e:
            raise LLMUnavailable(f"anthropic: {type(e).__name__}") from e
        return "".join(b.text for b in msg.content if b.type == "text")
