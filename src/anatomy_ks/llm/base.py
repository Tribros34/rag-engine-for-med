"""Sağlayıcıdan bağımsız LLM arayüzü (Bölüm 9.5). Model adı yalnızca LLM_MODEL env'den gelir."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol

from anatomy_ks.config import Settings


class LLMUnavailable(RuntimeError):
    """LLM yapılandırılmamış veya sağlayıcıya ulaşılamıyor."""


class LLMClient(Protocol):
    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]: ...

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str: ...


class UnconfiguredLLM:
    """API anahtarı / model girilmeden önce kullanılan yer tutucu."""

    def __init__(self, reason: str = "LLM yapılandırılmadı (LLM_MODEL / LLM_API_KEY)") -> None:
        self.reason = reason

    def stream(self, system: str, user: str, max_tokens: int = 800) -> Iterator[str]:
        raise LLMUnavailable(self.reason)

    def complete(self, system: str, user: str, max_tokens: int = 800) -> str:
        raise LLMUnavailable(self.reason)


def get_llm(settings: Settings) -> LLMClient:
    if not settings.llm_configured:
        return UnconfiguredLLM()
    match settings.llm_provider:
        case "anthropic":
            from anatomy_ks.llm.anthropic_llm import AnthropicLLM

            return AnthropicLLM(settings)
        case "openai_compatible":
            from anatomy_ks.llm.openai_compat import OpenAICompatLLM

            return OpenAICompatLLM(settings)
        case _:
            raise ValueError(settings.llm_provider)
