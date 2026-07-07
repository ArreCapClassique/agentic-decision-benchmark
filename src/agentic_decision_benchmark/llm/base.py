from __future__ import annotations

from typing import Protocol

from agentic_decision_benchmark.schemas import ProviderStats


class LLMProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        ...

    @property
    def stats(self) -> ProviderStats:
        ...


def estimate_tokens(text: str) -> int:
    return max(1, len(text.split())) if text else 0


class BaseLLMProvider:
    def __init__(self) -> None:
        self._stats = ProviderStats()

    @property
    def stats(self) -> ProviderStats:
        return self._stats

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        response = self._generate(prompt, temperature=temperature, max_tokens=max_tokens)
        self._stats.model_calls += 1
        self._stats.estimated_input_tokens += estimate_tokens(prompt)
        self._stats.estimated_output_tokens += estimate_tokens(response)
        return response

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        raise NotImplementedError

