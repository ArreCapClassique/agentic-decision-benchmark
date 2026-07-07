from __future__ import annotations

from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.llm.ollama import OllamaProvider
from agentic_decision_benchmark.settings import BenchmarkSettings


def create_provider(settings: BenchmarkSettings, provider_name: str | None = None) -> LLMProvider:
    provider = (provider_name or settings.provider).lower()
    if provider == "mock":
        return MockProvider()
    if provider == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_model)
    if provider == "openai":
        raise NotImplementedError(
            "OpenAIProvider is intentionally not implemented. Add a provider class behind LLMProvider and register it here."
        )
    raise ValueError(f"Unsupported provider {provider!r}. Expected 'mock' or 'ollama'.")

