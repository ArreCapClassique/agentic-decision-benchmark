from __future__ import annotations

from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.llm.ollama import OllamaProvider
from agentic_decision_benchmark.llm.openai import OpenAIProvider
from agentic_decision_benchmark.settings import BenchmarkSettings


def create_provider(settings: BenchmarkSettings, provider_name: str | None = None) -> LLMProvider:
    provider = (provider_name or settings.provider).lower()
    if provider == "mock":
        return MockProvider()
    if provider == "ollama":
        return OllamaProvider(base_url=settings.ollama_base_url, model=settings.ollama_model)
    if provider == "openai":
        return OpenAIProvider(
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            model=settings.openai_model,
            timeout=settings.openai_timeout,
            store=settings.openai_store,
        )
    raise ValueError(f"Unsupported provider {provider!r}. Expected 'mock', 'ollama', or 'openai'.")

