from __future__ import annotations

from agentic_decision_benchmark.llm.factory import create_provider
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.llm.ollama import OllamaProvider
from agentic_decision_benchmark.settings import load_settings


def test_provider_factory_returns_mock(tmp_path) -> None:
    settings = load_settings(provider="mock", output_dir=str(tmp_path))
    provider = create_provider(settings)
    assert isinstance(provider, MockProvider)


def test_provider_factory_returns_ollama(tmp_path) -> None:
    settings = load_settings(provider="ollama", output_dir=str(tmp_path))
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_provider_interface_is_consistent() -> None:
    provider = MockProvider()
    output = provider.generate("TASK: EVALUATOR\n{}", temperature=0.2, max_tokens=100)
    assert isinstance(output, str)
    assert provider.stats.model_calls == 1

