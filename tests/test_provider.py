from __future__ import annotations

import json

import httpx

from agentic_decision_benchmark.graphs.common_nodes import provider_json
from agentic_decision_benchmark.llm.factory import create_provider
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.llm.ollama import OllamaProvider
from agentic_decision_benchmark.llm.openai import OpenAIProvider
from agentic_decision_benchmark.schemas import ProviderStats, SingleRecommendation
from agentic_decision_benchmark.settings import load_settings


class RepairingProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self._stats = ProviderStats()

    @property
    def stats(self) -> ProviderStats:
        return self._stats

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        self.calls.append(prompt)
        if len(self.calls) == 1:
            return json.dumps(
                {
                    "recommended_strategy": "C",
                    "recommendation": "Use a staged dual-track strategy.",
                    "reasoning": "This should be an array.",
                    "risks": "This should be an array.",
                    "assumptions": "This should be an array.",
                    "confidence": 0.7,
                }
            )
        return json.dumps(
            {
                "recommended_strategy": "C",
                "recommendation": "Use a staged dual-track strategy.",
                "reasoning": ["The strategy preserves option value."],
                "risks": ["The strategy creates coordination risk."],
                "assumptions": ["Management can enforce investment gates."],
                "confidence": 0.7,
            }
        )


def test_provider_factory_returns_mock(tmp_path) -> None:
    settings = load_settings(provider="mock", output_dir=str(tmp_path))
    provider = create_provider(settings)
    assert isinstance(provider, MockProvider)


def test_provider_factory_returns_ollama(tmp_path) -> None:
    settings = load_settings(provider="ollama", output_dir=str(tmp_path))
    provider = create_provider(settings)
    assert isinstance(provider, OllamaProvider)


def test_provider_factory_returns_openai(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    settings = load_settings(provider="openai", output_dir=str(tmp_path))
    provider = create_provider(settings)
    assert isinstance(provider, OpenAIProvider)


def test_provider_interface_is_consistent() -> None:
    provider = MockProvider()
    output = provider.generate("TASK: EVALUATOR\n{}", temperature=0.2, max_tokens=100)
    assert isinstance(output, str)
    assert provider.stats.model_calls == 1


def test_provider_json_retries_schema_invalid_output() -> None:
    provider = RepairingProvider()

    output = provider_json(
        provider,
        "TASK: GENERALIST_RECOMMENDATION",
        SingleRecommendation,
        temperature=0.2,
        max_tokens=100,
    )

    assert output.reasoning == ["The strategy preserves option value."]
    assert len(provider.calls) == 2
    assert "PREVIOUS_MODEL_OUTPUT_FAILED_SCHEMA_VALIDATION" in provider.calls[1]
    assert "SingleRecommendation" in provider.calls[1]


def test_openai_provider_uses_responses_api_json_mode() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"ok": true}',
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 3,
                },
            },
        )

    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://api.openai.test/v1",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
    )

    output = provider.generate("TASK: EVALUATOR\n{}", temperature=0.2, max_tokens=100)

    assert output == '{"ok": true}'
    assert provider.stats.model_calls == 1
    assert provider.stats.estimated_input_tokens == 11
    assert provider.stats.estimated_output_tokens == 3

    request = requests[0]
    assert request.url == "https://api.openai.test/v1/responses"
    assert request.headers["authorization"] == "Bearer test-key"
    payload = json.loads(request.content)
    assert payload["model"] == "gpt-test"
    assert payload["input"] == "TASK: EVALUATOR\n{}"
    assert payload["max_output_tokens"] == 100
    assert payload["temperature"] == 0.2
    assert payload["store"] is False
    assert payload["text"]["format"]["type"] == "json_object"


def test_openai_provider_retries_rate_limit(monkeypatch) -> None:
    requests: list[httpx.Request] = []
    sleeps: list[float] = []

    monkeypatch.setattr("agentic_decision_benchmark.llm.openai.time.sleep", sleeps.append)

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                headers={"retry-after": "0.01"},
                json={"error": {"message": "Rate limit reached."}},
            )
        return httpx.Response(
            200,
            json={
                "output_text": '{"ok": true}',
                "usage": {
                    "input_tokens": 2,
                    "output_tokens": 3,
                },
            },
        )

    provider = OpenAIProvider(
        api_key="test-key",
        base_url="https://api.openai.test/v1",
        model="gpt-test",
        transport=httpx.MockTransport(handler),
        max_retries=1,
    )

    assert provider.generate("{}", temperature=0.2, max_tokens=100) == '{"ok": true}'
    assert len(requests) == 2
    assert sleeps == [0.01]

