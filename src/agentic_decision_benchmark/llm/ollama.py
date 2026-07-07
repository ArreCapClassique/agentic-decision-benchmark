from __future__ import annotations

import httpx

from agentic_decision_benchmark.llm.base import BaseLLMProvider


class OllamaProvider(BaseLLMProvider):
    def __init__(self, *, base_url: str, model: str, timeout: float = 120.0) -> None:
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            data = response.json()
        text = data.get("response")
        if not isinstance(text, str):
            raise ValueError(f"Ollama response missing text response: {data}")
        return text

