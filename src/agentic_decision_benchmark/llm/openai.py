from __future__ import annotations

import re
import time
from typing import Any

import httpx

from agentic_decision_benchmark.llm.base import BaseLLMProvider, estimate_tokens


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MAX_RETRIES = 4
DEFAULT_OPENAI_RETRY_DELAY = 5.0


class OpenAIProvider(BaseLLMProvider):
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        timeout: float = 120.0,
        store: bool = False,
        max_retries: int = DEFAULT_OPENAI_MAX_RETRIES,
        retry_delay: float = DEFAULT_OPENAI_RETRY_DELAY,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        super().__init__()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_PROVIDER=openai.")
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.store = store
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.transport = transport

    def generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": prompt,
            "instructions": (
                "You generate machine-parseable benchmark outputs. "
                "Return exactly one valid JSON object and no surrounding text."
            ),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "store": self.store,
            "text": {"format": {"type": "json_object"}},
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = self._post_with_retries(payload, headers)
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"OpenAI API request failed: {response.status_code} {response.text[:500]}"
            ) from exc

        data = response.json()
        text = _extract_output_text(data)
        self._record_stats(prompt, text, data)
        return text

    def _post_with_retries(self, payload: dict[str, Any], headers: dict[str, str]) -> httpx.Response:
        with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(
                        f"{self.base_url}/responses",
                        json=payload,
                        headers=headers,
                    )
                except httpx.TransportError as exc:
                    if attempt >= self.max_retries:
                        raise RuntimeError(
                            f"OpenAI API request failed after transport retries: {exc}"
                        ) from exc
                    time.sleep(self.retry_delay * float(attempt + 1))
                    continue
                if response.status_code != 429 or attempt >= self.max_retries:
                    return response
                time.sleep(_retry_delay(response, attempt=attempt, fallback=self.retry_delay))
        raise RuntimeError("OpenAI retry loop exited without a response.")

    def _record_stats(self, prompt: str, response_text: str, data: dict[str, Any]) -> None:
        usage = data.get("usage")
        input_tokens = _int_value(usage, "input_tokens") if isinstance(usage, dict) else None
        output_tokens = _int_value(usage, "output_tokens") if isinstance(usage, dict) else None

        self._stats.model_calls += 1
        self._stats.estimated_input_tokens += (
            input_tokens
            if input_tokens is not None
            else estimate_tokens(prompt)
        )
        self._stats.estimated_output_tokens += (
            output_tokens
            if output_tokens is not None
            else estimate_tokens(response_text)
        )


def _int_value(data: dict[str, Any], key: str) -> int | None:
    value = data.get(key)
    return value if isinstance(value, int) else None


def _retry_delay(response: httpx.Response, *, attempt: int, fallback: float) -> float:
    retry_after = response.headers.get("retry-after")
    if retry_after is not None:
        try:
            return max(0.0, float(retry_after))
        except ValueError:
            pass

    message = ""
    try:
        error = response.json().get("error", {})
        if isinstance(error, dict) and isinstance(error.get("message"), str):
            message = error["message"]
    except ValueError:
        message = response.text

    match = re.search(r"try again in ([0-9.]+)s", message, flags=re.IGNORECASE)
    if match:
        return max(0.0, float(match.group(1)) + 1.0)
    return fallback * float(attempt + 1)


def _extract_output_text(data: dict[str, Any]) -> str:
    output_text = data.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text

    parts: list[str] = []
    output = data.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for content_item in content:
                if not isinstance(content_item, dict):
                    continue
                if (
                    content_item.get("type") == "output_text"
                    and isinstance(content_item.get("text"), str)
                ):
                    parts.append(content_item["text"])
    if parts:
        return "\n".join(parts)

    error = data.get("error")
    if isinstance(error, dict) and error.get("message"):
        raise RuntimeError(f"OpenAI response error: {error['message']}")

    incomplete = data.get("incomplete_details")
    if isinstance(incomplete, dict) and incomplete.get("reason"):
        raise RuntimeError(f"OpenAI response incomplete: {incomplete['reason']}")

    raise ValueError(f"OpenAI response missing output text. Response keys: {sorted(data)}")
