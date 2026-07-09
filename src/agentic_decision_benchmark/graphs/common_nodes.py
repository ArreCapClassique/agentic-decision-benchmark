from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from agentic_decision_benchmark.evaluation.evaluator import Evaluator
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.schemas import AgentOutput, ModeName
from agentic_decision_benchmark.settings import BenchmarkSettings
from agentic_decision_benchmark.utils.json_utils import parse_model_json, to_jsonable


def _schema_repair_prompt(
    *,
    failed_response: str,
    schema: type[BaseModel],
    error: ValueError,
) -> str:
    schema_json = json.dumps(schema.model_json_schema(), indent=2, sort_keys=True)
    return "\n".join(
        [
            "PREVIOUS_MODEL_OUTPUT_FAILED_SCHEMA_VALIDATION",
            "Your previous output was valid JSON but did not match the required schema.",
            "Preserve the same substantive content, but correct only the JSON shape and enum values.",
            "Return a corrected JSON object only. Do not include Markdown or explanations.",
            "If a schema field is an array, return a JSON array even when it has only one item.",
            "If a strategy field expects a strategy ID, return exactly one of A, B, C, or D.",
            f"SCHEMA_NAME: {schema.__name__}",
            f"VALIDATION_ERROR: {error}",
            f"PREVIOUS_OUTPUT: {failed_response}",
            f"JSON_SCHEMA: {schema_json}",
        ]
    )


def provider_json(
    provider: LLMProvider,
    prompt: str,
    schema: type[BaseModel],
    *,
    temperature: float,
    max_tokens: int,
) -> BaseModel:
    response = provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    try:
        return parse_model_json(response, schema)  # type: ignore[return-value]
    except ValueError as first_error:
        repair_prompt = _schema_repair_prompt(
            failed_response=response,
            schema=schema,
            error=first_error,
        )
        repaired_response = provider.generate(repair_prompt, temperature=temperature, max_tokens=max_tokens)
        try:
            return parse_model_json(repaired_response, schema)  # type: ignore[return-value]
        except ValueError as second_error:
            raise ValueError(
                f"Model output failed {schema.__name__} validation after repair retry. "
                f"Initial error: {first_error}. Repair error: {second_error}"
            ) from second_error


def make_agent_output(
    *,
    mode: ModeName,
    phase: str,
    agent: str,
    round_id: int,
    output: BaseModel | dict[str, Any],
    confidence: float,
) -> AgentOutput:
    return AgentOutput(
        mode=mode,
        phase=phase,
        agent=agent,
        round_id=round_id,
        output=to_jsonable(output),
        confidence=confidence,
    )


def make_evaluator_node(provider: LLMProvider, settings: BenchmarkSettings):
    def evaluator_node(state: dict[str, Any]) -> dict[str, Any]:
        evaluator = Evaluator(provider, temperature=settings.temperature, max_tokens=settings.max_tokens)
        evaluation = evaluator.evaluate(state)
        return {"evaluation": evaluation.model_dump(mode="json")}

    return evaluator_node

