from __future__ import annotations

from typing import Any

from pydantic import BaseModel

from agentic_decision_benchmark.evaluation.evaluator import Evaluator
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.schemas import AgentOutput, ModeName
from agentic_decision_benchmark.settings import BenchmarkSettings
from agentic_decision_benchmark.utils.json_utils import parse_model_json, to_jsonable


def provider_json(
    provider: LLMProvider,
    prompt: str,
    schema: type[BaseModel],
    *,
    temperature: float,
    max_tokens: int,
) -> BaseModel:
    response = provider.generate(prompt, temperature=temperature, max_tokens=max_tokens)
    return parse_model_json(response, schema)  # type: ignore[return-value]


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

