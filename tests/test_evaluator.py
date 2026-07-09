from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentic_decision_benchmark.evaluation.evaluator import Evaluator
from agentic_decision_benchmark.llm.base import BaseLLMProvider
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.schemas import EvaluationResult
from agentic_decision_benchmark.settings import load_settings


class StaticEvaluationProvider(BaseLLMProvider):
    def __init__(self, payload: dict) -> None:
        super().__init__()
        self.payload = payload

    def _generate(self, prompt: str, *, temperature: float, max_tokens: int) -> str:
        return json.dumps(self.payload)


def _valid_payload(**overrides):
    payload = {
        "decision_quality": 4.1,
        "convergence": 3.8,
        "resilience": 3.4,
        "explainability": 4.2,
        "adaptability": 3.3,
        "cost_efficiency": 2.6,
        "runtime_efficiency": 3.1,
        "overall": 1.0,
        "rationale": "Test evaluation.",
    }
    payload.update(overrides)
    return payload


def test_evaluator_returns_required_fields_in_range(tmp_path) -> None:
    settings = load_settings(provider="mock", output_dir=str(tmp_path))
    evaluator = Evaluator(MockProvider(), temperature=settings.temperature, max_tokens=settings.max_tokens)
    result = evaluator.evaluate(
        {
            "final_recommendation": {"recommended_strategy": "C", "recommendation": "Use staged dual track."},
            "blackboard": [],
            "scorecards": [],
            "agent_outputs": [],
            "new_info_injection": False,
        }
    )
    data = result.model_dump()
    for key in [
        "decision_quality",
        "convergence",
        "resilience",
        "explainability",
        "adaptability",
        "cost_efficiency",
        "runtime_efficiency",
        "overall",
    ]:
        assert isinstance(data[key], float)
        assert 1.0 <= data[key] <= 5.0
    assert data["rationale"]


def test_evaluation_result_accepts_and_rounds_decimal_scores() -> None:
    result = EvaluationResult.model_validate(_valid_payload(decision_quality=4.25))

    assert result.decision_quality == 4.3


def test_evaluation_result_accepts_integer_scores_as_floats() -> None:
    result = EvaluationResult.model_validate(
        _valid_payload(
            decision_quality=4,
            convergence=4,
            resilience=4,
            explainability=4,
            adaptability=4,
            cost_efficiency=4,
            runtime_efficiency=4,
            overall=4,
        )
    )

    assert result.decision_quality == 4.0
    assert isinstance(result.decision_quality, float)


def test_evaluation_result_rejects_scores_below_minimum() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(_valid_payload(decision_quality=0.9))


def test_evaluation_result_rejects_scores_above_maximum() -> None:
    with pytest.raises(ValidationError):
        EvaluationResult.model_validate(_valid_payload(decision_quality=5.1))


def test_evaluator_overrides_llm_overall_with_weighted_decimal(tmp_path) -> None:
    settings = load_settings(provider="mock", output_dir=str(tmp_path))
    evaluator = Evaluator(
        StaticEvaluationProvider(_valid_payload(decision_quality=4.25, overall=1.0)),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    result = evaluator.evaluate(
        {
            "final_recommendation": {"recommended_strategy": "C", "recommendation": "Use staged dual track."},
            "blackboard": [],
            "scorecards": [],
            "agent_outputs": [],
            "new_info_injection": False,
        }
    )

    assert result.decision_quality == 4.3
    assert result.overall == 3.7

