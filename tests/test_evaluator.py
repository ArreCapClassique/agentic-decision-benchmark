from __future__ import annotations

from agentic_decision_benchmark.evaluation.evaluator import Evaluator
from agentic_decision_benchmark.llm.mock import MockProvider
from agentic_decision_benchmark.settings import load_settings


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
        assert 1 <= data[key] <= 5
    assert data["rationale"]

