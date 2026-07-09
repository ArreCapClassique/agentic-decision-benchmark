from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP

from agentic_decision_benchmark.evaluation.metrics import deterministic_metrics_snapshot
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.prompts import build_evaluator_prompt
from agentic_decision_benchmark.schemas import EvaluationResult
from agentic_decision_benchmark.settings import PROJECT_ROOT, load_yaml
from agentic_decision_benchmark.utils.json_utils import extract_json_object


EVALUATOR_SCORE_FIELDS: tuple[str, ...] = (
    "decision_quality",
    "convergence",
    "resilience",
    "explainability",
    "adaptability",
    "cost_efficiency",
    "runtime_efficiency",
)


def _round_decimal(value: float, places: int) -> float:
    quantizer = Decimal("1").scaleb(-places)
    return float(Decimal(str(value)).quantize(quantizer, rounding=ROUND_HALF_UP))


class Evaluator:
    def __init__(self, provider: LLMProvider, *, temperature: float, max_tokens: int) -> None:
        self.provider = provider
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.rubric = load_yaml(PROJECT_ROOT / "config" / "evaluator_rubric.yaml")

    def evaluate(self, state: dict) -> EvaluationResult:
        metrics = deterministic_metrics_snapshot(state, self.provider)
        prompt = build_evaluator_prompt(
            final_recommendation=state.get("final_recommendation", {}),
            deterministic_metrics=metrics,
            rubric=self.rubric,
        )
        response = self.provider.generate(prompt, temperature=self.temperature, max_tokens=self.max_tokens)
        payload = extract_json_object(response)
        if "overall" not in payload:
            raise ValueError("Model output missing required overall score.")
        payload["overall"] = 1.0
        validated = EvaluationResult.model_validate(payload)
        data = validated.model_dump()
        data["overall"] = self._weighted_overall(validated)
        return EvaluationResult.model_validate(data)

    def _weighted_overall(self, evaluation: EvaluationResult) -> float:
        overall_config = self.rubric.get("overall_score", {})
        if overall_config.get("method") != "weighted_average":
            raise ValueError("Evaluator rubric overall_score.method must be 'weighted_average'.")
        weights = overall_config.get("weights", {})
        if not isinstance(weights, dict) or not weights:
            raise ValueError("Evaluator rubric overall_score.weights must be a non-empty mapping.")
        rounding = int(overall_config.get("rounding", 1))

        weighted_total = 0.0
        weight_total = 0.0
        for field in EVALUATOR_SCORE_FIELDS:
            if field not in weights:
                raise ValueError(f"Evaluator rubric missing overall weight for {field!r}.")
            weight = float(weights[field])
            weighted_total += getattr(evaluation, field) * weight
            weight_total += weight
        if weight_total <= 0:
            raise ValueError("Evaluator rubric overall weights must sum to a positive value.")
        return _round_decimal(weighted_total / weight_total, rounding)

