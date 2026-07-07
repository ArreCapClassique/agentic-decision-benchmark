from __future__ import annotations

from agentic_decision_benchmark.evaluation.metrics import deterministic_metrics_snapshot
from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.prompts import build_evaluator_prompt
from agentic_decision_benchmark.schemas import EvaluationResult
from agentic_decision_benchmark.settings import PROJECT_ROOT, load_yaml
from agentic_decision_benchmark.utils.json_utils import parse_model_json


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
        return parse_model_json(response, EvaluationResult)

