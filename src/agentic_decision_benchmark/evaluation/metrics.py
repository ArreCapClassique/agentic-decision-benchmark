from __future__ import annotations

from typing import Any

from agentic_decision_benchmark.llm.base import LLMProvider
from agentic_decision_benchmark.schemas import BlackboardItem, Metrics, ProviderStats, Scorecard
from agentic_decision_benchmark.utils.json_utils import to_jsonable


def _provider_stats(provider: LLMProvider) -> ProviderStats:
    return provider.stats


def _flatten_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return " ".join(_flatten_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_flatten_text(item) for item in value)
    return str(value)


def compute_metrics(state: dict[str, Any], provider: LLMProvider, runtime_seconds: float = 0.0) -> Metrics:
    blackboard: list[BlackboardItem] = list(state.get("blackboard", []))
    scorecards: list[Scorecard] = list(state.get("scorecards", []))
    final_recommendation = state.get("final_recommendation", {})
    stats = _provider_stats(provider)
    aggregate_scores = final_recommendation.get("aggregate_scores", {}) if isinstance(final_recommendation, dict) else {}
    stdev_values = [
        float(item.get("stdev", 0.0))
        for item in aggregate_scores.values()
        if isinstance(item, dict) and "stdev" in item
    ]
    agent_output_rounds = []
    for item in state.get("agent_outputs", []):
        if isinstance(item, dict):
            agent_output_rounds.append(int(item.get("round_id", 0)))
        else:
            agent_output_rounds.append(int(getattr(item, "round_id", 0)))
    max_round = max(
        [int(state.get("round_id", 0))]
        + [item.round_id for item in blackboard]
        + [scorecard.round_id for scorecard in scorecards]
        + agent_output_rounds
    )
    final_strategy = None
    if isinstance(final_recommendation, dict):
        final_strategy = final_recommendation.get("recommended_strategy")

    state_text = _flatten_text(to_jsonable(state)).lower()
    faulty_present = "3 months with minimal cost and minimal integration risk" in state_text
    fault_corrected = faulty_present and any(
        word in state_text for word in ["reject", "rejected", "overconfident", "unsupported", "underestimating"]
    )
    new_info_incorporated = bool(
        state.get("new_info_injection")
        and ("24-month" in state_text or "24 months" in state_text)
        and any(word in state_text for word in ["belief", "feasibility", "deadline", "phased"])
    )

    return Metrics(
        runtime_seconds=round(runtime_seconds, 4),
        model_calls=stats.model_calls,
        estimated_input_tokens=stats.estimated_input_tokens,
        estimated_output_tokens=stats.estimated_output_tokens,
        total_estimated_tokens=stats.total_estimated_tokens,
        number_of_rounds=max_round,
        number_of_blackboard_items=len(blackboard),
        number_of_claim_items=sum(1 for item in blackboard if item.item_type == "claim"),
        number_of_critique_items=sum(1 for item in blackboard if item.item_type == "critique"),
        number_of_belief_update_items=sum(1 for item in blackboard if item.item_type == "belief_update"),
        number_of_scorecards=len(scorecards),
        agreement_ratio=final_recommendation.get("agreement_ratio") if isinstance(final_recommendation, dict) else None,
        score_variance=round(sum(stdev_values) / len(stdev_values), 4) if stdev_values else None,
        final_selected_strategy=final_strategy,
        fault_correction_detected=fault_corrected,
        new_information_incorporated=new_info_incorporated,
    )


def deterministic_metrics_snapshot(state: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    return compute_metrics(state, provider, runtime_seconds=0.0).model_dump(mode="json")

