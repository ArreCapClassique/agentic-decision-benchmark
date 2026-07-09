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
    conflict_map = [to_jsonable(item) for item in state.get("conflict_map", [])]
    salience_map = {key: to_jsonable(value) for key, value in state.get("salience_map", {}).items()}
    convergence_history = [to_jsonable(item) for item in state.get("convergence_history", [])]
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
    decision_artifact_text = _flatten_text(
        {
            "final_recommendation": final_recommendation,
            "agent_outputs": [to_jsonable(item) for item in state.get("agent_outputs", [])],
        }
    ).lower()
    faulty_present = "3 months with minimal cost and minimal integration risk" in state_text
    fault_corrected = faulty_present and any(
        word in state_text for word in ["reject", "rejected", "overconfident", "unsupported", "underestimating"]
    )
    new_info_incorporated = bool(
        state.get("new_info_injection")
        and state.get("new_information_injected")
        and ("24-month" in decision_artifact_text or "24 months" in decision_artifact_text)
        and any(word in decision_artifact_text for word in ["belief", "feasibility", "deadline", "phased"])
    )
    conflict_type_counts = {
        conflict_type: sum(1 for item in conflict_map if item.get("conflict_type") == conflict_type)
        for conflict_type in ("direct_critique", "opposing_stances", "risk_cluster")
    }
    unresolved_high_severity_conflicts = sum(
        1
        for item in conflict_map
        if int(item.get("severity", 0)) >= 4 and item.get("status") == "unresolved"
    )
    salience_values = [float(item.get("salience", 0.0)) for item in salience_map.values()]
    blackboard_by_id = {item.id: item for item in blackboard}
    top_salience_items = []
    for item_id, record in sorted(salience_map.items(), key=lambda entry: (-float(entry[1].get("salience", 0.0)), entry[0]))[:5]:
        blackboard_item = blackboard_by_id.get(item_id)
        top_salience_items.append(
            {
                "item_id": item_id,
                "salience": record.get("salience", 0.0),
                "author": blackboard_item.author if blackboard_item else None,
                "item_type": blackboard_item.item_type if blackboard_item else None,
                "content": blackboard_item.content if blackboard_item else None,
            }
        )
    deliberation_cycle = int(state.get("deliberation_cycle", 0) or 0)
    max_deliberation_cycles = int(state.get("max_deliberation_cycles", 0) or 0)
    latest_convergence = to_jsonable(state.get("convergence", {}))

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
        deliberation_cycles_used=deliberation_cycle,
        max_deliberation_cycles=max_deliberation_cycles,
        converged_before_max_cycles=bool(latest_convergence.get("converged") and deliberation_cycle < max_deliberation_cycles),
        number_of_conflicts=len(conflict_map),
        number_of_direct_critique_conflicts=conflict_type_counts["direct_critique"],
        number_of_opposing_stance_conflicts=conflict_type_counts["opposing_stances"],
        number_of_risk_cluster_conflicts=conflict_type_counts["risk_cluster"],
        unresolved_high_severity_conflicts=unresolved_high_severity_conflicts,
        agreement_ratio_after_each_cycle=[
            float(item.get("agreement_ratio", 0.0))
            for item in convergence_history
        ],
        low_confidence_agents_after_each_cycle=[
            list(item.get("low_confidence_agents", []))
            for item in convergence_history
        ],
        average_salience=round(sum(salience_values) / len(salience_values), 4) if salience_values else 0.0,
        top_salience_items=top_salience_items,
    )


def deterministic_metrics_snapshot(state: dict[str, Any], provider: LLMProvider) -> dict[str, Any]:
    return compute_metrics(state, provider, runtime_seconds=0.0).model_dump(mode="json")

