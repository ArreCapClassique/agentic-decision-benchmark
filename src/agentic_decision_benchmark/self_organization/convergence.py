from __future__ import annotations

from collections import Counter
from typing import Any

from agentic_decision_benchmark.schemas import AgentOutput, ConflictItem, ConvergenceStatus, StrategyId
from agentic_decision_benchmark.utils.json_utils import to_jsonable


def _output_dict(output: AgentOutput | dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(output)


def _conflict_dict(conflict: ConflictItem | dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(conflict)


def evaluate_convergence(
    *,
    agent_outputs: list[AgentOutput | dict[str, Any]],
    conflict_map: list[ConflictItem | dict[str, Any]],
    cycle_id: int,
    agent_names: list[str] | None = None,
) -> ConvergenceStatus:
    latest_recommendations: dict[str, StrategyId] = {}
    latest_confidence: dict[str, float] = {}
    fallback_recommendations: dict[str, StrategyId] = {}
    fallback_confidence: dict[str, float] = {}

    for raw_output in agent_outputs:
        output = _output_dict(raw_output)
        agent = str(output.get("agent", ""))
        payload = output.get("output", {})
        phase = str(output.get("phase", ""))
        if not agent or not isinstance(payload, dict):
            continue
        if phase == "round_1_independent_analysis" and payload.get("initial_recommendation"):
            fallback_recommendations[agent] = payload["initial_recommendation"]
            fallback_confidence[agent] = float(output.get("confidence", payload.get("confidence", 0.0)))
        if phase.startswith("round_3_belief_update") and payload.get("updated_recommendation"):
            latest_recommendations[agent] = payload["updated_recommendation"]
            latest_confidence[agent] = float(output.get("confidence", payload.get("confidence", 0.0)))

    expected_agents = agent_names or sorted(set(fallback_recommendations) | set(latest_recommendations))
    recommendations: dict[str, StrategyId] = {}
    confidences: dict[str, float] = {}
    for agent in expected_agents:
        if agent in latest_recommendations:
            recommendations[agent] = latest_recommendations[agent]
            confidences[agent] = latest_confidence.get(agent, 0.0)
        elif agent in fallback_recommendations:
            recommendations[agent] = fallback_recommendations[agent]
            confidences[agent] = fallback_confidence.get(agent, 0.0)

    total_agents = len(expected_agents) or len(recommendations)
    recommendation_counts = Counter(recommendations.values())
    if recommendation_counts:
        leading_strategy, leading_count = sorted(
            recommendation_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )[0]
        agreement_ratio = round(leading_count / total_agents, 4) if total_agents else 0.0
    else:
        leading_strategy = None
        agreement_ratio = 0.0

    unresolved_high_severity_conflicts = sum(
        1
        for conflict in conflict_map
        if int(_conflict_dict(conflict).get("severity", 0)) >= 4
        and _conflict_dict(conflict).get("status") == "unresolved"
    )
    low_confidence_agents = sorted(
        agent
        for agent, confidence in confidences.items()
        if confidence < 0.50
    )
    converged = (
        agreement_ratio >= 0.67
        and unresolved_high_severity_conflicts <= 1
        and len(low_confidence_agents) <= 1
    )
    reason = (
        f"agreement_ratio={agreement_ratio}; "
        f"unresolved_high_severity_conflicts={unresolved_high_severity_conflicts}; "
        f"low_confidence_agents={len(low_confidence_agents)}"
    )
    return ConvergenceStatus(
        cycle_id=cycle_id,
        converged=converged,
        leading_strategy=leading_strategy,
        agreement_ratio=agreement_ratio,
        unresolved_high_severity_conflicts=unresolved_high_severity_conflicts,
        low_confidence_agents=low_confidence_agents,
        reason=reason,
    )


def route_after_convergence(
    *,
    convergence: ConvergenceStatus | dict[str, Any],
    deliberation_cycle: int,
    max_deliberation_cycles: int,
) -> str:
    convergence_data = to_jsonable(convergence)
    if convergence_data.get("converged"):
        return "round_4_mcda_scorecard"
    if deliberation_cycle >= max_deliberation_cycles:
        return "round_4_mcda_scorecard"
    return "increment_deliberation_cycle"

