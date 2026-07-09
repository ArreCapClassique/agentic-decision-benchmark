from __future__ import annotations

import statistics
from collections import Counter
from typing import Any

from agentic_decision_benchmark.schemas import (
    AggregateStrategyScore,
    BlackboardItem,
    ConsensusResult,
    MCDA_CRITERIA,
    STRATEGY_IDS,
    Scorecard,
    StrategyId,
)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def _stdev(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    return round(statistics.pstdev(values), 4)


def _strategy_description(strategy_id: StrategyId | None, candidate_strategies: dict[str, Any]) -> str:
    if strategy_id is None:
        return "Unresolved tie after deterministic MCDA tie-breakers."
    details = candidate_strategies.get(strategy_id, {})
    return str(details.get("description") or details.get("name") or strategy_id)


def _strategy_label(strategy_id: StrategyId | None, candidate_strategies: dict[str, Any]) -> str:
    if strategy_id is None:
        return "unresolved"
    details = candidate_strategies.get(strategy_id, {})
    name = details.get("name") if isinstance(details, dict) else None
    return f"Strategy {strategy_id} ({name})" if name else f"Strategy {strategy_id}"


def _scorecard_top_vote(scorecard: Scorecard) -> StrategyId:
    ranked = sorted(scorecard.scores.items(), key=lambda item: (-item[1].overall, item[0]))
    return ranked[0][0]


def _equal_best(candidates: list[StrategyId], key_values: dict[StrategyId, float], *, higher: bool) -> list[StrategyId]:
    if not candidates:
        return []
    best_value = max(key_values[item] for item in candidates) if higher else min(key_values[item] for item in candidates)
    return [item for item in candidates if abs(key_values[item] - best_value) < 1e-9]


def _resolve_tie(finalists: list[StrategyId], aggregate_scores: dict[StrategyId, AggregateStrategyScore]) -> list[StrategyId]:
    remaining = finalists
    for criterion in ("legal_compliance_safety", "financial_viability", "operational_feasibility"):
        values = {
            strategy: aggregate_scores[strategy].criteria_means.get(criterion, 0.0)
            for strategy in remaining
        }
        remaining = _equal_best(remaining, values, higher=True)
        if len(remaining) <= 1:
            return remaining
    variance_values = {strategy: aggregate_scores[strategy].stdev for strategy in remaining}
    return _equal_best(remaining, variance_values, higher=False)


def run_mcda_consensus(
    *,
    scorecards: list[Scorecard],
    candidate_strategies: dict[str, Any],
    blackboard: list[BlackboardItem],
) -> ConsensusResult:
    if not scorecards:
        raise ValueError("Cannot run MCDA consensus without scorecards.")

    top_votes = Counter(_scorecard_top_vote(scorecard) for scorecard in scorecards)
    aggregate: dict[StrategyId, AggregateStrategyScore] = {}

    for strategy in STRATEGY_IDS:
        overall_scores = [scorecard.scores[strategy].overall for scorecard in scorecards]
        criteria_means = {
            criterion: _mean(
                [float(getattr(scorecard.scores[strategy].criteria, criterion)) for scorecard in scorecards]
            )
            for criterion in MCDA_CRITERIA
        }
        aggregate[strategy] = AggregateStrategyScore(
            mean=_mean(overall_scores),
            stdev=_stdev(overall_scores),
            top_votes=int(top_votes.get(strategy, 0)),
            criteria_means=criteria_means,
        )

    highest_mean = max(item.mean for item in aggregate.values())
    finalists = [strategy for strategy, item in aggregate.items() if abs(item.mean - highest_mean) < 1e-9]
    resolved = _resolve_tie(finalists, aggregate) if len(finalists) > 1 else finalists

    if len(resolved) == 1:
        recommended_strategy: StrategyId | None = resolved[0]
        result_status = "selected"
        unresolved_strategies: list[StrategyId] = []
    else:
        recommended_strategy = None
        result_status = "unresolved"
        unresolved_strategies = resolved

    winner_for_agreement = recommended_strategy or sorted(unresolved_strategies)[0]
    agreement_ratio = round(top_votes.get(winner_for_agreement, 0) / len(scorecards), 4)

    minority_concerns = [
        item.content
        for item in blackboard
        if item.item_type == "minority_concern"
        or (item.item_type == "critique" and item.severity is not None and item.severity >= 4)
    ]
    unresolved_risks = [
        item.content
        for item in blackboard
        if item.item_type == "risk"
        or (item.item_type == "critique" and item.severity is not None and item.severity >= 4)
    ]

    if recommended_strategy is not None:
        for scorecard in scorecards:
            agent_top = _scorecard_top_vote(scorecard)
            if agent_top != recommended_strategy:
                minority_concerns.append(
                    f"{scorecard.agent} top-scored {_strategy_label(agent_top, candidate_strategies)} "
                    f"while consensus selected {_strategy_label(recommended_strategy, candidate_strategies)}."
                )

    average_variance = _mean([item.stdev for item in aggregate.values()])
    confidence = max(0.4, min(0.95, round(0.55 + agreement_ratio * 0.35 - average_variance * 0.05, 4)))
    summary_target = (
        _strategy_label(recommended_strategy, candidate_strategies)
        if recommended_strategy
        else ", ".join(_strategy_label(strategy, candidate_strategies) for strategy in unresolved_strategies)
    )
    deliberation_summary = [
        f"Collected {len(scorecards)} MCDA scorecards.",
        f"Highest mean finalist(s): {summary_target}.",
        f"Agreement ratio based on top votes: {agreement_ratio}.",
    ]

    return ConsensusResult(
        recommended_strategy=recommended_strategy,
        strategy_description=_strategy_description(recommended_strategy, candidate_strategies),
        aggregate_scores=aggregate,
        agreement_ratio=agreement_ratio,
        minority_concerns=minority_concerns,
        unresolved_risks=unresolved_risks,
        deliberation_summary=deliberation_summary,
        confidence=confidence,
        result_status=result_status,
        unresolved_strategies=unresolved_strategies,
    )

