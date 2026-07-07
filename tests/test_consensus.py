from __future__ import annotations

from agentic_decision_benchmark.consensus.mcda import run_mcda_consensus
from agentic_decision_benchmark.schemas import BlackboardItem, CriteriaScores, Scorecard, StrategyScore


STRATEGIES = {
    "A": {"name": "A", "description": "A"},
    "B": {"name": "B", "description": "B"},
    "C": {"name": "C", "description": "C"},
    "D": {"name": "D", "description": "D"},
}


def _score(overall: float, legal: int = 3, financial: int = 3, operations: int = 3) -> StrategyScore:
    return StrategyScore(
        criteria=CriteriaScores(
            financial_viability=financial,
            operational_feasibility=operations,
            strategic_value=int(round(overall)),
            workforce_feasibility=3,
            legal_compliance_safety=legal,
        ),
        overall=overall,
        rationale="test",
    )


def _card(agent: str, scores: dict[str, float]) -> Scorecard:
    return Scorecard(
        agent=agent,
        scores={
            "A": _score(scores["A"]),
            "B": _score(scores["B"]),
            "C": _score(scores["C"]),
            "D": _score(scores["D"]),
        },
        confidence=0.8,
    )


def test_mcda_selects_correct_top_strategy() -> None:
    result = run_mcda_consensus(
        scorecards=[
            _card("one", {"A": 3, "B": 2, "C": 4, "D": 1}),
            _card("two", {"A": 3, "B": 2, "C": 5, "D": 1}),
        ],
        candidate_strategies=STRATEGIES,
        blackboard=[],
    )
    assert result.recommended_strategy == "C"
    assert result.aggregate_scores["C"].mean == 4.5
    assert result.aggregate_scores["C"].stdev == 0.5
    assert result.agreement_ratio == 1.0


def test_tie_breaking_uses_legal_then_financial_then_operations() -> None:
    card = Scorecard(
        agent="one",
        scores={
            "A": _score(4, legal=3, financial=5, operations=5),
            "B": _score(4, legal=5, financial=3, operations=3),
            "C": _score(2),
            "D": _score(1),
        },
        confidence=0.8,
    )
    result = run_mcda_consensus(scorecards=[card], candidate_strategies=STRATEGIES, blackboard=[])
    assert result.recommended_strategy == "B"


def test_minority_concerns_are_preserved() -> None:
    concern = BlackboardItem(
        round_id=3,
        author="HR Agent",
        item_type="minority_concern",
        content="Hiring may constrain the ramp.",
        confidence=0.8,
    )
    result = run_mcda_consensus(
        scorecards=[
            _card("one", {"A": 3, "B": 2, "C": 4, "D": 1}),
            _card("two", {"A": 3, "B": 2, "C": 4, "D": 1}),
        ],
        candidate_strategies=STRATEGIES,
        blackboard=[concern],
    )
    assert "Hiring may constrain the ramp." in result.minority_concerns

