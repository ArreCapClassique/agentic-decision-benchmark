from __future__ import annotations

import pytest

from agentic_decision_benchmark.schemas import BlackboardContribution, BlackboardItem, DomainAnalysis, Round1Analysis
from agentic_decision_benchmark.state import append_reducer, create_initial_state, validate_mode


def test_state_initializes_correctly() -> None:
    state = create_initial_state(
        mode="self_organizing",
        scenario="case",
        candidate_strategies={"A": {}},
    )
    assert state["mode"] == "self_organizing"
    assert state["blackboard"] == []
    assert state["agent_outputs"] == []
    assert state["scorecards"] == []
    assert state["deliberation_cycle"] == 0
    assert state["max_deliberation_cycles"] == 2
    assert state["salience_map"] == {}
    assert state["conflict_map"] == []
    assert state["convergence"] == {}
    assert state["convergence_history"] == []
    assert state["new_information_injected"] is False
    assert state["fault_injection"] is False
    assert state["metadata"]["information_setting"] == "equal_total"
    assert state["metadata"]["private_role_briefs_enabled"] is True


def test_append_reducer_does_not_overwrite_blackboard_items() -> None:
    first = [BlackboardItem(round_id=1, author="A", item_type="claim", content="one", confidence=0.7)]
    second = [BlackboardItem(round_id=2, author="B", item_type="critique", content="two", confidence=0.8)]
    merged = append_reducer(first, second)
    assert [item.content for item in merged] == ["one", "two"]


def test_mode_validation() -> None:
    assert validate_mode("single") == "single"
    assert validate_mode("supervisor") == "supervisor"
    assert validate_mode("self_organizing") == "self_organizing"
    with pytest.raises(ValueError):
        validate_mode("committee")


def test_strategy_id_fields_normalize_unique_strategy_mentions() -> None:
    analysis = DomainAnalysis(
        claims=["claim"],
        risks=["risk"],
        opportunities=["opportunity"],
        assumptions=["assumption"],
        recommendation="Prefer Strategy A because it protects revenue.",
        confidence=0.7,
    )

    assert analysis.recommendation == "A"


def test_text_list_fields_accept_model_text_objects() -> None:
    analysis = Round1Analysis(
        claims=[{"text": "Claim text", "confidence": 0.9}],
        risks=[{"text": "Risk text", "severity": "high"}],
        opportunities=[{"text": "Opportunity text"}],
        assumptions=[{"text": "Assumption text"}],
        questions_for_others=[{"text": "Question text?"}],
        domain_preferred_strategy="Strategy A",
        organization_preferred_strategy="Strategy C",
        domain_ranking=["Strategy A", "Strategy C"],
        organization_ranking=["Strategy C", "Strategy A"],
        strategies_supported=["Strategy A", "Strategy C"],
        strategies_challenged=["Strategy B"],
        non_negotiable_constraints=[{"text": "Constraint text"}],
        conditions_for_accepting_other_strategy={"Strategy C": [{"text": "Condition text"}]},
        confidence=0.7,
    )

    assert analysis.claims == ["Claim text"]
    assert analysis.risks == ["Risk text"]
    assert analysis.domain_preferred_strategy == "A"
    assert analysis.organization_preferred_strategy == "C"
    assert analysis.domain_ranking == ["A", "C"]
    assert analysis.organization_ranking == ["C", "A"]
    assert analysis.strategies_supported == ["A", "C"]
    assert analysis.strategies_challenged == ["B"]
    assert analysis.non_negotiable_constraints == ["Constraint text"]
    assert analysis.conditions_for_accepting_other_strategy == {"C": ["Condition text"]}


def test_blackboard_contribution_normalizes_item_type_and_severity_labels() -> None:
    contribution = BlackboardContribution(
        item_type="liquidity_risk",
        content="Liquidity may tighten.",
        severity="high",
        confidence=0.8,
    )

    assert contribution.item_type == "risk"
    assert contribution.severity == 5

