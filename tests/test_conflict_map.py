from __future__ import annotations

from agentic_decision_benchmark.schemas import BlackboardItem
from agentic_decision_benchmark.self_organization.conflict_map import build_conflict_map


def test_direct_critique_conflict_created_from_targeted_critique() -> None:
    claim = BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Technology Agent",
        item_type="claim",
        content="AI deployment is feasible.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="supports",
        confidence=0.8,
        severity=4,
    )
    critique = BlackboardItem(
        round_id=2,
        cycle_id=1,
        author="Operations Agent",
        item_type="critique",
        content="This underestimates integration work.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="challenges",
        target_item_id=claim.id,
        confidence=0.8,
        severity=4,
    )
    conflicts = build_conflict_map([claim, critique], cycle_id=1)
    assert [item.conflict_type for item in conflicts] == ["direct_critique"]
    assert conflicts[0].central_item_id == claim.id


def test_opposing_stances_conflict_created_for_same_strategy_criterion_and_topic() -> None:
    support = BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Technology Agent",
        item_type="claim",
        content="AI deployment can support Strategy A.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="supports",
        confidence=0.8,
        severity=3,
    )
    oppose = BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Operations Agent",
        item_type="risk",
        content="AI deployment timing makes Strategy A operationally risky.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="opposes",
        confidence=0.8,
        severity=4,
    )
    conflicts = build_conflict_map([support, oppose], cycle_id=1)
    assert [item.conflict_type for item in conflicts] == ["opposing_stances"]
    assert conflicts[0].topic == "ai_feasibility"


def test_high_severity_risks_from_multiple_agents_create_risk_cluster() -> None:
    finance = BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Finance Agent",
        item_type="risk",
        content="Strategy B strains cash flow.",
        topic_tags=["cash_flow"],
        related_strategy="B",
        criterion="financial_viability",
        stance="opposes",
        confidence=0.8,
        severity=4,
    )
    operations = BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Operations Agent",
        item_type="risk",
        content="Strategy B strains cash flow through disruption.",
        topic_tags=["cash_flow"],
        related_strategy="B",
        criterion="financial_viability",
        stance="opposes",
        confidence=0.8,
        severity=5,
    )
    conflicts = build_conflict_map([finance, operations], cycle_id=1)
    assert [item.conflict_type for item in conflicts] == ["risk_cluster"]
    assert conflicts[0].agents_involved == ["Finance Agent", "Operations Agent"]


def test_conflict_map_is_deterministic_for_same_blackboard_input() -> None:
    claim = BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Technology Agent",
        item_type="claim",
        content="AI deployment is feasible.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="supports",
        confidence=0.8,
        severity=4,
    )
    critique = BlackboardItem(
        round_id=2,
        cycle_id=1,
        author="Finance Agent",
        item_type="critique",
        content="This needs financial gating.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="challenges",
        target_item_id=claim.id,
        confidence=0.8,
        severity=4,
    )
    first = [item.model_dump(mode="json") for item in build_conflict_map([claim, critique], cycle_id=1)]
    second = [item.model_dump(mode="json") for item in build_conflict_map([claim, critique], cycle_id=1)]
    assert first == second

