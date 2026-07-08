from __future__ import annotations

from agentic_decision_benchmark.schemas import BlackboardItem
from agentic_decision_benchmark.self_organization.salience import compute_salience_map


def _claim(*, severity: int = 3) -> BlackboardItem:
    return BlackboardItem(
        round_id=1,
        cycle_id=1,
        author="Technology Agent",
        item_type="claim",
        content=f"AI deployment feasibility claim severity {severity}.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="supports",
        confidence=0.7,
        severity=severity,
    )


def test_salience_records_are_produced_for_blackboard_items() -> None:
    item = _claim()
    salience_map = compute_salience_map([item], ["Technology"])
    assert item.id in salience_map
    assert salience_map[item.id].salience > 0


def test_higher_severity_increases_salience() -> None:
    low = _claim(severity=1)
    high = _claim(severity=5)
    assert compute_salience_map([high], ["Technology"])[high.id].salience > compute_salience_map([low], ["Technology"])[low.id].salience


def test_critique_count_increases_salience() -> None:
    item = _claim()
    critique = BlackboardItem(
        round_id=2,
        cycle_id=1,
        author="Operations Agent",
        item_type="critique",
        content="This claim needs stronger data-readiness evidence.",
        topic_tags=["ai_feasibility"],
        related_strategy="A",
        criterion="operational_feasibility",
        stance="challenges",
        target_item_id=item.id,
        confidence=0.8,
        severity=4,
    )
    baseline = compute_salience_map([item], ["Technology"])[item.id].salience
    critiqued = compute_salience_map([item, critique], ["Technology"])[item.id].salience
    assert critiqued > baseline

